"""Apple-Silicon MLX backend for Meteo's NAGISA-style value-only NNUE.

The model matches the public NAGISA V3.1 topology: HalfKA_hm2 input,
1024-wide pairwise feature transformer, 16/64 hidden dimensions, and nine
progress8kpabs LayerStacks.  PackedSfenValue decoding and feature indexing are
delegated to a small CPU-only Rust companion compiled from the pinned Tatara
sources, while MLX performs forward/backward/update work on the local GPU.
"""

from __future__ import annotations

import gc
import hashlib
import http.client
import json
import math
import os
import platform
import queue
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil
from mlx.utils import tree_flatten, tree_unflatten
from numpy.typing import NDArray

from .mlx_training_primitives import (
    DaemonPrefetch,
    NumericPhase,
    QuantisationSchedule,
    fake_quantise_rust_f32_raw,
    ste_floor,
)
from .nnue_training import (
    DEFAULT_NET_ID,
    NagisaNnueArchitecture,
    PublicPsvShard,
    _dataloader_threads,
    _exclusive_run_lock,
    _fsync_directory,
    _fsync_file,
    _json_bytes,
    _sha256_file,
    _shards_from_index,
    _strict_json,
    _write_json_atomic,
    _write_new,
    _yaneuraou_architecture,
    download_nagisa_shard,
    ensure_nagisa_heldout,
    load_nagisa_plan,
    nagisa_wrm_contract,
    probe_value_only_psv,
)

MLX_BACKEND_PLAN_SCHEMA = "meteo-nagisa-mlx-backend-plan-v3"
LEGACY_MLX_BACKEND_PLAN_SCHEMAS = frozenset({"meteo-nagisa-mlx-backend-plan-v2"})
MLX_BACKEND_STATE_SCHEMA = "meteo-nagisa-mlx-backend-state-v1"
MLX_CHECKPOINT_SCHEMA = "meteo-nagisa-mlx-checkpoint-v1"
MLX_METRIC_SCHEMA = "meteo-nagisa-mlx-metric-v1"
MLX_NUMERIC_REPAIR_SCHEMA = "meteo-nagisa-mlx-numeric-repair-v1"
MLX_EXPORT_ROUNDING_REPAIR_SCHEMA = "meteo-nagisa-mlx-export-rounding-repair-v1"
NATIVE_BUILD_SCHEMA = "meteo-nagisa-mlx-native-build-v1"
NATIVE_PROTOCOL_MAGIC = b"MTMLX001"
QUANTIZED_PROTOCOL_MAGIC = b"MTQNT001"
NATIVE_PROTOCOL_VERSION = 1
MAX_ACTIVE = 40
FT_IN = 73_305
PIECE_INPUTS = 1_629
FT_OUT = 1_024
L1_OUT = 16
L1_EFFECTIVE = 15
L2_IN = 30
L2_OUT = 64
NUM_BUCKETS = 9
PAIRWISE_SCALE = 127.0 / 128.0
DENSE_CLAMP = 127.0 / 64.0
QUANTISATION_QA = 127.0
QUANTISATION_QB = 64.0
QUANTISATION_BIAS_SCALE = QUANTISATION_QA * QUANTISATION_QB
DEFAULT_QUANTISATION_WARMUP_STEPS = 1_024
DEFAULT_MLX_BATCH_SIZE = 65_536
DEFAULT_CHECKPOINT_BATCHES = 512
DEFAULT_LOG_BATCHES = 32
DEFAULT_VALIDATION_POSITIONS = 262_144
DEFAULT_SMOKE_POSITIONS = 8_192
DEFAULT_MLX_CACHE_BYTES = 24 * 1024**3
SOURCE_PREFETCH_STORAGE_RESERVE_BYTES = 12 * 1024**3
LEGACY_QAT_FEATURE_ACCUMULATION = "float16_audited_against_float32_reference"
QAT_FEATURE_ACCUMULATION = (
    "fused_metal_int32_forward_fp16_scatter_vjp_audited_against_float32_reference"
)
LEGACY_EXPORT_ROUNDING = "mlx_float32_product_round_ties_to_even"
TATARA_EXPORT_ROUNDING = "rust_f32_widen_f64_scale_round_half_away_fast_two_sum_ste"
MODEL_KEYS = (
    "ft_real",
    "ft_virtual",
    "ft_b",
    "l1_w",
    "l1_b",
    "l1f_w",
    "l1f_b",
    "l2_w",
    "l2_b",
    "l3_w",
    "l3_b",
)
MODEL_SHAPES: dict[str, tuple[int, ...]] = {
    "ft_real": (FT_IN, FT_OUT),
    "ft_virtual": (PIECE_INPUTS, FT_OUT),
    "ft_b": (FT_OUT,),
    "l1_w": (NUM_BUCKETS, L1_OUT, FT_OUT),
    "l1_b": (NUM_BUCKETS, L1_OUT),
    "l1f_w": (FT_OUT, L1_OUT),
    "l1f_b": (L1_OUT,),
    "l2_w": (NUM_BUCKETS, L2_OUT, L2_IN),
    "l2_b": (NUM_BUCKETS, L2_OUT),
    "l3_w": (NUM_BUCKETS, L2_OUT),
    "l3_b": (NUM_BUCKETS,),
}

_EXACT_SPARSE_FT_ACCUMULATION_KERNEL = mx.fast.metal_kernel(
    name="meteo_exact_sparse_ft_accumulation",
    input_names=("ft_raw", "indices", "nnz"),
    output_names=("accumulated",),
    source=f"""
        uint element = thread_position_in_grid.x;
        uint batch = element / {FT_OUT};
        uint output = element % {FT_OUT};
        int total = 0;
        for (uint slot = 0; slot < {MAX_ACTIVE}; ++slot) {{
            if (slot < uint(nnz[batch])) {{
                int feature = indices[batch * {MAX_ACTIVE} + slot];
                total += int(ft_raw[uint(feature) * {FT_OUT} + output]);
            }}
        }}
        accumulated[element] = float(total);
    """,
)


@mx.custom_function  # type: ignore[untyped-decorator]
def _exact_sparse_ft_accumulation(
    ft_raw: mx.array,
    indices: mx.array,
    nnz: mx.array,
) -> mx.array:
    """Accumulate quantised sparse FT rows with native int32 semantics."""

    if ft_raw.ndim != 2 or ft_raw.shape[1] != FT_OUT:
        raise ValueError("quantised FT table must have shape [features, 1024]")
    if indices.ndim != 2 or indices.shape[1] != MAX_ACTIVE:
        raise ValueError("sparse FT indices must have shape [batch, 40]")
    if nnz.ndim != 1 or nnz.shape[0] != indices.shape[0]:
        raise ValueError("sparse FT NNZ must have one value per batch row")
    return cast(
        list[mx.array],
        _EXACT_SPARSE_FT_ACCUMULATION_KERNEL(
            inputs=(ft_raw, indices, nnz),
            grid=(indices.shape[0] * FT_OUT, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=((indices.shape[0], FT_OUT),),
            output_dtypes=(mx.float32,),
        ),
    )[0]


@_exact_sparse_ft_accumulation.vjp  # type: ignore[untyped-decorator]
def _exact_sparse_ft_accumulation_vjp(
    primals: tuple[mx.array, mx.array, mx.array],
    cotangent: mx.array,
    _output: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Scatter the exact linear sum gradient back to active FT rows."""

    ft_raw, indices, nnz = primals
    mask = (mx.arange(MAX_ACTIVE, dtype=mx.int32)[None, :] < nnz[:, None]).astype(cotangent.dtype)
    safe_indices = mx.maximum(indices, 0)
    updates = (cotangent[:, None, :] * mask[..., None]).astype(mx.float16)
    ft_gradient = (mx.zeros(ft_raw.shape, dtype=mx.float16).at[safe_indices].add(updates)).astype(
        ft_raw.dtype
    )
    return ft_gradient, mx.zeros_like(indices), mx.zeros_like(nnz)


CLAMPED_MODEL_KEYS = (
    "l1_w",
    "l1_b",
    "l1f_w",
    "l1f_b",
    "l2_w",
    "l2_b",
    "l3_w",
)
_METRIC_LOCK = threading.Lock()
ModelForwardMode = Literal[
    "float32",
    "mixed_precision_warmup",
    "quantisation_aware",
    "quantised_reference",
]


@dataclass(frozen=True, slots=True)
class WrmLossParameters:
    """Tatara WRM parameters with an inference-scale consistency check."""

    nnue2score: float
    in_scaling: float
    in_offset: float
    target_scaling: float
    target_offset: float
    wdl_lambda: float = 0.0

    @classmethod
    def from_training(cls, training: Mapping[str, object]) -> WrmLossParameters:
        score_scale = float(cast(float, training["score_scale"]))
        expected = nagisa_wrm_contract(score_scale)
        observed = training.get("loss")
        if not isinstance(observed, dict) or observed != expected:
            raise ValueError("MLX run does not contain the scale-aligned NAGISA WRM contract")
        if float(cast(float, observed["loss_pow_exp"])) != 2.0:
            raise ValueError("MLX currently requires Tatara's non-extended squared WRM loss")
        if any(
            float(cast(float, observed[key])) != expected_value
            for key, expected_value in (
                ("loss_qp_asymmetry", 0.0),
                ("loss_weight_boost_w1", 0.0),
                ("wdl_lambda", 0.0),
            )
        ):
            raise ValueError("MLX WRM extension or WDL weights differ from the production recipe")
        return cls(
            nnue2score=float(cast(float, observed["nnue2score"])),
            in_scaling=float(cast(float, observed["in_scaling"])),
            in_offset=float(cast(float, observed["in_offset"])),
            target_scaling=float(cast(float, observed["target_scaling"])),
            target_offset=float(cast(float, observed["target_offset"])),
            wdl_lambda=float(cast(float, observed["wdl_lambda"])),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "nnue2score": self.nnue2score,
            "in_scaling": self.in_scaling,
            "in_offset": self.in_offset,
            "target_scaling": self.target_scaling,
            "target_offset": self.target_offset,
            "wdl_lambda": self.wdl_lambda,
        }


class TataraRanger(optim.Optimizer):
    """MLX port of Tatara's default RAdam + Lookahead update.

    The defaults and update order follow ``nnue-train`` exactly: beta1=.99,
    beta2=.999, epsilon=1e-8, n_sma threshold=5, alpha=.5, k=6, and zeroed
    initial slow weights.  The production recipe keeps decay at Tatara CLI's
    explicit default of zero.
    """

    def __init__(
        self,
        learning_rate: float | Callable[[mx.array], mx.array],
        *,
        decay: float = 0.0,
        beta1: float = 0.99,
        beta2: float = 0.999,
        eps: float = 1e-8,
        alpha: float = 0.5,
        lookahead_k: int = 6,
        n_sma_threshold: float = 5.0,
    ) -> None:
        super().__init__()
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("Ranger betas must be in (0, 1)")
        if eps <= 0.0 or decay < 0.0 or not 0.0 <= alpha <= 1.0:
            raise ValueError("Ranger epsilon/decay/alpha are invalid")
        if lookahead_k < 1 or n_sma_threshold <= 0.0:
            raise ValueError("Ranger lookahead and n_sma threshold must be positive")
        self._maybe_schedule("learning_rate", learning_rate)
        self.decay = decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.alpha = alpha
        self.lookahead_k = lookahead_k
        self.n_sma_threshold = n_sma_threshold

    def init_single(self, parameter: mx.array, state: dict[str, mx.array]) -> None:
        state["m"] = mx.zeros_like(parameter)
        state["v"] = mx.zeros_like(parameter)
        state["slow"] = mx.zeros_like(parameter)

    def apply_single(
        self,
        gradient: mx.array,
        parameter: mx.array,
        state: dict[str, mx.array],
    ) -> mx.array:
        step = self.step.astype(mx.float32)
        learning_rate = self.learning_rate.astype(gradient.dtype)
        beta1 = self.beta1
        beta2 = self.beta2
        beta2_t = mx.power(beta2, step)
        n_sma_max = 2.0 / (1.0 - beta2) - 1.0
        n_sma = n_sma_max - 2.0 * step * beta2_t / (1.0 - beta2_t)
        bc1 = 1.0 - mx.power(beta1, step)
        rectified = n_sma > self.n_sma_threshold
        p1 = (n_sma - 4.0) / (n_sma_max - 4.0)
        p2 = (n_sma - 2.0) / n_sma
        p3 = n_sma_max / (n_sma_max - 2.0)
        radicand = mx.maximum((1.0 - beta2_t) * p1 * p2 * p3, 0.0)
        step_size = mx.where(rectified, mx.sqrt(radicand) / bc1, 1.0 / bc1)
        rate = learning_rate * step_size.astype(gradient.dtype)

        m = beta1 * state["m"] + (1.0 - beta1) * gradient
        v = beta2 * state["v"] + (1.0 - beta2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v
        value = mx.where(rectified, m / (mx.sqrt(v) + self.eps), m)
        fast = parameter * (1.0 - self.decay * rate) - rate * value

        slow = state["slow"]
        lookahead = mx.remainder(self.step, self.lookahead_k) == 0
        blended = self.alpha * fast + (1.0 - self.alpha) * slow
        state["slow"] = mx.where(lookahead, blended, slow)
        return mx.where(lookahead, blended, fast)


@dataclass(frozen=True, slots=True)
class MlxNnueBatch:
    """One native-decoded sparse NNUE batch."""

    stm: NDArray[np.int32]
    nstm: NDArray[np.int32]
    nnz: NDArray[np.uint8]
    buckets: NDArray[np.uint8]
    scores: NDArray[np.int16]

    @property
    def positions(self) -> int:
        return int(self.scores.shape[0])

    def mlx(self) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        return (
            mx.array(self.stm),
            mx.array(self.nstm),
            mx.array(self.nnz.astype(np.int32, copy=False)),
            mx.array(self.buckets.astype(np.int32, copy=False)),
            mx.array(self.scores.astype(np.float32, copy=False)),
        )


class MeteoValueNnue(nn.Module):
    """NAGISA-compatible value-only HalfKA_hm2 progress LayerStack."""

    def __init__(
        self,
        *,
        initial_weights: Mapping[str, mx.array] | None = None,
        seed: int = 0x4D4554454F,
    ) -> None:
        super().__init__()
        if initial_weights is not None:
            observed = set(initial_weights)
            if observed != set(MODEL_SHAPES):
                raise ValueError(
                    "exact Tatara initialization has wrong tensor names: "
                    f"missing={sorted(set(MODEL_SHAPES) - observed)} "
                    f"extra={sorted(observed - set(MODEL_SHAPES))}"
                )
            for name, shape in MODEL_SHAPES.items():
                value = initial_weights[name]
                if value.dtype != mx.float32 or tuple(value.shape) != shape:
                    raise ValueError(
                        f"exact Tatara initializer tensor {name} has "
                        f"{value.dtype}/{tuple(value.shape)}, expected float32/{shape}"
                    )
                setattr(self, name, value)
            return
        keys = mx.random.split(mx.random.key(seed), 5)
        ft_half_width = math.sqrt(1.0 / FT_IN)
        self.ft_real = mx.random.uniform(
            -ft_half_width,
            ft_half_width,
            shape=(FT_IN, FT_OUT),
            dtype=mx.float32,
            key=keys[0],
        )
        # Tatara samples only real FT rows and appends zero factorizer rows.
        # This fallback initializer keeps that distributional invariant; the
        # production path uses the native Tatara xorshift initializer below.
        self.ft_virtual = mx.zeros((PIECE_INPUTS, FT_OUT), dtype=mx.float32)
        self.ft_b = mx.zeros((FT_OUT,), dtype=mx.float32)
        self.l1_w = mx.random.uniform(
            -0.01,
            0.01,
            shape=(NUM_BUCKETS, L1_OUT, FT_OUT),
            dtype=mx.float32,
            key=keys[1],
        )
        self.l1_b = mx.zeros((NUM_BUCKETS, L1_OUT), dtype=mx.float32)
        self.l1f_w = mx.random.uniform(
            -0.01,
            0.01,
            shape=(FT_OUT, L1_OUT),
            dtype=mx.float32,
            key=keys[2],
        )
        self.l1f_b = mx.zeros((L1_OUT,), dtype=mx.float32)
        self.l2_w = mx.random.uniform(
            -0.01,
            0.01,
            shape=(NUM_BUCKETS, L2_OUT, L2_IN),
            dtype=mx.float32,
            key=keys[3],
        )
        self.l2_b = mx.zeros((NUM_BUCKETS, L2_OUT), dtype=mx.float32)
        self.l3_w = mx.random.uniform(
            -0.01,
            0.01,
            shape=(NUM_BUCKETS, L2_OUT),
            dtype=mx.float32,
            key=keys[4],
        )
        self.l3_b = mx.zeros((NUM_BUCKETS,), dtype=mx.float32)

    @staticmethod
    def _bucket_mm(inputs: mx.array, weights: mx.array, buckets: mx.array) -> mx.array:
        batch_indices = mx.arange(inputs.shape[0], dtype=mx.int32)
        return mx.gather_mm(
            inputs[:, None, :],
            mx.swapaxes(weights, -1, -2),
            batch_indices,
            buckets,
        )[:, 0, :]

    @staticmethod
    def _validate_inputs(
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
    ) -> None:
        if stm.ndim != 2 or stm.shape[1] != MAX_ACTIVE or nstm.shape != stm.shape:
            raise ValueError("HalfKA_hm2 indices must have shape [batch, 40]")
        if (
            nnz.ndim != 1
            or buckets.ndim != 1
            or nnz.shape[0] != stm.shape[0]
            or buckets.shape[0] != stm.shape[0]
        ):
            raise ValueError("NNZ and progress buckets must be one value per position")

    def _float_forward(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
        *,
        feature_dtype: mx.Dtype,
    ) -> mx.array:
        self._validate_inputs(stm, nstm, nnz, buckets)
        effective_ft = (self.ft_real + mx.tile(self.ft_virtual, (45, 1))).astype(feature_dtype)
        mask = (mx.arange(MAX_ACTIVE, dtype=mx.int32)[None, :] < nnz[:, None]).astype(feature_dtype)
        safe_stm = mx.maximum(stm, 0)
        safe_nstm = mx.maximum(nstm, 0)
        stm_ft = mx.sum(mx.take(effective_ft, safe_stm, axis=0) * mask[..., None], axis=1).astype(
            mx.float32
        )
        nstm_ft = mx.sum(mx.take(effective_ft, safe_nstm, axis=0) * mask[..., None], axis=1).astype(
            mx.float32
        )

        def post_perspective(values: mx.array) -> mx.array:
            activated = mx.clip(values + self.ft_b, 0.0, 1.0)
            return activated[:, : FT_OUT // 2] * activated[:, FT_OUT // 2 :] * PAIRWISE_SCALE

        combined = mx.concatenate((post_perspective(stm_ft), post_perspective(nstm_ft)), axis=1)
        l1_bucket = self._bucket_mm(combined, self.l1_w, buckets) + mx.take(
            self.l1_b, buckets, axis=0
        )
        l1_shared = combined @ self.l1f_w + self.l1f_b
        l1_total = l1_bucket + l1_shared
        l1_main = l1_total[:, :L1_EFFECTIVE]
        l1_skip = l1_total[:, L1_EFFECTIVE]
        l2_input = mx.clip(
            mx.concatenate((mx.square(mx.abs(l1_main)) * PAIRWISE_SCALE, l1_main), axis=1),
            0.0,
            1.0,
        )
        l2 = self._bucket_mm(l2_input, self.l2_w, buckets) + mx.take(self.l2_b, buckets, axis=0)
        l2 = mx.clip(l2, 0.0, 1.0)
        l3 = self._bucket_mm(l2, self.l3_w[:, None, :], buckets)[:, 0]
        return l3 + mx.take(self.l3_b, buckets, axis=0) + l1_skip

    def mixed_precision_forward(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
    ) -> mx.array:
        """Fast warm-up: FP16 sparse feature gather, FP32 dense layers."""

        return self._float_forward(stm, nstm, nnz, buckets, feature_dtype=mx.float16)

    def _quantised_forward(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
        *,
        exact_sparse_accumulation: bool,
    ) -> mx.array:
        """Integer-equivalent deployment graph with straight-through gradients.

        Forward values reproduce Tatara/YaneuraOu's QA=127, QB=64 network,
        including factorizer folding, right shifts, and saturating casts.  The
        training path uses a fused Metal int32 sparse accumulator with an exact
        custom scatter VJP.  The independent reference path uses ordinary MLX
        FP32 operations and is checked against the native Rust evaluator before
        a generation is accepted.
        """

        self._validate_inputs(stm, nstm, nnz, buckets)
        effective_ft = self.ft_real + mx.tile(self.ft_virtual, (45, 1))
        exact_ft_raw = fake_quantise_rust_f32_raw(
            effective_ft,
            scale=QUANTISATION_QA,
            minimum=-32_768.0,
            maximum=32_767.0,
        )
        ft_bias_raw = fake_quantise_rust_f32_raw(
            self.ft_b,
            scale=QUANTISATION_QA,
            minimum=-32_768.0,
            maximum=32_767.0,
        )
        mask = (mx.arange(MAX_ACTIVE, dtype=mx.int32)[None, :] < nnz[:, None]).astype(mx.float32)
        safe_stm = mx.maximum(stm, 0)
        safe_nstm = mx.maximum(nstm, 0)

        def accumulate(indices: mx.array) -> mx.array:
            if exact_sparse_accumulation:
                return _exact_sparse_ft_accumulation(exact_ft_raw, indices, nnz)
            gathered = mx.take(exact_ft_raw, indices, axis=0) * mask[..., None]
            return mx.sum(gathered, axis=1)

        stm_raw = accumulate(safe_stm) + ft_bias_raw
        nstm_raw = accumulate(safe_nstm) + ft_bias_raw

        def pairwise_crelu(values: mx.array) -> mx.array:
            activated = mx.clip(values, 0.0, QUANTISATION_QA)
            paired = activated[:, : FT_OUT // 2] * activated[:, FT_OUT // 2 :]
            return mx.clip(ste_floor(paired / 128.0), 0.0, 126.0)

        combined_raw = mx.concatenate((pairwise_crelu(stm_raw), pairwise_crelu(nstm_raw)), axis=1)
        # Tatara folds the shared L1 factorizer into every bucket before the
        # *single* i8 rounding operation.  Quantising both branches separately
        # is measurably wrong around half-unit boundaries.
        merged_l1_w = self.l1_w + mx.swapaxes(self.l1f_w, 0, 1)[None, :, :]
        merged_l1_b = self.l1_b + self.l1f_b[None, :]
        l1_w_raw = fake_quantise_rust_f32_raw(
            merged_l1_w,
            scale=QUANTISATION_QB,
            minimum=-128.0,
            maximum=127.0,
        )
        l1_b_raw = fake_quantise_rust_f32_raw(merged_l1_b, scale=QUANTISATION_BIAS_SCALE)
        l1_total_raw = self._bucket_mm(combined_raw, l1_w_raw, buckets) + mx.take(
            l1_b_raw, buckets, axis=0
        )
        l1_main_raw = l1_total_raw[:, :L1_EFFECTIVE]
        l1_skip_raw = l1_total_raw[:, L1_EFFECTIVE]
        squared_raw = mx.clip(ste_floor(mx.square(l1_main_raw) / float(1 << 19)), 0.0, 127.0)
        crelu_raw = mx.clip(ste_floor(l1_main_raw / 64.0), 0.0, 127.0)
        l2_input_raw = mx.concatenate((squared_raw, crelu_raw), axis=1)

        l2_w_raw = fake_quantise_rust_f32_raw(
            self.l2_w,
            scale=QUANTISATION_QB,
            minimum=-128.0,
            maximum=127.0,
        )
        l2_b_raw = fake_quantise_rust_f32_raw(self.l2_b, scale=QUANTISATION_BIAS_SCALE)
        l2_dense_raw = self._bucket_mm(l2_input_raw, l2_w_raw, buckets) + mx.take(
            l2_b_raw, buckets, axis=0
        )
        l2_raw = mx.clip(ste_floor(l2_dense_raw / 64.0), 0.0, 127.0)

        l3_w_raw = fake_quantise_rust_f32_raw(
            self.l3_w,
            scale=QUANTISATION_QB,
            minimum=-128.0,
            maximum=127.0,
        )
        l3_b_raw = fake_quantise_rust_f32_raw(self.l3_b, scale=QUANTISATION_BIAS_SCALE)
        output_raw = self._bucket_mm(l2_raw, l3_w_raw[:, None, :], buckets)[:, 0]
        output_raw = output_raw + mx.take(l3_b_raw, buckets, axis=0) + l1_skip_raw
        return output_raw / QUANTISATION_BIAS_SCALE

    def quantisation_aware_forward(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
    ) -> mx.array:
        """QAT graph optimized for Apple unified-memory bandwidth."""

        return self._quantised_forward(
            stm,
            nstm,
            nnz,
            buckets,
            exact_sparse_accumulation=True,
        )

    def quantised_reference_forward(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
    ) -> mx.array:
        """High-fidelity QAT graph used to audit the native exported network."""

        return self._quantised_forward(
            stm,
            nstm,
            nnz,
            buckets,
            exact_sparse_accumulation=False,
        )

    def __call__(
        self,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
    ) -> mx.array:
        return self._float_forward(stm, nstm, nnz, buckets, feature_dtype=mx.float32)


def _model_outputs(
    model: MeteoValueNnue,
    stm: mx.array,
    nstm: mx.array,
    nnz: mx.array,
    buckets: mx.array,
    *,
    forward_mode: ModelForwardMode,
) -> mx.array:
    if forward_mode == "float32":
        return model(stm, nstm, nnz, buckets)
    if forward_mode == "mixed_precision_warmup":
        return model.mixed_precision_forward(stm, nstm, nnz, buckets)
    if forward_mode == "quantisation_aware":
        return model.quantisation_aware_forward(stm, nstm, nnz, buckets)
    if forward_mode == "quantised_reference":
        return model.quantised_reference_forward(stm, nstm, nnz, buckets)
    raise ValueError(f"unknown model forward mode: {forward_mode}")


def wrm_probabilities(
    outputs: mx.array,
    scores: mx.array,
    *,
    parameters: WrmLossParameters,
) -> tuple[mx.array, mx.array]:
    """Apply Tatara's non-extended WRM prediction and target transforms."""

    score_net = outputs * parameters.nnue2score
    q = mx.sigmoid((score_net - parameters.in_offset) / parameters.in_scaling)
    qm = mx.sigmoid((-score_net - parameters.in_offset) / parameters.in_scaling)
    predictions = 0.5 * (1.0 + q - qm)
    pt = (scores - parameters.target_offset) / parameters.target_scaling
    pmt = (-scores - parameters.target_offset) / parameters.target_scaling
    target_wrm = 0.5 * (1.0 + mx.sigmoid(pt) - mx.sigmoid(pmt))
    if parameters.wdl_lambda != 0.0:
        raise ValueError("value-only PSV has no WDL target available for a nonzero lambda")
    return predictions, target_wrm


def value_loss(
    model: MeteoValueNnue,
    stm: mx.array,
    nstm: mx.array,
    nnz: mx.array,
    buckets: mx.array,
    scores: mx.array,
    *,
    parameters: WrmLossParameters,
    forward_mode: ModelForwardMode = "float32",
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Tatara-compatible value-only WRM MSE (WDL lambda is zero)."""

    outputs = _model_outputs(
        model,
        stm,
        nstm,
        nnz,
        buckets,
        forward_mode=forward_mode,
    )
    predictions, targets = wrm_probabilities(outputs, scores, parameters=parameters)
    loss = mx.mean(mx.square(predictions - targets))
    return loss, (mx.mean(outputs), mx.mean(targets))


def _read_exact(stream: Any, size: int) -> bytes:
    output = bytearray(size)
    view = memoryview(output)
    offset = 0
    while offset < size:
        read = stream.readinto(view[offset:])
        if read is None or read == 0:
            raise EOFError(f"native MLX stream ended after {offset} of {size} bytes")
        offset += read
    return bytes(output)


class NativeBatchStream(Iterator[MlxNnueBatch]):
    """Bounded asynchronous reader for the Rust feature-stream protocol."""

    def __init__(
        self,
        binary: Path,
        *,
        psv: Path,
        progress: Path,
        start_record: int,
        records: int,
        batch_size: int,
        threads: int,
        prefetch: int = 2,
    ) -> None:
        command = [
            str(binary),
            "stream",
            "--input",
            str(psv),
            "--progress",
            str(progress),
            "--start-record",
            str(start_record),
            "--records",
            str(records),
            "--batch-size",
            str(batch_size),
            "--threads",
            str(threads),
        ]
        self.command = tuple(command)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._queue: queue.Queue[MlxNnueBatch | BaseException | None] = queue.Queue(
            maxsize=max(1, prefetch)
        )
        self._closed = False
        self._thread = threading.Thread(
            target=self._read_loop,
            name="meteo-mlx-native-reader",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            stdout = self._process.stdout
            if stdout is None:
                raise RuntimeError("native feature process has no stdout pipe")
            header = _read_exact(stdout, 16)
            if header[:8] != NATIVE_PROTOCOL_MAGIC:
                raise ValueError(f"native feature protocol magic mismatch: {header[:8]!r}")
            version, active = struct.unpack("<II", header[8:])
            if version != NATIVE_PROTOCOL_VERSION or active != MAX_ACTIVE:
                raise ValueError(
                    f"native feature protocol mismatch: version={version} active={active}"
                )
            while True:
                count = struct.unpack("<I", _read_exact(stdout, 4))[0]
                if count == 0:
                    break
                stm = np.frombuffer(_read_exact(stdout, count * MAX_ACTIVE * 4), dtype="<i4")
                nstm = np.frombuffer(_read_exact(stdout, count * MAX_ACTIVE * 4), dtype="<i4")
                nnz = np.frombuffer(_read_exact(stdout, count), dtype=np.uint8)
                buckets = np.frombuffer(_read_exact(stdout, count), dtype=np.uint8)
                scores = np.frombuffer(_read_exact(stdout, count * 2), dtype="<i2")
                self._queue.put(
                    MlxNnueBatch(
                        stm=stm.reshape(count, MAX_ACTIVE),
                        nstm=nstm.reshape(count, MAX_ACTIVE),
                        nnz=nnz,
                        buckets=buckets,
                        scores=scores,
                    )
                )
            returncode = self._process.wait()
            stderr = b"" if self._process.stderr is None else self._process.stderr.read()
            if returncode:
                raise RuntimeError(
                    f"native feature stream exited {returncode}: "
                    f"{stderr.decode(errors='replace').strip()}"
                )
            self._queue.put(None)
        except BaseException as error:
            self._queue.put(error)

    def __iter__(self) -> NativeBatchStream:
        return self

    def __next__(self) -> MlxNnueBatch:
        item = self._queue.get()
        if item is None:
            self.close()
            raise StopIteration
        if isinstance(item, BaseException):
            self.close()
            raise item
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._thread.join(timeout=5)

    def __enter__(self) -> NativeBatchStream:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _quantised_raw_values(
    native_binary: Path,
    *,
    network: Path,
    psv: Path,
    progress: Path,
    positions: int,
    batch_size: int,
    threads: int,
) -> NDArray[np.int32]:
    command = [
        str(native_binary.resolve(strict=True)),
        "evaluate-quantised",
        "--network",
        str(network.resolve(strict=True)),
        "--input",
        str(psv.resolve(strict=True)),
        "--progress",
        str(progress.resolve(strict=True)),
        "--start-record",
        "0",
        "--records",
        str(positions),
        "--batch-size",
        str(batch_size),
        "--threads",
        str(threads),
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "quantised NNUE evaluation failed "
            f"({completed.returncode}): {completed.stderr[-4000:].decode(errors='replace')}"
        )
    payload = completed.stdout
    if len(payload) < 16 or payload[:8] != QUANTIZED_PROTOCOL_MAGIC:
        raise ValueError("quantised NNUE evaluator returned an invalid protocol header")
    version = struct.unpack_from("<I", payload, 8)[0]
    if version != NATIVE_PROTOCOL_VERSION:
        raise ValueError(f"quantised NNUE protocol version mismatch: {version}")
    offset = 12
    arrays: list[NDArray[np.int32]] = []
    observed = 0
    while True:
        if offset + 4 > len(payload):
            raise ValueError("quantised NNUE protocol ended before a batch count")
        count = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if count == 0:
            break
        byte_count = count * 4
        if offset + byte_count > len(payload):
            raise ValueError("quantised NNUE protocol ended inside a raw-output batch")
        arrays.append(np.frombuffer(payload, dtype="<i4", count=count, offset=offset).copy())
        observed += count
        offset += byte_count
    if offset != len(payload) or observed != positions:
        raise ValueError(
            f"quantised NNUE protocol count/trailer mismatch: {observed}/{positions}, "
            f"trailing={len(payload) - offset}"
        )
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)


def _copy_native_sources(template: Path, destination: Path) -> None:
    source_files = (Path("Cargo.toml"), Path("src/main.rs"))
    if destination.exists():
        for relative in source_files:
            expected = template / relative
            observed = destination / relative
            if not observed.is_file() or _sha256_file(observed) != _sha256_file(expected):
                raise ValueError(f"MLX native source collision at {observed}")
        return
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (stage / "src").mkdir()
        for relative in source_files:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template / relative, target)
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build_mlx_native(run_directory: Path) -> dict[str, object]:
    """Build and hash the CPU-only Tatara companion without CUDA or cloud use."""

    root = run_directory.expanduser().resolve(strict=True)
    template = Path(__file__).resolve().with_name("mlx_native")
    tatara = root / "dependencies" / "tatara"
    if not template.is_dir() or not tatara.is_dir() or tatara.is_symlink():
        raise ValueError("MLX native template or pinned Tatara checkout is missing")
    destination = root / "dependencies" / "meteo_mlx_native"
    _copy_native_sources(template, destination)
    receipt_path = root / "receipts" / "mlx-native-build.json"
    binary = destination / "target" / "release" / "meteo-mlx-native"
    if receipt_path.is_file() and binary.is_file() and not binary.is_symlink():
        existing_receipt = _strict_json(receipt_path.read_bytes(), label="MLX native build receipt")
        identity = cast(dict[str, Any], existing_receipt.get("binary"))
        source = existing_receipt.get("source")
        expected_source = {
            str(relative): _sha256_file(template / relative)
            for relative in (Path("Cargo.toml"), Path("src/main.rs"))
        }
        if (
            identity.get("sha256") == _sha256_file(binary)
            and isinstance(source, dict)
            and source == expected_source
        ):
            return existing_receipt
        raise ValueError("MLX native binary/source no longer matches its immutable build receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("partial MLX native build receipt requires operator inspection")
    cargo_lock = destination / "Cargo.lock"
    commands: list[list[str]] = []
    if not cargo_lock.exists():
        commands.append(
            [
                "cargo",
                "generate-lockfile",
                "--manifest-path",
                str(destination / "Cargo.toml"),
                "--ignore-rust-version",
            ]
        )
    commands.append(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--ignore-rust-version",
            "--manifest-path",
            str(destination / "Cargo.toml"),
        ]
    )
    command_receipts: list[dict[str, object]] = []
    for command in commands:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        command_receipts.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4_000:],
                "stderr_tail": completed.stderr[-4_000:],
            }
        )
        if completed.returncode:
            raise RuntimeError(
                f"MLX native build failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
    if not binary.is_file() or binary.is_symlink():
        raise FileNotFoundError(f"MLX native build did not produce {binary}")
    os.chmod(binary, 0o700)
    _fsync_file(binary)
    rustc = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True)
    cargo = subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True)
    build_receipt: dict[str, object] = {
        "schema": NATIVE_BUILD_SCHEMA,
        "created_unix": time.time(),
        "cloud_gpu_used": False,
        "tatara_commit": cast(dict[str, Any], load_nagisa_plan(root)["tatara"])["commit"],
        "source": {
            str(relative): _sha256_file(template / relative)
            for relative in (Path("Cargo.toml"), Path("src/main.rs"))
        },
        "cargo_lock_sha256": _sha256_file(cargo_lock),
        "rustc": rustc.stdout.strip(),
        "cargo": cargo.stdout.strip(),
        "commands": command_receipts,
        "binary": {
            "path": str(binary),
            "bytes": binary.stat().st_size,
            "sha256": _sha256_file(binary),
        },
    }
    _write_new(receipt_path, _json_bytes(build_receipt), mode=0o600)
    return build_receipt


def initialise_exact_tatara_model(
    native_binary: Path,
    *,
    temporary_parent: Path,
) -> tuple[MeteoValueNnue, dict[str, object]]:
    """Create the exact Tatara default LayerStack initialization on MLX.

    The native companion calls Tatara's own ``LayerStackInit`` and xorshift
    sampler.  In particular, the factorizer rows and every bias are exactly
    zero at step 0; this avoids the variance shift caused by independently
    sampling virtual rows.
    """

    binary = native_binary.expanduser().resolve(strict=True)
    parent = temporary_parent.expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix=".meteo-tatara-init-", dir=parent) as raw_tmp:
        initializer = Path(raw_tmp) / "initial.safetensors"
        command = [str(binary), "initialise", "--output", str(initializer)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                "exact Tatara initialization failed "
                f"({completed.returncode}): {completed.stderr[-4000:]}"
            )
        if not initializer.is_file() or initializer.is_symlink():
            raise FileNotFoundError("native Tatara initializer did not publish its output")
        initializer_sha256 = _sha256_file(initializer)
        raw = mx.load(initializer)
        if not isinstance(raw, dict):
            raise TypeError("Tatara initializer did not load as a tensor mapping")
        weights = cast(dict[str, mx.array], raw)
        model = MeteoValueNnue(initial_weights=weights)
        mx.eval(model.parameters())
        arrays = _model_arrays(model)
        statistics: dict[str, dict[str, float | int]] = {}
        for name, value in arrays.items():
            minimum = mx.min(value)
            maximum = mx.max(value)
            mean = mx.mean(value)
            mean_square = mx.mean(mx.square(value))
            zero_count = mx.sum(value == 0.0)
            mx.eval(minimum, maximum, mean, mean_square, zero_count)
            mean_value = float(mean.item())
            statistics[name] = {
                "elements": int(value.size),
                "minimum": float(minimum.item()),
                "maximum": float(maximum.item()),
                "mean": mean_value,
                "standard_deviation": math.sqrt(
                    max(0.0, float(mean_square.item()) - mean_value * mean_value)
                ),
                "zero_elements": int(zero_count.item()),
            }
        for name in ("ft_virtual", "ft_b", "l1_b", "l1f_b", "l2_b", "l3_b"):
            if statistics[name]["zero_elements"] != statistics[name]["elements"]:
                raise ValueError(f"Tatara step-0 tensor {name} is not exactly zero")
        ft_limit = math.sqrt(1.0 / FT_IN)
        if (
            statistics["ft_real"]["minimum"] < -ft_limit
            or statistics["ft_real"]["maximum"] > ft_limit
        ):
            raise ValueError("Tatara FT initialization escaped its exact fan-in bounds")
        for name in ("l1_w", "l1f_w", "l2_w", "l3_w"):
            if statistics[name]["minimum"] < -0.01 or statistics[name]["maximum"] > 0.01:
                raise ValueError(f"Tatara dense initializer {name} escaped [-0.01, 0.01]")
        receipt: dict[str, object] = {
            "schema": "meteo-nagisa-mlx-tatara-initialization-v1",
            "created_unix": time.time(),
            "generator": "pinned_tatara_nnue_train_init_layerstack_default_uniform",
            "factorizer_virtual_rows": "exact_zero_append",
            "initializer_sha256": initializer_sha256,
            "native_binary_sha256": _sha256_file(binary),
            "statistics": statistics,
        }
        return model, receipt


def _tensor_signature(tree: Mapping[str, object]) -> dict[str, dict[str, object]]:
    signature: dict[str, dict[str, object]] = {}
    for name, value in cast(list[tuple[str, object]], tree_flatten(tree)):
        if name in signature:
            raise ValueError(f"duplicate tensor name: {name}")
        if not isinstance(value, mx.array):
            raise TypeError(f"checkpoint tree has a non-MLX leaf: {name}")
        signature[name] = {
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
        }
    return signature


def _model_arrays(model: MeteoValueNnue) -> dict[str, mx.array]:
    arrays = dict(cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())))
    if tuple(sorted(arrays)) != tuple(sorted(MODEL_KEYS)):
        raise ValueError(
            f"MLX model parameter contract changed: expected={sorted(MODEL_KEYS)} "
            f"observed={sorted(arrays)}"
        )
    return arrays


def _optimizer_arrays(optimizer: TataraRanger) -> dict[str, mx.array]:
    arrays = dict(cast(list[tuple[str, mx.array]], tree_flatten(optimizer.state)))
    if not arrays or "step" not in arrays or "learning_rate" not in arrays:
        raise ValueError("Ranger optimizer state is incomplete")
    return arrays


def _new_optimizer(model: MeteoValueNnue, *, learning_rate: float) -> TataraRanger:
    optimizer = TataraRanger(learning_rate, decay=0.0)
    optimizer.init(model.trainable_parameters())
    mx.eval(optimizer.state)
    return optimizer


def _set_dense_clamps(model: MeteoValueNnue) -> None:
    updated: list[tuple[str, mx.array]] = []
    for name, value in cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())):
        updated.append(
            (name, mx.clip(value, -DENSE_CLAMP, DENSE_CLAMP))
            if name in CLAMPED_MODEL_KEYS
            else (name, value)
        )
    model.update(cast(dict[str, Any], tree_unflatten(updated)))


def _compiled_training_step(
    model: MeteoValueNnue,
    optimizer: TataraRanger,
    *,
    parameters: WrmLossParameters,
    forward_mode: NumericPhase,
) -> Any:
    def loss_fn(
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
        scores: mx.array,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        return value_loss(
            model,
            stm,
            nstm,
            nnz,
            buckets,
            scores,
            parameters=parameters,
            forward_mode=forward_mode,
        )

    value_and_grad = nn.value_and_grad(model, loss_fn)
    compiled_state = [model.state, optimizer.state]

    def optimization_step(
        learning_rate: mx.array,
        stm: mx.array,
        nstm: mx.array,
        nnz: mx.array,
        buckets: mx.array,
        scores: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        optimizer.state["learning_rate"] = learning_rate
        (loss, (output_mean, target_mean)), gradients = value_and_grad(
            stm, nstm, nnz, buckets, scores
        )
        optimizer.update(model, gradients)
        _set_dense_clamps(model)
        return loss, output_mean, target_mean

    return mx.compile(
        optimization_step,
        inputs=compiled_state,
        outputs=compiled_state,
    )


def _evaluate_mlx_batch(
    model: MeteoValueNnue,
    batch: MlxNnueBatch,
    *,
    parameters: WrmLossParameters,
    forward_mode: ModelForwardMode = "float32",
) -> tuple[float, float, float]:
    loss, (output_mean, target_mean) = value_loss(
        model,
        *batch.mlx(),
        parameters=parameters,
        forward_mode=forward_mode,
    )
    mx.eval(loss, output_mean, target_mean)
    return float(loss.item()), float(output_mean.item()), float(target_mean.item())


def _save_safetensors(
    path: Path,
    arrays: Mapping[str, mx.array],
    *,
    metadata: Mapping[str, str],
) -> None:
    mx.eval(arrays)
    mx.save_safetensors(path, dict(arrays), metadata=dict(metadata))
    os.chmod(path, 0o600)
    _fsync_file(path)


def save_mlx_checkpoint(
    run_directory: Path,
    *,
    model: MeteoValueNnue,
    optimizer: TataraRanger,
    optimizer_step: int,
    superbatch: int,
    completed_batches_in_superbatch: int,
    presentations: int,
    parent: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Publish model + every Ranger state tensor as one atomic generation."""

    root = run_directory.expanduser().resolve(strict=True)
    destination_parent = (root / "checkpoints") if parent is None else parent.resolve(strict=True)
    destination = destination_parent / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite MLX checkpoint: {destination}")
    model_arrays = _model_arrays(model)
    optimizer_arrays = _optimizer_arrays(optimizer)
    observed_step = int(optimizer.state["step"].item())
    if observed_step != optimizer_step:
        raise ValueError(
            f"optimizer step mismatch: expected={optimizer_step} observed={observed_step}"
        )
    mlx_plan = load_mlx_plan(root)
    numeric_identity: dict[str, object]
    if mlx_plan.get("schema") == MLX_BACKEND_PLAN_SCHEMA:
        numeric_schedule = _quantisation_schedule_from_plan(mlx_plan)
        numeric_identity = {
            "next_update_phase": numeric_schedule.phase_for_completed_steps(optimizer_step),
            "warmup_optimizer_steps": numeric_schedule.warmup_optimizer_steps,
            "deployment_graph": cast(dict[str, Any], mlx_plan["numerics"])["deployment_graph"],
            "feature_accumulation_implementation": (
                effective_mlx_qat_feature_accumulation(root, mlx_plan)
            ),
            "export_rounding_implementation": effective_mlx_export_rounding(root, mlx_plan),
        }
    else:
        numeric_identity = {"next_update_phase": "legacy_float32"}
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination_parent))
    try:
        metadata = {
            "schema": MLX_CHECKPOINT_SCHEMA,
            "optimizer_step": str(optimizer_step),
            "superbatch": str(superbatch),
            "presentations": str(presentations),
        }
        model_file = stage / "model.safetensors"
        optimizer_file = stage / "optimizer.safetensors"
        _save_safetensors(model_file, model_arrays, metadata=metadata)
        _save_safetensors(optimizer_file, optimizer_arrays, metadata=metadata)
        manifest: dict[str, object] = {
            "schema": MLX_CHECKPOINT_SCHEMA,
            "complete": True,
            "created_unix": time.time(),
            "backend": "mlx",
            "model": NagisaNnueArchitecture().to_dict(),
            "optimizer": {
                "name": "tatara_ranger_mlx_port",
                "step": optimizer_step,
                "beta1": optimizer.beta1,
                "beta2": optimizer.beta2,
                "eps": optimizer.eps,
                "alpha": optimizer.alpha,
                "lookahead_k": optimizer.lookahead_k,
                "n_sma_threshold": optimizer.n_sma_threshold,
                "decay": optimizer.decay,
            },
            "coordinate": {
                "superbatch": superbatch,
                "completed_batches_in_superbatch": completed_batches_in_superbatch,
                "presentations": presentations,
            },
            "numerics": numeric_identity,
            "model_file": {
                "name": model_file.name,
                "bytes": model_file.stat().st_size,
                "sha256": _sha256_file(model_file),
                "signature": _tensor_signature(model.parameters()),
            },
            "optimizer_file": {
                "name": optimizer_file.name,
                "bytes": optimizer_file.stat().st_size,
                "sha256": _sha256_file(optimizer_file),
                "signature": _tensor_signature(optimizer.state),
            },
            "source_plan_sha256": _sha256_file(root / "plan.json"),
            "mlx_plan_sha256": _sha256_file(root / "mlx-plan.json"),
        }
        _write_new(stage / "manifest.json", _json_bytes(manifest), mode=0o600)
        _fsync_directory(stage)
        os.rename(stage, destination)
        _fsync_directory(destination_parent)
        return destination, manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_mlx_checkpoint(
    checkpoint: Path,
    *,
    learning_rate: float,
) -> tuple[MeteoValueNnue, TataraRanger, dict[str, object]]:
    requested = checkpoint.expanduser()
    if requested.is_symlink():
        raise ValueError("MLX checkpoint must be a regular directory")
    source = requested.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("MLX checkpoint must be a regular directory")
    manifest = _strict_json((source / "manifest.json").read_bytes(), label="MLX manifest")
    if manifest.get("schema") != MLX_CHECKPOINT_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("MLX checkpoint has no valid complete marker")
    for key in ("model_file", "optimizer_file"):
        identity = cast(dict[str, Any], manifest[key])
        path = source / cast(str, identity["name"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"MLX checkpoint artifact is not a regular file: {path}")
        if path.stat().st_size != int(cast(int, identity["bytes"])):
            raise ValueError(f"MLX checkpoint size mismatch: {path}")
        if _sha256_file(path) != cast(str, identity["sha256"]):
            raise ValueError(f"MLX checkpoint hash mismatch: {path}")
    raw_model = mx.load(source / cast(str, cast(dict[str, Any], manifest["model_file"])["name"]))
    if not isinstance(raw_model, dict):
        raise TypeError("MLX model safetensors did not load as a tensor mapping")
    model_weights = cast(dict[str, mx.array], raw_model)
    if tuple(sorted(model_weights)) != tuple(sorted(MODEL_KEYS)):
        raise ValueError("MLX model checkpoint parameter names do not match the architecture")
    model = MeteoValueNnue(initial_weights=model_weights)
    optimizer = _new_optimizer(model, learning_rate=learning_rate)
    raw_optimizer = mx.load(
        source / cast(str, cast(dict[str, Any], manifest["optimizer_file"])["name"])
    )
    if not isinstance(raw_optimizer, dict):
        raise TypeError("MLX optimizer safetensors did not load as a tensor mapping")
    restored_state = cast(dict[str, Any], tree_unflatten(cast(dict[str, mx.array], raw_optimizer)))
    if _tensor_signature(restored_state) != cast(
        dict[str, dict[str, object]], cast(dict[str, Any], manifest["optimizer_file"])["signature"]
    ):
        raise ValueError("MLX optimizer tensor signature changed during reload")
    optimizer.state = restored_state
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)
    expected_step = int(cast(dict[str, Any], manifest["optimizer"])["step"])
    if int(optimizer.state["step"].item()) != expected_step:
        raise ValueError("MLX Ranger global step did not survive checkpoint reload")
    return model, optimizer, manifest


def _prune_mlx_checkpoints(directory: Path, *, keep: int = 2) -> tuple[Path, ...]:
    if keep < 1:
        raise ValueError("MLX checkpoint retention must be positive")
    candidates: list[tuple[int, Path]] = []
    prefix = f"{DEFAULT_NET_ID}-mlx-step-"
    for path in directory.iterdir():
        if not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if not suffix.isdigit():
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"MLX checkpoint generation must be a regular directory: {path}")
        candidates.append((int(suffix), path))
    candidates.sort(reverse=True)
    for _, path in candidates[keep:]:
        shutil.rmtree(path)
    _fsync_directory(directory)
    return tuple(path for _, path in candidates[:keep])


def _prune_mlx_exports(directory: Path, *, keep: int = 2) -> tuple[Path, ...]:
    prefix = f"{DEFAULT_NET_ID}-mlx-step-"
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        suffix = path.name.removeprefix(prefix)
        if path.name.startswith(prefix) and suffix.isdigit():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"MLX export generation must be a regular directory: {path}")
            candidates.append((int(suffix), path))
    candidates.sort(reverse=True)
    for _, path in candidates[keep:]:
        shutil.rmtree(path)
    _fsync_directory(directory)
    return tuple(path for _, path in candidates[:keep])


def export_mlx_generation(
    run_directory: Path,
    *,
    checkpoint: Path,
    optimizer_step: int,
    native_binary: Path,
    parent: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    root = run_directory.expanduser().resolve(strict=True)
    destination_parent = root / "exports" if parent is None else parent.resolve(strict=True)
    destination = destination_parent / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite MLX NNUE export: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model_file = checkpoint.resolve(strict=True) / "model.safetensors"
        tatara_bin = stage / "network.bin"
        nn_bin = stage / "nn.bin"
        plan = load_nagisa_plan(root)
        training = cast(dict[str, Any], plan["training"])
        fv_scale = int(cast(int, training["yaneuraou_fv_scale"]))
        command = [
            str(native_binary.resolve(strict=True)),
            "export",
            "--model",
            str(model_file),
            "--tatara-output",
            str(tatara_bin),
            "--yaneuraou-output",
            str(nn_bin),
            "--fv-scale",
            str(fv_scale),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"MLX NNUE export failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
        if "saturating" in completed.stderr.lower():
            raise ValueError(
                f"MLX NNUE export reported quantisation saturation: {completed.stderr[-4000:]}"
            )
        for artifact in (tatara_bin, nn_bin):
            if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size < 1:
                raise ValueError(f"MLX native exporter omitted {artifact}")
            os.chmod(artifact, 0o600)
            _fsync_file(artifact)
        architecture = _yaneuraou_architecture(nn_bin)
        expected = NagisaNnueArchitecture().yaneuraou_header
        if architecture != expected:
            raise ValueError(
                "MLX YaneuraOu architecture mismatch: "
                f"expected={expected!r} observed={architecture!r}"
            )
        progress = root / "private" / "progress.bin"
        shutil.copyfile(progress, stage / "progress.bin")
        os.chmod(stage / "progress.bin", 0o600)
        _fsync_file(stage / "progress.bin")
        _write_new(
            stage / "eval_options.txt",
            (
                "LS_BUCKET_MODE progress8kpabs\n"
                "LS_PROGRESS_COEFF progress.bin\n"
                f"FV_SCALE {fv_scale}\n"
            ).encode(),
            mode=0o600,
        )
        receipt: dict[str, object] = {
            "schema": "meteo-nagisa-mlx-yaneuraou-export-v1",
            "created_unix": time.time(),
            "optimizer_step": optimizer_step,
            "checkpoint": str(checkpoint),
            "checkpoint_model_sha256": _sha256_file(model_file),
            "architecture": architecture,
            "routing": "progress8kpabs",
            "score_scale": float(cast(float, training["score_scale"])),
            "loss": training["loss"],
            "yaneuraou_fv_scale": fv_scale,
            "effective_cp_per_float_output": float(127 * 64) / fv_scale,
            "systematic_export_scale_ratio": 1.0,
            "factorizer_fold": "exact_virtual_piece_row_addition",
            "quantisation_saturation_reported": False,
            "exporter_stderr_tail": completed.stderr[-4_000:],
            "tatara_bin": {
                "bytes": tatara_bin.stat().st_size,
                "sha256": _sha256_file(tatara_bin),
            },
            "nn_bin": {
                "bytes": nn_bin.stat().st_size,
                "sha256": _sha256_file(nn_bin),
            },
            "progress_bin_sha256": _sha256_file(stage / "progress.bin"),
            "local_only": True,
        }
        _write_new(stage / "receipt.json", _json_bytes(receipt), mode=0o600)
        _fsync_directory(stage)
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
        return destination, receipt
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_mlx_plan(run_directory: Path) -> dict[str, Any]:
    root = run_directory.expanduser().resolve(strict=True)
    path = root / "mlx-plan.json"
    value = _strict_json(path.read_bytes(), label="MLX backend plan")
    schema = value.get("schema")
    if schema != MLX_BACKEND_PLAN_SCHEMA and schema not in LEGACY_MLX_BACKEND_PLAN_SCHEMAS:
        raise ValueError("MLX backend plan schema mismatch")
    if value.get("source_plan_sha256") != _sha256_file(root / "plan.json"):
        raise ValueError("MLX backend plan no longer matches the immutable source plan")
    return value


def _qat_feature_accumulation_contract(plan: Mapping[str, object]) -> str:
    numerics = plan.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("QAT MLX plan has no numeric training contract")
    contract = numerics.get("qat_feature_accumulation")
    if not isinstance(contract, str):
        raise ValueError("QAT MLX plan has no feature accumulation contract")
    return contract


def _load_mlx_numeric_repair_receipt(root: Path) -> dict[str, Any]:
    path = root / "receipts" / "mlx-numeric-repair-v1.json"
    value = _strict_json(path.read_bytes(), label="MLX numeric repair receipt")
    if (
        value.get("schema") != MLX_NUMERIC_REPAIR_SCHEMA
        or value.get("complete") is not True
        or value.get("source_mlx_plan_sha256") != _sha256_file(root / "mlx-plan.json")
        or value.get("from_feature_accumulation") != LEGACY_QAT_FEATURE_ACCUMULATION
        or value.get("to_feature_accumulation") != QAT_FEATURE_ACCUMULATION
        or value.get("model_or_optimizer_values_changed") is not False
    ):
        raise ValueError("MLX numeric repair receipt does not match the approved migration")
    source = value.get("source_checkpoint")
    if not isinstance(source, dict):
        raise ValueError("MLX numeric repair receipt has no source checkpoint identity")
    for key in ("manifest_sha256", "model_sha256", "optimizer_sha256"):
        digest = source.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("MLX numeric repair checkpoint identity is invalid")
    return value


def effective_mlx_qat_feature_accumulation(
    root: Path,
    plan: Mapping[str, object],
) -> str:
    """Return the effective QAT accumulator, requiring a receipt for v3 migration."""

    contract = _qat_feature_accumulation_contract(plan)
    if contract == QAT_FEATURE_ACCUMULATION:
        return contract
    if contract != LEGACY_QAT_FEATURE_ACCUMULATION:
        raise ValueError("QAT feature accumulation contract changed")
    return cast(str, _load_mlx_numeric_repair_receipt(root)["to_feature_accumulation"])


def _export_rounding_contract(plan: Mapping[str, object]) -> str:
    numerics = plan.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("QAT MLX plan has no numeric training contract")
    contract = numerics.get("export_rounding")
    if contract is None:
        return LEGACY_EXPORT_ROUNDING
    if not isinstance(contract, str):
        raise ValueError("QAT MLX export rounding contract is invalid")
    return contract


def _rounding_repair_source_files() -> dict[str, str]:
    return {
        "mlx_nnue.py": _sha256_file(Path(__file__).resolve(strict=True)),
        "mlx_training_primitives.py": _sha256_file(
            Path(__file__).with_name("mlx_training_primitives.py").resolve(strict=True)
        ),
    }


def _load_mlx_export_rounding_repair_receipt(root: Path) -> dict[str, Any]:
    path = root / "receipts" / "mlx-export-rounding-repair-v1.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("MLX export rounding repair receipt is not a regular file")
    value = _strict_json(path.read_bytes(), label="MLX export rounding repair receipt")
    if (
        value.get("schema") != MLX_EXPORT_ROUNDING_REPAIR_SCHEMA
        or value.get("complete") is not True
        or value.get("source_mlx_plan_sha256") != _sha256_file(root / "mlx-plan.json")
        or value.get("from_export_rounding") != LEGACY_EXPORT_ROUNDING
        or value.get("to_export_rounding") != TATARA_EXPORT_ROUNDING
        or value.get("model_or_optimizer_values_changed") is not False
        or value.get("source_files") != _rounding_repair_source_files()
    ):
        raise ValueError("MLX export rounding repair receipt does not match the approved migration")
    source_checkpoint = value.get("source_checkpoint")
    source_export = value.get("source_export")
    if not isinstance(source_checkpoint, dict) or not isinstance(source_export, dict):
        raise ValueError("MLX export rounding repair identities are incomplete")
    for identity in (source_checkpoint, source_export):
        for key, digest in identity.items():
            if not key.endswith("_sha256"):
                continue
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("MLX export rounding repair has an invalid SHA-256")
    return value


def effective_mlx_export_rounding(root: Path, plan: Mapping[str, object]) -> str:
    """Return the exporter rounding contract, requiring a receipt for old runs."""

    contract = _export_rounding_contract(plan)
    if contract == TATARA_EXPORT_ROUNDING:
        return contract
    if contract != LEGACY_EXPORT_ROUNDING:
        raise ValueError("QAT export rounding contract changed")
    return cast(str, _load_mlx_export_rounding_repair_receipt(root)["to_export_rounding"])


def _ensure_mlx_export_rounding_repair_receipt(
    root: Path,
    *,
    plan: Mapping[str, object],
    state: Mapping[str, object],
    checkpoint: Path,
) -> dict[str, Any] | None:
    contract = _export_rounding_contract(plan)
    if contract == TATARA_EXPORT_ROUNDING:
        return None
    if contract != LEGACY_EXPORT_ROUNDING:
        raise ValueError("QAT export rounding contract changed")
    path = root / "receipts" / "mlx-export-rounding-repair-v1.json"
    if path.is_file() and not path.is_symlink():
        return _load_mlx_export_rounding_repair_receipt(root)
    if path.exists() or path.is_symlink():
        raise FileExistsError("partial MLX export rounding repair receipt requires inspection")
    failure = state.get("failure")
    if not isinstance(failure, dict) or failure.get("kind") != "ValueError":
        raise ValueError("legacy rounding run requires its recorded native parity failure")
    message = failure.get("message")
    if not isinstance(message, str) or not message.startswith(
        "MLX integer reference differs from the exported native network:"
    ):
        raise ValueError("legacy rounding run did not stop at the approved native parity gate")
    manifest_path = checkpoint / "manifest.json"
    manifest = _strict_json(manifest_path.read_bytes(), label="MLX repair checkpoint manifest")
    model_identity = manifest.get("model_file")
    optimizer_identity = manifest.get("optimizer_file")
    if not isinstance(model_identity, dict) or not isinstance(optimizer_identity, dict):
        raise ValueError("MLX export rounding repair checkpoint identity is incomplete")
    model_path = checkpoint / cast(str, model_identity["name"])
    optimizer_path = checkpoint / cast(str, optimizer_identity["name"])
    if model_identity.get("sha256") != _sha256_file(model_path) or optimizer_identity.get(
        "sha256"
    ) != _sha256_file(optimizer_path):
        raise ValueError("MLX export rounding repair checkpoint artifacts changed")
    optimizer_step = int(cast(dict[str, Any], manifest["optimizer"])["step"])
    export = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    export_receipt_path = export / "receipt.json"
    export_receipt = _strict_json(
        export_receipt_path.read_bytes(), label="failed MLX export receipt"
    )
    if (
        export_receipt.get("schema") != "meteo-nagisa-mlx-yaneuraou-export-v1"
        or export_receipt.get("optimizer_step") != optimizer_step
        or export_receipt.get("checkpoint_model_sha256") != model_identity.get("sha256")
    ):
        raise ValueError("failed MLX export does not match the rounding repair checkpoint")
    source_export: dict[str, object] = {
        "receipt_sha256": _sha256_file(export_receipt_path),
    }
    for filename in ("network.bin", "nn.bin"):
        artifact = export / filename
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"failed MLX export artifact is not regular: {artifact}")
        source_export[f"{filename}_sha256"] = _sha256_file(artifact)
    receipt: dict[str, Any] = {
        "schema": MLX_EXPORT_ROUNDING_REPAIR_SCHEMA,
        "complete": True,
        "created_unix": time.time(),
        "source_mlx_plan_sha256": _sha256_file(root / "mlx-plan.json"),
        "from_export_rounding": LEGACY_EXPORT_ROUNDING,
        "to_export_rounding": TATARA_EXPORT_ROUNDING,
        "reason": "qa127_float32_half_boundary_differs_from_rust_f64_export_rounding",
        "repair": {
            "forward": "FastTwoSum residual plus Rust half-away-from-zero decision",
            "backward": "straight-through scale gradient",
            "approved_scales": [64, 127, 8128],
        },
        "source_files": _rounding_repair_source_files(),
        "source_checkpoint": {
            "optimizer_step": optimizer_step,
            "manifest_sha256": _sha256_file(manifest_path),
            "model_sha256": model_identity["sha256"],
            "optimizer_sha256": optimizer_identity["sha256"],
        },
        "source_export": source_export,
        "original_failure": dict(failure),
        "model_or_optimizer_values_changed": False,
        "local_only": True,
    }
    _write_new(path, _json_bytes(receipt), mode=0o600)
    _fsync_directory(path.parent)
    return receipt


def _ensure_mlx_numeric_repair_receipt(
    root: Path,
    *,
    plan: Mapping[str, object],
    state: Mapping[str, object],
    checkpoint: Path,
) -> dict[str, Any] | None:
    contract = _qat_feature_accumulation_contract(plan)
    if contract == QAT_FEATURE_ACCUMULATION:
        return None
    if contract != LEGACY_QAT_FEATURE_ACCUMULATION:
        raise ValueError("QAT feature accumulation contract changed")
    path = root / "receipts" / "mlx-numeric-repair-v1.json"
    if path.is_file() and not path.is_symlink():
        return _load_mlx_numeric_repair_receipt(root)
    if path.exists() or path.is_symlink():
        raise FileExistsError("partial MLX numeric repair receipt requires inspection")
    failure = state.get("failure")
    if not isinstance(failure, dict) or failure.get("kind") != "ValueError":
        raise ValueError("legacy FP16 run requires its recorded numeric audit failure")
    message = failure.get("message")
    if not isinstance(message, str) or not message.startswith(
        "FP16 feature accumulation differs from the exported native network:"
    ):
        raise ValueError("legacy FP16 run did not stop at the approved numeric repair gate")
    manifest_path = checkpoint / "manifest.json"
    manifest = _strict_json(manifest_path.read_bytes(), label="MLX repair checkpoint manifest")
    model_identity = manifest.get("model_file")
    optimizer_identity = manifest.get("optimizer_file")
    if not isinstance(model_identity, dict) or not isinstance(optimizer_identity, dict):
        raise ValueError("MLX repair checkpoint has no model/optimizer identities")
    model_path = checkpoint / cast(str, model_identity["name"])
    optimizer_path = checkpoint / cast(str, optimizer_identity["name"])
    if model_identity.get("sha256") != _sha256_file(model_path) or optimizer_identity.get(
        "sha256"
    ) != _sha256_file(optimizer_path):
        raise ValueError("MLX repair checkpoint artifacts no longer match their manifest")
    receipt: dict[str, Any] = {
        "schema": MLX_NUMERIC_REPAIR_SCHEMA,
        "complete": True,
        "created_unix": time.time(),
        "source_mlx_plan_sha256": _sha256_file(root / "mlx-plan.json"),
        "from_feature_accumulation": LEGACY_QAT_FEATURE_ACCUMULATION,
        "to_feature_accumulation": QAT_FEATURE_ACCUMULATION,
        "reason": "generation_8_fp16_forward_failed_native_probability_parity_gate",
        "repair": {
            "forward": "fused Metal int32 sparse FT accumulation",
            "vjp": "FP16 scatter-add straight-through gradient",
            "independent_reference": "ordinary MLX float32 sparse FT accumulation",
        },
        "source_checkpoint": {
            "path": str(checkpoint),
            "optimizer_step": cast(dict[str, Any], manifest["optimizer"])["step"],
            "manifest_sha256": _sha256_file(manifest_path),
            "model_sha256": model_identity["sha256"],
            "optimizer_sha256": optimizer_identity["sha256"],
        },
        "original_failure": dict(failure),
        "model_or_optimizer_values_changed": False,
        "local_only": True,
    }
    _write_new(path, _json_bytes(receipt), mode=0o600)
    _fsync_directory(path.parent)
    return receipt


def _quantisation_schedule_from_plan(plan: Mapping[str, object]) -> QuantisationSchedule:
    if plan.get("schema") != MLX_BACKEND_PLAN_SCHEMA:
        raise ValueError("legacy float-only MLX plans cannot silently resume under QAT code")
    numerics = plan.get("numerics")
    if not isinstance(numerics, dict):
        raise ValueError("QAT MLX plan has no numeric training contract")
    if numerics.get("deployment_graph") != "tatara_yaneuraou_integer_equivalent_ste":
        raise ValueError("QAT deployment graph contract changed")
    if numerics.get("warmup_forward") != "ft_float16_dense_float32":
        raise ValueError("QAT warm-up precision contract changed")
    if numerics.get("qat_feature_accumulation") not in {
        LEGACY_QAT_FEATURE_ACCUMULATION,
        QAT_FEATURE_ACCUMULATION,
    }:
        raise ValueError("QAT feature accumulation contract changed")
    if _export_rounding_contract(plan) not in {
        LEGACY_EXPORT_ROUNDING,
        TATARA_EXPORT_ROUNDING,
    }:
        raise ValueError("QAT export rounding contract changed")
    return QuantisationSchedule(
        warmup_optimizer_steps=int(cast(int, numerics["warmup_optimizer_steps"]))
    )


def load_mlx_state(run_directory: Path) -> dict[str, Any]:
    root = run_directory.expanduser().resolve(strict=True)
    value = _strict_json((root / "mlx-state.json").read_bytes(), label="MLX backend state")
    if value.get("schema") != MLX_BACKEND_STATE_SCHEMA:
        raise ValueError("MLX backend state schema mismatch")
    return value


def mlx_hardware_preflight(run_directory: Path) -> dict[str, object]:
    """Verify that this run can use the local Apple GPU without cloud compute."""

    root = run_directory.expanduser().resolve(strict=True)
    plan = load_nagisa_plan(root)
    reasons: list[str] = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        reasons.append("the MLX backend requires Apple-Silicon macOS")
    device = mx.default_device()
    device_text = str(device)
    if "gpu" not in device_text.lower():
        reasons.append(f"MLX default device is not the Apple GPU: {device_text}")
    try:
        device_info = cast(dict[str, str | int], mx.device_info(device))
    except Exception as error:  # pragma: no cover - defensive driver boundary
        device_info = {"error": repr(error)}
        reasons.append(f"MLX device query failed: {error}")
    progress = root / "private" / "progress.bin"
    expected_progress = cast(
        dict[str, Any],
        cast(dict[str, Any], plan["initialization"])["progress_router"],
    )
    if (
        not progress.is_file()
        or progress.is_symlink()
        or progress.stat().st_size != int(cast(int, expected_progress["bytes"]))
        or _sha256_file(progress) != cast(str, expected_progress["sha256"])
    ):
        reasons.append("progress.bin is missing or differs from the immutable plan")
    heldout = root / "private" / "heldout.psv"
    heldout_receipt = root / "receipts" / "heldout.json"
    if not heldout.is_file() or heldout.is_symlink() or not heldout_receipt.is_file():
        reasons.append("the immutable held-out PSV tail has not been prepared")
    native_receipt = root / "receipts" / "mlx-native-build.json"
    binary = root / "dependencies" / "meteo_mlx_native" / "target" / "release" / "meteo-mlx-native"
    if not native_receipt.is_file() or not binary.is_file() or binary.is_symlink():
        reasons.append("the pinned Tatara CPU companion has not been built")
    else:
        receipt = _strict_json(native_receipt.read_bytes(), label="MLX native build receipt")
        identity = receipt.get("binary")
        if not isinstance(identity, dict) or identity.get("sha256") != _sha256_file(binary):
            reasons.append("the Tatara CPU companion differs from its build receipt")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mlx_device": device_text,
        "mlx_device_info": device_info,
        "native_binary": str(binary),
        "cloud_gpu_used": False,
    }


def prepare_mlx_backend(
    run_directory: Path,
    *,
    checkpoint_batches: int = DEFAULT_CHECKPOINT_BATCHES,
    log_batches: int = DEFAULT_LOG_BATCHES,
    validation_positions: int = DEFAULT_VALIDATION_POSITIONS,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Add an immutable local-MLX execution plan to a prepared NNUE run."""

    root = run_directory.expanduser().resolve(strict=True)
    common = load_nagisa_plan(root)
    training = cast(dict[str, Any], common["training"])
    parameters = WrmLossParameters.from_training(training)
    batch_size = int(cast(int, training["batch_size"]))
    if batch_size != DEFAULT_MLX_BATCH_SIZE:
        raise ValueError(
            f"MLX production requires the pinned logical batch {DEFAULT_MLX_BATCH_SIZE}"
        )
    if checkpoint_batches < 1 or log_batches < 1 or validation_positions < 1:
        raise ValueError("MLX checkpoint/log/validation intervals must be positive")
    if validation_positions % batch_size:
        raise ValueError("MLX validation positions must be a whole number of logical batches")
    ensure_nagisa_heldout(root, timeout_seconds=timeout_seconds)
    native_receipt = build_mlx_native(root)
    hardware = mlx_hardware_preflight(root)
    if not hardware["ready"]:
        raise RuntimeError(f"local MLX preflight failed: {hardware['reasons']}")
    plan_path = root / "mlx-plan.json"
    state_path = root / "mlx-state.json"
    if plan_path.is_file() and state_path.is_file():
        existing = load_mlx_plan(root)
        configured = cast(dict[str, Any], existing["execution"])
        expected = {
            "batch_size": batch_size,
            "checkpoint_batches": checkpoint_batches,
            "log_batches": log_batches,
            "validation_positions": validation_positions,
            "quantisation_audit_positions": DEFAULT_SMOKE_POSITIONS,
            "mlx_cache_limit_bytes": DEFAULT_MLX_CACHE_BYTES,
        }
        observed = {key: configured.get(key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"immutable MLX execution plan differs: expected={expected} observed={observed}"
            )
        return existing
    if (
        plan_path.exists()
        or plan_path.is_symlink()
        or state_path.exists()
        or state_path.is_symlink()
    ):
        raise FileExistsError("partial MLX plan/state artifacts require operator inspection")
    device_info = cast(dict[str, Any], hardware["mlx_device_info"])
    recommended = int(device_info.get("max_recommended_working_set_size", 48 * 1024**3))
    physical = int(device_info.get("memory_size", recommended))
    memory_limit = min(int(recommended * 0.90), int(physical * 0.80))
    memory_limit = max(memory_limit, 24 * 1024**3)
    plan: dict[str, object] = {
        "schema": MLX_BACKEND_PLAN_SCHEMA,
        "created_unix": time.time(),
        "source_plan_sha256": _sha256_file(root / "plan.json"),
        "mode": "nagisa_value_only_nnue_local_mlx",
        "cloud_gpu_used": False,
        "legacy_policy_value_mcts_used": False,
        "architecture": common["architecture"],
        "labels": {
            "source": "PackedSfenValue.score_i16",
            "move16_used": False,
            "policy_head": False,
            "wdl_lambda": 0.0,
            "mate_stamps": "retained_without_clipping",
            "loss_kind": "Tatara_WRM_non_extended_MSE",
            "target_transform": f"sigmoid(score/{float(training['score_scale'])})",
            "prediction_transform": (
                f"sigmoid(network_output*{parameters.nnue2score}/{parameters.in_scaling})"
            ),
            "wrm": parameters.to_dict(),
            "distribution_preservation": {
                "target_transform_changed_by_output_normalization": False,
                "quantization_gain": 127 * 64,
                "engine_fv_scale": training["yaneuraou_fv_scale"],
                "effective_cp_per_float_output": parameters.nnue2score,
                "systematic_export_scale_ratio": 1.0,
            },
        },
        "initialization": {
            "generator": "pinned Tatara LayerStackInit::default_uniform",
            "real_ft": "Tatara xorshift uniform fan_in_73305 seed_0x100",
            "factorizer_virtual_rows": "exact_zero",
            "dense_weights": "Tatara xorshift uniform_abs_0.01 fixed_group_seeds",
            "biases": "exact_zero",
            "teacher_nnue_weights_copied": False,
        },
        "optimizer": {
            "name": "Ranger",
            "implementation": "MLX port of pinned Tatara RAdam plus Lookahead",
            "beta1": 0.99,
            "beta2": 0.999,
            "eps": 1e-8,
            "decay": 0.0,
            "alpha": 0.5,
            "lookahead_k": 6,
            "n_sma_threshold": 5.0,
            "slow_weights_initialization": "exact_zero",
            "state_precision": "float32",
            "state_precision_reason": (
                "measured FP16 moments reduced memory but did not improve throughput; "
                "retain full precision for learning stability"
            ),
        },
        "numerics": {
            "warmup_optimizer_steps": DEFAULT_QUANTISATION_WARMUP_STEPS,
            "warmup_presentations": DEFAULT_QUANTISATION_WARMUP_STEPS * batch_size,
            "warmup_forward": "ft_float16_dense_float32",
            "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
            "qat_feature_accumulation": QAT_FEATURE_ACCUMULATION,
            "export_rounding": TATARA_EXPORT_ROUNDING,
            "master_weight_precision": "float32",
            "qa": int(QUANTISATION_QA),
            "qb": int(QUANTISATION_QB),
            "bias_scale": int(QUANTISATION_BIAS_SCALE),
            "factorizer_export_rounding": "fold_then_round_once",
            "right_shift_emulation": "ste_floor_on_nonnegative_or_clipped_values",
            "reason_for_warmup": "avoid_all_zero_quantised_random_feature_transformer",
        },
        "execution": {
            "device": hardware["mlx_device"],
            "device_info": device_info,
            "precision": "mixed_fp16_fp32_then_integer_equivalent_qat",
            "batch_size": batch_size,
            "checkpoint_batches": checkpoint_batches,
            "log_batches": log_batches,
            "validation_positions": validation_positions,
            "quantisation_audit_positions": DEFAULT_SMOKE_POSITIONS,
            "native_decode_threads": _dataloader_threads(),
            "native_prefetch_batches": 2,
            "mlx_memory_limit_bytes": memory_limit,
            "mlx_cache_limit_bytes": DEFAULT_MLX_CACHE_BYTES,
            "memory_policy": (
                "measured_working_set_plus_headroom; consuming unused unified memory "
                "does not increase arithmetic throughput and risks swap"
            ),
            "target_presentations": training["target_presentations"],
            "planned_presentations": training["planned_presentations"],
            "total_optimizer_steps": sum(
                int(cast(int, item["batches"]))
                for item in cast(list[dict[str, object]], common["schedule"])
            ),
        },
        "storage": {
            "checkpoint_generations": 2,
            "export_generations": 2,
            "cached_source_shards": 2,
            "source_prefetch": "one resumable next shard overlaps accelerator compute",
            "source_prefetch_storage_reserve_bytes": SOURCE_PREFETCH_STORAGE_RESERVE_BYTES,
            "atomic_checkpoint_directories": True,
            "model_optimizer_reload_required_before_retention": True,
        },
        "native": native_receipt,
    }
    state: dict[str, object] = {
        "schema": MLX_BACKEND_STATE_SCHEMA,
        "status": "prepared",
        "pid": None,
        "durable_optimizer_step": 0,
        "durable_presentations": 0,
        "observed_optimizer_step": 0,
        "observed_presentations": 0,
        "completed_superbatch": 0,
        "latest_checkpoint": None,
        "latest_export": None,
        "last_metrics": None,
        "failure": None,
        "updated_unix": time.time(),
    }
    _write_new(plan_path, _json_bytes(plan), mode=0o600)
    _write_new(state_path, _json_bytes(state), mode=0o600)
    _fsync_directory(root)
    return plan


def _native_binary_from_plan(root: Path, plan: Mapping[str, object]) -> Path:
    native = cast(dict[str, Any], plan["native"])
    identity = cast(dict[str, Any], native["binary"])
    binary = Path(cast(str, identity["path"])).resolve(strict=True)
    if binary.is_symlink() or _sha256_file(binary) != cast(str, identity["sha256"]):
        raise ValueError("native MLX companion no longer matches the immutable plan")
    return binary


def _append_jsonl(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with _METRIC_LOCK, path.open("ab", buffering=0) as stream:
        stream.write(payload)
        os.fsync(stream.fileno())


def _log_mlx_event(root: Path, event: str, **fields: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": MLX_METRIC_SCHEMA,
        "time_unix": time.time(),
        "event": event,
        **fields,
    }
    _append_jsonl(root / "logs" / "mlx-training.jsonl", record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return record


def _learning_rate(common_plan: Mapping[str, object], superbatch: int) -> float:
    training = cast(dict[str, Any], common_plan["training"])
    initial = float(cast(float, training["learning_rate"]))
    final = float(cast(float, training["learning_rate_final"]))
    horizon = int(cast(int, training["superbatches"]))
    if superbatch >= horizon:
        return final
    progress = float(superbatch) / float(horizon)
    weight = 1.0 - 0.5 * (1.0 + math.cos(math.pi * progress))
    return initial + weight * (final - initial)


def _coordinate_for_step(
    schedule: Sequence[Mapping[str, object]], optimizer_step: int
) -> tuple[int, int, int]:
    """Map a global optimizer step to (schedule index, superbatch, batch offset)."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    consumed = 0
    for schedule_index, segment in enumerate(schedule):
        batches = int(cast(int, segment["batches"]))
        if optimizer_step < consumed + batches:
            return schedule_index, int(cast(int, segment["superbatch"])), optimizer_step - consumed
        consumed += batches
    if optimizer_step == consumed:
        return len(schedule), len(schedule) + 1, 0
    raise ValueError(f"optimizer step {optimizer_step} exceeds planned {consumed}")


def _latest_mlx_checkpoint(directory: Path) -> Path | None:
    prefix = f"{DEFAULT_NET_ID}-mlx-step-"
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if not suffix.isdigit():
            continue
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"MLX checkpoint must be a regular directory: {path}")
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"published MLX checkpoint has no manifest: {path}")
        manifest = _strict_json(manifest_path.read_bytes(), label="MLX checkpoint manifest")
        if manifest.get("schema") != MLX_CHECKPOINT_SCHEMA or manifest.get("complete") is not True:
            raise ValueError(f"published MLX checkpoint is incomplete: {path}")
        candidates.append((int(suffix), path))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def _model_batch_statistics(
    model: MeteoValueNnue,
    batch: MlxNnueBatch,
    *,
    parameters: WrmLossParameters,
    forward_mode: ModelForwardMode,
) -> dict[str, float | int]:
    stm, nstm, nnz, buckets, scores = batch.mlx()
    outputs = _model_outputs(
        model,
        stm,
        nstm,
        nnz,
        buckets,
        forward_mode=forward_mode,
    )
    predictions, targets = wrm_probabilities(outputs, scores, parameters=parameters)
    squared = mx.square(predictions - targets)
    values = (
        mx.sum(squared),
        mx.sum(outputs),
        mx.sum(mx.square(outputs)),
        mx.min(outputs),
        mx.max(outputs),
        mx.sum(predictions),
        mx.sum(mx.square(predictions)),
        mx.sum(targets),
        mx.sum(mx.square(targets)),
        mx.sum(predictions * targets),
    )
    mx.eval(values)
    return {
        "positions": batch.positions,
        "loss_sum": float(values[0].item()),
        "output_sum": float(values[1].item()),
        "output_square_sum": float(values[2].item()),
        "output_minimum": float(values[3].item()),
        "output_maximum": float(values[4].item()),
        "prediction_sum": float(values[5].item()),
        "prediction_square_sum": float(values[6].item()),
        "target_sum": float(values[7].item()),
        "target_square_sum": float(values[8].item()),
        "prediction_target_sum": float(values[9].item()),
    }


def evaluate_mlx_psv(
    model: MeteoValueNnue,
    *,
    psv: Path,
    progress: Path,
    native_binary: Path,
    positions: int,
    batch_size: int,
    parameters: WrmLossParameters,
    threads: int,
    forward_mode: ModelForwardMode = "float32",
) -> dict[str, object]:
    """Run a deterministic forward-only evaluation and report distributions."""

    if positions < 1 or batch_size < 1 or positions % batch_size:
        raise ValueError("evaluation positions must be positive and divisible by batch size")
    totals = {
        "positions": 0,
        "loss_sum": 0.0,
        "output_sum": 0.0,
        "output_square_sum": 0.0,
        "prediction_sum": 0.0,
        "prediction_square_sum": 0.0,
        "target_sum": 0.0,
        "target_square_sum": 0.0,
        "prediction_target_sum": 0.0,
        "score_sum": 0.0,
        "score_square_sum": 0.0,
    }
    output_minimum = math.inf
    output_maximum = -math.inf
    score_minimum = 32767
    score_maximum = -32768
    mate_stamps = 0
    bucket_counts = [0] * NUM_BUCKETS
    started = time.perf_counter()
    with NativeBatchStream(
        native_binary,
        psv=psv,
        progress=progress,
        start_record=0,
        records=positions,
        batch_size=batch_size,
        threads=threads,
    ) as batches:
        for batch in batches:
            observed = _model_batch_statistics(
                model,
                batch,
                parameters=parameters,
                forward_mode=forward_mode,
            )
            totals["positions"] += int(cast(int, observed["positions"]))
            for key in (
                "loss_sum",
                "output_sum",
                "output_square_sum",
                "prediction_sum",
                "prediction_square_sum",
                "target_sum",
                "target_square_sum",
                "prediction_target_sum",
            ):
                totals[key] += float(cast(float, observed[key]))
            output_minimum = min(output_minimum, float(cast(float, observed["output_minimum"])))
            output_maximum = max(output_maximum, float(cast(float, observed["output_maximum"])))
            score_minimum = min(score_minimum, int(batch.scores.min()))
            score_maximum = max(score_maximum, int(batch.scores.max()))
            score_f64 = batch.scores.astype(np.float64)
            totals["score_sum"] += float(score_f64.sum())
            totals["score_square_sum"] += float(np.square(score_f64).sum())
            mate_stamps += int(np.count_nonzero(np.abs(batch.scores.astype(np.int32)) >= 32_000))
            counts = np.bincount(batch.buckets, minlength=NUM_BUCKETS)
            for bucket, count in enumerate(counts[:NUM_BUCKETS]):
                bucket_counts[bucket] += int(count)
    elapsed = max(time.perf_counter() - started, 1e-9)
    count = int(totals["positions"])
    if count != positions:
        raise ValueError(f"held-out evaluator consumed {count}, expected {positions}")
    output_mean = totals["output_sum"] / count
    prediction_mean = totals["prediction_sum"] / count
    target_mean = totals["target_sum"] / count
    output_std = math.sqrt(
        max(0.0, totals["output_square_sum"] / count - output_mean * output_mean)
    )
    prediction_std = math.sqrt(
        max(0.0, totals["prediction_square_sum"] / count - prediction_mean * prediction_mean)
    )
    target_std = math.sqrt(
        max(0.0, totals["target_square_sum"] / count - target_mean * target_mean)
    )
    prediction_target_covariance = (
        totals["prediction_target_sum"] / count - prediction_mean * target_mean
    )
    probability_correlation = (
        prediction_target_covariance / (prediction_std * target_std)
        if prediction_std > 0.0 and target_std > 0.0
        else None
    )
    score_mean = totals["score_sum"] / count
    score_std = math.sqrt(max(0.0, totals["score_square_sum"] / count - score_mean * score_mean))
    return {
        "positions": count,
        "forward_mode": forward_mode,
        "loss": totals["loss_sum"] / count,
        "positions_per_second": count / elapsed,
        "elapsed_seconds": elapsed,
        "network_output_normalized": {
            "minimum": output_minimum,
            "maximum": output_maximum,
            "mean": output_mean,
            "standard_deviation": output_std,
        },
        "network_output_cp": {
            "minimum": output_minimum * parameters.nnue2score,
            "maximum": output_maximum * parameters.nnue2score,
            "mean": output_mean * parameters.nnue2score,
            "standard_deviation": output_std * parameters.nnue2score,
            "multiplier": parameters.nnue2score,
        },
        "prediction_probability": {
            "mean": prediction_mean,
            "standard_deviation": prediction_std,
        },
        "target_probability": {
            "mean": target_mean,
            "standard_deviation": target_std,
        },
        "prediction_target_probability_correlation": probability_correlation,
        "prediction_target_probability_mean_shift": prediction_mean - target_mean,
        "raw_score": {
            "minimum": score_minimum,
            "maximum": score_maximum,
            "mean": score_mean,
            "standard_deviation": score_std,
            "mate_stamp_abs_32000_or_more": mate_stamps,
        },
        "progress_bucket_counts": bucket_counts,
    }


def _distribution_summary(values: NDArray[np.float64]) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("distribution audit requires a finite non-empty vector")
    quantiles = np.quantile(values, [0.01, 0.5, 0.99])
    return {
        "minimum": float(values.min()),
        "p01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def _rounding_and_range_summary(
    values: NDArray[np.float64],
    *,
    scale: float,
    minimum_raw: float,
    maximum_raw: float,
) -> dict[str, object]:
    """Describe master-weight distance from one deployed integer tensor."""

    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weight audit requires finite non-empty tensors")
    raw = values * scale
    rounded = np.rint(raw)
    below = rounded < minimum_raw
    above = rounded > maximum_raw
    out_of_range = below | above
    clipped = np.clip(rounded, minimum_raw, maximum_raw)
    residual = np.abs(raw - clipped)
    count = int(raw.size)
    return {
        "elements": count,
        "scale": scale,
        "representable_raw": [minimum_raw, maximum_raw],
        "representable_float": [minimum_raw / scale, maximum_raw / scale],
        "master_minimum": float(values.min()),
        "master_maximum": float(values.max()),
        "below_range_elements": int(below.sum()),
        "above_range_elements": int(above.sum()),
        "out_of_range_elements": int(out_of_range.sum()),
        "out_of_range_fraction": float(out_of_range.mean()),
        "mean_absolute_raw_rounding_or_clipping_residual": float(residual.mean()),
        "p99_absolute_raw_rounding_or_clipping_residual": float(np.quantile(residual, 0.99)),
        "maximum_absolute_raw_rounding_or_clipping_residual": float(residual.max()),
    }


def audit_mlx_master_weight_ranges(checkpoint: Path) -> dict[str, object]:
    """Audit a checkpoint without changing its weights, optimizer, or run state.

    The effective HalfKA transformer and merged L1 weights are audited because
    those are the tensors actually quantised by the exporter.  This avoids the
    false comfort of inspecting factoriser branches independently.
    """

    requested = checkpoint.expanduser()
    if requested.is_symlink():
        raise ValueError("MLX checkpoint must be a regular directory")
    source = requested.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("MLX checkpoint must be a regular directory")
    manifest = _strict_json((source / "manifest.json").read_bytes(), label="MLX manifest")
    if manifest.get("schema") != MLX_CHECKPOINT_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("MLX checkpoint has no valid complete marker")
    identity = cast(dict[str, Any], manifest["model_file"])
    model_path = source / cast(str, identity["name"])
    if (
        not model_path.is_file()
        or model_path.is_symlink()
        or model_path.stat().st_size != int(cast(int, identity["bytes"]))
        or _sha256_file(model_path) != cast(str, identity["sha256"])
    ):
        raise ValueError("MLX checkpoint model identity does not match its manifest")
    loaded = mx.load(model_path)
    if not isinstance(loaded, dict):
        raise TypeError("MLX model safetensors did not load as a tensor mapping")
    arrays = cast(dict[str, mx.array], loaded)
    if tuple(sorted(arrays)) != tuple(sorted(MODEL_KEYS)):
        raise ValueError("MLX model checkpoint parameter names do not match the architecture")
    numpy_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    effective_ft = numpy_arrays["ft_real"] + np.tile(numpy_arrays["ft_virtual"], (45, 1))
    merged_l1_w = numpy_arrays["l1_w"] + np.swapaxes(numpy_arrays["l1f_w"], 0, 1)[None, :, :]
    merged_l1_b = numpy_arrays["l1_b"] + numpy_arrays["l1f_b"][None, :]
    audited = {
        "effective_ft_w": _rounding_and_range_summary(
            effective_ft,
            scale=QUANTISATION_QA,
            minimum_raw=-32_768.0,
            maximum_raw=32_767.0,
        ),
        "ft_b": _rounding_and_range_summary(
            numpy_arrays["ft_b"],
            scale=QUANTISATION_QA,
            minimum_raw=-32_768.0,
            maximum_raw=32_767.0,
        ),
        "merged_l1_w": _rounding_and_range_summary(
            merged_l1_w,
            scale=QUANTISATION_QB,
            minimum_raw=-128.0,
            maximum_raw=127.0,
        ),
        "merged_l1_b": _rounding_and_range_summary(
            merged_l1_b,
            scale=QUANTISATION_BIAS_SCALE,
            minimum_raw=-2_147_483_648.0,
            maximum_raw=2_147_483_647.0,
        ),
        "l2_w": _rounding_and_range_summary(
            numpy_arrays["l2_w"],
            scale=QUANTISATION_QB,
            minimum_raw=-128.0,
            maximum_raw=127.0,
        ),
        "l2_b": _rounding_and_range_summary(
            numpy_arrays["l2_b"],
            scale=QUANTISATION_BIAS_SCALE,
            minimum_raw=-2_147_483_648.0,
            maximum_raw=2_147_483_647.0,
        ),
        "l3_w": _rounding_and_range_summary(
            numpy_arrays["l3_w"],
            scale=QUANTISATION_QB,
            minimum_raw=-128.0,
            maximum_raw=127.0,
        ),
        "l3_b": _rounding_and_range_summary(
            numpy_arrays["l3_b"],
            scale=QUANTISATION_BIAS_SCALE,
            minimum_raw=-2_147_483_648.0,
            maximum_raw=2_147_483_647.0,
        ),
    }
    out_of_range = sum(cast(int, summary["out_of_range_elements"]) for summary in audited.values())
    elements = sum(cast(int, summary["elements"]) for summary in audited.values())
    return {
        "schema": "meteo-nagisa-mlx-master-range-audit-v1",
        "checkpoint": str(source),
        "checkpoint_model_sha256": identity["sha256"],
        "read_only": True,
        "factoriser_audit": "effective_ft_and_merged_l1_after_combining_virtual_branches",
        "out_of_range_elements": out_of_range,
        "audited_elements": elements,
        "out_of_range_fraction": out_of_range / elements,
        "tensors": audited,
    }


def _wrm_probability_numpy(
    centipawns: NDArray[np.float64],
    *,
    scaling: float,
    offset: float,
) -> NDArray[np.float64]:
    positive = 1.0 / (1.0 + np.exp(-(centipawns - offset) / scaling))
    negative = 1.0 / (1.0 + np.exp(-(-centipawns - offset) / scaling))
    return cast(NDArray[np.float64], 0.5 * (1.0 + positive - negative))


def _probability_error_summary(
    observed: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> dict[str, float]:
    residual = observed - reference
    absolute = np.abs(residual)
    return {
        "mean": float(residual.mean()),
        "mean_absolute": float(absolute.mean()),
        "p99_absolute": float(np.quantile(absolute, 0.99)),
        "maximum_absolute": float(absolute.max()),
    }


def audit_quantised_distribution(
    model: MeteoValueNnue,
    *,
    network: Path,
    psv: Path,
    progress: Path,
    native_binary: Path,
    positions: int,
    batch_size: int,
    parameters: WrmLossParameters,
    fv_scale: int,
    threads: int,
) -> dict[str, object]:
    """Prove that QAT forward values agree with the exported integer network.

    Float-master versus integer output drift remains useful diagnostic data,
    but it is not the acceptance criterion: training optimises the STE
    deployment graph.  Acceptance instead compares (1) the high-fidelity MLX
    integer reference and (2) the fused int32-forward QAT graph against the
    native Rust evaluator that reads the actual exported network.
    """

    if positions < 1 or batch_size < 1 or positions % batch_size:
        raise ValueError("quantisation audit positions must be divisible by batch_size")
    exact_multiplier = float(127 * 64) / fv_scale
    if fv_scale <= 0 or exact_multiplier != parameters.nnue2score:
        raise ValueError(
            "WRM/output scale contract is not exactly representable by YaneuraOu: "
            f"8128/{fv_scale}={exact_multiplier}, nnue2score={parameters.nnue2score}"
        )
    float_parts: list[NDArray[np.float64]] = []
    reference_parts: list[NDArray[np.float64]] = []
    training_parts: list[NDArray[np.float64]] = []
    score_parts: list[NDArray[np.float64]] = []
    with NativeBatchStream(
        native_binary,
        psv=psv,
        progress=progress,
        start_record=0,
        records=positions,
        batch_size=batch_size,
        threads=threads,
    ) as batches:
        for batch in batches:
            stm, nstm, nnz, buckets, _ = batch.mlx()
            float_output = model(stm, nstm, nnz, buckets)
            reference_output = model.quantised_reference_forward(stm, nstm, nnz, buckets)
            training_output = model.quantisation_aware_forward(stm, nstm, nnz, buckets)
            mx.eval(float_output, reference_output, training_output)
            float_parts.append(np.asarray(float_output, dtype=np.float64) * parameters.nnue2score)
            reference_parts.append(
                np.asarray(reference_output, dtype=np.float64) * parameters.nnue2score
            )
            training_parts.append(
                np.asarray(training_output, dtype=np.float64) * parameters.nnue2score
            )
            score_parts.append(batch.scores.astype(np.float64, copy=False))
    float_cp = np.concatenate(float_parts)
    reference_cp = np.concatenate(reference_parts)
    training_cp = np.concatenate(training_parts)
    target_scores = np.concatenate(score_parts)
    raw = _quantised_raw_values(
        native_binary,
        network=network,
        psv=psv,
        progress=progress,
        positions=positions,
        batch_size=batch_size,
        threads=threads,
    )
    quantised_cp = raw.astype(np.float64) / float(fv_scale)
    vectors = (float_cp, reference_cp, training_cp, target_scores, quantised_cp)
    if (
        any(vector.shape != quantised_cp.shape for vector in vectors)
        or quantised_cp.size != positions
    ):
        raise ValueError("MLX, native, and target evaluation vectors are not aligned")
    float_probability = _wrm_probability_numpy(
        float_cp,
        scaling=parameters.in_scaling,
        offset=parameters.in_offset,
    )
    reference_probability = _wrm_probability_numpy(
        reference_cp,
        scaling=parameters.in_scaling,
        offset=parameters.in_offset,
    )
    training_probability = _wrm_probability_numpy(
        training_cp,
        scaling=parameters.in_scaling,
        offset=parameters.in_offset,
    )
    quantised_probability = _wrm_probability_numpy(
        quantised_cp,
        scaling=parameters.in_scaling,
        offset=parameters.in_offset,
    )
    target_probability = _wrm_probability_numpy(
        target_scores,
        scaling=parameters.target_scaling,
        offset=parameters.target_offset,
    )
    reference_error = _probability_error_summary(reference_probability, quantised_probability)
    training_error = _probability_error_summary(training_probability, quantised_probability)
    float_drift = _probability_error_summary(float_probability, quantised_probability)
    if reference_error["maximum_absolute"] > 3.0e-3 or reference_error["p99_absolute"] > 1.0e-4:
        raise ValueError(
            f"MLX integer reference differs from the exported native network: {reference_error}"
        )
    if training_error["maximum_absolute"] > 5.0e-3 or training_error["p99_absolute"] > 5.0e-4:
        raise ValueError(
            f"training QAT accumulation differs from the exported native network: {training_error}"
        )
    residual_cp = quantised_cp - float_cp
    reference_residual_cp = quantised_cp - reference_cp
    training_residual_cp = quantised_cp - training_cp
    reference_loss = float(np.mean(np.square(reference_probability - target_probability)))
    training_loss = float(np.mean(np.square(training_probability - target_probability)))
    native_loss = float(np.mean(np.square(quantised_probability - target_probability)))
    float_loss = float(np.mean(np.square(float_probability - target_probability)))
    if abs(reference_loss - native_loss) > 1.0e-5 or abs(training_loss - native_loss) > 1.0e-4:
        raise ValueError(
            "QAT/native held-out loss mismatch exceeded its implementation budget: "
            f"reference={reference_loss} training={training_loss} native={native_loss}"
        )
    reference_std = float(reference_cp.std())
    quantised_std = float(quantised_cp.std())
    correlation: float | None = None
    regression_slope: float | None = None
    if reference_std > 1e-9 and quantised_std > 1e-9:
        correlation = float(np.corrcoef(reference_cp, quantised_cp)[0, 1])
        regression_slope = float(
            np.mean((reference_cp - reference_cp.mean()) * (quantised_cp - quantised_cp.mean()))
            / np.var(reference_cp)
        )
    decisive = np.abs(reference_cp) >= 100.0
    sign_disagreements = int(
        np.count_nonzero(np.signbit(reference_cp[decisive]) != np.signbit(quantised_cp[decisive]))
    )
    return {
        "schema": "meteo-nagisa-mlx-quantisation-audit-v2",
        "qat_feature_accumulation_implementation": QAT_FEATURE_ACCUMULATION,
        "export_rounding_implementation": TATARA_EXPORT_ROUNDING,
        "qat_gradient_accumulation": "fp16_scatter_add_vjp",
        "positions": positions,
        "quantization_gain": 127 * 64,
        "yaneuraou_fv_scale": fv_scale,
        "nnue2score": parameters.nnue2score,
        "systematic_scale_ratio": exact_multiplier / parameters.nnue2score,
        "float_master_cp": _distribution_summary(float_cp),
        "qat_reference_cp": _distribution_summary(reference_cp),
        "qat_training_cp": _distribution_summary(training_cp),
        "quantised_cp": _distribution_summary(quantised_cp),
        "native_minus_float_master_cp": _distribution_summary(residual_cp),
        "native_minus_qat_reference_cp": _distribution_summary(reference_residual_cp),
        "native_minus_qat_training_cp": _distribution_summary(training_residual_cp),
        "implementation_reference_probability_error": reference_error,
        "training_emulation_probability_error": training_error,
        "float_master_to_deployment_probability_drift": float_drift,
        "heldout_loss": {
            "float_master": float_loss,
            "qat_reference": reference_loss,
            "qat_training": training_loss,
            "native_export": native_loss,
        },
        "pearson_correlation": correlation,
        "regression_slope_native_on_qat_reference": regression_slope,
        "decisive_abs_cp_at_least_100": int(np.count_nonzero(decisive)),
        "decisive_sign_disagreements": sign_disagreements,
        "probability_error": training_error,
        "acceptance_basis": "native_export_matches_qat_forward_not_float_master",
        "safety_gate_passed": True,
    }


def _raw_psv_scores(path: Path, positions: int) -> NDArray[np.int16]:
    byte_size = path.stat().st_size
    if positions < 1 or byte_size < positions * 40 or byte_size % 40:
        raise ValueError("raw PSV score request is outside a valid 40-byte record file")
    mapped = np.memmap(path, mode="r", dtype=np.uint8, shape=(byte_size // 40, 40))
    low = mapped[:positions, 32].astype(np.uint16)
    high = mapped[:positions, 33].astype(np.uint16)
    unsigned = low | (high << 8)
    scores = unsigned.view(np.int16).copy()
    del mapped
    return scores


def _feature_stream_smoke_receipt(
    *,
    native_binary: Path,
    heldout: Path,
    progress: Path,
    positions: int,
    batch_size: int,
    threads: int,
    parameters: WrmLossParameters,
) -> tuple[list[MlxNnueBatch], dict[str, object]]:
    batches: list[MlxNnueBatch] = []
    digest = hashlib.sha256()
    with NativeBatchStream(
        native_binary,
        psv=heldout,
        progress=progress,
        start_record=0,
        records=positions,
        batch_size=batch_size,
        threads=threads,
    ) as stream:
        for batch in stream:
            batches.append(batch)
            for array in (batch.stm, batch.nstm, batch.nnz, batch.buckets, batch.scores):
                digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    streamed_scores = np.concatenate([batch.scores for batch in batches])
    raw_scores = _raw_psv_scores(heldout, positions)
    if not np.array_equal(streamed_scores, raw_scores):
        differences = int(np.count_nonzero(streamed_scores != raw_scores))
        raise ValueError(f"native stream changed {differences} raw PSV scalar labels")
    mlx_scores = mx.array(streamed_scores.astype(np.float32))
    zero_outputs = mx.zeros_like(mlx_scores)
    _, mlx_targets = wrm_probabilities(zero_outputs, mlx_scores, parameters=parameters)
    mx.eval(mlx_targets)
    observed_targets = np.asarray(mlx_targets)
    score_values = streamed_scores.astype(np.float32)
    expected_targets = (
        0.5
        * (
            1.0
            + 1.0
            / (
                1.0
                + np.exp(
                    -(score_values - np.float32(parameters.target_offset))
                    / np.float32(parameters.target_scaling)
                )
            )
            - 1.0
            / (
                1.0
                + np.exp(
                    -(-score_values - np.float32(parameters.target_offset))
                    / np.float32(parameters.target_scaling)
                )
            )
        )
    ).astype(np.float32)
    transform_error = float(np.max(np.abs(observed_targets - expected_targets)))
    if transform_error > 2.0e-7:
        raise ValueError(
            f"MLX WRM target transform drifted from float32 reference: {transform_error}"
        )
    identity_targets = (
        1.0 / (1.0 + np.exp(-score_values / np.float32(parameters.target_scaling)))
    ).astype(np.float32)
    identity_error = float(np.max(np.abs(observed_targets - identity_targets)))
    if parameters.target_offset != 0.0 or identity_error > 2.0e-7:
        raise ValueError(
            "configured WRM no longer preserves the sigmoid teacher distribution: "
            f"offset={parameters.target_offset} max_error={identity_error}"
        )
    nnz = np.concatenate([batch.nnz for batch in batches])
    buckets = np.concatenate([batch.buckets for batch in batches])
    if int(nnz.min()) < 1 or int(nnz.max()) > MAX_ACTIVE:
        raise ValueError("Tatara feature stream emitted an invalid active-feature count")
    if int(buckets.min()) < 0 or int(buckets.max()) >= NUM_BUCKETS:
        raise ValueError("Tatara progress stream emitted an invalid LayerStack bucket")
    return batches, {
        "positions": positions,
        "native_payload_sha256": digest.hexdigest(),
        "raw_score_exact_match": True,
        "raw_score_minimum": int(streamed_scores.min()),
        "raw_score_maximum": int(streamed_scores.max()),
        "mate_stamp_abs_32000_or_more": int(
            np.count_nonzero(np.abs(streamed_scores.astype(np.int32)) >= 32_000)
        ),
        "active_features": {"minimum": int(nnz.min()), "maximum": int(nnz.max())},
        "progress_bucket_counts": [
            int(value) for value in np.bincount(buckets, minlength=NUM_BUCKETS)[:NUM_BUCKETS]
        ],
        "target_transform": "Tatara WRM target",
        "wrm": parameters.to_dict(),
        "target_transform_max_abs_error_vs_numpy_float32": transform_error,
        "target_distribution_max_abs_error_vs_sigmoid_score_scale": identity_error,
    }


def smoke_mlx_backend(
    run_directory: Path,
    *,
    positions: int = DEFAULT_SMOKE_POSITIONS,
    training_steps: int = 12,
) -> dict[str, object]:
    """Exercise real PSV decode, MLX updates, save/reload, and NNUE export."""

    root = run_directory.expanduser().resolve(strict=True)
    mlx_plan = load_mlx_plan(root)
    common = load_nagisa_plan(root)
    execution = cast(dict[str, Any], mlx_plan["execution"])
    training = cast(dict[str, Any], common["training"])
    parameters = WrmLossParameters.from_training(training)
    batch_size = positions
    if positions < 16 or training_steps < 6 or positions > int(execution["validation_positions"]):
        raise ValueError("MLX smoke positions/training steps are outside the safe range")
    if positions % 16:
        raise ValueError("MLX smoke positions must be divisible by 16")
    receipt_path = root / "receipts" / "mlx-smoke.json"
    if receipt_path.is_file():
        existing_receipt = _strict_json(receipt_path.read_bytes(), label="MLX smoke receipt")
        if existing_receipt.get("schema") == "meteo-nagisa-mlx-smoke-v1" and existing_receipt.get(
            "mlx_plan_sha256"
        ) == _sha256_file(root / "mlx-plan.json"):
            return cast(dict[str, object], existing_receipt)
        raise ValueError("existing MLX smoke receipt belongs to a different immutable plan")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("partial MLX smoke receipt requires operator inspection")
    hardware = mlx_hardware_preflight(root)
    if not hardware["ready"]:
        raise RuntimeError(f"MLX smoke preflight failed: {hardware['reasons']}")
    memory_limit = int(execution["mlx_memory_limit_bytes"])
    cache_limit = int(execution["mlx_cache_limit_bytes"])
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(cache_limit)
    mx.reset_peak_memory()
    native_binary = _native_binary_from_plan(root, mlx_plan)
    heldout = root / "private" / "heldout.psv"
    progress = root / "private" / "progress.bin"
    threads = int(execution["native_decode_threads"])
    batches, stream_receipt = _feature_stream_smoke_receipt(
        native_binary=native_binary,
        heldout=heldout,
        progress=progress,
        positions=positions,
        batch_size=batch_size,
        threads=threads,
        parameters=parameters,
    )
    batch = batches[0]
    model, initialization = initialise_exact_tatara_model(
        native_binary, temporary_parent=root / "private"
    )
    initialization_path = root / "receipts" / "mlx-tatara-initialization.json"
    if initialization_path.is_file():
        prior = _strict_json(initialization_path.read_bytes(), label="Tatara initialization")
        if prior.get("initializer_sha256") != initialization.get("initializer_sha256"):
            raise ValueError("Tatara deterministic initializer changed between invocations")
    elif initialization_path.exists() or initialization_path.is_symlink():
        raise FileExistsError("partial Tatara initialization receipt requires inspection")
    else:
        _write_new(initialization_path, _json_bytes(initialization), mode=0o600)
    optimizer = _new_optimizer(model, learning_rate=_learning_rate(common, 1))
    initial_arrays = _model_arrays(model)
    initial_probe = mx.array(initial_arrays["l3_b"])
    mx.eval(initial_probe)
    before = _evaluate_mlx_batch(model, batch, parameters=parameters)
    optimization_step = _compiled_training_step(
        model,
        optimizer,
        parameters=parameters,
        forward_mode="mixed_precision_warmup",
    )
    train_losses: list[float] = []
    started = time.perf_counter()
    mlx_batch = batch.mlx()
    learning_rate = mx.array(_learning_rate(common, 1), dtype=mx.float32)
    for _ in range(training_steps):
        loss, output_mean, target_mean = optimization_step(learning_rate, *mlx_batch)
        mx.eval(loss, output_mean, target_mean, model.state, optimizer.state)
        observed_loss = float(loss.item())
        if not math.isfinite(observed_loss):
            raise FloatingPointError("MLX smoke training produced a non-finite loss")
        train_losses.append(observed_loss)
    warmup_after = _evaluate_mlx_batch(
        model,
        batch,
        parameters=parameters,
        forward_mode="mixed_precision_warmup",
    )
    if warmup_after[0] > before[0] + 1e-7:
        raise ValueError(
            "MLX smoke warm-up loss increased on its repeated real batch: "
            f"before={before[0]} after={warmup_after[0]}"
        )
    qat_probe_steps = 2
    optimization_step = _compiled_training_step(
        model,
        optimizer,
        parameters=parameters,
        forward_mode="quantisation_aware",
    )
    qat_losses: list[float] = []
    for _ in range(qat_probe_steps):
        loss, output_mean, target_mean = optimization_step(learning_rate, *mlx_batch)
        mx.eval(loss, output_mean, target_mean, model.state, optimizer.state)
        observed_loss = float(loss.item())
        if not math.isfinite(observed_loss):
            raise FloatingPointError("MLX QAT smoke training produced a non-finite loss")
        qat_losses.append(observed_loss)
    elapsed = max(time.perf_counter() - started, 1e-9)
    after = _evaluate_mlx_batch(model, batch, parameters=parameters)
    qat_after = _evaluate_mlx_batch(
        model,
        batch,
        parameters=parameters,
        forward_mode="quantised_reference",
    )
    parameter_delta = mx.max(mx.abs(_model_arrays(model)["l3_b"] - initial_probe))
    mx.eval(parameter_delta)
    parameter_delta_value = float(parameter_delta.item())
    if not math.isfinite(parameter_delta_value) or parameter_delta_value <= 0.0:
        raise ValueError("MLX smoke optimizer did not change the output-layer bias")
    total_steps = training_steps + qat_probe_steps
    with tempfile.TemporaryDirectory(prefix=".meteo-mlx-smoke-", dir=root / "private") as raw_tmp:
        temporary = Path(raw_tmp)
        temporary_checkpoints = temporary / "checkpoints"
        temporary_exports = temporary / "exports"
        temporary_checkpoints.mkdir()
        temporary_exports.mkdir()
        checkpoint, manifest = save_mlx_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            optimizer_step=total_steps,
            superbatch=1,
            completed_batches_in_superbatch=total_steps,
            presentations=total_steps * positions,
            parent=temporary_checkpoints,
        )
        reloaded_model, reloaded_optimizer, _ = load_mlx_checkpoint(
            checkpoint, learning_rate=_learning_rate(common, 1)
        )
        reloaded = _evaluate_mlx_batch(reloaded_model, batch, parameters=parameters)
        if max(abs(a - b) for a, b in zip(after, reloaded, strict=True)) > 1e-9:
            raise ValueError(f"MLX checkpoint reload changed outputs: {after} != {reloaded}")
        qat_reloaded = _evaluate_mlx_batch(
            reloaded_model,
            batch,
            parameters=parameters,
            forward_mode="quantised_reference",
        )
        if max(abs(a - b) for a, b in zip(qat_after, qat_reloaded, strict=True)) > 1e-9:
            raise ValueError(
                "MLX checkpoint reload changed the QAT deployment graph: "
                f"{qat_after} != {qat_reloaded}"
            )
        if int(reloaded_optimizer.state["step"].item()) != total_steps:
            raise ValueError("MLX checkpoint reload changed the Ranger optimizer step")
        exported, export_receipt = export_mlx_generation(
            root,
            checkpoint=checkpoint,
            optimizer_step=total_steps,
            native_binary=native_binary,
            parent=temporary_exports,
        )
        quantisation = audit_quantised_distribution(
            model,
            network=exported / "network.bin",
            psv=heldout,
            progress=progress,
            native_binary=native_binary,
            positions=positions,
            batch_size=positions,
            parameters=parameters,
            fv_scale=int(training["yaneuraou_fv_scale"]),
            threads=threads,
        )
        export_identity = {
            "architecture": export_receipt["architecture"],
            "tatara_bin": export_receipt["tatara_bin"],
            "nn_bin": export_receipt["nn_bin"],
            "progress_bin_sha256": export_receipt["progress_bin_sha256"],
            "directory_created": exported.name,
        }
        checkpoint_identity = {
            "model_bytes": cast(dict[str, Any], manifest["model_file"])["bytes"],
            "optimizer_bytes": cast(dict[str, Any], manifest["optimizer_file"])["bytes"],
            "reload_loss_exact_within": 1e-9,
            "qat_reload_loss_exact_within": 1e-9,
            "optimizer_step_restored": total_steps,
        }
    smoke_receipt: dict[str, object] = {
        "schema": "meteo-nagisa-mlx-smoke-v1",
        "created_unix": time.time(),
        "mlx_plan_sha256": _sha256_file(root / "mlx-plan.json"),
        "native_binary_sha256": _sha256_file(native_binary),
        "real_psv": stream_receipt,
        "initialization": initialization,
        "training": {
            "steps": total_steps,
            "mixed_precision_warmup_steps": training_steps,
            "qat_probe_steps": qat_probe_steps,
            "positions_per_step": positions,
            "before_loss": before[0],
            "after_loss": after[0],
            "per_step_loss": train_losses,
            "qat_probe_loss": qat_losses,
            "l3_bias_max_abs_parameter_delta": parameter_delta_value,
            "positions_per_second_including_initial_compile": positions * total_steps / elapsed,
            "elapsed_including_initial_compile_seconds": elapsed,
            "peak_mlx_memory_bytes": mx.get_peak_memory(),
            "finite": True,
        },
        "checkpoint": checkpoint_identity,
        "export": export_identity,
        "quantisation": quantisation,
        "checks": {
            "raw_scalar_labels_exact": True,
            "policy_targets_used": False,
            "mate_stamps_retained": True,
            "save_reload_verified": True,
            "optimizer_resume_verified": True,
            "yaneuraou_architecture_verified": True,
            "target_distribution_preserved": True,
            "quantised_distribution_gate_passed": True,
            "integer_equivalent_qat_update_executed": True,
            "cloud_gpu_used": False,
        },
    }
    _write_new(receipt_path, _json_bytes(smoke_receipt), mode=0o600)
    _fsync_directory(receipt_path.parent)
    return smoke_receipt


def _download_mlx_source_with_progress(
    root: Path,
    shard: PublicPsvShard,
    *,
    timeout_seconds: float,
) -> Path:
    stop = threading.Event()
    cache = root / "cache"
    part = cache / f"{shard.filename}.part"
    destination = cache / shard.filename

    def report() -> None:
        last_bytes = 0
        last_time = time.monotonic()
        while not stop.wait(15.0):
            path = destination if destination.is_file() else part
            observed = path.stat().st_size if path.is_file() and not path.is_symlink() else 0
            now = time.monotonic()
            elapsed = max(now - last_time, 1e-9)
            speed = max(0, observed - last_bytes) / elapsed
            remaining = max(0, shard.byte_size - observed)
            _log_mlx_event(
                root,
                "source_download_progress",
                source=shard.filename,
                bytes=observed,
                total_bytes=shard.byte_size,
                fraction=observed / shard.byte_size,
                bytes_per_second=speed,
                eta_seconds=(remaining / speed) if speed > 0 else None,
                free_storage_bytes=shutil.disk_usage(cache).free,
            )
            last_bytes = observed
            last_time = now

    monitor = threading.Thread(target=report, name="meteo-source-download-monitor", daemon=True)
    _log_mlx_event(
        root,
        "source_download_start",
        source=shard.filename,
        bytes=(
            destination.stat().st_size
            if destination.is_file() and not destination.is_symlink()
            else part.stat().st_size
            if part.is_file() and not part.is_symlink()
            else 0
        ),
        total_bytes=shard.byte_size,
        complete_cache_hit=(
            destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == shard.byte_size
        ),
    )
    monitor.start()
    try:
        maximum_attempts = 5
        for attempt in range(1, maximum_attempts + 1):
            try:
                output = download_nagisa_shard(root, shard, timeout_seconds=timeout_seconds)
                break
            except (OSError, http.client.HTTPException, urllib.error.URLError) as error:
                if attempt == maximum_attempts:
                    raise
                retry_seconds = min(60.0, 5.0 * (2 ** (attempt - 1)))
                _log_mlx_event(
                    root,
                    "source_download_retry",
                    source=shard.filename,
                    failed_attempt=attempt,
                    maximum_attempts=maximum_attempts,
                    retry_seconds=retry_seconds,
                    error_type=type(error).__name__,
                    error=str(error),
                    resumable_bytes=part.stat().st_size if part.is_file() else 0,
                )
                time.sleep(retry_seconds)
        else:  # pragma: no cover - the loop either returns a path or raises.
            raise AssertionError("source download retry loop exited without a result")
    finally:
        stop.set()
        monitor.join(timeout=5)
    _log_mlx_event(
        root,
        "source_download_complete",
        source=shard.filename,
        bytes=output.stat().st_size,
        sha256=shard.sha256,
    )
    return output


def _cached_source_shard_count(cache: Path) -> int:
    return sum(1 for path in cache.glob("split_*.bin") if path.is_file() and not path.is_symlink())


def _start_source_prefetch(
    root: Path,
    shard: PublicPsvShard,
    *,
    timeout_seconds: float,
) -> DaemonPrefetch[Path] | None:
    cache = root / "cache"
    destination = cache / shard.filename
    part = cache / f"{shard.filename}.part"
    existing_bytes = 0
    if destination.is_file() and not destination.is_symlink():
        existing_bytes = destination.stat().st_size
    elif part.is_file() and not part.is_symlink():
        existing_bytes = part.stat().st_size
    remaining = max(0, shard.byte_size - existing_bytes)
    free = shutil.disk_usage(cache).free
    required = remaining + SOURCE_PREFETCH_STORAGE_RESERVE_BYTES
    if free < required:
        _log_mlx_event(
            root,
            "source_prefetch_skipped_storage_guard",
            source=shard.filename,
            remaining_download_bytes=remaining,
            required_free_bytes=required,
            observed_free_bytes=free,
        )
        return None
    _log_mlx_event(
        root,
        "source_prefetch_scheduled",
        source=shard.filename,
        remaining_download_bytes=remaining,
        observed_free_bytes=free,
    )
    return DaemonPrefetch(
        name=f"meteo-prefetch-{shard.filename}",
        producer=lambda: _download_mlx_source_with_progress(
            root,
            shard,
            timeout_seconds=timeout_seconds,
        ),
    )


def _source_probe_receipt(root: Path, shard: PublicPsvShard, cached: Path) -> dict[str, object]:
    path = root / "receipts" / f"source-{shard.sha256}.json"
    if path.is_file():
        existing_receipt = _strict_json(path.read_bytes(), label="MLX source probe receipt")
        source = existing_receipt.get("source")
        if (
            existing_receipt.get("schema") != "meteo-nagisa-source-probe-v1"
            or not isinstance(source, dict)
            or source.get("sha256") != shard.sha256
        ):
            raise ValueError(f"source probe receipt does not match {shard.filename}")
        return cast(dict[str, object], existing_receipt)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"partial source receipt requires inspection: {path}")
    _log_mlx_event(root, "source_probe_start", source=shard.filename)
    probe = probe_value_only_psv(cached)
    source_receipt: dict[str, object] = {
        "schema": "meteo-nagisa-source-probe-v1",
        "source": shard.to_dict(),
        "probe": probe,
    }
    _write_new(path, _json_bytes(source_receipt), mode=0o600)
    _log_mlx_event(
        root,
        "source_probe_complete",
        source=shard.filename,
        records=probe["records"],
        score=probe["score"],
        policy_targets=0,
    )
    return source_receipt


def _release_consumed_source_cache(root: Path, shard: PublicPsvShard) -> bool:
    """Delete exactly one receipted shard after its superbatch is durable."""

    cache = (root / "cache").resolve(strict=True)
    cached = cache / shard.filename
    if not cached.exists() and not cached.is_symlink():
        return False
    if cached.is_symlink() or not cached.is_file() or cached.parent.resolve(strict=True) != cache:
        raise ValueError(f"consumed source cache target is not a regular child: {cached}")
    if cached.stat().st_size != shard.byte_size:
        raise ValueError(f"consumed source cache size changed before release: {cached}")
    receipt_path = root / "receipts" / f"source-{shard.sha256}.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ValueError(f"consumed source cache has no regular source receipt: {cached}")
    receipt = _strict_json(receipt_path.read_bytes(), label="MLX source probe receipt")
    probe = receipt.get("probe")
    if (
        receipt.get("schema") != "meteo-nagisa-source-probe-v1"
        or receipt.get("source") != shard.to_dict()
        or not isinstance(probe, dict)
        or probe.get("bytes") != shard.byte_size
        or probe.get("records") != shard.records
        or probe.get("sha256") != shard.sha256
    ):
        raise ValueError(f"consumed source cache receipt does not match: {cached}")
    cached.unlink()
    _fsync_directory(cache)
    _log_mlx_event(
        root,
        "source_cache_released",
        source=shard.filename,
        free_storage_bytes=shutil.disk_usage(root).free,
        cached_source_shards=_cached_source_shard_count(cache),
    )
    return True


def _remove_partial_mlx_stages(root: Path) -> list[str]:
    removed: list[str] = []
    token = f".{DEFAULT_NET_ID}-mlx-step-"
    for parent in (root / "checkpoints", root / "exports"):
        for path in parent.iterdir():
            if not path.name.startswith(token):
                continue
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected partial MLX stage type: {path}")
            shutil.rmtree(path)
            removed.append(str(path))
        _fsync_directory(parent)
    return removed


def _completed_superbatch(schedule: Sequence[Mapping[str, object]], optimizer_step: int) -> int:
    consumed = 0
    completed = 0
    for segment in schedule:
        consumed += int(cast(int, segment["batches"]))
        if optimizer_step < consumed:
            break
        completed = int(cast(int, segment["superbatch"]))
    return completed


def _validate_checkpoint_reload(
    root: Path,
    *,
    checkpoint: Path,
    reference_model: MeteoValueNnue,
    reference_batch: MlxNnueBatch,
    parameters: WrmLossParameters,
    learning_rate: float,
) -> tuple[MeteoValueNnue, TataraRanger, dict[str, object]]:
    mlx_plan = load_mlx_plan(root)
    modes: tuple[ModelForwardMode, ...]
    active: NumericPhase | None = None
    if mlx_plan.get("schema") == MLX_BACKEND_PLAN_SCHEMA:
        step = int(
            cast(
                dict[str, Any],
                _strict_json(
                    (checkpoint / "manifest.json").read_bytes(),
                    label="MLX checkpoint reload manifest",
                )["optimizer"],
            )["step"]
        )
        active = _quantisation_schedule_from_plan(mlx_plan).phase_for_completed_steps(step)
        modes = ("float32", active, "quantised_reference")
    else:
        modes = ("float32",)
    expected = {
        mode: _evaluate_mlx_batch(
            reference_model,
            reference_batch,
            parameters=parameters,
            forward_mode=mode,
        )
        for mode in dict.fromkeys(modes)
    }
    model, optimizer, manifest = load_mlx_checkpoint(checkpoint, learning_rate=learning_rate)
    observed = {
        mode: _evaluate_mlx_batch(
            model,
            reference_batch,
            parameters=parameters,
            forward_mode=mode,
        )
        for mode in expected
    }
    differences = {
        mode: max(abs(a - b) for a, b in zip(expected[mode], observed[mode], strict=True))
        for mode in expected
    }
    maximum_difference = max(differences.values())
    if maximum_difference > 1e-9:
        raise ValueError(
            f"published MLX checkpoint changed evaluation after reload: {maximum_difference}"
        )
    qat_reference_max_abs_difference: float | None = None
    if active == "quantisation_aware":
        stm, nstm, nnz, buckets, _ = reference_batch.mlx()
        qat_output = model.quantisation_aware_forward(stm, nstm, nnz, buckets)
        reference_output = model.quantised_reference_forward(stm, nstm, nnz, buckets)
        difference = mx.max(mx.abs(qat_output - reference_output))
        mx.eval(difference)
        qat_reference_max_abs_difference = float(difference.item())
        if qat_reference_max_abs_difference > 1.0e-7:
            raise ValueError(
                "checkpoint QAT forward differs from its independent float32 reference: "
                f"{qat_reference_max_abs_difference}"
            )
    _log_mlx_event(
        root,
        "checkpoint_reload_verified",
        checkpoint=str(checkpoint),
        optimizer_step=int(optimizer.state["step"].item()),
        evaluation_max_abs_difference=maximum_difference,
        evaluation_max_abs_difference_by_forward_mode=differences,
        qat_reference_max_abs_difference=qat_reference_max_abs_difference,
    )
    return model, optimizer, manifest


def _ensure_mlx_export(
    root: Path,
    *,
    checkpoint: Path,
    optimizer_step: int,
    native_binary: Path,
) -> tuple[Path, dict[str, object]]:
    destination = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    if not destination.exists():
        return export_mlx_generation(
            root,
            checkpoint=checkpoint,
            optimizer_step=optimizer_step,
            native_binary=native_binary,
        )
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"MLX export must be a regular directory: {destination}")
    receipt = _strict_json((destination / "receipt.json").read_bytes(), label="MLX export")
    model_file = checkpoint / "model.safetensors"
    if (
        receipt.get("schema") != "meteo-nagisa-mlx-yaneuraou-export-v1"
        or receipt.get("optimizer_step") != optimizer_step
        or receipt.get("checkpoint_model_sha256") != _sha256_file(model_file)
    ):
        raise ValueError(f"existing MLX export does not match checkpoint {checkpoint}")
    for filename, key in (("network.bin", "tatara_bin"), ("nn.bin", "nn_bin")):
        identity = receipt.get(key)
        artifact = destination / filename
        if (
            not isinstance(identity, dict)
            or not artifact.is_file()
            or artifact.is_symlink()
            or identity.get("bytes") != artifact.stat().st_size
            or identity.get("sha256") != _sha256_file(artifact)
        ):
            raise ValueError(f"existing MLX export artifact is corrupt: {artifact}")
    return destination, cast(dict[str, object], receipt)


def _finalize_mlx_segment(
    root: Path,
    *,
    common: Mapping[str, object],
    mlx_plan: Mapping[str, object],
    segment: Mapping[str, object],
    shard: PublicPsvShard,
    model: MeteoValueNnue,
    checkpoint: Path,
    optimizer_step: int,
    segment_train_metrics: Mapping[str, object] | None,
) -> tuple[Path, dict[str, object]]:
    superbatch = int(cast(int, segment["superbatch"]))
    receipt_path = root / "receipts" / f"mlx-superbatch-{superbatch:04d}.json"
    if receipt_path.is_file():
        existing_receipt = _strict_json(receipt_path.read_bytes(), label="MLX superbatch receipt")
        export_path = Path(cast(str, cast(dict[str, Any], existing_receipt["export"])["path"]))
        return export_path.resolve(strict=True), cast(dict[str, object], existing_receipt)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"partial MLX superbatch receipt requires inspection: {receipt_path}")
    execution = cast(dict[str, Any], mlx_plan["execution"])
    training = cast(dict[str, Any], common["training"])
    parameters = WrmLossParameters.from_training(training)
    validation_positions = int(execution["validation_positions"])
    _log_mlx_event(
        root,
        "heldout_validation_start",
        superbatch=superbatch,
        optimizer_step=optimizer_step,
        positions=validation_positions,
    )
    validation = evaluate_mlx_psv(
        model,
        psv=root / "private" / "heldout.psv",
        progress=root / "private" / "progress.bin",
        native_binary=_native_binary_from_plan(root, mlx_plan),
        positions=validation_positions,
        batch_size=int(execution["batch_size"]),
        parameters=parameters,
        threads=int(execution["native_decode_threads"]),
        forward_mode="quantised_reference",
    )
    _log_mlx_event(
        root,
        "heldout_validation_complete",
        superbatch=superbatch,
        optimizer_step=optimizer_step,
        validation_loss=validation["loss"],
        positions_per_second=validation["positions_per_second"],
    )
    export, export_receipt = _ensure_mlx_export(
        root,
        checkpoint=checkpoint,
        optimizer_step=optimizer_step,
        native_binary=_native_binary_from_plan(root, mlx_plan),
    )
    quantisation_positions = int(execution["quantisation_audit_positions"])
    quantisation = audit_quantised_distribution(
        model,
        network=export / "network.bin",
        psv=root / "private" / "heldout.psv",
        progress=root / "private" / "progress.bin",
        native_binary=_native_binary_from_plan(root, mlx_plan),
        positions=quantisation_positions,
        batch_size=quantisation_positions,
        parameters=parameters,
        fv_scale=int(training["yaneuraou_fv_scale"]),
        threads=int(execution["native_decode_threads"]),
    )
    _log_mlx_event(
        root,
        "quantisation_distribution_verified",
        superbatch=superbatch,
        optimizer_step=optimizer_step,
        positions=quantisation_positions,
        probability_error=quantisation["probability_error"],
        implementation_reference_probability_error=quantisation[
            "implementation_reference_probability_error"
        ],
        float_master_to_deployment_probability_drift=quantisation[
            "float_master_to_deployment_probability_drift"
        ],
        systematic_scale_ratio=quantisation["systematic_scale_ratio"],
    )
    manifest = _strict_json((checkpoint / "manifest.json").read_bytes(), label="MLX manifest")
    superbatch_receipt: dict[str, object] = {
        "schema": "meteo-nagisa-mlx-superbatch-receipt-v1",
        "created_unix": time.time(),
        "superbatch": superbatch,
        "optimizer_step": optimizer_step,
        "segment": dict(segment),
        "source": shard.to_dict(),
        "label_contract": {
            "scalar_score": "PackedSfenValue.score_i16",
            "policy_targets": 0,
            "wdl_lambda": 0.0,
            "mate_stamps_retained": True,
        },
        "training_metrics": dict(segment_train_metrics or {}),
        "heldout": validation,
        "quantisation": quantisation,
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": _sha256_file(checkpoint / "manifest.json"),
            "model": manifest["model_file"],
            "optimizer": manifest["optimizer_file"],
        },
        "export": {"path": str(export), "receipt": export_receipt},
    }
    _write_new(receipt_path, _json_bytes(superbatch_receipt), mode=0o600)
    _fsync_directory(receipt_path.parent)
    _log_mlx_event(
        root,
        "superbatch_complete",
        superbatch=superbatch,
        optimizer_step=optimizer_step,
        presentations=optimizer_step * int(execution["batch_size"]),
        heldout_loss=validation["loss"],
        checkpoint=str(checkpoint),
        export=str(export),
    )
    return export, superbatch_receipt


def _publish_mlx_checkpoint(
    root: Path,
    *,
    model: MeteoValueNnue,
    optimizer: TataraRanger,
    optimizer_step: int,
    superbatch: int,
    completed_batches: int,
    batch_size: int,
    reference_batch: MlxNnueBatch,
    parameters: WrmLossParameters,
    learning_rate: float,
) -> tuple[MeteoValueNnue, TataraRanger, Path, dict[str, object]]:
    checkpoint, _ = save_mlx_checkpoint(
        root,
        model=model,
        optimizer=optimizer,
        optimizer_step=optimizer_step,
        superbatch=superbatch,
        completed_batches_in_superbatch=completed_batches,
        presentations=optimizer_step * batch_size,
    )
    reloaded_model, reloaded_optimizer, manifest = _validate_checkpoint_reload(
        root,
        checkpoint=checkpoint,
        reference_model=model,
        reference_batch=reference_batch,
        parameters=parameters,
        learning_rate=learning_rate,
    )
    _prune_mlx_checkpoints(root / "checkpoints", keep=2)
    return reloaded_model, reloaded_optimizer, checkpoint, manifest


def _set_mlx_state(root: Path, state: dict[str, Any], **values: object) -> None:
    state.update(values)
    state["updated_unix"] = time.time()
    _write_json_atomic(root / "mlx-state.json", state)


def run_mlx_backend(
    run_directory: Path,
    *,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Run or exactly resume the local Apple-GPU 100B value-only schedule."""

    root = run_directory.expanduser().resolve(strict=True)
    common = load_nagisa_plan(root)
    mlx_plan = load_mlx_plan(root)
    execution = cast(dict[str, Any], mlx_plan["execution"])
    batch_size = int(execution["batch_size"])
    quantisation_schedule = _quantisation_schedule_from_plan(mlx_plan)
    training = cast(dict[str, Any], common["training"])
    parameters = WrmLossParameters.from_training(training)
    schedule = cast(list[dict[str, object]], common["schedule"])
    shards = {
        shard.filename: shard
        for shard in _shards_from_index(cast(dict[str, Any], common["corpus"]))
    }
    preflight = mlx_hardware_preflight(root)
    state = load_mlx_state(root)
    if not preflight["ready"]:
        _set_mlx_state(
            root,
            state,
            status="failed_preflight",
            pid=None,
            failure={"kind": "mlx_hardware_preflight", "details": preflight},
        )
        raise RuntimeError(f"MLX production preflight failed: {preflight['reasons']}")
    mx.set_memory_limit(int(execution["mlx_memory_limit_bytes"]))
    mx.set_cache_limit(int(execution["mlx_cache_limit_bytes"]))
    mx.reset_peak_memory()
    native_binary = _native_binary_from_plan(root, mlx_plan)
    stop_requested = threading.Event()

    def request_signal_stop(signum: int, _frame: object) -> None:
        stop_requested.set()
        _log_mlx_event(root, "signal_stop_requested", signal=signum)

    old_sigint = signal.signal(signal.SIGINT, request_signal_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_signal_stop)
    try:
        with _exclusive_run_lock(root):
            state = load_mlx_state(root)
            removed_stages = _remove_partial_mlx_stages(root)
            if removed_stages:
                _log_mlx_event(root, "partial_stages_removed", paths=removed_stages)
            latest = _latest_mlx_checkpoint(root / "checkpoints")
            if latest is None:
                model, initialization = initialise_exact_tatara_model(
                    native_binary, temporary_parent=root / "private"
                )
                initialization_receipt = _strict_json(
                    (root / "receipts" / "mlx-tatara-initialization.json").read_bytes(),
                    label="Tatara initialization receipt",
                )
                if initialization.get("initializer_sha256") != initialization_receipt.get(
                    "initializer_sha256"
                ):
                    raise ValueError("production initialization differs from passed MLX smoke")
                optimizer = _new_optimizer(model, learning_rate=_learning_rate(common, 1))
                with NativeBatchStream(
                    native_binary,
                    psv=root / "private" / "heldout.psv",
                    progress=root / "private" / "progress.bin",
                    start_record=0,
                    records=batch_size,
                    batch_size=batch_size,
                    threads=int(execution["native_decode_threads"]),
                ) as initial_batches:
                    reference_batch = next(initial_batches)
                model, optimizer, latest, _ = _publish_mlx_checkpoint(
                    root,
                    model=model,
                    optimizer=optimizer,
                    optimizer_step=0,
                    superbatch=1,
                    completed_batches=0,
                    batch_size=batch_size,
                    reference_batch=reference_batch,
                    parameters=parameters,
                    learning_rate=_learning_rate(common, 1),
                )
                initial_export, _ = _ensure_mlx_export(
                    root,
                    checkpoint=latest,
                    optimizer_step=0,
                    native_binary=native_binary,
                )
                _log_mlx_event(
                    root,
                    "production_random_initialization_published",
                    checkpoint=str(latest),
                    export=str(initial_export),
                    teacher_nnue_weights_copied=False,
                )
            else:
                repair = _ensure_mlx_numeric_repair_receipt(
                    root,
                    plan=mlx_plan,
                    state=state,
                    checkpoint=latest,
                )
                if repair is not None:
                    _log_mlx_event(
                        root,
                        "numeric_repair_verified",
                        from_feature_accumulation=repair["from_feature_accumulation"],
                        to_feature_accumulation=repair["to_feature_accumulation"],
                        source_checkpoint=repair["source_checkpoint"],
                    )
                rounding_repair = _ensure_mlx_export_rounding_repair_receipt(
                    root,
                    plan=mlx_plan,
                    state=state,
                    checkpoint=latest,
                )
                if rounding_repair is not None:
                    _log_mlx_event(
                        root,
                        "export_rounding_repair_verified",
                        from_export_rounding=rounding_repair["from_export_rounding"],
                        to_export_rounding=rounding_repair["to_export_rounding"],
                        source_checkpoint=rounding_repair["source_checkpoint"],
                    )
                manifest = _strict_json(
                    (latest / "manifest.json").read_bytes(), label="MLX manifest"
                )
                optimizer_step = int(cast(dict[str, Any], manifest["optimizer"])["step"])
                coordinate_index, coordinate_superbatch, _ = _coordinate_for_step(
                    schedule, optimizer_step
                )
                learning_superbatch = min(coordinate_superbatch, len(schedule))
                model, optimizer, _ = load_mlx_checkpoint(
                    latest, learning_rate=_learning_rate(common, learning_superbatch)
                )
                if int(optimizer.state["step"].item()) != optimizer_step:
                    raise ValueError("latest MLX checkpoint optimizer step failed to reload")
                _log_mlx_event(
                    root,
                    "production_resume_verified",
                    checkpoint=str(latest),
                    optimizer_step=optimizer_step,
                    schedule_index=coordinate_index,
                )
            optimizer_step = int(optimizer.state["step"].item())
            completed = _completed_superbatch(schedule, optimizer_step)
            _set_mlx_state(
                root,
                state,
                status="running",
                pid=os.getpid(),
                durable_optimizer_step=optimizer_step,
                durable_presentations=optimizer_step * batch_size,
                observed_optimizer_step=optimizer_step,
                observed_presentations=optimizer_step * batch_size,
                completed_superbatch=completed,
                latest_checkpoint=str(latest),
                failure=None,
            )
            # If a crash happened after the boundary checkpoint but before its
            # receipt/export, finish that transaction before reading new data.
            if completed > 0:
                previous = schedule[completed - 1]
                previous_shard = shards[cast(str, previous["filename"])]
                previous_receipt = root / "receipts" / f"mlx-superbatch-{completed:04d}.json"
                if not previous_receipt.is_file():
                    export, _ = _finalize_mlx_segment(
                        root,
                        common=common,
                        mlx_plan=mlx_plan,
                        segment=previous,
                        shard=previous_shard,
                        model=model,
                        checkpoint=latest,
                        optimizer_step=optimizer_step,
                        segment_train_metrics=None,
                    )
                    _set_mlx_state(root, state, latest_export=str(export))
                    _prune_mlx_exports(root / "exports", keep=2)
                _release_consumed_source_cache(root, previous_shard)
            schedule_index, _, batch_offset = _coordinate_for_step(schedule, optimizer_step)
            if schedule_index == len(schedule):
                _set_mlx_state(root, state, status="complete", pid=None)
                return mlx_run_status(root)
            active_schedule = schedule[schedule_index:]
            source_prefetch: DaemonPrefetch[Path] | None = None
            prefetched_filename: str | None = None
            for relative_index, segment in enumerate(active_schedule):
                superbatch = int(cast(int, segment["superbatch"]))
                segment_batches = int(cast(int, segment["batches"]))
                if superbatch != int(cast(int, schedule[schedule_index]["superbatch"])):
                    batch_offset = 0
                if (root / "STOP").exists() or stop_requested.is_set():
                    _set_mlx_state(root, state, status="stopped_safely", pid=None)
                    return mlx_run_status(root)
                shard = shards[cast(str, segment["filename"])]
                if source_prefetch is not None:
                    if prefetched_filename != shard.filename:
                        raise AssertionError(
                            "source prefetch target does not match the active schedule segment"
                        )
                    waited_for_prefetch = not source_prefetch.done
                    cached = source_prefetch.result()
                    _log_mlx_event(
                        root,
                        "source_prefetch_consumed",
                        source=shard.filename,
                        compute_had_to_wait=waited_for_prefetch,
                    )
                    source_prefetch = None
                    prefetched_filename = None
                else:
                    cached = _download_mlx_source_with_progress(
                        root, shard, timeout_seconds=timeout_seconds
                    )
                _source_probe_receipt(root, shard, cached)
                if relative_index + 1 < len(active_schedule):
                    next_segment = active_schedule[relative_index + 1]
                    next_shard = shards[cast(str, next_segment["filename"])]
                    source_prefetch = _start_source_prefetch(
                        root,
                        next_shard,
                        timeout_seconds=timeout_seconds,
                    )
                    if source_prefetch is not None:
                        prefetched_filename = next_shard.filename
                remaining_batches = segment_batches - batch_offset
                if remaining_batches < 1:
                    raise ValueError("MLX schedule resumed beyond the active segment")
                learning_rate_value = _learning_rate(common, superbatch)
                learning_rate = mx.array(learning_rate_value, dtype=mx.float32)
                numeric_phase = quantisation_schedule.phase_for_completed_steps(optimizer_step)
                optimization_step_fn = _compiled_training_step(
                    model,
                    optimizer,
                    parameters=parameters,
                    forward_mode=numeric_phase,
                )
                segment_started = time.perf_counter()
                interval_started = segment_started
                interval_positions = 0
                interval_loss = 0.0
                segment_loss = 0.0
                segment_positions = 0
                last_batch: MlxNnueBatch | None = None
                _log_mlx_event(
                    root,
                    "superbatch_start",
                    superbatch=superbatch,
                    source=shard.filename,
                    batch_offset=batch_offset,
                    remaining_batches=remaining_batches,
                    learning_rate=learning_rate_value,
                    numeric_phase=numeric_phase,
                    qat_warmup_optimizer_steps=quantisation_schedule.warmup_optimizer_steps,
                )
                with NativeBatchStream(
                    native_binary,
                    psv=cached,
                    progress=root / "private" / "progress.bin",
                    start_record=batch_offset * batch_size,
                    records=remaining_batches * batch_size,
                    batch_size=batch_size,
                    threads=int(execution["native_decode_threads"]),
                    prefetch=int(execution["native_prefetch_batches"]),
                ) as batches:
                    for batch in batches:
                        requested_phase = quantisation_schedule.phase_for_completed_steps(
                            optimizer_step
                        )
                        if requested_phase != numeric_phase:
                            del optimization_step_fn
                            gc.collect()
                            mx.clear_cache()
                            numeric_phase = requested_phase
                            optimization_step_fn = _compiled_training_step(
                                model,
                                optimizer,
                                parameters=parameters,
                                forward_mode=numeric_phase,
                            )
                            _log_mlx_event(
                                root,
                                "numeric_phase_start",
                                optimizer_step=optimizer_step,
                                presentations=optimizer_step * batch_size,
                                numeric_phase=numeric_phase,
                            )
                        last_batch = batch
                        loss, output_mean, target_mean = optimization_step_fn(
                            learning_rate, *batch.mlx()
                        )
                        mx.eval(loss, output_mean, target_mean, model.state, optimizer.state)
                        loss_value = float(loss.item())
                        if not math.isfinite(loss_value):
                            raise FloatingPointError(
                                f"MLX loss is non-finite at optimizer step {optimizer_step + 1}"
                            )
                        optimizer_step += 1
                        batch_offset += 1
                        segment_positions += batch.positions
                        interval_positions += batch.positions
                        segment_loss += loss_value * batch.positions
                        interval_loss += loss_value * batch.positions
                        is_segment_end = batch_offset == segment_batches
                        should_checkpoint = (
                            optimizer_step % int(execution["checkpoint_batches"]) == 0
                            or is_segment_end
                            or stop_requested.is_set()
                            or (root / "STOP").exists()
                        )
                        should_log = (
                            optimizer_step % int(execution["log_batches"]) == 0
                            or is_segment_end
                            or should_checkpoint
                        )
                        if should_log:
                            now = time.perf_counter()
                            interval_elapsed = max(now - interval_started, 1e-9)
                            metrics = _log_mlx_event(
                                root,
                                "training_progress",
                                superbatch=superbatch,
                                batch_in_superbatch=batch_offset,
                                batches_in_superbatch=segment_batches,
                                optimizer_step=optimizer_step,
                                presentations=optimizer_step * batch_size,
                                target_presentations=execution["target_presentations"],
                                loss=interval_loss / max(interval_positions, 1),
                                network_output_normalized_mean=float(output_mean.item()),
                                network_output_cp_mean=(
                                    float(output_mean.item()) * parameters.nnue2score
                                ),
                                target_probability_mean=float(target_mean.item()),
                                learning_rate=learning_rate_value,
                                numeric_phase=numeric_phase,
                                positions_per_second=interval_positions / interval_elapsed,
                                peak_mlx_memory_bytes=mx.get_peak_memory(),
                                active_mlx_memory_bytes=mx.get_active_memory(),
                                cached_mlx_memory_bytes=mx.get_cache_memory(),
                                durable_optimizer_step=int(state["durable_optimizer_step"]),
                            )
                            _set_mlx_state(
                                root,
                                state,
                                observed_optimizer_step=optimizer_step,
                                observed_presentations=optimizer_step * batch_size,
                                last_metrics=metrics,
                            )
                            interval_started = now
                            interval_positions = 0
                            interval_loss = 0.0
                        if should_checkpoint:
                            _log_mlx_event(
                                root,
                                "checkpoint_start",
                                optimizer_step=optimizer_step,
                                superbatch=superbatch,
                                batch_in_superbatch=batch_offset,
                            )
                            old_model = model
                            old_optimizer = optimizer
                            old_step_fn = optimization_step_fn
                            model, optimizer, latest, manifest = _publish_mlx_checkpoint(
                                root,
                                model=old_model,
                                optimizer=old_optimizer,
                                optimizer_step=optimizer_step,
                                superbatch=superbatch,
                                completed_batches=batch_offset,
                                batch_size=batch_size,
                                reference_batch=batch,
                                parameters=parameters,
                                learning_rate=learning_rate_value,
                            )
                            optimization_step_fn = _compiled_training_step(
                                model,
                                optimizer,
                                parameters=parameters,
                                forward_mode=numeric_phase,
                            )
                            del old_step_fn, old_optimizer, old_model
                            gc.collect()
                            mx.clear_cache()
                            _set_mlx_state(
                                root,
                                state,
                                durable_optimizer_step=optimizer_step,
                                durable_presentations=optimizer_step * batch_size,
                                latest_checkpoint=str(latest),
                            )
                            _log_mlx_event(
                                root,
                                "checkpoint_complete",
                                optimizer_step=optimizer_step,
                                checkpoint=str(latest),
                                model_bytes=cast(dict[str, Any], manifest["model_file"])["bytes"],
                                optimizer_bytes=cast(dict[str, Any], manifest["optimizer_file"])[
                                    "bytes"
                                ],
                                retained_generations=2,
                            )
                            if stop_requested.is_set() or (root / "STOP").exists():
                                _set_mlx_state(root, state, status="stopped_safely", pid=None)
                                return mlx_run_status(root)
                if last_batch is None or batch_offset != segment_batches:
                    raise ValueError("native MLX stream ended before its scheduled segment")
                elapsed = max(time.perf_counter() - segment_started, 1e-9)
                segment_metrics: dict[str, object] = {
                    "positions": segment_positions,
                    "mean_loss": segment_loss / max(segment_positions, 1),
                    "elapsed_seconds_including_checkpoints": elapsed,
                    "positions_per_second_including_checkpoints": segment_positions / elapsed,
                    "learning_rate": learning_rate_value,
                    "peak_mlx_memory_bytes": mx.get_peak_memory(),
                }
                export, receipt = _finalize_mlx_segment(
                    root,
                    common=common,
                    mlx_plan=mlx_plan,
                    segment=segment,
                    shard=shard,
                    model=model,
                    checkpoint=latest,
                    optimizer_step=optimizer_step,
                    segment_train_metrics=segment_metrics,
                )
                completed = superbatch
                _set_mlx_state(
                    root,
                    state,
                    completed_superbatch=completed,
                    latest_export=str(export),
                    last_metrics={
                        **segment_metrics,
                        "heldout_loss": cast(dict[str, Any], receipt["heldout"])["loss"],
                    },
                )
                _prune_mlx_exports(root / "exports", keep=2)
                _release_consumed_source_cache(root, shard)
                schedule_index += 1
                batch_offset = 0
            _set_mlx_state(root, state, status="complete", pid=None)
            return mlx_run_status(root)
    except BaseException as error:
        try:
            current = load_mlx_state(root)
            _set_mlx_state(
                root,
                current,
                status="failed",
                pid=None,
                failure={
                    "kind": type(error).__name__,
                    "message": str(error),
                    "repr": repr(error),
                },
            )
            _log_mlx_event(
                root,
                "training_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        except Exception:
            pass
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def mlx_run_status(run_directory: Path) -> dict[str, object]:
    root = run_directory.expanduser().resolve(strict=True)
    common = load_nagisa_plan(root)
    plan = load_mlx_plan(root)
    state = load_mlx_state(root)
    execution = cast(dict[str, Any], plan["execution"])
    target = int(execution["target_presentations"])
    durable = int(state.get("durable_presentations", 0))
    observed = int(state.get("observed_presentations", durable))
    pid_raw = state.get("pid")
    process_alive = False
    process_command: list[str] | None = None
    if isinstance(pid_raw, int) and not isinstance(pid_raw, bool) and pid_raw > 0:
        try:
            process = psutil.Process(pid_raw)
            process_command = process.cmdline()
            process_alive = process.is_running() and any(
                "run-mlx" in part for part in process_command
            )
        except (psutil.Error, OSError):
            process_alive = False
    last_metrics = state.get("last_metrics")
    speed: float | None = None
    if isinstance(last_metrics, dict):
        raw_speed = last_metrics.get("positions_per_second")
        if (
            isinstance(raw_speed, (int, float))
            and not isinstance(raw_speed, bool)
            and raw_speed > 0
        ):
            speed = float(raw_speed)
        else:
            raw_speed = last_metrics.get("positions_per_second_including_checkpoints")
            if (
                isinstance(raw_speed, (int, float))
                and not isinstance(raw_speed, bool)
                and raw_speed > 0
            ):
                speed = float(raw_speed)
    remaining = max(0, target - observed)
    eta_seconds = remaining / speed if speed else None
    eta_100m_seconds = max(0, 100_000_000 - observed) / speed if speed else None
    checkpoint_paths = _prune_free_checkpoint_listing(root / "checkpoints")
    export_paths = _prune_free_checkpoint_listing(root / "exports")
    training = cast(dict[str, Any], common["training"])
    parameters = WrmLossParameters.from_training(training)
    return {
        "schema": "meteo-nagisa-mlx-status-v1",
        "run_directory": str(root),
        "status": state.get("status"),
        "process_alive": process_alive,
        "pid": pid_raw,
        "process_command": process_command,
        "backend": "local Apple GPU via MLX",
        "cloud_gpu_used": False,
        "mode": "value_only_nnue_sfnn",
        "policy_head": False,
        "loss_kind": "Tatara_WRM_non_extended_MSE",
        "target_transform": f"sigmoid(score/{training['score_scale']})",
        "network_output_cp_multiplier": parameters.nnue2score,
        "yaneuraou_fv_scale": training["yaneuraou_fv_scale"],
        "durable_presentations": durable,
        "observed_presentations": observed,
        "target_presentations": target,
        "completion_fraction": observed / target,
        "completed_superbatch": state.get("completed_superbatch"),
        "planned_superbatches": training["superbatches"],
        "last_metrics": last_metrics,
        "positions_per_second": speed,
        "eta_to_100m_seconds": eta_100m_seconds,
        "eta_to_target_seconds": eta_seconds,
        "latest_checkpoint": state.get("latest_checkpoint"),
        "latest_export": state.get("latest_export"),
        "checkpoint_generations": checkpoint_paths,
        "export_generations": export_paths,
        "stop_requested": (root / "STOP").exists(),
        "failure": state.get("failure"),
        "free_storage_bytes": shutil.disk_usage(root).free,
    }


def _prune_free_checkpoint_listing(directory: Path) -> list[str]:
    output: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(f"{DEFAULT_NET_ID}-mlx-step-"):
            output.append(str(path))
    return output
