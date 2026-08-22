"""Reusable MLX numerics for quantisation-aware neural-network training.

The helpers in this module deliberately know nothing about shogi or NNUE.
They are small enough to reuse in CNN, transformer, and language-model
experiments that must train float master weights while matching a discrete
deployment graph in the forward pass.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import mlx.core as mx

NumericPhase = Literal["mixed_precision_warmup", "quantisation_aware"]


class DaemonPrefetch[T]:
    """Run one resumable producer ahead of compute without blocking shutdown.

    Training pipelines often have a large network/disk item that should be
    fetched while the accelerator consumes the current item.  A daemon thread
    lets process-level checkpoint/stop semantics remain authoritative; any
    partial artifact must be resumable by the supplied producer.
    """

    def __init__(self, *, name: str, producer: Callable[[], T]) -> None:
        if not name:
            raise ValueError("prefetch task name must not be empty")
        self.name = name
        self._producer = producer
        self._result: T | None = None
        self._error: BaseException | None = None
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._result = self._producer()
        except BaseException as error:  # propagate at the consumption boundary
            self._error = error
        finally:
            self._finished.set()

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    def result(self, timeout: float | None = None) -> T:
        if not self._finished.wait(timeout):
            raise TimeoutError(f"prefetch task did not finish in time: {self.name}")
        if self._error is not None:
            raise self._error
        return cast(T, self._result)


@dataclass(frozen=True, slots=True)
class QuantisationSchedule:
    """Select a numerically safe warm-up before exact fake quantisation.

    Very small random initial weights can all round to zero.  If an activation
    multiplies two such branches, a straight-through estimator alone cannot
    revive the dead path because the other factor is also zero.  A finite
    float/mixed-precision warm-up is therefore part of the training contract,
    not merely a performance option.
    """

    warmup_optimizer_steps: int

    def __post_init__(self) -> None:
        if self.warmup_optimizer_steps < 1:
            raise ValueError("quantisation warm-up must contain at least one optimizer step")

    def phase_for_completed_steps(self, completed_steps: int) -> NumericPhase:
        if completed_steps < 0:
            raise ValueError("completed optimizer steps must be non-negative")
        if completed_steps < self.warmup_optimizer_steps:
            return "mixed_precision_warmup"
        return "quantisation_aware"


def straight_through(surrogate: mx.array, deployed: mx.array) -> mx.array:
    """Use ``deployed`` forward values and ``surrogate`` backward gradients."""

    if surrogate.shape != deployed.shape:
        raise ValueError("straight-through tensors must have identical shapes")
    return surrogate + mx.stop_gradient(deployed - surrogate)


def ste_round(values: mx.array) -> mx.array:
    """Round in the forward graph while differentiating as the identity."""

    return straight_through(values, mx.round(values))


def ste_round_rust_f32_scaled(values: mx.array, *, scale: float) -> mx.array:
    """Match Rust ``(f32 as f64 * scale).round()`` with an STE gradient.

    MLX's ``round`` uses ties-to-even, whereas Rust's ``f64::round`` uses
    half-away-from-zero.  There is an additional, less obvious difference for
    Tatara's QA=127 path: MLX first rounds the f32 multiplication, while the
    exporter widens the master f32 value to f64 before multiplying.  A product
    that lands on ``n + 0.5`` in f32 can therefore be just above or below the
    boundary in the exporter.

    The three production scales have an exact difference-of-powers-of-two
    decomposition.  FastTwoSum recovers the multiplication residual using
    only f32 operations, so the deployed branch makes the same rounding
    decision as the f64 Rust exporter without requiring unsupported f64 Metal
    arithmetic.  The surrogate branch preserves the original ``scale``
    gradient.
    """

    if values.dtype != mx.float32:
        raise TypeError("Rust-compatible fake quantisation requires float32 master weights")
    scale_parts = {
        64.0: (64.0, 0.0),
        127.0: (128.0, 1.0),
        8128.0: (8192.0, 64.0),
    }
    if scale not in scale_parts:
        raise ValueError(f"unsupported Rust-compatible quantisation scale: {scale}")
    high_scale, low_scale = scale_parts[scale]
    magnitude = mx.abs(values)
    high = magnitude * high_scale
    low = magnitude * low_scale
    product = high - low
    # |high| >= |low|, so FastTwoSum recovers the exact residual of high-low.
    residual = (-low) - (product - high)
    integral = mx.floor(product)
    fraction = product - integral
    round_up = (fraction > 0.5) | ((fraction == 0.5) & (residual >= 0.0))
    rounded_magnitude = integral + round_up.astype(mx.float32)
    deployed = mx.where(values < 0.0, -rounded_magnitude, rounded_magnitude)
    return straight_through(values * scale, deployed)


def ste_floor(values: mx.array) -> mx.array:
    """Floor in the forward graph while differentiating as the identity."""

    return straight_through(values, mx.floor(values))


def fake_quantise_raw(
    values: mx.array,
    *,
    scale: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> mx.array:
    """Return integer-valued raw units with an STE back to float weights.

    The result stays in floating-point storage so MLX can differentiate it,
    but its forward values are exactly rounded integer units.  Bounds model
    the saturating integer cast of the deployment format.
    """

    if not scale > 0.0:
        raise ValueError("fake-quantisation scale must be positive")
    if (minimum is None) != (maximum is None):
        raise ValueError("fake-quantisation bounds must be both set or both omitted")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("fake-quantisation minimum exceeds maximum")
    raw = ste_round(values * scale)
    if minimum is not None and maximum is not None:
        raw = mx.clip(raw, minimum, maximum)
    return raw


def fake_quantise_rust_f32_raw(
    values: mx.array,
    *,
    scale: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> mx.array:
    """Fake-quantise f32 weights with Tatara/Rust export rounding semantics."""

    if (minimum is None) != (maximum is None):
        raise ValueError("fake-quantisation bounds must be both set or both omitted")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("fake-quantisation minimum exceeds maximum")
    raw = ste_round_rust_f32_scaled(values, scale=scale)
    if minimum is not None and maximum is not None:
        raw = mx.clip(raw, minimum, maximum)
    return raw
