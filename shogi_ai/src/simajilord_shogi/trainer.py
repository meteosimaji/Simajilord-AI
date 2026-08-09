"""MLX policy-value training over versioned self-play records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from numpy.typing import NDArray
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT, move_label

from .compute_interlock import TrainingStepLease
from .compute_interlock import training_step as acquire_training_step
from .distillation_targets import (
    CANONICAL_SCORER_IDS,
    CanonicalDistillationTarget,
    ProvenMateStatus,
)
from .domain import PositionSample
from .encoding import combined_features
from .ensemble import normalized_sfen
from .model import PolicyValueResNet


@dataclass(frozen=True, slots=True)
class TrainingInterlockConfig:
    """Runtime-only exclusion between MLX training work and a human USI game.

    ``state_root`` is deliberately excluded from checkpoint lineage by callers:
    it is a local coordination path, not a mathematical training input.  Pauses
    occur before RNG sampling, so waiting for a human game cannot alter the
    exact-resume stream.
    """

    state_root: Path
    generation: int | None = None
    ttl_seconds: float = 60.0
    heartbeat_interval_seconds: float = 5.0
    wait_timeout_seconds: float = 86_400.0
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.generation is not None and self.generation < 0:
            raise ValueError("training interlock generation must be non-negative")
        if not 3 <= self.ttl_seconds <= 3_600:
            raise ValueError("training interlock TTL must be in [3, 3600] seconds")
        if not 0.1 <= self.heartbeat_interval_seconds < self.ttl_seconds:
            raise ValueError("training interlock heartbeat must be in [0.1, TTL) seconds")
        if self.wait_timeout_seconds < 0:
            raise ValueError("training interlock wait timeout must be non-negative")
        if not 0.01 <= self.poll_interval_seconds <= 1:
            raise ValueError("training interlock poll interval must be in [0.01, 1] seconds")

    def public_metadata(self) -> dict[str, int | float | bool | None]:
        """Return reproducibility metadata without exposing a private local path."""

        return {
            "enabled": True,
            "generation": self.generation,
            "ttl_seconds": self.ttl_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "wait_timeout_seconds": self.wait_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "state_root_recorded": False,
        }


@contextmanager
def _training_compute_lease(
    config: TrainingInterlockConfig | None,
) -> Iterator[TrainingStepLease | None]:
    if config is None:
        yield None
        return
    with acquire_training_step(
        config.state_root,
        generation=config.generation,
        ttl_seconds=config.ttl_seconds,
        heartbeat_interval_seconds=config.heartbeat_interval_seconds,
        wait_timeout_seconds=config.wait_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    ) as lease:
        yield lease


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    steps: int
    model_step_start: int
    model_step_end: int
    optimizer_step_start: int
    optimizer_step_end: int
    samples: int
    probe_samples: int
    initial_loss: float
    final_loss: float
    policy_loss: float
    value_loss: float
    gradient_clip_norm: float
    observed_preclip_gradient_norm_max: float
    observed_preclip_gradient_norm_mean: float
    gradient_clipped_steps: int
    gradient_clipped_fraction: float
    teacher_policy_mix: float
    teacher_value_mix: float
    probe_effective_teacher_policy_mix_min: float
    probe_effective_teacher_policy_mix_mean: float
    probe_effective_teacher_policy_mix_max: float
    probe_loss_ratio: float
    legal_label_smoothing: float
    mlx_peak_memory_bytes: int
    telemetry_points: int
    compute_interlock_enabled: bool
    compute_interlock_acquisitions: int
    compute_interlock_wait_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingTracePoint:
    """One reproducible optimization observation, sampled at a fixed interval."""

    model_step: int
    optimizer_step: int
    batch_loss: float
    policy_loss: float
    value_loss: float
    preclip_gradient_norm: float
    gradient_was_clipped: bool
    learning_rate: float
    sample_indices_sha256: str


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Everything required to continue the current AdamW stream exactly.

    MLX does not expose a setter for its global PRNG state.  Training therefore
    reseeds MLX deterministically from ``base_seed`` and ``optimizer_step`` at
    every step.  NumPy's batch sampler state and every AdamW tensor are stored
    directly.
    """

    model_step: int
    optimizer_step: int
    base_seed: int
    dataset_fingerprint: str
    configuration_fingerprint: str
    exact_resume_eligible: bool
    optimizer_state: dict[str, Any]
    numpy_rng_state: dict[str, Any]
    numpy_bit_generator: str
    mlx_version: str
    numpy_version: str
    optimizer_name: str = "AdamW"
    scheduler_name: str = "constant"
    mlx_step_seed_scheme: str = "sha256(base_seed:optimizer_step)-low32"


@dataclass(frozen=True, slots=True)
class TrainingRun:
    metrics: TrainingMetrics
    state: TrainingState
    trace: tuple[TrainingTracePoint, ...]


def _effective_teacher_mix(
    sample: PositionSample,
    base_mix: float,
    *,
    curriculum_depth_ratio: float,
    minimum_mix: float,
) -> float:
    """Introduce a vastly deeper teacher gradually until the actor catches up."""

    ratio = sample.teacher_depth_ratio
    if ratio is None or ratio <= curriculum_depth_ratio:
        return base_mix
    curriculum_scale = float(
        max(minimum_mix / base_mix, (curriculum_depth_ratio / ratio) ** 0.5)
    )
    return base_mix * min(1.0, curriculum_scale)


def _make_batch(
    samples: list[PositionSample],
    *,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
) -> tuple[mx.array, mx.array, mx.array]:
    features = np.stack([combined_features(Board(sample.sfen)) for sample in samples])
    policies = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.float32)
    values = np.empty((len(samples), 1), dtype=np.float32)
    for row, sample in enumerate(samples):
        board = Board(sample.sfen)
        teacher_mix = (
            _effective_teacher_mix(
                sample,
                teacher_policy_mix,
                curriculum_depth_ratio=curriculum_depth_ratio,
                minimum_mix=minimum_teacher_policy_mix,
            )
            if sample.teacher_policy is not None
            else 0.0
        )
        targets: tuple[tuple[dict[str, float], float], ...] = (
            ((sample.policy, 1.0 - teacher_mix), (sample.teacher_policy, teacher_mix))
            if sample.teacher_policy is not None
            else ((sample.policy, 1.0),)
        )
        for target, target_weight in targets:
            for move_usi, probability in target.items():
                move = Move.from_usi(move_usi)
                if not board.is_legal_move(move):
                    raise ValueError(f"illegal target move {move_usi} at ply {sample.ply}")
                policies[row, move_label(move, board.turn)] += float(probability) * target_weight
        legal_moves = board.legal_moves()
        if legal_label_smoothing:
            policies[row] *= 1.0 - legal_label_smoothing
            legal_probability = legal_label_smoothing / len(legal_moves)
            for move in legal_moves:
                policies[row, move_label(move, board.turn)] += legal_probability
        total = float(policies[row].sum())
        if total <= 0:
            raise ValueError(f"empty policy target at ply {sample.ply}")
        policies[row] /= total
        values[row, 0] = sample.value_target
        if sample.teacher_value is not None:
            value_curriculum_scale = (
                teacher_mix / teacher_policy_mix if teacher_policy_mix > 0 else 0.0
            )
            effective_teacher_value_mix = teacher_value_mix * value_curriculum_scale
            values[row, 0] = (
                effective_teacher_value_mix * sample.teacher_value
                + (1.0 - effective_teacher_value_mix) * sample.value_target
            )
    return mx.array(features), mx.array(policies), mx.array(values)


def _make_canonical_batch(
    samples: Sequence[PositionSample],
    targets: Sequence[CanonicalDistillationTarget],
) -> tuple[mx.array, ...]:
    """Build a target batch with no actor-policy or terminal-result channel."""

    if len(samples) != len(targets) or not samples:
        raise ValueError("canonical samples and targets must be non-empty and aligned")
    features = np.stack([combined_features(Board(sample.sfen)) for sample in samples])
    scorer_values = np.empty((len(samples), len(CANONICAL_SCORER_IDS)), dtype=np.float32)
    legal_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    maximum_groups = max(len(target.equivalence_groups) for target in targets)
    group_masks = np.zeros((len(samples), maximum_groups, MOVE_LABEL_COUNT), dtype=np.bool_)
    group_masses = np.zeros((len(samples), maximum_groups), dtype=np.float32)
    proof_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    proof_active = np.zeros((len(samples),), dtype=np.bool_)
    union_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    union_sizes = np.empty((len(samples),), dtype=np.int32)
    for row, (sample, target) in enumerate(zip(samples, targets, strict=True)):
        board = Board(sample.sfen)
        if target.normalized_sfen != normalized_sfen(sample.sfen):
            raise ValueError(f"canonical target does not match sample SFEN at row {row}")
        legal_labels = tuple(move_label(move, board.turn) for move in board.legal_moves())
        if not legal_labels:
            raise ValueError(f"canonical target row {row} is terminal")
        if len(set(legal_labels)) != len(legal_labels):
            raise ValueError(f"legal moves collide in the policy label space at row {row}")
        legal_masks[row, list(legal_labels)] = True

        scorer_by_id = {scorer.scorer_id: scorer for scorer in target.scorers}
        if tuple(scorer_by_id) != CANONICAL_SCORER_IDS:
            raise ValueError(f"canonical target row {row} lacks the required scorer pair")
        scorer_values[row] = [
            target.canonical_values[scorer_id] for scorer_id in CANONICAL_SCORER_IDS
        ]
        for group_index, group in enumerate(target.equivalence_groups):
            labels: list[int] = []
            for move_usi in group.moves:
                move = Move.from_usi(move_usi)
                if not board.is_legal_move(move):
                    raise ValueError(f"illegal canonical group move {move_usi} at row {row}")
                labels.append(move_label(move, board.turn))
            if len(set(labels)) != len(labels):
                raise ValueError(f"canonical group policy labels collide at row {row}")
            group_masks[row, group_index, labels] = True
            group_masses[row, group_index] = group.target_mass
        mate_set = target.proven_winning_mate_set
        proof_active[row] = mate_set.status is ProvenMateStatus.PROVEN
        if proof_active[row] and not mate_set.moves:
            raise ValueError(f"canonical proof-mate set is empty at row {row}")
        for move_usi in mate_set.moves:
            proof_masks[row, move_label(Move.from_usi(move_usi), board.turn)] = True
        union_sizes[row] = len(target.canonical_best_union)
        for move_usi in target.canonical_best_union:
            union_masks[row, move_label(Move.from_usi(move_usi), board.turn)] = True
    return (
        mx.array(features),
        mx.array(scorer_values),
        mx.array(legal_masks),
        mx.array(group_masks),
        mx.array(group_masses),
        mx.array(proof_masks),
        mx.array(proof_active),
        mx.array(union_masks),
        mx.array(union_sizes),
    )


def _masked_logsumexp(
    logits: mx.array,
    mask: mx.array,
    *,
    axis: int,
) -> mx.array:
    floor = mx.array(-1e30, dtype=logits.dtype)
    return mx.logsumexp(mx.where(mask, logits, floor), axis=axis)


def _proof_mate_set_mass_loss(
    logits: mx.array,
    legal_mask: mx.array,
    proof_mask: mx.array,
) -> mx.array:
    """Mean ``-log P(M)`` for an internally proven winning-move set."""

    return mx.mean(_proof_mate_set_mass_losses(logits, legal_mask, proof_mask))


def _proof_mate_set_mass_losses(
    logits: mx.array,
    legal_mask: mx.array,
    proof_mask: mx.array,
) -> mx.array:
    legal_log_mass = _masked_logsumexp(logits, legal_mask, axis=1)
    proof_log_mass = _masked_logsumexp(logits, proof_mask, axis=1)
    return legal_log_mass - proof_log_mass


def _equivalence_group_mass_loss(
    logits: mx.array,
    legal_mask: mx.array,
    group_masks: mx.array,
    group_masses: mx.array,
) -> mx.array:
    """Cross entropy over group probability mass, not individual moves."""

    return mx.mean(
        _equivalence_group_mass_losses(logits, legal_mask, group_masks, group_masses)
    )


def _equivalence_group_mass_losses(
    logits: mx.array,
    legal_mask: mx.array,
    group_masks: mx.array,
    group_masses: mx.array,
) -> mx.array:
    legal_log_mass = _masked_logsumexp(logits, legal_mask, axis=1)
    group_log_mass = (
        _masked_logsumexp(logits[:, None, :], group_masks, axis=2) - legal_log_mass[:, None]
    )
    return -mx.sum(group_masses * group_log_mass, axis=1)


def _best_union_top5_loss(
    logits: mx.array,
    legal_mask: mx.array,
    union_mask: mx.array,
    union_sizes: mx.array,
    *,
    margin: float = 0.0,
    row_active: mx.array | None = None,
) -> mx.array:
    """Require every disagreeing canonical best move to fit inside legal top-5."""

    if logits.ndim != 2 or logits.shape[1] < 5:
        raise ValueError("best-union top-5 loss requires batched logits with at least 5 labels")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("best-union top-5 margin must be finite and non-negative")
    floor = mx.array(-1e30, dtype=logits.dtype)
    ceiling = mx.array(1e30, dtype=logits.dtype)
    nonunion_mask = mx.logical_and(legal_mask, mx.logical_not(union_mask))
    sorted_nonunion = mx.sort(mx.where(nonunion_mask, logits, floor), axis=1)
    five_largest_ascending = sorted_nonunion[:, -5:]
    threshold_indices = mx.clip(union_sizes - 1, 0, 4)[:, None]
    threshold = mx.take_along_axis(five_largest_ascending, threshold_indices, axis=1)[:, 0]
    weakest_union = mx.min(mx.where(union_mask, logits, ceiling), axis=1)
    required_nonunion_count = 6 - union_sizes
    active = mx.logical_and(
        union_sizes > 1,
        mx.sum(nonunion_mask, axis=1) >= required_nonunion_count,
    )
    if row_active is not None:
        if row_active.ndim != 1 or row_active.shape[0] != logits.shape[0]:
            raise ValueError("best-union top-5 row mask must match the batch")
        active = mx.logical_and(active, row_active)
    losses = nn.softplus(threshold + margin - weakest_union)
    return mx.mean(mx.where(active, losses, mx.zeros_like(losses)))


def _canonical_policy_loss(
    logits: mx.array,
    legal_mask: mx.array,
    group_masks: mx.array,
    group_masses: mx.array,
    proof_mask: mx.array,
    proof_active: mx.array,
    union_mask: mx.array,
    union_sizes: mx.array,
    *,
    best_union_top5_weight: float,
) -> mx.array:
    group_losses = _equivalence_group_mass_losses(
        logits,
        legal_mask,
        group_masks,
        group_masses,
    )
    proof_losses = _proof_mate_set_mass_losses(logits, legal_mask, proof_mask)
    policy_loss = mx.mean(mx.where(proof_active, proof_losses, group_losses))
    if best_union_top5_weight:
        policy_loss = policy_loss + best_union_top5_weight * _best_union_top5_loss(
            logits,
            legal_mask,
            union_mask,
            union_sizes,
            row_active=mx.logical_not(proof_active),
        )
    return policy_loss


def _combine_policy_value_loss(
    policy_loss: mx.array,
    value_loss: mx.array,
    *,
    value_loss_weight: float,
) -> mx.array:
    """Preserve the exact historical addition when the new weight is neutral."""

    if value_loss_weight == 1.0:
        return policy_loss + value_loss
    return policy_loss + value_loss_weight * value_loss


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mlx_step_seed(base_seed: int, optimizer_step: int) -> int:
    payload = f"{base_seed}:{optimizer_step}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _json_clone_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    cloned = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    if not isinstance(cloned, dict):
        raise TypeError("expected a JSON object")
    return cloned


def _copy_optimizer_state(state: dict[str, Any]) -> dict[str, Any]:
    copied = tree_map(
        lambda item: mx.array(item) if isinstance(item, mx.array) else item,
        state,
    )
    if not isinstance(copied, dict):
        raise TypeError("AdamW state must be a dictionary")
    return copied


def _optimizer_tensor_signature(state: dict[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    signature: dict[str, tuple[tuple[int, ...], str]] = {}
    flattened = cast(list[tuple[str, Any]], tree_flatten(state))
    for name, value in flattened:
        if not isinstance(value, mx.array):
            raise TypeError(f"non-tensor AdamW state leaf: {name}")
        if name in signature:
            raise ValueError(f"duplicate AdamW state tensor: {name}")
        signature[name] = (tuple(int(size) for size in value.shape), str(value.dtype))
    return signature


def _restore_optimizer(
    model: PolicyValueResNet,
    *,
    learning_rate: float,
    state: TrainingState | None,
) -> optim.AdamW:
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    optimizer.init(model.trainable_parameters())
    if state is None:
        return optimizer
    expected = _optimizer_tensor_signature(optimizer.state)
    observed = _optimizer_tensor_signature(state.optimizer_state)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(observed) if expected[name] != observed[name]
        )
        raise ValueError(
            "AdamW state is incompatible with the model/runtime: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    restored = _copy_optimizer_state(state.optimizer_state)
    optimizer.state.clear()
    optimizer.state.update(restored)
    mx.eval(optimizer.state)
    restored_step = int(optimizer.state["step"].item())
    if restored_step != state.optimizer_step:
        raise ValueError(
            f"AdamW step mismatch: state={state.optimizer_step}, tensor={restored_step}"
        )
    restored_learning_rate = float(optimizer.state["learning_rate"].item())
    expected_learning_rate = float(mx.array(learning_rate, dtype=mx.float32).item())
    if restored_learning_rate != expected_learning_rate:
        raise ValueError(
            "AdamW learning rate mismatch: "
            f"state={restored_learning_rate}, requested={expected_learning_rate}"
        )
    return optimizer


def _training_configuration_fingerprint(
    model: PolicyValueResNet,
    *,
    sample_count: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    reversal_priority: float,
    blunder_priority: float,
    require_teacher: bool,
    prevalidated_teacher_samples: bool,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
    maximum_gradient_norm: float,
    maximum_probe_loss_ratio: float,
    canonical_teacher_only: bool,
    value_loss_weight: float,
    best_union_top5_weight: float,
) -> str:
    legacy_compatible = not canonical_teacher_only and (
        value_loss_weight == 1.0 and best_union_top5_weight == 0.0
    )
    payload: dict[str, Any] = {
        "schema": (
            "meteo-training-configuration-v1"
            if legacy_compatible
            else "meteo-training-configuration-v2"
        ),
        "model": asdict(model.config),
        "sample_count": sample_count,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "reversal_priority": reversal_priority,
        "blunder_priority": blunder_priority,
        "require_teacher": require_teacher,
        "prevalidated_teacher_samples": prevalidated_teacher_samples,
        "teacher_policy_mix": teacher_policy_mix,
        "teacher_value_mix": teacher_value_mix,
        "legal_label_smoothing": legal_label_smoothing,
        "curriculum_depth_ratio": curriculum_depth_ratio,
        "minimum_teacher_policy_mix": minimum_teacher_policy_mix,
        "maximum_gradient_norm": maximum_gradient_norm,
        "maximum_probe_loss_ratio": maximum_probe_loss_ratio,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "bias_correction": False,
        },
        "scheduler": "constant",
        "mlx_version": version("mlx"),
        "numpy_version": np.__version__,
        "mlx_step_seed_scheme": "sha256(base_seed:optimizer_step)-low32",
    }
    if not legacy_compatible:
        payload.update(
            {
                "canonical_teacher_only": canonical_teacher_only,
                "value_loss_weight": value_loss_weight,
                "best_union_top5_weight": best_union_top5_weight,
                "canonical_policy_loss": (
                    "legal-equivalence-group-mass-with-proven-mate-set-precedence-v1"
                ),
                "canonical_value_loss": "mean-per-canonical-scorer-mse-v1",
                "actor_policy_target_mass_in_canonical_mode": 0.0,
                "game_outcome_value_contribution_in_canonical_mode": 0.0,
            }
        )
    return _stable_fingerprint(payload)


def _train_impl(
    model: PolicyValueResNet,
    samples: Sequence[PositionSample],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    reversal_priority: float,
    blunder_priority: float,
    require_teacher: bool,
    prevalidated_teacher_samples: bool,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
    maximum_gradient_norm: float,
    maximum_probe_loss_ratio: float,
    canonical_teacher_only: bool,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None,
    value_loss_weight: float,
    best_union_top5_weight: float,
    dataset_fingerprint: str,
    exact_resume_eligible: bool,
    initial_model_step: int | None,
    resume_state: TrainingState | None,
    telemetry_interval: int,
    training_interlock: TrainingInterlockConfig | None,
) -> TrainingRun:
    training_samples: Sequence[PositionSample]
    training_targets: Sequence[CanonicalDistillationTarget] | None = None
    if canonical_teacher_only:
        if canonical_targets is None:
            raise ValueError("canonical teacher-only training requires canonical targets")
        if not require_teacher:
            raise ValueError("canonical teacher-only training cannot allow actor targets")
        if prevalidated_teacher_samples:
            raise ValueError("packed PSV data cannot use canonical teacher-only targets")
        if len(canonical_targets) != len(samples):
            raise ValueError("canonical target count must exactly match the training samples")
        training_samples = samples
        training_targets = canonical_targets
    elif canonical_targets is not None:
        raise ValueError("canonical targets require canonical_teacher_only=True")
    elif require_teacher and not prevalidated_teacher_samples:
        training_samples = [sample for sample in samples if sample.teacher_policy is not None]
    else:
        training_samples = samples
    if not training_samples:
        requirement = "deep-teacher" if require_teacher else "training"
        raise ValueError(f"at least one {requirement} sample is required")
    if steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("steps, batch_size, and learning_rate must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if telemetry_interval < 1:
        raise ValueError("telemetry interval must be positive")
    if not 0 < teacher_policy_mix <= 1 or not 0 <= teacher_value_mix <= 1:
        raise ValueError("teacher policy/value mixes must be in (0, 1] and [0, 1]")
    if not 0 <= legal_label_smoothing < 1:
        raise ValueError("legal_label_smoothing must be in [0, 1)")
    if curriculum_depth_ratio < 1:
        raise ValueError("curriculum_depth_ratio must be at least one")
    if not 0 < minimum_teacher_policy_mix <= teacher_policy_mix:
        raise ValueError("minimum teacher policy mix must be positive and no larger than base mix")
    if maximum_gradient_norm <= 0 or maximum_probe_loss_ratio < 1:
        raise ValueError("gradient norm must be positive and probe loss ratio must be at least one")
    if not math.isfinite(value_loss_weight) or value_loss_weight < 0:
        raise ValueError("value_loss_weight must be finite and non-negative")
    if not math.isfinite(best_union_top5_weight) or best_union_top5_weight < 0:
        raise ValueError("best_union_top5_weight must be finite and non-negative")
    if not canonical_teacher_only and best_union_top5_weight != 0.0:
        raise ValueError("best-union top-5 loss requires canonical teacher-only training")
    if exact_resume_eligible and (
        len(dataset_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in dataset_fingerprint)
    ):
        raise ValueError("exact resume requires a lowercase SHA-256 dataset fingerprint")
    configuration_fingerprint = _training_configuration_fingerprint(
        model,
        sample_count=len(training_samples),
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        reversal_priority=reversal_priority,
        blunder_priority=blunder_priority,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        value_loss_weight=value_loss_weight,
        best_union_top5_weight=best_union_top5_weight,
    )
    if resume_state is not None:
        if not resume_state.exact_resume_eligible:
            raise ValueError("checkpoint training state is not eligible for exact resume")
        if not exact_resume_eligible:
            raise ValueError("exact resume cannot use an unverified dataset")
        if resume_state.dataset_fingerprint != dataset_fingerprint:
            raise ValueError("training dataset fingerprint changed since the checkpoint")
        if resume_state.configuration_fingerprint != configuration_fingerprint:
            raise ValueError(
                "training configuration changed since the checkpoint; use an explicit "
                "warm start with optimizer reset instead of silently resuming"
            )
        if resume_state.base_seed != seed:
            raise ValueError("training seed changed since the checkpoint")
        if resume_state.mlx_version != version("mlx"):
            raise ValueError("MLX version changed since the checkpoint")
        if resume_state.numpy_version != np.__version__:
            raise ValueError("NumPy version changed since the checkpoint")
        if initial_model_step is not None and initial_model_step != resume_state.model_step:
            raise ValueError("requested model step does not match the resume checkpoint")
        model_step_start = resume_state.model_step
        optimizer_step_start = resume_state.optimizer_step
    else:
        model_step_start = 0 if initial_model_step is None else initial_model_step
        optimizer_step_start = 0
    if model_step_start < 0:
        raise ValueError("initial model step must be non-negative")

    interlock_acquisitions = 0
    interlock_wait_seconds = 0.0
    restore_wait_started = perf_counter()
    with _training_compute_lease(training_interlock) as restore_lease:
        if training_interlock is not None:
            interlock_acquisitions += 1
            interlock_wait_seconds += perf_counter() - restore_wait_started
        optimizer = _restore_optimizer(
            model,
            learning_rate=learning_rate,
            state=resume_state,
        )
        if restore_lease is not None:
            restore_lease.assert_healthy()
    mx.reset_peak_memory()
    if resume_state is None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        if rng.bit_generator.__class__.__name__ != resume_state.numpy_bit_generator:
            raise ValueError("NumPy bit generator implementation changed since checkpoint")
        rng.bit_generator.state = _json_clone_dict(resume_state.numpy_rng_state)
    sample_probabilities: NDArray[np.float64] | None = None
    if not prevalidated_teacher_samples and not canonical_teacher_only:
        sample_probabilities = np.asarray(
            [
                (reversal_priority if sample.policy_reversal else 1.0)
                * (1.0 + blunder_priority * max(0.0, sample.teacher_regret or 0.0))
                for sample in training_samples
            ],
            dtype=np.float64,
        )
        sample_probabilities /= sample_probabilities.sum()

    def loss_fn(*batch: mx.array) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        inputs = batch[0]
        logits, predictions = model(inputs)
        if canonical_teacher_only:
            if len(batch) != 9:
                raise ValueError("canonical training batch has an invalid tensor contract")
            scorer_values = batch[1]
            legal_mask = batch[2]
            group_masks = batch[3]
            group_masses = batch[4]
            proof_mask = batch[5]
            proof_active = batch[6]
            union_mask = batch[7]
            union_sizes = batch[8]
            policy_loss = _canonical_policy_loss(
                logits,
                legal_mask,
                group_masks,
                group_masses,
                proof_mask,
                proof_active,
                union_mask,
                union_sizes,
                best_union_top5_weight=best_union_top5_weight,
            )
            value_loss = mx.mean(mx.square(predictions - scorer_values))
        else:
            if len(batch) != 3:
                raise ValueError("legacy training batch has an invalid tensor contract")
            policies = batch[1]
            values = batch[2]
            policy_loss = nn.losses.cross_entropy(logits, policies, reduction="mean")
            value_loss = mx.mean(mx.square(predictions - values))
        return _combine_policy_value_loss(
            policy_loss,
            value_loss,
            value_loss_weight=value_loss_weight,
        ), (policy_loss, value_loss)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    batch_options: dict[str, float] = {
        "teacher_policy_mix": teacher_policy_mix,
        "teacher_value_mix": teacher_value_mix,
        "legal_label_smoothing": legal_label_smoothing,
        "curriculum_depth_ratio": curriculum_depth_ratio,
        "minimum_teacher_policy_mix": minimum_teacher_policy_mix,
    }
    probe_count = min(256, len(training_samples))
    probe_indices = np.linspace(0, len(training_samples) - 1, probe_count, dtype=np.int64)
    probe_samples = [training_samples[int(index)] for index in probe_indices]
    probe_effective_mixes = [
        (
            _effective_teacher_mix(
                sample,
                teacher_policy_mix,
                curriculum_depth_ratio=curriculum_depth_ratio,
                minimum_mix=minimum_teacher_policy_mix,
            )
            if sample.teacher_policy is not None
            else 0.0
        )
        for sample in probe_samples
    ]
    if canonical_teacher_only:
        assert training_targets is not None
        probe_targets = [training_targets[int(index)] for index in probe_indices]
        probe_batch = _make_canonical_batch(probe_samples, probe_targets)
        probe_effective_mixes = [1.0 for _sample in probe_samples]
    else:
        probe_batch = _make_batch(
            probe_samples,
            **batch_options,
        )
    model.eval()
    initial_wait_started = perf_counter()
    with _training_compute_lease(training_interlock) as initial_lease:
        if training_interlock is not None:
            interlock_acquisitions += 1
            interlock_wait_seconds += perf_counter() - initial_wait_started
        initial, _ = loss_fn(*probe_batch)
        mx.eval(initial)
        if initial_lease is not None:
            initial_lease.assert_healthy()
    initial_loss = float(initial.item())
    if not np.isfinite(initial_loss):
        raise RuntimeError("training rejected because the initial fixed-probe loss is non-finite")
    model.train()
    observed_max_gradient_norm = 0.0
    observed_sum_gradient_norm = 0.0
    clipped_steps = 0
    trace: list[TrainingTracePoint] = []
    for local_step in range(steps):
        optimizer_step_before = optimizer_step_start + local_step
        step_wait_started = perf_counter()
        with _training_compute_lease(training_interlock) as step_lease:
            if training_interlock is not None:
                interlock_acquisitions += 1
                interlock_wait_seconds += perf_counter() - step_wait_started
            # RNG and sampling happen only after the lease is acquired.  A
            # human-play pause therefore cannot advance either exact-resume
            # stream without completing and persisting an optimizer step.
            mx.random.seed(_mlx_step_seed(seed, optimizer_step_before))
            indices = rng.choice(
                len(training_samples),
                size=min(batch_size, len(training_samples)),
                replace=prevalidated_teacher_samples,
                p=sample_probabilities,
            )
            selected_samples = [training_samples[int(index)] for index in indices]
            if canonical_teacher_only:
                assert training_targets is not None
                selected_targets = [training_targets[int(index)] for index in indices]
                batch = _make_canonical_batch(selected_samples, selected_targets)
            else:
                batch = _make_batch(
                    selected_samples,
                    **batch_options,
                )
            (loss, (policy_loss, value_loss)), gradients = value_and_grad(*batch)
            clipped_gradients, gradient_norm = optim.clip_grad_norm(  # type: ignore[no-untyped-call]
                gradients, max_norm=maximum_gradient_norm
            )
            optimizer.update(model, clipped_gradients)
            # MLX is lazy: the lease must cover mx.eval of model and optimizer
            # state, not merely optimizer.update.
            mx.eval(
                loss,
                policy_loss,
                value_loss,
                gradient_norm,
                model.parameters(),
                optimizer.state,
            )
            if step_lease is not None:
                step_lease.assert_healthy()
        batch_loss_value = float(loss.item())
        policy_loss_value = float(policy_loss.item())
        value_loss_value = float(value_loss.item())
        gradient_norm_value = float(gradient_norm.item())
        if not all(
            np.isfinite(value)
            for value in (
                batch_loss_value,
                policy_loss_value,
                value_loss_value,
                gradient_norm_value,
            )
        ):
            raise RuntimeError(
                "training rejected because a non-finite optimization value was observed at "
                f"optimizer step {optimizer_step_before + 1}"
            )
        was_clipped = gradient_norm_value > maximum_gradient_norm
        clipped_steps += int(was_clipped)
        observed_max_gradient_norm = max(observed_max_gradient_norm, gradient_norm_value)
        observed_sum_gradient_norm += gradient_norm_value
        optimizer_step_after = optimizer_step_before + 1
        model_step_after = model_step_start + local_step + 1
        if (
            optimizer_step_after % telemetry_interval == 0
            or local_step == 0
            or local_step == steps - 1
        ):
            index_bytes = np.asarray(indices, dtype="<i8").tobytes()
            trace.append(
                TrainingTracePoint(
                    model_step=model_step_after,
                    optimizer_step=optimizer_step_after,
                    batch_loss=batch_loss_value,
                    policy_loss=policy_loss_value,
                    value_loss=value_loss_value,
                    preclip_gradient_norm=gradient_norm_value,
                    gradient_was_clipped=was_clipped,
                    learning_rate=float(optimizer.state["learning_rate"].item()),
                    sample_indices_sha256=hashlib.sha256(index_bytes).hexdigest(),
                )
            )
    model.eval()
    final_wait_started = perf_counter()
    with _training_compute_lease(training_interlock) as final_lease:
        if training_interlock is not None:
            interlock_acquisitions += 1
            interlock_wait_seconds += perf_counter() - final_wait_started
        final, (final_policy, final_value) = loss_fn(*probe_batch)
        mx.eval(final, final_policy, final_value)
        if final_lease is not None:
            final_lease.assert_healthy()
    final_loss = float(final.item())
    if not np.isfinite(final_loss) or final_loss > initial_loss * maximum_probe_loss_ratio:
        raise RuntimeError(
            "training rejected because fixed-probe loss became unstable: "
            f"{initial_loss} -> {final_loss}"
        )
    optimizer_step_end = int(optimizer.state["step"].item())
    expected_optimizer_step_end = optimizer_step_start + steps
    if optimizer_step_end != expected_optimizer_step_end:
        raise RuntimeError(
            "AdamW step counter diverged from the training loop: "
            f"{optimizer_step_end} != {expected_optimizer_step_end}"
        )
    model_step_end = model_step_start + steps
    metrics = TrainingMetrics(
        steps=steps,
        model_step_start=model_step_start,
        model_step_end=model_step_end,
        optimizer_step_start=optimizer_step_start,
        optimizer_step_end=optimizer_step_end,
        samples=len(training_samples),
        probe_samples=probe_count,
        initial_loss=initial_loss,
        final_loss=final_loss,
        policy_loss=float(final_policy.item()),
        value_loss=float(final_value.item()),
        gradient_clip_norm=maximum_gradient_norm,
        observed_preclip_gradient_norm_max=observed_max_gradient_norm,
        observed_preclip_gradient_norm_mean=observed_sum_gradient_norm / steps,
        gradient_clipped_steps=clipped_steps,
        gradient_clipped_fraction=clipped_steps / steps,
        teacher_policy_mix=1.0 if canonical_teacher_only else teacher_policy_mix,
        teacher_value_mix=1.0 if canonical_teacher_only else teacher_value_mix,
        probe_effective_teacher_policy_mix_min=min(probe_effective_mixes),
        probe_effective_teacher_policy_mix_mean=float(np.mean(probe_effective_mixes)),
        probe_effective_teacher_policy_mix_max=max(probe_effective_mixes),
        probe_loss_ratio=final_loss / max(initial_loss, 1e-12),
        legal_label_smoothing=0.0 if canonical_teacher_only else legal_label_smoothing,
        mlx_peak_memory_bytes=int(mx.get_peak_memory()),
        telemetry_points=len(trace),
        compute_interlock_enabled=training_interlock is not None,
        compute_interlock_acquisitions=interlock_acquisitions,
        compute_interlock_wait_seconds=interlock_wait_seconds,
    )
    state = TrainingState(
        model_step=model_step_end,
        optimizer_step=optimizer_step_end,
        base_seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        exact_resume_eligible=exact_resume_eligible,
        optimizer_state=_copy_optimizer_state(optimizer.state),
        numpy_rng_state=_json_clone_dict(rng.bit_generator.state),
        numpy_bit_generator=rng.bit_generator.__class__.__name__,
        mlx_version=version("mlx"),
        numpy_version=np.__version__,
    )
    return TrainingRun(metrics=metrics, state=state, trace=tuple(trace))


def train_resumable(
    model: PolicyValueResNet,
    samples: Sequence[PositionSample],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    dataset_fingerprint: str,
    seed: int = 0,
    reversal_priority: float = 4.0,
    blunder_priority: float = 3.0,
    require_teacher: bool = True,
    prevalidated_teacher_samples: bool = False,
    teacher_policy_mix: float = 0.75,
    teacher_value_mix: float = 0.5,
    legal_label_smoothing: float = 0.01,
    curriculum_depth_ratio: float = 64.0,
    minimum_teacher_policy_mix: float = 0.1,
    maximum_gradient_norm: float = 1.0,
    maximum_probe_loss_ratio: float = 1.25,
    canonical_teacher_only: bool = False,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None = None,
    value_loss_weight: float = 1.0,
    best_union_top5_weight: float = 0.0,
    initial_model_step: int | None = None,
    resume_state: TrainingState | None = None,
    telemetry_interval: int = 1,
    training_interlock: TrainingInterlockConfig | None = None,
) -> TrainingRun:
    """Train and return a serializable, fail-closed exact-resume state."""

    return _train_impl(
        model,
        samples,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        reversal_priority=reversal_priority,
        blunder_priority=blunder_priority,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        canonical_targets=canonical_targets,
        value_loss_weight=value_loss_weight,
        best_union_top5_weight=best_union_top5_weight,
        dataset_fingerprint=dataset_fingerprint,
        exact_resume_eligible=True,
        initial_model_step=initial_model_step,
        resume_state=resume_state,
        telemetry_interval=telemetry_interval,
        training_interlock=training_interlock,
    )


def train(
    model: PolicyValueResNet,
    samples: Sequence[PositionSample],
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    reversal_priority: float = 4.0,
    blunder_priority: float = 3.0,
    require_teacher: bool = True,
    prevalidated_teacher_samples: bool = False,
    teacher_policy_mix: float = 0.75,
    teacher_value_mix: float = 0.5,
    legal_label_smoothing: float = 0.01,
    curriculum_depth_ratio: float = 64.0,
    minimum_teacher_policy_mix: float = 0.1,
    maximum_gradient_norm: float = 1.0,
    maximum_probe_loss_ratio: float = 1.25,
    canonical_teacher_only: bool = False,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None = None,
    value_loss_weight: float = 1.0,
    best_union_top5_weight: float = 0.0,
    training_interlock: TrainingInterlockConfig | None = None,
) -> TrainingMetrics:
    """Backward-compatible warm-start API without a resumable data identity."""

    return _train_impl(
        model,
        samples,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        reversal_priority=reversal_priority,
        blunder_priority=blunder_priority,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        canonical_targets=canonical_targets,
        value_loss_weight=value_loss_weight,
        best_union_top5_weight=best_union_top5_weight,
        dataset_fingerprint="legacy-warm-start-without-stable-dataset-identity",
        exact_resume_eligible=False,
        initial_model_step=0,
        resume_state=None,
        telemetry_interval=max(1, steps),
        training_interlock=training_interlock,
    ).metrics
