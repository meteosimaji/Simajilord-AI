"""MLX policy-value training over versioned self-play records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from numpy.typing import NDArray
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT, move_label

from .compute_interlock import TrainingStepLease
from .compute_interlock import training_step as acquire_training_step
from .config import model_config_payload
from .distillation_targets import (
    CANONICAL_SCORER_IDS,
    CanonicalDistillationTarget,
    ProvenMateStatus,
)
from .distillation_targets_v2 import (
    CanonicalDistillationTargetV2,
    PlayTargetKind,
)
from .domain import PositionSample
from .encoding import combined_features, combined_features_with_history
from .ensemble import normalized_sfen
from .model import PolicyValueResNet
from .multi_objective_distillation import (
    blend_visit_and_value_policy,
    proven_mate_in_one_moves,
)

TRAINING_BATCH_CACHE_LIMIT_BYTES = 8 * 1024**3


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
    implicit_policy_mix: float
    proven_mate_policy_loss_weight: float
    probe_proven_mate_rows: int
    checkmate_sample_priority: float
    checkmate_horizon_plies: int
    probe_effective_teacher_policy_mix_min: float
    probe_effective_teacher_policy_mix_mean: float
    probe_effective_teacher_policy_mix_max: float
    probe_loss_ratio: float
    legal_label_smoothing: float
    training_batch_cache_enabled: bool
    training_batch_cache_bytes: int
    mlx_peak_memory_bytes: int
    telemetry_points: int
    compute_interlock_enabled: bool
    compute_interlock_acquisitions: int
    compute_interlock_wait_seconds: float
    canonical_teacher_policy_losses: tuple[float, ...] | None
    canonical_teacher_value_losses: tuple[float, ...] | None
    canonical_play_policy_loss: float | None
    canonical_play_interval_value_loss: float | None
    canonical_wdl_loss: float | None
    canonical_uncertainty_loss: float | None


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


def _validated_unit_value(value: float, *, label: str, ply: int) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not -1.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be finite and within [-1, 1] at ply {ply}")
    return numeric


def _sample_sampling_weight(
    sample: PositionSample,
    *,
    reversal_priority: float,
    blunder_priority: float,
    checkmate_sample_priority: float,
    checkmate_horizon_plies: int,
) -> float:
    weight = (reversal_priority if sample.policy_reversal else 1.0) * (
        1.0 + blunder_priority * max(0.0, sample.teacher_regret or 0.0)
    )
    distance = sample.terminal_checkmate_distance
    if distance is not None and (
        not isinstance(distance, int) or isinstance(distance, bool) or distance < 1
    ):
        raise ValueError("terminal checkmate distance must be a positive integer")
    if distance is not None and distance <= checkmate_horizon_plies:
        weight *= 1.0 + checkmate_sample_priority / math.sqrt(distance)
    return weight


def _make_batch(
    samples: list[PositionSample],
    *,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
    implicit_policy_mix: float = 0.0,
    with_policy_constraints: bool = False,
) -> tuple[mx.array, ...]:
    features = np.stack([combined_features(Board(sample.sfen)) for sample in samples])
    policies = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.float32)
    values = np.empty((len(samples), 1), dtype=np.float32)
    legal_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    proof_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    proof_active = np.zeros((len(samples),), dtype=np.bool_)
    for row, sample in enumerate(samples):
        board = Board(sample.sfen)
        if sample.ply < 0:
            raise ValueError("training sample ply must be non-negative")
        if sample.turn != board.turn.value:
            raise ValueError(f"training sample turn disagrees with SFEN at ply {sample.ply}")
        _validated_unit_value(sample.root_value, label="actor root value", ply=sample.ply)
        outcome_value = _validated_unit_value(
            sample.value_target,
            label="game outcome value target",
            ply=sample.ply,
        )
        teacher_value = (
            None
            if sample.teacher_value is None
            else _validated_unit_value(
                sample.teacher_value,
                label="teacher value target",
                ply=sample.ply,
            )
        )
        if sample.teacher_depth_ratio is not None and (
            not math.isfinite(sample.teacher_depth_ratio) or sample.teacher_depth_ratio <= 0
        ):
            raise ValueError(f"teacher depth ratio must be finite and positive at ply {sample.ply}")
        if sample.terminal_checkmate_distance is not None and (
            sample.terminal_checkmate_distance < 1
        ):
            raise ValueError(
                f"terminal checkmate distance must be positive at ply {sample.ply}"
            )
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
        actor_policy = blend_visit_and_value_policy(
            sample.policy,
            sample.actor_implicit_policy,
            implicit_weight=implicit_policy_mix,
        )
        teacher_policy = (
            None
            if sample.teacher_policy is None
            else blend_visit_and_value_policy(
                sample.teacher_policy,
                sample.teacher_implicit_policy,
                implicit_weight=implicit_policy_mix,
            )
        )
        if sample.teacher_policy is not None:
            if teacher_policy is None:
                raise AssertionError("teacher policy blend unexpectedly disappeared")
            targets: tuple[tuple[dict[str, float], float], ...] = (
                (actor_policy, 1.0 - teacher_mix),
                (teacher_policy, teacher_mix),
            )
        else:
            targets = ((actor_policy, 1.0),)
        for target, target_weight in targets:
            if not target:
                raise ValueError(f"empty source policy target at ply {sample.ply}")
            source_total = 0.0
            for move_usi, probability in target.items():
                numeric_probability = float(probability)
                if not math.isfinite(numeric_probability) or numeric_probability < 0:
                    raise ValueError(
                        f"policy probability must be finite and non-negative at ply {sample.ply}"
                    )
                move = Move.from_usi(move_usi)
                if not board.is_legal_move(move):
                    raise ValueError(f"illegal target move {move_usi} at ply {sample.ply}")
                policies[row, move_label(move, board.turn)] += (
                    numeric_probability * target_weight
                )
                source_total += numeric_probability
            if source_total <= 0:
                raise ValueError(f"zero-mass source policy target at ply {sample.ply}")
        legal_moves = tuple(board.legal_moves())
        if not legal_moves:
            raise ValueError(f"terminal position cannot be a training sample at ply {sample.ply}")
        legal_move_usi = {move.to_usi() for move in legal_moves}
        legal_labels = [move_label(move, board.turn) for move in legal_moves]
        legal_masks[row, legal_labels] = True
        claimed_mates = tuple(
            sorted(
                set(sample.actor_proven_mate_moves)
                | set(sample.teacher_proven_mate_moves)
            )
        )
        if claimed_mates:
            actual_mates = proven_mate_in_one_moves(board)
            if not set(claimed_mates).issubset(actual_mates):
                raise ValueError(
                    f"unverified proven mate move at ply {sample.ply}: "
                    f"claimed={claimed_mates!r}, verified={actual_mates!r}"
                )
            if not set(claimed_mates).issubset(legal_move_usi):
                raise ValueError(f"illegal proven mate move at ply {sample.ply}")
            proof_active[row] = True
            # Once one claimed proof has been independently verified, enumerate
            # the complete mate-in-one set so an unlisted but equally valid mate
            # is never penalized by the set-mass objective.
            for move_usi in actual_mates:
                proof_masks[row, move_label(Move.from_usi(move_usi), board.turn)] = True
        if legal_label_smoothing:
            policies[row] *= 1.0 - legal_label_smoothing
            legal_probability = legal_label_smoothing / len(legal_moves)
            for move in legal_moves:
                policies[row, move_label(move, board.turn)] += legal_probability
        total = float(policies[row].sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"empty policy target at ply {sample.ply}")
        policies[row] /= total
        values[row, 0] = outcome_value
        if teacher_value is not None:
            value_curriculum_scale = (
                teacher_mix / teacher_policy_mix if teacher_policy_mix > 0 else 0.0
            )
            effective_teacher_value_mix = teacher_value_mix * value_curriculum_scale
            values[row, 0] = (
                effective_teacher_value_mix * teacher_value
                + (1.0 - effective_teacher_value_mix) * outcome_value
            )
    result = (mx.array(features), mx.array(policies), mx.array(values))
    if not with_policy_constraints:
        return result
    return (
        *result,
        mx.array(legal_masks),
        mx.array(proof_masks),
        mx.array(proof_active),
    )


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
            raise ValueError(f"canonical target row {row} lacks the required scorer set")
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


def _make_canonical_v2_batch(
    samples: Sequence[PositionSample],
    targets: Sequence[CanonicalDistillationTargetV2],
) -> tuple[mx.array, ...]:
    """Build independent teacher-head and guarded play-head supervision.

    Every row trains the detached teacher heads and uncertainty head.  The play
    mask disables policy/value/WDL supervision for unresolved rows, which remain
    in the additional-search queue.
    """

    if len(samples) != len(targets) or not samples:
        raise ValueError("canonical v2 samples and targets must be non-empty and aligned")
    features = np.stack(
        [combined_features_with_history(target.history) for target in targets]
    )
    teacher_policies = np.zeros(
        (len(samples), len(CANONICAL_SCORER_IDS), MOVE_LABEL_COUNT),
        dtype=np.float32,
    )
    teacher_values = np.empty(
        (len(samples), len(CANONICAL_SCORER_IDS)), dtype=np.float32
    )
    legal_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    play_active = np.zeros((len(samples),), dtype=np.bool_)
    play_policies = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.float32)
    value_lower = np.zeros((len(samples), 1), dtype=np.float32)
    value_upper = np.zeros((len(samples), 1), dtype=np.float32)
    wdl_targets = np.zeros((len(samples), 3), dtype=np.float32)
    wdl_active = np.zeros((len(samples),), dtype=np.bool_)
    play_proof_masks = np.zeros((len(samples), MOVE_LABEL_COUNT), dtype=np.bool_)
    play_proof_active = np.zeros((len(samples),), dtype=np.bool_)
    uncertainty_targets = np.empty((len(samples), 1), dtype=np.float32)
    for row, (sample, target) in enumerate(zip(samples, targets, strict=True)):
        if target.play.train_play == target.play.additional_search_required:
            raise ValueError(f"canonical v2 play/search flags are inconsistent at row {row}")
        if target.normalized_sfen != normalized_sfen(sample.sfen):
            raise ValueError(f"canonical v2 target does not match sample SFEN at row {row}")
        board = Board(sample.sfen)
        history_board = Board(target.history.target_sfen)
        if board.to_sfen() != history_board.to_sfen():
            raise ValueError(f"canonical v2 exact history does not match sample at row {row}")
        legal_moves = tuple(board.legal_moves())
        if not legal_moves:
            raise ValueError(f"canonical v2 target row {row} is terminal")
        legal_labels = tuple(move_label(move, board.turn) for move in legal_moves)
        if len(set(legal_labels)) != len(legal_labels):
            raise ValueError(f"legal moves collide in policy-label space at row {row}")
        legal_masks[row, list(legal_labels)] = True
        if tuple(scorer.scorer_id for scorer in target.scorers) != CANONICAL_SCORER_IDS:
            raise ValueError(f"canonical v2 target row {row} lacks the required scorer set")
        for scorer_index, scorer in enumerate(target.scorers):
            teacher_values[row, scorer_index] = scorer.value
            for move_usi, probability in scorer.policy.items():
                move = Move.from_usi(move_usi)
                if not board.is_legal_move(move):
                    raise ValueError(
                        f"illegal canonical v2 teacher move {move_usi} at row {row}"
                    )
                teacher_policies[
                    row, scorer_index, move_label(move, board.turn)
                ] = probability
        play_active[row] = target.play.train_play
        uncertainty_targets[row, 0] = target.uncertainty_target
        if not play_active[row]:
            if (
                target.play.policy
                or target.play.robust_best_moves
                or target.play.value_interval is not None
                or target.play.wdl is not None
            ):
                raise ValueError(f"unresolved canonical v2 row {row} contains play targets")
            continue
        for move_usi, probability in target.play.policy.items():
            move = Move.from_usi(move_usi)
            if not board.is_legal_move(move):
                raise ValueError(f"illegal guarded play move {move_usi} at row {row}")
            play_policies[row, move_label(move, board.turn)] = probability
        if not np.isclose(play_policies[row].sum(), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError(f"guarded play policy is not normalized at row {row}")
        robust_labels = [
            move_label(Move.from_usi(move_usi), board.turn)
            for move_usi in target.play.robust_best_moves
        ]
        if not robust_labels:
            raise ValueError(f"guarded play target lacks robust best moves at row {row}")
        maximum_probability = float(play_policies[row].max())
        if any(
            not math.isclose(
                float(play_policies[row, label]),
                maximum_probability,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            for label in robust_labels
        ):
            raise ValueError(f"guarded robust best is not policy top-1 at row {row}")
        if target.play.value_interval is None:
            raise ValueError(f"guarded play target lacks a value interval at row {row}")
        value_lower[row, 0], value_upper[row, 0] = target.play.value_interval
        if target.play.wdl is not None:
            wdl_targets[row] = target.play.wdl
            wdl_active[row] = True
        if target.play.kind is PlayTargetKind.PROVEN_MATE:
            play_proof_active[row] = True
            play_proof_masks[row, robust_labels] = True
    resolved_rows = np.flatnonzero(play_active)
    if resolved_rows.size == 0:
        raise ValueError("canonical v2 optimizer batch requires at least one resolved row")
    return (
        mx.array(features),
        mx.array(teacher_policies),
        mx.array(teacher_values),
        mx.array(legal_masks),
        mx.array(features[resolved_rows]),
        mx.array(legal_masks[resolved_rows]),
        mx.array(play_policies[resolved_rows]),
        mx.array(value_lower[resolved_rows]),
        mx.array(value_upper[resolved_rows]),
        mx.array(wdl_targets[resolved_rows]),
        mx.array(wdl_active[resolved_rows]),
        mx.array(play_proof_masks[resolved_rows]),
        mx.array(play_proof_active[resolved_rows]),
        mx.array(uncertainty_targets),
    )


def _masked_mean(values: mx.array, active: mx.array) -> mx.array:
    if values.ndim != 1 or active.ndim != 1 or values.shape != active.shape:
        raise ValueError("masked mean inputs must be aligned rank-one tensors")
    weights = active.astype(values.dtype)
    denominator = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=values.dtype))
    return mx.sum(values * weights) / denominator


def _canonical_v2_sample_indices(
    rng: np.random.Generator,
    targets: Sequence[CanonicalDistillationTargetV2],
    *,
    batch_size: int,
    auxiliary_seed: int,
) -> NDArray[np.int64]:
    """Choose independent deterministic play and auxiliary batches.

    ``batch_size`` is the resolved play-batch capacity.  Unresolved rows are an
    additional detached auxiliary batch and never displace a resolved row or
    advance the persisted resolved-sampling RNG.  This keeps the play/trunk
    update byte-identical when an unresolved queue is attached, including when
    ``batch_size == 1``.
    """

    if batch_size < 1 or not targets:
        raise ValueError("canonical v2 sampling requires targets and a positive batch size")
    resolved = np.asarray(
        [index for index, target in enumerate(targets) if target.play.train_play],
        dtype=np.int64,
    )
    unresolved = np.asarray(
        [index for index, target in enumerate(targets) if not target.play.train_play],
        dtype=np.int64,
    )
    if resolved.size == 0:
        raise ValueError("canonical v2 has no play-eligible rows")
    resolved_selected = np.asarray(
        rng.choice(
            resolved,
            size=min(batch_size, int(resolved.size)),
            replace=False,
        ),
        dtype=np.int64,
    )
    if unresolved.size == 0:
        return resolved_selected
    auxiliary_rng = np.random.default_rng(auxiliary_seed)
    unresolved_selected = np.asarray(
        auxiliary_rng.choice(
            unresolved,
            size=min(batch_size, int(unresolved.size)),
            replace=False,
        ),
        dtype=np.int64,
    )
    return np.concatenate((resolved_selected, unresolved_selected))


_CANONICAL_AUXILIARY_PARAMETER_PREFIXES = (
    "teacher_policies.",
    "teacher_policy_biases.",
    "teacher_value_outputs.",
    "uncertainty_output.",
)


def _canonical_v2_clip_gradients(
    gradients: dict[str, Any],
    *,
    max_norm: float,
) -> tuple[dict[str, Any], mx.array]:
    """Clip play/shared and detached auxiliary gradients independently.

    A single global norm would let a high-loss unresolved teacher row scale
    down the resolved play/trunk gradient even though its play loss is masked.
    Separate normalizers remove that indirect coupling.
    """

    flattened = cast(list[tuple[str, mx.array]], tree_flatten(gradients))
    play = [
        (name, gradient)
        for name, gradient in flattened
        if not name.startswith(_CANONICAL_AUXILIARY_PARAMETER_PREFIXES)
    ]
    auxiliary = [
        (name, gradient)
        for name, gradient in flattened
        if name.startswith(_CANONICAL_AUXILIARY_PARAMETER_PREFIXES)
    ]
    if not play or not auxiliary:
        raise AssertionError("canonical v2 gradient parameter groups are incomplete")

    def group_norm(group: list[tuple[str, mx.array]]) -> mx.array:
        return mx.sqrt(sum((gradient.square().sum() for _name, gradient in group), 0.0))

    play_norm = group_norm(play)
    auxiliary_norm = group_norm(auxiliary)
    play_scale = mx.minimum(max_norm / (play_norm + 1e-6), 1.0)
    auxiliary_scale = mx.minimum(max_norm / (auxiliary_norm + 1e-6), 1.0)
    clipped = [
        (
            name,
            gradient
            * (
                auxiliary_scale
                if name.startswith(_CANONICAL_AUXILIARY_PARAMETER_PREFIXES)
                else play_scale
            ),
        )
        for name, gradient in flattened
    ]
    return cast(dict[str, Any], tree_unflatten(clipped)), mx.maximum(
        play_norm, auxiliary_norm
    )


def _canonical_v2_loss(
    model: PolicyValueResNet,
    *batch: mx.array,
    value_loss_weight: float,
    proven_mate_policy_loss_weight: float = 0.25,
    uncertainty_loss_weight: float = 0.1,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    """Independent teacher losses plus evidence-gated play losses.

    Teacher value predictions have shape ``[B, T]`` and are compared
    elementwise with ``[B, T]`` targets.  No scalar-to-multi-teacher broadcast
    is possible.  The model detaches auxiliary heads from the shared trunk.
    """

    if len(batch) != 14:
        raise ValueError("canonical v2 training batch has an invalid tensor contract")
    (
        teacher_policy_losses,
        play_policy_loss,
        teacher_value_losses,
        interval_value_loss,
        wdl_loss,
        uncertainty_loss,
    ) = _canonical_v2_monitor_losses(
        model,
        *batch,
        proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
    )
    teacher_policy_loss = mx.mean(teacher_policy_losses)
    teacher_value_loss = mx.mean(teacher_value_losses)
    policy_component = (
        teacher_policy_loss
        + play_policy_loss
        + wdl_loss
        + uncertainty_loss_weight * uncertainty_loss
    )
    value_component = teacher_value_loss + interval_value_loss
    total = policy_component + value_loss_weight * value_component
    return total, (policy_component, value_component)


def _canonical_v2_monitor_losses(
    model: PolicyValueResNet,
    *batch: mx.array,
    proven_mate_policy_loss_weight: float = 0.25,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Return per-teacher and play-head losses without averaging their identities."""

    if len(batch) != 14:
        raise ValueError("canonical v2 training batch has an invalid tensor contract")
    if model.training:
        raise ValueError(
            "canonical v2 loss requires eval mode so unresolved rows cannot alter "
            "BatchNorm statistics"
        )
    (
        inputs,
        teacher_policies,
        teacher_values,
        legal_mask,
        play_inputs,
        play_legal_mask,
        play_policies,
        value_lower,
        value_upper,
        wdl_targets,
        wdl_active,
        play_proof_mask,
        play_proof_active,
        uncertainty_targets,
    ) = batch
    output = model.forward_canonical(inputs)
    play_output = model.forward_canonical(play_inputs)
    floor = mx.array(-1e30, dtype=output.policy_play.dtype)
    play_logits = mx.where(play_legal_mask, play_output.policy_play, floor)
    teacher_logits = mx.where(
        legal_mask[:, None, :],
        output.policy_teachers,
        floor,
    )
    teacher_policy_rows = nn.losses.cross_entropy(
        mx.reshape(teacher_logits, (-1, MOVE_LABEL_COUNT)),
        mx.reshape(teacher_policies, (-1, MOVE_LABEL_COUNT)),
        reduction="none",
    )
    teacher_policy_losses = mx.mean(
        mx.reshape(teacher_policy_rows, (-1, len(CANONICAL_SCORER_IDS))),
        axis=0,
    )
    dense_play_policy_losses = nn.losses.cross_entropy(
        play_logits,
        play_policies,
        reduction="none",
    )
    mate_play_policy_losses = _proof_mate_set_mass_losses(
        play_output.policy_play,
        play_legal_mask,
        play_proof_mask,
    )
    if (
        not math.isfinite(proven_mate_policy_loss_weight)
        or not 0.0 <= proven_mate_policy_loss_weight <= 1.0
    ):
        raise ValueError("proven-mate policy loss weight must be finite in [0, 1]")
    play_policy_loss = mx.mean(
        mx.where(
            play_proof_active,
            proven_mate_policy_loss_weight * mate_play_policy_losses,
            dense_play_policy_losses,
        )
    )
    teacher_value_losses = mx.mean(
        mx.square(output.value_teachers - teacher_values), axis=0
    )
    below = mx.maximum(value_lower - play_output.value_play, 0.0)
    above = mx.maximum(play_output.value_play - value_upper, 0.0)
    interval_value_loss = mx.mean(mx.square(below) + mx.square(above))
    wdl_losses = nn.losses.cross_entropy(
        play_output.wdl_play_logits,
        wdl_targets,
        reduction="none",
    )
    wdl_loss = _masked_mean(wdl_losses, wdl_active)
    uncertainty_loss = mx.mean(mx.square(output.uncertainty - uncertainty_targets))
    return (
        teacher_policy_losses,
        play_policy_loss,
        teacher_value_losses,
        interval_value_loss,
        wdl_loss,
        uncertainty_loss,
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


def _legacy_policy_loss(
    logits: mx.array,
    policies: mx.array,
    legal_mask: mx.array,
    proof_mask: mx.array,
    proof_active: mx.array,
    *,
    proven_mate_policy_loss_weight: float,
) -> mx.array:
    """Use dense CE normally and a capped set-mass objective for proven mates.

    Dense cross entropy over a uniform mate distribution would train arbitrary
    preferences *inside* a set of equally proven mating moves.  The set-mass
    loss only asks the model to put probability somewhere in that verified set.
    Its independent weight prevents saturated mate rows from dominating the
    ordinary policy distribution while value/outcome learning remains active.
    """

    if len(logits.shape) != 2 or logits.shape != policies.shape:
        raise ValueError("legacy policy logits and targets must be aligned rank-two tensors")
    if legal_mask.shape != logits.shape or proof_mask.shape != logits.shape:
        raise ValueError("legacy policy masks must match the logits")
    if proof_active.ndim != 1 or proof_active.shape[0] != logits.shape[0]:
        raise ValueError("legacy proof-active mask must match the batch")
    if (
        not math.isfinite(proven_mate_policy_loss_weight)
        or not 0.0 <= proven_mate_policy_loss_weight <= 1.0
    ):
        raise ValueError("proven-mate policy loss weight must be finite in [0, 1]")
    normal_losses = nn.losses.cross_entropy(logits, policies, reduction="none")
    proof_losses = _proof_mate_set_mass_losses(logits, legal_mask, proof_mask)
    return mx.mean(
        mx.where(
            proof_active,
            proven_mate_policy_loss_weight * proof_losses,
            normal_losses,
        )
    )


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
    checkmate_sample_priority: float,
    checkmate_horizon_plies: int,
    require_teacher: bool,
    prevalidated_teacher_samples: bool,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    implicit_policy_mix: float,
    proven_mate_policy_loss_weight: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
    maximum_gradient_norm: float,
    maximum_probe_loss_ratio: float,
    canonical_teacher_only: bool,
    canonical_target_version: int | None,
    value_loss_weight: float,
    best_union_top5_weight: float,
) -> str:
    legacy_compatible = not canonical_teacher_only and (
        value_loss_weight == 1.0
        and best_union_top5_weight == 0.0
        and implicit_policy_mix == 0.0
        and proven_mate_policy_loss_weight == 1.0
    )
    payload: dict[str, Any] = {
        "schema": (
            "meteo-training-configuration-v1"
            if legacy_compatible
            else "meteo-training-configuration-v3"
        ),
        "model": model_config_payload(model.config),
        "sample_count": sample_count,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "reversal_priority": reversal_priority,
        "blunder_priority": blunder_priority,
        "checkmate_sample_priority": checkmate_sample_priority,
        "checkmate_horizon_plies": checkmate_horizon_plies,
        "require_teacher": require_teacher,
        "prevalidated_teacher_samples": prevalidated_teacher_samples,
        "teacher_policy_mix": teacher_policy_mix,
        "teacher_value_mix": teacher_value_mix,
        "implicit_policy_mix": implicit_policy_mix,
        "proven_mate_policy_loss_weight": proven_mate_policy_loss_weight,
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
                "canonical_target_version": canonical_target_version,
                "value_loss_weight": value_loss_weight,
                "best_union_top5_weight": best_union_top5_weight,
                "canonical_policy_loss": (
                    "independent-teacher-distribution-ce-plus-worst-regret-distribution-ce-v3"
                    if canonical_target_version == 2
                    else "canonical-v1-training-disabled"
                ),
                "canonical_value_loss": (
                    "independent-teacher-head-mse-plus-play-interval-loss-v2"
                    if canonical_target_version == 2
                    else "canonical-v1-training-disabled"
                ),
                "unresolved_play_policy_value_wdl_masked": True,
                "auxiliary_teacher_gradient_to_play_trunk": "stopped",
                "canonical_batchnorm_mode": "frozen_eval",
                "canonical_batch_sampling": (
                    "resolved-capacity-plus-independent-unresolved-auxiliary-v3"
                ),
                "canonical_gradient_clipping": "separate-play-and-auxiliary-global-norm-v2",
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
    checkmate_sample_priority: float,
    checkmate_horizon_plies: int,
    require_teacher: bool,
    prevalidated_teacher_samples: bool,
    teacher_policy_mix: float,
    teacher_value_mix: float,
    implicit_policy_mix: float,
    proven_mate_policy_loss_weight: float,
    legal_label_smoothing: float,
    curriculum_depth_ratio: float,
    minimum_teacher_policy_mix: float,
    maximum_gradient_norm: float,
    maximum_probe_loss_ratio: float,
    canonical_teacher_only: bool,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None,
    canonical_targets_v2: Sequence[CanonicalDistillationTargetV2] | None,
    value_loss_weight: float,
    best_union_top5_weight: float,
    dataset_fingerprint: str,
    exact_resume_eligible: bool,
    initial_model_step: int | None,
    resume_state: TrainingState | None,
    telemetry_interval: int,
    training_interlock: TrainingInterlockConfig | None,
    progress_callback: Callable[[dict[str, int | float]], None] | None,
) -> TrainingRun:
    if canonical_targets is not None:
        raise ValueError(
            "canonical target v1 training is disabled because one value head broadcast over "
            "all teacher values converges to their arithmetic mean; rebuild a complete "
            "canonical-target-contract-v2 sidecar"
        )
    if canonical_teacher_only:
        raise RuntimeError(
            "canonical-target-contract-v2 optimizer runs are disabled until the real "
            "score-matrix builder, calibration replay verification, and independent "
            "held-out receipt are implemented"
        )
    training_samples: Sequence[PositionSample]
    training_targets_v2: Sequence[CanonicalDistillationTargetV2] | None = None
    if not canonical_teacher_only and (
        model.config.history_input_version != 1
        or model.config.canonical_head_version != 1
    ):
        raise ValueError(
            "history/canonical-v2 checkpoints cannot enter the legacy actor/outcome training "
            "path; use a complete canonical-target-contract-v2 sidecar or a future explicitly "
            "history-aware Reanalyse contract"
        )
    if canonical_teacher_only:
        if canonical_targets_v2 is None:
            raise ValueError("canonical teacher-only training requires canonical v2 targets")
        if not require_teacher:
            raise ValueError("canonical teacher-only training cannot allow actor targets")
        if prevalidated_teacher_samples:
            raise ValueError("packed PSV data cannot use canonical teacher-only targets")
        if len(canonical_targets_v2) != len(samples):
            raise ValueError("canonical target count must exactly match the training samples")
        if (
            model.config.canonical_head_version != 2
            or model.config.history_input_version != 2
            or model.config.canonical_teacher_count != len(CANONICAL_SCORER_IDS)
        ):
            raise ValueError(
                "canonical-target-contract-v2 requires an explicitly upgraded model with "
                "independent heads and history-input-v2"
            )
        if not any(
            target.play.train_play and not target.play.additional_search_required
            for target in canonical_targets_v2
        ):
            raise ValueError(
                "canonical v2 contains no play-eligible positions; unresolved positions remain "
                "an additional-search queue and cannot create optimizer steps by themselves"
            )
        training_samples = samples
        training_targets_v2 = canonical_targets_v2
    elif canonical_targets_v2 is not None:
        raise ValueError("canonical v2 targets require canonical_teacher_only=True")
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
    if not math.isfinite(implicit_policy_mix) or not 0.0 <= implicit_policy_mix <= 1.0:
        raise ValueError("implicit policy mix must be finite in [0, 1]")
    if (
        not math.isfinite(proven_mate_policy_loss_weight)
        or not 0.0 <= proven_mate_policy_loss_weight <= 1.0
    ):
        raise ValueError("proven-mate policy loss weight must be finite in [0, 1]")
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
    if not math.isfinite(checkmate_sample_priority) or checkmate_sample_priority < 0:
        raise ValueError("checkmate sample priority must be finite and non-negative")
    if checkmate_horizon_plies < 1:
        raise ValueError("checkmate horizon plies must be positive")
    if best_union_top5_weight != 0.0:
        raise ValueError(
            "best-union top-5 is a disabled canonical-v1 auxiliary; canonical v2 derives a "
            "top-1-preserving worst-teacher-regret play distribution"
        )
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
        checkmate_sample_priority=checkmate_sample_priority,
        checkmate_horizon_plies=checkmate_horizon_plies,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        implicit_policy_mix=implicit_policy_mix,
        proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        canonical_target_version=(2 if canonical_teacher_only else None),
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
                _sample_sampling_weight(
                    sample,
                    reversal_priority=reversal_priority,
                    blunder_priority=blunder_priority,
                    checkmate_sample_priority=checkmate_sample_priority,
                    checkmate_horizon_plies=checkmate_horizon_plies,
                )
                for sample in training_samples
            ],
            dtype=np.float64,
        )
        sample_probabilities /= sample_probabilities.sum()

    def loss_fn(*batch: mx.array) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        if canonical_teacher_only:
            return _canonical_v2_loss(
                model,
                *batch,
                value_loss_weight=value_loss_weight,
                proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
            )
        if len(batch) != 6:
            raise ValueError("legacy training batch has an invalid tensor contract")
        inputs, policies, values, legal_mask, proof_mask, proof_active = batch
        logits, predictions = model(inputs)
        policy_loss = _legacy_policy_loss(
            logits,
            policies,
            legal_mask,
            proof_mask,
            proof_active,
            proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
        )
        value_loss = mx.mean(mx.square(predictions - values))
        return _combine_policy_value_loss(
            policy_loss,
            value_loss,
            value_loss_weight=value_loss_weight,
        ), (policy_loss, value_loss)

    value_and_grad = nn.value_and_grad(model, loss_fn)
    compiled_state = [model.state, optimizer.state, mx.random.state]

    def optimization_step(*batch: mx.array) -> tuple[mx.array, ...]:
        (loss, (policy_loss, value_loss)), gradients = value_and_grad(*batch)
        if canonical_teacher_only:
            clipped_gradients, gradient_norm = _canonical_v2_clip_gradients(
                gradients,
                max_norm=maximum_gradient_norm,
            )
        else:
            clipped_gradients, gradient_norm = optim.clip_grad_norm(
                gradients, max_norm=maximum_gradient_norm
            )
        optimizer.update(model, clipped_gradients)
        return loss, policy_loss, value_loss, gradient_norm

    # MLX compilation captures and updates BatchNorm/model, AdamW, and random
    # state as one graph.  Keeping the outer optimization step intact lets MLX
    # fuse forward/backward/update work instead of launching hundreds of small
    # kernels per batch.
    compiled_optimization_step = mx.compile(
        optimization_step,
        inputs=compiled_state,
        outputs=compiled_state,
    )
    batch_options: dict[str, Any] = {
        "teacher_policy_mix": teacher_policy_mix,
        "teacher_value_mix": teacher_value_mix,
        "implicit_policy_mix": implicit_policy_mix,
        "legal_label_smoothing": legal_label_smoothing,
        "curriculum_depth_ratio": curriculum_depth_ratio,
        "minimum_teacher_policy_mix": minimum_teacher_policy_mix,
        "with_policy_constraints": True,
    }
    probe_count = min(256, len(training_samples))
    probe_indices = np.linspace(0, len(training_samples) - 1, probe_count, dtype=np.int64)
    if canonical_teacher_only:
        assert training_targets_v2 is not None
        probe_eligible = {
            index for index, target in enumerate(training_targets_v2) if target.play.train_play
        }
        probe_unresolved = set(range(len(training_targets_v2))) - probe_eligible
        if not any(int(index) in probe_eligible for index in probe_indices):
            probe_indices[0] = min(probe_eligible)
        if probe_count > 1 and probe_unresolved and not any(
            int(index) in probe_unresolved for index in probe_indices
        ):
            probe_indices[-1] = min(probe_unresolved)
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
        assert training_targets_v2 is not None
        probe_targets_v2 = [training_targets_v2[int(index)] for index in probe_indices]
        probe_batch = _make_canonical_v2_batch(probe_samples, probe_targets_v2)
        probe_effective_mixes = [1.0 for _sample in probe_samples]
    else:
        probe_batch = _make_batch(
            probe_samples,
            **batch_options,
        )
    cached_training_batch: tuple[mx.array, ...] | None = None
    training_batch_cache_bytes = 0
    if not canonical_teacher_only and not prevalidated_teacher_samples:
        probe_tensor_bytes = sum(int(tensor.nbytes) for tensor in probe_batch)
        estimated_cache_bytes = math.ceil(
            probe_tensor_bytes * len(training_samples) / len(probe_samples)
        )
        if estimated_cache_bytes <= TRAINING_BATCH_CACHE_LIMIT_BYTES:
            cached_training_batch = _make_batch(
                list(training_samples),
                **batch_options,
            )
            mx.eval(cached_training_batch)
            training_batch_cache_bytes = sum(
                int(tensor.nbytes) for tensor in cached_training_batch
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
    # Canonical auxiliary rows must not change BatchNorm running statistics.
    # Gradients still flow through the resolved play rows in eval mode.
    if canonical_teacher_only:
        model.eval()
    else:
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
            if canonical_teacher_only:
                assert training_targets_v2 is not None
                indices = _canonical_v2_sample_indices(
                    rng,
                    training_targets_v2,
                    batch_size=batch_size,
                    auxiliary_seed=_mlx_step_seed(
                        seed ^ 0xA5A5A5A5,
                        optimizer_step_before,
                    ),
                )
            else:
                indices = rng.choice(
                    len(training_samples),
                    size=min(batch_size, len(training_samples)),
                    replace=prevalidated_teacher_samples,
                    p=sample_probabilities,
                )
            selected_samples = [training_samples[int(index)] for index in indices]
            if canonical_teacher_only:
                assert training_targets_v2 is not None
                selected_targets_v2 = [
                    training_targets_v2[int(index)] for index in indices
                ]
                batch = _make_canonical_v2_batch(
                    selected_samples,
                    selected_targets_v2,
                )
            else:
                if cached_training_batch is None:
                    batch = _make_batch(
                        selected_samples,
                        **batch_options,
                    )
                else:
                    index_array = mx.array(np.asarray(indices, dtype=np.int32))
                    batch = tuple(
                        mx.take(tensor, index_array, axis=0)
                        for tensor in cached_training_batch
                    )
            loss, policy_loss, value_loss, gradient_norm = compiled_optimization_step(
                *batch
            )
            # MLX is lazy: the lease must cover mx.eval of model and optimizer
            # state, not merely optimizer.update.
            mx.eval(
                loss,
                policy_loss,
                value_loss,
                gradient_norm,
                compiled_state,
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
        progress_interval = max(1, steps // 100)
        if progress_callback is not None and (
            local_step == 0
            or local_step == steps - 1
            or (local_step + 1) % progress_interval == 0
        ):
            progress_callback(
                {
                    "steps_completed": local_step + 1,
                    "steps_total": steps,
                    "model_step": model_step_after,
                    "optimizer_step": optimizer_step_after,
                    "batch_loss": batch_loss_value,
                    "policy_loss": policy_loss_value,
                    "value_loss": value_loss_value,
                }
            )
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
    canonical_monitor_values: tuple[mx.array, ...] | None = None
    with _training_compute_lease(training_interlock) as final_lease:
        if training_interlock is not None:
            interlock_acquisitions += 1
            interlock_wait_seconds += perf_counter() - final_wait_started
        final, (final_policy, final_value) = loss_fn(*probe_batch)
        if canonical_teacher_only:
            canonical_monitor_values = _canonical_v2_monitor_losses(
                model,
                *probe_batch,
                proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
            )
            mx.eval(
                final,
                final_policy,
                final_value,
                *canonical_monitor_values,
            )
        else:
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
    canonical_teacher_policy_losses: tuple[float, ...] | None = None
    canonical_teacher_value_losses: tuple[float, ...] | None = None
    canonical_play_policy_loss: float | None = None
    canonical_play_interval_value_loss: float | None = None
    canonical_wdl_loss: float | None = None
    canonical_uncertainty_loss: float | None = None
    if canonical_monitor_values is not None:
        (
            teacher_policy_losses,
            play_policy_loss,
            teacher_value_losses,
            play_interval_value_loss,
            wdl_loss,
            uncertainty_loss,
        ) = canonical_monitor_values
        teacher_policy_array = np.asarray(teacher_policy_losses, dtype=np.float64)
        teacher_value_array = np.asarray(teacher_value_losses, dtype=np.float64)
        canonical_teacher_policy_losses = tuple(
            float(value) for value in teacher_policy_array
        )
        canonical_teacher_value_losses = tuple(
            float(value) for value in teacher_value_array
        )
        canonical_play_policy_loss = float(play_policy_loss.item())
        canonical_play_interval_value_loss = float(play_interval_value_loss.item())
        canonical_wdl_loss = float(wdl_loss.item())
        canonical_uncertainty_loss = float(uncertainty_loss.item())
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
        implicit_policy_mix=0.0 if canonical_teacher_only else implicit_policy_mix,
        proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
        probe_proven_mate_rows=(
            sum(
                target.play.kind is PlayTargetKind.PROVEN_MATE
                for target in (training_targets_v2 or ())
            )
            if canonical_teacher_only
            else sum(
                bool(sample.actor_proven_mate_moves or sample.teacher_proven_mate_moves)
                for sample in probe_samples
            )
        ),
        checkmate_sample_priority=checkmate_sample_priority,
        checkmate_horizon_plies=checkmate_horizon_plies,
        probe_effective_teacher_policy_mix_min=min(probe_effective_mixes),
        probe_effective_teacher_policy_mix_mean=float(np.mean(probe_effective_mixes)),
        probe_effective_teacher_policy_mix_max=max(probe_effective_mixes),
        probe_loss_ratio=final_loss / max(initial_loss, 1e-12),
        legal_label_smoothing=0.0 if canonical_teacher_only else legal_label_smoothing,
        training_batch_cache_enabled=cached_training_batch is not None,
        training_batch_cache_bytes=training_batch_cache_bytes,
        mlx_peak_memory_bytes=int(mx.get_peak_memory()),
        telemetry_points=len(trace),
        compute_interlock_enabled=training_interlock is not None,
        compute_interlock_acquisitions=interlock_acquisitions,
        compute_interlock_wait_seconds=interlock_wait_seconds,
        canonical_teacher_policy_losses=canonical_teacher_policy_losses,
        canonical_teacher_value_losses=canonical_teacher_value_losses,
        canonical_play_policy_loss=canonical_play_policy_loss,
        canonical_play_interval_value_loss=canonical_play_interval_value_loss,
        canonical_wdl_loss=canonical_wdl_loss,
        canonical_uncertainty_loss=canonical_uncertainty_loss,
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
    checkmate_sample_priority: float = 0.0,
    checkmate_horizon_plies: int = 16,
    require_teacher: bool = True,
    prevalidated_teacher_samples: bool = False,
    teacher_policy_mix: float = 0.75,
    teacher_value_mix: float = 0.5,
    implicit_policy_mix: float = 0.5,
    proven_mate_policy_loss_weight: float = 0.25,
    legal_label_smoothing: float = 0.01,
    curriculum_depth_ratio: float = 64.0,
    minimum_teacher_policy_mix: float = 0.1,
    maximum_gradient_norm: float = 1.0,
    maximum_probe_loss_ratio: float = 1.25,
    canonical_teacher_only: bool = False,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None = None,
    canonical_targets_v2: Sequence[CanonicalDistillationTargetV2] | None = None,
    value_loss_weight: float = 1.0,
    best_union_top5_weight: float = 0.0,
    initial_model_step: int | None = None,
    resume_state: TrainingState | None = None,
    telemetry_interval: int = 1,
    training_interlock: TrainingInterlockConfig | None = None,
    progress_callback: Callable[[dict[str, int | float]], None] | None = None,
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
        checkmate_sample_priority=checkmate_sample_priority,
        checkmate_horizon_plies=checkmate_horizon_plies,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        implicit_policy_mix=implicit_policy_mix,
        proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        canonical_targets=canonical_targets,
        canonical_targets_v2=canonical_targets_v2,
        value_loss_weight=value_loss_weight,
        best_union_top5_weight=best_union_top5_weight,
        dataset_fingerprint=dataset_fingerprint,
        exact_resume_eligible=True,
        initial_model_step=initial_model_step,
        resume_state=resume_state,
        telemetry_interval=telemetry_interval,
        training_interlock=training_interlock,
        progress_callback=progress_callback,
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
    checkmate_sample_priority: float = 0.0,
    checkmate_horizon_plies: int = 16,
    require_teacher: bool = True,
    prevalidated_teacher_samples: bool = False,
    teacher_policy_mix: float = 0.75,
    teacher_value_mix: float = 0.5,
    implicit_policy_mix: float = 0.5,
    proven_mate_policy_loss_weight: float = 0.25,
    legal_label_smoothing: float = 0.01,
    curriculum_depth_ratio: float = 64.0,
    minimum_teacher_policy_mix: float = 0.1,
    maximum_gradient_norm: float = 1.0,
    maximum_probe_loss_ratio: float = 1.25,
    canonical_teacher_only: bool = False,
    canonical_targets: Sequence[CanonicalDistillationTarget] | None = None,
    canonical_targets_v2: Sequence[CanonicalDistillationTargetV2] | None = None,
    value_loss_weight: float = 1.0,
    best_union_top5_weight: float = 0.0,
    training_interlock: TrainingInterlockConfig | None = None,
    progress_callback: Callable[[dict[str, int | float]], None] | None = None,
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
        checkmate_sample_priority=checkmate_sample_priority,
        checkmate_horizon_plies=checkmate_horizon_plies,
        require_teacher=require_teacher,
        prevalidated_teacher_samples=prevalidated_teacher_samples,
        teacher_policy_mix=teacher_policy_mix,
        teacher_value_mix=teacher_value_mix,
        implicit_policy_mix=implicit_policy_mix,
        proven_mate_policy_loss_weight=proven_mate_policy_loss_weight,
        legal_label_smoothing=legal_label_smoothing,
        curriculum_depth_ratio=curriculum_depth_ratio,
        minimum_teacher_policy_mix=minimum_teacher_policy_mix,
        maximum_gradient_norm=maximum_gradient_norm,
        maximum_probe_loss_ratio=maximum_probe_loss_ratio,
        canonical_teacher_only=canonical_teacher_only,
        canonical_targets=canonical_targets,
        canonical_targets_v2=canonical_targets_v2,
        value_loss_weight=value_loss_weight,
        best_union_top5_weight=best_union_top5_weight,
        dataset_fingerprint="legacy-warm-start-without-stable-dataset-identity",
        exact_resume_eligible=False,
        initial_model_step=0,
        resume_state=None,
        telemetry_interval=max(1, steps),
        training_interlock=training_interlock,
        progress_callback=progress_callback,
    ).metrics
