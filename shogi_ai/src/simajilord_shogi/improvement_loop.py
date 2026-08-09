"""Append-only contracts for continuous, teacher-assisted self-improvement.

This module deliberately separates orchestration and promotion policy from the
expensive engine runners.  It plans reproducible games, constrains exploration,
freezes leakage-free dataset growth, gates every opponent role, and records an
append-only generation/state chain.  Callers may execute the planned jobs with
Meteo or USI engines, but cannot weaken the production gate by changing runner
defaults.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rsshogi.core import Board

from .opening_suite import OpeningPosition

LOOP_PLAN_SCHEMA = "meteo-continuous-improvement-plan-v1"
STAGE_RECEIPT_SCHEMA = "meteo-continuous-improvement-stage-v1"
DATASET_GROWTH_SCHEMA = "meteo-dataset-growth-v1"
PROMOTION_DECISION_SCHEMA = "meteo-role-gated-promotion-v1"
LOOP_STATE_SCHEMA = "meteo-continuous-improvement-state-v1"

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_CANDIDATE_ID = "@candidate"
_REFERENCE_ID = "@champion-reference"
_FLAT_START = OpeningPosition.from_sfen(Board().to_sfen())


class OpponentRole(StrEnum):
    """Why an engine belongs in the continuous evaluation league."""

    CHAMPION = "champion"
    HISTORICAL_METEO = "historical_meteo"
    EXTERNAL_TEACHER = "external_teacher"
    WEAK = "weak"
    RANDOM = "random"


class DatasetSplit(StrEnum):
    """Mutually exclusive uses of a completed game artifact."""

    TRAIN = "train"
    VALIDATION = "validation"
    SEALED_TEST = "sealed_test"
    ARENA = "arena"


class GenerationStage(StrEnum):
    """Ordered, append-only receipts for one candidate generation."""

    GAMES_COLLECTED = "games_collected"
    TEACHER_LABELLED = "teacher_labelled"
    DATASET_FROZEN = "dataset_frozen"
    CANDIDATE_TRAINED = "candidate_trained"
    BENCHMARK_COMPLETE = "benchmark_complete"
    FINALIZED = "finalized"


_STAGE_ORDER = (
    GenerationStage.GAMES_COLLECTED,
    GenerationStage.TEACHER_LABELLED,
    GenerationStage.DATASET_FROZEN,
    GenerationStage.CANDIDATE_TRAINED,
    GenerationStage.BENCHMARK_COMPLETE,
    GenerationStage.FINALIZED,
)


def _require_identifier(value: str, *, label: str) -> str:
    if not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in _SHA256_CHARACTERS for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _stable_sha256(payload: object) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"refusing symlinked provenance artifact: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")

    try:
        payload: Any = json.loads(
            path.read_bytes(),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid UTF-8 JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> str:
    """Atomically install JSON without ever replacing an existing artifact.

    A byte-identical existing artifact is an idempotent resume.  A different
    artifact at the same path is a fail-closed configuration or lineage change.
    """

    serialized = _json_bytes(payload)
    digest = hashlib.sha256(serialized).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError(f"refusing create-only artifact symlink: {path}")
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot verify existing create-only artifact: {path}") from error
        if existing != serialized:
            raise FileExistsError(f"refusing to replace different create-only artifact: {path}")
        return digest

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            if path.is_symlink():
                raise FileExistsError(f"refusing concurrently-created symlink: {path}") from error
            if path.read_bytes() != serialized:
                raise FileExistsError(
                    f"refusing to replace concurrently-created artifact: {path}"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest


@dataclass(frozen=True, slots=True)
class Participant:
    """One pinned opponent artifact in the role-separated league."""

    participant_id: str
    role: OpponentRole
    artifact_sha256: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.participant_id, label="participant_id")
        _require_sha256(self.artifact_sha256, label="participant artifact")
        if self.participant_id in {_CANDIDATE_ID, _REFERENCE_ID}:
            raise ValueError(f"participant_id {self.participant_id!r} is reserved")
        if self.display_name is not None:
            _require_identifier(self.display_name, label="participant display_name")

    def to_manifest(self) -> dict[str, object]:
        return {
            "participant_id": self.participant_id,
            "role": self.role.value,
            "artifact_sha256": self.artifact_sha256,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, slots=True)
class TeacherSource:
    """A reviewed output-only teacher used after game generation."""

    teacher_id: str
    family: str
    artifact_sha256: str
    nodes: int
    multipv: int
    training_outputs_allowed: bool

    def __post_init__(self) -> None:
        _require_identifier(self.teacher_id, label="teacher_id")
        _require_identifier(self.family, label="teacher family")
        _require_sha256(self.artifact_sha256, label="teacher artifact")
        if self.nodes < 1:
            raise ValueError("teacher nodes must be positive")
        if self.multipv < 2:
            raise ValueError("continuous teaching requires MultiPV >= 2")
        if not self.training_outputs_allowed:
            raise PermissionError(
                f"teacher {self.teacher_id!r} is not approved for training-output generation"
            )

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundedExplorationConfig:
    """Training-only stochasticity restricted to deeply safe opening moves."""

    max_ply_exclusive: int = 24
    temperature: float = 0.8
    dirichlet_fraction: float = 0.0
    max_regret_upper_bound: float = 0.02
    minimum_nodes: int = 200_000
    minimum_multipv: int = 4
    minimum_teacher_agreement: int = 1

    def __post_init__(self) -> None:
        if self.max_ply_exclusive < 1:
            raise ValueError("exploration must be limited to at least one opening ply")
        if self.temperature <= 0:
            raise ValueError("training exploration temperature must be positive")
        if not 0 <= self.dirichlet_fraction < 1:
            raise ValueError("training exploration noise fraction must be in [0, 1)")
        if not 0 <= self.max_regret_upper_bound <= 2:
            raise ValueError("exploration regret bound must be in [0, 2]")
        if self.minimum_nodes < 1 or self.minimum_multipv < 2:
            raise ValueError("exploration requires a deep MultiPV search")
        if self.minimum_teacher_agreement < 1:
            raise ValueError("minimum teacher agreement must be positive")

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContinuousLoopConfig:
    """Immutable production policy for every generation in one lineage."""

    training_games_per_generation: int = 1024
    formal_opening_pairs: int = 500
    bootstrap_iterations: int = 20_000
    superiority_threshold: float = 0.5
    external_max_regression: float = 0.0
    weak_score_floor: float = 0.90
    random_score_floor: float = 1.0
    minimum_new_unique_train_games: int = 1
    require_flat_start_training: bool = True
    required_roles: tuple[OpponentRole, ...] = (
        OpponentRole.CHAMPION,
        OpponentRole.HISTORICAL_METEO,
        OpponentRole.EXTERNAL_TEACHER,
        OpponentRole.WEAK,
        OpponentRole.RANDOM,
    )
    exploration: BoundedExplorationConfig = BoundedExplorationConfig()
    evaluation_temperature: float = 0.0
    evaluation_dirichlet_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.training_games_per_generation < 2:
            raise ValueError("each generation must create at least two training games")
        if self.formal_opening_pairs < 500:
            raise ValueError("production promotion requires at least 500 unique opening pairs")
        if self.bootstrap_iterations < 100:
            raise ValueError("production bootstrap requires at least 100 iterations")
        if not 0 <= self.superiority_threshold < 1:
            raise ValueError("superiority threshold must be in [0, 1)")
        if not 0 <= self.external_max_regression <= 1:
            raise ValueError("external regression allowance must be in [0, 1]")
        if not 0 <= self.weak_score_floor <= 1:
            raise ValueError("weak score floor must be in [0, 1]")
        if not 0 <= self.random_score_floor <= 1:
            raise ValueError("random score floor must be in [0, 1]")
        if self.minimum_new_unique_train_games < 1:
            raise ValueError("dataset growth must add at least one unique training game")
        if not self.required_roles or len(set(self.required_roles)) != len(self.required_roles):
            raise ValueError("required opponent roles must be unique and non-empty")
        if OpponentRole.CHAMPION not in self.required_roles:
            raise ValueError("current champion must always be a required role")
        if self.evaluation_temperature != 0 or self.evaluation_dirichlet_fraction != 0:
            raise ValueError("formal evaluation must have temperature=0 and noise=0")

    def to_manifest(self) -> dict[str, object]:
        return {
            "training_games_per_generation": self.training_games_per_generation,
            "formal_opening_pairs": self.formal_opening_pairs,
            "formal_games_per_opponent": self.formal_opening_pairs * 2,
            "bootstrap_iterations": self.bootstrap_iterations,
            "superiority_threshold": self.superiority_threshold,
            "external_max_regression": self.external_max_regression,
            "weak_score_floor": self.weak_score_floor,
            "random_score_floor": self.random_score_floor,
            "minimum_new_unique_train_games": self.minimum_new_unique_train_games,
            "require_flat_start_training": self.require_flat_start_training,
            "required_roles": [role.value for role in self.required_roles],
            "exploration": self.exploration.to_manifest(),
            "evaluation_temperature": self.evaluation_temperature,
            "evaluation_dirichlet_fraction": self.evaluation_dirichlet_fraction,
        }


@dataclass(frozen=True, slots=True)
class ExplorationCandidate:
    """One move admitted by a deep MultiPV safety review."""

    move: str
    rank: int
    selection_weight: float
    regret_upper_bound: float
    nodes: int
    multipv_width: int
    agreeing_teachers: int
    proves_winning_mate: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.move, label="exploration move")
        if self.rank < 1 or self.selection_weight < 0 or not math.isfinite(self.selection_weight):
            raise ValueError("invalid exploration rank or selection weight")
        if not math.isfinite(self.regret_upper_bound) or self.regret_upper_bound < 0:
            raise ValueError("exploration regret must be a finite upper bound")
        if self.nodes < 1 or self.multipv_width < 1 or self.agreeing_teachers < 1:
            raise ValueError("exploration search evidence counts must be positive")


@dataclass(frozen=True, slots=True)
class ExplorationDecision:
    move: str
    reason: str
    eligible_moves: tuple[str, ...]
    regret_upper_bound: float
    temperature: float
    dirichlet_fraction: float


def choose_bounded_exploration_move(
    candidates: Sequence[ExplorationCandidate],
    *,
    ply: int,
    config: BoundedExplorationConfig,
    seed: int,
    evaluation: bool = False,
) -> ExplorationDecision:
    """Sample only among deep MultiPV moves with a proven regret upper bound.

    The rank-one move is returned deterministically in formal evaluation, after
    the opening window, or when no alternative clears every safety condition.
    If rank one proves a forced win, an alternative must also prove a forced win.
    """

    if not candidates:
        raise ValueError("at least one MultiPV candidate is required")
    if ply < 0 or seed < 0:
        raise ValueError("ply and exploration seed must be non-negative")
    moves = [candidate.move for candidate in candidates]
    if len(set(moves)) != len(moves):
        raise ValueError("MultiPV candidates must have unique root moves")
    ranks = [candidate.rank for candidate in candidates]
    if len(set(ranks)) != len(ranks) or 1 not in ranks:
        raise ValueError("MultiPV candidates must have unique ranks including rank one")
    best = next(candidate for candidate in candidates if candidate.rank == 1)

    if evaluation:
        return ExplorationDecision(
            best.move,
            "evaluation_deterministic",
            (best.move,),
            0.0,
            0.0,
            0.0,
        )
    if ply >= config.max_ply_exclusive:
        return ExplorationDecision(
            best.move,
            "after_opening_window",
            (best.move,),
            0.0,
            0.0,
            0.0,
        )
    if (
        best.nodes < config.minimum_nodes
        or best.multipv_width < config.minimum_multipv
        or best.agreeing_teachers < config.minimum_teacher_agreement
    ):
        return ExplorationDecision(
            best.move,
            "insufficient_deep_multipv_evidence",
            (best.move,),
            0.0,
            0.0,
            0.0,
        )

    eligible = tuple(
        candidate
        for candidate in sorted(candidates, key=lambda item: (item.rank, item.move))
        if candidate.nodes >= config.minimum_nodes
        and candidate.multipv_width >= config.minimum_multipv
        and candidate.agreeing_teachers >= config.minimum_teacher_agreement
        and candidate.regret_upper_bound <= config.max_regret_upper_bound
        and (not best.proves_winning_mate or candidate.proves_winning_mate)
    )
    if best not in eligible:
        eligible = (best, *eligible)
    if len(eligible) == 1:
        return ExplorationDecision(
            best.move,
            "no_safe_alternative",
            (best.move,),
            0.0,
            0.0,
            0.0,
        )

    weights = np.asarray(
        [max(candidate.selection_weight, 0.0) for candidate in eligible],
        dtype=np.float64,
    )
    if float(weights.sum()) <= 0:
        weights = np.ones(len(eligible), dtype=np.float64)
    weights = weights ** (1.0 / config.temperature)
    weights /= weights.sum()
    if config.dirichlet_fraction:
        generator = np.random.default_rng(seed)
        noise = generator.dirichlet(np.ones(len(eligible), dtype=np.float64))
        weights = (1.0 - config.dirichlet_fraction) * weights + config.dirichlet_fraction * noise
    chosen_index = int(np.random.default_rng(seed).choice(len(eligible), p=weights))
    chosen = eligible[chosen_index]
    return ExplorationDecision(
        chosen.move,
        "bounded_opening_exploration",
        tuple(candidate.move for candidate in eligible),
        chosen.regret_upper_bound,
        config.temperature,
        config.dirichlet_fraction,
    )


@dataclass(frozen=True, slots=True)
class PlannedGame:
    """A deterministic work item; randomness is carried only by its pinned seed."""

    job_id: str
    purpose: Literal["training", "evaluation", "reference_evaluation"]
    opening_sfen: str
    opening_key: str
    subject_id: str
    opponent_id: str
    subject_color: Literal["black", "white"]
    seed: int
    exploration_enabled: bool
    temperature: float
    dirichlet_fraction: float

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Complete immutable work schedule for one candidate generation."""

    generation: int
    lineage_id: str
    previous_state_sha256: str | None
    dataset_parent_sha256: str
    champion: Participant
    participants: tuple[Participant, ...]
    teachers: tuple[TeacherSource, ...]
    required_historical_artifact_sha256: tuple[str, ...]
    training_openings: tuple[OpeningPosition, ...]
    evaluation_openings: tuple[OpeningPosition, ...]
    training_jobs: tuple[PlannedGame, ...]
    evaluation_jobs: tuple[PlannedGame, ...]
    reference_evaluation_jobs: tuple[PlannedGame, ...]
    config: ContinuousLoopConfig
    seed: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": LOOP_PLAN_SCHEMA,
            "generation": self.generation,
            "lineage_id": self.lineage_id,
            "previous_state_sha256": self.previous_state_sha256,
            "dataset_parent_sha256": self.dataset_parent_sha256,
            "champion": self.champion.to_manifest(),
            "participants": [participant.to_manifest() for participant in self.participants],
            "teachers": [teacher.to_manifest() for teacher in self.teachers],
            "required_historical_artifact_sha256": list(self.required_historical_artifact_sha256),
            "training_openings": [opening.to_manifest() for opening in self.training_openings],
            "evaluation_openings": [opening.to_manifest() for opening in self.evaluation_openings],
            "training_jobs": [job.to_manifest() for job in self.training_jobs],
            "evaluation_jobs": [job.to_manifest() for job in self.evaluation_jobs],
            "reference_evaluation_jobs": [
                job.to_manifest() for job in self.reference_evaluation_jobs
            ],
            "config": self.config.to_manifest(),
            "seed": self.seed,
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_manifest())


def _job_seed(
    *,
    lineage_id: str,
    generation: int,
    purpose: str,
    index: int,
    opening_key: str,
    subject_id: str,
    opponent_id: str,
    color: str,
    seed: int,
) -> int:
    identity = {
        "lineage_id": lineage_id,
        "generation": generation,
        "purpose": purpose,
        "index": index,
        "opening_key": opening_key,
        "subject_id": subject_id,
        "opponent_id": opponent_id,
        "color": color,
        "base_seed": seed,
    }
    return int.from_bytes(hashlib.sha256(_json_bytes(identity)).digest()[:8], "big") & (2**63 - 1)


def _planned_game(
    *,
    lineage_id: str,
    generation: int,
    purpose: Literal["training", "evaluation", "reference_evaluation"],
    index: int,
    opening: OpeningPosition,
    subject_id: str,
    opponent_id: str,
    color: Literal["black", "white"],
    seed: int,
    exploration: BoundedExplorationConfig | None,
) -> PlannedGame:
    resolved_seed = _job_seed(
        lineage_id=lineage_id,
        generation=generation,
        purpose=purpose,
        index=index,
        opening_key=opening.normalized_key,
        subject_id=subject_id,
        opponent_id=opponent_id,
        color=color,
        seed=seed,
    )
    identity = {
        "generation": generation,
        "purpose": purpose,
        "index": index,
        "opening_key": opening.normalized_key,
        "subject_id": subject_id,
        "opponent_id": opponent_id,
        "subject_color": color,
        "seed": resolved_seed,
    }
    return PlannedGame(
        job_id=_stable_sha256(identity),
        purpose=purpose,
        opening_sfen=opening.sfen,
        opening_key=opening.normalized_key,
        subject_id=subject_id,
        opponent_id=opponent_id,
        subject_color=color,
        seed=resolved_seed,
        exploration_enabled=exploration is not None,
        temperature=0.0 if exploration is None else exploration.temperature,
        dirichlet_fraction=0.0 if exploration is None else exploration.dirichlet_fraction,
    )


def _canonical_unique_openings(sfens: Sequence[str], *, label: str) -> tuple[OpeningPosition, ...]:
    if not sfens:
        raise ValueError(f"{label} openings must not be empty")
    positions = tuple(OpeningPosition.from_sfen(sfen) for sfen in sfens)
    keys = [position.normalized_key for position in positions]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} openings contain duplicate normalized SFEN positions")
    return positions


def build_generation_plan(
    *,
    generation: int,
    lineage_id: str,
    dataset_parent_sha256: str,
    champion: Participant,
    participants: Sequence[Participant],
    teachers: Sequence[TeacherSource],
    training_opening_sfens: Sequence[str],
    evaluation_opening_sfens: Sequence[str],
    config: ContinuousLoopConfig | None = None,
    seed: int = 0,
    previous_state_sha256: str | None = None,
    previous_state: dict[str, Any] | None = None,
) -> GenerationPlan:
    """Build a balanced schedule with flat-start training and 1000-game gates."""

    if generation < 1 or seed < 0:
        raise ValueError("generation must be positive and seed must be non-negative")
    if config is None:
        config = ContinuousLoopConfig()
    _require_identifier(lineage_id, label="lineage_id")
    _require_sha256(dataset_parent_sha256, label="parent dataset")
    required_historical_artifacts: tuple[str, ...] = ()
    if previous_state is not None:
        if previous_state.get("schema") != LOOP_STATE_SCHEMA:
            raise ValueError("previous loop state has an unsupported schema")
        if previous_state.get("generation") != generation - 1:
            raise ValueError("previous loop state generation is not contiguous")
        computed_previous_sha256 = _stable_sha256(previous_state)
        if previous_state_sha256 is None:
            previous_state_sha256 = computed_previous_sha256
        elif previous_state_sha256 != computed_previous_sha256:
            raise ValueError("previous loop state digest does not match its payload")
        raw_history = previous_state.get("model_history")
        if not isinstance(raw_history, list):
            raise ValueError("previous loop state model_history must be a list")
        history_digests: list[str] = []
        for item in raw_history:
            if not isinstance(item, dict):
                raise ValueError("previous model_history contains a non-object entry")
            digest = item.get("artifact_sha256")
            if not isinstance(digest, str):
                raise ValueError("previous model_history artifact digest is missing")
            history_digests.append(_require_sha256(digest, label="historical model"))
        required_historical_artifacts = tuple(sorted(set(history_digests)))
    elif previous_state_sha256 is not None:
        raise ValueError("previous state payload is required to prove all-generation coverage")
    if champion.role is not OpponentRole.CHAMPION:
        raise ValueError("champion participant must have role=champion")

    participant_tuple = tuple(participants)
    if not participant_tuple:
        raise ValueError("at least one evaluation participant is required")
    participant_ids = [participant.participant_id for participant in participant_tuple]
    if len(set(participant_ids)) != len(participant_ids):
        raise ValueError("participant IDs must be unique")
    champion_matches = [
        participant
        for participant in participant_tuple
        if participant.role is OpponentRole.CHAMPION
    ]
    if champion_matches != [champion]:
        raise ValueError("participants must contain exactly the pinned current champion")
    present_roles = {participant.role for participant in participant_tuple}
    missing_roles = set(config.required_roles) - present_roles
    if missing_roles:
        raise ValueError(
            "missing required opponent roles: "
            + ", ".join(sorted(role.value for role in missing_roles))
        )
    covered_history = {
        participant.artifact_sha256
        for participant in participant_tuple
        if participant.role in {OpponentRole.CHAMPION, OpponentRole.HISTORICAL_METEO}
    }
    missing_history = set(required_historical_artifacts) - covered_history
    if missing_history:
        raise ValueError(
            "not every prior Meteo generation is scheduled for evaluation: "
            + ", ".join(sorted(missing_history))
        )

    teacher_tuple = tuple(teachers)
    teacher_ids = [teacher.teacher_id for teacher in teacher_tuple]
    if not teacher_tuple or len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("teacher-assisted generations require unique non-empty teacher IDs")
    training_openings = _canonical_unique_openings(
        training_opening_sfens,
        label="training",
    )
    evaluation_openings = _canonical_unique_openings(
        evaluation_opening_sfens,
        label="evaluation",
    )
    training_keys = {opening.normalized_key for opening in training_openings}
    evaluation_keys = {opening.normalized_key for opening in evaluation_openings}
    overlap = training_keys & evaluation_keys
    if overlap:
        raise ValueError("training and formal evaluation opening splits overlap")
    if config.require_flat_start_training and _FLAT_START.normalized_key not in training_keys:
        raise ValueError("training openings must include the ordinary flat starting position")
    if len(evaluation_openings) < config.formal_opening_pairs:
        raise ValueError(
            "formal evaluation requires at least "
            f"{config.formal_opening_pairs} unique openings (1000 games per opponent)"
        )

    opponents = tuple(
        sorted(
            participant_tuple,
            key=lambda item: (item.role.value, item.participant_id),
        )
    )
    flat = next(
        opening
        for opening in training_openings
        if opening.normalized_key == _FLAT_START.normalized_key
    )
    training_schedule_openings = (flat, flat, *training_openings)
    training_jobs: list[PlannedGame] = []
    for index in range(config.training_games_per_generation):
        opening = training_schedule_openings[index % len(training_schedule_openings)]
        opponent = opponents[index % len(opponents)]
        color: Literal["black", "white"] = "black" if index % 2 == 0 else "white"
        training_jobs.append(
            _planned_game(
                lineage_id=lineage_id,
                generation=generation,
                purpose="training",
                index=index,
                opening=opening,
                subject_id=champion.participant_id,
                opponent_id=opponent.participant_id,
                color=color,
                seed=seed,
                exploration=config.exploration,
            )
        )

    evaluation_jobs: list[PlannedGame] = []
    reference_jobs: list[PlannedGame] = []
    job_index = 0
    reference_index = 0
    for opponent in opponents:
        for opening in evaluation_openings:
            for color in ("black", "white"):
                evaluation_jobs.append(
                    _planned_game(
                        lineage_id=lineage_id,
                        generation=generation,
                        purpose="evaluation",
                        index=job_index,
                        opening=opening,
                        subject_id=_CANDIDATE_ID,
                        opponent_id=opponent.participant_id,
                        color=color,
                        seed=seed,
                        exploration=None,
                    )
                )
                job_index += 1
                if opponent.role is OpponentRole.EXTERNAL_TEACHER:
                    reference_jobs.append(
                        _planned_game(
                            lineage_id=lineage_id,
                            generation=generation,
                            purpose="reference_evaluation",
                            index=reference_index,
                            opening=opening,
                            subject_id=_REFERENCE_ID,
                            opponent_id=opponent.participant_id,
                            color=color,
                            seed=seed,
                            exploration=None,
                        )
                    )
                    reference_index += 1

    all_jobs = (*training_jobs, *evaluation_jobs, *reference_jobs)
    job_ids = [job.job_id for job in all_jobs]
    seeds = [job.seed for job in all_jobs]
    if len(set(job_ids)) != len(job_ids) or len(set(seeds)) != len(seeds):
        raise AssertionError("generation planning produced duplicate work identities")
    return GenerationPlan(
        generation=generation,
        lineage_id=lineage_id,
        previous_state_sha256=previous_state_sha256,
        dataset_parent_sha256=dataset_parent_sha256,
        champion=champion,
        participants=participant_tuple,
        teachers=teacher_tuple,
        required_historical_artifact_sha256=required_historical_artifacts,
        training_openings=training_openings,
        evaluation_openings=evaluation_openings,
        training_jobs=tuple(training_jobs),
        evaluation_jobs=tuple(evaluation_jobs),
        reference_evaluation_jobs=tuple(reference_jobs),
        config=config,
        seed=seed,
    )


def write_generation_plan(generation_dir: Path, plan: GenerationPlan) -> str:
    """Create or idempotently resume an immutable generation plan."""

    expected_name = f"generation-{plan.generation:06d}"
    if generation_dir.name != expected_name:
        raise ValueError(f"generation directory must be named {expected_name!r}")
    return _write_json_create_only(generation_dir / "plan.json", plan.to_manifest())


def _receipt_path(generation_dir: Path, stage: GenerationStage) -> Path:
    return generation_dir / f"receipt-{_STAGE_ORDER.index(stage) + 1:02d}-{stage.value}.json"


def validate_stage_chain(generation_dir: Path) -> tuple[GenerationStage | None, str | None]:
    """Verify every receipt against the plan and its predecessor digest."""

    plan_path = generation_dir / "plan.json"
    plan = _read_strict_json(plan_path)
    if plan.get("schema") != LOOP_PLAN_SCHEMA:
        raise ValueError("unsupported or missing generation plan schema")
    plan_sha256 = _file_sha256(plan_path)
    previous: str | None = None
    last_stage: GenerationStage | None = None
    gap_seen = False
    for stage in _STAGE_ORDER:
        path = _receipt_path(generation_dir, stage)
        if not path.exists():
            gap_seen = True
            continue
        if gap_seen:
            raise ValueError("generation receipt chain contains a stage gap")
        payload = _read_strict_json(path)
        if payload.get("schema") != STAGE_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported receipt schema: {path}")
        if payload.get("stage") != stage.value:
            raise ValueError(f"receipt filename/stage mismatch: {path}")
        if payload.get("generation") != plan.get("generation"):
            raise ValueError(f"receipt generation does not match plan: {path}")
        if payload.get("plan_sha256") != plan_sha256:
            raise ValueError(f"receipt plan digest does not match current plan: {path}")
        if payload.get("previous_receipt_sha256") != previous:
            raise ValueError(f"receipt predecessor digest mismatch: {path}")
        previous = _file_sha256(path)
        last_stage = stage
    return last_stage, previous


def append_stage_receipt(
    generation_dir: Path,
    stage: GenerationStage,
    evidence: dict[str, Any],
) -> str:
    """Append exactly the next receipt, idempotently, without rewriting history."""

    plan_path = generation_dir / "plan.json"
    plan = _read_strict_json(plan_path)
    if plan.get("schema") != LOOP_PLAN_SCHEMA:
        raise ValueError("unsupported or missing generation plan schema")
    current_stage, previous = validate_stage_chain(generation_dir)
    requested_index = _STAGE_ORDER.index(stage)
    current_index = -1 if current_stage is None else _STAGE_ORDER.index(current_stage)
    target = _receipt_path(generation_dir, stage)
    if requested_index > current_index + 1:
        next_stage = _STAGE_ORDER[current_index + 1]
        raise ValueError(f"cannot append {stage.value!r}; next stage is {next_stage.value!r}")
    predecessor_sha256 = (
        None
        if requested_index == 0
        else _file_sha256(_receipt_path(generation_dir, _STAGE_ORDER[requested_index - 1]))
    )
    payload = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "generation": plan["generation"],
        "stage": stage.value,
        "plan_sha256": _file_sha256(plan_path),
        "previous_receipt_sha256": predecessor_sha256,
        "evidence": evidence,
    }
    if requested_index <= current_index:
        return _write_json_create_only(target, payload)
    if payload["previous_receipt_sha256"] != previous:
        raise AssertionError("new receipt does not extend the validated chain")
    return _write_json_create_only(target, payload)


@dataclass(frozen=True, slots=True)
class GameArtifact:
    """Identity and usage contract for one completed game file."""

    job_id: str
    trajectory_sha256: str
    artifact_sha256: str
    split: DatasetSplit
    opening_key: str
    producer_id: str
    execution_seed: int
    retry: int
    complete: bool
    eligible_for_training: bool
    teacher_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.job_id, label="game job_id")
        _require_sha256(self.trajectory_sha256, label="game trajectory")
        _require_sha256(self.artifact_sha256, label="game artifact")
        _require_identifier(self.opening_key, label="game opening key")
        _require_identifier(self.producer_id, label="game producer_id")
        if self.execution_seed < 0 or self.retry < 0:
            raise ValueError("game execution seed and retry must be non-negative")
        if len(set(self.teacher_ids)) != len(self.teacher_ids):
            raise ValueError("game teacher IDs must be unique")
        for teacher_id in self.teacher_ids:
            _require_identifier(teacher_id, label="game teacher_id")

    def to_manifest(self) -> dict[str, object]:
        result = asdict(self)
        result["split"] = self.split.value
        return result


@dataclass(frozen=True, slots=True)
class DatasetGrowthManifest:
    payload: dict[str, Any]
    snapshot_sha256: str


def freeze_dataset_growth(
    plan: GenerationPlan,
    entries: Sequence[GameArtifact],
    *,
    parent_train_trajectories: Sequence[str],
) -> DatasetGrowthManifest:
    """Freeze unique training growth while keeping every evaluation split sealed."""

    entry_tuple = tuple(entries)
    if not entry_tuple:
        raise ValueError("dataset growth requires completed game artifacts")
    parent = tuple(parent_train_trajectories)
    for digest in parent:
        _require_sha256(digest, label="parent training trajectory")
    if len(set(parent)) != len(parent):
        raise ValueError("parent training trajectory identities contain duplicates")

    job_ids = [entry.job_id for entry in entry_tuple]
    trajectories = [entry.trajectory_sha256 for entry in entry_tuple]
    artifacts = [entry.artifact_sha256 for entry in entry_tuple]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("dataset entries contain duplicate job IDs")
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("dataset entries contain deterministic duplicate trajectories")
    if len(set(artifacts)) != len(artifacts):
        raise ValueError("one artifact cannot be reused across dataset splits")
    if any(not entry.complete for entry in entry_tuple):
        raise ValueError("incomplete games cannot enter a frozen dataset snapshot")

    known_teachers = {teacher.teacher_id for teacher in plan.teachers}
    training_job_ids = {job.job_id for job in plan.training_jobs}
    evaluation_job_ids = {
        job.job_id for job in (*plan.evaluation_jobs, *plan.reference_evaluation_jobs)
    }
    train_entries = tuple(entry for entry in entry_tuple if entry.split is DatasetSplit.TRAIN)
    if len(train_entries) < plan.config.minimum_new_unique_train_games:
        raise ValueError("generation did not add enough unique training games")
    parent_set = set(parent)
    if parent_set & {entry.trajectory_sha256 for entry in train_entries}:
        raise ValueError("new training games duplicate the parent dataset")
    planned_by_job = {
        job.job_id: job
        for job in (*plan.training_jobs, *plan.evaluation_jobs, *plan.reference_evaluation_jobs)
    }
    for entry in entry_tuple:
        planned_job = planned_by_job.get(entry.job_id)
        if planned_job is None:
            raise ValueError("game artifact does not belong to the generation plan")
        if entry.retry == 0:
            if entry.execution_seed != planned_job.seed:
                raise ValueError("game execution seed does not match its immutable plan")
        elif entry.split is not DatasetSplit.TRAIN:
            raise ValueError("formal evaluation games cannot use stochastic retry seeds")
        elif entry.execution_seed != deterministic_retry_seed(planned_job, retry=entry.retry):
            raise ValueError("training retry seed does not match its deterministic derivation")
        if entry.split is DatasetSplit.TRAIN:
            if entry.job_id not in training_job_ids:
                raise ValueError("training artifact does not belong to the generation plan")
            if not entry.eligible_for_training:
                raise ValueError("training split entry must explicitly permit training")
            if not entry.teacher_ids:
                raise ValueError("teacher-assisted training entry lacks teacher labels")
            unknown = set(entry.teacher_ids) - known_teachers
            if unknown:
                raise ValueError(f"training entry references unknown teachers: {sorted(unknown)}")
        elif entry.split is DatasetSplit.ARENA:
            if entry.job_id not in evaluation_job_ids:
                raise ValueError("arena artifact does not belong to the formal evaluation plan")
            if entry.eligible_for_training:
                raise ValueError("arena games are permanently ineligible for training")
        elif entry.eligible_for_training:
            raise ValueError(f"{entry.split.value} entries cannot be marked for training")

    split_openings: dict[DatasetSplit, set[str]] = {
        split: {entry.opening_key for entry in entry_tuple if entry.split is split}
        for split in DatasetSplit
    }
    split_pairs = (
        (DatasetSplit.TRAIN, DatasetSplit.VALIDATION),
        (DatasetSplit.TRAIN, DatasetSplit.SEALED_TEST),
        (DatasetSplit.TRAIN, DatasetSplit.ARENA),
        (DatasetSplit.VALIDATION, DatasetSplit.SEALED_TEST),
        (DatasetSplit.VALIDATION, DatasetSplit.ARENA),
        (DatasetSplit.SEALED_TEST, DatasetSplit.ARENA),
    )
    for left, right in split_pairs:
        if split_openings[left] & split_openings[right]:
            raise ValueError(f"{left.value} and {right.value} opening identities overlap")
    if (
        plan.config.require_flat_start_training
        and _FLAT_START.normalized_key not in split_openings[DatasetSplit.TRAIN]
    ):
        raise ValueError("frozen generation training games must include flat start")

    parent_identity_sha256 = _stable_sha256(sorted(parent))
    payload: dict[str, Any] = {
        "schema": DATASET_GROWTH_SCHEMA,
        "generation": plan.generation,
        "plan_sha256": plan.sha256,
        "parent_snapshot_sha256": plan.dataset_parent_sha256,
        "parent_train_trajectory_count": len(parent),
        "parent_train_trajectory_set_sha256": parent_identity_sha256,
        "new_unique_train_games": len(train_entries),
        "cumulative_unique_train_games": len(parent) + len(train_entries),
        "failed_candidates_may_contribute_after_teacher_review": True,
        "arena_training_allowed": False,
        "entries": [entry.to_manifest() for entry in entry_tuple],
        "split_counts": {
            split.value: sum(entry.split is split for entry in entry_tuple)
            for split in DatasetSplit
        },
        "split_opening_set_sha256": {
            split.value: _stable_sha256(sorted(keys)) for split, keys in split_openings.items()
        },
    }
    return DatasetGrowthManifest(payload=payload, snapshot_sha256=_stable_sha256(payload))


@dataclass(frozen=True, slots=True)
class BenchmarkEvidence:
    """Candidate points for one opponent, clustered by unique opening."""

    opponent_id: str
    opponent_role: OpponentRole
    opening_keys: tuple[str, ...]
    candidate_black_points: tuple[float, ...]
    candidate_white_points: tuple[float, ...]
    game_sha256: tuple[str, ...]
    incomplete_games: int = 0
    temperature: float = 0.0
    dirichlet_fraction: float = 0.0
    reference_black_points: tuple[float, ...] | None = None
    reference_white_points: tuple[float, ...] | None = None
    reference_game_sha256: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class GateMetric:
    score: float
    lower_95: float
    upper_95: float

    def to_manifest(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OpponentGateResult:
    opponent_id: str
    role: OpponentRole
    pairs: int
    games: int
    black: GateMetric
    white: GateMetric
    paired: GateMetric
    reference_delta_black: GateMetric | None
    reference_delta_white: GateMetric | None
    reference_delta_paired: GateMetric | None
    external_target_beaten: bool | None
    passed: bool
    blockers: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "opponent_id": self.opponent_id,
            "role": self.role.value,
            "pairs": self.pairs,
            "games": self.games,
            "black": self.black.to_manifest(),
            "white": self.white.to_manifest(),
            "paired": self.paired.to_manifest(),
            "reference_delta_black": (
                None
                if self.reference_delta_black is None
                else self.reference_delta_black.to_manifest()
            ),
            "reference_delta_white": (
                None
                if self.reference_delta_white is None
                else self.reference_delta_white.to_manifest()
            ),
            "reference_delta_paired": (
                None
                if self.reference_delta_paired is None
                else self.reference_delta_paired.to_manifest()
            ),
            "external_target_beaten": self.external_target_beaten,
            "passed": self.passed,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    plan_sha256: str
    promoted: bool
    blockers: tuple[str, ...]
    opponent_results: tuple[OpponentGateResult, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema": PROMOTION_DECISION_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "promoted": self.promoted,
            "blockers": list(self.blockers),
            "opponent_results": [result.to_manifest() for result in self.opponent_results],
        }

    @property
    def sha256(self) -> str:
        return _stable_sha256(self.to_manifest())


def _bootstrap_metric(values: Sequence[float], *, iterations: int, seed: int) -> GateMetric:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("benchmark metric values must be finite and non-empty")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    chunk_size = max(1, min(iterations, 1_000_000 // len(array)))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        indices = rng.integers(0, len(array), size=(stop - start, len(array)))
        means[start:stop] = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.05, 0.95))
    return GateMetric(float(array.mean()), float(lower), float(upper))


def _validate_benchmark_evidence(
    plan: GenerationPlan,
    evidence: BenchmarkEvidence,
) -> tuple[Participant, tuple[float, ...], tuple[float, ...]]:
    participant_by_id = {
        participant.participant_id: participant for participant in plan.participants
    }
    participant = participant_by_id.get(evidence.opponent_id)
    if participant is None or participant.role is not evidence.opponent_role:
        raise ValueError("benchmark opponent identity/role does not match the generation plan")
    expected_keys = tuple(opening.normalized_key for opening in plan.evaluation_openings)
    if evidence.opening_keys != expected_keys:
        raise ValueError("benchmark opening keys/order do not match the immutable evaluation split")
    if len(set(evidence.opening_keys)) != len(evidence.opening_keys):
        raise ValueError("benchmark contains repeated normalized openings")
    count = len(evidence.opening_keys)
    if count < plan.config.formal_opening_pairs:
        raise ValueError("benchmark has fewer than 500 independent opening pairs")
    if (
        len(evidence.candidate_black_points) != count
        or len(evidence.candidate_white_points) != count
    ):
        raise ValueError("benchmark color result counts do not match opening pairs")
    allowed = {0.0, 0.5, 1.0}
    candidate_points = (*evidence.candidate_black_points, *evidence.candidate_white_points)
    if any(point not in allowed for point in candidate_points):
        raise ValueError("benchmark points must be losses, draws, or wins")
    if evidence.incomplete_games != 0:
        raise ValueError("formal benchmark cannot contain incomplete games")
    if evidence.temperature != 0 or evidence.dirichlet_fraction != 0:
        raise ValueError("formal benchmark must use temperature=0 and noise=0")
    if len(evidence.game_sha256) != 2 * count:
        raise ValueError("benchmark must identify exactly two games per opening")
    for digest in evidence.game_sha256:
        _require_sha256(digest, label="benchmark game")
    if len(set(evidence.game_sha256)) != len(evidence.game_sha256):
        raise ValueError("benchmark contains duplicate game artifacts")

    references = (evidence.reference_black_points, evidence.reference_white_points)
    if participant.role is OpponentRole.EXTERNAL_TEACHER:
        if any(reference is None for reference in references):
            raise ValueError("external-teacher gate requires same-opening champion reference games")
        if any(len(reference or ()) != count for reference in references):
            raise ValueError("external reference counts do not match opening pairs")
        if any(point not in allowed for reference in references for point in (reference or ())):
            raise ValueError("external reference points must be losses, draws, or wins")
        if (
            evidence.reference_game_sha256 is None
            or len(evidence.reference_game_sha256) != 2 * count
        ):
            raise ValueError("external reference must identify exactly two games per opening")
        for digest in evidence.reference_game_sha256:
            _require_sha256(digest, label="external reference game")
        if len(set(evidence.reference_game_sha256)) != len(evidence.reference_game_sha256):
            raise ValueError("external reference contains duplicate game artifacts")
    elif any(reference is not None for reference in references):
        raise ValueError("reference points are accepted only for external-teacher gates")
    elif evidence.reference_game_sha256 is not None:
        raise ValueError("reference games are accepted only for external-teacher gates")
    return participant, evidence.candidate_black_points, evidence.candidate_white_points


def evaluate_promotion(
    plan: GenerationPlan,
    evidence: Sequence[BenchmarkEvidence],
) -> PromotionDecision:
    """Apply 1000-game, color-stable, role-specific promotion gates."""

    evidence_tuple = tuple(evidence)
    if len({item.opponent_id for item in evidence_tuple}) != len(evidence_tuple):
        raise ValueError("each planned opponent must have exactly one benchmark report")
    expected_ids = {participant.participant_id for participant in plan.participants}
    actual_ids = {item.opponent_id for item in evidence_tuple}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"benchmark opponent set mismatch; missing={missing}, unexpected={unexpected}"
        )
    all_game_digests = [digest for item in evidence_tuple for digest in item.game_sha256]
    all_game_digests.extend(
        digest for item in evidence_tuple for digest in (item.reference_game_sha256 or ())
    )
    if len(set(all_game_digests)) != len(all_game_digests):
        raise ValueError("a benchmark game artifact was reused across opponents")

    results: list[OpponentGateResult] = []
    aggregate_blockers: list[str] = []
    for report_index, item in enumerate(evidence_tuple):
        participant, black_points, white_points = _validate_benchmark_evidence(plan, item)
        pair_points = tuple(
            (black + white) / 2.0 for black, white in zip(black_points, white_points, strict=True)
        )
        base_seed = plan.seed + 10_000_000 + report_index * 20
        black = _bootstrap_metric(
            black_points,
            iterations=plan.config.bootstrap_iterations,
            seed=base_seed,
        )
        white = _bootstrap_metric(
            white_points,
            iterations=plan.config.bootstrap_iterations,
            seed=base_seed + 1,
        )
        paired = _bootstrap_metric(
            pair_points,
            iterations=plan.config.bootstrap_iterations,
            seed=base_seed + 2,
        )
        delta_black: GateMetric | None = None
        delta_white: GateMetric | None = None
        delta_paired: GateMetric | None = None
        external_target_beaten: bool | None = None
        blockers: list[str] = []

        if participant.role is OpponentRole.CHAMPION:
            threshold = plan.config.superiority_threshold
            for color, metric in (("black", black), ("white", white), ("paired", paired)):
                if metric.lower_95 <= threshold:
                    blockers.append(f"{color}_superiority_not_proven")
        elif participant.role is OpponentRole.HISTORICAL_METEO:
            threshold = plan.config.superiority_threshold
            for color, metric in (("black", black), ("white", white), ("paired", paired)):
                if metric.lower_95 < threshold:
                    blockers.append(f"{color}_historical_regression")
        elif participant.role is OpponentRole.EXTERNAL_TEACHER:
            reference_black = item.reference_black_points
            reference_white = item.reference_white_points
            if reference_black is None or reference_white is None:
                raise AssertionError("validated external evidence lost its references")
            reference_pair = tuple(
                (black_point + white_point) / 2.0
                for black_point, white_point in zip(
                    reference_black,
                    reference_white,
                    strict=True,
                )
            )
            delta_black = _bootstrap_metric(
                tuple(
                    candidate - reference
                    for candidate, reference in zip(black_points, reference_black, strict=True)
                ),
                iterations=plan.config.bootstrap_iterations,
                seed=base_seed + 3,
            )
            delta_white = _bootstrap_metric(
                tuple(
                    candidate - reference
                    for candidate, reference in zip(white_points, reference_white, strict=True)
                ),
                iterations=plan.config.bootstrap_iterations,
                seed=base_seed + 4,
            )
            delta_paired = _bootstrap_metric(
                tuple(
                    candidate - reference
                    for candidate, reference in zip(pair_points, reference_pair, strict=True)
                ),
                iterations=plan.config.bootstrap_iterations,
                seed=base_seed + 5,
            )
            allowed_regression = -plan.config.external_max_regression
            for color, metric in (
                ("black", delta_black),
                ("white", delta_white),
                ("paired", delta_paired),
            ):
                if metric.lower_95 < allowed_regression:
                    blockers.append(f"{color}_external_teacher_regression")
            external_target_beaten = (
                black.lower_95 > plan.config.superiority_threshold
                and white.lower_95 > plan.config.superiority_threshold
                and paired.lower_95 > plan.config.superiority_threshold
            )
        elif participant.role is OpponentRole.WEAK:
            for color, metric in (("black", black), ("white", white), ("paired", paired)):
                if metric.lower_95 < plan.config.weak_score_floor:
                    blockers.append(f"{color}_weak_opponent_floor_missed")
        elif participant.role is OpponentRole.RANDOM:
            for color, metric in (("black", black), ("white", white), ("paired", paired)):
                if metric.lower_95 < plan.config.random_score_floor:
                    blockers.append(f"{color}_random_floor_missed")
        else:  # pragma: no cover - exhaustive StrEnum defense
            raise AssertionError(f"unhandled opponent role: {participant.role}")

        prefixed = tuple(f"{participant.participant_id}:{blocker}" for blocker in blockers)
        aggregate_blockers.extend(prefixed)
        results.append(
            OpponentGateResult(
                opponent_id=participant.participant_id,
                role=participant.role,
                pairs=len(item.opening_keys),
                games=2 * len(item.opening_keys),
                black=black,
                white=white,
                paired=paired,
                reference_delta_black=delta_black,
                reference_delta_white=delta_white,
                reference_delta_paired=delta_paired,
                external_target_beaten=external_target_beaten,
                passed=not blockers,
                blockers=prefixed,
            )
        )
    return PromotionDecision(
        plan_sha256=plan.sha256,
        promoted=not aggregate_blockers,
        blockers=tuple(aggregate_blockers),
        opponent_results=tuple(results),
    )


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    candidate_id: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_id, label="candidate_id")
        _require_sha256(self.artifact_sha256, label="candidate artifact")

    def to_manifest(self) -> dict[str, str]:
        return asdict(self)


def build_loop_state(
    plan: GenerationPlan,
    decision: PromotionDecision,
    dataset: DatasetGrowthManifest,
    candidate: CandidateArtifact,
    *,
    previous_state: dict[str, Any] | None = None,
    final_receipt_sha256: str,
) -> dict[str, Any]:
    """Advance data every generation, but the champion only after every gate passes."""

    _require_sha256(final_receipt_sha256, label="final receipt")
    if decision.plan_sha256 != plan.sha256:
        raise ValueError("promotion decision belongs to a different generation plan")
    if dataset.payload.get("plan_sha256") != plan.sha256:
        raise ValueError("dataset snapshot belongs to a different generation plan")
    history: list[dict[str, Any]] = []
    if previous_state is not None:
        if previous_state.get("schema") != LOOP_STATE_SCHEMA:
            raise ValueError("previous loop state has an unsupported schema")
        if previous_state.get("generation") != plan.generation - 1:
            raise ValueError("previous loop state generation is not contiguous")
        if _stable_sha256(previous_state) != plan.previous_state_sha256:
            raise ValueError("generation plan does not extend the supplied previous state")
        raw_history = previous_state.get("model_history")
        if not isinstance(raw_history, list):
            raise ValueError("previous model_history must be a list")
        history = [dict(item) for item in raw_history if isinstance(item, dict)]
        if len(history) != len(raw_history):
            raise ValueError("previous model_history contains a non-object entry")
    elif plan.previous_state_sha256 is not None:
        raise ValueError("previous state is required by this generation plan")

    history.append(
        {
            **candidate.to_manifest(),
            "generation": plan.generation,
            "promotion_status": "promoted" if decision.promoted else "rejected",
            "eligible_as_historical_opponent": True,
        }
    )
    champion_after: dict[str, str]
    if decision.promoted:
        champion_after = candidate.to_manifest()
    else:
        champion_after = {
            "candidate_id": plan.champion.participant_id,
            "artifact_sha256": plan.champion.artifact_sha256,
        }
    return {
        "schema": LOOP_STATE_SCHEMA,
        "generation": plan.generation,
        "next_generation": plan.generation + 1,
        "previous_state_sha256": plan.previous_state_sha256,
        "plan_sha256": plan.sha256,
        "final_receipt_sha256": final_receipt_sha256,
        "promotion_decision_sha256": decision.sha256,
        "promotion_status": "promoted" if decision.promoted else "rejected",
        "champion_before": {
            "candidate_id": plan.champion.participant_id,
            "artifact_sha256": plan.champion.artifact_sha256,
        },
        "champion_after": champion_after,
        "dataset_before_sha256": plan.dataset_parent_sha256,
        "dataset_after_sha256": dataset.snapshot_sha256,
        "dataset_advanced_even_if_candidate_rejected": True,
        "model_history": history,
    }


def append_loop_state(workdir: Path, state: dict[str, Any]) -> str:
    """Append one contiguous state record and validate the whole state chain."""

    if state.get("schema") != LOOP_STATE_SCHEMA:
        raise ValueError("loop state has an unsupported schema")
    generation = state.get("generation")
    if not isinstance(generation, int) or generation < 1:
        raise ValueError("loop state generation must be a positive integer")
    states_dir = workdir / "states"
    existing = sorted(states_dir.glob("state-*.json")) if states_dir.exists() else []
    if existing:
        expected_names = [f"state-{index:06d}.json" for index in range(1, len(existing) + 1)]
        if [path.name for path in existing] != expected_names:
            raise ValueError("loop state chain contains a numbering gap")
        previous_path = existing[-1]
        previous = _read_strict_json(previous_path)
        if previous.get("schema") != LOOP_STATE_SCHEMA:
            raise ValueError("existing loop state has an unsupported schema")
        if previous.get("generation") != generation - 1:
            target = states_dir / f"state-{generation:06d}.json"
            if target.exists():
                return _write_json_create_only(target, state)
            raise ValueError("new loop state generation is not contiguous")
        if state.get("previous_state_sha256") != _stable_sha256(previous):
            raise ValueError("new loop state does not extend the current state head")
    elif generation != 1 or state.get("previous_state_sha256") is not None:
        raise ValueError("the first loop state must be generation one without a predecessor")
    return _write_json_create_only(states_dir / f"state-{generation:06d}.json", state)


def load_latest_loop_state(workdir: Path) -> tuple[dict[str, Any], str] | None:
    """Read and hash the latest fully validated append-only loop state."""

    states_dir = workdir / "states"
    paths = sorted(states_dir.glob("state-*.json")) if states_dir.exists() else []
    if not paths:
        return None
    previous_sha256: str | None = None
    latest: dict[str, Any] | None = None
    expected_names = [f"state-{index:06d}.json" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("loop state chain contains a numbering gap")
    for generation, path in enumerate(paths, start=1):
        payload = _read_strict_json(path)
        if payload.get("schema") != LOOP_STATE_SCHEMA or payload.get("generation") != generation:
            raise ValueError(f"invalid loop state identity: {path}")
        if payload.get("previous_state_sha256") != previous_sha256:
            raise ValueError(f"loop state predecessor digest mismatch: {path}")
        previous_sha256 = _stable_sha256(payload)
        latest = payload
    if latest is None or previous_sha256 is None:  # pragma: no cover - paths is non-empty
        raise AssertionError("state chain validation lost its latest entry")
    return latest, previous_sha256


def trajectory_sha256(initial_sfen: str, moves: Sequence[str]) -> str:
    """Return the game identity used to reject deterministic duplicate trajectories."""

    canonical = Board(initial_sfen).to_sfen()
    return _stable_sha256({"initial_sfen": canonical, "moves": list(moves)})


def deterministic_retry_seed(job: PlannedGame, *, retry: int) -> int:
    """Derive a new seed for a duplicate/failed training job without changing its plan."""

    if retry < 1:
        raise ValueError("retry number must be positive")
    digest = hashlib.sha256(
        _json_bytes({"job_id": job.job_id, "original_seed": job.seed, "retry": retry})
    ).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)
