"""Typed, replay-bound targets for canonical multi-teacher distillation.

The ordinary :class:`~simajilord_shogi.domain.PositionSample` deliberately
keeps actor search and terminal game facts.  Canonical distillation targets
live in an adjacent sidecar so those facts cannot silently become target mass.
This module validates the complete sidecar against the exact replay and returns
targets in the same order as ``replay.position_samples``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from rsshogi.core import Board, Move

from .artifact_provenance import sha256_file
from .domain import GameRecord, PositionSample, Termination
from .ensemble import normalized_sfen
from .external_usi import UsiPositionHistory
from .replay import load_games

CANONICAL_TARGET_SIDECAR_SCHEMA = "meteo-canonical-distillation-targets-v1"
CANONICAL_TARGET_MODE = "canonical_multi_teacher"
CANONICAL_POSITION_IDENTITY = (
    "game_index + sample_index + validated exact history + normalized SFEN(board/turn/hands)"
)
CANONICAL_TARGET_SUFFIX = ".distillation-targets.json"
NAGISA_SCORER_ID = "nagisa-v3.1"
SUISHO11PLUS_SCORER_ID = "suisho11plus-wcsc36-20260525-local"
SOUJOU_TSEC7_SCORER_ID = "soujou-tsec7-paid"
CANONICAL_SCORER_IDS = (
    NAGISA_SCORER_ID,
    SUISHO11PLUS_SCORER_ID,
    SOUJOU_TSEC7_SCORER_ID,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ProvenMateStatus(StrEnum):
    NONE = "none"
    PROVEN = "proven"


@dataclass(frozen=True, slots=True)
class CanonicalReplayIdentity:
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class CanonicalScorerDescriptor:
    scorer_id: str
    family: str
    role: str


@dataclass(frozen=True, slots=True)
class CanonicalScorerTarget:
    scorer_id: str
    best_move: str
    policy: dict[str, float]
    value: float


@dataclass(frozen=True, slots=True)
class PolicyEquivalenceGroup:
    group_id: int
    kind: str
    moves: tuple[str, ...]
    target_mass: float
    worst_teacher_regret: float
    teacher_depth_dispersion: float
    reply_dispersion: float


@dataclass(frozen=True, slots=True)
class ProvenWinningMateSet:
    status: ProvenMateStatus
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalDistillationTarget:
    game_index: int
    sample_index: int
    normalized_sfen: str
    history_context_sha256: str
    scorers: tuple[CanonicalScorerTarget, ...]
    canonical_best_union: tuple[str, ...]
    equivalence_groups: tuple[PolicyEquivalenceGroup, ...]
    proven_winning_mate_set: ProvenWinningMateSet
    canonical_values: dict[str, float]
    history_training_weight: float


@dataclass(frozen=True, slots=True)
class CanonicalTargetSidecar:
    schema: str
    replay: CanonicalReplayIdentity
    position_identity: str
    target_mode: str
    canonical_scorers: tuple[CanonicalScorerDescriptor, ...]
    positions: tuple[CanonicalDistillationTarget, ...]


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _exact_fields(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields do not match schema: missing={missing}, extra={extra}")


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"canonical target sidecar is not valid UTF-8 JSON: {path}") from error
    return _mapping(payload, label="canonical target sidecar")


def _validated_regular_file(path: Path, *, label: str) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path.expanduser())))
    if resolved.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def resolve_canonical_target_sidecar(
    replay: Path,
    *,
    explicit: Path | None = None,
) -> Path | None:
    """Resolve at most one explicit-or-adjacent sidecar, rejecting ambiguity."""

    replay_path = replay.expanduser().resolve()
    adjacent = replay_path.with_suffix(replay_path.suffix + CANONICAL_TARGET_SUFFIX)
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        if any(os.path.samefile(candidate, existing) for existing in candidates):
            return
        candidates.append(candidate)

    if adjacent.exists() or adjacent.is_symlink():
        resolved_adjacent = _validated_regular_file(
            adjacent, label="adjacent canonical target sidecar"
        )
        add_candidate(resolved_adjacent)
    if explicit is not None:
        resolved_explicit = _validated_regular_file(
            explicit, label="explicit canonical target sidecar"
        )
        add_candidate(resolved_explicit)
    if len(candidates) > 1:
        raise ValueError(
            "multiple canonical target sidecars are ambiguous: "
            f"{sorted(str(path) for path in candidates)}"
        )
    return next(iter(candidates), None)


def canonical_history_context_sha256(game: GameRecord, sample: PositionSample) -> str:
    """Hash the exact validated USI history prefix used to reach one sample."""

    history = UsiPositionHistory.from_game(game, sample)
    encoded = (
        history.initial_sfen.encode("utf-8")
        + b"\0"
        + " ".join(history.moves).encode("ascii")
    )
    return hashlib.sha256(encoded).hexdigest()


def _legal_move(move_usi: object, *, board: Board, label: str) -> str:
    if not isinstance(move_usi, str) or not move_usi:
        raise ValueError(f"{label} must be a non-empty USI move")
    try:
        move = Move.from_usi(move_usi)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not valid USI: {move_usi!r}") from error
    if not board.is_legal_move(move):
        raise ValueError(f"illegal {label} {move_usi!r} at {board.to_sfen()}")
    return move_usi


def _scorer_descriptor(value: object, *, index: int) -> CanonicalScorerDescriptor:
    row = _mapping(value, label=f"canonical scorer descriptor {index}")
    _exact_fields(row, {"scorer_id", "family", "role"}, label=f"scorer descriptor {index}")
    scorer_id = row["scorer_id"]
    family = row["family"]
    role = row["role"]
    if not all(isinstance(item, str) and item for item in (scorer_id, family, role)):
        raise ValueError("canonical scorer descriptor strings must be non-empty")
    if role != "canonical_scorer":
        raise ValueError("canonical scorer role must be canonical_scorer")
    expected_family = {
        NAGISA_SCORER_ID: "nagisa",
        SUISHO11PLUS_SCORER_ID: "suisho",
        SOUJOU_TSEC7_SCORER_ID: "soujou",
    }.get(cast(str, scorer_id))
    if expected_family is None or family != expected_family:
        raise ValueError(f"unexpected canonical scorer descriptor: {scorer_id!r}/{family!r}")
    return CanonicalScorerDescriptor(
        scorer_id=cast(str, scorer_id),
        family=cast(str, family),
        role=cast(str, role),
    )


def _normalized_policy(value: object, *, board: Board, label: str) -> dict[str, float]:
    mapping = _mapping(value, label=label)
    if not mapping:
        raise ValueError(f"{label} must not be empty")
    policy: dict[str, float] = {}
    for raw_move in sorted(mapping):
        move = _legal_move(raw_move, board=board, label=f"{label} move")
        probability = _finite_float(mapping[raw_move], label=f"{label}[{move}]")
        if probability <= 0.0:
            raise ValueError(f"{label} probabilities must be strictly positive")
        policy[move] = probability
    total = math.fsum(policy.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{label} probability mass must equal one, got {total}")
    return policy


def _scorer_target(value: object, *, board: Board, index: int) -> CanonicalScorerTarget:
    row = _mapping(value, label=f"canonical scorer target {index}")
    _exact_fields(
        row,
        {"scorer_id", "best_move", "policy", "value"},
        label=f"canonical scorer target {index}",
    )
    scorer_id = row["scorer_id"]
    if not isinstance(scorer_id, str):
        raise ValueError("canonical scorer target scorer_id must be a string")
    best_move = _legal_move(
        row["best_move"], board=board, label=f"canonical scorer {scorer_id} best_move"
    )
    policy = _normalized_policy(
        row["policy"], board=board, label=f"canonical scorer {scorer_id} policy"
    )
    if best_move not in policy:
        raise ValueError(f"canonical scorer {scorer_id} best_move is absent from its policy")
    maximum_probability = max(policy.values())
    if not math.isclose(
        policy[best_move],
        maximum_probability,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"canonical scorer {scorer_id} best_move is not a maximum-probability move"
        )
    target_value = _finite_float(row["value"], label=f"canonical scorer {scorer_id} value")
    if not -1.0 <= target_value <= 1.0:
        raise ValueError(f"canonical scorer {scorer_id} value must be in [-1, 1]")
    return CanonicalScorerTarget(
        scorer_id=scorer_id,
        best_move=best_move,
        policy=policy,
        value=target_value,
    )


def validate_canonical_scorer_target(
    value: object,
    *,
    board: Board,
    index: int,
) -> CanonicalScorerTarget:
    """Public strict parser shared by canonical target contract versions."""

    return _scorer_target(value, board=board, index=index)


def _equivalence_group(
    value: object,
    *,
    board: Board,
    index: int,
) -> PolicyEquivalenceGroup:
    row = _mapping(value, label=f"policy equivalence group {index}")
    _exact_fields(
        row,
        {
            "group_id",
            "kind",
            "moves",
            "target_mass",
            "worst_teacher_regret",
            "teacher_depth_dispersion",
            "reply_dispersion",
        },
        label=f"policy equivalence group {index}",
    )
    group_id = _integer(row["group_id"], label=f"equivalence group {index} group_id")
    if group_id != index:
        raise ValueError("policy equivalence group IDs must be contiguous and ordered")
    kind = row["kind"]
    if not isinstance(kind, str) or not kind.strip() or kind != kind.strip():
        raise ValueError("policy equivalence group kind must be a trimmed non-empty string")
    moves = tuple(
        _legal_move(move, board=board, label=f"equivalence group {index} move")
        for move in _sequence(row["moves"], label=f"equivalence group {index} moves")
    )
    if not moves or moves != tuple(sorted(set(moves))):
        raise ValueError("policy equivalence group moves must be non-empty, unique, and sorted")
    target_mass = _finite_float(row["target_mass"], label=f"equivalence group {index} target_mass")
    if target_mass < 0.0:
        raise ValueError("policy equivalence group target_mass must be non-negative")
    evidence = tuple(
        _finite_float(row[field], label=f"equivalence group {index} {field}")
        for field in (
            "worst_teacher_regret",
            "teacher_depth_dispersion",
            "reply_dispersion",
        )
    )
    if any(item < 0.0 for item in evidence):
        raise ValueError("policy equivalence evidence values must be non-negative")
    return PolicyEquivalenceGroup(
        group_id=group_id,
        kind=kind,
        moves=moves,
        target_mass=target_mass,
        worst_teacher_regret=evidence[0],
        teacher_depth_dispersion=evidence[1],
        reply_dispersion=evidence[2],
    )


def _proven_mate_set(value: object, *, board: Board) -> ProvenWinningMateSet:
    row = _mapping(value, label="proven winning mate set")
    _exact_fields(row, {"status", "moves"}, label="proven winning mate set")
    try:
        status = ProvenMateStatus(row["status"])
    except (TypeError, ValueError) as error:
        raise ValueError("proven mate status must be none or proven") from error
    moves = tuple(
        _legal_move(move, board=board, label="proven winning mate move")
        for move in _sequence(row["moves"], label="proven winning mate moves")
    )
    if moves != tuple(sorted(set(moves))):
        raise ValueError("proven winning mate moves must be unique and sorted")
    if (status is ProvenMateStatus.PROVEN) != bool(moves):
        raise ValueError("only a proven mate set may contain moves, and it must contain moves")
    if status is ProvenMateStatus.PROVEN:
        raise ValueError(
            "canonical sidecar v1 rejects self-declared proven mates until an internal "
            "proof artifact is replayed and verified"
        )
    return ProvenWinningMateSet(status=status, moves=moves)


def _position_target(
    value: object,
    *,
    game: GameRecord,
    sample: PositionSample,
    game_index: int,
    sample_index: int,
) -> CanonicalDistillationTarget:
    row = _mapping(value, label=f"canonical target {game_index}:{sample_index}")
    _exact_fields(
        row,
        {
            "game_index",
            "sample_index",
            "normalized_sfen",
            "history_context_sha256",
            "scorers",
            "canonical_best_union",
            "equivalence_groups",
            "proven_winning_mate_set",
            "canonical_values",
            "history_training_weight",
        },
        label=f"canonical target {game_index}:{sample_index}",
    )
    if _integer(row["game_index"], label="canonical target game_index") != game_index:
        raise ValueError("canonical target game_index does not match replay order")
    if _integer(row["sample_index"], label="canonical target sample_index") != sample_index:
        raise ValueError("canonical target sample_index does not match replay order")
    board = Board(sample.sfen)
    expected_normalized = normalized_sfen(sample.sfen)
    if row["normalized_sfen"] != expected_normalized:
        raise ValueError("canonical target normalized_sfen does not match replay sample")
    history_sha256 = _sha256(
        row["history_context_sha256"], label="canonical target history_context_sha256"
    )
    if history_sha256 != canonical_history_context_sha256(game, sample):
        raise ValueError("canonical target history identity does not match replay sample")

    scorers = tuple(
        _scorer_target(scorer, board=board, index=index)
        for index, scorer in enumerate(_sequence(row["scorers"], label="canonical target scorers"))
    )
    if tuple(scorer.scorer_id for scorer in scorers) != CANONICAL_SCORER_IDS:
        raise ValueError(
            "every canonical target requires exactly NAGISA, Suisho11Plus, and Soujou TSEC7"
        )
    expected_best_union = tuple(sorted({scorer.best_move for scorer in scorers}))
    best_union = tuple(
        _legal_move(move, board=board, label="canonical best-union move")
        for move in _sequence(row["canonical_best_union"], label="canonical best union")
    )
    if best_union != expected_best_union:
        raise ValueError("canonical_best_union must equal the sorted scorer best-move union")
    if len(best_union) > 5:
        raise ValueError("canonical best union cannot exceed five moves")

    groups = tuple(
        _equivalence_group(group, board=board, index=index)
        for index, group in enumerate(
            _sequence(row["equivalence_groups"], label="policy equivalence groups")
        )
    )
    if not groups:
        raise ValueError("canonical target requires at least one policy equivalence group")
    grouped_moves: set[str] = set()
    for group in groups:
        overlap = grouped_moves.intersection(group.moves)
        if overlap:
            raise ValueError(f"policy equivalence groups overlap: {sorted(overlap)}")
        grouped_moves.update(group.moves)
    moves_scored_by_every_canonical_teacher = set.intersection(
        *(set(scorer.policy) for scorer in scorers)
    )
    unscored_group_moves = grouped_moves - moves_scored_by_every_canonical_teacher
    if unscored_group_moves:
        raise ValueError(
            "equivalence-group moves must be scored by every canonical teacher: "
            f"{sorted(unscored_group_moves)}"
        )
    if not set(best_union).issubset(grouped_moves):
        raise ValueError("policy equivalence groups must cover every canonical best-union move")
    group_mass = math.fsum(group.target_mass for group in groups)
    if not math.isclose(group_mass, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"policy equivalence group target mass must total one, got {group_mass}")

    mate_set = _proven_mate_set(row["proven_winning_mate_set"], board=board)
    if not set(mate_set.moves).issubset(grouped_moves):
        raise ValueError(
            "every proven mate move must belong to the accepted equivalence-group candidates"
        )
    raw_values = _mapping(row["canonical_values"], label="canonical values")
    if set(raw_values) != set(CANONICAL_SCORER_IDS):
        raise ValueError(
            "canonical_values must contain exactly NAGISA, Suisho11Plus, and Soujou TSEC7"
        )
    canonical_values = {
        scorer_id: _finite_float(raw_values[scorer_id], label=f"canonical value {scorer_id}")
        for scorer_id in CANONICAL_SCORER_IDS
    }
    for scorer in scorers:
        if canonical_values[scorer.scorer_id] != scorer.value:
            raise ValueError("canonical_values must exactly repeat the scorer values")
    history_weight = _finite_float(row["history_training_weight"], label="history_training_weight")
    if history_weight != 1.0:
        raise ValueError(
            "canonical sidecar v1 does not yet permit trajectory credit; "
            "history_training_weight must be 1"
        )
    return CanonicalDistillationTarget(
        game_index=game_index,
        sample_index=sample_index,
        normalized_sfen=expected_normalized,
        history_context_sha256=history_sha256,
        scorers=scorers,
        canonical_best_union=best_union,
        equivalence_groups=groups,
        proven_winning_mate_set=mate_set,
        canonical_values=canonical_values,
        history_training_weight=history_weight,
    )


def load_canonical_target_sidecar(
    replay: Path,
    sidecar: Path,
    *,
    games: Sequence[GameRecord] | None = None,
) -> CanonicalTargetSidecar:
    """Load and validate a canonical target sidecar against every replay sample."""

    replay_path = _validated_regular_file(replay, label="canonical target replay")
    sidecar_path = _validated_regular_file(sidecar, label="canonical target sidecar")
    source_games = tuple(load_games(replay_path) if games is None else games)
    payload = _read_strict_json(sidecar_path)
    _exact_fields(
        payload,
        {
            "schema",
            "replay",
            "position_identity",
            "target_mode",
            "canonical_scorers",
            "positions",
        },
        label="canonical target sidecar",
    )
    if payload["schema"] != CANONICAL_TARGET_SIDECAR_SCHEMA:
        raise ValueError("unsupported canonical target sidecar schema")
    if payload["position_identity"] != CANONICAL_POSITION_IDENTITY:
        raise ValueError("canonical target position_identity contract does not match")
    if payload["target_mode"] != CANONICAL_TARGET_MODE:
        raise ValueError("canonical target mode must be canonical_multi_teacher")

    replay_row = _mapping(payload["replay"], label="canonical target replay identity")
    _exact_fields(replay_row, {"sha256", "bytes"}, label="canonical target replay identity")
    replay_identity = CanonicalReplayIdentity(
        sha256=_sha256(replay_row["sha256"], label="canonical replay SHA-256"),
        bytes=_integer(replay_row["bytes"], label="canonical replay bytes"),
    )
    if replay_identity.bytes != replay_path.stat().st_size:
        raise ValueError("canonical target replay byte count does not match")
    if replay_identity.sha256 != sha256_file(replay_path):
        raise ValueError("canonical target replay SHA-256 does not match")

    descriptors = tuple(
        _scorer_descriptor(descriptor, index=index)
        for index, descriptor in enumerate(
            _sequence(payload["canonical_scorers"], label="canonical scorers")
        )
    )
    if tuple(descriptor.scorer_id for descriptor in descriptors) != CANONICAL_SCORER_IDS:
        raise ValueError(
            "canonical sidecar requires NAGISA, Suisho11Plus, then Soujou TSEC7 descriptors"
        )

    for game_index, game in enumerate(source_games):
        if game.termination is Termination.MAX_PLIES:
            raise ValueError(f"canonical training rejects incomplete max_plies game {game_index}")
    expected_samples = [
        (game_index, sample_index, game, sample)
        for game_index, game in enumerate(source_games)
        for sample_index, sample in enumerate(game.samples)
    ]
    raw_positions = _sequence(payload["positions"], label="canonical target positions")
    if len(raw_positions) != len(expected_samples):
        raise ValueError(
            "canonical target count does not cover the replay exactly: "
            f"{len(raw_positions)} != {len(expected_samples)}"
        )
    positions = tuple(
        _position_target(
            raw,
            game=game,
            sample=sample,
            game_index=game_index,
            sample_index=sample_index,
        )
        for raw, (game_index, sample_index, game, sample) in zip(
            raw_positions, expected_samples, strict=True
        )
    )
    return CanonicalTargetSidecar(
        schema=CANONICAL_TARGET_SIDECAR_SCHEMA,
        replay=replay_identity,
        position_identity=CANONICAL_POSITION_IDENTITY,
        target_mode=CANONICAL_TARGET_MODE,
        canonical_scorers=descriptors,
        positions=positions,
    )
