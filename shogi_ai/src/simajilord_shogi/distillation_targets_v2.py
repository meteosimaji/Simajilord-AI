"""Fail-closed canonical multi-teacher targets with guarded play supervision.

Contract v1 preserved NAGISA and Suisho11Plus as separate records, but its
training path still compared one scalar value head with both values.  The
mathematical optimum of that broadcast MSE is their arithmetic midpoint.
Contract v2 therefore has two deliberately different responsibilities:

* independent teacher policy/value targets train independent auxiliary heads;
* the actual play heads train only where a complete cross-teacher score matrix,
  depth/reply stability, exact history, and (when applicable) an internal mate
  proof establish a robust target.

Unresolved positions remain in the sidecar as an additional-search queue.  In
mixed batches they update only detached teacher heads and uncertainty: play
policy/value/WDL losses are masked and canonical training freezes BatchNorm
statistics.  A corpus containing no resolved position cannot start optimizer
steps, so unresolved data alone cannot decay or move the play model.
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
from .distillation_targets import (
    CANONICAL_POSITION_IDENTITY,
    CANONICAL_SCORER_IDS,
    NAGISA_SCORER_ID,
    SUISHO11PLUS_SCORER_ID,
    CanonicalReplayIdentity,
    CanonicalScorerTarget,
    canonical_history_context_sha256,
    validate_canonical_scorer_target,
)
from .domain import GameRecord, PositionSample, TeacherScoreBound, TeacherScoreKind, Termination
from .encoding import HistoryInput
from .ensemble import normalized_sfen
from .external_usi import UsiPositionHistory
from .replay import load_games
from .research_context import game_phase
from .rights_lineage import validate_rights_restriction_summary
from .tsume import TsumeSolver

CANONICAL_TARGET_V2_SCHEMA = "meteo-canonical-distillation-targets-v2"
CANONICAL_TARGET_V2_MODE = "independent_teacher_heads_with_guarded_play"
CANONICAL_TARGET_V2_SUFFIX = ".distillation-targets-v2.json"
CANONICAL_Q_VALUE_PERSPECTIVE = "root_player_signed_value"
CANONICAL_RAW_SCORE_PERSPECTIVE = "search_position_side_to_move"
CANONICAL_CALIBRATION_SCHEMA = "meteo-teacher-value-calibration-v1"
CANONICAL_CALIBRATION_FORMULA = (
    "root_q=root_player_sign*tanh(raw_cp/tanh_denominator);"
    "mate=root_player_sign*raw_mate_sign"
)
CANONICAL_POLICY_CALIBRATION_FORMULA = (
    "winning_mates_uniform_else_softmax_raw_cp_over_temperature"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROBABILITY_TOLERANCE = 1e-8
_DERIVATION_TOLERANCE = 1e-7


class MateProofState(StrEnum):
    """Three-valued bounded proof state plus the non-mate-applicable case."""

    NOT_APPLICABLE = "not_applicable"
    PROVEN_WIN = "proven_win"
    PROVEN_NO_MATE_WITHIN_BOUND = "proven_no_mate_within_bound"
    UNKNOWN_LIMIT = "unknown_limit"


class PlayTargetKind(StrEnum):
    """Why the play head is active, or why it is deliberately masked."""

    UNRESOLVED = "unresolved"
    ROBUST_CONSENSUS = "robust_consensus"
    PROVEN_MATE = "proven_mate"


class CanonicalDisagreementReason(StrEnum):
    """Machine-readable reasons why the play target needed caution or more search."""

    TEACHER_BEST_MOVE = "teacher_best_move"
    TEACHER_POLICY_DISTRIBUTION = "teacher_policy_distribution"
    TEACHER_VALUE_SIGN = "teacher_value_sign"
    CANDIDATE_FAMILY_COVERAGE = "candidate_family_coverage_incomplete"
    REPLY_COVERAGE = "reply_coverage_incomplete"
    CALIBRATION = "calibration_unverified"
    NON_EXACT_BOUND = "non_exact_score_bound"
    DEPTH_INSTABILITY = "depth_instability"
    REPLY_INSTABILITY = "reply_instability"
    EMPTY_CONSENSUS = "empty_cross_teacher_regret_consensus"
    REPORTED_MATE_UNPROVEN = "reported_winning_mate_unproven"
    PROOF_UNKNOWN_LIMIT = "mate_proof_unknown_limit"


@dataclass(frozen=True, slots=True)
class PrincipalReplyScore:
    move: str
    q_value: float
    raw_score_perspective: str
    root_player_sign: int
    score_kind: TeacherScoreKind
    bound: TeacherScoreBound
    score_cp: int | None
    mate_plies: int | None
    mate_unknown_sign: int | None
    reported_nodes: int
    depth: int
    time_ms: int
    pv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateScoreEvidence:
    move: str
    q_value: float
    raw_score_perspective: str
    root_player_sign: int
    score_kind: TeacherScoreKind
    bound: TeacherScoreBound
    score_cp: int | None
    mate_plies: int | None
    mate_unknown_sign: int | None
    reported_nodes: int
    depth: int
    time_ms: int
    pv: tuple[str, ...]
    principal_replies: tuple[PrincipalReplyScore, ...]
    legal_reply_coverage_complete: bool


@dataclass(frozen=True, slots=True)
class CandidateFamilyProposal:
    """One candidate source's lossless, provenance-bound proposal set."""

    family: str
    producer: str
    provenance_sha256: str
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BudgetScoreEvidence:
    requested_nodes: int
    reported_nodes: int
    candidates: tuple[CandidateScoreEvidence, ...]


@dataclass(frozen=True, slots=True)
class ScoreCalibration:
    """Teacher- and phase-specific conversion from raw USI scores to root Q."""

    schema: str
    phase: str
    formula: str
    policy_formula: str
    policy_temperature: float
    ponanza_coefficient: float
    tanh_denominator: float
    artifact_sha256: str
    independent_validation_passed: bool


@dataclass(frozen=True, slots=True)
class CanonicalScorerMatrix:
    scorer_id: str
    q_value_perspective: str
    calibration: ScoreCalibration
    budgets: tuple[BudgetScoreEvidence, ...]


@dataclass(frozen=True, slots=True)
class RobustnessThresholds:
    regret: float
    depth_dispersion: float
    reply_dispersion: float


@dataclass(frozen=True, slots=True)
class CanonicalScorerDescriptorV2:
    """Exact local scorer/runtime identity without copying an external artifact."""

    scorer_id: str
    family: str
    role: str
    engine_sha256: str
    evaluation_artifacts_sha256: str
    startup_provenance_sha256: str
    option_fingerprint_sha256: str
    history_mode: str
    book_enabled: bool
    threads: int
    hash_mb: int
    multipv: int


@dataclass(frozen=True, slots=True)
class VerifiedMateProof:
    state: MateProofState
    max_plies: int | None
    node_limit: int | None
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuardedPlayTarget:
    kind: PlayTargetKind
    train_play: bool
    additional_search_required: bool
    policy: dict[str, float]
    robust_best_moves: tuple[str, ...]
    value_interval: tuple[float, float] | None
    wdl: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class CanonicalDistillationTargetV2:
    game_index: int
    sample_index: int
    normalized_sfen: str
    history_context_sha256: str
    history: HistoryInput
    scorers: tuple[CanonicalScorerTarget, ...]
    candidate_proposals: tuple[CandidateFamilyProposal, ...]
    candidate_families: tuple[str, ...]
    candidate_family_coverage_complete: bool
    reply_coverage_complete: bool
    candidate_moves: tuple[str, ...]
    budgets: tuple[int, ...]
    score_matrix: tuple[CanonicalScorerMatrix, ...]
    thresholds: RobustnessThresholds
    proof: VerifiedMateProof
    play: GuardedPlayTarget
    uncertainty_target: float
    worst_teacher_regret: dict[str, float]
    maximum_depth_dispersion: dict[str, float]
    maximum_reply_dispersion: dict[str, float]
    disagreement_reasons: tuple[CanonicalDisagreementReason, ...]


@dataclass(frozen=True, slots=True)
class CanonicalTargetSidecarV2:
    schema: str
    replay: CanonicalReplayIdentity
    position_identity: str
    target_mode: str
    canonical_scorers: tuple[CanonicalScorerDescriptorV2, ...]
    required_candidate_families: tuple[str, ...]
    rights_restriction_summary: dict[str, object]
    positions: tuple[CanonicalDistillationTargetV2, ...]

    @property
    def play_eligible_positions(self) -> int:
        return sum(target.play.train_play for target in self.positions)

    @property
    def unresolved_positions(self) -> int:
        return len(self.positions) - self.play_eligible_positions

    @property
    def production_training_eligible(self) -> bool:
        """Remain false until the real score-matrix builder receipt is implemented.

        The v2 loader independently checks the internal matrix mathematics.  It
        cannot yet prove that the claimed calibration artifact, candidate
        proposers, and held-out split were produced by the unfinished builder.
        Keeping this gate closed prevents a hand-authored sidecar from turning
        those self-assertions into a real optimizer run.
        """

        return False

    @property
    def production_training_blockers(self) -> tuple[str, ...]:
        return (
            "score_matrix_builder_receipt_unimplemented",
            "calibration_artifact_replay_verification_unimplemented",
            "independent_heldout_split_receipt_unimplemented",
        )


def _v2_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def _v2_sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return cast(list[object], value)


def _v2_exact_fields(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields do not match schema: missing={missing}, extra={extra}")


def _v2_integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _v2_optional_integer(value: object, *, label: str) -> int | None:
    return None if value is None else _v2_integer(value, label=label)


def _v2_optional_signed_integer(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer or null")
    return value


def _v2_finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _v2_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _v2_strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _v2_reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _v2_read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_v2_strict_object,
            parse_constant=_v2_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"canonical v2 sidecar is not valid UTF-8 JSON: {path}") from error
    return _v2_mapping(payload, label="canonical v2 sidecar")


def _v2_regular_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    if absolute.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {absolute}")
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return absolute


def resolve_canonical_target_sidecar_v2(
    replay: Path,
    *,
    explicit: Path | None = None,
) -> Path | None:
    """Resolve one v2 sidecar and reject explicit/adjacent ambiguity."""

    replay_path = replay.expanduser().resolve()
    adjacent = replay_path.with_suffix(replay_path.suffix + CANONICAL_TARGET_V2_SUFFIX)
    candidates: list[Path] = []
    for candidate, label in (
        (adjacent, "adjacent canonical v2 sidecar"),
        (explicit, "explicit canonical v2 sidecar"),
    ):
        if candidate is None or (
            candidate == adjacent and not (candidate.exists() or candidate.is_symlink())
        ):
            continue
        resolved = _v2_regular_file(candidate, label=label)
        if not any(os.path.samefile(resolved, existing) for existing in candidates):
            candidates.append(resolved)
    if len(candidates) > 1:
        raise ValueError(
            "multiple canonical v2 sidecars are ambiguous: "
            f"{sorted(str(path) for path in candidates)}"
        )
    return next(iter(candidates), None)


def _v2_legal_move(value: object, *, board: Board, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty USI move")
    try:
        move = Move.from_usi(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{label} is not valid USI: {value!r}") from error
    if not board.is_legal_move(move):
        raise ValueError(f"illegal {label} {value!r} at {board.to_sfen()}")
    return value


def _v2_descriptor(value: object, *, index: int) -> CanonicalScorerDescriptorV2:
    row = _v2_mapping(value, label=f"canonical v2 scorer descriptor {index}")
    _v2_exact_fields(
        row,
        {
            "scorer_id",
            "family",
            "role",
            "engine_sha256",
            "evaluation_artifacts_sha256",
            "startup_provenance_sha256",
            "option_fingerprint_sha256",
            "history_mode",
            "book_enabled",
            "threads",
            "hash_mb",
            "multipv",
        },
        label="scorer descriptor",
    )
    scorer_id = row["scorer_id"]
    family = row["family"]
    role = row["role"]
    expected_family = {
        NAGISA_SCORER_ID: "nagisa",
        SUISHO11PLUS_SCORER_ID: "suisho",
    }.get(cast(str, scorer_id))
    if (
        not isinstance(scorer_id, str)
        or not isinstance(family, str)
        or not isinstance(role, str)
        or role != "canonical_scorer"
        or family != expected_family
    ):
        raise ValueError("canonical v2 scorer descriptor is not the exact reviewed pair")
    if row["history_mode"] != "game_prefix":
        raise ValueError("canonical v2 scorer must receive the exact game-prefix history")
    if row["book_enabled"] is not False:
        raise ValueError("canonical v2 scorer must disable its opening book")
    return CanonicalScorerDescriptorV2(
        scorer_id=scorer_id,
        family=family,
        role=role,
        engine_sha256=_v2_sha256(row["engine_sha256"], label="scorer engine SHA-256"),
        evaluation_artifacts_sha256=_v2_sha256(
            row["evaluation_artifacts_sha256"],
            label="scorer evaluation-artifacts SHA-256",
        ),
        startup_provenance_sha256=_v2_sha256(
            row["startup_provenance_sha256"],
            label="scorer startup-provenance SHA-256",
        ),
        option_fingerprint_sha256=_v2_sha256(
            row["option_fingerprint_sha256"],
            label="scorer option-fingerprint SHA-256",
        ),
        history_mode="game_prefix",
        book_enabled=False,
        threads=_v2_integer(row["threads"], label="scorer threads", minimum=1),
        hash_mb=_v2_integer(row["hash_mb"], label="scorer hash_mb", minimum=1),
        multipv=_v2_integer(row["multipv"], label="scorer MultiPV", minimum=1),
    )


def _v2_score_kind_fields(
    row: Mapping[str, object], *, label: str
) -> tuple[TeacherScoreKind, TeacherScoreBound, int | None, int | None, int | None]:
    try:
        score_kind = TeacherScoreKind(cast(str, row["score_kind"]))
        bound = TeacherScoreBound(cast(str, row["bound"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} has an invalid score kind or bound") from error
    score_cp = _v2_optional_signed_integer(row["score_cp"], label=f"{label} score_cp")
    mate_plies = _v2_optional_signed_integer(
        row["mate_plies"], label=f"{label} mate_plies"
    )
    mate_unknown_sign = _v2_optional_signed_integer(
        row["mate_unknown_sign"], label=f"{label} mate_unknown_sign"
    )
    if score_kind is TeacherScoreKind.CENTIPAWN:
        if score_cp is None or mate_plies is not None or mate_unknown_sign is not None:
            raise ValueError(f"{label} centipawn score fields are inconsistent")
    elif score_cp is not None or (mate_plies is None) == (mate_unknown_sign is None):
        raise ValueError(f"{label} mate score fields are inconsistent")
    if mate_unknown_sign not in {None, -1, 1}:
        raise ValueError(f"{label} mate_unknown_sign must be -1 or 1")
    return score_kind, bound, score_cp, mate_plies, mate_unknown_sign


def _v2_calibration(
    value: object,
    *,
    scorer_id: str,
    expected_phase: str,
) -> ScoreCalibration:
    row = _v2_mapping(value, label=f"{scorer_id} value calibration")
    _v2_exact_fields(
        row,
        {
            "schema",
            "phase",
            "formula",
            "policy_formula",
            "policy_temperature",
            "ponanza_coefficient",
            "tanh_denominator",
            "artifact_sha256",
            "independent_validation_passed",
        },
        label=f"{scorer_id} value calibration",
    )
    if row["schema"] != CANONICAL_CALIBRATION_SCHEMA:
        raise ValueError(f"{scorer_id} value calibration schema does not match")
    phase = row["phase"]
    if not isinstance(phase, str) or not phase or phase.strip() != phase:
        raise ValueError(f"{scorer_id} calibration phase must be a non-empty identifier")
    if phase != expected_phase:
        raise ValueError(
            f"{scorer_id} calibration phase does not match the replay position: "
            f"{phase!r} != {expected_phase!r}"
        )
    if row["formula"] != CANONICAL_CALIBRATION_FORMULA:
        raise ValueError(f"{scorer_id} value calibration formula does not match")
    if row["policy_formula"] != CANONICAL_POLICY_CALIBRATION_FORMULA:
        raise ValueError(f"{scorer_id} policy calibration formula does not match")
    policy_temperature = _v2_finite(
        row["policy_temperature"], label=f"{scorer_id} policy temperature"
    )
    if policy_temperature <= 0.0:
        raise ValueError(f"{scorer_id} policy temperature must be positive")
    coefficient = _v2_finite(
        row["ponanza_coefficient"], label=f"{scorer_id} Ponanza coefficient"
    )
    denominator = _v2_finite(
        row["tanh_denominator"], label=f"{scorer_id} tanh denominator"
    )
    if coefficient <= 0.0 or denominator <= 0.0:
        raise ValueError(f"{scorer_id} value calibration scales must be positive")
    if not math.isclose(denominator, 2.0 * coefficient, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{scorer_id} tanh denominator must equal 2C")
    if not isinstance(row["independent_validation_passed"], bool):
        raise ValueError("independent_validation_passed must be boolean")
    return ScoreCalibration(
        schema=CANONICAL_CALIBRATION_SCHEMA,
        phase=phase,
        formula=CANONICAL_CALIBRATION_FORMULA,
        policy_formula=CANONICAL_POLICY_CALIBRATION_FORMULA,
        policy_temperature=policy_temperature,
        ponanza_coefficient=coefficient,
        tanh_denominator=denominator,
        artifact_sha256=_v2_sha256(
            row["artifact_sha256"], label=f"{scorer_id} calibration artifact SHA-256"
        ),
        independent_validation_passed=row["independent_validation_passed"],
    )


def _v2_root_q(
    *,
    score_kind: TeacherScoreKind,
    score_cp: int | None,
    mate_plies: int | None,
    mate_unknown_sign: int | None,
    root_player_sign: int,
    calibration: ScoreCalibration,
) -> float:
    if root_player_sign not in {-1, 1}:
        raise ValueError("root_player_sign must be -1 or 1")
    if score_kind is TeacherScoreKind.CENTIPAWN:
        assert score_cp is not None
        return root_player_sign * math.tanh(score_cp / calibration.tanh_denominator)
    if mate_unknown_sign is not None:
        raw_mate_sign = mate_unknown_sign
    else:
        assert mate_plies is not None
        raw_mate_sign = 1 if mate_plies >= 0 else -1
    return float(root_player_sign * raw_mate_sign)


def _v2_score_and_q(
    row: Mapping[str, object],
    *,
    label: str,
    expected_root_player_sign: int,
    calibration: ScoreCalibration,
) -> tuple[
    float,
    int,
    TeacherScoreKind,
    TeacherScoreBound,
    int | None,
    int | None,
    int | None,
]:
    if row["raw_score_perspective"] != CANONICAL_RAW_SCORE_PERSPECTIVE:
        raise ValueError(f"{label} raw score perspective does not match")
    root_player_sign = _v2_optional_signed_integer(
        row["root_player_sign"], label=f"{label} root_player_sign"
    )
    if root_player_sign != expected_root_player_sign:
        raise ValueError(
            f"{label} root_player_sign must be {expected_root_player_sign} for this search root"
        )
    score_kind, bound, score_cp, mate_plies, mate_unknown_sign = _v2_score_kind_fields(
        row, label=label
    )
    q_value = _v2_finite(row["q_value"], label=f"{label} q_value")
    expected_q = _v2_root_q(
        score_kind=score_kind,
        score_cp=score_cp,
        mate_plies=mate_plies,
        mate_unknown_sign=mate_unknown_sign,
        root_player_sign=root_player_sign,
        calibration=calibration,
    )
    if not math.isclose(q_value, expected_q, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} q_value does not match its raw score and calibration")
    return (
        q_value,
        root_player_sign,
        score_kind,
        bound,
        score_cp,
        mate_plies,
        mate_unknown_sign,
    )


def _v2_reply(
    value: object,
    *,
    child: Board,
    requested_nodes: int,
    calibration: ScoreCalibration,
    label: str,
) -> PrincipalReplyScore:
    row = _v2_mapping(value, label=label)
    _v2_exact_fields(
        row,
        {
            "move",
            "q_value",
            "raw_score_perspective",
            "root_player_sign",
            "score_kind",
            "bound",
            "score_cp",
            "mate_plies",
            "mate_unknown_sign",
            "reported_nodes",
            "depth",
            "time_ms",
            "pv",
        },
        label=label,
    )
    move = _v2_legal_move(row["move"], board=child, label=f"{label} move")
    (
        q_value,
        root_player_sign,
        score_kind,
        bound,
        score_cp,
        mate_plies,
        mate_unknown_sign,
    ) = _v2_score_and_q(
        row,
        label=label,
        expected_root_player_sign=-1,
        calibration=calibration,
    )
    pv_values = _v2_sequence(row["pv"], label=f"{label} pv")
    if not pv_values or pv_values[0] != move:
        raise ValueError(f"{label} PV must start with its constrained reply")
    pv_board = child.copy()
    pv: list[str] = []
    for ply, raw_move in enumerate(pv_values):
        pv_move = _v2_legal_move(raw_move, board=pv_board, label=f"{label} PV[{ply}]")
        pv.append(pv_move)
        pv_board.apply_move(Move.from_usi(pv_move))
    return PrincipalReplyScore(
        move=move,
        q_value=q_value,
        raw_score_perspective=CANONICAL_RAW_SCORE_PERSPECTIVE,
        root_player_sign=root_player_sign,
        score_kind=score_kind,
        bound=bound,
        score_cp=score_cp,
        mate_plies=mate_plies,
        mate_unknown_sign=mate_unknown_sign,
        reported_nodes=_v2_integer(
            row["reported_nodes"], label=f"{label} reported_nodes", minimum=requested_nodes
        ),
        depth=_v2_integer(row["depth"], label=f"{label} depth", minimum=1),
        time_ms=_v2_integer(row["time_ms"], label=f"{label} time_ms"),
        pv=tuple(pv),
    )


def _v2_candidate_score(
    value: object,
    *,
    root: Board,
    expected_move: str,
    requested_nodes: int,
    calibration: ScoreCalibration,
    label: str,
) -> CandidateScoreEvidence:
    row = _v2_mapping(value, label=label)
    _v2_exact_fields(
        row,
        {
            "move",
            "q_value",
            "raw_score_perspective",
            "root_player_sign",
            "score_kind",
            "bound",
            "score_cp",
            "mate_plies",
            "mate_unknown_sign",
            "reported_nodes",
            "depth",
            "time_ms",
            "pv",
            "principal_replies",
        },
        label=label,
    )
    move = _v2_legal_move(row["move"], board=root, label=f"{label} move")
    if move != expected_move:
        raise ValueError(f"{label} candidates must follow the canonical sorted move order")
    (
        q_value,
        root_player_sign,
        score_kind,
        bound,
        score_cp,
        mate_plies,
        mate_unknown_sign,
    ) = _v2_score_and_q(
        row,
        label=label,
        expected_root_player_sign=1,
        calibration=calibration,
    )
    reported_nodes = _v2_integer(
        row["reported_nodes"], label=f"{label} reported_nodes", minimum=requested_nodes
    )
    depth = _v2_integer(row["depth"], label=f"{label} depth", minimum=1)
    time_ms = _v2_integer(row["time_ms"], label=f"{label} time_ms")
    pv_values = _v2_sequence(row["pv"], label=f"{label} pv")
    if not pv_values or pv_values[0] != move:
        raise ValueError(f"{label} PV must start with its constrained root move")
    pv_board = root.copy()
    pv: list[str] = []
    for ply, raw_move in enumerate(pv_values):
        pv_move = _v2_legal_move(raw_move, board=pv_board, label=f"{label} PV[{ply}]")
        pv.append(pv_move)
        pv_board.apply_move(Move.from_usi(pv_move))
    child = root.copy()
    child.apply_move(Move.from_usi(move))
    expected_reply_moves = tuple(sorted(reply.to_usi() for reply in child.legal_moves()))
    replies = tuple(
        _v2_reply(
            reply,
            child=child,
            requested_nodes=requested_nodes,
            calibration=calibration,
            label=f"{label} principal reply {index}",
        )
        for index, reply in enumerate(
            _v2_sequence(row["principal_replies"], label=f"{label} principal_replies")
        )
    )
    observed_reply_moves = tuple(reply.move for reply in replies)
    if observed_reply_moves != tuple(sorted(set(observed_reply_moves))):
        raise ValueError(f"{label} principal replies must be unique and sorted")
    if len(pv) > 1 and pv[1] not in {reply.move for reply in replies}:
        raise ValueError(f"{label} PV reply must be represented in principal_replies")
    if expected_reply_moves and not replies:
        raise ValueError(f"{label} requires at least one scored principal reply")
    return CandidateScoreEvidence(
        move=move,
        q_value=q_value,
        raw_score_perspective=CANONICAL_RAW_SCORE_PERSPECTIVE,
        root_player_sign=root_player_sign,
        score_kind=score_kind,
        bound=bound,
        score_cp=score_cp,
        mate_plies=mate_plies,
        mate_unknown_sign=mate_unknown_sign,
        reported_nodes=reported_nodes,
        depth=depth,
        time_ms=time_ms,
        pv=tuple(pv),
        principal_replies=replies,
        legal_reply_coverage_complete=observed_reply_moves == expected_reply_moves,
    )


def _v2_scorer_matrix(
    value: object,
    *,
    root: Board,
    scorer_id: str,
    budgets: tuple[int, ...],
    candidates: tuple[str, ...],
) -> CanonicalScorerMatrix:
    row = _v2_mapping(value, label=f"score matrix {scorer_id}")
    _v2_exact_fields(
        row,
        {
            "scorer_id",
            "q_value_perspective",
            "calibration",
            "budgets",
        },
        label=f"score matrix {scorer_id}",
    )
    if row["scorer_id"] != scorer_id:
        raise ValueError("score-matrix scorer order does not match canonical scorer order")
    if row["q_value_perspective"] != CANONICAL_Q_VALUE_PERSPECTIVE:
        raise ValueError(
            "canonical score-matrix q_values must use the root player's signed perspective"
        )
    calibration = _v2_calibration(
        row["calibration"],
        scorer_id=scorer_id,
        expected_phase=game_phase(max(root.game_ply - 1, 0)).value,
    )
    raw_budgets = _v2_sequence(row["budgets"], label=f"{scorer_id} budget rows")
    if len(raw_budgets) != len(budgets):
        raise ValueError(f"{scorer_id} must cover every requested node budget")
    parsed: list[BudgetScoreEvidence] = []
    for index, (raw_budget, requested_nodes) in enumerate(
        zip(raw_budgets, budgets, strict=True)
    ):
        budget_row = _v2_mapping(raw_budget, label=f"{scorer_id} budget {index}")
        _v2_exact_fields(
            budget_row,
            {"requested_nodes", "reported_nodes", "candidates"},
            label=f"{scorer_id} budget {index}",
        )
        if _v2_integer(
            budget_row["requested_nodes"],
            label=f"{scorer_id} requested_nodes",
            minimum=1,
        ) != requested_nodes:
            raise ValueError(f"{scorer_id} requested node budgets do not match")
        reported_nodes = _v2_integer(
            budget_row["reported_nodes"],
            label=f"{scorer_id} reported_nodes",
            minimum=requested_nodes * len(candidates),
        )
        raw_candidates = _v2_sequence(
            budget_row["candidates"], label=f"{scorer_id} budget candidates"
        )
        if len(raw_candidates) != len(candidates):
            raise ValueError(f"{scorer_id} budget does not score every candidate")
        parsed_candidates = tuple(
            _v2_candidate_score(
                raw_candidate,
                root=root,
                expected_move=candidate,
                requested_nodes=requested_nodes,
                calibration=calibration,
                label=f"{scorer_id} budget {requested_nodes} candidate {candidate}",
            )
            for raw_candidate, candidate in zip(raw_candidates, candidates, strict=True)
        )
        measured_nodes = sum(
            candidate.reported_nodes
            + sum(reply.reported_nodes for reply in candidate.principal_replies)
            for candidate in parsed_candidates
        )
        if reported_nodes != measured_nodes:
            raise ValueError(
                f"{scorer_id} aggregate reported_nodes must equal all candidate/reply searches"
            )
        parsed.append(
            BudgetScoreEvidence(
                requested_nodes=requested_nodes,
                reported_nodes=reported_nodes,
                candidates=parsed_candidates,
            )
        )
    return CanonicalScorerMatrix(
        scorer_id=scorer_id,
        q_value_perspective=CANONICAL_Q_VALUE_PERSPECTIVE,
        calibration=calibration,
        budgets=tuple(parsed),
    )


def _v2_thresholds(value: object) -> RobustnessThresholds:
    row = _v2_mapping(value, label="robustness thresholds")
    _v2_exact_fields(
        row,
        {"regret", "depth_dispersion", "reply_dispersion"},
        label="robustness thresholds",
    )
    regret = _v2_finite(row["regret"], label="regret threshold")
    depth = _v2_finite(row["depth_dispersion"], label="depth-dispersion threshold")
    reply = _v2_finite(row["reply_dispersion"], label="reply-dispersion threshold")
    if regret < 0.0 or depth < 0.0 or reply < 0.0:
        raise ValueError("robustness thresholds must be non-negative")
    return RobustnessThresholds(
        regret=regret,
        depth_dispersion=depth,
        reply_dispersion=reply,
    )


def _v2_expected_scorer_target(
    matrix: CanonicalScorerMatrix,
) -> tuple[dict[str, float], tuple[str, ...], float]:
    """Derive one teacher head target from its deepest lossless score row."""

    deepest = matrix.budgets[-1].candidates
    winning_mates = tuple(
        candidate
        for candidate in deepest
        if candidate.score_kind is TeacherScoreKind.MATE and candidate.q_value == 1.0
    )
    if winning_mates:
        mass = 1.0 / len(winning_mates)
        policy = {candidate.move: mass for candidate in winning_mates}
        best_moves = tuple(sorted(policy))
        return policy, best_moves, 1.0

    centipawns = tuple(
        candidate
        for candidate in deepest
        if candidate.score_kind is TeacherScoreKind.CENTIPAWN
    )
    if centipawns:
        maximum_cp = max(cast(int, candidate.score_cp) for candidate in centipawns)
        weights = {
            candidate.move: math.exp(
                (cast(int, candidate.score_cp) - maximum_cp)
                / matrix.calibration.policy_temperature
            )
            for candidate in centipawns
        }
        total = math.fsum(weights.values())
        policy = {move: weight / total for move, weight in weights.items()}
        maximum_probability = max(policy.values())
        best_moves = tuple(
            sorted(
                move
                for move, probability in policy.items()
                if math.isclose(
                    probability,
                    maximum_probability,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        )
        value = max(candidate.q_value for candidate in centipawns)
        return policy, best_moves, value

    mass = 1.0 / len(deepest)
    policy = {candidate.move: mass for candidate in deepest}
    return policy, tuple(sorted(policy)), -1.0


def _v2_validate_scorer_targets_against_matrices(
    scorers: tuple[CanonicalScorerTarget, ...],
    matrices: tuple[CanonicalScorerMatrix, ...],
) -> None:
    for scorer, matrix in zip(scorers, matrices, strict=True):
        expected_policy, expected_best_moves, expected_value = _v2_expected_scorer_target(
            matrix
        )
        if set(scorer.policy) != set(expected_policy) or any(
            not math.isclose(
                scorer.policy[move],
                expected_policy[move],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for move in expected_policy
        ):
            raise ValueError(
                f"{scorer.scorer_id} teacher policy does not match its calibrated deepest scores"
            )
        if scorer.best_move not in expected_best_moves:
            raise ValueError(
                f"{scorer.scorer_id} teacher best move does not match its calibrated deepest scores"
            )
        if not math.isclose(
            scorer.value, expected_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{scorer.scorer_id} teacher value does not match its calibrated deepest score"
            )


def _v2_proof(value: object, *, board: Board) -> VerifiedMateProof:
    row = _v2_mapping(value, label="internal mate proof")
    _v2_exact_fields(
        row, {"state", "max_plies", "node_limit", "moves"}, label="internal mate proof"
    )
    try:
        state = MateProofState(row["state"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid internal mate proof state") from error
    raw_moves = _v2_sequence(row["moves"], label="internal mate proof moves")
    moves = tuple(
        _v2_legal_move(move, board=board, label="internal proven mate move")
        for move in raw_moves
    )
    if moves != tuple(sorted(set(moves))):
        raise ValueError("internal mate proof moves must be unique and sorted")
    if state is MateProofState.NOT_APPLICABLE:
        if row["max_plies"] is not None or row["node_limit"] is not None or moves:
            raise ValueError("not_applicable mate proof must not contain search evidence")
        return VerifiedMateProof(state=state, max_plies=None, node_limit=None, moves=())
    max_plies = _v2_integer(row["max_plies"], label="proof max_plies", minimum=1)
    node_limit = _v2_integer(row["node_limit"], label="proof node_limit", minimum=1)
    if max_plies % 2 == 0:
        raise ValueError("proof max_plies must be odd")
    if state is MateProofState.UNKNOWN_LIMIT:
        if moves:
            raise ValueError("UNKNOWN_LIMIT cannot claim proven moves")
        return VerifiedMateProof(
            state=state, max_plies=max_plies, node_limit=node_limit, moves=()
        )
    try:
        solution = TsumeSolver(node_limit=node_limit, history_exact=True).solve_all(
            board, max_plies
        )
    except RuntimeError as error:
        raise ValueError("declared mate proof exceeded its recorded node limit") from error
    proven = tuple(sorted(solution.first_moves))
    if state is MateProofState.PROVEN_WIN:
        if not moves or moves != proven:
            raise ValueError(
                "PROVEN_WIN must equal every internally re-proved root mating move"
            )
    elif moves or proven:
        raise ValueError(
            "PROVEN_NO_MATE_WITHIN_BOUND requires an exhaustive empty internal result"
        )
    return VerifiedMateProof(
        state=state,
        max_plies=max_plies,
        node_limit=node_limit,
        moves=moves,
    )


def _v2_play_declaration(value: object) -> tuple[PlayTargetKind, bool, bool]:
    row = _v2_mapping(value, label="guarded play declaration")
    _v2_exact_fields(
        row,
        {"kind", "train_play", "additional_search_required"},
        label="guarded play declaration",
    )
    try:
        kind = PlayTargetKind(row["kind"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid guarded play target kind") from error
    if not isinstance(row["train_play"], bool) or not isinstance(
        row["additional_search_required"], bool
    ):
        raise ValueError("guarded play booleans must be boolean")
    return kind, row["train_play"], row["additional_search_required"]


def _v2_candidate_proposal(
    value: object,
    *,
    board: Board,
    index: int,
) -> CandidateFamilyProposal:
    label = f"candidate-family proposal {index}"
    row = _v2_mapping(value, label=label)
    _v2_exact_fields(
        row,
        {"family", "producer", "provenance_sha256", "moves"},
        label=label,
    )
    family = row["family"]
    producer = row["producer"]
    if not isinstance(family, str) or not family or family.strip() != family:
        raise ValueError(f"{label} family must be a non-empty trimmed identifier")
    if not isinstance(producer, str) or not producer or producer.strip() != producer:
        raise ValueError(f"{label} producer must be a non-empty trimmed identifier")
    moves = tuple(
        _v2_legal_move(move, board=board, label=f"{label} move")
        for move in _v2_sequence(row["moves"], label=f"{label} moves")
    )
    if moves != tuple(sorted(set(moves))):
        raise ValueError(f"{label} moves must be unique and sorted")
    return CandidateFamilyProposal(
        family=family,
        producer=producer,
        provenance_sha256=_v2_sha256(
            row["provenance_sha256"], label=f"{label} provenance SHA-256"
        ),
        moves=moves,
    )


def _v2_policy_tv(scorers: tuple[CanonicalScorerTarget, ...]) -> float:
    moves = set().union(*(set(scorer.policy) for scorer in scorers))
    return 0.5 * math.fsum(
        abs(scorers[0].policy.get(move, 0.0) - scorers[1].policy.get(move, 0.0))
        for move in moves
    )


def _v2_derive_evidence(
    *,
    candidates: tuple[str, ...],
    matrices: tuple[CanonicalScorerMatrix, ...],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], bool]:
    worst_regret = {move: 0.0 for move in candidates}
    maximum_depth = {move: 0.0 for move in candidates}
    maximum_reply = {move: 0.0 for move in candidates}
    exact = True

    def worst_reply_score(candidate: CandidateScoreEvidence) -> float:
        # Every q_value in contract v2 is from the root player's perspective.
        # A reply is selected by the opponent, so the fail-closed backed-up
        # value is the lowest scored reply, never their average.
        return min(
            candidate.q_value,
            *(reply.q_value for reply in candidate.principal_replies),
        )

    for matrix in matrices:
        by_budget = [
            {candidate.move: candidate for candidate in budget.candidates}
            for budget in matrix.budgets
        ]
        deepest = by_budget[-1]
        best = max(worst_reply_score(candidate) for candidate in deepest.values())
        for move in candidates:
            scores = [budget[move] for budget in by_budget]
            direct_scores = [score.q_value for score in scores]
            backed_up_scores = [worst_reply_score(score) for score in scores]
            worst_regret[move] = max(
                worst_regret[move], best - worst_reply_score(deepest[move])
            )
            maximum_depth[move] = max(
                maximum_depth[move],
                max(direct_scores) - min(direct_scores),
            )
            # Reply stability is the cross-budget stability of the adversarial
            # backup, not max-minus-min across legal replies.  The latter would
            # punish a sound move merely because the opponent also has bad
            # replies; only the lowest reply is relevant to rational play.
            maximum_reply[move] = max(
                maximum_reply[move],
                max(backed_up_scores) - min(backed_up_scores),
            )
            exact = exact and all(
                score.bound is TeacherScoreBound.EXACT
                and all(
                    reply.bound is TeacherScoreBound.EXACT
                    for reply in score.principal_replies
                )
                for score in scores
            )
    return worst_regret, maximum_depth, maximum_reply, exact


def _v2_robust_policy(
    candidates: Sequence[str],
    regrets: Mapping[str, float],
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Put target mass only on exact minimax-regret argmin moves.

    A softmax over regret necessarily assigns non-zero mass to inferior moves.
    That is useful as an exploratory search prior, but it is not acceptable as
    the production best-move label requested by this contract.
    """

    minimum = min(regrets[move] for move in candidates)
    robust_best = tuple(
        sorted(
            move
            for move in candidates
            if math.isclose(regrets[move], minimum, rel_tol=0.0, abs_tol=1e-12)
        )
    )
    policy = {move: 1.0 / len(robust_best) for move in robust_best}
    return policy, robust_best


def _v2_proven_mate_policy(moves: Sequence[str]) -> dict[str, float]:
    """Give every independently proven winning root move equal top mass."""

    if not moves or len(set(moves)) != len(moves):
        raise ValueError("proven mate policy requires a non-empty unique move set")
    return {move: 1.0 / len(moves) for move in moves}


def _v2_position(
    value: object,
    *,
    game: GameRecord,
    sample: PositionSample,
    game_index: int,
    sample_index: int,
    required_candidate_families: tuple[str, ...],
) -> CanonicalDistillationTargetV2:
    label = f"canonical v2 target {game_index}:{sample_index}"
    row = _v2_mapping(value, label=label)
    _v2_exact_fields(
        row,
        {
            "game_index",
            "sample_index",
            "normalized_sfen",
            "history_context_sha256",
            "scorers",
            "candidate_proposals",
            "candidate_family_coverage_complete",
            "reply_coverage_complete",
            "candidate_moves",
            "budgets",
            "score_matrix",
            "thresholds",
            "proof",
            "play",
        },
        label=label,
    )
    if _v2_integer(row["game_index"], label="game_index") != game_index or _v2_integer(
        row["sample_index"], label="sample_index"
    ) != sample_index:
        raise ValueError("canonical v2 position indices do not match replay order")
    if row["normalized_sfen"] != normalized_sfen(sample.sfen):
        raise ValueError("canonical v2 normalized SFEN does not match replay")
    history_sha256 = _v2_sha256(
        row["history_context_sha256"], label="history_context_sha256"
    )
    if history_sha256 != canonical_history_context_sha256(game, sample):
        raise ValueError("canonical v2 history identity does not match replay")
    history = UsiPositionHistory.from_game(game, sample)
    history_input = HistoryInput(
        initial_sfen=history.initial_sfen,
        moves=history.moves,
        target_sfen=history.target_sfen,
        complete=Board(history.initial_sfen).to_sfen() == Board().to_sfen(),
    )
    board = history.target_board()
    scorers = tuple(
        validate_canonical_scorer_target(scorer, board=board, index=index)
        for index, scorer in enumerate(_v2_sequence(row["scorers"], label="scorers"))
    )
    if tuple(scorer.scorer_id for scorer in scorers) != CANONICAL_SCORER_IDS:
        raise ValueError("canonical v2 requires exactly NAGISA then Suisho11Plus targets")
    candidate_proposals = tuple(
        _v2_candidate_proposal(proposal, board=board, index=index)
        for index, proposal in enumerate(
            _v2_sequence(row["candidate_proposals"], label="candidate_proposals")
        )
    )
    candidate_families = tuple(proposal.family for proposal in candidate_proposals)
    if candidate_families != tuple(sorted(set(candidate_families))):
        raise ValueError("candidate family identifiers must be unique and sorted")
    if not isinstance(row["candidate_family_coverage_complete"], bool) or not isinstance(
        row["reply_coverage_complete"], bool
    ):
        raise ValueError("coverage fields must be boolean")
    candidates = tuple(
        _v2_legal_move(move, board=board, label="canonical v2 candidate")
        for move in _v2_sequence(row["candidate_moves"], label="candidate_moves")
    )
    if not candidates or candidates != tuple(sorted(set(candidates))):
        raise ValueError("candidate moves must be non-empty, unique, and sorted")
    proposal_union = set().union(*(set(proposal.moves) for proposal in candidate_proposals))
    if proposal_union != set(candidates):
        raise ValueError(
            "candidate moves must exactly equal the union of provenance-bound family proposals"
        )
    required_moves = set().union(*(set(scorer.policy) for scorer in scorers))
    if not required_moves.issubset(candidates):
        raise ValueError("candidate union must contain every canonical teacher policy move")
    budgets = tuple(
        _v2_integer(item, label="node budget", minimum=1)
        for item in _v2_sequence(row["budgets"], label="node budgets")
    )
    if len(budgets) < 2 or budgets != tuple(sorted(set(budgets))):
        raise ValueError("canonical v2 requires at least two strictly increasing node budgets")
    raw_matrices = _v2_sequence(row["score_matrix"], label="score_matrix")
    if len(raw_matrices) != len(CANONICAL_SCORER_IDS):
        raise ValueError("score matrix requires exactly two canonical scorers")
    matrices = tuple(
        _v2_scorer_matrix(
            raw,
            root=board,
            scorer_id=scorer_id,
            budgets=budgets,
            candidates=candidates,
        )
        for raw, scorer_id in zip(raw_matrices, CANONICAL_SCORER_IDS, strict=True)
    )
    _v2_validate_scorer_targets_against_matrices(scorers, matrices)
    thresholds = _v2_thresholds(row["thresholds"])
    proof = _v2_proof(row["proof"], board=board)
    declared_kind, declared_train, declared_search = _v2_play_declaration(row["play"])
    regrets, depth_dispersion, reply_dispersion, exact_bounds = _v2_derive_evidence(
        candidates=candidates, matrices=matrices
    )
    coverage_complete = row["candidate_family_coverage_complete"]
    declared_reply_complete = row["reply_coverage_complete"]
    derived_reply_complete = all(
        candidate.legal_reply_coverage_complete
        for matrix in matrices
        for budget in matrix.budgets
        for candidate in budget.candidates
    )
    if declared_reply_complete and not derived_reply_complete:
        raise ValueError(
            "reply_coverage_complete cannot be asserted without scoring every legal reply"
        )
    reply_complete = declared_reply_complete and derived_reply_complete
    if coverage_complete and candidate_families != required_candidate_families:
        raise ValueError(
            "complete candidate-family coverage must equal the sidecar's required families"
        )
    calibration_complete = all(
        matrix.calibration.independent_validation_passed for matrix in matrices
    )
    reported_winning_mate = any(
        candidate.score_kind is TeacherScoreKind.MATE
        and (
            candidate.mate_unknown_sign == 1
            or (candidate.mate_plies is not None and candidate.mate_plies >= 0)
        )
        for matrix in matrices
        for budget in matrix.budgets
        for candidate in budget.candidates
    )
    if proof.state is MateProofState.PROVEN_WIN and not set(proof.moves).issubset(candidates):
        raise ValueError("every internally proven mate move must be in the candidate union")
    robust_policy, robust_best = _v2_robust_policy(candidates, regrets)
    regret_consensus_exists = all(
        regrets[move] <= thresholds.regret + _DERIVATION_TOLERANCE
        for move in robust_best
    )
    depth_stable = all(
        depth_dispersion[move] <= thresholds.depth_dispersion + _DERIVATION_TOLERANCE
        for move in candidates
    )
    reply_stable = all(
        reply_dispersion[move] <= thresholds.reply_dispersion + _DERIVATION_TOLERANCE
        for move in candidates
    )

    if proof.state is MateProofState.PROVEN_WIN:
        expected_kind = PlayTargetKind.PROVEN_MATE
        expected_train = True
        expected_search = False
        play_policy = _v2_proven_mate_policy(proof.moves)
        robust_best = proof.moves
        value_interval: tuple[float, float] | None = (1.0, 1.0)
        wdl: tuple[float, float, float] | None = (0.0, 0.0, 1.0)
    elif (
        coverage_complete
        and reply_complete
        and calibration_complete
        and exact_bounds
        and regret_consensus_exists
        and depth_stable
        and reply_stable
        and not reported_winning_mate
        and proof.state
        in {MateProofState.NOT_APPLICABLE, MateProofState.PROVEN_NO_MATE_WITHIN_BOUND}
    ):
        expected_kind = PlayTargetKind.ROBUST_CONSENSUS
        expected_train = True
        expected_search = False
        play_policy = robust_policy
        scorer_values = [scorer.value for scorer in scorers]
        value_interval = (min(scorer_values), max(scorer_values))
        wdl = None
    else:
        expected_kind = PlayTargetKind.UNRESOLVED
        expected_train = False
        expected_search = True
        play_policy = {}
        robust_best = ()
        value_interval = None
        wdl = None
    if (declared_kind, declared_train, declared_search) != (
        expected_kind,
        expected_train,
        expected_search,
    ):
        raise ValueError(
            "guarded play declaration disagrees with the independently derived evidence gate"
        )
    if play_policy:
        total = math.fsum(play_policy.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
            raise AssertionError("derived guarded play policy is not normalized")
        maximum = max(play_policy.values())
        if any(
            not math.isclose(play_policy[move], maximum, rel_tol=0.0, abs_tol=1e-12)
            for move in robust_best
        ):
            raise AssertionError("derived robust best move is not policy top-1")
    policy_uncertainty = _v2_policy_tv(scorers)
    value_uncertainty = abs(scorers[0].value - scorers[1].value) / 2.0
    uncertainty = (
        1.0
        if not expected_train
        else min(1.0, max(policy_uncertainty, value_uncertainty))
    )
    reasons: set[CanonicalDisagreementReason] = set()
    if scorers[0].best_move != scorers[1].best_move:
        reasons.add(CanonicalDisagreementReason.TEACHER_BEST_MOVE)
    if policy_uncertainty > _DERIVATION_TOLERANCE:
        reasons.add(CanonicalDisagreementReason.TEACHER_POLICY_DISTRIBUTION)
    if scorers[0].value * scorers[1].value < 0.0:
        reasons.add(CanonicalDisagreementReason.TEACHER_VALUE_SIGN)
    if not coverage_complete:
        reasons.add(CanonicalDisagreementReason.CANDIDATE_FAMILY_COVERAGE)
    if not reply_complete:
        reasons.add(CanonicalDisagreementReason.REPLY_COVERAGE)
    if not calibration_complete:
        reasons.add(CanonicalDisagreementReason.CALIBRATION)
    if not exact_bounds:
        reasons.add(CanonicalDisagreementReason.NON_EXACT_BOUND)
    if not depth_stable:
        reasons.add(CanonicalDisagreementReason.DEPTH_INSTABILITY)
    if not reply_stable:
        reasons.add(CanonicalDisagreementReason.REPLY_INSTABILITY)
    if not regret_consensus_exists:
        reasons.add(CanonicalDisagreementReason.EMPTY_CONSENSUS)
    if reported_winning_mate and proof.state is not MateProofState.PROVEN_WIN:
        reasons.add(CanonicalDisagreementReason.REPORTED_MATE_UNPROVEN)
    if proof.state is MateProofState.UNKNOWN_LIMIT:
        reasons.add(CanonicalDisagreementReason.PROOF_UNKNOWN_LIMIT)
    return CanonicalDistillationTargetV2(
        game_index=game_index,
        sample_index=sample_index,
        normalized_sfen=normalized_sfen(sample.sfen),
        history_context_sha256=history_sha256,
        history=history_input,
        scorers=scorers,
        candidate_proposals=candidate_proposals,
        candidate_families=candidate_families,
        candidate_family_coverage_complete=coverage_complete,
        reply_coverage_complete=reply_complete,
        candidate_moves=candidates,
        budgets=budgets,
        score_matrix=matrices,
        thresholds=thresholds,
        proof=proof,
        play=GuardedPlayTarget(
            kind=expected_kind,
            train_play=expected_train,
            additional_search_required=expected_search,
            policy=play_policy,
            robust_best_moves=robust_best,
            value_interval=value_interval,
            wdl=wdl,
        ),
        uncertainty_target=uncertainty,
        worst_teacher_regret=regrets,
        maximum_depth_dispersion=depth_dispersion,
        maximum_reply_dispersion=reply_dispersion,
        disagreement_reasons=tuple(sorted(reasons, key=str)),
    )


def _v2_validate_rights_summary(value: object) -> dict[str, object]:
    summary = validate_rights_restriction_summary(
        _v2_mapping(value, label="canonical v2 rights restriction summary")
    )
    raw_sources = cast(list[dict[str, object]], summary["sources"])
    if tuple(cast(str, source["rights_id"]) for source in raw_sources) != tuple(
        sorted(CANONICAL_SCORER_IDS)
    ):
        raise ValueError("canonical v2 rights summary must bind exactly NAGISA and Suisho11Plus")
    if any(not cast(list[str], source["sidecar_sha256s"]) for source in raw_sources):
        raise ValueError("each canonical v2 teacher requires source-sidecar evidence")
    if summary["publication_allowed"] is not False:
        raise ValueError("Suisho11Plus canonical v2 targets must remain local-only")
    return summary


def load_canonical_target_sidecar_v2(
    replay: Path,
    sidecar: Path,
    *,
    games: Sequence[GameRecord] | None = None,
) -> CanonicalTargetSidecarV2:
    """Validate the complete v2 target contract against an immutable replay."""

    replay_path = _v2_regular_file(replay, label="canonical v2 replay")
    sidecar_path = _v2_regular_file(sidecar, label="canonical v2 sidecar")
    source_games = tuple(load_games(replay_path) if games is None else games)
    payload = _v2_read_json(sidecar_path)
    schema = payload.get("schema")
    if schema != CANONICAL_TARGET_V2_SCHEMA:
        if schema == "meteo-canonical-distillation-targets-v1":
            raise ValueError(
                "canonical target v1 is training-disabled: its single value head broadcasts "
                "over two teacher values and converges to their arithmetic midpoint; rebuild "
                "a complete canonical-target-contract-v2 sidecar"
            )
        raise ValueError("unsupported canonical target sidecar schema")
    _v2_exact_fields(
        payload,
        {
            "schema",
            "replay",
            "position_identity",
            "target_mode",
            "canonical_scorers",
            "required_candidate_families",
            "rights_restriction_summary",
            "positions",
        },
        label="canonical v2 sidecar",
    )
    if payload["position_identity"] != CANONICAL_POSITION_IDENTITY:
        raise ValueError("canonical v2 position identity contract does not match")
    if payload["target_mode"] != CANONICAL_TARGET_V2_MODE:
        raise ValueError("canonical v2 target mode does not match")
    replay_row = _v2_mapping(payload["replay"], label="canonical v2 replay identity")
    _v2_exact_fields(replay_row, {"sha256", "bytes"}, label="canonical v2 replay identity")
    replay_identity = CanonicalReplayIdentity(
        sha256=_v2_sha256(replay_row["sha256"], label="canonical v2 replay SHA-256"),
        bytes=_v2_integer(replay_row["bytes"], label="canonical v2 replay bytes"),
    )
    if replay_identity.bytes != replay_path.stat().st_size or replay_identity.sha256 != sha256_file(
        replay_path
    ):
        raise ValueError("canonical v2 replay identity does not match")
    descriptors = tuple(
        _v2_descriptor(descriptor, index=index)
        for index, descriptor in enumerate(
            _v2_sequence(payload["canonical_scorers"], label="canonical v2 scorers")
        )
    )
    if tuple(descriptor.scorer_id for descriptor in descriptors) != CANONICAL_SCORER_IDS:
        raise ValueError("canonical v2 scorer order must be NAGISA then Suisho11Plus")
    shared_search_settings = {
        (descriptor.threads, descriptor.hash_mb, descriptor.multipv)
        for descriptor in descriptors
    }
    if len(shared_search_settings) != 1:
        raise ValueError("canonical scorers must use the same threads, hash, and MultiPV")
    raw_required_families = _v2_sequence(
        payload["required_candidate_families"],
        label="required candidate families",
    )
    if not all(
        isinstance(item, str) and item.strip() == item and item
        for item in raw_required_families
    ):
        raise ValueError("required candidate families must be non-empty trimmed strings")
    required_candidate_families = tuple(cast(list[str], raw_required_families))
    if required_candidate_families != tuple(sorted(set(required_candidate_families))):
        raise ValueError("required candidate families must be unique and sorted")
    mandatory_families = {"meteo", "nagisa", "suisho", "tactical"}
    if not mandatory_families.issubset(required_candidate_families):
        raise ValueError(
            "canonical v2 requires nagisa, suisho, meteo, and tactical candidate families"
        )
    rights_summary = _v2_validate_rights_summary(payload["rights_restriction_summary"])
    for game_index, game in enumerate(source_games):
        if game.termination is Termination.MAX_PLIES:
            raise ValueError(f"canonical v2 rejects incomplete max_plies game {game_index}")
    expected = [
        (game_index, sample_index, game, sample)
        for game_index, game in enumerate(source_games)
        for sample_index, sample in enumerate(game.samples)
    ]
    raw_positions = _v2_sequence(payload["positions"], label="canonical v2 positions")
    if len(raw_positions) != len(expected):
        raise ValueError("canonical v2 sidecar must cover every replay sample exactly")
    positions = tuple(
        _v2_position(
            raw,
            game=game,
            sample=sample,
            game_index=game_index,
            sample_index=sample_index,
            required_candidate_families=required_candidate_families,
        )
        for raw, (game_index, sample_index, game, sample) in zip(
            raw_positions, expected, strict=True
        )
    )
    return CanonicalTargetSidecarV2(
        schema=CANONICAL_TARGET_V2_SCHEMA,
        replay=replay_identity,
        position_identity=CANONICAL_POSITION_IDENTITY,
        target_mode=CANONICAL_TARGET_V2_MODE,
        canonical_scorers=descriptors,
        required_candidate_families=required_candidate_families,
        rights_restriction_summary=rights_summary,
        positions=positions,
    )


def canonical_v2_sidecar_sha256(path: Path) -> str:
    """Return a public-safe identity for an already validated v2 sidecar."""

    source = _v2_regular_file(path, label="canonical v2 sidecar")
    return hashlib.sha256(source.read_bytes()).hexdigest()
