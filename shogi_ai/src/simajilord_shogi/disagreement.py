"""Deterministic, family-aware selection of positions that merit deeper search."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from rsshogi.core import Board, Move

from .artifact_provenance import (
    FileIdentity,
    InputArtifact,
    canonical_json_sha256,
    identify_file,
    identify_input_artifact,
    sha256_file,
)
from .domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from .ensemble import ENSEMBLE_REPORT_SCHEMA, normalized_sfen
from .replay import append_games, load_games
from .research_context import TrajectoryContext, trajectory_contexts

DISAGREEMENT_REPORT_SCHEMA = "meteo-deep-disagreement-selection-v1"


class DisagreementReason(StrEnum):
    """Independent reasons that can send a position to expensive deep search."""

    TOP1_DISAGREEMENT = "top1_disagreement"
    VALUE_DIRECTION_DISAGREEMENT = "value_direction_disagreement"
    HIGH_JENSEN_SHANNON = "high_jensen_shannon"
    FAMILY_UNIQUE_BEST_MOVE = "family_unique_best_move"
    REPORTED_WINNING_MATE = "reported_winning_mate"


@dataclass(frozen=True, slots=True)
class DisagreementConfig:
    """Selection thresholds; zero ``maximum_positions`` is represented by ``None``."""

    high_js_threshold: float = 0.25
    value_neutral_threshold: float = 0.05
    maximum_positions: int | None = None
    opening_priority_max_ply: int = 24
    opening_priority_nodes: int = 2_000_000
    centipawn_value_scale: float = 1_200.0
    family_value_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.high_js_threshold) or not 0 <= self.high_js_threshold <= 1:
            raise ValueError("high_js_threshold must be finite and in [0, 1]")
        if (
            not math.isfinite(self.value_neutral_threshold)
            or not 0 <= self.value_neutral_threshold < 1
        ):
            raise ValueError("value_neutral_threshold must be finite and in [0, 1)")
        if self.maximum_positions is not None and self.maximum_positions < 1:
            raise ValueError("maximum_positions must be positive when supplied")
        if self.opening_priority_max_ply < 1 or self.opening_priority_nodes < 1:
            raise ValueError("opening priority ply and nodes must be positive")
        if not math.isfinite(self.centipawn_value_scale) or self.centipawn_value_scale <= 0:
            raise ValueError("centipawn_value_scale must be finite and positive")
        normalized_names: set[str] = set()
        for family, scale in self.family_value_scales.items():
            name = family.strip()
            if not name:
                raise ValueError("family value-scale name must not be empty")
            key = name.casefold()
            if key in normalized_names:
                raise ValueError(f"duplicate family value-scale name: {family}")
            normalized_names.add(key)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"family value scale must be finite and positive: {family}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DisagreementTeacherInput:
    """One manifest teacher and its correlation family."""

    label: str
    replay: Path
    family: str


@dataclass(frozen=True, slots=True)
class ReportedWinningMate:
    """A positive USI mate score, explicitly not an internal mate proof."""

    teacher_label: str
    teacher_family: str
    teacher_source: str
    game_index: int
    sample_index: int
    move: str
    bound: str
    mate_plies: int | None
    mate_unknown_sign: int | None


@dataclass(frozen=True, slots=True)
class SelectedPosition:
    normalized_sfen: str
    output_sfen: str
    selection_rank: int
    reasons: tuple[str, ...]
    family_top1: dict[str, str]
    family_values: dict[str, float | None]
    family_unique_best_moves: dict[str, str]
    family_normalized_jensen_shannon: float
    manifest_normalized_jensen_shannon: float
    reported_winning_mates: tuple[ReportedWinningMate, ...]
    trajectory: TrajectoryContext
    recommended_nodes: int | None
    teacher_stories: tuple[TeacherPositionStory, ...]


@dataclass(frozen=True, slots=True)
class TeacherPositionStory:
    """One teacher occurrence kept whole; PVs are never spliced across families."""

    teacher_label: str
    teacher_family: str
    teacher_source: str
    game_index: int
    sample_index: int
    top_move: str
    raw_exact_root_cp: float | None
    posthoc_value: float | None
    legacy_teacher_value: float
    variations: tuple[TeacherVariation, ...]


@dataclass(frozen=True, slots=True)
class DisagreementTeacherReport:
    label: str
    family: str
    artifact: InputArtifact
    manifest_recorded_path: str
    manifest_recorded_sha256: str
    eligible_observations: int
    unique_normalized_positions: int
    duplicate_observations_collapsed: int


@dataclass(frozen=True, slots=True)
class DisagreementSelectionReport:
    schema: str
    top1_definition: str
    family_aggregation: str
    value_transform_semantics: str
    ensemble_replay: InputArtifact
    ensemble_manifest: FileIdentity
    teachers: tuple[DisagreementTeacherReport, ...]
    config: DisagreementConfig
    base_normalized_positions: int
    candidate_positions: int
    selected_positions: int
    candidate_reason_counts: dict[str, int]
    selected_reason_counts: dict[str, int]
    positions: tuple[SelectedPosition, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class DisagreementSelectionBuild:
    games: tuple[GameRecord, ...]
    report: DisagreementSelectionReport


@dataclass(frozen=True, slots=True)
class _TeacherObservation:
    game_index: int
    sample_index: int
    sample: PositionSample
    policy: dict[str, float]
    raw_exact_root_cp: float | None
    posthoc_value: float | None
    legacy_teacher_value: float


@dataclass(frozen=True, slots=True)
class _TeacherAggregate:
    label: str
    family: str
    policy: dict[str, float]
    value: float | None
    reported_winning_mates: tuple[ReportedWinningMate, ...]
    stories: tuple[TeacherPositionStory, ...]


@dataclass(frozen=True, slots=True)
class _FamilyAggregate:
    family: str
    policy: dict[str, float]
    value: float | None


@dataclass(frozen=True, slots=True)
class _Candidate:
    normalized_sfen: str
    output_sfen: str
    reasons: tuple[DisagreementReason, ...]
    family_top1: dict[str, str]
    family_values: dict[str, float | None]
    family_unique_best_moves: dict[str, str]
    family_normalized_jensen_shannon: float
    manifest_normalized_jensen_shannon: float
    reported_winning_mates: tuple[ReportedWinningMate, ...]
    trajectory: TrajectoryContext
    recommended_nodes: int | None
    teacher_stories: tuple[TeacherPositionStory, ...]


def _validate_teacher_inputs(
    teachers: Sequence[DisagreementTeacherInput],
) -> tuple[DisagreementTeacherInput, ...]:
    if len(teachers) < 2:
        raise ValueError("disagreement selection requires at least two teacher replays")
    labels: set[str] = set()
    paths: set[str] = set()
    validated: list[DisagreementTeacherInput] = []
    for supplied in teachers:
        label = supplied.label.strip()
        family = supplied.family.strip()
        if not label:
            raise ValueError("teacher label must not be empty")
        if not family:
            raise ValueError(f"teacher family must not be empty: {label}")
        normalized_label = label.casefold()
        if normalized_label in labels:
            raise ValueError(f"duplicate teacher label: {label}")
        labels.add(normalized_label)
        replay = supplied.replay.expanduser().resolve()
        normalized_path = str(replay).casefold()
        if normalized_path in paths:
            raise ValueError(f"duplicate teacher replay: {replay}")
        paths.add(normalized_path)
        validated.append(DisagreementTeacherInput(label=label, replay=replay, family=family))
    return tuple(validated)


def _json_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _json_array(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return value


def _load_and_validate_manifest(
    path: Path,
    *,
    ensemble_replay: FileIdentity,
    teachers: Sequence[DisagreementTeacherInput],
) -> tuple[FileIdentity, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    identity = identify_file(path)
    raw: object = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    manifest = _json_object(raw, context="ensemble manifest")
    if manifest.get("schema") != ENSEMBLE_REPORT_SCHEMA:
        raise ValueError(
            f"unsupported ensemble manifest schema: {manifest.get('schema')!r}"
        )
    output = _json_object(manifest.get("output"), context="ensemble manifest output")
    if output.get("replay_sha256") != ensemble_replay.sha256:
        raise ValueError("ensemble replay SHA-256 does not match the manifest")

    manifest_inputs: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(
        _json_array(manifest.get("inputs"), context="ensemble manifest inputs")
    ):
        row = _json_object(value, context=f"ensemble manifest input {index}")
        label = str(row.get("label", "")).strip()
        if not label:
            raise ValueError(f"ensemble manifest input {index} has no label")
        key = label.casefold()
        if key in manifest_inputs:
            raise ValueError(f"duplicate ensemble manifest teacher label: {label}")
        manifest_inputs[key] = row
    supplied_labels = {teacher.label.casefold() for teacher in teachers}
    if set(manifest_inputs) != supplied_labels:
        missing = sorted(set(manifest_inputs) - supplied_labels)
        extra = sorted(supplied_labels - set(manifest_inputs))
        raise ValueError(
            "teacher labels do not exactly match ensemble manifest; "
            f"missing={missing}, extra={extra}"
        )

    positions: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(
        _json_array(manifest.get("positions"), context="ensemble manifest positions")
    ):
        row = _json_object(value, context=f"ensemble manifest position {index}")
        key = str(row.get("normalized_sfen", ""))
        if not key:
            raise ValueError(f"ensemble manifest position {index} has no normalized_sfen")
        if key in positions:
            raise ValueError(f"duplicate normalized SFEN in ensemble manifest: {key}")
        positions[key] = row
    return identity, manifest_inputs, positions


def _normalized_policy(
    policy: dict[str, float], *, sfen: str, teacher_label: str
) -> dict[str, float]:
    board = Board(sfen)
    if not board.is_valid():
        raise ValueError(f"invalid teacher SFEN for {teacher_label}: {sfen}")
    normalized: dict[str, float] = defaultdict(float)
    for move_usi, raw_probability in policy.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(
                f"teacher {teacher_label!r} has negative or non-finite policy mass"
            )
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(
                f"teacher {teacher_label!r} has illegal policy move {move_usi} at {sfen}"
            )
        if probability > 0:
            normalized[move_usi] += probability
    total = math.fsum(normalized.values())
    if total <= 0:
        raise ValueError(f"teacher {teacher_label!r} has empty policy at {sfen}")
    return {move: normalized[move] / total for move in sorted(normalized)}


def _average_policies(policies: Sequence[dict[str, float]]) -> dict[str, float]:
    if not policies:
        raise ValueError("at least one policy is required")
    averaged: dict[str, float] = defaultdict(float)
    for policy in policies:
        for move, probability in policy.items():
            averaged[move] += probability / len(policies)
    total = math.fsum(averaged.values())
    return {move: averaged[move] / total for move in sorted(averaged)}


def _policy_top1(policy: dict[str, float]) -> str:
    return min(policy, key=lambda move: (-policy[move], move))


def _policy_entropy(policy: dict[str, float]) -> float:
    return -math.fsum(
        probability * math.log(probability)
        for probability in policy.values()
        if probability > 0
    )


def _normalized_js(families: Sequence[_FamilyAggregate]) -> float:
    if len(families) < 2:
        return 0.0
    mixture = _average_policies([family.policy for family in families])
    js = max(
        0.0,
        _policy_entropy(mixture)
        - math.fsum(_policy_entropy(family.policy) for family in families) / len(families),
    )
    return min(1.0, js / math.log(len(families)))


def _is_reported_winning_mate(variation: TeacherVariation) -> bool:
    return variation.score_kind is TeacherScoreKind.MATE and (
        (variation.mate_plies is not None and variation.mate_plies > 0)
        or variation.mate_unknown_sign == 1
    )


def _value_scale_for_family(family: str, config: DisagreementConfig) -> float:
    configured = {
        name.casefold(): scale for name, scale in config.family_value_scales.items()
    }
    return float(configured.get(family.casefold(), config.centipawn_value_scale))


def _load_teacher_observations(
    teacher: DisagreementTeacherInput,
    config: DisagreementConfig,
) -> tuple[dict[str, tuple[_TeacherObservation, ...]], int, int]:
    grouped: dict[str, list[_TeacherObservation]] = defaultdict(list)
    eligible = 0
    for game_index, game in enumerate(load_games(teacher.replay)):
        if game.termination is Termination.MAX_PLIES:
            continue
        for sample_index, sample in enumerate(game.samples):
            has_policy = sample.teacher_policy is not None
            has_value = sample.teacher_value is not None
            if not has_policy and not has_value:
                continue
            if has_policy != has_value:
                raise ValueError(
                    f"teacher {teacher.label!r} must provide policy and value together at "
                    f"game {game_index}, sample {sample_index}"
                )
            legacy_value = float(cast(float, sample.teacher_value))
            if not math.isfinite(legacy_value) or not -1 <= legacy_value <= 1:
                raise ValueError(
                    f"teacher {teacher.label!r} has value outside [-1, 1] at "
                    f"game {game_index}, sample {sample_index}"
                )
            board = Board(sample.sfen)
            policy = _normalized_policy(
                cast(dict[str, float], sample.teacher_policy),
                sfen=sample.sfen,
                teacher_label=teacher.label,
            )
            exact_scores: dict[str, tuple[int, float]] = {}
            for variation in sample.teacher_variations or ():
                root_board_move = Move.from_usi(variation.move)
                if not board.is_legal_move(root_board_move):
                    raise ValueError(
                        f"teacher {teacher.label!r} has illegal variation root "
                        f"{variation.move} at {sample.sfen}"
                    )
                if (
                    variation.score_kind is TeacherScoreKind.CENTIPAWN
                    and variation.bound is TeacherScoreBound.EXACT
                    and variation.score_cp is not None
                ):
                    existing = exact_scores.get(variation.move)
                    if existing is None or variation.rank < existing[0]:
                        exact_scores[variation.move] = (
                            variation.rank,
                            float(variation.score_cp),
                        )
            preferred_moves = tuple(
                move
                for move in (sample.teacher_best_move, _policy_top1(policy))
                if move is not None
            )
            root_cp = next(
                (
                    exact_scores[move][1]
                    for move in preferred_moves
                    if move in exact_scores
                ),
                None,
            )
            if root_cp is None and exact_scores:
                root_score_move = min(
                    exact_scores,
                    key=lambda move: (exact_scores[move][0], move),
                )
                root_cp = exact_scores[root_score_move][1]
            value_scale = _value_scale_for_family(teacher.family, config)
            key = normalized_sfen(sample.sfen)
            grouped[key].append(
                _TeacherObservation(
                    game_index=game_index,
                    sample_index=sample_index,
                    sample=sample,
                    policy=policy,
                    raw_exact_root_cp=root_cp,
                    posthoc_value=(
                        None if root_cp is None else math.tanh(root_cp / value_scale)
                    ),
                    legacy_teacher_value=legacy_value,
                )
            )
            eligible += 1
    frozen = {key: tuple(observations) for key, observations in grouped.items()}
    return frozen, eligible, eligible - len(frozen)


def _teacher_aggregate(
    teacher: DisagreementTeacherInput,
    observations: tuple[_TeacherObservation, ...],
) -> _TeacherAggregate:
    reports: list[ReportedWinningMate] = []
    stories: list[TeacherPositionStory] = []
    for observation in observations:
        source = observation.sample.teacher_source or "unknown"
        stories.append(
            TeacherPositionStory(
                teacher_label=teacher.label,
                teacher_family=teacher.family,
                teacher_source=source,
                game_index=observation.game_index,
                sample_index=observation.sample_index,
                top_move=_policy_top1(observation.policy),
                raw_exact_root_cp=observation.raw_exact_root_cp,
                posthoc_value=observation.posthoc_value,
                legacy_teacher_value=observation.legacy_teacher_value,
                variations=observation.sample.teacher_variations or (),
            )
        )
        for variation in observation.sample.teacher_variations or ():
            if _is_reported_winning_mate(variation):
                reports.append(
                    ReportedWinningMate(
                        teacher_label=teacher.label,
                        teacher_family=teacher.family,
                        teacher_source=source,
                        game_index=observation.game_index,
                        sample_index=observation.sample_index,
                        move=variation.move,
                        bound=variation.bound.value,
                        mate_plies=variation.mate_plies,
                        mate_unknown_sign=variation.mate_unknown_sign,
                    )
                )
    reports.sort(
        key=lambda report: (
            report.teacher_family.casefold(),
            report.teacher_label.casefold(),
            report.game_index,
            report.sample_index,
            report.move,
        )
    )
    return _TeacherAggregate(
        label=teacher.label,
        family=teacher.family,
        policy=_average_policies([observation.policy for observation in observations]),
        value=(
            math.fsum(
                observation.posthoc_value
                for observation in observations
                if observation.posthoc_value is not None
            )
            / sum(observation.posthoc_value is not None for observation in observations)
            if any(observation.posthoc_value is not None for observation in observations)
            else None
        ),
        reported_winning_mates=tuple(reports),
        stories=tuple(
            sorted(stories, key=lambda story: (story.game_index, story.sample_index))
        ),
    )


def _family_aggregates(
    teachers: Sequence[_TeacherAggregate],
) -> tuple[_FamilyAggregate, ...]:
    grouped: dict[str, list[_TeacherAggregate]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for teacher in teachers:
        key = teacher.family.casefold()
        grouped[key].append(teacher)
        display_names.setdefault(key, teacher.family)
    return tuple(
        _FamilyAggregate(
            family=display_names[key],
            policy=_average_policies([teacher.policy for teacher in grouped[key]]),
            value=(
                math.fsum(
                    teacher.value
                    for teacher in grouped[key]
                    if teacher.value is not None
                )
                / sum(teacher.value is not None for teacher in grouped[key])
                if any(teacher.value is not None for teacher in grouped[key])
                else None
            ),
        )
        for key in sorted(grouped)
    )


def _value_direction(value: float, *, neutral_threshold: float) -> str:
    if value > neutral_threshold:
        return "positive"
    if value < -neutral_threshold:
        return "negative"
    return "neutral"


def _reason_counts(candidates: Sequence[_Candidate]) -> dict[str, int]:
    counts = Counter(reason.value for candidate in candidates for reason in candidate.reasons)
    return {reason.value: counts[reason.value] for reason in DisagreementReason}


def _selection_order(candidate: _Candidate) -> tuple[int, int, int, float, float, str]:
    mate_priority = int(DisagreementReason.REPORTED_WINNING_MATE in candidate.reasons)
    opening_priority = int(candidate.recommended_nodes is not None)
    values = tuple(value for value in candidate.family_values.values() if value is not None)
    value_width = max(values) - min(values) if values else 0.0
    return (
        -mate_priority,
        -opening_priority,
        -len(candidate.reasons),
        -candidate.family_normalized_jensen_shannon,
        -value_width,
        candidate.normalized_sfen,
    )


def build_disagreement_replay(
    ensemble_replay: Path,
    ensemble_manifest: Path,
    teachers: Sequence[DisagreementTeacherInput],
    *,
    config: DisagreementConfig | None = None,
) -> DisagreementSelectionBuild:
    """Select expensive-search candidates without trusting a convex target as truth."""

    effective_config = config or DisagreementConfig()
    validated_teachers = _validate_teacher_inputs(teachers)
    available_families = {teacher.family.casefold() for teacher in validated_teachers}
    unknown_scale_families = sorted(
        family.casefold()
        for family in effective_config.family_value_scales
        if family.casefold() not in available_families
    )
    if unknown_scale_families:
        raise ValueError(
            f"family value scales name unavailable families: {unknown_scale_families}"
        )
    ensemble_artifact = identify_input_artifact(ensemble_replay)
    manifest_identity, manifest_inputs, manifest_positions = _load_and_validate_manifest(
        ensemble_manifest,
        ensemble_replay=ensemble_artifact.file,
        teachers=validated_teachers,
    )

    ensemble_games = tuple(load_games(Path(ensemble_artifact.file.path)))
    ensemble_contexts = trajectory_contexts(ensemble_games)
    ensemble_locations: dict[str, tuple[int, int, PositionSample]] = {}
    for game_index, game in enumerate(ensemble_games):
        for sample_index, sample in enumerate(game.samples):
            key = normalized_sfen(sample.sfen)
            if key in ensemble_locations:
                raise ValueError(f"duplicate normalized SFEN in ensemble replay: {key}")
            ensemble_locations[key] = (game_index, sample_index, sample)
    if set(ensemble_locations) != set(manifest_positions):
        replay_only = sorted(set(ensemble_locations) - set(manifest_positions))[:3]
        manifest_only = sorted(set(manifest_positions) - set(ensemble_locations))[:3]
        raise ValueError(
            "ensemble replay positions do not match manifest positions; "
            f"replay_only={replay_only}, manifest_only={manifest_only}"
        )

    observations_by_label: dict[str, dict[str, tuple[_TeacherObservation, ...]]] = {}
    teacher_reports: list[DisagreementTeacherReport] = []
    for teacher in validated_teachers:
        artifact = identify_input_artifact(teacher.replay)
        manifest_row = manifest_inputs[teacher.label.casefold()]
        recorded_sha = str(manifest_row.get("replay_sha256", ""))
        if recorded_sha != artifact.file.sha256:
            raise ValueError(
                f"teacher {teacher.label!r} replay SHA-256 does not match ensemble manifest"
            )
        observations, eligible, duplicate_count = _load_teacher_observations(
            teacher,
            effective_config,
        )
        observations_by_label[teacher.label.casefold()] = observations
        teacher_reports.append(
            DisagreementTeacherReport(
                label=teacher.label,
                family=teacher.family,
                artifact=artifact,
                manifest_recorded_path=str(manifest_row.get("replay", "")),
                manifest_recorded_sha256=recorded_sha,
                eligible_observations=eligible,
                unique_normalized_positions=len(observations),
                duplicate_observations_collapsed=duplicate_count,
            )
        )

    candidates: list[_Candidate] = []
    for key in sorted(ensemble_locations):
        teacher_aggregates = [
            _teacher_aggregate(teacher, observations_by_label[teacher.label.casefold()][key])
            for teacher in validated_teachers
            if key in observations_by_label[teacher.label.casefold()]
        ]
        if len(teacher_aggregates) != len(validated_teachers):
            missing = sorted(
                teacher.label
                for teacher in validated_teachers
                if key not in observations_by_label[teacher.label.casefold()]
            )
            raise ValueError(f"teacher replays lack ensemble position {key}; missing={missing}")
        families = _family_aggregates(teacher_aggregates)
        family_top1 = {family.family: _policy_top1(family.policy) for family in families}
        family_values = {family.family: family.value for family in families}
        top1_counts = Counter(family_top1.values())
        unique_best = {
            family: move
            for family, move in family_top1.items()
            if len(families) > 1 and top1_counts[move] == 1
        }
        directions = {
            _value_direction(
                family.value,
                neutral_threshold=effective_config.value_neutral_threshold,
            )
            for family in families
            if family.value is not None
        }
        family_js = _normalized_js(families)
        reported_mates = tuple(
            report
            for teacher in teacher_aggregates
            for report in teacher.reported_winning_mates
        )
        teacher_stories = tuple(
            story for teacher in teacher_aggregates for story in teacher.stories
        )
        reasons: list[DisagreementReason] = []
        if len(set(family_top1.values())) > 1:
            reasons.append(DisagreementReason.TOP1_DISAGREEMENT)
        if len(directions) > 1:
            reasons.append(DisagreementReason.VALUE_DIRECTION_DISAGREEMENT)
        if family_js >= effective_config.high_js_threshold:
            reasons.append(DisagreementReason.HIGH_JENSEN_SHANNON)
        if unique_best:
            reasons.append(DisagreementReason.FAMILY_UNIQUE_BEST_MOVE)
        if reported_mates:
            reasons.append(DisagreementReason.REPORTED_WINNING_MATE)
        if not reasons:
            continue
        manifest_position = manifest_positions[key]
        candidates.append(
            _Candidate(
                normalized_sfen=key,
                output_sfen=ensemble_locations[key][2].sfen,
                reasons=tuple(sorted(reasons, key=lambda reason: reason.value)),
                family_top1=dict(sorted(family_top1.items())),
                family_values=dict(sorted(family_values.items())),
                family_unique_best_moves=dict(sorted(unique_best.items())),
                family_normalized_jensen_shannon=family_js,
                manifest_normalized_jensen_shannon=float(
                    manifest_position.get("normalized_jensen_shannon_divergence", 0.0)
                ),
                reported_winning_mates=reported_mates,
                trajectory=ensemble_contexts[key],
                recommended_nodes=(
                    effective_config.opening_priority_nodes
                    if ensemble_contexts[key].absolute_ply + 1
                    <= effective_config.opening_priority_max_ply
                    else None
                ),
                teacher_stories=teacher_stories,
            )
        )

    ranked = sorted(candidates, key=_selection_order)
    selected = (
        ranked[: effective_config.maximum_positions]
        if effective_config.maximum_positions is not None
        else ranked
    )
    if not selected:
        raise ValueError("no ensemble position matched the disagreement selection criteria")
    selected_keys = {candidate.normalized_sfen for candidate in selected}
    filtered_games: list[GameRecord] = []
    for game in ensemble_games:
        samples = tuple(
            sample for sample in game.samples if normalized_sfen(sample.sfen) in selected_keys
        )
        if samples:
            filtered_games.append(replace(game, samples=samples))

    position_reports = tuple(
        SelectedPosition(
            normalized_sfen=candidate.normalized_sfen,
            output_sfen=candidate.output_sfen,
            selection_rank=rank,
            reasons=tuple(reason.value for reason in candidate.reasons),
            family_top1=candidate.family_top1,
            family_values=candidate.family_values,
            family_unique_best_moves=candidate.family_unique_best_moves,
            family_normalized_jensen_shannon=candidate.family_normalized_jensen_shannon,
            manifest_normalized_jensen_shannon=candidate.manifest_normalized_jensen_shannon,
            reported_winning_mates=candidate.reported_winning_mates,
            trajectory=candidate.trajectory,
            recommended_nodes=candidate.recommended_nodes,
            teacher_stories=candidate.teacher_stories,
        )
        for rank, candidate in enumerate(selected, start=1)
    )
    report = DisagreementSelectionReport(
        schema=DISAGREEMENT_REPORT_SCHEMA,
        top1_definition=(
            "lexicographically deterministic argmax of each normalized policy; duplicate "
            "positions are averaged within a teacher and correlated teachers are averaged "
            "within one family before comparison"
        ),
        family_aggregation=(
            "each family contributes one equally weighted policy/value regardless of how many "
            "correlated teacher replays belong to that family"
        ),
        value_transform_semantics=(
            "Value-direction selection ignores legacy replay teacher_value. Exact CP root "
            "scores are converted posthoc as tanh(cp / configured_scale), default 1200; this "
            "matches the centered form of sigmoid(cp/C) when C=600. Families without an exact "
            "CP root score are recorded with null value and do not invent a direction vote. "
            "When one engine repeats a root move in MultiPV, the smallest-rank exact CP is the "
            "representative value while every original typed variation remains in the story."
        ),
        ensemble_replay=ensemble_artifact,
        ensemble_manifest=manifest_identity,
        teachers=tuple(teacher_reports),
        config=effective_config,
        base_normalized_positions=len(ensemble_locations),
        candidate_positions=len(candidates),
        selected_positions=len(selected),
        candidate_reason_counts=_reason_counts(candidates),
        selected_reason_counts=_reason_counts(selected),
        positions=position_reports,
        limitations=(
            "A positive USI score-mate entry is only a reported winning mate and is never "
            "labeled proven by this selector.",
            "Selection flags expensive-search candidates; it does not reject unusual moves, "
            "modify the ensemble target, or treat the manifest mixture as ground truth.",
            "Family labels are explicit experimental inputs because architecture labels alone "
            "cannot establish error independence; for example, Hao and tanuki-dr4 should share "
            "one tanuki family label.",
            "opening_family is a reproducible initial-position/first-24-move fingerprint, not "
            "a guessed human joseki name. Full teacher variation trees remain separated in "
            "teacher_stories and are never stitched into one synthetic PV.",
        ),
    )
    return DisagreementSelectionBuild(games=tuple(filtered_games), report=report)


def write_disagreement_replay(
    build: DisagreementSelectionBuild,
    output: Path,
    *,
    report: Path | None = None,
) -> dict[str, Any]:
    """Write a filtered deep-search input and hashed provenance without overwriting."""

    output_path = output.expanduser().resolve()
    report_path = (
        report.expanduser().resolve()
        if report is not None
        else output_path.with_suffix(output_path.suffix + ".disagreement.json")
    )
    if output_path == report_path:
        raise ValueError("disagreement replay and report paths must differ")
    for target in (output_path, report_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite disagreement artifact: {target}")
    append_games(output_path, build.games)
    payload: dict[str, Any] = build.report.to_dict()
    payload["output"] = {
        "replay": str(output_path),
        "replay_sha256": sha256_file(output_path),
        "replay_bytes": output_path.stat().st_size,
        "report": str(report_path),
    }
    payload["provenance_sha256"] = canonical_json_sha256(payload)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
