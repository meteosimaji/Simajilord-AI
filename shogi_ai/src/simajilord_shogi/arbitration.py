"""Family-aware arbitration of independently generated deep-search replay passes."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
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
)
from .ensemble import normalized_sfen
from .replay import append_games, load_games
from .research_context import GamePhase, TrajectoryContext, trajectory_contexts
from .tsume import TsumeSolver, TsumeVariation

ARBITRATION_REPORT_SCHEMA = "meteo-adaptive-deep-arbitration-v1"
ARBITRATION_TEACHER_SOURCE = "meteo-adaptive-deep-arbitration"
OPPONENT_EXPLOIT_EVIDENCE_SCHEMA = "meteo-opponent-exploit-evidence-v1"


@dataclass(frozen=True, slots=True)
class DepthPassInput:
    """One externally produced replay at a declared search-effort depth."""

    label: str
    replay: Path
    family: str
    depth: int
    validation_scope: str = "in_domain"


@dataclass(frozen=True, slots=True)
class ArbitrationConfig:
    """Exact mate proof and research-risk thresholds."""

    proof_max_plies: int = 3
    proof_node_limit: int = 1_000_000
    deep_cp_drop_threshold: float = 300.0
    deep_value_drop_threshold: float = 0.25
    require_complete_passes: bool = True
    family_weights: dict[str, float] = field(default_factory=dict)
    phase_family_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    centipawn_value_scale: float = 1_200.0
    family_value_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.proof_max_plies < 1 or self.proof_max_plies % 2 == 0:
            raise ValueError("proof_max_plies must be a positive odd number")
        if self.proof_node_limit < 1:
            raise ValueError("proof_node_limit must be positive")
        if (
            not math.isfinite(self.deep_cp_drop_threshold)
            or self.deep_cp_drop_threshold <= 0
        ):
            raise ValueError("deep_cp_drop_threshold must be finite and positive")
        if (
            not math.isfinite(self.deep_value_drop_threshold)
            or not 0 <= self.deep_value_drop_threshold <= 2
        ):
            raise ValueError("deep_value_drop_threshold must be finite and in [0, 2]")
        if not math.isfinite(self.centipawn_value_scale) or self.centipawn_value_scale <= 0:
            raise ValueError("centipawn_value_scale must be finite and positive")
        normalized_names: set[str] = set()
        for family, weight in self.family_weights.items():
            name = family.strip()
            if not name:
                raise ValueError("family weight name must not be empty")
            key = name.casefold()
            if key in normalized_names:
                raise ValueError(f"duplicate family weight name: {family}")
            normalized_names.add(key)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(f"family weight must be finite and positive: {family}")
        known_phases = {phase.value.casefold() for phase in GamePhase}
        normalized_phases: set[str] = set()
        for phase, family_weights in self.phase_family_weights.items():
            phase_name = phase.strip()
            phase_key = phase_name.casefold()
            if not phase_name or phase_key not in known_phases:
                raise ValueError(f"unknown phase family-weight phase: {phase!r}")
            if phase_key in normalized_phases:
                raise ValueError(f"duplicate phase family-weight phase: {phase}")
            normalized_phases.add(phase_key)
            if not family_weights:
                raise ValueError(f"phase family weights must not be empty: {phase}")
            phase_families: set[str] = set()
            for family, weight in family_weights.items():
                family_name = family.strip()
                family_key = family_name.casefold()
                if not family_name:
                    raise ValueError("phase family weight name must not be empty")
                if family_key in phase_families:
                    raise ValueError(
                        f"duplicate phase family weight name in {phase}: {family}"
                    )
                phase_families.add(family_key)
                if not math.isfinite(weight) or weight <= 0:
                    raise ValueError(
                        f"phase family weight must be finite and positive: "
                        f"{phase}:{family}"
                    )
        normalized_scale_names: set[str] = set()
        for family, scale in self.family_value_scales.items():
            name = family.strip()
            if not name:
                raise ValueError("family value-scale name must not be empty")
            key = name.casefold()
            if key in normalized_scale_names:
                raise ValueError(f"duplicate family value-scale name: {family}")
            normalized_scale_names.add(key)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"family value scale must be finite and positive: {family}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DepthPassReport:
    label: str
    family: str
    depth: int
    validation_scope: str
    artifact: InputArtifact
    normalized_positions: int
    reported_teacher_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    move: str
    probability: float


@dataclass(frozen=True, slots=True)
class DepthReportedWinningMate:
    pass_label: str
    family: str
    declared_depth: int
    validation_scope: str
    teacher_source: str
    move: str
    bound: str
    mate_plies: int | None
    mate_unknown_sign: int | None


@dataclass(frozen=True, slots=True)
class FamilyDepthPoint:
    depth: int
    pass_labels: tuple[str, ...]
    validation_scopes: tuple[str, ...]
    reported_search_depths: tuple[int, ...]
    top1: str
    raw_exact_root_cp: float | None
    posthoc_value: float | None
    legacy_teacher_value: float
    centipawn_value_scale: float
    policy: tuple[PolicyTarget, ...]
    exact_scores_cp: dict[str, float]
    reported_winning_mates: tuple[DepthReportedWinningMate, ...]


@dataclass(frozen=True, slots=True)
class FamilyTop1Stability:
    family: str
    points: tuple[FamilyDepthPoint, ...]
    stable_across_depths: bool
    top1_transitions: int
    deepest_stable_run: int


@dataclass(frozen=True, slots=True)
class MoveRobustness:
    move: str
    top1_families: tuple[str, ...]
    top1_depths: tuple[int, ...]
    validation_scopes: tuple[str, ...]
    reproduced_across_families: bool
    reproduced_at_multiple_depths: bool
    reproduced_outside_in_domain: bool


@dataclass(frozen=True, slots=True)
class ArbitrationRiskFlag:
    """Advisory evidence; flags never remove a move from the union target."""

    code: str
    move: str | None
    families: tuple[str, ...]
    depths: tuple[int, ...]
    validation_scopes: tuple[str, ...]
    evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class OpponentExploitEvidence:
    """Held-out match evidence, kept separate from the objective teacher label."""

    normalized_sfen: str
    move: str
    opponent_family: str
    expected_score_gain: float
    games: int
    validation_scope: str
    unknown_opponent_family: bool


@dataclass(frozen=True, slots=True)
class OpponentEvidenceReport:
    artifact: FileIdentity
    schema: str
    total_records: int
    used_records: int


@dataclass(frozen=True, slots=True)
class MovePracticalTradeoff:
    """Two-axis evidence; neither axis silently replaces the other."""

    move: str
    objective_deep_regret_cp: float | None
    opponent_exploit_gain: float | None
    opponent_families: tuple[str, ...]
    validation_scopes: tuple[str, ...]
    unknown_opponent_family_reproduced: bool
    pareto_candidate: bool | None
    core_training_target: bool
    usage_classification: str


@dataclass(frozen=True, slots=True)
class PassPVStory:
    """One pass's intact PV and its temporal neighbors in the same replay chunk."""

    pass_label: str
    family: str
    declared_depth: int
    validation_scope: str
    trajectory: TrajectoryContext
    top_move: str
    raw_exact_root_cp: float | None
    posthoc_value: float | None
    legacy_teacher_value: float
    previous_top_move: str | None
    previous_posthoc_value: float | None
    top_move_matches_previous: bool | None
    posthoc_value_delta_from_previous: float | None
    next_top_move: str | None
    next_posthoc_value: float | None
    top_move_matches_next: bool | None
    posthoc_value_delta_to_next: float | None
    variations: tuple[TeacherVariation, ...]


@dataclass(frozen=True, slots=True)
class StrategyBranch:
    """A family-specific terminal concept; branches are never spliced together."""

    family: str
    declared_depth: int
    top_move: str
    pass_stories: tuple[PassPVStory, ...]


@dataclass(frozen=True, slots=True)
class MateProofReport:
    """Result from internally enumerating standard-tsume checking continuations."""

    status: str
    max_plies: int
    node_limit: int
    nodes: int
    proven_moves: tuple[str, ...]
    solutions: tuple[TsumeVariation, ...]


@dataclass(frozen=True, slots=True)
class ArbitrationPositionReport:
    normalized_sfen: str
    output_sfen: str
    trajectory: TrajectoryContext
    contributing_families: tuple[str, ...]
    effective_family_weights: dict[str, float]
    top1_stability: tuple[FamilyTop1Stability, ...]
    deepest_family_top1: dict[str, str]
    unresolved_top1_disagreement: bool
    top1_union: tuple[str, ...]
    final_target_kind: str
    final_policy: tuple[PolicyTarget, ...]
    final_value: float | None
    reported_winning_mates: tuple[DepthReportedWinningMate, ...]
    mate_proof: MateProofReport
    move_robustness: tuple[MoveRobustness, ...]
    practical_tradeoffs: tuple[MovePracticalTradeoff, ...]
    strategy_branches: tuple[StrategyBranch, ...]
    risk_flags: tuple[ArbitrationRiskFlag, ...]


@dataclass(frozen=True, slots=True)
class ArbitrationReport:
    schema: str
    base_replay: InputArtifact
    depth_passes: tuple[DepthPassReport, ...]
    opponent_evidence: OpponentEvidenceReport | None
    config: ArbitrationConfig
    family_aggregation: str
    mate_precedence: str
    value_transform_semantics: str
    risk_flags_are_advisory: bool
    objective_strategy_separation: str
    base_normalized_positions: int
    output_normalized_positions: int
    unresolved_top1_positions: int
    internally_proven_mate_positions: int
    risk_flag_counts: dict[str, int]
    positions: tuple[ArbitrationPositionReport, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class ArbitrationBuild:
    games: tuple[GameRecord, ...]
    report: ArbitrationReport


@dataclass(frozen=True, slots=True)
class _PassObservation:
    spec: DepthPassInput
    sample: PositionSample
    policy: dict[str, float]
    raw_exact_root_cp: float | None
    posthoc_value: float | None
    legacy_teacher_value: float
    centipawn_value_scale: float
    reported_winning_mates: tuple[DepthReportedWinningMate, ...]
    exact_scores_cp: dict[str, float]
    trajectory: TrajectoryContext
    previous_top_move: str | None
    previous_posthoc_value: float | None
    next_top_move: str | None
    next_posthoc_value: float | None


@dataclass(frozen=True, slots=True)
class _TerminalFamily:
    family: str
    point: FamilyDepthPoint
    policy: dict[str, float]
    weight: float


class _HistoryExactTsumeSolver(TsumeSolver):
    """Use ``solve_all`` without history-blind transposition memoization.

    ``TsumeSolver`` keys its performance cache by board hash and remaining
    depth. Repetition legality also depends on path history, so proof-bearing
    arbitration recomputes attacker nodes instead of reusing that cache.
    """

    def _solve_attacker(self, board: Board, remaining: int) -> TsumeVariation | None:
        for move in board.legal_moves():
            result = self._try_attack_move(board, move, remaining)
            if result is not None:
                return result
        return None


def _validate_passes(passes: Sequence[DepthPassInput]) -> tuple[DepthPassInput, ...]:
    if len(passes) < 2:
        raise ValueError("deep arbitration requires at least two depth-pass replays")
    labels: set[str] = set()
    paths: set[str] = set()
    family_names: dict[str, str] = {}
    validated: list[DepthPassInput] = []
    for supplied in passes:
        label = supplied.label.strip()
        family = supplied.family.strip()
        validation_scope = supplied.validation_scope.strip()
        if not label or not family or not validation_scope:
            raise ValueError("pass label, family, and validation_scope must not be empty")
        if supplied.depth < 1:
            raise ValueError(f"pass depth must be positive: {label}")
        label_key = label.casefold()
        if label_key in labels:
            raise ValueError(f"duplicate depth-pass label: {label}")
        labels.add(label_key)
        family_key = family.casefold()
        previous_family = family_names.setdefault(family_key, family)
        if previous_family != family:
            raise ValueError(
                f"family spelling must be consistent: {previous_family!r} versus {family!r}"
            )
        replay = supplied.replay.expanduser().resolve()
        path_key = str(replay).casefold()
        if path_key in paths:
            raise ValueError(f"duplicate depth-pass replay: {replay}")
        paths.add(path_key)
        validated.append(
            DepthPassInput(
                label=label,
                replay=replay,
                family=family,
                depth=supplied.depth,
                validation_scope=validation_scope,
            )
        )
    return tuple(validated)


def _normalized_policy(
    policy: dict[str, float], *, sfen: str, pass_label: str
) -> dict[str, float]:
    board = Board(sfen)
    if not board.is_valid():
        raise ValueError(f"invalid SFEN in pass {pass_label!r}: {sfen}")
    accumulated: dict[str, float] = defaultdict(float)
    for move_usi, raw_probability in policy.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(f"pass {pass_label!r} has invalid policy probability")
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(
                f"pass {pass_label!r} has illegal policy move {move_usi} at {sfen}"
            )
        if probability > 0:
            accumulated[move_usi] += probability
    total = math.fsum(accumulated.values())
    if total <= 0:
        raise ValueError(f"pass {pass_label!r} has empty policy at {sfen}")
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _average_policies(policies: Sequence[dict[str, float]]) -> dict[str, float]:
    if not policies:
        raise ValueError("at least one policy is required")
    accumulated: dict[str, float] = defaultdict(float)
    for policy in policies:
        for move, probability in policy.items():
            accumulated[move] += probability / len(policies)
    total = math.fsum(accumulated.values())
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _policy_top1(policy: Mapping[str, float]) -> str:
    return min(policy, key=lambda move: (-policy[move], move))


def _policy_target(policy: Mapping[str, float]) -> tuple[PolicyTarget, ...]:
    return tuple(
        PolicyTarget(move=move, probability=probability)
        for move, probability in sorted(policy.items(), key=lambda item: (-item[1], item[0]))
    )


def _reported_winning_mate(variation: TeacherVariation) -> bool:
    return variation.score_kind is TeacherScoreKind.MATE and (
        (variation.mate_plies is not None and variation.mate_plies > 0)
        or variation.mate_unknown_sign == 1
    )


def _value_scale_for_family(family: str, config: ArbitrationConfig) -> float:
    configured = {
        name.casefold(): scale for name, scale in config.family_value_scales.items()
    }
    return float(configured.get(family.casefold(), config.centipawn_value_scale))


def _pass_observation(
    spec: DepthPassInput,
    sample: PositionSample,
    config: ArbitrationConfig,
    trajectory: TrajectoryContext,
) -> _PassObservation:
    if sample.teacher_policy is None or sample.teacher_value is None:
        raise ValueError(
            f"pass {spec.label!r} has a position without both teacher policy and value"
        )
    legacy_value = float(sample.teacher_value)
    if not math.isfinite(legacy_value) or not -1 <= legacy_value <= 1:
        raise ValueError(f"pass {spec.label!r} has teacher value outside [-1, 1]")
    board = Board(sample.sfen)
    policy = _normalized_policy(
        sample.teacher_policy,
        sfen=sample.sfen,
        pass_label=spec.label,
    )
    reports: list[DepthReportedWinningMate] = []
    exact_scores: dict[str, float] = {}
    exact_ranks: dict[str, int] = {}
    for variation in sample.teacher_variations or ():
        move = Move.from_usi(variation.move)
        if not board.is_legal_move(move):
            raise ValueError(
                f"pass {spec.label!r} has illegal variation root {variation.move} "
                f"at {sample.sfen}"
            )
        if (
            variation.score_kind is TeacherScoreKind.CENTIPAWN
            and variation.bound is TeacherScoreBound.EXACT
            and variation.score_cp is not None
        ):
            existing_rank = exact_ranks.get(variation.move)
            if existing_rank is None or variation.rank < existing_rank:
                exact_scores[variation.move] = float(variation.score_cp)
                exact_ranks[variation.move] = variation.rank
        if _reported_winning_mate(variation):
            reports.append(
                DepthReportedWinningMate(
                    pass_label=spec.label,
                    family=spec.family,
                    declared_depth=spec.depth,
                    validation_scope=spec.validation_scope,
                    teacher_source=sample.teacher_source or "unknown",
                    move=variation.move,
                    bound=variation.bound.value,
                    mate_plies=variation.mate_plies,
                    mate_unknown_sign=variation.mate_unknown_sign,
                )
            )
    reports.sort(key=lambda report: (report.move, report.bound, report.pass_label.casefold()))
    preferred_moves = tuple(
        move
        for move in (sample.teacher_best_move, _policy_top1(policy))
        if move is not None
    )
    root_cp = next(
        (exact_scores[move] for move in preferred_moves if move in exact_scores),
        None,
    )
    if root_cp is None and exact_scores:
        root_move = min(exact_scores, key=lambda move: (exact_ranks[move], move))
        root_cp = exact_scores[root_move]
    value_scale = _value_scale_for_family(spec.family, config)
    return _PassObservation(
        spec=spec,
        sample=sample,
        policy=policy,
        raw_exact_root_cp=root_cp,
        posthoc_value=None if root_cp is None else math.tanh(root_cp / value_scale),
        legacy_teacher_value=legacy_value,
        centipawn_value_scale=value_scale,
        reported_winning_mates=tuple(reports),
        exact_scores_cp=dict(sorted(exact_scores.items())),
        trajectory=trajectory,
        previous_top_move=None,
        previous_posthoc_value=None,
        next_top_move=None,
        next_posthoc_value=None,
    )


def _load_pass(
    spec: DepthPassInput,
    config: ArbitrationConfig,
) -> tuple[dict[str, _PassObservation], DepthPassReport]:
    artifact = identify_input_artifact(spec.replay)
    observations: dict[str, _PassObservation] = {}
    sources: set[str] = set()
    games = tuple(load_games(Path(artifact.file.path)))
    contexts = trajectory_contexts(games)
    for game in games:
        for sample in game.samples:
            key = normalized_sfen(sample.sfen)
            if key in observations:
                raise ValueError(f"pass {spec.label!r} repeats normalized position {key}")
            observation = _pass_observation(spec, sample, config, contexts[key])
            observations[key] = observation
            sources.add(sample.teacher_source or "unknown")
    for key, observation in tuple(observations.items()):
        previous_key = observation.trajectory.previous_normalized_sfen
        previous = (
            observations.get(previous_key)
            if observation.trajectory.previous_is_consecutive and previous_key is not None
            else None
        )
        next_key = observation.trajectory.next_normalized_sfen
        following = (
            observations.get(next_key)
            if observation.trajectory.next_is_consecutive and next_key is not None
            else None
        )
        observations[key] = replace(
            observation,
            previous_top_move=(
                _policy_top1(previous.policy) if previous is not None else None
            ),
            previous_posthoc_value=(
                previous.posthoc_value if previous is not None else None
            ),
            next_top_move=(
                _policy_top1(following.policy) if following is not None else None
            ),
            next_posthoc_value=(
                following.posthoc_value if following is not None else None
            ),
        )
    return observations, DepthPassReport(
        label=spec.label,
        family=spec.family,
        depth=spec.depth,
        validation_scope=spec.validation_scope,
        artifact=artifact,
        normalized_positions=len(observations),
        reported_teacher_sources=tuple(sorted(sources)),
    )


def _family_depth_points(
    observations: Sequence[_PassObservation],
) -> tuple[FamilyTop1Stability, ...]:
    by_family_depth: dict[tuple[str, int], list[_PassObservation]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for observation in observations:
        family_key = observation.spec.family.casefold()
        display_names.setdefault(family_key, observation.spec.family)
        by_family_depth[(family_key, observation.spec.depth)].append(observation)
    by_family: dict[str, list[FamilyDepthPoint]] = defaultdict(list)
    for (family_key, depth), members in sorted(by_family_depth.items()):
        policy = _average_policies([member.policy for member in members])
        exact_moves = sorted({move for member in members for move in member.exact_scores_cp})
        exact_scores = {
            move: math.fsum(
                member.exact_scores_cp[move]
                for member in members
                if move in member.exact_scores_cp
            )
            / sum(move in member.exact_scores_cp for member in members)
            for move in exact_moves
        }
        reported_search_depths = tuple(
            sorted(
                sample_depth
                for member in members
                if (sample_depth := member.sample.teacher_depth) is not None
            )
        )
        raw_root_scores = [
            member.raw_exact_root_cp
            for member in members
            if member.raw_exact_root_cp is not None
        ]
        posthoc_values = [
            member.posthoc_value for member in members if member.posthoc_value is not None
        ]
        value_scales = {member.centipawn_value_scale for member in members}
        if len(value_scales) != 1:
            raise ValueError(
                f"family {display_names[family_key]!r} uses inconsistent value scales "
                f"at declared depth {depth}"
            )
        by_family[family_key].append(
            FamilyDepthPoint(
                depth=depth,
                pass_labels=tuple(sorted(member.spec.label for member in members)),
                validation_scopes=tuple(
                    sorted({member.spec.validation_scope for member in members})
                ),
                reported_search_depths=reported_search_depths,
                top1=_policy_top1(policy),
                raw_exact_root_cp=(
                    math.fsum(raw_root_scores) / len(raw_root_scores)
                    if raw_root_scores
                    else None
                ),
                posthoc_value=(
                    math.fsum(posthoc_values) / len(posthoc_values)
                    if posthoc_values
                    else None
                ),
                legacy_teacher_value=(
                    math.fsum(member.legacy_teacher_value for member in members) / len(members)
                ),
                centipawn_value_scale=next(iter(value_scales)),
                policy=_policy_target(policy),
                exact_scores_cp=dict(sorted(exact_scores.items())),
                reported_winning_mates=tuple(
                    sorted(
                        (
                            report
                            for member in members
                            for report in member.reported_winning_mates
                        ),
                        key=lambda report: (
                            report.move,
                            report.pass_label.casefold(),
                            report.bound,
                        ),
                    )
                ),
            )
        )
    stability: list[FamilyTop1Stability] = []
    for family_key in sorted(by_family):
        points = tuple(sorted(by_family[family_key], key=lambda point: point.depth))
        top1_moves = [point.top1 for point in points]
        transitions = sum(
            left != right for left, right in pairwise(top1_moves)
        )
        deepest_run = 1
        for move in reversed(top1_moves[:-1]):
            if move != top1_moves[-1]:
                break
            deepest_run += 1
        stability.append(
            FamilyTop1Stability(
                family=display_names[family_key],
                points=points,
                stable_across_depths=len(set(top1_moves)) == 1,
                top1_transitions=transitions,
                deepest_stable_run=deepest_run,
            )
        )
    return tuple(stability)


def _policy_from_target(target: Sequence[PolicyTarget]) -> dict[str, float]:
    return {item.move: item.probability for item in target}


def _family_weight_map(
    stability: Sequence[FamilyTop1Stability],
    configured: Mapping[str, float],
    phase_configured: Mapping[str, Mapping[str, float]],
    *,
    phase: str,
) -> dict[str, float]:
    available = {item.family.casefold(): item.family for item in stability}
    normalized_config = {name.casefold(): weight for name, weight in configured.items()}
    unknown = sorted(set(normalized_config) - set(available))
    if unknown:
        raise ValueError(f"family weights name unavailable families: {unknown}")
    phase_weights = next(
        (
            {name.casefold(): weight for name, weight in weights.items()}
            for configured_phase, weights in phase_configured.items()
            if configured_phase.casefold() == phase.casefold()
        ),
        {},
    )
    unknown_phase_families = sorted(set(phase_weights) - set(available))
    if unknown_phase_families:
        raise ValueError(
            f"phase family weights name unavailable families at {phase}: "
            f"{unknown_phase_families}"
        )
    raw = {
        display: float(normalized_config.get(key, 1.0))
        * float(phase_weights.get(key, 1.0))
        for key, display in sorted(available.items())
    }
    total = math.fsum(raw.values())
    return {family: raw[family] / total for family in sorted(raw)}


def _terminal_families(
    stability: Sequence[FamilyTop1Stability], weights: Mapping[str, float]
) -> tuple[_TerminalFamily, ...]:
    return tuple(
        _TerminalFamily(
            family=item.family,
            point=item.points[-1],
            policy=_policy_from_target(item.points[-1].policy),
            weight=weights[item.family],
        )
        for item in stability
    )


def _family_union_policy(families: Sequence[_TerminalFamily]) -> dict[str, float]:
    accumulated: dict[str, float] = defaultdict(float)
    for family in families:
        for move, probability in family.policy.items():
            accumulated[move] += family.weight * probability
    total = math.fsum(accumulated.values())
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"family union policy is not normalized: {total}")
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _prove_standard_tsume(board: Board, config: ArbitrationConfig) -> MateProofReport:
    solver = _HistoryExactTsumeSolver(node_limit=config.proof_node_limit)
    try:
        solution_set = solver.solve_all(board, config.proof_max_plies)
    except RuntimeError as error:
        if str(error) != "tsume node limit exceeded":
            raise
        return MateProofReport(
            status="node_limit_exceeded",
            max_plies=config.proof_max_plies,
            node_limit=config.proof_node_limit,
            nodes=solver.nodes,
            proven_moves=(),
            solutions=(),
        )
    solutions = tuple(sorted(solution_set.solutions, key=lambda solution: solution.move))
    return MateProofReport(
        status="proven" if solutions else "not_proven_within_bound",
        max_plies=config.proof_max_plies,
        node_limit=config.proof_node_limit,
        nodes=solution_set.nodes,
        proven_moves=tuple(solution.move for solution in solutions),
        solutions=solutions,
    )


def _move_robustness(
    observations: Sequence[_PassObservation],
) -> tuple[MoveRobustness, ...]:
    evidence: dict[str, dict[str, set[str] | set[int]]] = defaultdict(
        lambda: {"families": set(), "depths": set(), "scopes": set()}
    )
    for observation in observations:
        move = _policy_top1(observation.policy)
        cast(set[str], evidence[move]["families"]).add(observation.spec.family)
        cast(set[int], evidence[move]["depths"]).add(observation.spec.depth)
        cast(set[str], evidence[move]["scopes"]).add(observation.spec.validation_scope)
    reports: list[MoveRobustness] = []
    for move in sorted(evidence):
        families = tuple(sorted(cast(set[str], evidence[move]["families"])))
        depths = tuple(sorted(cast(set[int], evidence[move]["depths"])))
        scopes = tuple(sorted(cast(set[str], evidence[move]["scopes"])))
        reports.append(
            MoveRobustness(
                move=move,
                top1_families=families,
                top1_depths=depths,
                validation_scopes=scopes,
                reproduced_across_families=len(families) > 1,
                reproduced_at_multiple_depths=len(depths) > 1,
                reproduced_outside_in_domain=any(
                    scope.casefold() != "in_domain" for scope in scopes
                ),
            )
        )
    return tuple(reports)


def _json_object(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _load_opponent_evidence(
    path: Path | None,
    base_samples: Mapping[str, PositionSample],
) -> tuple[
    dict[str, tuple[OpponentExploitEvidence, ...]],
    OpponentEvidenceReport | None,
]:
    if path is None:
        return {}, None
    identity = identify_file(path)
    raw: object = json.loads(Path(identity.path).read_text(encoding="utf-8"))
    payload = _json_object(raw, context="opponent exploit evidence")
    if payload.get("schema") != OPPONENT_EXPLOIT_EVIDENCE_SCHEMA:
        raise ValueError(
            f"unsupported opponent evidence schema: {payload.get('schema')!r}"
        )
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("opponent exploit evidence records must be a JSON array")
    grouped: dict[str, list[OpponentExploitEvidence]] = defaultdict(list)
    used = 0
    for index, raw_record in enumerate(raw_records):
        record = _json_object(raw_record, context=f"opponent evidence record {index}")
        supplied_sfen = str(record.get("normalized_sfen", "")).strip()
        if len(supplied_sfen.split()) == 3:
            key = supplied_sfen
        elif len(supplied_sfen.split()) == 4:
            key = normalized_sfen(supplied_sfen)
        else:
            raise ValueError(
                f"opponent evidence record {index} has invalid normalized_sfen"
            )
        move = str(record.get("move", "")).strip()
        opponent_family = str(record.get("opponent_family", "")).strip()
        validation_scope = str(record.get("validation_scope", "")).strip()
        if not move or not opponent_family or not validation_scope:
            raise ValueError(
                f"opponent evidence record {index} requires move, opponent_family, "
                "and validation_scope"
            )
        gain = float(record.get("expected_score_gain", math.nan))
        games = int(record.get("games", 0))
        unknown_family = record.get("unknown_opponent_family")
        if not isinstance(unknown_family, bool):
            raise ValueError(
                f"opponent evidence record {index} requires boolean unknown_opponent_family"
            )
        if not math.isfinite(gain) or not -2 <= gain <= 2:
            raise ValueError(
                f"opponent evidence record {index} expected_score_gain must be in [-2, 2]"
            )
        if games < 1:
            raise ValueError(f"opponent evidence record {index} games must be positive")
        if key not in base_samples:
            continue
        board = Board(base_samples[key].sfen)
        parsed_move = Move.from_usi(move)
        if not board.is_legal_move(parsed_move):
            raise ValueError(
                f"opponent evidence record {index} has illegal move {move} at {key}"
            )
        grouped[key].append(
            OpponentExploitEvidence(
                normalized_sfen=key,
                move=move,
                opponent_family=opponent_family,
                expected_score_gain=gain,
                games=games,
                validation_scope=validation_scope,
                unknown_opponent_family=unknown_family,
            )
        )
        used += 1
    frozen = {
        key: tuple(
            sorted(
                records,
                key=lambda record: (
                    record.move,
                    record.opponent_family.casefold(),
                    record.validation_scope,
                    record.games,
                    record.expected_score_gain,
                ),
            )
        )
        for key, records in grouped.items()
    }
    return frozen, OpponentEvidenceReport(
        artifact=identity,
        schema=OPPONENT_EXPLOIT_EVIDENCE_SCHEMA,
        total_records=len(raw_records),
        used_records=used,
    )


def _objective_regret_cp(
    move: str,
    stability: Sequence[FamilyTop1Stability],
    weights: Mapping[str, float],
    proof: MateProofReport,
) -> float | None:
    if proof.proven_moves:
        return 0.0 if move in proof.proven_moves else None
    observations: list[tuple[float, float]] = []
    for item in stability:
        exact_scores = item.points[-1].exact_scores_cp
        if move not in exact_scores or not exact_scores:
            continue
        regret = max(0.0, max(exact_scores.values()) - exact_scores[move])
        observations.append((weights[item.family], regret))
    total_weight = math.fsum(weight for weight, _regret in observations)
    if total_weight <= 0:
        return None
    return math.fsum(weight * regret for weight, regret in observations) / total_weight


def _practical_tradeoffs(
    final_policy: Mapping[str, float],
    stability: Sequence[FamilyTop1Stability],
    weights: Mapping[str, float],
    proof: MateProofReport,
    evidence: Sequence[OpponentExploitEvidence],
) -> tuple[MovePracticalTradeoff, ...]:
    evidence_by_move: dict[str, list[OpponentExploitEvidence]] = defaultdict(list)
    for record in evidence:
        evidence_by_move[record.move].append(record)
    moves = sorted(set(final_policy) | set(evidence_by_move))
    rows: list[dict[str, object]] = []
    for move in moves:
        records = evidence_by_move[move]
        total_games = sum(record.games for record in records)
        gain = (
            math.fsum(record.expected_score_gain * record.games for record in records)
            / total_games
            if total_games
            else None
        )
        unknown_reproduced = any(record.unknown_opponent_family for record in records)
        if proof.proven_moves and move not in proof.proven_moves:
            usage = "dominated_by_internally_proven_mate"
        elif gain is not None and gain > 0 and unknown_reproduced:
            usage = "general_practical_strength_candidate"
        elif gain is not None and gain > 0:
            usage = "targeted_match_strategy_candidate"
        elif move in final_policy:
            usage = "opponent_independent_core_target"
        else:
            usage = "insufficient_positive_match_evidence"
        rows.append(
            {
                "move": move,
                "regret": _objective_regret_cp(move, stability, weights, proof),
                "gain": gain,
                "families": tuple(sorted({record.opponent_family for record in records})),
                "scopes": tuple(sorted({record.validation_scope for record in records})),
                "unknown": unknown_reproduced,
                "core": move in final_policy,
                "usage": usage,
            }
        )
    result: list[MovePracticalTradeoff] = []
    for row in rows:
        regret = cast(float | None, row["regret"])
        gain = cast(float | None, row["gain"])
        pareto: bool | None = None
        if regret is not None and gain is not None:
            pareto = True
            for competitor in rows:
                competitor_regret = cast(float | None, competitor["regret"])
                competitor_gain = cast(float | None, competitor["gain"])
                if competitor_regret is None or competitor_gain is None:
                    continue
                weakly_better = competitor_regret <= regret and competitor_gain >= gain
                strictly_better = competitor_regret < regret or competitor_gain > gain
                if weakly_better and strictly_better:
                    pareto = False
                    break
        if proof.proven_moves and cast(str, row["move"]) not in proof.proven_moves:
            pareto = False
        result.append(
            MovePracticalTradeoff(
                move=cast(str, row["move"]),
                objective_deep_regret_cp=regret,
                opponent_exploit_gain=gain,
                opponent_families=cast(tuple[str, ...], row["families"]),
                validation_scopes=cast(tuple[str, ...], row["scopes"]),
                unknown_opponent_family_reproduced=cast(bool, row["unknown"]),
                pareto_candidate=pareto,
                core_training_target=cast(bool, row["core"]),
                usage_classification=cast(str, row["usage"]),
            )
        )
    return tuple(result)


def _strategy_branches(
    stability: Sequence[FamilyTop1Stability],
    observations: Sequence[_PassObservation],
) -> tuple[StrategyBranch, ...]:
    branches: list[StrategyBranch] = []
    for item in stability:
        terminal = item.points[-1]
        members = sorted(
            (
                observation
                for observation in observations
                if observation.spec.family.casefold() == item.family.casefold()
            ),
            key=lambda observation: (
                observation.spec.depth,
                observation.spec.label.casefold(),
            ),
        )
        stories = tuple(
            PassPVStory(
                pass_label=observation.spec.label,
                family=observation.spec.family,
                declared_depth=observation.spec.depth,
                validation_scope=observation.spec.validation_scope,
                trajectory=observation.trajectory,
                top_move=_policy_top1(observation.policy),
                raw_exact_root_cp=observation.raw_exact_root_cp,
                posthoc_value=observation.posthoc_value,
                legacy_teacher_value=observation.legacy_teacher_value,
                previous_top_move=observation.previous_top_move,
                previous_posthoc_value=observation.previous_posthoc_value,
                top_move_matches_previous=(
                    None
                    if observation.previous_top_move is None
                    else observation.previous_top_move == _policy_top1(observation.policy)
                ),
                posthoc_value_delta_from_previous=(
                    None
                    if observation.posthoc_value is None
                    or observation.previous_posthoc_value is None
                    else observation.posthoc_value - observation.previous_posthoc_value
                ),
                next_top_move=observation.next_top_move,
                next_posthoc_value=observation.next_posthoc_value,
                top_move_matches_next=(
                    None
                    if observation.next_top_move is None
                    else observation.next_top_move == _policy_top1(observation.policy)
                ),
                posthoc_value_delta_to_next=(
                    None
                    if observation.posthoc_value is None
                    or observation.next_posthoc_value is None
                    else observation.next_posthoc_value - observation.posthoc_value
                ),
                variations=observation.sample.teacher_variations or (),
            )
            for observation in members
        )
        branches.append(
            StrategyBranch(
                family=item.family,
                declared_depth=terminal.depth,
                top_move=terminal.top1,
                pass_stories=stories,
            )
        )
    return tuple(branches)


def _risk_flags(
    stability: Sequence[FamilyTop1Stability],
    observations: Sequence[_PassObservation],
    config: ArbitrationConfig,
) -> tuple[ArbitrationRiskFlag, ...]:
    flags: list[ArbitrationRiskFlag] = []
    terminal_counts = Counter(item.points[-1].top1 for item in stability)
    for item in stability:
        points = item.points
        scopes = tuple(sorted({scope for point in points for scope in point.validation_scopes}))
        if points[0].top1 != points[-1].top1:
            flags.append(
                ArbitrationRiskFlag(
                    code="shallow_deep_top1_reversal",
                    move=points[0].top1,
                    families=(item.family,),
                    depths=(points[0].depth, points[-1].depth),
                    validation_scopes=scopes,
                    evidence={"deep_top1": points[-1].top1},
                )
            )
        collapsed_moves = sorted(
            {point.top1 for point in points[:-1] if point.top1 != points[-1].top1}
        )
        for move in collapsed_moves:
            observed_depths = tuple(point.depth for point in points if point.top1 == move)
            flags.append(
                ArbitrationRiskFlag(
                    code="search_effort_move_collapse",
                    move=move,
                    families=(item.family,),
                    depths=observed_depths,
                    validation_scopes=scopes,
                    evidence={
                        "deepest_depth": points[-1].depth,
                        "deepest_top1": points[-1].top1,
                    },
                )
            )
        terminal_move = points[-1].top1
        if len(stability) > 1 and terminal_counts[terminal_move] == 1:
            flags.append(
                ArbitrationRiskFlag(
                    code="family_unique_terminal_move",
                    move=terminal_move,
                    families=(item.family,),
                    depths=(points[-1].depth,),
                    validation_scopes=points[-1].validation_scopes,
                    evidence={
                        "terminal_family_count": len(stability),
                        "supporting_family_count": 1,
                    },
                )
            )
        if (
            len(points) > 1
            and points[0].posthoc_value is not None
            and points[-1].posthoc_value is not None
            and points[0].posthoc_value - points[-1].posthoc_value
            >= config.deep_value_drop_threshold
        ):
            shallow_value = points[0].posthoc_value
            deep_value = points[-1].posthoc_value
            flags.append(
                ArbitrationRiskFlag(
                    code="deep_position_value_drop",
                    move=points[0].top1,
                    families=(item.family,),
                    depths=(points[0].depth, points[-1].depth),
                    validation_scopes=scopes,
                    evidence={
                        "shallow_posthoc_value": shallow_value,
                        "deep_posthoc_value": deep_value,
                        "drop": shallow_value - deep_value,
                        "transform": "tanh(raw_exact_root_cp / centipawn_value_scale)",
                    },
                )
            )
        score_points: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for point in points:
            for move, score in point.exact_scores_cp.items():
                score_points[move].append((point.depth, score))
        for move, values in sorted(score_points.items()):
            if len(values) < 2:
                continue
            shallow_depth, shallow_score = values[0]
            deep_depth, deep_score = values[-1]
            score_drop = shallow_score - deep_score
            if score_drop >= config.deep_cp_drop_threshold:
                flags.append(
                    ArbitrationRiskFlag(
                        code="deep_exact_cp_drop",
                        move=move,
                        families=(item.family,),
                        depths=(shallow_depth, deep_depth),
                        validation_scopes=scopes,
                        evidence={
                            "shallow_score_cp": shallow_score,
                            "deep_score_cp": deep_score,
                            "drop_cp": score_drop,
                        },
                    )
                )
        if len(points) > 1:
            earlier_reports = {
                report.move for point in points[:-1] for report in point.reported_winning_mates
            }
            deepest_reports = {report.move for report in points[-1].reported_winning_mates}
            for move in sorted(earlier_reports - deepest_reports):
                flags.append(
                    ArbitrationRiskFlag(
                        code="reported_mate_not_reproduced_at_deepest",
                        move=move,
                        families=(item.family,),
                        depths=(points[0].depth, points[-1].depth),
                        validation_scopes=scopes,
                        evidence={"reported_only": True},
                    )
                )

    # A second engine/family is an experimental axis even when declared depths match.
    top1_by_observation = defaultdict(set)
    for observation in observations:
        top1_by_observation[_policy_top1(observation.policy)].add(observation.spec.family)
    for move, families in sorted(top1_by_observation.items()):
        observed_family_count = len(
            {observation.spec.family for observation in observations}
        )
        if observed_family_count > 1 and len(families) == 1:
            family = min(families)
            matching = [
                observation
                for observation in observations
                if _policy_top1(observation.policy) == move
            ]
            flags.append(
                ArbitrationRiskFlag(
                    code="cross_teacher_move_fragility",
                    move=move,
                    families=(family,),
                    depths=tuple(sorted({observation.spec.depth for observation in matching})),
                    validation_scopes=tuple(
                        sorted({observation.spec.validation_scope for observation in matching})
                    ),
                    evidence={"supporting_family_count": 1},
                )
            )
    return tuple(
        sorted(
            flags,
            key=lambda flag: (
                flag.code,
                flag.move or "",
                flag.families,
                flag.depths,
            ),
        )
    )


def _arbitrate_position(
    sample: PositionSample,
    trajectory: TrajectoryContext,
    observations: Sequence[_PassObservation],
    config: ArbitrationConfig,
    opponent_evidence: Sequence[OpponentExploitEvidence],
) -> tuple[PositionSample, ArbitrationPositionReport]:
    stability = _family_depth_points(observations)
    weights = _family_weight_map(
        stability,
        config.family_weights,
        config.phase_family_weights,
        phase=trajectory.phase,
    )
    terminals = _terminal_families(stability, weights)
    family_policy = _family_union_policy(terminals)
    deepest_top1 = {family.family: family.point.top1 for family in terminals}
    top1_union = tuple(sorted(set(deepest_top1.values())))
    unresolved = len(top1_union) > 1
    value_families = [
        family for family in terminals if family.point.posthoc_value is not None
    ]
    value_weight = math.fsum(family.weight for family in value_families)
    family_value = (
        math.fsum(
            family.weight * cast(float, family.point.posthoc_value)
            for family in value_families
        )
        / value_weight
        if value_families
        else None
    )
    reported_mates = tuple(
        sorted(
            (
                reported_mate
                for observation in observations
                for reported_mate in observation.reported_winning_mates
            ),
            key=lambda report: (
                report.family.casefold(),
                report.declared_depth,
                report.pass_label.casefold(),
                report.move,
            ),
        )
    )
    proof = _prove_standard_tsume(Board(sample.sfen), config)
    final_value: float | None
    if proof.proven_moves:
        final_policy = {
            move: 1.0 / len(proof.proven_moves) for move in sorted(proof.proven_moves)
        }
        final_value = 1.0
        final_kind = "internally_proven_mate_uniform"
        acceptable_moves = set(proof.proven_moves)
        teacher_best = proof.proven_moves[0] if len(proof.proven_moves) == 1 else None
    else:
        final_policy = family_policy
        final_value = family_value
        final_kind = "family_union_soft_target"
        acceptable_moves = set(top1_union)
        teacher_best = top1_union[0] if not unresolved else None
    actor_best = sample.actor_best_move
    if actor_best is None and sample.policy:
        actor_best = _policy_top1(sample.policy)
    output = replace(
        sample,
        teacher_policy=final_policy,
        teacher_value=final_value,
        policy_reversal=actor_best is not None and actor_best not in acceptable_moves,
        teacher_best_move=teacher_best,
        teacher_regret=None,
        teacher_regret_is_lower_bound=False,
        teacher_nodes=None,
        teacher_depth_ratio=None,
        teacher_time_ms=None,
        teacher_nps=None,
        teacher_depth=None,
        teacher_source=ARBITRATION_TEACHER_SOURCE,
        teacher_context="adaptive-deep-arbitration",
        teacher_variations=None,
        teacher_policy_temperature=None,
        teacher_value_scale=None,
    )
    risks = _risk_flags(stability, observations, config)
    report = ArbitrationPositionReport(
        normalized_sfen=normalized_sfen(sample.sfen),
        output_sfen=sample.sfen,
        trajectory=trajectory,
        contributing_families=tuple(family.family for family in terminals),
        effective_family_weights=dict(sorted(weights.items())),
        top1_stability=stability,
        deepest_family_top1=dict(sorted(deepest_top1.items())),
        unresolved_top1_disagreement=unresolved,
        top1_union=top1_union,
        final_target_kind=final_kind,
        final_policy=_policy_target(final_policy),
        final_value=final_value,
        reported_winning_mates=reported_mates,
        mate_proof=proof,
        move_robustness=_move_robustness(observations),
        practical_tradeoffs=_practical_tradeoffs(
            final_policy,
            stability,
            weights,
            proof,
            opponent_evidence,
        ),
        strategy_branches=_strategy_branches(stability, observations),
        risk_flags=risks,
    )
    return output, report


def build_depth_arbitration(
    base_replay: Path,
    depth_passes: Sequence[DepthPassInput],
    *,
    config: ArbitrationConfig | None = None,
    opponent_evidence: Path | None = None,
) -> ArbitrationBuild:
    """Merge deep passes while preserving proof and unresolved policy uncertainty."""

    effective_config = config or ArbitrationConfig()
    validated_passes = _validate_passes(depth_passes)
    available_families = {depth_pass.family.casefold() for depth_pass in validated_passes}
    unknown_scale_families = sorted(
        {
            family.casefold()
            for family in effective_config.family_value_scales
            if family.casefold() not in available_families
        }
    )
    if unknown_scale_families:
        raise ValueError(
            f"family value scales name unavailable families: {unknown_scale_families}"
        )
    unknown_phase_weight_families = sorted(
        {
            family.casefold()
            for weights in effective_config.phase_family_weights.values()
            for family in weights
            if family.casefold() not in available_families
        }
    )
    if unknown_phase_weight_families:
        raise ValueError(
            "phase family weights name unavailable families: "
            f"{unknown_phase_weight_families}"
        )
    base_artifact = identify_input_artifact(base_replay)
    base_games = tuple(load_games(Path(base_artifact.file.path)))
    base_contexts = trajectory_contexts(base_games)
    base_samples: dict[str, PositionSample] = {}
    for game in base_games:
        for sample in game.samples:
            key = normalized_sfen(sample.sfen)
            if key in base_samples:
                raise ValueError(f"base replay repeats normalized position {key}")
            base_samples[key] = sample
    if not base_samples:
        raise ValueError("base replay has no positions")
    evidence_by_position, evidence_report = _load_opponent_evidence(
        opponent_evidence,
        base_samples,
    )

    loaded_passes: list[dict[str, _PassObservation]] = []
    pass_reports: list[DepthPassReport] = []
    for depth_pass in validated_passes:
        pass_observations, pass_report = _load_pass(depth_pass, effective_config)
        extra = sorted(set(pass_observations) - set(base_samples))[:3]
        if extra:
            raise ValueError(
                f"pass {depth_pass.label!r} contains positions outside base replay: {extra}"
            )
        if (
            effective_config.require_complete_passes
            and set(pass_observations) != set(base_samples)
        ):
            missing = sorted(set(base_samples) - set(pass_observations))[:3]
            raise ValueError(
                f"pass {depth_pass.label!r} is incomplete for base replay; examples={missing}"
            )
        loaded_passes.append(pass_observations)
        pass_reports.append(pass_report)

    arbitrated_samples: dict[str, PositionSample] = {}
    position_reports: list[ArbitrationPositionReport] = []
    for key in sorted(base_samples):
        position_observations = [loaded[key] for loaded in loaded_passes if key in loaded]
        if not position_observations:
            raise ValueError(f"no depth pass contains base position {key}")
        output, position_report = _arbitrate_position(
            base_samples[key],
            base_contexts[key],
            position_observations,
            effective_config,
            evidence_by_position.get(key, ()),
        )
        arbitrated_samples[key] = output
        position_reports.append(position_report)

    output_games = tuple(
        replace(
            game,
            samples=tuple(
                arbitrated_samples[normalized_sfen(sample.sfen)] for sample in game.samples
            ),
        )
        for game in base_games
    )
    risk_counts = Counter(
        flag.code for position in position_reports for flag in position.risk_flags
    )
    final_report = ArbitrationReport(
        schema=ARBITRATION_REPORT_SCHEMA,
        base_replay=base_artifact,
        depth_passes=tuple(pass_reports),
        opponent_evidence=evidence_report,
        config=effective_config,
        family_aggregation=(
            "passes at one family/depth are averaged; each family then contributes only its "
            "deepest available point once, so more correlated engines or depth passes do not "
            "create extra votes. Optional phase-family weights are strictly positive "
            "multipliers on the global family priors, never a hard engine switch; they are "
            "renormalized into one teacher target for one Meteo model"
        ),
        mate_precedence=(
            "TsumeSolver.solve_all traversal with history-exact recomputation enumerates every "
            "legal standard-tsume root within the configured bound; a non-empty proven set "
            "replaces every non-mate target and all proven root moves receive equal mass"
        ),
        value_transform_semantics=(
            "Final teacher_value never uses legacy replay teacher_value. For an exact CP root "
            "score it uses tanh(cp / configured_scale), with default scale 1200 because "
            "sigmoid(cp / C) has the equivalent centered value 2*sigmoid(cp/C)-1 = "
            "tanh(cp/(2*C)); therefore published C=600 corresponds to Meteo scale=1200. "
            "Per-family posthoc scales are recorded and may override the default. If no exact "
            "CP root score exists and no internal mate is proven, teacher_value remains unset. "
            "Repeated MultiPV roots use the smallest-rank exact CP as the representative score "
            "without deleting any typed PV from strategy_branches."
        ),
        risk_flags_are_advisory=True,
        objective_strategy_separation=(
            "The output teacher_policy is opponent-independent and is never changed by match "
            "evidence. objective_deep_regret_cp and opponent_exploit_gain remain separate axes. "
            "Positive gain reproduced for an explicitly unknown opponent family is classified "
            "as a general practical-strength candidate; known-family-only gain is classified "
            "for targeted match strategy/repertoire use. Pareto status is reported without a "
            "hard exclusion threshold."
        ),
        base_normalized_positions=len(base_samples),
        output_normalized_positions=len(arbitrated_samples),
        unresolved_top1_positions=sum(
            position.unresolved_top1_disagreement for position in position_reports
        ),
        internally_proven_mate_positions=sum(
            bool(position.mate_proof.proven_moves) for position in position_reports
        ),
        risk_flag_counts=dict(sorted(risk_counts.items())),
        positions=tuple(position_reports),
        limitations=(
            "USI score-mate reports remain reported_winning_mates even when exact or repeated; "
            "only the internal legal all-defense standard-tsume proof receives mate precedence.",
            "The internal proof covers standard tsume, where every attacker move checks. A quiet "
            "game-theoretic forced mate outside that definition is not labeled proven.",
            "deep_exact_cp_drop compares exact root scores for the same MultiPV move across "
            "declared passes; replay data does not contain per-ply evaluation traces inside a PV.",
            "Research-risk flags are provenance, not filters. Unusual moves remain in the union "
            "soft target unless an internally proven mate set takes precedence; validation_scope "
            "records whether independent or unknown-split reproduction was actually supplied.",
            "opening_family is a reproducible opening fingerprint rather than a guessed name. "
            "strategy_branches retain terminal family PVs and same-pass temporal neighbors as "
            "separate stories; arbitration never splices different teachers' PVs into one line.",
        ),
    )
    return ArbitrationBuild(games=output_games, report=final_report)


def write_depth_arbitration(
    build: ArbitrationBuild,
    output: Path,
    *,
    report: Path | None = None,
) -> dict[str, Any]:
    """Write an arbitration replay and self-authenticating report without overwriting."""

    output_path = output.expanduser().resolve()
    report_path = (
        report.expanduser().resolve()
        if report is not None
        else output_path.with_suffix(output_path.suffix + ".arbitration.json")
    )
    if output_path == report_path:
        raise ValueError("arbitration replay and report paths must differ")
    for target in (output_path, report_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite arbitration artifact: {target}")
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
