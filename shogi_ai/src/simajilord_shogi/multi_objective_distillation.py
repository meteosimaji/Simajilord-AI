"""Fail-closed targets for search-value policy and autonomous distillation.

The ordinary AlphaZero target records root visit counts.  That distribution is
useful, but it also contains the search policy's exploration bias.  Meteo keeps
it and adds a second target derived from the searched value of *every* legal
root move.  A target is accepted only when the move/value/visit maps exactly
cover the legal move set and every move received at least one real visit.

Internally proven mates are deliberately outside that soft target.  A mate set
is a set-valued constraint (put probability mass on any proven winning move),
not a reason to invent an ordering between equivalent mating moves or to let a
saturated softmax dominate ordinary-policy learning.

The curriculum objects in this module are evidence gates, not a scheduler.
Generations, elapsed time, or a desired phase can never advance the phase on
their own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from rsshogi.core import Board

from .adjudication import adjudicate_board
from .domain import Termination


class SearchPolicyKind(StrEnum):
    """Whether a root supplies an ordinary distribution or a mate-set target."""

    ALL_LEGAL_VALUE_DISTRIBUTION = "all_legal_value_distribution"
    PROVEN_MATE_SET = "proven_mate_set"


@dataclass(frozen=True, slots=True)
class SearchValuePolicyTarget:
    """Lossless all-legal root-search evidence and its policy interpretation."""

    kind: SearchPolicyKind
    legal_moves: tuple[str, ...]
    move_values: dict[str, float]
    move_visits: dict[str, int]
    policy: dict[str, float]
    temperature: float
    proven_mate_moves: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateUnion:
    """Deterministic candidate union without silently losing proposer identity."""

    proposer_moves: tuple[tuple[str, tuple[str, ...]], ...]
    moves: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LinkedNegamaxConstraint:
    """A rank/value consistency check for an actual parent -> child edge."""

    parent_move: str
    parent_q: float
    child_value: float
    absolute_residual: float
    tolerance: float
    satisfied: bool


@dataclass(frozen=True, slots=True)
class DepthRankConstraint:
    """Cross-budget ranking stability over the same position and candidate set."""

    budgets: tuple[int, ...]
    candidates: tuple[str, ...]
    deepest_best_moves: tuple[str, ...]
    top_move_stable: bool
    pairwise_reversals: int
    pairwise_comparisons: int
    reversal_fraction: float


class DistillationPhase(StrEnum):
    """External-teacher dependence stage selected only from held-out evidence."""

    BOOTSTRAP = "bootstrap"
    TRANSITION = "transition"
    AUTONOMOUS = "autonomous"


@dataclass(frozen=True, slots=True)
class CurriculumEvidence:
    """Independent evidence needed before reducing external-teacher weight."""

    complete_games: int
    heldout_positions: int
    arena_pairs: int
    complete_game_fraction: float
    self_reanalysis_coverage: float
    self_reanalysis_policy_js: float
    external_teacher_worst_regret: float
    external_teacher_policy_js: float
    arena_score_lower_bound: float
    no_heldout_regression: bool
    mate_holdout_passed: bool


@dataclass(frozen=True, slots=True)
class CurriculumThresholds:
    """Two nested, auditable phase gates."""

    transition_min_complete_games: int = 1_000
    transition_min_heldout_positions: int = 10_000
    transition_min_arena_pairs: int = 64
    transition_min_complete_fraction: float = 0.98
    transition_min_reanalysis_coverage: float = 0.50
    transition_max_reanalysis_policy_js: float = 0.20
    transition_max_external_regret: float = 0.20
    transition_max_external_policy_js: float = 0.25
    transition_min_arena_lower_bound: float = 0.50
    autonomous_min_complete_games: int = 10_000
    autonomous_min_heldout_positions: int = 50_000
    autonomous_min_arena_pairs: int = 256
    autonomous_min_complete_fraction: float = 0.995
    autonomous_min_reanalysis_coverage: float = 0.80
    autonomous_max_reanalysis_policy_js: float = 0.10
    autonomous_max_external_regret: float = 0.10
    autonomous_max_external_policy_js: float = 0.15
    autonomous_min_arena_lower_bound: float = 0.52

    def __post_init__(self) -> None:
        integer_thresholds = (
            self.transition_min_complete_games,
            self.transition_min_heldout_positions,
            self.transition_min_arena_pairs,
            self.autonomous_min_complete_games,
            self.autonomous_min_heldout_positions,
            self.autonomous_min_arena_pairs,
        )
        if any(value < 1 for value in integer_thresholds):
            raise ValueError("curriculum evidence-count thresholds must be positive")
        probabilities = (
            self.transition_min_complete_fraction,
            self.transition_min_reanalysis_coverage,
            self.transition_max_reanalysis_policy_js,
            self.transition_max_external_regret,
            self.transition_max_external_policy_js,
            self.transition_min_arena_lower_bound,
            self.autonomous_min_complete_fraction,
            self.autonomous_min_reanalysis_coverage,
            self.autonomous_max_reanalysis_policy_js,
            self.autonomous_max_external_regret,
            self.autonomous_max_external_policy_js,
            self.autonomous_min_arena_lower_bound,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("curriculum probability thresholds must be finite in [0, 1]")
        if (
            self.autonomous_min_complete_games < self.transition_min_complete_games
            or self.autonomous_min_heldout_positions < self.transition_min_heldout_positions
            or self.autonomous_min_arena_pairs < self.transition_min_arena_pairs
            or self.autonomous_min_complete_fraction < self.transition_min_complete_fraction
            or self.autonomous_min_reanalysis_coverage
            < self.transition_min_reanalysis_coverage
            or self.autonomous_max_reanalysis_policy_js
            > self.transition_max_reanalysis_policy_js
            or self.autonomous_max_external_regret > self.transition_max_external_regret
            or self.autonomous_max_external_policy_js
            > self.transition_max_external_policy_js
            or self.autonomous_min_arena_lower_bound
            < self.transition_min_arena_lower_bound
        ):
            raise ValueError("autonomous curriculum thresholds must be stricter than transition")


@dataclass(frozen=True, slots=True)
class CurriculumDecision:
    """Selected stage, target-source weights, and unmet evidence gates."""

    phase: DistillationPhase
    actor_visit_weight: float
    self_reanalysis_weight: float
    external_teacher_weight: float
    blockers: tuple[str, ...]


def proven_mate_in_one_moves(board: Board) -> tuple[str, ...]:
    """Return every legal move that is independently verified as immediate mate."""

    if adjudicate_board(board) is not None:
        raise ValueError("cannot derive a mate target from a terminal position")
    root_turn = board.turn.value
    proven: list[str] = []
    for move in board.legal_moves():
        child = Board(board.to_sfen())
        child.apply_move(move)
        result = adjudicate_board(child)
        if (
            result is not None
            and result.termination is Termination.CHECKMATE
            and result.winner == root_turn
        ):
            proven.append(move.to_usi())
    return tuple(sorted(proven))


def all_legal_value_policy(
    board: Board,
    move_values: Mapping[str, float],
    move_visits: Mapping[str, int],
    *,
    temperature: float,
) -> SearchValuePolicyTarget:
    """Create an implicit policy only from complete, actually visited root values.

    Mate-in-one roots return a set-valued target with an empty ordinary policy.
    Callers must route that set to a proof-set mass loss instead of dense cross
    entropy or label smoothing.
    """

    if adjudicate_board(board) is not None:
        raise ValueError("cannot derive an all-legal policy from a terminal position")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("all-legal policy temperature must be finite and positive")
    legal_moves = tuple(sorted(move.to_usi() for move in board.legal_moves()))
    if not legal_moves:
        raise ValueError("all-legal policy requires at least one legal move")
    expected = set(legal_moves)
    values = dict(move_values)
    visits = dict(move_visits)
    for label, observed in (("value", set(values)), ("visit", set(visits))):
        if observed != expected:
            raise ValueError(
                f"all-legal {label} map does not exactly cover legal moves: "
                f"missing={sorted(expected - observed)!r}, outside={sorted(observed - expected)!r}"
            )
    normalized_values: dict[str, float] = {}
    normalized_visits: dict[str, int] = {}
    for move in legal_moves:
        value = float(values[move])
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"root Q for {move!r} must be finite in [-1, 1]")
        visit = visits[move]
        if isinstance(visit, bool) or not isinstance(visit, int) or visit < 1:
            raise ValueError(f"root visits for {move!r} must be a positive integer")
        normalized_values[move] = value
        normalized_visits[move] = visit

    mate_moves = proven_mate_in_one_moves(board)
    if mate_moves:
        return SearchValuePolicyTarget(
            kind=SearchPolicyKind.PROVEN_MATE_SET,
            legal_moves=legal_moves,
            move_values=normalized_values,
            move_visits=normalized_visits,
            policy={},
            temperature=temperature,
            proven_mate_moves=mate_moves,
        )

    maximum = max(normalized_values.values())
    weights = {
        move: math.exp((normalized_values[move] - maximum) / temperature)
        for move in legal_moves
    }
    total = math.fsum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("all-legal value softmax produced invalid mass")
    policy = {move: weights[move] / total for move in legal_moves}
    return SearchValuePolicyTarget(
        kind=SearchPolicyKind.ALL_LEGAL_VALUE_DISTRIBUTION,
        legal_moves=legal_moves,
        move_values=normalized_values,
        move_visits=normalized_visits,
        policy=policy,
        temperature=temperature,
    )


def blend_visit_and_value_policy(
    visit_policy: Mapping[str, float],
    implicit_policy: Mapping[str, float] | None,
    *,
    implicit_weight: float,
) -> dict[str, float]:
    """Blend exploration-aware visits with all-legal value ranking."""

    if not math.isfinite(implicit_weight) or not 0.0 <= implicit_weight <= 1.0:
        raise ValueError("implicit policy weight must be finite in [0, 1]")
    visits = dict(visit_policy)
    implicit = None if implicit_policy is None else dict(implicit_policy)
    if not visits:
        raise ValueError("visit policy must not be empty")
    if implicit is None:
        implicit_weight = 0.0
    elif set(implicit) != set(visits):
        raise ValueError("visit and implicit policies must cover the same moves")

    def normalized(policy: Mapping[str, float], *, label: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for move, raw in policy.items():
            probability = float(raw)
            if not math.isfinite(probability) or probability < 0.0:
                raise ValueError(
                    f"{label} probability for {move!r} must be finite and non-negative"
                )
            result[move] = probability
        total = math.fsum(result.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{label} policy must sum to one, observed {total}")
        return {move: probability / total for move, probability in result.items()}

    normalized_visits = normalized(visits, label="visit")
    if implicit is None:
        return normalized_visits
    normalized_implicit = normalized(implicit, label="implicit")
    return {
        move: (1.0 - implicit_weight) * normalized_visits[move]
        + implicit_weight * normalized_implicit[move]
        for move in normalized_visits
    }


def merge_candidate_proposals(
    board: Board,
    proposals: Sequence[tuple[str, Sequence[str]]],
    *,
    required_moves: Sequence[str] = (),
) -> CandidateUnion:
    """Merge teacher/actor/book/tactical candidates while preserving provenance."""

    if not proposals:
        raise ValueError("candidate union requires at least one proposer")
    legal = {move.to_usi() for move in board.legal_moves()}
    rows: list[tuple[str, tuple[str, ...]]] = []
    proposer_ids: set[str] = set()
    union: set[str] = set()
    for proposer, raw_moves in proposals:
        if not proposer or proposer != proposer.strip() or proposer in proposer_ids:
            raise ValueError("candidate proposer IDs must be unique, non-empty, and trimmed")
        proposer_ids.add(proposer)
        moves = tuple(sorted(set(raw_moves)))
        outside = sorted(set(moves) - legal)
        if outside:
            raise ValueError(f"candidate proposer {proposer!r} supplied illegal moves: {outside!r}")
        rows.append((proposer, moves))
        union.update(moves)
    required = tuple(sorted(set(required_moves)))
    outside_required = sorted(set(required) - legal)
    if outside_required:
        raise ValueError(f"required candidate moves are illegal: {outside_required!r}")
    union.update(required)
    if not union:
        raise ValueError("candidate proposal union is empty")
    return CandidateUnion(proposer_moves=tuple(rows), moves=tuple(sorted(union)))


def linked_negamax_constraint(
    *,
    parent_move: str,
    parent_q: float,
    child_value: float,
    tolerance: float,
) -> LinkedNegamaxConstraint:
    """Check ``Q(s,a) ~= -V(s')`` only for a verified linked child position."""

    if not parent_move:
        raise ValueError("linked negamax constraint requires a parent move")
    values = (parent_q, child_value, tolerance)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("linked negamax values must be finite")
    if not -1.0 <= parent_q <= 1.0 or not -1.0 <= child_value <= 1.0:
        raise ValueError("linked negamax Q/value must be in [-1, 1]")
    if tolerance < 0.0:
        raise ValueError("linked negamax tolerance must be non-negative")
    residual = abs(parent_q + child_value)
    return LinkedNegamaxConstraint(
        parent_move=parent_move,
        parent_q=parent_q,
        child_value=child_value,
        absolute_residual=residual,
        tolerance=tolerance,
        satisfied=residual <= tolerance,
    )


def depth_rank_constraint(
    budget_values: Sequence[tuple[int, Mapping[str, float]]],
    *,
    tie_tolerance: float = 1e-9,
) -> DepthRankConstraint:
    """Measure ranking reversals across common-condition search budgets."""

    if len(budget_values) < 2:
        raise ValueError("depth-rank constraint requires at least two budgets")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("rank tie tolerance must be finite and non-negative")
    budgets = tuple(budget for budget, _values in budget_values)
    if budgets != tuple(sorted(set(budgets))) or budgets[0] < 1:
        raise ValueError("rank budgets must be positive and strictly increasing")
    candidate_set: set[str] | None = None
    normalized: list[dict[str, float]] = []
    for budget, raw in budget_values:
        values = {move: float(value) for move, value in raw.items()}
        if not values:
            raise ValueError(f"rank budget {budget} has no candidates")
        if candidate_set is None:
            candidate_set = set(values)
        elif set(values) != candidate_set:
            raise ValueError("every rank budget must score the same candidate set")
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in values.values()):
            raise ValueError(f"rank budget {budget} contains an invalid Q value")
        normalized.append(values)
    if candidate_set is None:
        raise AssertionError("validated rank evidence unexpectedly has no candidates")
    candidates = tuple(sorted(candidate_set))
    deepest = normalized[-1]
    maximum = max(deepest.values())
    deepest_best = tuple(
        move for move in candidates if maximum - deepest[move] <= tie_tolerance
    )
    top_stable = all(
        bool(
            set(deepest_best)
            & {
                move
                for move in candidates
                if max(values.values()) - values[move] <= tie_tolerance
            }
        )
        for values in normalized[:-1]
    )
    reversals = 0
    comparisons = 0
    for shallower, deeper in pairwise(normalized):
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                shallow_delta = shallower[left] - shallower[right]
                deep_delta = deeper[left] - deeper[right]
                if abs(shallow_delta) <= tie_tolerance or abs(deep_delta) <= tie_tolerance:
                    continue
                comparisons += 1
                reversals += int((shallow_delta > 0.0) != (deep_delta > 0.0))
    return DepthRankConstraint(
        budgets=budgets,
        candidates=candidates,
        deepest_best_moves=deepest_best,
        top_move_stable=top_stable,
        pairwise_reversals=reversals,
        pairwise_comparisons=comparisons,
        reversal_fraction=(0.0 if comparisons == 0 else reversals / comparisons),
    )


def choose_distillation_phase(
    evidence: CurriculumEvidence,
    thresholds: CurriculumThresholds | None = None,
) -> CurriculumDecision:
    """Reduce teacher dependence only after every independent gate passes."""

    thresholds = CurriculumThresholds() if thresholds is None else thresholds

    if evidence.complete_games < 0 or evidence.heldout_positions < 0 or evidence.arena_pairs < 0:
        raise ValueError("curriculum evidence counts must be non-negative")
    probabilities = (
        evidence.complete_game_fraction,
        evidence.self_reanalysis_coverage,
        evidence.self_reanalysis_policy_js,
        evidence.external_teacher_worst_regret,
        evidence.external_teacher_policy_js,
        evidence.arena_score_lower_bound,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("curriculum evidence rates must be finite in [0, 1]")

    def blockers(*, autonomous: bool) -> list[str]:
        prefix = "autonomous" if autonomous else "transition"
        checks = (
            (
                evidence.complete_games
                >= (
                    thresholds.autonomous_min_complete_games
                    if autonomous
                    else thresholds.transition_min_complete_games
                ),
                f"{prefix}:complete_games",
            ),
            (
                evidence.heldout_positions
                >= (
                    thresholds.autonomous_min_heldout_positions
                    if autonomous
                    else thresholds.transition_min_heldout_positions
                ),
                f"{prefix}:heldout_positions",
            ),
            (
                evidence.arena_pairs
                >= (
                    thresholds.autonomous_min_arena_pairs
                    if autonomous
                    else thresholds.transition_min_arena_pairs
                ),
                f"{prefix}:arena_pairs",
            ),
            (
                evidence.complete_game_fraction
                >= (
                    thresholds.autonomous_min_complete_fraction
                    if autonomous
                    else thresholds.transition_min_complete_fraction
                ),
                f"{prefix}:complete_game_fraction",
            ),
            (
                evidence.self_reanalysis_coverage
                >= (
                    thresholds.autonomous_min_reanalysis_coverage
                    if autonomous
                    else thresholds.transition_min_reanalysis_coverage
                ),
                f"{prefix}:self_reanalysis_coverage",
            ),
            (
                evidence.self_reanalysis_policy_js
                <= (
                    thresholds.autonomous_max_reanalysis_policy_js
                    if autonomous
                    else thresholds.transition_max_reanalysis_policy_js
                ),
                f"{prefix}:self_reanalysis_policy_js",
            ),
            (
                evidence.external_teacher_worst_regret
                <= (
                    thresholds.autonomous_max_external_regret
                    if autonomous
                    else thresholds.transition_max_external_regret
                ),
                f"{prefix}:external_teacher_worst_regret",
            ),
            (
                evidence.external_teacher_policy_js
                <= (
                    thresholds.autonomous_max_external_policy_js
                    if autonomous
                    else thresholds.transition_max_external_policy_js
                ),
                f"{prefix}:external_teacher_policy_js",
            ),
            (
                evidence.arena_score_lower_bound
                >= (
                    thresholds.autonomous_min_arena_lower_bound
                    if autonomous
                    else thresholds.transition_min_arena_lower_bound
                ),
                f"{prefix}:arena_score_lower_bound",
            ),
            (evidence.no_heldout_regression, f"{prefix}:heldout_regression"),
            (evidence.mate_holdout_passed, f"{prefix}:mate_holdout"),
        )
        return [reason for passed, reason in checks if not passed]

    autonomous_blockers = blockers(autonomous=True)
    if not autonomous_blockers:
        return CurriculumDecision(
            phase=DistillationPhase.AUTONOMOUS,
            actor_visit_weight=0.30,
            self_reanalysis_weight=0.60,
            external_teacher_weight=0.10,
            blockers=(),
        )
    transition_blockers = blockers(autonomous=False)
    if not transition_blockers:
        return CurriculumDecision(
            phase=DistillationPhase.TRANSITION,
            actor_visit_weight=0.20,
            self_reanalysis_weight=0.45,
            external_teacher_weight=0.35,
            blockers=tuple(autonomous_blockers),
        )
    return CurriculumDecision(
        phase=DistillationPhase.BOOTSTRAP,
        actor_visit_weight=0.10,
        self_reanalysis_weight=0.25,
        external_teacher_weight=0.65,
        blockers=tuple(transition_blockers),
    )
