"""Interval-safe targets for multi-proposer, single-scorer distillation.

Candidate discovery may use every approved teacher.  This module only accepts
scores from one declared scorer and therefore never averages incompatible
teacher value scales.  Alpha-beta node counts are retained as evidence but are
never converted into policy mass.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final

from .domain import TeacherScoreBound

_INTERVAL_TOLERANCE: Final = 1e-12


class AnchorResolutionKind(StrEnum):
    PROVEN_MATE = "proven_mate"
    EXACT_Q_DISTRIBUTION = "exact_q_distribution"
    INTERVAL_DOMINANCE = "interval_dominance"
    UNRESOLVED = "unresolved"


class AnchorValueLossKind(StrEnum):
    POINT = "point"
    INTERVAL = "interval"
    DISABLED = "disabled"


class AnchorReanalysisReason(StrEnum):
    CHALLENGER_TOP1 = "challenger_top1_disagreement"
    REPORTED_MATE = "reported_mate_requires_proof"
    VALUE_SIGN = "teacher_value_sign_conflict"
    BUDGET_TOP1 = "budget_top1_instability"
    REPLY_TOP1 = "reply_backup_changed_top1"
    NARROW_GAP = "anchor_top2_gap_below_threshold"
    NON_EXACT_BOUND = "non_exact_anchor_bound"
    OVERLAPPING_INTERVALS = "anchor_intervals_overlap"


@dataclass(frozen=True, slots=True)
class AnchorCandidateScore:
    """One equal-requested-budget branch score from the declared anchor."""

    move: str
    q_value: float
    bound: TeacherScoreBound
    requested_nodes: int
    reported_nodes: int | None = None

    def __post_init__(self) -> None:
        if not self.move or any(character.isspace() for character in self.move):
            raise ValueError("candidate move must be a non-empty USI token")
        if not math.isfinite(self.q_value) or not -1.0 <= self.q_value <= 1.0:
            raise ValueError("candidate Q must be finite in [-1, 1]")
        if self.requested_nodes < 1:
            raise ValueError("requested nodes must be positive")
        if self.reported_nodes is not None and self.reported_nodes < 0:
            raise ValueError("reported nodes must be non-negative")

    @property
    def interval(self) -> tuple[float, float]:
        if self.bound is TeacherScoreBound.EXACT:
            return self.q_value, self.q_value
        if self.bound is TeacherScoreBound.LOWER:
            return self.q_value, 1.0
        if self.bound is TeacherScoreBound.UPPER:
            return -1.0, self.q_value
        raise AssertionError(f"unsupported teacher score bound: {self.bound!r}")


@dataclass(frozen=True, slots=True)
class AnchorLabelTarget:
    scorer_id: str
    resolution: AnchorResolutionKind
    candidate_scores: tuple[AnchorCandidateScore, ...]
    policy: dict[str, float]
    best_moves: tuple[str, ...]
    chosen_move: str | None
    value_interval: tuple[float, float] | None
    policy_loss_enabled: bool
    value_loss_kind: AnchorValueLossKind
    additional_search_required: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validated_scores(
    scores: tuple[AnchorCandidateScore, ...],
) -> tuple[AnchorCandidateScore, ...]:
    if not scores:
        raise ValueError("anchor target requires at least one candidate score")
    moves = tuple(score.move for score in scores)
    if len(set(moves)) != len(moves):
        raise ValueError("anchor candidate moves must be unique")
    budgets = {score.requested_nodes for score in scores}
    if len(budgets) != 1:
        raise ValueError("policy Q values must come from one equal requested node budget")
    return tuple(sorted(scores, key=lambda score: score.move))


def _softmax_q_policy(
    scores: tuple[AnchorCandidateScore, ...], *, temperature: float
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("policy temperature must be finite and positive")
    if any(score.bound is not TeacherScoreBound.EXACT for score in scores):
        raise ValueError("a Q-distribution policy requires exact scores")
    maximum = max(score.q_value for score in scores)
    weights = {
        score.move: math.exp((score.q_value - maximum) / temperature) for score in scores
    }
    total = math.fsum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


def _dominant_move(
    scores: tuple[AnchorCandidateScore, ...], *, margin: float
) -> str | None:
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("interval-dominance margin must be finite and non-negative")
    if len(scores) == 1:
        return scores[0].move
    dominant: list[str] = []
    for candidate in scores:
        lower, _upper = candidate.interval
        other_upper = max(
            other.interval[1] for other in scores if other.move != candidate.move
        )
        if lower > other_upper + margin:
            dominant.append(candidate.move)
    if len(dominant) > 1:
        raise AssertionError("two disjoint intervals cannot both dominate")
    return None if not dominant else dominant[0]


def resolve_anchor_target(
    scorer_id: str,
    scores: tuple[AnchorCandidateScore, ...],
    *,
    policy_temperature: float,
    interval_margin: float = 0.0,
    proven_mate_moves: tuple[str, ...] = (),
) -> AnchorLabelTarget:
    """Resolve one anchor matrix without inventing values inside search bounds."""

    if not scorer_id:
        raise ValueError("scorer ID must not be empty")
    ordered = _validated_scores(scores)
    score_moves = {score.move for score in ordered}
    if len(set(proven_mate_moves)) != len(proven_mate_moves):
        raise ValueError("proven mate moves must be unique")
    if not set(proven_mate_moves).issubset(score_moves):
        raise ValueError("every proven mate move must belong to the candidate matrix")
    if proven_mate_moves:
        mates = tuple(sorted(proven_mate_moves))
        probability = 1.0 / len(mates)
        return AnchorLabelTarget(
            scorer_id=scorer_id,
            resolution=AnchorResolutionKind.PROVEN_MATE,
            candidate_scores=ordered,
            policy={move: probability for move in mates},
            best_moves=mates,
            chosen_move=mates[0],
            value_interval=(1.0, 1.0),
            policy_loss_enabled=True,
            value_loss_kind=AnchorValueLossKind.POINT,
            additional_search_required=False,
        )

    if all(score.bound is TeacherScoreBound.EXACT for score in ordered):
        maximum = max(score.q_value for score in ordered)
        best = tuple(
            score.move
            for score in ordered
            if math.isclose(
                score.q_value, maximum, rel_tol=0.0, abs_tol=_INTERVAL_TOLERANCE
            )
        )
        return AnchorLabelTarget(
            scorer_id=scorer_id,
            resolution=AnchorResolutionKind.EXACT_Q_DISTRIBUTION,
            candidate_scores=ordered,
            policy=_softmax_q_policy(ordered, temperature=policy_temperature),
            best_moves=best,
            chosen_move=best[0],
            value_interval=(maximum, maximum),
            policy_loss_enabled=True,
            value_loss_kind=AnchorValueLossKind.POINT,
            additional_search_required=False,
        )

    dominant = _dominant_move(ordered, margin=interval_margin)
    if dominant is not None:
        winning_score = next(score for score in ordered if score.move == dominant)
        return AnchorLabelTarget(
            scorer_id=scorer_id,
            resolution=AnchorResolutionKind.INTERVAL_DOMINANCE,
            candidate_scores=ordered,
            policy={dominant: 1.0},
            best_moves=(dominant,),
            chosen_move=dominant,
            value_interval=winning_score.interval,
            policy_loss_enabled=True,
            value_loss_kind=(
                AnchorValueLossKind.POINT
                if winning_score.bound is TeacherScoreBound.EXACT
                else AnchorValueLossKind.INTERVAL
            ),
            additional_search_required=False,
        )

    value_interval = (
        max(score.interval[0] for score in ordered),
        max(score.interval[1] for score in ordered),
    )
    informative_value = value_interval != (-1.0, 1.0)
    return AnchorLabelTarget(
        scorer_id=scorer_id,
        resolution=AnchorResolutionKind.UNRESOLVED,
        candidate_scores=ordered,
        policy={},
        best_moves=(),
        chosen_move=None,
        value_interval=value_interval if informative_value else None,
        policy_loss_enabled=False,
        value_loss_kind=(
            AnchorValueLossKind.INTERVAL
            if informative_value
            else AnchorValueLossKind.DISABLED
        ),
        additional_search_required=True,
    )


def interval_squared_error(
    prediction: float, value_interval: tuple[float, float]
) -> float:
    """Squared hinge loss: zero anywhere inside a trustworthy value interval."""

    lower, upper = value_interval
    if (
        not math.isfinite(prediction)
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not -1.0 <= lower <= upper <= 1.0
    ):
        raise ValueError("prediction and value interval must be finite and bounded")
    if prediction < lower:
        return (lower - prediction) ** 2
    if prediction > upper:
        return (prediction - upper) ** 2
    return 0.0


def anchor_reanalysis_reasons(
    *,
    anchor_top_move: str,
    challenger_top_moves: tuple[str, ...],
    teacher_values: tuple[float, ...],
    reported_mate: bool,
    budget_top_moves: tuple[str, ...],
    reply_backed_top_move: str | None,
    exact_top_gap: float | None,
    narrow_gap_threshold: float,
    scores: tuple[AnchorCandidateScore, ...],
) -> tuple[AnchorReanalysisReason, ...]:
    """Derive auditable reasons for allocating a larger anchor search budget."""

    if not anchor_top_move:
        raise ValueError("anchor top move must not be empty")
    if not math.isfinite(narrow_gap_threshold) or narrow_gap_threshold < 0.0:
        raise ValueError("narrow-gap threshold must be finite and non-negative")
    if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in teacher_values):
        raise ValueError("teacher values must be finite in [-1, 1]")
    reasons: set[AnchorReanalysisReason] = set()
    if any(move != anchor_top_move for move in challenger_top_moves):
        reasons.add(AnchorReanalysisReason.CHALLENGER_TOP1)
    if reported_mate:
        reasons.add(AnchorReanalysisReason.REPORTED_MATE)
    if any(value < 0.0 for value in teacher_values) and any(
        value > 0.0 for value in teacher_values
    ):
        reasons.add(AnchorReanalysisReason.VALUE_SIGN)
    if len(set(budget_top_moves)) > 1:
        reasons.add(AnchorReanalysisReason.BUDGET_TOP1)
    if reply_backed_top_move is not None and reply_backed_top_move != anchor_top_move:
        reasons.add(AnchorReanalysisReason.REPLY_TOP1)
    if exact_top_gap is not None:
        if not math.isfinite(exact_top_gap) or exact_top_gap < 0.0:
            raise ValueError("exact top gap must be finite and non-negative")
        if exact_top_gap < narrow_gap_threshold:
            reasons.add(AnchorReanalysisReason.NARROW_GAP)
    if any(score.bound is not TeacherScoreBound.EXACT for score in scores):
        reasons.add(AnchorReanalysisReason.NON_EXACT_BOUND)
    ordered = _validated_scores(scores)
    if (
        any(score.bound is not TeacherScoreBound.EXACT for score in ordered)
        and _dominant_move(ordered, margin=0.0) is None
    ):
        reasons.add(AnchorReanalysisReason.OVERLAPPING_INTERVALS)
    return tuple(sorted(reasons, key=str))
