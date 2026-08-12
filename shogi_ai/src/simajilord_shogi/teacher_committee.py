"""Online three-teacher signal synthesis for descriptive direct-play probes.

This module deliberately keeps the committee separate from the canonical-v2
training-data builder.  A committee game is evidence about an online decision
rule; its actual moves are never teacher labels.  The reusable pure function
below combines a complete cross-teacher score matrix by minimising the largest
per-teacher regret, while the player obtains that matrix from fresh USI
``searchmoves`` analyses at every configured node budget.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from rsshogi.core import Board, Move

from .adjudication import adjudicate_board, can_declare_win_csa27
from .arena import MoveDecision
from .domain import TeacherScoreBound
from .external_usi import ExternalUsiTeacher, UsiPositionHistory


@dataclass(frozen=True, slots=True)
class RobustSignal:
    """A full-distribution minimax-regret synthesis of one score matrix."""

    policy: dict[str, float]
    chosen_move: str
    robust_best_moves: tuple[str, ...]
    worst_regret: dict[str, float]
    mean_value: dict[str, float]
    chosen_conservative_value: float


@dataclass(frozen=True, slots=True)
class CommitteeBudgetEvidence:
    requested_nodes: int
    reported_nodes: int
    branch_reanalyses: int
    conservative_bound_fallbacks: int
    teacher_values: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    robust_policy: tuple[tuple[str, float], ...]
    worst_regret: tuple[tuple[str, float], ...]
    robust_best_moves: tuple[str, ...]
    chosen_move: str


@dataclass(frozen=True, slots=True)
class CommitteeDecisionEvidence:
    target_sfen: str
    teacher_proposals: tuple[tuple[str, tuple[str, ...]], ...]
    candidate_moves: tuple[str, ...]
    budgets: tuple[CommitteeBudgetEvidence, ...]
    chosen_move: str
    budget_choice_stable: bool
    reported_nodes: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HighDepthCommitteeBudgetEvidence:
    requested_nodes_per_candidate: int
    bound_retry_multipliers: tuple[int, ...]
    reported_nodes: int
    branch_searches: int
    bound_retries: int
    persistent_bounds: int
    teacher_intervals: tuple[
        tuple[str, tuple[tuple[str, float, float, str], ...]], ...
    ]
    robust_policy: tuple[tuple[str, float], ...]
    worst_case_regret: tuple[tuple[str, float], ...]
    robust_best_moves: tuple[str, ...]
    chosen_move: str


@dataclass(frozen=True, slots=True)
class HighDepthCommitteeDecisionEvidence:
    target_sfen: str
    teacher_proposals: tuple[tuple[str, tuple[str, ...]], ...]
    candidate_moves: tuple[str, ...]
    budgets: tuple[HighDepthCommitteeBudgetEvidence, ...]
    chosen_move: str
    budget_choice_stable: bool
    reply_reanalysis: HighDepthReplyReanalysisEvidence
    reported_nodes: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HighDepthReplyCandidateEvidence:
    move: str
    terminal_root_value: float | None
    reply_proposals: tuple[tuple[str, tuple[str, ...]], ...]
    reply_moves: tuple[str, ...]
    teacher_reply_intervals: tuple[
        tuple[str, tuple[tuple[str, float, float, str], ...]], ...
    ]
    teacher_backed_intervals: tuple[tuple[str, float, float], ...]
    reported_nodes: int
    elapsed_seconds: float
    branch_searches: int
    bound_retries: int
    persistent_bounds: int


@dataclass(frozen=True, slots=True)
class HighDepthReplyReanalysisEvidence:
    root_choice: str
    reanalysed_candidates: tuple[str, ...]
    reply_proposal_nodes_per_teacher: int
    reply_score_nodes_per_teacher_per_reply: int
    bound_retry_multipliers: tuple[int, ...]
    candidates: tuple[HighDepthReplyCandidateEvidence, ...]
    robust_policy: tuple[tuple[str, float], ...]
    worst_case_regret: tuple[tuple[str, float], ...]
    chosen_move: str
    changed_root_choice: bool
    reported_nodes: int
    elapsed_seconds: float
    branch_searches: int
    bound_retries: int
    persistent_bounds: int


def score_interval_from_bound(
    value: float,
    bound: TeacherScoreBound,
) -> tuple[float, float]:
    """Convert one root-side USI score and bound to a closed Q interval."""

    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("bounded teacher value must be finite in [-1, 1]")
    if bound is TeacherScoreBound.EXACT:
        return value, value
    if bound is TeacherScoreBound.LOWER:
        return value, 1.0
    if bound is TeacherScoreBound.UPPER:
        return -1.0, value
    raise AssertionError(f"unsupported teacher score bound: {bound!r}")


def backup_root_interval_from_child_replies(
    child_reply_intervals: Mapping[str, tuple[float, float]],
) -> tuple[float, float]:
    """Negamax-back up an opponent reply set without collapsing bounds.

    Child scores are from the opponent-to-move perspective.  Negation maps a
    child interval ``[L, U]`` to the root interval ``[-U, -L]``.  The opponent
    then chooses the reply that is worst for the root player, so the interval
    for that minimum is ``[min(-U), min(-L)]``.
    """

    if not child_reply_intervals:
        raise ValueError("reply backup requires at least one child interval")
    root_reply_intervals: list[tuple[float, float]] = []
    for move, interval in child_reply_intervals.items():
        if len(interval) != 2:
            raise ValueError(f"reply interval for {move!r} must have two endpoints")
        lower, upper = interval
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not -1.0 <= lower <= upper <= 1.0
        ):
            raise ValueError(f"reply interval for {move!r} is invalid")
        root_reply_intervals.append((-upper, -lower))
    return (
        min(lower for lower, _upper in root_reply_intervals),
        min(upper for _lower, upper in root_reply_intervals),
    )


def synthesize_worst_case_interval_signal(
    teacher_intervals: Sequence[
        tuple[str, Mapping[str, tuple[float, float]]]
    ],
    *,
    temperature: float,
) -> RobustSignal:
    """Minimise a valid worst-case regret upper bound over score intervals.

    For teacher ``t`` and candidate ``m``, the true regret is no larger than
    ``max_j upper[t,j] - lower[t,m]``.  Minimising the largest such quantity
    across teachers is conservative under every value assignment allowed by
    the USI bounds.  Persistent bounds therefore widen uncertainty rather than
    being mislabeled as exact scalar supervision.
    """

    if len(teacher_intervals) < 2:
        raise ValueError("robust signal synthesis requires at least two teachers")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("robust signal temperature must be finite and positive")
    teacher_ids = tuple(teacher_id for teacher_id, _values in teacher_intervals)
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("robust signal teacher IDs must be unique")
    expected_moves: set[str] | None = None
    normalized: list[tuple[str, dict[str, tuple[float, float]]]] = []
    for teacher_id, raw_intervals in teacher_intervals:
        intervals = dict(raw_intervals)
        if not intervals:
            raise ValueError(f"teacher {teacher_id!r} supplied no candidate intervals")
        moves = set(intervals)
        if expected_moves is None:
            expected_moves = moves
        elif moves != expected_moves:
            missing = sorted(expected_moves - moves)
            outside = sorted(moves - expected_moves)
            raise ValueError(
                "robust signal requires a complete common candidate matrix: "
                f"teacher={teacher_id!r} missing={missing!r} outside={outside!r}"
            )
        for move, interval in intervals.items():
            if len(interval) != 2:
                raise ValueError(
                    f"teacher {teacher_id!r} interval for {move!r} must have two endpoints"
                )
            lower, upper = interval
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or not -1.0 <= lower <= upper <= 1.0
            ):
                raise ValueError(
                    f"teacher {teacher_id!r} interval for {move!r} is invalid"
                )
        normalized.append((teacher_id, intervals))
    if expected_moves is None:
        raise AssertionError("validated interval matrix unexpectedly has no moves")

    candidates = tuple(sorted(expected_moves))
    best_upper = {
        teacher_id: max(upper for _lower, upper in intervals.values())
        for teacher_id, intervals in normalized
    }
    worst_regret = {
        move: max(
            best_upper[teacher_id] - intervals[move][0]
            for teacher_id, intervals in normalized
        )
        for move in candidates
    }
    mean_lower = {
        move: math.fsum(intervals[move][0] for _teacher_id, intervals in normalized)
        / len(normalized)
        for move in candidates
    }
    minimum_regret = min(worst_regret.values())
    robust_best_moves = tuple(
        move
        for move in candidates
        if math.isclose(
            worst_regret[move], minimum_regret, rel_tol=0.0, abs_tol=1e-12
        )
    )
    chosen_move = min(robust_best_moves, key=lambda move: (-mean_lower[move], move))
    weights = {
        move: math.exp(-(worst_regret[move] - minimum_regret) / temperature)
        for move in candidates
    }
    total_weight = math.fsum(weights.values())
    policy = {move: weights[move] / total_weight for move in candidates}
    return RobustSignal(
        policy=policy,
        chosen_move=chosen_move,
        robust_best_moves=robust_best_moves,
        worst_regret=worst_regret,
        mean_value=mean_lower,
        chosen_conservative_value=min(
            intervals[chosen_move][0] for _teacher_id, intervals in normalized
        ),
    )


def synthesize_worst_regret_signal(
    teacher_values: Sequence[tuple[str, Mapping[str, float]]],
    *,
    temperature: float,
) -> RobustSignal:
    """Combine a complete score matrix without averaging teachers away.

    Each teacher first defines its own best value over the common candidate
    union.  A move's regret is the gap from that best value, and its robust
    regret is the largest such gap across teachers.  The play distribution is
    an exponential distribution over robust regret; actual play selects an
    exact minimum-regret move.  Mean value is used only as a deterministic
    tie-break after the minimax criterion.
    """

    return synthesize_worst_case_interval_signal(
        tuple(
            (
                teacher_id,
                {move: (value, value) for move, value in values.items()},
            )
            for teacher_id, values in teacher_values
        ),
        temperature=temperature,
    )


class RobustTeacherCommitteePlayer:
    """Cross-score a candidate union with every teacher before each move."""

    source = "meteo-three-teacher-worst-regret-committee-v1"

    def __init__(
        self,
        teachers: Sequence[ExternalUsiTeacher],
        *,
        proposal_nodes: int,
        score_budgets: Sequence[int],
        candidates_per_teacher: int,
        temperature: float,
    ) -> None:
        if len(teachers) < 2:
            raise ValueError("committee player requires at least two teacher instances")
        teacher_ids = tuple(teacher.policy.policy_id for teacher in teachers)
        if len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError("committee teacher policy IDs must be unique")
        if proposal_nodes < 1 or candidates_per_teacher < 1:
            raise ValueError("proposal nodes and candidates per teacher must be positive")
        budgets = tuple(int(nodes) for nodes in score_budgets)
        if len(budgets) < 2 or budgets != tuple(sorted(set(budgets))):
            raise ValueError(
                "committee score budgets must contain at least two strictly increasing values"
            )
        if budgets[0] < 1:
            raise ValueError("committee score budgets must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("committee temperature must be finite and positive")
        self.teachers = tuple(teachers)
        self.teacher_ids = teacher_ids
        self.proposal_nodes = proposal_nodes
        self.score_budgets = budgets
        self.candidates_per_teacher = candidates_per_teacher
        self.temperature = temperature
        self.evidence: list[CommitteeDecisionEvidence] = []

    def new_game(self) -> None:
        for teacher in self.teachers:
            teacher.new_game()

    def choose_move(self, board: Board) -> MoveDecision:
        return self.choose_move_with_history(
            board,
            UsiPositionHistory(
                initial_sfen=board.to_sfen(),
                moves=(),
                target_sfen=board.to_sfen(),
            ),
        )

    def choose_move_with_history(
        self, board: Board, history: UsiPositionHistory
    ) -> MoveDecision:
        if board.to_sfen() != history.target_sfen:
            raise ValueError("committee board does not match supplied USI history target")
        legal = {move.to_usi() for move in board.legal_moves()}
        proposal_rows: list[tuple[str, tuple[str, ...]]] = []
        proposal_analyses = []
        for teacher in self.teachers:
            analysis = teacher.analyse_with_history_searchmoves(
                history,
                tuple(sorted(legal)),
                nodes=self.proposal_nodes,
            )
            proposed = tuple(
                variation.move
                for variation in analysis.candidates
                if variation.move in legal
            )[: self.candidates_per_teacher]
            proposal_rows.append((teacher.policy.policy_id, proposed))
            proposal_analyses.append(analysis)
        candidates = tuple(
            sorted(
                {
                    move
                    for _teacher_id, proposals in proposal_rows
                    for move in proposals
                }
            )
        )
        if not candidates:
            bestmoves = {analysis.bestmove for analysis in proposal_analyses}
            if bestmoves == {"win"} and can_declare_win_csa27(board):
                return MoveDecision("win", {}, 1.0, source=self.source)
            raise RuntimeError(
                "committee teachers supplied no legal continuation; refusing to turn an "
                "engine resignation into a supposedly board-terminal game"
            )
        for teacher in self.teachers:
            if teacher.multipv < len(candidates):
                raise ValueError(
                    "committee teacher MultiPV is smaller than the candidate union: "
                    f"teacher={teacher.policy.policy_id!r} multipv={teacher.multipv} "
                    f"candidates={len(candidates)}"
                )

        total_nodes = sum(analysis.nodes or 0 for analysis in proposal_analyses)
        total_seconds = math.fsum(analysis.elapsed_seconds for analysis in proposal_analyses)
        budget_evidence: list[CommitteeBudgetEvidence] = []
        deepest_signal: RobustSignal | None = None
        for requested_nodes in self.score_budgets:
            score_rows: list[tuple[str, dict[str, float]]] = []
            budget_nodes = 0
            branch_reanalyses = 0
            conservative_bound_fallbacks = 0
            for teacher in self.teachers:
                analysis = teacher.analyse_with_history_searchmoves(
                    history,
                    candidates,
                    nodes=requested_nodes,
                )
                target = teacher.target_from_analysis(board, analysis)
                values = dict(target.move_values)
                outside = sorted(set(values) - set(candidates))
                if outside:
                    raise RuntimeError(
                        "committee scorer returned a move outside the requested union: "
                        f"teacher={teacher.policy.policy_id!r} outside={outside!r}"
                    )
                bounds = {
                    variation.move: variation.bound for variation in target.candidates
                }
                repairs = tuple(
                    move
                    for move in candidates
                    if move not in values
                    or bounds.get(move) is not TeacherScoreBound.EXACT
                )
                measured_nodes = analysis.nodes or 0
                budget_nodes += measured_nodes
                total_nodes += measured_nodes
                total_seconds += analysis.elapsed_seconds
                for move in repairs:
                    final_value: float | None = None
                    final_bound: TeacherScoreBound | None = None
                    for repair_nodes_requested in (
                        requested_nodes,
                        requested_nodes * 4,
                    ):
                        branch_analysis = teacher.analyse_with_history_searchmoves(
                            history,
                            (move,),
                            nodes=repair_nodes_requested,
                        )
                        branch_target = teacher.target_from_analysis(board, branch_analysis)
                        branch_values = dict(branch_target.move_values)
                        branch_bounds = {
                            variation.move: variation.bound
                            for variation in branch_target.candidates
                        }
                        if set(branch_values) != {move}:
                            raise RuntimeError(
                                "committee branch reanalysis returned the wrong move set: "
                                f"teacher={teacher.policy.policy_id!r} move={move!r} "
                                f"returned={sorted(branch_values)!r}"
                            )
                        final_value = branch_values[move]
                        final_bound = branch_bounds.get(move)
                        repair_nodes = branch_analysis.nodes or 0
                        budget_nodes += repair_nodes
                        total_nodes += repair_nodes
                        total_seconds += branch_analysis.elapsed_seconds
                        branch_reanalyses += 1
                        if final_bound is TeacherScoreBound.EXACT:
                            break
                    if final_value is None or final_bound is None:
                        raise AssertionError("validated branch reanalysis lost its score")
                    if final_bound is TeacherScoreBound.UPPER:
                        # An upper bound cannot establish any finite lower value.  Treat the
                        # branch as maximally bad rather than letting an optimistic bound win.
                        values[move] = -1.0
                        conservative_bound_fallbacks += 1
                    else:
                        # EXACT is the normal path.  A persistent LOWER bound is safe as a
                        # pessimistic floor for this live-play probe, but remains explicitly
                        # ineligible for the canonical training sidecar.
                        values[move] = final_value
                        conservative_bound_fallbacks += int(
                            final_bound is TeacherScoreBound.LOWER
                        )
                if set(values) != set(candidates):
                    missing = sorted(set(candidates) - set(values))
                    raise RuntimeError(
                        "committee scorer did not return the complete candidate union after "
                        f"exact repair: teacher={teacher.policy.policy_id!r} "
                        f"missing={missing!r}"
                    )
                score_rows.append((teacher.policy.policy_id, values))
            signal = synthesize_worst_regret_signal(
                score_rows,
                temperature=self.temperature,
            )
            deepest_signal = signal
            budget_evidence.append(
                CommitteeBudgetEvidence(
                    requested_nodes=requested_nodes,
                    reported_nodes=budget_nodes,
                    branch_reanalyses=branch_reanalyses,
                    conservative_bound_fallbacks=conservative_bound_fallbacks,
                    teacher_values=tuple(
                        (teacher_id, tuple(sorted(values.items())))
                        for teacher_id, values in score_rows
                    ),
                    robust_policy=tuple(sorted(signal.policy.items())),
                    worst_regret=tuple(sorted(signal.worst_regret.items())),
                    robust_best_moves=signal.robust_best_moves,
                    chosen_move=signal.chosen_move,
                )
            )
        if deepest_signal is None:
            raise AssertionError("validated committee score budgets unexpectedly produced none")
        chosen = deepest_signal.chosen_move
        stable = all(row.chosen_move == chosen for row in budget_evidence)
        self.evidence.append(
            CommitteeDecisionEvidence(
                target_sfen=board.to_sfen(),
                teacher_proposals=tuple(proposal_rows),
                candidate_moves=candidates,
                budgets=tuple(budget_evidence),
                chosen_move=chosen,
                budget_choice_stable=stable,
                reported_nodes=total_nodes,
                elapsed_seconds=total_seconds,
            )
        )
        return MoveDecision(
            chosen,
            deepest_signal.policy,
            deepest_signal.chosen_conservative_value,
            nodes=total_nodes,
            elapsed_seconds=total_seconds,
            nps=(None if total_seconds <= 0.0 else total_nodes / total_seconds),
            source=self.source,
        )


class HighDepthRobustTeacherCommitteePlayer:
    """Deep interval committee with adversarial reply reanalysis."""

    source = "meteo-three-teacher-reply-reanalysed-interval-committee-v3"

    def __init__(
        self,
        teachers: Sequence[ExternalUsiTeacher],
        *,
        proposal_nodes: int,
        score_budgets: Sequence[int],
        candidates_per_teacher: int,
        temperature: float,
        reply_reanalysis_candidates: int,
        reply_candidates_per_teacher: int,
        reply_proposal_nodes: int,
        reply_score_nodes: int,
        bound_retry_multipliers: Sequence[int] = (1, 4, 4, 8, 8),
    ) -> None:
        if len(teachers) < 2:
            raise ValueError("committee player requires at least two teacher instances")
        teacher_ids = tuple(teacher.policy.policy_id for teacher in teachers)
        if len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError("committee teacher policy IDs must be unique")
        if proposal_nodes < 1 or candidates_per_teacher < 1:
            raise ValueError("proposal nodes and candidates per teacher must be positive")
        budgets = tuple(int(nodes) for nodes in score_budgets)
        if len(budgets) < 2 or budgets != tuple(sorted(set(budgets))):
            raise ValueError(
                "committee score budgets must contain at least two strictly increasing values"
            )
        if budgets[0] < 1:
            raise ValueError("committee score budgets must be positive")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("committee temperature must be finite and positive")
        retry_multipliers = tuple(int(multiplier) for multiplier in bound_retry_multipliers)
        if (
            len(retry_multipliers) < 2
            or retry_multipliers[0] != 1
            or retry_multipliers != tuple(sorted(retry_multipliers))
        ):
            raise ValueError(
                "bound retry multipliers must start at one and contain at least two "
                "nondecreasing positive entries"
            )
        if reply_reanalysis_candidates < 1 or reply_candidates_per_teacher < 1:
            raise ValueError("reply candidate counts must be positive")
        if reply_proposal_nodes < 1 or reply_score_nodes < 1:
            raise ValueError("reply search node budgets must be positive")
        for teacher in teachers:
            required_multipv = max(candidates_per_teacher, reply_candidates_per_teacher)
            if teacher.multipv < required_multipv:
                raise ValueError(
                    "high-depth committee MultiPV must cover root and reply proposals: "
                    f"teacher={teacher.policy.policy_id!r} multipv={teacher.multipv}"
                )
        self.teachers = tuple(teachers)
        self.teacher_ids = teacher_ids
        self.proposal_nodes = proposal_nodes
        self.score_budgets = budgets
        self.candidates_per_teacher = candidates_per_teacher
        self.temperature = temperature
        self.reply_reanalysis_candidates = reply_reanalysis_candidates
        self.reply_candidates_per_teacher = reply_candidates_per_teacher
        self.reply_proposal_nodes = reply_proposal_nodes
        self.reply_score_nodes = reply_score_nodes
        self.bound_retry_multipliers = retry_multipliers
        self.evidence: list[HighDepthCommitteeDecisionEvidence] = []

    def new_game(self) -> None:
        for teacher in self.teachers:
            teacher.new_game()

    def choose_move(self, board: Board) -> MoveDecision:
        return self.choose_move_with_history(
            board,
            UsiPositionHistory(
                initial_sfen=board.to_sfen(),
                moves=(),
                target_sfen=board.to_sfen(),
            ),
        )

    def _propose(
        self,
        teacher: ExternalUsiTeacher,
        history: UsiPositionHistory,
        legal: tuple[str, ...],
        *,
        nodes: int,
        limit: int,
    ) -> tuple[str, tuple[str, ...], int, float, str]:
        analysis = teacher.analyse_with_history_searchmoves(
            history,
            legal,
            nodes=nodes,
        )
        legal_set = set(legal)
        proposed = tuple(
            variation.move
            for variation in analysis.candidates
            if variation.move in legal_set
        )[:limit]
        return (
            teacher.policy.policy_id,
            proposed,
            analysis.nodes or 0,
            analysis.elapsed_seconds,
            analysis.bestmove,
        )

    def _score_teacher_budget(
        self,
        teacher: ExternalUsiTeacher,
        history: UsiPositionHistory,
        candidates: tuple[str, ...],
        requested_nodes: int,
    ) -> tuple[
        str,
        dict[str, tuple[float, float]],
        tuple[tuple[str, float, float, str], ...],
        int,
        float,
        int,
        int,
        int,
    ]:
        intervals: dict[str, tuple[float, float]] = {}
        rows: list[tuple[str, float, float, str]] = []
        reported_nodes = 0
        elapsed_seconds = 0.0
        branch_searches = 0
        bound_retries = 0
        persistent_bounds = 0
        board = history.target_board()
        for move in candidates:
            final_value: float | None = None
            final_bound: TeacherScoreBound | None = None
            for attempt, multiplier in enumerate(self.bound_retry_multipliers):
                nodes = requested_nodes * multiplier
                analysis = teacher.analyse_with_history_searchmoves(
                    history,
                    (move,),
                    nodes=nodes,
                )
                target = teacher.target_from_analysis(board, analysis)
                values = dict(target.move_values)
                bounds = {
                    variation.move: variation.bound for variation in target.candidates
                }
                if set(values) != {move} or move not in bounds:
                    raise RuntimeError(
                        "high-depth branch search returned an incomplete move score: "
                        f"teacher={teacher.policy.policy_id!r} move={move!r} "
                        f"values={sorted(values)!r} bounds={sorted(bounds)!r}"
                    )
                final_value = values[move]
                final_bound = bounds[move]
                measured_nodes = analysis.nodes or 0
                reported_nodes += measured_nodes
                elapsed_seconds += analysis.elapsed_seconds
                branch_searches += 1
                bound_retries += int(attempt > 0)
                if final_bound is TeacherScoreBound.EXACT:
                    break
            if final_value is None or final_bound is None:
                raise AssertionError("validated high-depth branch lost its score")
            lower, upper = score_interval_from_bound(final_value, final_bound)
            intervals[move] = (lower, upper)
            rows.append((move, lower, upper, final_bound.value))
            persistent_bounds += int(final_bound is not TeacherScoreBound.EXACT)
        return (
            teacher.policy.policy_id,
            intervals,
            tuple(rows),
            reported_nodes,
            elapsed_seconds,
            branch_searches,
            bound_retries,
            persistent_bounds,
        )

    @staticmethod
    def _terminal_root_value(board: Board, root_turn: int) -> float | None:
        adjudication = adjudicate_board(board)
        if adjudication is None:
            return None
        if adjudication.winner is None:
            return 0.0
        return 1.0 if adjudication.winner == root_turn else -1.0

    def _reanalyse_replies(
        self,
        board: Board,
        history: UsiPositionHistory,
        root_signal: RobustSignal,
        candidates: tuple[str, ...],
    ) -> tuple[
        RobustSignal,
        HighDepthReplyReanalysisEvidence,
        int,
        float,
    ]:
        ranked_candidates = tuple(
            sorted(
                candidates,
                key=lambda move: (
                    root_signal.worst_regret[move],
                    -root_signal.mean_value[move],
                    move,
                ),
            )[: self.reply_reanalysis_candidates]
        )
        if not ranked_candidates:
            raise AssertionError("validated root candidates unexpectedly produced no reply set")

        backed_by_teacher: dict[str, dict[str, tuple[float, float]]] = {
            teacher_id: {} for teacher_id in self.teacher_ids
        }
        candidate_evidence: list[HighDepthReplyCandidateEvidence] = []
        total_nodes = 0
        total_seconds = 0.0
        total_branch_searches = 0
        total_bound_retries = 0
        total_persistent_bounds = 0
        root_turn = board.turn.value

        for candidate in ranked_candidates:
            child = history.target_board()
            candidate_move = Move.from_usi(candidate)
            if not child.is_legal_move(candidate_move):
                raise RuntimeError(
                    f"reply reanalysis received illegal root candidate {candidate!r}"
                )
            child.apply_move(candidate_move)
            terminal_root_value = self._terminal_root_value(child, root_turn)
            if terminal_root_value is not None:
                backed = tuple(
                    (teacher_id, terminal_root_value, terminal_root_value)
                    for teacher_id in self.teacher_ids
                )
                for teacher_id in self.teacher_ids:
                    backed_by_teacher[teacher_id][candidate] = (
                        terminal_root_value,
                        terminal_root_value,
                    )
                candidate_evidence.append(
                    HighDepthReplyCandidateEvidence(
                        move=candidate,
                        terminal_root_value=terminal_root_value,
                        reply_proposals=(),
                        reply_moves=(),
                        teacher_reply_intervals=(),
                        teacher_backed_intervals=backed,
                        reported_nodes=0,
                        elapsed_seconds=0.0,
                        branch_searches=0,
                        bound_retries=0,
                        persistent_bounds=0,
                    )
                )
                continue

            child_history = UsiPositionHistory(
                initial_sfen=history.initial_sfen,
                moves=(*history.moves, candidate),
                target_sfen=child.to_sfen(),
            )
            reply_legal = tuple(sorted(move.to_usi() for move in child.legal_moves()))
            if not reply_legal:
                raise AssertionError("nonterminal child position has no legal replies")
            with ThreadPoolExecutor(max_workers=len(self.teachers)) as executor:
                proposals = tuple(
                    executor.map(
                        lambda teacher,
                        child_history=child_history,
                        reply_legal=reply_legal: self._propose(
                            teacher,
                            child_history,
                            reply_legal,
                            nodes=self.reply_proposal_nodes,
                            limit=self.reply_candidates_per_teacher,
                        ),
                        self.teachers,
                    )
                )
            proposal_rows = tuple(
                (teacher_id, moves) for teacher_id, moves, *_rest in proposals
            )
            reply_moves = tuple(
                sorted(
                    {
                        move
                        for _teacher_id, teacher_moves in proposal_rows
                        for move in teacher_moves
                    }
                )
            )
            if not reply_moves:
                raise RuntimeError(
                    "nonterminal reply proposals were empty: "
                    f"candidate={candidate!r} bestmoves={tuple(row[4] for row in proposals)!r}"
                )

            proposal_nodes = sum(row[2] for row in proposals)
            proposal_seconds = math.fsum(row[3] for row in proposals)
            with ThreadPoolExecutor(max_workers=len(self.teachers)) as executor:
                score_rows = tuple(
                    executor.map(
                        lambda teacher,
                        child_history=child_history,
                        reply_moves=reply_moves: self._score_teacher_budget(
                            teacher,
                            child_history,
                            reply_moves,
                            self.reply_score_nodes,
                        ),
                        self.teachers,
                    )
                )
            score_nodes = sum(row[3] for row in score_rows)
            score_seconds = math.fsum(row[4] for row in score_rows)
            branch_searches = sum(row[5] for row in score_rows)
            bound_retries = sum(row[6] for row in score_rows)
            persistent_bounds = sum(row[7] for row in score_rows)
            backed_rows: list[tuple[str, float, float]] = []
            for teacher_id, intervals, _rows, *_metrics in score_rows:
                lower, upper = backup_root_interval_from_child_replies(intervals)
                backed_by_teacher[teacher_id][candidate] = (lower, upper)
                backed_rows.append((teacher_id, lower, upper))

            candidate_nodes = proposal_nodes + score_nodes
            candidate_seconds = proposal_seconds + score_seconds
            total_nodes += candidate_nodes
            total_seconds += candidate_seconds
            total_branch_searches += branch_searches
            total_bound_retries += bound_retries
            total_persistent_bounds += persistent_bounds
            candidate_evidence.append(
                HighDepthReplyCandidateEvidence(
                    move=candidate,
                    terminal_root_value=None,
                    reply_proposals=proposal_rows,
                    reply_moves=reply_moves,
                    teacher_reply_intervals=tuple(
                        (teacher_id, rows)
                        for teacher_id, _intervals, rows, *_metrics in score_rows
                    ),
                    teacher_backed_intervals=tuple(backed_rows),
                    reported_nodes=candidate_nodes,
                    elapsed_seconds=candidate_seconds,
                    branch_searches=branch_searches,
                    bound_retries=bound_retries,
                    persistent_bounds=persistent_bounds,
                )
            )

        reply_signal = synthesize_worst_case_interval_signal(
            tuple(
                (teacher_id, backed_by_teacher[teacher_id])
                for teacher_id in self.teacher_ids
            ),
            temperature=self.temperature,
        )
        evidence = HighDepthReplyReanalysisEvidence(
            root_choice=root_signal.chosen_move,
            reanalysed_candidates=ranked_candidates,
            reply_proposal_nodes_per_teacher=self.reply_proposal_nodes,
            reply_score_nodes_per_teacher_per_reply=self.reply_score_nodes,
            bound_retry_multipliers=self.bound_retry_multipliers,
            candidates=tuple(candidate_evidence),
            robust_policy=tuple(sorted(reply_signal.policy.items())),
            worst_case_regret=tuple(sorted(reply_signal.worst_regret.items())),
            chosen_move=reply_signal.chosen_move,
            changed_root_choice=reply_signal.chosen_move != root_signal.chosen_move,
            reported_nodes=total_nodes,
            elapsed_seconds=total_seconds,
            branch_searches=total_branch_searches,
            bound_retries=total_bound_retries,
            persistent_bounds=total_persistent_bounds,
        )
        return reply_signal, evidence, total_nodes, total_seconds

    def choose_move_with_history(
        self, board: Board, history: UsiPositionHistory
    ) -> MoveDecision:
        if board.to_sfen() != history.target_sfen:
            raise ValueError("committee board does not match supplied USI history target")
        legal = tuple(sorted(move.to_usi() for move in board.legal_moves()))
        with ThreadPoolExecutor(max_workers=len(self.teachers)) as executor:
            proposals = tuple(
                executor.map(
                    lambda teacher: self._propose(
                        teacher,
                        history,
                        legal,
                        nodes=self.proposal_nodes,
                        limit=self.candidates_per_teacher,
                    ),
                    self.teachers,
                )
            )
        proposal_rows = tuple((teacher_id, moves) for teacher_id, moves, *_rest in proposals)
        candidates = tuple(
            sorted(
                {
                    move
                    for _teacher_id, teacher_moves in proposal_rows
                    for move in teacher_moves
                }
            )
        )
        if not candidates:
            bestmoves = {row[4] for row in proposals}
            if bestmoves == {"win"} and can_declare_win_csa27(board):
                return MoveDecision("win", {}, 1.0, source=self.source)
            raise RuntimeError(
                "committee teachers supplied no legal continuation; refusing to turn an "
                "engine resignation into a supposedly board-terminal game"
            )

        total_nodes = sum(row[2] for row in proposals)
        total_seconds = math.fsum(row[3] for row in proposals)
        budget_evidence: list[HighDepthCommitteeBudgetEvidence] = []
        deepest_signal: RobustSignal | None = None
        for requested_nodes in self.score_budgets:
            with ThreadPoolExecutor(max_workers=len(self.teachers)) as executor:
                score_rows = tuple(
                    executor.map(
                        lambda teacher, nodes=requested_nodes: self._score_teacher_budget(
                            teacher,
                            history,
                            candidates,
                            nodes,
                        ),
                        self.teachers,
                    )
                )
            interval_matrix = tuple(
                (teacher_id, intervals)
                for teacher_id, intervals, _rows, *_metrics in score_rows
            )
            signal = synthesize_worst_case_interval_signal(
                interval_matrix,
                temperature=self.temperature,
            )
            deepest_signal = signal
            budget_nodes = sum(row[3] for row in score_rows)
            budget_seconds = math.fsum(row[4] for row in score_rows)
            total_nodes += budget_nodes
            total_seconds += budget_seconds
            budget_evidence.append(
                HighDepthCommitteeBudgetEvidence(
                    requested_nodes_per_candidate=requested_nodes,
                    bound_retry_multipliers=self.bound_retry_multipliers,
                    reported_nodes=budget_nodes,
                    branch_searches=sum(row[5] for row in score_rows),
                    bound_retries=sum(row[6] for row in score_rows),
                    persistent_bounds=sum(row[7] for row in score_rows),
                    teacher_intervals=tuple(
                        (teacher_id, rows)
                        for teacher_id, _intervals, rows, *_metrics in score_rows
                    ),
                    robust_policy=tuple(sorted(signal.policy.items())),
                    worst_case_regret=tuple(sorted(signal.worst_regret.items())),
                    robust_best_moves=signal.robust_best_moves,
                    chosen_move=signal.chosen_move,
                )
            )
        if deepest_signal is None:
            raise AssertionError("validated high-depth budgets unexpectedly produced none")
        root_chosen = deepest_signal.chosen_move
        stable = all(row.chosen_move == root_chosen for row in budget_evidence)
        reply_signal, reply_evidence, reply_nodes, reply_seconds = (
            self._reanalyse_replies(board, history, deepest_signal, candidates)
        )
        total_nodes += reply_nodes
        total_seconds += reply_seconds
        chosen = reply_signal.chosen_move
        self.evidence.append(
            HighDepthCommitteeDecisionEvidence(
                target_sfen=board.to_sfen(),
                teacher_proposals=proposal_rows,
                candidate_moves=candidates,
                budgets=tuple(budget_evidence),
                chosen_move=chosen,
                budget_choice_stable=stable,
                reply_reanalysis=reply_evidence,
                reported_nodes=total_nodes,
                elapsed_seconds=total_seconds,
            )
        )
        return MoveDecision(
            chosen,
            reply_signal.policy,
            reply_signal.chosen_conservative_value,
            nodes=total_nodes,
            elapsed_seconds=total_seconds,
            nps=(None if total_seconds <= 0.0 else total_nodes / total_seconds),
            source=self.source,
        )
