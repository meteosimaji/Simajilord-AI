"""Fair checkpoint arenas and direct games against external USI engines."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import count
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.types import Color, RepetitionState

from .checkpoint import load_checkpoint
from .compute_interlock import ComputeLeaseSettings, InterlockedEvaluator
from .config import SearchConfig
from .domain import GameRecord, PositionSample, Termination
from .evaluator import Evaluator
from .external_usi import ExternalUsiTeacher, UsiAnalysis, UsiPositionHistory
from .game import play_match
from .mcts import MCTS
from .model import MLXEvaluator
from .opening_suite import OpeningPosition


@dataclass(frozen=True, slots=True)
class MoveDecision:
    move: str
    policy: dict[str, float]
    value: float
    nodes: int | None = None
    elapsed_seconds: float | None = None
    nps: float | None = None
    source: str | None = None
    peak_tree_nodes: int | None = None
    tree_recycles: int = 0


class DirectPlayer(Protocol):
    def choose_move(self, board: Board) -> MoveDecision: ...


@runtime_checkable
class HistoryAwareDirectPlayer(Protocol):
    """Optional direct-player capability for decisions that depend on the game path."""

    def choose_move_with_history(
        self, board: Board, history: UsiPositionHistory
    ) -> MoveDecision: ...


class MctsPlayer:
    """A Meteo checkpoint acting directly through its own MCTS."""

    def __init__(self, evaluator: Evaluator, config: SearchConfig, *, seed: int = 0) -> None:
        self.search = MCTS(evaluator, config, seed=seed)

    def choose_move(self, board: Board) -> MoveDecision:
        result = self.search.search(board, add_root_noise=False)
        return MoveDecision(
            result.best_move,
            result.policy,
            result.root_value,
            nodes=result.simulations,
            elapsed_seconds=result.elapsed_seconds,
            nps=result.nodes_per_second,
            source="meteo",
            peak_tree_nodes=result.peak_tree_nodes,
            tree_recycles=result.tree_recycles,
        )


class ExternalUsiPlayer:
    """An external engine that selects one USI bestmove per actual game ply."""

    def __init__(self, engine: ExternalUsiTeacher) -> None:
        self.engine = engine

    def choose_move(self, board: Board) -> MoveDecision:
        return self._decision_from_analysis(board, self.engine.analyse(board))

    def choose_move_with_history(
        self, board: Board, history: UsiPositionHistory
    ) -> MoveDecision:
        """Use the actual game prefix for history-sensitive external engines."""

        if board.to_sfen() != history.target_sfen:
            raise ValueError(
                "live direct-game board does not match the supplied USI history target"
            )
        return self._decision_from_analysis(board, self.engine.analyse_with_history(history))

    def _decision_from_analysis(self, board: Board, analysis: UsiAnalysis) -> MoveDecision:
        legal = {move.to_usi() for move in board.legal_moves()}
        if analysis.bestmove in {"resign", "win"}:
            return MoveDecision(
                analysis.bestmove,
                {},
                -1.0 if analysis.bestmove == "resign" else 1.0,
                nodes=analysis.nodes,
                elapsed_seconds=analysis.elapsed_seconds,
                nps=analysis.nps,
                source=self.engine.policy.policy_id,
            )
        if analysis.bestmove not in legal:
            raise ValueError(
                f"external engine selected illegal move {analysis.bestmove} at {board.to_sfen()}"
            )
        policy = {move: 0.0 for move in legal}
        policy[analysis.bestmove] = 1.0
        calibrated = self.engine.target_from_analysis(board, analysis)
        return MoveDecision(
            analysis.bestmove,
            policy,
            calibrated.value,
            nodes=analysis.nodes,
            elapsed_seconds=analysis.elapsed_seconds,
            nps=analysis.nps,
            source=self.engine.policy.policy_id,
        )


@dataclass(frozen=True, slots=True)
class ArenaSummary:
    games: int
    wins: int
    draws: int
    losses: int
    incomplete_games: int
    score: float
    lower_95: float
    upper_95: float
    elo: float | None
    promoted: bool
    promotion_min_games: int
    promotion_lower_bound: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedClusterResult:
    """The two candidate-color results attached to one normalized opening."""

    index: int
    opening_sfen: str
    normalized_key: str
    opening_sha256: str
    candidate_black_point: float
    candidate_white_point: float
    score: float
    candidate_black_termination: str | None
    candidate_white_termination: str | None
    incomplete: bool


@dataclass(frozen=True, slots=True)
class PairedArenaSummary:
    """Opening-cluster statistics for two color-swapped games per position."""

    opening_pairs: int
    independent_opening_pairs: int
    games: int
    pair_wins: int
    pair_ties: int
    pair_losses: int
    incomplete_pairs: int
    incomplete_games: int
    repeated_opening_pairs: int
    training_arena_overlap_count: int
    legacy_single_opening: bool
    promotion_eligible: bool
    score: float
    cluster_bootstrap_lower_95: float
    cluster_bootstrap_upper_95: float
    ci_method: str
    confidence_level: float
    sign_flip_p_value_one_sided: float
    promoted: bool
    promotion_min_pairs: int
    promotion_lower_bound: float
    bootstrap_iterations: int
    seed: int
    promotion_blockers: tuple[str, ...]
    training_arena_overlap_keys: tuple[str, ...]
    cluster_results: tuple[PairedClusterResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PairedArenaSummary:
        fields: dict[str, Any] = dict(value)
        raw_clusters = fields.get("cluster_results", ())
        if not isinstance(raw_clusters, (list, tuple)):
            raise ValueError("paired arena cluster_results must be a list")
        fields["cluster_results"] = tuple(
            PairedClusterResult(**cluster) if isinstance(cluster, dict) else cluster
            for cluster in raw_clusters
        )
        fields["promotion_blockers"] = tuple(fields.get("promotion_blockers", ()))
        fields["training_arena_overlap_keys"] = tuple(fields.get("training_arena_overlap_keys", ()))
        return cls(**fields)


def _adjudicate(board: Board) -> tuple[int | None, Termination] | None:
    repetition = board.repetition_state()
    if repetition != RepetitionState.NONE:
        if repetition in (RepetitionState.WIN, RepetitionState.SUPERIOR):
            return board.turn.value, Termination.REPETITION
        if repetition in (RepetitionState.LOSE, RepetitionState.INFERIOR):
            return board.turn.opponent().value, Termination.REPETITION
        return None, Termination.REPETITION
    if board.can_declare_win():
        return board.turn.value, Termination.DECLARATION
    if board.is_mated() or not board.legal_moves():
        return board.turn.opponent().value, Termination.CHECKMATE
    return None


def play_direct_game(
    black: DirectPlayer,
    white: DirectPlayer,
    *,
    initial_sfen: str | None = None,
    max_plies: int | None = None,
) -> GameRecord:
    """Play direct move providers under the same legal terminal adjudication."""

    if max_plies is not None and max_plies < 1:
        raise ValueError("max_plies must be positive when configured")
    board = Board(initial_sfen) if initial_sfen else Board()
    starting_sfen = board.to_sfen()
    players = {Color.BLACK.value: black, Color.WHITE.value: white}
    moves: list[str] = []
    pending_samples: list[PositionSample] = []
    winner: int | None = None
    termination = Termination.MAX_PLIES
    plies = count() if max_plies is None else range(max_plies)
    for ply in plies:
        adjudication = _adjudicate(board)
        if adjudication is not None:
            winner, termination = adjudication
            break
        turn = board.turn.value
        player = players[turn]
        if isinstance(player, HistoryAwareDirectPlayer):
            history = UsiPositionHistory(
                initial_sfen=starting_sfen,
                moves=tuple(moves),
                target_sfen=board.to_sfen(),
            )
            decision = player.choose_move_with_history(board, history)
        else:
            decision = player.choose_move(board)
        if decision.move == "resign":
            winner = board.turn.opponent().value
            termination = Termination.RESIGNATION
            break
        if decision.move == "win":
            if not board.can_declare_win():
                raise ValueError("external engine claimed an illegal entering-king win")
            winner = turn
            termination = Termination.DECLARATION
            break
        move = Move.from_usi(decision.move)
        if not board.is_legal_move(move):
            raise ValueError(f"player selected illegal move {decision.move} at {board.to_sfen()}")
        pending_samples.append(
            PositionSample(
                sfen=board.to_sfen(),
                ply=ply,
                turn=turn,
                policy=decision.policy,
                root_value=decision.value,
                chosen_move=decision.move,
                actor_best_move=decision.move,
                actor_simulations=decision.nodes,
                actor_search_seconds=decision.elapsed_seconds,
                actor_nps=decision.nps,
                actor_source=decision.source,
                actor_peak_tree_nodes=decision.peak_tree_nodes,
                actor_tree_recycles=decision.tree_recycles,
            )
        )
        board.apply_move(move)
        moves.append(decision.move)
    else:
        termination = Termination.MAX_PLIES

    samples = tuple(
        replace(
            sample,
            value_target=(0.0 if winner is None else (1.0 if sample.turn == winner else -1.0)),
        )
        for sample in pending_samples
    )
    return GameRecord(starting_sfen, tuple(moves), samples, winner, termination)


def _score_interval(
    score: float, games: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Wilson interval over game points, with a draw represented as half a point."""

    if games < 1:
        return 0.0, 1.0
    denominator = 1.0 + z * z / games
    center = (score + z * z / (2.0 * games)) / denominator
    spread = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * games)) / games)
    spread /= denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def summarize_arena(
    results: list[float],
    *,
    incomplete_games: int = 0,
    promotion_min_games: int = 100,
    promotion_lower_bound: float = 0.5,
) -> ArenaSummary:
    if not results:
        raise ValueError("at least one arena result is required")
    if any(result not in (0.0, 0.5, 1.0) for result in results):
        raise ValueError("arena results must be losses, draws, or wins")
    if not 0 <= incomplete_games <= len(results):
        raise ValueError("incomplete_games is out of range")
    if promotion_min_games < 1 or not 0 <= promotion_lower_bound <= 1:
        raise ValueError("invalid promotion criteria")
    wins = results.count(1.0)
    draws = results.count(0.5)
    losses = results.count(0.0)
    score = sum(results) / len(results)
    lower, upper = _score_interval(score, len(results))
    elo = None if score in (0.0, 1.0) else 400.0 * math.log10(score / (1.0 - score))
    promoted = (
        len(results) >= promotion_min_games
        and incomplete_games == 0
        and lower > promotion_lower_bound
    )
    return ArenaSummary(
        games=len(results),
        wins=wins,
        draws=draws,
        losses=losses,
        incomplete_games=incomplete_games,
        score=score,
        lower_95=lower,
        upper_95=upper,
        elo=elo,
        promoted=promoted,
        promotion_min_games=promotion_min_games,
        promotion_lower_bound=promotion_lower_bound,
    )


def summarize_paired_arena(
    results_by_opening: Sequence[tuple[float, float]],
    *,
    incomplete_pairs: int = 0,
    incomplete_games: int | None = None,
    opening_sfens: Sequence[str] | None = None,
    opening_keys: Sequence[str] | None = None,
    opening_hashes: Sequence[str] | None = None,
    terminations: Sequence[tuple[str | None, str | None]] | None = None,
    incomplete_by_opening: Sequence[bool] | None = None,
    training_arena_overlap_keys: Sequence[str] = (),
    legacy_single_opening: bool = False,
    promotion_eligible: bool = True,
    promotion_min_pairs: int = 32,
    promotion_lower_bound: float = 0.5,
    bootstrap_iterations: int = 20_000,
    seed: int = 0,
) -> PairedArenaSummary:
    """Treat each color-swapped opening as one statistical cluster.

    The one-sided 95% percentile lower bound resamples independent opening
    clusters, never individual games. Repeated normalized keys are collapsed for
    statistics and are always a promotion blocker.

    The one-sided randomization test sign-flips each pair's deviation around
    0.5 and is recorded as a diagnostic; promotion is gated by the bootstrap
    lower bound and the explicit data-integrity checks.
    """

    if not results_by_opening:
        raise ValueError("at least one paired opening result is required")
    allowed = {0.0, 0.5, 1.0}
    if any(first not in allowed or second not in allowed for first, second in results_by_opening):
        raise ValueError("paired arena results must be losses, draws, or wins")
    pair_count = len(results_by_opening)
    if not 0 <= incomplete_pairs <= pair_count:
        raise ValueError("incomplete_pairs is out of range")
    if incomplete_games is None:
        incomplete_games = incomplete_pairs
    if not incomplete_pairs <= incomplete_games <= pair_count * 2:
        raise ValueError("incomplete_games is out of range")
    if promotion_min_pairs < 1 or not 0 <= promotion_lower_bound <= 1:
        raise ValueError("invalid paired promotion criteria")
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    if seed < 0:
        raise ValueError("paired arena seed must be non-negative")

    def _checked_metadata(
        values: Sequence[str] | None,
        *,
        label: str,
        default: Sequence[str],
    ) -> tuple[str, ...]:
        resolved = tuple(default if values is None else values)
        if len(resolved) != pair_count:
            raise ValueError(f"{label} must match paired arena result count")
        return resolved

    synthetic_keys = tuple(f"cluster-{index:06d}" for index in range(pair_count))
    keys = _checked_metadata(opening_keys, label="opening_keys", default=synthetic_keys)
    sfens = _checked_metadata(opening_sfens, label="opening_sfens", default=keys)
    if opening_sfens is not None and any(
        OpeningPosition.from_sfen(sfen).normalized_key != key
        for sfen, key in zip(sfens, keys, strict=True)
    ):
        raise ValueError("opening_sfens and opening_keys identify different positions")
    expected_hashes = tuple(hashlib.sha256(key.encode("utf-8")).hexdigest() for key in keys)
    hashes = _checked_metadata(
        opening_hashes,
        label="opening_hashes",
        default=expected_hashes,
    )
    if hashes != expected_hashes:
        raise ValueError("opening_hashes must be SHA-256 digests of opening_keys")
    resolved_terminations: tuple[tuple[str | None, str | None], ...]
    if terminations is None:
        resolved_terminations = tuple((None, None) for _ in range(pair_count))
    else:
        resolved_terminations = tuple(terminations)
        if len(resolved_terminations) != pair_count:
            raise ValueError("terminations must match paired arena result count")
    if incomplete_by_opening is None:
        incomplete_flags = tuple(index < incomplete_pairs for index in range(pair_count))
    else:
        incomplete_flags = tuple(incomplete_by_opening)
        if len(incomplete_flags) != pair_count:
            raise ValueError("incomplete_by_opening must match paired arena result count")
        if sum(incomplete_flags) != incomplete_pairs:
            raise ValueError("incomplete_by_opening disagrees with incomplete_pairs")

    cluster_results = tuple(
        PairedClusterResult(
            index=index,
            opening_sfen=sfens[index],
            normalized_key=keys[index],
            opening_sha256=hashes[index],
            candidate_black_point=first,
            candidate_white_point=second,
            score=(first + second) / 2.0,
            candidate_black_termination=resolved_terminations[index][0],
            candidate_white_termination=resolved_terminations[index][1],
            incomplete=incomplete_flags[index],
        )
        for index, (first, second) in enumerate(results_by_opening)
    )
    grouped_scores: dict[str, list[float]] = {}
    for cluster in cluster_results:
        grouped_scores.setdefault(cluster.normalized_key, []).append(cluster.score)
    independent_scores = np.asarray(
        [float(np.mean(scores)) for scores in grouped_scores.values()],
        dtype=np.float64,
    )
    independent_count = len(independent_scores)
    repeated_opening_pairs = pair_count - independent_count
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(bootstrap_iterations, dtype=np.float64)
    maximum_indices_per_chunk = 1_000_000
    chunk_size = max(1, min(bootstrap_iterations, maximum_indices_per_chunk // independent_count))
    for start in range(0, bootstrap_iterations, chunk_size):
        stop = min(start + chunk_size, bootstrap_iterations)
        sampled_indices = rng.integers(
            0,
            independent_count,
            size=(stop - start, independent_count),
        )
        bootstrap_means[start:stop] = independent_scores[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.05, 0.95))

    observed_difference = float(independent_scores.mean() - 0.5)
    deviations = independent_scores - 0.5
    exceedances = 0
    for start in range(0, bootstrap_iterations, chunk_size):
        stop = min(start + chunk_size, bootstrap_iterations)
        signs = rng.choice(
            np.asarray((-1.0, 1.0), dtype=np.float64),
            size=(stop - start, independent_count),
        )
        null_differences = (signs * deviations).mean(axis=1)
        exceedances += int(np.count_nonzero(null_differences >= observed_difference - 1e-15))
    p_value = (
        1.0 if observed_difference <= 0 else float((exceedances + 1) / (bootstrap_iterations + 1))
    )
    score = float(independent_scores.mean())
    overlap_keys = tuple(dict.fromkeys(training_arena_overlap_keys))
    blockers: list[str] = []
    if legacy_single_opening:
        blockers.append("legacy_single_opening_debug_only")
    elif not promotion_eligible:
        blockers.append("promotion_disabled")
    if repeated_opening_pairs:
        blockers.append("repeated_normalized_openings")
    if independent_count < promotion_min_pairs:
        blockers.append("insufficient_independent_opening_pairs")
    if incomplete_games:
        blockers.append("incomplete_games")
    if overlap_keys:
        blockers.append("training_arena_split_overlap")
    if float(lower) <= promotion_lower_bound:
        blockers.append("lower_bound_not_above_threshold")
    return PairedArenaSummary(
        opening_pairs=pair_count,
        independent_opening_pairs=independent_count,
        games=2 * pair_count,
        pair_wins=int(np.count_nonzero(independent_scores > 0.5)),
        pair_ties=int(np.count_nonzero(independent_scores == 0.5)),
        pair_losses=int(np.count_nonzero(independent_scores < 0.5)),
        incomplete_pairs=incomplete_pairs,
        incomplete_games=incomplete_games,
        repeated_opening_pairs=repeated_opening_pairs,
        training_arena_overlap_count=len(overlap_keys),
        legacy_single_opening=legacy_single_opening,
        promotion_eligible=promotion_eligible and not legacy_single_opening,
        score=score,
        cluster_bootstrap_lower_95=float(lower),
        cluster_bootstrap_upper_95=float(upper),
        ci_method="deterministic_percentile_cluster_bootstrap_one_sided",
        confidence_level=0.95,
        sign_flip_p_value_one_sided=p_value,
        promoted=not blockers,
        promotion_min_pairs=promotion_min_pairs,
        promotion_lower_bound=promotion_lower_bound,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        promotion_blockers=tuple(blockers),
        training_arena_overlap_keys=overlap_keys,
        cluster_results=cluster_results,
    )


def _candidate_point(record: GameRecord, candidate_color: int) -> float:
    if record.winner is None:
        return 0.5
    return 1.0 if record.winner == candidate_color else 0.0


def evaluate_checkpoint_pair(
    candidate_checkpoint: Path,
    champion_checkpoint: Path,
    search_config: SearchConfig,
    *,
    openings: Sequence[str],
    seed: int = 0,
    promotion_min_pairs: int = 32,
    promotion_lower_bound: float = 0.5,
    bootstrap_iterations: int = 20_000,
    allow_repeated_openings: bool = False,
    legacy_single_opening: bool = False,
    promotion_eligible: bool = True,
    training_arena_overlap_keys: Sequence[str] = (),
    promotion_min_games: int | None = None,
    compute_interlock: ComputeLeaseSettings | None = None,
) -> tuple[PairedArenaSummary, list[GameRecord]]:
    """Run every opening twice with colors swapped and no self-play noise."""

    if not openings:
        raise ValueError("at least one opening SFEN is required")
    if promotion_min_games is not None:
        if not legacy_single_opening:
            raise ValueError("promotion_min_games is supported only in legacy debug mode")
        if promotion_min_games < 1:
            raise ValueError("promotion_min_games must be positive when supplied")
        promotion_min_pairs = max(1, (promotion_min_games + 1) // 2)
    positions = tuple(OpeningPosition.from_sfen(opening) for opening in openings)
    keys = tuple(position.normalized_key for position in positions)
    if len(set(keys)) != len(keys) and not allow_repeated_openings:
        raise ValueError("duplicate normalized opening SFEN is not allowed in paired arena")
    candidate_model, _ = load_checkpoint(candidate_checkpoint)
    champion_model, _ = load_checkpoint(champion_checkpoint)
    candidate_evaluator = MLXEvaluator(candidate_model)
    champion_evaluator = MLXEvaluator(champion_model)
    candidate = (
        candidate_evaluator
        if compute_interlock is None
        else InterlockedEvaluator(candidate_evaluator, compute_interlock)
    )
    champion = (
        champion_evaluator
        if compute_interlock is None
        else InterlockedEvaluator(champion_evaluator, compute_interlock)
    )
    records: list[GameRecord] = []
    paired_points: list[tuple[float, float]] = []
    paired_terminations: list[tuple[str, str]] = []
    incomplete_flags: list[bool] = []
    for index, position in enumerate(positions):
        candidate_black = play_match(
            candidate,
            champion,
            search_config,
            search_config,
            initial_sfen=position.sfen,
            seed=seed + 2 * index,
            self_play_noise=False,
        )
        candidate_white = play_match(
            champion,
            candidate,
            search_config,
            search_config,
            initial_sfen=position.sfen,
            seed=seed + 2 * index + 1,
            self_play_noise=False,
        )
        records.extend((candidate_black, candidate_white))
        paired_points.append(
            (
                _candidate_point(candidate_black, Color.BLACK.value),
                _candidate_point(candidate_white, Color.WHITE.value),
            )
        )
        paired_terminations.append(
            (candidate_black.termination.value, candidate_white.termination.value)
        )
        incomplete_flags.append(
            candidate_black.termination == Termination.MAX_PLIES
            or candidate_white.termination == Termination.MAX_PLIES
        )
    incomplete_games = sum(record.termination == Termination.MAX_PLIES for record in records)
    return (
        summarize_paired_arena(
            paired_points,
            incomplete_pairs=sum(incomplete_flags),
            incomplete_games=incomplete_games,
            opening_sfens=tuple(position.sfen for position in positions),
            opening_keys=keys,
            opening_hashes=tuple(position.sha256 for position in positions),
            terminations=paired_terminations,
            incomplete_by_opening=incomplete_flags,
            training_arena_overlap_keys=training_arena_overlap_keys,
            legacy_single_opening=legacy_single_opening,
            promotion_eligible=promotion_eligible,
            promotion_min_pairs=promotion_min_pairs,
            promotion_lower_bound=promotion_lower_bound,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        records,
    )


def benchmark_checkpoint_vs_external(
    checkpoint: Path,
    external: ExternalUsiTeacher,
    search_config: SearchConfig,
    *,
    openings: Sequence[str],
    seed: int = 0,
    promotion_min_pairs: int = 32,
    promotion_lower_bound: float = 0.5,
    bootstrap_iterations: int = 20_000,
    legacy_single_opening: bool = False,
    promotion_eligible: bool = True,
    compute_interlock: ComputeLeaseSettings | None = None,
) -> tuple[PairedArenaSummary, list[GameRecord]]:
    """Benchmark Meteo against a USI opponent as color-swapped opening clusters.

    Each normalized opening is one independent observation containing a Meteo-
    black game and a Meteo-white game.  Repeating an opening is rejected rather
    than silently inflating either confidence or an Elo estimate.
    """

    if not openings:
        raise ValueError("at least one opening SFEN is required")
    if legacy_single_opening and len(openings) != 1:
        raise ValueError("legacy external benchmark accepts exactly one debug opening")
    positions = tuple(OpeningPosition.from_sfen(opening) for opening in openings)
    keys = tuple(position.normalized_key for position in positions)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate normalized opening SFEN is not allowed in USI benchmark")
    model, _ = load_checkpoint(checkpoint)
    base_evaluator = MLXEvaluator(model)
    evaluator = (
        base_evaluator
        if compute_interlock is None
        else InterlockedEvaluator(base_evaluator, compute_interlock)
    )
    opponent = ExternalUsiPlayer(external)
    records: list[GameRecord] = []
    paired_points: list[tuple[float, float]] = []
    paired_terminations: list[tuple[str, str]] = []
    incomplete_flags: list[bool] = []
    for index, position in enumerate(positions):
        meteo_black_player = MctsPlayer(evaluator, search_config, seed=seed + 2 * index)
        external.new_game()
        meteo_black = play_direct_game(
            meteo_black_player,
            opponent,
            initial_sfen=position.sfen,
            max_plies=search_config.max_plies,
        )
        meteo_white_player = MctsPlayer(evaluator, search_config, seed=seed + 2 * index + 1)
        external.new_game()
        meteo_white = play_direct_game(
            opponent,
            meteo_white_player,
            initial_sfen=position.sfen,
            max_plies=search_config.max_plies,
        )
        records.extend((meteo_black, meteo_white))
        paired_points.append(
            (
                _candidate_point(meteo_black, Color.BLACK.value),
                _candidate_point(meteo_white, Color.WHITE.value),
            )
        )
        paired_terminations.append(
            (meteo_black.termination.value, meteo_white.termination.value)
        )
        incomplete_flags.append(
            meteo_black.termination == Termination.MAX_PLIES
            or meteo_white.termination == Termination.MAX_PLIES
        )
    incomplete_games = sum(record.termination == Termination.MAX_PLIES for record in records)
    return (
        summarize_paired_arena(
            paired_points,
            incomplete_pairs=sum(incomplete_flags),
            incomplete_games=incomplete_games,
            opening_sfens=tuple(position.sfen for position in positions),
            opening_keys=keys,
            opening_hashes=tuple(position.sha256 for position in positions),
            terminations=paired_terminations,
            incomplete_by_opening=incomplete_flags,
            legacy_single_opening=legacy_single_opening,
            promotion_eligible=promotion_eligible,
            promotion_min_pairs=promotion_min_pairs,
            promotion_lower_bound=promotion_lower_bound,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        records,
    )
