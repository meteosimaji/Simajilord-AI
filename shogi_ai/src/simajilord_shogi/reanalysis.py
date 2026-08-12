"""Deep same-network teacher search and policy-reversal mining."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from rsshogi.core import Board, Move

from .config import ReanalysisConfig, SearchConfig
from .domain import GameRecord, PositionSample, TeacherScoreBound, TeacherVariation
from .evaluator import Evaluator
from .external_usi import ExternalUsiTeacher, UsiPositionHistory
from .mcts import MCTS, SearchResult
from .multi_objective_distillation import all_legal_value_policy


def _actor_reference_move(sample: PositionSample) -> str:
    """Return observed actor evidence without upgrading it to teacher truth."""

    if sample.actor_best_move is not None:
        return sample.actor_best_move
    if sample.chosen_move is not None:
        return sample.chosen_move
    if sample.policy:
        return max(sample.policy, key=sample.policy.__getitem__)
    raise ValueError("actor sample has neither policy, chosen move, nor best move")


def _uncertainty(sample: PositionSample) -> float:
    ordered = sorted(sample.policy.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return 1.0 - (ordered[0] - ordered[1])


def _select_reanalysis_indices(game: GameRecord, config: ReanalysisConfig) -> set[int]:
    if not game.samples or config.reanalyse_fraction == 0:
        return set()
    count = max(1, round(len(game.samples) * config.reanalyse_fraction))
    eligible = [
        index
        for index, sample in enumerate(game.samples)
        if _uncertainty(sample) >= config.uncertainty_threshold
    ]
    if len(eligible) < count:
        eligible = list(range(len(game.samples)))
    ranked = sorted(
        eligible,
        key=lambda index: (_uncertainty(game.samples[index]), -index),
        reverse=True,
    )[:count]
    return set(ranked)


def _is_tactical_sample(sample: PositionSample) -> bool:
    """Select forcing actor moves without consulting future teacher output."""

    board = Board(sample.sfen)
    if board.is_in_check():
        return True
    move_usi = sample.chosen_move or sample.actor_best_move
    if move_usi is None:
        return False
    move = Move.from_usi(move_usi)
    if not board.is_legal_move(move):
        return False
    is_capture = move.is_normal() and not board.is_square_empty(move.to_sq)
    if is_capture or move.is_promotion():
        return True
    board.apply_move(move)
    return board.is_in_check()


def _select_external_indices(
    game: GameRecord,
    config: ReanalysisConfig,
    selection: str,
) -> set[int]:
    if selection == "uncertainty":
        return _select_reanalysis_indices(game, config)
    if selection not in {"all", "tactical"}:
        raise ValueError("external reanalysis selection must be uncertainty, tactical, or all")
    if not game.samples or config.reanalyse_fraction == 0:
        return set()
    eligible = [
        index
        for index, sample in enumerate(game.samples)
        if selection == "all" or _is_tactical_sample(sample)
    ]
    count = min(len(eligible), max(1, round(len(game.samples) * config.reanalyse_fraction)))
    return set(
        sorted(
            eligible,
            key=lambda index: (_uncertainty(game.samples[index]), -index),
            reverse=True,
        )[:count]
    )


def reanalyse_game(
    game: GameRecord,
    evaluator: Evaluator,
    actor_search: SearchConfig,
    config: ReanalysisConfig,
    *,
    seed: int | None = None,
) -> GameRecord:
    """Replace selected actor targets with a much deeper same-model search.

    Positions are selected deterministically by uncertainty and a spread across
    the game. A best-move rank reversal is retained explicitly for prioritized
    replay, which trains the network to predict discoveries made only at depth.
    """

    return reanalyse_games(
        (game,),
        evaluator,
        actor_search,
        config,
        seed=seed,
        max_parallel_positions=1,
    )[0]


def reanalyse_games(
    games: Sequence[GameRecord],
    evaluator: Evaluator,
    actor_search: SearchConfig,
    config: ReanalysisConfig,
    *,
    seed: int | None = None,
    max_parallel_positions: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[GameRecord]:
    """Deep-reanalyse independent positions in bounded shared neural batches.

    Each root retains the same PUCT budget and has no cross-root statistics.  The
    only shared work is the neural forward pass, so batching does not turn one
    game's visits into evidence for another game.  Chunks bound Python-tree and
    unified-memory growth while avoiding the old scalar-GPU reanalysis path.
    """

    if max_parallel_positions < 1:
        raise ValueError("max_parallel_positions must be positive")
    if not games:
        return []
    teacher_simulations = max(
        config.minimum_teacher_simulations,
        actor_search.simulations * config.teacher_simulation_multiplier,
    )
    teacher_config = replace(
        actor_search,
        simulations=teacher_simulations,
        dirichlet_fraction=0.0,
        temperature=0.0,
        root_min_visits=max(1, actor_search.root_min_visits),
    )
    selected_by_game = [_select_reanalysis_indices(game, config) for game in games]
    tasks: list[tuple[int, int, PositionSample, Board]] = []
    for game_index, (game, selected) in enumerate(
        zip(games, selected_by_game, strict=True)
    ):
        for sample_index in sorted(selected):
            sample = game.samples[sample_index]
            history = UsiPositionHistory.from_game(game, sample)
            tasks.append((game_index, sample_index, sample, history.target_board()))
    if not tasks:
        return list(games)

    replacements: dict[tuple[int, int], PositionSample] = {}
    for start in range(0, len(tasks), max_parallel_positions):
        chunk = tasks[start : start + max_parallel_positions]
        chunk_seed = None if seed is None else seed + start
        search = MCTS(evaluator, teacher_config, seed=chunk_seed)
        results = search.search_many(
            [board for _game_index, _sample_index, _sample, board in chunk],
            add_root_noise=False,
        )
        for (game_index, sample_index, sample, board), teacher in zip(
            chunk, results, strict=True
        ):
            replacements[(game_index, sample_index)] = _deep_teacher_sample(
                sample,
                board,
                teacher,
                teacher_config=teacher_config,
            )
        if progress_callback is not None:
            progress_callback(min(start + len(chunk), len(tasks)), len(tasks))

    revised_games: list[GameRecord] = []
    for game_index, game in enumerate(games):
        revised_games.append(
            replace(
                game,
                samples=tuple(
                    replacements.get((game_index, sample_index), sample)
                    for sample_index, sample in enumerate(game.samples)
                ),
            )
        )
    return revised_games


def _deep_teacher_sample(
    sample: PositionSample,
    board: Board,
    teacher: SearchResult,
    *,
    teacher_config: SearchConfig,
) -> PositionSample:
    """Attach one completed same-model deep-search result to its source sample."""

    result = teacher
    implicit_target = all_legal_value_policy(
        board,
        result.q_values,
        result.root_visits,
        temperature=teacher_config.implicit_policy_temperature,
    )
    actor_reference = _actor_reference_move(sample)
    teacher_chosen_q = (
        result.q_values.get(sample.chosen_move) if sample.chosen_move is not None else None
    )
    return replace(
        sample,
        teacher_policy=result.policy,
        teacher_value=result.root_value,
        policy_reversal=result.best_move != actor_reference,
        discovery_simulation=result.discovery_simulation,
        teacher_best_move=result.best_move,
        teacher_nodes=result.simulations,
        teacher_depth_ratio=(
            result.simulations / sample.actor_simulations
            if sample.actor_simulations
            else None
        ),
        teacher_source="meteo-self-deep",
        teacher_context="same-position-deep",
        teacher_move_values=implicit_target.move_values,
        teacher_move_visits=implicit_target.move_visits,
        teacher_implicit_policy=(implicit_target.policy or None),
        teacher_proven_mate_moves=implicit_target.proven_mate_moves,
        teacher_regret=(
            None
            if teacher_chosen_q is None
            else max(
                0.0,
                result.q_values[result.best_move] - teacher_chosen_q,
            )
        ),
    )


def reanalyse_game_external(
    game: GameRecord,
    teacher: ExternalUsiTeacher,
    config: ReanalysisConfig,
    *,
    selection: str = "uncertainty",
    teacher_context: str | None = None,
) -> GameRecord:
    """Attach direct MultiPV targets from a rights-approved external USI teacher."""

    selected = _select_external_indices(game, config, selection)
    if not selected:
        return game
    revised: list[PositionSample] = []
    for index, sample in enumerate(game.samples):
        if index not in selected:
            revised.append(sample)
            continue
        history = UsiPositionHistory.from_game(game, sample)
        target = teacher.training_target_with_history(history)
        actor_reference = _actor_reference_move(sample)
        move_values = dict(target.move_values)
        best_value = move_values.get(target.bestmove)
        best_variation = next(
            (
                variation
                for variation in target.candidates
                if variation.move == target.bestmove
            ),
            None,
        )
        played_target = (
            None
            if sample.chosen_move is None
            else teacher.training_target_with_history_searchmoves(
                history,
                (sample.chosen_move,),
            )
        )
        chosen_value = None if played_target is None else played_target.value
        played_variation: TeacherVariation | None = (
            None
            if played_target is None or sample.chosen_move is None
            else next(
                (
                    variation
                    for variation in played_target.candidates
                    if variation.move == sample.chosen_move
                ),
                None,
            )
        )
        played_move_exact = (
            played_variation is not None
            and played_variation.bound is TeacherScoreBound.EXACT
        )
        best_move_exact = (
            best_variation is not None
            and best_variation.bound is TeacherScoreBound.EXACT
        )
        regret_is_lower_bound = (
            best_value is not None
            and chosen_value is not None
            and not (best_move_exact and played_move_exact)
        )
        regret = (
            None
            if best_value is None or chosen_value is None
            else max(0.0, best_value - chosen_value)
        )
        revised.append(
            replace(
                sample,
                teacher_policy=target.policy,
                teacher_value=target.value,
                policy_reversal=target.bestmove != actor_reference,
                teacher_best_move=target.bestmove,
                teacher_regret=regret,
                teacher_regret_is_lower_bound=regret_is_lower_bound,
                teacher_nodes=target.nodes,
                teacher_depth_ratio=(
                    target.nodes / sample.actor_simulations
                    if target.nodes is not None and sample.actor_simulations
                    else None
                ),
                teacher_time_ms=target.time_ms,
                teacher_nps=target.nps,
                teacher_depth=target.depth,
                teacher_source=teacher.policy.policy_id,
                teacher_context=teacher_context or selection,
                teacher_variations=target.candidates,
                teacher_policy_temperature=target.policy_temperature,
                teacher_value_scale=target.value_scale,
                teacher_played_move_value=chosen_value,
                teacher_played_move_nodes=(
                    None if played_target is None else played_target.nodes
                ),
                teacher_played_move_time_ms=(
                    None if played_target is None else played_target.time_ms
                ),
                teacher_played_move_nps=(
                    None if played_target is None else played_target.nps
                ),
                teacher_played_move_depth=(
                    None if played_target is None else played_target.depth
                ),
                teacher_played_move_exact=played_move_exact,
                teacher_played_move_variation=played_variation,
            )
        )
    return replace(game, samples=tuple(revised))


def sample_weight(sample: PositionSample, config: ReanalysisConfig) -> float:
    reversal = config.reversal_priority if sample.policy_reversal else 1.0
    regret = 1.0 + 3.0 * max(0.0, sample.teacher_regret or 0.0)
    return reversal * regret
