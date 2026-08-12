"""Complete legal-game runner shared by CLI, Discord adapter, and self-play."""

from __future__ import annotations

from dataclasses import replace
from itertools import count

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.types import Color

from .adjudication import adjudicate_board
from .config import SearchConfig
from .domain import GameRecord, PositionSample, Termination
from .evaluator import Evaluator
from .mcts import MCTS, choose_move
from .multi_objective_distillation import all_legal_value_policy


def play_game(
    evaluator: Evaluator,
    config: SearchConfig,
    *,
    initial_sfen: str | None = None,
    seed: int | None = None,
    self_play_noise: bool = True,
) -> GameRecord:
    """Play until a rules result, resignation, or the configured safety draw."""

    return play_match(
        evaluator,
        evaluator,
        config,
        config,
        initial_sfen=initial_sfen,
        seed=seed,
        self_play_noise=self_play_noise,
    )


def play_match(
    black_evaluator: Evaluator,
    white_evaluator: Evaluator,
    black_config: SearchConfig,
    white_config: SearchConfig,
    *,
    initial_sfen: str | None = None,
    seed: int | None = None,
    self_play_noise: bool = False,
) -> GameRecord:
    """Play arbitrary actors, including shallow-vs-deep same-model matches."""

    board = Board(initial_sfen) if initial_sfen else Board()
    starting_sfen = board.to_sfen()
    rng = np.random.default_rng(seed)
    searches = {
        Color.BLACK.value: MCTS(black_evaluator, black_config, seed=seed),
        Color.WHITE.value: MCTS(
            white_evaluator, white_config, seed=None if seed is None else seed + 1
        ),
    }
    configs = {Color.BLACK.value: black_config, Color.WHITE.value: white_config}
    moves: list[str] = []
    pending_samples: list[PositionSample] = []
    winner: int | None = None
    termination = Termination.MAX_PLIES

    configured_limits = [
        limit for limit in (black_config.max_plies, white_config.max_plies) if limit is not None
    ]
    max_plies = max(configured_limits) if len(configured_limits) == 2 else None
    plies = count() if max_plies is None else range(max_plies)
    for ply in plies:
        adjudication = adjudicate_board(board)
        if adjudication is not None:
            winner = adjudication.winner
            termination = adjudication.termination
            break

        side_config = configs[board.turn.value]
        result = searches[board.turn.value].search(board, add_root_noise=self_play_noise)
        if (
            side_config.resign_threshold is not None
            and ply >= side_config.resign_min_ply
            and result.root_value <= side_config.resign_threshold
        ):
            winner = board.turn.opponent().value
            termination = Termination.RESIGNATION
            break

        temperature = side_config.temperature if ply < side_config.temperature_moves else 0.0
        move_usi = choose_move(result, temperature=temperature, rng=rng)
        implicit_target = all_legal_value_policy(
            board,
            result.q_values,
            result.root_visits,
            temperature=side_config.implicit_policy_temperature,
        )
        pending_samples.append(
            PositionSample(
                sfen=board.to_sfen(),
                ply=ply,
                turn=board.turn.value,
                policy=result.policy,
                root_value=result.root_value,
                discovery_simulation=result.discovery_simulation,
                chosen_move=move_usi,
                actor_best_move=result.best_move,
                actor_regret=max(
                    0.0,
                    result.q_values[result.best_move] - result.q_values[move_usi],
                ),
                actor_simulations=result.simulations,
                actor_search_seconds=result.elapsed_seconds,
                actor_nps=result.nodes_per_second,
                actor_source="meteo",
                actor_peak_tree_nodes=result.peak_tree_nodes,
                actor_tree_recycles=result.tree_recycles,
                actor_move_values=implicit_target.move_values,
                actor_move_visits=implicit_target.move_visits,
                actor_implicit_policy=(implicit_target.policy or None),
                actor_proven_mate_moves=implicit_target.proven_mate_moves,
            )
        )
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise AssertionError(f"search selected illegal move {move_usi} at {board.to_sfen()}")
        board.apply_move(move)
        moves.append(move_usi)
    else:
        termination = Termination.MAX_PLIES

    samples = tuple(
        replace(
            sample,
            value_target=(0.0 if winner is None else (1.0 if sample.turn == winner else -1.0)),
        )
        for sample in pending_samples
    )
    return GameRecord(
        initial_sfen=starting_sfen,
        moves=tuple(moves),
        samples=samples,
        winner=winner,
        termination=termination,
    )


def replay_and_validate(record: GameRecord) -> Board:
    """Replay every move and reject corrupted or illegal training records."""

    board = Board(record.initial_sfen)
    for ply, move_usi in enumerate(record.moves):
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(f"illegal move at ply {ply}: {move_usi}")
        board.apply_move(move)
    return board


def winner_name(winner: int | None) -> str:
    if winner is None:
        return "draw"
    return "black" if winner == Color.BLACK.value else "white"
