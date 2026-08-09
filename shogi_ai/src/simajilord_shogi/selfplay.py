"""Parallel process self-play orchestration."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.types import RepetitionState

from .checkpoint import load_checkpoint
from .compute_interlock import ComputeLeaseSettings, InterlockedEvaluator
from .config import SearchConfig
from .domain import GameRecord, PositionSample, Termination
from .game import play_game
from .mcts import MCTS, choose_move
from .model import MLXEvaluator


@dataclass(frozen=True, slots=True)
class SelfPlayJob:
    checkpoint: str
    search: SearchConfig
    seed: int
    compute_interlock: ComputeLeaseSettings | None = None


def _run_job(job: SelfPlayJob) -> GameRecord:
    model, _ = load_checkpoint(Path(job.checkpoint))
    evaluator = MLXEvaluator(model)
    protected_evaluator = (
        evaluator
        if job.compute_interlock is None
        else InterlockedEvaluator(evaluator, job.compute_interlock)
    )
    return play_game(
        protected_evaluator,
        job.search,
        seed=job.seed,
        self_play_noise=True,
    )


def parallel_self_play(
    checkpoint: Path,
    search: SearchConfig,
    *,
    games: int,
    workers: int,
    seed: int = 0,
    compute_interlock: ComputeLeaseSettings | None = None,
) -> list[GameRecord]:
    if games < 1 or workers < 1:
        raise ValueError("games and workers must be positive")
    jobs = [
        SelfPlayJob(
            str(checkpoint.resolve()),
            search,
            seed + index,
            compute_interlock,
        )
        for index in range(games)
    ]
    if workers == 1:
        return [_run_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_run_job, jobs))


def batched_self_play(
    checkpoint: Path,
    search_config: SearchConfig,
    *,
    games: int,
    seed: int = 0,
    initial_sfens: list[str] | None = None,
    compute_interlock: ComputeLeaseSettings | None = None,
) -> list[GameRecord]:
    """Advance many games together and batch every neural leaf evaluation."""

    if games < 1:
        raise ValueError("games must be positive")
    model, _ = load_checkpoint(checkpoint)
    evaluator = MLXEvaluator(model)
    protected_evaluator = (
        evaluator
        if compute_interlock is None
        else InterlockedEvaluator(evaluator, compute_interlock)
    )
    search = MCTS(protected_evaluator, search_config, seed=seed)
    if initial_sfens is not None and len(initial_sfens) != games:
        raise ValueError("initial_sfens length must equal games")
    boards = (
        [Board(sfen) for sfen in initial_sfens]
        if initial_sfens is not None
        else [Board() for _ in range(games)]
    )
    initial_sfens = [board.to_sfen() for board in boards]
    rngs = [np.random.default_rng(seed + index) for index in range(games)]
    move_lists: list[list[str]] = [[] for _ in range(games)]
    sample_lists: list[list[PositionSample]] = [[] for _ in range(games)]
    winners: list[int | None] = [None] * games
    terminations: list[Termination | None] = [None] * games

    plies = count() if search_config.max_plies is None else range(search_config.max_plies)
    for ply in plies:
        active: list[int] = []
        for index, board in enumerate(boards):
            if terminations[index] is not None:
                continue
            adjudication = _adjudicate(board)
            if adjudication is not None:
                winners[index], terminations[index] = adjudication
            else:
                active.append(index)
        if not active:
            break

        results = search.search_many([boards[index] for index in active], add_root_noise=True)
        for index, result in zip(active, results, strict=True):
            board = boards[index]
            if (
                search_config.resign_threshold is not None
                and ply >= search_config.resign_min_ply
                and result.root_value <= search_config.resign_threshold
            ):
                winners[index] = board.turn.opponent().value
                terminations[index] = Termination.RESIGNATION
                continue
            temperature = (
                search_config.temperature if ply < search_config.temperature_moves else 0.0
            )
            move_usi = choose_move(result, temperature=temperature, rng=rngs[index])
            sample_lists[index].append(
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
                )
            )
            move = Move.from_usi(move_usi)
            if not board.is_legal_move(move):
                raise AssertionError(f"batched search selected illegal move {move_usi}")
            board.apply_move(move)
            move_lists[index].append(move_usi)

    for index, board in enumerate(boards):
        if terminations[index] is None:
            adjudication = _adjudicate(board)
            if adjudication is None:
                terminations[index] = Termination.MAX_PLIES
            else:
                winners[index], terminations[index] = adjudication

    records: list[GameRecord] = []
    for index in range(games):
        winner = winners[index]
        samples = tuple(
            replace(
                sample,
                value_target=(0.0 if winner is None else (1.0 if sample.turn == winner else -1.0)),
            )
            for sample in sample_lists[index]
        )
        termination = terminations[index]
        if termination is None:
            raise AssertionError("self-play game was not adjudicated")
        records.append(
            GameRecord(
                initial_sfen=initial_sfens[index],
                moves=tuple(move_lists[index]),
                samples=samples,
                winner=winner,
                termination=termination,
            )
        )
    return records


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
