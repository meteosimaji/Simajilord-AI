"""Deep same-network teacher search and policy-reversal mining."""

from __future__ import annotations

from dataclasses import replace

from rsshogi.core import Board, Move

from .config import ReanalysisConfig, SearchConfig
from .domain import GameRecord, PositionSample
from .evaluator import Evaluator
from .external_usi import ExternalUsiTeacher, UsiPositionHistory
from .mcts import MCTS


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

    selected = _select_reanalysis_indices(game, config)
    if not selected:
        return game
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
    search = MCTS(evaluator, teacher_config, seed=seed)
    revised: list[PositionSample] = []
    for index, sample in enumerate(game.samples):
        if index not in selected:
            revised.append(sample)
            continue
        teacher = search.search(Board(sample.sfen), add_root_noise=False)
        actor_reference = _actor_reference_move(sample)
        teacher_chosen_q = (
            teacher.q_values.get(sample.chosen_move) if sample.chosen_move is not None else None
        )
        revised.append(
            replace(
                sample,
                teacher_policy=teacher.policy,
                teacher_value=teacher.root_value,
                policy_reversal=teacher.best_move != actor_reference,
                discovery_simulation=teacher.discovery_simulation,
                teacher_best_move=teacher.best_move,
                teacher_nodes=teacher.simulations,
                teacher_depth_ratio=(
                    teacher.simulations / sample.actor_simulations
                    if sample.actor_simulations
                    else None
                ),
                teacher_source="meteo-self-deep",
                teacher_context="same-position-deep",
                teacher_regret=(
                    None
                    if teacher_chosen_q is None
                    else max(
                        0.0,
                        teacher.q_values[teacher.best_move] - teacher_chosen_q,
                    )
                ),
            )
        )
    return replace(game, samples=tuple(revised))


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
        chosen_value = (
            move_values.get(sample.chosen_move) if sample.chosen_move is not None else None
        )
        regret_is_lower_bound = False
        if best_value is not None and chosen_value is None and move_values:
            chosen_value = min(move_values.values())
            regret_is_lower_bound = True
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
            )
        )
    return replace(game, samples=tuple(revised))


def sample_weight(sample: PositionSample, config: ReanalysisConfig) -> float:
    reversal = config.reversal_priority if sample.policy_reversal else 1.0
    regret = 1.0 + 3.0 * max(0.0, sample.teacher_regret or 0.0)
    return reversal * regret
