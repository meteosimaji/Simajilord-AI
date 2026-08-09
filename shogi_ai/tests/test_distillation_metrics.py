from __future__ import annotations

import math
from dataclasses import replace

import pytest
from rsshogi.core import Board

from simajilord_shogi.distillation_metrics import (
    evaluate_teacher_alignment,
    normalized_position_key,
)
from simajilord_shogi.domain import PositionSample
from simajilord_shogi.encoding import HistoryInput, board_from_history_input
from simajilord_shogi.evaluator import Evaluation

from .test_mcts_game import MATE_IN_ONE_SFEN


class _FixedEvaluator:
    def evaluate(self, board: Board) -> Evaluation:
        return self.evaluate_batch([board])[0]

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        evaluations: list[Evaluation] = []
        for board in boards:
            legal = [move.to_usi() for move in board.legal_moves()]
            policy = {move: 0.0 for move in legal}
            policy["G*5b"] = 0.5
            policy["6c7d"] = 0.5
            evaluations.append(Evaluation(policy=policy, value=0.5))
        return evaluations


class _HistoryFixedEvaluator:
    def evaluate(self, board: Board) -> Evaluation:
        raise AssertionError("history-aware evaluation must not use a bare board")

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        raise AssertionError("history-aware evaluation must not use bare boards")

    def evaluate_history_batch(self, histories: list[HistoryInput]) -> list[Evaluation]:
        results: list[Evaluation] = []
        for history in histories:
            board = board_from_history_input(history)
            legal = [move.to_usi() for move in board.legal_moves()]
            results.append(
                Evaluation(
                    policy={move: 1.0 / len(legal) for move in legal},
                    value=0.0,
                )
            )
        return results


def _sample() -> PositionSample:
    return PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=0.0,
        teacher_policy={"G*5b": 0.6, "6c7d": 0.4},
        teacher_value=1.0,
        teacher_source="teacher-a",
    )


def test_alignment_scores_full_teacher_distribution_and_deduplicates() -> None:
    sample = _sample()
    duplicate_with_new_counter = replace(sample, sfen=MATE_IN_ONE_SFEN.rsplit(" ", 1)[0] + " 9")

    report = evaluate_teacher_alignment(
        _FixedEvaluator(), [sample, duplicate_with_new_counter], batch_size=1
    )
    metrics = report.overall

    assert metrics.samples == 1
    assert metrics.unique_positions == 1
    assert metrics.policy_cross_entropy == pytest.approx(-math.log(0.5))
    assert 0 < metrics.policy_js_divergence < math.log(2)
    assert metrics.teacher_mass_at_1 == pytest.approx(0.6) or (
        metrics.teacher_mass_at_1 == pytest.approx(0.4)
    )
    assert metrics.teacher_mass_at_3 == pytest.approx(1.0)
    assert metrics.teacher_mass_at_5 == pytest.approx(1.0)
    assert metrics.teacher_best_top_3 == 1.0
    assert metrics.value_mse == pytest.approx(0.25)
    assert metrics.value_brier == pytest.approx(0.0625)
    assert report.by_teacher["teacher-a"] == metrics
    assert report.by_phase["opening"] == metrics


def test_alignment_rejects_conflicting_duplicate_from_same_teacher() -> None:
    sample = _sample()
    conflicting = replace(sample, teacher_policy={"G*5b": 1.0})

    with pytest.raises(ValueError, match="conflicting duplicate"):
        evaluate_teacher_alignment(_FixedEvaluator(), [sample, conflicting])


def test_history_alignment_keeps_same_sfen_different_prefixes_independent() -> None:
    first_moves = ("7g7f", "3c3d", "2g2f", "8c8d")
    second_moves = ("2g2f", "8c8d", "7g7f", "3c3d")
    first_board = Board()
    second_board = Board()
    for move_usi in first_moves:
        first_board.apply_usi(move_usi)
    for move_usi in second_moves:
        second_board.apply_usi(move_usi)
    assert first_board.to_sfen() == second_board.to_sfen()
    legal_move = first_board.legal_moves()[0].to_usi()
    sample = PositionSample(
        sfen=first_board.to_sfen(),
        ply=4,
        turn=first_board.turn.value,
        policy={legal_move: 1.0},
        root_value=0.0,
        teacher_policy={legal_move: 1.0},
        teacher_value=0.0,
        teacher_source="teacher-a",
    )
    histories = [
        HistoryInput(Board().to_sfen(), first_moves, first_board.to_sfen()),
        HistoryInput(Board().to_sfen(), second_moves, second_board.to_sfen()),
    ]

    report = evaluate_teacher_alignment(
        _HistoryFixedEvaluator(),
        [sample, sample],
        histories=histories,
        batch_size=2,
    )

    assert report.overall.samples == 2
    assert report.overall.unique_positions == 2


def test_history_alignment_rejects_sample_target_mismatch() -> None:
    board = Board()
    move = board.legal_moves()[0].to_usi()
    sample = PositionSample(
        sfen=board.to_sfen().rsplit(" ", 1)[0] + " 100",
        ply=100,
        turn=board.turn.value,
        policy={move: 1.0},
        root_value=0.0,
        teacher_policy={move: 1.0},
        teacher_value=0.0,
        teacher_source="teacher-a",
    )
    history = HistoryInput(
        initial_sfen=board.to_sfen(),
        moves=(),
        target_sfen=board.to_sfen(),
    )

    with pytest.raises(ValueError, match=r"history target SFEN.*evaluation sample"):
        evaluate_teacher_alignment(
            _HistoryFixedEvaluator(),
            [sample],
            histories=[history],
        )


def test_normalized_position_key_ignores_only_move_counter() -> None:
    assert normalized_position_key(MATE_IN_ONE_SFEN) == normalized_position_key(
        MATE_IN_ONE_SFEN.rsplit(" ", 1)[0] + " 123"
    )

    with pytest.raises(ValueError, match="field count"):
        normalized_position_key("not sfen")
