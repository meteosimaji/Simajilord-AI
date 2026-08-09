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


def test_normalized_position_key_ignores_only_move_counter() -> None:
    assert normalized_position_key(MATE_IN_ONE_SFEN) == normalized_position_key(
        MATE_IN_ONE_SFEN.rsplit(" ", 1)[0] + " 123"
    )

    with pytest.raises(ValueError, match="field count"):
        normalized_position_key("not sfen")
