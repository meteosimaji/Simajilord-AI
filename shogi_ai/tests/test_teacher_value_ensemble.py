from __future__ import annotations

import math

import pytest

from simajilord_shogi.teacher_value_ensemble import (
    TeacherValueEnsembleMode,
    TeacherWinProbability,
    combine_teacher_win_probabilities,
    cp_to_win_probability,
    win_probability_to_cp,
)


def test_probability_and_score_space_ensembles_are_explicitly_distinct() -> None:
    teachers = (
        TeacherWinProbability("ryfamate", 0.90),
        TeacherWinProbability("dl-suisho", 0.60),
        TeacherWinProbability("aobazero", 0.55),
    )

    probability_mean = combine_teacher_win_probabilities(
        teachers,
        mode=TeacherValueEnsembleMode.PROBABILITY_ARITHMETIC_MEAN,
        ponanza_coefficient=600.0,
    )
    converted_score_mean = combine_teacher_win_probabilities(
        teachers,
        mode=TeacherValueEnsembleMode.CONVERTED_SCORE_ARITHMETIC_MEAN,
        ponanza_coefficient=600.0,
    )

    assert probability_mean.output_probability == pytest.approx((0.90 + 0.60 + 0.55) / 3)
    assert converted_score_mean.output_probability != pytest.approx(
        probability_mean.output_probability
    )
    assert probability_mean.suisho11_formula_claimed is False
    assert converted_score_mean.suisho11_formula_claimed is False


def test_converted_score_mean_matches_public_predecessor_recipe() -> None:
    teachers = (
        TeacherWinProbability("teacher-b", 0.75, weight=3.0),
        TeacherWinProbability("teacher-a", 0.25, weight=1.0),
    )
    result = combine_teacher_win_probabilities(
        teachers,
        mode=TeacherValueEnsembleMode.CONVERTED_SCORE_ARITHMETIC_MEAN,
        ponanza_coefficient=700.0,
    )
    expected_cp = (
        3.0 * win_probability_to_cp(0.75, ponanza_coefficient=700.0)
        + win_probability_to_cp(0.25, ponanza_coefficient=700.0)
    ) / 4.0

    assert result.output_evaluation_cp == pytest.approx(expected_cp)
    assert result.output_probability == pytest.approx(
        cp_to_win_probability(expected_cp, ponanza_coefficient=700.0)
    )
    assert tuple(result.normalized_weights) == ("teacher-a", "teacher-b")


def test_exact_probability_endpoints_are_clipped_and_receipted() -> None:
    result = combine_teacher_win_probabilities(
        (
            TeacherWinProbability("certain-loss", 0.0),
            TeacherWinProbability("certain-win", 1.0),
        ),
        mode=TeacherValueEnsembleMode.PROBABILITY_ARITHMETIC_MEAN,
        ponanza_coefficient=600.0,
        probability_epsilon=1e-5,
    )

    assert result.clipped_teacher_ids == ("certain-loss", "certain-win")
    assert result.output_probability == pytest.approx(0.5)
    assert math.isfinite(result.output_evaluation_cp)


def test_value_ensemble_rejects_duplicate_teacher_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        combine_teacher_win_probabilities(
            (
                TeacherWinProbability("same", 0.4),
                TeacherWinProbability("same", 0.6),
            ),
            mode=TeacherValueEnsembleMode.PROBABILITY_ARITHMETIC_MEAN,
            ponanza_coefficient=600.0,
        )
