from __future__ import annotations

import math

import pytest

from simajilord_shogi.domain import TeacherScoreBound
from simajilord_shogi.teacher_committee import (
    backup_root_interval_from_child_replies,
    score_interval_from_bound,
    synthesize_worst_case_interval_signal,
    synthesize_worst_regret_signal,
)


def test_worst_regret_signal_keeps_full_distribution_and_strict_best() -> None:
    signal = synthesize_worst_regret_signal(
        (
            ("nagisa", {"7g7f": 0.90, "2g2f": 0.70, "5g5f": 0.20}),
            ("suisho11", {"7g7f": 0.80, "2g2f": 0.85, "5g5f": 0.10}),
            ("soujou", {"7g7f": 0.88, "2g2f": 0.76, "5g5f": 0.30}),
        ),
        temperature=0.05,
    )

    assert signal.chosen_move == "7g7f"
    assert signal.robust_best_moves == ("7g7f",)
    assert signal.worst_regret == pytest.approx(
        {"7g7f": 0.05, "2g2f": 0.20, "5g5f": 0.75}
    )
    assert set(signal.policy) == {"7g7f", "2g2f", "5g5f"}
    assert math.fsum(signal.policy.values()) == pytest.approx(1.0)
    assert signal.policy["7g7f"] > signal.policy["2g2f"] > signal.policy["5g5f"]
    assert signal.chosen_conservative_value == pytest.approx(0.80)


def test_worst_regret_signal_preserves_a_universally_near_best_compromise() -> None:
    signal = synthesize_worst_regret_signal(
        (
            ("nagisa", {"A": 1.0, "B": 0.6, "C": 0.9}),
            ("suisho11", {"A": 0.6, "B": 1.0, "C": 0.9}),
            ("soujou", {"A": 0.7, "B": 0.7, "C": 0.95}),
        ),
        temperature=0.1,
    )

    assert signal.chosen_move == "C"
    assert signal.worst_regret["C"] == pytest.approx(0.1)
    assert signal.worst_regret["A"] == pytest.approx(0.4)
    assert signal.worst_regret["B"] == pytest.approx(0.4)


def test_score_bounds_become_intervals_in_root_perspective() -> None:
    assert score_interval_from_bound(0.25, TeacherScoreBound.EXACT) == (0.25, 0.25)
    assert score_interval_from_bound(0.25, TeacherScoreBound.LOWER) == (0.25, 1.0)
    assert score_interval_from_bound(0.25, TeacherScoreBound.UPPER) == (-1.0, 0.25)


def test_reply_backup_negates_perspective_and_selects_opponents_worst_reply() -> None:
    backed = backup_root_interval_from_child_replies(
        {
            "reply-a": (0.20, 0.40),
            "reply-b": (-0.10, 0.80),
        }
    )

    assert backed == pytest.approx((-0.80, -0.20))


def test_reply_backup_can_reverse_a_shallow_root_preference() -> None:
    backed = {
        "nagisa": {
            "root-favorite": backup_root_interval_from_child_replies(
                {"tactical-refutation": (0.80, 0.80)}
            ),
            "safe": backup_root_interval_from_child_replies(
                {"normal-reply": (-0.10, -0.10)}
            ),
        },
        "suisho11": {
            "root-favorite": backup_root_interval_from_child_replies(
                {"tactical-refutation": (0.70, 0.70)}
            ),
            "safe": backup_root_interval_from_child_replies(
                {"normal-reply": (-0.05, -0.05)}
            ),
        },
        "soujou": {
            "root-favorite": backup_root_interval_from_child_replies(
                {"tactical-refutation": (0.75, 0.75)}
            ),
            "safe": backup_root_interval_from_child_replies(
                {"normal-reply": (-0.08, -0.08)}
            ),
        },
    }

    signal = synthesize_worst_case_interval_signal(
        tuple((teacher_id, intervals) for teacher_id, intervals in backed.items()),
        temperature=0.05,
    )

    assert signal.chosen_move == "safe"
    assert signal.chosen_conservative_value == pytest.approx(0.05)


def test_reply_backup_rejects_empty_or_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="at least one"):
        backup_root_interval_from_child_replies({})
    with pytest.raises(ValueError, match="invalid"):
        backup_root_interval_from_child_replies({"bad": (0.5, 0.4)})


def test_interval_signal_minimizes_worst_case_regret_not_bound_midpoints() -> None:
    signal = synthesize_worst_case_interval_signal(
        (
            (
                "nagisa",
                {
                    "safe": (0.70, 0.70),
                    "uncertain": (-1.0, 0.95),
                },
            ),
            (
                "suisho11",
                {
                    "safe": (0.72, 0.72),
                    "uncertain": (0.60, 1.0),
                },
            ),
            (
                "soujou",
                {
                    "safe": (0.68, 0.68),
                    "uncertain": (0.65, 0.65),
                },
            ),
        ),
        temperature=0.1,
    )

    assert signal.chosen_move == "safe"
    assert signal.worst_regret["safe"] == pytest.approx(0.28)
    assert signal.worst_regret["uncertain"] == pytest.approx(1.95)


@pytest.mark.parametrize(
    "teacher_values,temperature,error",
    (
        ((("only", {"A": 0.0}),), 0.1, "at least two"),
        (
            (("one", {"A": 0.0}), ("two", {"B": 0.0})),
            0.1,
            "complete common candidate matrix",
        ),
        (
            (("same", {"A": 0.0}), ("same", {"A": 0.0})),
            0.1,
            "IDs must be unique",
        ),
        (
            (("one", {"A": 0.0}), ("two", {"A": 0.0})),
            0.0,
            "temperature",
        ),
    ),
)
def test_worst_regret_signal_rejects_incomplete_or_ambiguous_inputs(
    teacher_values: tuple[tuple[str, dict[str, float]], ...],
    temperature: float,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        synthesize_worst_regret_signal(teacher_values, temperature=temperature)
