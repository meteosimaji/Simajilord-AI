"""Explicit scalar-value ensemble comparators for teacher ablations.

Suisho 11 publicly identifies its three component models but does not publish
the combination formula.  The methods here therefore carry their own names and
must never be presented as a reproduction of Suisho 11.  They implement the
public INUGAMI predecessor (win rate -> score -> arithmetic mean) and a
scale-independent probability-mean comparator.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import cast


class TeacherValueEnsembleMode(StrEnum):
    CONVERTED_SCORE_ARITHMETIC_MEAN = "converted_score_arithmetic_mean"
    PROBABILITY_ARITHMETIC_MEAN = "probability_arithmetic_mean"


@dataclass(frozen=True, slots=True)
class TeacherWinProbability:
    teacher_id: str
    probability: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.teacher_id:
            raise ValueError("teacher ID must not be empty")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("teacher probability must be finite in [0, 1]")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("teacher weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class TeacherValueEnsembleResult:
    mode: TeacherValueEnsembleMode
    ponanza_coefficient: float
    probability_epsilon: float
    teachers: tuple[TeacherWinProbability, ...]
    normalized_weights: dict[str, float]
    clipped_teacher_ids: tuple[str, ...]
    output_probability: float
    output_evaluation_cp: float
    suisho11_formula_claimed: bool = False

    def to_dict(self) -> dict[str, object]:
        return cast(
            dict[str, object],
            json.loads(json.dumps(asdict(self), allow_nan=False, sort_keys=True)),
        )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def win_probability_to_cp(probability: float, *, ponanza_coefficient: float) -> float:
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("finite CP conversion requires probability strictly inside (0, 1)")
    if not math.isfinite(ponanza_coefficient) or ponanza_coefficient <= 0.0:
        raise ValueError("Ponanza coefficient must be finite and positive")
    return ponanza_coefficient * math.log(probability / (1.0 - probability))


def cp_to_win_probability(cp: float, *, ponanza_coefficient: float) -> float:
    if not math.isfinite(cp):
        raise ValueError("evaluation CP must be finite")
    if not math.isfinite(ponanza_coefficient) or ponanza_coefficient <= 0.0:
        raise ValueError("Ponanza coefficient must be finite and positive")
    return _sigmoid(cp / ponanza_coefficient)


def combine_teacher_win_probabilities(
    teachers: tuple[TeacherWinProbability, ...],
    *,
    mode: TeacherValueEnsembleMode,
    ponanza_coefficient: float,
    probability_epsilon: float = 1e-6,
) -> TeacherValueEnsembleResult:
    """Combine calibrated probabilities under one explicitly receipted method."""

    if not teachers:
        raise ValueError("value ensemble requires at least one teacher")
    teacher_ids = tuple(teacher.teacher_id for teacher in teachers)
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("value ensemble teacher IDs must be unique")
    if not math.isfinite(ponanza_coefficient) or ponanza_coefficient <= 0.0:
        raise ValueError("Ponanza coefficient must be finite and positive")
    if not math.isfinite(probability_epsilon) or not 0.0 < probability_epsilon < 0.5:
        raise ValueError("probability epsilon must be finite and inside (0, 0.5)")
    total_weight = math.fsum(teacher.weight for teacher in teachers)
    normalized_weights = {
        teacher.teacher_id: teacher.weight / total_weight for teacher in teachers
    }
    clipped_probabilities: dict[str, float] = {}
    clipped_teacher_ids: list[str] = []
    for teacher in teachers:
        clipped = min(1.0 - probability_epsilon, max(probability_epsilon, teacher.probability))
        clipped_probabilities[teacher.teacher_id] = clipped
        if clipped != teacher.probability:
            clipped_teacher_ids.append(teacher.teacher_id)

    if mode is TeacherValueEnsembleMode.PROBABILITY_ARITHMETIC_MEAN:
        output_probability = math.fsum(
            normalized_weights[teacher.teacher_id]
            * clipped_probabilities[teacher.teacher_id]
            for teacher in teachers
        )
        output_cp = win_probability_to_cp(
            output_probability,
            ponanza_coefficient=ponanza_coefficient,
        )
    elif mode is TeacherValueEnsembleMode.CONVERTED_SCORE_ARITHMETIC_MEAN:
        output_cp = math.fsum(
            normalized_weights[teacher.teacher_id]
            * win_probability_to_cp(
                clipped_probabilities[teacher.teacher_id],
                ponanza_coefficient=ponanza_coefficient,
            )
            for teacher in teachers
        )
        output_probability = cp_to_win_probability(
            output_cp,
            ponanza_coefficient=ponanza_coefficient,
        )
    else:
        raise AssertionError(f"unsupported teacher value ensemble mode: {mode!r}")

    return TeacherValueEnsembleResult(
        mode=mode,
        ponanza_coefficient=ponanza_coefficient,
        probability_epsilon=probability_epsilon,
        teachers=tuple(sorted(teachers, key=lambda teacher: teacher.teacher_id)),
        normalized_weights=dict(sorted(normalized_weights.items())),
        clipped_teacher_ids=tuple(sorted(clipped_teacher_ids)),
        output_probability=output_probability,
        output_evaluation_cp=output_cp,
    )
