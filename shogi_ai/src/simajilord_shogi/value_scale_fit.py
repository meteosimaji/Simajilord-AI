"""Game-clustered calibration of external-teacher centipawn value scales."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import cast

from .domain import (
    GameRecord,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)


@dataclass(frozen=True, slots=True)
class ValueScaleObservation:
    """One exact root cp score and its eventual side-to-move game result."""

    game_id: int
    cp: int
    label: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.label) or not 0.0 <= self.label <= 1.0:
            raise ValueError("value-scale label must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class FitWarning:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ScaleEstimate:
    value_scale: float
    bce: float
    boundary: str | None


@dataclass(frozen=True, slots=True)
class CrossValidationSummary:
    mode: str
    folds: int
    held_out_samples: int
    fitted_bce: float
    baseline_bce: float
    scale_min: float
    scale_median: float
    scale_max: float
    boundary_folds: int


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    resamples: int
    seed: int
    successful_resamples: int
    boundary_resamples: int
    scale_p2_5: float
    scale_median: float
    scale_p97_5: float


@dataclass(frozen=True, slots=True)
class TeacherValueScaleFit:
    teacher_source: str
    status: str
    provisional: bool
    samples: int
    games: int
    cp_min: int
    cp_max: int
    cp_median: float
    outcomes: dict[str, int]
    exclusions: dict[str, int]
    baseline_scale: float
    baseline_bce: float
    fitted_scale: float
    fitted_bce: float
    bce_improvement: float
    fit_boundary: str | None
    cross_validation: CrossValidationSummary | None
    cluster_bootstrap: BootstrapSummary | None
    warnings: tuple[FitWarning, ...]


@dataclass(frozen=True, slots=True)
class ValueScaleFitReport:
    schema_version: int
    method: dict[str, object]
    provisional: bool
    extraction: dict[str, int]
    teachers: dict[str, TeacherValueScaleFit]
    warnings: tuple[FitWarning, ...]

    def to_dict(self) -> dict[str, object]:
        payload = cast(dict[str, object], asdict(self))
        teacher_payloads = cast(dict[str, dict[str, object]], payload["teachers"])
        for teacher_payload in teacher_payloads.values():
            baseline_denominator = cast(float, teacher_payload["baseline_scale"])
            fitted_denominator = cast(float, teacher_payload["fitted_scale"])
            # Preserve the old *_scale fields as raw tanh denominators while
            # making the Ponanza C versus signed-value 2C distinction explicit.
            teacher_payload["baseline_ponanza_coefficient"] = baseline_denominator / 2.0
            teacher_payload["baseline_value_tanh_denominator"] = baseline_denominator
            teacher_payload["fitted_ponanza_coefficient"] = fitted_denominator / 2.0
            teacher_payload["fitted_value_tanh_denominator"] = fitted_denominator
        return payload


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def binary_cross_entropy(
    observations: Sequence[ValueScaleObservation], value_scale: float
) -> float:
    """Mean BCE for ``p = sigmoid(2 * cp / value_scale)``."""

    if not observations:
        raise ValueError("at least one value-scale observation is required")
    if not math.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError("value_scale must be finite and positive")
    return sum(
        _softplus(logit) - observation.label * logit
        for observation in observations
        for logit in (2.0 * observation.cp / value_scale,)
    ) / len(observations)


def fit_value_scale(
    observations: Sequence[ValueScaleObservation],
    *,
    minimum_scale: float = 1.0,
    maximum_scale: float = 1_000_000.0,
) -> ScaleEstimate:
    """Fit the positive scale by convex bisection in inverse-scale space."""

    if not observations:
        raise ValueError("at least one value-scale observation is required")
    if (
        not math.isfinite(minimum_scale)
        or not math.isfinite(maximum_scale)
        or minimum_scale <= 0.0
        or maximum_scale <= minimum_scale
    ):
        raise ValueError("scale bounds must be finite, positive, and increasing")

    def gradient(inverse_scale_twice: float) -> float:
        return sum(
            (_sigmoid(inverse_scale_twice * observation.cp) - observation.label)
            * observation.cp
            for observation in observations
        ) / len(observations)

    lower_beta = 2.0 / maximum_scale
    upper_beta = 2.0 / minimum_scale
    lower_gradient = gradient(lower_beta)
    upper_gradient = gradient(upper_beta)
    if lower_gradient >= 0.0:
        scale = maximum_scale
        return ScaleEstimate(scale, binary_cross_entropy(observations, scale), "maximum")
    if upper_gradient <= 0.0:
        scale = minimum_scale
        return ScaleEstimate(scale, binary_cross_entropy(observations, scale), "minimum")

    for _ in range(80):
        midpoint = (lower_beta + upper_beta) / 2.0
        if gradient(midpoint) < 0.0:
            lower_beta = midpoint
        else:
            upper_beta = midpoint
    scale = 2.0 / ((lower_beta + upper_beta) / 2.0)
    return ScaleEstimate(scale, binary_cross_entropy(observations, scale), None)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _root_variation(
    variations: Sequence[TeacherVariation], teacher_best_move: str | None
) -> TeacherVariation | None:
    if teacher_best_move is not None:
        matching = [variation for variation in variations if variation.move == teacher_best_move]
        return min(matching, key=lambda variation: variation.rank) if matching else None
    return min(variations, key=lambda variation: variation.rank) if variations else None


def _extract_observations(
    games: Sequence[GameRecord],
) -> tuple[
    dict[str, list[ValueScaleObservation]],
    dict[str, int],
    dict[str, Counter[str]],
]:
    grouped: dict[str, list[ValueScaleObservation]] = defaultdict(list)
    exclusions: dict[str, Counter[str]] = defaultdict(Counter)
    extraction: Counter[str] = Counter(
        games=len(games),
        complete_games=0,
        incomplete_games=0,
        samples=sum(len(game.samples) for game in games),
        eligible_samples=0,
    )
    for game_id, game in enumerate(games):
        if game.termination is Termination.MAX_PLIES:
            extraction["incomplete_games"] += 1
            extraction["excluded_incomplete_samples"] += len(game.samples)
            continue
        extraction["complete_games"] += 1
        if game.winner not in {None, 0, 1}:
            extraction["excluded_invalid_winner_samples"] += len(game.samples)
            continue
        for sample in game.samples:
            if not sample.teacher_variations:
                extraction["excluded_without_variations"] += 1
                continue
            source = sample.teacher_source or "unknown"
            if sample.teacher_source is None:
                exclusions[source]["unknown_teacher_source"] += 1
            root = _root_variation(sample.teacher_variations, sample.teacher_best_move)
            if root is None:
                exclusions[source]["missing_root_variation"] += 1
                extraction["excluded_missing_root_variation"] += 1
                continue
            if root.score_kind is TeacherScoreKind.MATE:
                exclusions[source]["mate_score"] += 1
                extraction["excluded_mate_scores"] += 1
                continue
            if root.bound is not TeacherScoreBound.EXACT:
                exclusions[source]["bounded_score"] += 1
                extraction["excluded_bounded_scores"] += 1
                continue
            if root.score_cp is None or sample.turn not in {0, 1}:
                exclusions[source]["invalid_cp_or_turn"] += 1
                extraction["excluded_invalid_cp_or_turn"] += 1
                continue
            label = 0.5 if game.winner is None else float(game.winner == sample.turn)
            grouped[source].append(ValueScaleObservation(game_id, root.score_cp, label))
            extraction["eligible_samples"] += 1
    return dict(grouped), dict(extraction), exclusions


def _cross_validate(
    observations: Sequence[ValueScaleObservation],
    *,
    baseline_scale: float,
    minimum_scale: float,
    maximum_scale: float,
    maximum_folds: int,
    seed: int,
) -> CrossValidationSummary | None:
    game_ids = sorted({observation.game_id for observation in observations})
    if len(game_ids) < 2:
        return None
    folds = len(game_ids) if len(game_ids) <= maximum_folds else maximum_folds
    shuffled = list(game_ids)
    random.Random(seed).shuffle(shuffled)
    held_out_groups = [set(shuffled[index::folds]) for index in range(folds)]
    fitted_loss_sum = 0.0
    baseline_loss_sum = 0.0
    held_out_samples = 0
    scales: list[float] = []
    boundary_folds = 0
    for held_out_ids in held_out_groups:
        training = [
            observation for observation in observations if observation.game_id not in held_out_ids
        ]
        held_out = [
            observation for observation in observations if observation.game_id in held_out_ids
        ]
        if not training or not held_out:
            continue
        estimate = fit_value_scale(
            training,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        scales.append(estimate.value_scale)
        boundary_folds += estimate.boundary is not None
        fitted_loss_sum += binary_cross_entropy(held_out, estimate.value_scale) * len(held_out)
        baseline_loss_sum += binary_cross_entropy(held_out, baseline_scale) * len(held_out)
        held_out_samples += len(held_out)
    if not scales or held_out_samples == 0:
        return None
    return CrossValidationSummary(
        mode="leave_one_game_out" if folds == len(game_ids) else "group_k_fold",
        folds=len(scales),
        held_out_samples=held_out_samples,
        fitted_bce=fitted_loss_sum / held_out_samples,
        baseline_bce=baseline_loss_sum / held_out_samples,
        scale_min=min(scales),
        scale_median=median(scales),
        scale_max=max(scales),
        boundary_folds=boundary_folds,
    )


def _cluster_bootstrap(
    observations: Sequence[ValueScaleObservation],
    *,
    resamples: int,
    seed: int,
    minimum_scale: float,
    maximum_scale: float,
) -> BootstrapSummary | None:
    game_groups: dict[int, list[ValueScaleObservation]] = defaultdict(list)
    for observation in observations:
        game_groups[observation.game_id].append(observation)
    game_ids = sorted(game_groups)
    if len(game_ids) < 2 or resamples == 0:
        return None
    generator = random.Random(seed)
    scales: list[float] = []
    boundary_resamples = 0
    for _ in range(resamples):
        sampled_ids = [generator.choice(game_ids) for _ in game_ids]
        sampled = [
            observation
            for game_id in sampled_ids
            for observation in game_groups[game_id]
        ]
        estimate = fit_value_scale(
            sampled,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
        )
        scales.append(estimate.value_scale)
        boundary_resamples += estimate.boundary is not None
    return BootstrapSummary(
        resamples=resamples,
        seed=seed,
        successful_resamples=len(scales),
        boundary_resamples=boundary_resamples,
        scale_p2_5=_percentile(scales, 0.025),
        scale_median=_percentile(scales, 0.5),
        scale_p97_5=_percentile(scales, 0.975),
    )


def _teacher_seed(seed: int, teacher_source: str) -> int:
    digest = hashlib.sha256(teacher_source.encode()).digest()
    return seed ^ int.from_bytes(digest[:4], "big")


def _fit_teacher(
    teacher_source: str,
    observations: Sequence[ValueScaleObservation],
    exclusions: Counter[str],
    *,
    baseline_scale: float,
    minimum_scale: float,
    maximum_scale: float,
    bootstrap_resamples: int,
    seed: int,
    maximum_cross_validation_folds: int,
    minimum_reliable_games: int,
    minimum_reliable_samples: int,
) -> TeacherValueScaleFit:
    estimate = fit_value_scale(
        observations,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    baseline_bce = binary_cross_entropy(observations, baseline_scale)
    game_count = len({observation.game_id for observation in observations})
    source_seed = _teacher_seed(seed, teacher_source)
    cross_validation = _cross_validate(
        observations,
        baseline_scale=baseline_scale,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        maximum_folds=maximum_cross_validation_folds,
        seed=source_seed,
    )
    bootstrap = _cluster_bootstrap(
        observations,
        resamples=bootstrap_resamples,
        seed=source_seed,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
    )
    warnings: list[FitWarning] = []
    if len(observations) < minimum_reliable_samples:
        warnings.append(
            FitWarning(
                "small_sample_count",
                f"{len(observations)} eligible samples is below {minimum_reliable_samples}",
            )
        )
    if game_count < minimum_reliable_games:
        warnings.append(
            FitWarning(
                "small_game_count",
                f"{game_count} independent game clusters is below {minimum_reliable_games}",
            )
        )
    win_count = sum(observation.label == 1.0 for observation in observations)
    draw_count = sum(observation.label == 0.5 for observation in observations)
    loss_count = sum(observation.label == 0.0 for observation in observations)
    if win_count == 0 or loss_count == 0:
        warnings.append(
            FitWarning(
                "limited_outcome_support",
                "both decisive outcome classes are required for a stable no-intercept scale fit",
            )
        )
    cp_values = [observation.cp for observation in observations]
    if min(cp_values) >= 0 or max(cp_values) <= 0:
        warnings.append(
            FitWarning(
                "one_sided_cp_support",
                "eligible root scores do not cover both negative and positive cp",
            )
        )
    if estimate.boundary is not None:
        warnings.append(
            FitWarning(
                "scale_at_search_boundary",
                f"optimum reached the configured {estimate.boundary} scale boundary",
            )
        )
    if cross_validation is None:
        warnings.append(
            FitWarning(
                "cross_validation_unavailable",
                "at least two independent games are required for game-level validation",
            )
        )
    elif cross_validation.fitted_bce >= cross_validation.baseline_bce:
        warnings.append(
            FitWarning(
                "no_cross_validated_improvement",
                "game-fold fitted BCE did not improve on the configured baseline scale",
            )
        )
    if bootstrap is None:
        warnings.append(
            FitWarning(
                "bootstrap_unavailable",
                "game-cluster bootstrap requires at least two games and one resample",
            )
        )
    else:
        if bootstrap.boundary_resamples * 5 > bootstrap.successful_resamples:
            warnings.append(
                FitWarning(
                    "bootstrap_boundary_instability",
                    "more than 20% of game-cluster bootstrap fits reached a scale boundary",
                )
            )
        if bootstrap.scale_p97_5 / bootstrap.scale_p2_5 > 4.0:
            warnings.append(
                FitWarning(
                    "wide_bootstrap_interval",
                    "the 95% game-cluster bootstrap scale interval spans more than fourfold",
                )
            )
    if teacher_source == "unknown":
        warnings.append(
            FitWarning(
                "unknown_teacher_source",
                "teacher_source is missing, so this calibration cannot be tied "
                "to an engine version",
            )
        )
    provisional = bool(warnings)
    return TeacherValueScaleFit(
        teacher_source=teacher_source,
        status="provisional" if provisional else "fitted",
        provisional=provisional,
        samples=len(observations),
        games=game_count,
        cp_min=min(cp_values),
        cp_max=max(cp_values),
        cp_median=median(cp_values),
        outcomes={"win": win_count, "draw": draw_count, "loss": loss_count},
        exclusions=dict(sorted(exclusions.items())),
        baseline_scale=baseline_scale,
        baseline_bce=baseline_bce,
        fitted_scale=estimate.value_scale,
        fitted_bce=estimate.bce,
        bce_improvement=baseline_bce - estimate.bce,
        fit_boundary=estimate.boundary,
        cross_validation=cross_validation,
        cluster_bootstrap=bootstrap,
        warnings=tuple(warnings),
    )


def fit_teacher_value_scales(
    games: Sequence[GameRecord],
    *,
    baseline_scale: float = 1_200.0,
    minimum_scale: float = 1.0,
    maximum_scale: float = 1_000_000.0,
    bootstrap_resamples: int = 500,
    seed: int = 0,
    maximum_cross_validation_folds: int = 10,
    minimum_reliable_games: int = 30,
    minimum_reliable_samples: int = 200,
) -> ValueScaleFitReport:
    """Fit every teacher source independently and report game-cluster uncertainty."""

    if not math.isfinite(baseline_scale) or baseline_scale <= 0.0:
        raise ValueError("baseline_scale must be finite and positive")
    if (
        not math.isfinite(minimum_scale)
        or not math.isfinite(maximum_scale)
        or minimum_scale <= 0.0
        or maximum_scale <= minimum_scale
    ):
        raise ValueError("scale bounds must be finite, positive, and increasing")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")
    if maximum_cross_validation_folds < 2:
        raise ValueError("maximum_cross_validation_folds must be at least two")
    if minimum_reliable_games < 1 or minimum_reliable_samples < 1:
        raise ValueError("reliability thresholds must be positive")
    grouped, extraction, exclusions = _extract_observations(games)
    teachers = {
        teacher_source: _fit_teacher(
            teacher_source,
            observations,
            exclusions[teacher_source],
            baseline_scale=baseline_scale,
            minimum_scale=minimum_scale,
            maximum_scale=maximum_scale,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            maximum_cross_validation_folds=maximum_cross_validation_folds,
            minimum_reliable_games=minimum_reliable_games,
            minimum_reliable_samples=minimum_reliable_samples,
        )
        for teacher_source, observations in sorted(grouped.items())
    }
    warnings: list[FitWarning] = []
    if not teachers:
        warnings.append(
            FitWarning(
                "no_eligible_samples",
                "no exact root centipawn teacher variations were available for calibration",
            )
        )
    if extraction.get("incomplete_games", 0):
        warnings.append(
            FitWarning(
                "incomplete_games_excluded",
                "max_plies games were excluded because they have no observed terminal outcome",
            )
        )
    if teachers:
        warnings.append(
            FitWarning(
                "selection_bias_not_identified",
                "the fit measures this replay only and cannot identify opening or sampling bias",
            )
        )
    provisional = not teachers or any(result.provisional for result in teachers.values())
    return ValueScaleFitReport(
        schema_version=1,
        method={
            "probability": "sigmoid(2 * cp / value_scale)",
            "signed_value": "2 * probability - 1 = tanh(cp / value_scale)",
            "scale_semantics": {
                "ponanza_coefficient": "C in sigmoid(cp / C)",
                "value_scale": "2C, the denominator in tanh(cp / value_scale)",
                "baseline_ponanza_coefficient": baseline_scale / 2.0,
                "baseline_value_tanh_denominator": baseline_scale,
            },
            "objective": "mean binary cross entropy",
            "label": "side-to-move win=1, draw=0.5, loss=0",
            "candidate": "teacher_best_move exact cp, otherwise MultiPV rank 1 exact cp",
            "excluded_scores": ["mate", "lowerbound", "upperbound"],
            "scale_bounds": {"minimum": minimum_scale, "maximum": maximum_scale},
            "cross_validation": (
                "leave-one-game-out when game count <= maximum folds; otherwise group K-fold"
            ),
            "uncertainty": "percentile game-cluster bootstrap",
            "bootstrap_resamples": bootstrap_resamples,
            "seed": seed,
        },
        provisional=provisional,
        extraction=dict(sorted(extraction.items())),
        teachers=teachers,
        warnings=tuple(warnings),
    )
