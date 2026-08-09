"""Experimental variance-normalized policies from lossless USI MultiPV scores."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from statistics import mean
from typing import Literal, cast

from rsshogi.core import Board

from .domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
)

PolicyCalibrationMode = Literal["off", "legacy", "variance-normalized"]
POLICY_CALIBRATION_SCHEMA = "meteo-teacher-policy-calibration-v1"
LOGIT_STANDARDIZATION_PAPER = (
    "https://openaccess.thecvf.com/content/CVPR2024/html/"
    "Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html"
)


@dataclass(frozen=True, slots=True)
class PolicyCalibrationConfig:
    mode: PolicyCalibrationMode = "variance-normalized"
    prior_strength: float = 5.0
    standard_deviation_floor: float = 25.0
    standard_deviation_cap: float = 2_000.0
    normalized_temperature: float = 1.0
    default_prior_standard_deviation: float = 200.0
    variance_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.mode not in {"off", "legacy", "variance-normalized"}:
            raise ValueError("unsupported teacher-policy calibration mode")
        positive = (
            self.prior_strength,
            self.standard_deviation_floor,
            self.standard_deviation_cap,
            self.normalized_temperature,
            self.default_prior_standard_deviation,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("policy calibration scales and prior strength must be positive")
        if self.standard_deviation_cap < self.standard_deviation_floor:
            raise ValueError("standard deviation cap must not be below its floor")
        if not math.isfinite(self.variance_epsilon) or self.variance_epsilon < 0.0:
            raise ValueError("variance_epsilon must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GlobalDispersionReport:
    teacher_source: str
    positions_seen: int
    cp_positions: int
    multi_candidate_positions: int
    pooled_degrees_of_freedom: int
    pooled_variance_cp2: float
    pooled_standard_deviation_cp: float
    prior_variance_cp2: float
    prior_standard_deviation_cp: float
    prior_source: str


@dataclass(frozen=True, slots=True)
class EntropyComparison:
    positions_compared: int
    legacy_mean_nats: float | None
    variance_normalized_mean_nats: float | None
    output_mean_nats: float | None
    output_minus_legacy_mean_nats: float | None


@dataclass(frozen=True, slots=True)
class TeacherPolicyCalibrationReport:
    teacher_source: str
    global_dispersion: GlobalDispersionReport
    samples_seen: int
    candidate_positions: int
    applied_positions: int
    cp_positions: int
    winning_mate_positions: int
    single_candidate_positions: int
    zero_variance_positions: int
    fallback_positions: int
    fallback_counts: dict[str, int]
    excluded_candidates: dict[str, int]
    effective_standard_deviation_cp: dict[str, float] | None
    entropy: EntropyComparison
    mean_subtraction_max_probability_delta: float


@dataclass(frozen=True, slots=True)
class PolicyCalibrationReport:
    schema: str
    mode: PolicyCalibrationMode
    experimental: bool
    method: dict[str, object]
    config: dict[str, object]
    games: int
    samples: int
    teachers: dict[str, TeacherPolicyCalibrationReport]
    totals: dict[str, int]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class PolicyCalibrationBuild:
    games: tuple[GameRecord, ...]
    report: PolicyCalibrationReport


@dataclass(frozen=True, slots=True)
class _RawCandidates:
    legal_moves: frozenset[str] | None
    cp: dict[str, int]
    winning_mates: tuple[str, ...]
    excluded: Counter[str]
    missing_variations: bool
    invalid_sfen: bool


@dataclass(slots=True)
class _DispersionAccumulator:
    positions_seen: int = 0
    cp_positions: int = 0
    multi_candidate_positions: int = 0
    pooled_degrees_of_freedom: int = 0
    pooled_sum_squares: float = 0.0


@dataclass(slots=True)
class _TeacherAccumulator:
    samples_seen: int = 0
    candidate_positions: int = 0
    applied_positions: int = 0
    cp_positions: int = 0
    winning_mate_positions: int = 0
    single_candidate_positions: int = 0
    zero_variance_positions: int = 0
    fallback_positions: int = 0
    fallback_counts: Counter[str] = field(default_factory=Counter)
    excluded_candidates: Counter[str] = field(default_factory=Counter)
    effective_standard_deviations: list[float] = field(default_factory=list)
    legacy_entropies: list[float] = field(default_factory=list)
    normalized_entropies: list[float] = field(default_factory=list)
    output_entropies: list[float] = field(default_factory=list)
    mean_subtraction_max_probability_delta: float = 0.0


def _teacher_source(sample: PositionSample) -> str:
    return sample.teacher_source or "unknown"


def _mate_sign(variation: TeacherVariation) -> int:
    if variation.score_kind is not TeacherScoreKind.MATE:
        raise TypeError("centipawn variation has no mate sign")
    if variation.mate_unknown_sign is not None:
        return variation.mate_unknown_sign
    assert variation.mate_plies is not None
    return 1 if variation.mate_plies >= 0 else -1


def _legal_moves(sfen: str) -> frozenset[str] | None:
    try:
        board = Board(sfen)
    except (RuntimeError, ValueError):
        return None
    if not board.is_valid():
        return None
    return frozenset(move.to_usi() for move in board.legal_moves())


def _raw_candidates(sample: PositionSample) -> _RawCandidates:
    legal_moves = _legal_moves(sample.sfen)
    excluded: Counter[str] = Counter()
    if legal_moves is None:
        excluded["invalid_sfen"] += 1
        return _RawCandidates(None, {}, (), excluded, not sample.teacher_variations, True)
    if not sample.teacher_variations:
        return _RawCandidates(legal_moves, {}, (), excluded, True, False)
    cp: dict[str, int] = {}
    winning_mates: list[str] = []
    seen_moves: set[str] = set()
    for variation in sorted(sample.teacher_variations, key=lambda item: item.rank):
        if variation.bound is not TeacherScoreBound.EXACT:
            excluded["bounded"] += 1
            continue
        if variation.move not in legal_moves:
            excluded["illegal_root_move"] += 1
            continue
        if variation.move in seen_moves:
            excluded["duplicate_root_move"] += 1
            continue
        seen_moves.add(variation.move)
        if variation.score_kind is TeacherScoreKind.MATE:
            if _mate_sign(variation) > 0:
                winning_mates.append(variation.move)
            else:
                excluded["losing_mate"] += 1
            continue
        assert variation.score_cp is not None
        cp[variation.move] = variation.score_cp
    return _RawCandidates(
        legal_moves,
        cp,
        tuple(winning_mates),
        excluded,
        False,
        False,
    )


def _position_sum_squares(values: Sequence[int]) -> tuple[float, float]:
    center = math.fsum(values) / len(values)
    return center, math.fsum((value - center) ** 2 for value in values)


def _global_dispersions(
    games: Sequence[GameRecord], config: PolicyCalibrationConfig
) -> dict[str, GlobalDispersionReport]:
    accumulators: dict[str, _DispersionAccumulator] = defaultdict(_DispersionAccumulator)
    for game in games:
        for sample in game.samples:
            if sample.teacher_policy is None and not sample.teacher_variations:
                continue
            source = _teacher_source(sample)
            accumulator = accumulators[source]
            accumulator.positions_seen += 1
            raw = _raw_candidates(sample)
            if raw.winning_mates or not raw.cp:
                continue
            if sample.teacher_best_move is not None and sample.teacher_best_move not in raw.cp:
                continue
            values = tuple(raw.cp.values())
            accumulator.cp_positions += 1
            if len(values) < 2:
                continue
            accumulator.multi_candidate_positions += 1
            _, sum_squares = _position_sum_squares(values)
            accumulator.pooled_sum_squares += sum_squares
            accumulator.pooled_degrees_of_freedom += len(values) - 1
    reports: dict[str, GlobalDispersionReport] = {}
    for source, accumulator in sorted(accumulators.items()):
        pooled_variance = (
            accumulator.pooled_sum_squares / accumulator.pooled_degrees_of_freedom
            if accumulator.pooled_degrees_of_freedom
            else 0.0
        )
        if pooled_variance > config.variance_epsilon:
            prior_variance = pooled_variance
            prior_source = "teacher_pooled_within_position_variance"
        else:
            prior_variance = config.default_prior_standard_deviation**2
            prior_source = "configured_default"
        reports[source] = GlobalDispersionReport(
            teacher_source=source,
            positions_seen=accumulator.positions_seen,
            cp_positions=accumulator.cp_positions,
            multi_candidate_positions=accumulator.multi_candidate_positions,
            pooled_degrees_of_freedom=accumulator.pooled_degrees_of_freedom,
            pooled_variance_cp2=pooled_variance,
            pooled_standard_deviation_cp=math.sqrt(max(0.0, pooled_variance)),
            prior_variance_cp2=prior_variance,
            prior_standard_deviation_cp=math.sqrt(prior_variance),
            prior_source=prior_source,
        )
    return reports


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        raise ValueError("softmax requires at least one logit")
    maximum = max(logits.values())
    weights = {move: math.exp(value - maximum) for move, value in logits.items()}
    total = math.fsum(weights.values())
    return {move: weight / total for move, weight in weights.items()}


def _normalized_legacy_policy(
    policy: dict[str, float] | None, legal_moves: frozenset[str] | None
) -> tuple[dict[str, float] | None, bool]:
    if policy is None or legal_moves is None:
        return None, False
    filtered = {
        move: probability
        for move, probability in policy.items()
        if move in legal_moves and math.isfinite(probability) and probability > 0.0
    }
    if not filtered:
        return None, bool(policy)
    total = math.fsum(filtered.values())
    normalized = {move: probability / total for move, probability in filtered.items()}
    changed = len(filtered) != len(policy) or not math.isclose(total, 1.0, abs_tol=1e-12)
    return normalized, changed


def _entropy(policy: dict[str, float]) -> float:
    return -math.fsum(probability * math.log(probability) for probability in policy.values())


def _variance_normalized_cp_policy(
    cp: dict[str, int],
    dispersion: GlobalDispersionReport,
    config: PolicyCalibrationConfig,
) -> tuple[dict[str, float], float, float, bool]:
    values = tuple(cp.values())
    center, sum_squares = _position_sum_squares(values)
    degrees_of_freedom = len(values) - 1
    posterior_variance = (
        sum_squares + config.prior_strength * dispersion.prior_variance_cp2
    ) / (degrees_of_freedom + config.prior_strength)
    effective_standard_deviation = min(
        config.standard_deviation_cap,
        max(config.standard_deviation_floor, math.sqrt(max(0.0, posterior_variance))),
    )
    denominator = effective_standard_deviation * config.normalized_temperature
    centered_logits = {move: (score - center) / denominator for move, score in cp.items()}
    uncentered_logits = {move: score / denominator for move, score in cp.items()}
    centered_policy = _softmax(centered_logits)
    uncentered_policy = _softmax(uncentered_logits)
    invariance_delta = max(
        abs(centered_policy[move] - uncentered_policy[move]) for move in centered_policy
    )
    raw_variance = sum_squares / degrees_of_freedom if degrees_of_freedom else 0.0
    return (
        centered_policy,
        effective_standard_deviation,
        invariance_delta,
        raw_variance <= config.variance_epsilon,
    )


def _candidate_policy(
    sample: PositionSample,
    raw: _RawCandidates,
    dispersion: GlobalDispersionReport,
    config: PolicyCalibrationConfig,
    accumulator: _TeacherAccumulator,
) -> tuple[dict[str, float] | None, bool]:
    accumulator.excluded_candidates.update(raw.excluded)
    legacy, legacy_changed = _normalized_legacy_policy(sample.teacher_policy, raw.legal_moves)
    if legacy_changed:
        accumulator.excluded_candidates["legacy_policy_entries_sanitized"] += 1
    fallback_reason: str | None = None
    candidate: dict[str, float] | None = None
    from_raw = False
    if raw.invalid_sfen:
        fallback_reason = "invalid_sfen"
    elif raw.missing_variations:
        fallback_reason = "missing_raw_variations"
    elif raw.winning_mates:
        probability = 1.0 / len(raw.winning_mates)
        candidate = {move: probability for move in raw.winning_mates}
        accumulator.excluded_candidates["cp_ignored_when_winning_mate"] += len(raw.cp)
        accumulator.winning_mate_positions += 1
        from_raw = True
    elif raw.cp:
        candidate, effective_std, invariance_delta, zero_variance = (
            _variance_normalized_cp_policy(raw.cp, dispersion, config)
        )
        accumulator.cp_positions += 1
        accumulator.effective_standard_deviations.append(effective_std)
        accumulator.mean_subtraction_max_probability_delta = max(
            accumulator.mean_subtraction_max_probability_delta,
            invariance_delta,
        )
        if len(raw.cp) == 1:
            accumulator.single_candidate_positions += 1
        if zero_variance:
            accumulator.zero_variance_positions += 1
        from_raw = True
    else:
        fallback_reason = "no_eligible_raw_candidates"
    if (
        from_raw
        and sample.teacher_best_move is not None
        and candidate is not None
        and sample.teacher_best_move not in candidate
    ):
        candidate = None
        from_raw = False
        fallback_reason = "best_move_not_eligible"
    if not from_raw:
        accumulator.fallback_positions += 1
        assert fallback_reason is not None
        accumulator.fallback_counts[fallback_reason] += 1
        candidate = legacy
        if candidate is None:
            accumulator.fallback_counts["missing_usable_legacy_policy"] += 1
    else:
        accumulator.candidate_positions += 1
    if legacy is not None and candidate is not None:
        accumulator.legacy_entropies.append(_entropy(legacy))
        accumulator.normalized_entropies.append(_entropy(candidate))
    return candidate, from_raw


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _final_teacher_report(
    source: str,
    dispersion: GlobalDispersionReport,
    accumulator: _TeacherAccumulator,
) -> TeacherPolicyCalibrationReport:
    legacy_mean = _mean_or_none(accumulator.legacy_entropies)
    output_mean = _mean_or_none(accumulator.output_entropies)
    effective = accumulator.effective_standard_deviations
    return TeacherPolicyCalibrationReport(
        teacher_source=source,
        global_dispersion=dispersion,
        samples_seen=accumulator.samples_seen,
        candidate_positions=accumulator.candidate_positions,
        applied_positions=accumulator.applied_positions,
        cp_positions=accumulator.cp_positions,
        winning_mate_positions=accumulator.winning_mate_positions,
        single_candidate_positions=accumulator.single_candidate_positions,
        zero_variance_positions=accumulator.zero_variance_positions,
        fallback_positions=accumulator.fallback_positions,
        fallback_counts=dict(sorted(accumulator.fallback_counts.items())),
        excluded_candidates=dict(sorted(accumulator.excluded_candidates.items())),
        effective_standard_deviation_cp=(
            {
                "minimum": min(effective),
                "mean": mean(effective),
                "maximum": max(effective),
            }
            if effective
            else None
        ),
        entropy=EntropyComparison(
            positions_compared=len(accumulator.legacy_entropies),
            legacy_mean_nats=legacy_mean,
            variance_normalized_mean_nats=_mean_or_none(
                accumulator.normalized_entropies
            ),
            output_mean_nats=output_mean,
            output_minus_legacy_mean_nats=(
                output_mean - legacy_mean
                if output_mean is not None and legacy_mean is not None
                else None
            ),
        ),
        mean_subtraction_max_probability_delta=(
            accumulator.mean_subtraction_max_probability_delta
        ),
    )


def calibrate_teacher_policies(
    games: Sequence[GameRecord],
    config: PolicyCalibrationConfig | None = None,
) -> PolicyCalibrationBuild:
    """Create a replay with either legacy or experimental standardized policies."""

    if config is None:
        config = PolicyCalibrationConfig()
    dispersions = _global_dispersions(games, config)
    accumulators: dict[str, _TeacherAccumulator] = defaultdict(_TeacherAccumulator)
    calibrated_games: list[GameRecord] = []
    for game in games:
        calibrated_samples: list[PositionSample] = []
        for sample in game.samples:
            if sample.teacher_policy is None and not sample.teacher_variations:
                calibrated_samples.append(sample)
                continue
            source = _teacher_source(sample)
            accumulator = accumulators[source]
            accumulator.samples_seen += 1
            raw = _raw_candidates(sample)
            candidate, from_raw = _candidate_policy(
                sample,
                raw,
                dispersions[source],
                config,
                accumulator,
            )
            apply_candidate = config.mode == "variance-normalized" and candidate is not None
            output_sample = replace(sample, teacher_policy=candidate) if apply_candidate else sample
            if apply_candidate and from_raw:
                accumulator.applied_positions += 1
            legacy_policy, _ = _normalized_legacy_policy(
                sample.teacher_policy, raw.legal_moves
            )
            if legacy_policy is not None and candidate is not None:
                output_policy = candidate if apply_candidate else legacy_policy
                accumulator.output_entropies.append(_entropy(output_policy))
            calibrated_samples.append(output_sample)
        calibrated_games.append(replace(game, samples=tuple(calibrated_samples)))
    teacher_reports = {
        source: _final_teacher_report(source, dispersions[source], accumulators[source])
        for source in sorted(accumulators)
    }
    totals = Counter[str]()
    for report in teacher_reports.values():
        totals["samples_seen"] += report.samples_seen
        totals["candidate_positions"] += report.candidate_positions
        totals["applied_positions"] += report.applied_positions
        totals["fallback_positions"] += report.fallback_positions
        totals["cp_positions"] += report.cp_positions
        totals["winning_mate_positions"] += report.winning_mate_positions
    return PolicyCalibrationBuild(
        games=tuple(calibrated_games),
        report=PolicyCalibrationReport(
            schema=POLICY_CALIBRATION_SCHEMA,
            mode=config.mode,
            experimental=True,
            method={
                "cp_policy": (
                    "softmax((cp - position_mean_cp) / "
                    "shrunk_position_standard_deviation_cp / normalized_temperature)"
                ),
                "position_variance": "unbiased within-position candidate variance",
                "variance_shrinkage": (
                    "(within_sum_squares + prior_strength * teacher_prior_variance) / "
                    "(candidate_degrees_of_freedom + prior_strength)"
                ),
                "teacher_prior": "pooled exact-cp within-position variance",
                "winning_mates": "separate score domain; all legal exact winning mates equal",
                "mean_subtraction_softmax_invariant": True,
                "mean_subtraction_identity": "softmax(logits - constant) = softmax(logits)",
                "reference": LOGIT_STANDARDIZATION_PAPER,
            },
            config=cast(dict[str, object], asdict(config)),
            games=len(games),
            samples=sum(len(game.samples) for game in games),
            teachers=teacher_reports,
            totals=dict(sorted(totals.items())),
            limitations=(
                "Logit Standardization was evaluated on classification logits, not shogi cp.",
                "Shogi cp candidates are search outputs, not neural-class logits; "
                "this transfer is experimental.",
                "Variance normalization preserves rank but deliberately discards "
                "absolute cp dispersion.",
                "Use held-out distillation and paired playing-strength tests before promotion.",
            ),
        ),
    )
