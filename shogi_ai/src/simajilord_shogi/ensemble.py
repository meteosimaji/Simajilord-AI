"""Reproducible same-position aggregation for independent teacher replays.

The training replay contains one convex mixture of the root MultiPV policies
and values.  Full, typed PV sequences cannot be represented as one unambiguous
ranked list after teachers disagree, so they remain separated by teacher in the
sidecar report instead of being mislabeled as properties of the mixture.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from rsshogi.core import Board, Move

from .domain import GameRecord, PositionSample, TeacherVariation, Termination
from .replay import append_games, load_games

ENSEMBLE_REPORT_SCHEMA = "meteo-teacher-ensemble-v1"
ENSEMBLE_TEACHER_SOURCE = "meteo-teacher-ensemble"


@dataclass(frozen=True, slots=True)
class TeacherReplayInput:
    """One independently produced teacher replay and its prior mixture weight."""

    label: str
    replay: Path
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RankedMove:
    move: str
    probability: float


@dataclass(frozen=True, slots=True)
class TeacherOccurrenceReport:
    input_label: str
    input_replay: str
    game_index: int
    sample_index: int
    teacher_source: str
    teacher_context: str | None
    teacher_nodes: int | None
    teacher_time_ms: int | None
    teacher_nps: int | None
    teacher_depth: int | None
    teacher_depth_ratio: float | None
    teacher_value: float
    teacher_best_move: str | None
    teacher_regret: float | None
    teacher_regret_is_lower_bound: bool
    teacher_variations: tuple[TeacherVariation, ...]
    teacher_policy_temperature: float | None
    teacher_value_scale: float | None
    policy_entropy_nats: float
    top_moves: tuple[RankedMove, ...]


@dataclass(frozen=True, slots=True)
class TeacherContributionReport:
    input_label: str
    input_weight: float
    effective_weight: float
    observations_collapsed: int
    averaged_value: float
    averaged_policy_entropy_nats: float
    averaged_top_moves: tuple[RankedMove, ...]
    occurrences: tuple[TeacherOccurrenceReport, ...]


@dataclass(frozen=True, slots=True)
class PositionEnsembleReport:
    normalized_sfen: str
    output_sfen: str
    contributor_count: int
    contributor_labels: tuple[str, ...]
    mixture_top_moves: tuple[RankedMove, ...]
    mixture_policy_entropy_nats: float
    mixture_policy_support_size: int
    teacher_policy_support_union_size: int
    teacher_policy_support_intersection_size: int
    weighted_teacher_policy_entropy_nats: float
    jensen_shannon_divergence_nats: float
    normalized_jensen_shannon_divergence: float
    soft_policy_agreement: float
    mixture_best_move: str
    mixture_best_move_teacher_weight: float
    dominant_teacher_best_move_weight: float
    unanimous_teacher_best_move: bool
    weighted_pairwise_policy_overlap: float
    weighted_pairwise_top_k_jaccard: float
    unanimous_top_k_moves: tuple[str, ...]
    teacher_value: float
    teacher_value_minimum: float
    teacher_value_maximum: float
    teacher_value_width: float
    teacher_value_standard_deviation: float
    positive_value_weight: float
    neutral_value_weight: float
    negative_value_weight: float
    dominant_value_direction_weight: float
    unanimous_value_direction: bool
    contributions: tuple[TeacherContributionReport, ...]


@dataclass(frozen=True, slots=True)
class TeacherInputReport:
    label: str
    replay: str
    replay_sha256: str
    replay_bytes: int
    weight: float
    games: int
    eligible_games: int
    incomplete_games_skipped: int
    teacher_observations: int
    unique_normalized_positions: int
    duplicate_observations_collapsed: int
    provenance_path: str | None
    provenance_sha256: str | None
    provenance: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class OverlapGuardReport:
    replay: str
    replay_sha256: str
    normalized_positions: int


@dataclass(frozen=True, slots=True)
class TeacherEnsembleReport:
    schema: str
    split: str
    sfen_identity: str
    within_input_aggregation: str
    between_teacher_aggregation: str
    minimum_teachers: int
    top_k: int
    value_neutral_threshold: float
    inputs: tuple[TeacherInputReport, ...]
    overlap_guards: tuple[OverlapGuardReport, ...]
    candidate_normalized_positions: int
    output_normalized_positions: int
    output_games: int
    positions_with_all_teachers: int
    dropped_below_minimum_teachers: int
    duplicate_observations_collapsed: int
    contributor_count_histogram: dict[str, int]
    positions: tuple[PositionEnsembleReport, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class TeacherEnsembleBuild:
    games: tuple[GameRecord, ...]
    report: TeacherEnsembleReport


@dataclass(frozen=True, slots=True)
class _Observation:
    input_index: int
    input_label: str
    input_replay: str
    input_weight: float
    game_index: int
    sample_index: int
    game: GameRecord
    sample: PositionSample
    normalized_policy: dict[str, float]
    teacher_value: float


@dataclass(frozen=True, slots=True)
class _TeacherContribution:
    input_index: int
    label: str
    input_weight: float
    policy: dict[str, float]
    value: float
    observations: tuple[_Observation, ...]


@dataclass(frozen=True, slots=True)
class _LoadedInput:
    games: tuple[GameRecord, ...]
    observations: dict[str, tuple[_Observation, ...]]
    report: TeacherInputReport


def normalized_sfen(sfen: str) -> str:
    """Return canonical board/turn/hands fields, deliberately omitting move count."""

    if len(sfen.split()) != 4:
        raise ValueError(f"SFEN must contain exactly four fields: {sfen!r}")
    board = Board(sfen)
    if not board.is_valid():
        raise ValueError(f"invalid SFEN: {sfen!r}")
    canonical_fields = board.to_sfen().split()
    if len(canonical_fields) != 4:
        raise AssertionError("rsshogi returned a non-standard SFEN")
    return " ".join(canonical_fields[:3])


def _output_sfen(normalized: str) -> str:
    return f"{normalized} 1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_policy(
    policy: dict[str, float],
    *,
    sfen: str,
    input_label: str,
) -> dict[str, float]:
    board = Board(sfen)
    accumulated: dict[str, float] = {}
    for move_usi, raw_probability in policy.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0:
            raise ValueError(
                f"teacher {input_label!r} has a negative or non-finite policy probability"
            )
        move = Move.from_usi(move_usi)
        if not board.is_legal_move(move):
            raise ValueError(f"teacher {input_label!r} has illegal move {move_usi} at {sfen}")
        if probability > 0:
            accumulated[move_usi] = accumulated.get(move_usi, 0.0) + probability
    total = math.fsum(accumulated.values())
    if total <= 0:
        raise ValueError(f"teacher {input_label!r} has an empty policy at {sfen}")
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _policy_entropy(policy: dict[str, float]) -> float:
    return -math.fsum(probability * math.log(probability) for probability in policy.values())


def _ranked_moves(policy: dict[str, float], *, limit: int) -> tuple[RankedMove, ...]:
    ordered = sorted(policy.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return tuple(RankedMove(move=move, probability=probability) for move, probability in ordered)


def _policy_average(policies: Sequence[dict[str, float]]) -> dict[str, float]:
    if not policies:
        raise ValueError("at least one policy is required")
    accumulated: dict[str, float] = defaultdict(float)
    for policy in policies:
        for move, probability in policy.items():
            accumulated[move] += probability / len(policies)
    total = math.fsum(accumulated.values())
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _policy_mixture(
    contributions: Sequence[_TeacherContribution], effective_weights: Sequence[float]
) -> dict[str, float]:
    accumulated: dict[str, float] = defaultdict(float)
    for contribution, weight in zip(contributions, effective_weights, strict=True):
        for move, probability in contribution.policy.items():
            accumulated[move] += weight * probability
    total = math.fsum(accumulated.values())
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"teacher policy mixture is not normalized: {total}")
    return {move: accumulated[move] / total for move in sorted(accumulated)}


def _weighted_pairwise_top_k_jaccard(
    contributions: Sequence[_TeacherContribution],
    weights: Sequence[float],
    *,
    top_k: int,
) -> tuple[float, tuple[str, ...]]:
    top_sets = [
        {ranked.move for ranked in _ranked_moves(contribution.policy, limit=top_k)}
        for contribution in contributions
    ]
    unanimous = tuple(sorted(set.intersection(*top_sets))) if top_sets else ()
    if len(top_sets) < 2:
        return 1.0, unanimous
    weighted_overlap = 0.0
    pair_weight_total = 0.0
    for left in range(len(top_sets)):
        for right in range(left + 1, len(top_sets)):
            pair_weight = weights[left] * weights[right]
            union = top_sets[left] | top_sets[right]
            overlap = len(top_sets[left] & top_sets[right]) / len(union)
            weighted_overlap += pair_weight * overlap
            pair_weight_total += pair_weight
    return weighted_overlap / pair_weight_total, unanimous


def _weighted_pairwise_policy_overlap(
    contributions: Sequence[_TeacherContribution], weights: Sequence[float]
) -> float:
    """Weighted soft overlap, sum(min(p, q)), across every teacher pair."""

    if len(contributions) < 2:
        return 1.0
    weighted_overlap = 0.0
    pair_weight_total = 0.0
    for left in range(len(contributions)):
        for right in range(left + 1, len(contributions)):
            pair_weight = weights[left] * weights[right]
            support = contributions[left].policy.keys() | contributions[right].policy.keys()
            overlap = math.fsum(
                min(
                    contributions[left].policy.get(move, 0.0),
                    contributions[right].policy.get(move, 0.0),
                )
                for move in support
            )
            weighted_overlap += pair_weight * overlap
            pair_weight_total += pair_weight
    return weighted_overlap / pair_weight_total


def _read_provenance(replay: Path) -> tuple[str | None, str | None, dict[str, object] | None]:
    provenance_path = replay.with_suffix(replay.suffix + ".provenance.json")
    if not provenance_path.exists():
        return None, None, None
    if not provenance_path.is_file():
        raise ValueError(f"provenance sidecar is not a file: {provenance_path}")
    raw_payload: object = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"provenance sidecar must contain a JSON object: {provenance_path}")
    payload = {str(key): value for key, value in raw_payload.items()}
    return str(provenance_path), _file_sha256(provenance_path), payload


def _load_teacher_input(spec: TeacherReplayInput, input_index: int) -> _LoadedInput:
    replay = spec.replay.expanduser().resolve()
    if not replay.is_file():
        raise FileNotFoundError(replay)
    games = tuple(load_games(replay))
    grouped: dict[str, list[_Observation]] = defaultdict(list)
    incomplete_games = 0
    for game_index, game in enumerate(games):
        if game.termination == Termination.MAX_PLIES:
            incomplete_games += 1
            continue
        for sample_index, sample in enumerate(game.samples):
            has_policy = sample.teacher_policy is not None
            has_value = sample.teacher_value is not None
            if not has_policy and not has_value:
                continue
            if has_policy != has_value:
                raise ValueError(
                    f"teacher {spec.label!r} must provide both policy and value at "
                    f"game {game_index}, sample {sample_index}"
                )
            if sample.teacher_source is None or not sample.teacher_source.strip():
                raise ValueError(
                    f"teacher {spec.label!r} is missing teacher_source at "
                    f"game {game_index}, sample {sample_index}"
                )
            teacher_value = float(cast(float, sample.teacher_value))
            if not math.isfinite(teacher_value) or not -1.0 <= teacher_value <= 1.0:
                raise ValueError(
                    f"teacher {spec.label!r} has value outside [-1, 1] at "
                    f"game {game_index}, sample {sample_index}"
                )
            key = normalized_sfen(sample.sfen)
            canonical_sfen = _output_sfen(key)
            grouped[key].append(
                _Observation(
                    input_index=input_index,
                    input_label=spec.label,
                    input_replay=str(replay),
                    input_weight=spec.weight,
                    game_index=game_index,
                    sample_index=sample_index,
                    game=game,
                    sample=sample,
                    normalized_policy=_normalized_policy(
                        cast(dict[str, float], sample.teacher_policy),
                        sfen=canonical_sfen,
                        input_label=spec.label,
                    ),
                    teacher_value=teacher_value,
                )
            )
    frozen_grouped = {key: tuple(observations) for key, observations in grouped.items()}
    observation_count = sum(len(observations) for observations in frozen_grouped.values())
    provenance_path, provenance_sha256, provenance = _read_provenance(replay)
    report = TeacherInputReport(
        label=spec.label,
        replay=str(replay),
        replay_sha256=_file_sha256(replay),
        replay_bytes=replay.stat().st_size,
        weight=spec.weight,
        games=len(games),
        eligible_games=len(games) - incomplete_games,
        incomplete_games_skipped=incomplete_games,
        teacher_observations=observation_count,
        unique_normalized_positions=len(frozen_grouped),
        duplicate_observations_collapsed=observation_count - len(frozen_grouped),
        provenance_path=provenance_path,
        provenance_sha256=provenance_sha256,
        provenance=provenance,
    )
    return _LoadedInput(games=games, observations=frozen_grouped, report=report)


def _teacher_contribution(observations: tuple[_Observation, ...]) -> _TeacherContribution:
    first = observations[0]
    return _TeacherContribution(
        input_index=first.input_index,
        label=first.input_label,
        input_weight=first.input_weight,
        policy=_policy_average([observation.normalized_policy for observation in observations]),
        value=math.fsum(observation.teacher_value for observation in observations)
        / len(observations),
        observations=observations,
    )


def _occurrence_report(
    observation: _Observation, *, top_k: int
) -> TeacherOccurrenceReport:
    sample = observation.sample
    return TeacherOccurrenceReport(
        input_label=observation.input_label,
        input_replay=observation.input_replay,
        game_index=observation.game_index,
        sample_index=observation.sample_index,
        teacher_source=cast(str, sample.teacher_source),
        teacher_context=sample.teacher_context,
        teacher_nodes=sample.teacher_nodes,
        teacher_time_ms=sample.teacher_time_ms,
        teacher_nps=sample.teacher_nps,
        teacher_depth=sample.teacher_depth,
        teacher_depth_ratio=sample.teacher_depth_ratio,
        teacher_value=observation.teacher_value,
        teacher_best_move=sample.teacher_best_move,
        teacher_regret=sample.teacher_regret,
        teacher_regret_is_lower_bound=sample.teacher_regret_is_lower_bound,
        teacher_variations=sample.teacher_variations or (),
        teacher_policy_temperature=sample.teacher_policy_temperature,
        teacher_value_scale=sample.teacher_value_scale,
        policy_entropy_nats=_policy_entropy(observation.normalized_policy),
        top_moves=_ranked_moves(observation.normalized_policy, limit=top_k),
    )


def _weighted_optional_regret(
    contributions: Sequence[_TeacherContribution], weights: Sequence[float]
) -> tuple[float | None, bool]:
    actor_choices = {
        observation.sample.chosen_move
        for contribution in contributions
        for observation in contribution.observations
    }
    if len(actor_choices) != 1:
        return None, False
    regrets: list[float] = []
    lower_bound = False
    for contribution in contributions:
        contribution_regrets = [
            observation.sample.teacher_regret for observation in contribution.observations
        ]
        if any(regret is None for regret in contribution_regrets):
            return None, False
        regrets.append(
            math.fsum(cast(float, regret) for regret in contribution_regrets)
            / len(contribution_regrets)
        )
        lower_bound = lower_bound or any(
            observation.sample.teacher_regret_is_lower_bound
            for observation in contribution.observations
        )
    return math.fsum(weight * regret for weight, regret in zip(weights, regrets, strict=True)), (
        lower_bound
    )


def _value_direction(value: float, *, neutral_threshold: float) -> str:
    if value > neutral_threshold:
        return "positive"
    if value < -neutral_threshold:
        return "negative"
    return "neutral"


def _position_ensemble(
    normalized: str,
    contributions: Sequence[_TeacherContribution],
    *,
    split: str,
    top_k: int,
    value_neutral_threshold: float,
) -> tuple[PositionSample, PositionEnsembleReport, _Observation]:
    weight_total = math.fsum(contribution.input_weight for contribution in contributions)
    effective_weights = [contribution.input_weight / weight_total for contribution in contributions]
    mixture = _policy_mixture(contributions, effective_weights)
    mixture_entropy = _policy_entropy(mixture)
    teacher_entropies = [_policy_entropy(contribution.policy) for contribution in contributions]
    weighted_teacher_entropy = math.fsum(
        weight * entropy
        for weight, entropy in zip(effective_weights, teacher_entropies, strict=True)
    )
    js_divergence = max(0.0, mixture_entropy - weighted_teacher_entropy)
    normalized_js = (
        js_divergence / math.log(len(contributions)) if len(contributions) > 1 else 0.0
    )
    teacher_value = math.fsum(
        weight * contribution.value
        for weight, contribution in zip(effective_weights, contributions, strict=True)
    )
    value_variance = math.fsum(
        weight * (contribution.value - teacher_value) ** 2
        for weight, contribution in zip(effective_weights, contributions, strict=True)
    )
    value_directions = [
        _value_direction(contribution.value, neutral_threshold=value_neutral_threshold)
        for contribution in contributions
    ]
    direction_weights = {
        direction: math.fsum(
            weight
            for weight, observed_direction in zip(
                effective_weights, value_directions, strict=True
            )
            if observed_direction == direction
        )
        for direction in ("positive", "neutral", "negative")
    }
    best_moves = [
        _ranked_moves(contribution.policy, limit=1)[0].move
        for contribution in contributions
    ]
    mixture_best = _ranked_moves(mixture, limit=1)[0].move
    best_move_weights: dict[str, float] = defaultdict(float)
    for best_move, weight in zip(best_moves, effective_weights, strict=True):
        best_move_weights[best_move] += weight
    pairwise_overlap, unanimous_top_k = _weighted_pairwise_top_k_jaccard(
        contributions, effective_weights, top_k=top_k
    )
    soft_pairwise_overlap = _weighted_pairwise_policy_overlap(
        contributions, effective_weights
    )
    policy_supports = [set(contribution.policy) for contribution in contributions]
    support_union = set.union(*policy_supports)
    support_intersection = set.intersection(*policy_supports)
    contribution_reports = tuple(
        TeacherContributionReport(
            input_label=contribution.label,
            input_weight=contribution.input_weight,
            effective_weight=effective_weight,
            observations_collapsed=len(contribution.observations),
            averaged_value=contribution.value,
            averaged_policy_entropy_nats=teacher_entropy,
            averaged_top_moves=_ranked_moves(contribution.policy, limit=top_k),
            occurrences=tuple(
                _occurrence_report(observation, top_k=top_k)
                for observation in contribution.observations
            ),
        )
        for contribution, effective_weight, teacher_entropy in zip(
            contributions, effective_weights, teacher_entropies, strict=True
        )
    )
    representative = contributions[0].observations[0]
    representative_sample = representative.sample
    actor_best = representative_sample.actor_best_move
    if actor_best is None and representative_sample.policy:
        actor_best = max(
            representative_sample.policy,
            key=representative_sample.policy.__getitem__,
        )
    teacher_regret, regret_is_lower_bound = _weighted_optional_regret(
        contributions, effective_weights
    )
    output_sfen = _output_sfen(normalized)
    output_sample = replace(
        representative_sample,
        sfen=output_sfen,
        turn=Board(output_sfen).turn.value,
        teacher_policy=mixture,
        teacher_value=teacher_value,
        policy_reversal=actor_best is not None and mixture_best != actor_best,
        teacher_best_move=mixture_best,
        teacher_regret=teacher_regret,
        teacher_regret_is_lower_bound=regret_is_lower_bound,
        teacher_nodes=None,
        teacher_depth_ratio=None,
        teacher_time_ms=None,
        teacher_nps=None,
        teacher_depth=None,
        teacher_source=ENSEMBLE_TEACHER_SOURCE,
        teacher_context=f"ensemble:{split}",
        # A single ranked variation list or calibration value would falsely
        # look like a property of the mixture.  The lossless teacher-specific
        # values are retained in the sidecar occurrence reports instead.
        teacher_variations=None,
        teacher_policy_temperature=None,
        teacher_value_scale=None,
    )
    values = [contribution.value for contribution in contributions]
    report = PositionEnsembleReport(
        normalized_sfen=normalized,
        output_sfen=output_sfen,
        contributor_count=len(contributions),
        contributor_labels=tuple(contribution.label for contribution in contributions),
        mixture_top_moves=_ranked_moves(mixture, limit=top_k),
        mixture_policy_entropy_nats=mixture_entropy,
        mixture_policy_support_size=len(mixture),
        teacher_policy_support_union_size=len(support_union),
        teacher_policy_support_intersection_size=len(support_intersection),
        weighted_teacher_policy_entropy_nats=weighted_teacher_entropy,
        jensen_shannon_divergence_nats=js_divergence,
        normalized_jensen_shannon_divergence=normalized_js,
        soft_policy_agreement=max(0.0, 1.0 - normalized_js),
        mixture_best_move=mixture_best,
        mixture_best_move_teacher_weight=best_move_weights.get(mixture_best, 0.0),
        dominant_teacher_best_move_weight=max(best_move_weights.values()),
        unanimous_teacher_best_move=len(set(best_moves)) == 1,
        weighted_pairwise_policy_overlap=soft_pairwise_overlap,
        weighted_pairwise_top_k_jaccard=pairwise_overlap,
        unanimous_top_k_moves=unanimous_top_k,
        teacher_value=teacher_value,
        teacher_value_minimum=min(values),
        teacher_value_maximum=max(values),
        teacher_value_width=max(values) - min(values),
        teacher_value_standard_deviation=math.sqrt(max(0.0, value_variance)),
        positive_value_weight=direction_weights["positive"],
        neutral_value_weight=direction_weights["neutral"],
        negative_value_weight=direction_weights["negative"],
        dominant_value_direction_weight=max(direction_weights.values()),
        unanimous_value_direction=len(set(value_directions)) == 1,
        contributions=contribution_reports,
    )
    return output_sample, report, representative


def _validate_inputs(inputs: Sequence[TeacherReplayInput]) -> tuple[TeacherReplayInput, ...]:
    if len(inputs) < 2:
        raise ValueError("teacher ensemble requires at least two input replays")
    labels: set[str] = set()
    paths: set[str] = set()
    validated: list[TeacherReplayInput] = []
    for spec in inputs:
        label = spec.label.strip()
        if not label:
            raise ValueError("teacher input label must not be empty")
        normalized_label = label.casefold()
        if normalized_label in labels:
            raise ValueError(f"duplicate teacher input label: {label}")
        labels.add(normalized_label)
        replay = spec.replay.expanduser().resolve()
        normalized_path = str(replay).casefold()
        if normalized_path in paths:
            raise ValueError(f"duplicate teacher input replay: {replay}")
        paths.add(normalized_path)
        if not math.isfinite(spec.weight) or spec.weight <= 0:
            raise ValueError(f"teacher input weight must be finite and positive: {label}")
        validated.append(TeacherReplayInput(label=label, replay=replay, weight=spec.weight))
    return tuple(validated)


def _guard_against_split_overlap(
    output_keys: set[str], forbidden_replays: Sequence[Path]
) -> tuple[OverlapGuardReport, ...]:
    reports: list[OverlapGuardReport] = []
    seen: set[str] = set()
    for supplied_path in forbidden_replays:
        replay = supplied_path.expanduser().resolve()
        normalized_path = str(replay).casefold()
        if normalized_path in seen:
            raise ValueError(f"duplicate overlap-guard replay: {replay}")
        seen.add(normalized_path)
        if not replay.is_file():
            raise FileNotFoundError(replay)
        guarded_keys = {
            normalized_sfen(sample.sfen)
            for game in load_games(replay)
            for sample in game.samples
        }
        overlap = output_keys & guarded_keys
        if overlap:
            examples = "; ".join(sorted(overlap)[:3])
            raise ValueError(
                f"split overlap guard rejected {len(overlap)} normalized positions from "
                f"{replay}; examples: {examples}"
            )
        reports.append(
            OverlapGuardReport(
                replay=str(replay),
                replay_sha256=_file_sha256(replay),
                normalized_positions=len(guarded_keys),
            )
        )
    return tuple(reports)


def build_teacher_ensemble(
    inputs: Sequence[TeacherReplayInput],
    *,
    split: str,
    minimum_teachers: int = 2,
    top_k: int = 3,
    value_neutral_threshold: float = 0.05,
    forbid_overlap_with: Sequence[Path] = (),
) -> TeacherEnsembleBuild:
    """Build one split without ever combining positions across split boundaries.

    Every input is first deduplicated by normalized SFEN.  Duplicate observations
    inside one input are averaged equally; independent input distributions are
    then mixed with the supplied positive prior weights, renormalized over the
    teachers available at that position.
    """

    split_name = split.strip()
    if not split_name:
        raise ValueError("split must not be empty")
    validated_inputs = _validate_inputs(inputs)
    if not 2 <= minimum_teachers <= len(validated_inputs):
        raise ValueError("minimum_teachers must be between two and the input count")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not math.isfinite(value_neutral_threshold) or not 0 <= value_neutral_threshold < 1:
        raise ValueError("value_neutral_threshold must be finite and in [0, 1)")

    loaded = tuple(
        _load_teacher_input(spec, input_index)
        for input_index, spec in enumerate(validated_inputs)
    )
    all_keys = set().union(*(teacher_input.observations for teacher_input in loaded))
    contributor_histogram: dict[str, int] = defaultdict(int)
    output_records: dict[tuple[int, int], list[tuple[int, str, PositionSample]]] = defaultdict(
        list
    )
    position_reports: list[PositionEnsembleReport] = []
    output_keys: set[str] = set()
    dropped = 0
    complete_coverage = 0
    for key in sorted(all_keys):
        contributions = [
            _teacher_contribution(teacher_input.observations[key])
            for teacher_input in loaded
            if key in teacher_input.observations
        ]
        contributor_histogram[str(len(contributions))] += 1
        if len(contributions) < minimum_teachers:
            dropped += 1
            continue
        if len(contributions) == len(validated_inputs):
            complete_coverage += 1
        output_sample, position_report, representative = _position_ensemble(
            key,
            contributions,
            split=split_name,
            top_k=top_k,
            value_neutral_threshold=value_neutral_threshold,
        )
        output_records[(representative.input_index, representative.game_index)].append(
            (representative.sample_index, key, output_sample)
        )
        position_reports.append(position_report)
        output_keys.add(key)

    if not output_keys:
        raise ValueError("no normalized position has the required teacher coverage")
    overlap_guards = _guard_against_split_overlap(output_keys, forbid_overlap_with)
    output_games: list[GameRecord] = []
    for input_index, game_index in sorted(output_records):
        source_game = loaded[input_index].games[game_index]
        samples = tuple(
            sample
            for _sample_index, _key, sample in sorted(
                output_records[(input_index, game_index)],
                key=lambda row: (row[0], row[1]),
            )
        )
        output_games.append(replace(source_game, samples=samples))

    report = TeacherEnsembleReport(
        schema=ENSEMBLE_REPORT_SCHEMA,
        split=split_name,
        sfen_identity=(
            "canonical SFEN board, side-to-move, and hands; move counter is ignored"
        ),
        within_input_aggregation=(
            "normalize every observation policy, then arithmetic-mean duplicate "
            "observations for the same normalized SFEN"
        ),
        between_teacher_aggregation=(
            "positive input weights are renormalized over contributors at each position; "
            "policy and value use the same convex weights"
        ),
        minimum_teachers=minimum_teachers,
        top_k=top_k,
        value_neutral_threshold=value_neutral_threshold,
        inputs=tuple(teacher_input.report for teacher_input in loaded),
        overlap_guards=overlap_guards,
        candidate_normalized_positions=len(all_keys),
        output_normalized_positions=len(output_keys),
        output_games=len(output_games),
        positions_with_all_teachers=complete_coverage,
        dropped_below_minimum_teachers=dropped,
        duplicate_observations_collapsed=sum(
            teacher_input.report.duplicate_observations_collapsed for teacher_input in loaded
        ),
        contributor_count_histogram=dict(sorted(contributor_histogram.items())),
        positions=tuple(position_reports),
        limitations=(
            "Each invocation creates exactly one declared split; use forbid_overlap_with to "
            "fail closed against train/validation/test leakage.",
            "The mixed PositionSample deliberately has no single variation list or calibration "
            "scale. Full typed PV sequences and per-teacher calibration values remain separated "
            "in positions[].contributions[].occurrences[].",
            "Best-move disagreement is diagnostic only: it never removes or downweights a "
            "teacher candidate. The output policy preserves the union support and soft mass of "
            "every contributing teacher, including alternative mating moves.",
            "Aggregated search nodes, depth, time, and NPS are intentionally unset on the output "
            "sample because those quantities are not additive across engines; exact per-teacher "
            "values remain in positions[].contributions[].occurrences[].",
        ),
    )
    return TeacherEnsembleBuild(games=tuple(output_games), report=report)


def write_teacher_ensemble(
    build: TeacherEnsembleBuild,
    output: Path,
    *,
    manifest: Path | None = None,
) -> dict[str, object]:
    """Write a new replay and its lineage report without overwriting either target."""

    output_path = output.expanduser().resolve()
    manifest_path = (
        manifest.expanduser().resolve()
        if manifest is not None
        else output_path.with_suffix(output_path.suffix + ".ensemble.json")
    )
    if output_path == manifest_path:
        raise ValueError("ensemble output and manifest paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite ensemble replay: {output_path}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite ensemble manifest: {manifest_path}")
    append_games(output_path, build.games)
    payload = build.report.to_dict()
    payload["output"] = {
        "replay": str(output_path),
        "replay_sha256": _file_sha256(output_path),
        "replay_bytes": output_path.stat().st_size,
        "manifest": str(manifest_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
