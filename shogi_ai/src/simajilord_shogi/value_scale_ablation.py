"""Create-only, held-out-gated ablations of cp-to-value calibration.

The coefficients discussed by AobaNNUE and dlshogi are denominators of a
win-probability sigmoid, ``p = sigmoid(cp / coefficient)``.  Meteo's network
predicts a signed value in ``[-1, 1]`` instead, so the equivalent target is
``2p - 1 = tanh(cp / (2 * coefficient))``.  The factor of two is deliberate:
the pre-existing ``teacher_value_scale`` field records the denominator of the
``tanh`` expression, not the Ponanza coefficient.

This module changes only teacher value targets.  Teacher policies and their
independent centipawn temperature are retained byte-for-byte at the data-model
level.  A teacher-specific fitted arm is emitted only when a training-only
fit is reliable and improves on both fixed coefficients on an independent
validation split.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from .artifact_provenance import (
    canonical_json_sha256,
    identify_file,
    identify_input_artifact,
    sha256_file,
)
from .domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from .ensemble import normalized_sfen
from .replay import append_games, load_games
from .value_scale_fit import (
    TeacherValueScaleFit,
    ValueScaleObservation,
    binary_cross_entropy,
    fit_teacher_value_scales,
)

VALUE_SCALE_ABLATION_SCHEMA = "meteo-value-scale-ablation-v1"
PONANZA_COEFFICIENT_600 = 600.0
DLSHOGI_COEFFICIENT_756 = 756.0864962951762

# Primary-source revisions audited when the formula contract was introduced.
AOBA_NNUE_AUDIT_COMMIT = "8613a0ac911fe35e3fa70aaf012ab6243b7096eb"
AOBA_SCORE_CONVERTER_AUDIT_COMMIT = "b092f262bff8f9e4a0a375d6881285c278923995"
DLSHOGI_AUDIT_COMMIT = "5969e3165ab195f305940623c8a55160fc05a0a5"


@dataclass(frozen=True, slots=True)
class ValueScaleAblationConfig:
    """One shared experimental contract for every emitted training arm."""

    training_seed: int = 0
    steps: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    teacher_policy_mix: float = 0.75
    teacher_value_mix: float = 0.5
    legal_label_smoothing: float = 0.01
    curriculum_depth_ratio: float = 64.0
    minimum_teacher_policy_mix: float = 0.1
    maximum_gradient_norm: float = 1.0
    maximum_probe_loss_ratio: float = 1.25
    bootstrap_resamples: int = 500
    maximum_cross_validation_folds: int = 10
    minimum_fit_games: int = 30
    minimum_fit_samples: int = 200
    minimum_validation_games: int = 10
    minimum_validation_samples: int = 100
    minimum_heldout_bce_improvement: float = 0.0
    minimum_coefficient: float = 1.0
    maximum_coefficient: float = 500_000.0

    def __post_init__(self) -> None:
        if self.steps < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("steps, batch size, and learning rate must be positive")
        if not 0 < self.teacher_policy_mix <= 1 or not 0 <= self.teacher_value_mix <= 1:
            raise ValueError("teacher policy/value mixes must be in (0, 1] and [0, 1]")
        if not 0 <= self.legal_label_smoothing < 1:
            raise ValueError("legal label smoothing must be in [0, 1)")
        if self.curriculum_depth_ratio < 1:
            raise ValueError("curriculum depth ratio must be at least one")
        if not 0 < self.minimum_teacher_policy_mix <= self.teacher_policy_mix:
            raise ValueError(
                "minimum teacher policy mix must be positive and no larger than base mix"
            )
        if self.maximum_gradient_norm <= 0 or self.maximum_probe_loss_ratio < 1:
            raise ValueError("gradient norm must be positive and probe loss ratio at least one")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap resamples must be positive for fit adoption")
        if self.maximum_cross_validation_folds < 2:
            raise ValueError("maximum cross-validation folds must be at least two")
        if min(
            self.minimum_fit_games,
            self.minimum_fit_samples,
            self.minimum_validation_games,
            self.minimum_validation_samples,
        ) < 1:
            raise ValueError("fit and validation reliability thresholds must be positive")
        if self.minimum_heldout_bce_improvement < 0:
            raise ValueError("minimum held-out BCE improvement must be non-negative")
        if (
            not math.isfinite(self.minimum_coefficient)
            or not math.isfinite(self.maximum_coefficient)
            or self.minimum_coefficient <= 0
            or self.maximum_coefficient <= self.minimum_coefficient
        ):
            raise ValueError("coefficient bounds must be finite, positive, and increasing")


@dataclass(frozen=True, slots=True)
class _RootEvidence:
    variation: TeacherVariation
    mate_sign: int | None


@dataclass(frozen=True, slots=True)
class _PreparedSplit:
    games: tuple[GameRecord, ...]
    observations: dict[str, tuple[ValueScaleObservation, ...]]
    exclusions: dict[str, int]
    teacher_sources: tuple[str, ...]
    raw_samples: int
    eligible_samples: int
    exact_cp_samples: int
    exact_mate_samples: int
    position_fingerprint: str
    policy_fingerprint: str


def ponanza_win_probability(cp: int | float, coefficient: float) -> float:
    """Map a side-to-move cp score to win probability using the Ponanza form."""

    if not math.isfinite(float(cp)):
        raise ValueError("cp must be finite")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Ponanza coefficient must be finite and positive")
    logit = float(cp) / coefficient
    if logit >= 0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def ponanza_signed_value(cp: int | float, coefficient: float) -> float:
    """Return the exactly equivalent Meteo target in ``[-1, 1]``."""

    # tanh(cp / (2C)) is algebraically equal to 2*sigmoid(cp/C)-1 and is
    # numerically stable for very large engine evaluations.
    if not math.isfinite(float(cp)):
        raise ValueError("cp must be finite")
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("Ponanza coefficient must be finite and positive")
    return math.tanh(float(cp) / (2.0 * coefficient))


def rescale_cp_between_ponanza_coefficients(
    cp: int | float, *, source_coefficient: float, target_coefficient: float
) -> float:
    """Re-encode one probability while changing its cp coefficient.

    This is the direction used by AobaNNUE's converter: a score encoded with
    756.086... is multiplied by ``600 / 756.086...`` to encode the same
    probability with coefficient 600.
    """

    if not math.isfinite(float(cp)):
        raise ValueError("cp must be finite")
    if (
        not math.isfinite(source_coefficient)
        or not math.isfinite(target_coefficient)
        or source_coefficient <= 0
        or target_coefficient <= 0
    ):
        raise ValueError("source and target coefficients must be finite and positive")
    return float(cp) * target_coefficient / source_coefficient


def _mate_sign(variation: TeacherVariation) -> int:
    if variation.score_kind is not TeacherScoreKind.MATE:
        raise TypeError("centipawn variation has no mate sign")
    if variation.mate_unknown_sign is not None:
        return variation.mate_unknown_sign
    assert variation.mate_plies is not None
    return 1 if variation.mate_plies >= 0 else -1


def _root_evidence(sample: PositionSample) -> tuple[_RootEvidence | None, str | None]:
    if sample.teacher_policy is None or sample.teacher_value is None:
        return None, "without_complete_teacher_target"
    if sample.teacher_source is None or not sample.teacher_source.strip():
        return None, "missing_teacher_source"
    variations = sample.teacher_variations
    if not variations:
        return None, "missing_teacher_variations"
    if sample.teacher_best_move is None:
        root = min(variations, key=lambda variation: variation.rank)
    else:
        matching = [
            variation for variation in variations if variation.move == sample.teacher_best_move
        ]
        if not matching:
            return None, "teacher_best_move_missing_from_variations"
        root = min(matching, key=lambda variation: variation.rank)
    if root.bound is not TeacherScoreBound.EXACT:
        return None, "bounded_root_score"
    return _RootEvidence(
        variation=root,
        mate_sign=(_mate_sign(root) if root.score_kind is TeacherScoreKind.MATE else None),
    ), None


def _terminal_probability(game: GameRecord, sample: PositionSample) -> float:
    if sample.turn not in {0, 1}:
        raise ValueError("sample turn must be black=0 or white=1")
    return 0.5 if game.winner is None else float(game.winner == sample.turn)


def _fingerprint_rows(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        serialized = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(serialized).to_bytes(8, "big"))
        digest.update(serialized)
    return digest.hexdigest()


def _prepare_split(games: Sequence[GameRecord]) -> _PreparedSplit:
    exclusions: Counter[str] = Counter()
    observations: dict[str, list[ValueScaleObservation]] = defaultdict(list)
    retained: list[GameRecord] = []
    position_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    raw_samples = sum(len(game.samples) for game in games)
    exact_cp_samples = 0
    exact_mate_samples = 0
    teacher_sources: set[str] = set()
    for game_id, game in enumerate(games):
        if game.termination is Termination.MAX_PLIES:
            exclusions["incomplete_game_samples"] += len(game.samples)
            retained.append(replace(game, samples=()))
            continue
        retained_samples: list[PositionSample] = []
        for sample_index, sample in enumerate(game.samples):
            evidence, exclusion = _root_evidence(sample)
            if evidence is None:
                assert exclusion is not None
                exclusions[exclusion] += 1
                continue
            source = cast(str, sample.teacher_source)
            teacher_sources.add(source)
            retained_samples.append(sample)
            normalized = normalized_sfen(sample.sfen)
            position_rows.append(
                {
                    "game_index": game_id,
                    "sample_index": sample_index,
                    "normalized_sfen": normalized,
                    "teacher_source": source,
                }
            )
            policy_rows.append(
                {
                    "game_index": game_id,
                    "sample_index": sample_index,
                    "normalized_sfen": normalized,
                    "teacher_source": source,
                    "teacher_policy": sample.teacher_policy,
                    "teacher_policy_temperature": sample.teacher_policy_temperature,
                }
            )
            if evidence.variation.score_kind is TeacherScoreKind.CENTIPAWN:
                assert evidence.variation.score_cp is not None
                exact_cp_samples += 1
                observations[source].append(
                    ValueScaleObservation(
                        game_id=game_id,
                        cp=evidence.variation.score_cp,
                        label=_terminal_probability(game, sample),
                    )
                )
            else:
                exact_mate_samples += 1
        retained.append(replace(game, samples=tuple(retained_samples)))
    eligible_samples = exact_cp_samples + exact_mate_samples
    if eligible_samples == 0:
        raise ValueError("split has no exact root cp or mate teacher targets")
    return _PreparedSplit(
        games=tuple(retained),
        observations={source: tuple(rows) for source, rows in sorted(observations.items())},
        exclusions=dict(sorted(exclusions.items())),
        teacher_sources=tuple(sorted(teacher_sources)),
        raw_samples=raw_samples,
        eligible_samples=eligible_samples,
        exact_cp_samples=exact_cp_samples,
        exact_mate_samples=exact_mate_samples,
        position_fingerprint=_fingerprint_rows(position_rows),
        policy_fingerprint=_fingerprint_rows(policy_rows),
    )


def _normalized_positions(games: Sequence[GameRecord]) -> set[str]:
    return {normalized_sfen(sample.sfen) for game in games for sample in game.samples}


def _outcomes(observations: Sequence[ValueScaleObservation]) -> dict[str, int]:
    return {
        "win": sum(observation.label == 1.0 for observation in observations),
        "draw": sum(observation.label == 0.5 for observation in observations),
        "loss": sum(observation.label == 0.0 for observation in observations),
    }


def _coefficient_bce(
    observations: Sequence[ValueScaleObservation], coefficient: float
) -> float:
    # value_scale_fit historically names the tanh denominator.  Passing 2C
    # yields sigmoid(2cp/(2C)) == sigmoid(cp/C), the Ponanza convention.
    return binary_cross_entropy(observations, 2.0 * coefficient)


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


def _heldout_improvement_bootstrap(
    source: str,
    observations: Sequence[ValueScaleObservation],
    fitted_coefficient: float,
    *,
    resamples: int,
    seed: int,
) -> dict[str, int | float] | None:
    groups: dict[int, list[ValueScaleObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.game_id].append(observation)
    game_ids = sorted(groups)
    if len(game_ids) < 2:
        return None
    source_digest = hashlib.sha256(source.encode("utf-8")).digest()
    source_seed = seed ^ int.from_bytes(source_digest[:4], "big")
    generator = random.Random(source_seed)
    improvements: list[float] = []
    for _ in range(resamples):
        sampled_ids = [generator.choice(game_ids) for _ in game_ids]
        sampled = [observation for game_id in sampled_ids for observation in groups[game_id]]
        fixed_loss = min(
            _coefficient_bce(sampled, PONANZA_COEFFICIENT_600),
            _coefficient_bce(sampled, DLSHOGI_COEFFICIENT_756),
        )
        fitted_loss = _coefficient_bce(sampled, fitted_coefficient)
        improvements.append(fixed_loss - fitted_loss)
    point_fixed = min(
        _coefficient_bce(observations, PONANZA_COEFFICIENT_600),
        _coefficient_bce(observations, DLSHOGI_COEFFICIENT_756),
    )
    point_fitted = _coefficient_bce(observations, fitted_coefficient)
    return {
        "resamples": resamples,
        "seed": source_seed,
        "point_improvement": point_fixed - point_fitted,
        "improvement_p2_5": _percentile(improvements, 0.025),
        "improvement_median": _percentile(improvements, 0.5),
        "improvement_p97_5": _percentile(improvements, 0.975),
    }


def _fit_source_report(
    source: str,
    fit: TeacherValueScaleFit | None,
    validation_observations: Sequence[ValueScaleObservation],
    *,
    config: ValueScaleAblationConfig,
) -> tuple[dict[str, object], float | None]:
    blocking_reasons: list[str] = []
    validation_games = len({observation.game_id for observation in validation_observations})
    if len(validation_observations) < config.minimum_validation_samples:
        blocking_reasons.append("insufficient_validation_samples")
    if validation_games < config.minimum_validation_games:
        blocking_reasons.append("insufficient_validation_game_clusters")
    validation_outcomes = _outcomes(validation_observations)
    if validation_outcomes["win"] == 0 or validation_outcomes["loss"] == 0:
        blocking_reasons.append("limited_validation_outcome_support")
    fixed_bce = {
        "ponanza-600": (
            _coefficient_bce(validation_observations, PONANZA_COEFFICIENT_600)
            if validation_observations
            else None
        ),
        "dlshogi-756.086496": (
            _coefficient_bce(validation_observations, DLSHOGI_COEFFICIENT_756)
            if validation_observations
            else None
        ),
    }
    fitted_coefficient: float | None = None
    fitted_heldout_bce: float | None = None
    heldout_improvement_bootstrap: dict[str, int | float] | None = None
    if fit is None:
        blocking_reasons.append("no_training_fit")
    else:
        fitted_coefficient = fit.fitted_scale / 2.0
        fit_warning_codes = [warning.code for warning in fit.warnings]
        blocking_reasons.extend(f"training_fit:{code}" for code in fit_warning_codes)
        if validation_observations:
            fitted_heldout_bce = _coefficient_bce(
                validation_observations, fitted_coefficient
            )
            fixed_losses = [loss for loss in fixed_bce.values() if loss is not None]
            if fixed_losses and (
                fitted_heldout_bce
                >= min(fixed_losses) - config.minimum_heldout_bce_improvement
            ):
                blocking_reasons.append("no_heldout_improvement_over_best_fixed_coefficient")
            heldout_improvement_bootstrap = _heldout_improvement_bootstrap(
                source,
                validation_observations,
                fitted_coefficient,
                resamples=config.bootstrap_resamples,
                seed=config.training_seed,
            )
            if heldout_improvement_bootstrap is None:
                blocking_reasons.append("heldout_improvement_bootstrap_unavailable")
            elif (
                heldout_improvement_bootstrap["improvement_p2_5"]
                <= config.minimum_heldout_bce_improvement
            ):
                blocking_reasons.append("heldout_improvement_ci_not_above_threshold")
    eligible = not blocking_reasons and fitted_coefficient is not None
    return (
        {
            "teacher_source": source,
            "status": "eligible" if eligible else "blocked",
            "blocking_reasons": blocking_reasons,
            "training_fit": (None if fit is None else asdict(fit)),
            "fitted_ponanza_coefficient": fitted_coefficient,
            "fitted_tanh_denominator": (
                None if fitted_coefficient is None else 2.0 * fitted_coefficient
            ),
            "validation": {
                "samples": len(validation_observations),
                "games": validation_games,
                "outcomes": validation_outcomes,
                "fixed_bce": fixed_bce,
                "fitted_bce": fitted_heldout_bce,
                "fitted_improvement_cluster_bootstrap": heldout_improvement_bootstrap,
            },
        },
        fitted_coefficient if eligible else None,
    )


def _transform_games(
    games: Sequence[GameRecord], coefficients: Mapping[str, float]
) -> tuple[GameRecord, ...]:
    transformed: list[GameRecord] = []
    for game in games:
        samples: list[PositionSample] = []
        for sample in game.samples:
            source = cast(str, sample.teacher_source)
            coefficient = coefficients.get(source)
            if coefficient is None:
                raise ValueError(f"no coefficient configured for teacher {source!r}")
            evidence, exclusion = _root_evidence(sample)
            if evidence is None:
                raise AssertionError(f"prepared sample became ineligible: {exclusion}")
            variation = evidence.variation
            if variation.score_kind is TeacherScoreKind.MATE:
                assert evidence.mate_sign is not None
                teacher_value = float(evidence.mate_sign)
            else:
                assert variation.score_cp is not None
                teacher_value = ponanza_signed_value(variation.score_cp, coefficient)
            samples.append(
                replace(
                    sample,
                    teacher_value=teacher_value,
                    # This field is explicitly the denominator in tanh(cp / scale).
                    teacher_value_scale=2.0 * coefficient,
                )
            )
        transformed.append(replace(game, samples=tuple(samples)))
    return tuple(transformed)


def _checkpoint_identity(directory: Path) -> dict[str, object]:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    metadata = identify_file(resolved / "metadata.json")
    weights = identify_file(resolved / "weights.safetensors")
    digest = hashlib.sha256()
    for path in (Path(metadata.path), Path(weights.path)):
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(resolved),
        "checkpoint_sha256": digest.hexdigest(),
        "metadata": metadata.to_dict(),
        "weights": weights.to_dict(),
    }


def _split_summary(split: _PreparedSplit) -> dict[str, object]:
    return {
        "games": len(split.games),
        "raw_samples": split.raw_samples,
        "eligible_samples": split.eligible_samples,
        "exact_cp_samples": split.exact_cp_samples,
        "exact_mate_samples": split.exact_mate_samples,
        "teacher_sources": list(split.teacher_sources),
        "exclusions": split.exclusions,
        "position_fingerprint": split.position_fingerprint,
        "policy_fingerprint": split.policy_fingerprint,
    }


def _training_command(
    checkpoint: Path,
    train_replay: Path,
    output_checkpoint: Path,
    config: ValueScaleAblationConfig,
) -> list[str]:
    return [
        "simajilord-shogi",
        "train",
        str(checkpoint),
        str(train_replay),
        str(output_checkpoint),
        "--steps",
        str(config.steps),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        repr(config.learning_rate),
        "--seed",
        str(config.training_seed),
        "--teacher-policy-mix",
        repr(config.teacher_policy_mix),
        "--teacher-value-mix",
        repr(config.teacher_value_mix),
        "--legal-label-smoothing",
        repr(config.legal_label_smoothing),
        "--curriculum-depth-ratio",
        repr(config.curriculum_depth_ratio),
        "--minimum-teacher-policy-mix",
        repr(config.minimum_teacher_policy_mix),
        "--max-gradient-norm",
        repr(config.maximum_gradient_norm),
        "--max-probe-loss-ratio",
        repr(config.maximum_probe_loss_ratio),
    ]


def prepare_value_scale_ablation(
    train_replay: Path,
    validation_replay: Path,
    output_directory: Path,
    parent_checkpoint: Path,
    *,
    train_split: str = "train",
    validation_split: str = "validation",
    config: ValueScaleAblationConfig | None = None,
) -> dict[str, object]:
    """Materialize fixed arms and, only if justified, a teacher-fit arm.

    The output directory is create-only and installed atomically.  Any overlap
    in normalized positions between the requested train and validation splits
    fails before an artifact is created.
    """

    resolved_config = config or ValueScaleAblationConfig()
    if not train_split.strip() or not validation_split.strip() or train_split == validation_split:
        raise ValueError("train and validation split names must be non-empty and distinct")
    train_path = train_replay.expanduser().resolve()
    validation_path = validation_replay.expanduser().resolve()
    if train_path == validation_path:
        raise ValueError("train and validation replays must be distinct files")
    output_path = output_directory.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite value-scale ablation: {output_path}")
    train_input = identify_input_artifact(train_path)
    validation_input = identify_input_artifact(validation_path)
    checkpoint = _checkpoint_identity(parent_checkpoint)
    train = _prepare_split(load_games(train_path))
    validation = _prepare_split(load_games(validation_path))
    overlap = _normalized_positions(train.games) & _normalized_positions(validation.games)
    if overlap:
        example = sorted(overlap)[0]
        raise ValueError(
            f"train/validation normalized-position overlap ({len(overlap)}), e.g. {example}"
        )
    if train.teacher_sources != validation.teacher_sources:
        raise ValueError(
            "train and validation must contain the same teacher sources: "
            f"{train.teacher_sources} != {validation.teacher_sources}"
        )

    # The existing fitter's scale is the tanh denominator.  Bounds and baseline
    # are doubled here so its sigmoid(2cp/scale) objective exactly becomes the
    # Ponanza sigmoid(cp/coefficient) objective.
    fit_report = fit_teacher_value_scales(
        train.games,
        baseline_scale=2.0 * PONANZA_COEFFICIENT_600,
        minimum_scale=2.0 * resolved_config.minimum_coefficient,
        maximum_scale=2.0 * resolved_config.maximum_coefficient,
        bootstrap_resamples=resolved_config.bootstrap_resamples,
        seed=resolved_config.training_seed,
        maximum_cross_validation_folds=resolved_config.maximum_cross_validation_folds,
        minimum_reliable_games=resolved_config.minimum_fit_games,
        minimum_reliable_samples=resolved_config.minimum_fit_samples,
    )
    source_fit_reports: dict[str, object] = {}
    fitted_coefficients: dict[str, float] = {}
    for source in train.teacher_sources:
        report, coefficient = _fit_source_report(
            source,
            fit_report.teachers.get(source),
            validation.observations.get(source, ()),
            config=resolved_config,
        )
        source_fit_reports[source] = report
        if coefficient is not None:
            fitted_coefficients[source] = coefficient
    fitted_arm_eligible = len(fitted_coefficients) == len(train.teacher_sources)

    arms: list[tuple[str, dict[str, float], str]] = [
        (
            "ponanza-600",
            {source: PONANZA_COEFFICIENT_600 for source in train.teacher_sources},
            "fixed",
        ),
        (
            "dlshogi-756.086496",
            {source: DLSHOGI_COEFFICIENT_756 for source in train.teacher_sources},
            "fixed",
        ),
    ]
    if fitted_arm_eligible:
        arms.append(("teacher-fit", fitted_coefficients, "training-fit-heldout-gated"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        arm_reports: dict[str, object] = {}
        parent_resolved = Path(cast(str, checkpoint["path"]))
        for arm_name, coefficients, selection in arms:
            train_output_relative = Path("train") / f"{arm_name}.jsonl"
            validation_output_relative = Path("validation") / f"{arm_name}.jsonl"
            train_temporary = temporary_path / train_output_relative
            validation_temporary = temporary_path / validation_output_relative
            append_games(train_temporary, _transform_games(train.games, coefficients))
            append_games(
                validation_temporary,
                _transform_games(validation.games, coefficients),
            )
            train_final = output_path / train_output_relative
            validation_final = output_path / validation_output_relative
            checkpoint_output = output_path / "checkpoints" / arm_name
            arm_reports[arm_name] = {
                "status": "ready",
                "selection": selection,
                "ponanza_coefficients_by_teacher": dict(sorted(coefficients.items())),
                "tanh_denominators_by_teacher": {
                    source: 2.0 * coefficient
                    for source, coefficient in sorted(coefficients.items())
                },
                "train_replay": {
                    "path": str(train_final),
                    "sha256": sha256_file(train_temporary),
                    "bytes": train_temporary.stat().st_size,
                    "samples": train.eligible_samples,
                    "position_fingerprint": train.position_fingerprint,
                    "policy_fingerprint": train.policy_fingerprint,
                },
                "validation_replay": {
                    "path": str(validation_final),
                    "sha256": sha256_file(validation_temporary),
                    "bytes": validation_temporary.stat().st_size,
                    "samples": validation.eligible_samples,
                    "position_fingerprint": validation.position_fingerprint,
                    "policy_fingerprint": validation.policy_fingerprint,
                },
                "training_command": _training_command(
                    parent_resolved,
                    train_final,
                    checkpoint_output,
                    resolved_config,
                ),
                "validation_command": [
                    "simajilord-shogi",
                    "evaluate-distillation",
                    str(checkpoint_output),
                    str(validation_final),
                    "--output",
                    str(output_path / "evaluation" / f"{arm_name}.json"),
                ],
            }
        if not fitted_arm_eligible:
            arm_reports["teacher-fit"] = {
                "status": "blocked",
                "selection": "training-fit-heldout-gated",
                "reason": "one or more teacher-specific fits failed the adoption gate",
                "train_replay": None,
                "validation_replay": None,
                "training_command": None,
                "validation_command": None,
            }

        package_root = Path(__file__).resolve().parent
        source_files = tuple(
            identify_file(package_root / name).to_dict()
            for name in (
                "value_scale_ablation.py",
                "value_scale_fit.py",
                "external_usi.py",
                "model.py",
                "trainer.py",
                "cli.py",
            )
        )
        manifest: dict[str, object] = {
            "schema": VALUE_SCALE_ABLATION_SCHEMA,
            "status": "ready" if fitted_arm_eligible else "fixed-arms-ready-fit-blocked",
            "three_arm_training_ready": fitted_arm_eligible,
            "splits": {
                "train": train_split,
                "validation": validation_split,
                "overlap_guard": "normalized board, side-to-move, and hands; move count omitted",
                "normalized_position_overlap": 0,
            },
            "inputs": {
                "train": train_input.to_dict(),
                "validation": validation_input.to_dict(),
                "parent_checkpoint": checkpoint,
            },
            "data_contract": {
                "train": _split_summary(train),
                "validation": _split_summary(validation),
                "same_positions_across_arms": True,
                "same_teacher_policies_across_arms": True,
                "only_fields_changed": ["teacher_value", "teacher_value_scale"],
            },
            "formula_contract": {
                "coefficient_name": "Ponanza win-probability coefficient C",
                "win_probability": "p = sigmoid(cp / C)",
                "meteo_signed_teacher_value": "v = 2p - 1 = tanh(cp / (2C))",
                "stored_teacher_value_scale": "2C (denominator of tanh, not C)",
                "aobannue_conversion_direction": (
                    "cp_600 = cp_756.0864962951762 * 600 / 756.0864962951762"
                ),
                "trainer_value_target": (
                    "effective_mix * teacher_value + (1-effective_mix) * terminal_signed_result"
                ),
                "trainer_effective_mix": (
                    "teacher_value_mix * effective_teacher_policy_mix / teacher_policy_mix"
                ),
                "trainer_value_loss": "mean((network_signed_value - target)^2)",
                "policy_temperature": (
                    "independent exp(cp / teacher_policy_temperature); unchanged by this ablation"
                ),
                "mate_targets": "exact reported winning/lost mate maps to +1/-1; not fitted",
            },
            "primary_source_audit": {
                "aobannue_readme": {
                    "url": "https://github.com/yssaya/AobaNNUE/blob/"
                    f"{AOBA_NNUE_AUDIT_COMMIT}/README.md",
                    "commit": AOBA_NNUE_AUDIT_COMMIT,
                },
                "aobannue_score_converter": {
                    "url": "https://github.com/yssaya/cshogi_aoba/blob/"
                    f"{AOBA_SCORE_CONVERTER_AUDIT_COMMIT}/psv_shuffle/psvs.cpp",
                    "commit": AOBA_SCORE_CONVERTER_AUDIT_COMMIT,
                },
                "dlshogi_score_to_value": {
                    "url": "https://github.com/TadaoYamaoka/DeepLearningShogi/blob/"
                    f"{DLSHOGI_AUDIT_COMMIT}/cppshogi/cppshogi.h",
                    "commit": DLSHOGI_AUDIT_COMMIT,
                },
                "aobannue_author_claim": (
                    "600 was about +40 Elo stronger than 756 in the author's AobaNNUE setup; "
                    "this is prior evidence, not a guaranteed Meteo gain"
                ),
            },
            "training_contract": asdict(resolved_config),
            "fit_gate": {
                "fit_source": "train split terminal outcomes only",
                "adoption_source": "independent validation split terminal outcomes",
                "sealed_final_test_used": False,
                "all_teachers_must_pass": True,
                "teacher_reports": source_fit_reports,
            },
            "arms": arm_reports,
            "limitations": [
                (
                    "AobaNNUE's reported +40 Elo is prior evidence from that author's model, "
                    "data, and training setup; it does not establish a Meteo gain."
                ),
                (
                    "USI cp scales depend on engine output calibration, including FV_SCALE and "
                    "search strength, so each teacher requires independent held-out evidence."
                ),
                (
                    "Validation selects the coefficient; it is not the sealed final test and "
                    "must not be reused to claim final playing-strength improvement."
                ),
                (
                    "This label ablation holds data and optimizer settings fixed but still "
                    "requires paired arena games to decide playing strength."
                ),
            ],
            "source_files": source_files,
        }
        manifest["experiment_contract_sha256"] = canonical_json_sha256(manifest)
        manifest_path = temporary_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(output_path)
        return manifest
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
