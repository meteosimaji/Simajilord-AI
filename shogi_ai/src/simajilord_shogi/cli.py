"""Command-line surface for play, self-play, training, verification, and USI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import rsshogi
from rsshogi.core import Board, Move
from rsshogi.types import Color

from .arbitration import (
    ArbitrationConfig,
    DepthPassInput,
    build_depth_arbitration,
    write_depth_arbitration,
)
from .arena import PairedArenaSummary, benchmark_checkpoint_vs_external
from .checkpoint import load_checkpoint, load_checkpoint_with_training_state, save_checkpoint
from .compute_interlock import InterlockedEvaluator
from .config import (
    ModelProfile,
    ReanalysisConfig,
    SearchConfig,
    model_profile,
    optional_max_plies,
)
from .disagreement import (
    DisagreementConfig,
    DisagreementTeacherInput,
    build_disagreement_replay,
    write_disagreement_replay,
)
from .distillation_metrics import evaluate_teacher_alignment, normalized_position_key
from .distillation_targets import (
    CANONICAL_SCORER_IDS,
    CANONICAL_TARGET_SIDECAR_SCHEMA,
    load_canonical_target_sidecar,
    resolve_canonical_target_sidecar,
)
from .domain import GameRecord
from .ensemble import (
    TeacherReplayInput,
    build_teacher_ensemble,
    write_teacher_ensemble,
)
from .external_usi import (
    ExternalUsiTeacher,
    UsiHistoryMode,
    UsiOptionValueVerification,
)
from .floodgate import (
    FloodgateCorpusConfig,
    create_floodgate_corpus,
    list_7z_csa_members,
    parse_rating_snapshot,
    stream_7z_csa_sources,
)
from .game import play_game, replay_and_validate
from .mcts import MCTS
from .memory import MemoryBudget, configure_mlx_memory
from .model import MLXEvaluator, PolicyValueResNet, assert_model_shapes, parameter_count
from .model_rights import (
    MODEL_RIGHTS,
    RightsDecision,
    analysable_rights_ids,
    distillable_rights_ids,
    locally_distillable_rights_ids,
    model_rights,
)
from .opening_suite import OpeningPosition, load_opening_suite
from .opponent import OpponentProfile
from .policy_calibration import PolicyCalibrationConfig, calibrate_teacher_policies
from .reanalysis import reanalyse_game, reanalyse_game_external
from .replay import (
    append_games,
    evaluation_value_telemetry,
    largest_blunders,
    load_games,
    position_samples,
    search_telemetry,
)
from .rights_lineage import (
    expected_lineage_rights_summary,
    summarize_teacher_sidecar,
)
from .self_improvement import SelfImprovementConfig, run_self_improvement
from .selfplay import batched_self_play, parallel_self_play
from .teacher_data import PackedSfenValueDataset
from .teacher_dedup import build_single_teacher_dedup, write_single_teacher_dedup
from .trainer import TrainingInterlockConfig, train, train_resumable
from .tsume import mine_unique_tsume
from .usi import UsiEngine
from .value_scale_ablation import (
    PONANZA_COEFFICIENT_600,
    ValueScaleAblationConfig,
    prepare_value_scale_ablation,
)
from .value_scale_fit import fit_teacher_value_scales
from .ybb_book import build_ybb_reanalysis_plan, write_ybb_reanalysis_plan


def _profile(value: str) -> ModelProfile:
    allowed = ("smoke", "development", "competition_v1", "competition_v2")
    if value not in allowed:
        raise argparse.ArgumentTypeError(f"choose one of: {', '.join(allowed)}")
    return value  # type: ignore[return-value]


def _search_from_args(args: argparse.Namespace) -> SearchConfig:
    configured_max = optional_max_plies(getattr(args, "max_plies", 0))
    return SearchConfig(
        simulations=args.simulations,
        max_plies=configured_max,
        temperature_moves=getattr(args, "temperature_moves", 24),
        resign_threshold=getattr(args, "resign_threshold", -0.98),
        max_tree_nodes=(getattr(args, "max_tree_nodes", 0) or None),
        memory_check_interval=getattr(args, "memory_check_interval", 1024),
        intra_root_batch_size=getattr(args, "intra_root_batch_size", 1),
        intra_root_virtual_loss=getattr(args, "intra_root_virtual_loss", 0.0),
        max_evaluation_batch_size=getattr(args, "max_evaluation_batch_size", 256),
    )


def _engine_options(values: Sequence[str]) -> dict[str, str | int]:
    options: dict[str, str | int] = {}
    normalized_names: set[str] = set()
    for option in values:
        if "=" not in option:
            raise ValueError(f"engine option must be NAME=VALUE: {option}")
        raw_name, value = option.split("=", 1)
        name = raw_name.strip()
        if not name:
            raise ValueError("engine option name must not be empty")
        normalized = name.casefold()
        if normalized == "multipv":
            raise ValueError("MultiPV is reserved; use --multipv")
        if normalized in normalized_names:
            raise ValueError(f"duplicate engine option name: {name}")
        normalized_names.add(normalized)
        options[name] = value
    return options


def _require_suisho11plus_teacher_options(
    options: dict[str, str | int], *, multipv: int
) -> None:
    """Reject a runnable-looking but misconfigured Suisho11Plus teacher."""

    normalized = {name.casefold(): str(value).strip() for name, value in options.items()}
    required = {
        "fv_scale": "40",
        "usi_ownbook": "false",
        "bookfile": "no_book",
        "pvinterval": "0",
    }
    for name, expected in required.items():
        observed = normalized.get(name)
        if observed is None:
            raise ValueError(f"Suisho11Plus requires explicit engine option {name}")
        if observed.casefold() != expected:
            raise ValueError(
                f"Suisho11Plus engine option {name} must be {expected!r}, got {observed!r}"
            )
    eval_directory = normalized.get("evaldir")
    if eval_directory is None:
        raise ValueError("Suisho11Plus requires explicit engine option EvalDir")
    resolved_eval = Path(eval_directory).expanduser().resolve()
    if not resolved_eval.is_dir() or resolved_eval.is_symlink():
        raise ValueError("Suisho11Plus EvalDir must be an existing non-symlink directory")
    for name in ("threads", "usi_hash"):
        raw_value = normalized.get(name)
        if raw_value is None:
            raise ValueError(f"Suisho11Plus requires explicit engine option {name}")
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"Suisho11Plus engine option {name} must be an integer") from error
        if value < 1:
            raise ValueError(f"Suisho11Plus engine option {name} must be positive")
    if "hash" in normalized:
        raise ValueError("Suisho11Plus uses USI_Hash, not Hash")
    if multipv < 8:
        raise ValueError("Suisho11Plus distillation requires MultiPV of at least 8")


def _resolve_ponanza_value_scale(
    *,
    ponanza_coefficient: float | None,
    legacy_tanh_denominator: float | None,
) -> tuple[float, float, str]:
    """Resolve probability coefficient C and signed-value tanh denominator 2C.

    Historical Meteo flags called the raw denominator in ``tanh(cp / D)`` a
    "scale".  Explicit uses of those flags keep that exact meaning.  New flags
    accept the Ponanza/dlshogi coefficient in ``sigmoid(cp / C)`` instead.
    """

    if ponanza_coefficient is not None and legacy_tanh_denominator is not None:
        raise ValueError(
            "Ponanza coefficient and legacy tanh denominator are mutually exclusive"
        )
    if ponanza_coefficient is None and legacy_tanh_denominator is None:
        ponanza_coefficient = PONANZA_COEFFICIENT_600
        convention = "default_ponanza_coefficient"
    elif ponanza_coefficient is not None:
        convention = "explicit_ponanza_coefficient"
    else:
        convention = "explicit_legacy_tanh_denominator"

    if ponanza_coefficient is not None:
        if not math.isfinite(ponanza_coefficient) or ponanza_coefficient <= 0.0:
            raise ValueError("Ponanza coefficient must be finite and positive")
        tanh_denominator = 2.0 * ponanza_coefficient
        if not math.isfinite(tanh_denominator):
            raise ValueError("twice the Ponanza coefficient must be finite")
        return ponanza_coefficient, tanh_denominator, convention

    assert legacy_tanh_denominator is not None
    if not math.isfinite(legacy_tanh_denominator) or legacy_tanh_denominator <= 0.0:
        raise ValueError("legacy tanh denominator must be finite and positive")
    return legacy_tanh_denominator / 2.0, legacy_tanh_denominator, convention


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(directory: Path) -> str:
    """Hash the exact metadata and weights consumed by load_checkpoint, in order."""

    digest = hashlib.sha256()
    for name in ("metadata.json", "weights.safetensors"):
        file_path = directory / name
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _artifact_provenance(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Resolve and hash every separately licensed teacher artifact once."""

    artifacts: list[dict[str, object]] = []
    seen: set[str] = set()
    for supplied_path in paths:
        artifact_path = supplied_path.expanduser().resolve()
        normalized = str(artifact_path).casefold()
        if normalized in seen:
            raise ValueError(f"duplicate teacher artifact: {supplied_path}")
        seen.add(normalized)
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        artifacts.append(
            {
                "path": str(artifact_path),
                "sha256": _sha256_file(artifact_path),
                "bytes": artifact_path.stat().st_size,
            }
        )
    return artifacts


def _source_tree_provenance() -> dict[str, object]:
    """Hash the exact importable Python source, including an uncommitted tree."""

    package_root = Path(__file__).resolve().parent
    paths = sorted(package_root.rglob("*.py"))
    marker = package_root / "py.typed"
    if marker.is_file():
        paths.append(marker)
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return {
        "package_root": str(package_root),
        "sha256": digest.hexdigest(),
        "files": len(paths),
    }


def _git_source_identity() -> dict[str, object]:
    """Describe the Git commit and dirty state without assuming Git is installed."""

    package_root = Path(__file__).resolve().parent

    def git_output(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=package_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.rstrip("\n")

    try:
        repository_root = Path(git_output("rev-parse", "--show-toplevel")).resolve()
        commit = git_output("rev-parse", "HEAD")
        status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "available": False,
            "repository_root": None,
            "commit": None,
            "branch": None,
            "dirty": None,
            "status_entries": None,
            "status_sha256": None,
        }
    try:
        branch: str | None = git_output("symbolic-ref", "--quiet", "--short", "HEAD")
    except subprocess.CalledProcessError:
        branch = None
    status_bytes = status.encode("utf-8")
    return {
        "available": True,
        "repository_root": str(repository_root),
        "source_root_relative": str(package_root.relative_to(repository_root)),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
        # Hash rather than embed local filenames from unrelated dirty work.
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
    }


def _checkpoint_provenance(directory: Path) -> dict[str, object]:
    """Hash every checkpoint file and preserve its complete JSON metadata/lineage."""

    try:
        resolved = directory.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"checkpoint does not exist or is not readable: {directory}") from error
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    entries = sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    symlinks = [entry for entry in entries if entry.is_symlink()]
    if symlinks:
        raise ValueError(f"checkpoint must not contain symlinks: {symlinks[0]}")
    files = [entry for entry in entries if entry.is_file()]
    if not files:
        raise ValueError(f"checkpoint contains no files: {resolved}")
    aggregate = hashlib.sha256()
    file_records: list[dict[str, object]] = []
    for file_path in files:
        relative = file_path.relative_to(resolved).as_posix()
        byte_count = file_path.stat().st_size
        relative_bytes = relative.encode("utf-8")
        aggregate.update(len(relative_bytes).to_bytes(8, "big"))
        aggregate.update(relative_bytes)
        aggregate.update(byte_count.to_bytes(8, "big"))
        file_digest = hashlib.sha256()
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                aggregate.update(chunk)
                file_digest.update(chunk)
        file_records.append(
            {
                "relative_path": relative,
                "sha256": file_digest.hexdigest(),
                "bytes": byte_count,
            }
        )
    metadata_path = resolved / "metadata.json"
    weights_path = resolved / "weights.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise ValueError("checkpoint requires metadata.json and weights.safetensors")
    try:
        metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint metadata is not valid UTF-8 JSON: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return {
        "path": str(resolved),
        "all_files_sha256": aggregate.hexdigest(),
        "load_checkpoint_sha256": _checkpoint_sha256(resolved),
        "file_count": len(file_records),
        "files": file_records,
        "metadata": metadata,
        "lineage": metadata.get("lineage"),
    }


def _score_to_elo(score: float) -> float | None:
    """Map a finite interior score to logistic Elo; endpoints are unbounded."""

    if score <= 0.0 or score >= 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def _paired_elo_report(summary: PairedArenaSummary) -> dict[str, object]:
    """Expose Elo only for integrity-valid production opening-pair results."""

    blockers: list[str] = []
    if summary.legacy_single_opening:
        blockers.append("legacy_single_opening_debug_only")
    if not summary.promotion_eligible:
        blockers.append("production_promotion_disabled")
    if summary.repeated_opening_pairs:
        blockers.append("repeated_normalized_openings")
    if summary.independent_opening_pairs < summary.promotion_min_pairs:
        blockers.append("insufficient_independent_opening_pairs")
    if summary.incomplete_games:
        blockers.append("incomplete_games")
    if summary.training_arena_overlap_count:
        blockers.append("training_arena_split_overlap")
    if blockers:
        return {
            "eligible": False,
            "estimate": None,
            "cluster_bootstrap_lower_95": None,
            "cluster_bootstrap_upper_95": None,
            "method": None,
            "blockers": blockers,
        }
    return {
        "eligible": True,
        "estimate": _score_to_elo(summary.score),
        "cluster_bootstrap_lower_95": _score_to_elo(
            summary.cluster_bootstrap_lower_95
        ),
        "cluster_bootstrap_upper_95": _score_to_elo(
            summary.cluster_bootstrap_upper_95
        ),
        "method": "logistic transform of opening-cluster mean and one-sided bootstrap bounds",
        "unbounded_endpoint_is_null": True,
        "blockers": [],
    }


def _benchmark_opening_configuration(
    args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, object], bool]:
    """Resolve either a production suite split or one explicit legacy debug pair."""

    if args.promotion_min_pairs < 1:
        raise ValueError("promotion minimum opening pairs must be positive")
    if not 0.0 <= args.promotion_lower_bound <= 1.0:
        raise ValueError("promotion lower bound must be between zero and one")
    if args.bootstrap_iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if args.seed < 0:
        raise ValueError("benchmark seed must be non-negative")

    if args.opening_suite is not None:
        if args.initial_sfen:
            raise ValueError("--initial-sfen is available only in legacy debug mode")
        if args.promotion_min_pairs < 32:
            raise ValueError("production USI benchmark requires promotion-min-pairs >= 32")
        suite = load_opening_suite(args.opening_suite)
        split = suite.split(args.opening_split)
        if len(split.positions) < args.promotion_min_pairs:
            raise ValueError(
                "production USI benchmark opening split has fewer positions than "
                f"promotion-min-pairs: {len(split.positions)} < {args.promotion_min_pairs}"
            )
        expected_games = 2 * len(split.positions)
        if args.games is not None and args.games != expected_games:
            raise ValueError(
                f"--games must equal twice the selected opening count ({expected_games})"
            )
        openings = tuple(position.sfen for position in split.positions)
        opening_manifest: dict[str, object] = {
            "mode": "immutable_split_suite",
            "source": str(suite.source),
            "source_sha256": suite.source_sha256,
            "normalized_suite_sha256": suite.normalized_sha256,
            "split": split.name,
            "split_sha256": split.sha256,
            "count": len(split.positions),
            "normalized_keys": [position.normalized_key for position in split.positions],
            "opening_sha256": [position.sha256 for position in split.positions],
            "promotion_and_elo_eligible_input": True,
        }
        return openings, opening_manifest, False

    if not args.legacy_single_opening_debug:
        raise AssertionError("argparse did not select a benchmark opening mode")
    if len(args.initial_sfen) > 1:
        raise ValueError("legacy external benchmark accepts at most one --initial-sfen")
    if args.games is not None and args.games != 2:
        raise ValueError("legacy single-opening debug benchmark always runs exactly two games")
    position = OpeningPosition.from_sfen(
        args.initial_sfen[0] if args.initial_sfen else Board().to_sfen()
    )
    return (
        (position.sfen,),
        {
            "mode": "legacy_single_opening_debug_only",
            "source": "cli --initial-sfen" if args.initial_sfen else "rsshogi standard start",
            "source_sha256": position.sha256,
            "normalized_suite_sha256": None,
            "split": None,
            "split_sha256": None,
            "count": 1,
            "normalized_keys": [position.normalized_key],
            "opening_sha256": [position.sha256],
            "promotion_and_elo_eligible_input": False,
        },
        True,
    )


def _training_input_provenance(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Hash replays and link, without embedding, their adjacent lineage sidecars."""

    records: list[dict[str, object]] = []
    for input_record in _artifact_provenance(paths):
        replay = Path(str(input_record["path"]))
        sidecars: list[dict[str, object]] = []
        for suffix in (".ensemble.json", ".provenance.json"):
            sidecar = replay.with_suffix(replay.suffix + suffix)
            if sidecar.is_file():
                resolved_sidecar = sidecar.expanduser().resolve()
                sidecar_bytes = resolved_sidecar.read_bytes()
                sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
                try:
                    sidecar_payload: object = json.loads(sidecar_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"training lineage sidecar is not valid UTF-8 JSON: {sidecar}"
                    ) from error
                if not isinstance(sidecar_payload, dict) or not all(
                    isinstance(key, str) for key in sidecar_payload
                ):
                    raise ValueError(
                        f"training lineage sidecar must be a JSON object: {sidecar}"
                    )
                sidecars.append(
                    {
                        "path": str(resolved_sidecar),
                        "sha256": sidecar_sha256,
                        "bytes": len(sidecar_bytes),
                        "rights_restriction_summary": summarize_teacher_sidecar(
                            sidecar_payload,
                            sidecar_sha256=sidecar_sha256,
                        ),
                    }
                )
        records.append({**input_record, "lineage_sidecars": sidecars})
    return records


def _training_dataset_fingerprint(
    paths: Sequence[Path],
    *,
    selection: dict[str, object] | None = None,
    input_provenance: Sequence[dict[str, object]] | None = None,
) -> str:
    """Hash consumed bytes, lineage bytes, ordering, and target-selection semantics."""

    identities: list[dict[str, object]] = []
    records = (
        _training_input_provenance(paths)
        if input_provenance is None
        else list(input_provenance)
    )
    if len(records) != len(paths):
        raise ValueError("training provenance count does not match the ordered inputs")
    for supplied_path, record in zip(paths, records, strict=True):
        if Path(str(record.get("path"))) != supplied_path.expanduser().resolve():
            raise ValueError("training provenance path does not match its ordered input")
        raw_sidecars = record["lineage_sidecars"]
        if not isinstance(raw_sidecars, list):
            raise TypeError("training lineage sidecars must be a list")
        identities.append(
            {
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "lineage": [
                    {"sha256": sidecar["sha256"], "bytes": sidecar["bytes"]}
                    for sidecar in raw_sidecars
                ],
            }
        )
    payload = {
        "schema": "meteo-training-dataset-identity-v1",
        "ordered_inputs": identities,
        "selection": selection or {},
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_new_checkpoint_output(output: Path, *, parent: Path) -> None:
    output_path = output.expanduser().resolve()
    parent_path = parent.expanduser().resolve()
    if str(output_path).casefold() == str(parent_path).casefold():
        raise ValueError("training output must be a new checkpoint, not the parent")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {output_path}")


def _verify_training_input_provenance_unchanged(
    paths: Sequence[Path], expected: Sequence[dict[str, object]]
) -> None:
    """Fail before checkpoint creation if any replay or sidecar changed mid-run."""

    if _training_input_provenance(paths) != list(expected):
        raise ValueError("training input or lineage sidecar changed during training")


def _training_lineage(
    *,
    parent: Path,
    inputs: Sequence[Path],
    current_step: int,
    teacher_sources: Sequence[str],
    hyperparameters: dict[str, object],
    dataset_fingerprint: str,
    optimizer_state_restored: bool,
    input_provenance: Sequence[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Describe either the first exact-state segment or an exact continuation."""

    parent_path = parent.expanduser().resolve()
    training_inputs = (
        _training_input_provenance(inputs)
        if input_provenance is None
        else json.loads(json.dumps(list(input_provenance), allow_nan=False, sort_keys=True))
    )
    if not isinstance(training_inputs, list):
        raise TypeError("training input provenance must be a JSON list")
    lineage: dict[str, object] = {
        "schema": "meteo-training-lineage-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "method": (
            "exact-resume teacher distillation"
            if optimizer_state_restored
            else "warm-start first exact-state teacher-distillation segment"
        ),
        "parent_checkpoint": {
            **_checkpoint_provenance(parent_path),
            "step": current_step,
        },
        "training_inputs": training_inputs,
        "dataset_fingerprint": dataset_fingerprint,
        "teacher_sources": sorted(set(teacher_sources)),
        "hyperparameters": hyperparameters,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "constant",
            "state_restored": optimizer_state_restored,
        },
        "rng_state_restored": optimizer_state_restored,
        "exact_resume_from_parent": optimizer_state_restored,
        "output_contains_exact_resume_state": True,
        "source_tree": _source_tree_provenance(),
        "git": _git_source_identity(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": getattr(mx, "__version__", None),
            "rsshogi": getattr(rsshogi, "__version__", None),
        },
    }
    lineage["rights_restriction_summary"] = expected_lineage_rights_summary(lineage)
    return lineage


def _ensemble_teacher_inputs(
    teacher_specs: Sequence[str], weight_specs: Sequence[str]
) -> tuple[TeacherReplayInput, ...]:
    """Parse LABEL=REPLAY and LABEL=WEIGHT pairs without positional ambiguity."""

    replay_by_label: dict[str, tuple[str, Path]] = {}
    for teacher_spec in teacher_specs:
        raw_label, separator, raw_path = teacher_spec.partition("=")
        label = raw_label.strip()
        if not separator or not label or not raw_path.strip():
            raise ValueError(f"teacher must be LABEL=REPLAY: {teacher_spec}")
        normalized_label = label.casefold()
        if normalized_label in replay_by_label:
            raise ValueError(f"duplicate teacher label: {label}")
        replay_by_label[normalized_label] = (label, Path(raw_path.strip()))

    weight_by_label: dict[str, float] = {}
    for weight_spec in weight_specs:
        raw_label, separator, raw_weight = weight_spec.partition("=")
        label = raw_label.strip()
        if not separator or not label or not raw_weight.strip():
            raise ValueError(f"weight must be LABEL=WEIGHT: {weight_spec}")
        normalized_label = label.casefold()
        if normalized_label in weight_by_label:
            raise ValueError(f"duplicate teacher weight: {label}")
        if normalized_label not in replay_by_label:
            raise ValueError(f"weight names unknown teacher: {label}")
        try:
            weight_by_label[normalized_label] = float(raw_weight)
        except ValueError as error:
            raise ValueError(f"teacher weight must be numeric: {weight_spec}") from error

    return tuple(
        TeacherReplayInput(
            label=label,
            replay=replay,
            weight=weight_by_label.get(normalized_label, 1.0),
        )
        for normalized_label, (label, replay) in replay_by_label.items()
    )


def _named_text_specs(
    specs: Sequence[str], *, option_name: str
) -> dict[str, tuple[str, str]]:
    """Parse repeatable NAME=VALUE options with case-insensitive collision checks."""

    parsed: dict[str, tuple[str, str]] = {}
    for spec in specs:
        raw_name, separator, raw_value = spec.partition("=")
        name = raw_name.strip()
        value = raw_value.strip()
        if not separator or not name or not value:
            raise ValueError(f"{option_name} must be NAME=VALUE: {spec}")
        key = name.casefold()
        if key in parsed:
            raise ValueError(f"duplicate {option_name} name: {name}")
        parsed[key] = (name, value)
    return parsed


def _disagreement_teacher_inputs(
    teacher_specs: Sequence[str], family_specs: Sequence[str]
) -> tuple[DisagreementTeacherInput, ...]:
    replays = _named_text_specs(teacher_specs, option_name="teacher")
    families = _named_text_specs(family_specs, option_name="family")
    if set(replays) != set(families):
        missing = sorted(set(replays) - set(families))
        extra = sorted(set(families) - set(replays))
        raise ValueError(
            f"family labels must exactly match teachers; missing={missing}, extra={extra}"
        )
    return tuple(
        DisagreementTeacherInput(
            label=label,
            replay=Path(replay),
            family=families[key][1],
        )
        for key, (label, replay) in replays.items()
    )


def _depth_pass_inputs(
    replay_specs: Sequence[str],
    family_specs: Sequence[str],
    depth_specs: Sequence[str],
    scope_specs: Sequence[str],
) -> tuple[DepthPassInput, ...]:
    replays = _named_text_specs(replay_specs, option_name="pass")
    families = _named_text_specs(family_specs, option_name="pass-family")
    depths = _named_text_specs(depth_specs, option_name="pass-depth")
    scopes = _named_text_specs(scope_specs, option_name="pass-scope")
    for option_name, supplied in (("pass-family", families), ("pass-depth", depths)):
        if set(supplied) != set(replays):
            missing = sorted(set(replays) - set(supplied))
            extra = sorted(set(supplied) - set(replays))
            raise ValueError(
                f"{option_name} labels must exactly match passes; "
                f"missing={missing}, extra={extra}"
            )
    unknown_scopes = sorted(set(scopes) - set(replays))
    if unknown_scopes:
        raise ValueError(f"pass-scope names unknown passes: {unknown_scopes}")
    parsed: list[DepthPassInput] = []
    for key, (label, replay) in replays.items():
        try:
            depth = int(depths[key][1])
        except ValueError as error:
            raise ValueError(f"pass-depth must be an integer: {depths[key][1]}") from error
        parsed.append(
            DepthPassInput(
                label=label,
                replay=Path(replay),
                family=families[key][1],
                depth=depth,
                validation_scope=scopes.get(key, (label, "in_domain"))[1],
            )
        )
    return tuple(parsed)


def _positive_named_floats(
    specs: Sequence[str], *, option_name: str
) -> dict[str, float]:
    parsed = _named_text_specs(specs, option_name=option_name)
    result: dict[str, float] = {}
    for _key, (name, raw_value) in parsed.items():
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"{option_name} must be numeric: {name}={raw_value}") from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{option_name} must be finite and positive: {name}")
        result[name] = value
    return result


def _positive_phase_family_floats(
    specs: Sequence[str], *, option_name: str
) -> dict[str, dict[str, float]]:
    """Parse repeatable PHASE:FAMILY=WEIGHT options without silent collisions."""

    result: dict[str, dict[str, float]] = {}
    phase_spellings: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for spec in specs:
        raw_name, separator, raw_value = spec.partition("=")
        raw_phase, family_separator, raw_family = raw_name.partition(":")
        phase = raw_phase.strip()
        family = raw_family.strip()
        if (
            not separator
            or not family_separator
            or not phase
            or not family
            or not raw_value.strip()
        ):
            raise ValueError(f"{option_name} must be PHASE:FAMILY=WEIGHT: {spec}")
        phase_key = phase.casefold()
        family_key = family.casefold()
        pair = (phase_key, family_key)
        if pair in seen_pairs:
            raise ValueError(f"duplicate {option_name} name: {phase}:{family}")
        seen_pairs.add(pair)
        previous_phase = phase_spellings.setdefault(phase_key, phase)
        if previous_phase != phase:
            raise ValueError(
                f"{option_name} phase spelling must be consistent: "
                f"{previous_phase!r} versus {phase!r}"
            )
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(
                f"{option_name} must be numeric: {phase}:{family}={raw_value}"
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"{option_name} must be finite and positive: {phase}:{family}"
            )
        result.setdefault(previous_phase, {})[family] = value
    return result


def _require_limited_local_destinations(
    local_root: Path, destinations: Sequence[Path]
) -> Path:
    """Keep restricted teacher artifacts outside Git, or under an ignored local root."""

    root = local_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("local-only root must be an existing, non-symlink directory")
    resolved_destinations = tuple(
        destination.expanduser().resolve() for destination in destinations
    )
    for destination in resolved_destinations:
        if destination == root or not destination.is_relative_to(root):
            raise ValueError(
                f"limited-local output must be a child of local-only root: {destination}"
            )

    repository = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if repository.returncode != 0:
        return root
    repository_root = Path(repository.stdout.strip()).resolve()
    for destination in resolved_destinations:
        if not destination.is_relative_to(repository_root):
            continue
        relative = destination.relative_to(repository_root)
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", str(relative)],
            cwd=repository_root,
            check=False,
        )
        if ignored.returncode != 0:
            raise ValueError(
                "limited-local output inside the repository must be covered by .gitignore: "
                f"{destination}"
            )
    return root


def _add_distillation_safety_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--teacher-policy-mix", type=float, default=0.75)
    parser.add_argument("--teacher-value-mix", type=float, default=0.5)
    parser.add_argument("--legal-label-smoothing", type=float, default=0.01)
    parser.add_argument("--curriculum-depth-ratio", type=float, default=64.0)
    parser.add_argument("--minimum-teacher-policy-mix", type=float, default=0.1)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--max-probe-loss-ratio", type=float, default=1.25)


def _add_training_interlock_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--human-play-state-root",
        type=Path,
        help=(
            "local ShogiHome/Meteo state root; when set, MLX batches/steps serialize with "
            "each active human USI go while evolution continues between human searches"
        ),
    )
    parser.add_argument(
        "--human-play-wait-timeout-seconds",
        type=float,
        default=86_400.0,
        help=(
            "maximum time each evolution compute lease waits for an active human USI go or "
            "another evolution lease (default 24h)"
        ),
    )
    parser.add_argument("--human-play-poll-interval-seconds", type=float, default=0.05)
    parser.add_argument("--training-lease-ttl-seconds", type=float, default=60.0)
    parser.add_argument("--training-lease-heartbeat-seconds", type=float, default=5.0)


def _training_interlock_from_args(
    args: argparse.Namespace, *, generation: int | None = None
) -> TrainingInterlockConfig | None:
    state_root = getattr(args, "human_play_state_root", None)
    if state_root is None:
        return None
    return TrainingInterlockConfig(
        state_root=state_root,
        generation=generation,
        ttl_seconds=args.training_lease_ttl_seconds,
        heartbeat_interval_seconds=args.training_lease_heartbeat_seconds,
        wait_timeout_seconds=args.human_play_wait_timeout_seconds,
        poll_interval_seconds=args.human_play_poll_interval_seconds,
    )


def _add_search_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-tree-nodes",
        type=int,
        default=0,
        help="tree high-water mark; zero derives a safe limit from currently available memory",
    )
    parser.add_argument("--memory-check-interval", type=int, default=1024)
    parser.add_argument(
        "--intra-root-batch-size",
        type=int,
        default=1,
        help=(
            "neural leaves reserved per root; 1 preserves historical PUCT, while values above "
            "1 are experimental until a paired playing-strength gate passes"
        ),
    )
    parser.add_argument(
        "--intra-root-virtual-loss",
        type=float,
        default=0.0,
        help="experimental virtual loss for pending batched leaves; zero is quality-first",
    )
    parser.add_argument(
        "--max-evaluation-batch-size",
        type=int,
        default=256,
        help="hard cap on one neural evaluator batch",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simajilord-shogi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check rules, MLX, and model shapes")
    doctor.add_argument("--profile", type=_profile, default="smoke")

    nps_benchmark = subparsers.add_parser(
        "bench-nps", help="measure neural MCTS nodes/s, elapsed time, and peak unified memory"
    )
    nps_benchmark.add_argument("--profile", type=_profile, default="competition_v1")
    nps_benchmark.add_argument("--parallel-roots", type=int, default=32)
    nps_benchmark.add_argument("--simulations", type=int, default=100)
    nps_benchmark.add_argument("--repeats", type=int, default=3)
    nps_benchmark.add_argument(
        "--target-nodes-per-root",
        type=int,
        default=100_000_000,
        help="estimate wall time for this many nodes on every parallel root",
    )
    _add_search_memory_arguments(nps_benchmark)

    initialize = subparsers.add_parser("init", help="create a random initial checkpoint")
    initialize.add_argument("output", type=Path)
    initialize.add_argument("--profile", type=_profile, default="smoke")

    selfplay = subparsers.add_parser("selfplay", help="generate parallel self-play games")
    selfplay.add_argument("checkpoint", type=Path)
    selfplay.add_argument("output", type=Path)
    selfplay.add_argument("--games", type=int, default=2)
    selfplay.add_argument("--workers", type=int, default=2)
    selfplay.add_argument("--mode", choices=("batched", "process"), default="batched")
    selfplay.add_argument("--simulations", type=int, default=800)
    selfplay.add_argument(
        "--max-plies",
        type=int,
        default=0,
        help="debug safety cutoff; zero means play to a rules result",
    )
    selfplay.add_argument("--temperature-moves", type=int, default=24)
    selfplay.add_argument("--resign-threshold", type=float, default=-0.98)
    selfplay.add_argument("--seed", type=int, default=0)
    _add_search_memory_arguments(selfplay)
    _add_training_interlock_arguments(selfplay)

    training = subparsers.add_parser("train", help="update a checkpoint from replay data")
    training.add_argument("checkpoint", type=Path)
    training.add_argument("replay", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--telemetry-interval",
        type=int,
        default=1,
        help="persist one optimizer trace point every N steps (clip aggregates cover every step)",
    )
    training.add_argument("--anchor-replay", type=Path, action="append", default=[])
    training.add_argument("--allow-actor-targets", action="store_true")
    training.add_argument("--canonical-teacher-only", action="store_true")
    training.add_argument("--canonical-target-sidecar", type=Path)
    training.add_argument("--value-loss-weight", type=float, default=1.0)
    training.add_argument("--best-union-top5-weight", type=float, default=0.0)
    training.add_argument(
        "--reset-optimizer-state",
        action="store_true",
        help=(
            "explicitly warm-start model weights while resetting AdamW/RNG state; required "
            "when moving an exact-resume v1 checkpoint to a changed v2 loss contract"
        ),
    )
    _add_distillation_safety_arguments(training)
    _add_training_interlock_arguments(training)

    train_psv = subparsers.add_parser(
        "train-psv", help="stream-train a checkpoint from a YaneuraOu PSV file"
    )
    train_psv.add_argument("checkpoint", type=Path)
    train_psv.add_argument("psv", type=Path)
    train_psv.add_argument("output", type=Path)
    train_psv.add_argument("--source-name", required=True)
    train_psv.add_argument("--steps", type=int, default=1000)
    train_psv.add_argument("--batch-size", type=int, default=64)
    train_psv.add_argument("--learning-rate", type=float, default=1e-4)
    train_psv.add_argument("--seed", type=int, default=0)
    train_psv.add_argument(
        "--telemetry-interval",
        type=int,
        default=1,
        help="persist one optimizer trace point every N steps (clip aggregates cover every step)",
    )
    train_psv.add_argument("--offset", type=int, default=0)
    train_psv.add_argument("--stride", type=int, default=1)
    train_psv.add_argument("--limit", type=int)
    _add_training_interlock_arguments(train_psv)
    psv_value_scale = train_psv.add_mutually_exclusive_group()
    psv_value_scale.add_argument(
        "--score-ponanza-coefficient",
        type=float,
        help=(
            "Ponanza probability coefficient C in sigmoid(cp/C); default C=600, "
            "which uses tanh denominator 2C=1200"
        ),
    )
    psv_value_scale.add_argument(
        "--score-scale",
        type=float,
        help=(
            "legacy compatibility: raw denominator D in tanh(cp/D); an explicit "
            "--score-scale keeps its old meaning and cannot be combined with "
            "--score-ponanza-coefficient"
        ),
    )
    train_psv.add_argument("--evaluation-weight", type=float, default=0.5)
    _add_distillation_safety_arguments(train_psv)

    teacher_dedup = subparsers.add_parser(
        "deduplicate-teacher-replay",
        help="merge history-specific targets for one teacher by observable position",
    )
    teacher_dedup.add_argument("input", type=Path)
    teacher_dedup.add_argument("output", type=Path)
    teacher_dedup.add_argument("--split", required=True)
    teacher_dedup.add_argument("--forbid-overlap-with", type=Path, action="append", default=[])
    teacher_dedup.add_argument("--manifest", type=Path)

    floodgate_corpus = subparsers.add_parser(
        "prepare-floodgate-corpus",
        help="build a local-only, history-reanalysis-required corpus from official CSA games",
    )
    floodgate_corpus.add_argument("archive", type=Path)
    floodgate_corpus.add_argument("rating_html", type=Path)
    floodgate_corpus.add_argument("output", type=Path)
    floodgate_corpus.add_argument("--rating-url", required=True)
    floodgate_corpus.add_argument("--rating-scope", default="14-day")
    floodgate_corpus.add_argument("--min-rating", type=int, default=3_900)
    floodgate_corpus.add_argument("--min-games", type=int, default=20)
    floodgate_corpus.add_argument("--exclude-zero-loss", action="store_true")
    floodgate_corpus.add_argument("--split-seed", default="meteo-floodgate-corpus-v1")
    floodgate_corpus.add_argument("--seven-zip", default="7z")
    floodgate_corpus.add_argument("--maximum-csa-bytes", type=int, default=2_000_000)
    floodgate_corpus.add_argument(
        "--extraction-batch-bytes",
        type=int,
        default=768 * 1024 * 1024,
        help="upper bound for uncompressed CSA bytes in one temporary extraction batch",
    )
    floodgate_corpus.add_argument(
        "--extraction-batch-members",
        type=int,
        default=50_000,
        help="upper bound for CSA members in one temporary extraction batch",
    )
    floodgate_corpus.add_argument(
        "--extraction-timeout-seconds",
        type=float,
        default=600,
        help="timeout for each bounded 7-Zip batch extraction",
    )
    floodgate_corpus.add_argument(
        "--limit-members",
        type=int,
        help="deterministic prefix for smoke runs only; omit for a full corpus",
    )

    ybb_plan = subparsers.add_parser(
        "prepare-ybb-reanalysis-plan",
        help=(
            "audit every YANE-BINBOOK-V1 position and create an exactly-covering "
            "local no-book MultiPV plan"
        ),
    )
    ybb_plan.add_argument("book", type=Path)
    ybb_plan.add_argument("output", type=Path)
    ybb_plan.add_argument("--source-archive", type=Path, required=True)
    ybb_plan.add_argument("--source-url", required=True)
    ybb_plan.add_argument("--teacher-id", action="append", required=True)
    ybb_plan.add_argument("--shard-size", type=int, default=100_000)
    ybb_plan.add_argument("--screen-nodes", type=int, default=20_000)
    ybb_plan.add_argument("--deep-nodes", type=int, default=200_000)
    ybb_plan.add_argument("--priority-nodes", type=int, default=2_000_000)
    ybb_plan.add_argument("--multipv", type=int, default=8)
    ybb_plan.add_argument(
        "--local-only-user-authorized",
        action="store_true",
        help=(
            "required acknowledgement: the 13.5M archive has no reviewed publication grant; "
            "the plan and derived corpus remain local"
        ),
    )

    reanalysis = subparsers.add_parser(
        "reanalyse", help="attach deep same-position teacher targets"
    )
    reanalysis.add_argument("checkpoint", type=Path)
    reanalysis.add_argument("replay", type=Path)
    reanalysis.add_argument("output", type=Path)
    reanalysis.add_argument("--actor-simulations", type=int, default=800)
    reanalysis.add_argument("--teacher-simulations", type=int, default=6400)
    reanalysis.add_argument("--fraction", type=float, default=0.5)
    reanalysis.add_argument("--seed", type=int, default=0)
    _add_search_memory_arguments(reanalysis)
    _add_training_interlock_arguments(reanalysis)

    external_reanalysis = subparsers.add_parser(
        "reanalyse-usi",
        help="attach MultiPV targets from a reviewed output-distillable USI engine",
    )
    external_reanalysis.add_argument("replay", type=Path)
    external_reanalysis.add_argument("output", type=Path)
    external_reanalysis.add_argument("--engine", type=Path, required=True)
    external_reanalysis.add_argument("--engine-cwd", type=Path)
    external_reanalysis.add_argument("--engine-arg", action="append", default=[])
    external_reanalysis.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="separately licensed eval/model/data file to hash in provenance; repeatable",
    )
    external_reanalysis.add_argument(
        "--rights-profile",
        required=True,
        choices=locally_distillable_rights_ids(),
    )
    external_reanalysis.add_argument(
        "--local-only-user-authorized",
        action="store_true",
        help=(
            "explicitly enable a LIMITED_LOCAL teacher that the user lawfully acquired; "
            "raw labels and derived checkpoints remain unpublished"
        ),
    )
    external_reanalysis.add_argument(
        "--local-only-root",
        type=Path,
        help="existing private root that must contain every LIMITED_LOCAL output artifact",
    )
    external_reanalysis.add_argument(
        "--verify-yaneuraou-options",
        action="store_true",
        help="query YaneuraOu getoption after isready and reject any option value mismatch",
    )
    external_reanalysis.add_argument("--nodes", type=int, default=1_000_000)
    external_reanalysis.add_argument("--multipv", type=int, default=32)
    external_reanalysis.add_argument(
        "--teacher-policy-temperature",
        type=float,
        default=200.0,
        help="centipawn temperature for the full MultiPV policy target",
    )
    external_value_scale = external_reanalysis.add_mutually_exclusive_group()
    external_value_scale.add_argument(
        "--teacher-ponanza-coefficient",
        type=float,
        help=(
            "Ponanza probability coefficient C in sigmoid(cp/C); default C=600, "
            "which uses signed-value tanh denominator 2C=1200"
        ),
    )
    external_value_scale.add_argument(
        "--teacher-value-scale",
        type=float,
        help=(
            "legacy compatibility: raw denominator D in tanh(cp/D); an explicit "
            "--teacher-value-scale keeps its old meaning and cannot be combined with "
            "--teacher-ponanza-coefficient"
        ),
    )
    external_reanalysis.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    external_reanalysis.add_argument("--fraction", type=float, default=0.5)
    external_reanalysis.add_argument("--uncertainty-threshold", type=float, default=0.15)
    external_reanalysis.add_argument(
        "--selection",
        choices=("uncertainty", "tactical", "all"),
        default="uncertainty",
    )
    external_reanalysis.add_argument(
        "--teacher-tag",
        help="provenance tag such as tactics, yagura, or deep-endgame",
    )
    external_reanalysis.add_argument("--timeout-seconds", type=float, default=300.0)

    teacher_ensemble = subparsers.add_parser(
        "ensemble-teachers",
        help="merge same-position MultiPV policies and values from independent teacher replays",
    )
    teacher_ensemble.add_argument("output", type=Path)
    teacher_ensemble.add_argument(
        "--teacher",
        action="append",
        required=True,
        metavar="LABEL=REPLAY",
        help="teacher label and reanalysed replay; repeat for every independent teacher",
    )
    teacher_ensemble.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT",
        help="positive mixture weight; omitted teachers default to 1.0",
    )
    teacher_ensemble.add_argument("--split", required=True)
    teacher_ensemble.add_argument("--minimum-teachers", type=int, default=2)
    teacher_ensemble.add_argument("--top-k", type=int, default=3)
    teacher_ensemble.add_argument("--value-neutral-threshold", type=float, default=0.05)
    teacher_ensemble.add_argument(
        "--forbid-overlap-with",
        type=Path,
        action="append",
        default=[],
        metavar="OTHER_SPLIT_REPLAY",
        help="fail if any normalized SFEN is also present in another split; repeatable",
    )
    teacher_ensemble.add_argument("--manifest", type=Path)

    disagreement_selection = subparsers.add_parser(
        "select-deep-disagreements",
        help="create a family-aware, hashed replay of positions needing deeper arbitration",
    )
    disagreement_selection.add_argument("ensemble_replay", type=Path)
    disagreement_selection.add_argument("ensemble_manifest", type=Path)
    disagreement_selection.add_argument("output", type=Path)
    disagreement_selection.add_argument(
        "--teacher", action="append", required=True, metavar="LABEL=REPLAY"
    )
    disagreement_selection.add_argument(
        "--family",
        action="append",
        required=True,
        metavar="LABEL=CORRELATION_FAMILY",
        help="one family per teacher; correlated teachers share the same family value",
    )
    disagreement_selection.add_argument("--high-js-threshold", type=float, default=0.25)
    disagreement_selection.add_argument(
        "--value-neutral-threshold", type=float, default=0.05
    )
    disagreement_selection.add_argument(
        "--maximum",
        type=int,
        default=0,
        help="maximum selected positions; zero preserves every match",
    )
    disagreement_selection.add_argument("--opening-priority-max-ply", type=int, default=24)
    disagreement_selection.add_argument("--opening-priority-nodes", type=int, default=2_000_000)
    disagreement_selection.add_argument(
        "--centipawn-value-scale",
        type=float,
        default=1_200.0,
        help=(
            "raw denominator D in tanh(cp/D); D=2C, so default D=1200 is "
            "Ponanza probability coefficient C=600"
        ),
    )
    disagreement_selection.add_argument(
        "--family-value-scale", action="append", default=[], metavar="FAMILY=D"
    )
    disagreement_selection.add_argument("--report", type=Path)

    depth_arbitration = subparsers.add_parser(
        "arbitrate-depth-passes",
        help="merge deep reanalysis passes without inventing one answer at disagreement",
    )
    depth_arbitration.add_argument("base_replay", type=Path)
    depth_arbitration.add_argument("output", type=Path)
    depth_arbitration.add_argument(
        "--pass", dest="depth_pass", action="append", required=True, metavar="LABEL=REPLAY"
    )
    depth_arbitration.add_argument(
        "--pass-family", action="append", required=True, metavar="LABEL=CORRELATION_FAMILY"
    )
    depth_arbitration.add_argument(
        "--pass-depth", action="append", required=True, metavar="LABEL=DECLARED_DEPTH"
    )
    depth_arbitration.add_argument(
        "--pass-scope",
        action="append",
        default=[],
        metavar="LABEL=VALIDATION_SCOPE",
        help="for example in_domain, independent, or unknown_split",
    )
    depth_arbitration.add_argument("--proof-plies", type=int, choices=(1, 3, 5, 7), default=3)
    depth_arbitration.add_argument("--proof-node-limit", type=int, default=1_000_000)
    depth_arbitration.add_argument("--deep-cp-drop-threshold", type=float, default=300.0)
    depth_arbitration.add_argument("--deep-value-drop-threshold", type=float, default=0.25)
    depth_arbitration.add_argument("--allow-incomplete-passes", action="store_true")
    depth_arbitration.add_argument(
        "--family-weight", action="append", default=[], metavar="FAMILY=WEIGHT"
    )
    depth_arbitration.add_argument(
        "--phase-family-weight",
        action="append",
        default=[],
        metavar="PHASE:FAMILY=WEIGHT",
        help=(
            "strictly positive training-time multiplier on a family prior for one phase; "
            "all families remain present and the result is one distilled target"
        ),
    )
    depth_arbitration.add_argument(
        "--centipawn-value-scale",
        type=float,
        default=1_200.0,
        help=(
            "raw denominator D in tanh(cp/D); D=2C, so default D=1200 is "
            "Ponanza probability coefficient C=600"
        ),
    )
    depth_arbitration.add_argument(
        "--family-value-scale", action="append", default=[], metavar="FAMILY=D"
    )
    depth_arbitration.add_argument(
        "--opponent-evidence",
        type=Path,
        help="optional meteo-opponent-exploit-evidence-v1 JSON; never changes the core target",
    )
    depth_arbitration.add_argument("--report", type=Path)

    value_scale_fit = subparsers.add_parser(
        "fit-teacher-value-scale",
        help="fit per-teacher cp-to-win-probability scales with game-cluster uncertainty",
    )
    value_scale_fit.add_argument("replay", type=Path)
    value_scale_fit.add_argument("--output", type=Path, required=True)
    fit_baseline_scale = value_scale_fit.add_mutually_exclusive_group()
    fit_baseline_scale.add_argument(
        "--baseline-ponanza-coefficient",
        type=float,
        help=(
            "baseline Ponanza coefficient C in sigmoid(cp/C); default C=600, "
            "equivalent to internal tanh denominator 2C=1200"
        ),
    )
    fit_baseline_scale.add_argument(
        "--baseline-scale",
        type=float,
        help=(
            "legacy compatibility: raw signed-value denominator D in tanh(cp/D); "
            "cannot be combined with --baseline-ponanza-coefficient"
        ),
    )
    value_scale_fit.add_argument(
        "--minimum-scale",
        type=float,
        default=1.0,
        help="minimum fitted raw tanh denominator D=2C",
    )
    value_scale_fit.add_argument(
        "--maximum-scale",
        type=float,
        default=1_000_000.0,
        help="maximum fitted raw tanh denominator D=2C",
    )
    value_scale_fit.add_argument("--bootstrap-resamples", type=int, default=500)
    value_scale_fit.add_argument("--seed", type=int, default=0)
    value_scale_fit.add_argument("--maximum-cross-validation-folds", type=int, default=10)
    value_scale_fit.add_argument("--minimum-reliable-games", type=int, default=30)
    value_scale_fit.add_argument("--minimum-reliable-samples", type=int, default=200)

    value_scale_ablation = subparsers.add_parser(
        "prepare-value-scale-ablation",
        help=(
            "create equal-data Ponanza-600, dlshogi-756, and heldout-gated "
            "teacher-fit value arms"
        ),
    )
    value_scale_ablation.add_argument("train_replay", type=Path)
    value_scale_ablation.add_argument("validation_replay", type=Path)
    value_scale_ablation.add_argument("output_directory", type=Path)
    value_scale_ablation.add_argument("--parent-checkpoint", type=Path, required=True)
    value_scale_ablation.add_argument("--train-split", default="train")
    value_scale_ablation.add_argument("--validation-split", default="validation")
    value_scale_ablation.add_argument("--training-seed", type=int, default=0)
    value_scale_ablation.add_argument("--steps", type=int, default=100)
    value_scale_ablation.add_argument("--batch-size", type=int, default=64)
    value_scale_ablation.add_argument("--learning-rate", type=float, default=1e-3)
    value_scale_ablation.add_argument("--bootstrap-resamples", type=int, default=500)
    value_scale_ablation.add_argument("--maximum-cross-validation-folds", type=int, default=10)
    value_scale_ablation.add_argument("--minimum-fit-games", type=int, default=30)
    value_scale_ablation.add_argument("--minimum-fit-samples", type=int, default=200)
    value_scale_ablation.add_argument("--minimum-validation-games", type=int, default=10)
    value_scale_ablation.add_argument("--minimum-validation-samples", type=int, default=100)
    value_scale_ablation.add_argument(
        "--minimum-heldout-bce-improvement", type=float, default=0.0
    )
    value_scale_ablation.add_argument("--minimum-coefficient", type=float, default=1.0)
    value_scale_ablation.add_argument("--maximum-coefficient", type=float, default=500_000.0)
    _add_distillation_safety_arguments(value_scale_ablation)

    policy_calibration = subparsers.add_parser(
        "calibrate-teacher-policy",
        help="compare or apply experimental variance-normalized MultiPV teacher policies",
    )
    policy_calibration.add_argument("input", type=Path)
    policy_calibration.add_argument("output", type=Path)
    policy_calibration.add_argument(
        "--mode",
        choices=("off", "legacy", "variance-normalized"),
        default="variance-normalized",
    )
    policy_calibration.add_argument("--prior-strength", type=float, default=5.0)
    policy_calibration.add_argument("--std-floor", type=float, default=25.0)
    policy_calibration.add_argument("--std-cap", type=float, default=2_000.0)
    policy_calibration.add_argument("--normalized-temperature", type=float, default=1.0)
    policy_calibration.add_argument("--default-prior-std", type=float, default=200.0)
    policy_calibration.add_argument("--variance-epsilon", type=float, default=1e-9)

    distillation_evaluation = subparsers.add_parser(
        "evaluate-distillation",
        help="score a checkpoint against held-out full-policy teacher targets",
    )
    distillation_evaluation.add_argument("checkpoint", type=Path)
    distillation_evaluation.add_argument("replay", type=Path)
    distillation_evaluation.add_argument("--batch-size", type=int, default=32)
    distillation_evaluation.add_argument("--output", type=Path)

    rights = subparsers.add_parser(
        "model-rights", help="print the reviewed external model and distillation-rights registry"
    )
    rights.add_argument("--rights-id", choices=tuple(item.rights_id for item in MODEL_RIGHTS))
    rights.add_argument("--distillable-only", action="store_true")

    tsume = subparsers.add_parser(
        "tsume-mine", help="mine unique forced mates from replay positions"
    )
    tsume.add_argument("replay", type=Path)
    tsume.add_argument("output", type=Path)
    tsume.add_argument("--plies", type=int, choices=(1, 3, 5, 7), default=3)
    tsume.add_argument("--node-limit", type=int, default=1_000_000)
    tsume.add_argument("--maximum", type=int, default=100)

    blunders = subparsers.add_parser(
        "blunders", help="report moves with the largest deep-teacher regret"
    )
    blunders.add_argument("replay", type=Path)
    blunders.add_argument("--limit", type=int, default=20)

    opponent = subparsers.add_parser(
        "opponent-learn", help="learn an opponent's tendencies and weaknesses"
    )
    opponent.add_argument("replay", type=Path)
    opponent.add_argument("output", type=Path)
    opponent.add_argument("--name", required=True)
    opponent.add_argument("--color", choices=("black", "white"), required=True)

    play = subparsers.add_parser("play", help="play a complete terminal game")
    play.add_argument("checkpoint", type=Path)
    play.add_argument("--human", choices=("black", "white"), default="black")
    play.add_argument("--simulations", type=int, default=800)
    play.add_argument(
        "--max-plies",
        type=int,
        default=0,
        help="debug safety cutoff; zero means play to a rules result",
    )
    _add_search_memory_arguments(play)

    usi = subparsers.add_parser("usi", help="run as a USI engine")
    usi.add_argument("checkpoint", type=Path)
    usi.add_argument("--simulations", type=int, default=800)
    _add_search_memory_arguments(usi)

    improve = subparsers.add_parser(
        "improve", help="run resumable self-play, deep teaching, training, and arena promotion"
    )
    improve.add_argument("champion", type=Path)
    improve.add_argument("workdir", type=Path)
    improve.add_argument("--generations", type=int, default=1)
    improve.add_argument("--games", type=int, default=32)
    improve.add_argument("--actor-simulations", type=int, default=800)
    improve.add_argument("--actor-temperature-moves", type=int, default=24)
    improve.add_argument("--teacher-simulations", type=int, default=6400)
    improve.add_argument("--reanalyse-fraction", type=float, default=0.5)
    improve.add_argument("--steps", type=int, default=1000)
    improve.add_argument("--batch-size", type=int, default=64)
    improve.add_argument("--learning-rate", type=float, default=1e-4)
    improve.add_argument(
        "--opening-suite",
        type=Path,
        help=(
            "read-only meteo-opening-suite-v1 JSON with distinct actor/arena splits; "
            "required for promotion"
        ),
    )
    improve.add_argument("--actor-opening-split", default="actor")
    improve.add_argument("--arena-opening-split", default="arena")
    improve.add_argument(
        "--arena-games",
        type=int,
        default=None,
        help="deprecated legacy single-opening debug game count; incompatible with --opening-suite",
    )
    improve.add_argument("--arena-simulations", type=int, default=1600)
    improve.add_argument(
        "--promotion-min-games",
        type=int,
        default=None,
        help="deprecated legacy debug threshold; cannot enable promotion",
    )
    improve.add_argument("--promotion-min-pairs", type=int, default=32)
    improve.add_argument("--promotion-lower-bound", type=float, default=0.5)
    improve.add_argument("--arena-bootstrap-iterations", type=int, default=20_000)
    improve.add_argument("--max-plies", type=int, default=0)
    improve.add_argument(
        "--initial-sfen",
        help="legacy single-opening smoke/debug mode only; candidate promotion is disabled",
    )
    improve.add_argument("--anchor-replay", type=Path, action="append", default=[])
    improve.add_argument("--seed", type=int, default=0)
    _add_distillation_safety_arguments(improve)
    _add_search_memory_arguments(improve)
    _add_training_interlock_arguments(improve)

    benchmark = subparsers.add_parser(
        "benchmark-usi", help="play paired-color games against a reviewed free USI engine"
    )
    benchmark.add_argument("checkpoint", type=Path)
    benchmark.add_argument("output", type=Path)
    benchmark.add_argument("--engine", type=Path, required=True)
    benchmark.add_argument("--engine-cwd", type=Path)
    benchmark.add_argument("--engine-arg", action="append", default=[])
    benchmark.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="separately licensed eval/model/data file to hash in the report; repeatable",
    )
    benchmark.add_argument(
        "--rights-profile",
        required=True,
        choices=analysable_rights_ids(),
    )
    benchmark.add_argument("--nodes", type=int, default=100_000)
    benchmark.add_argument("--multipv", type=int, default=1)
    benchmark.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    opening_mode = benchmark.add_mutually_exclusive_group(required=True)
    opening_mode.add_argument(
        "--opening-suite",
        type=Path,
        help=(
            "read-only meteo-opening-suite-v1 JSON; required for a production/Elo benchmark"
        ),
    )
    opening_mode.add_argument(
        "--legacy-single-opening-debug",
        action="store_true",
        help="run one descriptive color-swapped pair with promotion and Elo disabled",
    )
    benchmark.add_argument("--opening-split", default="arena")
    benchmark.add_argument(
        "--games",
        type=int,
        default=None,
        help="optional assertion; must equal twice the selected opening count",
    )
    benchmark.add_argument("--simulations", type=int, default=800)
    benchmark.add_argument("--max-plies", type=int, default=0)
    benchmark.add_argument(
        "--initial-sfen",
        action="append",
        default=[],
        help="at most one SFEN, and only with --legacy-single-opening-debug",
    )
    benchmark.add_argument("--promotion-min-pairs", type=int, default=32)
    benchmark.add_argument("--promotion-lower-bound", type=float, default=0.5)
    benchmark.add_argument("--bootstrap-iterations", type=int, default=20_000)
    benchmark.add_argument("--seed", type=int, default=0)
    _add_search_memory_arguments(benchmark)
    _add_training_interlock_arguments(benchmark)

    verify = subparsers.add_parser("verify", help="run the local end-to-end acceptance path")
    verify.add_argument("output", type=Path)
    verify.add_argument("--workers", type=int, default=2)
    verify.add_argument("--profile", type=_profile, default="smoke")
    return parser


def _doctor(profile: ModelProfile, memory_budget: MemoryBudget) -> dict[str, object]:
    board = Board()
    if len(board.legal_moves()) != 30 or not board.is_valid():
        raise AssertionError("rsshogi initial position validation failed")
    model = PolicyValueResNet(model_profile(profile))
    assert_model_shapes(model)
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "mlx": getattr(mx, "__version__", "unknown"),
        "rsshogi": getattr(rsshogi, "__version__", "unknown"),
        "device": str(mx.default_device()),
        "profile": model.config.name,
        "parameters": parameter_count(model),
        "initial_legal_moves": len(board.legal_moves()),
        "memory": memory_budget.to_dict(),
        "status": "ok",
    }


def _distillation_evaluation_report(
    checkpoint: Path,
    replay: Path,
    *,
    batch_size: int,
    memory_budget: MemoryBudget,
) -> dict[str, object]:
    checkpoint_path = checkpoint.expanduser().resolve()
    replay_path = replay.expanduser().resolve()
    model, step = load_checkpoint(checkpoint_path)
    games = load_games(replay_path)
    samples = position_samples(games)
    teacher_samples = [sample for sample in samples if sample.teacher_policy is not None]
    alignment = evaluate_teacher_alignment(
        MLXEvaluator(model),
        samples,
        batch_size=batch_size,
    )
    metadata_path = checkpoint_path / "metadata.json"
    weights_path = checkpoint_path / "weights.safetensors"
    teacher_unique_positions = {
        normalized_position_key(sample.sfen) for sample in teacher_samples
    }
    teacher_source_positions = {
        (sample.teacher_source or "unknown", normalized_position_key(sample.sfen))
        for sample in teacher_samples
    }
    return {
        "schema": "meteo-distillation-evaluation-v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _checkpoint_sha256(checkpoint_path),
            "step": step,
            "model_profile": model.config.name,
            "parameters": parameter_count(model),
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256_file(metadata_path),
                "bytes": metadata_path.stat().st_size,
            },
            "weights": {
                "path": str(weights_path),
                "sha256": _sha256_file(weights_path),
                "bytes": weights_path.stat().st_size,
            },
        },
        "replay": {
            "path": str(replay_path),
            "sha256": _sha256_file(replay_path),
            "bytes": replay_path.stat().st_size,
            "games": len(games),
            "raw_samples": sum(len(game.samples) for game in games),
            "eligible_samples": len(samples),
            "teacher_labelled_samples": len(teacher_samples),
            "teacher_source_position_pairs": len(teacher_source_positions),
            "teacher_unique_positions": len(teacher_unique_positions),
            "evaluated_samples": alignment.overall.samples,
            "evaluated_unique_positions": alignment.overall.unique_positions,
        },
        "batch_size": batch_size,
        "metrics": alignment.to_dict(),
        "memory": memory_budget.to_dict(),
    }


def _benchmark_nps(
    profile: ModelProfile,
    *,
    parallel_roots: int,
    simulations: int,
    repeats: int,
    target_nodes_per_root: int,
    max_tree_nodes: int | None,
    memory_check_interval: int,
    intra_root_batch_size: int,
    intra_root_virtual_loss: float,
    max_evaluation_batch_size: int,
    memory_budget: MemoryBudget,
) -> dict[str, object]:
    if parallel_roots < 1 or simulations < 1 or repeats < 1 or target_nodes_per_root < 1:
        raise ValueError("parallel roots, simulations, repeats, and target nodes must be positive")
    model = PolicyValueResNet(model_profile(profile))
    evaluator = MLXEvaluator(model)
    search_config = SearchConfig(
        simulations=simulations,
        root_min_visits=1,
        dirichlet_fraction=0,
        max_tree_nodes=max_tree_nodes,
        memory_check_interval=memory_check_interval,
        intra_root_batch_size=intra_root_batch_size,
        intra_root_virtual_loss=intra_root_virtual_loss,
        max_evaluation_batch_size=max_evaluation_batch_size,
    )
    measurements: list[dict[str, int | float]] = []
    mx.reset_peak_memory()
    for _repeat in range(repeats):
        results = MCTS(evaluator, search_config).search_many(
            [Board() for _ in range(parallel_roots)]
        )
        nodes = sum(result.simulations for result in results)
        seconds = sum(result.elapsed_seconds for result in results)
        measurements.append(
            {
                "nodes": nodes,
                "seconds": seconds,
                "nps": nodes / seconds,
                "peak_tree_nodes": results[0].peak_tree_nodes,
                "tree_recycles": results[0].tree_recycles,
            }
        )
    total_nodes = sum(int(item["nodes"]) for item in measurements)
    total_seconds = sum(float(item["seconds"]) for item in measurements)
    aggregate_nps = total_nodes / total_seconds
    retained_tree_node_limit = max_tree_nodes or memory_budget.tree_node_limit
    target_wall_seconds = target_nodes_per_root * parallel_roots / aggregate_nps
    return {
        "profile": model.config.name,
        "parameters": parameter_count(model),
        "parallel_roots": parallel_roots,
        "simulations_per_root": simulations,
        "repeats": repeats,
        "measurements": measurements,
        "total_nodes": total_nodes,
        "total_seconds": total_seconds,
        "aggregate_nps": aggregate_nps,
        "target_nodes_per_root": target_nodes_per_root,
        "estimated_wall_seconds_for_target_per_root": target_wall_seconds,
        "estimated_wall_hours_for_target_per_root": target_wall_seconds / 3600.0,
        "retained_tree_node_limit": retained_tree_node_limit,
        "intra_root_batch_size": intra_root_batch_size,
        "intra_root_virtual_loss": intra_root_virtual_loss,
        "max_evaluation_batch_size": max_evaluation_batch_size,
        "target_requires_tree_recycling": target_nodes_per_root > retained_tree_node_limit,
        "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
        "memory": memory_budget.to_dict(),
        "note": "throughput benchmark uses random weights; it measures speed, not playing strength",
    }


def _interactive_play(
    checkpoint: Path,
    human: str,
    simulations: int,
    max_plies: int | None,
    max_tree_nodes: int | None,
    memory_check_interval: int,
    intra_root_batch_size: int,
    intra_root_virtual_loss: float,
    max_evaluation_batch_size: int,
) -> int:
    model, _ = load_checkpoint(checkpoint)
    evaluator = MLXEvaluator(model)
    search = SearchConfig(
        simulations=simulations,
        max_plies=max_plies,
        dirichlet_fraction=0,
        max_tree_nodes=max_tree_nodes,
        memory_check_interval=memory_check_interval,
        intra_root_batch_size=intra_root_batch_size,
        intra_root_virtual_loss=intra_root_virtual_loss,
        max_evaluation_batch_size=max_evaluation_batch_size,
    )
    engine = MCTS(evaluator, search)
    board = Board()
    human_color = Color.BLACK if human == "black" else Color.WHITE
    plies = 0
    while max_plies is None or plies < max_plies:
        print(board.to_bod())
        if board.can_declare_win():
            print(f"{board.turn} wins by declaration")
            return 0
        if board.is_mated() or not board.legal_moves():
            print(f"{board.turn.opponent()} wins by checkmate")
            return 0
        if board.turn == human_color:
            raw = input("your move (USI, or resign): ").strip()
            if raw == "resign":
                print("you resigned")
                return 0
            move = Move.from_usi(raw)
            if not board.is_legal_move(move):
                print("illegal move")
                continue
            board.apply_move(move)
            plies += 1
        else:
            result = engine.search(board)
            print(f"engine: {result.best_move} value={result.root_value:+.3f}")
            board.apply_usi(result.best_move)
            plies += 1
    print("draw by configured maximum plies")
    return 0


def _verify(output: Path, workers: int, profile: ModelProfile) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    model = PolicyValueResNet(model_profile(profile))
    assert_model_shapes(model, batch_size=1)
    initial = output / "initial"
    save_checkpoint(model, initial, step=0)
    search = SearchConfig(
        simulations=1,
        root_min_visits=1,
        temperature_moves=0,
        max_plies=4,
        resign_threshold=None,
        dirichlet_fraction=0,
    )
    mate_sfens = (
        "4k4/9/3B5/9/9/9/9/9/4K4 b G 1",
        "5k3/9/4B4/9/9/9/9/9/4K4 b G 1",
        "3k5/9/2B6/9/9/9/9/9/4K4 b G 1",
        "6k2/9/5B3/9/9/9/9/9/4K4 b G 1",
        "2k6/9/1B7/9/9/9/9/9/4K4 b G 1",
    )
    mate_sfen = mate_sfens[0]
    game_count = max(4, workers)
    games = batched_self_play(
        initial,
        search,
        games=game_count,
        initial_sfens=[mate_sfens[index % len(mate_sfens)] for index in range(game_count)],
    )
    for game in games:
        replay_and_validate(game)
    teacher_evaluator = MLXEvaluator(model)
    reanalysis_config = ReanalysisConfig(
        teacher_simulation_multiplier=8,
        minimum_teacher_simulations=200,
        reanalyse_fraction=1.0,
    )
    games = [
        reanalyse_game(
            game,
            teacher_evaluator,
            search,
            reanalysis_config,
            seed=index,
        )
        for index, game in enumerate(games)
    ]
    replay_path = output / "replay.jsonl"
    append_games(replay_path, games)
    samples = position_samples(load_games(replay_path))
    metrics = train(
        model,
        samples,
        steps=8,
        batch_size=min(8, len(samples)),
        learning_rate=1e-4,
    )
    if metrics.final_loss >= metrics.initial_loss:
        raise AssertionError(
            "acceptance training did not reduce loss: "
            f"{metrics.initial_loss} -> {metrics.final_loss}"
        )
    trained = output / "trained"
    save_checkpoint(model, trained, step=metrics.steps)
    reloaded, step = load_checkpoint(trained)
    assert_model_shapes(reloaded, batch_size=1)
    evaluation = MLXEvaluator(reloaded).evaluate(Board())
    if set(evaluation.policy) != {move.to_usi() for move in Board().legal_moves()}:
        raise AssertionError("reloaded model did not produce exactly the legal policy")
    mate_probe = play_game(
        MLXEvaluator(reloaded),
        SearchConfig(
            simulations=1,
            root_min_visits=1,
            max_plies=4,
            temperature_moves=0,
            resign_threshold=None,
            dirichlet_fraction=0,
        ),
        initial_sfen=mate_sfen,
        self_play_noise=False,
    )
    if mate_probe.termination.value != "checkmate":
        raise AssertionError("reloaded neural engine failed the checkmate acceptance probe")
    return {
        "status": "ok",
        "profile": profile,
        "games": len(games),
        "terminations": [game.termination.value for game in games],
        "legal_replay": True,
        "samples": len(samples),
        "deep_teacher_samples": sum(sample.teacher_policy is not None for sample in samples),
        "policy_reversals": sum(sample.policy_reversal for sample in samples),
        "training": asdict(metrics),
        "checkpoint_step": step,
        "reload_legal_moves": len(evaluation.policy),
        "neural_checkmate_plies": len(mate_probe.moves),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    memory_commands = {
        "doctor",
        "bench-nps",
        "init",
        "selfplay",
        "train",
        "train-psv",
        "reanalyse",
        "play",
        "usi",
        "improve",
        "benchmark-usi",
        "evaluate-distillation",
        "verify",
    }
    memory_budget = configure_mlx_memory() if args.command in memory_commands else None
    if args.command == "doctor":
        if memory_budget is None:
            raise AssertionError("doctor memory budget was not configured")
        print(json.dumps(_doctor(args.profile, memory_budget), indent=2))
        return 0
    if args.command == "bench-nps":
        if memory_budget is None:
            raise AssertionError("NPS benchmark memory budget was not configured")
        print(
            json.dumps(
                _benchmark_nps(
                    args.profile,
                    parallel_roots=args.parallel_roots,
                    simulations=args.simulations,
                    repeats=args.repeats,
                    target_nodes_per_root=args.target_nodes_per_root,
                    max_tree_nodes=args.max_tree_nodes or None,
                    memory_check_interval=args.memory_check_interval,
                    intra_root_batch_size=args.intra_root_batch_size,
                    intra_root_virtual_loss=args.intra_root_virtual_loss,
                    max_evaluation_batch_size=args.max_evaluation_batch_size,
                    memory_budget=memory_budget,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "init":
        model = PolicyValueResNet(model_profile(args.profile))
        assert_model_shapes(model, batch_size=1)
        print(save_checkpoint(model, args.output, step=0))
        return 0
    if args.command == "selfplay":
        search_config = _search_from_args(args)
        selfplay_interlock = _training_interlock_from_args(args)
        selfplay_games = (
            batched_self_play(
                args.checkpoint,
                search_config,
                games=args.games,
                seed=args.seed,
                compute_interlock=selfplay_interlock,
            )
            if args.mode == "batched"
            else parallel_self_play(
                args.checkpoint,
                search_config,
                games=args.games,
                workers=args.workers,
                seed=args.seed,
                compute_interlock=selfplay_interlock,
            )
        )
        append_games(args.output, selfplay_games)
        print(
            json.dumps(
                {
                    "games": len(selfplay_games),
                    "output": str(args.output),
                    "search": search_telemetry(selfplay_games),
                    "memory": memory_budget.to_dict() if memory_budget is not None else None,
                    "human_play_compute_interlock": (
                        {"enabled": False}
                        if selfplay_interlock is None
                        else selfplay_interlock.public_metadata()
                    ),
                }
            )
        )
        return 0
    if args.command == "deduplicate-teacher-replay":
        build = build_single_teacher_dedup(
            args.input,
            split=args.split,
            forbid_overlap_with=args.forbid_overlap_with,
        )
        payload = write_single_teacher_dedup(
            build,
            args.output,
            manifest=args.manifest,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "prepare-floodgate-corpus":
        if args.limit_members is not None and args.limit_members < 1:
            raise ValueError("--limit-members must be positive when configured")
        if args.maximum_csa_bytes < 1:
            raise ValueError("--maximum-csa-bytes must be positive")
        if args.extraction_batch_bytes < args.maximum_csa_bytes:
            raise ValueError(
                "--extraction-batch-bytes must be at least --maximum-csa-bytes"
            )
        if args.extraction_batch_members < 1:
            raise ValueError("--extraction-batch-members must be positive")
        if args.extraction_timeout_seconds <= 0:
            raise ValueError("--extraction-timeout-seconds must be positive")
        rating_bytes = args.rating_html.expanduser().resolve().read_bytes()
        rating_snapshot = parse_rating_snapshot(
            rating_bytes,
            source_url=args.rating_url,
            scope=args.rating_scope,
        )
        members = list_7z_csa_members(args.archive, seven_zip=args.seven_zip)
        if args.limit_members is not None:
            members = members[: args.limit_members]
        with stream_7z_csa_sources(
            args.archive,
            members,
            seven_zip=args.seven_zip,
            maximum_bytes=args.maximum_csa_bytes,
            maximum_batch_bytes=args.extraction_batch_bytes,
            maximum_batch_members=args.extraction_batch_members,
            timeout_seconds=args.extraction_timeout_seconds,
        ) as sources:
            manifest = create_floodgate_corpus(
                args.output,
                archive=args.archive,
                csa_sources=sources,
                rating_snapshot=rating_snapshot,
                config=FloodgateCorpusConfig(
                    min_rating=args.min_rating,
                    min_games=args.min_games,
                    include_zero_loss_players=not args.exclude_zero_loss,
                    split_seed=args.split_seed,
                ),
            )
        output_path = args.output.expanduser().resolve()
        inspection = manifest["inspection"]
        if not isinstance(inspection, dict):
            raise AssertionError("Floodgate corpus inspection must be an object")
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "output": str(output_path),
                    "manifest_sha256": _sha256_file(output_path / "manifest.json"),
                    "rating_snapshot_sha256": rating_snapshot.source_sha256,
                    "rating_snapshot_date": rating_snapshot.snapshot_date,
                    "archive_members": len(members),
                    "inspection": {
                        key: inspection[key]
                        for key in (
                            "sources",
                            "parsed_games",
                            "parse_failures",
                            "selected_games",
                            "excluded_games",
                        )
                    },
                    "splits": manifest["splits"],
                    "rights": manifest["rights"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "prepare-ybb-reanalysis-plan":
        archive_identity = _artifact_provenance([args.source_archive])[0]
        plan = build_ybb_reanalysis_plan(
            args.book,
            source_url=args.source_url,
            source_archive_sha256=str(archive_identity["sha256"]),
            required_teacher_ids=args.teacher_id,
            shard_size=args.shard_size,
            screen_nodes=args.screen_nodes,
            deep_nodes=args.deep_nodes,
            priority_nodes=args.priority_nodes,
            multipv=args.multipv,
            local_only_user_authorized=args.local_only_user_authorized,
        )
        output_sha256 = write_ybb_reanalysis_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "schema": plan.to_dict()["schema"],
                    "output": str(args.output.expanduser().resolve()),
                    "output_sha256": output_sha256,
                    "source": plan.source.to_dict(),
                    "source_archive": archive_identity,
                    "positions": plan.header.record_count,
                    "move_records_referenced": (
                        plan.structural_audit.move_records_referenced
                    ),
                    "teachers": list(plan.required_teacher_ids),
                    "shards": len(plan.shards),
                    "book_moves_used_as_labels": False,
                    "rights_scope": plan.rights_scope,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "train":
        _require_new_checkpoint_output(args.output, parent=args.checkpoint)
        training_interlock = _training_interlock_from_args(args)
        model, current_step, resume_state, _parent_trace = (
            load_checkpoint_with_training_state(args.checkpoint)
        )
        canonical_sidecar_path = resolve_canonical_target_sidecar(
            args.replay,
            explicit=args.canonical_target_sidecar,
        )
        if args.canonical_teacher_only:
            if args.anchor_replay:
                raise ValueError(
                    "canonical teacher-only v1 accepts exactly one replay and no anchors"
                )
            if args.allow_actor_targets:
                raise ValueError("canonical teacher-only training cannot allow actor targets")
            if canonical_sidecar_path is None:
                raise ValueError(
                    "canonical teacher-only training requires one explicit or adjacent sidecar"
                )
        elif canonical_sidecar_path is not None:
            raise ValueError(
                "canonical target sidecar is present but --canonical-teacher-only is absent"
            )
        if not args.canonical_teacher_only and args.best_union_top5_weight != 0.0:
            raise ValueError("--best-union-top5-weight requires --canonical-teacher-only")
        replay_files = [args.replay, *args.anchor_replay]
        training_input_provenance = _training_input_provenance(replay_files)
        canonical_sidecar_identity: dict[str, object] | None = None
        canonical_targets = None
        replay_games = load_games(args.replay)
        if canonical_sidecar_path is not None:
            canonical_sidecar_identity = {
                "sha256": _sha256_file(canonical_sidecar_path),
                "bytes": canonical_sidecar_path.stat().st_size,
            }
            canonical_targets = load_canonical_target_sidecar(
                args.replay,
                canonical_sidecar_path,
                games=replay_games,
            ).positions
        dataset_fingerprint = _training_dataset_fingerprint(
            replay_files,
            selection=(
                None
                if canonical_sidecar_identity is None
                else {
                    "target_mode": "canonical_dual_teacher",
                    "canonical_target_sidecar": canonical_sidecar_identity,
                }
            ),
            input_provenance=training_input_provenance,
        )
        samples = position_samples(
            game
            for replay_file in replay_files
            for game in (
                replay_games if replay_file == args.replay else load_games(replay_file)
            )
        )
        training_hyperparameters: dict[str, object] = {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "require_teacher": not args.allow_actor_targets,
            "teacher_policy_mix": args.teacher_policy_mix,
            "teacher_value_mix": args.teacher_value_mix,
            "legal_label_smoothing": args.legal_label_smoothing,
            "curriculum_depth_ratio": args.curriculum_depth_ratio,
            "minimum_teacher_policy_mix": args.minimum_teacher_policy_mix,
            "maximum_gradient_norm": args.max_gradient_norm,
            "maximum_probe_loss_ratio": args.max_probe_loss_ratio,
            "telemetry_interval": args.telemetry_interval,
            "human_play_compute_interlock": (
                {"enabled": False}
                if training_interlock is None
                else training_interlock.public_metadata()
            ),
        }
        if (
            args.canonical_teacher_only
            or args.value_loss_weight != 1.0
            or args.best_union_top5_weight != 0.0
            or args.reset_optimizer_state
        ):
            training_hyperparameters.update(
                {
                    "canonical_teacher_only": args.canonical_teacher_only,
                    "actor_policy_target_mass": (
                        0.0 if args.canonical_teacher_only else None
                    ),
                    "game_outcome_value_contribution": (
                        0.0 if args.canonical_teacher_only else None
                    ),
                    "value_loss_weight": args.value_loss_weight,
                    "best_union_top5_weight": args.best_union_top5_weight,
                    "optimizer_state_reset_requested": args.reset_optimizer_state,
                    "canonical_target_contract": (
                        {
                            "schema": CANONICAL_TARGET_SIDECAR_SCHEMA,
                            "scorer_ids": list(CANONICAL_SCORER_IDS),
                            "positions": len(canonical_targets or ()),
                            "sidecar_bytes_bound_only_in_dataset_fingerprint": True,
                            "sidecar_path_or_raw_hash_recorded": False,
                        }
                        if args.canonical_teacher_only
                        else None
                    ),
                }
            )
        effective_resume_state = None if args.reset_optimizer_state else resume_state
        training_run = train_resumable(
            model,
            samples,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset_fingerprint=dataset_fingerprint,
            seed=args.seed,
            require_teacher=not args.allow_actor_targets,
            teacher_policy_mix=args.teacher_policy_mix,
            teacher_value_mix=args.teacher_value_mix,
            legal_label_smoothing=args.legal_label_smoothing,
            curriculum_depth_ratio=args.curriculum_depth_ratio,
            minimum_teacher_policy_mix=args.minimum_teacher_policy_mix,
            maximum_gradient_norm=args.max_gradient_norm,
            maximum_probe_loss_ratio=args.max_probe_loss_ratio,
            canonical_teacher_only=args.canonical_teacher_only,
            canonical_targets=canonical_targets,
            value_loss_weight=args.value_loss_weight,
            best_union_top5_weight=args.best_union_top5_weight,
            initial_model_step=current_step,
            resume_state=effective_resume_state,
            telemetry_interval=args.telemetry_interval,
            training_interlock=training_interlock,
        )
        _verify_training_input_provenance_unchanged(
            replay_files, training_input_provenance
        )
        if canonical_sidecar_path is not None:
            assert canonical_sidecar_identity is not None
            if canonical_sidecar_identity != {
                "sha256": _sha256_file(canonical_sidecar_path),
                "bytes": canonical_sidecar_path.stat().st_size,
            }:
                raise ValueError("canonical target sidecar changed during training")
        save_checkpoint(
            model,
            args.output,
            step=training_run.state.model_step,
            lineage=_training_lineage(
                parent=args.checkpoint,
                inputs=replay_files,
                current_step=current_step,
                teacher_sources=[
                    *(
                        CANONICAL_SCORER_IDS
                        if args.canonical_teacher_only
                        else tuple(
                            sample.teacher_source
                            for sample in samples
                            if sample.teacher_source is not None
                        )
                    )
                ],
                hyperparameters={
                    **training_hyperparameters,
                    "metrics": asdict(training_run.metrics),
                },
                dataset_fingerprint=dataset_fingerprint,
                optimizer_state_restored=effective_resume_state is not None,
                input_provenance=training_input_provenance,
            ),
            training_state=training_run.state,
            training_trace=training_run.trace,
        )
        print(
            json.dumps(
                {
                    **asdict(training_run.metrics),
                    "dataset_fingerprint": dataset_fingerprint,
                    "exact_resume_from_parent": effective_resume_state is not None,
                    "output_contains_exact_resume_state": True,
                    **(
                        {
                            "canonical_teacher_only": True,
                            "actor_policy_target_mass": 0.0,
                            "game_outcome_value_contribution": 0.0,
                            "value_loss_weight": args.value_loss_weight,
                            "best_union_top5_weight": args.best_union_top5_weight,
                            "optimizer_state_reset_requested": args.reset_optimizer_state,
                        }
                        if args.canonical_teacher_only
                        else {}
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "train-psv":
        _require_new_checkpoint_output(args.output, parent=args.checkpoint)
        training_interlock = _training_interlock_from_args(args)
        (
            score_ponanza_coefficient,
            score_value_tanh_denominator,
            score_scale_convention,
        ) = _resolve_ponanza_value_scale(
            ponanza_coefficient=args.score_ponanza_coefficient,
            legacy_tanh_denominator=args.score_scale,
        )
        psv_input_provenance = _training_input_provenance([args.psv])
        dataset_fingerprint = _training_dataset_fingerprint(
            [args.psv],
            selection={
                "format": "packed-sfen-value",
                "source_name": args.source_name,
                "offset": args.offset,
                "stride": args.stride,
                "limit": args.limit,
                "score_value_tanh_denominator": score_value_tanh_denominator,
                "evaluation_weight": args.evaluation_weight,
            },
            input_provenance=psv_input_provenance,
        )
        model, current_step, resume_state, _parent_trace = (
            load_checkpoint_with_training_state(args.checkpoint)
        )
        dataset = PackedSfenValueDataset(
            args.psv,
            source_name=args.source_name,
            offset=args.offset,
            stride=args.stride,
            limit=args.limit,
            score_scale=score_value_tanh_denominator,
            evaluation_weight=args.evaluation_weight,
        )
        psv_training_hyperparameters: dict[str, object] = {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "offset": args.offset,
            "stride": args.stride,
            "limit": args.limit,
            "evaluation_weight": args.evaluation_weight,
            "score_ponanza_coefficient": score_ponanza_coefficient,
            "score_value_tanh_denominator": score_value_tanh_denominator,
            "score_scale": score_value_tanh_denominator,
            "score_scale_input_convention": score_scale_convention,
            "value_probability_formula": "sigmoid(cp / C)",
            "signed_value_formula": "2p - 1 = tanh(cp / (2C))",
            "teacher_policy_mix": args.teacher_policy_mix,
            "teacher_value_mix": args.teacher_value_mix,
            "legal_label_smoothing": args.legal_label_smoothing,
            "curriculum_depth_ratio": args.curriculum_depth_ratio,
            "minimum_teacher_policy_mix": args.minimum_teacher_policy_mix,
            "maximum_gradient_norm": args.max_gradient_norm,
            "maximum_probe_loss_ratio": args.max_probe_loss_ratio,
            "telemetry_interval": args.telemetry_interval,
            "human_play_compute_interlock": (
                {"enabled": False}
                if training_interlock is None
                else training_interlock.public_metadata()
            ),
        }
        training_run = train_resumable(
            model,
            dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset_fingerprint=dataset_fingerprint,
            seed=args.seed,
            require_teacher=True,
            prevalidated_teacher_samples=True,
            teacher_policy_mix=args.teacher_policy_mix,
            teacher_value_mix=args.teacher_value_mix,
            legal_label_smoothing=args.legal_label_smoothing,
            curriculum_depth_ratio=args.curriculum_depth_ratio,
            minimum_teacher_policy_mix=args.minimum_teacher_policy_mix,
            maximum_gradient_norm=args.max_gradient_norm,
            maximum_probe_loss_ratio=args.max_probe_loss_ratio,
            initial_model_step=current_step,
            resume_state=resume_state,
            telemetry_interval=args.telemetry_interval,
            training_interlock=training_interlock,
        )
        _verify_training_input_provenance_unchanged([args.psv], psv_input_provenance)
        save_checkpoint(
            model,
            args.output,
            step=training_run.state.model_step,
            lineage=_training_lineage(
                parent=args.checkpoint,
                inputs=[args.psv],
                current_step=current_step,
                teacher_sources=[args.source_name],
                hyperparameters={
                    **psv_training_hyperparameters,
                    "metrics": asdict(training_run.metrics),
                },
                dataset_fingerprint=dataset_fingerprint,
                optimizer_state_restored=resume_state is not None,
                input_provenance=psv_input_provenance,
            ),
            training_state=training_run.state,
            training_trace=training_run.trace,
        )
        print(
            json.dumps(
                {
                    **asdict(training_run.metrics),
                    "source": args.source_name,
                    "psv": str(args.psv),
                    "offset": args.offset,
                    "stride": args.stride,
                    "score_ponanza_coefficient": score_ponanza_coefficient,
                    "score_value_tanh_denominator": score_value_tanh_denominator,
                    "score_scale": score_value_tanh_denominator,
                    "score_scale_input_convention": score_scale_convention,
                    "dataset_fingerprint": dataset_fingerprint,
                    "exact_resume_from_parent": resume_state is not None,
                    "output_contains_exact_resume_state": True,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "reanalyse":
        model, _ = load_checkpoint(args.checkpoint)
        reanalysis_interlock = _training_interlock_from_args(args)
        base_evaluator = MLXEvaluator(model)
        evaluator = (
            base_evaluator
            if reanalysis_interlock is None
            else InterlockedEvaluator(base_evaluator, reanalysis_interlock)
        )
        actor_search = SearchConfig(
            simulations=args.actor_simulations,
            max_tree_nodes=args.max_tree_nodes or None,
            memory_check_interval=args.memory_check_interval,
            intra_root_batch_size=args.intra_root_batch_size,
            intra_root_virtual_loss=args.intra_root_virtual_loss,
            max_evaluation_batch_size=args.max_evaluation_batch_size,
        )
        config = ReanalysisConfig(
            teacher_simulation_multiplier=max(
                2, args.teacher_simulations // args.actor_simulations
            ),
            minimum_teacher_simulations=args.teacher_simulations,
            reanalyse_fraction=args.fraction,
        )
        reanalysed_games = [
            reanalyse_game(game, evaluator, actor_search, config, seed=args.seed + index)
            for index, game in enumerate(load_games(args.replay))
        ]
        append_games(args.output, reanalysed_games)
        teacher_count = sum(
            sample.teacher_policy is not None
            for game in reanalysed_games
            for sample in game.samples
        )
        print(
            json.dumps(
                {
                    "games": len(reanalysed_games),
                    "deep_teacher_samples": teacher_count,
                    "human_play_compute_interlock": (
                        {"enabled": False}
                        if reanalysis_interlock is None
                        else reanalysis_interlock.public_metadata()
                    ),
                }
            )
        )
        return 0
    if args.command == "reanalyse-usi":
        if args.output.exists():
            raise FileExistsError(f"refusing to append duplicate teacher replay: {args.output}")
        provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
        startup_provenance_path = args.output.with_suffix(
            args.output.suffix + ".startup.json"
        )
        if provenance_path.exists():
            raise FileExistsError(f"refusing to overwrite provenance: {provenance_path}")
        if startup_provenance_path.exists() or startup_provenance_path.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite startup provenance: {startup_provenance_path}"
            )
        rights = model_rights(args.rights_profile)
        limited_local = rights.output_distillation == RightsDecision.LIMITED
        local_only_root: Path | None = None
        if limited_local:
            if not args.local_only_user_authorized or args.local_only_root is None:
                raise PermissionError(
                    "LIMITED_LOCAL teacher use requires --local-only-user-authorized "
                    "and --local-only-root"
                )
            local_only_root = _require_limited_local_destinations(
                args.local_only_root,
                (args.output, provenance_path, startup_provenance_path),
            )
        elif args.local_only_user_authorized or args.local_only_root is not None:
            raise ValueError(
                "local-only acknowledgement options are only valid for a LIMITED_LOCAL profile"
            )
        options = _engine_options(args.option)
        if rights.rights_id == "suisho11plus-wcsc36-20260525-local":
            _require_suisho11plus_teacher_options(options, multipv=args.multipv)
        config = ReanalysisConfig(
            reanalyse_fraction=args.fraction,
            uncertainty_threshold=args.uncertainty_threshold,
        )
        command = [str(args.engine), *args.engine_arg]
        source_games = load_games(args.replay)
        external_games: list[GameRecord] = []
        (
            teacher_ponanza_coefficient,
            teacher_value_tanh_denominator,
            teacher_value_scale_convention,
        ) = _resolve_ponanza_value_scale(
            ponanza_coefficient=args.teacher_ponanza_coefficient,
            legacy_tanh_denominator=args.teacher_value_scale,
        )
        option_value_verification = (
            UsiOptionValueVerification.YANEURAOU_GETOPTION
            if args.verify_yaneuraou_options
            or rights.rights_id == "suisho11plus-wcsc36-20260525-local"
            else UsiOptionValueVerification.NONE
        )
        startup_provenance: dict[str, object]
        with ExternalUsiTeacher(
            command,
            rights.teacher_policy(allow_limited_local=limited_local),
            nodes=args.nodes,
            multipv=args.multipv,
            options=options,
            timeout_seconds=args.timeout_seconds,
            training_use=True,
            working_directory=args.engine_cwd,
            policy_temperature=args.teacher_policy_temperature,
            value_scale=teacher_value_tanh_denominator,
            option_value_verification=option_value_verification,
            startup_provenance_path=startup_provenance_path,
        ) as external:
            startup_provenance = external.startup_provenance.to_dict()
            for game in source_games:
                external.new_game()
                external_games.append(
                    reanalyse_game_external(
                        game,
                        external,
                        config,
                        selection=args.selection,
                        teacher_context=args.teacher_tag,
                    )
                )
        append_games(args.output, external_games)
        teacher_samples = [
            sample
            for game in external_games
            for sample in game.samples
            if sample.teacher_source == rights.rights_id
        ]
        engine_path = args.engine.expanduser().resolve()
        provenance = {
            "rights": rights.to_dict(),
            "rights_mode": "limited_local" if limited_local else "public_output_only",
            "local_only_user_authorized": args.local_only_user_authorized,
            "local_only_root": str(local_only_root) if local_only_root is not None else None,
            "publication_allowed": (
                rights.output_only_meteo_publication == RightsDecision.ALLOWED
            ),
            "public_release_gate": (
                "blocked_pending_rights_holder_permission"
                if limited_local
                else "rights_profile_allows_output_only_meteo_publication"
            ),
            "engine": str(engine_path),
            "engine_arguments": list(args.engine_arg),
            "engine_working_directory": (
                str(args.engine_cwd.expanduser().resolve()) if args.engine_cwd is not None else None
            ),
            "engine_options": options,
            "option_value_verification": option_value_verification.value,
            "startup_provenance": startup_provenance,
            "startup_provenance_file": str(startup_provenance_path.resolve()),
            "startup_provenance_sha256": _sha256_file(startup_provenance_path),
            "engine_sha256": _sha256_file(engine_path) if engine_path.is_file() else None,
            "artifacts": _artifact_provenance(args.artifact),
            "nodes_per_position": args.nodes,
            "multipv": args.multipv,
            "teacher_policy_temperature": args.teacher_policy_temperature,
            "teacher_policy_formula": "softmax(cp / teacher_policy_temperature)",
            "teacher_ponanza_coefficient": teacher_ponanza_coefficient,
            "teacher_value_tanh_denominator": teacher_value_tanh_denominator,
            "teacher_value_scale": teacher_value_tanh_denominator,
            "teacher_value_scale_input_convention": teacher_value_scale_convention,
            "teacher_value_probability_formula": "sigmoid(cp / C)",
            "teacher_signed_value_formula": "2p - 1 = tanh(cp / (2C))",
            "policy_and_value_calibration_are_independent": True,
            "history_mode": UsiHistoryMode.GAME_PREFIX.value,
            "history_mode_missing_field_means": UsiHistoryMode.BOARD_ONLY.value,
            "history_source": "GameRecord.initial_sfen + moves[:PositionSample.ply]",
            "selection": args.selection,
            "teacher_tag": args.teacher_tag or args.selection,
            "fraction": args.fraction,
            "uncertainty_threshold": args.uncertainty_threshold,
            "timeout_seconds": args.timeout_seconds,
            "source_replay": str(args.replay.resolve()),
            "output_replay": str(args.output.resolve()),
            "games": len(external_games),
            "teacher_samples": len(teacher_samples),
            "policy_reversals": sum(sample.policy_reversal for sample in teacher_samples),
            "lower_bound_regrets": sum(
                sample.teacher_regret_is_lower_bound for sample in teacher_samples
            ),
            "reported_teacher_nodes": sum(
                sample.teacher_nodes or 0 for sample in teacher_samples
            ),
            "teacher_nps": {
                "minimum": min(
                    (sample.teacher_nps for sample in teacher_samples if sample.teacher_nps),
                    default=None,
                ),
                "maximum": max(
                    (sample.teacher_nps for sample in teacher_samples if sample.teacher_nps),
                    default=None,
                ),
            },
        }
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(provenance, indent=2))
        return 0
    if args.command == "ensemble-teachers":
        teacher_inputs = _ensemble_teacher_inputs(args.teacher, args.weight)
        ensemble_build = build_teacher_ensemble(
            teacher_inputs,
            split=args.split,
            minimum_teachers=args.minimum_teachers,
            top_k=args.top_k,
            value_neutral_threshold=args.value_neutral_threshold,
            forbid_overlap_with=args.forbid_overlap_with,
        )
        ensemble_manifest = write_teacher_ensemble(
            ensemble_build,
            args.output,
            manifest=args.manifest,
        )
        print(json.dumps(ensemble_manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "select-deep-disagreements":
        selection_build = build_disagreement_replay(
            args.ensemble_replay,
            args.ensemble_manifest,
            _disagreement_teacher_inputs(args.teacher, args.family),
            config=DisagreementConfig(
                high_js_threshold=args.high_js_threshold,
                value_neutral_threshold=args.value_neutral_threshold,
                maximum_positions=args.maximum or None,
                opening_priority_max_ply=args.opening_priority_max_ply,
                opening_priority_nodes=args.opening_priority_nodes,
                centipawn_value_scale=args.centipawn_value_scale,
                family_value_scales=_positive_named_floats(
                    args.family_value_scale, option_name="family-value-scale"
                ),
            ),
        )
        selection_payload = write_disagreement_replay(
            selection_build,
            args.output,
            report=args.report,
        )
        print(
            json.dumps(
                {
                    "schema": selection_payload["schema"],
                    "selected_positions": selection_payload["selected_positions"],
                    "selected_reason_counts": selection_payload["selected_reason_counts"],
                    "output": selection_payload["output"],
                    "provenance_sha256": selection_payload["provenance_sha256"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "arbitrate-depth-passes":
        arbitration_build = build_depth_arbitration(
            args.base_replay,
            _depth_pass_inputs(
                args.depth_pass,
                args.pass_family,
                args.pass_depth,
                args.pass_scope,
            ),
            config=ArbitrationConfig(
                proof_max_plies=args.proof_plies,
                proof_node_limit=args.proof_node_limit,
                deep_cp_drop_threshold=args.deep_cp_drop_threshold,
                deep_value_drop_threshold=args.deep_value_drop_threshold,
                require_complete_passes=not args.allow_incomplete_passes,
                family_weights=_positive_named_floats(
                    args.family_weight, option_name="family-weight"
                ),
                phase_family_weights=_positive_phase_family_floats(
                    args.phase_family_weight, option_name="phase-family-weight"
                ),
                centipawn_value_scale=args.centipawn_value_scale,
                family_value_scales=_positive_named_floats(
                    args.family_value_scale, option_name="family-value-scale"
                ),
            ),
            opponent_evidence=args.opponent_evidence,
        )
        arbitration_payload = write_depth_arbitration(
            arbitration_build,
            args.output,
            report=args.report,
        )
        print(
            json.dumps(
                {
                    "schema": arbitration_payload["schema"],
                    "output_normalized_positions": arbitration_payload[
                        "output_normalized_positions"
                    ],
                    "unresolved_top1_positions": arbitration_payload[
                        "unresolved_top1_positions"
                    ],
                    "internally_proven_mate_positions": arbitration_payload[
                        "internally_proven_mate_positions"
                    ],
                    "risk_flag_counts": arbitration_payload["risk_flag_counts"],
                    "output": arbitration_payload["output"],
                    "provenance_sha256": arbitration_payload["provenance_sha256"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "fit-teacher-value-scale":
        replay_path = args.replay.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if replay_path == output_path:
            raise ValueError("value-scale report output must differ from the source replay")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite value-scale report: {output_path}")
        (
            baseline_ponanza_coefficient,
            baseline_value_tanh_denominator,
            baseline_scale_convention,
        ) = _resolve_ponanza_value_scale(
            ponanza_coefficient=args.baseline_ponanza_coefficient,
            legacy_tanh_denominator=args.baseline_scale,
        )
        scale_report = fit_teacher_value_scales(
            load_games(replay_path),
            baseline_scale=baseline_value_tanh_denominator,
            minimum_scale=args.minimum_scale,
            maximum_scale=args.maximum_scale,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
            maximum_cross_validation_folds=args.maximum_cross_validation_folds,
            minimum_reliable_games=args.minimum_reliable_games,
            minimum_reliable_samples=args.minimum_reliable_samples,
        )
        payload = {
            **scale_report.to_dict(),
            "baseline_ponanza_coefficient": baseline_ponanza_coefficient,
            "baseline_value_tanh_denominator": baseline_value_tanh_denominator,
            "baseline_scale_input_convention": baseline_scale_convention,
            "source_replay": str(replay_path),
            "source_replay_sha256": _sha256_file(replay_path),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output_path.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "prepare-value-scale-ablation":
        ablation_manifest = prepare_value_scale_ablation(
            args.train_replay,
            args.validation_replay,
            args.output_directory,
            args.parent_checkpoint,
            train_split=args.train_split,
            validation_split=args.validation_split,
            config=ValueScaleAblationConfig(
                training_seed=args.training_seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                teacher_policy_mix=args.teacher_policy_mix,
                teacher_value_mix=args.teacher_value_mix,
                legal_label_smoothing=args.legal_label_smoothing,
                curriculum_depth_ratio=args.curriculum_depth_ratio,
                minimum_teacher_policy_mix=args.minimum_teacher_policy_mix,
                maximum_gradient_norm=args.max_gradient_norm,
                maximum_probe_loss_ratio=args.max_probe_loss_ratio,
                bootstrap_resamples=args.bootstrap_resamples,
                maximum_cross_validation_folds=args.maximum_cross_validation_folds,
                minimum_fit_games=args.minimum_fit_games,
                minimum_fit_samples=args.minimum_fit_samples,
                minimum_validation_games=args.minimum_validation_games,
                minimum_validation_samples=args.minimum_validation_samples,
                minimum_heldout_bce_improvement=args.minimum_heldout_bce_improvement,
                minimum_coefficient=args.minimum_coefficient,
                maximum_coefficient=args.maximum_coefficient,
            ),
        )
        print(json.dumps(ablation_manifest, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "calibrate-teacher-policy":
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        provenance_path = output_path.with_suffix(output_path.suffix + ".provenance.json")
        if input_path in {output_path, provenance_path}:
            raise ValueError("policy calibration input, output, and provenance must differ")
        for target in (output_path, provenance_path):
            if target.exists():
                raise FileExistsError(
                    f"refusing to overwrite policy calibration artifact: {target}"
                )
        calibration = calibrate_teacher_policies(
            load_games(input_path),
            PolicyCalibrationConfig(
                mode=args.mode,
                prior_strength=args.prior_strength,
                standard_deviation_floor=args.std_floor,
                standard_deviation_cap=args.std_cap,
                normalized_temperature=args.normalized_temperature,
                default_prior_standard_deviation=args.default_prior_std,
                variance_epsilon=args.variance_epsilon,
            ),
        )
        append_games(output_path, calibration.games)
        provenance = {
            **calibration.report.to_dict(),
            "input_replay": str(input_path),
            "input_sha256": _sha256_file(input_path),
            "output_replay": str(output_path),
            "output_sha256": _sha256_file(output_path),
            "provenance": str(provenance_path),
        }
        serialized = json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n"
        provenance_path.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    if args.command == "evaluate-distillation":
        output_path = args.output.expanduser().resolve() if args.output is not None else None
        if output_path is not None and output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite distillation evaluation report: {output_path}"
            )
        if memory_budget is None:
            raise AssertionError("distillation evaluation memory budget was not configured")
        evaluation_report = _distillation_evaluation_report(
            args.checkpoint,
            args.replay,
            batch_size=args.batch_size,
            memory_budget=memory_budget,
        )
        if output_path is not None:
            evaluation_report["output"] = str(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(evaluation_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(evaluation_report, indent=2, sort_keys=True))
        return 0
    if args.command == "model-rights":
        records = list(MODEL_RIGHTS)
        if args.rights_id is not None:
            records = [model_rights(args.rights_id)]
        if args.distillable_only:
            approved = set(distillable_rights_ids())
            records = [record for record in records if record.rights_id in approved]
        print(json.dumps([record.to_dict() for record in records], indent=2))
        return 0
    if args.command == "tsume-mine":
        puzzles = mine_unique_tsume(
            load_games(args.replay),
            max_plies=args.plies,
            node_limit=args.node_limit,
            maximum_puzzles=args.maximum,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(puzzle.to_dict()) + "\n" for puzzle in puzzles),
            encoding="utf-8",
        )
        print(json.dumps({"puzzles": len(puzzles), "output": str(args.output)}))
        return 0
    if args.command == "blunders":
        samples = largest_blunders(load_games(args.replay), limit=args.limit)
        print(
            json.dumps(
                [
                    {
                        "sfen": sample.sfen,
                        "ply": sample.ply,
                        "played": sample.chosen_move,
                        "teacher_best": sample.teacher_best_move,
                        "regret": sample.teacher_regret,
                    }
                    for sample in samples
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "opponent-learn":
        profile = (
            OpponentProfile.load(args.output)
            if args.output.exists()
            else OpponentProfile(args.name)
        )
        opponent_color = 0 if args.color == "black" else 1
        for game in load_games(args.replay):
            profile.observe_game(game, opponent_color)
        profile.save(args.output)
        print(
            json.dumps(
                {
                    "name": profile.name,
                    "games": profile.games,
                    "known_positions": len(profile.position_moves),
                    "weaknesses": len(profile.weaknesses),
                    "output": str(args.output),
                }
            )
        )
        return 0
    if args.command == "play":
        return _interactive_play(
            args.checkpoint,
            args.human,
            args.simulations,
            optional_max_plies(args.max_plies),
            args.max_tree_nodes or None,
            args.memory_check_interval,
            args.intra_root_batch_size,
            args.intra_root_virtual_loss,
            args.max_evaluation_batch_size,
        )
    if args.command == "usi":
        model, _ = load_checkpoint(args.checkpoint)
        UsiEngine(
            MLXEvaluator(model),
            SearchConfig(
                simulations=args.simulations,
                max_tree_nodes=args.max_tree_nodes or None,
                memory_check_interval=args.memory_check_interval,
                intra_root_batch_size=args.intra_root_batch_size,
                intra_root_virtual_loss=args.intra_root_virtual_loss,
                max_evaluation_batch_size=args.max_evaluation_batch_size,
            ),
        ).run()
        return 0
    if args.command == "improve":
        improvement_config = SelfImprovementConfig(
            actor_games=args.games,
            actor_simulations=args.actor_simulations,
            actor_temperature_moves=args.actor_temperature_moves,
            teacher_simulations=args.teacher_simulations,
            reanalyse_fraction=args.reanalyse_fraction,
            training_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            arena_games=args.arena_games,
            arena_simulations=args.arena_simulations,
            promotion_min_games=args.promotion_min_games,
            promotion_min_pairs=args.promotion_min_pairs,
            promotion_lower_bound=args.promotion_lower_bound,
            arena_bootstrap_iterations=args.arena_bootstrap_iterations,
            max_plies=optional_max_plies(args.max_plies),
            initial_sfen=args.initial_sfen,
            opening_suite=(
                None if args.opening_suite is None else str(args.opening_suite.resolve())
            ),
            actor_opening_split=args.actor_opening_split,
            arena_opening_split=args.arena_opening_split,
            seed=args.seed,
            anchor_replays=tuple(str(path.resolve()) for path in args.anchor_replay),
            teacher_policy_mix=args.teacher_policy_mix,
            teacher_value_mix=args.teacher_value_mix,
            legal_label_smoothing=args.legal_label_smoothing,
            curriculum_depth_ratio=args.curriculum_depth_ratio,
            minimum_teacher_policy_mix=args.minimum_teacher_policy_mix,
            maximum_gradient_norm=args.max_gradient_norm,
            maximum_probe_loss_ratio=args.max_probe_loss_ratio,
            max_tree_nodes=args.max_tree_nodes or None,
            memory_check_interval=args.memory_check_interval,
            intra_root_batch_size=args.intra_root_batch_size,
            intra_root_virtual_loss=args.intra_root_virtual_loss,
            max_evaluation_batch_size=args.max_evaluation_batch_size,
        )
        improvement_interlock = _training_interlock_from_args(args)
        results = (
            run_self_improvement(
                args.champion,
                args.workdir,
                improvement_config,
                generations=args.generations,
            )
            if improvement_interlock is None
            else run_self_improvement(
                args.champion,
                args.workdir,
                improvement_config,
                generations=args.generations,
                training_interlock=improvement_interlock,
            )
        )
        print(json.dumps([asdict(result) for result in results], indent=2))
        return 0
    if args.command == "benchmark-usi":
        output_path = args.output.expanduser().resolve()
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite benchmark bundle: {output_path}")
        checkpoint_provenance = _checkpoint_provenance(args.checkpoint)
        checkpoint_path = Path(str(checkpoint_provenance["path"]))
        if output_path == checkpoint_path or output_path.is_relative_to(checkpoint_path):
            raise ValueError("benchmark output must not be the checkpoint or a checkpoint child")
        options = _engine_options(args.option)
        rights = model_rights(args.rights_profile)
        benchmark_interlock = _training_interlock_from_args(args)
        openings, opening_manifest, legacy_single_opening = (
            _benchmark_opening_configuration(args)
        )
        search = SearchConfig(
            simulations=args.simulations,
            max_plies=optional_max_plies(args.max_plies),
            temperature_moves=0,
            temperature=0.0,
            dirichlet_fraction=0.0,
            max_tree_nodes=args.max_tree_nodes or None,
            memory_check_interval=args.memory_check_interval,
        )
        if args.nodes < 1 or args.multipv < 1:
            raise ValueError("external nodes and MultiPV must be positive")
        requested_engine = args.engine.expanduser()
        if requested_engine.exists():
            engine_path = requested_engine.resolve(strict=True)
        else:
            located_engine = shutil.which(str(args.engine))
            if located_engine is None:
                raise FileNotFoundError(args.engine)
            engine_path = Path(located_engine).resolve(strict=True)
        if not engine_path.is_file():
            raise ValueError(f"USI engine is not a regular file: {engine_path}")
        engine_identity = _artifact_provenance([engine_path])[0]
        artifact_identities = _artifact_provenance(args.artifact)
        engine_working_directory = (
            args.engine_cwd.expanduser().resolve(strict=True)
            if args.engine_cwd is not None
            else engine_path.parent
        )
        if not engine_working_directory.is_dir():
            raise NotADirectoryError(engine_working_directory)
        source_tree = _source_tree_provenance()
        git_identity = _git_source_identity()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite benchmark bundle: {output_path}")
        temporary_path = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.name}.tmp-",
                dir=str(output_path.parent),
            )
        )
        report: dict[str, object]
        try:
            command = [str(engine_path), *args.engine_arg]
            with ExternalUsiTeacher(
                command,
                rights.teacher_policy(),
                nodes=args.nodes,
                multipv=args.multipv,
                options=options,
                training_use=False,
                working_directory=engine_working_directory,
            ) as external:
                summary, benchmark_games = benchmark_checkpoint_vs_external(
                    checkpoint_path,
                    external,
                    search,
                    openings=openings,
                    seed=args.seed,
                    promotion_min_pairs=args.promotion_min_pairs,
                    promotion_lower_bound=args.promotion_lower_bound,
                    bootstrap_iterations=args.bootstrap_iterations,
                    legacy_single_opening=legacy_single_opening,
                    promotion_eligible=not legacy_single_opening,
                    compute_interlock=benchmark_interlock,
                )

            if len(benchmark_games) != 2 * len(openings) or summary.games != len(
                benchmark_games
            ):
                raise RuntimeError("USI benchmark returned a mismatched game count")
            reported_keys = tuple(
                cluster.normalized_key for cluster in summary.cluster_results
            )
            expected_keys = tuple(
                OpeningPosition.from_sfen(opening).normalized_key for opening in openings
            )
            if reported_keys != expected_keys:
                raise RuntimeError("USI benchmark summary does not match the opening manifest")
            for opening_index, expected_key in enumerate(expected_keys):
                records_for_opening = benchmark_games[2 * opening_index : 2 * opening_index + 2]
                if any(
                    OpeningPosition.from_sfen(record.initial_sfen).normalized_key
                    != expected_key
                    for record in records_for_opening
                ):
                    raise RuntimeError("USI benchmark replay does not match the opening manifest")

            temporary_replay_path = temporary_path / "games.jsonl"
            append_games(temporary_replay_path, benchmark_games)
            replay_path = output_path / "games.jsonl"
            replay_identity = {
                "path": str(replay_path),
                "sha256": _sha256_file(temporary_replay_path),
                "bytes": temporary_replay_path.stat().st_size,
                "games": len(benchmark_games),
            }

            # Refuse a report whose supposedly immutable inputs changed during a long run.
            if _checkpoint_provenance(checkpoint_path) != checkpoint_provenance:
                raise RuntimeError("checkpoint changed during USI benchmark")
            if _artifact_provenance([engine_path])[0] != engine_identity:
                raise RuntimeError("USI engine executable changed during benchmark")
            if _artifact_provenance(args.artifact) != artifact_identities:
                raise RuntimeError("USI engine artifact changed during benchmark")
            if _source_tree_provenance() != source_tree:
                raise RuntimeError("Meteo source tree changed during benchmark")
            if opening_manifest["mode"] == "immutable_split_suite":
                opening_source = Path(str(opening_manifest["source"]))
                if _sha256_file(opening_source) != opening_manifest["source_sha256"]:
                    raise RuntimeError("opening suite changed during USI benchmark")

            termination_counts: dict[str, int] = {}
            for benchmark_game in benchmark_games:
                key = benchmark_game.termination.value
                termination_counts[key] = termination_counts.get(key, 0) + 1
            report = {
                "schema": "meteo-external-usi-benchmark-v2",
                "created_at": datetime.now(UTC).isoformat(),
                "output": str(output_path),
                "benchmark_contract": {
                    "unit": "normalized_opening_color_swapped_pair",
                    "games_per_opening": 2,
                    "duplicate_normalized_openings": "rejected",
                    "production_requires_immutable_opening_suite": True,
                    "legacy_single_opening_is_descriptive_only": True,
                },
                "checkpoint": checkpoint_provenance,
                "source": {
                    "tree": source_tree,
                    "git": git_identity,
                },
                "opening": opening_manifest,
                "engine": {
                    "name": rights.name,
                    "executable": engine_identity,
                    "arguments": list(args.engine_arg),
                    "working_directory": str(engine_working_directory),
                    "options": options,
                    "nodes_per_move": args.nodes,
                    "multipv": args.multipv,
                    "value_conversion": {
                        "input": "root MultiPV score from the side-to-move perspective",
                        "centipawn_formula": "tanh(cp / D)",
                        "tanh_denominator_D": external.value_scale,
                        "ponanza_coefficient_C": external.value_scale / 2.0,
                        "mate": "winning=+1; losing=-1",
                        "resign": -1.0,
                        "declaration_win": 1.0,
                    },
                    "artifacts": artifact_identities,
                },
                "rights": rights.to_dict(),
                "meteo_search": asdict(search),
                "human_play_compute_interlock": (
                    {"enabled": False}
                    if benchmark_interlock is None
                    else benchmark_interlock.public_metadata()
                ),
                "history_mode": UsiHistoryMode.GAME_PREFIX.value,
                "history_mode_missing_field_means": UsiHistoryMode.BOARD_ONLY.value,
                "history_source": "play_direct_game.initial_sfen + moves_so_far",
                "ci": {
                    "method": summary.ci_method,
                    "confidence_level": summary.confidence_level,
                    "seed": summary.seed,
                    "bootstrap_iterations": summary.bootstrap_iterations,
                    "resampling_unit": "independent_normalized_opening_pair",
                },
                "summary": summary.to_dict(),
                "elo": _paired_elo_report(summary),
                "termination": {
                    "counts": termination_counts,
                    "incomplete_games": summary.incomplete_games,
                    "incomplete_pairs": summary.incomplete_pairs,
                    "complete": summary.incomplete_games == 0,
                },
                "search": search_telemetry(benchmark_games),
                "evaluation_values": evaluation_value_telemetry(benchmark_games),
                "memory": memory_budget.to_dict() if memory_budget is not None else None,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "mlx": getattr(mx, "__version__", None),
                    "rsshogi": getattr(rsshogi, "__version__", None),
                },
                "replay": replay_identity,
            }
            (temporary_path / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if output_path.exists() or output_path.is_symlink():
                raise FileExistsError(f"refusing to overwrite benchmark bundle: {output_path}")
            temporary_path.rename(output_path)
        except BaseException:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "verify":
        print(json.dumps(_verify(args.output, args.workers, args.profile), indent=2))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
