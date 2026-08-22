"""Private post-bootstrap MLX continuation for exact value-only replay.

This module deliberately creates a new run root.  It never mutates the
completed 100B run, its base checkpoint, or any replay PSV.  The base model and
every Ranger tensor are resumed at their existing optimizer step; incremental
updates remain in the already-active integer-equivalent QAT phase.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import mlx.core as mx
import psutil

from .artifact_provenance import sha256_file
from .distillation_targets import CANONICAL_SCORER_IDS
from .incremental_value_replay import load_incremental_value_split
from .mlx_nnue import (
    MLX_BACKEND_PLAN_SCHEMA,
    MLX_CHECKPOINT_SCHEMA,
    QAT_FEATURE_ACCUMULATION,
    MeteoValueNnue,
    MlxNnueBatch,
    NativeBatchStream,
    TataraRanger,
    WrmLossParameters,
    _compiled_training_step,
    _ensure_mlx_export,
    _evaluate_mlx_batch,
    _latest_mlx_checkpoint,
    _native_binary_from_plan,
    _prune_mlx_checkpoints,
    _prune_mlx_exports,
    audit_mlx_master_weight_ranges,
    audit_quantised_distribution,
    effective_mlx_qat_feature_accumulation,
    evaluate_mlx_psv,
    load_mlx_checkpoint,
    save_mlx_checkpoint,
)
from .nnue_training import (
    DEFAULT_NET_ID,
    _exclusive_run_lock,
    _fsync_directory,
    _json_bytes,
    _strict_json,
    _write_json_atomic,
    _write_new,
    load_nagisa_plan,
    probe_value_only_psv,
)
from .teacher_data import PSV_RECORD_BYTES

INCREMENTAL_MLX_PLAN_SCHEMA = "meteo-incremental-mlx-plan-v1"
INCREMENTAL_MLX_STATE_SCHEMA = "meteo-incremental-mlx-state-v1"
INCREMENTAL_MLX_METRIC_SCHEMA = "meteo-incremental-mlx-metric-v1"
INCREMENTAL_MLX_CHECKPOINT_RECEIPT_SCHEMA = "meteo-incremental-mlx-checkpoint-receipt-v1"
SCALAR_TARGET_CONTRACT_SCHEMA = "meteo-scalar-target-contract-v1"

_CALIBRATED_ANCHOR_LABEL_SPACE = "calibrated_anchor_centipawn_v1"
_SIDE_TO_MOVE_PERSPECTIVE = "side_to_move"

ReplaySourceName = Literal["broad_anchor", "hard_exact"]
_SOURCE_NAMES: tuple[ReplaySourceName, ReplaySourceName] = (
    "broad_anchor",
    "hard_exact",
)
_SHA256 = frozenset("0123456789abcdef")


def _require_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _target_contract(
    receipt: Mapping[str, object], *, name: str
) -> tuple[dict[str, object], dict[str, object]]:
    raw = receipt.get("target_contract")
    if not isinstance(raw, dict):
        raise ValueError(f"{name} receipt must declare an explicit scalar target contract")
    expected = {
        "schema",
        "label_space",
        "score_perspective",
        "target_anchor_scorer_id",
        "target_anchor_identity_sha256",
        "calibration_sha256",
        "score_scale",
        "yaneuraou_fv_scale",
        "source_label_space",
        "conversion_kind",
        "conversion_artifact_sha256",
    }
    if set(raw) != expected:
        raise ValueError(f"{name} scalar target contract fields do not match the schema")
    scorer_id = _require_string(
        raw["target_anchor_scorer_id"], label=f"{name} target anchor scorer"
    )
    if scorer_id not in CANONICAL_SCORER_IDS:
        raise ValueError(f"{name} target anchor scorer is not canonical")
    label_space = _require_string(raw["label_space"], label=f"{name} label space")
    source_label_space = _require_string(
        raw["source_label_space"], label=f"{name} source label space"
    )
    conversion_kind = _require_string(
        raw["conversion_kind"], label=f"{name} scalar conversion kind"
    )
    if (
        raw["schema"] != SCALAR_TARGET_CONTRACT_SCHEMA
        or label_space != _CALIBRATED_ANCHOR_LABEL_SPACE
        or raw["score_perspective"] != _SIDE_TO_MOVE_PERSPECTIVE
    ):
        raise ValueError(f"{name} scalar target label space is unsupported")
    anchor_identity = _require_sha256(
        raw["target_anchor_identity_sha256"], label=f"{name} anchor identity"
    )
    calibration = _require_sha256(raw["calibration_sha256"], label=f"{name} scalar calibration")
    conversion_artifact = _require_sha256(
        raw["conversion_artifact_sha256"], label=f"{name} conversion artifact"
    )
    score_scale = _require_float(raw["score_scale"], label=f"{name} score scale")
    fv_scale = _require_int(
        raw["yaneuraou_fv_scale"], label=f"{name} YaneuraOu FV_SCALE", minimum=1
    )
    if score_scale <= 0.0:
        raise ValueError(f"{name} score scale must be positive")
    if conversion_kind == "identity_same_scale":
        if source_label_space != label_space or conversion_artifact != calibration:
            raise ValueError(
                f"{name} identity scalar conversion must already be on the calibrated target scale"
            )
    elif conversion_kind == "calibrated_to_target":
        if source_label_space == label_space:
            raise ValueError(
                f"{name} calibrated conversion must identify its distinct source label space"
            )
    else:
        raise ValueError(f"{name} scalar conversion kind is unsupported")
    target_identity: dict[str, object] = {
        "schema": SCALAR_TARGET_CONTRACT_SCHEMA,
        "label_space": label_space,
        "score_perspective": _SIDE_TO_MOVE_PERSPECTIVE,
        "target_anchor_scorer_id": scorer_id,
        "target_anchor_identity_sha256": anchor_identity,
        "calibration_sha256": calibration,
        "score_scale": score_scale,
        "yaneuraou_fv_scale": fv_scale,
    }
    normalized = {
        **target_identity,
        **{
            "source_label_space": source_label_space,
            "conversion_kind": conversion_kind,
            "conversion_artifact_sha256": conversion_artifact,
        },
    }
    return normalized, target_identity


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    _reject_symlink_components(absolute, label=label)
    source = absolute.resolve(strict=True)
    metadata = source.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return source


def _regular_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    _reject_symlink_components(absolute, label=label)
    source = absolute.resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return source


def _strict_object(path: Path, *, label: str) -> dict[str, Any]:
    source = _regular_file(path, label=label)
    return _strict_json(source.read_bytes(), label=label)


def _stat_identity(path: Path) -> dict[str, int]:
    metadata = path.stat(follow_symlinks=False)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
    }


def _label_position_ids(
    receipt_path: Path,
    receipt_value: Mapping[str, object],
    *,
    name: str,
    records: int,
    hard_exact: bool,
    target_anchor_scorer_id: str,
) -> tuple[Path, tuple[str, ...], dict[str, object]]:
    labels_identity = receipt_value.get("labels")
    if not isinstance(labels_identity, dict):
        raise ValueError(f"{name} receipt must identify labels.jsonl")
    filename = labels_identity.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
    ):
        raise ValueError(f"{name} labels filename must be one local basename")
    labels_path = _regular_file(receipt_path.parent / filename, label=f"{name} labels")
    if labels_identity.get("bytes") != labels_path.stat().st_size or labels_identity.get(
        "sha256"
    ) != sha256_file(labels_path):
        raise ValueError(f"{name} receipt does not identify labels.jsonl")
    position_ids: list[str] = []
    scorer_ids: set[str | None] = set()
    requested_nodes: set[int | None] = set()
    kinds: set[str | None] = set()
    with labels_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{name} labels.jsonl contains an empty row")
            row = _strict_json(line, label=f"{name} labels row {line_number}")
            position_ids.append(
                _require_sha256(
                    row.get("position_id"),
                    label=f"{name} labels row {line_number} position_id",
                )
            )
            raw_scorer = row.get("scorer_id")
            if raw_scorer is not None and not isinstance(raw_scorer, str):
                raise ValueError(f"{name} labels row {line_number} scorer_id is invalid")
            raw_nodes = row.get("requested_nodes")
            if raw_nodes is not None and (
                isinstance(raw_nodes, bool) or not isinstance(raw_nodes, int) or raw_nodes < 1
            ):
                raise ValueError(f"{name} labels row {line_number} requested_nodes is invalid")
            raw_kind = row.get("kind")
            if raw_kind is not None and not isinstance(raw_kind, str):
                raise ValueError(f"{name} labels row {line_number} kind is invalid")
            scorer_ids.add(raw_scorer)
            requested_nodes.add(raw_nodes)
            kinds.add(raw_kind)
    if len(position_ids) != records:
        raise ValueError(f"{name} labels count does not match its PSV")
    if len(set(position_ids)) != len(position_ids):
        raise ValueError(f"{name} labels contain duplicate normalized positions")
    non_null_scorers = {value for value in scorer_ids if value is not None}
    non_null_nodes = {value for value in requested_nodes if value is not None}
    if len(non_null_scorers) > 1:
        raise ValueError(f"{name} labels mix multiple raw scorer scales")
    if len(non_null_nodes) > 1:
        raise ValueError(f"{name} labels mix multiple search budgets")
    if hard_exact and (
        scorer_ids != {target_anchor_scorer_id}
        or len(non_null_nodes) != 1
        or kinds != {"anchor_search_exact"}
    ):
        raise ValueError(
            "hard exact labels must use one anchor scorer, one node budget, and exact search kind"
        )
    summary: dict[str, object] = {
        "scorer_ids": sorted(non_null_scorers),
        "requested_nodes": sorted(non_null_nodes),
        "kinds": sorted(value for value in kinds if value is not None),
        "all_rows_declared_semantics": (
            None not in scorer_ids and None not in requested_nodes and None not in kinds
        ),
    }
    return labels_path, tuple(position_ids), summary


def _source_identity(
    *,
    name: str,
    psv: Path,
    receipt: Path,
    split_receipt: Path,
    expected_split: str,
    hard_exact: bool,
    batch_size: int,
) -> dict[str, object]:
    source = _regular_file(psv, label=f"{name} PSV")
    receipt_path = _regular_file(receipt, label=f"{name} receipt")
    split_path = _regular_file(split_receipt, label=f"{name} split receipt")
    if source.stat().st_size % PSV_RECORD_BYTES:
        raise ValueError(f"{name} PSV byte length is not a whole PackedSfenValue stream")
    records = source.stat().st_size // PSV_RECORD_BYTES
    if records < batch_size or records % batch_size:
        raise ValueError(f"{name} PSV records must be a positive multiple of batch_size")
    split = load_incremental_value_split(split_path)
    if split.split_id != expected_split:
        raise ValueError(f"{name} split must be {expected_split!r}")
    receipt_value = _strict_object(receipt_path, label=f"{name} receipt")
    target_contract, target_identity = _target_contract(receipt_value, name=name)
    psv_identity = receipt_value.get("psv")
    if not isinstance(psv_identity, dict):
        raise ValueError(f"{name} receipt must contain a PSV identity")
    psv_sha256 = sha256_file(source)
    if (
        psv_identity.get("sha256") != psv_sha256
        or psv_identity.get("bytes") != source.stat().st_size
        or receipt_value.get("records") != records
    ):
        raise ValueError(f"{name} receipt does not identify its PSV")
    split_identity = receipt_value.get("split")
    if not isinstance(split_identity, dict) or split_identity.get("receipt_sha256") != sha256_file(
        split_path
    ):
        raise ValueError(f"{name} receipt does not identify its split receipt")
    labels_path, selected_position_ids, label_semantics = _label_position_ids(
        receipt_path,
        receipt_value,
        name=name,
        records=records,
        hard_exact=hard_exact,
        target_anchor_scorer_id=cast(str, target_identity["target_anchor_scorer_id"]),
    )
    semantic_scorers = cast(list[str], label_semantics["scorer_ids"])
    if (
        target_contract["conversion_kind"] == "identity_same_scale"
        and semantic_scorers
        and semantic_scorers != [target_identity["target_anchor_scorer_id"]]
    ):
        raise ValueError(f"{name} identity-scale labels do not come from the target anchor")
    if not set(selected_position_ids).issubset(split.position_ids):
        raise ValueError(f"{name} labels are not a subset of the pre-labelled split")
    if hard_exact and (
        receipt_value.get("schema") != "meteo-incremental-value-replay-v1"
        or receipt_value.get("mode") != "value_only_nnue_incremental_exact_labels"
        or receipt_value.get("complete") is not True
        or receipt_value.get("policy_targets") != 0
        or receipt_value.get("cross_teacher_value_average") is not False
        or receipt_value.get("bounds_written_as_point_targets") is not False
        or receipt_value.get("unresolved_positions_written") is not False
        or receipt_value.get("history_dependent_positions_written") is not False
        or receipt_value.get("publication_allowed") is not False
    ):
        raise ValueError("hard exact receipt does not prove exact board-only scalar labels")
    probe = probe_value_only_psv(source, board_samples=min(records, 4_096))
    if probe.get("records") != records or probe.get("policy_targets") != 0:
        raise ValueError(f"{name} PSV is not a complete value-only stream")
    return {
        "name": name,
        "path": str(source),
        "sha256": psv_sha256,
        "bytes": source.stat().st_size,
        "records": records,
        "batches": records // batch_size,
        "stat_identity": _stat_identity(source),
        "receipt": {
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
            "schema": receipt_value.get("schema"),
        },
        "labels": {
            "path": str(labels_path),
            "sha256": sha256_file(labels_path),
            "bytes": labels_path.stat().st_size,
            "position_ids": list(selected_position_ids),
            "semantics": label_semantics,
        },
        "target_contract": target_contract,
        "target_identity": target_identity,
        "target_identity_sha256": hashlib.sha256(_json_bytes(target_identity)).hexdigest(),
        "split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "split_id": split.split_id,
            "source_games_sha256": split.source_games_sha256,
            "position_ids": list(split.position_ids),
        },
        "probe": probe,
    }


def _base_checkpoint_identity(
    base_run_directory: Path, base_checkpoint: Path
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_root = _regular_directory(base_run_directory, label="base 100B run")
    checkpoint = _regular_directory(base_checkpoint, label="base MLX checkpoint")
    try:
        checkpoint.relative_to(base_root / "checkpoints")
    except ValueError as error:
        raise ValueError("base checkpoint must be inside the completed base run") from error
    state = _strict_object(base_root / "mlx-state.json", label="base MLX state")
    if state.get("status") != "complete":
        raise ValueError("base 100B MLX run must be complete before incremental training")
    latest = state.get("latest_checkpoint")
    if (
        not isinstance(latest, str)
        or _regular_directory(Path(latest), label="base latest checkpoint") != checkpoint
    ):
        raise ValueError("base checkpoint must be the completed run's latest checkpoint")
    common = _strict_object(base_root / "plan.json", label="base source plan")
    mlx_plan = _strict_object(base_root / "mlx-plan.json", label="base MLX plan")
    if mlx_plan.get("schema") != MLX_BACKEND_PLAN_SCHEMA:
        raise ValueError("base MLX plan must use the QAT-capable v3 contract")
    if mlx_plan.get("source_plan_sha256") != sha256_file(base_root / "plan.json"):
        raise ValueError("base MLX plan no longer matches its source plan")
    manifest = _strict_object(checkpoint / "manifest.json", label="base checkpoint manifest")
    if manifest.get("schema") != MLX_CHECKPOINT_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("base checkpoint is incomplete")
    for key in ("model_file", "optimizer_file"):
        identity = manifest.get(key)
        if not isinstance(identity, dict) or not isinstance(identity.get("name"), str):
            raise ValueError(f"base checkpoint has no {key} identity")
        artifact = _regular_file(checkpoint / identity["name"], label=f"base {key}")
        if identity.get("bytes") != artifact.stat().st_size or identity.get(
            "sha256"
        ) != sha256_file(artifact):
            raise ValueError(f"base checkpoint {key} identity mismatch")
    optimizer = manifest.get("optimizer")
    numerics = manifest.get("numerics")
    if not isinstance(optimizer, dict) or not isinstance(numerics, dict):
        raise ValueError("base checkpoint optimizer/numeric identity is missing")
    optimizer_step = _require_int(optimizer.get("step"), label="base optimizer step")
    plan_numerics = mlx_plan.get("numerics")
    if not isinstance(plan_numerics, dict):
        raise ValueError("base MLX plan has no numeric contract")
    warmup = _require_int(
        plan_numerics.get("warmup_optimizer_steps"),
        label="base QAT warmup steps",
        minimum=1,
    )
    if (
        optimizer_step < warmup
        or numerics.get("next_update_phase") != "quantisation_aware"
        or numerics.get("warmup_optimizer_steps") != warmup
        or numerics.get("deployment_graph") != plan_numerics.get("deployment_graph")
        or plan_numerics.get("deployment_graph") != "tatara_yaneuraou_integer_equivalent_ste"
        or plan_numerics.get("warmup_forward") != "ft_float16_dense_float32"
        or effective_mlx_qat_feature_accumulation(base_root, mlx_plan) != QAT_FEATURE_ACCUMULATION
    ):
        raise ValueError("base checkpoint has not entered the production QAT phase")
    return base_root, checkpoint, common, mlx_plan, manifest


def _mix_cycle(broad_weight: int, hard_weight: int) -> tuple[ReplaySourceName, ...]:
    _require_int(broad_weight, label="broad anchor weight", minimum=1)
    _require_int(hard_weight, label="hard exact weight", minimum=1)
    if broad_weight + hard_weight > 1_024:
        raise ValueError("replay mix cycle is unreasonably large")
    return cast(
        tuple[ReplaySourceName, ...],
        ("broad_anchor",) * broad_weight + ("hard_exact",) * hard_weight,
    )


def weighted_batch_source(mix_cycle: Sequence[str], incremental_batch: int) -> ReplaySourceName:
    """Return the deterministic source selected for a zero-based added batch."""

    if incremental_batch < 0 or not mix_cycle:
        raise ValueError("incremental batch and mix cycle are invalid")
    source = mix_cycle[incremental_batch % len(mix_cycle)]
    if source not in _SOURCE_NAMES:
        raise ValueError(f"unknown replay source in mix cycle: {source!r}")
    return source


def _source_batches_before(
    mix_cycle: Sequence[str], source: ReplaySourceName, incremental_batch: int
) -> int:
    cycles, remainder = divmod(incremental_batch, len(mix_cycle))
    per_cycle = sum(item == source for item in mix_cycle)
    return cycles * per_cycle + sum(item == source for item in mix_cycle[:remainder])


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_writable_layout(root: Path) -> None:
    """Reject redirected mutable children before any executor write."""

    verified_root = _regular_directory(root, label="incremental MLX writable root")
    for name in ("checkpoints", "exports", "logs", "receipts", "private"):
        child = _regular_directory(verified_root / name, label=f"incremental {name}")
        if child.parent != verified_root:
            raise PermissionError(f"incremental writable directory escaped its root: {child}")
    lock_path = verified_root / "RUNNING.lock"
    if os.path.lexists(lock_path):
        _regular_file(lock_path, label="incremental run lock")


def _publish_create_only_directory(stage: Path, target: Path) -> None:
    """Atomically reserve a name without POSIX rename replacement semantics."""

    try:
        target.mkdir(mode=0o700)
    except FileExistsError:
        raise FileExistsError(f"incremental run destination appeared: {target}") from None
    try:
        for child in stage.iterdir():
            os.rename(child, target / child.name)
        _fsync_directory(target)
        stage.rmdir()
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def prepare_incremental_mlx_run(
    output_directory: Path,
    *,
    base_run_directory: Path,
    base_checkpoint: Path,
    broad_anchor_psv: Path,
    broad_anchor_receipt: Path,
    broad_anchor_split_receipt: Path,
    hard_exact_psv: Path,
    hard_exact_receipt: Path,
    hard_exact_split_receipt: Path,
    calibration_psv: Path,
    calibration_receipt: Path,
    calibration_split_receipt: Path,
    additional_optimizer_steps: int,
    learning_rate: float,
    broad_weight: int = 3,
    hard_weight: int = 1,
    checkpoint_batches: int = 128,
    log_batches: int = 16,
    calibration_validation_positions: int = 262_144,
    quantisation_audit_positions: int = 8_192,
) -> dict[str, object]:
    """Create an immutable, separate continuation plan without copying PSV data."""

    target = _absolute(output_directory)
    _reject_symlink_components(target, label="incremental MLX output")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite incremental MLX run: {target}")
    base_root, checkpoint, common, mlx_plan, manifest = _base_checkpoint_identity(
        base_run_directory, base_checkpoint
    )
    if _path_is_within(target, base_root):
        raise ValueError("incremental run root must be outside the completed base run")
    execution = mlx_plan.get("execution")
    training = common.get("training")
    if not isinstance(execution, dict) or not isinstance(training, dict):
        raise ValueError("base execution/training contract is incomplete")
    batch_size = _require_int(execution.get("batch_size"), label="batch size", minimum=1)
    additional_steps = _require_int(
        additional_optimizer_steps,
        label="additional optimizer steps",
        minimum=1,
    )
    rate = _require_float(learning_rate, label="incremental learning rate")
    if rate <= 0.0:
        raise ValueError("incremental learning rate must be positive")
    for label, value in (
        ("checkpoint batches", checkpoint_batches),
        ("log batches", log_batches),
        ("validation positions", calibration_validation_positions),
        ("quantisation audit positions", quantisation_audit_positions),
    ):
        _require_int(value, label=label, minimum=1)
    if (
        calibration_validation_positions % batch_size
        or quantisation_audit_positions > calibration_validation_positions
    ):
        raise ValueError("validation must use full batches and contain the QAT audit sample")
    sources = {
        "broad_anchor": _source_identity(
            name="broad_anchor",
            psv=broad_anchor_psv,
            receipt=broad_anchor_receipt,
            split_receipt=broad_anchor_split_receipt,
            expected_split="train",
            hard_exact=False,
            batch_size=batch_size,
        ),
        "hard_exact": _source_identity(
            name="hard_exact",
            psv=hard_exact_psv,
            receipt=hard_exact_receipt,
            split_receipt=hard_exact_split_receipt,
            expected_split="train",
            hard_exact=True,
            batch_size=batch_size,
        ),
        "calibration": _source_identity(
            name="calibration",
            psv=calibration_psv,
            receipt=calibration_receipt,
            split_receipt=calibration_split_receipt,
            expected_split="calibration",
            hard_exact=False,
            batch_size=batch_size,
        ),
    }
    target_identities = {
        cast(str, identity["target_identity_sha256"]): cast(
            dict[str, object], identity["target_identity"]
        )
        for identity in sources.values()
    }
    if len(target_identities) != 1:
        raise ValueError(
            "broad, hard, and calibration PSV receipts target incompatible scalar spaces"
        )
    scalar_target_identity = next(iter(target_identities.values()))
    training_score_scale = _require_float(
        training.get("score_scale"), label="base training score scale"
    )
    training_fv_scale = _require_int(
        training.get("yaneuraou_fv_scale"),
        label="base training YaneuraOu FV_SCALE",
        minimum=1,
    )
    if (
        scalar_target_identity["score_scale"] != training_score_scale
        or scalar_target_identity["yaneuraou_fv_scale"] != training_fv_scale
    ):
        raise ValueError("replay scalar target contract is incompatible with the base model scale")
    position_sets = {
        name: set(cast(dict[str, Any], identity["labels"])["position_ids"])
        for name, identity in sources.items()
    }
    overlap: dict[str, int] = {}
    names = tuple(sources)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap[f"{left}__{right}"] = len(position_sets[left] & position_sets[right])
    if any(overlap.values()):
        raise ValueError("replay sources contain duplicate normalized position identities")
    for identity in sources.values():
        labels_identity = cast(dict[str, Any], identity["labels"])
        selected_position_ids = cast(list[str], labels_identity.pop("position_ids"))
        labels_identity["position_ids_sha256"] = hashlib.sha256(
            _json_bytes(selected_position_ids)
        ).hexdigest()
        labels_identity["positions"] = len(selected_position_ids)
        split_identity = cast(dict[str, Any], identity["split"])
        position_ids = cast(list[str], split_identity.pop("position_ids"))
        split_identity["position_ids_sha256"] = hashlib.sha256(
            _json_bytes(position_ids)
        ).hexdigest()
        split_identity["positions"] = len(position_ids)
    if cast(int, sources["calibration"]["records"]) < calibration_validation_positions:
        raise ValueError("calibration PSV is smaller than calibration_validation_positions")
    cycle = _mix_cycle(broad_weight, hard_weight)
    optimizer = cast(dict[str, Any], manifest["optimizer"])
    base_step = int(optimizer["step"])
    progress_source = _regular_file(
        base_root / "private" / "progress.bin", label="base progress.bin"
    )
    plan: dict[str, object] = {
        "schema": INCREMENTAL_MLX_PLAN_SCHEMA,
        "complete": True,
        "created_unix": time.time(),
        "mode": "post_100b_value_only_qat_continuation",
        "scope": "private_local_only",
        "publication_allowed": False,
        "base": {
            "run_directory": str(base_root),
            "checkpoint": str(checkpoint),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "model_sha256": cast(dict[str, Any], manifest["model_file"])["sha256"],
            "optimizer_sha256": cast(dict[str, Any], manifest["optimizer_file"])["sha256"],
            "optimizer_step": base_step,
            "source_plan_sha256": sha256_file(base_root / "plan.json"),
            "mlx_plan_sha256": sha256_file(base_root / "mlx-plan.json"),
            "state_sha256": sha256_file(base_root / "mlx-state.json"),
            "required_status": "complete",
            "read_only": True,
        },
        "sources": sources,
        "scalar_target_contract": {
            **scalar_target_identity,
            "identity_sha256": hashlib.sha256(_json_bytes(scalar_target_identity)).hexdigest(),
            "all_sources_compatible": True,
            "raw_cross_teacher_cp_mixing_allowed": False,
        },
        "duplicate_contract": {
            "identity": "normalized_board_position_sha256",
            "pairwise_overlap_counts": overlap,
            "duplicates_allowed": False,
            "verified_during_prepare_from_full_split_position_ids": True,
        },
        "schedule": {
            "kind": "deterministic_weighted_cycle",
            "cycle": list(cycle),
            "cycle_sha256": hashlib.sha256(_json_bytes(list(cycle))).hexdigest(),
            "broad_weight": broad_weight,
            "hard_weight": hard_weight,
            "additional_optimizer_steps": additional_steps,
            "base_optimizer_step": base_step,
            "target_optimizer_step": base_step + additional_steps,
            "physical_psv_copy": False,
        },
        "optimizer": {
            "continuation": "model_and_all_ranger_state_tensors",
            "state_reset_allowed": False,
            "learning_rate": rate,
        },
        "numerics": {
            "phase": "quantisation_aware",
            "phase_reset_allowed": False,
            "warmup_reentry_allowed": False,
            "float_master_weights_retained": True,
            "deployment_graph": "tatara_yaneuraou_integer_equivalent_ste",
        },
        "execution": {
            "batch_size": batch_size,
            "checkpoint_batches": checkpoint_batches,
            "log_batches": log_batches,
            "calibration_validation_positions": calibration_validation_positions,
            "quantisation_audit_positions": quantisation_audit_positions,
            "native_decode_threads": int(execution.get("native_decode_threads", 1)),
            "native_prefetch_batches": int(execution.get("native_prefetch_batches", 2)),
            "mlx_memory_limit_bytes": int(execution.get("mlx_memory_limit_bytes", 0)),
            "mlx_cache_limit_bytes": int(execution.get("mlx_cache_limit_bytes", 0)),
        },
        "retention": {
            "scope": "incremental_run_local_only",
            "checkpoint_generations": 2,
            "export_generations": 2,
            "checkpoint_reload_before_prune": True,
            "global_base_plus_incremental_accounting_implemented": False,
        },
        "compatibility": {
            "source_plan_file": "plan.json",
            "mlx_plan_file": "mlx-plan.json",
            "progress_file": "private/progress.bin",
            "progress_sha256": sha256_file(progress_source),
        },
        "write_guard": {
            "writable_root_only": True,
            "forbidden_base_root": str(base_root),
            "external_sources_read_only": True,
            "source_content_verification": (
                "full_sha256_at_prepare_then_device_inode_size_mtime_on_resume"
            ),
        },
    }
    state: dict[str, object] = {
        "schema": INCREMENTAL_MLX_STATE_SCHEMA,
        "status": "prepared",
        "pid": None,
        "plan_sha256": "",
        "base_optimizer_step": base_step,
        "durable_incremental_batches": 0,
        "observed_incremental_batches": 0,
        "durable_optimizer_step": base_step,
        "observed_optimizer_step": base_step,
        "latest_checkpoint": str(checkpoint),
        "latest_export": None,
        "last_metrics": None,
        "last_calibration_validation": None,
        "last_quantisation_audit": None,
        "failure": None,
        "updated_unix": time.time(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _regular_directory(target.parent, label="incremental MLX output parent") / target.name
    if _path_is_within(target, base_root):
        raise ValueError("incremental run root must be outside the completed base run")
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for child in ("checkpoints", "exports", "logs", "receipts", "private"):
            (stage / child).mkdir(mode=0o700)
        plan_bytes = (base_root / "plan.json").read_bytes()
        mlx_plan_bytes = (base_root / "mlx-plan.json").read_bytes()
        _write_new(stage / "plan.json", plan_bytes, mode=0o600)
        _write_new(stage / "mlx-plan.json", mlx_plan_bytes, mode=0o600)
        _write_new(stage / "private" / "progress.bin", progress_source.read_bytes(), mode=0o600)
        incremental_plan_bytes = _json_bytes(plan)
        state["plan_sha256"] = hashlib.sha256(incremental_plan_bytes).hexdigest()
        _write_new(stage / "incremental-plan.json", incremental_plan_bytes, mode=0o600)
        _write_new(stage / "incremental-state.json", _json_bytes(state), mode=0o600)
        _fsync_directory(stage / "private")
        _fsync_directory(stage)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"incremental run destination appeared: {target}")
        _publish_create_only_directory(stage, target)
        _fsync_directory(target.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return load_incremental_mlx_plan(target)


def load_incremental_mlx_plan(run_directory: Path) -> dict[str, object]:
    """Load the plan and deeply re-hash every pinned external input."""

    return _load_incremental_mlx_plan(run_directory, verify_external_content=True)


def _load_incremental_mlx_plan(
    run_directory: Path, *, verify_external_content: bool
) -> dict[str, object]:
    root = _regular_directory(run_directory, label="incremental MLX run")
    plan_path = _regular_file(root / "incremental-plan.json", label="incremental plan")
    plan = _strict_json(plan_path.read_bytes(), label="incremental MLX plan")
    if plan.get("schema") != INCREMENTAL_MLX_PLAN_SCHEMA or plan.get("complete") is not True:
        raise ValueError("incremental MLX plan schema/complete marker mismatch")
    state = _strict_object(root / "incremental-state.json", label="incremental MLX state")
    if state.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("incremental MLX state no longer matches its immutable plan")
    guard = plan.get("write_guard")
    if not isinstance(guard, dict) or not isinstance(guard.get("forbidden_base_root"), str):
        raise ValueError("incremental MLX write guard is missing")
    base_root = _regular_directory(Path(guard["forbidden_base_root"]), label="guarded base run")
    if _path_is_within(root, base_root):
        raise PermissionError("incremental executor refuses to write inside the base run")
    compatibility = plan.get("compatibility")
    base = plan.get("base")
    if not isinstance(compatibility, dict) or not isinstance(base, dict):
        raise ValueError("incremental compatibility/base identity is missing")
    for filename, expected in (
        ("plan.json", base.get("source_plan_sha256")),
        ("mlx-plan.json", base.get("mlx_plan_sha256")),
        ("private/progress.bin", compatibility.get("progress_sha256")),
    ):
        if sha256_file(_regular_file(root / filename, label=filename)) != expected:
            raise ValueError(f"incremental compatibility artifact changed: {filename}")
    if verify_external_content:
        _validate_external_identities(plan)
    return cast(dict[str, object], plan)


def _validate_external_identities(plan: Mapping[str, object]) -> None:
    base = plan.get("base")
    sources = plan.get("sources")
    if not isinstance(base, dict) or not isinstance(sources, dict):
        raise ValueError("incremental external identities are missing")
    base_root = _regular_directory(Path(cast(str, base["run_directory"])), label="base run")
    if sha256_file(base_root / "mlx-state.json") != base.get("state_sha256"):
        raise ValueError("completed base run state changed after incremental prepare")
    checkpoint = _regular_directory(Path(cast(str, base["checkpoint"])), label="base checkpoint")
    if sha256_file(checkpoint / "manifest.json") != base.get("checkpoint_manifest_sha256"):
        raise ValueError("base checkpoint manifest changed after incremental prepare")
    manifest = _strict_object(checkpoint / "manifest.json", label="base checkpoint manifest")
    for key, expected_key in (
        ("model_file", "model_sha256"),
        ("optimizer_file", "optimizer_sha256"),
    ):
        identity = manifest.get(key)
        if not isinstance(identity, dict) or not isinstance(identity.get("name"), str):
            raise ValueError(f"base checkpoint {key} identity is missing")
        artifact = _regular_file(checkpoint / identity["name"], label=f"base {key}")
        if sha256_file(artifact) != base.get(expected_key):
            raise ValueError(f"base checkpoint {key} changed after incremental prepare")
    for name, raw_identity in sources.items():
        if not isinstance(name, str) or not isinstance(raw_identity, dict):
            raise ValueError("incremental source identity is malformed")
        psv = _regular_file(Path(cast(str, raw_identity["path"])), label=f"{name} PSV")
        if _stat_identity(psv) != raw_identity.get("stat_identity") or sha256_file(
            psv
        ) != raw_identity.get("sha256"):
            raise ValueError(f"{name} PSV identity changed after hashing")
        receipt = raw_identity.get("receipt")
        labels = raw_identity.get("labels")
        split = raw_identity.get("split")
        if (
            not isinstance(receipt, dict)
            or not isinstance(labels, dict)
            or not isinstance(split, dict)
        ):
            raise ValueError(f"{name} receipt/labels/split identity is missing")
        receipt_path = _regular_file(Path(cast(str, receipt["path"])), label=f"{name} receipt")
        if sha256_file(receipt_path) != receipt.get("sha256"):
            raise ValueError(f"{name} receipt changed after incremental prepare")
        labels_path = _regular_file(Path(cast(str, labels["path"])), label=f"{name} labels")
        if labels_path.stat().st_size != labels.get("bytes") or sha256_file(
            labels_path
        ) != labels.get("sha256"):
            raise ValueError(f"{name} labels changed after incremental prepare")
        split_path = _regular_file(Path(cast(str, split["path"])), label=f"{name} split")
        if sha256_file(split_path) != split.get("sha256"):
            raise ValueError(f"{name} split changed after incremental prepare")


def load_incremental_mlx_state(run_directory: Path) -> dict[str, Any]:
    root = _regular_directory(run_directory, label="incremental MLX run")
    state = _strict_object(root / "incremental-state.json", label="incremental MLX state")
    if state.get("schema") != INCREMENTAL_MLX_STATE_SCHEMA:
        raise ValueError("incremental MLX state schema mismatch")
    return state


def _set_state(root: Path, state: dict[str, Any], **values: object) -> None:
    _validate_writable_layout(root)
    state.update(values)
    state["updated_unix"] = time.time()
    _write_json_atomic(root / "incremental-state.json", state)


def _log(root: Path, event: str, **fields: object) -> dict[str, object]:
    _validate_writable_layout(root)
    record: dict[str, object] = {
        "schema": INCREMENTAL_MLX_METRIC_SCHEMA,
        "time_unix": time.time(),
        "event": event,
        **fields,
    }
    payload = _json_bytes(record)
    with (root / "logs" / "incremental-training.jsonl").open("ab", buffering=0) as stream:
        stream.write(payload)
        os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return record


def _source_mapping(plan: Mapping[str, object], source: ReplaySourceName) -> dict[str, Any]:
    sources = plan.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get(source), dict):
        raise ValueError(f"incremental source is missing: {source}")
    return cast(dict[str, Any], sources[source])


def _validate_source_stat(identity: Mapping[str, object], *, name: str) -> Path:
    source = _regular_file(Path(cast(str, identity["path"])), label=f"{name} PSV")
    if _stat_identity(source) != identity.get("stat_identity"):
        raise ValueError(f"{name} PSV identity changed during incremental training")
    return source


def _execute_training_batch(
    step_fn: Callable[..., tuple[mx.array, mx.array, mx.array]],
    learning_rate: mx.array,
    batch: MlxNnueBatch,
    *,
    model: MeteoValueNnue,
    optimizer: TataraRanger,
) -> tuple[float, float, float]:
    loss, output_mean, target_mean = step_fn(learning_rate, *batch.mlx())
    mx.eval(loss, output_mean, target_mean, model.state, optimizer.state)
    values = float(loss.item()), float(output_mean.item()), float(target_mean.item())
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("incremental MLX update produced non-finite metrics")
    return values


def _checkpoint_transaction_unprotected(
    root: Path,
    *,
    plan: Mapping[str, object],
    model: MeteoValueNnue,
    optimizer: TataraRanger,
    optimizer_step: int,
    incremental_batches: int,
    reference_batch: MlxNnueBatch,
    parameters: WrmLossParameters,
    native_binary: Path,
) -> tuple[MeteoValueNnue, TataraRanger, Path, Path, dict[str, object]]:
    execution = cast(dict[str, Any], plan["execution"])
    schedule = cast(dict[str, Any], plan["schedule"])
    batch_size = int(execution["batch_size"])
    checkpoint, _manifest = save_mlx_checkpoint(
        root,
        model=model,
        optimizer=optimizer,
        optimizer_step=optimizer_step,
        superbatch=1 + incremental_batches // len(cast(list[str], schedule["cycle"])),
        completed_batches_in_superbatch=incremental_batches
        % len(cast(list[str], schedule["cycle"])),
        presentations=optimizer_step * batch_size,
    )
    modes = ("float32", "quantisation_aware", "quantised_reference")
    expected = {
        mode: _evaluate_mlx_batch(
            model,
            reference_batch,
            parameters=parameters,
            forward_mode=cast(Any, mode),
        )
        for mode in modes
    }
    reloaded_model, reloaded_optimizer, reloaded_manifest = load_mlx_checkpoint(
        checkpoint,
        learning_rate=float(cast(dict[str, Any], plan["optimizer"])["learning_rate"]),
    )
    observed = {
        mode: _evaluate_mlx_batch(
            reloaded_model,
            reference_batch,
            parameters=parameters,
            forward_mode=cast(Any, mode),
        )
        for mode in modes
    }
    maximum_difference = max(
        abs(left - right)
        for mode in modes
        for left, right in zip(expected[mode], observed[mode], strict=True)
    )
    if maximum_difference > 1e-9:
        raise ValueError("incremental checkpoint model/optimizer reload was not exact")
    if int(reloaded_optimizer.state["step"].item()) != optimizer_step:
        raise ValueError("incremental checkpoint optimizer step did not reload exactly")
    calibration = cast(dict[str, Any], cast(dict[str, Any], plan["sources"])["calibration"])
    validation = evaluate_mlx_psv(
        reloaded_model,
        psv=Path(cast(str, calibration["path"])),
        progress=root / "private" / "progress.bin",
        native_binary=native_binary,
        positions=int(execution["calibration_validation_positions"]),
        batch_size=batch_size,
        parameters=parameters,
        threads=int(execution["native_decode_threads"]),
        forward_mode="quantised_reference",
    )
    export, export_receipt = _ensure_mlx_export(
        root,
        checkpoint=checkpoint,
        optimizer_step=optimizer_step,
        native_binary=native_binary,
    )
    audit_positions = int(execution["quantisation_audit_positions"])
    quantisation = audit_quantised_distribution(
        reloaded_model,
        network=export / "network.bin",
        psv=Path(cast(str, calibration["path"])),
        progress=root / "private" / "progress.bin",
        native_binary=native_binary,
        positions=audit_positions,
        batch_size=audit_positions,
        parameters=parameters,
        fv_scale=int(
            cast(dict[str, Any], load_nagisa_plan(root)["training"])["yaneuraou_fv_scale"]
        ),
        threads=int(execution["native_decode_threads"]),
    )
    master_audit = audit_mlx_master_weight_ranges(checkpoint)
    receipt: dict[str, object] = {
        "schema": INCREMENTAL_MLX_CHECKPOINT_RECEIPT_SCHEMA,
        "complete": True,
        "optimizer_step": optimizer_step,
        "incremental_batches": incremental_batches,
        "numeric_phase": "quantisation_aware",
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": sha256_file(checkpoint / "manifest.json"),
            "model": reloaded_manifest["model_file"],
            "optimizer": reloaded_manifest["optimizer_file"],
            "reload_max_abs_difference": maximum_difference,
        },
        "calibration_validation": validation,
        "export": {"path": str(export), "receipt": export_receipt},
        "quantisation_audit": quantisation,
        "master_weight_audit": master_audit,
        "source_manifest_sha256": sha256_file(root / "incremental-plan.json"),
        "local_only": True,
    }
    receipt_path = root / "receipts" / f"incremental-step-{optimizer_step:08d}.json"
    if receipt_path.is_file():
        existing = _strict_json(receipt_path.read_bytes(), label="incremental checkpoint receipt")
        if existing != receipt:
            raise ValueError("existing incremental checkpoint receipt differs")
    else:
        _write_new(receipt_path, _json_bytes(receipt), mode=0o600)
    _fsync_directory(root / "receipts")
    return reloaded_model, reloaded_optimizer, checkpoint, export, receipt


def _checkpoint_transaction(
    root: Path,
    *,
    plan: Mapping[str, object],
    model: MeteoValueNnue,
    optimizer: TataraRanger,
    optimizer_step: int,
    incremental_batches: int,
    reference_batch: MlxNnueBatch,
    parameters: WrmLossParameters,
    native_binary: Path,
) -> tuple[MeteoValueNnue, TataraRanger, Path, Path, dict[str, object]]:
    """Publish one all-or-rollback checkpoint/validation/export/audit transaction."""

    _validate_writable_layout(root)
    checkpoint = root / "checkpoints" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    export = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{optimizer_step:08d}"
    receipt = root / "receipts" / f"incremental-step-{optimizer_step:08d}.json"
    try:
        return _checkpoint_transaction_unprotected(
            root,
            plan=plan,
            model=model,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            incremental_batches=incremental_batches,
            reference_batch=reference_batch,
            parameters=parameters,
            native_binary=native_binary,
        )
    except BaseException:
        for directory in (checkpoint, export):
            if directory.is_symlink():
                raise ValueError(
                    f"refusing to clean a symlinked failed transaction: {directory}"
                ) from None
            if directory.is_dir():
                shutil.rmtree(directory)
        if receipt.is_symlink():
            raise ValueError(f"refusing to clean a symlinked receipt: {receipt}") from None
        receipt.unlink(missing_ok=True)
        _fsync_directory(root / "checkpoints")
        _fsync_directory(root / "exports")
        _fsync_directory(root / "receipts")
        raise


def _checkpoint_step(path: Path) -> int:
    manifest = _strict_object(path / "manifest.json", label="incremental checkpoint manifest")
    optimizer = manifest.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("incremental checkpoint optimizer identity is missing")
    return _require_int(optimizer.get("step"), label="incremental checkpoint step")


def _completed_transaction_receipt(
    root: Path, checkpoint: Path, step: int, *, base_step: int
) -> dict[str, Any] | None:
    receipt_path = root / "receipts" / f"incremental-step-{step:08d}.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    receipt = _strict_json(receipt_path.read_bytes(), label="incremental checkpoint receipt")
    identity = receipt.get("checkpoint")
    export_identity = receipt.get("export")
    expected_export = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{step:08d}"
    if (
        receipt.get("schema") != INCREMENTAL_MLX_CHECKPOINT_RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("optimizer_step") != step
        or receipt.get("incremental_batches") != step - base_step
        or receipt.get("numeric_phase") != "quantisation_aware"
        or receipt.get("source_manifest_sha256") != sha256_file(root / "incremental-plan.json")
        or receipt.get("local_only") is not True
        or not isinstance(identity, dict)
        or identity.get("path") != str(checkpoint)
        or identity.get("manifest_sha256") != sha256_file(checkpoint / "manifest.json")
        or not isinstance(export_identity, dict)
        or export_identity.get("path") != str(expected_export)
        or not isinstance(export_identity.get("receipt"), dict)
        or not isinstance(receipt.get("calibration_validation"), dict)
        or not isinstance(receipt.get("quantisation_audit"), dict)
        or not isinstance(receipt.get("master_weight_audit"), dict)
    ):
        return None
    _regular_directory(expected_export, label="incremental export")
    return receipt


def _remove_uncommitted_transaction(root: Path, checkpoint: Path, step: int) -> None:
    expected = root / "checkpoints" / f"{DEFAULT_NET_ID}-mlx-step-{step:08d}"
    if checkpoint != expected or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("refusing to remove an unexpected incremental checkpoint")
    export = root / "exports" / f"{DEFAULT_NET_ID}-mlx-step-{step:08d}"
    receipt = root / "receipts" / f"incremental-step-{step:08d}.json"
    shutil.rmtree(checkpoint)
    if export.is_symlink() or receipt.is_symlink():
        raise ValueError("uncommitted transaction cleanup encountered a symlink")
    if export.is_dir():
        shutil.rmtree(export)
    receipt.unlink(missing_ok=True)
    for parent in (root / "checkpoints", root / "exports", root / "receipts"):
        _fsync_directory(parent)


def _prune_incremental_receipts(root: Path) -> None:
    checkpoint_prefix = f"{DEFAULT_NET_ID}-mlx-step-"
    retained_steps = {
        path.name.removeprefix(checkpoint_prefix)
        for path in (root / "checkpoints").iterdir()
        if path.name.startswith(checkpoint_prefix)
        and path.name.removeprefix(checkpoint_prefix).isdigit()
    }
    prefix = "incremental-step-"
    for path in (root / "receipts").iterdir():
        suffix = path.name.removeprefix(prefix).removesuffix(".json")
        if not (path.name.startswith(prefix) and path.name.endswith(".json") and suffix.isdigit()):
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"incremental receipt must be a regular file: {path}")
        if suffix not in retained_steps:
            path.unlink()
    _fsync_directory(root / "receipts")


def run_incremental_mlx(run_directory: Path) -> dict[str, object]:
    """Execute or exactly resume the deterministic post-100B replay schedule."""

    root = _regular_directory(run_directory, label="incremental MLX run")
    _validate_writable_layout(root)
    stop_requested = threading.Event()

    def signal_stop(signum: int, _frame: object) -> None:
        stop_requested.set()
        _log(root, "signal_stop_requested", signal=signum)

    install_signals = threading.current_thread() is threading.main_thread()
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    if install_signals:
        signal.signal(signal.SIGINT, signal_stop)
        signal.signal(signal.SIGTERM, signal_stop)
    try:
        with _exclusive_run_lock(root):
            _validate_writable_layout(root)
            # The immutable plan and mutable state must be loaded only after
            # the run lock is held; otherwise two executors can derive resume
            # coordinates from the same stale pre-lock state.
            plan = load_incremental_mlx_plan(root)
            state = load_incremental_mlx_state(root)
            execution = cast(dict[str, Any], plan["execution"])
            schedule = cast(dict[str, Any], plan["schedule"])
            base = cast(dict[str, Any], plan["base"])
            optimizer_contract = cast(dict[str, Any], plan["optimizer"])
            batch_size = int(execution["batch_size"])
            base_step = int(schedule["base_optimizer_step"])
            target_step = int(schedule["target_optimizer_step"])
            target_added = int(schedule["additional_optimizer_steps"])
            mix_cycle = cast(list[str], schedule["cycle"])
            learning_rate_value = float(optimizer_contract["learning_rate"])
            common = load_nagisa_plan(root)
            parameters = WrmLossParameters.from_training(cast(dict[str, Any], common["training"]))
            mlx_plan = _strict_object(
                root / "mlx-plan.json", label="incremental compatibility MLX plan"
            )
            native_binary = _native_binary_from_plan(root, mlx_plan)
            if int(execution["mlx_memory_limit_bytes"]) > 0:
                mx.set_memory_limit(int(execution["mlx_memory_limit_bytes"]))
            if int(execution["mlx_cache_limit_bytes"]) > 0:
                mx.set_cache_limit(int(execution["mlx_cache_limit_bytes"]))
            mx.reset_peak_memory()
            local_latest = _latest_mlx_checkpoint(root / "checkpoints")
            durable_step = int(state["durable_optimizer_step"])
            if local_latest is not None and _checkpoint_step(local_latest) > durable_step:
                orphan_step = _checkpoint_step(local_latest)
                recovered_receipt = _completed_transaction_receipt(
                    root, local_latest, orphan_step, base_step=base_step
                )
                if recovered_receipt is not None:
                    _log(
                        root,
                        "complete_transaction_recovered",
                        previous_durable_step=durable_step,
                        recovered_step=orphan_step,
                    )
                    durable_step = orphan_step
                    recovered_export = cast(dict[str, Any], recovered_receipt["export"])
                    _set_state(
                        root,
                        state,
                        durable_incremental_batches=orphan_step - base_step,
                        durable_optimizer_step=orphan_step,
                        latest_checkpoint=str(local_latest),
                        latest_export=recovered_export["path"],
                        last_calibration_validation=recovered_receipt["calibration_validation"],
                        last_quantisation_audit=recovered_receipt["quantisation_audit"],
                    )
                else:
                    _remove_uncommitted_transaction(root, local_latest, orphan_step)
                    _log(
                        root,
                        "uncommitted_transaction_removed",
                        durable_step=durable_step,
                        removed_step=orphan_step,
                    )
                    local_latest = _latest_mlx_checkpoint(root / "checkpoints")
            checkpoint = (
                _regular_directory(Path(cast(str, base["checkpoint"])), label="base checkpoint")
                if local_latest is None
                else local_latest
            )
            checkpoint_step = base_step if local_latest is None else _checkpoint_step(checkpoint)
            if checkpoint_step < durable_step or checkpoint_step > target_step:
                raise ValueError("checkpoint/state coordinate is not exactly resumable")
            if checkpoint_step > durable_step:
                _log(
                    root,
                    "orphan_checkpoint_recovered",
                    previous_durable_step=durable_step,
                    recovered_step=checkpoint_step,
                )
                durable_step = checkpoint_step
            if (
                local_latest is not None
                and _completed_transaction_receipt(
                    root, local_latest, checkpoint_step, base_step=base_step
                )
                is None
            ):
                raise ValueError("durable incremental checkpoint transaction receipt is invalid")
            model, optimizer, _ = load_mlx_checkpoint(checkpoint, learning_rate=learning_rate_value)
            if int(optimizer.state["step"].item()) != durable_step:
                raise ValueError("Ranger optimizer state does not match durable coordinate")
            added = durable_step - base_step
            if added < 0 or added > target_added:
                raise ValueError("incremental batch coordinate is outside the immutable plan")
            _set_state(
                root,
                state,
                status="running",
                pid=os.getpid(),
                durable_incremental_batches=added,
                observed_incremental_batches=added,
                durable_optimizer_step=durable_step,
                observed_optimizer_step=durable_step,
                latest_checkpoint=str(checkpoint),
                failure=None,
            )
            if added == target_added:
                _prune_mlx_checkpoints(root / "checkpoints", keep=2)
                _prune_mlx_exports(root / "exports", keep=2)
                _prune_incremental_receipts(root)
                _set_state(root, state, status="complete", pid=None)
                return incremental_mlx_status(root)
            if (root / "STOP").exists():
                _set_state(root, state, status="stopped_safely", pid=None)
                return incremental_mlx_status(root)
            step_fn = _compiled_training_step(
                model,
                optimizer,
                parameters=parameters,
                forward_mode="quantisation_aware",
            )
            learning_rate = mx.array(learning_rate_value, dtype=mx.float32)
            interval_started = time.perf_counter()
            interval_positions = 0
            interval_loss = 0.0
            last_batch: MlxNnueBatch | None = None
            while added < target_added:
                source_name = weighted_batch_source(mix_cycle, added)
                source = _source_mapping(plan, source_name)
                source_psv = _validate_source_stat(source, name=source_name)
                source_batches = int(source["batches"])
                consumed = _source_batches_before(mix_cycle, source_name, added)
                source_batch = consumed % source_batches
                run_batches = 1
                while (
                    added + run_batches < target_added
                    and weighted_batch_source(mix_cycle, added + run_batches) == source_name
                    and source_batch + run_batches < source_batches
                ):
                    run_batches += 1
                with NativeBatchStream(
                    native_binary,
                    psv=source_psv,
                    progress=root / "private" / "progress.bin",
                    start_record=source_batch * batch_size,
                    records=run_batches * batch_size,
                    batch_size=batch_size,
                    threads=int(execution["native_decode_threads"]),
                    prefetch=int(execution["native_prefetch_batches"]),
                ) as batches:
                    for batch in batches:
                        last_batch = batch
                        loss, output_mean, target_mean = _execute_training_batch(
                            step_fn,
                            learning_rate,
                            batch,
                            model=model,
                            optimizer=optimizer,
                        )
                        added += 1
                        optimizer_step = base_step + added
                        if int(optimizer.state["step"].item()) != optimizer_step:
                            raise ValueError("Ranger optimizer step skipped or reset")
                        interval_positions += batch.positions
                        interval_loss += loss * batch.positions
                        stop_now = stop_requested.is_set() or (root / "STOP").exists()
                        final = added == target_added
                        checkpoint_now = (
                            added % int(execution["checkpoint_batches"]) == 0 or stop_now or final
                        )
                        log_now = added % int(execution["log_batches"]) == 0 or checkpoint_now
                        if log_now:
                            elapsed = max(time.perf_counter() - interval_started, 1e-9)
                            speed = interval_positions / elapsed
                            metrics = _log(
                                root,
                                "training_progress",
                                source=source_name,
                                optimizer_step=optimizer_step,
                                incremental_batches=added,
                                incremental_presentations=added * batch_size,
                                target_incremental_presentations=target_added * batch_size,
                                loss=interval_loss / max(interval_positions, 1),
                                output_mean=output_mean,
                                target_probability_mean=target_mean,
                                positions_per_second=speed,
                                eta_seconds=(target_added - added) * batch_size / speed,
                                learning_rate=learning_rate_value,
                                numeric_phase="quantisation_aware",
                                peak_mlx_memory_bytes=mx.get_peak_memory(),
                            )
                            _set_state(
                                root,
                                state,
                                observed_incremental_batches=added,
                                observed_optimizer_step=optimizer_step,
                                last_metrics=metrics,
                            )
                            interval_started = time.perf_counter()
                            interval_positions = 0
                            interval_loss = 0.0
                        if checkpoint_now:
                            old_model, old_optimizer, old_step_fn = model, optimizer, step_fn
                            model, optimizer, checkpoint, export, receipt = _checkpoint_transaction(
                                root,
                                plan=plan,
                                model=old_model,
                                optimizer=old_optimizer,
                                optimizer_step=optimizer_step,
                                incremental_batches=added,
                                reference_batch=batch,
                                parameters=parameters,
                                native_binary=native_binary,
                            )
                            step_fn = _compiled_training_step(
                                model,
                                optimizer,
                                parameters=parameters,
                                forward_mode="quantisation_aware",
                            )
                            del old_step_fn, old_optimizer, old_model
                            gc.collect()
                            mx.clear_cache()
                            _set_state(
                                root,
                                state,
                                durable_incremental_batches=added,
                                durable_optimizer_step=optimizer_step,
                                latest_checkpoint=str(checkpoint),
                                latest_export=str(export),
                                last_calibration_validation=receipt["calibration_validation"],
                                last_quantisation_audit=receipt["quantisation_audit"],
                            )
                            # Prune only after the new transaction is durable in
                            # state.  A failed validation/export/audit must leave
                            # both earlier rollback generations untouched.
                            _prune_mlx_checkpoints(root / "checkpoints", keep=2)
                            _prune_mlx_exports(root / "exports", keep=2)
                            _prune_incremental_receipts(root)
                        if stop_now:
                            _set_state(root, state, status="stopped_safely", pid=None)
                            return incremental_mlx_status(root)
                _validate_source_stat(source, name=source_name)
                if last_batch is None:
                    raise ValueError("native incremental stream produced no batch")
            _set_state(root, state, status="complete", pid=None)
            return incremental_mlx_status(root)
    except BaseException as error:
        try:
            current = load_incremental_mlx_state(root)
            _set_state(
                root,
                current,
                status="failed",
                pid=None,
                failure={"kind": type(error).__name__, "message": str(error)},
            )
            _log(root, "training_failed", error_type=type(error).__name__, error=str(error))
        except Exception:
            pass
        raise
    finally:
        if install_signals:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)


def incremental_mlx_status(run_directory: Path) -> dict[str, object]:
    root = _regular_directory(run_directory, label="incremental MLX run")
    # Status/ETA must remain O(1) even when the immutable corpora are terabytes.
    # Full external SHA verification is deliberately reserved for prepare/run.
    plan = _load_incremental_mlx_plan(root, verify_external_content=False)
    state = load_incremental_mlx_state(root)
    schedule = cast(dict[str, Any], plan["schedule"])
    execution = cast(dict[str, Any], plan["execution"])
    batch_size = int(execution["batch_size"])
    target_batches = int(schedule["additional_optimizer_steps"])
    observed = int(state["observed_incremental_batches"])
    last_metrics = state.get("last_metrics")
    speed: float | None = None
    if isinstance(last_metrics, dict):
        candidate = last_metrics.get("positions_per_second")
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and candidate > 0
        ):
            speed = float(candidate)
    pid = state.get("pid")
    alive = False
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        try:
            process = psutil.Process(pid)
            alive = process.is_running() and any(
                "incremental" in part.casefold() for part in process.cmdline()
            )
        except (OSError, psutil.Error):
            alive = False
    checkpoints = sorted(
        str(path)
        for path in (root / "checkpoints").iterdir()
        if path.name.startswith(f"{DEFAULT_NET_ID}-mlx-step-")
    )
    exports = sorted(
        str(path)
        for path in (root / "exports").iterdir()
        if path.name.startswith(f"{DEFAULT_NET_ID}-mlx-step-")
    )
    remaining_positions = max(0, target_batches - observed) * batch_size
    return {
        "schema": "meteo-incremental-mlx-status-v1",
        "status": state.get("status"),
        "run_directory": str(root),
        "process_alive": alive,
        "pid": pid,
        "mode": "post_100b_value_only_qat_continuation",
        "numeric_phase": "quantisation_aware",
        "phase_reset_allowed": False,
        "policy_head": False,
        "base_optimizer_step": schedule["base_optimizer_step"],
        "durable_optimizer_step": state["durable_optimizer_step"],
        "observed_optimizer_step": state["observed_optimizer_step"],
        "durable_incremental_batches": state["durable_incremental_batches"],
        "observed_incremental_batches": observed,
        "target_incremental_batches": target_batches,
        "observed_incremental_presentations": observed * batch_size,
        "target_incremental_presentations": target_batches * batch_size,
        "completion_fraction": observed / target_batches,
        "mix_cycle": schedule["cycle"],
        "positions_per_second": speed,
        "eta_seconds": remaining_positions / speed if speed else None,
        "latest_checkpoint": state.get("latest_checkpoint"),
        "latest_export": state.get("latest_export"),
        "checkpoint_generations": checkpoints,
        "export_generations": exports,
        "last_metrics": last_metrics,
        "last_calibration_validation": state.get("last_calibration_validation"),
        "last_quantisation_audit": state.get("last_quantisation_audit"),
        "stop_requested": (root / "STOP").exists(),
        "failure": state.get("failure"),
    }
