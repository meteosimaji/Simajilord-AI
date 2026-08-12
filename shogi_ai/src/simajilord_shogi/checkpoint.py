"""Atomic model checkpoint metadata and MLX weight persistence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .branding import ENGINE_AUTHOR, ENGINE_DISPLAY_NAME, ENGINE_ROMANIZED_NAME
from .config import ModelConfig, model_config_payload
from .model import PolicyValueResNet
from .rights_lineage import (
    expected_lineage_rights_summary,
    lineage_has_teacher_evidence,
    validate_rights_restriction_summary,
)
from .trainer import TrainingState, TrainingTracePoint

FORMAT_VERSION = 1
TRAINING_STATE_SCHEMA = "meteo-exact-training-state-v1"
CHECKPOINT_IDENTITY_SCHEMA = "meteo-bounded-checkpoint-identity-v1"
CHECKPOINT_COMPLETE_SCHEMA = "meteo-checkpoint-complete-v1"
CHECKPOINT_COMPLETE_MARKER = "COMPLETE.json"
# Checkpoint metadata is control-plane data, not a place to copy prior
# checkpoints.  One MiB is ample for the model contract, direct-parent
# identity, rights summary, and exact-resume file receipts while making a
# recursively embedded ancestry fail before another large artifact is written.
MAX_CHECKPOINT_METADATA_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_strict_json(path: Path, *, label: str) -> object:
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON: {path}") from error


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_durable(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _checkpoint_file_records(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.name == CHECKPOINT_COMPLETE_MARKER:
            continue
        if child.is_symlink() or not child.is_file():
            raise ValueError(f"checkpoint contains a non-regular artifact: {child}")
        records.append(
            {
                "name": child.name,
                "sha256": _sha256_file(child),
                "bytes": child.stat().st_size,
            }
        )
    return records


def _complete_marker_payload(directory: Path, *, step: int) -> dict[str, object]:
    records = _checkpoint_file_records(directory)
    return {
        "schema": CHECKPOINT_COMPLETE_SCHEMA,
        "format_version": FORMAT_VERSION,
        "step": step,
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(cast(int, record["bytes"]) for record in records),
    }


def validate_checkpoint_complete(
    directory: Path,
    *,
    require_marker: bool = False,
) -> dict[str, object] | None:
    """Verify the durable completion marker when present.

    Markerless format-v1 checkpoints remain readable for backward compatibility.
    A checkpoint carrying a marker, however, is accepted only when the marker
    names every other top-level artifact and every byte count and hash matches.
    """

    directory = directory.expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"checkpoint must be a non-symlink directory: {directory}")
    marker_path = directory / CHECKPOINT_COMPLETE_MARKER
    if marker_path.is_symlink():
        raise ValueError(f"checkpoint completion marker must not be a symlink: {marker_path}")
    if not marker_path.exists():
        if require_marker:
            raise ValueError(f"checkpoint completion marker is missing: {marker_path}")
        return None
    if not marker_path.is_file():
        raise ValueError(f"checkpoint completion marker is not a regular file: {marker_path}")
    raw_marker = _read_strict_json(marker_path, label="checkpoint completion marker")
    marker = _mapping(raw_marker, label="checkpoint completion marker")
    expected_keys = {
        "schema",
        "format_version",
        "step",
        "files",
        "file_count",
        "total_bytes",
    }
    if set(marker) != expected_keys:
        raise ValueError("checkpoint completion marker fields do not match its schema")
    if marker["schema"] != CHECKPOINT_COMPLETE_SCHEMA:
        raise ValueError("unsupported checkpoint completion marker schema")
    if marker["format_version"] != FORMAT_VERSION:
        raise ValueError("checkpoint completion marker format version is unsupported")
    if not isinstance(marker["step"], int) or marker["step"] < 0:
        raise ValueError("checkpoint completion marker step must be non-negative")
    raw_records = marker["files"]
    if not isinstance(raw_records, list):
        raise ValueError("checkpoint completion marker files must be a list")
    records: list[dict[str, object]] = []
    names: set[str] = set()
    for raw_record in raw_records:
        record = _mapping(raw_record, label="checkpoint completion file record")
        if set(record) != {"name", "sha256", "bytes"}:
            raise ValueError("checkpoint completion file record fields are invalid")
        name = record["name"]
        digest = record["sha256"]
        byte_count = record["bytes"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name == CHECKPOINT_COMPLETE_MARKER
            or name in names
        ):
            raise ValueError("checkpoint completion marker contains an unsafe or duplicate name")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"checkpoint completion marker has an invalid SHA-256 for {name}")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"checkpoint completion marker has an invalid byte count for {name}")
        names.add(name)
        records.append(record)
    actual_records = _checkpoint_file_records(directory)
    if records != actual_records:
        raise ValueError("checkpoint completion marker does not match checkpoint artifacts")
    if marker["file_count"] != len(records):
        raise ValueError("checkpoint completion marker file count is inconsistent")
    if marker["total_bytes"] != sum(cast(int, record["bytes"]) for record in records):
        raise ValueError("checkpoint completion marker byte count is inconsistent")
    return cast(dict[str, object], marker)


def _read_checkpoint_metadata(directory: Path) -> dict[str, Any]:
    marker = validate_checkpoint_complete(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FileNotFoundError(metadata_path)
    metadata_bytes = metadata_path.stat().st_size
    if metadata_bytes > MAX_CHECKPOINT_METADATA_BYTES:
        raise ValueError(
            "checkpoint metadata exceeds the bounded metadata limit: "
            f"{metadata_bytes} > {MAX_CHECKPOINT_METADATA_BYTES} bytes"
        )
    value = _read_strict_json(metadata_path, label="checkpoint metadata")
    metadata = _mapping(value, label="checkpoint metadata")
    if marker is not None and (
        marker["format_version"] != metadata.get("format_version")
        or marker["step"] != metadata.get("step")
    ):
        raise ValueError("checkpoint completion marker and metadata disagree")
    return metadata


def bounded_checkpoint_provenance(directory: Path) -> dict[str, object]:
    """Return a constant-depth identity for one complete checkpoint.

    The result deliberately excludes paths, complete metadata, file listings,
    and parent lineage.  It can therefore be embedded as a direct-parent
    receipt without recursively copying every ancestor into every descendant.
    """

    supplied = directory.expanduser()
    if supplied.is_symlink():
        raise ValueError(f"checkpoint directory must not be a symlink: {supplied}")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"checkpoint does not exist or is not readable: {directory}") from error
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    entries = sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix())
    symlinks = [entry for entry in entries if entry.is_symlink()]
    if symlinks:
        raise ValueError(f"checkpoint must not contain symlinks: {symlinks[0]}")
    files = [entry for entry in entries if entry.is_file()]
    metadata_path = resolved / "metadata.json"
    weights_path = resolved / "weights.safetensors"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise ValueError("checkpoint requires metadata.json and weights.safetensors")
    metadata = _read_checkpoint_metadata(resolved)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    raw_model = metadata.get("model")
    if not isinstance(raw_model, dict):
        raise ValueError("checkpoint model metadata must be a JSON object")

    aggregate = hashlib.sha256()
    total_bytes = 0
    identities: dict[str, dict[str, object]] = {}
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
        total_bytes += byte_count
        if file_path in (metadata_path, weights_path):
            identities[file_path.name] = {
                "sha256": file_digest.hexdigest(),
                "bytes": byte_count,
            }

    load_digest = hashlib.sha256()
    for file_path in (metadata_path, weights_path):
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                load_digest.update(chunk)

    identity: dict[str, object] = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "format_version": FORMAT_VERSION,
        "step": int(metadata["step"]),
        "model": json.loads(json.dumps(raw_model, allow_nan=False, sort_keys=True)),
        "all_files_sha256": aggregate.hexdigest(),
        "load_checkpoint_sha256": load_digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "metadata": identities["metadata.json"],
        "weights": identities["weights.safetensors"],
    }
    raw_lineage = metadata.get("lineage")
    if isinstance(raw_lineage, dict):
        raw_summary = raw_lineage.get("rights_restriction_summary")
        if raw_summary is not None:
            if not isinstance(raw_summary, dict):
                raise ValueError("checkpoint rights restriction summary must be a JSON object")
            identity["rights_restriction_summary"] = validate_rights_restriction_summary(
                raw_summary
            )
        elif lineage_has_teacher_evidence(raw_lineage):
            identity["legacy_teacher_lineage_summary_missing"] = True
    encoded_identity = json.dumps(identity, allow_nan=False, sort_keys=True).encode("utf-8")
    if len(encoded_identity) > MAX_CHECKPOINT_METADATA_BYTES:
        raise ValueError("bounded checkpoint identity unexpectedly exceeds its size limit")
    return identity


def _encoded_training_state(
    state: TrainingState,
    trace: Sequence[TrainingTracePoint],
) -> tuple[dict[str, Any], dict[str, mx.array], bytes]:
    if not state.exact_resume_eligible:
        raise ValueError("refusing to persist an inexact training state as resumable")
    if state.model_step < 0 or state.optimizer_step < 0 or state.base_seed < 0:
        raise ValueError("training state steps and seed must be non-negative")
    if len(state.dataset_fingerprint) != 64:
        raise ValueError("training state dataset fingerprint must be SHA-256")

    flattened = cast(list[tuple[str, Any]], tree_flatten(state.optimizer_state))
    tensors: dict[str, mx.array] = {}
    tensor_specs: list[dict[str, object]] = []
    for name, value in sorted(flattened):
        if not name or name in tensors:
            raise ValueError(f"invalid or duplicate optimizer tensor name: {name!r}")
        if not isinstance(value, mx.array):
            raise TypeError(f"optimizer state leaf is not an MLX tensor: {name}")
        tensors[name] = value
        tensor_specs.append(
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
            }
        )
    if not tensors:
        raise ValueError("optimizer state must contain at least one tensor")

    trace_rows = [asdict(point) for point in trace]
    previous_optimizer_step = -1
    for row in trace_rows:
        optimizer_step = int(row["optimizer_step"])
        model_step = int(row["model_step"])
        if optimizer_step <= previous_optimizer_step:
            raise ValueError("training trace optimizer steps must be strictly increasing")
        if optimizer_step > state.optimizer_step or model_step > state.model_step:
            raise ValueError("training trace extends beyond its training state")
        previous_optimizer_step = optimizer_step
    trace_bytes = b"".join(
        (json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for row in trace_rows
    )
    numpy_rng_state = json.loads(json.dumps(state.numpy_rng_state, allow_nan=False, sort_keys=True))
    if not isinstance(numpy_rng_state, dict):
        raise TypeError("NumPy RNG state must be a JSON object")
    payload: dict[str, Any] = {
        "schema": TRAINING_STATE_SCHEMA,
        "model_step": state.model_step,
        "optimizer_step": state.optimizer_step,
        "base_seed": state.base_seed,
        "dataset_fingerprint": state.dataset_fingerprint,
        "configuration_fingerprint": state.configuration_fingerprint,
        "exact_resume_eligible": True,
        "optimizer": {
            "name": state.optimizer_name,
            "scheduler": state.scheduler_name,
            "tensor_specs": tensor_specs,
        },
        "rng": {
            "numpy_bit_generator": state.numpy_bit_generator,
            "numpy_state": numpy_rng_state,
            "mlx_step_seed_scheme": state.mlx_step_seed_scheme,
        },
        "runtime": {
            "mlx": state.mlx_version,
            "numpy": state.numpy_version,
        },
        "trace": {
            "schema": "meteo-training-trace-v1",
            "points": len(trace_rows),
        },
    }
    # Validate all scalar metadata before creating the temporary directory.
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload, tensors, trace_bytes


def save_checkpoint(
    model: PolicyValueResNet,
    directory: Path,
    *,
    step: int,
    lineage: Mapping[str, Any] | None = None,
    training_state: TrainingState | None = None,
    training_trace: Sequence[TrainingTracePoint] = (),
) -> Path:
    """Create one complete checkpoint without replacing an existing artifact."""

    directory = directory.expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {directory}")
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    encoded_state: tuple[dict[str, Any], dict[str, mx.array], bytes] | None = None
    if training_state is not None:
        if training_state.model_step != step:
            raise ValueError(
                "checkpoint step must match exact training state model step: "
                f"{step} != {training_state.model_step}"
            )
        encoded_state = _encoded_training_state(training_state, training_trace)
    elif training_trace:
        raise ValueError("training trace requires an exact training state")
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "model": model_config_payload(model.config),
        "backend": "mlx",
        "engine": {
            "display_name": ENGINE_DISPLAY_NAME,
            "romanized_name": ENGINE_ROMANIZED_NAME,
            "author": ENGINE_AUTHOR,
        },
    }
    if lineage is not None:
        # Round-trip once so checkpoint metadata cannot silently contain Path,
        # NaN, or another value that is not stable JSON provenance.
        encoded_lineage = json.dumps(lineage, allow_nan=False, sort_keys=True)
        decoded_lineage = json.loads(encoded_lineage)
        if not isinstance(decoded_lineage, dict):
            raise TypeError("checkpoint lineage must be a JSON object")
        raw_rights_summary = decoded_lineage.get("rights_restriction_summary")
        if raw_rights_summary is None and lineage_has_teacher_evidence(decoded_lineage):
            raise ValueError(
                "teacher-bearing training lineage requires a rights restriction summary"
            )
        if raw_rights_summary is not None:
            if not isinstance(raw_rights_summary, dict):
                raise ValueError("checkpoint rights restriction summary must be a JSON object")
            canonical_rights_summary = validate_rights_restriction_summary(raw_rights_summary)
            if decoded_lineage.get("schema") == "meteo-training-lineage-v2":
                expected_rights_summary = expected_lineage_rights_summary(decoded_lineage)
                if canonical_rights_summary != expected_rights_summary:
                    raise ValueError(
                        "checkpoint rights restriction summary does not cover its lineage"
                    )
            decoded_lineage["rights_restriction_summary"] = canonical_rights_summary
        metadata["lineage"] = decoded_lineage

    preflight_metadata = (
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if len(preflight_metadata) > MAX_CHECKPOINT_METADATA_BYTES:
        raise ValueError(
            "checkpoint metadata exceeds the bounded metadata limit: "
            f"{len(preflight_metadata)} > {MAX_CHECKPOINT_METADATA_BYTES} bytes"
        )

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=str(directory.parent)))
    published = False
    try:
        weights_path = temporary / "weights.safetensors"
        model.save_weights(str(weights_path))
        _fsync_file(weights_path)
        if encoded_state is not None:
            state_payload, optimizer_tensors, trace_bytes = encoded_state
            optimizer_path = temporary / "optimizer.safetensors"
            trace_path = temporary / "training_trace.jsonl"
            state_path = temporary / "training_state.json"
            mx.save_safetensors(str(optimizer_path), optimizer_tensors)
            _fsync_file(optimizer_path)
            _write_bytes_durable(trace_path, trace_bytes)
            state_payload["files"] = {
                "optimizer": {
                    "name": optimizer_path.name,
                    "sha256": _sha256_file(optimizer_path),
                    "bytes": optimizer_path.stat().st_size,
                },
                "trace": {
                    "name": trace_path.name,
                    "sha256": _sha256_file(trace_path),
                    "bytes": trace_path.stat().st_size,
                },
            }
            _write_bytes_durable(
                state_path,
                (
                    json.dumps(state_payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
            )
            metadata["training_state"] = {
                "schema": TRAINING_STATE_SCHEMA,
                "exact_resume": True,
                "name": state_path.name,
                "sha256": _sha256_file(state_path),
                "bytes": state_path.stat().st_size,
            }
        encoded_metadata = json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
        encoded_metadata_bytes = encoded_metadata.encode("utf-8")
        if len(encoded_metadata_bytes) > MAX_CHECKPOINT_METADATA_BYTES:
            raise ValueError(
                "checkpoint metadata exceeds the bounded metadata limit after state encoding: "
                f"{len(encoded_metadata_bytes)} > {MAX_CHECKPOINT_METADATA_BYTES} bytes"
            )
        _write_bytes_durable(temporary / "metadata.json", encoded_metadata_bytes)

        # A file existing is not sufficient evidence that a resumable checkpoint
        # was saved.  Exercise the same strict loader used by a future process
        # before publishing the directory or deleting any previous generation.
        _reloaded_model, reloaded_step, reloaded_state, _reloaded_trace = (
            load_checkpoint_with_training_state(temporary)
        )
        if reloaded_step != step:
            raise ValueError("staged checkpoint reloaded with the wrong model step")
        if encoded_state is not None:
            if training_state is None:
                raise AssertionError("encoded training state requires its source state")
            if reloaded_state is None:
                raise ValueError("staged checkpoint lost its exact training state")
            if (
                reloaded_state.model_step != training_state.model_step
                or reloaded_state.optimizer_step != training_state.optimizer_step
            ):
                raise ValueError("staged checkpoint reloaded with the wrong training state")
        elif reloaded_state is not None:
            raise ValueError("staged inference checkpoint unexpectedly contains training state")

        marker_payload = _complete_marker_payload(temporary, step=step)
        marker_bytes = (
            json.dumps(marker_payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_bytes_durable(temporary / CHECKPOINT_COMPLETE_MARKER, marker_bytes)
        validate_checkpoint_complete(temporary, require_marker=True)
        _fsync_directory(temporary)
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {directory}")
        temporary.rename(directory)
        published = True
        _fsync_directory(directory.parent)
        validate_checkpoint_complete(directory, require_marker=True)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if published:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory


def load_checkpoint(directory: Path) -> tuple[PolicyValueResNet, int]:
    directory = directory.expanduser().resolve()
    metadata = _read_checkpoint_metadata(directory)
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    config = ModelConfig(**metadata["model"])
    model = PolicyValueResNet(config)
    model.load_weights(str(directory / "weights.safetensors"), strict=True)
    mx.eval(model.parameters())
    return model, int(metadata["step"])


def _verified_child(directory: Path, record: Mapping[str, Any]) -> Path:
    name = record.get("name")
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("checkpoint state contains an invalid child filename")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"checkpoint state contains an invalid SHA-256 for {name}")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise ValueError(f"checkpoint state contains an invalid byte count for {name}")
    child = directory / name
    if child.is_symlink() or not child.is_file():
        raise FileNotFoundError(child)
    if child.stat().st_size != expected_bytes or _sha256_file(child) != expected_sha256:
        raise ValueError(f"checkpoint state file identity mismatch: {child}")
    return child


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def load_training_state(
    directory: Path,
) -> tuple[TrainingState, tuple[TrainingTracePoint, ...]] | None:
    """Load and hash-verify optional exact-resume state and step telemetry."""

    directory = directory.expanduser().resolve()
    metadata = _read_checkpoint_metadata(directory)
    summary_value = metadata.get("training_state")
    if summary_value is None:
        return None
    summary = _mapping(summary_value, label="checkpoint training-state summary")
    if summary.get("schema") != TRAINING_STATE_SCHEMA or summary.get("exact_resume") is not True:
        raise ValueError("unsupported or inexact checkpoint training state")
    state_path = _verified_child(directory, summary)
    payload = _mapping(
        _read_strict_json(state_path, label="checkpoint training state"),
        label="checkpoint training state",
    )
    if payload.get("schema") != TRAINING_STATE_SCHEMA:
        raise ValueError("unsupported checkpoint training-state format")
    if payload.get("exact_resume_eligible") is not True:
        raise ValueError("checkpoint training state is not exact-resume eligible")
    model_step = int(payload["model_step"])
    if model_step != int(metadata["step"]):
        raise ValueError("checkpoint model step and exact training state disagree")

    files = _mapping(payload.get("files"), label="checkpoint training-state files")
    optimizer_path = _verified_child(
        directory,
        _mapping(files.get("optimizer"), label="optimizer file identity"),
    )
    trace_path = _verified_child(
        directory,
        _mapping(files.get("trace"), label="trace file identity"),
    )
    optimizer_record = _mapping(payload.get("optimizer"), label="optimizer state")
    raw_specs = optimizer_record.get("tensor_specs")
    if not isinstance(raw_specs, list):
        raise ValueError("optimizer tensor_specs must be a list")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for raw_spec in raw_specs:
        spec = _mapping(raw_spec, label="optimizer tensor spec")
        name = spec.get("name")
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if (
            not isinstance(name, str)
            or not name
            or name in specs
            or not isinstance(shape, list)
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError("invalid or duplicate optimizer tensor spec")
        specs[name] = (tuple(shape), dtype)
    loaded_value = mx.load(str(optimizer_path))
    if not isinstance(loaded_value, dict):
        raise ValueError("optimizer safetensors must contain a tensor mapping")
    loaded = cast(dict[str, mx.array], loaded_value)
    if set(loaded) != set(specs):
        raise ValueError("optimizer safetensors keys do not match the state manifest")
    for name, tensor in loaded.items():
        observed = (tuple(int(size) for size in tensor.shape), str(tensor.dtype))
        if observed != specs[name]:
            raise ValueError(f"optimizer tensor shape/dtype mismatch: {name}")
    mx.eval(loaded)
    optimizer_state_value = tree_unflatten(sorted(loaded.items()))
    if not isinstance(optimizer_state_value, dict):
        raise ValueError("optimizer state did not reconstruct to a dictionary")

    trace_record = _mapping(payload.get("trace"), label="training trace record")
    expected_points = int(trace_record.get("points", -1))
    trace: list[TrainingTracePoint] = []
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank training trace row at line {line_number}")
            try:
                raw_row = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_json_object,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"training trace row {line_number} is not strict JSON"
                ) from error
            row = _mapping(raw_row, label=f"training trace row {line_number}")
            trace.append(TrainingTracePoint(**row))
    if len(trace) != expected_points:
        raise ValueError("training trace point count does not match the state manifest")

    rng = _mapping(payload.get("rng"), label="training RNG state")
    runtime = _mapping(payload.get("runtime"), label="training runtime state")
    state = TrainingState(
        model_step=model_step,
        optimizer_step=int(payload["optimizer_step"]),
        base_seed=int(payload["base_seed"]),
        dataset_fingerprint=str(payload["dataset_fingerprint"]),
        configuration_fingerprint=str(payload["configuration_fingerprint"]),
        exact_resume_eligible=True,
        optimizer_state=cast(dict[str, Any], optimizer_state_value),
        numpy_rng_state=_mapping(rng.get("numpy_state"), label="NumPy RNG state"),
        numpy_bit_generator=str(rng["numpy_bit_generator"]),
        optimizer_name=str(optimizer_record["name"]),
        scheduler_name=str(optimizer_record["scheduler"]),
        mlx_step_seed_scheme=str(rng["mlx_step_seed_scheme"]),
        mlx_version=str(runtime["mlx"]),
        numpy_version=str(runtime["numpy"]),
    )
    return state, tuple(trace)


def load_checkpoint_with_training_state(
    directory: Path,
) -> tuple[
    PolicyValueResNet,
    int,
    TrainingState | None,
    tuple[TrainingTracePoint, ...],
]:
    """Load model weights plus optional exact-resume state without weakening legacy loads."""

    model, step = load_checkpoint(directory)
    loaded = load_training_state(directory)
    if loaded is None:
        return model, step, None, ()
    state, trace = loaded
    return model, step, state, trace
