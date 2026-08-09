"""Atomic model checkpoint metadata and MLX weight persistence."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .branding import ENGINE_AUTHOR, ENGINE_DISPLAY_NAME, ENGINE_ROMANIZED_NAME
from .config import ModelConfig
from .model import PolicyValueResNet
from .rights_lineage import (
    expected_lineage_rights_summary,
    lineage_has_teacher_evidence,
    validate_rights_restriction_summary,
)
from .trainer import TrainingState, TrainingTracePoint

FORMAT_VERSION = 1
TRAINING_STATE_SCHEMA = "meteo-exact-training-state-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        (
            json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in trace_rows
    )
    numpy_rng_state = json.loads(
        json.dumps(state.numpy_rng_state, allow_nan=False, sort_keys=True)
    )
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
        "model": asdict(model.config),
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
            canonical_rights_summary = validate_rights_restriction_summary(
                raw_rights_summary
            )
            if decoded_lineage.get("schema") == "meteo-training-lineage-v2":
                expected_rights_summary = expected_lineage_rights_summary(decoded_lineage)
                if canonical_rights_summary != expected_rights_summary:
                    raise ValueError(
                        "checkpoint rights restriction summary does not cover its lineage"
                    )
            decoded_lineage["rights_restriction_summary"] = canonical_rights_summary
        metadata["lineage"] = decoded_lineage

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=str(directory.parent))
    )
    try:
        model.save_weights(str(temporary / "weights.safetensors"))
        if encoded_state is not None:
            state_payload, optimizer_tensors, trace_bytes = encoded_state
            optimizer_path = temporary / "optimizer.safetensors"
            trace_path = temporary / "training_trace.jsonl"
            state_path = temporary / "training_state.json"
            mx.save_safetensors(str(optimizer_path), optimizer_tensors)
            trace_path.write_bytes(trace_bytes)
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
            state_path.write_text(
                json.dumps(state_payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            metadata["training_state"] = {
                "schema": TRAINING_STATE_SCHEMA,
                "exact_resume": True,
                "name": state_path.name,
                "sha256": _sha256_file(state_path),
                "bytes": state_path.stat().st_size,
            }
        encoded_metadata = (
            json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "metadata.json").write_text(encoded_metadata, encoding="utf-8")
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {directory}")
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def load_checkpoint(directory: Path) -> tuple[PolicyValueResNet, int]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
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
    if not child.is_file():
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
    metadata = _mapping(
        json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
        label="checkpoint metadata",
    )
    summary_value = metadata.get("training_state")
    if summary_value is None:
        return None
    summary = _mapping(summary_value, label="checkpoint training-state summary")
    if summary.get("schema") != TRAINING_STATE_SCHEMA or summary.get("exact_resume") is not True:
        raise ValueError("unsupported or inexact checkpoint training state")
    state_path = _verified_child(directory, summary)
    payload = _mapping(
        json.loads(state_path.read_text(encoding="utf-8")),
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
            row = _mapping(json.loads(line), label=f"training trace row {line_number}")
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
