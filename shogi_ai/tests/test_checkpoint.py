from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import pytest

import simajilord_shogi.checkpoint as checkpoint_module
from simajilord_shogi.checkpoint import (
    CHECKPOINT_COMPLETE_MARKER,
    CHECKPOINT_COMPLETE_SCHEMA,
    CHECKPOINT_IDENTITY_SCHEMA,
    MAX_CHECKPOINT_METADATA_BYTES,
    bounded_checkpoint_provenance,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_complete,
)
from simajilord_shogi.config import model_profile
from simajilord_shogi.model import PolicyValueResNet
from simajilord_shogi.rights_lineage import merge_rights_restriction_summaries
from simajilord_shogi.trainer import TrainingState


def test_checkpoint_writes_and_verifies_complete_marker(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"

    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=4)

    marker = validate_checkpoint_complete(checkpoint, require_marker=True)
    assert marker is not None
    assert marker["schema"] == CHECKPOINT_COMPLETE_SCHEMA
    assert marker["step"] == 4
    records = {record["name"]: record for record in marker["files"]}
    assert set(records) == {"metadata.json", "weights.safetensors"}
    for name, record in records.items():
        path = checkpoint / name
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_rejects_bytes_changed_after_completion(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=4)
    weights = checkpoint / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="does not match checkpoint artifacts"):
        load_checkpoint(checkpoint)


def test_markerless_format_v1_checkpoint_remains_loadable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=4)
    (checkpoint / CHECKPOINT_COMPLETE_MARKER).unlink()

    _model, step = load_checkpoint(checkpoint)

    assert step == 4
    assert validate_checkpoint_complete(checkpoint) is None


def test_optimizer_save_failure_leaves_no_checkpoint_or_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    state = TrainingState(
        model_step=4,
        optimizer_step=2,
        base_seed=1,
        dataset_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
        exact_resume_eligible=True,
        optimizer_state={"state": mx.array([1.0])},
        numpy_rng_state={"state": 1},
        numpy_bit_generator="fixture",
        mlx_version="fixture",
        numpy_version="fixture",
    )
    original_save_safetensors = checkpoint_module.mx.save_safetensors

    def fail_optimizer(path: str, tensors: object) -> None:
        if Path(path).name == "optimizer.safetensors":
            raise RuntimeError("injected optimizer write failure")
        original_save_safetensors(path, tensors)

    monkeypatch.setattr(checkpoint_module.mx, "save_safetensors", fail_optimizer)

    with pytest.raises(RuntimeError, match="injected optimizer write failure"):
        save_checkpoint(
            PolicyValueResNet(model_profile("smoke")),
            checkpoint,
            step=4,
            training_state=state,
        )

    assert not checkpoint.exists()
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_checkpoint_round_trips_optional_training_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    lineage = {
        "schema": "meteo-training-lineage-v1",
        "parent_checkpoint": {"sha256": "a" * 64},
        "exact_resume": False,
    }

    save_checkpoint(
        PolicyValueResNet(model_profile("smoke")),
        checkpoint,
        step=12,
        lineage=lineage,
    )

    _model, step = load_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert step == 12
    assert metadata["lineage"] == lineage


def test_checkpoint_rejects_non_finite_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(ValueError, match="Out of range float"):
        save_checkpoint(
            PolicyValueResNet(model_profile("smoke")),
            checkpoint,
            step=0,
            lineage={"loss": float("nan")},
        )
    assert not checkpoint.exists()
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_checkpoint_refuses_to_replace_existing_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=1)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint.iterdir()
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite checkpoint"):
        save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=2)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint.iterdir()
    }
    assert after == before
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_checkpoint_requires_summary_for_teacher_bearing_v2_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(ValueError, match="requires a rights restriction summary"):
        save_checkpoint(
            PolicyValueResNet(model_profile("smoke")),
            checkpoint,
            step=1,
            lineage={
                "schema": "meteo-training-lineage-v2",
                "teacher_sources": ["nagisa-v3.1"],
            },
        )
    assert not checkpoint.exists()


def test_checkpoint_rejects_summary_that_does_not_cover_teacher_sources(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    nagisa_summary = merge_rights_restriction_summaries([], teacher_sources=["nagisa-v3.1"])

    with pytest.raises(ValueError, match="does not cover its lineage"):
        save_checkpoint(
            PolicyValueResNet(model_profile("smoke")),
            checkpoint,
            step=1,
            lineage={
                "schema": "meteo-training-lineage-v2",
                "teacher_sources": ["suisho11plus-wcsc36-20260525-local"],
                "rights_restriction_summary": nagisa_summary,
            },
        )
    assert not checkpoint.exists()


def test_checkpoint_rejects_oversized_metadata_before_creating_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    oversized_lineage = {"payload": "x" * MAX_CHECKPOINT_METADATA_BYTES}

    with pytest.raises(ValueError, match="bounded metadata limit"):
        save_checkpoint(
            PolicyValueResNet(model_profile("smoke")),
            checkpoint,
            step=1,
            lineage=oversized_lineage,
        )

    assert not checkpoint.exists()
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_bounded_checkpoint_provenance_has_constant_depth(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    save_checkpoint(
        PolicyValueResNet(model_profile("smoke")),
        parent,
        step=3,
        lineage={
            "schema": "meteo-training-lineage-v1",
            "parent_checkpoint": {
                "metadata": {"lineage": {"metadata": {"ancestor": "must-not-copy"}}}
            },
        },
    )

    identity = bounded_checkpoint_provenance(parent)

    assert identity["schema"] == CHECKPOINT_IDENTITY_SCHEMA
    assert identity["step"] == 3
    assert "path" not in identity
    assert "lineage" not in identity
    assert identity["metadata"]["bytes"] == (parent / "metadata.json").stat().st_size
    assert "must-not-copy" not in json.dumps(identity, sort_keys=True)
