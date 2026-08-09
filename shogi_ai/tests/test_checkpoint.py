from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simajilord_shogi.checkpoint import load_checkpoint, save_checkpoint
from simajilord_shogi.config import model_profile
from simajilord_shogi.model import PolicyValueResNet
from simajilord_shogi.rights_lineage import merge_rights_restriction_summaries


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
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.iterdir()
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite checkpoint"):
        save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=2)

    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in checkpoint.iterdir()
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
    nagisa_summary = merge_rights_restriction_summaries(
        [], teacher_sources=["nagisa-v3.1"]
    )

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
