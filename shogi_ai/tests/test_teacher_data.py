from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

from simajilord_shogi.checkpoint import save_checkpoint
from simajilord_shogi.cli import main
from simajilord_shogi.config import model_profile
from simajilord_shogi.model import PolicyValueResNet
from simajilord_shogi.teacher_data import PackedSfenValueDataset


def _write_psv(path: Path) -> None:
    initial = Board()
    drop_board = Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
    first = initial.to_psv(
        mv=Move.from_usi("7g7f"),
        score=300,
        game_result=1,
        game_ply=1,
    )
    second = bytearray(
        drop_board.to_psv(
            mv=Move.from_usi("P*5e"),
            score=-600,
            game_result=0,
            game_ply=101,
        )
    )
    second_record = np.frombuffer(second, dtype=PackedSfenValue, count=1)
    second_record["game_result"][0] = -1
    path.write_bytes(first + second)


def test_psv_is_lazy_and_decodes_yaneuraou_drop_moves(tmp_path: Path) -> None:
    path = tmp_path / "teacher.psv"
    _write_psv(path)
    dataset = PackedSfenValueDataset(
        path,
        source_name="fixture",
        score_scale=600,
        evaluation_weight=0.5,
    )

    assert len(dataset) == 2
    assert dataset[0].teacher_best_move == "7g7f"
    assert dataset[0].value_target == 1.0
    assert dataset[0].teacher_value == pytest.approx((math.tanh(0.5) + 1.0) / 2)
    assert dataset[0].teacher_value_scale == 600.0
    assert dataset[1].teacher_best_move == "P*5e"
    assert dataset[1].value_target == -1.0
    assert dataset[1].teacher_source == "fixture"
    assert Board(dataset[1].sfen).is_legal_move(Move.from_usi("P*5e"))


def test_psv_stride_limit_and_corrupt_size_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "teacher.psv"
    _write_psv(path)

    selected = PackedSfenValueDataset(
        path,
        source_name="fixture",
        offset=1,
        stride=2,
        limit=1,
    )
    assert len(selected) == 1
    assert selected[0].teacher_best_move == "P*5e"
    assert selected[0].teacher_value == pytest.approx((math.tanh(-0.5) - 1.0) / 2)
    assert selected[0].teacher_value_scale == 1_200.0

    corrupt = tmp_path / "corrupt.psv"
    corrupt.write_bytes(b"not-a-psv")
    with pytest.raises(ValueError, match="multiple of 40"):
        PackedSfenValueDataset(corrupt, source_name="fixture")


def test_psv_rejects_non_side_to_move_results(tmp_path: Path) -> None:
    board = Board()
    path = tmp_path / "invalid-result.psv"
    path.write_bytes(
        board.to_psv(
            mv=Move.from_usi("7g7f"),
            score=0,
            game_result=2,
            game_ply=1,
        )
    )
    dataset = PackedSfenValueDataset(path, source_name="fixture")

    with pytest.raises(ValueError, match="game_result 2"):
        _ = dataset[0]


def test_policy_supervised_psv_rejects_value_only_move16_zero(tmp_path: Path) -> None:
    board = Board()
    raw = bytearray(
        board.to_psv(
            mv=Move.from_usi("7g7f"),
            score=10,
            game_result=0,
            game_ply=1,
        )
    )
    record = np.frombuffer(raw, dtype=PackedSfenValue, count=1)
    record["move"][0] = 0
    path = tmp_path / "value-only.psv"
    path.write_bytes(raw)
    dataset = PackedSfenValueDataset(path, source_name="value-only-fixture")

    with pytest.raises(ValueError, match="Move16=0 and is value-only"):
        _ = dataset[0]


def test_train_psv_records_ponanza_coefficient_and_tanh_denominator_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "initial"
    psv = tmp_path / "teacher.psv"
    output = tmp_path / "candidate"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=3)
    _write_psv(psv)

    assert (
        main(
            [
                "train-psv",
                str(checkpoint),
                str(psv),
                str(output),
                "--source-name",
                "fixture",
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--score-ponanza-coefficient",
                "756.0864962951762",
            ]
        )
        == 0
    )

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    hyperparameters = metadata["lineage"]["hyperparameters"]
    assert metadata["step"] == 4
    assert hyperparameters["score_ponanza_coefficient"] == 756.0864962951762
    assert hyperparameters["score_value_tanh_denominator"] == pytest.approx(
        1_512.1729925903524
    )
    assert hyperparameters["score_scale"] == pytest.approx(1_512.1729925903524)
    assert hyperparameters["score_scale_input_convention"] == (
        "explicit_ponanza_coefficient"
    )

    legacy_output = tmp_path / "legacy-scale600"
    assert (
        main(
            [
                "train-psv",
                str(checkpoint),
                str(psv),
                str(legacy_output),
                "--source-name",
                "fixture",
                "--steps",
                "1",
                "--batch-size",
                "1",
                "--score-scale",
                "600",
            ]
        )
        == 0
    )
    legacy_metadata = json.loads(
        (legacy_output / "metadata.json").read_text(encoding="utf-8")
    )
    legacy_hyperparameters = legacy_metadata["lineage"]["hyperparameters"]
    assert legacy_hyperparameters["score_ponanza_coefficient"] == 300.0
    assert legacy_hyperparameters["score_value_tanh_denominator"] == 600.0
    assert legacy_hyperparameters["score_scale"] == 600.0
    assert legacy_hyperparameters["score_scale_input_convention"] == (
        "explicit_legacy_tanh_denominator"
    )
