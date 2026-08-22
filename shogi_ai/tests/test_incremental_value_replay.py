from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

from simajilord_shogi.distillation_targets import SOUJOU_TSEC7_SCORER_ID
from simajilord_shogi.incremental_value_replay import (
    INCREMENTAL_VALUE_REPLAY_SCHEMA,
    IncrementalValueSplit,
    ScalarLabelKind,
    ScalarValueLabel,
    build_incremental_value_replay,
    load_scalar_value_labels,
    validate_incremental_split_set,
)
from simajilord_shogi.nnue_runner import main as nnue_main
from simajilord_shogi.nnue_training import probe_value_only_psv


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _labels() -> tuple[ScalarValueLabel, ScalarValueLabel]:
    first = ScalarValueLabel(
        sfen=Board().to_sfen(),
        score_cp=123,
        game_ply=1,
        game_result=0,
        game_result_known=False,
        kind=ScalarLabelKind.ANCHOR_SEARCH_EXACT,
        scorer_id=SOUJOU_TSEC7_SCORER_ID,
        requested_nodes=1_000_000,
        source_receipt_sha256=_digest("matrix"),
        qsearch_leaf_rescored=True,
    )
    child = Board()
    child.apply_move(Move.from_usi("7g7f"))
    second = ScalarValueLabel(
        sfen=child.to_sfen(),
        score_cp=-32_000,
        game_ply=2,
        game_result=-1,
        game_result_known=True,
        kind=ScalarLabelKind.PROVEN_MATE,
        scorer_id=None,
        requested_nodes=None,
        source_receipt_sha256=_digest("proof"),
        qsearch_leaf_rescored=False,
    )
    return first, second


def test_incremental_replay_is_move16_zero_exact_and_receipted(tmp_path: Path) -> None:
    labels = _labels()
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=_digest("games"),
        position_ids=tuple(label.position_id for label in labels),
    )

    receipt = build_incremental_value_replay(
        labels,
        split=split,
        output_directory=tmp_path / "value-replay",
    )

    assert receipt["schema"] == INCREMENTAL_VALUE_REPLAY_SCHEMA
    assert receipt["cross_teacher_value_average"] is False
    assert receipt["bounds_written_as_point_targets"] is False
    assert receipt["publication_allowed"] is False
    psv = tmp_path / "value-replay" / "train.psv"
    audit = probe_value_only_psv(psv, board_samples=2)
    assert audit["records"] == 2
    assert audit["policy_targets"] == 0
    records = np.memmap(psv, dtype=PackedSfenValue, mode="r")
    assert set(int(value) for value in records["move"]) == {0}
    assert sorted(int(value) for value in records["score"]) == [-32_000, 123]
    rows = [
        json.loads(line)
        for line in (tmp_path / "value-replay" / "labels.jsonl").read_text().splitlines()
    ]
    assert {row["position_id"] for row in rows} == set(split.position_ids)


def test_incremental_replay_rejects_bounds_history_and_post_label_split(tmp_path: Path) -> None:
    label = _labels()[0]
    with pytest.raises(ValueError, match="history-dependent"):
        replace(label, history_dependent=True)
    with pytest.raises(ValueError, match="qsearch leaf"):
        replace(label, qsearch_leaf_rescored=False)
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=_digest("games"),
        position_ids=(_digest("some-other-position"),),
    )
    with pytest.raises(ValueError, match="absent from the pre-labelled"):
        build_incremental_value_replay([label], split=split, output_directory=tmp_path / "bad")


def test_incremental_replay_is_create_only_and_train_only(tmp_path: Path) -> None:
    labels = _labels()
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=_digest("games"),
        position_ids=tuple(label.position_id for label in labels),
    )
    output = tmp_path / "value-replay"
    build_incremental_value_replay(labels, split=split, output_directory=output)
    with pytest.raises(FileExistsError, match="overwrite"):
        build_incremental_value_replay(labels, split=split, output_directory=output)
    heldout = IncrementalValueSplit(
        split_id="held_out_test",
        source_games_sha256=_digest("games"),
        position_ids=tuple(label.position_id for label in labels),
    )
    with pytest.raises(ValueError, match="only the train split"):
        build_incremental_value_replay(labels, split=heldout, output_directory=tmp_path / "heldout")


def test_scalar_label_domains_fail_closed() -> None:
    label = _labels()[0]
    with pytest.raises(ValueError, match="canonical scorer"):
        replace(label, scorer_id="unknown-teacher")
    with pytest.raises(ValueError, match="proven-mate"):
        replace(label, score_cp=32_000)
    with pytest.raises(ValueError, match="unknown WDL"):
        replace(label, game_result=1, game_result_known=False)


def test_incremental_value_replay_cli_replays_strict_label_and_split_receipts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = _labels()
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=_digest("games"),
        position_ids=tuple(label.position_id for label in labels),
    )
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(label.to_input_row()) + "\n" for label in labels),
        encoding="utf-8",
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split.to_dict()), encoding="utf-8")

    assert (
        nnue_main(
            [
                "build-incremental-psv",
                str(labels_path),
                str(split_path),
                str(tmp_path / "output"),
            ]
        )
        == 0
    )

    result = json.loads(capsys.readouterr().out)
    assert result["records"] == 2
    assert result["policy_targets"] == 0


def test_incremental_split_set_requires_three_disjoint_prelabelled_sets() -> None:
    source = _digest("games")
    splits = tuple(
        IncrementalValueSplit(
            split_id=split_id,
            source_games_sha256=source,
            position_ids=(_digest(split_id),),
        )
        for split_id in ("train", "calibration", "held_out_test")
    )

    receipt = validate_incremental_split_set(splits)

    assert receipt["verified_zero_cross_split_overlap"] is True
    assert all(not overlap for overlap in receipt["normalized_position_overlap"].values())
    with pytest.raises(ValueError, match="overlap"):
        validate_incremental_split_set(
            (splits[0], splits[1], replace(splits[2], position_ids=splits[0].position_ids))
        )


def test_scalar_label_json_rejects_casefold_key_collisions(tmp_path: Path) -> None:
    label = _labels()[0].to_input_row()
    payload = json.dumps(label)[:-1] + ',"SFEN":"duplicate"}\n'
    path = tmp_path / "labels.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="case-insensitive"):
        load_scalar_value_labels(path)
