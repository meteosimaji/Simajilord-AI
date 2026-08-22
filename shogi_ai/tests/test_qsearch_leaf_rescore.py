from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

from simajilord_shogi.artifact_provenance import sha256_file
from simajilord_shogi.distillation_targets import NAGISA_SCORER_ID
from simajilord_shogi.external_usi import (
    ExternalUsiTeacher,
    UsiOptionValueVerification,
)
from simajilord_shogi.incremental_value_replay import load_scalar_value_labels
from simajilord_shogi.model_rights import model_rights
from simajilord_shogi.production_score_matrix import (
    CanonicalScorerIdentity,
    ExternalUsiProductionScorer,
)
from simajilord_shogi.qsearch_leaf import QSEARCH_LEAF_RECEIPT_SCHEMA
from simajilord_shogi.qsearch_leaf_rescore import (
    BOARD_ONLY_CONTEXT,
    QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA,
    rescore_qsearch_leaves,
)


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _fake_engine(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys

options = {"BookFile": "no_book", "Hash": "64", "MultiPV": "1", "Threads": "1"}
position = ""
for raw in sys.stdin:
    line = raw.rstrip("\\r\\n")
    if line == "usi":
        print("id name LeafRescoreFake")
        print("option name MultiPV type spin default 1 min 1 max 32")
        print("option name Threads type spin default 1 min 1 max 8")
        print("option name Hash type spin default 64 min 1 max 4096")
        print("option name BookFile type string default no_book")
        print("usiok", flush=True)
    elif line.startswith("setoption name "):
        payload = line.removeprefix("setoption name ")
        name, separator, value = payload.partition(" value ")
        options[name] = value if separator else ""
    elif line == "isready":
        print("readyok", flush=True)
    elif line.startswith("getoption "):
        name = line.removeprefix("getoption ")
        print(f"Options[{name}] = {options[name]}", flush=True)
    elif line.startswith("position "):
        position = line
    elif line.startswith("go nodes "):
        nodes = int(line.split()[2])
        move = "7g7f" if " b " in position else "3c3d"
        score = "score cp 123" if " b " in position else "score mate 3"
        print(
            f"info depth 9 seldepth 13 nodes {nodes} time 1 nps {nodes * 1000} "
            f"{score} pv {move}"
        )
        print(f"bestmove {move}", flush=True)
    elif line == "quit":
        break
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _qsearch_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "qsearch"
    directory.mkdir()
    root = Board()
    child = root.copy()
    child.apply_move(Move.from_usi("7g7f"))
    records = np.zeros(2, dtype=PackedSfenValue)
    for index, (board, score, ply) in enumerate(((root, 999, 1), (child, -888, 2))):
        records[index]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
        records[index]["score"] = score
        records[index]["move"] = 0
        records[index]["game_ply"] = ply
        records[index]["game_result"] = 0
    leaves = directory / "leaves.psv"
    leaves.write_bytes(records.tobytes())
    receipt = {
        "schema": QSEARCH_LEAF_RECEIPT_SCHEMA,
        "source": {"bytes": 80, "sha256": "0" * 64, "records": 2},
        "engine": {"sha256": "1" * 64, "profile": {}},
        "output": {
            "file": "leaves.psv",
            "bytes": leaves.stat().st_size,
            "sha256": sha256_file(leaves),
            "records": 2,
        },
        "transcript": {
            "file": "transcript.txt",
            "sha256": "2" * 64,
            "done_line": "done",
            "done_fields": {},
        },
        "score_was_recomputed_at_leaf": False,
        "eligible_for_value_training": False,
        "required_next_stage": "single_anchor_rescore_every_leaf",
        "complete": True,
    }
    (directory / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def _anchor(tmp_path: Path) -> tuple[ExternalUsiTeacher, ExternalUsiProductionScorer]:
    engine_path = tmp_path / "anchor.py"
    _fake_engine(engine_path)
    policy = model_rights(NAGISA_SCORER_ID).teacher_policy()
    engine = ExternalUsiTeacher(
        [str(engine_path)],
        policy,
        nodes=64,
        multipv=1,
        options={"BookFile": "no_book", "Hash": 64, "Threads": 1},
        timeout_seconds=5.0,
        training_use=True,
        working_directory=tmp_path,
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
    )
    engine.start()
    identity = CanonicalScorerIdentity(
        scorer_id=NAGISA_SCORER_ID,
        engine_sha256=sha256_file(engine_path),
        evaluation_artifacts_sha256=hashlib.sha256(b"eval").hexdigest(),
        options=(
            ("BookFile", "no_book"),
            ("Hash", "64"),
            ("MultiPV", "1"),
            ("Threads", "1"),
        ),
        startup_transcript_sha256=_json_sha(engine.startup_provenance.to_dict()),
        parser_sha256=hashlib.sha256(b"parser").hexdigest(),
        scorer_code_sha256=hashlib.sha256(b"scorer").hexdigest(),
        calibration_sha256=hashlib.sha256(b"calibration").hexdigest(),
        threads=1,
        hash_mb=64,
        multipv=1,
        book_enabled=False,
    )
    return engine, ExternalUsiProductionScorer(engine, identity)


def test_rescore_emits_only_exact_cp_and_queues_mate(tmp_path: Path) -> None:
    qsearch = _qsearch_directory(tmp_path)
    engine, anchor = _anchor(tmp_path)
    output = tmp_path / "rescored"
    try:
        receipt = rescore_qsearch_leaves(
            qsearch,
            anchor=anchor,
            requested_nodes=64,
            output_directory=output,
        )
    finally:
        engine.close()

    assert receipt["schema"] == QSEARCH_LEAF_RESCORE_RECEIPT_SCHEMA
    assert receipt["source_record_count"] == 2
    assert receipt["unique_board_count"] == 2
    assert receipt["labels"]["rows"] == 1
    assert receipt["unresolved_queue"]["rows"] == 1
    assert receipt["unresolved_queue"]["reason_counts"] == {"mate_requires_separate_proof": 1}
    assert receipt["old_score_used_as_target"] is False
    assert receipt["position_context"] == BOARD_ONLY_CONTEXT
    assert receipt["local_only"] is True
    assert receipt["publication_allowed"] is False

    labels = load_scalar_value_labels(output / "labels.jsonl")
    assert len(labels) == 1
    assert labels[0].score_cp == 123
    assert labels[0].score_cp not in {999, -888}
    assert labels[0].qsearch_leaf_rescored is True
    assert labels[0].history_dependent is False
    search_rows = [
        json.loads(line) for line in (output / "searches.jsonl").read_text().splitlines()
    ]
    source_row = next(
        row
        for row in search_rows
        if row["search_receipt_sha256"] == labels[0].source_receipt_sha256
    )
    unsigned = dict(source_row)
    unsigned.pop("search_receipt_sha256")
    assert _json_sha(unsigned) == labels[0].source_receipt_sha256
    assert source_row["old_score_used_as_target"] is False
    unresolved = [
        json.loads(line) for line in (output / "unresolved.jsonl").read_text().splitlines()
    ]
    assert unresolved[0]["variation"]["score_kind"] == "mate"

    with pytest.raises(FileExistsError, match="overwrite"):
        rescore_qsearch_leaves(
            qsearch,
            anchor=anchor,
            requested_nodes=64,
            output_directory=output,
        )


def test_rescore_rejects_conversion_marked_training_eligible(tmp_path: Path) -> None:
    qsearch = _qsearch_directory(tmp_path)
    receipt_path = qsearch / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["eligible_for_value_training"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    engine, anchor = _anchor(tmp_path)
    try:
        with pytest.raises(ValueError, match="ineligible"):
            rescore_qsearch_leaves(
                qsearch,
                anchor=anchor,
                requested_nodes=64,
                output_directory=tmp_path / "rescored",
            )
    finally:
        engine.close()


def test_rescore_requires_anchor_fixed_node_contract(tmp_path: Path) -> None:
    qsearch = _qsearch_directory(tmp_path)
    engine, anchor = _anchor(tmp_path)
    try:
        with pytest.raises(ValueError, match="same fixed node budget"):
            rescore_qsearch_leaves(
                qsearch,
                anchor=anchor,
                requested_nodes=65,
                output_directory=tmp_path / "rescored",
            )
    finally:
        engine.close()


def test_rescore_rejects_symlinked_output_parent_before_search(tmp_path: Path) -> None:
    qsearch = _qsearch_directory(tmp_path)
    engine, anchor = _anchor(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="must not traverse a symlink"):
            rescore_qsearch_leaves(
                qsearch,
                anchor=anchor,
                requested_nodes=64,
                output_directory=linked_parent / "rescored",
            )
    finally:
        engine.close()
    assert not (real_parent / "rescored").exists()
