from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.numpy import PackedSfenValue

import simajilord_shogi.external_usi as external_usi_module
import simajilord_shogi.production_score_matrix as production_score_matrix_module
from simajilord_shogi.artifact_provenance import sha256_file
from simajilord_shogi.distillation_targets import CANONICAL_SCORER_IDS, NAGISA_SCORER_ID
from simajilord_shogi.incremental_value_replay import IncrementalValueSplit
from simajilord_shogi.model_rights import MODEL_RIGHTS
from simajilord_shogi.nnue_game_generation import NNUE_GAME_RECEIPT_SCHEMA
from simajilord_shogi.nnue_game_runner import (
    NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
    NNUE_GAME_JSONL_SCHEMA,
)
from simajilord_shogi.post_bootstrap_runner import (
    GAME_POSITION_BUNDLE_SCHEMA,
    GAME_POSITION_SOURCE_SCHEMA,
    QSEARCH_RESCORE_RUNNER_CONFIG_SCHEMA,
    build_incremental_replay_from_rescore,
    extract_game_position_source,
    load_qsearch_anchor_config,
    main,
    run_qsearch_rescore,
    validate_registered_model_usage_receipts,
)
from simajilord_shogi.qsearch_leaf import QSEARCH_LEAF_RECEIPT_SCHEMA
from simajilord_shogi.teacher_data import PSV_RECORD_BYTES


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _artifact_set(paths: list[Path]) -> tuple[list[dict[str, str]], str]:
    rows = [_artifact(path) for path in sorted(paths, key=lambda item: str(item))]
    return rows, _json_sha({"artifacts": rows})


def _fake_anchor(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys

options = {"BookFile": "no_book", "Hash": "64", "MultiPV": "1", "Threads": "1"}
position = ""
for raw in sys.stdin:
    line = raw.rstrip("\\r\\n")
    if line == "usi":
        print("id name PostBootstrapAnchor")
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
        print(
            f"info depth 9 seldepth 13 nodes {nodes} time 1 nps {nodes * 1000} "
            f"score cp 123 pv {move}"
        )
        print(f"bestmove {move}", flush=True)
    elif line == "quit":
        break
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _qsearch_conversion(tmp_path: Path) -> Path:
    root = tmp_path / "qsearch"
    root.mkdir()
    board = Board()
    records = np.zeros(1, dtype=PackedSfenValue)
    records[0]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
    records[0]["score"] = 999
    records[0]["move"] = 0
    records[0]["game_ply"] = 1
    records[0]["game_result"] = 0
    leaves = root / "leaves.psv"
    leaves.write_bytes(records.tobytes())
    receipt = {
        "schema": QSEARCH_LEAF_RECEIPT_SCHEMA,
        "source": {"bytes": PSV_RECORD_BYTES, "sha256": "0" * 64, "records": 1},
        "engine": {"sha256": "1" * 64, "profile": {}},
        "output": {
            "file": "leaves.psv",
            "bytes": leaves.stat().st_size,
            "sha256": sha256_file(leaves),
            "records": 1,
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
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return root


def _qsearch_config(tmp_path: Path) -> Path:
    engine = tmp_path / "anchor.py"
    _fake_anchor(engine)
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"scale":1200}\n', encoding="utf-8")
    evaluation, evaluation_sha = _artifact_set([engine])
    anchor = {
        "scorer_id": NAGISA_SCORER_ID,
        "engine": {"path": str(engine), "sha256": sha256_file(engine), "arguments": []},
        "working_directory": str(tmp_path),
        "evaluation_artifacts": evaluation,
        "evaluation_artifacts_sha256": evaluation_sha,
        "options": [
            {"name": "BookFile", "value": "no_book"},
            {"name": "Hash", "value": 64},
            {"name": "Threads", "value": 1},
        ],
        "hash_option_name": "Hash",
        "proposal_multipv": 3,
        "book_disable": {
            "enabled": False,
            "option_name": "BookFile",
            "option_value": "no_book",
        },
        "option_value_verification": "yaneuraou_getoption",
        "parser_artifact": _artifact(Path(external_usi_module.__file__).resolve()),
        "scorer_code_artifact": _artifact(Path(production_score_matrix_module.__file__).resolve()),
        "calibration_artifact": _artifact(calibration),
        "timeout_seconds": 5.0,
        "value_scale": 1200.0,
        "expected_fatal_startup_diagnostics": [],
    }
    payload = {
        "schema": QSEARCH_RESCORE_RUNNER_CONFIG_SCHEMA,
        "output_scope": "private_local_only",
        "limited_local_authorization": {
            "authorized": True,
            "scope": "private_local_distillation_only",
            "rights_ids": [],
        },
        "requested_nodes": 64,
        "anchor": anchor,
    }
    path = tmp_path / "rescore-config.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_rescore_cli_and_verified_replay_are_end_to_end_create_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _qsearch_config(tmp_path)
    qsearch = _qsearch_conversion(tmp_path)
    rescored = tmp_path / "rescored"
    assert main(["rescore-qsearch", str(config), str(qsearch), str(rescored)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["labels"] == 1
    assert summary["local_only"] is True

    label = json.loads((rescored / "labels.jsonl").read_text())
    # Split receipts use normalized-position hashes, not raw SFEN strings.
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=hashlib.sha256(b"game source").hexdigest(),
        position_ids=(hashlib.sha256(" ".join(label["sfen"].split()[:3]).encode()).hexdigest(),),
    )
    split_path = tmp_path / "train-split.json"
    split_path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    replay = tmp_path / "incremental-replay"

    assert (
        main(
            [
                "build-incremental-replay",
                str(rescored),
                str(split_path),
                str(replay),
            ]
        )
        == 0
    )
    replay_summary = json.loads(capsys.readouterr().out)
    assert replay_summary["records"] == 1
    assert replay_summary["policy_targets"] == 0
    assert (replay / "train.psv").stat().st_size == PSV_RECORD_BYTES
    with pytest.raises(FileExistsError, match="overwrite"):
        build_incremental_replay_from_rescore(rescored, split_path, replay)


def test_replay_builder_rejects_tampered_search_receipt(tmp_path: Path) -> None:
    config = _qsearch_config(tmp_path)
    qsearch = _qsearch_conversion(tmp_path)
    rescored = tmp_path / "rescored"
    run_qsearch_rescore(config, qsearch, rescored)
    labels = json.loads((rescored / "labels.jsonl").read_text())
    split = IncrementalValueSplit(
        split_id="train",
        source_games_sha256=hashlib.sha256(b"games").hexdigest(),
        position_ids=(hashlib.sha256(" ".join(labels["sfen"].split()[:3]).encode()).hexdigest(),),
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split.to_dict()), encoding="utf-8")
    (rescored / "searches.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        build_incremental_replay_from_rescore(rescored, split_path, tmp_path / "must-not-exist")


def test_qsearch_config_rejects_duplicate_casefold_keys(tmp_path: Path) -> None:
    config = _qsearch_config(tmp_path)
    raw = config.read_text()
    config.write_text(raw[:-1] + ',"Schema":"collision"}', encoding="utf-8")
    with pytest.raises(ValueError, match="case-insensitive"):
        load_qsearch_anchor_config(config)


def _sample(*, sfen: str, move: str) -> dict[str, Any]:
    return {
        "sfen": sfen,
        "ply": 0,
        "turn": 0,
        "policy": {move: 1.0},
        "root_value": 0.0,
        "value_target": 0.0,
        "chosen_move": move,
        "actor_source": "meteo-test",
    }


def _game_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "games"
    root.mkdir()
    board = Board()
    initial = board.to_sfen()
    move = "7g7f"
    final = board.copy()
    final.apply_move(Move.from_usi(move))
    game_id = hashlib.sha256(b"game-1").hexdigest()
    history_sha = hashlib.sha256(initial.encode() + b"\0").hexdigest()
    game = {
        "initial_sfen": initial,
        "moves": [move],
        "samples": [_sample(sfen=initial, move=move)],
        "winner": None,
        "termination": "max_plies",
    }
    ply = {
        "generated_ply": 0,
        "absolute_ply": 0,
        "turn": 0,
        "actor_source": "meteo-test",
        "rights_policy_id": "meteo-test",
        "sfen": initial,
        "move": move,
        "root_value": 0.0,
        "nodes": 64,
        "nps": 64000.0,
        "search_seconds": 0.001,
        "history_prefix_length": 0,
        "history_context_sha256": history_sha,
        "resignation_overridden": False,
        "termination": "max_plies",
        "rules_terminal_trajectory": False,
        "strong_value_target": None,
        "history_prefix": {
            "source": "game.full_game_moves",
            "length": 0,
            "sha256": history_sha,
        },
    }
    trajectory = {
        "schema": NNUE_GAME_RECEIPT_SCHEMA,
        "game_id": game_id,
        "opening_index": 0,
        "color_swap_leg": 0,
        "initial_sfen": initial,
        "opening_prefix_moves": [],
        "starting_sfen": initial,
        "opening_history_context_sha256": history_sha,
        "black_actor_source": "meteo-test",
        "white_actor_source": "teacher-test",
        "generated_moves": [move],
        "full_game_moves": [move],
        "final_sfen": final.to_sfen(),
        "plies": [ply],
        "winner": None,
        "termination": "max_plies",
        "rules_terminal_trajectory": False,
        "safety_truncated": True,
        "strong_wdl_label_allowed": False,
        "strong_wdl_label_black": None,
        "game_seconds": 0.001,
    }
    game_row = {
        "schema": NNUE_GAME_JSONL_SCHEMA,
        "game": game,
        "trajectory_receipt": trajectory,
    }
    games_bytes = (json.dumps(game_row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (root / "games.jsonl").write_bytes(games_bytes)
    unsigned_receipt: dict[str, object] = {
        "schema": NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
        "local_only": True,
        "publication_allowed": False,
        "strong_wdl_eligibility": {"allowed_only_for_rules_terminal_trajectory": True},
        "config": {},
        "engine_inputs": [],
        "games_jsonl": {
            "path": "games.jsonl",
            "schema": NNUE_GAME_JSONL_SCHEMA,
            "games": 1,
            "bytes": len(games_bytes),
            "sha256": hashlib.sha256(games_bytes).hexdigest(),
            "contains_complete_trajectory_receipts": True,
        },
        "generation": {},
    }
    receipt = {
        **unsigned_receipt,
        "payload_sha256": _json_sha(unsigned_receipt),
        "payload_sha256_scope": "receipt_without_payload_sha256_fields",
    }
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return root


def test_game_extractor_preserves_history_and_never_labels_played_move(
    tmp_path: Path,
) -> None:
    bundle = _game_bundle(tmp_path)
    output = tmp_path / "position-source"
    receipt = extract_game_position_source(bundle, output)
    assert receipt["schema"] == GAME_POSITION_BUNDLE_SCHEMA
    assert receipt["labels_emitted"] == 0
    row = json.loads((output / "positions.jsonl").read_text())
    assert row["schema"] == GAME_POSITION_SOURCE_SCHEMA
    assert row["initial_sfen"] == Board().to_sfen()
    assert row["history_moves"] == []
    assert row["full_game_moves"] == ["7g7f"]
    assert row["actor_source"] == "meteo-test"
    assert row["played_move"] == "7g7f"
    assert row["played_move_role"] == "proposal_only_not_label"
    assert row["label_generated"] is False
    split = json.loads((output / "split-receipt.json").read_text())
    assert split["label_generation_happens_after_split"] is True
    assert sum(item["game_count"] for item in split["splits"].values()) == 1
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_game_position_source(bundle, output)


def test_game_extractor_rejects_modified_bundle_bytes(tmp_path: Path) -> None:
    bundle = _game_bundle(tmp_path)
    with (bundle / "games.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        extract_game_position_source(bundle, tmp_path / "bad-output")


def test_incremental_mlx_cli_routes_prepare_run_and_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import simajilord_shogi.post_bootstrap_runner as runner

    captured: dict[str, object] = {}

    def fake_prepare(output_directory: Path, **kwargs: object) -> dict[str, object]:
        captured["output"] = output_directory
        captured.update(kwargs)
        return {"stage": "prepared"}

    monkeypatch.setattr(runner, "prepare_incremental_mlx_run", fake_prepare)
    required_paths = [
        "base-run-directory",
        "base-checkpoint",
        "broad-anchor-psv",
        "broad-anchor-receipt",
        "broad-anchor-split-receipt",
        "hard-exact-psv",
        "hard-exact-receipt",
        "hard-exact-split-receipt",
        "calibration-psv",
        "calibration-receipt",
        "calibration-split-receipt",
    ]
    argv = ["prepare-incremental-mlx", str(tmp_path / "incremental")]
    for option in required_paths:
        argv.extend((f"--{option}", str(tmp_path / option)))
    argv.extend(("--additional-optimizer-steps", "10", "--learning-rate", "0.00001"))
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out) == {"stage": "prepared"}
    assert captured["additional_optimizer_steps"] == 10
    assert captured["calibration_validation_positions"] == 262_144

    monkeypatch.setattr(runner, "run_incremental_mlx", lambda path: {"stage": "run"})
    assert main(["run-incremental-mlx", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"stage": "run"}
    monkeypatch.setattr(runner, "incremental_mlx_status", lambda path: {"stage": "status"})
    assert main(["incremental-status", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {"stage": "status"}


def test_registered_model_usage_receipt_cli_is_all_nine_and_create_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_paths: list[Path] = []
    for index, rights_id in enumerate(sorted(record.rights_id for record in MODEL_RIGHTS)):
        unsigned = {
            "schema": NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
            "local_only": True,
            "publication_allowed": False,
            "engine_inputs": [
                {"kind": "rights_profile", "rights_id": rights_id},
                {"kind": "meteo_runtime_contract", "rights_id": None},
            ],
            "generation": {
                "metrics": {"games": 240 if rights_id in set(CANONICAL_SCORER_IDS) else 20}
            },
        }
        receipt = {
            **unsigned,
            "payload_sha256": _json_sha(unsigned),
            "payload_sha256_scope": "receipt_without_payload_sha256_fields",
        }
        path = tmp_path / f"bundle-{index:02d}.json"
        path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        receipt_paths.append(path)
    output = tmp_path / "model-usage-gate.json"

    assert main(["validate-model-usage", str(output), *(str(path) for path in receipt_paths)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["complete"] is True
    assert len(result["observed_games"]) == 9
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError, match="overwrite"):
        validate_registered_model_usage_receipts(receipt_paths, output)


def test_registered_model_usage_receipts_reject_missing_and_tampered(
    tmp_path: Path,
) -> None:
    rights_id = sorted(record.rights_id for record in MODEL_RIGHTS)[0]
    unsigned = {
        "schema": NNUE_GAME_BUNDLE_RECEIPT_SCHEMA,
        "local_only": True,
        "publication_allowed": False,
        "engine_inputs": [
            {"kind": "rights_profile", "rights_id": rights_id},
            {"kind": "meteo_runtime_contract", "rights_id": None},
        ],
        "generation": {"metrics": {"games": 20}},
    }
    receipt = {
        **unsigned,
        "payload_sha256": _json_sha(unsigned),
        "payload_sha256_scope": "receipt_without_payload_sha256_fields",
    }
    source = tmp_path / "one.json"
    source.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="missing registered-model games"):
        validate_registered_model_usage_receipts([source], tmp_path / "gate.json")
    receipt["generation"]["metrics"]["games"] = 21
    source.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA-256 does not verify"):
        validate_registered_model_usage_receipts([source], tmp_path / "gate.json")
