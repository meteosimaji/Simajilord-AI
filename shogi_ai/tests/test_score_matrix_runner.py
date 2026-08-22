from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from rsshogi.core import Board

import simajilord_shogi.external_usi as external_usi_module
import simajilord_shogi.production_score_matrix as production_score_matrix_module
from simajilord_shogi.artifact_provenance import sha256_file
from simajilord_shogi.distillation_targets import CANONICAL_SCORER_IDS
from simajilord_shogi.production_score_matrix import (
    ACTUAL_PLAYED_CANDIDATE_SOURCE_ID,
    AUXILIARY_PROPOSER_IDS,
    METEO_CANDIDATE_SOURCE_ID,
    PRODUCTION_SCORE_MATRIX_SCHEMA,
)
from simajilord_shogi.score_matrix_runner import (
    LIMITED_LOCAL_SCOPE,
    PRIVATE_OUTPUT_SCOPE,
    SCORE_MATRIX_RUNNER_CONFIG_SCHEMA,
    SCORE_MATRIX_RUNNER_SUMMARY_SCHEMA,
    load_runner_config,
    main,
    run_score_matrix,
)


def _sha(path: Path) -> str:
    return sha256_file(path)


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


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _artifact_set(paths: list[Path]) -> tuple[list[dict[str, str]], str]:
    artifacts = [_artifact(path) for path in sorted(paths, key=lambda item: str(item))]
    return artifacts, _json_sha({"artifacts": artifacts})


def _write_fake_usi(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys

options = {
    "BookFile": "no_book",
    "Hash": "64",
    "USI_Hash": "64",
    "MultiPV": "1",
    "Threads": "1",
}


def emit(line):
    print(line, flush=True)


for raw in sys.stdin:
    line = raw.rstrip("\\r\\n")
    if line == "usi":
        emit("id name StrictFakeUSI")
        emit("option name MultiPV type spin default 1 min 1 max 32")
        emit("option name Threads type spin default 1 min 1 max 8")
        emit("option name Hash type spin default 64 min 1 max 4096")
        emit("option name USI_Hash type spin default 64 min 1 max 4096")
        emit("option name BookFile type string default no_book")
        emit("usiok")
    elif line.startswith("setoption name "):
        payload = line.removeprefix("setoption name ")
        name, separator, value = payload.partition(" value ")
        options[name] = value if separator else ""
    elif line == "isready":
        emit("readyok")
    elif line.startswith("getoption "):
        name = line.removeprefix("getoption ")
        if name not in options:
            emit("error no such option")
        else:
            emit(f"Options[{name}] = {options[name]}")
    elif line.startswith("go nodes "):
        tokens = line.split()
        nodes = int(tokens[2])
        move_index = tokens.index("searchmoves") + 1
        move = tokens[move_index]
        emit(
            f"info depth 8 seldepth 12 nodes {nodes} time 1 nps {nodes * 1000} "
            f"score cp 25 pv {move}"
        )
        emit(f"bestmove {move}")
    elif line == "quit":
        break
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _proposal_payload(
    *,
    producer: str,
    moves: list[str],
    history_command: str,
    artifact_path: Path,
    reset_search_state: bool = False,
    requested_nodes: int = 0,
    multipv: int = 0,
) -> dict[str, object]:
    sent_commands = [
        *((["usinewgame"]) if reset_search_state else []),
        history_command,
        f"propose {' '.join(moves)}",
    ]
    transcript_lines = [f"proposal {' '.join(moves)}"]
    producer_artifacts, producer_identity = _artifact_set([artifact_path])
    return {
        "producer": producer,
        "moves": moves,
        "requested_nodes": requested_nodes,
        "reported_nodes": requested_nodes,
        "multipv": multipv,
        "sent_commands": sent_commands,
        "transcript_lines": transcript_lines,
        "transcript_sha256": _json_sha(
            {
                "sent_commands": sent_commands,
                "transcript_lines": transcript_lines,
            }
        ),
        "producer_artifacts": producer_artifacts,
        "producer_identity_sha256": producer_identity,
        "producer_code_artifact": _artifact(artifact_path),
    }


def _write_config(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    engine = tmp_path / "fake_usi.py"
    _write_fake_usi(engine)
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"scale":1200}\n', encoding="utf-8")
    meteo_artifact = tmp_path / "meteo-proposer.bin"
    meteo_artifact.write_bytes(b"meteo proposal model")

    parser_path = Path(external_usi_module.__file__).resolve(strict=True)
    scorer_code_path = Path(production_score_matrix_module.__file__).resolve(strict=True)
    evaluation_artifacts, evaluation_digest = _artifact_set([engine])
    options = [
        {"name": "BookFile", "value": "no_book"},
        {"name": "Hash", "value": 64},
        {"name": "Threads", "value": 1},
    ]
    scorer_configs: list[dict[str, object]] = []
    for scorer_id in CANONICAL_SCORER_IDS:
        scorer_configs.append(
            {
                "scorer_id": scorer_id,
                "engine": {
                    "path": str(engine),
                    "sha256": _sha(engine),
                    "arguments": [],
                },
                "working_directory": str(tmp_path),
                "evaluation_artifacts": evaluation_artifacts,
                "evaluation_artifacts_sha256": evaluation_digest,
                "options": options,
                "hash_option_name": "Hash",
                "proposal_multipv": 4,
                "book_disable": {
                    "enabled": False,
                    "option_name": "BookFile",
                    "option_value": "no_book",
                },
                "option_value_verification": "yaneuraou_getoption",
                "parser_artifact": _artifact(parser_path),
                "scorer_code_artifact": _artifact(scorer_code_path),
                "calibration_artifact": _artifact(calibration),
                "timeout_seconds": 5.0,
                "value_scale": 1200.0,
                "expected_fatal_startup_diagnostics": [],
            }
        )

    board = Board()
    initial_sfen = board.to_sfen()
    history_command = "position startpos"
    payload: dict[str, Any] = {
        "schema": SCORE_MATRIX_RUNNER_CONFIG_SCHEMA,
        "output_scope": PRIVATE_OUTPUT_SCOPE,
        "limited_local_authorization": {
            "authorized": True,
            "scope": LIMITED_LOCAL_SCOPE,
            "rights_ids": ["soujou-tsec7-paid", "suisho11plus-wcsc36-20260525-local"],
        },
        "history": {
            "initial_sfen": initial_sfen,
            "moves": [],
            "target_sfen": initial_sfen,
        },
        "search": {
            "requested_nodes": 64,
            "root_proposal_nodes": 32,
            "reply_proposal_nodes": 32,
            "interval_dominance_margin": 0.0,
        },
        "experiment": {
            "arm_id": "B",
            "scoring_anchor_id": CANONICAL_SCORER_IDS[2],
            "proposal_source_ids": list(CANONICAL_SCORER_IDS),
            "auxiliary_proposal_source_ids": [],
        },
        "canonical_scorers": scorer_configs,
        "meteo_proposal": _proposal_payload(
            producer="Meteo proposal-only model",
            moves=["7g7f"],
            history_command=history_command,
            artifact_path=meteo_artifact,
        ),
        "actual_played_move": _proposal_payload(
            producer="immutable game replay",
            moves=["2g2f"],
            history_command=history_command,
            artifact_path=meteo_artifact,
        ),
        "auxiliary_proposals": [],
    }
    config = tmp_path / "score-matrix-config.json"
    config.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return config, tmp_path / "score-matrix.json", payload


def test_cli_builds_create_only_external_usi_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output, _payload = _write_config(tmp_path)

    assert main([str(config), str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == SCORE_MATRIX_RUNNER_SUMMARY_SCHEMA
    assert summary["local_only"] is True
    assert summary["publication_allowed"] is False
    assert summary["qsearch_leaf_re_evaluated"] is False
    assert summary["cross_teacher_value_average"] is False
    assert summary["majority_vote"] is False
    assert summary["experiment_arm_id"] == "B"
    assert summary["scoring_anchor_id"] == CANONICAL_SCORER_IDS[2]
    assert summary["proposal_source_ids"] == list(CANONICAL_SCORER_IDS)

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == PRODUCTION_SCORE_MATRIX_SCHEMA
    assert receipt["position_mode"] == "exact_history_root_with_one_ply_reply_reanalysis"
    assert receipt["qsearch_leaf_re_evaluated"] is False
    assert receipt["cross_teacher_value_average"] is False
    assert receipt["majority_vote"] is False
    assert receipt["proposal_membership_is_truth"] is False
    assert receipt["local_only"] is True
    assert receipt["publication_allowed"] is False
    assert [row["scorer"]["scorer_id"] for row in receipt["scorer_matrices"]] == [
        CANONICAL_SCORER_IDS[2]
    ]
    assert receipt["experiment_arm_id"] == "B"
    assert receipt["proposal_role"] == "top_k_candidate_generation_only_not_policy_or_value_truth"
    assert receipt["scoring_role"] == "single_anchor_equal_node_multipv1_searchmoves"
    assert [identity["multipv"] for identity in receipt["proposal_identities"]] == [4, 4, 4]
    source_kinds = {
        source["source_id"]: source["source_kind"] for source in receipt["candidate_sources"]
    }
    assert source_kinds[METEO_CANDIDATE_SOURCE_ID] == "proposal_only"
    assert source_kinds[ACTUAL_PLAYED_CANDIDATE_SOURCE_ID] == "actual_played_move_proposal_only"
    assert {candidate["move"] for candidate in receipt["candidates"]} >= {"2g2f", "7g7f"}
    for matrix in receipt["scorer_matrices"]:
        assert matrix["scorer"]["multipv"] == 1
        assert matrix["scorer"]["threads"] == 1
        assert matrix["scorer"]["book_enabled"] is False
        for candidate in matrix["candidates"]:
            searches = [candidate["root_search"], *candidate["reply_searches"]]
            for search in searches:
                expected = f"go nodes 64 searchmoves {search['searchmove']}"
                assert expected in search["sent_commands"]
                assert search["sent_commands"][0] == "usinewgame"
                assert search["sent_commands"].count("usinewgame") == 1
                assert search["requested_nodes"] == 64
                assert search["reported_nodes"] == 64

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_score_matrix(config, output)


def test_loader_fails_closed_without_exact_limited_local_authorization(tmp_path: Path) -> None:
    _config, _output, payload = _write_config(tmp_path)
    payload["limited_local_authorization"]["authorized"] = False
    denied = tmp_path / "denied.json"
    denied.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PermissionError, match="explicit authorization"):
        load_runner_config(denied)

    payload["limited_local_authorization"]["authorized"] = True
    payload["limited_local_authorization"]["rights_ids"] = ["soujou-tsec7-paid"]
    denied.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="exact required rights IDs"):
        load_runner_config(denied)


@pytest.mark.parametrize(
    "raw,match",
    [
        ('{"schema":"x","Schema":"y"}', "case-insensitive JSON object key collision"),
        ('{"schema":NaN}', "non-finite JSON constant"),
        ('{"schema":Infinity}', "non-finite JSON constant"),
    ],
)
def test_loader_rejects_ambiguous_or_nonfinite_json(tmp_path: Path, raw: str, match: str) -> None:
    config = tmp_path / "invalid.json"
    config.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_runner_config(config)


def test_loader_verifies_artifact_hashes_before_engine_start(tmp_path: Path) -> None:
    _config, _output, payload = _write_config(tmp_path)
    payload["canonical_scorers"][0]["engine"]["sha256"] = "0" * 64
    invalid = tmp_path / "invalid-hash.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="engine SHA-256 mismatch"):
        load_runner_config(invalid)


def test_loader_requires_exact_canonical_scorer_order(tmp_path: Path) -> None:
    _config, _output, payload = _write_config(tmp_path)
    payload["canonical_scorers"] = list(reversed(payload["canonical_scorers"]))
    invalid = tmp_path / "wrong-order.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical order"):
        load_runner_config(invalid)


def test_loader_accepts_per_scorer_hash_or_usi_hash_with_equal_numeric_mb(
    tmp_path: Path,
) -> None:
    _config, _output, payload = _write_config(tmp_path)
    first = payload["canonical_scorers"][0]
    first["hash_option_name"] = "USI_Hash"
    first["options"] = sorted(
        [
            row if row["name"] != "Hash" else {"name": "USI_Hash", "value": 64}
            for row in first["options"]
        ],
        key=lambda row: str(row["name"]).casefold(),
    )
    mixed = tmp_path / "mixed-hash-options.json"
    mixed.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_runner_config(mixed)
    assert loaded.canonical_scorers[0].hash_option_name == "USI_Hash"
    assert {scorer.hash_mb for scorer in loaded.canonical_scorers} == {64}

    payload["canonical_scorers"][0]["options"] = sorted(
        [
            row if row["name"] != "USI_Hash" else {"name": "USI_Hash", "value": 128}
            for row in payload["canonical_scorers"][0]["options"]
        ],
        key=lambda row: str(row["name"]).casefold(),
    )
    mismatched = tmp_path / "mismatched-hash-mb.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="common numeric hash size"):
        load_runner_config(mismatched)


@pytest.mark.parametrize(
    ("arm_id", "proposal_source_ids"),
    [
        ("A-N", [CANONICAL_SCORER_IDS[0]]),
        ("A-W", [CANONICAL_SCORER_IDS[1]]),
        ("A-S", [CANONICAL_SCORER_IDS[2]]),
        ("B", list(CANONICAL_SCORER_IDS)),
    ],
)
def test_runner_pins_baseline_arm_anchor_and_proposal_sources(
    tmp_path: Path,
    arm_id: str,
    proposal_source_ids: list[str],
) -> None:
    _config, output, payload = _write_config(tmp_path)
    payload["experiment"].update(
        {
            "arm_id": arm_id,
            "proposal_source_ids": proposal_source_ids,
            "auxiliary_proposal_source_ids": [],
        }
    )
    config = tmp_path / f"arm-{arm_id}.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    receipt = run_score_matrix(config, output)
    assert receipt.experiment_arm_id == arm_id
    assert receipt.scoring_anchor_id == CANONICAL_SCORER_IDS[2]
    assert receipt.proposal_source_ids == tuple(proposal_source_ids)
    assert tuple(matrix.scorer.scorer_id for matrix in receipt.scorer_matrices) == (
        CANONICAL_SCORER_IDS[2],
    )
    assert receipt.cross_teacher_value_average is False
    assert receipt.majority_vote is False


def test_b9_arm_accepts_all_six_auxiliary_proposers_as_nontruth_evidence(
    tmp_path: Path,
) -> None:
    _config, output, payload = _write_config(tmp_path)
    legal_moves = tuple(sorted(move.to_usi() for move in Board().legal_moves()))
    auxiliary_rows: list[dict[str, object]] = []
    for index, source_id in enumerate(AUXILIARY_PROPOSER_IDS):
        artifact = tmp_path / f"aux-{index}.bin"
        artifact.write_bytes(f"auxiliary proposal model {source_id}".encode())
        auxiliary_rows.append(
            {
                "source_id": source_id,
                "rights_id": source_id,
                "proposal": _proposal_payload(
                    producer=f"auxiliary {source_id}",
                    moves=[legal_moves[index]],
                    history_command="position startpos",
                    artifact_path=artifact,
                    reset_search_state=True,
                    requested_nodes=32,
                    multipv=4,
                ),
            }
        )
    payload["experiment"].update(
        {
            "arm_id": "B9",
            "proposal_source_ids": list(CANONICAL_SCORER_IDS),
            "auxiliary_proposal_source_ids": list(AUXILIARY_PROPOSER_IDS),
        }
    )
    payload["auxiliary_proposals"] = auxiliary_rows
    config = tmp_path / "arm-b9.json"
    config.write_text(json.dumps(payload), encoding="utf-8")

    receipt = run_score_matrix(config, output)
    assert receipt.experiment_arm_id == "B9"
    assert receipt.auxiliary_proposal_source_ids == AUXILIARY_PROPOSER_IDS
    assert (
        tuple(
            authorization.rights_id for authorization in receipt.auxiliary_proposal_authorizations
        )
        == AUXILIARY_PROPOSER_IDS
    )
    source_by_id = {source.source_id: source for source in receipt.candidate_sources}
    assert set(AUXILIARY_PROPOSER_IDS).issubset(source_by_id)
    assert all(
        source_by_id[source_id].source_kind.value == "proposal_only"
        and source_by_id[source_id].sent_commands[0] == "usinewgame"
        for source_id in AUXILIARY_PROPOSER_IDS
    )
    assert len(receipt.scorer_matrices) == 1
    assert receipt.proposal_membership_is_truth is False
