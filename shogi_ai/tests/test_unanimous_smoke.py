from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.distillation_targets import CANONICAL_SCORER_IDS
from simajilord_shogi.replay import load_games, position_samples
from simajilord_shogi.rights_lineage import summarize_teacher_sidecar
from simajilord_shogi.unanimous_smoke import (
    COMMITTEE_BENCHMARK_SCHEMA,
    COMMITTEE_CONFIG_SCHEMA,
    COMMITTEE_PAIR_SCHEMA,
    UNANIMOUS_SMOKE_SCHEMA,
    build_unanimous_smoke_corpus,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evidence_row(board: Board, *, game_index: int, decision_index: int) -> dict[str, object]:
    candidates = tuple(sorted(move.to_usi() for move in board.legal_moves())[:3])
    best = candidates[0]
    values = {best: 0.5, candidates[1]: 0.1, candidates[2]: -0.2}
    teacher_intervals = [
        [
            teacher_id,
            [[move, values[move], values[move], "exact"] for move in candidates],
        ]
        for teacher_id in CANONICAL_SCORER_IDS
    ]
    budgets = [
        {
            "requested_nodes_per_candidate": nodes,
            "chosen_move": best,
            "teacher_intervals": teacher_intervals,
        }
        for nodes in (100, 1000)
    ]
    reply_candidates = [
        {
            "move": move,
            "terminal_root_value": None,
            "persistent_bounds": 0,
            "teacher_backed_intervals": [
                [teacher_id, values[move], values[move]]
                for teacher_id in CANONICAL_SCORER_IDS
            ],
        }
        for move in candidates
    ]
    return {
        "target_sfen": board.to_sfen(),
        "game_index": game_index,
        "opening_index": 0,
        "committee_decision_index": decision_index,
        "teacher_proposals": [
            [teacher_id, list(candidates)] for teacher_id in CANONICAL_SCORER_IDS
        ],
        "candidate_moves": list(candidates),
        "budgets": budgets,
        "budget_choice_stable": True,
        "reply_reanalysis": {
            "root_choice": best,
            "chosen_move": best,
            "reply_score_nodes_per_teacher_per_reply": 1000,
            "reanalysed_candidates": list(candidates),
            "candidates": reply_candidates,
        },
        "chosen_move": best,
    }


def _committee_fixture(root: Path) -> Path:
    config = {
        "schema": COMMITTEE_CONFIG_SCHEMA,
        "seed": 7,
        "contract": {
            "committee_id": "meteo-three-teacher-reply-reanalysed-interval-committee-v3",
            "canonical_teacher_ids": list(CANONICAL_SCORER_IDS),
            "score_budgets_per_teacher_per_candidate": [100, 1000],
            "reply_score_nodes_per_teacher_per_reply": 1000,
            "actual_moves_are_training_labels": False,
        },
    }
    _write_json(root / "config.json", config)

    board = Board()
    evidence_by_opponent: dict[str, list[dict[str, object]]] = {
        teacher_id: [] for teacher_id in CANONICAL_SCORER_IDS
    }
    owners = (
        CANONICAL_SCORER_IDS[0],
        CANONICAL_SCORER_IDS[0],
        CANONICAL_SCORER_IDS[1],
        CANONICAL_SCORER_IDS[2],
    )
    for index, owner in enumerate(owners):
        evidence_by_opponent[owner].append(
            _evidence_row(board, game_index=index, decision_index=0)
        )
        board.apply_move(sorted(board.legal_moves(), key=lambda move: move.to_usi())[0])

    for opponent_id, evidence in evidence_by_opponent.items():
        pair = root / "opponents" / opponent_id
        evidence_path = pair / "committee-evidence.json"
        _write_json(evidence_path, evidence)
        raw = evidence_path.read_bytes()
        _write_json(
            pair / "report.json",
            {
                "schema": COMMITTEE_PAIR_SCHEMA,
                "opponent": opponent_id,
                "contract": {
                    "canonical_teacher_ids": list(CANONICAL_SCORER_IDS),
                    "actual_moves_are_training_labels": False,
                },
                "committee_decisions": len(evidence),
                "committee_evidence": {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                },
            },
        )
    _write_json(
        root / "report.json",
        {
            "schema": COMMITTEE_BENCHMARK_SCHEMA,
            "config": config,
            "completed_opponents": 3,
            "all_games_complete": True,
            "results": [{"opponent": teacher_id} for teacher_id in CANONICAL_SCORER_IDS],
        },
    )
    return root


def test_build_unanimous_smoke_is_receipted_local_only_and_nonpromotable(
    tmp_path: Path,
) -> None:
    benchmark = _committee_fixture(tmp_path / "benchmark")
    output = tmp_path / "smoke"

    manifest = build_unanimous_smoke_corpus(
        benchmark,
        output,
        train_count=2,
        calibration_count=1,
        heldout_count=1,
        split_seed="unit-test",
    )

    assert manifest["schema"] == UNANIMOUS_SMOKE_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"train": 2, "calibration": 1, "heldout": 1}
    assert manifest["promotable_checkpoint"] is False
    assert (output / "manifest.json").is_file()

    samples = position_samples(load_games(output / "train.jsonl"))
    assert len(samples) == 2
    for sample in samples:
        assert sample.teacher_source == "soujou-tsec7-paid"
        assert sample.teacher_best_move == sample.chosen_move
        assert sample.teacher_policy is not None
        assert sample.teacher_move_values is not None
        assert max(sample.teacher_policy, key=sample.teacher_policy.get) == sample.chosen_move
        assert pytest.approx(sum(sample.teacher_policy.values())) == 1.0

    split_receipt = json.loads((output / "split-receipt.json").read_text(encoding="utf-8"))
    assert all(not overlap for overlap in split_receipt["normalized_position_overlap"].values())
    assert all(not overlap for overlap in split_receipt["game_group_overlap"].values())
    assert any(split_receipt["opening_group_overlap"].values())
    assert "opening_root_shared_across_diagnostic_splits" in split_receipt["promotion_blockers"]

    sidecar_path = output / "train.jsonl.provenance.json"
    sidecar_raw = sidecar_path.read_bytes()
    summary = summarize_teacher_sidecar(
        json.loads(sidecar_raw),
        sidecar_sha256=hashlib.sha256(sidecar_raw).hexdigest(),
    )
    assert summary["publication_allowed"] is False
    assert [source["rights_id"] for source in summary["sources"]] == sorted(
        CANONICAL_SCORER_IDS
    )

    score_receipt = json.loads(
        (output / "score-matrix-receipt.json").read_text(encoding="utf-8")
    )
    assert score_receipt["selected_rows"] == 4
    assert score_receipt["selection_contract"]["actual_benchmark_moves_used_as_labels"] is False
    assert score_receipt["selection_contract"]["cross_teacher_value_average"] is False


def test_unanimous_smoke_rejects_evidence_changed_after_pair_receipt(tmp_path: Path) -> None:
    benchmark = _committee_fixture(tmp_path / "benchmark")
    evidence = (
        benchmark
        / "opponents"
        / CANONICAL_SCORER_IDS[0]
        / "committee-evidence.json"
    )
    evidence.write_bytes(evidence.read_bytes() + b" ")

    with pytest.raises(ValueError, match="does not bind the exact evidence"):
        build_unanimous_smoke_corpus(
            benchmark,
            tmp_path / "smoke",
            train_count=2,
            calibration_count=1,
            heldout_count=1,
        )
    assert not (tmp_path / "smoke").exists()


def test_unanimous_smoke_excludes_bound_rows_instead_of_using_midpoints(
    tmp_path: Path,
) -> None:
    benchmark = _committee_fixture(tmp_path / "benchmark")
    opponent = CANONICAL_SCORER_IDS[2]
    evidence_path = benchmark / "opponents" / opponent / "committee-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[0]["budgets"][-1]["teacher_intervals"][0][1][0][3] = "lowerbound"
    _write_json(evidence_path, evidence)
    raw = evidence_path.read_bytes()
    report_path = evidence_path.with_name("report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["committee_evidence"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="not enough strict positions"):
        build_unanimous_smoke_corpus(
            benchmark,
            tmp_path / "smoke",
            train_count=2,
            calibration_count=1,
            heldout_count=1,
        )
    assert not (tmp_path / "smoke").exists()


def test_unanimous_smoke_excludes_untyped_q_endpoint_that_may_be_mate(
    tmp_path: Path,
) -> None:
    benchmark = _committee_fixture(tmp_path / "benchmark")
    opponent = CANONICAL_SCORER_IDS[2]
    evidence_path = benchmark / "opponents" / opponent / "committee-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    best = evidence[0]["chosen_move"]
    for budget in evidence[0]["budgets"]:
        for _teacher_id, matrix in budget["teacher_intervals"]:
            for score_row in matrix:
                if score_row[0] == best:
                    score_row[1] = 1.0
                    score_row[2] = 1.0
    for candidate in evidence[0]["reply_reanalysis"]["candidates"]:
        if candidate["move"] == best:
            for backed_row in candidate["teacher_backed_intervals"]:
                backed_row[1] = 1.0
                backed_row[2] = 1.0
    _write_json(evidence_path, evidence)
    raw = evidence_path.read_bytes()
    report_path = evidence_path.with_name("report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["committee_evidence"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    _write_json(report_path, report)

    with pytest.raises(ValueError, match="not enough strict positions"):
        build_unanimous_smoke_corpus(
            benchmark,
            tmp_path / "smoke",
            train_count=2,
            calibration_count=1,
            heldout_count=1,
        )
    assert not (tmp_path / "smoke").exists()
