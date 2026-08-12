from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
from rsshogi.core import Board
from rsshogi.numpy import PackedSfenValue

import simajilord_shogi.public_psv as public_psv
from simajilord_shogi.public_psv import (
    PUBLIC_POSITION_SEED_SCHEMA,
    PUBLIC_PSV_SEED_SCHEMA,
    HttpRangeResult,
    fetch_public_psv_position_seeds,
)

_CORPUS_ID = "nodchip-shogi-hao-depth9"
_REVISION = "7bc19a9e880ea307a52c57f57ea6c752301b25bc"
_LOCAL_CORPUS_ID = "soujou-team-datasets-1"
_LOCAL_REVISION = "4dfad115d4a808ebe20b6f65f6416ad75a69a6e7"


def _records(count: int, *, zero_moves: bool = False) -> bytes:
    array = np.zeros(count, dtype=PackedSfenValue)
    board = Board()
    for index in range(count):
        legal = sorted(board.legal_moves(), key=lambda move: move.to_usi())
        move = legal[index % len(legal)]
        array[index]["sfen"] = np.frombuffer(board.to_packed_sfen(), dtype=np.uint8)
        array[index]["score"] = index - count // 2
        array[index]["move"] = 0 if zero_moves else int(move)
        array[index]["game_ply"] = index
        array[index]["game_result"] = (-1, 0, 1)[index % 3]
        board.apply_move(move)
    return array.tobytes()


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, bytes],
    *,
    revision: str = _REVISION,
) -> None:
    api = json.dumps(
        {
            "sha": revision,
            "siblings": [{"rfilename": name} for name in files],
        },
        sort_keys=True,
    ).encode()
    monkeypatch.setattr(
        public_psv,
        "_fetch_url",
        lambda _url, *, timeout_seconds: api,
    )

    def fake_range(
        url: str,
        *,
        start: int,
        end: int,
        timeout_seconds: float,
    ) -> HttpRangeResult:
        del timeout_seconds
        name = unquote(url.rsplit("/", 1)[-1])
        payload = files[name]
        assert 0 <= start <= end < len(payload)
        return HttpRangeResult(
            payload=payload[start : end + 1],
            start=start,
            end=end,
            total_bytes=len(payload),
            etag=f'"{hashlib.sha256(payload).hexdigest()}"',
            final_url=url,
        )

    monkeypatch.setattr(public_psv, "_fetch_range", fake_range)


def test_fetch_public_psv_emits_only_unlabelled_receipted_position_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {
        "a.bin": _records(12),
        "b.bin": _records(12),
        "ignored.txt": b"not a PSV",
    }
    _install_fake_hub(monkeypatch, files)

    output = tmp_path / "public-seeds"
    receipt = fetch_public_psv_position_seeds(
        _CORPUS_ID,
        output,
        file_count=2,
        records_per_file=4,
        seed="unit-test",
    )

    assert receipt["schema"] == PUBLIC_PSV_SEED_SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["contract"] == {
        "source_rights_scope": "public_licensed",
        "license_verified": "MIT",
        "operator_acknowledged_user_attested_local_reuse": False,
        "source_and_derived_artifacts_local_only": False,
        "source_move_score_and_result_are_training_labels": False,
        "position_seed_only": True,
        "qsearch_or_root_reanalysis_required": True,
        "current_three_teacher_reanalysis_required": True,
        "optimizer_input_allowed": False,
        "heldout_or_promotion_evidence_allowed": False,
        "source_game_lineage_available": False,
    }
    assert (output / "receipt.json").is_file()
    rows = [json.loads(line) for line in (output / "positions.jsonl").read_text().splitlines()]
    assert rows
    assert len(rows) <= 8
    assert all(row["schema"] == PUBLIC_POSITION_SEED_SCHEMA for row in rows)
    assert all(
        row["discarded_legacy_annotations"]["allowed_as_meteo_training_label"] is False
        for row in rows
    )
    assert all(row["required_next_stage"] == "current_three_teacher_reanalysis" for row in rows)
    assert len({row["normalized_sfen"] for row in rows}) == len(rows)


def test_fetch_public_psv_fails_closed_on_value_only_move_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_hub(monkeypatch, {"value-only.bin": _records(8, zero_moves=True)})
    output = tmp_path / "bad-seeds"

    with pytest.raises(ValueError, match="Move16=0"):
        fetch_public_psv_position_seeds(
            _CORPUS_ID,
            output,
            file_count=1,
            records_per_file=4,
        )

    assert not output.exists()


def test_local_only_value_corpus_requires_acknowledgement_and_discards_zero_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "local-seeds"
    with pytest.raises(PermissionError, match="local-only corpus requires"):
        fetch_public_psv_position_seeds(
            _LOCAL_CORPUS_ID,
            output,
            file_count=1,
            records_per_file=4,
        )

    _install_fake_hub(
        monkeypatch,
        {"split_000.bin": _records(8, zero_moves=True)},
        revision=_LOCAL_REVISION,
    )
    receipt = fetch_public_psv_position_seeds(
        _LOCAL_CORPUS_ID,
        output,
        file_count=1,
        records_per_file=4,
        allow_user_attested_local_only=True,
    )

    assert receipt["contract"]["source_rights_scope"] == "user_attested_local_only"
    assert receipt["contract"]["license_verified"] is None
    assert receipt["contract"]["operator_acknowledged_user_attested_local_reuse"] is True
    assert receipt["contract"]["source_and_derived_artifacts_local_only"] is True
    assert "source_rights_are_user_attested_local_only" in receipt["promotion_blockers"]
    rows = [json.loads(line) for line in (output / "positions.jsonl").read_text().splitlines()]
    assert rows
    assert all(row["discarded_legacy_annotations"]["move"] is None for row in rows)
    assert all(
        row["discarded_legacy_annotations"]["move16_contract"]
        == "zero_value_only_required"
        for row in rows
    )


def test_fetch_public_psv_rejects_unreviewed_corpus_before_network(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        fetch_public_psv_position_seeds(
            "dlsuisho15b-unique-public",
            tmp_path / "forbidden",
        )
