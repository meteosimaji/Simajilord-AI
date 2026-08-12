from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

from simajilord_shogi.ybb_book import (
    YBB_HEADER,
    YBB_INDEX,
    YBB_MAGIC,
    YBB_MOVE,
    YBB_MOVE_DEPTH_FLAG,
    YaneuraOuBinaryBook,
    YbbReanalysisReceipt,
    build_ybb_reanalysis_plan,
    verify_ybb_reanalysis_coverage,
    write_ybb_reanalysis_plan,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fixture_book(path: Path) -> None:
    first = Board()
    second = Board()
    second.push_usi("7g7f")
    records = [
        (first, 1, (("7g7f", 120, 20), ("2g2f", 100, 18))),
        (second, 2, (("3c3d", -30, 16),)),
    ]
    records.sort(key=lambda record: bytes(record[0].to_packed_sfen()))
    move_offset = 0
    index = bytearray()
    moves = bytearray()
    for board, ply, candidates in records:
        index.extend(
            YBB_INDEX.pack(
                bytes(board.to_packed_sfen()),
                move_offset,
                ply,
                len(candidates),
            )
        )
        for move_usi, evaluation, depth in candidates:
            moves.extend(YBB_MOVE.pack(int(Move.from_usi(move_usi)), evaluation, depth))
            move_offset += YBB_MOVE.size
    path.write_bytes(
        YBB_HEADER.pack(YBB_MAGIC, len(records), YBB_MOVE_DEPTH_FLAG) + index + moves
    )


def test_ybb_reader_streams_unique_positions_and_keeps_book_data_out_of_targets(
    tmp_path: Path,
) -> None:
    book_path = tmp_path / "fixture.ybb"
    _fixture_book(book_path)

    with YaneuraOuBinaryBook(book_path) as book:
        audit = book.structural_audit()
        positions = tuple(book.iter_position_candidates())
        moves = tuple(book.book_moves_for_audit(index) for index in range(len(book)))

        initial = Board()
        initial_index = book.find_position_index(initial)
        initial_moves = book.book_moves_for_position_audit(initial)
        after_76 = Board()
        after_76.push_usi("7g7f")
        after_76_index = book.find_position_index(after_76)
        different = Board()
        different.push_usi("2g2f")
        different_index = book.find_position_index(different)
        wrong_ply = Board(initial.to_sfen().rsplit(" ", 1)[0] + " 99")
        wrong_ply_exact = book.find_position_index(wrong_ply)
        wrong_ply_ignored = book.find_position_index(wrong_ply, exact_ply=False)

    assert audit.entries_scanned == 2
    assert audit.move_records_referenced == 3
    assert audit.duplicate_packed_sfens == 0
    assert [position.index for position in positions] == [0, 1]
    assert all(position.reanalysis_required for position in positions)
    assert not any(position.book_moves_are_training_targets for position in positions)
    assert {move.move_usi for group in moves for move in group} == {"7g7f", "2g2f", "3c3d"}
    assert initial_index is not None
    assert initial_moves is not None
    assert {move.move_usi for move in initial_moves} == {"7g7f", "2g2f"}
    assert after_76_index is not None
    assert different_index is None
    assert wrong_ply_exact is None
    assert wrong_ply_ignored == initial_index


def test_ybb_plan_covers_every_index_and_receipts_require_every_teacher_shard(
    tmp_path: Path,
) -> None:
    book_path = tmp_path / "fixture.ybb"
    _fixture_book(book_path)
    archive_sha256 = _digest("source archive")
    plan = build_ybb_reanalysis_plan(
        book_path,
        source_url="https://storage.example/book.7z",
        source_archive_sha256=archive_sha256,
        required_teacher_ids=("nagisa-v3.1", "suisho11plus-wcsc36"),
        shard_size=1,
        local_only_user_authorized=True,
    )

    assert [(shard.start_index, shard.stop_index) for shard in plan.shards] == [
        (0, 1),
        (1, 2),
    ]
    assert plan.to_dict()["training_contract"] == {
        "book_moves_are_labels": False,
        "book_evaluations_are_labels": False,
        "book_depths_are_labels": False,
        "fresh_no_book_multipv_required": True,
        "all_source_records_covered_exactly_once": True,
        "deep_escalation": (
            "depth flips, teacher disagreement, mate, defense, early deviation, "
            "or held-out weakness"
        ),
        "publication": "source book and its derived move/eval data are excluded",
    }
    output = tmp_path / "plan.json"
    output_sha256 = write_ybb_reanalysis_plan(plan, output)
    assert output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_text())["source"]["sha256"] == plan.source.sha256
    with pytest.raises(FileExistsError):
        write_ybb_reanalysis_plan(plan, output)

    receipts = [
        YbbReanalysisReceipt(
            teacher_id=teacher_id,
            shard_id=shard.shard_id,
            source_sha256=plan.source.sha256,
            range_sha256=shard.range_sha256,
            output_replay_sha256=_digest(f"{teacher_id}-{shard.shard_id}"),
            positions_attempted=shard.positions,
            positions_succeeded=shard.positions,
            positions_failed=0,
            nodes=plan.screen_nodes,
            multipv=plan.multipv,
            book_enabled=False,
        )
        for teacher_id in plan.required_teacher_ids
        for shard in plan.shards
    ]
    with pytest.raises(ValueError, match="incomplete"):
        verify_ybb_reanalysis_coverage(plan, receipts[:-1])
    coverage = verify_ybb_reanalysis_coverage(plan, receipts)
    assert coverage["complete"] is True
    assert coverage["source_records"] == 2
    assert coverage["book_moves_used_as_labels"] is False


def test_ybb_plan_is_local_only_and_reader_rejects_unknown_flags(tmp_path: Path) -> None:
    book_path = tmp_path / "fixture.ybb"
    _fixture_book(book_path)
    with pytest.raises(PermissionError, match="local-only"):
        build_ybb_reanalysis_plan(
            book_path,
            source_url="https://storage.example/book.7z",
            source_archive_sha256=_digest("archive"),
            required_teacher_ids=("nagisa-v3.1",),
        )

    invalid_path = tmp_path / "unknown-flags.ybb"
    invalid_path.write_bytes(YBB_HEADER.pack(YBB_MAGIC, 1, 2) + bytes(YBB_INDEX.size))
    with pytest.raises(ValueError, match="unknown flags"):
        YaneuraOuBinaryBook(invalid_path)
