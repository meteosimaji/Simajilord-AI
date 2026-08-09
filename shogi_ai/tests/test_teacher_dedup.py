from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

import simajilord_shogi.teacher_dedup as teacher_dedup
from simajilord_shogi.artifact_provenance import sha256_file
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.replay import append_games, load_games
from simajilord_shogi.teacher_dedup import (
    SINGLE_TEACHER_DEDUP_METHOD,
    SINGLE_TEACHER_DEDUP_SCHEMA,
    SingleTeacherDedupBuild,
    SingleTeacherDedupReport,
    TeacherDedupOccurrence,
    TeacherDedupPositionReport,
    build_single_teacher_dedup,
    write_single_teacher_dedup,
)

KING_CYCLE = ("5i6h", "5a6b", "6h5i", "6b5a")


def _board_after(moves: tuple[str, ...]) -> Board:
    board = Board()
    for move_usi in moves:
        move = Move.from_usi(move_usi)
        assert board.is_legal_move(move)
        board.apply_move(move)
    return board


def _teacher_sample(
    *,
    sfen: str,
    ply: int,
    chosen_move: str | None,
    actor_move: str,
    teacher_policy: dict[str, float],
    teacher_value: float,
    teacher_source: str = "nagisa-v3.1",
    teacher_context: str = "history-aware",
) -> PositionSample:
    teacher_best = min(
        teacher_policy,
        key=lambda move: (-teacher_policy[move], move),
    )
    variations = tuple(
        TeacherVariation(
            rank=rank,
            move=move,
            score_kind=TeacherScoreKind.CENTIPAWN,
            bound=TeacherScoreBound.EXACT,
            pv=(move,),
            score_cp=400 - 100 * rank,
        )
        for rank, move in enumerate(teacher_policy, start=1)
    )
    board = Board(sfen)
    assert all(board.is_legal_move(Move.from_usi(move)) for move in teacher_policy)
    return PositionSample(
        sfen=sfen,
        ply=ply,
        turn=board.turn.value,
        policy={actor_move: 1.0},
        root_value=0.0,
        value_target=0.0,
        chosen_move=chosen_move,
        actor_best_move=actor_move,
        teacher_policy=teacher_policy,
        teacher_value=teacher_value,
        teacher_best_move=teacher_best,
        teacher_nodes=20_000,
        teacher_time_ms=20,
        teacher_nps=1_000_000,
        teacher_depth=12,
        teacher_source=teacher_source,
        teacher_context=teacher_context,
        teacher_variations=variations,
        teacher_policy_temperature=100.0,
        teacher_value_scale=1_200.0,
    )


def _different_history_games(*, identical_targets: bool = False) -> tuple[GameRecord, ...]:
    start = Board().to_sfen()
    returned = _board_after(KING_CYCLE).to_sfen()
    first_policy = {"7g7f": 3.0, "2g2f": 1.0}
    second_policy = (
        first_policy if identical_targets else {"7g7f": 1.0, "2g2f": 3.0}
    )
    first_value = 0.8
    second_value = first_value if identical_targets else -0.2
    first = GameRecord(
        initial_sfen=start,
        moves=KING_CYCLE,
        samples=(
            _teacher_sample(
                sfen=start,
                ply=0,
                chosen_move="5i6h",
                actor_move="5i6h",
                teacher_policy=first_policy,
                teacher_value=first_value,
                teacher_context="direct-start",
            ),
        ),
        winner=None,
        termination=Termination.REPETITION,
    )
    second = GameRecord(
        initial_sfen=start,
        moves=KING_CYCLE,
        samples=(
            _teacher_sample(
                sfen=returned,
                ply=4,
                chosen_move=None,
                actor_move="7g7f",
                teacher_policy=second_policy,
                teacher_value=second_value,
                teacher_context="after-cycle",
            ),
        ),
        winner=None,
        termination=Termination.REPETITION,
    )
    return first, second


def _write_replay(path: Path, games: tuple[GameRecord, ...]) -> None:
    assert append_games(path, games) == len(games)


def test_dedup_averages_different_history_targets_without_stitching_story(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nagisa-history.jsonl"
    games = _different_history_games()
    _write_replay(source, games)
    source_bytes = source.read_bytes()
    source_hash = sha256_file(source)

    build = build_single_teacher_dedup(source, split="train")

    assert isinstance(build, SingleTeacherDedupBuild)
    assert source.read_bytes() == source_bytes
    assert sha256_file(source) == source_hash
    assert len(build.games) == 1
    assert build.games[0].initial_sfen == games[0].initial_sfen
    assert build.games[0].moves == games[0].moves
    assert len(build.games[0].samples) == 1
    output_sample = build.games[0].samples[0]
    assert output_sample.ply == 0
    assert output_sample.sfen == games[0].samples[0].sfen
    assert output_sample.teacher_policy == pytest.approx({"2g2f": 0.5, "7g7f": 0.5})
    assert math.isclose(sum(output_sample.teacher_policy.values()), 1.0)
    assert output_sample.teacher_value == pytest.approx(0.3)
    assert output_sample.teacher_best_move == "2g2f"
    assert output_sample.teacher_variations is None
    assert output_sample.teacher_nodes is None
    assert output_sample.teacher_time_ms is None
    assert output_sample.teacher_depth is None
    assert output_sample.teacher_policy_temperature == 100.0
    assert output_sample.teacher_value_scale == 1_200.0

    report = build.report
    assert isinstance(report, SingleTeacherDedupReport)
    assert report.schema == SINGLE_TEACHER_DEDUP_SCHEMA
    assert report.method == SINGLE_TEACHER_DEDUP_METHOD
    assert report.split == "train"
    assert report.input.artifact.file.sha256 == source_hash
    assert report.output_samples == 1
    assert report.duplicate_positions == 1
    assert report.duplicate_observations_collapsed == 1
    assert report.positions_with_multiple_histories == 1
    assert report.positions_with_policy_conflicts == 1
    assert report.positions_with_value_conflicts == 1
    assert report.positions_with_any_target_conflict == 1
    position = report.positions[0]
    assert isinstance(position, TeacherDedupPositionReport)
    assert position.occurrence_count == 2
    assert position.distinct_input_games == 2
    assert position.distinct_history_contexts == 2
    assert position.distinct_teacher_contexts == 2
    assert position.policy_conflict
    assert position.value_conflict
    assert position.target_conflict
    assert position.generalized_policy_js_divergence_nats > 0.0
    assert position.maximum_policy_l1_to_average == pytest.approx(0.5)
    assert position.teacher_value_width == pytest.approx(1.0)
    assert all(
        isinstance(occurrence, TeacherDedupOccurrence)
        for occurrence in position.occurrences
    )
    assert [occurrence.history_move_count for occurrence in position.occurrences] == [0, 4]
    assert all(occurrence.typed_variation_count == 2 for occurrence in position.occurrences)
    assert all(
        occurrence.typed_variations_sha256 is not None
        for occurrence in position.occurrences
    )
    assert len({occurrence.sample_sha256 for occurrence in position.occurrences}) == 2
    assert len({occurrence.history_context_sha256 for occurrence in position.occurrences}) == 2


def test_dedup_identical_targets_are_not_reported_as_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "identical.jsonl"
    _write_replay(source, _different_history_games(identical_targets=True))

    build = build_single_teacher_dedup(source, split="validation")
    position = build.report.positions[0]

    assert position.distinct_history_contexts == 2
    assert not position.policy_conflict
    assert not position.value_conflict
    assert not position.target_conflict
    assert position.generalized_policy_js_divergence_nats == pytest.approx(0.0)
    assert position.maximum_policy_l1_to_average == pytest.approx(0.0)
    assert build.report.positions_with_any_target_conflict == 0


def test_dedup_writer_is_create_only_atomic_hashed_and_deterministic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_replay(source, _different_history_games())
    build = build_single_teacher_dedup(source, split="train")
    repeated = build_single_teacher_dedup(source, split="train")
    assert build == repeated

    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    first_payload = write_single_teacher_dedup(build, first_output)
    second_payload = write_single_teacher_dedup(build, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert tuple(load_games(first_output)) == build.games
    assert first_payload["schema"] == SINGLE_TEACHER_DEDUP_SCHEMA
    first_output_report = first_payload["output"]
    second_output_report = second_payload["output"]
    assert isinstance(first_output_report, dict)
    assert isinstance(second_output_report, dict)
    first_replay_identity = first_output_report["replay"]
    second_replay_identity = second_output_report["replay"]
    assert isinstance(first_replay_identity, dict)
    assert isinstance(second_replay_identity, dict)
    assert first_replay_identity["sha256"] == sha256_file(first_output)
    assert first_replay_identity["sha256"] == second_replay_identity["sha256"]
    first_manifest = first_output.with_suffix(first_output.suffix + ".dedup.json")
    assert first_manifest.is_file()
    assert not list(tmp_path.glob(".*.tmp-*"))

    source_hash = sha256_file(source)
    with pytest.raises(FileExistsError, match="refusing to overwrite dedup replay"):
        write_single_teacher_dedup(build, first_output)
    assert sha256_file(source) == source_hash
    assert sha256_file(first_output) == first_replay_identity["sha256"]


def test_dedup_rejects_illegal_targets_overlap_and_path_collisions(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    games = _different_history_games()
    invalid_sample = replace(
        games[0].samples[0],
        teacher_policy={"7g7e": 1.0},
        teacher_best_move="7g7e",
        teacher_variations=(
            TeacherVariation(
                rank=1,
                move="7g7e",
                score_kind=TeacherScoreKind.CENTIPAWN,
                bound=TeacherScoreBound.EXACT,
                pv=("7g7e",),
                score_cp=1,
            ),
        ),
    )
    invalid_game = replace(games[0], samples=(invalid_sample,))
    invalid_source = tmp_path / "invalid.jsonl"
    _write_replay(invalid_source, (invalid_game,))
    with pytest.raises(ValueError, match="illegal teacher policy move"):
        build_single_teacher_dedup(invalid_source, split="train")

    _write_replay(source, games)
    with pytest.raises(ValueError, match="split overlap guard rejected"):
        build_single_teacher_dedup(
            source,
            split="train",
            forbid_overlap_with=(source,),
        )

    disjoint_board = _board_after(("7g7f",))
    disjoint_sample = PositionSample(
        sfen=disjoint_board.to_sfen(),
        ply=1,
        turn=disjoint_board.turn.value,
        policy={"3c3d": 1.0},
        root_value=0.0,
    )
    disjoint_guard = tmp_path / "guard.jsonl"
    _write_replay(
        disjoint_guard,
        (
            GameRecord(
                initial_sfen=Board().to_sfen(),
                moves=("7g7f",),
                samples=(disjoint_sample,),
                winner=None,
                termination=Termination.REPETITION,
            ),
        ),
    )
    with pytest.raises(ValueError, match="duplicate overlap-guard replay"):
        build_single_teacher_dedup(
            source,
            split="train",
            forbid_overlap_with=(disjoint_guard, disjoint_guard),
        )

    build = build_single_teacher_dedup(source, split="train")
    with pytest.raises(ValueError, match="paths must differ"):
        write_single_teacher_dedup(build, source)
    same_target = tmp_path / "same.jsonl"
    with pytest.raises(ValueError, match="paths must differ"):
        write_single_teacher_dedup(build, same_target, manifest=same_target)

    existing_output = tmp_path / "existing.jsonl"
    existing_output.write_text("owned by user\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite dedup replay"):
        write_single_teacher_dedup(build, existing_output)
    assert existing_output.read_text(encoding="utf-8") == "owned by user\n"
    assert not existing_output.with_suffix(existing_output.suffix + ".dedup.json").exists()

    new_output = tmp_path / "new.jsonl"
    existing_manifest = tmp_path / "owned-manifest.json"
    existing_manifest.write_text("owned by user\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite dedup manifest"):
        write_single_teacher_dedup(build, new_output, manifest=existing_manifest)
    assert not new_output.exists()
    assert existing_manifest.read_text(encoding="utf-8") == "owned by user\n"


def test_atomic_pair_rolls_back_its_manifest_when_output_install_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_replay(source, _different_history_games())
    build = build_single_teacher_dedup(source, split="train")
    output = tmp_path / "raced.jsonl"
    manifest = output.with_suffix(output.suffix + ".dedup.json")
    real_link = os.link
    calls = 0

    def fail_second_link(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError(target_path)
        real_link(source_path, target_path)

    monkeypatch.setattr(teacher_dedup.os, "link", fail_second_link)

    with pytest.raises(FileExistsError, match="refusing to overwrite dedup replay"):
        write_single_teacher_dedup(build, output)
    assert not output.exists()
    assert not manifest.exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_single_teacher_and_public_names_have_no_case_or_type_collision(
    tmp_path: Path,
) -> None:
    assert len(teacher_dedup.__all__) == len(set(teacher_dedup.__all__))
    assert teacher_dedup.__all__ == tuple(dict.fromkeys(teacher_dedup.__all__))
    assert teacher_dedup.SingleTeacherDedupBuild is SingleTeacherDedupBuild
    assert teacher_dedup.SingleTeacherDedupReport is SingleTeacherDedupReport
    assert teacher_dedup.TeacherDedupOccurrence is TeacherDedupOccurrence
    assert teacher_dedup.TeacherDedupPositionReport is TeacherDedupPositionReport
    assert len(
        {
            SingleTeacherDedupBuild,
            SingleTeacherDedupReport,
            TeacherDedupOccurrence,
            TeacherDedupPositionReport,
        }
    ) == 4

    source = tmp_path / "mixed-sources.jsonl"
    games = _different_history_games()
    mixed = replace(
        games[1],
        samples=(replace(games[1].samples[0], teacher_source="suisho5"),),
    )
    _write_replay(source, (games[0], mixed))
    with pytest.raises(ValueError, match="exactly one teacher_source"):
        build_single_teacher_dedup(source, split="train")
