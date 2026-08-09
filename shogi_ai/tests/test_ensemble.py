from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

from simajilord_shogi.cli import _ensemble_teacher_inputs, main
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.ensemble import (
    ENSEMBLE_TEACHER_SOURCE,
    TeacherReplayInput,
    build_teacher_ensemble,
    normalized_sfen,
    write_teacher_ensemble,
)
from simajilord_shogi.replay import append_games, load_games, position_samples


def _teacher_sample(
    *,
    move_count: int,
    teacher_source: str,
    teacher_context: str,
    teacher_policy: dict[str, float],
    teacher_value: float,
    teacher_nodes: int,
) -> PositionSample:
    fields = Board().to_sfen().split()
    sfen = " ".join([*fields[:3], str(move_count)])
    return PositionSample(
        sfen=sfen,
        ply=move_count - 1,
        turn=0,
        policy={"7g7f": 1.0},
        root_value=0.0,
        value_target=0.0,
        chosen_move="7g7f",
        actor_best_move="7g7f",
        actor_simulations=10,
        teacher_policy=teacher_policy,
        teacher_value=teacher_value,
        teacher_best_move=max(teacher_policy, key=teacher_policy.__getitem__),
        teacher_regret=0.2,
        teacher_nodes=teacher_nodes,
        teacher_time_ms=20,
        teacher_nps=teacher_nodes * 50,
        teacher_depth=8,
        teacher_depth_ratio=teacher_nodes / 10,
        teacher_source=teacher_source,
        teacher_context=teacher_context,
        teacher_variations=(
            TeacherVariation(
                rank=1,
                move=max(teacher_policy, key=teacher_policy.__getitem__),
                score_kind=TeacherScoreKind.CENTIPAWN,
                bound=TeacherScoreBound.EXACT,
                pv=(max(teacher_policy, key=teacher_policy.__getitem__),),
                score_cp=123,
            ),
        ),
        teacher_policy_temperature=175.0,
        teacher_value_scale=450.0,
    )


def _write_teacher_replays(tmp_path: Path) -> tuple[Path, Path]:
    teacher_a = tmp_path / "teacher-a.jsonl"
    teacher_b = tmp_path / "teacher-b.jsonl"
    game_a = GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=(),
        samples=(
            _teacher_sample(
                move_count=1,
                teacher_source="suisho5-yaneuraou-v7.50",
                teacher_context="endgame",
                teacher_policy={"7g7f": 3.0, "2g2f": 1.0},
                teacher_value=0.8,
                teacher_nodes=100,
            ),
            _teacher_sample(
                move_count=42,
                teacher_source="suisho5-yaneuraou-v7.50",
                teacher_context="endgame",
                teacher_policy={"7g7f": 2.0},
                teacher_value=0.6,
                teacher_nodes=110,
            ),
        ),
        winner=None,
        termination=Termination.REPETITION,
    )
    game_b = GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=(),
        samples=(
            _teacher_sample(
                move_count=900,
                teacher_source="gikou2-v2.0.2",
                teacher_context="tactical",
                teacher_policy={"7g7f": 1.0, "2g2f": 3.0},
                teacher_value=-0.2,
                teacher_nodes=200,
            ),
        ),
        winner=None,
        termination=Termination.REPETITION,
    )
    append_games(teacher_a, [game_a])
    append_games(teacher_b, [game_b])
    teacher_a.with_suffix(teacher_a.suffix + ".provenance.json").write_text(
        json.dumps({"rights": {"rights_id": "suisho5-yaneuraou-v7.50"}}),
        encoding="utf-8",
    )
    return teacher_a, teacher_b


def test_teacher_ensemble_normalizes_deduplicates_and_preserves_disagreement(
    tmp_path: Path,
) -> None:
    teacher_a, teacher_b = _write_teacher_replays(tmp_path)
    inputs = (
        TeacherReplayInput("suisho", teacher_a, weight=1.0),
        TeacherReplayInput("gikou", teacher_b, weight=3.0),
    )

    build = build_teacher_ensemble(inputs, split="train", top_k=1)
    repeated = build_teacher_ensemble(inputs, split="train", top_k=1)

    assert build == repeated
    samples = position_samples(build.games)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sfen.endswith(" 1")
    assert sample.teacher_source == ENSEMBLE_TEACHER_SOURCE
    assert sample.teacher_context == "ensemble:train"
    assert sample.teacher_policy is not None
    assert math.isclose(sum(sample.teacher_policy.values()), 1.0)
    # Suisho's two normalized observations average to (.875, .125), then the
    # per-teacher mixture uses weights (.25, .75).
    assert sample.teacher_policy["7g7f"] == pytest.approx(0.40625)
    assert sample.teacher_policy["2g2f"] == pytest.approx(0.59375)
    assert sample.teacher_value == pytest.approx(0.025)
    assert sample.teacher_best_move == "2g2f"
    assert sample.policy_reversal
    assert sample.teacher_nodes is None
    assert sample.teacher_depth is None
    assert sample.teacher_variations is None
    assert sample.teacher_policy_temperature is None
    assert sample.teacher_value_scale is None

    report = build.report
    assert report.output_normalized_positions == 1
    assert report.duplicate_observations_collapsed == 1
    assert report.inputs[0].provenance == {
        "rights": {"rights_id": "suisho5-yaneuraou-v7.50"}
    }
    position = report.positions[0]
    assert position.jensen_shannon_divergence_nats > 0
    assert 0 < position.normalized_jensen_shannon_divergence < 1
    assert position.weighted_pairwise_top_k_jaccard == 0
    assert not position.unanimous_teacher_best_move
    assert position.mixture_best_move_teacher_weight == pytest.approx(0.75)
    assert position.teacher_value_width == pytest.approx(0.9)
    assert position.positive_value_weight == pytest.approx(0.25)
    assert position.negative_value_weight == pytest.approx(0.75)
    assert position.dominant_value_direction_weight == pytest.approx(0.75)
    assert not position.unanimous_value_direction
    assert [item.teacher_nodes for item in position.contributions[0].occurrences] == [
        100,
        110,
    ]
    assert position.contributions[1].occurrences[0].teacher_context == "tactical"
    occurrence = position.contributions[1].occurrences[0]
    assert occurrence.teacher_variations[0].score_cp == 123
    assert occurrence.teacher_policy_temperature == 175.0
    assert occurrence.teacher_value_scale == 450.0
    assert any("Full typed PV" in limitation for limitation in report.limitations)


def test_teacher_ensemble_keeps_union_of_alternative_mating_moves(tmp_path: Path) -> None:
    mate_sfen = "3pkp3/9/3B5/9/9/9/9/9/4K4 b GR 1"
    for move_usi in ("G*5b", "R*5b"):
        board = Board(mate_sfen)
        move = Move.from_usi(move_usi)
        assert board.is_legal_move(move)
        board.apply_move(move)
        assert board.is_mated()

    replay_paths: list[Path] = []
    for label, move_usi, value in (
        ("gold-mate", "G*5b", 1.0),
        ("rook-mate", "R*5b", 0.999),
    ):
        replay_path = tmp_path / f"{label}.jsonl"
        sample = PositionSample(
            sfen=mate_sfen,
            ply=0,
            turn=0,
            policy={"G*5b": 1.0},
            root_value=1.0,
            value_target=1.0,
            chosen_move="G*5b",
            actor_best_move="G*5b",
            teacher_policy={move_usi: 1.0},
            teacher_value=value,
            teacher_best_move=move_usi,
            teacher_nodes=1_000_000,
            teacher_source=label,
            teacher_context="forced-mate",
            teacher_variations=(
                TeacherVariation(
                    rank=1,
                    move=move_usi,
                    score_kind=TeacherScoreKind.MATE,
                    bound=TeacherScoreBound.EXACT,
                    pv=(move_usi,),
                    mate_plies=1,
                ),
            ),
        )
        append_games(
            replay_path,
            [
                GameRecord(
                    initial_sfen=mate_sfen,
                    moves=("G*5b",),
                    samples=(sample,),
                    winner=0,
                    termination=Termination.CHECKMATE,
                )
            ],
        )
        replay_paths.append(replay_path)

    build = build_teacher_ensemble(
        (
            TeacherReplayInput("gold-mate", replay_paths[0]),
            TeacherReplayInput("rook-mate", replay_paths[1]),
        ),
        split="train",
        top_k=2,
    )

    sample = position_samples(build.games)[0]
    assert sample.teacher_policy == {"G*5b": 0.5, "R*5b": 0.5}
    assert sample.teacher_variations is None
    assert not sample.policy_reversal
    position = build.report.positions[0]
    assert position.mixture_policy_support_size == 2
    assert position.teacher_policy_support_union_size == 2
    assert position.teacher_policy_support_intersection_size == 0
    assert position.weighted_pairwise_policy_overlap == 0
    assert position.normalized_jensen_shannon_divergence == pytest.approx(1.0)
    assert position.soft_policy_agreement == pytest.approx(0.0)
    assert position.unanimous_value_direction
    retained_mates = {
        occurrence.teacher_variations[0].move
        for contribution in position.contributions
        for occurrence in contribution.occurrences
    }
    assert retained_mates == {"G*5b", "R*5b"}
    assert any("alternative mating moves" in item for item in build.report.limitations)


def test_teacher_ensemble_writer_and_cli_are_reproducible(tmp_path: Path, capsys: object) -> None:
    teacher_a, teacher_b = _write_teacher_replays(tmp_path)
    inputs = (
        TeacherReplayInput("suisho", teacher_a),
        TeacherReplayInput("gikou", teacher_b),
    )
    build = build_teacher_ensemble(inputs, split="validation")
    first_output = tmp_path / "ensemble-first.jsonl"
    second_output = tmp_path / "ensemble-second.jsonl"

    first_manifest = write_teacher_ensemble(build, first_output)
    write_teacher_ensemble(build, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_manifest["schema"] == "meteo-teacher-ensemble-v1"
    assert first_manifest["output_normalized_positions"] == 1
    assert load_games(first_output) == list(build.games)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_teacher_ensemble(build, first_output)

    cli_output = tmp_path / "ensemble-cli.jsonl"
    assert (
        main(
            [
                "ensemble-teachers",
                str(cli_output),
                "--teacher",
                f"suisho={teacher_a}",
                "--teacher",
                f"gikou={teacher_b}",
                "--weight",
                "suisho=2",
                "--split",
                "train",
            ]
        )
        == 0
    )
    assert cli_output.is_file()
    assert cli_output.with_suffix(cli_output.suffix + ".ensemble.json").is_file()


def test_teacher_ensemble_rejects_labels_paths_and_cross_split_overlap(tmp_path: Path) -> None:
    teacher_a, teacher_b = _write_teacher_replays(tmp_path)
    same_position_with_different_counter = " ".join(
        [*Board().to_sfen().split()[:3], "1234"]
    )
    assert normalized_sfen(Board().to_sfen()) == normalized_sfen(
        same_position_with_different_counter
    )

    with pytest.raises(ValueError, match="duplicate teacher input label"):
        build_teacher_ensemble(
            (
                TeacherReplayInput("Teacher", teacher_a),
                TeacherReplayInput("teacher", teacher_b),
            ),
            split="train",
        )
    with pytest.raises(ValueError, match="duplicate teacher input replay"):
        build_teacher_ensemble(
            (
                TeacherReplayInput("a", teacher_a),
                TeacherReplayInput("b", teacher_a),
            ),
            split="train",
        )
    with pytest.raises(ValueError, match="split overlap guard rejected"):
        build_teacher_ensemble(
            (
                TeacherReplayInput("a", teacher_a),
                TeacherReplayInput("b", teacher_b),
            ),
            split="train",
            forbid_overlap_with=[teacher_b],
        )

    parsed = _ensemble_teacher_inputs(
        [f"Suisho={teacher_a}", f"Gikou={teacher_b}"],
        ["suisho=2.5"],
    )
    assert parsed[0].weight == 2.5
    assert parsed[1].weight == 1.0
    with pytest.raises(ValueError, match="duplicate teacher label"):
        _ensemble_teacher_inputs([f"X={teacher_a}", f"x={teacher_b}"], [])
