from __future__ import annotations

import json
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.artifact_provenance import canonical_json_sha256
from simajilord_shogi.disagreement import (
    DISAGREEMENT_REPORT_SCHEMA,
    DisagreementConfig,
    DisagreementReason,
    DisagreementTeacherInput,
    build_disagreement_replay,
    write_disagreement_replay,
)
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.ensemble import (
    TeacherReplayInput,
    build_teacher_ensemble,
    write_teacher_ensemble,
)
from simajilord_shogi.replay import append_games, position_samples


def _teacher_game(
    *,
    label: str,
    move: str,
    value: float,
    reported_mate: bool = False,
) -> GameRecord:
    variation = TeacherVariation(
        rank=1,
        move=move,
        score_kind=(TeacherScoreKind.MATE if reported_mate else TeacherScoreKind.CENTIPAWN),
        bound=TeacherScoreBound.EXACT,
        pv=(move,),
        score_cp=None if reported_mate else int(value * 1_000),
        mate_plies=3 if reported_mate else None,
    )
    sample = PositionSample(
        sfen=Board().to_sfen(),
        ply=0,
        turn=0,
        policy={"7g7f": 1.0},
        root_value=0.0,
        chosen_move="7g7f",
        actor_best_move="7g7f",
        teacher_policy={move: 1.0},
        teacher_value=value,
        teacher_best_move=move,
        teacher_source=label,
        teacher_context="screen-20k",
        teacher_variations=(variation,),
    )
    return GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=(move,),
        samples=(sample,),
        winner=None,
        termination=Termination.REPETITION,
    )


def test_family_aware_selection_preserves_unique_move_and_unproven_mate_report(
    tmp_path: Path,
) -> None:
    teacher_rows = (
        ("hao", "tanuki", "7g7f", 0.8, True),
        ("tanuki-dr4", "tanuki", "7g7f", 0.6, False),
        ("gikou", "gikou", "2g2f", -0.8, False),
    )
    teacher_paths: list[Path] = []
    for label, _family, move, value, mate in teacher_rows:
        replay = tmp_path / f"{label}.jsonl"
        append_games(
            replay,
            [_teacher_game(label=label, move=move, value=value, reported_mate=mate)],
        )
        teacher_paths.append(replay)

    ensemble_path = tmp_path / "ensemble.jsonl"
    manifest_path = tmp_path / "manifest.json"
    ensemble = build_teacher_ensemble(
        tuple(
            TeacherReplayInput(label, replay)
            for (label, _family, _move, _value, _mate), replay in zip(
                teacher_rows, teacher_paths, strict=True
            )
        ),
        split="train",
        minimum_teachers=3,
    )
    write_teacher_ensemble(ensemble, ensemble_path, manifest=manifest_path)

    inputs = tuple(
        DisagreementTeacherInput(label, replay, family)
        for (label, family, _move, _value, _mate), replay in zip(
            teacher_rows, teacher_paths, strict=True
        )
    )
    build = build_disagreement_replay(
        ensemble_path,
        manifest_path,
        inputs,
        config=DisagreementConfig(high_js_threshold=0.1),
    )
    repeated = build_disagreement_replay(
        ensemble_path,
        manifest_path,
        inputs,
        config=DisagreementConfig(high_js_threshold=0.1),
    )

    assert build == repeated
    assert build.report.schema == DISAGREEMENT_REPORT_SCHEMA
    assert build.report.selected_positions == 1
    position = build.report.positions[0]
    assert set(position.reasons) == {reason.value for reason in DisagreementReason}
    # Hao and tanuki-dr4 are one correlated vote, versus Gikou's one vote.
    assert position.family_top1 == {"gikou": "2g2f", "tanuki": "7g7f"}
    assert position.family_normalized_jensen_shannon == pytest.approx(1.0)
    assert position.recommended_nodes == 2_000_000
    assert position.trajectory.phase == "opening_plies_1_24"
    assert position.trajectory.opening_family.startswith("opening-fingerprint:")
    assert len(position.teacher_stories) == 3
    assert position.teacher_stories[0].variations
    assert len(position.reported_winning_mates) == 1
    # A score-mate report only triggers selection; this module never calls it a proof.
    assert position.reported_winning_mates[0].move == "7g7f"

    output = tmp_path / "deep-input.jsonl"
    payload = write_disagreement_replay(build, output)
    report_path = output.with_suffix(output.suffix + ".disagreement.json")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_hash = saved.pop("provenance_sha256")
    assert recorded_hash == canonical_json_sha256(saved)
    assert payload["provenance_sha256"] == recorded_hash
    assert len(position_samples(build.games, include_incomplete=True)) == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_disagreement_replay(build, output)


def test_selection_rejects_teacher_hash_not_in_manifest(tmp_path: Path) -> None:
    replay_a = tmp_path / "a.jsonl"
    replay_b = tmp_path / "b.jsonl"
    append_games(replay_a, [_teacher_game(label="a", move="7g7f", value=0.5)])
    append_games(replay_b, [_teacher_game(label="b", move="2g2f", value=-0.5)])
    ensemble_path = tmp_path / "ensemble.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_teacher_ensemble(
        build_teacher_ensemble(
            (TeacherReplayInput("a", replay_a), TeacherReplayInput("b", replay_b)),
            split="train",
        ),
        ensemble_path,
        manifest=manifest_path,
    )
    replay_b.write_text(replay_b.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 does not match"):
        build_disagreement_replay(
            ensemble_path,
            manifest_path,
            (
                DisagreementTeacherInput("a", replay_a, "a-family"),
                DisagreementTeacherInput("b", replay_b, "b-family"),
            ),
        )
