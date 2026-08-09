from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from simajilord_shogi.cli import build_parser, main
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.replay import append_games
from simajilord_shogi.value_scale_fit import (
    ValueScaleObservation,
    fit_teacher_value_scales,
    fit_value_scale,
)


def _variation(
    cp: int,
    *,
    kind: TeacherScoreKind = TeacherScoreKind.CENTIPAWN,
    bound: TeacherScoreBound = TeacherScoreBound.EXACT,
) -> TeacherVariation:
    if kind is TeacherScoreKind.MATE:
        return TeacherVariation(
            rank=1,
            move="7g7f",
            score_kind=kind,
            bound=bound,
            pv=("7g7f",),
            mate_plies=3,
        )
    return TeacherVariation(
        rank=1,
        move="7g7f",
        score_kind=kind,
        bound=bound,
        pv=("7g7f",),
        score_cp=cp,
    )


def _sample(
    cp: int,
    turn: int,
    *,
    source: str = "nagisa-v3.1",
    kind: TeacherScoreKind = TeacherScoreKind.CENTIPAWN,
    bound: TeacherScoreBound = TeacherScoreBound.EXACT,
) -> PositionSample:
    return PositionSample(
        sfen=f"synthetic-{source}-{cp}-{turn}-{kind}-{bound}",
        ply=0,
        turn=turn,
        policy={"7g7f": 1.0},
        root_value=0.0,
        teacher_best_move="7g7f",
        teacher_source=source,
        teacher_variations=(_variation(cp, kind=kind, bound=bound),),
    )


def _calibration_games() -> list[GameRecord]:
    cps = (-600, -300, -100, 100, 300, 600)
    labels = (0, 1, 0, 1, 0, 1)
    games: list[GameRecord] = []
    for index, (cp, label) in enumerate(zip(cps, labels, strict=True)):
        turn = index % 2
        winner = turn if label else 1 - turn
        games.append(
            GameRecord(
                initial_sfen=f"game-{index}",
                moves=("7g7f",),
                samples=(
                    _sample(cp, turn),
                    _sample(cp, turn, kind=TeacherScoreKind.MATE),
                    _sample(cp, turn, bound=TeacherScoreBound.LOWER),
                ),
                winner=winner,
                termination=Termination.RESIGNATION,
            )
        )
    games.append(
        GameRecord(
            initial_sfen="incomplete",
            moves=(),
            samples=(_sample(200, 0),),
            winner=None,
            termination=Termination.MAX_PLIES,
        )
    )
    return games


def test_convex_scale_fit_recovers_known_fractional_probability_curve() -> None:
    expected_scale = 375.0
    cps = (-900, -600, -300, -100, 0, 100, 300, 600, 900)
    observations = [
        ValueScaleObservation(
            game_id=index,
            cp=cp,
            label=1.0 / (1.0 + math.exp(-2.0 * cp / expected_scale)),
        )
        for index, cp in enumerate(cps)
    ]

    estimate = fit_value_scale(observations)

    assert estimate.boundary is None
    assert estimate.value_scale == pytest.approx(expected_scale, rel=1e-8)


def test_report_excludes_mate_bounds_and_incomplete_games_with_cluster_uncertainty() -> None:
    report = fit_teacher_value_scales(
        _calibration_games(),
        bootstrap_resamples=40,
        seed=7,
        minimum_reliable_games=10,
        minimum_reliable_samples=10,
    )

    assert report.provisional
    assert report.extraction["games"] == 7
    assert report.extraction["eligible_samples"] == 6
    assert report.extraction["excluded_mate_scores"] == 6
    assert report.extraction["excluded_bounded_scores"] == 6
    assert report.extraction["incomplete_games"] == 1
    teacher = report.teachers["nagisa-v3.1"]
    assert teacher.samples == 6
    assert teacher.games == 6
    assert teacher.cp_min == -600
    assert teacher.cp_max == 600
    assert teacher.outcomes == {"win": 3, "draw": 0, "loss": 3}
    assert teacher.exclusions == {"bounded_score": 6, "mate_score": 6}
    assert teacher.baseline_bce > 0
    assert teacher.fitted_bce > 0
    assert teacher.cross_validation is not None
    assert teacher.cross_validation.mode == "leave_one_game_out"
    assert teacher.cross_validation.held_out_samples == 6
    assert teacher.cluster_bootstrap is not None
    assert teacher.cluster_bootstrap.resamples == 40
    assert teacher.cluster_bootstrap.scale_p2_5 > 0
    warning_codes = {warning.code for warning in teacher.warnings}
    assert {"small_sample_count", "small_game_count"} <= warning_codes
    json.dumps(report.to_dict(), allow_nan=False)


def test_cli_writes_hashed_provisional_json_report(tmp_path: Path) -> None:
    replay = tmp_path / "teacher.jsonl"
    output = tmp_path / "reports" / "scale.json"
    append_games(replay, _calibration_games())

    args = build_parser().parse_args(
        [
            "fit-teacher-value-scale",
            str(replay),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "20",
            "--seed",
            "11",
        ]
    )
    assert args.command == "fit-teacher-value-scale"
    assert main(
        [
            "fit-teacher-value-scale",
            str(replay),
            "--output",
            str(output),
            "--bootstrap-resamples",
            "20",
            "--seed",
            "11",
        ]
    ) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["provisional"] is True
    assert payload["source_replay"] == str(replay.resolve())
    assert len(payload["source_replay_sha256"]) == 64
    assert payload["teachers"]["nagisa-v3.1"]["samples"] == 6
    assert payload["method"]["probability"] == "sigmoid(2 * cp / value_scale)"
    assert payload["baseline_ponanza_coefficient"] == 600.0
    assert payload["baseline_value_tanh_denominator"] == 1_200.0
    assert payload["baseline_scale_input_convention"] == "default_ponanza_coefficient"
    assert payload["method"]["scale_semantics"]["baseline_ponanza_coefficient"] == 600.0
    teacher = payload["teachers"]["nagisa-v3.1"]
    assert teacher["baseline_ponanza_coefficient"] == 600.0
    assert teacher["baseline_value_tanh_denominator"] == 1_200.0
    assert teacher["fitted_ponanza_coefficient"] * 2 == pytest.approx(
        teacher["fitted_value_tanh_denominator"]
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "fit-teacher-value-scale",
                str(replay),
                "--output",
                str(output),
            ]
        )
