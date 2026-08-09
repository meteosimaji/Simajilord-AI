from __future__ import annotations

import json
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.cli import build_parser, main
from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    TeacherVariation,
    Termination,
)
from simajilord_shogi.policy_calibration import (
    PolicyCalibrationConfig,
    calibrate_teacher_policies,
)
from simajilord_shogi.replay import append_games, load_games

START_SFEN = Board().to_sfen()
MULTIPLE_MATES_SFEN = "3pkp3/9/3B5/9/9/9/9/9/4K4 b GR 1"
SOURCE = "nagisa-v3.1"


def _cp(
    rank: int,
    move: str,
    score: int,
    *,
    bound: TeacherScoreBound = TeacherScoreBound.EXACT,
) -> TeacherVariation:
    return TeacherVariation(
        rank=rank,
        move=move,
        score_kind=TeacherScoreKind.CENTIPAWN,
        bound=bound,
        pv=(move,),
        score_cp=score,
    )


def _mate(rank: int, move: str, plies: int) -> TeacherVariation:
    return TeacherVariation(
        rank=rank,
        move=move,
        score_kind=TeacherScoreKind.MATE,
        bound=TeacherScoreBound.EXACT,
        pv=(move,),
        mate_plies=plies,
    )


def _sample(
    variations: tuple[TeacherVariation, ...] | None,
    *,
    sfen: str = START_SFEN,
    best_move: str = "7g7f",
    policy: dict[str, float] | None = None,
    value: float = 0.25,
) -> PositionSample:
    return PositionSample(
        sfen=sfen,
        ply=0,
        turn=0,
        policy={"7g7f": 1.0},
        root_value=0.0,
        teacher_policy=policy or {"7g7f": 0.5, "2g2f": 0.5},
        teacher_value=value,
        teacher_best_move=best_move,
        teacher_source=SOURCE,
        teacher_variations=variations,
        teacher_policy_temperature=200.0,
        teacher_value_scale=600.0,
    )


def _games() -> list[GameRecord]:
    samples = (
        _sample((_cp(1, "7g7f", 300), _cp(2, "2g2f", -100))),
        _sample((_cp(1, "7g7f", 50), _cp(2, "2g2f", 50))),
        _sample((_cp(1, "7g7f", 75),), policy={"7g7f": 1.0}),
        _sample(
            (_cp(1, "7g7f", 100), _cp(2, "9z9z", 10_000)),
            policy={"7g7f": 1.0},
        ),
        _sample(
            (
                _cp(1, "7g7f", 100),
                _cp(2, "2g2f", 50, bound=TeacherScoreBound.LOWER),
            ),
            policy={"7g7f": 0.8, "2g2f": 0.2},
        ),
        _sample(
            (
                _cp(1, "7g7f", 100),
                _cp(2, "2g2f", 50, bound=TeacherScoreBound.LOWER),
            ),
            best_move="2g2f",
            policy={"7g7f": 0.2, "2g2f": 0.8},
        ),
        _sample(None, policy={"7g7f": 0.6, "2g2f": 0.4}),
    )
    mate_sample = _sample(
        (
            _mate(1, "G*5b", 1),
            _mate(2, "R*5b", 3),
            _cp(3, "6c7d", 5_000),
        ),
        sfen=MULTIPLE_MATES_SFEN,
        best_move="G*5b",
        policy={"G*5b": 0.7, "R*5b": 0.2, "6c7d": 0.1},
        value=0.875,
    )
    return [
        GameRecord(
            initial_sfen=START_SFEN,
            moves=(),
            samples=samples,
            winner=0,
            termination=Termination.RESIGNATION,
        ),
        GameRecord(
            initial_sfen=MULTIPLE_MATES_SFEN,
            moves=(),
            samples=(mate_sample,),
            winner=0,
            termination=Termination.CHECKMATE,
        ),
    ]


def test_variance_normalization_shrinks_degenerate_positions_and_preserves_mates() -> None:
    source_games = _games()
    build = calibrate_teacher_policies(
        source_games,
        PolicyCalibrationConfig(
            mode="variance-normalized",
            prior_strength=4.0,
            standard_deviation_floor=20.0,
            standard_deviation_cap=500.0,
            normalized_temperature=1.0,
            default_prior_standard_deviation=150.0,
        ),
    )

    calibrated = build.games[0].samples
    assert calibrated[0].teacher_policy is not None
    assert calibrated[0].teacher_policy["7g7f"] > calibrated[0].teacher_policy["2g2f"]
    assert calibrated[0].teacher_policy != source_games[0].samples[0].teacher_policy
    assert calibrated[1].teacher_policy == pytest.approx({"7g7f": 0.5, "2g2f": 0.5})
    assert calibrated[2].teacher_policy == {"7g7f": 1.0}
    assert calibrated[3].teacher_policy == {"7g7f": 1.0}
    assert calibrated[4].teacher_policy == {"7g7f": 1.0}
    assert calibrated[5].teacher_policy == pytest.approx({"7g7f": 0.2, "2g2f": 0.8})
    assert calibrated[6].teacher_policy == pytest.approx({"7g7f": 0.6, "2g2f": 0.4})

    mate = build.games[1].samples[0]
    assert mate.teacher_policy == pytest.approx({"G*5b": 0.5, "R*5b": 0.5})
    assert mate.teacher_value == 0.875
    assert mate.teacher_variations == source_games[1].samples[0].teacher_variations
    assert mate.teacher_policy_temperature == 200.0
    assert mate.teacher_value_scale == 600.0

    report = build.report.teachers[SOURCE]
    assert report.global_dispersion.multi_candidate_positions >= 2
    assert report.global_dispersion.prior_standard_deviation_cp > 0
    assert report.cp_positions == 6
    assert report.winning_mate_positions == 1
    assert report.single_candidate_positions >= 3
    assert report.zero_variance_positions >= 3
    assert report.fallback_counts == {
        "best_move_not_eligible": 1,
        "missing_raw_variations": 1,
    }
    assert report.excluded_candidates["bounded"] == 2
    assert report.excluded_candidates["cp_ignored_when_winning_mate"] == 1
    assert report.excluded_candidates["illegal_root_move"] == 1
    assert report.mean_subtraction_max_probability_delta < 1e-12
    assert report.entropy.positions_compared == 8
    assert report.entropy.output_mean_nats is not None
    assert build.report.experimental
    assert build.report.method["mean_subtraction_softmax_invariant"] is True
    json.dumps(build.report.to_dict(), allow_nan=False)


def test_legacy_and_off_modes_leave_policy_and_value_unchanged() -> None:
    games = _games()
    for mode in ("legacy", "off"):
        build = calibrate_teacher_policies(
            games,
            PolicyCalibrationConfig(mode=mode),
        )
        assert [sample.teacher_policy for game in build.games for sample in game.samples] == [
            sample.teacher_policy for game in games for sample in game.samples
        ]
        assert [sample.teacher_value for game in build.games for sample in game.samples] == [
            sample.teacher_value for game in games for sample in game.samples
        ]
        report = build.report.teachers[SOURCE]
        assert report.applied_positions == 0
        assert report.entropy.output_mean_nats == pytest.approx(
            report.entropy.legacy_mean_nats
        )


def test_calibration_cli_writes_replay_and_hashed_provenance_without_overwrite(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "teacher.jsonl"
    output_path = tmp_path / "calibrated.jsonl"
    append_games(input_path, _games())
    args = build_parser().parse_args(
        [
            "calibrate-teacher-policy",
            str(input_path),
            str(output_path),
            "--mode",
            "variance-normalized",
            "--prior-strength",
            "3",
        ]
    )
    assert args.command == "calibrate-teacher-policy"

    argv = [
        "calibrate-teacher-policy",
        str(input_path),
        str(output_path),
        "--mode",
        "variance-normalized",
        "--prior-strength",
        "3",
    ]
    assert main(argv) == 0

    provenance_path = output_path.with_suffix(".jsonl.provenance.json")
    assert output_path.is_file()
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema"] == "meteo-teacher-policy-calibration-v1"
    assert provenance["input_replay"] == str(input_path.resolve())
    assert provenance["output_replay"] == str(output_path.resolve())
    assert len(provenance["input_sha256"]) == 64
    assert len(provenance["output_sha256"]) == 64
    assert provenance["config"]["prior_strength"] == 3.0
    assert provenance["experimental"] is True
    calibrated = load_games(output_path)
    assert calibrated[1].samples[0].teacher_value == 0.875

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(argv)
