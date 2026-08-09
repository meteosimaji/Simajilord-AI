from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

from simajilord_shogi.domain import (
    GameRecord,
    PositionSample,
    TeacherScoreBound,
    TeacherScoreKind,
    Termination,
)
from simajilord_shogi.external_usi import (
    ExternalTeacherPolicy,
    ExternalUsiTeacher,
    UsiHistoryMode,
    UsiOptionValueVerification,
    UsiPositionHistory,
)

MATE_IN_ONE_SFEN = "4k4/9/3B5/9/9/9/9/9/4K4 b G 1"
MULTIPLE_MATES_SFEN = "3pkp3/9/3B5/9/9/9/9/9/4K4 b GR 1"


def _fake_multipv_usi(path: Path) -> None:
    path.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-multipv', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print(
            'info depth 8 multipv 1 score cp 1200 lowerbound nodes 321 '
            'time 10 nps 32100 pv G*5b 5a4b',
            flush=True,
        )
        print(
            'info depth 8 multipv 2 score cp -100 upperbound nodes 321 '
            'time 10 nps 32100 pv 6c7d 5a4b',
            flush=True,
        )
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _fake_mate_multipv_usi(path: Path) -> None:
    path.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-mates', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 4', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print('info depth 12 multipv 1 score mate 1 lowerbound pv G*5b 5a4b', flush=True)
        print('info depth 12 multipv 2 score mate 3 upperbound pv R*5b 5a4b', flush=True)
        print('info depth 12 multipv 3 score mate + pv 6c7d 5a4b', flush=True)
        print('info depth 12 multipv 4 score mate -5 pv 6c8e 5a4b', flush=True)
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _fake_history_sensitive_usi(path: Path, command_log: Path) -> None:
    path.write_text(
        f"""import sys
last_position = ''
log_path = {str(command_log)!r}
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-history-sensitive', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 2', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('position '):
        last_position = command
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(command + '\\n')
    elif command.startswith('go '):
        move = '3c3d' if last_position == 'position startpos moves 7g7f' else 'G*5b'
        print(f'info depth 4 score cp 100 nodes 1 pv {{move}}', flush=True)
        print(f'bestmove {{move}}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _fake_yaneuraou_getoption_usi(
    path: Path,
    *,
    force_fv_scale_after_ready: int | None = None,
) -> None:
    path.write_text(
        f"""import sys
options = {{'MultiPV': '1', 'USI_Hash': '16', 'FV_SCALE': '16', 'Threads': '1'}}
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-yaneuraou-getoption', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('option name USI_Hash type spin default 16 min 1 max 1048576', flush=True)
        print('option name FV_SCALE type spin default 16 min 1 max 128', flush=True)
        print('option name Threads type spin default 1 min 1 max 512', flush=True)
        print('usiok', flush=True)
    elif command.startswith('setoption name '):
        setting = command.removeprefix('setoption name ')
        name, value = setting.split(' value ', 1)
        options[name] = value
    elif command == 'isready':
        if {force_fv_scale_after_ready!r} is not None:
            options['FV_SCALE'] = str({force_fv_scale_after_ready!r})
        print('NNUE hash mismatch: fixture warning', file=sys.stderr, flush=True)
        print('readyok', flush=True)
    elif command.startswith('getoption '):
        name = command.removeprefix('getoption ')
        if name in options:
            print(f'Options[{{name}}] = {{options[name]}}', flush=True)
        else:
            print(f'No such option: {{name}}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _board_after(initial_sfen: str, *moves: str) -> Board:
    board = Board(initial_sfen)
    for move_usi in moves:
        move = Move.from_usi(move_usi)
        assert board.is_legal_move(move)
        board.apply_move(move)
    return board


def test_position_history_serializes_startpos_and_arbitrary_sfen_prefixes() -> None:
    start = Board().to_sfen()
    after_black = _board_after(start, "7g7f")
    startpos_history = UsiPositionHistory(start, ("7g7f",), after_black.to_sfen())

    assert startpos_history.mode is UsiHistoryMode.GAME_PREFIX
    assert startpos_history.position_command() == "position startpos moves 7g7f"

    arbitrary_initial = after_black.to_sfen()
    after_white = _board_after(arbitrary_initial, "3c3d")
    sfen_history = UsiPositionHistory(
        arbitrary_initial,
        ("3c3d",),
        after_white.to_sfen(),
    )

    assert sfen_history.position_command() == (f"position sfen {arbitrary_initial} moves 3c3d")


def test_position_history_and_target_mismatches_fail_closed() -> None:
    start = Board().to_sfen()
    after_black = _board_after(start, "7g7f")
    with pytest.raises(ValueError, match="illegal history move"):
        UsiPositionHistory(start, ("7g7e",), start)
    with pytest.raises(ValueError, match="does not match target position"):
        UsiPositionHistory(start, ("7g7f",), start)

    wrong_turn = PositionSample(
        sfen=after_black.to_sfen(),
        ply=1,
        turn=0,
        policy={"3c3d": 1.0},
        root_value=0.0,
    )
    game = GameRecord(
        initial_sfen=start,
        moves=("7g7f",),
        samples=(wrong_turn,),
        winner=None,
        termination=Termination.MAX_PLIES,
    )
    with pytest.raises(ValueError, match="does not match sample turn"):
        UsiPositionHistory.from_game(game, wrong_turn)
    with pytest.raises(ValueError, match="outside recorded move range"):
        UsiPositionHistory.from_game(game, replace(wrong_turn, ply=2, turn=1))

    recorded_moves = ("7g7f", "3c3d")
    wrong_chosen_move = replace(wrong_turn, turn=1, chosen_move="8c8d")
    game_with_reply = replace(game, moves=recorded_moves, samples=(wrong_chosen_move,))
    with pytest.raises(ValueError, match="does not match recorded move"):
        UsiPositionHistory.from_game(game_with_reply, wrong_chosen_move)


def test_history_aware_target_sends_prefix_and_board_only_api_remains_compatible(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_history.py"
    command_log = tmp_path / "positions.txt"
    _fake_history_sensitive_usi(script, command_log)
    policy = ExternalTeacherPolicy(
        policy_id="test-history",
        name="history-sensitive",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )
    start = Board().to_sfen()
    after_black = _board_after(start, "7g7f")
    history = UsiPositionHistory(start, ("7g7f",), after_black.to_sfen())

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=1,
    ) as teacher:
        history_target = teacher.training_target_with_history(history)
        board_only_target = teacher.training_target(Board(MATE_IN_ONE_SFEN))

    assert history_target.bestmove == "3c3d"
    assert board_only_target.bestmove == "G*5b"
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "position startpos moves 7g7f",
        f"position sfen {MATE_IN_ONE_SFEN}",
    ]


def test_unknown_teacher_rights_fail_closed() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-unreviewed",
        name="unreviewed-free-model",
        source="contract-not-reviewed",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )

    with pytest.raises(PermissionError):
        ExternalUsiTeacher(["/does/not/matter"], policy, nodes=1)


def test_paid_teacher_is_disabled_even_when_output_training_is_allowed() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-commercial",
        name="commercial-model",
        source="commercial-contract",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
        requires_payment=True,
    )

    with pytest.raises(PermissionError, match="free-only policy"):
        ExternalUsiTeacher(["/does/not/matter"], policy, nodes=1)


def test_local_only_teacher_policy_cannot_claim_redistribution() -> None:
    with pytest.raises(ValueError, match="cannot be marked redistributable"):
        ExternalTeacherPolicy(
            policy_id="test-local-only-conflict",
            name="local-only-conflict",
            source="test",
            analysis_allowed=True,
            training_outputs_allowed=True,
            redistribution_allowed=True,
            training_outputs_local_only=True,
        )


def test_missing_approved_engine_is_reported() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-local",
        name="local-teacher",
        source="local-license",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=False,
    )
    teacher = ExternalUsiTeacher(["/does/not/exist"], policy, nodes=1)

    with pytest.raises(FileNotFoundError):
        teacher.evaluate(Board())


def test_analysis_only_engine_cannot_silently_become_a_teacher() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-analysis-only",
        name="analysis-only",
        source="reviewed-free-license",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )

    ExternalUsiTeacher(["/does/not/matter"], policy, nodes=1, training_use=False)
    with pytest.raises(PermissionError, match="training-data generation"):
        ExternalUsiTeacher(["/does/not/matter"], policy, nodes=1, training_use=True)


def test_training_target_preserves_legal_multipv_scores_and_nodes(tmp_path: Path) -> None:
    script = tmp_path / "fake_multipv.py"
    _fake_multipv_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-multipv",
        name="fake-multipv",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=321,
        multipv=2,
    ) as teacher:
        target = teacher.training_target(Board(MATE_IN_ONE_SFEN))

    assert target.bestmove == "G*5b"
    assert target.scores == (("G*5b", 1200), ("6c7d", -100))
    assert target.nodes == 321
    assert target.time_ms == 10
    assert target.nps == 32100
    assert target.depth == 8
    assert target.elapsed_seconds > 0
    assert sum(target.policy.values()) == pytest.approx(1.0)
    assert target.policy["G*5b"] > target.policy["6c7d"]
    assert target.value == pytest.approx(0.7615941559557649)
    first, second = target.candidates
    assert first.rank == 1
    assert first.score_kind is TeacherScoreKind.CENTIPAWN
    assert first.score_cp == 1200
    assert first.bound is TeacherScoreBound.LOWER
    assert first.pv == ("G*5b", "5a4b")
    assert second.bound is TeacherScoreBound.UPPER
    assert second.pv == ("6c7d", "5a4b")
    assert target.policy_temperature == 200.0
    assert target.value_scale == 1_200.0


def test_teacher_scales_are_configurable_and_invalid_values_fail_closed(tmp_path: Path) -> None:
    script = tmp_path / "fake_multipv.py"
    _fake_multipv_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-scale",
        name="fake-scale",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=321,
        multipv=2,
        policy_temperature=1_000.0,
        value_scale=1_200.0,
    ) as teacher:
        target = teacher.training_target(Board(MATE_IN_ONE_SFEN))

    assert target.policy["G*5b"] < 0.8
    assert target.value == pytest.approx(0.7615941559557649)
    assert target.policy_temperature == 1_000.0
    assert target.value_scale == 1_200.0

    for argument, value in (
        ("policy_temperature", 0.0),
        ("policy_temperature", float("nan")),
        ("value_scale", -1.0),
        ("value_scale", float("inf")),
    ):
        with pytest.raises(ValueError, match=argument):
            ExternalUsiTeacher(
                ["/does/not/matter"],
                policy,
                nodes=1,
                **{argument: value},
            )


def test_mate_scores_remain_separate_from_centipawn_scaling(tmp_path: Path) -> None:
    script = tmp_path / "fake_mates.py"
    _fake_mate_multipv_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-mates",
        name="fake-mates",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=4,
        policy_temperature=0.01,
        value_scale=1.0,
    ) as teacher:
        target = teacher.training_target(Board(MULTIPLE_MATES_SFEN))

    assert target.value == 1.0
    assert target.policy == pytest.approx({"G*5b": 1 / 3, "R*5b": 1 / 3, "6c7d": 1 / 3})
    assert target.candidates[0].mate_plies == 1
    assert target.candidates[1].mate_plies == 3
    assert target.candidates[2].mate_unknown_sign == 1
    assert target.candidates[3].mate_plies == -5
    assert dict(target.move_values)["6c8e"] == -1.0


def test_reserved_multipv_option_cannot_be_shadowed() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-shadow",
        name="fake",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with pytest.raises(ValueError, match="MultiPV is reserved"):
        ExternalUsiTeacher(
            ["/does/not/matter"],
            policy,
            nodes=1,
            options={"multipv": 99},
        )


def test_relative_engine_path_is_resolved_before_changing_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_dir = tmp_path / "bin"
    engine_dir.mkdir()
    script = engine_dir / "fake-multipv"
    _fake_multipv_usi(script)
    script.write_text(
        "#!/usr/bin/env python3\n" + script.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    policy = ExternalTeacherPolicy(
        policy_id="test-relative-path",
        name="relative-path-engine",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        ["bin/fake-multipv"],
        policy,
        nodes=321,
        multipv=2,
    ) as teacher:
        target = teacher.training_target(Board(MATE_IN_ONE_SFEN))

    assert target.bestmove == "G*5b"


def test_requested_option_must_match_engine_declaration_not_similar_name(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_yaneuraou.py"
    _fake_yaneuraou_getoption_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-option-name",
        name="fake-yaneuraou",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )
    teacher = ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        options={"Hash": 64},
    )

    with pytest.raises(ValueError, match="did not declare requested option 'Hash'"):
        teacher.start()

    assert teacher.process is None


def test_yaneuraou_getoption_verifies_values_and_hashes_complete_startup_transcript(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_yaneuraou.py"
    sidecar = tmp_path / "startup.provenance.json"
    _fake_yaneuraou_getoption_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-getoption",
        name="fake-yaneuraou",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=8,
        options={"USI_Hash": 64, "FV_SCALE": 40, "Threads": 1},
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
        startup_provenance_path=sidecar,
    ) as teacher:
        startup = teacher.startup_provenance

    assert startup.schema == "meteo-usi-startup-provenance-v1"
    assert startup.identity_lines == ("id name fake-yaneuraou-getoption",)
    assert {declaration.name for declaration in startup.option_declarations} == {
        "MultiPV",
        "USI_Hash",
        "FV_SCALE",
        "Threads",
    }
    applied = {option.name: option for option in startup.applied_options}
    assert applied["MultiPV"].applied_value == "8"
    assert applied["USI_Hash"].applied_value == "64"
    assert applied["FV_SCALE"].applied_value == "40"
    assert applied["Threads"].applied_value == "1"
    assert all(option.verified for option in applied.values())
    assert startup.stderr_lines == ("NNUE hash mismatch: fixture warning",)
    assert startup.warnings == (("stderr", "NNUE hash mismatch: fixture warning"),)
    assert startup.stdout_sha256 == hashlib.sha256(
        "\n".join(startup.stdout_lines).encode() + b"\n"
    ).hexdigest()
    assert startup.stderr_sha256 == hashlib.sha256(
        b"NNUE hash mismatch: fixture warning\n"
    ).hexdigest()

    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    recorded_hash = saved.pop("provenance_sha256")
    assert recorded_hash == hashlib.sha256(
        json.dumps(
            saved,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert saved["warnings"] == [["stderr", "NNUE hash mismatch: fixture warning"]]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        startup.write_create_only(sidecar)


def test_yaneuraou_getoption_rejects_post_ready_value_override(tmp_path: Path) -> None:
    script = tmp_path / "fake_yaneuraou.py"
    sidecar = tmp_path / "must-not-exist.json"
    _fake_yaneuraou_getoption_usi(script, force_fv_scale_after_ready=16)
    policy = ExternalTeacherPolicy(
        policy_id="test-getoption-mismatch",
        name="fake-yaneuraou",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )
    teacher = ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        options={"FV_SCALE": 40},
        option_value_verification=UsiOptionValueVerification.YANEURAOU_GETOPTION,
        startup_provenance_path=sidecar,
    )

    with pytest.raises(
        RuntimeError,
        match=r"FV_SCALE.*requested='40' current='16'",
    ):
        teacher.start()

    assert teacher.process is None
    assert not sidecar.exists()


def test_standard_usi_teacher_keeps_legacy_compatibility_without_getoption(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_standard_usi.py"
    _fake_multipv_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-standard-usi",
        name="fake-standard-usi",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=2,
    ) as teacher:
        startup = teacher.startup_provenance

    assert startup.option_value_verification == "none"
    assert startup.applied_options[0].name == "MultiPV"
    assert startup.applied_options[0].requested_value == "2"
    assert startup.applied_options[0].applied_value is None
    assert startup.applied_options[0].verified is False


def test_option_names_cannot_collide_by_case() -> None:
    policy = ExternalTeacherPolicy(
        policy_id="test-option-collision",
        name="fake",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with pytest.raises(ValueError, match="duplicate USI option name ignoring case"):
        ExternalUsiTeacher(
            ["/does/not/matter"],
            policy,
            nodes=1,
            options={"USI_Hash": 64, "usi_hash": 32},
        )
