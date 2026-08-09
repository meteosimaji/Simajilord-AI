from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

from simajilord_shogi.cli import main
from simajilord_shogi.config import ReanalysisConfig, SearchConfig
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.external_usi import ExternalTeacherPolicy, ExternalUsiTeacher
from simajilord_shogi.reanalysis import (
    reanalyse_game,
    reanalyse_game_external,
    sample_weight,
)
from simajilord_shogi.replay import append_games, load_games

from .test_mcts_game import MATE_IN_ONE_SFEN, MateMoveHasTinyPrior


def test_same_position_deep_reanalysis_records_policy_reversal() -> None:
    actor_policy = {"6c7d": 1.0}
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy=actor_policy,
        root_value=0.0,
        value_target=1.0,
        chosen_move="6c7d",
        actor_best_move="6c7d",
    )
    game = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=(),
        samples=(sample,),
        winner=0,
        termination=Termination.CHECKMATE,
    )
    reanalysis = ReanalysisConfig(
        teacher_simulation_multiplier=2,
        minimum_teacher_simulations=200,
        reanalyse_fraction=1.0,
        reversal_priority=5.0,
    )

    revised = reanalyse_game(
        game,
        MateMoveHasTinyPrior(),
        SearchConfig(simulations=1, root_min_visits=0, dirichlet_fraction=0),
        reanalysis,
    )
    revised_sample = revised.samples[0]
    assert revised_sample.teacher_policy is not None
    teacher_best = max(revised_sample.teacher_policy, key=revised_sample.teacher_policy.__getitem__)
    assert teacher_best == "G*5b"
    assert revised_sample.policy_reversal
    assert revised_sample.discovery_simulation is not None
    assert revised_sample.teacher_best_move == "G*5b"
    assert revised_sample.teacher_regret is not None
    assert revised_sample.teacher_regret > 0
    assert sample_weight(revised_sample, reanalysis) > 5.0


def test_external_tactical_reanalysis_records_versioned_teacher_source(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_tactical_usi.py"
    script.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-tactical', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print(
            'info depth 8 multipv 1 score cp 1200 nodes 321 time 10 nps 32100 pv G*5b',
            flush=True,
        )
        print(
            'info depth 8 multipv 2 score cp -100 nodes 321 time 10 nps 32100 pv 6c7d',
            flush=True,
        )
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
        chosen_move="G*5b",
        actor_best_move="G*5b",
        actor_simulations=3,
    )
    game = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=("G*5b",),
        samples=(sample,),
        winner=0,
        termination=Termination.CHECKMATE,
    )
    policy = ExternalTeacherPolicy(
        policy_id="gikou2-v2.0.2",
        name="Gikou 2",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )
    config = ReanalysisConfig(reanalyse_fraction=1.0)

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=321,
        multipv=2,
        policy_temperature=250.0,
        value_scale=750.0,
    ) as teacher:
        revised = reanalyse_game_external(
            game,
            teacher,
            config,
            selection="tactical",
            teacher_context="gikou-tactics",
        )

    revised_sample = revised.samples[0]
    assert revised_sample.teacher_source == "gikou2-v2.0.2"
    assert revised_sample.teacher_context == "gikou-tactics"
    assert revised_sample.teacher_nodes == 321
    assert revised_sample.teacher_depth_ratio == 107
    assert revised_sample.teacher_time_ms == 10
    assert revised_sample.teacher_nps == 32100
    assert revised_sample.teacher_depth == 8
    assert revised_sample.teacher_best_move == "G*5b"
    assert revised_sample.teacher_policy is not None
    assert revised_sample.teacher_policy["G*5b"] > revised_sample.teacher_policy["6c7d"]
    assert revised_sample.teacher_variations is not None
    assert revised_sample.teacher_variations[0].pv == ("G*5b",)
    assert revised_sample.teacher_policy_temperature == 250.0
    assert revised_sample.teacher_value_scale == 750.0
    assert not revised_sample.policy_reversal

    serialized = json.loads(json.dumps(revised.to_dict()))
    restored = GameRecord.from_dict(serialized).samples[0]
    assert restored.teacher_variations == revised_sample.teacher_variations
    assert restored.teacher_policy_temperature == 250.0
    assert restored.teacher_value_scale == 750.0


def test_external_reanalysis_cli_defaults_to_c600_and_records_both_scales(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "fake_tactical_usi.py"
    script.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-tactical', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print('info depth 8 multipv 1 score cp 1200 nodes 321 pv G*5b', flush=True)
        print('info depth 8 multipv 2 score cp -100 nodes 321 pv 6c7d', flush=True)
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )
    source = tmp_path / "actor.jsonl"
    output = tmp_path / "teacher.jsonl"
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=0.0,
        chosen_move="G*5b",
        actor_best_move="G*5b",
    )
    append_games(
        source,
        [
            GameRecord(
                initial_sfen=MATE_IN_ONE_SFEN,
                moves=("G*5b",),
                samples=(sample,),
                winner=0,
                termination=Termination.CHECKMATE,
            )
        ],
    )

    assert (
        main(
            [
                "reanalyse-usi",
                str(source),
                str(output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--selection",
                "all",
                "--fraction",
                "1",
                "--nodes",
                "321",
                "--multipv",
                "2",
            ]
        )
        == 0
    )

    revised = load_games(output)[0].samples[0]
    assert revised.teacher_value == pytest.approx(0.7615941559557649)
    assert revised.teacher_value_scale == 1_200.0
    provenance_path = output.with_suffix(".jsonl.provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == provenance
    assert provenance["teacher_ponanza_coefficient"] == 600.0
    assert provenance["teacher_value_tanh_denominator"] == 1_200.0
    assert provenance["teacher_value_scale"] == 1_200.0
    assert provenance["teacher_value_scale_input_convention"] == (
        "default_ponanza_coefficient"
    )
    assert provenance["teacher_policy_temperature"] == 200.0
    assert provenance["policy_and_value_calibration_are_independent"] is True
    assert provenance["history_mode"] == "game_prefix"
    assert provenance["history_mode_missing_field_means"] == "board_only"
    assert provenance["history_source"] == "GameRecord.initial_sfen + moves[:PositionSample.ply]"

    legacy_output = tmp_path / "teacher-legacy-scale600.jsonl"
    assert (
        main(
            [
                "reanalyse-usi",
                str(source),
                str(legacy_output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--selection",
                "all",
                "--fraction",
                "1",
                "--nodes",
                "321",
                "--multipv",
                "2",
                "--teacher-value-scale",
                "600",
            ]
        )
        == 0
    )
    legacy_sample = load_games(legacy_output)[0].samples[0]
    assert legacy_sample.teacher_value == pytest.approx(0.9640275800758169)
    assert legacy_sample.teacher_value_scale == 600.0
    legacy_provenance = json.loads(
        legacy_output.with_suffix(".jsonl.provenance.json").read_text(encoding="utf-8")
    )
    assert json.loads(capsys.readouterr().out) == legacy_provenance
    assert legacy_provenance["teacher_ponanza_coefficient"] == 300.0
    assert legacy_provenance["teacher_value_tanh_denominator"] == 600.0
    assert legacy_provenance["teacher_value_scale_input_convention"] == (
        "explicit_legacy_tanh_denominator"
    )


def test_external_reanalysis_sends_the_exact_prefix_to_each_sample(tmp_path: Path) -> None:
    command_log = tmp_path / "position-commands.txt"
    script = tmp_path / "fake_history_usi.py"
    script.write_text(
        f"""import sys
last_position = ''
log_path = {str(command_log)!r}
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-history-reanalysis', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 1', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('position '):
        last_position = command
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(command + '\\n')
    elif command.startswith('go '):
        if last_position == 'position startpos':
            move = '7g7f'
        elif last_position == 'position startpos moves 7g7f':
            move = '3c3d'
        else:
            raise RuntimeError(f'unexpected position command: {{last_position}}')
        print(f'info depth 4 multipv 1 score cp 100 nodes 1 pv {{move}}', flush=True)
        print(f'bestmove {{move}}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )
    board = Board()
    initial_sfen = board.to_sfen()
    moves = ("7g7f", "3c3d")
    samples: list[PositionSample] = []
    for ply, move_usi in enumerate(moves):
        samples.append(
            PositionSample(
                sfen=board.to_sfen(),
                ply=ply,
                turn=board.turn.value,
                policy={move_usi: 1.0},
                root_value=0.0,
                chosen_move=move_usi,
                actor_best_move=move_usi,
                actor_simulations=1,
            )
        )
        move = Move.from_usi(move_usi)
        assert board.is_legal_move(move)
        board.apply_move(move)
    game = GameRecord(
        initial_sfen=initial_sfen,
        moves=moves,
        samples=tuple(samples),
        winner=None,
        termination=Termination.MAX_PLIES,
    )
    policy = ExternalTeacherPolicy(
        policy_id="test-history-reanalysis",
        name="history-sensitive",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=1,
    ) as teacher:
        revised = reanalyse_game_external(
            game,
            teacher,
            ReanalysisConfig(reanalyse_fraction=1.0),
            selection="all",
        )

    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "position startpos",
        "position startpos moves 7g7f",
    ]
    assert [sample.teacher_best_move for sample in revised.samples] == ["7g7f", "3c3d"]
