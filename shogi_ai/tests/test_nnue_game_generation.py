from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi import nnue_game_runner
from simajilord_shogi.domain import Termination
from simajilord_shogi.external_usi import ExternalTeacherPolicy
from simajilord_shogi.nnue_game_generation import (
    NNUE_GAME_GENERATION_SCHEMA,
    NnueGameGenerationConfig,
    NnueGameGenerator,
    NnueGenerationOpening,
    UsiEngineSpec,
)
from simajilord_shogi.nnue_runtime import RuntimeProfile, stage_nnue_runtime

MATE_IN_ONE_SFEN = "4k4/9/3B5/9/9/9/9/9/4K4 b G 1"
SECOND_MATE_IN_ONE_SFEN = "3pkp3/9/3B5/9/9/9/9/9/4K4 b GR 1"
BLACK_CSA27_DECLARATION_SFEN = (
    "K+N5+L1/G+L+P+B1+R+P2/3+P2G2/9/2+p+n5/3s2+ss1/3+p+p1+s1+r/7g+n/6g+nk b 2L8Pb4p 1"
)
REPETITION_MOVES = (
    "5i5h",
    "5a5b",
    "5h5i",
    "5b5a",
) * 3


def _fake_full_game_usi(path: Path) -> None:
    path.write_text(
        """from __future__ import annotations

import sys
import time

log_path = sys.argv[1]
delay = float(sys.argv[2])
resign_with_pv = sys.argv[3] == 'resign-with-pv'
position = ''
options = {'MultiPV': '1', 'Threads': '1'}


def selected_move(command: str) -> str:
    if command.startswith('position sfen 4k4/') or command.startswith(
        'position sfen 3pkp3/'
    ):
        return 'G*5b'
    moves = command.split(' moves ', 1)[1].split() if ' moves ' in command else []
    cycle = ('5i5h', '5a5b', '5h5i', '5b5a')
    return cycle[len(moves) % len(cycle)]


for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-full-game-nnue', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 1', flush=True)
        print('option name Threads type spin default 1 min 1 max 64', flush=True)
        print('usiok', flush=True)
    elif command.startswith('setoption name '):
        setting = command.removeprefix('setoption name ')
        name, value = setting.split(' value ', 1)
        options[name] = value
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('getoption '):
        name = command.removeprefix('getoption ')
        if name not in options:
            print(f'No such option: {name}', flush=True)
        else:
            print(f'Options[{name}] = {options[name]}', flush=True)
    elif command.startswith('position '):
        position = command
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(f'{command}\\n')
    elif command.startswith('go '):
        time.sleep(delay)
        move = selected_move(position)
        score = 'mate 1' if move == 'G*5b' else 'cp 20'
        print(
            f'info depth 4 nodes 17 time 2 nps 8500 score {score} pv {move}',
            flush=True,
        )
        print('bestmove resign' if resign_with_pv else f'bestmove {move}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_meteo_runtime_export(path: Path) -> Path:
    path.mkdir()
    nn_bin = path / "nn.bin"
    progress_bin = path / "progress.bin"
    eval_options = path / "eval_options.txt"
    nn_bin.write_bytes(b"Meteo NNUE fixture\0weights")
    progress_bin.write_bytes(b"progress-8kpabs-v1")
    eval_options.write_text(
        "LS_BUCKET_MODE progress8kpabs\nLS_PROGRESS_COEFF progress.bin\nFV_SCALE 16\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "meteo-nagisa-mlx-yaneuraou-export-v1",
        "architecture": "HalfKAv2_hm^ + progress LayerStack",
        "optimizer_step": 123,
        "local_only": True,
        "nn_bin": {"bytes": nn_bin.stat().st_size, "sha256": _sha256_file(nn_bin)},
        "progress_bin_sha256": _sha256_file(progress_bin),
        "routing": "progress8kpabs",
        "yaneuraou_fv_scale": 16,
    }
    (path / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _fake_meteo_yaneuraou(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
from pathlib import Path
import sys

options = {
    'MultiPV': '1',
    'BookFile': 'standard_book.db',
    'EnteringKingRule': 'CSARule27',
    'EvalDir': 'eval',
    'FV_SCALE': '16',
    'USI_Hash': '16',
    'LS_BUCKET_MODE': 'progress8kpabs',
    'LS_PROGRESS_COEFF': 'progress.bin',
    'PvInterval': '300',
    'Threads': '1',
    'USI_OwnBook': 'true',
}

for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-meteo-yaneuraou', flush=True)
        print('id author fixture', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print(
            'option name BookFile type combo default standard_book.db '
            'var no_book var standard_book.db',
            flush=True,
        )
        print(
            'option name EnteringKingRule type combo default CSARule27 '
            'var CSARule24 var CSARule27',
            flush=True,
        )
        print('option name EvalDir type string default eval', flush=True)
        print('option name FV_SCALE type spin default 16 min 1 max 128', flush=True)
        print('option name USI_Hash type spin default 16 min 1 max 1048576', flush=True)
        print(
            'option name LS_BUCKET_MODE type combo default progress8kpabs '
            'var progress8kpabs',
            flush=True,
        )
        print('option name LS_PROGRESS_COEFF type string default progress.bin', flush=True)
        print('option name PvInterval type spin default 300 min 0 max 100000', flush=True)
        print('option name Threads type spin default 1 min 1 max 512', flush=True)
        print('option name USI_OwnBook type check default true', flush=True)
        print('usiok', flush=True)
    elif command.startswith('setoption name '):
        setting = command.removeprefix('setoption name ')
        name, value = setting.split(' value ', 1)
        options[name] = value
    elif command == 'isready':
        required = (
            Path(options['EvalDir']) / 'nn.bin',
            Path(options['EvalDir']) / options['LS_PROGRESS_COEFF'],
        )
        if not all(item.is_file() for item in required):
            print('error NNUE runtime artifact not found', flush=True)
        print('readyok', flush=True)
    elif command.startswith('getoption '):
        name = command.removeprefix('getoption ')
        if name in options:
            print(f'Options[{name}] = {options[name]}', flush=True)
        else:
            print(f'No such option: {name}', flush=True)
    elif command.startswith('go nodes '):
        nodes = command.split()[2]
        print(
            f'info depth 4 seldepth 5 multipv 1 score cp -42 '
            f'nodes {nodes} time 2 nps 8500 pv 7g7f',
            flush=True,
        )
        print('bestmove 7g7f', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _fixture_policy(source: str) -> ExternalTeacherPolicy:
    return ExternalTeacherPolicy(
        policy_id=f"fixture-rights-{source}",
        name=f"fixture {source}",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=False,
    )


def _spec(
    script: Path,
    log: Path,
    source: str,
    *,
    delay: float = 0.0,
    resign_with_pv: bool = False,
) -> UsiEngineSpec:
    return UsiEngineSpec(
        actor_source=source,
        command=(
            sys.executable,
            str(script),
            str(log),
            str(delay),
            "resign-with-pv" if resign_with_pv else "normal",
        ),
        policy=_fixture_policy(source),
        nodes=17,
        timeout_seconds=5.0,
        artifact_paths=(script,),
    )


def _runner_config(script: Path, log: Path) -> dict[str, object]:
    pinned_artifacts = sorted(
        (Path(sys.executable), script),
        key=lambda artifact: str(artifact.resolve()),
    )

    def engine(actor_source: str, rights_id: str) -> dict[str, object]:
        return {
            "actor_source": actor_source,
            "rights_id": rights_id,
            "allow_limited_local": False,
            "command": [sys.executable, str(script), str(log), "0", "normal"],
            "nodes": 17,
            "options": [{"name": "Threads", "value": 1}],
            "timeout_seconds": 5.0,
            "working_directory": None,
            "policy_temperature": 200.0,
            "value_scale": 1_200.0,
            "option_value_verification": "yaneuraou_getoption",
            "expected_fatal_startup_diagnostics": [],
            "artifact_identities": [
                {
                    "path": str(artifact.resolve()),
                    "sha256": _sha256_file(artifact.resolve()),
                }
                for artifact in pinned_artifacts
            ],
        }

    return {
        "schema": nnue_game_runner.NNUE_GAME_RUNNER_CONFIG_SCHEMA,
        "engines": [
            engine("nagisa-v3.1-fixture", "nagisa-v3.1"),
            engine("suisho5-fixture", "suisho5"),
        ],
        "generation": {
            "generation_id": "runner-integration-mate",
            "max_plies": 1,
            "parallelism": 1,
            "openings": [{"initial_sfen": MATE_IN_ONE_SFEN, "moves": []}],
        },
    }


def test_parallel_color_swapped_generation_overrides_resign_and_receipts_metrics(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    first = _spec(
        script,
        command_log,
        "meteo-nnue-candidate",
        delay=0.05,
        resign_with_pv=True,
    )
    second = _spec(
        script,
        command_log,
        "nagisa-v3.1",
        delay=0.05,
        resign_with_pv=True,
    )
    result = NnueGameGenerator(
        first,
        second,
        NnueGameGenerationConfig(
            generation_id="fixture-parallel-mates",
            openings=(MATE_IN_ONE_SFEN, SECOND_MATE_IN_ONE_SFEN),
            max_plies=1,
            parallelism=2,
        ),
    ).generate()

    assert len(result.games) == 4
    assert len(result.strong_wdl_records()) == 4
    assert len(result.position_source_records()) == 4
    for opening_index in range(2):
        black_first, black_second = result.games[2 * opening_index : 2 * opening_index + 2]
        assert black_first.receipt.black_actor_source == "meteo-nnue-candidate"
        assert black_second.receipt.black_actor_source == "nagisa-v3.1"
        assert black_first.receipt.color_swap_leg == 0
        assert black_second.receipt.color_swap_leg == 1
        for game in (black_first, black_second):
            assert game.record.moves == ("G*5b",)
            assert game.record.termination is Termination.CHECKMATE
            assert game.record.winner == 0
            assert game.record.samples[0].value_target == 1.0
            assert game.receipt.rules_terminal_trajectory
            assert game.receipt.strong_wdl_label_allowed
            assert game.receipt.strong_wdl_label_black == 1.0
            assert not game.receipt.safety_truncated
            assert game.receipt.plies[0].resignation_overridden
            assert game.receipt.plies[0].nodes == 17
            assert game.receipt.plies[0].nps == 8_500
            assert game.receipt.plies[0].search_seconds is not None
            assert (
                game.receipt.plies[0].resolve_history_prefix(
                    initial_sfen=game.receipt.initial_sfen,
                    full_game_moves=game.receipt.full_game_moves,
                )
                == ()
            )
            assert game.receipt.plies[0].termination is Termination.CHECKMATE
            assert game.receipt.plies[0].strong_value_target == 1.0

    metrics = result.receipt.metrics
    assert result.receipt.schema == NNUE_GAME_GENERATION_SCHEMA
    assert result.receipt.no_resignation_adjudication
    assert result.receipt.no_evaluation_adjudication
    assert result.receipt.max_plies_is_not_a_strong_wdl_label
    assert metrics.requested_parallelism == 2
    assert metrics.effective_parallelism == 2
    assert metrics.peak_concurrent_games == 2
    assert metrics.games == 4
    assert metrics.rules_complete_games == 4
    assert metrics.incomplete_games == 0
    assert metrics.positions == 4
    assert metrics.game_seconds_median > 0
    assert metrics.game_seconds_p95 >= metrics.game_seconds_median
    assert metrics.move_search_seconds_median > 0
    assert metrics.move_search_seconds_p95 >= metrics.move_search_seconds_median
    assert metrics.games_per_hour > 0
    assert metrics.positions_per_second > 0
    assert {actor.actor_source for actor in metrics.actors} == {
        "meteo-nnue-candidate",
        "nagisa-v3.1",
    }
    assert all(actor.positions == 2 for actor in metrics.actors)
    assert all(actor.nodes == 34 for actor in metrics.actors)
    assert all(actor.reported_nps_median == 8_500 for actor in metrics.actors)
    assert len(result.receipt.engine_startups) == 4
    assert {startup.worker_index for startup in result.receipt.engine_startups} == {0, 1}
    assert {startup.engine_slot for startup in result.receipt.engine_startups} == {
        "first",
        "second",
    }
    assert all(
        startup.executable.sha256 == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        for startup in result.receipt.engine_startups
    )
    expected_script_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    for profile in result.receipt.engine_profiles:
        artifacts = profile["artifacts"]
        assert isinstance(artifacts, list)
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert isinstance(artifact, dict)
        assert artifact["sha256"] == expected_script_hash

    receipt_path = tmp_path / "generation-receipt.json"
    result.receipt.write_create_only(receipt_path)
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_hash = stored.pop("payload_sha256")
    stored.pop("payload_sha256_scope")
    canonical = json.dumps(
        stored,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert expected_hash == hashlib.sha256(canonical).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        result.receipt.write_create_only(receipt_path)


def test_full_prefix_repetition_is_a_rules_terminal_strong_draw(tmp_path: Path) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    first = _spec(script, command_log, "first-nnue")
    second = _spec(script, command_log, "second-nnue")
    result = NnueGameGenerator(
        first,
        second,
        NnueGameGenerationConfig(
            generation_id="fixture-repetition",
            openings=(Board().to_sfen(),),
            max_plies=12,
            parallelism=1,
        ),
    ).generate()

    assert len(result.games) == 2
    assert len(result.strong_wdl_records()) == 2
    assert result.receipt.metrics.rules_complete_games == 2
    for game in result.games:
        assert game.record.moves == REPETITION_MOVES
        assert game.record.termination is Termination.REPETITION
        assert game.record.winner is None
        assert game.receipt.rules_terminal_trajectory
        assert game.receipt.strong_wdl_label_allowed
        assert game.receipt.strong_wdl_label_black == 0.5
        assert len(game.receipt.plies) == 12
        assert (
            game.receipt.plies[-1].resolve_history_prefix(
                initial_sfen=game.receipt.initial_sfen,
                full_game_moves=game.receipt.full_game_moves,
            )
            == REPETITION_MOVES[:-1]
        )
        assert all(ply.termination is Termination.REPETITION for ply in game.receipt.plies)
        assert all(ply.strong_value_target == 0.0 for ply in game.receipt.plies)
        assert all(sample.value_target == 0.0 for sample in game.record.samples)

    first_leg_sources = tuple(ply.actor_source for ply in result.games[0].receipt.plies[:4])
    second_leg_sources = tuple(ply.actor_source for ply in result.games[1].receipt.plies[:4])
    assert first_leg_sources == (
        "first-nnue",
        "second-nnue",
        "first-nnue",
        "second-nnue",
    )
    assert second_leg_sources == (
        "second-nnue",
        "first-nnue",
        "second-nnue",
        "first-nnue",
    )
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert "position startpos" in commands
    assert f"position startpos moves {' '.join(REPETITION_MOVES[:-1])}" in commands


def test_safety_cutoff_preserves_opening_history_but_forbids_strong_wdl(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    opening = NnueGenerationOpening(
        initial_sfen=Board().to_sfen(),
        moves=("7g7f",),
    )
    result = NnueGameGenerator(
        _spec(script, command_log, "candidate-nnue"),
        _spec(script, command_log, "champion-nnue"),
        NnueGameGenerationConfig(
            generation_id="fixture-incomplete",
            openings=(opening,),
            max_plies=1,
            parallelism=8,
        ),
    ).generate()

    assert len(result.position_source_records()) == 2
    assert result.strong_wdl_records() == ()
    assert result.receipt.metrics.requested_parallelism == 8
    assert result.receipt.metrics.effective_parallelism == 1
    assert result.receipt.metrics.incomplete_games == 2
    for game in result.games:
        assert game.record.initial_sfen == Board().to_sfen()
        assert game.record.moves == ("7g7f", "5a5b")
        assert game.record.samples[0].ply == 1
        assert game.record.samples[0].value_target == 0.0
        assert game.record.termination is Termination.MAX_PLIES
        assert game.record.winner is None
        assert game.receipt.opening_prefix_moves == ("7g7f",)
        assert game.receipt.generated_moves == ("5a5b",)
        assert game.receipt.full_game_moves == ("7g7f", "5a5b")
        assert game.receipt.safety_truncated
        assert not game.receipt.rules_terminal_trajectory
        assert not game.receipt.strong_wdl_label_allowed
        assert game.receipt.strong_wdl_label_black is None
        assert game.receipt.plies[0].resolve_history_prefix(
            initial_sfen=game.receipt.initial_sfen,
            full_game_moves=game.receipt.full_game_moves,
        ) == ("7g7f",)
        assert game.receipt.plies[0].termination is Termination.MAX_PLIES
        assert game.receipt.plies[0].strong_value_target is None


def test_entering_king_is_rule_adjudicated_without_engine_claim(tmp_path: Path) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    result = NnueGameGenerator(
        _spec(script, command_log, "candidate-nnue"),
        _spec(script, command_log, "teacher-nnue"),
        NnueGameGenerationConfig(
            generation_id="fixture-declaration",
            openings=(BLACK_CSA27_DECLARATION_SFEN,),
            max_plies=1,
        ),
    ).generate()

    assert len(result.strong_wdl_records()) == 2
    assert result.receipt.metrics.positions == 0
    for game in result.games:
        assert game.record.moves == ()
        assert game.record.samples == ()
        assert game.record.termination is Termination.DECLARATION
        assert game.record.winner == 0
        assert game.receipt.plies == ()
        assert game.receipt.rules_terminal_trajectory
        assert game.receipt.strong_wdl_label_allowed
        assert game.receipt.strong_wdl_label_black == 1.0
        assert game.receipt.final_sfen == Board(BLACK_CSA27_DECLARATION_SFEN).to_sfen()


def test_config_and_rights_profile_construction_fail_closed() -> None:
    start = Board().to_sfen()
    with pytest.raises(ValueError, match="continues after a rules-terminal position"):
        NnueGenerationOpening(
            initial_sfen=start,
            moves=(*REPETITION_MOVES, "5i5h"),
        )
    with pytest.raises(ValueError, match="duplicate exact opening history"):
        NnueGameGenerationConfig(
            generation_id="duplicates",
            openings=(start, start),
        ).normalized_openings()
    with pytest.raises(ValueError, match="fixes MultiPV=1"):
        UsiEngineSpec(
            actor_source="fixture",
            command=(sys.executable,),
            policy=_fixture_policy("fixture"),
            nodes=1,
            options=(("MultiPV", 2),),
        )
    with pytest.raises(ValueError, match="cannot share one actor_source"):
        NnueGameGenerator(
            UsiEngineSpec(
                actor_source="ambiguous-meteo",
                command=(sys.executable, "candidate"),
                policy=_fixture_policy("candidate"),
                nodes=1,
            ),
            UsiEngineSpec(
                actor_source="ambiguous-meteo",
                command=(sys.executable, "champion"),
                policy=_fixture_policy("champion"),
                nodes=1,
            ),
            NnueGameGenerationConfig(generation_id="ambiguous", openings=(start,)),
        )

    limited = UsiEngineSpec.from_rights_profile(
        actor_source="suisho11plus-local",
        command=(sys.executable,),
        rights_id="suisho11plus-wcsc36-20260525-local",
        nodes=1,
    )
    with pytest.raises(PermissionError, match="public-release-safe teacher set"):
        limited.open_engine()

    authorized = UsiEngineSpec.from_rights_profile(
        actor_source="suisho11plus-local",
        command=(sys.executable,),
        rights_id="suisho11plus-wcsc36-20260525-local",
        nodes=1,
        allow_limited_local=True,
    )
    authorized.require_game_generation_permission()
    profile = authorized.to_profile_dict(())
    hard_game_rights = profile["hard_game_training_rights"]
    assert isinstance(hard_game_rights, dict)
    assert hard_game_rights == {
        "registered": True,
        "rights_id": "suisho11plus-wcsc36-20260525-local",
        "decision": "limited",
        "limited_local_authorized": True,
    }


def test_standalone_runner_atomically_publishes_games_receipt_and_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    config_path = tmp_path / "games-config.json"
    config_path.write_text(
        json.dumps(_runner_config(script, command_log)),
        encoding="utf-8",
    )
    output = tmp_path / "generated-games"

    assert nnue_game_runner.main([str(config_path), str(output)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["schema"] == nnue_game_runner.NNUE_GAME_BUNDLE_RECEIPT_SCHEMA
    assert summary["output"] == str(output)
    assert summary["metrics"]["games"] == 2
    assert summary["metrics"]["rules_complete_games"] == 2
    assert summary["metrics"]["positions"] == 2
    assert summary["metrics"]["games_per_hour"] > 0
    assert summary["metrics"]["positions_per_second"] > 0

    games_path = output / "games.jsonl"
    receipt_path = output / "receipt.json"
    assert output.is_dir()
    assert games_path.is_file()
    assert receipt_path.is_file()
    lines = games_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    games = [json.loads(line) for line in lines]
    assert all(game["schema"] == nnue_game_runner.NNUE_GAME_JSONL_SCHEMA for game in games)
    assert [game["trajectory_receipt"]["color_swap_leg"] for game in games] == [0, 1]
    assert [game["trajectory_receipt"]["black_actor_source"] for game in games] == [
        "nagisa-v3.1-fixture",
        "suisho5-fixture",
    ]
    assert all(game["game"]["termination"] == "checkmate" for game in games)
    assert all(game["trajectory_receipt"]["strong_wdl_label_allowed"] for game in games)
    assert all(
        game["trajectory_receipt"]["plies"][0]["history_prefix"]
        == {
            "source": "game.full_game_moves",
            "length": 0,
            "sha256": hashlib.sha256(f"{Board(MATE_IN_ONE_SFEN).to_sfen()}\0".encode()).hexdigest(),
        }
        for game in games
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_hash = receipt.pop("payload_sha256")
    receipt.pop("payload_sha256_scope")
    canonical_receipt = json.dumps(
        receipt,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert receipt_hash == hashlib.sha256(canonical_receipt).hexdigest()
    assert receipt["schema"] == nnue_game_runner.NNUE_GAME_BUNDLE_RECEIPT_SCHEMA
    assert receipt["local_only"] is True
    assert receipt["publication_allowed"] is False
    assert receipt["strong_wdl_eligibility"] == {
        "allowed_only_for_rules_terminal_trajectory": True,
        "allowed_terminations": [
            "checkmate",
            "repetition",
            "entering_king_declaration",
        ],
        "resignation_is_eligible": False,
        "evaluation_adjudication_is_eligible": False,
        "max_plies_is_eligible": False,
        "per_game_gate": "trajectory_receipt.strong_wdl_label_allowed",
        "per_ply_gate": "trajectory_receipt.plies[].strong_value_target",
    }
    assert receipt["config"]["sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert receipt["config"]["path"] == str(config_path)
    assert receipt["config"]["path_scope"] == "private_local_provenance_only"
    assert receipt["config"]["path_publication_allowed"] is False
    assert receipt["engine_inputs"] == [
        {
            "actor_source": "nagisa-v3.1-fixture",
            "kind": "rights_profile",
            "rights_id": "nagisa-v3.1",
            "meteo_runtime_contract": None,
            "pinned_artifacts": [
                {
                    "path": str(artifact.resolve()),
                    "path_scope": "private_local_provenance_only",
                    "path_publication_allowed": False,
                    "sha256": _sha256_file(artifact.resolve()),
                }
                for artifact in sorted(
                    (Path(sys.executable), script),
                    key=lambda item: str(item.resolve()),
                )
            ],
        },
        {
            "actor_source": "suisho5-fixture",
            "kind": "rights_profile",
            "rights_id": "suisho5",
            "meteo_runtime_contract": None,
            "pinned_artifacts": [
                {
                    "path": str(artifact.resolve()),
                    "path_scope": "private_local_provenance_only",
                    "path_publication_allowed": False,
                    "sha256": _sha256_file(artifact.resolve()),
                }
                for artifact in sorted(
                    (Path(sys.executable), script),
                    key=lambda item: str(item.resolve()),
                )
            ],
        },
    ]
    assert receipt["games_jsonl"] == {
        "path": "games.jsonl",
        "schema": nnue_game_runner.NNUE_GAME_JSONL_SCHEMA,
        "games": 2,
        "bytes": games_path.stat().st_size,
        "sha256": hashlib.sha256(games_path.read_bytes()).hexdigest(),
        "contains_complete_trajectory_receipts": True,
    }
    assert "games" not in receipt["generation"]
    assert receipt["generation"]["metrics"] == summary["metrics"]
    assert all(
        option["verified"]
        for startup in receipt["generation"]["engine_startups"]
        for option in startup["startup_provenance"]["applied_options"]
    )
    assert not list(tmp_path.glob(".generated-games.tmp-*"))
    assert not list(tmp_path.glob(".generated-games.nnue-games.lock"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        nnue_game_runner.main([str(config_path), str(output)])


def test_standalone_runner_uses_verified_meteo_runtime_contract_as_local_engine(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export = _fake_meteo_runtime_export(tmp_path / "meteo-export")
    yaneuraou = _fake_meteo_yaneuraou(tmp_path / "source-yaneuraou")
    runtime = tmp_path / "meteo-runtime"
    contract = stage_nnue_runtime(
        export,
        yaneuraou,
        runtime,
        profile=RuntimeProfile(
            profile_id="meteo-candidate-fixture",
            hash_megabytes=32,
            smoke_nodes=17,
            timeout_seconds=5.0,
        ),
    )
    third_party_script = tmp_path / "fake_full_game.py"
    third_party_log = tmp_path / "positions.log"
    _fake_full_game_usi(third_party_script)
    config = _runner_config(third_party_script, third_party_log)
    config["engines"][0] = {
        "actor_source": "meteo-runtime-candidate",
        "meteo_runtime_contract": str(runtime / "runtime-contract.json"),
        "nodes": 17,
        "timeout_seconds": 5.0,
        "policy_temperature": 200.0,
        "value_scale": 1_200.0,
    }
    config["generation"] = {
        "generation_id": "meteo-runtime-contract-integration",
        "max_plies": 1,
        "parallelism": 1,
        "openings": [{"initial_sfen": Board().to_sfen(), "moves": []}],
    }
    config_path = tmp_path / "meteo-games-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = nnue_game_runner.load_nnue_game_runner_config(config_path)
    assert loaded.first.command == (str(runtime / "engine" / "YaneuraOu"),)
    assert loaded.first.working_directory == runtime
    assert loaded.first.policy.policy_id == (f"meteo-nnue-runtime-{contract.contract_sha256}")
    assert loaded.first.policy.training_outputs_local_only
    assert not loaded.first.policy.redistribution_allowed
    assert loaded.first.option_value_verification.value == "yaneuraou_getoption"
    assert all(name.casefold() != "multipv" for name, _ in loaded.first.options)
    assert dict(loaded.first.options)["BookFile"] == "no_book"
    assert dict(loaded.first.options)["Threads"] == "1"
    assert {path.relative_to(runtime).as_posix() for path in loaded.first.artifact_paths} == {
        "runtime-contract.json",
        "engine/YaneuraOu",
        "eval/nn.bin",
        "eval/progress.bin",
        "eval/eval_options.txt",
        "source-export-receipt.json",
    }
    assert loaded.engine_inputs[0].kind == "meteo_runtime_contract"
    assert loaded.engine_inputs[0].runtime_contract == contract
    loaded.revalidate_meteo_runtime_contracts()

    output = tmp_path / "meteo-generated-games"
    assert nnue_game_runner.main([str(config_path), str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["metrics"]["games"] == 2
    assert summary["metrics"]["positions"] == 2
    assert summary["metrics"]["rules_complete_games"] == 0

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["local_only"] is True
    assert receipt["publication_allowed"] is False
    meteo_input = receipt["engine_inputs"][0]
    assert meteo_input["kind"] == "meteo_runtime_contract"
    assert meteo_input["rights_id"] is None
    assert meteo_input["pinned_artifacts"] == []
    runtime_receipt = meteo_input["meteo_runtime_contract"]
    assert runtime_receipt["runtime_directory"] == str(runtime)
    assert runtime_receipt["path_scope"] == "private_local_provenance_only"
    assert runtime_receipt["path_publication_allowed"] is False
    assert runtime_receipt["contract"]["contract_sha256"] == contract.contract_sha256
    assert runtime_receipt["contract"]["runtime_content_sha256"] == (
        contract.runtime_content_sha256
    )
    profiles = receipt["generation"]["engine_profiles"]
    assert profiles[0]["rights_policy"]["training_outputs_local_only"] is True
    assert profiles[0]["rights_policy"]["redistribution_allowed"] is False
    games = [
        json.loads(line)
        for line in (output / "games.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [game["game"]["termination"] for game in games] == [
        "max_plies",
        "max_plies",
    ]
    assert all(not game["trajectory_receipt"]["strong_wdl_label_allowed"] for game in games)


def test_standalone_runner_config_rejects_unknown_duplicate_and_rights_ambiguity(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    config = _runner_config(script, command_log)

    unknown = tmp_path / "unknown.json"
    unknown.write_text(
        json.dumps({**config, "unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        nnue_game_runner.load_nnue_game_runner_config(unknown)

    duplicate_options = json.loads(json.dumps(config))
    duplicate_options["engines"][0]["options"] = [
        {"name": "Threads", "value": 1},
        {"name": "threads", "value": 2},
    ]
    duplicate_options_path = tmp_path / "duplicate-options.json"
    duplicate_options_path.write_text(json.dumps(duplicate_options), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate USI option ignoring case"):
        nnue_game_runner.load_nnue_game_runner_config(duplicate_options_path)

    paid_without_ack = json.loads(json.dumps(config))
    paid_without_ack["engines"][0]["rights_id"] = "suisho11plus-wcsc36-20260525-local"
    paid_path = tmp_path / "paid-without-ack.json"
    paid_path.write_text(json.dumps(paid_without_ack), encoding="utf-8")
    with pytest.raises(PermissionError, match="allow_limited_local=true"):
        nnue_game_runner.load_nnue_game_runner_config(paid_path)

    public_with_ack = json.loads(json.dumps(config))
    public_with_ack["engines"][0]["allow_limited_local"] = True
    public_path = tmp_path / "public-with-ack.json"
    public_path.write_text(json.dumps(public_with_ack), encoding="utf-8")
    with pytest.raises(ValueError, match="only valid for a LOCAL_AUTHORIZED_ONLY"):
        nnue_game_runner.load_nnue_game_runner_config(public_path)

    duplicate_key = tmp_path / "duplicate-key.json"
    valid_text = json.dumps(config)
    duplicate_key.write_text(
        valid_text.replace(
            '"schema":',
            '"schema":"duplicate", "schema":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key 'schema'"):
        nnue_game_runner.load_nnue_game_runner_config(duplicate_key)

    duplicate_key_ignoring_case = tmp_path / "duplicate-key-ignoring-case.json"
    duplicate_key_ignoring_case.write_text(
        valid_text.replace(
            '"schema":',
            '"Schema":"duplicate", "schema":',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object keys ignoring case"):
        nnue_game_runner.load_nnue_game_runner_config(duplicate_key_ignoring_case)

    both_engine_sources = json.loads(json.dumps(config))
    both_engine_sources["engines"][0]["meteo_runtime_contract"] = "runtime"
    both_path = tmp_path / "both-engine-sources.json"
    both_path.write_text(json.dumps(both_engine_sources), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one of rights_id or meteo_runtime_contract"):
        nnue_game_runner.load_nnue_game_runner_config(both_path)

    no_engine_source = json.loads(json.dumps(config))
    no_engine_source["engines"][0].pop("rights_id")
    neither_path = tmp_path / "no-engine-source.json"
    neither_path.write_text(json.dumps(no_engine_source), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one of rights_id or meteo_runtime_contract"):
        nnue_game_runner.load_nnue_game_runner_config(neither_path)

    meteo_with_overrides = json.loads(json.dumps(config))
    meteo_with_overrides["engines"][0].pop("rights_id")
    meteo_with_overrides["engines"][0]["meteo_runtime_contract"] = "runtime"
    overrides_path = tmp_path / "meteo-with-overrides.json"
    overrides_path.write_text(json.dumps(meteo_with_overrides), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        nnue_game_runner.load_nnue_game_runner_config(overrides_path)


def test_rights_runner_pins_executable_and_artifacts_before_and_after_games(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_full_game.py"
    command_log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    config = _runner_config(script, command_log)
    config_path = tmp_path / "pinned.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = nnue_game_runner.load_nnue_game_runner_config(config_path)
    loaded.revalidate_engine_inputs()
    script.write_text(script.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        loaded.revalidate_engine_inputs()

    empty = _runner_config(script, command_log)
    empty["engines"][0]["artifact_identities"] = []
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps(empty), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        nnue_game_runner.load_nnue_game_runner_config(empty_path)

    missing_executable = _runner_config(script, command_log)
    missing_executable["engines"][0]["artifact_identities"] = [
        identity
        for identity in missing_executable["engines"][0]["artifact_identities"]
        if Path(identity["path"]) != Path(sys.executable).resolve()
    ]
    missing_path = tmp_path / "missing-executable.json"
    missing_path.write_text(json.dumps(missing_executable), encoding="utf-8")
    with pytest.raises(ValueError, match="command executable must be included"):
        nnue_game_runner.load_nnue_game_runner_config(missing_path)


def test_meteo_game_runner_rejects_runtime_not_pinned_to_multipv_one(
    tmp_path: Path,
) -> None:
    export = _fake_meteo_runtime_export(tmp_path / "meteo-export-mpv2")
    yaneuraou = _fake_meteo_yaneuraou(tmp_path / "source-yaneuraou-mpv2")
    runtime = tmp_path / "meteo-runtime-mpv2"
    stage_nnue_runtime(
        export,
        yaneuraou,
        runtime,
        profile=RuntimeProfile(
            profile_id="meteo-mpv2-rejected",
            multipv=2,
            smoke_nodes=17,
            timeout_seconds=5.0,
        ),
    )
    script = tmp_path / "fake_full_game.py"
    log = tmp_path / "positions.log"
    _fake_full_game_usi(script)
    config = _runner_config(script, log)
    config["engines"][0] = {
        "actor_source": "meteo-runtime-mpv2",
        "meteo_runtime_contract": str(runtime),
        "nodes": 17,
        "timeout_seconds": 5.0,
        "policy_temperature": 200.0,
        "value_scale": 1_200.0,
    }
    config_path = tmp_path / "meteo-mpv2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must pin MultiPV=1"):
        nnue_game_runner.load_nnue_game_runner_config(config_path)
