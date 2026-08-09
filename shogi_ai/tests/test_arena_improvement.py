from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from rsshogi.core import Board

import simajilord_shogi.arena as arena_module
import simajilord_shogi.cli as shogi_cli
import simajilord_shogi.self_improvement as improvement
from simajilord_shogi.arena import (
    ExternalUsiPlayer,
    MoveDecision,
    play_direct_game,
    summarize_arena,
    summarize_paired_arena,
)
from simajilord_shogi.checkpoint import save_checkpoint
from simajilord_shogi.config import SearchConfig, model_profile
from simajilord_shogi.domain import GameRecord, Termination
from simajilord_shogi.external_usi import ExternalTeacherPolicy, ExternalUsiTeacher
from simajilord_shogi.model import PolicyValueResNet, upgrade_model_to_canonical_v2
from simajilord_shogi.opening_suite import OpeningPosition, load_opening_suite
from simajilord_shogi.self_improvement import SelfImprovementConfig, run_generation
from simajilord_shogi.trainer import TrainingInterlockConfig

MATE_IN_ONE_SFEN = "4k4/9/3B5/9/9/9/9/9/4K4 b G 1"


def _unique_legal_sfens(count: int) -> list[str]:
    board = Board()
    result: list[str] = []
    for _ in range(count):
        result.append(board.to_sfen())
        legal_moves = list(board.legal_moves())
        if not legal_moves:
            raise AssertionError("test opening generator unexpectedly reached a terminal position")
        board.apply_move(legal_moves[0])
    return result


def _fake_usi(path: Path) -> None:
    path.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-suisho', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        print('info depth 1 nodes 1 score cp 9999 pv G*5b', flush=True)
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _fake_yaneuraou_benchmark_usi(path: Path) -> None:
    path.write_text(
        """import sys
options = {
    'MultiPV': '1',
    'EvalDir': '',
    'FV_SCALE': '16',
    'Threads': '1',
    'USI_Hash': '16',
    'USI_OwnBook': 'true',
    'BookFile': 'standard_book.db',
    'PvInterval': '300',
}
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-yaneuraou-benchmark', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('option name EvalDir type string default .', flush=True)
        print('option name FV_SCALE type spin default 16 min 1 max 128', flush=True)
        print('option name Threads type spin default 1 min 1 max 512', flush=True)
        print('option name USI_Hash type spin default 16 min 1 max 1048576', flush=True)
        print('option name USI_OwnBook type check default true', flush=True)
        print('option name BookFile type string default standard_book.db', flush=True)
        print('option name PvInterval type spin default 300 min 0 max 10000', flush=True)
        print('usiok', flush=True)
    elif command.startswith('setoption name '):
        setting = command.removeprefix('setoption name ')
        name, value = setting.split(' value ', 1)
        options[name] = value
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('getoption '):
        name = command.removeprefix('getoption ')
        if name in options:
            print(f'Options[{name}] = {options[name]}', flush=True)
        else:
            print(f'No such option: {name}', flush=True)
    elif command.startswith('go '):
        print('info depth 1 nodes 1 score cp 9999 pv G*5b', flush=True)
        print('bestmove G*5b', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


def _fake_history_usi(path: Path, command_log: Path) -> None:
    path.write_text(
        f"""import sys
last_position = ''
log_path = {str(command_log)!r}
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-history-arena', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
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
        print(f'info depth 1 nodes 1 score cp 0 pv {{move}}', flush=True)
        print(f'bestmove {{move}}', flush=True)
    elif command == 'quit':
        break
""",
        encoding="utf-8",
    )


class _LegacyMatePlayer:
    """A pre-history-protocol DirectPlayer used to lock backward compatibility."""

    def choose_move(self, _board: object) -> MoveDecision:
        return MoveDecision("G*5b", {"G*5b": 1.0}, 1.0, source="legacy-fixture")


def test_direct_usi_player_plays_a_legal_terminal_game(tmp_path: Path) -> None:
    script = tmp_path / "fake_usi.py"
    _fake_usi(script)
    policy = ExternalTeacherPolicy(
        policy_id="test-fake-free",
        name="fake-free-engine",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )
    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=1,
        training_use=False,
    ) as engine:
        external = ExternalUsiPlayer(engine)
        record = play_direct_game(
            external,
            external,
            initial_sfen=MATE_IN_ONE_SFEN,
            max_plies=4,
        )

    assert record.moves == ("G*5b",)
    assert record.termination.value == "checkmate"
    assert record.winner == 0
    assert record.samples[0].root_value == pytest.approx(math.tanh(9999 / 1200))
    assert record.samples[0].actor_source == "test-fake-free"


def test_direct_external_game_sends_the_actual_prefix_to_history_sensitive_engine(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_history_usi.py"
    command_log = tmp_path / "position-commands.txt"
    _fake_history_usi(script, command_log)
    policy = ExternalTeacherPolicy(
        policy_id="test-history-arena",
        name="history-sensitive-arena",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )
    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=1,
        training_use=False,
    ) as engine:
        external = ExternalUsiPlayer(engine)
        record = play_direct_game(external, external, max_plies=2)

    assert record.moves == ("7g7f", "3c3d")
    assert record.termination.value == "max_plies"
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "position startpos",
        "position startpos moves 7g7f",
    ]


def test_direct_player_without_history_capability_remains_supported() -> None:
    legacy = _LegacyMatePlayer()

    record = play_direct_game(
        legacy,
        legacy,
        initial_sfen=MATE_IN_ONE_SFEN,
        max_plies=2,
    )

    assert record.moves == ("G*5b",)
    assert record.termination.value == "checkmate"


def test_external_benchmark_real_smoke_path_returns_one_paired_cluster(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_usi.py"
    _fake_usi(script)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=0)
    policy = ExternalTeacherPolicy(
        policy_id="test-real-external-benchmark",
        name="fake-real-external-benchmark",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=False,
        redistribution_allowed=False,
    )
    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=1,
        multipv=1,
        training_use=False,
    ) as engine:
        interlock = TrainingInterlockConfig(
            state_root=tmp_path / "benchmark-state",
            ttl_seconds=3,
            heartbeat_interval_seconds=1,
            wait_timeout_seconds=2,
            poll_interval_seconds=0.01,
        )
        summary, games = arena_module.benchmark_checkpoint_vs_external(
            checkpoint,
            engine,
            SearchConfig(simulations=1, max_plies=1),
            openings=[MATE_IN_ONE_SFEN],
            seed=5,
            legacy_single_opening=True,
            promotion_eligible=False,
            bootstrap_iterations=100,
            compute_interlock=interlock,
        )

    assert len(games) == 2
    assert summary.opening_pairs == 1
    assert summary.independent_opening_pairs == 1
    assert summary.incomplete_games == 2
    assert summary.cluster_results[0].normalized_key == OpeningPosition.from_sfen(
        MATE_IN_ONE_SFEN
    ).normalized_key
    assert "legacy_single_opening_debug_only" in summary.promotion_blockers
    assert not summary.promoted
    assert not any((interlock.state_root / "training-step-leases").iterdir())


def test_benchmark_report_records_game_prefix_history_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "fake_usi.py"
    _fake_usi(script)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=0)
    placeholder = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=(),
        samples=(),
        winner=None,
        termination=Termination.MAX_PLIES,
    )

    def fake_benchmark(*_args: object, **_kwargs: object) -> tuple[object, list[GameRecord]]:
        return (
            summarize_paired_arena(
                [(0.5, 0.5)],
                incomplete_pairs=1,
                incomplete_games=2,
                opening_sfens=[MATE_IN_ONE_SFEN],
                opening_keys=[OpeningPosition.from_sfen(MATE_IN_ONE_SFEN).normalized_key],
                legacy_single_opening=True,
                promotion_eligible=False,
                bootstrap_iterations=100,
            ),
            [placeholder, placeholder],
        )

    monkeypatch.setattr(shogi_cli, "benchmark_checkpoint_vs_external", fake_benchmark)
    output = tmp_path / "benchmark"

    assert (
        shogi_cli.main(
            [
                "benchmark-usi",
                str(checkpoint),
                str(output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--nodes",
                "1",
                "--legacy-single-opening-debug",
                "--initial-sfen",
                MATE_IN_ONE_SFEN,
                "--games",
                "2",
                "--simulations",
                "1",
            ]
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["history_mode"] == "game_prefix"
    assert report["history_mode_missing_field_means"] == "board_only"
    assert report["history_source"] == "play_direct_game.initial_sfen + moves_so_far"
    assert report["human_play_compute_interlock"] == {"enabled": False}
    assert report["opening"]["mode"] == "legacy_single_opening_debug_only"
    assert report["summary"]["legacy_single_opening"] is True
    assert report["elo"]["eligible"] is False
    assert report["replay"]["sha256"]
    assert report["checkpoint"]["files"]


def test_benchmark_output_is_create_only_before_engine_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing-benchmark"
    output.mkdir()
    marker = output / "owned-by-user.txt"
    marker.write_text("preserve", encoding="utf-8")

    def must_not_construct(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("engine must not start when output already exists")

    monkeypatch.setattr(shogi_cli, "ExternalUsiTeacher", must_not_construct)
    with pytest.raises(FileExistsError, match="refusing to overwrite benchmark bundle"):
        shogi_cli.main(
            [
                "benchmark-usi",
                str(tmp_path / "missing-checkpoint"),
                str(output),
                "--engine",
                str(tmp_path / "missing-engine"),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--legacy-single-opening-debug",
            ]
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_suisho11plus_benchmark_requires_explicit_local_authorization(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=0)

    with pytest.raises(PermissionError, match="LOCAL_AUTHORIZED_ONLY benchmark use"):
        shogi_cli.main(
            [
                "benchmark-usi",
                str(checkpoint),
                str(tmp_path / "private" / "benchmark"),
                "--engine",
                str(tmp_path / "missing-engine"),
                "--rights-profile",
                "suisho11plus-wcsc36-20260525-local",
                "--legacy-single-opening-debug",
            ]
        )


def test_suisho11plus_benchmark_is_verified_private_and_unpublishable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "fake_yaneuraou.py"
    _fake_yaneuraou_benchmark_usi(script)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=0)
    eval_directory = tmp_path / "eval"
    eval_directory.mkdir()
    private_root = tmp_path / "private"
    private_root.mkdir()
    output = private_root / "benchmark"
    placeholder = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=(),
        samples=(),
        winner=None,
        termination=Termination.MAX_PLIES,
    )

    def fake_benchmark(*_args: object, **_kwargs: object) -> tuple[object, list[GameRecord]]:
        return (
            summarize_paired_arena(
                [(0.5, 0.5)],
                incomplete_pairs=1,
                incomplete_games=2,
                opening_sfens=[MATE_IN_ONE_SFEN],
                opening_keys=[OpeningPosition.from_sfen(MATE_IN_ONE_SFEN).normalized_key],
                legacy_single_opening=True,
                promotion_eligible=False,
                bootstrap_iterations=100,
            ),
            [placeholder, placeholder],
        )

    monkeypatch.setattr(shogi_cli, "benchmark_checkpoint_vs_external", fake_benchmark)
    assert (
        shogi_cli.main(
            [
                "benchmark-usi",
                str(checkpoint),
                str(output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "suisho11plus-wcsc36-20260525-local",
                "--local-only-user-authorized",
                "--local-only-root",
                str(private_root),
                "--nodes",
                "1",
                "--multipv",
                "1",
                "--option",
                f"EvalDir={eval_directory}",
                "--option",
                "FV_SCALE=40",
                "--option",
                "Threads=1",
                "--option",
                "USI_Hash=64",
                "--option",
                "USI_OwnBook=false",
                "--option",
                "BookFile=no_book",
                "--option",
                "PvInterval=0",
                "--legacy-single-opening-debug",
                "--initial-sfen",
                MATE_IN_ONE_SFEN,
                "--games",
                "2",
                "--simulations",
                "1",
            ]
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["rights_mode"] == "limited_local"
    assert report["local_only_user_authorized"] is True
    assert report["publication_allowed"] is False
    assert report["public_release_gate"] == "blocked_pending_rights_holder_permission"
    assert report["rights"]["distillation_scope"] == "local_authorized_only"
    assert report["engine"]["option_value_verification"] == "yaneuraou_getoption"
    applied = report["engine"]["startup_provenance"]["applied_options"]
    assert applied
    assert all(option["verified"] for option in applied)


def test_failed_benchmark_cleans_sibling_temporary_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "fake_usi.py"
    _fake_usi(script)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), checkpoint, step=0)
    output = tmp_path / "failed-benchmark"

    def fail_after_engine_start(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated arena failure")

    monkeypatch.setattr(shogi_cli, "benchmark_checkpoint_vs_external", fail_after_engine_start)
    with pytest.raises(RuntimeError, match="simulated arena failure"):
        shogi_cli.main(
            [
                "benchmark-usi",
                str(checkpoint),
                str(output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--nodes",
                "1",
                "--legacy-single-opening-debug",
                "--simulations",
                "1",
            ]
        )

    assert not output.exists()
    assert list(tmp_path.glob(".failed-benchmark.tmp-*")) == []


def test_production_benchmark_pins_opening_checkpoint_source_and_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = tmp_path / "fake_usi.py"
    _fake_usi(script)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        PolicyValueResNet(model_profile("smoke")),
        checkpoint,
        step=7,
        lineage={"parent_checkpoint": "fixture-parent"},
    )
    openings = _unique_legal_sfens(32)
    suite = tmp_path / "opening-suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema": "meteo-opening-suite-v1",
                "splits": {"heldout": openings},
            }
        ),
        encoding="utf-8",
    )
    private_state = tmp_path / "private-benchmark-state-token"

    def fake_benchmark(
        _checkpoint: Path,
        _external: object,
        _search: SearchConfig,
        *,
        openings: tuple[str, ...],
        seed: int,
        promotion_min_pairs: int,
        promotion_lower_bound: float,
        bootstrap_iterations: int,
        legacy_single_opening: bool,
        promotion_eligible: bool,
        compute_interlock: TrainingInterlockConfig | None,
    ) -> tuple[object, list[GameRecord]]:
        assert seed == 41
        assert promotion_min_pairs == 32
        assert promotion_lower_bound == 0.5
        assert bootstrap_iterations == 200
        assert not legacy_single_opening
        assert promotion_eligible
        assert compute_interlock is not None
        assert compute_interlock.state_root == private_state
        positions = tuple(OpeningPosition.from_sfen(opening) for opening in openings)
        games = [
            GameRecord(position.sfen, (), (), None, Termination.REPETITION)
            for position in positions
            for _ in range(2)
        ]
        return (
            summarize_paired_arena(
                [(0.5, 0.5)] * len(positions),
                opening_sfens=[position.sfen for position in positions],
                opening_keys=[position.normalized_key for position in positions],
                opening_hashes=[position.sha256 for position in positions],
                terminations=[("repetition", "repetition")] * len(positions),
                promotion_min_pairs=promotion_min_pairs,
                promotion_lower_bound=promotion_lower_bound,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            ),
            games,
        )

    monkeypatch.setattr(shogi_cli, "benchmark_checkpoint_vs_external", fake_benchmark)
    output = tmp_path / "production-benchmark"
    assert (
        shogi_cli.main(
            [
                "benchmark-usi",
                str(checkpoint),
                str(output),
                "--engine",
                sys.executable,
                "--engine-arg",
                str(script),
                "--rights-profile",
                "gikou2-v2.0.2",
                "--nodes",
                "1",
                "--opening-suite",
                str(suite),
                "--opening-split",
                "heldout",
                "--bootstrap-iterations",
                "200",
                "--seed",
                "41",
                "--simulations",
                "1",
                "--human-play-state-root",
                str(private_state),
            ]
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["schema"] == "meteo-external-usi-benchmark-v2"
    assert report["opening"]["mode"] == "immutable_split_suite"
    assert report["opening"]["split"] == "heldout"
    assert report["opening"]["count"] == 32
    assert len(report["opening"]["normalized_keys"]) == 32
    assert report["summary"]["independent_opening_pairs"] == 32
    assert report["ci"] == {
        "bootstrap_iterations": 200,
        "confidence_level": 0.95,
        "method": "deterministic_percentile_cluster_bootstrap_one_sided",
        "resampling_unit": "independent_normalized_opening_pair",
        "seed": 41,
    }
    assert report["elo"]["eligible"] is True
    assert report["elo"]["estimate"] == 0.0
    assert report["checkpoint"]["lineage"] == {"parent_checkpoint": "fixture-parent"}
    assert report["checkpoint"]["file_count"] == 2
    assert report["human_play_compute_interlock"]["enabled"] is True
    assert report["human_play_compute_interlock"]["state_root_recorded"] is False
    assert "private-benchmark-state-token" not in json.dumps(
        report["human_play_compute_interlock"]
    )
    assert report["source"]["tree"]["sha256"]
    assert report["source"]["git"]["commit"]
    assert report["engine"]["executable"]["sha256"]
    assert report["engine"]["value_conversion"] == {
        "centipawn_formula": "tanh(cp / D)",
        "declaration_win": 1.0,
        "input": "root MultiPV score from the side-to-move perspective",
        "mate": "winning=+1; losing=-1",
        "ponanza_coefficient_C": 600.0,
        "resign": -1.0,
        "tanh_denominator_D": 1200.0,
    }
    assert report["evaluation_values"] == {}
    assert report["replay"]["sha256"]
    assert report["termination"] == {
        "complete": True,
        "counts": {"repetition": 64},
        "incomplete_games": 0,
        "incomplete_pairs": 0,
    }


def test_arena_promotion_requires_confident_score_and_complete_games() -> None:
    decisive = summarize_arena(
        [1.0] * 100,
        promotion_min_games=100,
        promotion_lower_bound=0.5,
    )
    incomplete = summarize_arena(
        [1.0] * 100,
        incomplete_games=1,
        promotion_min_games=100,
        promotion_lower_bound=0.5,
    )

    assert decisive.lower_95 > 0.5
    assert decisive.promoted
    assert not incomplete.promoted


def test_paired_arena_uses_openings_as_clusters() -> None:
    decisive = summarize_paired_arena(
        [(1.0, 1.0)] * 32,
        bootstrap_iterations=2_000,
        seed=7,
    )
    duplicated_tie = summarize_paired_arena(
        [(1.0, 0.0)] * 32,
        bootstrap_iterations=2_000,
        seed=7,
    )

    assert decisive.opening_pairs == 32
    assert decisive.games == 64
    assert decisive.cluster_bootstrap_lower_95 == 1.0
    assert decisive.sign_flip_p_value_one_sided < 0.05
    assert decisive.promoted
    assert duplicated_tie.score == 0.5
    assert duplicated_tie.cluster_bootstrap_lower_95 == 0.5
    assert duplicated_tie.sign_flip_p_value_one_sided == 1.0
    assert not duplicated_tie.promoted


def test_repeating_one_opening_one_hundred_times_never_promotes() -> None:
    normalized_key = "one-debug-opening"
    summary = summarize_paired_arena(
        [(1.0, 1.0)] * 100,
        opening_keys=[normalized_key] * 100,
        bootstrap_iterations=2_000,
        seed=11,
    )

    assert summary.opening_pairs == 100
    assert summary.independent_opening_pairs == 1
    assert summary.repeated_opening_pairs == 99
    assert summary.cluster_bootstrap_lower_95 == 1.0
    assert "repeated_normalized_openings" in summary.promotion_blockers
    assert "insufficient_independent_opening_pairs" in summary.promotion_blockers
    assert not summary.promoted


def test_thirty_two_unique_complete_pairs_use_one_sided_cluster_bound() -> None:
    summary = summarize_paired_arena(
        [(1.0, 1.0)] * 32,
        opening_keys=[f"opening-{index}" for index in range(32)],
        bootstrap_iterations=2_000,
        seed=17,
    )

    assert summary.independent_opening_pairs == 32
    assert summary.cluster_bootstrap_lower_95 == 1.0
    assert summary.ci_method == "deterministic_percentile_cluster_bootstrap_one_sided"
    assert summary.confidence_level == 0.95
    assert summary.seed == 17
    assert summary.promoted

    too_few = summarize_paired_arena(
        [(1.0, 1.0)] * 31,
        opening_keys=[f"opening-{index}" for index in range(31)],
        bootstrap_iterations=2_000,
        seed=17,
    )
    assert too_few.independent_opening_pairs == 31
    assert "insufficient_independent_opening_pairs" in too_few.promotion_blockers
    assert not too_few.promoted


def test_paired_bootstrap_uses_the_five_percent_one_sided_quantile() -> None:
    results = [(1.0, 1.0)] * 20 + [(0.5, 0.5)] * 8 + [(0.0, 0.0)] * 4
    summary = summarize_paired_arena(
        results,
        opening_keys=[f"opening-{index}" for index in range(32)],
        bootstrap_iterations=20_000,
        seed=123,
    )
    scores = np.asarray([1.0] * 20 + [0.5] * 8 + [0.0] * 4, dtype=np.float64)
    generator = np.random.default_rng(123)
    indices = generator.integers(0, 32, size=(20_000, 32))
    means = scores[indices].mean(axis=1)
    expected_one_sided = float(np.quantile(means, 0.05))
    old_two_sided_lower = float(np.quantile(means, 0.025))

    assert summary.cluster_bootstrap_lower_95 == pytest.approx(expected_one_sided)
    assert expected_one_sided > old_two_sided_lower


def test_paired_arena_rejects_incomplete_games_and_training_split_overlap() -> None:
    keys = [f"opening-{index}" for index in range(32)]
    incomplete = summarize_paired_arena(
        [(1.0, 1.0)] * 32,
        opening_keys=keys,
        incomplete_pairs=1,
        incomplete_games=1,
        bootstrap_iterations=1_000,
        seed=19,
    )
    overlapping = summarize_paired_arena(
        [(1.0, 1.0)] * 32,
        opening_keys=keys,
        training_arena_overlap_keys=(keys[0],),
        bootstrap_iterations=1_000,
        seed=19,
    )

    assert "incomplete_games" in incomplete.promotion_blockers
    assert not incomplete.promoted
    assert overlapping.training_arena_overlap_count == 1
    assert "training_arena_split_overlap" in overlapping.promotion_blockers
    assert not overlapping.promoted


def test_opening_suite_rejects_normalized_and_duplicate_json_keys(tmp_path: Path) -> None:
    sfens = _unique_legal_sfens(3)
    same_position_new_counter = " ".join([*sfens[0].split()[:3], "999"])
    duplicate_position = tmp_path / "duplicate-position.json"
    duplicate_position.write_text(
        json.dumps(
            {
                "schema": "meteo-opening-suite-v1",
                "splits": {
                    "actor": [sfens[0]],
                    "arena": [same_position_new_counter],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate normalized opening SFEN"):
        load_opening_suite(duplicate_position)

    duplicate_split_key = tmp_path / "duplicate-split-key.json"
    duplicate_split_key.write_text(
        '{"schema":"meteo-opening-suite-v1","splits":{'
        f'"actor":[{json.dumps(sfens[0])}],'
        f'"arena":[{json.dumps(sfens[1])}],'
        f'"arena":[{json.dumps(sfens[2])}]}}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key 'arena'"):
        load_opening_suite(duplicate_split_key)

    duplicate_within_split = tmp_path / "duplicate-within-split.json"
    duplicate_within_split.write_text(
        json.dumps(
            {
                "schema": "meteo-opening-suite-v1",
                "splits": {
                    "actor": [sfens[1]],
                    "arena": [sfens[0], same_position_new_counter],
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate normalized opening SFEN"):
        load_opening_suite(duplicate_within_split)


def test_opening_suite_is_hash_pinned_and_create_only_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sfens = _unique_legal_sfens(34)
    suite_path = tmp_path / "openings.json"

    def write_suite(actor_sfen: str) -> None:
        suite_path.write_text(
            json.dumps(
                {
                    "schema": "meteo-opening-suite-v1",
                    "splits": {
                        "actor": [actor_sfen],
                        "arena": sfens[1:33],
                    },
                }
            ),
            encoding="utf-8",
        )

    write_suite(sfens[0])
    suite = load_opening_suite(suite_path)
    assert len(suite.split("arena").positions) == 32
    assert suite.source_sha256
    assert suite.normalized_sha256

    champion = tmp_path / "champion"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), champion, step=0)
    config = SelfImprovementConfig(
        actor_games=1,
        actor_simulations=1,
        actor_temperature_moves=0,
        teacher_simulations=2,
        reanalyse_fraction=1.0,
        training_steps=1,
        batch_size=1,
        learning_rate=1e-6,
        arena_simulations=1,
        max_plies=2,
        opening_suite=str(suite_path),
    )

    def interrupt(*_args: object, **_kwargs: object) -> list[object]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(improvement, "batched_self_play", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation(champion, tmp_path / "loop", config)

    write_suite(sfens[33])
    with pytest.raises(ValueError, match="opening suite changed after generation creation"):
        run_generation(champion, tmp_path / "loop", config)


def test_checkpoint_pair_runs_candidate_black_then_white_for_each_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = object()
    champion = object()
    calls: list[tuple[object, object, str, int, bool]] = []

    def fake_load(path: Path) -> tuple[object, int]:
        return (candidate if path.name == "candidate" else champion), 0

    def fake_match(
        black: object,
        white: object,
        _black_config: SearchConfig,
        _white_config: SearchConfig,
        *,
        initial_sfen: str,
        seed: int,
        self_play_noise: bool,
    ) -> GameRecord:
        calls.append((black, white, initial_sfen, seed, self_play_noise))
        winner = 0 if black is candidate else 1
        return GameRecord(initial_sfen, (), (), winner, Termination.CHECKMATE)

    monkeypatch.setattr(arena_module, "load_checkpoint", fake_load)
    monkeypatch.setattr(arena_module, "MLXEvaluator", lambda model: model)
    monkeypatch.setattr(arena_module, "play_match", fake_match)
    openings = _unique_legal_sfens(2)

    summary, records = arena_module.evaluate_checkpoint_pair(
        tmp_path / "candidate",
        tmp_path / "champion",
        SearchConfig(simulations=1),
        openings=openings,
        promotion_min_pairs=2,
        bootstrap_iterations=500,
        seed=23,
    )

    assert len(records) == 4
    assert [(black, white) for black, white, *_rest in calls] == [
        (candidate, champion),
        (champion, candidate),
        (candidate, champion),
        (champion, candidate),
    ]
    assert [seed for *_prefix, seed, _noise in calls] == [23, 24, 25, 26]
    assert all(not noise for *_prefix, noise in calls)
    assert all(cluster.candidate_black_point == 1.0 for cluster in summary.cluster_results)
    assert all(cluster.candidate_white_point == 1.0 for cluster in summary.cluster_results)
    assert summary.promoted


def test_external_benchmark_uses_unique_opening_pairs_and_per_game_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = object()
    interlocked_evaluator = object()
    opponent = object()
    openings = _unique_legal_sfens(2)
    calls: list[tuple[object, object, str]] = []
    interlock = TrainingInterlockConfig(state_root=tmp_path / "shared-state")

    class FakeExternal:
        new_games = 0

        def new_game(self) -> None:
            self.new_games += 1

    external = FakeExternal()

    monkeypatch.setattr(arena_module, "load_checkpoint", lambda _path: (model, 0))
    monkeypatch.setattr(arena_module, "MLXEvaluator", lambda loaded: loaded)

    def fake_interlocked_evaluator(evaluator: object, settings: object) -> object:
        assert evaluator is model
        assert settings is interlock
        return interlocked_evaluator

    monkeypatch.setattr(arena_module, "InterlockedEvaluator", fake_interlocked_evaluator)
    monkeypatch.setattr(
        arena_module,
        "MctsPlayer",
        lambda evaluator, _config, *, seed: (evaluator, seed),
    )
    monkeypatch.setattr(arena_module, "ExternalUsiPlayer", lambda _external: opponent)

    def fake_direct_game(
        black: object,
        white: object,
        *,
        initial_sfen: str | None,
        max_plies: int | None,
    ) -> GameRecord:
        assert initial_sfen is not None
        assert max_plies == 9
        calls.append((black, white, initial_sfen))
        winner = 0 if isinstance(black, tuple) else 1
        return GameRecord(initial_sfen, (), (), winner, Termination.CHECKMATE)

    monkeypatch.setattr(arena_module, "play_direct_game", fake_direct_game)
    summary, records = arena_module.benchmark_checkpoint_vs_external(
        tmp_path / "checkpoint",
        external,  # type: ignore[arg-type]
        SearchConfig(simulations=1, max_plies=9),
        openings=openings,
        seed=31,
        promotion_min_pairs=2,
        bootstrap_iterations=500,
        compute_interlock=interlock,
    )

    assert len(records) == 4
    assert external.new_games == 4
    assert all(player[0] is interlocked_evaluator for player, _opponent, _sfen in calls[::2])
    assert all(player[0] is interlocked_evaluator for _opponent, player, _sfen in calls[1::2])
    assert [player[1] for player, _opponent, _sfen in calls[::2]] == [31, 33]
    assert [player[1] for _opponent, player, _sfen in calls[1::2]] == [32, 34]
    assert [cluster.normalized_key for cluster in summary.cluster_results] == [
        OpeningPosition.from_sfen(opening).normalized_key for opening in openings
    ]
    assert summary.promoted

    duplicate_counter = " ".join([*openings[0].split()[:3], "999"])
    with pytest.raises(ValueError, match="duplicate normalized opening"):
        arena_module.benchmark_checkpoint_vs_external(
            tmp_path / "checkpoint",
            external,  # type: ignore[arg-type]
            SearchConfig(simulations=1),
            openings=[openings[0], duplicate_counter],
            promotion_min_pairs=2,
            bootstrap_iterations=10,
        )


def test_improve_cli_wires_opening_suite_and_pair_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[SelfImprovementConfig] = []

    def fake_run(
        _champion: Path,
        _workdir: Path,
        config: SelfImprovementConfig,
        *,
        generations: int,
    ) -> list[object]:
        assert generations == 1
        captured.append(config)
        return []

    monkeypatch.setattr(shogi_cli, "run_self_improvement", fake_run)
    suite = tmp_path / "openings.json"
    assert (
        shogi_cli.main(
            [
                "improve",
                str(tmp_path / "champion"),
                str(tmp_path / "workdir"),
                "--opening-suite",
                str(suite),
                "--actor-opening-split",
                "train",
                "--arena-opening-split",
                "heldout",
                "--promotion-min-pairs",
                "40",
                "--arena-bootstrap-iterations",
                "1234",
            ]
        )
        == 0
    )

    assert len(captured) == 1
    assert captured[0].opening_suite == str(suite.resolve())
    assert captured[0].actor_opening_split == "train"
    assert captured[0].arena_opening_split == "heldout"
    assert captured[0].promotion_min_pairs == 40
    assert captured[0].arena_bootstrap_iterations == 1234


def test_one_self_improvement_generation_is_manifested_and_gated(tmp_path: Path) -> None:
    mx.random.seed(0)
    champion = tmp_path / "champion"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), champion, step=0)
    workdir = tmp_path / "loop"
    result = run_generation(
        champion,
        workdir,
        SelfImprovementConfig(
            actor_games=2,
            actor_simulations=1,
            actor_temperature_moves=0,
            teacher_simulations=2,
            reanalyse_fraction=1.0,
            training_steps=1,
            batch_size=2,
            learning_rate=1e-6,
            arena_games=2,
            arena_simulations=1,
            promotion_min_games=2,
            promotion_lower_bound=0.5,
            max_plies=4,
            initial_sfen=MATE_IN_ONE_SFEN,
        ),
    )

    manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
    state = json.loads((workdir / "state.json").read_text(encoding="utf-8"))
    assert result.status == "rejected"
    assert not result.promoted
    assert result.arena.score == 0.5
    assert result.arena.incomplete_games == 0
    assert result.arena.legacy_single_opening
    assert "legacy_single_opening_debug_only" in result.arena.promotion_blockers
    assert manifest["schema"] == 2
    assert manifest["stage"] == "complete"
    assert manifest["deep_teacher_samples"] == 2
    assert manifest["opening_suite"]["mode"] == "legacy_single_opening_debug_only"
    assert manifest["arena_opening_normalized_keys"]
    assert manifest["arena_opening_sha256"]
    assert manifest["arena"]["cluster_results"]
    assert manifest["arena_ci"] == {
        "method": "deterministic_percentile_cluster_bootstrap_one_sided",
        "confidence_level": 0.95,
        "seed": 1_000_000,
        "bootstrap_iterations": 20_000,
    }
    assert Path(result.candidate, "weights.safetensors").is_file()
    assert state["champion"] == str(champion.resolve())


def test_legacy_self_improvement_rejects_v2_champion_before_writing(tmp_path: Path) -> None:
    champion = tmp_path / "champion-v2"
    save_checkpoint(
        upgrade_model_to_canonical_v2(PolicyValueResNet(model_profile("smoke"))),
        champion,
        step=0,
    )
    workdir = tmp_path / "loop"

    with pytest.raises(ValueError, match=r"legacy-only.*no generation was started"):
        run_generation(
            champion,
            workdir,
            SelfImprovementConfig(
                actor_games=1,
                actor_simulations=1,
                actor_temperature_moves=0,
                teacher_simulations=2,
                reanalyse_fraction=1.0,
                training_steps=1,
                batch_size=1,
                learning_rate=1e-6,
                arena_games=2,
                arena_simulations=1,
                promotion_min_games=2,
                max_plies=4,
                initial_sfen=MATE_IN_ONE_SFEN,
            ),
        )
    assert not workdir.exists()


def test_split_opening_generation_records_production_pair_manifest(tmp_path: Path) -> None:
    mx.random.seed(0)
    champion = tmp_path / "champion"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), champion, step=0)
    suite_path = tmp_path / "openings.json"
    suite_path.write_text(
        json.dumps(
            {
                "schema": "meteo-opening-suite-v1",
                "splits": {
                    "actor": [MATE_IN_ONE_SFEN],
                    "arena": _unique_legal_sfens(32),
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_generation(
        champion,
        tmp_path / "loop",
        SelfImprovementConfig(
            actor_games=1,
            actor_simulations=1,
            actor_temperature_moves=0,
            teacher_simulations=2,
            reanalyse_fraction=1.0,
            training_steps=1,
            batch_size=1,
            learning_rate=1e-6,
            arena_simulations=1,
            promotion_min_pairs=32,
            arena_bootstrap_iterations=500,
            max_plies=1,
            opening_suite=str(suite_path),
        ),
    )

    manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
    assert not result.promoted
    assert result.arena.promotion_eligible
    assert result.arena.independent_opening_pairs == 32
    assert result.arena.incomplete_pairs == 32
    assert result.arena.repeated_opening_pairs == 0
    assert len(result.arena.cluster_results) == 32
    assert manifest["opening_suite"]["mode"] == "immutable_split_suite"
    assert len(manifest["arena_opening_normalized_keys"]) == 32
    assert len(manifest["arena_opening_sha256"]) == 32
    assert manifest["training_arena_overlap_keys"] == []
    assert manifest["arena_ci"]["method"] == result.arena.ci_method
    assert manifest["arena_ci"]["seed"] == 1_000_000


def test_interrupted_generation_resumes_from_json_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    champion = tmp_path / "champion"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), champion, step=0)
    config = SelfImprovementConfig(
        actor_games=1,
        actor_simulations=1,
        actor_temperature_moves=0,
        teacher_simulations=2,
        reanalyse_fraction=1.0,
        training_steps=1,
        batch_size=1,
        learning_rate=1e-6,
        arena_games=2,
        arena_simulations=1,
        promotion_min_games=2,
        max_plies=4,
        initial_sfen=MATE_IN_ONE_SFEN,
    )
    original = improvement.batched_self_play

    def interrupt(*_args: object, **_kwargs: object) -> list[object]:
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(improvement, "batched_self_play", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_generation(champion, tmp_path / "loop", config)

    monkeypatch.setattr(improvement, "batched_self_play", original)
    result = run_generation(champion, tmp_path / "loop", config)
    assert result.status == "rejected"
    assert json.loads(Path(result.manifest).read_text(encoding="utf-8"))["stage"] == "complete"
