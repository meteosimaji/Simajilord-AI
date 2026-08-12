from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rsshogi.core import Board, Move

from simajilord_shogi.cli import main
from simajilord_shogi.config import ReanalysisConfig, SearchConfig
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.encoding import history_input_from_board
from simajilord_shogi.evaluator import Evaluation
from simajilord_shogi.external_usi import ExternalTeacherPolicy, ExternalUsiTeacher
from simajilord_shogi.reanalysis import (
    reanalyse_game,
    reanalyse_game_external,
    reanalyse_games,
    sample_weight,
)
from simajilord_shogi.replay import append_games, load_games

from .test_mcts_game import MATE_IN_ONE_SFEN, MateMoveHasTinyPrior


class _HistoryInspectingEvaluator:
    def __init__(self, expected_prefix: tuple[str, ...]) -> None:
        self.expected_prefix = expected_prefix
        self.calls = 0

    def evaluate(self, board: Board) -> Evaluation:
        history = history_input_from_board(board)
        assert history.moves[: len(self.expected_prefix)] == self.expected_prefix
        legal = [move.to_usi() for move in board.legal_moves()]
        self.calls += 1
        return Evaluation(
            policy={move: 1.0 / len(legal) for move in legal},
            value=0.0,
        )


class _BatchInspectingEvaluator:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @staticmethod
    def evaluate(board: Board) -> Evaluation:
        legal = [move.to_usi() for move in board.legal_moves()]
        return Evaluation(
            policy={move: 1.0 / len(legal) for move in legal},
            value=0.0,
        )

    def evaluate_batch(self, boards: list[Board]) -> list[Evaluation]:
        self.batch_sizes.append(len(boards))
        return [self.evaluate(board) for board in boards]


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


def test_same_model_reanalysis_preserves_the_exact_game_prefix() -> None:
    moves = ("7g7f", "3c3d", "2g2f", "8c8d")
    board = Board()
    for move_usi in moves:
        board.apply_usi(move_usi)
    legal_move = board.legal_moves()[0].to_usi()
    sample = PositionSample(
        sfen=board.to_sfen(),
        ply=len(moves),
        turn=board.turn.value,
        policy={legal_move: 1.0},
        root_value=0.0,
        chosen_move=legal_move,
        actor_best_move=legal_move,
    )
    game = GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=moves,
        samples=(sample,),
        winner=None,
        termination=Termination.REPETITION,
    )
    evaluator = _HistoryInspectingEvaluator(moves)

    reanalyse_game(
        game,
        evaluator,
        SearchConfig(simulations=1, root_min_visits=1, dirichlet_fraction=0),
        ReanalysisConfig(
            teacher_simulation_multiplier=2,
            minimum_teacher_simulations=2,
            reanalyse_fraction=1.0,
        ),
    )

    assert evaluator.calls > 0


def test_same_model_reanalysis_batches_independent_game_positions() -> None:
    boards: list[Board] = []
    board = Board()
    for _index in range(4):
        boards.append(Board(board.to_sfen()))
        board.apply_move(board.legal_moves()[0])
    games: list[GameRecord] = []
    for item in boards:
        legal_move = item.legal_moves()[0].to_usi()
        sample = PositionSample(
            sfen=item.to_sfen(),
            ply=0,
            turn=item.turn.value,
            policy={legal_move: 1.0},
            root_value=0.0,
            chosen_move=legal_move,
            actor_best_move=legal_move,
            actor_simulations=1,
        )
        games.append(
            GameRecord(
                initial_sfen=item.to_sfen(),
                moves=(),
                samples=(sample,),
                winner=None,
                termination=Termination.AGREED_DRAW,
            )
        )
    evaluator = _BatchInspectingEvaluator()

    revised = reanalyse_games(
        games,
        evaluator,
        SearchConfig(simulations=1, root_min_visits=1, dirichlet_fraction=0),
        ReanalysisConfig(
            teacher_simulation_multiplier=2,
            minimum_teacher_simulations=2,
            reanalyse_fraction=1.0,
        ),
        max_parallel_positions=4,
    )

    assert max(evaluator.batch_sizes) == 4
    assert all(game.samples[0].teacher_policy is not None for game in revised)


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
        if ' searchmoves ' in command:
            move = command.rsplit(' searchmoves ', 1)[1]
            score = 1200 if move == 'G*5b' else -100
            print(
                f'info depth 8 multipv 1 score cp {score} nodes 321 '
                f'time 10 nps 32100 pv {move}',
                flush=True,
            )
        else:
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
    assert revised_sample.teacher_played_move_value == revised_sample.teacher_value
    assert revised_sample.teacher_played_move_nodes == 321
    assert revised_sample.teacher_played_move_time_ms == 10
    assert revised_sample.teacher_played_move_nps == 32100
    assert revised_sample.teacher_played_move_depth == 8
    assert revised_sample.teacher_played_move_exact
    assert revised_sample.teacher_played_move_variation is not None
    assert revised_sample.teacher_played_move_variation.move == "G*5b"
    assert not revised_sample.policy_reversal

    serialized = json.loads(json.dumps(revised.to_dict()))
    restored = GameRecord.from_dict(serialized).samples[0]
    assert restored.teacher_variations == revised_sample.teacher_variations
    assert restored.teacher_policy_temperature == 250.0
    assert restored.teacher_value_scale == 750.0
    assert (
        restored.teacher_played_move_variation
        == revised_sample.teacher_played_move_variation
    )


def test_external_reanalysis_scores_an_omitted_played_move_with_searchmoves(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_branch_usi.py"
    script.write_text(
        """import sys
for raw in sys.stdin:
    command = raw.strip()
    if command == 'usi':
        print('id name fake-branch', flush=True)
        print('option name MultiPV type spin default 1 min 1 max 32', flush=True)
        print('usiok', flush=True)
    elif command == 'isready':
        print('readyok', flush=True)
    elif command.startswith('go '):
        if ' searchmoves ' in command:
            move = command.rsplit(' searchmoves ', 1)[1]
            score = -300 if move == '6c7d' else 1200
            print(f'info depth 10 score cp {score} nodes 500 pv {move}', flush=True)
            print(f'bestmove {move}', flush=True)
        else:
            print('info depth 10 multipv 1 score cp 1200 nodes 500 pv G*5b', flush=True)
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
        policy={"6c7d": 1.0},
        root_value=0.0,
        chosen_move="6c7d",
        actor_best_move="6c7d",
        actor_simulations=8,
    )
    game = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=("6c7d",),
        samples=(sample,),
        winner=None,
        termination=Termination.RESIGNATION,
    )
    policy = ExternalTeacherPolicy(
        policy_id="test-exact-played-move",
        name="exact-played-move",
        source="test fixture",
        analysis_allowed=True,
        training_outputs_allowed=True,
        redistribution_allowed=True,
    )

    with ExternalUsiTeacher(
        [sys.executable, str(script)],
        policy,
        nodes=500,
        multipv=1,
        value_scale=1_200.0,
    ) as teacher:
        revised = reanalyse_game_external(
            game,
            teacher,
            ReanalysisConfig(reanalyse_fraction=1.0),
            selection="all",
        )

    result = revised.samples[0]
    assert result.teacher_best_move == "G*5b"
    assert result.teacher_policy is not None
    assert "6c7d" not in result.teacher_policy
    assert result.teacher_played_move_value == pytest.approx(
        -0.24491866240370913
    )
    assert result.teacher_played_move_exact
    assert result.teacher_regret == pytest.approx(
        0.7615941559557649 - (-0.24491866240370913)
    )
    assert not result.teacher_regret_is_lower_bound


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
        if ' searchmoves ' in command:
            move = command.rsplit(' searchmoves ', 1)[1]
            score = 1200 if move == 'G*5b' else -100
            print(f'info depth 8 multipv 1 score cp {score} nodes 321 pv {move}', flush=True)
        else:
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
    assert provenance["played_moves_searchmoves_evaluated"] == 1
    assert provenance["played_moves_exact_bound"] == 1
    assert provenance["reported_played_move_search_nodes"] == 321

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
        "position startpos",
        "position startpos moves 7g7f",
        "position startpos moves 7g7f",
    ]
    assert [sample.teacher_best_move for sample in revised.samples] == ["7g7f", "3c3d"]
    assert all(sample.teacher_played_move_exact for sample in revised.samples)
    assert [sample.teacher_played_move_value for sample in revised.samples] == [
        sample.teacher_value for sample in revised.samples
    ]
