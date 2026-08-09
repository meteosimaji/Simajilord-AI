from __future__ import annotations

import math
from pathlib import Path

import pytest
from rsshogi.core import Board

from simajilord_shogi.config import SearchConfig, optional_max_plies
from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.evaluator import UniformEvaluator
from simajilord_shogi.game import play_game
from simajilord_shogi.replay import (
    append_games,
    evaluation_value_telemetry,
    load_games,
    position_samples,
    search_telemetry,
)
from simajilord_shogi.usi import UsiEngine

from .test_mcts_game import MATE_IN_ONE_SFEN


def test_replay_round_trip(tmp_path: Path) -> None:
    game = play_game(
        UniformEvaluator(),
        SearchConfig(
            simulations=1,
            root_min_visits=1,
            max_plies=4,
            temperature_moves=0,
            resign_threshold=None,
            dirichlet_fraction=0,
        ),
        initial_sfen=MATE_IN_ONE_SFEN,
        self_play_noise=False,
    )
    path = tmp_path / "games.jsonl"

    assert append_games(path, [game]) == 1
    assert load_games(path) == [game]
    telemetry = search_telemetry([game])["meteo"]
    assert telemetry["nodes"] > 0
    assert telemetry["seconds"] > 0
    assert telemetry["nps"] > 0


def test_evaluation_value_telemetry_is_source_separated_and_fail_closed() -> None:
    start = Board().to_sfen()
    samples = (
        PositionSample(start, 0, 0, {}, -0.75, actor_source="meteo"),
        PositionSample(start, 1, 1, {}, 0.25, actor_source="meteo"),
        PositionSample(start, 2, 0, {}, 1.0, actor_source="teacher"),
    )
    game = GameRecord(start, (), samples, None, Termination.REPETITION)

    telemetry = evaluation_value_telemetry([game])

    assert telemetry["meteo"] == {
        "positions": 2,
        "mean": -0.25,
        "mean_absolute": 0.5,
        "root_mean_square": pytest.approx(math.sqrt(0.3125)),
        "minimum": -0.75,
        "maximum": 0.25,
        "negative": 1,
        "zero": 0,
        "positive": 1,
    }
    assert telemetry["teacher"]["mean"] == 1.0

    invalid = GameRecord(
        start,
        (),
        (PositionSample(start, 0, 0, {}, math.nan, actor_source="broken"),),
        None,
        Termination.REPETITION,
    )
    with pytest.raises(ValueError, match="root value must be finite"):
        evaluation_value_telemetry([invalid])


def test_usi_handshake_position_and_go() -> None:
    engine = UsiEngine(
        UniformEvaluator(),
        SearchConfig(simulations=1, root_min_visits=1, dirichlet_fraction=0),
    )
    handshake = engine.handle("usi")
    assert handshake[0] == "id name めてお / Meteo"
    assert handshake[1] == "id author meteosimaji"
    assert handshake[-1] == "usiok"
    assert engine.handle("isready") == ["readyok"]
    assert engine.handle(f"position sfen {MATE_IN_ONE_SFEN}") == []
    output = engine.handle("go nodes 1")
    assert " nodes " in f" {output[0]} "
    assert " nps " in f" {output[0]} "
    assert " time " in f" {output[0]} "
    assert output[-1] == "bestmove G*5b"


def test_incomplete_cutoff_is_not_training_data() -> None:
    incomplete = GameRecord(
        initial_sfen=Board().to_sfen(),
        moves=(),
        samples=(),
        winner=None,
        termination=Termination.MAX_PLIES,
    )

    assert position_samples([incomplete]) == []


def test_zero_ply_cutoff_means_a_rules_complete_game() -> None:
    assert optional_max_plies(0) is None
    assert optional_max_plies(-1) is None
    assert optional_max_plies(3001) == 3001
