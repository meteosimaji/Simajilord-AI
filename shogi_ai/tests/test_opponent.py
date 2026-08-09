from __future__ import annotations

from pathlib import Path

from rsshogi.core import Board

from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.evaluator import UniformEvaluator
from simajilord_shogi.opponent import OpponentConditionedEvaluator, OpponentProfile

from .test_mcts_game import MATE_IN_ONE_SFEN


def _observed_game() -> GameRecord:
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
        chosen_move="G*5b",
        actor_best_move="G*5b",
        teacher_best_move="6c7d",
        teacher_regret=0.4,
    )
    return GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=("G*5b",),
        samples=(sample,),
        winner=0,
        termination=Termination.CHECKMATE,
    )


def test_profile_learns_tendency_weakness_and_round_trips(tmp_path: Path) -> None:
    profile = OpponentProfile("opponent")
    profile.observe_game(_observed_game(), opponent_color=0)

    prediction = profile.predict(Board(MATE_IN_ONE_SFEN), history=[])
    assert prediction == {"G*5b": 1.0}
    assert profile.likely_continuations([]) == [("G*5b", 1.0)]
    assert profile.weaknesses[0].regret == 0.4

    path = tmp_path / "opponent.json"
    profile.save(path)
    assert OpponentProfile.load(path) == profile


def test_conditioned_prior_keeps_every_legal_reply() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    profile = OpponentProfile("opponent")
    profile.observe_game(_observed_game(), opponent_color=0)
    evaluator = OpponentConditionedEvaluator(
        UniformEvaluator(), profile, opponent_color=0, exploit_weight=0.3
    )

    result = evaluator.evaluate(board)
    assert result.policy["G*5b"] > min(result.policy.values())
    assert all(probability > 0 for probability in result.policy.values())
