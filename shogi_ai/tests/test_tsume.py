from __future__ import annotations

from rsshogi.core import Board

from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.tsume import TsumeSolver, mine_unique_tsume

from .test_mcts_game import MATE_IN_ONE_SFEN

MULTIPLE_MATE_IN_ONE_SFEN = (
    "3rkr3/3p1p3/9/6B2/9/9/9/9/4K4 b B4G4S4N4L16P 1"
)


def test_unique_standard_tsume_is_proved() -> None:
    puzzle = TsumeSolver(node_limit=10_000).solve_unique(Board(MATE_IN_ONE_SFEN), max_plies=1)

    assert puzzle is not None
    assert puzzle.first_move == "G*5b"
    assert puzzle.solution.replies == ()
    assert puzzle.nodes > 0


def test_all_alternative_mating_moves_are_preserved() -> None:
    board = Board(MULTIPLE_MATE_IN_ONE_SFEN)
    assert board.is_valid()
    solver = TsumeSolver(node_limit=10_000)

    solution_set = solver.solve_all(board, max_plies=1)

    assert len(solution_set.solutions) > 1
    assert {"L*5b", "S*5b", "G*5b"} <= set(solution_set.first_moves)
    assert solution_set.nodes > 0
    assert TsumeSolver(node_limit=10_000).solve_unique(board, max_plies=1) is None


def test_mine_tsume_from_replay_position() -> None:
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
    )
    game = GameRecord(
        initial_sfen=MATE_IN_ONE_SFEN,
        moves=("G*5b",),
        samples=(sample,),
        winner=0,
        termination=Termination.CHECKMATE,
    )

    puzzles = mine_unique_tsume([game], max_plies=1)
    assert len(puzzles) == 1
    assert puzzles[0].first_move == "G*5b"
