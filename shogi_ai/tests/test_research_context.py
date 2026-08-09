from __future__ import annotations

from rsshogi.core import Board, Move

from simajilord_shogi.domain import GameRecord, PositionSample, Termination
from simajilord_shogi.ensemble import normalized_sfen
from simajilord_shogi.research_context import trajectory_contexts


def _sample(board: Board, ply: int, move: str) -> PositionSample:
    return PositionSample(
        sfen=board.to_sfen(),
        ply=ply,
        turn=board.turn.value,
        policy={move: 1.0},
        root_value=0.0,
    )


def test_trajectory_context_tracks_phase_neighbors_and_contiguous_chunks() -> None:
    board = Board()
    initial = board.to_sfen()
    first = _sample(board, 0, "7g7f")
    board.apply_move(Move.from_usi("7g7f"))
    second = _sample(board, 1, "3c3d")
    board.apply_move(Move.from_usi("3c3d"))
    board.apply_move(Move.from_usi("2g2f"))
    fourth = _sample(board, 3, "8c8d")
    game = GameRecord(
        initial_sfen=initial,
        moves=("7g7f", "3c3d", "2g2f", "8c8d"),
        samples=(first, second, fourth),
        winner=None,
        termination=Termination.REPETITION,
    )

    contexts = trajectory_contexts((game,))
    first_context = contexts[normalized_sfen(first.sfen)]
    second_context = contexts[normalized_sfen(second.sfen)]
    fourth_context = contexts[normalized_sfen(fourth.sfen)]

    assert first_context.phase == "opening_plies_1_24"
    assert first_context.absolute_ply == 0
    assert first_context.next_is_consecutive
    assert second_context.previous_is_consecutive
    assert not second_context.next_is_consecutive
    assert first_context.chunk_start_ply == 0
    assert second_context.chunk_end_ply == 1
    assert fourth_context.chunk_start_ply == fourth_context.chunk_end_ply == 3
    assert first_context.game_id == second_context.game_id == fourth_context.game_id
    assert first_context.opening_family == fourth_context.opening_family
