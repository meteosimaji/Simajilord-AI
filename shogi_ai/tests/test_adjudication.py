from __future__ import annotations

from typing import cast

from rsshogi.core import Board, Move
from rsshogi.types import Color, RepetitionState

from simajilord_shogi.adjudication import (
    adjudicate_board,
    can_declare_win_csa27,
    terminal_repetition_adjudication,
)
from simajilord_shogi.domain import Termination
from simajilord_shogi.mcts import terminal_value

INFERIOR_CYCLE_SFEN = (
    "ln1g1gsnl/1r1s1k1b1/2p1pp2p/p2p2pp1/1p6P/2P1P4/"
    "PPGP1PPP1/1B2RK3/LNS2GSNL b - 17"
)
INFERIOR_CYCLE_MOVES = (
    "9g9f",
    "9a9c",
    "7f7e",
    "3a3b",
    "9i9g",
    "3b3c",
    "8g8f",
    "4c4d",
    "4g4f",
    "8e8f",
    "6g6f",
    "6b6c",
    "5f5e",
    "6c5d",
    "4h5i",
    "8f8g+",
    "7g8g",
    "P*8f",
    "8g7g",
)


class _SyntheticBoard:
    def __init__(self, repetition: RepetitionState) -> None:
        self._repetition = repetition
        self.turn = Color.BLACK

    def repetition_state(self) -> RepetitionState:
        return self._repetition

    def can_declare_win(self) -> bool:
        return False

    def is_mated(self) -> bool:
        return False

    def legal_moves(self) -> list[Move]:
        return [Move.from_usi("7g7f")]


def _synthetic_board(repetition: RepetitionState) -> Board:
    return cast(Board, _SyntheticBoard(repetition))


def test_only_rules_terminal_repetitions_are_adjudicated() -> None:
    assert terminal_repetition_adjudication(_synthetic_board(RepetitionState.NONE)) is None
    assert terminal_repetition_adjudication(_synthetic_board(RepetitionState.SUPERIOR)) is None
    assert terminal_repetition_adjudication(_synthetic_board(RepetitionState.INFERIOR)) is None

    draw = terminal_repetition_adjudication(_synthetic_board(RepetitionState.DRAW))
    win = terminal_repetition_adjudication(_synthetic_board(RepetitionState.WIN))
    loss = terminal_repetition_adjudication(_synthetic_board(RepetitionState.LOSE))

    assert draw is not None and draw.winner is None
    assert win is not None and win.winner == Color.BLACK.value
    assert loss is not None and loss.winner == Color.WHITE.value
    assert draw.termination is Termination.REPETITION
    assert win.termination is Termination.REPETITION
    assert loss.termination is Termination.REPETITION


def test_mcts_values_draw_and_perpetual_check_without_ending_dominance_cycles() -> None:
    assert terminal_value(_synthetic_board(RepetitionState.DRAW)) == 0.0
    assert terminal_value(_synthetic_board(RepetitionState.WIN)) == 1.0
    assert terminal_value(_synthetic_board(RepetitionState.LOSE)) == -1.0
    assert terminal_value(_synthetic_board(RepetitionState.SUPERIOR)) is None
    assert terminal_value(_synthetic_board(RepetitionState.INFERIOR)) is None


def test_observed_inferior_cycle_is_playable_instead_of_an_immediate_loss() -> None:
    board = Board(INFERIOR_CYCLE_SFEN)
    for move_usi in INFERIOR_CYCLE_MOVES:
        board.apply_move(Move.from_usi(move_usi))

    assert board.repetition_state() == RepetitionState.INFERIOR
    assert adjudicate_board(board) is None
    assert terminal_value(board) is None
    assert board.legal_moves()


def test_csa27_declaration_matches_yaneuraou_for_black_and_white() -> None:
    black = Board(
        "K+N5+L1/G+L+P+B1+R+P2/3+P2G2/9/2+p+n5/3s2+ss1/"
        "3+p+p1+s1+r/7g+n/6g+nk b 2L8Pb4p 1"
    )
    white = Board(
        "K+N5+L1/G+P+P+B1+R+P2/+P+L1+P2G2/9/9/9/6b1+r/"
        "4+ss+sg+n/5+pg+nk w 2L5Psn7p 1"
    )

    # rsshogi 1.1.0 leaves the native rule disabled for SFEN boards.
    assert not black.can_declare_win()
    assert not white.can_declare_win()
    assert can_declare_win_csa27(black)
    assert can_declare_win_csa27(white)
    assert adjudicate_board(black) is not None
    assert adjudicate_board(white) is not None


def test_csa27_declaration_rejects_insufficient_points() -> None:
    board = Board(
        "K+N5+L1/G+L+P+B1+R+P2/3+P2G2/9/2+p+n5/3s2+ss1/"
        "3+p+p1+s1+r/7g+n/6g+nk b 2L7Pb5p 1"
    )

    assert not can_declare_win_csa27(board)
