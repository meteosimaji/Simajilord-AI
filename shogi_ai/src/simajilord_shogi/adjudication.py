"""Shared legal terminal adjudication for games and search roots."""

from __future__ import annotations

from dataclasses import dataclass

from rsshogi.core import Board
from rsshogi.types import Color, PieceType, RepetitionState

from .domain import Termination


@dataclass(frozen=True, slots=True)
class BoardAdjudication:
    winner: int | None
    termination: Termination


_CSA27_MAJOR_TYPES = frozenset(
    {
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.HORSE,
        PieceType.DRAGON,
    }
)


def can_declare_win_csa27(board: Board) -> bool:
    """Return whether the side to move can declare under YaneuraOu CSARule27.

    ``rsshogi==1.1.0`` constructs Python ``Board`` objects with its internal
    entering-king rule set to ``None`` and exposes no Python rule setter.  Its
    otherwise-correct ``Board.can_declare_win()`` therefore always returns
    false for boards created from SFEN.  Meteo and its pinned YaneuraOu
    teachers use ``EnteringKingRule=CSARule27``, so enforce that exact standard
    rule here: king in the enemy camp, not in check, at least ten non-king
    friendly pieces in that camp, and 28 points for black or 27 for white.
    """

    # A few unit-test doubles intentionally expose only the old native method.
    # Production callers pass the concrete rsshogi Board and take the exact
    # branch below.
    if not isinstance(board, Board):
        return board.can_declare_win()

    if board.is_in_check():
        return False
    us = board.turn
    king = board.king_square(us)
    if king.is_none():
        return False

    def in_enemy_camp(rank: int) -> bool:
        return rank <= 2 if us == Color.BLACK else rank >= 6

    if not in_enemy_camp(king.rank):
        return False

    camp_pieces = tuple(
        piece
        for square, piece in board.iter_pieces()
        if piece.color == us and in_enemy_camp(square.rank)
    )
    # The king is included in this count, matching YaneuraOu's p1 >= 11.
    if len(camp_pieces) < 11:
        return False

    points = sum(
        0
        if piece.piece_type == PieceType.KING
        else 5
        if piece.piece_type in _CSA27_MAJOR_TYPES
        else 1
        for piece in camp_pieces
    )
    hand = board.hand_counts(us)
    points += sum(hand[piece] for piece in ("P", "L", "N", "S", "G"))
    points += 5 * (hand["B"] + hand["R"])
    required = 28 if us == Color.BLACK else 27
    return points >= required


def terminal_repetition_adjudication(board: Board) -> BoardAdjudication | None:
    """Return only a rules-terminal fourfold-repetition result.

    ``SUPERIOR`` and ``INFERIOR`` are dominance-cycle search states: the board
    pieces match an earlier position while one side's hand has grown or shrunk.
    They are useful search information, but they are not themselves a legal
    game termination. Only an exact draw or a perpetual-check win/loss ends
    the game here.
    """

    repetition = board.repetition_state()
    if repetition == RepetitionState.DRAW:
        return BoardAdjudication(None, Termination.REPETITION)
    if repetition == RepetitionState.WIN:
        return BoardAdjudication(board.turn.value, Termination.REPETITION)
    if repetition == RepetitionState.LOSE:
        return BoardAdjudication(board.turn.opponent().value, Termination.REPETITION)
    return None


def adjudicate_board(board: Board) -> BoardAdjudication | None:
    """Return a legal terminal result, or ``None`` while play may continue."""

    repetition = terminal_repetition_adjudication(board)
    if repetition is not None:
        return repetition
    if can_declare_win_csa27(board):
        return BoardAdjudication(board.turn.value, Termination.DECLARATION)
    if board.is_mated() or not board.legal_moves():
        return BoardAdjudication(board.turn.opponent().value, Termination.CHECKMATE)
    return None
