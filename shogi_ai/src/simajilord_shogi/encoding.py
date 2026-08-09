"""Exact standard dlshogi board features implemented on top of rsshogi."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT, move_label
from rsshogi.types import Color, PieceType, Square

BOARD_SIZE = 9
SQUARES = 81
PIECE_TYPES: tuple[PieceType, ...] = (
    PieceType.PAWN,
    PieceType.LANCE,
    PieceType.KNIGHT,
    PieceType.SILVER,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.GOLD,
    PieceType.KING,
    PieceType.PRO_PAWN,
    PieceType.PRO_LANCE,
    PieceType.PRO_KNIGHT,
    PieceType.PRO_SILVER,
    PieceType.HORSE,
    PieceType.DRAGON,
)
HAND_TYPES: tuple[PieceType, ...] = PIECE_TYPES[:7]
MAX_HAND_COUNTS: tuple[int, ...] = (8, 4, 4, 4, 4, 2, 2)
FEATURES1_NUM = 62
FEATURES2_NUM = 57
INPUT_CHANNELS = FEATURES1_NUM + FEATURES2_NUM

FloatFeatures = NDArray[np.float32]


def _oriented_index(square: Square, turn: Color) -> int:
    return square.value if turn == Color.BLACK else SQUARES - 1 - square.value


def encode_board(board: Board) -> tuple[FloatFeatures, FloatFeatures]:
    """Return the standard dlshogi (62, 9, 9) and (57, 9, 9) planes.

    The ordering and side-to-move rotation follow cshogi's
    `_dlshogi_make_input_features`; the final feature2 plane is check, not turn.
    """

    features1 = np.zeros((FEATURES1_NUM, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    features2 = np.zeros((FEATURES2_NUM, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    turn = board.turn
    occupied = board.pieces()

    for absolute_color in (Color.BLACK, Color.WHITE):
        relative_color = absolute_color.value if turn == Color.BLACK else 1 - absolute_color.value
        base = relative_color * 31

        for piece_index, piece_type in enumerate(PIECE_TYPES):
            pieces = board.pieces_for(piece_type, absolute_color)
            attacks = board.attacks_by(absolute_color, piece_type)
            for square_value in range(SQUARES):
                square = Square.from_index(square_value)
                oriented = _oriented_index(square, turn)
                rank, file = divmod(oriented, BOARD_SIZE)
                if pieces.test(square):
                    features1[base + piece_index, rank, file] = 1.0
                if attacks.test(square):
                    features1[base + 14 + piece_index, rank, file] = 1.0

        for square_value in range(SQUARES):
            square = Square.from_index(square_value)
            attack_count = min(
                3, board.attackers_to_color(absolute_color, square, occupied).count()
            )
            oriented = _oriented_index(square, turn)
            rank, file = divmod(oriented, BOARD_SIZE)
            for threshold in range(attack_count):
                features1[base + 28 + threshold, rank, file] = 1.0

        hand_offset = relative_color * sum(MAX_HAND_COUNTS)
        running_offset = 0
        hand = board.hand(absolute_color)
        for piece_type, maximum in zip(HAND_TYPES, MAX_HAND_COUNTS, strict=True):
            count = min(hand.count(piece_type), maximum)
            features2[hand_offset + running_offset : hand_offset + running_offset + count] = 1.0
            running_offset += maximum

    if board.is_in_check():
        features2[-1] = 1.0
    return features1, features2


def combined_features(board: Board) -> FloatFeatures:
    """Return all input planes concatenated in channel-first order."""

    feature1, feature2 = encode_board(board)
    return np.concatenate((feature1, feature2), axis=0)


def legal_policy_labels(board: Board) -> dict[str, int]:
    """Map each legal USI move to its side-oriented dlshogi policy label."""

    return {move.to_usi(): move_label(move, board.turn) for move in board.legal_moves()}


def legal_policy_mask(board: Board) -> NDArray[np.bool_]:
    mask = np.zeros(MOVE_LABEL_COUNT, dtype=np.bool_)
    for move in board.legal_moves():
        mask[move_label(move, board.turn)] = True
    return mask


def label_for_move(board: Board, move: Move) -> int:
    if not board.is_legal_move(move):
        raise ValueError(f"illegal move for current position: {move.to_usi()}")
    return move_label(move, board.turn)
