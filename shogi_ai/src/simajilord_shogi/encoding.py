"""Exact standard dlshogi board features implemented on top of rsshogi."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT, move_label
from rsshogi.types import Color, PieceType, Square

from .adjudication import can_declare_win_csa27

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
HISTORY_LENGTH = 8
HISTORY_FEATURES_PER_MOVE = 5
HISTORY_SUMMARY_CHANNELS = 6
HISTORY_INPUT_CHANNELS = HISTORY_LENGTH * HISTORY_FEATURES_PER_MOVE + HISTORY_SUMMARY_CHANNELS

FloatFeatures = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class HistoryInput:
    """Exact recorded prefix used to build history-input-v2 features.

    ``complete`` means the prefix reaches the standard initial position.  A
    replay that intentionally starts from a mid-game opening root remains
    exact relative to that root, but uses ``complete=False`` because moves
    before the root are not available to inference from the same position.
    """

    initial_sfen: str
    moves: tuple[str, ...]
    target_sfen: str
    complete: bool = True


def _position_key(board: Board) -> str:
    return " ".join(board.to_sfen().split()[:3])


def _canonical_history(history: HistoryInput) -> tuple[Board, list[tuple[Move, int]]]:
    try:
        board = Board(history.initial_sfen)
        target = Board(history.target_sfen)
    except (TypeError, ValueError) as error:
        raise ValueError("history input contains invalid SFEN") from error
    if not board.is_valid() or not target.is_valid():
        raise ValueError("history input contains an invalid board")
    expected_complete = board.to_sfen() == Board().to_sfen()
    if history.complete != expected_complete:
        raise ValueError(
            "history completeness must be derived from the standard initial position"
        )
    descriptors: list[tuple[Move, int]] = []
    for ply, move_usi in enumerate(history.moves):
        try:
            move = Move.from_usi(move_usi)
        except (TypeError, ValueError) as error:
            raise ValueError(f"history input has invalid move at ply {ply}") from error
        if not board.is_legal_move(move):
            raise ValueError(f"history input has illegal move {move_usi!r} at ply {ply}")
        if move.is_drop():
            dropped_piece_type = move.dropped_piece_type
            if dropped_piece_type is None:
                raise ValueError(f"history input drop lacks a piece type at ply {ply}")
            piece_type = dropped_piece_type.value
        else:
            source_piece = board.piece_on(move.from_sq)
            if source_piece is None:
                raise ValueError(f"history input move lacks a source piece at ply {ply}")
            piece_type = source_piece.piece_type.value
        descriptors.append((move, int(piece_type)))
        board.apply_move(move)
    if board.to_sfen() != target.to_sfen():
        raise ValueError("history input replay does not equal its target SFEN")
    return board, descriptors


def history_input_from_board(board: Board) -> HistoryInput:
    """Recover the complete available prefix from an rsshogi board history.

    Search boards created with ``Board.copy()`` retain their move stack.  A
    board reconstructed from bare SFEN has no stack and is represented with
    ``complete=False`` so the network cannot confuse missing history with a
    verified empty game prefix.
    """

    target_sfen = board.to_sfen()
    probe = board.copy()
    reversed_moves: list[str] = []
    while True:
        last = probe.last_move()
        if last is None:
            break
        move = Move.from_usi(last.to_usi())
        reversed_moves.append(move.to_usi())
        probe.undo_move(move)
    moves = tuple(reversed(reversed_moves))
    complete = probe.to_sfen() == Board().to_sfen()
    return HistoryInput(
        initial_sfen=probe.to_sfen(),
        moves=moves,
        target_sfen=target_sfen,
        complete=complete,
    )


def encode_history(history: HistoryInput) -> FloatFeatures:
    """Encode recent moves and rules-sensitive summaries into 46 planes.

    For each of the latest eight moves, newest first, five planes encode source
    square, destination square, piece type, drop, and promotion.  Six summary
    planes encode current-position occurrence count, black/white consecutive
    checking-move counts, entering-king availability, absolute game ply, and
    whether the supplied prefix reaches the standard initial position.
    Every square is oriented to the current side to move, matching the 119
    standard dlshogi planes.
    """

    board, descriptors = _canonical_history(history)
    result = np.zeros((HISTORY_INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    turn = board.turn
    for slot, (move, piece_type) in enumerate(reversed(descriptors[-HISTORY_LENGTH:])):
        base = slot * HISTORY_FEATURES_PER_MOVE
        if not move.is_drop():
            oriented_from = _oriented_index(move.from_sq, turn)
            from_rank, from_file = divmod(oriented_from, BOARD_SIZE)
            result[base, from_rank, from_file] = 1.0
        oriented_to = _oriented_index(move.to_sq, turn)
        to_rank, to_file = divmod(oriented_to, BOARD_SIZE)
        result[base + 1, to_rank, to_file] = 1.0
        result[base + 2] = min(max(piece_type, 1), 14) / 14.0
        result[base + 3] = float(move.is_drop())
        result[base + 4] = float(move.is_promotion())

    replay = Board(history.initial_sfen)
    current_key = _position_key(board)
    occurrence_count = int(_position_key(replay) == current_key)
    consecutive_checks = {Color.BLACK.value: 0, Color.WHITE.value: 0}
    for move_usi in history.moves:
        mover = replay.turn.value
        move = Move.from_usi(move_usi)
        replay.apply_move(move)
        occurrence_count += int(_position_key(replay) == current_key)
        if replay.is_in_check():
            consecutive_checks[mover] += 1
        else:
            consecutive_checks[mover] = 0
    summary = HISTORY_LENGTH * HISTORY_FEATURES_PER_MOVE
    result[summary] = min(occurrence_count, 4) / 4.0
    result[summary + 1] = min(consecutive_checks[Color.BLACK.value], 8) / 8.0
    result[summary + 2] = min(consecutive_checks[Color.WHITE.value], 8) / 8.0
    result[summary + 3] = float(can_declare_win_csa27(board))
    result[summary + 4] = min(max(board.game_ply - 1, 0), 511) / 511.0
    result[summary + 5] = float(history.complete)
    return result


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


def combined_features_with_history(history: HistoryInput) -> FloatFeatures:
    """Return 119 board planes plus the exact history-input-v2 planes."""

    board, _descriptors = _canonical_history(history)
    return np.concatenate((*encode_board(board), encode_history(history)), axis=0)


def board_from_history_input(history: HistoryInput) -> Board:
    """Replay a validated prefix while preserving the board's move stack."""

    board, _descriptors = _canonical_history(history)
    return board


def history_input_sha256(history: HistoryInput) -> str:
    """Return a stable exact-history identity for split and dedup boundaries."""

    board, _descriptors = _canonical_history(history)
    payload = {
        "schema": "meteo-history-input-v2-identity",
        "initial_sfen": Board(history.initial_sfen).to_sfen(),
        "moves": list(history.moves),
        "target_sfen": board.to_sfen(),
        "complete": history.complete,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combined_features_from_board(board: Board, *, history_input_version: int) -> FloatFeatures:
    """Encode an inference board according to a checkpoint's input contract."""

    if history_input_version == 1:
        return combined_features(board)
    if history_input_version == 2:
        return combined_features_with_history(history_input_from_board(board))
    raise ValueError(f"unsupported history input version: {history_input_version}")


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
