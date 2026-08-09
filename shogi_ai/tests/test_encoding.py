from __future__ import annotations

import numpy as np
import pytest
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT

from simajilord_shogi.config import model_profile
from simajilord_shogi.encoding import (
    FEATURES1_NUM,
    FEATURES2_NUM,
    HISTORY_INPUT_CHANNELS,
    HistoryInput,
    combined_features_with_history,
    encode_board,
    history_input_from_board,
    legal_policy_labels,
    legal_policy_mask,
)


def test_initial_features_and_legal_labels() -> None:
    board = Board()
    feature1, feature2 = encode_board(board)

    assert feature1.shape == (FEATURES1_NUM, 9, 9)
    assert feature2.shape == (FEATURES2_NUM, 9, 9)
    assert feature1.dtype == np.float32
    assert feature2.dtype == np.float32
    assert int(feature1[:14].sum() + feature1[31:45].sum()) == 40
    assert not feature2.any()

    labels = legal_policy_labels(board)
    mask = legal_policy_mask(board)
    assert len(labels) == 30
    assert len(set(labels.values())) == 30
    assert mask.shape == (MOVE_LABEL_COUNT,)
    assert int(mask.sum()) == 30


def test_check_plane_is_last_and_full() -> None:
    board = Board("4k4/9/4R4/9/9/9/9/9/4K4 b G 1")
    board.apply_move(Move.from_usi("G*5b"))
    _, feature2 = encode_board(board)

    assert board.is_mated()
    assert np.all(feature2[-1] == 1.0)


def test_competition_v1_uses_the_public_dlshogi_resnet_shape() -> None:
    profile = model_profile("competition_v1")

    assert profile.dlshogi_legacy is True
    assert profile.residual_blocks == 20
    assert profile.channels == 256
    assert profile.policy_channels == 27
    assert profile.value_channels == 27


def test_same_sfen_with_different_exact_histories_has_different_v2_tensor() -> None:
    first_moves = ("7g7f", "3c3d", "2g2f", "8c8d")
    second_moves = ("2g2f", "8c8d", "7g7f", "3c3d")
    first_board = Board()
    second_board = Board()
    for move_usi in first_moves:
        first_board.apply_move(Move.from_usi(move_usi))
    for move_usi in second_moves:
        second_board.apply_move(Move.from_usi(move_usi))
    assert first_board.to_sfen() == second_board.to_sfen()

    first = combined_features_with_history(
        HistoryInput(
            initial_sfen=Board().to_sfen(),
            moves=first_moves,
            target_sfen=first_board.to_sfen(),
        )
    )
    second = combined_features_with_history(
        HistoryInput(
            initial_sfen=Board().to_sfen(),
            moves=second_moves,
            target_sfen=second_board.to_sfen(),
        )
    )

    assert first.shape == (FEATURES1_NUM + FEATURES2_NUM + HISTORY_INPUT_CHANNELS, 9, 9)
    board_channels = FEATURES1_NUM + FEATURES2_NUM
    np.testing.assert_array_equal(first[:board_channels], second[:board_channels])
    assert not np.array_equal(first[board_channels:], second[board_channels:])


def test_bare_midgame_sfen_is_not_mislabelled_as_complete_history() -> None:
    bare_midgame = Board("4k4/9/9/9/9/9/9/9/4K4 b - 1")

    history = history_input_from_board(bare_midgame)

    assert history.moves == ()
    assert history.complete is False
    assert history_input_from_board(Board()).complete is True


def test_arbitrary_root_remains_incomplete_after_recorded_moves() -> None:
    initial_sfen = "4k4/9/9/9/9/9/9/9/4K4 b R 1"
    root = Board(initial_sfen)
    root.apply_move(Move.from_usi("R*5e"))

    history = history_input_from_board(root)

    assert history.moves == ("R*5e",)
    assert history.complete is False
    np.testing.assert_array_equal(
        combined_features_with_history(history),
        combined_features_with_history(
            HistoryInput(
                initial_sfen=initial_sfen,
                moves=("R*5e",),
                target_sfen=root.to_sfen(),
                complete=False,
            )
        ),
    )

    with pytest.raises(ValueError, match="must be derived"):
        combined_features_with_history(
            HistoryInput(
                initial_sfen=initial_sfen,
                moves=("R*5e",),
                target_sfen=root.to_sfen(),
                complete=True,
            )
        )
