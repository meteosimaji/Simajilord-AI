from __future__ import annotations

import numpy as np
from rsshogi.core import Board, Move
from rsshogi.policy import MOVE_LABEL_COUNT

from simajilord_shogi.config import model_profile
from simajilord_shogi.encoding import (
    FEATURES1_NUM,
    FEATURES2_NUM,
    encode_board,
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
