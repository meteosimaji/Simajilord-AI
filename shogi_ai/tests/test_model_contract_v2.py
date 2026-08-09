from __future__ import annotations

import mlx.core as mx
import numpy as np
from rsshogi.core import Board

from simajilord_shogi.config import model_profile
from simajilord_shogi.encoding import (
    HistoryInput,
    combined_features,
    combined_features_with_history,
)
from simajilord_shogi.model import (
    PolicyValueResNet,
    assert_model_shapes,
    upgrade_model_to_canonical_v2,
)


def test_v1_to_v2_upgrade_preserves_play_predictions_and_separates_teacher_heads() -> None:
    legacy = PolicyValueResNet(model_profile("smoke"))
    legacy.eval()
    board = Board()
    legacy_inputs = mx.array(np.stack([combined_features(board)]))
    legacy_policy, legacy_value = legacy(legacy_inputs)
    mx.eval(legacy_policy, legacy_value)

    upgraded = upgrade_model_to_canonical_v2(legacy)
    upgraded.eval()
    history = HistoryInput(
        initial_sfen=board.to_sfen(),
        moves=(),
        target_sfen=board.to_sfen(),
    )
    upgraded_inputs = mx.array(np.stack([combined_features_with_history(history)]))
    play_policy, play_value = upgraded(upgraded_inputs)
    canonical = upgraded.forward_canonical(upgraded_inputs)
    mx.eval(
        play_policy,
        play_value,
        canonical.policy_teachers,
        canonical.value_teachers,
    )

    np.testing.assert_array_equal(np.asarray(play_policy), np.asarray(legacy_policy))
    np.testing.assert_array_equal(np.asarray(play_value), np.asarray(legacy_value))
    np.testing.assert_array_equal(
        np.asarray(canonical.policy_teachers[:, 0]), np.asarray(play_policy)
    )
    np.testing.assert_array_equal(
        np.asarray(canonical.policy_teachers[:, 1]), np.asarray(play_policy)
    )
    np.testing.assert_array_equal(
        np.asarray(canonical.value_teachers[:, 0:1]), np.asarray(play_value)
    )
    np.testing.assert_array_equal(
        np.asarray(canonical.value_teachers[:, 1:2]), np.asarray(play_value)
    )
    assert upgraded.config.history_input_version == 2
    assert upgraded.config.canonical_head_version == 2
    assert_model_shapes(upgraded, batch_size=1)
