from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from rsshogi.core import Board, Move
from rsshogi.policy import move_label

from simajilord_shogi.config import model_profile
from simajilord_shogi.distillation_targets_v2 import (
    CanonicalDistillationTargetV2,
    GuardedPlayTarget,
    PlayTargetKind,
    load_canonical_target_sidecar_v2,
)
from simajilord_shogi.domain import GameRecord, PositionSample
from simajilord_shogi.model import PolicyValueResNet, upgrade_model_to_canonical_v2
from simajilord_shogi.trainer import (
    _canonical_v2_clip_gradients,
    _canonical_v2_loss,
    _canonical_v2_sample_indices,
    _make_canonical_v2_batch,
    train,
    train_resumable,
)

from .canonical_v2_fixtures import (
    build_canonical_v2_payload,
    write_canonical_v2_replay,
    write_canonical_v2_sidecar,
)


def _loaded_fixture(
    tmp_path: Path, *, coverage_complete: bool = True
) -> tuple[GameRecord, tuple[CanonicalDistillationTargetV2, ...]]:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    write_canonical_v2_sidecar(
        sidecar,
        build_canonical_v2_payload(
            replay,
            game,
            coverage_complete=coverage_complete,
        ),
    )
    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
    return game, loaded.positions


def _canonical_v2_test_step(
    model: PolicyValueResNet,
    samples: list[PositionSample],
    targets: list[CanonicalDistillationTargetV2],
    *,
    learning_rate: float = 1e-3,
) -> tuple[float, float]:
    """Exercise the v2 loss in a synthetic unit test without a production entry."""

    batch = _make_canonical_v2_batch(samples, targets)
    model.eval()

    def loss_fn(*items: mx.array) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        return _canonical_v2_loss(model, *items, value_loss_weight=1.0)

    initial, _ = loss_fn(*batch)
    mx.eval(initial)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    optimizer.init(model.trainable_parameters())
    value_and_grad = nn.value_and_grad(model, loss_fn)
    (_loss, _components), gradients = value_and_grad(*batch)
    clipped, _gradient_norm = _canonical_v2_clip_gradients(gradients, max_norm=1.0)
    optimizer.update(model, clipped)
    mx.eval(model.parameters(), optimizer.state)
    final, _ = loss_fn(*batch)
    mx.eval(final)
    return float(initial.item()), float(final.item())


def test_v2_batch_keeps_teacher_policy_and_value_targets_independent(
    tmp_path: Path,
) -> None:
    game, targets = _loaded_fixture(tmp_path)
    batch = _make_canonical_v2_batch([game.samples[0]], targets)
    board = Board()
    two_six = move_label(Move.from_usi("2g2f"), board.turn)
    seven_six = move_label(Move.from_usi("7g7f"), board.turn)
    teacher_policies = np.asarray(batch[1])
    teacher_values = np.asarray(batch[2])
    play_inputs = np.asarray(batch[4])
    play_policy = np.asarray(batch[6])

    assert teacher_policies.shape[1] == 2
    assert teacher_policies[0, 0, seven_six] == pytest.approx(
        targets[0].scorers[0].policy["7g7f"]
    )
    assert teacher_policies[0, 1, two_six] == pytest.approx(
        targets[0].scorers[1].policy["2g2f"]
    )
    np.testing.assert_array_equal(
        teacher_values,
        np.asarray(
            [[targets[0].scorers[0].value, targets[0].scorers[1].value]],
            dtype=np.float32,
        ),
    )
    assert teacher_values[0, 0] < 0.0 < teacher_values[0, 1]
    assert play_inputs.shape[0] == 1
    assert play_policy[0, two_six] == 1.0
    assert play_policy[0, seven_six] == 0.0


def test_v2_training_updates_distinct_teacher_heads_without_scalar_broadcast(
    tmp_path: Path,
) -> None:
    game, targets = _loaded_fixture(tmp_path)
    model = upgrade_model_to_canonical_v2(PolicyValueResNet(model_profile("smoke")))
    batch = _make_canonical_v2_batch([game.samples[0]], targets)
    before = model.forward_canonical(batch[0])
    mx.eval(before.value_teachers)
    np.testing.assert_array_equal(
        np.asarray(before.value_teachers[:, 0]),
        np.asarray(before.value_teachers[:, 1]),
    )

    initial_loss, final_loss = _canonical_v2_test_step(
        model,
        [game.samples[0]],
        list(targets),
    )
    after = model.forward_canonical(batch[0])
    mx.eval(after.value_teachers)

    assert float(after.value_teachers[0, 0].item()) < float(
        after.value_teachers[0, 1].item()
    )
    assert final_loss < initial_loss


def _unresolved_copy(
    target: CanonicalDistillationTargetV2,
) -> CanonicalDistillationTargetV2:
    scorers = (
        replace(
            target.scorers[0],
            best_move="2g2f",
            policy={"2g2f": 0.9, "7g7f": 0.1},
            value=0.8,
        ),
        replace(
            target.scorers[1],
            best_move="7g7f",
            policy={"2g2f": 0.1, "7g7f": 0.9},
            value=-0.8,
        ),
    )
    return replace(
        target,
        scorers=scorers,
        play=GuardedPlayTarget(
            kind=PlayTargetKind.UNRESOLVED,
            train_play=False,
            additional_search_required=True,
            policy={},
            robust_best_moves=(),
            value_interval=None,
            wdl=None,
        ),
        uncertainty_target=1.0,
    )


def test_unresolved_rows_train_only_detached_teacher_and_uncertainty_heads(
    tmp_path: Path,
) -> None:
    game, targets = _loaded_fixture(tmp_path)
    resolved = targets[0]
    unresolved = _unresolved_copy(resolved)
    batch = _make_canonical_v2_batch(
        [game.samples[0], game.samples[0]],
        [resolved, unresolved],
    )

    assert np.asarray(batch[0]).shape[0] == 2
    assert np.asarray(batch[4]).shape[0] == 1
    assert np.asarray(batch[6]).shape[0] == 1
    np.testing.assert_array_equal(
        np.asarray(batch[2])[1], np.asarray([0.8, -0.8], dtype=np.float32)
    )

    mx.random.seed(20260809)
    resolved_model = upgrade_model_to_canonical_v2(
        PolicyValueResNet(model_profile("smoke"))
    )
    mx.random.seed(20260809)
    mixed_model = upgrade_model_to_canonical_v2(PolicyValueResNet(model_profile("smoke")))
    mixed_model.update(resolved_model.parameters())
    mx.eval(resolved_model.parameters(), mixed_model.parameters())
    for name, resolved_parameter in tree_flatten(resolved_model.parameters()):
        np.testing.assert_array_equal(
            np.asarray(resolved_parameter),
            np.asarray(dict(tree_flatten(mixed_model.parameters()))[name]),
        )

    _canonical_v2_test_step(resolved_model, [game.samples[0]], [resolved])
    _canonical_v2_test_step(
        mixed_model,
        [game.samples[0], game.samples[0]],
        [resolved, unresolved],
    )
    mx.eval(resolved_model.parameters(), mixed_model.parameters())

    resolved_parameters = dict(tree_flatten(resolved_model.parameters()))
    mixed_parameters = dict(tree_flatten(mixed_model.parameters()))
    auxiliary_prefixes = (
        "teacher_policies.",
        "teacher_policy_biases.",
        "teacher_value_outputs.",
        "uncertainty_output.",
    )
    auxiliary_differences = 0
    for name, resolved_parameter in resolved_parameters.items():
        mixed_parameter = mixed_parameters[name]
        if name.startswith(auxiliary_prefixes):
            auxiliary_differences += int(
                not np.allclose(
                    np.asarray(resolved_parameter),
                    np.asarray(mixed_parameter),
                    rtol=0.0,
                    atol=1e-8,
                )
            )
        else:
            np.testing.assert_array_equal(
                np.asarray(resolved_parameter),
                np.asarray(mixed_parameter),
                err_msg=f"unresolved row changed play/shared parameter {name}",
            )
    assert auxiliary_differences > 0


def test_canonical_sampling_includes_resolved_and_unresolved_without_duplicates(
    tmp_path: Path,
) -> None:
    _game, targets = _loaded_fixture(tmp_path)
    resolved = targets[0]
    unresolved = _unresolved_copy(resolved)

    mixed_rng = np.random.default_rng(7)
    resolved_rng = np.random.default_rng(7)
    indices = _canonical_v2_sample_indices(
        mixed_rng,
        [resolved, unresolved],
        batch_size=1,
        auxiliary_seed=11,
    )
    resolved_only = _canonical_v2_sample_indices(
        resolved_rng,
        [resolved],
        batch_size=1,
        auxiliary_seed=11,
    )

    assert sorted(indices.tolist()) == [0, 1]
    assert resolved_only.tolist() == [0]
    assert mixed_rng.integers(0, 2**31) == resolved_rng.integers(0, 2**31)


def test_public_trainer_apis_cannot_start_unreceipted_v2_optimizer_training(
    tmp_path: Path,
) -> None:
    game, targets = _loaded_fixture(tmp_path)
    model = upgrade_model_to_canonical_v2(PolicyValueResNet(model_profile("smoke")))
    before = {
        name: np.asarray(parameter).copy()
        for name, parameter in tree_flatten(model.parameters())
    }

    with pytest.raises(RuntimeError, match="optimizer runs are disabled"):
        train_resumable(
            model,
            [game.samples[0]],
            steps=1,
            batch_size=1,
            learning_rate=1e-3,
            dataset_fingerprint="a" * 64,
            canonical_teacher_only=True,
            canonical_targets_v2=targets,
            maximum_probe_loss_ratio=100.0,
        )
    with pytest.raises(RuntimeError, match="optimizer runs are disabled"):
        train(
            model,
            [game.samples[0]],
            steps=1,
            batch_size=1,
            learning_rate=1e-3,
            canonical_teacher_only=True,
            canonical_targets_v2=targets,
            maximum_probe_loss_ratio=100.0,
        )
    for name, parameter in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(np.asarray(parameter), before[name])


def test_v2_checkpoint_cannot_fall_back_to_legacy_actor_outcome_training(
    tmp_path: Path,
) -> None:
    game, _targets = _loaded_fixture(tmp_path)
    model = upgrade_model_to_canonical_v2(PolicyValueResNet(model_profile("smoke")))

    with pytest.raises(ValueError, match="cannot enter the legacy actor/outcome"):
        train_resumable(
            model,
            [game.samples[0]],
            steps=1,
            batch_size=1,
            learning_rate=1e-3,
            dataset_fingerprint="a" * 64,
            canonical_teacher_only=False,
            maximum_probe_loss_ratio=100.0,
        )
