from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from rsshogi.core import Board, Move
from rsshogi.policy import move_label

from simajilord_shogi.checkpoint import (
    load_checkpoint,
    load_checkpoint_with_training_state,
    save_checkpoint,
)
from simajilord_shogi.config import model_profile
from simajilord_shogi.distillation_targets import (
    CANONICAL_SCORER_IDS,
    CanonicalDistillationTarget,
    CanonicalScorerTarget,
    PolicyEquivalenceGroup,
    ProvenMateStatus,
    ProvenWinningMateSet,
)
from simajilord_shogi.domain import PositionSample
from simajilord_shogi.ensemble import normalized_sfen
from simajilord_shogi.human_gui import CheckpointIdentity, ComputeInterlock
from simajilord_shogi.model import PolicyValueResNet
from simajilord_shogi.trainer import (
    TrainingInterlockConfig,
    _best_union_top5_loss,
    _canonical_policy_loss,
    _combine_policy_value_loss,
    _equivalence_group_mass_loss,
    _make_batch,
    _make_canonical_batch,
    _proof_mate_set_mass_loss,
    _sample_sampling_weight,
    train_resumable,
)

from .test_mcts_game import MATE_IN_ONE_SFEN


def _canonical_target(
    *,
    proven_moves: tuple[str, ...] = (),
) -> CanonicalDistillationTarget:
    best_union = ("6c5b+", "G*5b")
    return CanonicalDistillationTarget(
        game_index=0,
        sample_index=0,
        normalized_sfen=normalized_sfen(MATE_IN_ONE_SFEN),
        history_context_sha256="a" * 64,
        scorers=(
            CanonicalScorerTarget(
                scorer_id=CANONICAL_SCORER_IDS[0],
                best_move="G*5b",
                policy={"G*5b": 1.0},
                value=-0.8,
            ),
            CanonicalScorerTarget(
                scorer_id=CANONICAL_SCORER_IDS[1],
                best_move="6c5b+",
                policy={"6c5b+": 1.0},
                value=-0.6,
            ),
            CanonicalScorerTarget(
                scorer_id=CANONICAL_SCORER_IDS[2],
                best_move="G*5b",
                policy={"G*5b": 1.0},
                value=-0.7,
            ),
        ),
        canonical_best_union=best_union,
        equivalence_groups=(
            PolicyEquivalenceGroup(
                group_id=0,
                kind="near_optimal",
                moves=best_union,
                target_mass=1.0,
                worst_teacher_regret=0.0,
                teacher_depth_dispersion=0.0,
                reply_dispersion=0.0,
            ),
        ),
        proven_winning_mate_set=ProvenWinningMateSet(
            status=(ProvenMateStatus.PROVEN if proven_moves else ProvenMateStatus.NONE),
            moves=proven_moves,
        ),
        canonical_values={
            CANONICAL_SCORER_IDS[0]: -0.8,
            CANONICAL_SCORER_IDS[1]: -0.6,
            CANONICAL_SCORER_IDS[2]: -0.7,
        },
        history_training_weight=1.0,
    )


def test_extremely_deep_teacher_is_introduced_by_curriculum_not_hard_replacement() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"6c7d": 1.0},
        root_value=-1.0,
        value_target=-1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
        teacher_depth_ratio=256.0,
    )

    _, policies, values = _make_batch(
        [sample],
        teacher_policy_mix=0.8,
        teacher_value_mix=0.5,
        legal_label_smoothing=0.0,
        curriculum_depth_ratio=64.0,
        minimum_teacher_policy_mix=0.1,
    )
    policy = np.asarray(policies)[0]

    actor_label = move_label(Move.from_usi("6c7d"), board.turn)
    teacher_label = move_label(Move.from_usi("G*5b"), board.turn)
    assert policy[actor_label] == pytest.approx(0.6)
    assert policy[teacher_label] == pytest.approx(0.4)
    # The same curriculum scale softens the value target: a 0.5 configured
    # teacher mix becomes 0.25 when the policy teacher mix is halved.
    assert float(np.asarray(values)[0, 0]) == pytest.approx(-0.5)


def test_legal_label_smoothing_never_assigns_probability_to_illegal_labels() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )

    _, policies, _ = _make_batch(
        [sample],
        teacher_policy_mix=0.75,
        teacher_value_mix=0.5,
        legal_label_smoothing=0.1,
        curriculum_depth_ratio=64.0,
        minimum_teacher_policy_mix=0.1,
    )
    nonzero = set(np.flatnonzero(np.asarray(policies)[0]))
    legal_labels = {move_label(move, board.turn) for move in board.legal_moves()}

    assert nonzero == legal_labels
    assert float(np.asarray(policies).sum()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"turn": 1}, "turn disagrees with SFEN"),
        ({"root_value": math.inf}, "actor root value"),
        ({"value_target": 1.01}, "game outcome value target"),
        ({"teacher_value": math.nan}, "teacher value target"),
        ({"policy": {"G*5b": -0.1}}, "finite and non-negative"),
        ({"policy": {"G*5b": math.nan}}, "finite and non-negative"),
    ],
)
def test_training_batch_rejects_malformed_perspective_value_and_policy_targets(
    change: dict[str, object], message: str
) -> None:
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=0.0,
        value_target=1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )

    with pytest.raises(ValueError, match=message):
        _make_batch(
            [replace(sample, **change)],  # type: ignore[arg-type]
            teacher_policy_mix=0.75,
            teacher_value_mix=0.0,
            legal_label_smoothing=0.0,
            curriculum_depth_ratio=64.0,
            minimum_teacher_policy_mix=0.1,
        )


def test_checkmate_horizon_prioritizes_last_actionable_mating_position() -> None:
    base = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
    )

    ordinary = _sample_sampling_weight(
        base,
        reversal_priority=4.0,
        blunder_priority=3.0,
        checkmate_sample_priority=2.0,
        checkmate_horizon_plies=16,
    )
    mating = _sample_sampling_weight(
        replace(base, terminal_checkmate_distance=1),
        reversal_priority=4.0,
        blunder_priority=3.0,
        checkmate_sample_priority=2.0,
        checkmate_horizon_plies=16,
    )
    outside_horizon = _sample_sampling_weight(
        replace(base, terminal_checkmate_distance=17),
        reversal_priority=4.0,
        blunder_priority=3.0,
        checkmate_sample_priority=2.0,
        checkmate_horizon_plies=16,
    )

    assert ordinary == 1.0
    assert mating == 3.0
    assert outside_horizon == 1.0


def test_canonical_batch_assigns_zero_actor_mass_and_zero_game_outcome_contribution() -> None:
    board = Board(MATE_IN_ONE_SFEN)
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"6c7d": 1.0},
        root_value=1.0,
        value_target=1.0,
        chosen_move="6c7d",
        actor_best_move="6c7d",
    )

    batch = _make_canonical_batch([sample], [_canonical_target()])
    scorer_values = np.asarray(batch[1])[0]
    group_mask = np.asarray(batch[3])[0, 0]

    actor_label = move_label(Move.from_usi("6c7d"), board.turn)
    nagisa_label = move_label(Move.from_usi("G*5b"), board.turn)
    suisho_label = move_label(Move.from_usi("6c5b+"), board.turn)
    assert not group_mask[actor_label]
    assert group_mask[nagisa_label]
    assert group_mask[suisho_label]
    assert not np.asarray(batch[5]).any()
    np.testing.assert_array_equal(
        scorer_values,
        np.asarray([-0.8, -0.6, -0.7], dtype=np.float32),
    )
    assert 1.0 not in scorer_values


def test_canonical_batch_padding_and_inactive_proof_use_no_fake_move_fallback() -> None:
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"6c7d": 1.0},
        root_value=1.0,
        value_target=1.0,
    )
    target = _canonical_target()
    zero_mass_group = PolicyEquivalenceGroup(
        group_id=1,
        kind="candidate",
        moves=("6c7d",),
        target_mass=0.0,
        worst_teacher_regret=0.0,
        teacher_depth_dispersion=0.0,
        reply_dispersion=0.0,
    )
    wider_target = replace(
        target,
        equivalence_groups=(*target.equivalence_groups, zero_mass_group),
    )

    batch = _make_canonical_batch([sample, sample], [target, wider_target])
    group_masks = np.asarray(batch[3])
    group_masses = np.asarray(batch[4])
    proof_masks = np.asarray(batch[5])

    assert not group_masks[0, 1].any()
    assert group_masses[0, 1] == 0.0
    assert group_masks[1, 1].any()
    assert group_masses[1, 1] == 0.0
    assert not proof_masks.any()


def test_proven_mate_set_loss_is_symmetric_inside_set_and_penalizes_external_mass() -> None:
    legal = mx.array([[True, True, True]])
    proof = mx.array([[True, True, False]])
    left = _proof_mate_set_mass_loss(mx.array([[9.0, 1.0, 0.0]]), legal, proof)
    right = _proof_mate_set_mass_loss(mx.array([[1.0, 9.0, 0.0]]), legal, proof)
    external = _proof_mate_set_mass_loss(mx.array([[1.0, 1.0, 9.0]]), legal, proof)
    mx.eval(left, right, external)

    assert float(left.item()) == pytest.approx(float(right.item()), abs=1e-7)
    assert float(external.item()) > float(left.item()) + 5.0


def test_canonical_policy_loss_gives_proven_mate_set_precedence_over_groups() -> None:
    logits = mx.array([[8.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
    legal = mx.array([[True, True, True, True, True, True]])
    groups = mx.array([[[True, False, False, False, False, False]]])
    masses = mx.array([[1.0]])
    proof = mx.array([[False, True, False, False, False, False]])
    union = mx.array([[True, True, False, False, False, False]])
    actual = _canonical_policy_loss(
        logits,
        legal,
        groups,
        masses,
        proof,
        mx.array([True]),
        union,
        mx.array([2]),
        best_union_top5_weight=0.0,
    )
    expected = _proof_mate_set_mass_loss(logits, legal, proof)
    group_only = _equivalence_group_mass_loss(logits, legal, groups, masses)
    mx.eval(actual, expected, group_only)

    assert float(actual.item()) == pytest.approx(float(expected.item()), abs=1e-7)
    assert float(actual.item()) > float(group_only.item()) + 5.0


def test_proven_mate_precedence_disables_best_union_top5_auxiliary() -> None:
    logits = mx.array([[8.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
    legal = mx.array([[True, True, True, True, True, True]])
    groups = mx.array([[[True, False, False, False, False, False]]])
    masses = mx.array([[1.0]])
    proof = mx.array([[False, True, False, False, False, False]])
    union = mx.array([[True, False, True, False, False, False]])
    actual = _canonical_policy_loss(
        logits,
        legal,
        groups,
        masses,
        proof,
        mx.array([True]),
        union,
        mx.array([2]),
        best_union_top5_weight=100.0,
    )
    expected = _proof_mate_set_mass_loss(logits, legal, proof)
    mx.eval(actual, expected)

    assert float(actual.item()) == pytest.approx(float(expected.item()), abs=1e-7)


def test_equivalence_group_loss_uses_group_mass_not_within_group_ranking() -> None:
    legal = mx.array([[True, True, True]])
    groups = mx.array([[[True, True, False]]])
    masses = mx.array([[1.0]])
    left = _equivalence_group_mass_loss(
        mx.array([[8.0, 0.0, -1.0]]), legal, groups, masses
    )
    right = _equivalence_group_mass_loss(
        mx.array([[0.0, 8.0, -1.0]]), legal, groups, masses
    )
    outside = _equivalence_group_mass_loss(
        mx.array([[0.0, 0.0, 8.0]]), legal, groups, masses
    )
    padded = _equivalence_group_mass_loss(
        mx.array([[8.0, 0.0, -1.0]]),
        legal,
        mx.array([[[True, True, False], [False, False, False]]]),
        mx.array([[1.0, 0.0]]),
    )
    mx.eval(left, right, outside, padded)

    assert float(left.item()) == pytest.approx(float(right.item()), abs=1e-7)
    assert float(padded.item()) == pytest.approx(float(left.item()), abs=1e-7)
    assert float(outside.item()) > float(left.item()) + 5.0


def test_inactive_empty_proof_mask_leaves_group_loss_unchanged() -> None:
    logits = mx.array([[8.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
    legal = mx.array([[True, True, True, True, True, True]])
    groups = mx.array([[[True, False, False, False, False, False]]])
    masses = mx.array([[1.0]])
    union = mx.array([[True, False, False, False, False, False]])
    actual = _canonical_policy_loss(
        logits,
        legal,
        groups,
        masses,
        mx.array([[False, False, False, False, False, False]]),
        mx.array([False]),
        union,
        mx.array([1]),
        best_union_top5_weight=0.0,
    )
    expected = _equivalence_group_mass_loss(logits, legal, groups, masses)
    mx.eval(actual, expected)

    assert float(actual.item()) == pytest.approx(float(expected.item()), abs=1e-7)


def test_best_union_top5_loss_is_symmetric_legal_only_and_disables_same_best() -> None:
    legal = mx.array([[True, True, True, True, True, True, True, False]])
    union = mx.array([[True, True, False, False, False, False, False, False]])
    union_sizes = mx.array([2])
    good = _best_union_top5_loss(
        mx.array([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 100.0]]),
        legal,
        union,
        union_sizes,
    )
    symmetric = _best_union_top5_loss(
        mx.array([[9.0, 10.0, 8.0, 7.0, 6.0, 5.0, 4.0, 100.0]]),
        legal,
        union,
        union_sizes,
    )
    sixth = _best_union_top5_loss(
        mx.array([[10.0, 4.0, 8.0, 7.0, 6.0, 5.0, 3.0, 100.0]]),
        legal,
        union,
        union_sizes,
    )
    same_best = _best_union_top5_loss(
        mx.array([[1.0, 0.0, 8.0, 7.0, 6.0, 5.0, 4.0, 100.0]]),
        legal,
        mx.array([[True, False, False, False, False, False, False, False]]),
        mx.array([1]),
    )
    mx.eval(good, symmetric, sixth, same_best)

    assert float(good.item()) == pytest.approx(float(symmetric.item()), abs=1e-7)
    assert float(good.item()) < 0.03
    assert float(sixth.item()) > 1.0
    assert float(same_best.item()) == 0.0


def test_value_loss_weight_is_neutral_at_one_and_scales_only_value_component() -> None:
    policy = mx.array(2.0)
    value = mx.array(3.0)
    neutral = _combine_policy_value_loss(policy, value, value_loss_weight=1.0)
    weighted = _combine_policy_value_loss(policy, value, value_loss_weight=0.25)
    mx.eval(neutral, weighted)

    assert float(neutral.item()) == 5.0
    assert float(weighted.item()) == 2.75


def test_exact_resume_matches_one_uninterrupted_adamw_stream(tmp_path: Path) -> None:
    base = tmp_path / "base"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), base, step=7)
    continuous_model, _ = load_checkpoint(base)
    segmented_model, _ = load_checkpoint(base)
    samples = [
        PositionSample(
            sfen=MATE_IN_ONE_SFEN,
            ply=0,
            turn=0,
            policy={"6c7d": 1.0},
            root_value=-1.0,
            value_target=-1.0,
            teacher_policy={move: 1.0},
            teacher_value=1.0,
        )
        for move in ("G*5b", "6c5b+")
    ]
    dataset_fingerprint = hashlib.sha256(b"two-history-aware-targets").hexdigest()
    options = {
        "batch_size": 1,
        "learning_rate": 1e-4,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": 19,
        "maximum_probe_loss_ratio": 100.0,
        "telemetry_interval": 1,
    }

    continuous = train_resumable(
        continuous_model,
        samples,
        steps=4,
        initial_model_step=7,
        **options,
    )
    first = train_resumable(
        segmented_model,
        samples,
        steps=2,
        initial_model_step=7,
        **options,
    )
    midpoint = tmp_path / "midpoint"
    save_checkpoint(
        segmented_model,
        midpoint,
        step=first.state.model_step,
        training_state=first.state,
        training_trace=first.trace,
    )
    resumed_model, resumed_step, resumed_state, loaded_trace = (
        load_checkpoint_with_training_state(midpoint)
    )
    assert resumed_state is not None
    assert loaded_trace == first.trace
    second = train_resumable(
        resumed_model,
        samples,
        steps=2,
        initial_model_step=resumed_step,
        resume_state=resumed_state,
        **options,
    )

    continuous_parameters = dict(tree_flatten(continuous_model.parameters()))
    resumed_parameters = dict(tree_flatten(resumed_model.parameters()))
    assert continuous_parameters.keys() == resumed_parameters.keys()
    for name in continuous_parameters:
        np.testing.assert_array_equal(
            np.asarray(continuous_parameters[name]),
            np.asarray(resumed_parameters[name]),
        )
    continuous_optimizer = dict(tree_flatten(continuous.state.optimizer_state))
    resumed_optimizer = dict(tree_flatten(second.state.optimizer_state))
    assert continuous_optimizer.keys() == resumed_optimizer.keys()
    for name in continuous_optimizer:
        np.testing.assert_array_equal(
            np.asarray(continuous_optimizer[name]),
            np.asarray(resumed_optimizer[name]),
        )
    assert second.state.model_step == continuous.state.model_step == 11
    assert second.state.optimizer_step == continuous.state.optimizer_step == 4
    assert [point.sample_indices_sha256 for point in continuous.trace] == [
        *(point.sample_indices_sha256 for point in first.trace),
        *(point.sample_indices_sha256 for point in second.trace),
    ]


def test_compute_interlock_pause_preserves_exact_training_stream(tmp_path: Path) -> None:
    base = tmp_path / "base-interlock"
    save_checkpoint(PolicyValueResNet(model_profile("smoke")), base, step=3)
    uninterrupted_model, _ = load_checkpoint(base)
    interlocked_model, _ = load_checkpoint(base)
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"6c7d": 1.0},
        root_value=-1.0,
        value_target=-1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )
    dataset_fingerprint = hashlib.sha256(b"interlock-determinism").hexdigest()
    options = {
        "steps": 2,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": 23,
        "initial_model_step": 3,
        "maximum_probe_loss_ratio": 100.0,
    }

    uninterrupted = train_resumable(uninterrupted_model, [sample], **options)
    state_root = tmp_path / "human-state"
    interlocked = train_resumable(
        interlocked_model,
        [sample],
        training_interlock=TrainingInterlockConfig(
            state_root=state_root,
            generation=7,
            ttl_seconds=3,
            heartbeat_interval_seconds=1,
            wait_timeout_seconds=2,
            poll_interval_seconds=0.01,
        ),
        **options,
    )

    for name, parameter in tree_flatten(uninterrupted_model.parameters()):
        np.testing.assert_array_equal(
            np.asarray(parameter),
            np.asarray(dict(tree_flatten(interlocked_model.parameters()))[name]),
        )
    for name, parameter in tree_flatten(uninterrupted.state.optimizer_state):
        np.testing.assert_array_equal(
            np.asarray(parameter),
            np.asarray(dict(tree_flatten(interlocked.state.optimizer_state))[name]),
        )
    assert uninterrupted.state.numpy_rng_state == interlocked.state.numpy_rng_state
    assert uninterrupted.trace == interlocked.trace
    assert interlocked.metrics.compute_interlock_enabled is True
    assert interlocked.metrics.compute_interlock_acquisitions == options["steps"] + 3
    assert interlocked.metrics.compute_interlock_wait_seconds >= 0
    assert not any((state_root / "training-step-leases").iterdir())


def test_training_waits_for_human_before_advancing_optimizer(tmp_path: Path) -> None:
    state_root = tmp_path / "human-priority-state"
    interlock = ComputeInterlock(state_root)
    identity = CheckpointIdentity(
        generation=1,
        relative_path="generation-1",
        checkpoint_sha256="a" * 64,
        metadata_sha256="b" * 64,
        weights_sha256="c" * 64,
        step=0,
        published_unix_seconds=time.time(),
    )
    human = interlock.acquire_human(
        "d" * 32,
        identity,
        ttl_seconds=3,
        heartbeat_interval_seconds=1,
        wait_timeout_seconds=0,
    )
    released = threading.Event()

    def release_human() -> None:
        time.sleep(0.08)
        human.release()
        released.set()

    release_thread = threading.Thread(target=release_human)
    release_thread.start()
    model = PolicyValueResNet(model_profile("smoke"))
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )
    run = train_resumable(
        model,
        [sample],
        steps=1,
        batch_size=1,
        learning_rate=1e-4,
        dataset_fingerprint="d" * 64,
        maximum_probe_loss_ratio=100.0,
        training_interlock=TrainingInterlockConfig(
            state_root=state_root,
            ttl_seconds=3,
            heartbeat_interval_seconds=1,
            wait_timeout_seconds=2,
            poll_interval_seconds=0.01,
        ),
    )
    release_thread.join(timeout=2)

    assert released.is_set()
    assert run.state.optimizer_step == 1
    assert run.metrics.compute_interlock_wait_seconds >= 0.05
    assert interlock.snapshot().human_active_count == 0
    assert interlock.snapshot().training_step_active_count == 0


def test_exact_resume_rejects_changed_dataset_identity() -> None:
    model = PolicyValueResNet(model_profile("smoke"))
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"G*5b": 1.0},
        root_value=1.0,
        value_target=1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )
    first = train_resumable(
        model,
        [sample],
        steps=1,
        batch_size=1,
        learning_rate=1e-4,
        dataset_fingerprint="a" * 64,
        maximum_probe_loss_ratio=100.0,
    )

    with pytest.raises(ValueError, match="dataset fingerprint changed"):
        train_resumable(
            model,
            [sample],
            steps=1,
            batch_size=1,
            learning_rate=1e-4,
            dataset_fingerprint="b" * 64,
            resume_state=first.state,
            maximum_probe_loss_ratio=100.0,
        )


def test_v1_optimizer_state_cannot_silently_resume_canonical_v2_loss() -> None:
    model = PolicyValueResNet(model_profile("smoke"))
    sample = PositionSample(
        sfen=MATE_IN_ONE_SFEN,
        ply=0,
        turn=0,
        policy={"6c7d": 1.0},
        root_value=1.0,
        value_target=1.0,
        teacher_policy={"G*5b": 1.0},
        teacher_value=1.0,
    )
    first = train_resumable(
        model,
        [sample],
        steps=1,
        batch_size=1,
        learning_rate=1e-4,
        dataset_fingerprint="c" * 64,
        maximum_probe_loss_ratio=100.0,
    )

    with pytest.raises(ValueError, match="arithmetic mean"):
        train_resumable(
            model,
            [sample],
            steps=1,
            batch_size=1,
            learning_rate=1e-4,
            dataset_fingerprint="c" * 64,
            canonical_teacher_only=True,
            canonical_targets=[_canonical_target()],
            resume_state=first.state,
            maximum_probe_loss_ratio=100.0,
        )
