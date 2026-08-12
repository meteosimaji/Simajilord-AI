from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from rsshogi.core import Board

from simajilord_shogi.distillation_targets_v2 import (
    CANONICAL_TARGET_V2_SCHEMA,
    CanonicalDisagreementReason,
    MateProofState,
    PlayTargetKind,
    _v2_proven_mate_policy,
    load_canonical_target_sidecar_v2,
    resolve_canonical_target_sidecar_v2,
)
from simajilord_shogi.encoding import combined_features_with_history, history_input_from_board

from .canonical_v2_fixtures import (
    build_canonical_v2_payload,
    calibrated_cp_and_q,
    write_canonical_v2_replay,
    write_canonical_v2_sidecar,
)
from .test_distillation_targets import _payload as build_v1_payload


def test_v2_binds_runtime_rights_history_and_independent_teacher_targets(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = replay.with_suffix(replay.suffix + ".distillation-targets-v2.json")
    write_canonical_v2_sidecar(sidecar, build_canonical_v2_payload(replay, game))

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    assert loaded.schema == CANONICAL_TARGET_V2_SCHEMA
    assert loaded.play_eligible_positions == 1
    assert loaded.unresolved_positions == 0
    assert loaded.rights_restriction_summary["publication_allowed"] is False
    assert loaded.positions[0].play.kind is PlayTargetKind.UNANIMOUS_CONSENSUS
    assert set(loaded.positions[0].play.policy) == {"2g2f", "7g7f"}
    assert loaded.positions[0].play.policy["2g2f"] > loaded.positions[0].play.policy["7g7f"]
    assert math.fsum(loaded.positions[0].play.policy.values()) == pytest.approx(1.0)
    assert loaded.positions[0].play.robust_best_moves == ("2g2f",)
    assert len(loaded.positions[0].reply_proposals) == 2
    assert len(loaded.positions[0].reply_proposals[0].candidates[0].proposals) == 3
    assert (
        loaded.positions[0].reply_proposals[0].candidates[0].reply_moves
        == tuple(
            reply.move
            for reply in loaded.positions[0]
            .score_matrix[0]
            .budgets[0]
            .candidates[0]
            .principal_replies
        )
    )
    assert loaded.positions[0].scorers[0].value > 0.0
    assert loaded.positions[0].scorers[0].value < loaded.positions[0].scorers[2].value
    assert loaded.positions[0].scorers[2].value < loaded.positions[0].scorers[1].value
    assert loaded.positions[0].unanimous_bootstrap.best_move == "2g2f"
    assert loaded.positions[0].unanimous_bootstrap.stable_across_budgets is True
    reply = loaded.positions[0].score_matrix[0].budgets[0].candidates[0].principal_replies[0]
    assert reply.root_player_sign == -1
    assert reply.score_cp is not None
    assert reply.reported_nodes == 100
    assert reply.pv == (reply.move,)
    assert set(loaded.positions[0].disagreement_reasons) == {
        CanonicalDisagreementReason.TEACHER_POLICY_DISTRIBUTION,
    }
    assert resolve_canonical_target_sidecar_v2(replay) == sidecar.resolve()


def test_v2_empty_or_incomplete_consensus_stays_out_of_training(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    write_canonical_v2_sidecar(
        sidecar,
        build_canonical_v2_payload(replay, game, coverage_complete=False),
    )

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    target = loaded.positions[0]
    assert target.play.kind is PlayTargetKind.UNRESOLVED
    assert target.play.policy == {}
    assert target.play.train_play is False
    assert target.play.additional_search_required is True
    assert target.uncertainty_target == 1.0


def test_v2_round_zero_rejects_cross_teacher_top1_disagreement(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    write_canonical_v2_sidecar(
        sidecar,
        build_canonical_v2_payload(replay, game, unanimous=False),
    )

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
    target = loaded.positions[0]

    assert target.play.kind is PlayTargetKind.UNRESOLVED
    assert target.play.train_play is False
    assert target.play.additional_search_required is True
    assert target.unanimous_bootstrap.best_move is None
    assert target.unanimous_bootstrap.unanimous_across_teachers is False
    assert target.unanimous_bootstrap.value_sign_compatible is False
    assert CanonicalDisagreementReason.TEACHER_TOP1_NOT_UNANIMOUS in (
        target.disagreement_reasons
    )
    assert CanonicalDisagreementReason.VALUE_SIGN_CONFLICT in target.disagreement_reasons


def test_v2_arbitrary_opening_root_matches_inference_history_completeness(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    start_fields = Board().to_sfen().split()
    start_fields[-1] = "9"
    game = write_canonical_v2_replay(
        replay,
        initial_sfen=" ".join(start_fields),
    )
    sidecar = tmp_path / "targets.json"
    write_canonical_v2_sidecar(sidecar, build_canonical_v2_payload(replay, game))

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
    training_history = loaded.positions[0].history
    inference_history = history_input_from_board(Board(game.samples[0].sfen))

    assert training_history.complete is False
    assert inference_history.complete is False
    assert training_history.moves == inference_history.moves == ()
    np.testing.assert_array_equal(
        combined_features_with_history(training_history),
        combined_features_with_history(inference_history),
    )


def test_v2_rejects_missing_family_reply_and_mismatched_search_settings(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"

    payload = build_canonical_v2_payload(replay, game)
    payload["required_candidate_families"] = ["nagisa", "suisho"]
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match=r"meteo.*tactical"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    descriptors = payload["canonical_scorers"]
    assert isinstance(descriptors, list)
    assert isinstance(descriptors[1], dict)
    descriptors[1]["threads"] = 2
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="same threads, hash, and MultiPV"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list)
    assert isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list)
    assert isinstance(matrices[0], dict)
    budgets = matrices[0]["budgets"]
    assert isinstance(budgets, list)
    assert isinstance(budgets[0], dict)
    candidates = budgets[0]["candidates"]
    assert isinstance(candidates, list)
    assert isinstance(candidates[0], dict)
    candidates[0]["principal_replies"] = []
    candidates[0]["pv"] = [candidates[0]["move"]]
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="cross-teacher proposal union"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list) and isinstance(matrices[0], dict)
    budgets = matrices[0]["budgets"]
    assert isinstance(budgets, list) and isinstance(budgets[0], dict)
    candidates = budgets[0]["candidates"]
    assert isinstance(candidates, list) and isinstance(candidates[0], dict)
    replies = candidates[0]["principal_replies"]
    assert isinstance(replies, list) and len(replies) > 1
    removed = replies.pop()
    assert isinstance(removed, dict)
    budgets[0]["reported_nodes"] = int(budgets[0]["reported_nodes"]) - int(
        removed["reported_nodes"]
    )
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="cross-teacher proposal union"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list) and isinstance(matrices[0], dict)
    matrices[0]["q_value_perspective"] = "side_to_move"
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="root player's signed perspective"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])


def test_v2_recomputes_q_from_raw_score_and_teacher_phase_calibration(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list) and isinstance(matrices[0], dict)
    budgets = matrices[0]["budgets"]
    assert isinstance(budgets, list) and isinstance(budgets[0], dict)
    candidates = budgets[0]["candidates"]
    assert isinstance(candidates, list) and isinstance(candidates[0], dict)
    candidates[0]["q_value"] = 0.123
    write_canonical_v2_sidecar(sidecar, payload)

    with pytest.raises(ValueError, match="raw score and calibration"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list)
    for matrix in matrices:
        assert isinstance(matrix, dict)
        calibration = matrix["calibration"]
        assert isinstance(calibration, dict)
        calibration["independent_validation_passed"] = False
    positions[0]["play"] = {
        "kind": "unresolved",
        "train_play": False,
        "additional_search_required": True,
    }
    write_canonical_v2_sidecar(sidecar, payload)

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
    assert loaded.positions[0].play.kind is PlayTargetKind.UNRESOLVED
    assert (
        CanonicalDisagreementReason.CALIBRATION
        in loaded.positions[0].disagreement_reasons
    )


def test_v2_rejects_self_declared_unknown_calibration_phase(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list)
    for matrix in matrices:
        assert isinstance(matrix, dict)
        calibration = matrix["calibration"]
        assert isinstance(calibration, dict)
        calibration["phase"] = "self-declared-unregistered-phase"
    write_canonical_v2_sidecar(sidecar, payload)

    with pytest.raises(ValueError, match="phase does not match the replay position"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])


def test_v2_requires_per_family_candidate_and_provenance_receipts(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    proposals = positions[0]["candidate_proposals"]
    assert isinstance(proposals, list) and isinstance(proposals[0], dict)
    proposals[0].pop("provenance_sha256")
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="provenance_sha256"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    proposals = positions[0]["candidate_proposals"]
    assert isinstance(proposals, list) and all(isinstance(item, dict) for item in proposals)
    for proposal in proposals:
        assert isinstance(proposal, dict)
        proposal["moves"] = ["2g2f"]
    write_canonical_v2_sidecar(sidecar, payload)
    with pytest.raises(ValueError, match="exactly equal the union"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("value", 0.0, "teacher value does not match"),
        (
            "policy",
            {"2g2f": 0.9, "7g7f": 0.1},
            "teacher policy does not match",
        ),
    ],
)
def test_v2_rejects_teacher_labels_not_derived_from_the_score_matrix(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    scorers = positions[0]["scorers"]
    assert isinstance(scorers, list) and isinstance(scorers[0], dict)
    scorers[0][field] = replacement
    write_canonical_v2_sidecar(sidecar, payload)

    with pytest.raises(ValueError, match=message):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])


def test_unstable_global_best_is_not_replaced_by_a_stable_inferior_move(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list) and isinstance(matrices[0], dict)
    budgets = matrices[0]["budgets"]
    assert isinstance(budgets, list) and isinstance(budgets[0], dict)
    candidates = budgets[0]["candidates"]
    assert isinstance(candidates, list) and isinstance(candidates[0], dict)
    score_cp, q_value = calibrated_cp_and_q(0.2, root_player_sign=1)
    candidates[0]["score_cp"] = score_cp
    candidates[0]["q_value"] = q_value
    replies = candidates[0]["principal_replies"]
    assert isinstance(replies, list) and all(isinstance(reply, dict) for reply in replies)
    reply_cp, reply_q = calibrated_cp_and_q(0.15, root_player_sign=-1)
    for reply in replies:
        assert isinstance(reply, dict)
        reply["score_cp"] = reply_cp
        reply["q_value"] = reply_q
    positions[0]["play"] = {
        "kind": "unresolved",
        "train_play": False,
        "additional_search_required": True,
    }
    write_canonical_v2_sidecar(sidecar, payload)

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
    target = loaded.positions[0]

    assert target.play.kind is PlayTargetKind.UNRESOLVED
    assert target.play.robust_best_moves == ()
    assert (
        CanonicalDisagreementReason.DEPTH_INSTABILITY
        in target.disagreement_reasons
    )


def test_bad_opponent_replies_do_not_make_a_sound_worst_reply_unstable(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    positions = payload["positions"]
    assert isinstance(positions, list) and isinstance(positions[0], dict)
    matrices = positions[0]["score_matrix"]
    assert isinstance(matrices, list)
    for matrix in matrices:
        assert isinstance(matrix, dict)
        budgets = matrix["budgets"]
        assert isinstance(budgets, list)
        for budget in budgets:
            assert isinstance(budget, dict)
            candidates = budget["candidates"]
            assert isinstance(candidates, list)
            for candidate in candidates:
                assert isinstance(candidate, dict)
                replies = candidate["principal_replies"]
                assert isinstance(replies, list) and len(replies) > 1
                last_reply = replies[-1]
                assert isinstance(last_reply, dict)
                score_cp, q_value = calibrated_cp_and_q(0.9, root_player_sign=-1)
                last_reply["score_cp"] = score_cp
                last_reply["q_value"] = q_value
    write_canonical_v2_sidecar(sidecar, payload)

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    assert loaded.positions[0].play.kind is PlayTargetKind.UNANIMOUS_CONSENSUS
    assert CanonicalDisagreementReason.REPLY_INSTABILITY not in (
        loaded.positions[0].disagreement_reasons
    )


def test_multiple_proven_mates_receive_equal_top_mass() -> None:
    assert _v2_proven_mate_policy(("5c5b+", "G*5b")) == {
        "5c5b+": 0.5,
        "G*5b": 0.5,
    }


def test_reported_mate_without_internal_proof_cannot_train_play_head(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    payload = build_canonical_v2_payload(replay, game)
    position = payload["positions"]
    assert isinstance(position, list) and isinstance(position[0], dict)
    matrix = position[0]["score_matrix"]
    assert isinstance(matrix, list) and isinstance(matrix[0], dict)
    budgets = matrix[0]["budgets"]
    assert isinstance(budgets, list) and isinstance(budgets[-1], dict)
    candidates = budgets[-1]["candidates"]
    assert isinstance(candidates, list) and isinstance(candidates[0], dict)
    candidates[0].update(
        {
            "q_value": 1.0,
            "score_kind": "mate",
            "score_cp": None,
            "mate_plies": 5,
            "mate_unknown_sign": None,
        }
    )
    scorers = position[0]["scorers"]
    assert isinstance(scorers, list) and isinstance(scorers[0], dict)
    mate_move = candidates[0]["move"]
    scorers[0].update(
        {
            "best_move": mate_move,
            "policy": {mate_move: 1.0},
            "value": 1.0,
        }
    )
    position[0]["play"] = {
        "kind": "unresolved",
        "train_play": False,
        "additional_search_required": True,
    }
    write_canonical_v2_sidecar(sidecar, payload)

    loaded = load_canonical_target_sidecar_v2(replay, sidecar, games=[game])

    assert loaded.positions[0].proof.state is MateProofState.NOT_APPLICABLE
    assert loaded.positions[0].play.kind is PlayTargetKind.UNRESOLVED


def test_v1_sidecar_is_rejected_for_the_structural_mean_bug(tmp_path: Path) -> None:
    replay = tmp_path / "actor.jsonl"
    game = write_canonical_v2_replay(replay)
    sidecar = tmp_path / "targets.json"
    sidecar.write_text(json.dumps(build_v1_payload(replay, game)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="arithmetic mean"):
        load_canonical_target_sidecar_v2(replay, sidecar, games=[game])
