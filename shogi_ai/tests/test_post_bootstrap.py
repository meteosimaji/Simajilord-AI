from __future__ import annotations

import hashlib
import json

import pytest

from simajilord_shogi.cli import main
from simajilord_shogi.distillation_targets import CANONICAL_SCORER_IDS
from simajilord_shogi.post_bootstrap import (
    HISUI_SEALED_TARGET_ID,
    NAGISA_NEXT_SEALED_TARGET_ID,
    POST_BOOTSTRAP_ROADMAP_SCHEMA,
    REGISTERED_MODEL_USAGE_GATE_SCHEMA,
    MatchupAllocation,
    ReplayAllocation,
    post_bootstrap_roadmap_contract,
    validate_registered_model_usage,
)


def test_post_bootstrap_contract_uses_all_three_teachers_without_value_voting() -> None:
    contract = post_bootstrap_roadmap_contract()

    assert contract["schema"] == POST_BOOTSTRAP_ROADMAP_SCHEMA
    inventory = contract["inventory"]
    assert inventory["registered_local_model_count"] == 9
    assert tuple(inventory["canonical_teachers"]) == CANONICAL_SCORER_IDS
    assert len(inventory["secondary_registered_opponents"]) == 6
    assert "takewarabe" not in inventory["registered_local_models"]

    teaching = contract["teacher_reanalysis"]
    assert teaching["candidate_proposers"] == [*CANONICAL_SCORER_IDS, "meteo_current"]
    assert teaching["single_scorer_per_experiment_arm"] is True
    assert teaching["cross_teacher_raw_score_average"] is False
    assert teaching["majority_vote"] is False
    assert teaching["nagisa_and_soujou_are_independent_votes"] is False
    assert teaching["actual_played_move_is_best_move_label"] is False
    assert teaching["teacher_played_move_is_best_move_label"] is False
    assert teaching["both_sides_moves_are_reanalysed"] is True
    secondary_use = teaching["secondary_registered_model_use"]
    assert len(secondary_use["model_ids"]) == 6
    assert secondary_use["candidate_role"] == "proposal_only_optional_ablation"
    assert secondary_use["final_value_source"] == "selected_single_canonical_anchor_only"


def test_game_and_replay_allocations_are_complete_bounded_and_sum_to_one() -> None:
    contract = post_bootstrap_roadmap_contract()
    generation = contract["game_generation"]
    matchups = generation["matchups"]

    assert sum(item["fraction"] for item in matchups) == pytest.approx(1.0)
    assert generation["initial_games_per_generation"] == 2_400
    assert sum(item["games_per_generation"] for item in matchups) == 2_400
    assert all(item["games_per_generation"] % 2 == 0 for item in matchups)
    teacher_games = next(
        item for item in matchups if item["matchup_id"] == "meteo_vs_three_teachers"
    )
    assert tuple(teacher_games["opponents"]) == CANONICAL_SCORER_IDS
    assert generation["meteo_teacher_share_per_teacher"] == pytest.approx(0.10)
    assert len(generation["three_teacher_pairs"]) == 3
    assert all(left != right for left, right in generation["three_teacher_pairs"])
    assert generation["teacher_crossplay_share_per_pair"] == pytest.approx(0.05)
    assert generation["all_registered_models_used_each_generation"] is True
    usage = generation["registered_model_usage_gate"]
    assert len(usage["required_ids"]) == 9
    assert usage["raw_secondary_scores_enter_anchor_targets"] is False
    assert usage["same_lineage_models_count_as_independent_votes"] is False
    assert usage["generation_complete_requires_each_id_in_game_receipts"] is True
    assert generation["latest_previous_weight_artifacts_only"] is True
    performance = generation["performance_measurement"]
    assert performance["strength_claim_before_nnue_runtime_adapter"] is False
    assert performance["production_game_search"] == "multipv_1"
    assert performance["deep_three_teacher_reanalysis"] == "selected_positions_only"
    assert "completed_games_per_hour" in performance["required_metrics"]
    reference = performance["historical_teacher_reference_only"]
    assert reference["is_meteo_measurement"] is False
    assert reference["search_nodes_per_move"] == 20_000
    assert reference["serial_search_time_ceiling_games_per_hour"] == [1648.2, 1737.5]
    assert reference["must_not_be_used_as_meteo_production_eta"] is True

    retention = contract["artifact_retention"]
    assert retention["full_training_checkpoint_limit"] == 2
    assert retention["playable_weight_window_limit"] == 2
    assert retention["compact_champion_registry_limit"] == 1
    assert retention["compact_champion_must_reference_playable_weight_window"] is True
    assert retention["third_independent_full_checkpoint_allowed"] is False

    rules = generation["rules"]
    assert rules["resign"] is False
    assert rules["evaluation_adjudication"] is False
    assert rules["truncated_games_enter_strong_terminal_labels"] is False
    assert "checkmate" in rules["continue_to"]
    assert "legal_repetition_result" in rules["continue_to"]

    rl = contract["reinforcement_learning"]
    assert rl["direct_policy_gradient"] is False
    assert rl["student_policy_head"] is False
    assert sum(item["fraction"] for item in rl["replay_mix"]) == pytest.approx(1.0)
    broad = next(
        item for item in rl["replay_mix"] if item["source_id"] == "broad_100b_bootstrap_anchor"
    )
    assert broad["fraction"] == pytest.approx(0.50)


def test_registered_model_usage_gate_requires_actual_games_from_all_nine_models() -> None:
    contract = post_bootstrap_roadmap_contract()
    canonical = set(contract["inventory"]["canonical_teachers"])
    receipts = []
    for rights_id in contract["inventory"]["registered_local_models"]:
        receipt = {
            "schema": "meteo-nnue-game-bundle-receipt-v1",
            "local_only": True,
            "publication_allowed": False,
            "engine_inputs": [
                {"kind": "rights_profile", "rights_id": rights_id},
                {"kind": "meteo_runtime_contract", "rights_id": None},
            ],
            "generation": {"metrics": {"games": 240 if rights_id in canonical else 20}},
        }
        canonical_receipt = json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        receipt["payload_sha256"] = hashlib.sha256(canonical_receipt).hexdigest()
        receipt["payload_sha256_scope"] = "receipt_without_payload_sha256_fields"
        receipts.append(receipt)

    gate = validate_registered_model_usage(receipts)
    assert gate.complete
    assert gate.missing_games == ()
    assert gate.to_dict()["schema"] == REGISTERED_MODEL_USAGE_GATE_SCHEMA
    assert len(gate.to_dict()["observed_games"]) == 9

    with pytest.raises(ValueError, match="missing registered-model games"):
        validate_registered_model_usage(receipts[:-1])

    tampered = json.loads(json.dumps(receipts))
    tampered[0]["generation"]["metrics"]["games"] += 1
    with pytest.raises(ValueError, match="payload SHA-256 does not verify"):
        validate_registered_model_usage(tampered)


def test_hisui_stays_sealed_and_cannot_leak_into_training() -> None:
    contract = post_bootstrap_roadmap_contract()
    hisui = contract["inventory"]["hisui"]

    assert HISUI_SEALED_TARGET_ID in contract["objective"]["targets"]
    assert hisui["role"] == "sealed_external_completed_engine_target"
    assert hisui["local_artifact_available"] is False
    assert hisui["training_use_allowed"] is False
    assert hisui["automatic_scraping_or_unapproved_bulk_play"] is False


def test_unpublished_nagisa_v4_is_a_sealed_target_not_an_inferred_teacher() -> None:
    contract = post_bootstrap_roadmap_contract()
    target = contract["inventory"]["nagisa_v4_unpublished"]
    reserve = contract["promotion"]["unpublished_nagisa_v4_reserve"]

    assert NAGISA_NEXT_SEALED_TARGET_ID in contract["objective"]["targets"]
    assert target["role"] == "sealed_unpublished_future_engine_target"
    assert target["public_release_or_reproducible_benchmark_available"] is False
    assert target["local_artifact_available"] is False
    assert target["training_use_allowed"] is False
    assert target["v3_1_is_a_proxy_not_proof_of_v4_superiority"] is True
    assert reserve["v3_1_proxy_lower_95_score_target"] == pytest.approx(0.55)
    assert reserve["proxy_is_not_a_v4_strength_claim"] is True
    assert reserve["unknown_version_identity_must_fail_closed"] is True


def test_promotion_separates_incremental_generation_from_all_target_claim() -> None:
    promotion = post_bootstrap_roadmap_contract()["promotion"]

    assert "candidate_beats_previous_champion_on_paired_openings" in promotion["generation_gate"]
    assert "no_local_teacher_regression_each_teacher_separately" in promotion["generation_gate"]
    assert promotion["test_sequence"] == ["proxy", "stc_sprt", "ltc_sprt", "long_analysis"]
    assert promotion["minimum_formal_opening_pairs_per_opponent"] == 500
    assert promotion["both_colors_required"] is True
    assert promotion["do_not_stop_sprt_before_terminal_decision"] is True
    assert "lower_95_score_above_0_5_vs_each_local_teacher" in promotion["target_superiority_claim"]


def test_qat_changes_are_next_run_ablations_not_live_mutations() -> None:
    contract = post_bootstrap_roadmap_contract()
    quantisation = contract["quantisation_and_speed"]

    assert contract["entry_gate"]["current_run_is_immutable_while_active"] is True
    assert quantisation["native_export_must_match_qat_forward"] is True
    assert quantisation["float_master_is_not_a_deployable_model"] is True
    assert quantisation["unsafe_mid_run_numeric_change"] is False
    assert "representable_range_projection_after_optimizer" in quantisation["qat_ablation_arms"]
    assert "custom_fused_metal_sparse_gather_backward" in quantisation["next_run_speed_ablations"]


def test_roadmap_stage_and_allocation_identifiers_are_unique() -> None:
    contract = post_bootstrap_roadmap_contract()
    stage_ids = [stage["id"] for stage in contract["stages"]]
    matchup_ids = [matchup["matchup_id"] for matchup in contract["game_generation"]["matchups"]]
    replay_ids = [
        source["source_id"] for source in contract["reinforcement_learning"]["replay_mix"]
    ]

    assert len(stage_ids) == len(set(stage_ids))
    assert len(matchup_ids) == len(set(matchup_ids))
    assert len(replay_ids) == len(set(replay_ids))


def test_allocation_types_reject_invalid_or_shadowed_identities() -> None:
    with pytest.raises(ValueError, match="non-empty trimmed"):
        MatchupAllocation(" duplicated ", 0.5, ("a",), ("b",), "use", True)
    with pytest.raises(ValueError, match="subjects must be unique"):
        MatchupAllocation("duplicate-subject", 0.5, ("a", "a"), ("b",), "use", True)
    with pytest.raises(ValueError, match="non-empty trimmed strings"):
        MatchupAllocation("empty-subject", 0.5, (" ",), ("b",), "use", True)
    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        ReplayAllocation("bad", 0.0, "role")


def test_post_bootstrap_cli_prints_the_same_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["post-bootstrap-roadmap"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == post_bootstrap_roadmap_contract()
