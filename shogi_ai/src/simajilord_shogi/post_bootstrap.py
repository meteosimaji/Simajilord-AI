"""Authoritative roadmap after Meteo's 100B value-only NNUE bootstrap.

The bootstrap learns a broad scalar evaluation function from a published PSV
corpus.  It is deliberately followed by a different loop: complete games find
Meteo's current failures, the three reviewed teachers propose counterexamples,
one pinned scorer reanalyses every candidate on a common scale, and the scalar
targets are distilled back into the deployable NNUE.  This is fitted value /
expert iteration for an alpha-beta engine, not policy-gradient RL.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .distillation_targets import CANONICAL_SCORER_IDS
from .model_rights import MODEL_RIGHTS

POST_BOOTSTRAP_ROADMAP_SCHEMA = "meteo-post-100b-roadmap-v1"
HISUI_SEALED_TARGET_ID = "hisui-wcsc36-sealed-external"
NAGISA_NEXT_SEALED_TARGET_ID = "nagisa-v4-unpublished-sealed-external"
REGISTERED_MODEL_USAGE_GATE_SCHEMA = "meteo-registered-model-usage-gate-v1"
_GAME_BUNDLE_SCHEMA = "meteo-nnue-game-bundle-receipt-v1"


@dataclass(frozen=True, slots=True)
class MatchupAllocation:
    """Initial fraction of complete games assigned to one matchup family."""

    matchup_id: str
    fraction: float
    subjects: tuple[str, ...]
    opponents: tuple[str, ...]
    position_use: str
    distinct_subject_opponent_required: bool

    def __post_init__(self) -> None:
        if not self.matchup_id.strip() or self.matchup_id != self.matchup_id.strip():
            raise ValueError("matchup_id must be a non-empty trimmed string")
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("matchup fraction must be finite and in (0, 1]")
        if not self.subjects or not self.opponents:
            raise ValueError("matchup subjects and opponents must be non-empty")
        if any(not item.strip() or item != item.strip() for item in self.subjects):
            raise ValueError("matchup subjects must be non-empty trimmed strings")
        if any(not item.strip() or item != item.strip() for item in self.opponents):
            raise ValueError("matchup opponents must be non-empty trimmed strings")
        if len(set(self.subjects)) != len(self.subjects):
            raise ValueError("matchup subjects must be unique")
        if len(set(self.opponents)) != len(self.opponents):
            raise ValueError("matchup opponents must be unique")
        if not self.position_use.strip():
            raise ValueError("matchup position_use must not be empty")
        if type(self.distinct_subject_opponent_required) is not bool:
            raise TypeError("distinct_subject_opponent_required must be bool")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["subjects"] = list(self.subjects)
        result["opponents"] = list(self.opponents)
        return result


@dataclass(frozen=True, slots=True)
class ReplayAllocation:
    """Initial value-training mixture; every fraction is an ablation baseline."""

    source_id: str
    fraction: float
    role: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or self.source_id != self.source_id.strip():
            raise ValueError("replay source_id must be a non-empty trimmed string")
        if not math.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("replay fraction must be finite and in (0, 1]")
        if not self.role.strip():
            raise ValueError("replay role must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegisteredModelUsageGate:
    """Evidence that every registered model actually played its minimum games."""

    required_games: tuple[tuple[str, int], ...]
    observed_games: tuple[tuple[str, int], ...]
    source_bundle_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        required_ids = tuple(model_id for model_id, _ in self.required_games)
        observed_ids = tuple(model_id for model_id, _ in self.observed_games)
        if required_ids != tuple(sorted(set(required_ids))):
            raise ValueError("required model usage IDs must be unique and sorted")
        if observed_ids != required_ids:
            raise ValueError("observed model usage IDs must exactly match required IDs")
        if any(games < 1 for _, games in self.required_games):
            raise ValueError("required model game counts must be positive")
        if any(games < 0 for _, games in self.observed_games):
            raise ValueError("observed model game counts must be non-negative")
        if not self.source_bundle_sha256:
            raise ValueError("model usage gate requires source game bundles")
        if self.source_bundle_sha256 != tuple(sorted(set(self.source_bundle_sha256))):
            raise ValueError("source game-bundle hashes must be unique and sorted")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.source_bundle_sha256
        ):
            raise ValueError("source game-bundle hashes must be lowercase SHA-256")

    @property
    def complete(self) -> bool:
        return all(
            observed >= required
            for (_, required), (_, observed) in zip(
                self.required_games,
                self.observed_games,
                strict=True,
            )
        )

    @property
    def missing_games(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (model_id, required - observed)
            for (model_id, required), (_, observed) in zip(
                self.required_games,
                self.observed_games,
                strict=True,
            )
            if observed < required
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": REGISTERED_MODEL_USAGE_GATE_SCHEMA,
            "required_games": dict(self.required_games),
            "observed_games": dict(self.observed_games),
            "source_bundle_sha256": list(self.source_bundle_sha256),
            "complete": self.complete,
            "missing_games": dict(self.missing_games),
        }


def validate_registered_model_usage(
    bundle_receipts: Sequence[Mapping[str, object]],
) -> RegisteredModelUsageGate:
    """Require real local-only game receipts for all nine registered models."""

    registered_ids = tuple(sorted(record.rights_id for record in MODEL_RIGHTS))
    canonical_ids = set(CANONICAL_SCORER_IDS)
    required = tuple(
        (model_id, 240 if model_id in canonical_ids else 20) for model_id in registered_ids
    )
    observed = dict.fromkeys(registered_ids, 0)
    bundle_hashes: list[str] = []
    for index, receipt in enumerate(bundle_receipts):
        if receipt.get("schema") != _GAME_BUNDLE_SCHEMA:
            raise ValueError(f"game bundle {index} has an unsupported schema")
        if receipt.get("local_only") is not True or receipt.get("publication_allowed") is not False:
            raise ValueError(f"game bundle {index} is not a private local-only receipt")
        digest = receipt.get("payload_sha256")
        if not isinstance(digest, str):
            raise ValueError(f"game bundle {index} omits its payload SHA-256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"game bundle {index} has an invalid payload SHA-256")
        if receipt.get("payload_sha256_scope") != "receipt_without_payload_sha256_fields":
            raise ValueError(f"game bundle {index} has an unsupported payload hash scope")
        unsigned = dict(receipt)
        unsigned.pop("payload_sha256")
        unsigned.pop("payload_sha256_scope")
        try:
            canonical = json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError(f"game bundle {index} is not canonical JSON evidence") from error
        if hashlib.sha256(canonical).hexdigest() != digest:
            raise ValueError(f"game bundle {index} payload SHA-256 does not verify")
        bundle_hashes.append(digest)
        generation_value = receipt.get("generation")
        if not isinstance(generation_value, Mapping):
            raise ValueError(f"game bundle {index} omits generation evidence")
        metrics_value = generation_value.get("metrics")
        if not isinstance(metrics_value, Mapping):
            raise ValueError(f"game bundle {index} omits generation metrics")
        games = metrics_value.get("games")
        if isinstance(games, bool) or not isinstance(games, int) or games < 1:
            raise ValueError(f"game bundle {index} has an invalid game count")
        inputs_value = receipt.get("engine_inputs")
        if not isinstance(inputs_value, list) or len(inputs_value) != 2:
            raise ValueError(f"game bundle {index} must identify exactly two engine inputs")
        bundle_ids: set[str] = set()
        for raw_input in inputs_value:
            if not isinstance(raw_input, Mapping):
                raise ValueError(f"game bundle {index} contains an invalid engine input")
            if raw_input.get("kind") != "rights_profile":
                continue
            rights_id = raw_input.get("rights_id")
            if not isinstance(rights_id, str) or rights_id not in observed:
                raise ValueError(f"game bundle {index} contains an unknown rights ID")
            bundle_ids.add(rights_id)
        for rights_id in bundle_ids:
            observed[rights_id] += games
    gate = RegisteredModelUsageGate(
        required_games=required,
        observed_games=tuple((model_id, observed[model_id]) for model_id in registered_ids),
        source_bundle_sha256=tuple(sorted(bundle_hashes)),
    )
    if not gate.complete:
        raise ValueError(
            f"generation is missing registered-model games: {dict(gate.missing_games)}"
        )
    return gate


def _require_unit_sum(values: tuple[float, ...], *, label: str) -> None:
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} fractions must sum to exactly one")


def post_bootstrap_roadmap_contract() -> dict[str, object]:
    """Return the checked, JSON-serializable post-100B learning contract."""

    registered_ids = tuple(record.rights_id for record in MODEL_RIGHTS)
    if len(registered_ids) != len(set(registered_ids)):
        raise ValueError("registered model rights IDs must be unique")
    missing_teachers = set(CANONICAL_SCORER_IDS) - set(registered_ids)
    if missing_teachers:
        raise ValueError(
            f"canonical teachers are absent from the rights ledger: {missing_teachers}"
        )
    secondary_ids = tuple(
        rights_id for rights_id in registered_ids if rights_id not in CANONICAL_SCORER_IDS
    )
    teacher_pairs = tuple(
        (left, right)
        for index, left in enumerate(CANONICAL_SCORER_IDS)
        for right in CANONICAL_SCORER_IDS[index + 1 :]
    )
    initial_games_per_generation = 2_400

    matchups = (
        MatchupAllocation(
            "meteo_current_selfplay",
            0.25,
            ("meteo_current",),
            ("meteo_current",),
            "on_policy_coverage_and_evaluation_swing_mining",
            False,
        ),
        MatchupAllocation(
            "meteo_latest_vs_previous",
            0.20,
            ("meteo_latest_candidate",),
            ("meteo_previous_champion",),
            "incremental_regression_and_counterstrategy_mining",
            True,
        ),
        MatchupAllocation(
            "meteo_vs_three_teachers",
            0.30,
            ("meteo_current",),
            CANONICAL_SCORER_IDS,
            "teacher_specific_failure_mining_equal_share_per_teacher",
            True,
        ),
        MatchupAllocation(
            "three_teacher_crossplay",
            0.15,
            CANONICAL_SCORER_IDS,
            CANONICAL_SCORER_IDS,
            "attack_defence_and_strategy_diversity_not_played_move_labels",
            True,
        ),
        MatchupAllocation(
            "meteo_vs_secondary_registered_models",
            0.05,
            ("meteo_current",),
            secondary_ids,
            "off_lineage_diversity_and_blind_spot_mining",
            True,
        ),
        MatchupAllocation(
            "special_seeded_games",
            0.05,
            ("meteo_current",),
            ("mixed_strong_pool",),
            "tsec_book_exit_mate_repetition_entering_king_and_human_analysis_seeds",
            True,
        ),
    )
    _require_unit_sum(tuple(item.fraction for item in matchups), label="matchup")
    matchup_payloads: list[dict[str, object]] = []
    for item in matchups:
        exact_games = initial_games_per_generation * item.fraction
        if not exact_games.is_integer() or int(exact_games) % 2 != 0:
            raise ValueError("matchup fractions must produce even paired game counts")
        payload = item.to_dict()
        payload["games_per_generation"] = int(exact_games)
        matchup_payloads.append(payload)

    replay = (
        ReplayAllocation(
            "broad_100b_bootstrap_anchor",
            0.50,
            "prevent_forgetting_and_preserve_global_value_calibration",
        ),
        ReplayAllocation(
            "meteo_selfplay_failures",
            0.15,
            "learn_new_on_policy_weaknesses_after_deep_reanalysis",
        ),
        ReplayAllocation(
            "meteo_teacher_game_failures",
            0.15,
            "learn_teacher_specific_counterexamples_after_common_rescoring",
        ),
        ReplayAllocation(
            "teacher_disagreement_and_crossplay",
            0.10,
            "learn_difficult_strategy_boundaries_without_majority_vote",
        ),
        ReplayAllocation(
            "mate_terminal_repetition_and_entering_king",
            0.05,
            "retain_exact_rule_and_terminal_outcomes",
        ),
        ReplayAllocation(
            "tsec_book_exit_and_human_analysis",
            0.05,
            "improve_analysis_coverage_and_known_opening_responses",
        ),
    )
    _require_unit_sum(tuple(item.fraction for item in replay), label="replay")

    return {
        "schema": POST_BOOTSTRAP_ROADMAP_SCHEMA,
        "entry_gate": {
            "bootstrap_target_presentations": 100_000_000_000,
            "target_is_presentations_not_unique_positions": True,
            "current_run_is_immutable_while_active": True,
            "requires_complete_run_and_reloadable_latest_previous": True,
            "requires_all_superbatch_source_and_quantisation_receipts": True,
            "final_step_is_not_assumed_best_without_arena_evidence": True,
        },
        "objective": {
            "primary": "strongest_verified_local_meteo_engine",
            "targets": [
                *CANONICAL_SCORER_IDS,
                HISUI_SEALED_TARGET_ID,
                NAGISA_NEXT_SEALED_TARGET_ID,
            ],
            "uses": ["strongest_play", "multi_pv_analysis", "human_play"],
            "bootstrap_alone_proves_target_superiority": False,
        },
        "inventory": {
            "registered_local_models": list(registered_ids),
            "registered_local_model_count": len(registered_ids),
            "canonical_teachers": list(CANONICAL_SCORER_IDS),
            "secondary_registered_opponents": list(secondary_ids),
            "excluded_model": "takewarabe",
            "hisui": {
                "role": "sealed_external_completed_engine_target",
                "local_artifact_available": False,
                "training_use_allowed": False,
                "automatic_scraping_or_unapproved_bulk_play": False,
                "activation_requires": [
                    "lawfully_obtained_artifact_or_explicit_service_permission",
                    "binary_eval_progress_book_options_hash_receipt",
                ],
            },
            "nagisa_v4_unpublished": {
                "role": "sealed_unpublished_future_engine_target",
                "public_release_or_reproducible_benchmark_available": False,
                "local_artifact_available": False,
                "training_use_allowed": False,
                "v3_1_is_a_proxy_not_proof_of_v4_superiority": True,
                "activation_requires": [
                    "lawfully_obtained_exact_artifact_or_explicit_service_permission",
                    "binary_eval_progress_book_options_hash_receipt",
                    "same_opening_same_hardware_same_time_control_color_swapped_games",
                ],
            },
        },
        "game_generation": {
            "initial_games_per_generation": initial_games_per_generation,
            "allocations_are_initial_ablation_values_not_strength_claims": True,
            "matchups": matchup_payloads,
            "three_teacher_pairs": [list(pair) for pair in teacher_pairs],
            "all_registered_models_used_each_generation": True,
            "registered_model_usage_gate": {
                "required_ids": list(registered_ids),
                "canonical_teacher_roles": [
                    "complete_game_opponent",
                    "candidate_proposer",
                    "single_scale_scoring_anchor_ablation",
                ],
                "secondary_model_roles": [
                    "paired_complete_game_opponent",
                    "off_lineage_failure_mining",
                    "optional_proposal_only_challenger_after_anchor_rescoring",
                ],
                "raw_secondary_scores_enter_anchor_targets": False,
                "same_lineage_models_count_as_independent_votes": False,
                "generation_complete_requires_each_id_in_game_receipts": True,
            },
            "meteo_teacher_share_per_teacher": 0.10,
            "teacher_crossplay_share_per_pair": 0.05,
            "latest_previous_weight_artifacts_only": True,
            "old_generation_hash_and_result_metadata_retained": True,
            "rules": {
                "resign": False,
                "evaluation_adjudication": False,
                "continue_to": [
                    "checkmate",
                    "legal_repetition_result",
                    "entering_king_declaration",
                    "other_rule_terminal",
                ],
                "safety_ply_limit_is_debug_only": True,
                "truncated_games_enter_strong_terminal_labels": False,
                "paired_colors": True,
                "opening_exploration": "bounded_deep_regret_only",
            },
            "performance_measurement": {
                "strength_claim_before_nnue_runtime_adapter": False,
                "production_game_search": "multipv_1",
                "deep_three_teacher_reanalysis": "selected_positions_only",
                "concurrency_sweep": [1, 2, 4, 8],
                "required_metrics": [
                    "engine_nps_per_game",
                    "median_and_p95_seconds_per_game",
                    "completed_games_per_hour",
                    "positions_written_per_second",
                    "teacher_reanalysis_positions_per_hour",
                    "cpu_memory_thermal_and_disk_pressure",
                ],
                "selection_rule": (
                    "maximise_completed_games_per_hour_subject_to_"
                    "correct_terminal_games_and_fixed_search_budget"
                ),
                "receipt_required": True,
                "historical_teacher_reference_only": {
                    "is_meteo_measurement": False,
                    "search_nodes_per_move": 20_000,
                    "games_per_matchup_sample": 16,
                    "serial_search_time_ceiling_games_per_hour": [1648.2, 1737.5],
                    "excludes_orchestration_io_and_thermal_limits": True,
                    "must_not_be_used_as_meteo_production_eta": True,
                    "source": (
                        "artifacts/runs/20260809-multiteacher-distill-v1/"
                        "nagisa-suisho-limited-local-training-v1/"
                        "teacher-vs-teacher-corpus-v1/matches/"
                        "suisho-self-and-vs-nagisa-nobook20k-mpv1-8pairs-"
                        "20260809T1543"
                    ),
                },
            },
        },
        "artifact_retention": {
            "full_training_checkpoint_limit": 2,
            "playable_weight_window_limit": 2,
            "full_checkpoints": "latest_and_immediately_previous_only",
            "historical_weight_files_retained": False,
            "historical_metadata_append_only": True,
            "compact_champion_registry_limit": 1,
            "compact_champion_contains_optimizer_state": False,
            "compact_champion_must_reference_playable_weight_window": True,
            "content_addressed_deduplication_required": True,
            "third_independent_full_checkpoint_allowed": False,
        },
        "teacher_reanalysis": {
            "candidate_proposers": [*CANONICAL_SCORER_IDS, "meteo_current"],
            "single_scorer_per_experiment_arm": True,
            "permitted_scorer_arms": list(CANONICAL_SCORER_IDS),
            "cross_teacher_raw_score_average": False,
            "majority_vote": False,
            "nagisa_and_soujou_are_independent_votes": False,
            "multi_pv_role": "candidate_generation_only",
            "candidate_scoring": "equal_requested_nodes_searchmoves_multipv_1",
            "actual_played_move_is_best_move_label": False,
            "teacher_played_move_is_best_move_label": False,
            "both_sides_moves_are_reanalysed": True,
            "reply_backup_required": True,
            "qsearch_leaf_is_rerun_and_rescored": True,
            "mate_bounds_and_terminal_results_remain_separate": True,
            "unresolved_action": "increase_budget_then_keep_partial_interval_or_exclude",
            "priority_triggers": [
                "meteo_teacher_move_disagreement",
                "teacher_family_disagreement",
                "large_evaluation_swing",
                "high_confidence_meteo_error",
                "top_move_changes_at_tenfold_nodes",
                "reply_backup_changes_top_move",
                "mate_or_score_sign_conflict",
                "narrow_top_two_interval",
                "tsec_or_book_exit_position",
                "repetition_entering_king_or_extreme_value",
            ],
            "secondary_registered_model_use": {
                "model_ids": list(secondary_ids),
                "candidate_role": "proposal_only_optional_ablation",
                "played_move_role": "position_source_only",
                "final_value_source": "selected_single_canonical_anchor_only",
                "cross_family_agreement_increases_confidence_not_target_weight": True,
            },
        },
        "reinforcement_learning": {
            "algorithm_family": "teacher_stabilised_fitted_value_and_expert_iteration",
            "direct_policy_gradient": False,
            "student_policy_head": False,
            "search_policy_improver": "alpha_beta_with_current_meteo_value",
            "per_move_target": "sign_corrected_child_qsearch_leaf_scalar_value",
            "terminal_signal": "actual_complete_game_wdl_or_proven_rule_result",
            "teacher_search_signal": "deep_common_scale_scalar_value",
            "wdl_lambda_comparison_arms": [0.0, 0.10, 0.25],
            "wdl_lambda_is_selected_by_heldout_and_elo_not_assumed": True,
            "terminal_results_override_eval_adjudication": True,
            "replay_mix": [item.to_dict() for item in replay],
            "failed_candidate_data_may_survive_after_teacher_review": True,
        },
        "stages": [
            {
                "id": "B0",
                "name": "freeze_and_audit_100b_baseline",
                "exit": "complete_receipts_native_qat_parity_and_reloadable_exports",
            },
            {
                "id": "B1",
                "name": "three_axis_baseline",
                "exit": "fixed_node_fixed_time_and_completed_engine_results_recorded",
            },
            {
                "id": "B2",
                "name": "single_scorer_label_pilot",
                "exit": "anchor_only_and_three_multi_proposer_arms_compared_on_same_positions",
            },
            {
                "id": "B3",
                "name": "complete_game_failure_mining_and_deep_reanalysis",
                "exit": "unique_reanalysed_value_psv_and_split_receipts_frozen",
            },
            {
                "id": "B4",
                "name": "value_only_replay_training",
                "exit": "multiple_seed_qat_candidates_pass_value_and_worst_group_gates",
            },
            {
                "id": "B5",
                "name": "architecture_and_search_tournament",
                "exit": "fixed_time_winner_then_meteo_specific_spsa_and_time_management",
            },
            {
                "id": "B6",
                "name": "continuous_league",
                "exit": "repeat_until_stopped_with_generation_promotion_gates",
            },
        ],
        "architecture_tournament": {
            "same_data_seed_steps_and_search_conditions": True,
            "arms": [
                "current_nagisa_style_1024_layerstack9",
                "suisho11_style_halfkav2_1024",
                "soujou_tsec7_style_halfkahm_2048_layerstack9",
                "hisui_inspired_feature_layerstack_and_quantisation_ablations_without_topology_claim",
                "shared_ft_progress_layerstack_hybrid",
            ],
            "bigger_network_is_not_assumed_stronger": True,
            "selection_metric": "fixed_time_elo_after_nps_cost",
        },
        "quantisation_and_speed": {
            "native_export_must_match_qat_forward": True,
            "float_master_is_not_a_deployable_model": True,
            "required_diagnostics": [
                "native_qat_probability_error",
                "float_master_deployment_probability_drift",
                "combined_factorised_weight_saturation",
                "per_layer_rounding_residual",
                "heldout_loss_delta_float_vs_deployment",
                "extreme_value_and_progress_bucket_drift",
            ],
            "qat_ablation_arms": [
                "current_exact_qat",
                "representable_range_projection_after_optimizer",
                "master_to_deployment_consistency_regulariser",
                "fv_scale_and_output_calibration",
            ],
            "unsafe_mid_run_numeric_change": False,
            "next_run_speed_ablations": [
                "native_decode_threads",
                "batch_size",
                "checkpoint_interval",
                "custom_fused_metal_sparse_gather_backward",
            ],
            "more_reserved_memory_alone_is_expected_to_speed_compute": False,
        },
        "promotion": {
            "generation_gate": [
                "candidate_beats_previous_champion_on_paired_openings",
                "no_local_teacher_regression_each_teacher_separately",
                "fixed_node_and_fixed_time_both_pass",
                "worst_group_analysis_rule_and_quantisation_gates_pass",
            ],
            "test_sequence": ["proxy", "stc_sprt", "ltc_sprt", "long_analysis"],
            "minimum_formal_opening_pairs_per_opponent": 500,
            "both_colors_required": True,
            "do_not_stop_sprt_before_terminal_decision": True,
            "target_superiority_claim": [
                "lower_95_score_above_0_5_vs_each_local_teacher",
                "same_claim_vs_pinned_hisui_only_after_artifact_permission",
                "same_claim_vs_pinned_nagisa_v4_only_after_artifact_or_service_permission",
                "exact_versions_hashes_hardware_time_control_and_book_state_reported",
            ],
            "unpublished_nagisa_v4_reserve": {
                "v3_1_proxy_lower_95_score_target": 0.55,
                "proxy_is_not_a_v4_strength_claim": True,
                "required_proxy_time_controls": ["fixed_nodes", "stc", "ltc"],
                "required_direct_gate_when_available": (
                    "candidate black white and paired lower 95 score above 0.5"
                ),
                "unknown_version_identity_must_fail_closed": True,
            },
            "analysis_gate": [
                "top_move_regret",
                "pv_stability_at_1x_and_10x_nodes",
                "wdl_calibration",
                "multipv_candidate_independent_research",
                "mate_repetition_entering_king_correctness",
                "tsec_and_book_exit_worst_group",
            ],
            "human_profile": {
                "canonical_weight_is_not_weakened": True,
                "strength_control": ["time", "nodes", "book_scope", "handicap"],
            },
        },
        "runtime_profiles": {
            "strongest_play": "multipv_1_book_and_time_management_after_spsa",
            "analysis": "pv1_priority_bounded_extra_pvs_independent_candidate_research",
            "human_play": "same_champion_weight_strength_control_outside_evaluation",
            "llm_role": "explain_engine_and_rule_facts_never_choose_the_best_move",
        },
        "implementation_readiness": {
            "available": [
                "external_usi_game_and_search_primitives",
                "teacher_candidate_and_interval_arbitration_primitives",
                "append_only_generation_dataset_and_promotion_contracts",
                "mlx_value_only_qat_checkpoint_and_yaneuraou_export",
            ],
            "required_before_B3_production": [
                "execute_generation_plan_with_current_nnue_and_three_exact_teacher_profiles",
                "convert_reanalysed_child_values_and_terminal_wdl_to_incremental_psv",
                "resume_value_only_mlx_training_from_100b_champion_with_replay_mix",
                "run_spsa_and_multi_time_control_arena_receipts",
            ],
            "automatic_start_immediately_after_100b": False,
        },
    }
