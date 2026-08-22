"""Machine-readable contract for Meteo's current NNUE and legacy research routes.

The active production bootstrap is a NAGISA-compatible, value-only NNUE/SFNN
trained by Tatara.  The older MLX Policy+Value committee experiments remain
described as non-production research controls so their artifacts cannot be
mistaken for the current Meteo model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .distillation_targets import (
    CANONICAL_SCORER_IDS,
    NAGISA_SCORER_ID,
    SOUJOU_TSEC7_SCORER_ID,
    SUISHO11PLUS_SCORER_ID,
)
from .post_bootstrap import post_bootstrap_roadmap_contract
from .teacher_value_ensemble import TeacherValueEnsembleMode

LEARNING_STRATEGY_SCHEMA = "meteo-learning-strategy-v3"
LIFETIME_POSITIONS_SEEN_TARGET = 100_000_000_000


@dataclass(frozen=True, slots=True)
class TeacherRoleContract:
    """Teacher roles for one single-scorer experiment arm."""

    scorer_id: str
    proposer_ids: tuple[str, ...]
    challenger_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        canonical = set(CANONICAL_SCORER_IDS)
        if self.scorer_id not in canonical:
            raise ValueError(f"unknown canonical scorer: {self.scorer_id}")
        if len(set(self.proposer_ids)) != len(self.proposer_ids):
            raise ValueError("proposer IDs must be unique")
        if set(self.proposer_ids) != canonical:
            raise ValueError("every canonical teacher must propose candidates")
        if len(set(self.challenger_ids)) != len(self.challenger_ids):
            raise ValueError("challenger IDs must be unique")
        if set(self.challenger_ids) != canonical - {self.scorer_id}:
            raise ValueError("challengers must be exactly the two non-scoring teachers")

    def to_dict(self) -> dict[str, object]:
        return {
            "scorer_id": self.scorer_id,
            "proposer_ids": list(self.proposer_ids),
            "challenger_ids": list(self.challenger_ids),
        }


def teacher_role_contract(scorer_id: str) -> TeacherRoleContract:
    """Return the canonical multi-proposer/single-scorer role assignment."""

    return TeacherRoleContract(
        scorer_id=scorer_id,
        proposer_ids=CANONICAL_SCORER_IDS,
        challenger_ids=tuple(
            teacher_id for teacher_id in CANONICAL_SCORER_IDS if teacher_id != scorer_id
        ),
    )


def learning_strategy_contract() -> dict[str, object]:
    """Return the authoritative, JSON-serializable staged-learning contract."""

    scorer_arms = tuple(teacher_role_contract(scorer_id) for scorer_id in CANONICAL_SCORER_IDS)
    return {
        "schema": LEARNING_STRATEGY_SCHEMA,
        "production_learning": {
            "requested": True,
            "active_route": "nagisa_style_value_only_nnue_sfnn",
            "entrypoint": "simajilord-nnue",
            "start_only_after_corpus_heldout_hardware_and_checkpoint_gates": True,
            "parallel_with_teacher_benchmark": False,
        },
        "active_training_route": {
            "mode": "nagisa_style_value_only_nnue_sfnn",
            "learner": "Tatara",
            "student_weights": "random_init",
            "teacher_nnue_weights_copied": False,
            "progress_router": "NAGISA_V3.1_progress8kpabs_only",
            "features": "HalfKA_hm2(Friend)",
            "feature_dimensions": 73_305,
            "network": "SFNNWithoutPsqt_1024_15_64_LayerStack9",
            "policy_head": False,
            "policy_loss": 0.0,
            "primary_value_corpus": "soujou-team-datasets-1",
            "primary_unique_source_positions": 49_594_855_063,
            "target_presentations": LIFETIME_POSITIONS_SEEN_TARGET,
            "target_is_unique_positions": False,
            "score_scale": 600.0,
            "loss": "WRM_identity_sigmoid_target",
            "wrm_nnue2score": 508.0,
            "yaneuraou_fv_scale": 16,
            "systematic_export_scale_ratio": 1.0,
            "score_drop_abs": None,
            "extreme_score_policy": "keep_including_mate_stamps",
            "optimizer_device": "LOCAL_APPLE_GPU_MLX",
            "position_generation_for_primary_bootstrap": "already_published_qsearch_psv",
            "current_three_teacher_search_required_for_primary_bootstrap": False,
            "current_three_teachers_after_bootstrap": "challenge_reanalyse_and_arena",
            "checkpoint_retention": ["latest", "previous"],
            "public_checkpoint_release_allowed": False,
        },
        "primary_label_strategy": {
            "status": "legacy_policy_value_research_control_not_active_production",
            "mode": "multi_proposer_single_scorer",
            "provisional_anchor": SOUJOU_TSEC7_SCORER_ID,
            "permitted_primary_scorers": list(CANONICAL_SCORER_IDS),
            "scorer_arms": [arm.to_dict() for arm in scorer_arms],
            "cross_teacher_value_average": False,
            "majority_vote": False,
            "consensus_role": "confidence_only",
            "disagreement_role": "additional_search",
            "bounds": "interval_aware",
            "unresolved_policy_loss": "disabled",
            "alpha_beta_nodes_are_policy": False,
            "policy_source": "equal_requested_budget_anchor_q",
        },
        "teacher_selection": {
            "provisional_is_not_permanent": True,
            "required_candidates": [
                SOUJOU_TSEC7_SCORER_ID,
                NAGISA_SCORER_ID,
                SUISHO11PLUS_SCORER_ID,
            ],
            "metrics": [
                "deep_adjudicated_best_move_regret",
                "tenfold_budget_top1_stability",
                "fatal_blunder_rate",
                "top_5_percent_regret",
                "top_1_percent_regret",
                "mate_miss_rate",
                "wdl_calibration_error",
                "worst_group_regret",
                "fixed_node_strength",
                "fixed_time_strength",
            ],
        },
        "rounds": [
            {
                "id": "S",
                "name": "pipeline-smoke-unanimous",
                "dataset": "strict_unanimous_20",
                "model_profile": "smoke",
                "purpose": "pipeline_validation_only",
                "intentionally_overfit": True,
                "promotable_checkpoint": False,
            },
            {
                "id": "A",
                "name": "single-teacher-baselines",
                "scorer_arms": list(CANONICAL_SCORER_IDS),
                "candidate_sources": "same_scorer_only",
                "required": True,
            },
            {
                "id": "B",
                "name": "multi-proposer-single-scorer",
                "scorer_arms": list(CANONICAL_SCORER_IDS),
                "candidate_sources": list(CANONICAL_SCORER_IDS),
                "position": "primary_candidate",
            },
            {
                "id": "C",
                "name": "unanimous-only",
                "position": "high-purity-control",
            },
            {
                "id": "D",
                "name": "robust-three-teacher-ensemble",
                "position": "experimental-control",
                "suisho11_reference": {
                    "public_components": ["Ryfamate latest", "DL Suisho", "AobaZero"],
                    "exact_formula_public": False,
                    "must_not_claim_reproduction": True,
                },
                "required_scalar_comparators": [
                    TeacherValueEnsembleMode.CONVERTED_SCORE_ARITHMETIC_MEAN.value,
                    TeacherValueEnsembleMode.PROBABILITY_ARITHMETIC_MEAN.value,
                    "interval_robust_three_teacher",
                ],
            },
            {
                "id": "E",
                "name": "coarse-group-teacher-routing",
                "position": "late_experiment_after_measured_specialization",
            },
        ],
        "data_scale": {
            "lifetime_cumulative_positions_seen_target": LIFETIME_POSITIONS_SEEN_TARGET,
            "unique_positions_retained_is_separate": True,
            "raw_score_matrices_are_high_quality_shards": True,
            "training_buffer_is_compressed_deduplicated_and_bounded": True,
            "repeated_epoch_presentations_count_as_seen_not_unique": True,
            "quality_floor_must_not_be_lowered_to_reach_the_counter": True,
            "gates": [
                {"positions": "20-100", "purpose": "pipeline_overfit_smoke"},
                {"positions": "about_10_000", "purpose": "sign_loss_split_leakage_audit"},
                {"positions": "hundreds_of_thousands", "purpose": "A_B_C_D_direction"},
                {"positions": "millions_plus", "purpose": "competition_v1_bootstrap"},
                {"positions": "continuous", "purpose": "selfplay_and_failure_reanalysis"},
            ],
        },
        "public_corpus_reuse": {
            "catalog_command": "simajilord-shogi teacher-lineage",
            "human_audit_document": "DATASET_AUDIT.md",
            "soujou_datasets_1_records": 49_594_855_063,
            "soujou_datasets_1_use": "direct_local_only_scalar_value_nnue_training",
            "soujou_datasets_2_records": 111_945_173_977,
            "soujou_datasets_2_use": "blocked_by_manual_gate_rights_and_binary_probe",
            "nominal_100b_source_is_not_100b_unique_positions": True,
            "two_passes_plus_remainder_reach_100b_presentations": True,
            "dlsuisho_unique_records": 14_668_949_437,
            "dlsuisho_unique_policy_labels_available": False,
            "dlsuisho_unique_value_pretraining": "blocked_until_loader_and_rights_gates",
            "legacy_dlsuisho_precursor_records": 16_667_054_287,
            "legacy_and_unique_are_additive_independent_sources": False,
            "mit_nodchip_sources": "position_seed_after_qsearch_dedup_and_relabel",
            "nagisa_soujou_source_counts_are_independent": False,
            "related_hugging_face_publishers_audited": [
                "sojoteam",
                "penguinkumimanu",
                "washiun",
                "nodchip",
            ],
            "historical_scalar_used_without_current_reanalysis_in_active_nnue_route": True,
            "historical_policy_used": False,
        },
        "lineage_informed_recipe": {
            "compute_split": {
                "primary_bootstrap_position_generation": "reused_public_qsearch_psv",
                "future_teacher_search_and_selfplay": "CPU",
                "meteo_neural_optimization": "MLX_LOCAL_APPLE_GPU",
                "arena_and_fixed_node_evaluation": "CPU",
                "concurrent_cpu_teacher_and_gpu_training": "future_after_resource_isolation",
                "reason": (
                    "CPU decodes exact Tatara PSV/features while the local Apple GPU runs the "
                    "float32 MLX port; no cloud GPU is used"
                ),
            },
            "source_layers": [
                {
                    "id": "soujou_datasets_1_current_position_pool",
                    "records": 49_594_855_063,
                    "role": "primary_large_position_generation_shortcut",
                    "status": "active_local_only_streamed_value_nnue_source",
                    "legacy_policy_available": False,
                    "legacy_value_used_as_production_label": True,
                    "required_transform": [
                        "operator_local_only_acknowledgement",
                        "immutable_revision_size_and_lfs_sha256_index",
                        "move16_zero_and_board_decode_probe",
                        "immutable_tail_heldout_before_optimizer",
                        "score_scale_600_and_value_only_loss",
                    ],
                },
                {
                    "id": "soujou_datasets_2_gated_expansion",
                    "records": 111_945_173_977,
                    "role": "future_broad_position_pool_after_access_and_audit",
                    "status": "blocked_manual_gate_rights_probe_and_overlap_dedup",
                    "counts_toward_lifetime_target_now": False,
                    "reason": (
                        "metadata scale is verified, but binary content is inaccessible "
                        "without authentication and deeply overlaps datasets_1"
                    ),
                },
                {
                    "id": "mit_depth9_position_seeds",
                    "sources": [
                        "nodchip-shogi-hao-depth9",
                        "nodchip-tanuki-nnue-pytorch-2024-07-30.1",
                        "nodchip-shogi-suisho5-depth9-validation",
                        "nodchip-shogi-suisho5-depth9-entering-king",
                    ],
                    "role": "position_generation_shortcut_only",
                    "legacy_move_score_result_used_as_labels": False,
                    "required_transform": [
                        "deterministic_range_receipt",
                        "board_deduplication",
                        "current_three_teacher_candidate_generation",
                        "single_scorer_equal_budget_reanalysis",
                    ],
                },
                {
                    "id": "soujou_dlsuisho_unique_family",
                    "records": 14_668_949_437,
                    "role": "optional_value_only_pretraining",
                    "policy_loss": 0.0,
                    "status": "blocked_pending_explicit_rights_and_value_only_loader",
                    "double_count_for_nagisa_and_soujou": False,
                },
                {
                    "id": "public_game_and_strategy_diversity",
                    "sources": [
                        "aobazero-public-selfplay-games",
                        "aoba-komaochi-public-selfplay-games",
                        "aoba-furibisha-public-selfplay-games",
                        "gct-wcsc31-public-training-data",
                        "qhapaq-pretty-daabi-training-kit",
                        "denryu-public-game-records",
                    ],
                    "role": "later_rl_diversity_strategy_rule_and_failure_seed",
                    "status": "blocked_pending_external_archive_rights_and_generation_parser",
                    "raw_policy_used_as_current_teacher_policy": False,
                    "history_and_rule_generation_must_be_preserved": True,
                    "played_move_used_as_best_move_label": False,
                    "qpd_unique_source_records": 915_204,
                    "qpd_historical_repetition_factor_not_unique": 20,
                },
                {
                    "id": "meteo_teacher_and_selfplay_failures",
                    "role": "new_information_beyond_teacher_ancestry",
                    "required_labels": [
                        "actual_board_terminal_result",
                        "proven_mate_or_no_mate_contract",
                        "current_deep_reanalysis",
                    ],
                },
            ],
            "teacher_layers": {
                "candidate_proposers": list(CANONICAL_SCORER_IDS),
                "single_scorer_arms": [f"B-{scorer_id}" for scorer_id in CANONICAL_SCORER_IDS],
                "provisional_first_arm": f"B-{SOUJOU_TSEC7_SCORER_ID}",
                "nagisa_soujou_shared_ancestry_is_two_independent_votes": False,
                "challenger_disagreement_action": "increase_search_not_average",
                "suisho11_style_ensemble": "Round_D_comparator_only",
                "suisho11_exact_formula_claimed": False,
                "extreme_probability_positions_deleted": False,
            },
            "stage_gates": [
                {
                    "stage": "S",
                    "input": "strict_current_three_teacher_unanimous_exact",
                    "positions": "20_train_10_calibration_10_diagnostic",
                    "model": "smoke_2x32_random_init",
                    "optimizer": "GPU",
                    "checkpoint_retention": 2,
                    "promotion_allowed": False,
                },
                {
                    "stage": "P0",
                    "input": (
                        "rights_reviewed_position_seeds_after_cross_lineage_dedup_and_"
                        "current_teacher_relabel"
                    ),
                    "positions": "about_10_000_unique",
                    "purpose": "sign_history_split_policy_mask_and_receipt_audit",
                    "promotion_allowed": False,
                },
                {
                    "stage": "P1",
                    "input": "same_positions_for_A_N_A_W_A_S_B_N_B_W_B_S_C_D",
                    "positions": "hundreds_of_thousands_unique",
                    "purpose": "select_label_strategy_and_scorer_by_regret_and_strength",
                    "promotion_allowed": False,
                },
                {
                    "stage": "P2",
                    "input": "millions_of_broad_relabelled_and_special_positions",
                    "model": "competition_v1_20x256",
                    "purpose": "teacher_bootstrap",
                    "promotion_allowed": "only_after_fixed_node_time_and_worst_group_gates",
                },
                {
                    "stage": "RL",
                    "input": "meteo_league_selfplay_teacher_games_and_failure_mining",
                    "purpose": "escape_shared_teacher_blind_spots",
                    "teachers_become": "challengers_and_reanalysers",
                },
            ],
            "p2_initial_sampling_weights": {
                "broad_current_teacher_relabel": 0.45,
                "deep_disagreement_and_student_failure": 0.20,
                "mate_terminal_and_tactical": 0.10,
                "repetition_entering_king_and_rule_history": 0.10,
                "deduplicated_replay_anchor": 0.15,
            },
            "checkpoint_contract": {
                "retained_generations": ["latest", "previous"],
                "ancestral_checkpoints_retained": False,
                "arena_pair": "latest_vs_previous",
                "atomic_complete_marker_required": True,
            },
            "scale_contract": {
                "lifetime_positions_seen": LIFETIME_POSITIONS_SEEN_TARGET,
                "unique_and_repeated_seen_are_separate": True,
                "lineage_duplicate_positions_are_counted_once_as_unique": True,
                "quality_gate_may_not_be_relaxed_for_scale": True,
            },
        },
        "required_receipts": [
            "immutable_corpus_index",
            "local_only_acknowledgement",
            "progress_router_extraction_receipt",
            "heldout_range_receipt",
            "per_shard_value_contract_receipt",
            "checkpoint_complete_marker",
        ],
        "legacy_policy_value_required_receipts": [
            "split_receipt",
            "score_matrix_receipt",
            "anchor_calibration_receipt",
            "checkpoint_complete_marker",
        ],
        "promotion_rule": [
            "identical_positions",
            "identical_initial_seed_sets",
            "heldout_regret_improvement",
            "fixed_node_improvement",
            "fixed_time_improvement",
            "no_material_worst_group_regression",
            "multiple_seed_reproduction",
        ],
        "long_term": [
            "teacher_bootstrap",
            "disagreement_mining",
            "meteo_vs_teacher_failure_mining",
            "selfplay",
            "deep_reanalysis",
            "teacher_role_reduced_to_challenger",
        ],
        "opening_book": {
            "petashock_role": "starting_position_distribution_only",
            "book_moves_are_policy_labels": False,
            "split_by_book_root": True,
            "prioritize_book_exit_positions": True,
        },
        "post_100b": post_bootstrap_roadmap_contract(),
    }
