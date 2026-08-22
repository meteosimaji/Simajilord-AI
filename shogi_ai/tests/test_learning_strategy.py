from __future__ import annotations

import json
import math

import pytest

from simajilord_shogi.anchor_labeling import (
    AnchorCandidateScore,
    AnchorReanalysisReason,
    AnchorResolutionKind,
    AnchorValueLossKind,
    anchor_reanalysis_reasons,
    interval_squared_error,
    resolve_anchor_target,
)
from simajilord_shogi.cli import main
from simajilord_shogi.distillation_targets import (
    CANONICAL_SCORER_IDS,
    SOUJOU_TSEC7_SCORER_ID,
)
from simajilord_shogi.domain import TeacherScoreBound
from simajilord_shogi.learning_strategy import (
    LEARNING_STRATEGY_SCHEMA,
    LIFETIME_POSITIONS_SEEN_TARGET,
    learning_strategy_contract,
    teacher_role_contract,
)


def _score(
    move: str,
    value: float,
    bound: TeacherScoreBound = TeacherScoreBound.EXACT,
    *,
    requested_nodes: int = 1_000_000,
    reported_nodes: int | None = None,
) -> AnchorCandidateScore:
    return AnchorCandidateScore(
        move=move,
        q_value=value,
        bound=bound,
        requested_nodes=requested_nodes,
        reported_nodes=reported_nodes,
    )


def test_learning_strategy_keeps_all_three_primary_scorer_arms() -> None:
    contract = learning_strategy_contract()

    assert contract["schema"] == LEARNING_STRATEGY_SCHEMA
    active = contract["active_training_route"]
    assert active["mode"] == "nagisa_style_value_only_nnue_sfnn"
    assert active["policy_head"] is False
    assert active["student_weights"] == "random_init"
    assert active["target_presentations"] == 100_000_000_000
    assert active["primary_unique_source_positions"] == 49_594_855_063
    assert active["optimizer_device"] == "LOCAL_APPLE_GPU_MLX"
    assert active["score_scale"] == 600.0
    assert active["loss"] == "WRM_identity_sigmoid_target"
    assert active["wrm_nnue2score"] == 508.0
    assert active["yaneuraou_fv_scale"] == 16
    assert active["systematic_export_scale_ratio"] == 1.0
    assert active["score_drop_abs"] is None
    assert active["extreme_score_policy"] == "keep_including_mate_stamps"
    primary = contract["primary_label_strategy"]
    assert primary["status"] == "legacy_policy_value_research_control_not_active_production"
    assert primary["provisional_anchor"] == SOUJOU_TSEC7_SCORER_ID
    assert tuple(primary["permitted_primary_scorers"]) == CANONICAL_SCORER_IDS
    assert primary["cross_teacher_value_average"] is False
    assert primary["majority_vote"] is False
    assert len(primary["scorer_arms"]) == 3
    assert contract["data_scale"]["lifetime_cumulative_positions_seen_target"] == (
        LIFETIME_POSITIONS_SEEN_TARGET
    )
    assert LIFETIME_POSITIONS_SEEN_TARGET == 100_000_000_000

    recipe = contract["lineage_informed_recipe"]
    assert recipe["compute_split"]["meteo_neural_optimization"] == "MLX_LOCAL_APPLE_GPU"
    assert (
        recipe["teacher_layers"]["nagisa_soujou_shared_ancestry_is_two_independent_votes"] is False
    )
    assert recipe["checkpoint_contract"]["retained_generations"] == ["latest", "previous"]
    assert sum(recipe["p2_initial_sampling_weights"].values()) == pytest.approx(1.0)

    public_reuse = contract["public_corpus_reuse"]
    assert public_reuse["soujou_datasets_1_records"] == 49_594_855_063
    assert public_reuse["soujou_datasets_2_records"] == 111_945_173_977
    assert public_reuse["nominal_100b_source_is_not_100b_unique_positions"] is True
    assert public_reuse["two_passes_plus_remainder_reach_100b_presentations"] is True
    assert public_reuse["historical_policy_used"] is False
    assert public_reuse["legacy_and_unique_are_additive_independent_sources"] is False
    assert set(public_reuse["related_hugging_face_publishers_audited"]) == {
        "sojoteam",
        "penguinkumimanu",
        "washiun",
        "nodchip",
    }

    source_layers = {layer["id"]: layer for layer in recipe["source_layers"]}
    assert source_layers["soujou_datasets_1_current_position_pool"]["records"] == (49_594_855_063)
    assert (
        source_layers["soujou_datasets_1_current_position_pool"][
            "legacy_value_used_as_production_label"
        ]
        is True
    )
    assert (
        source_layers["soujou_datasets_2_gated_expansion"]["counts_toward_lifetime_target_now"]
        is False
    )
    diversity = source_layers["public_game_and_strategy_diversity"]
    assert diversity["played_move_used_as_best_move_label"] is False
    assert diversity["qpd_historical_repetition_factor_not_unique"] == 20

    post_100b = contract["post_100b"]
    assert post_100b["entry_gate"]["bootstrap_target_presentations"] == (100_000_000_000)
    assert post_100b["game_generation"]["latest_previous_weight_artifacts_only"] is True
    assert post_100b["reinforcement_learning"]["student_policy_head"] is False


@pytest.mark.parametrize("scorer_id", CANONICAL_SCORER_IDS)
def test_teacher_role_contract_has_one_scorer_and_two_challengers(scorer_id: str) -> None:
    role = teacher_role_contract(scorer_id)

    assert role.scorer_id == scorer_id
    assert set(role.proposer_ids) == set(CANONICAL_SCORER_IDS)
    assert set(role.challenger_ids) == set(CANONICAL_SCORER_IDS) - {scorer_id}


def test_exact_anchor_q_builds_policy_without_using_reported_node_counts() -> None:
    scores = (
        _score("2g2f", 0.4, reported_nodes=80_000),
        _score("7g7f", 0.2, reported_nodes=1_000_000),
    )

    target = resolve_anchor_target(
        SOUJOU_TSEC7_SCORER_ID,
        scores,
        policy_temperature=0.1,
    )

    assert target.resolution is AnchorResolutionKind.EXACT_Q_DISTRIBUTION
    assert target.policy_loss_enabled is True
    assert target.value_loss_kind is AnchorValueLossKind.POINT
    assert target.chosen_move == "2g2f"
    assert target.value_interval == (0.4, 0.4)
    assert math.isclose(sum(target.policy.values()), 1.0)
    assert target.policy["2g2f"] > target.policy["7g7f"]


def test_unequal_requested_budgets_are_rejected_even_if_q_is_exact() -> None:
    with pytest.raises(ValueError, match="equal requested node budget"):
        resolve_anchor_target(
            SOUJOU_TSEC7_SCORER_ID,
            (
                _score("2g2f", 0.4, requested_nodes=100_000),
                _score("7g7f", 0.2, requested_nodes=1_000_000),
            ),
            policy_temperature=0.1,
        )


def test_interval_dominance_proves_best_move_without_all_exact() -> None:
    target = resolve_anchor_target(
        SOUJOU_TSEC7_SCORER_ID,
        (
            _score("2g2f", 0.5, TeacherScoreBound.LOWER),
            _score("7g7f", 0.4, TeacherScoreBound.UPPER),
        ),
        policy_temperature=0.1,
        interval_margin=0.05,
    )

    assert target.resolution is AnchorResolutionKind.INTERVAL_DOMINANCE
    assert target.policy == {"2g2f": 1.0}
    assert target.value_interval == (0.5, 1.0)
    assert target.value_loss_kind is AnchorValueLossKind.INTERVAL
    assert target.additional_search_required is False


def test_overlapping_bounds_disable_policy_but_keep_interval_value() -> None:
    target = resolve_anchor_target(
        SOUJOU_TSEC7_SCORER_ID,
        (
            _score("2g2f", 0.2, TeacherScoreBound.LOWER),
            _score("7g7f", 0.3, TeacherScoreBound.UPPER),
        ),
        policy_temperature=0.1,
    )

    assert target.resolution is AnchorResolutionKind.UNRESOLVED
    assert target.policy == {}
    assert target.policy_loss_enabled is False
    assert target.value_loss_kind is AnchorValueLossKind.INTERVAL
    assert target.value_interval == (0.2, 1.0)
    assert target.additional_search_required is True


def test_internal_proven_mate_is_set_valued_and_overrides_reported_scores() -> None:
    target = resolve_anchor_target(
        SOUJOU_TSEC7_SCORER_ID,
        (_score("2g2f", -0.2), _score("7g7f", -0.3)),
        policy_temperature=0.1,
        proven_mate_moves=("7g7f", "2g2f"),
    )

    assert target.resolution is AnchorResolutionKind.PROVEN_MATE
    assert target.policy == {"2g2f": 0.5, "7g7f": 0.5}
    assert target.value_interval == (1.0, 1.0)


def test_interval_loss_is_zero_inside_and_hinge_outside() -> None:
    assert interval_squared_error(0.25, (0.2, 0.3)) == 0.0
    assert interval_squared_error(0.1, (0.2, 0.3)) == pytest.approx(0.01)
    assert interval_squared_error(0.5, (0.2, 0.3)) == pytest.approx(0.04)


def test_reanalysis_reasons_are_trigger_rights_not_cross_teacher_votes() -> None:
    scores = (
        _score("2g2f", 0.2, TeacherScoreBound.LOWER),
        _score("7g7f", 0.3, TeacherScoreBound.UPPER),
    )

    reasons = anchor_reanalysis_reasons(
        anchor_top_move="2g2f",
        challenger_top_moves=("7g7f", "2g2f"),
        teacher_values=(-0.1, 0.2, 0.3),
        reported_mate=True,
        budget_top_moves=("2g2f", "7g7f"),
        reply_backed_top_move="7g7f",
        exact_top_gap=0.01,
        narrow_gap_threshold=0.02,
        scores=scores,
    )

    assert set(reasons) == set(AnchorReanalysisReason)


def test_learning_strategy_cli_prints_the_same_contract(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["learning-strategy"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == learning_strategy_contract()
