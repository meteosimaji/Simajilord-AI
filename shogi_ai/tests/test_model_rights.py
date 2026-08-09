from __future__ import annotations

import pytest

from simajilord_shogi.model_rights import (
    MODEL_RIGHTS,
    RightsDecision,
    analysable_rights_ids,
    distillable_rights_ids,
    locally_distillable_rights_ids,
    model_rights,
)


def test_rights_registry_has_no_identifier_or_casefold_name_collisions() -> None:
    assert len({record.rights_id for record in MODEL_RIGHTS}) == len(MODEL_RIGHTS)
    assert len({record.name.casefold() for record in MODEL_RIGHTS}) == len(MODEL_RIGHTS)


@pytest.mark.parametrize(
    "rights_id",
    [
        "aobannue-v1.1",
        "dlshogi-gct-wcsc31",
        "nagisa-v3.1",
        "shinden3-2025-02-21",
        "suisho5",
    ],
)
def test_output_distillation_is_separate_from_artifact_reuse(rights_id: str) -> None:
    output_only = model_rights(rights_id)

    assert output_only.output_distillation == RightsDecision.ALLOWED
    assert output_only.direct_weight_use == RightsDecision.NOT_APPROVED
    assert output_only.original_artifact_redistribution == RightsDecision.NOT_APPROVED
    assert output_only.output_only_meteo_publication == RightsDecision.ALLOWED
    assert rights_id in distillable_rights_ids()


def test_artifacts_with_explicit_terms_retain_their_distinct_decisions() -> None:
    assert model_rights("gikou2-v2.0.2").direct_weight_use == RightsDecision.CONDITIONAL_GPL
    assert model_rights("hao-2023-05-08").direct_weight_use == RightsDecision.CONDITIONAL_GPL
    assert (
        model_rights("tanuki-dr4-2023-12-03").direct_weight_use
        == RightsDecision.CONDITIONAL_GPL
    )
    assert model_rights("aobazero-public-domain").direct_weight_use == RightsDecision.ALLOWED
    assert model_rights("yaneuraou-rezero").direct_weight_use == RightsDecision.ALLOWED
    apery = model_rights("apery-public")
    assert apery.direct_weight_use == RightsDecision.ALLOWED
    assert apery.original_artifact_redistribution == RightsDecision.ALLOWED


def test_takewarabe_approximation_is_an_opponent_without_external_weights() -> None:
    takewarabe = model_rights("takewarabe-approx-v7.50-material9")

    assert takewarabe.analysis == RightsDecision.ALLOWED
    assert takewarabe.hard_game_training == RightsDecision.ALLOWED
    assert takewarabe.direct_weight_use == RightsDecision.NOT_APPLICABLE
    assert takewarabe.original_artifact_redistribution == RightsDecision.CONDITIONAL_GPL
    assert takewarabe.rights_id in distillable_rights_ids()


def test_separate_dlshogi_model_terms_override_general_gpl_output_default() -> None:
    restricted = model_rights("dlshogi-dr2-exhi")

    assert restricted.output_distillation == RightsDecision.NOT_APPROVED
    assert restricted.hard_game_training == RightsDecision.LIMITED
    assert restricted.rights_id not in distillable_rights_ids()


def test_gikou_is_approved_but_hosted_hisui_is_not_locally_distillable() -> None:
    approved = set(distillable_rights_ids())
    analysable = set(analysable_rights_ids())

    assert "gikou2-v2.0.2" in approved
    assert "hao-2023-05-08" in approved
    assert "shinden3-2025-02-21" in approved
    assert "tanuki-dr4-2023-12-03" in approved
    assert "tanuki-wcsc36-paid" not in approved
    assert "hisui-wcsc36-hosted" not in approved
    assert "hisui-wcsc36-hosted" not in analysable
    assert approved <= analysable
    with pytest.raises(PermissionError, match="free-only policy"):
        model_rights("hisui-wcsc36-hosted").teacher_policy().require_analysis_permission()


def test_suisho11plus_is_explicit_local_only_and_never_public_by_default() -> None:
    rights_id = "suisho11plus-wcsc36-20260525-local"
    rights = model_rights(rights_id)

    assert rights.analysis == RightsDecision.LIMITED
    assert rights.output_distillation == RightsDecision.LIMITED
    assert rights.direct_weight_use == RightsDecision.NOT_APPROVED
    assert rights.output_only_meteo_publication == RightsDecision.NOT_APPROVED
    assert rights_id not in distillable_rights_ids()
    assert rights_id in locally_distillable_rights_ids()
    with pytest.raises(PermissionError, match="free-only policy"):
        rights.teacher_policy().require_training_permission()

    local_policy = rights.teacher_policy(allow_limited_local=True)
    local_policy.require_training_permission()
    assert local_policy.training_outputs_local_only
    assert not local_policy.redistribution_allowed
    assert not local_policy.requires_payment


def test_unknown_rights_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown model-rights profile"):
        model_rights("typo-or-unreviewed")
