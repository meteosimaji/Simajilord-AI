from __future__ import annotations

import pytest

from simajilord_shogi.model_rights import (
    MODEL_RIGHTS,
    DistillationScope,
    RightsDecision,
    analysable_rights_ids,
    distillable_rights_ids,
    local_only_distillable_rights_ids,
    locally_distillable_rights_ids,
    model_rights,
    not_authorized_rights_ids,
    public_distillable_rights_ids,
)

CURRENT_RIGHTS_IDS = {
    "aobannue-v1.1",
    "gikou2-v2.0.2",
    "hao-2023-05-08",
    "nagisa-v3.1",
    "shinden3-2025-02-21",
    "soujou-tsec7-paid",
    "suisho11plus-wcsc36-20260525-local",
    "suisho5",
    "tanuki-dr4-2023-12-03",
}

RETIRED_RIGHTS_IDS = {
    "aobazero-public-domain",
    "apery-public",
    "dlshogi-aoba-wcsc35",
    "dlshogi-dr2-exhi",
    "dlshogi-gct-wcsc31",
    "elmo-wcsc27",
    "gpsfish-public",
    "hisui-wcsc36-hosted",
    "kanade-wcsc35-paid",
    "suisho10-11-supporter",
    "sunfish4",
    "takewarabe-approx-v7.50-material9",
    "tanuki-wcsc36-paid",
    "yaneuraou-rezero",
    "zimetu-2026-01-26",
}


def test_rights_registry_has_no_identifier_or_casefold_name_collisions() -> None:
    assert len({record.rights_id for record in MODEL_RIGHTS}) == len(MODEL_RIGHTS)
    assert len({record.name.casefold() for record in MODEL_RIGHTS}) == len(MODEL_RIGHTS)


def test_rights_registry_contains_exactly_the_selected_nine_models() -> None:
    assert len(MODEL_RIGHTS) == 9
    assert {record.rights_id for record in MODEL_RIGHTS} == CURRENT_RIGHTS_IDS


@pytest.mark.parametrize(
    "rights_id",
    [
        "aobannue-v1.1",
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


def test_nagisa_progress_router_exception_does_not_approve_weight_copy() -> None:
    rights = model_rights("nagisa-v3.1")

    assert rights.direct_weight_use == RightsDecision.NOT_APPROVED
    assert any("progress.bin" in note and "local-only" in note for note in rights.notes)


def test_artifacts_with_explicit_terms_retain_their_distinct_decisions() -> None:
    assert model_rights("gikou2-v2.0.2").direct_weight_use == RightsDecision.CONDITIONAL_GPL
    assert model_rights("hao-2023-05-08").direct_weight_use == RightsDecision.CONDITIONAL_GPL
    assert (
        model_rights("tanuki-dr4-2023-12-03").direct_weight_use
        == RightsDecision.CONDITIONAL_GPL
    )


def test_selected_public_profiles_are_approved_for_analysis_and_distillation() -> None:
    approved = set(distillable_rights_ids())
    analysable = set(analysable_rights_ids())

    assert approved == {
        "aobannue-v1.1",
        "gikou2-v2.0.2",
        "hao-2023-05-08",
        "nagisa-v3.1",
        "shinden3-2025-02-21",
        "suisho5",
        "tanuki-dr4-2023-12-03",
    }
    assert approved <= analysable


@pytest.mark.parametrize(
    "rights_id",
    [
        "suisho11plus-wcsc36-20260525-local",
        "soujou-tsec7-paid",
    ],
)
def test_reviewed_paid_teacher_is_explicit_local_only_and_never_public_by_default(
    rights_id: str,
) -> None:
    rights = model_rights(rights_id)

    assert rights.analysis == RightsDecision.LIMITED
    assert rights.output_distillation == RightsDecision.LIMITED
    assert rights.direct_weight_use == RightsDecision.NOT_APPROVED
    assert rights.output_only_meteo_publication == RightsDecision.NOT_APPROVED
    assert rights.distillation_scope is DistillationScope.LOCAL_AUTHORIZED_ONLY
    assert rights_id not in distillable_rights_ids()
    assert rights_id not in public_distillable_rights_ids()
    assert rights_id in locally_distillable_rights_ids()
    assert rights_id in local_only_distillable_rights_ids()
    with pytest.raises(PermissionError, match="public-release-safe teacher set"):
        rights.teacher_policy().require_training_permission()

    local_policy = rights.teacher_policy(allow_limited_local=True)
    local_policy.require_training_permission()
    assert local_policy.training_outputs_local_only
    assert not local_policy.redistribution_allowed
    assert not local_policy.requires_explicit_local_authorization


def test_local_only_teacher_registry_contains_exact_reviewed_profiles() -> None:
    assert set(local_only_distillable_rights_ids()) == {
        "soujou-tsec7-paid",
        "suisho11plus-wcsc36-20260525-local",
    }


def test_rights_scopes_partition_the_registry_without_using_price_as_a_gate() -> None:
    public = set(public_distillable_rights_ids())
    local_only = set(local_only_distillable_rights_ids())
    not_authorized = set(not_authorized_rights_ids())

    assert public.isdisjoint(local_only)
    assert public.isdisjoint(not_authorized)
    assert local_only.isdisjoint(not_authorized)
    assert public | local_only | not_authorized == {
        record.rights_id for record in MODEL_RIGHTS
    }
    assert public | local_only == set(locally_distillable_rights_ids())
    assert not not_authorized


@pytest.mark.parametrize("rights_id", sorted(RETIRED_RIGHTS_IDS))
def test_retired_profiles_fail_closed(rights_id: str) -> None:
    assert rights_id not in CURRENT_RIGHTS_IDS
    assert rights_id not in analysable_rights_ids()
    assert rights_id not in locally_distillable_rights_ids()
    with pytest.raises(ValueError, match="unknown model-rights profile"):
        model_rights(rights_id)


def test_unknown_rights_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown model-rights profile"):
        model_rights("typo-or-unreviewed")
