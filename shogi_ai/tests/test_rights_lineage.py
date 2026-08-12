from __future__ import annotations

import json
from typing import cast

import pytest

from simajilord_shogi.rights_lineage import (
    RIGHTS_RESTRICTION_SUMMARY_SCHEMA,
    expected_lineage_rights_summary,
    legacy_ancestor_restriction_id,
    merge_rights_restriction_summaries,
    summarize_teacher_sidecar,
    validate_rights_restriction_summary,
)


def _nagisa_provenance() -> dict[str, object]:
    return {
        "engine_working_directory": "/" + "Users" + "/private/paid-engine",
        "source": "https://example.invalid/private-download",
        "engine_sha256": "e" * 64,
        "rights": {
            "rights_id": "nagisa-v3.1",
            "output_only_meteo_publication": "allowed",
            "sources": ["https://example.invalid/must-not-copy"],
        },
    }


def _suisho_provenance() -> dict[str, object]:
    return {
        "rights_mode": "limited_local",
        "publication_allowed": False,
        "public_release_gate": "blocked_pending_rights_holder_permission",
        "local_only_root": "/" + "private" + "/tmp/supporter-artifact",
        "startup_provenance_sha256": "f" * 64,
        "rights": {
            "rights_id": "suisho11plus-wcsc36-20260525-local",
            "output_only_meteo_publication": "not_approved",
            "sources": ["https://example.invalid/paid-supporter-post"],
        },
    }


def test_direct_nagisa_sidecar_is_publication_allowed_and_privacy_safe() -> None:
    sidecar_sha256 = "a" * 64
    summary = summarize_teacher_sidecar(_nagisa_provenance(), sidecar_sha256=sidecar_sha256)

    assert summary["schema"] == RIGHTS_RESTRICTION_SUMMARY_SCHEMA
    assert summary["publication_allowed"] is True
    assert summary["restriction_ids"] == []
    assert summary["sidecar_sha256s"] == [sidecar_sha256]
    assert summary["sources"] == [
        {
            "rights_id": "nagisa-v3.1",
            "decision": "allowed",
            "publication_allowed": True,
            "sidecar_sha256s": [sidecar_sha256],
            "restriction_ids": [],
        }
    ]
    encoded = json.dumps(summary, sort_keys=True)
    for private_token in (
        "/" + "Users",
        "/" + "private" + "/tmp",
        "example.invalid",
        "paid-engine",
        "e" * 64,
    ):
        assert private_token not in encoded


def test_ensemble_resolves_generic_marker_to_nagisa_and_restricted_suisho() -> None:
    sidecar_sha256 = "b" * 64
    summary = summarize_teacher_sidecar(
        {
            "schema": "meteo-teacher-ensemble-v1",
            "inputs": [
                {"label": "nagisa", "provenance": _nagisa_provenance()},
                {"label": "suisho", "provenance": _suisho_provenance()},
            ],
        },
        sidecar_sha256=sidecar_sha256,
    )
    merged = merge_rights_restriction_summaries(
        [summary], teacher_sources=["meteo-teacher-ensemble"]
    )

    sources = {
        cast(str, source["rights_id"]): source
        for source in cast(list[dict[str, object]], merged["sources"])
    }
    assert set(sources) == {
        "nagisa-v3.1",
        "suisho11plus-wcsc36-20260525-local",
    }
    assert sources["nagisa-v3.1"]["publication_allowed"] is True
    assert sources["suisho11plus-wcsc36-20260525-local"]["decision"] == ("not_approved")
    assert sources["suisho11plus-wcsc36-20260525-local"]["publication_allowed"] is False
    assert merged["publication_allowed"] is False
    assert len(cast(list[str], merged["restriction_ids"])) == 1
    assert "meteo-teacher-ensemble" not in json.dumps(merged)


def test_exact_resume_merge_preserves_parent_restrictions_and_sidecar_binding() -> None:
    parent = summarize_teacher_sidecar(_suisho_provenance(), sidecar_sha256="c" * 64)
    current = summarize_teacher_sidecar(_nagisa_provenance(), sidecar_sha256="d" * 64)

    merged = merge_rights_restriction_summaries([parent, current], teacher_sources=["nagisa-v3.1"])

    assert merged["publication_allowed"] is False
    assert merged["sidecar_sha256s"] == ["c" * 64, "d" * 64]
    assert parent["restriction_ids"] == merged["restriction_ids"]


def test_bounded_parent_identity_preserves_rights_without_parent_lineage() -> None:
    parent = summarize_teacher_sidecar(_suisho_provenance(), sidecar_sha256="c" * 64)
    resolved = expected_lineage_rights_summary(
        {
            "schema": "meteo-training-lineage-v2",
            "teacher_sources": [],
            "training_inputs": [],
            "parent_checkpoint": {"rights_restriction_summary": parent},
        }
    )

    assert resolved == parent


def test_legacy_parent_without_summary_gets_stable_fail_closed_marker() -> None:
    summary = merge_rights_restriction_summaries([], legacy_parent_missing_summary=True)

    assert summary["publication_allowed"] is False
    assert summary["inherited_restriction_ids"] == [legacy_ancestor_restriction_id()]
    assert summary["restriction_ids"] == [legacy_ancestor_restriction_id()]


def test_summary_rejects_removed_or_fabricated_restriction_ids() -> None:
    summary = summarize_teacher_sidecar(_suisho_provenance(), sidecar_sha256="f" * 64)
    tampered = json.loads(json.dumps(summary))
    tampered["restriction_ids"] = []

    with pytest.raises(ValueError, match=r"canonical|restriction_ids"):
        validate_rights_restriction_summary(tampered)


def test_unattributed_restrictive_sidecar_branch_fails_closed() -> None:
    summary = summarize_teacher_sidecar(
        {
            "teacher": _nagisa_provenance(),
            "separate_local_record": {"publication_allowed": False},
        },
        sidecar_sha256="1" * 64,
    )

    assert summary["publication_allowed"] is False
    assert len(cast(list[str], summary["unattributed_restriction_ids"])) == 1
