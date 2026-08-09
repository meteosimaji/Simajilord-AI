"""Privacy-safe teacher-rights summaries carried by Meteo checkpoints.

Training provenance sidecars intentionally remain local because they can contain
engine paths, supporter-only artifact identities, and other operational details.
This module extracts only the reviewed rights identifier, the publication
decision, an effective publication boolean, the enclosing sidecar digest, and
deterministic restriction identifiers.  The resulting summary can therefore
survive an inference-only export without copying the private sidecar itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .model_rights import RightsDecision, model_rights

RIGHTS_RESTRICTION_SUMMARY_SCHEMA = "meteo-teacher-rights-restriction-summary-v1"
RIGHTS_RESTRICTION_ID_SCHEMA = "meteo-teacher-rights-restriction-id-v1"

# These are generated target labels, not external rights-registry identifiers.
# They are permitted only when an adjacent sidecar resolves them to exact
# reviewed teacher rights IDs.
DERIVED_TEACHER_SOURCE_IDS = frozenset({"meteo-teacher-ensemble"})

_SAFE_RIGHTS_ID_RE = re.compile(r"[a-z0-9][a-z0-9._+-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RESTRICTED_MODES = frozenset(
    {
        "blocked_pending_rights_holder_permission",
        "limited",
        "limited-local",
        "limited_local",
        "not_approved",
        "user_authorized_local_analysis_only",
    }
)
_LEGACY_ANCESTOR_CODE = "legacy-teacher-lineage-summary-missing"
_UNATTRIBUTED_SIDECAR_CODE = "unattributed-sidecar-publication-block"


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _require_rights_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_RIGHTS_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a privacy-safe rights identifier")
    return value


def _registry_decision(rights_id: str) -> str:
    try:
        return model_rights(rights_id).output_only_meteo_publication.value
    except ValueError:
        return "unreviewed"


def teacher_rights_restriction_id(
    *,
    rights_id: str,
    decision: str,
    publication_allowed: bool,
) -> str:
    """Return the stable receipt identifier for one effective teacher decision."""

    _require_rights_id(rights_id, label="teacher rights_id")
    if not isinstance(decision, str) or not decision:
        raise ValueError("teacher rights decision must be a non-empty string")
    return _canonical_sha256(
        {
            "schema": RIGHTS_RESTRICTION_ID_SCHEMA,
            "code": "teacher-publication-restricted",
            "rights_id": rights_id,
            "decision": decision,
            "publication_allowed": publication_allowed,
        }
    )


def legacy_ancestor_restriction_id() -> str:
    """Stable fail-closed marker for a pre-summary teacher checkpoint ancestor."""

    return _canonical_sha256(
        {
            "schema": RIGHTS_RESTRICTION_ID_SCHEMA,
            "code": _LEGACY_ANCESTOR_CODE,
        }
    )


def _unattributed_sidecar_restriction_id(sidecar_sha256: str) -> str:
    return _canonical_sha256(
        {
            "schema": RIGHTS_RESTRICTION_ID_SCHEMA,
            "code": _UNATTRIBUTED_SIDECAR_CODE,
            "sidecar_sha256": sidecar_sha256,
        }
    )


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, Any], value)


def _local_publication_block(mapping: Mapping[str, object]) -> bool:
    if mapping.get("publication_allowed") is False:
        return True
    for key, value in mapping.items():
        normalized = key.casefold()
        if "local_only" in normalized and value is True:
            return True
        if (
            normalized in {"rights_mode", "public_release_gate"}
            and isinstance(value, str)
            and value.casefold() in _RESTRICTED_MODES
        ):
            return True
    return False


def _source_row(
    *,
    rights_id: str,
    publication_allowed: bool,
    sidecar_sha256s: Sequence[str],
) -> dict[str, object]:
    rights_id = _require_rights_id(rights_id, label="teacher rights_id")
    decision = _registry_decision(rights_id)
    effective_allowed = publication_allowed and decision == RightsDecision.ALLOWED.value
    restriction_ids = (
        []
        if effective_allowed
        else [
            teacher_rights_restriction_id(
                rights_id=rights_id,
                decision=decision,
                publication_allowed=effective_allowed,
            )
        ]
    )
    return {
        "rights_id": rights_id,
        "decision": decision,
        "publication_allowed": effective_allowed,
        "sidecar_sha256s": sorted(
            {_require_sha256(item, label="teacher sidecar SHA-256") for item in sidecar_sha256s}
        ),
        "restriction_ids": restriction_ids,
    }


def _assemble_summary(
    *,
    rows: Mapping[str, dict[str, object]],
    sidecar_sha256s: Sequence[str],
    inherited_restriction_ids: Sequence[str] = (),
    unattributed_restriction_ids: Sequence[str] = (),
) -> dict[str, object]:
    ordered_rows = [rows[rights_id] for rights_id in sorted(rows)]
    inherited = sorted(
        {
            _require_sha256(item, label="inherited rights restriction ID")
            for item in inherited_restriction_ids
        }
    )
    unattributed = sorted(
        {
            _require_sha256(item, label="unattributed rights restriction ID")
            for item in unattributed_restriction_ids
        }
    )
    restriction_ids = sorted(
        {
            *inherited,
            *unattributed,
            *(
                restriction_id
                for row in ordered_rows
                for restriction_id in cast(list[str], row["restriction_ids"])
            ),
        }
    )
    return {
        "schema": RIGHTS_RESTRICTION_SUMMARY_SCHEMA,
        "publication_allowed": not restriction_ids,
        "sidecar_sha256s": sorted(
            {
                _require_sha256(item, label="teacher sidecar SHA-256")
                for item in sidecar_sha256s
            }
        ),
        "sources": ordered_rows,
        "inherited_restriction_ids": inherited,
        "unattributed_restriction_ids": unattributed,
        "restriction_ids": restriction_ids,
    }


def summarize_teacher_sidecar(
    payload: Mapping[str, object],
    *,
    sidecar_sha256: str,
) -> dict[str, object]:
    """Extract a privacy-safe summary from a direct or ensemble sidecar.

    Rights records may appear directly under ``rights`` or inside an ensemble's
    ``inputs[].provenance``.  A restrictive wrapper applies to all descendants.
    No source URL, filesystem path, artifact identity, or unrelated digest is
    copied into the returned object.
    """

    sidecar_sha256 = _require_sha256(sidecar_sha256, label="teacher sidecar SHA-256")
    observations: dict[str, bool] = {}
    unattributed_block = False

    def observe(rights_value: Mapping[str, object], *, blocked: bool) -> None:
        rights_id = _require_rights_id(
            rights_value.get("rights_id"), label="sidecar teacher rights_id"
        )
        decision = _registry_decision(rights_id)
        reported = rights_value.get("output_only_meteo_publication")
        reported_matches = reported is None or reported == decision
        allowed = (
            not blocked
            and decision == RightsDecision.ALLOWED.value
            and reported_matches
        )
        observations[rights_id] = observations.get(rights_id, True) and allowed

    def visit(value: object, *, inherited_blocked: bool) -> bool:
        nonlocal unattributed_block
        if isinstance(value, dict):
            mapping = _mapping(value, label="teacher sidecar object")
            local_blocked = _local_publication_block(mapping)
            blocked = inherited_blocked or local_blocked
            found_rights = False
            rights_value = mapping.get("rights")
            if isinstance(rights_value, dict) and "rights_id" in rights_value:
                observe(
                    _mapping(rights_value, label="sidecar rights record"),
                    blocked=blocked,
                )
                found_rights = True
            if "rights_id" in mapping and "output_only_meteo_publication" in mapping:
                observe(mapping, blocked=blocked)
                found_rights = True
            for key in sorted(mapping):
                if key == "rights" and isinstance(rights_value, dict):
                    continue
                found_rights = (
                    visit(mapping[key], inherited_blocked=blocked) or found_rights
                )
            if local_blocked and not found_rights:
                unattributed_block = True
            return found_rights
        elif isinstance(value, list):
            found_rights = False
            for item in value:
                found_rights = (
                    visit(item, inherited_blocked=inherited_blocked) or found_rights
                )
            return found_rights
        return False

    visit(dict(payload), inherited_blocked=False)
    rows = {
        rights_id: _source_row(
            rights_id=rights_id,
            publication_allowed=allowed,
            sidecar_sha256s=(sidecar_sha256,),
        )
        for rights_id, allowed in observations.items()
    }
    unattributed = (
        [_unattributed_sidecar_restriction_id(sidecar_sha256)]
        if unattributed_block
        else []
    )
    summary = _assemble_summary(
        rows=rows,
        sidecar_sha256s=(sidecar_sha256,),
        unattributed_restriction_ids=unattributed,
    )
    return validate_rights_restriction_summary(summary)


def validate_rights_restriction_summary(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate and canonicalize a checkpoint-safe restriction summary."""

    summary = _mapping(dict(value), label="teacher rights restriction summary")
    expected_keys = {
        "schema",
        "publication_allowed",
        "sidecar_sha256s",
        "sources",
        "inherited_restriction_ids",
        "unattributed_restriction_ids",
        "restriction_ids",
    }
    if set(summary) != expected_keys:
        raise ValueError("teacher rights restriction summary fields do not match its schema")
    if summary["schema"] != RIGHTS_RESTRICTION_SUMMARY_SCHEMA:
        raise ValueError("unsupported teacher rights restriction summary schema")
    raw_sidecars = summary["sidecar_sha256s"]
    raw_sources = summary["sources"]
    raw_inherited = summary["inherited_restriction_ids"]
    raw_unattributed = summary["unattributed_restriction_ids"]
    raw_restrictions = summary["restriction_ids"]
    if not isinstance(raw_sidecars, list):
        raise ValueError("teacher rights sidecar_sha256s must be a list")
    if not isinstance(raw_sources, list):
        raise ValueError("teacher rights sources must be a list")
    if not isinstance(raw_inherited, list) or not isinstance(raw_unattributed, list):
        raise ValueError("teacher rights inherited/unattributed restrictions must be lists")
    if not isinstance(raw_restrictions, list):
        raise ValueError("teacher rights restriction_ids must be a list")

    rows: dict[str, dict[str, object]] = {}
    for index, raw_source in enumerate(raw_sources):
        source = _mapping(raw_source, label=f"teacher rights source {index}")
        if set(source) != {
            "rights_id",
            "decision",
            "publication_allowed",
            "sidecar_sha256s",
            "restriction_ids",
        }:
            raise ValueError("teacher rights source fields do not match its schema")
        rights_id = _require_rights_id(
            source["rights_id"], label="teacher rights source rights_id"
        )
        if rights_id in rows:
            raise ValueError("duplicate teacher rights source")
        decision = source["decision"]
        if decision != _registry_decision(rights_id):
            raise ValueError("teacher rights source decision differs from the reviewed registry")
        publication_allowed = source["publication_allowed"]
        if not isinstance(publication_allowed, bool):
            raise ValueError("teacher rights source publication_allowed must be boolean")
        source_sidecars = source["sidecar_sha256s"]
        source_restrictions = source["restriction_ids"]
        if not isinstance(source_sidecars, list) or not isinstance(source_restrictions, list):
            raise ValueError("teacher rights source hashes and restrictions must be lists")
        expected = _source_row(
            rights_id=rights_id,
            publication_allowed=publication_allowed,
            sidecar_sha256s=source_sidecars,
        )
        if source != expected:
            raise ValueError("teacher rights source is not canonical or has invalid restrictions")
        rows[rights_id] = expected

    canonical = _assemble_summary(
        rows=rows,
        sidecar_sha256s=raw_sidecars,
        inherited_restriction_ids=raw_inherited,
        unattributed_restriction_ids=raw_unattributed,
    )
    if summary != canonical:
        raise ValueError("teacher rights restriction summary is not canonical")
    if not isinstance(summary["publication_allowed"], bool):
        raise ValueError("teacher rights publication_allowed must be boolean")
    if raw_restrictions != canonical["restriction_ids"]:
        raise ValueError("teacher rights restriction_ids do not match source decisions")
    return canonical


def merge_rights_restriction_summaries(
    summaries: Sequence[Mapping[str, object]],
    *,
    teacher_sources: Sequence[str] = (),
    legacy_parent_missing_summary: bool = False,
) -> dict[str, object]:
    """Merge current sidecars, exact-resume ancestors, and direct teacher IDs."""

    rows: dict[str, dict[str, object]] = {}
    sidecars: set[str] = set()
    inherited: set[str] = set()
    unattributed: set[str] = set()
    current_sidecar_sources: set[str] = set()

    for raw_summary in summaries:
        summary = validate_rights_restriction_summary(raw_summary)
        sidecars.update(cast(list[str], summary["sidecar_sha256s"]))
        inherited.update(cast(list[str], summary["inherited_restriction_ids"]))
        unattributed.update(cast(list[str], summary["unattributed_restriction_ids"]))
        for raw_row in cast(list[dict[str, object]], summary["sources"]):
            rights_id = cast(str, raw_row["rights_id"])
            current_sidecar_sources.add(rights_id)
            existing = rows.get(rights_id)
            if existing is None:
                rows[rights_id] = dict(raw_row)
                continue
            merged_sidecars = {
                *cast(list[str], existing["sidecar_sha256s"]),
                *cast(list[str], raw_row["sidecar_sha256s"]),
            }
            rows[rights_id] = _source_row(
                rights_id=rights_id,
                publication_allowed=(
                    cast(bool, existing["publication_allowed"])
                    and cast(bool, raw_row["publication_allowed"])
                ),
                sidecar_sha256s=tuple(merged_sidecars),
            )

    for source in sorted(set(teacher_sources)):
        source = _require_rights_id(source, label="training teacher source")
        if source in DERIVED_TEACHER_SOURCE_IDS and current_sidecar_sources:
            continue
        existing = rows.get(source)
        direct = _source_row(
            rights_id=source,
            publication_allowed=_registry_decision(source) == RightsDecision.ALLOWED.value,
            sidecar_sha256s=(),
        )
        if existing is None:
            rows[source] = direct
        else:
            rows[source] = _source_row(
                rights_id=source,
                publication_allowed=(
                    cast(bool, existing["publication_allowed"])
                    and cast(bool, direct["publication_allowed"])
                ),
                sidecar_sha256s=cast(list[str], existing["sidecar_sha256s"]),
            )

    if legacy_parent_missing_summary:
        inherited.add(legacy_ancestor_restriction_id())
    return validate_rights_restriction_summary(
        _assemble_summary(
            rows=rows,
            sidecar_sha256s=tuple(sidecars),
            inherited_restriction_ids=tuple(inherited),
            unattributed_restriction_ids=tuple(unattributed),
        )
    )


def lineage_has_teacher_evidence(lineage: Mapping[str, object]) -> bool:
    """Whether a training lineage must carry a reviewed rights summary."""

    if lineage.get("schema") != "meteo-training-lineage-v2":
        return False
    teacher_sources = lineage.get("teacher_sources")
    if isinstance(teacher_sources, list) and bool(teacher_sources):
        return True
    training_inputs = lineage.get("training_inputs")
    if not isinstance(training_inputs, list):
        return False
    for item in training_inputs:
        if not isinstance(item, dict):
            continue
        sidecars = item.get("lineage_sidecars")
        if isinstance(sidecars, list) and bool(sidecars):
            return True
    return False


def expected_lineage_rights_summary(
    lineage: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the canonical summary required by a v2 training lineage.

    Every adjacent sidecar summary is bound to the digest in its provenance
    record.  A summarized parent is merged verbatim; a teacher-bearing legacy
    parent without the v1 summary contributes a stable fail-closed marker.
    """

    if lineage.get("schema") != "meteo-training-lineage-v2":
        raise ValueError("teacher rights summaries apply to training-lineage v2")
    raw_teacher_sources = lineage.get("teacher_sources", [])
    if not isinstance(raw_teacher_sources, list) or not all(
        isinstance(item, str) for item in raw_teacher_sources
    ):
        raise ValueError("training lineage teacher_sources must be a list of strings")

    current_sidecars: list[dict[str, object]] = []
    raw_inputs = lineage.get("training_inputs", [])
    if not isinstance(raw_inputs, list):
        raise ValueError("training lineage training_inputs must be a list")
    for input_index, raw_input in enumerate(raw_inputs):
        training_input = _mapping(
            raw_input, label=f"training lineage input {input_index}"
        )
        raw_sidecars = training_input.get("lineage_sidecars", [])
        if not isinstance(raw_sidecars, list):
            raise ValueError("training lineage sidecars must be a list")
        for sidecar_index, raw_sidecar in enumerate(raw_sidecars):
            sidecar = _mapping(
                raw_sidecar,
                label=f"training lineage sidecar {input_index}:{sidecar_index}",
            )
            sidecar_sha256 = _require_sha256(
                sidecar.get("sha256"), label="training lineage sidecar SHA-256"
            )
            raw_summary = sidecar.get("rights_restriction_summary")
            if not isinstance(raw_summary, dict):
                raise ValueError(
                    "teacher-bearing lineage sidecar lacks a rights restriction summary"
                )
            summary = validate_rights_restriction_summary(raw_summary)
            if summary["sidecar_sha256s"] != [sidecar_sha256]:
                raise ValueError(
                    "lineage sidecar rights summary is not bound to its exact SHA-256"
                )
            current_sidecars.append(summary)

    current = merge_rights_restriction_summaries(
        current_sidecars,
        teacher_sources=cast(list[str], raw_teacher_sources),
    )
    parent_lineage: dict[str, Any] | None = None
    raw_parent = lineage.get("parent_checkpoint")
    if isinstance(raw_parent, dict):
        parent = _mapping(raw_parent, label="training lineage parent checkpoint")
        raw_parent_lineage = parent.get("lineage")
        if isinstance(raw_parent_lineage, dict):
            parent_lineage = _mapping(
                raw_parent_lineage, label="parent checkpoint lineage"
            )

    if parent_lineage is None:
        return current
    raw_parent_summary = parent_lineage.get("rights_restriction_summary")
    if isinstance(raw_parent_summary, dict):
        parent_summary = validate_rights_restriction_summary(raw_parent_summary)
        return merge_rights_restriction_summaries([parent_summary, current])
    return merge_rights_restriction_summaries(
        [current],
        legacy_parent_missing_summary=lineage_has_teacher_evidence(parent_lineage),
    )
