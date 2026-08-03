"""Actor/task file isolation and source-label construction for agent tools."""

from __future__ import annotations

import hashlib

from simajilord.core import DisclosureObservation, InvocationContext
from simajilord.core.errors import UserError
from simajilord.services.files import (
    WorkspaceFileProvenance,
    WorkspaceVisibility,
)


def file_workspace_id(context: InvocationContext) -> str:
    """Return the configured file authority without changing the guild authority."""

    guild_id = context.workspace_id
    if guild_id is None:
        raise UserError("files.workspace_required")
    if context.file_workspace_mode == "guild_shared":
        return guild_id
    if context.file_workspace_mode not in {"actor", "actor_task"}:
        raise UserError("files.workspace_mode_invalid")
    components = ["v2", guild_id, context.actor_id]
    if context.file_workspace_mode == "actor_task":
        task_key = context.agent_task_id or f"request:{context.request_id}"
        components.append(task_key)
    digest = hashlib.sha256("\x1f".join(components).encode("utf-8")).hexdigest()
    return f"agent-files:{context.file_workspace_mode}:{digest}"


def file_provenance(
    context: InvocationContext,
    *,
    origin_guild_id: str | None = None,
    origin_channel_id: str | None = None,
    origin_message_id: str | None = None,
    origin_visibility: WorkspaceVisibility | None = None,
) -> WorkspaceFileProvenance:
    """Conservatively label output with every source read in the active turn."""

    observations = tuple(dict.fromkeys(context.disclosure_observations))
    resources = tuple(
        (
            item.source_workspace_id,
            item.source_resource_id,
            item.visibility,
        )
        for item in observations
    )
    effective_guild_id = origin_guild_id or context.workspace_id
    effective_channel_id = origin_channel_id or context.origin_resource_id
    effective_message_id = origin_message_id or context.active_message_id
    matching_origin = next(
        (
            item
            for item in reversed(observations)
            if item.source_workspace_id == effective_guild_id
            and item.source_resource_id == effective_channel_id
        ),
        None,
    )
    effective_visibility = origin_visibility or (
        matching_origin.visibility if matching_origin is not None else "actor_private"
    )
    if (
        effective_guild_id is not None
        and effective_channel_id is not None
        and effective_visibility != "actor_private"
    ):
        resources = tuple(
            dict.fromkeys(
                (
                    *resources,
                    (
                        effective_guild_id,
                        effective_channel_id,
                        effective_visibility,
                    ),
                )
            )
        )
    resources = tuple(dict.fromkeys(resources))
    sources_truncated = len(resources) > 32
    resources = resources[:32]
    sensitivity = _combined_sensitivity(
        (*tuple(item.visibility for item in observations), effective_visibility),
        fallback=effective_visibility,
    )
    if sources_truncated:
        sensitivity = "uncertain"
    return WorkspaceFileProvenance(
        owner_actor_ids=(context.actor_id,),
        origin_guild_id=effective_guild_id,
        origin_channel_id=effective_channel_id,
        origin_message_id=effective_message_id,
        origin_visibility=effective_visibility,
        created_task_id=context.agent_task_id,
        sensitivity=sensitivity,
        source_resources=resources,
        sources_truncated=sources_truncated,
    )


def provenance_observations(
    provenance: WorkspaceFileProvenance | None,
) -> tuple[DisclosureObservation, ...]:
    """Restore Discord source labels when a later tool reads a workspace file."""

    if provenance is None or provenance.declassified_at is not None:
        return ()
    observations: list[DisclosureObservation] = []
    for workspace_id, resource_id, visibility in provenance.source_resources:
        observations.append(
            DisclosureObservation(
                source_workspace_id=workspace_id,
                source_resource_id=resource_id,
                visibility=visibility,
                relation_to_origin="uncertain",
            )
        )
    if (
        not observations
        and provenance.origin_guild_id is not None
        and provenance.origin_channel_id is not None
        and provenance.origin_visibility != "actor_private"
    ):
        observations.append(
            DisclosureObservation(
                source_workspace_id=provenance.origin_guild_id,
                source_resource_id=provenance.origin_channel_id,
                visibility=provenance.origin_visibility,
                relation_to_origin="uncertain",
            )
        )
    return tuple(dict.fromkeys(observations))


def _combined_sensitivity(
    visibilities: tuple[str, ...],
    *,
    fallback: WorkspaceVisibility,
) -> WorkspaceVisibility:
    values = set(visibilities)
    if "uncertain" in values:
        return "uncertain"
    if "restricted" in values:
        return "restricted"
    if "actor_private" in values:
        return "actor_private"
    if "guild_public" in values:
        return "guild_public"
    return fallback
