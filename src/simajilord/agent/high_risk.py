"""Complete, requester-private presentations for high-risk agent actions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .actions import ActionClassification, action_policy
from .contracts import (
    AGENT_HIGH_RISK_CAPABILITIES,
    AgentHighRiskPresentation,
    AgentHighRiskReviewField,
)


class HighRiskPresentationError(ValueError):
    """The exact action cannot fit a complete private Discord review."""


@dataclass(frozen=True, slots=True)
class _PresentationSpec:
    action: str
    public_target: str
    target_keys: tuple[str, ...] = ()
    payload_keys: tuple[str, ...] = ()
    audience: str = "The selected Discord server audience may observe this change."
    external_transfer: str = "The listed mutation is sent to the Discord API."


_SPECS: dict[str, _PresentationSpec] = {
    "connector.destructive": _PresentationSpec(
        "Run a destructive connector action",
        "One configured external connector",
        target_keys=("connector_id", "tool", "contract_id"),
        payload_keys=("arguments",),
        audience="The destination and its readers are controlled by the selected connector.",
        external_transfer="The complete listed payload is sent to the configured connector.",
    ),
    "discord.add_thread_member": _PresentationSpec(
        "Add a thread member",
        "One Discord thread audience",
        target_keys=("thread_id", "user_id"),
        audience="The selected user may gain access to the thread and its retained history.",
    ),
    "discord.assign_role": _PresentationSpec(
        "Assign a role",
        "One server member and role",
        target_keys=("user_id", "role_id"),
    ),
    "discord.ban_member": _PresentationSpec(
        "Ban a member",
        "One server member",
        target_keys=("user_id",),
    ),
    "discord.bulk_delete_messages": _PresentationSpec(
        "Delete multiple messages",
        "One Discord channel",
        target_keys=("channel_id", "message_ids"),
        audience="Selected messages will disappear for everyone who can read the channel.",
    ),
    "discord.channel_operation": _PresentationSpec(
        "Change a channel resource",
        "One Discord channel or channel resource",
        target_keys=(
            "operation",
            "channel_id",
            "destination_channel_id",
            "guild_id",
            "sound_id",
        ),
    ),
    "discord.create_automod_rule": _PresentationSpec(
        "Create an AutoMod rule",
        "One server AutoMod policy",
        target_keys=("guild_id",),
        audience="The rule may moderate messages or members across the selected server.",
    ),
    "discord.create_channel": _PresentationSpec(
        "Create a channel",
        "One new server channel",
        target_keys=("name",),
    ),
    "discord.create_guild_resource": _PresentationSpec(
        "Create a server resource",
        "One new Discord server resource",
        target_keys=("kind", "guild_id", "channel_id", "category_id"),
    ),
    "discord.create_platform_asset": _PresentationSpec(
        "Create a platform asset",
        "One new Discord emoji, sticker, or sound",
        target_keys=("kind", "guild_id", "path"),
        external_transfer="The selected workspace file bytes are uploaded to Discord.",
    ),
    "discord.create_role": _PresentationSpec(
        "Create a role",
        "One new server role",
        target_keys=("name",),
    ),
    "discord.delete_automod_rule": _PresentationSpec(
        "Delete an AutoMod rule",
        "One server AutoMod policy",
        target_keys=("rule_id", "guild_id"),
    ),
    "discord.delete_guild_resource": _PresentationSpec(
        "Delete a server resource",
        "One Discord server resource",
        target_keys=("kind", "resource_id", "guild_id", "channel_id"),
    ),
    "discord.delete_message": _PresentationSpec(
        "Delete a message",
        "One Discord message",
        target_keys=("channel_id", "message_id"),
        audience="The selected message will disappear for everyone who can read the channel.",
    ),
    "discord.delete_platform_asset": _PresentationSpec(
        "Delete a platform asset",
        "One Discord emoji, sticker, or sound",
        target_keys=("kind", "resource_id", "guild_id"),
    ),
    "discord.kick_member": _PresentationSpec(
        "Kick a member",
        "One server member",
        target_keys=("user_id",),
    ),
    "discord.message_action": _PresentationSpec(
        "Apply a destructive message action",
        "One Discord message",
        target_keys=("action", "channel_id", "message_id", "guild_id", "emoji"),
    ),
    "discord.read_aloud_add_sources": _PresentationSpec(
        "Add read-aloud sources",
        "Server read-aloud routing",
        target_keys=("text_channel_ids", "audio_destination_id"),
        audience="Voice listeners in the selected destination may hear source-channel content.",
        external_transfer="Text may be sent to the configured speech provider and Discord voice.",
    ),
    "discord.read_aloud_announcements_set": _PresentationSpec(
        "Change read-aloud announcements",
        "Server read-aloud policy",
        audience="Voice listeners may hear the configured member events.",
        external_transfer=(
            "Announcement text may reach the configured speech provider and "
            "Discord voice."
        ),
    ),
    "discord.read_aloud_content_mode_set": _PresentationSpec(
        "Change read-aloud content mode",
        "Server read-aloud policy",
        audience="Voice listeners may hear content allowed by the selected mode.",
        external_transfer=(
            "Allowed text may reach the configured speech provider and Discord voice."
        ),
    ),
    "discord.read_aloud_dictionary_remove": _PresentationSpec(
        "Remove a read-aloud dictionary entry",
        "Server read-aloud dictionary",
        target_keys=("surface",),
        external_transfer="Future transformed text may reach the configured speech provider.",
    ),
    "discord.read_aloud_dictionary_set": _PresentationSpec(
        "Set a read-aloud dictionary entry",
        "Server read-aloud dictionary",
        target_keys=("surface",),
        external_transfer="Future transformed text may reach the configured speech provider.",
    ),
    "discord.read_aloud_disable": _PresentationSpec(
        "Disable read aloud",
        "Server read-aloud routing",
        audience="Automatic speech stops for the configured voice destination.",
        external_transfer="No new speech-provider transfer is created by disabling the route.",
    ),
    "discord.read_aloud_exclusion_set": _PresentationSpec(
        "Change a read-aloud exclusion",
        "One server member or role",
        target_keys=("target", "target_id"),
        audience="Voice listeners may hear or stop hearing content from the selected subject.",
        external_transfer=(
            "Allowed text may reach the configured speech provider and Discord voice."
        ),
    ),
    "discord.read_aloud_remove_source": _PresentationSpec(
        "Remove a read-aloud source",
        "One source channel",
        target_keys=("text_channel_id",),
        audience="Voice listeners stop receiving automatic speech from the selected source.",
        external_transfer="No new speech-provider transfer is created by removing the source.",
    ),
    "discord.read_aloud_semantics_set": _PresentationSpec(
        "Change read-aloud message semantics",
        "Server read-aloud policy",
        audience="Voice listeners may hear the newly enabled message attributes.",
        external_transfer=(
            "Allowed text may reach the configured speech provider and Discord voice."
        ),
    ),
    "discord.remove_role": _PresentationSpec(
        "Remove a role",
        "One server member and role",
        target_keys=("user_id", "role_id"),
    ),
    "discord.remove_thread_member": _PresentationSpec(
        "Remove a thread member",
        "One Discord thread audience",
        target_keys=("thread_id", "user_id"),
        audience="The selected user may lose access to the thread.",
    ),
    "discord.send_direct_message": _PresentationSpec(
        "Send a direct message",
        "One Discord user",
        target_keys=("user_id", "guild_id"),
        audience="Only the selected Discord user is the intended message audience.",
        external_transfer="The complete message body is sent to Discord for direct delivery.",
    ),
    "discord.set_bot_presence": _PresentationSpec(
        "Change the bot presence",
        "The bot's public Discord presence",
        target_keys=("guild_id",),
        audience="The bot presence may be visible to members across connected servers.",
    ),
    "discord.set_channel_overwrite": _PresentationSpec(
        "Change a channel permission overwrite",
        "One channel and member or role",
        target_keys=("channel_id", "target_kind", "target_id", "guild_id"),
        audience="The selected overwrite can expand or reduce who can access the channel.",
    ),
    "discord.set_timeout": _PresentationSpec(
        "Change a member timeout",
        "One server member",
        target_keys=("user_id",),
    ),
    "discord.unban_member": _PresentationSpec(
        "Unban a member",
        "One server member",
        target_keys=("user_id",),
    ),
    "discord.update_automod_rule": _PresentationSpec(
        "Update an AutoMod rule",
        "One server AutoMod policy",
        target_keys=("rule_id", "guild_id"),
        audience="The rule may moderate messages or members across the selected server.",
    ),
    "discord.update_channel_settings": _PresentationSpec(
        "Update channel settings",
        "One Discord channel",
        target_keys=("channel_id",),
    ),
    "discord.update_guild_resource": _PresentationSpec(
        "Update a server resource",
        "One Discord server resource",
        target_keys=(
            "kind",
            "resource_id",
            "guild_id",
            "voice_channel_id",
            "destination_channel_id",
        ),
    ),
    "discord.update_platform_asset": _PresentationSpec(
        "Update a platform asset",
        "One Discord emoji, sticker, or sound",
        target_keys=("kind", "resource_id", "guild_id"),
    ),
    "discord.update_thread": _PresentationSpec(
        "Update a thread",
        "One Discord thread",
        target_keys=("thread_id",),
        audience="Archiving or renaming changes how server members find or use the thread.",
    ),
    "system.shell": _PresentationSpec(
        "Run an isolated local command",
        "The requester's isolated task workspace",
        target_keys=("working_directory",),
        payload_keys=("argv", "timeout_seconds"),
        audience="Command output returns only to this agent task unless later shared explicitly.",
        external_transfer=(
            "A local sandboxed process receives the complete listed argv; network "
            "is denied."
        ),
    ),
}


def _validate_specs() -> None:
    missing = AGENT_HIGH_RISK_CAPABILITIES - _SPECS.keys()
    extra = _SPECS.keys() - AGENT_HIGH_RISK_CAPABILITIES
    if missing or extra:
        raise RuntimeError(
            "High-risk presentation coverage mismatch "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    for capability, spec in _SPECS.items():
        ordered_keys = (*spec.target_keys, *spec.payload_keys)
        if len(ordered_keys) != len(set(ordered_keys)):
            raise RuntimeError(
                f"High-risk presentation keys overlap for {capability}"
            )


_validate_specs()


def high_risk_presentation(
    capability: str,
    arguments: object,
) -> AgentHighRiskPresentation:
    """Present every effect argument privately, or reject without truncation."""

    spec = _SPECS.get(capability)
    if spec is None:
        raise HighRiskPresentationError(
            "No structured private presenter exists for this capability."
        )
    if not isinstance(arguments, Mapping) or not all(
        isinstance(key, str) for key in arguments
    ):
        raise HighRiskPresentationError(
            "High-risk confirmation arguments must be one object."
        )
    sanitized = {
        str(key): value
        for key, value in arguments.items()
        if key != "authorization_event_id"
    }
    remaining = dict(sanitized)
    fields: list[AgentHighRiskReviewField] = []

    _append_selected_field(fields, "Exact target", remaining, spec.target_keys)
    expected_keys = tuple(
        key for key in remaining if key.startswith("expected_")
    )
    if expected_keys:
        _append_selected_field(
            fields,
            "Expected current state",
            remaining,
            expected_keys,
        )
    else:
        fields.append(
            AgentHighRiskReviewField(
                name="Expected current state",
                value=(
                    "No compare-and-set value was supplied; live permissions and "
                    "target state are rechecked before dispatch."
                ),
            )
        )
    audit_keys = tuple(
        key for key in ("reason", "evidence_message_ids") if key in remaining
    )
    _append_selected_field(
        fields,
        "Authorization context",
        remaining,
        audit_keys,
    )
    _append_selected_field(
        fields,
        "Provider or process payload",
        remaining,
        spec.payload_keys,
    )
    if remaining:
        fields.append(
            AgentHighRiskReviewField(
                name="Requested change",
                value=_format_items(tuple(remaining.items())),
            )
        )
    else:
        fields.append(
            AgentHighRiskReviewField(
                name="Requested change",
                value=f"Apply this exact operation: {spec.action}.",
            )
        )
    fields.extend(
        (
            AgentHighRiskReviewField(
                name="Audience and visibility",
                value=spec.audience,
            ),
            AgentHighRiskReviewField(
                name="External transfer",
                value=spec.external_transfer,
            ),
            AgentHighRiskReviewField(
                name="Reversibility",
                value=_reversibility(capability),
            ),
        )
    )
    try:
        return AgentHighRiskPresentation(
            public_action=spec.action,
            public_target=spec.public_target,
            review_fields=tuple(fields),
        )
    except ValueError as exc:
        raise HighRiskPresentationError(
            "The exact details do not fit the secure private review. "
            "Use a narrower action or an operator-reviewed workflow."
        ) from exc


def _append_selected_field(
    fields: list[AgentHighRiskReviewField],
    name: str,
    remaining: dict[str, object],
    keys: tuple[str, ...],
) -> None:
    selected = tuple(
        (key, remaining.pop(key)) for key in keys if key in remaining
    )
    if selected:
        fields.append(
            AgentHighRiskReviewField(name=name, value=_format_items(selected))
        )


def _format_items(items: tuple[tuple[str, object], ...]) -> str:
    lines: list[str] = []
    for key, value in items:
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise HighRiskPresentationError(
                "High-risk confirmation values must be JSON values."
            ) from exc
        lines.append(f"{key}: {encoded}")
    formatted = "\n".join(lines)
    if not formatted or len(formatted) > 950:
        raise HighRiskPresentationError(
            "The exact details do not fit the secure private review. "
            "Use a narrower action or an operator-reviewed workflow."
        )
    return formatted


def _reversibility(capability: str) -> str:
    classification = action_policy(capability).classification
    if classification is ActionClassification.FULLY_REVERSIBLE:
        return "An exact, idempotent Undo is available after a tracked success."
    if classification is ActionClassification.COMPENSATING:
        return "A compensating Undo is available, but external history may remain."
    return "No automatic Undo is available after dispatch."
