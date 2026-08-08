"""Bounded action receipts and restart-safe undo for agent writes."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from simajilord.core import (
    AgentPrincipalKind,
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import CapabilityError, UserError

log = logging.getLogger(__name__)

ACTION_UNDO_ANY_GRANT = "action_undo_any"
_MAX_UNDO_ARGUMENT_CHARACTERS = 4_096
_MAX_TARGET_IDS = 20
_MAX_TARGET_ID_CHARACTERS = 128
_MAX_AUTHORIZATION_REFERENCE_CHARACTERS = 200
_MAX_EXTERNAL_EFFECT_SUMMARY_CHARACTERS = 512


class ActionClassification(StrEnum):
    """Whether an action can be restored without retaining a large snapshot."""

    FULLY_REVERSIBLE = "fully_reversible"
    COMPENSATING = "compensating"
    NON_UNDOABLE = "non_undoable"


class ActionStatus(StrEnum):
    """Durable action state used to make repeated undo idempotent."""

    SUCCEEDED = "succeeded"
    UNDOING = "undoing"
    UNDONE = "undone"


class ExternalEffectStatus(StrEnum):
    """Durable lifecycle for a write crossing the local process boundary."""

    PLANNED = "planned"
    DISPATCHED = "dispatched"
    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"
    RECONCILED = "reconciled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    """Small model-facing proof that a write succeeded."""

    action_id: str | None
    capability: str
    status: str
    tracked: bool
    undo_available: bool
    undo_capability: str | None
    classification: ActionClassification
    principal_kind: AgentPrincipalKind
    executor_principal_id: str | None = None
    delegator_principal_id: str | None = None
    trigger_actor_ids: tuple[str, ...] = ()
    requester_principal_id: str | None = None
    policy_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActionUndoRequest:
    """Undo one action by ID, or the most recent undoable action when omitted."""

    action_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActionUndoResponse:
    """Stable result for first and repeated undo requests."""

    action_id: str
    capability: str
    status: str
    undo_capability: str
    undo_action_id: str


UndoArgumentFactory = Callable[[object, object], Mapping[str, object] | None]


@dataclass(frozen=True, slots=True)
class ActionPolicy:
    """Static inverse definition; code, rather than SQLite, owns undo behavior."""

    capability: str
    classification: ActionClassification
    undo_capability: str | None = None
    undo_arguments: UndoArgumentFactory | None = None

    def __post_init__(self) -> None:
        available = self.undo_capability is not None or self.undo_arguments is not None
        if available and (
            self.undo_capability is None or self.undo_arguments is None
        ):
            raise ValueError("undo capability and argument factory must be defined together")
        if self.classification is ActionClassification.NON_UNDOABLE and available:
            raise ValueError("non-undoable actions cannot define an inverse")
        if self.classification is not ActionClassification.NON_UNDOABLE and not available:
            raise ValueError("reversible actions must define an inverse")


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Bounded SQLite projection; it deliberately contains no message or file body."""

    action_id: str
    capability: str
    actor_id: str
    workspace_id: str | None
    transport: str
    request_id: str
    principal_kind: AgentPrincipalKind | None
    executor_principal_id: str | None
    delegator_principal_id: str | None
    trigger_actor_ids: tuple[str, ...]
    requester_principal_id: str | None
    policy_id: str | None
    target_ids: tuple[tuple[str, str], ...]
    status: ActionStatus
    classification: ActionClassification
    undo_capability: str | None
    undo_arguments: Mapping[str, object] | None
    host_delivery: bool
    created_at: datetime
    expires_at: datetime
    undone_at: datetime | None = None
    undo_action_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalEffectRecord:
    """Body-free replay barrier for one provider tool call."""

    effect_id: str
    capability: str
    actor_id: str
    workspace_id: str | None
    transport: str
    request_id: str
    provider_thread_id: str | None
    provider_turn_id: str | None
    tool_call_id: str
    arguments_fingerprint: str
    target_ids: tuple[tuple[str, str], ...]
    authorization_reference: str | None
    summary: str
    status: ExternalEffectStatus
    action_id: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class _PostedMessageRequest:
    """Body-free projection of one Discord post created outside the tool catalog."""

    channel_id: str


@dataclass(frozen=True, slots=True)
class _PostedMessageResponse:
    """Only the IDs needed to delete one bot-authored Discord post."""

    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class _PostedMessagesResponse:
    """A bounded scalar projection for one multi-post Discord response."""

    channel_id: str
    message_ids: str


class ActionJournal(Protocol):
    """Narrow event-journal port used without coupling to one implementation."""

    async def append(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None = None,
        workspace_id: str | None = None,
        transport: str | None = None,
        request_id: str | None = None,
    ) -> int: ...


def _same_reaction(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    return _selected_values(request, ("channel_id", "message_id", "emoji"))


def _created_message(_request: object, response: object) -> Mapping[str, object]:
    selected = dict(_selected_values(response, ("channel_id", "message_id")))
    guild_id = getattr(response, "guild_id", None)
    if isinstance(guild_id, str):
        selected["guild_id"] = guild_id
    return selected


def _created_messages(_request: object, response: object) -> Mapping[str, object]:
    selected = dict(_selected_values(response, ("channel_id", "message_ids")))
    guild_id = getattr(response, "guild_id", None)
    if isinstance(guild_id, str):
        selected["guild_id"] = guild_id
    return selected


def _created_timer(_request: object, response: object) -> Mapping[str, object]:
    timer = getattr(response, "timer", None)
    return _selected_values(timer, ("timer_id",))


def _cancelled_timer(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    return _selected_values(request, ("timer_id",))


def _message_pin_change(
    expected_pinned: bool,
) -> UndoArgumentFactory:
    def capture(
        request: object,
        response: object,
    ) -> Mapping[str, object] | None:
        if getattr(response, "changed", True) is False:
            return None
        selected = dict(_selected_values(request, ("channel_id", "message_id")))
        selected["expected_pinned"] = expected_pinned
        return selected

    return capture


def _thread_update(
    _request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    thread_id = getattr(response, "thread_id", None)
    old_name = getattr(response, "old_name", None)
    old_archived = getattr(response, "old_archived", None)
    current_name = getattr(response, "name", None)
    current_archived = getattr(response, "archived", None)
    undo_fingerprint = getattr(response, "undo_fingerprint", None)
    if (
        not isinstance(thread_id, str)
        or not isinstance(old_name, str)
        or not isinstance(old_archived, bool)
        or not isinstance(current_name, str)
        or not isinstance(current_archived, bool)
        or not isinstance(undo_fingerprint, str)
    ):
        return {}
    return {
        "thread_id": thread_id,
        "name": old_name,
        "archived": old_archived,
        "expected_name": current_name,
        "expected_archived": current_archived,
        "expected_undo_fingerprint": undo_fingerprint,
    }


def _created_thread(
    _request: object,
    response: object,
) -> Mapping[str, object]:
    selected = dict(_selected_values(response, ("thread_id",)))
    current_name = getattr(response, "name", None)
    current_archived = getattr(response, "archived", None)
    undo_fingerprint = getattr(response, "undo_fingerprint", None)
    if (
        not isinstance(current_name, str)
        or not isinstance(current_archived, bool)
        or not isinstance(undo_fingerprint, str)
    ):
        return {}
    selected["archived"] = True
    selected["expected_name"] = current_name
    selected["expected_archived"] = current_archived
    selected["expected_undo_fingerprint"] = undo_fingerprint
    return selected


def _thread_member(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    expected_present = getattr(response, "present", None)
    if not isinstance(expected_present, bool):
        return {}
    selected = dict(_selected_values(request, ("thread_id", "user_id")))
    selected["expected_present"] = expected_present
    return selected


def _published_file_copy(
    _request: object,
    response: object,
) -> Mapping[str, object]:
    return _selected_values(response, ("publication_id", "revision"))


def _role_member(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    expected_assigned = getattr(response, "assigned", None)
    if not isinstance(expected_assigned, bool):
        return {}
    selected = dict(_selected_values(request, ("user_id", "role_id")))
    selected["expected_assigned"] = expected_assigned
    return selected


def _channel_settings(
    _request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    channel_id = getattr(response, "channel_id", None)
    old_topic = getattr(response, "old_topic", None)
    old_slowmode = getattr(response, "old_slowmode_seconds", None)
    current_topic = getattr(response, "topic", None)
    current_slowmode = getattr(response, "slowmode_seconds", None)
    if (
        not isinstance(channel_id, str)
        or (old_topic is not None and not isinstance(old_topic, str))
        or not isinstance(old_slowmode, int)
        or isinstance(old_slowmode, bool)
        or (current_topic is not None and not isinstance(current_topic, str))
        or not isinstance(current_slowmode, int)
        or isinstance(current_slowmode, bool)
    ):
        return {}
    return {
        "channel_id": channel_id,
        "topic": old_topic,
        "slowmode_seconds": old_slowmode,
        "expected_topic": current_topic,
        "expected_slowmode_seconds": current_slowmode,
    }


def _timeout_state(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    if getattr(response, "changed", True) is False:
        return None
    selected = dict(_selected_values(request, ("user_id",)))
    previous = getattr(response, "previous_until_iso", None)
    current = getattr(response, "until_iso", None)
    if (
        previous is not None
        and not isinstance(previous, str)
    ) or (
        current is not None
        and not isinstance(current, str)
    ):
        return {}
    selected["until_iso"] = previous
    selected["expected_until_iso"] = current
    selected["reason"] = "Undo a previous Simajilord timeout action"
    return selected


def _unban_member(request: object, _response: object) -> Mapping[str, object]:
    selected = dict(_selected_values(request, ("user_id",)))
    selected["reason"] = "Undo a previous Simajilord ban action"
    return selected


def _created_role(_request: object, response: object) -> Mapping[str, object]:
    return _selected_values(response, ("role_id", "undo_fingerprint"))


def _created_channel(_request: object, response: object) -> Mapping[str, object]:
    return _selected_values(response, ("channel_id", "undo_fingerprint"))


def _audio_volume(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    selected: dict[str, object] = {}
    for request_name, response_name in (
        ("music_percent", "previous_music_volume_percent"),
        ("speech_percent", "previous_speech_volume_percent"),
    ):
        if getattr(request, request_name, None) is None:
            continue
        previous = getattr(response, response_name, None)
        current = getattr(response, f"{request_name.removesuffix('_percent')}_volume_percent", None)
        if (
            not isinstance(previous, int)
            or isinstance(previous, bool)
            or not isinstance(current, int)
            or isinstance(current, bool)
        ):
            return None
        if current == previous:
            continue
        selected[request_name] = previous
        selected[f"expected_{request_name}"] = current
    return selected or None


def _read_aloud_announcements(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    selected: dict[str, object] = {}
    for argument, response_name in (
        ("join", "previous_announce_join"),
        ("leave", "previous_announce_leave"),
        ("move", "previous_announce_move"),
    ):
        if getattr(request, argument, None) is None:
            continue
        previous = getattr(response, response_name, None)
        current = getattr(response, f"announce_{argument}", None)
        if not isinstance(previous, bool) or not isinstance(current, bool):
            return None
        if current == previous:
            continue
        selected[argument] = previous
        selected[f"expected_{argument}"] = current
    return selected or None


def _read_aloud_semantics(
    request: object,
    response: object,
) -> Mapping[str, object] | None:
    selected: dict[str, object] = {}
    for argument, response_name in (
        ("author_names", "previous_read_author_names"),
        ("replies", "previous_read_replies"),
        ("attachments", "previous_read_attachments"),
        ("vc_members_only", "previous_vc_members_only"),
    ):
        if getattr(request, argument, None) is None:
            continue
        previous = getattr(response, response_name, None)
        current_name = {
            "author_names": "read_author_names",
            "replies": "read_replies",
            "attachments": "read_attachments",
            "vc_members_only": "vc_members_only",
        }[argument]
        current = getattr(response, current_name, None)
        if not isinstance(previous, bool) or not isinstance(current, bool):
            return None
        if current == previous:
            continue
        selected[argument] = previous
        selected[f"expected_{argument}"] = current
    return selected or None


def _read_aloud_content_state(
    _request: object,
    response: object,
) -> Mapping[str, object] | None:
    values: dict[str, object] = {}
    for argument, previous_name, current_name in (
        ("read_messages", "previous_read_messages", "read_messages"),
        ("announce_join", "previous_announce_join", "announce_join"),
        ("announce_leave", "previous_announce_leave", "announce_leave"),
        ("announce_move", "previous_announce_move", "announce_move"),
    ):
        previous = getattr(response, previous_name, None)
        current = getattr(response, current_name, None)
        if not isinstance(previous, bool) or not isinstance(current, bool):
            return None
        values[argument] = previous
        values[f"expected_{argument}"] = current
    if all(
        getattr(response, current_name) == values[argument]
        for argument, _previous_name, current_name in (
            ("read_messages", "previous_read_messages", "read_messages"),
            ("announce_join", "previous_announce_join", "announce_join"),
            ("announce_leave", "previous_announce_leave", "announce_leave"),
            ("announce_move", "previous_announce_move", "announce_move"),
        )
    ):
        return None
    return values


def _selected_values(value: object, names: tuple[str, ...]) -> Mapping[str, object]:
    selected: dict[str, object] = {}
    for name in names:
        item = getattr(value, name, None)
        if not isinstance(item, (str, int, float, bool)) or isinstance(item, bytes):
            return {}
        selected[name] = item
    return selected


_REVERSIBLE_POLICIES = (
    ActionPolicy(
        capability="discord.add_reaction",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.remove_own_reaction",
        undo_arguments=_same_reaction,
    ),
    ActionPolicy(
        capability="discord.remove_own_reaction",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.add_reaction",
        undo_arguments=_same_reaction,
    ),
    *(
        ActionPolicy(
            capability=capability,
            classification=ActionClassification.FULLY_REVERSIBLE,
            undo_capability="discord.delete_own_message",
            undo_arguments=_created_message,
        )
        for capability in (
            "discord.send_embed",
            "discord.send_message",
            "discord.reply_message",
            "discord.send_file",
            "discord.send_files",
            "discord.send_managed_file",
            "discord.send_published_file",
            "discord.open_file_manager",
            "discord.post_expanded_message",
            "discord.create_quote_image",
            "discord.create_poll",
        )
    ),
    ActionPolicy(
        capability="discord.send_messages",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.delete_own_messages",
        undo_arguments=_created_messages,
    ),
    ActionPolicy(
        capability="timer.create",
        classification=ActionClassification.COMPENSATING,
        undo_capability="timer.cancel",
        undo_arguments=_created_timer,
    ),
    ActionPolicy(
        capability="timer.cancel",
        classification=ActionClassification.COMPENSATING,
        undo_capability="timer.restore",
        undo_arguments=_cancelled_timer,
    ),
    ActionPolicy(
        capability="timer.restore",
        classification=ActionClassification.COMPENSATING,
        undo_capability="timer.cancel",
        undo_arguments=_cancelled_timer,
    ),
    ActionPolicy(
        capability="discord.pin_message",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.unpin_message",
        undo_arguments=_message_pin_change(True),
    ),
    ActionPolicy(
        capability="discord.unpin_message",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.pin_message",
        undo_arguments=_message_pin_change(False),
    ),
    ActionPolicy(
        capability="discord.create_thread",
        classification=ActionClassification.COMPENSATING,
        undo_capability="discord.update_thread",
        undo_arguments=_created_thread,
    ),
    ActionPolicy(
        capability="discord.create_forum_post",
        classification=ActionClassification.COMPENSATING,
        undo_capability="discord.update_thread",
        undo_arguments=_created_thread,
    ),
    ActionPolicy(
        capability="discord.update_thread",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.update_thread",
        undo_arguments=_thread_update,
    ),
    ActionPolicy(
        capability="discord.add_thread_member",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.remove_thread_member",
        undo_arguments=_thread_member,
    ),
    ActionPolicy(
        capability="files.publish_copy",
        classification=ActionClassification.COMPENSATING,
        undo_capability="files.revoke_publication",
        undo_arguments=_published_file_copy,
    ),
    ActionPolicy(
        capability="discord.assign_role",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.remove_role",
        undo_arguments=_role_member,
    ),
    ActionPolicy(
        capability="discord.remove_role",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.assign_role",
        undo_arguments=_role_member,
    ),
    ActionPolicy(
        capability="discord.update_channel_settings",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.update_channel_settings",
        undo_arguments=_channel_settings,
    ),
    ActionPolicy(
        capability="discord.set_timeout",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.set_timeout",
        undo_arguments=_timeout_state,
    ),
    ActionPolicy(
        capability="discord.ban_member",
        classification=ActionClassification.COMPENSATING,
        undo_capability="discord.unban_member",
        undo_arguments=_unban_member,
    ),
    ActionPolicy(
        capability="discord.create_role",
        classification=ActionClassification.COMPENSATING,
        undo_capability="discord.delete_created_role",
        undo_arguments=_created_role,
    ),
    ActionPolicy(
        capability="discord.create_channel",
        classification=ActionClassification.COMPENSATING,
        undo_capability="discord.delete_created_channel",
        undo_arguments=_created_channel,
    ),
    *(
        ActionPolicy(
            capability=capability,
            classification=ActionClassification.FULLY_REVERSIBLE,
            undo_capability=capability,
            undo_arguments=_audio_volume,
        )
        for capability in (
            "audio.set_volume",
            "discord.set_audio_volume",
        )
    ),
    *(
        ActionPolicy(
            capability=capability,
            classification=ActionClassification.FULLY_REVERSIBLE,
            undo_capability=capability,
            undo_arguments=_read_aloud_announcements,
        )
        for capability in (
            "speech.read_aloud_announcements_set",
            "discord.read_aloud_announcements_set",
        )
    ),
    *(
        ActionPolicy(
            capability=capability,
            classification=ActionClassification.FULLY_REVERSIBLE,
            undo_capability=capability,
            undo_arguments=_read_aloud_semantics,
        )
        for capability in (
            "speech.read_aloud_semantics_set",
            "discord.read_aloud_semantics_set",
        )
    ),
    *(
        ActionPolicy(
            capability=capability,
            classification=ActionClassification.FULLY_REVERSIBLE,
            undo_capability="speech.read_aloud_content_state_restore",
            undo_arguments=_read_aloud_content_state,
        )
        for capability in (
            "speech.read_aloud_content_mode_set",
            "discord.read_aloud_content_mode_set",
        )
    ),
)

# Every currently implemented mutation that cannot be restored from a few IDs or scalar
# values is explicit here. In particular, file/message bodies and deleted content are never
# copied into the action database merely to manufacture an Undo button.
NON_UNDOABLE_ACTION_CAPABILITIES = frozenset(
    {
        "action.undo",
        "authority.request",
        "authority.lease_create",
        "authority.lease_revoke",
        "audio.play",
        "audio.control",
        "audio.mix",
        "audio.pause",
        "audio.resume",
        "audio.skip",
        "audio.stop",
        "audio.leave",
        "audio.set_loop",
        "audio.remove",
        "audio.set_auto_leave",
        "audio.shuffle",
        "audio.seek",
        "audio.tune",
        "audio.move",
        "audio.clear_mine",
        "compute.run",
        "connector.write",
        "connector.destructive",
        "files.download_url",
        "files.copy_to_task",
        "files.delete",
        "files.write_text",
        "files.replace_text",
        "files.revoke_publication",
        "feedback.create",
        "image.generate",
        "media.download",
        "media.save",
        "memory.forget",
        "memory.remember",
        "memory.review",
        "memory.update",
        "speech.speak",
        "speech.manage_read_aloud",
        "speech.read_aloud_add_sources",
        "speech.read_aloud_remove_source",
        "speech.read_aloud_disable",
        "speech.read_aloud_content_state_restore",
        "speech.read_aloud_user_voice_set",
        "speech.read_aloud_server_voice_set",
        "speech.read_aloud_dictionary_set",
        "speech.read_aloud_dictionary_remove",
        "speech.read_aloud_exclusion_set",
        "system.shell",
        "discord.analyze_attachment",
        "discord.delete_own_message",
        "discord.delete_own_messages",
        "discord.edit_own_message",
        "discord.create_guild_resource",
        "discord.update_guild_resource",
        "discord.delete_guild_resource",
        "discord.message_action",
        "discord.set_channel_overwrite",
        "discord.create_platform_asset",
        "discord.update_platform_asset",
        "discord.delete_platform_asset",
        "discord.create_automod_rule",
        "discord.update_automod_rule",
        "discord.delete_automod_rule",
        "discord.channel_operation",
        "discord.forward_message",
        "discord.send_direct_message",
        "discord.set_bot_presence",
        "discord.delete_created_role",
        "discord.delete_created_channel",
        "discord.delete_message",
        "discord.bulk_delete_messages",
        "discord.kick_member",
        "discord.unban_member",
        "discord.import_attachment",
        "discord.connect_voice",
        "discord.play_audio",
        "discord.play_attachment",
        "discord.control_audio",
        "discord.pause_audio",
        "discord.resume_audio",
        "discord.skip_audio",
        "discord.stop_audio",
        "discord.leave_audio",
        "discord.set_audio_loop",
        "discord.remove_audio",
        "discord.set_audio_auto_leave",
        "discord.shuffle_audio",
        "discord.seek_audio",
        "discord.tune_audio",
        "discord.set_audio_radio",
        "discord.move_audio",
        "discord.clear_my_audio",
        "discord.speak",
        "discord.read_aloud_add_sources",
        "discord.read_aloud_remove_source",
        "discord.read_aloud_disable",
        "discord.read_aloud_dictionary_set",
        "discord.read_aloud_dictionary_remove",
        "discord.read_aloud_exclusion_set",
        "discord.manage_read_aloud",
        "discord.remove_thread_member",
    }
)

_ACTION_POLICIES = {policy.capability: policy for policy in _REVERSIBLE_POLICIES}
if len(_ACTION_POLICIES) != len(_REVERSIBLE_POLICIES):
    raise RuntimeError("Duplicate action policy capability")
if NON_UNDOABLE_ACTION_CAPABILITIES & _ACTION_POLICIES.keys():
    raise RuntimeError("Action capabilities have conflicting undo classifications")


def action_policy(capability: str) -> ActionPolicy:
    """Return an explicit policy, conservatively defaulting new actions to no Undo."""

    policy = _ACTION_POLICIES.get(capability)
    if policy is not None:
        return policy
    return ActionPolicy(
        capability=capability,
        classification=ActionClassification.NON_UNDOABLE,
    )


class ActionReceiptStore:
    """Small restart-safe ledger with TTL, total, and per-actor record caps."""

    def __init__(
        self,
        path: Path,
        *,
        ttl: timedelta = timedelta(days=7),
        max_records: int = 2_000,
        max_records_per_actor: int = 100,
        recover_interrupted: bool = True,
    ) -> None:
        if ttl <= timedelta(0):
            raise ValueError("action receipt TTL must be positive")
        if max_records < 1:
            raise ValueError("max_records must be positive")
        if not 1 <= max_records_per_actor <= max_records:
            raise ValueError("max_records_per_actor must be between 1 and max_records")
        self.path = path
        self.ttl = ttl
        self.max_records = max_records
        self.max_records_per_actor = max_records_per_actor
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize(recover_interrupted=recover_interrupted)

    async def add(
        self,
        *,
        action_id: str,
        capability: str,
        context: InvocationContext,
        target_ids: tuple[tuple[str, str], ...],
        classification: ActionClassification,
        undo_capability: str | None,
        undo_arguments: Mapping[str, object] | None,
        host_delivery: bool = False,
    ) -> ActionRecord:
        _validate_action_identities(context)
        now = datetime.now(UTC)
        record = ActionRecord(
            action_id=action_id,
            capability=capability,
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            transport=context.transport,
            request_id=context.request_id,
            principal_kind=context.principal_kind,
            executor_principal_id=context.executor_principal_id,
            delegator_principal_id=context.delegator_principal_id,
            trigger_actor_ids=context.trigger_actor_ids,
            requester_principal_id=context.requester_principal_id,
            policy_id=context.policy_id,
            target_ids=target_ids,
            status=ActionStatus.SUCCEEDED,
            classification=classification,
            undo_capability=undo_capability,
            undo_arguments=undo_arguments,
            host_delivery=host_delivery,
            created_at=now,
            expires_at=now + self.ttl,
        )
        async with self._lock:
            return await asyncio.to_thread(self._add, record)

    async def plan_external_effect(
        self,
        *,
        capability: str,
        request: object,
        context: InvocationContext,
        authorization_reference: str | None,
    ) -> ExternalEffectRecord:
        """Persist intent before a write is eligible to reach its handler."""

        record = _external_effect_record(
            capability=capability,
            request=request,
            context=context,
            authorization_reference=authorization_reference,
            ttl=self.ttl,
        )
        async with self._lock:
            return await asyncio.to_thread(self._plan_external_effect, record)

    async def dispatch_external_effect(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:
        """Cross the replay barrier immediately before external dispatch."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset({ExternalEffectStatus.PLANNED}),
                ExternalEffectStatus.DISPATCHED,
                None,
            )

    async def mark_external_effect_unknown(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:
        """Record that a dispatched write did not return definite evidence."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset(
                    {
                        ExternalEffectStatus.DISPATCHED,
                        ExternalEffectStatus.CANCELLED,
                    }
                ),
                ExternalEffectStatus.UNKNOWN,
                None,
            )

    async def reject_external_effect(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:
        """Close a plan that failed validation before any effect was dispatched."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset({ExternalEffectStatus.PLANNED}),
                ExternalEffectStatus.REJECTED,
                None,
            )

    async def cancel_external_effect(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord:
        """Close a plan cancelled before any effect was dispatched."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset({ExternalEffectStatus.PLANNED}),
                ExternalEffectStatus.CANCELLED,
                None,
            )

    async def confirm_external_effect(
        self,
        effect_id: str,
        *,
        action_id: str | None,
    ) -> ExternalEffectRecord:
        """Attach positive result evidence to a dispatched effect."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset(
                    {
                        ExternalEffectStatus.DISPATCHED,
                        ExternalEffectStatus.UNKNOWN,
                    }
                ),
                ExternalEffectStatus.CONFIRMED,
                action_id,
            )

    async def reconcile_external_effect(
        self,
        effect_id: str,
        *,
        action_id: str | None = None,
    ) -> ExternalEffectRecord:
        """Close an operator-verified uncertain effect without replaying it."""

        async with self._lock:
            return await asyncio.to_thread(
                self._transition_external_effect,
                effect_id,
                frozenset({ExternalEffectStatus.UNKNOWN}),
                ExternalEffectStatus.RECONCILED,
                action_id,
            )

    async def external_effect(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord | None:
        async with self._lock:
            return await asyncio.to_thread(self._external_effect, effect_id)

    async def external_effects(
        self,
        *,
        status: ExternalEffectStatus | None = None,
        limit: int = 100,
    ) -> tuple[ExternalEffectRecord, ...]:
        """List body-free effect metadata for bounded operator diagnostics."""

        if not 1 <= limit <= 1_000:
            raise ValueError("external effect limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(
                self._external_effects,
                status,
                limit,
            )

    async def is_confirmed_memory_evidence(
        self,
        *,
        action_id: str,
        context: InvocationContext,
        allow_any_actor: bool,
    ) -> bool:
        """Validate a current external Action without claiming or mutating it."""

        async with self._lock:
            return await asyncio.to_thread(
                self._is_confirmed_memory_evidence,
                action_id,
                context.actor_id,
                context.workspace_id,
                allow_any_actor,
            )

    async def request_has_replay_barrier(self, request_id: str) -> bool:
        """Return whether recovery must not repeat the model turn automatically."""

        async with self._lock:
            return await asyncio.to_thread(
                self._request_has_replay_barrier,
                request_id,
            )

    async def claim_latest_or_id(
        self,
        *,
        action_id: str | None,
        context: InvocationContext,
    ) -> ActionRecord:
        allow_any_actor = ACTION_UNDO_ANY_GRANT in context.grants
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_latest_or_id,
                action_id,
                context.actor_id,
                context.workspace_id,
                allow_any_actor,
            )

    async def release_failed(self, action_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._release_failed, action_id)

    async def complete(self, action_id: str, undo_action_id: str) -> ActionRecord:
        async with self._lock:
            return await asyncio.to_thread(self._complete, action_id, undo_action_id)

    async def count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._count)

    def _initialize(self, *, recover_interrupted: bool) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_actions (
                    action_id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    transport TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    principal_kind TEXT,
                    executor_principal_id TEXT,
                    delegator_principal_id TEXT,
                    trigger_actor_ids_json TEXT NOT NULL DEFAULT '[]',
                    requester_principal_id TEXT,
                    policy_id TEXT,
                    target_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    undo_capability TEXT,
                    undo_arguments_json TEXT,
                    host_delivery INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    undone_at TEXT,
                    undo_action_id TEXT
                );
                CREATE INDEX IF NOT EXISTS agent_actions_actor_recent
                ON agent_actions(actor_id, workspace_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS agent_actions_expiry
                ON agent_actions(expires_at);

                CREATE TABLE IF NOT EXISTS agent_external_effects (
                    effect_id TEXT PRIMARY KEY,
                    capability TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    transport TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    provider_thread_id TEXT,
                    provider_turn_id TEXT,
                    tool_call_id TEXT NOT NULL,
                    arguments_fingerprint TEXT NOT NULL,
                    target_ids_json TEXT NOT NULL,
                    authorization_reference TEXT,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    action_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS agent_external_effects_request
                ON agent_external_effects(request_id, status);
                CREATE INDEX IF NOT EXISTS agent_external_effects_expiry
                ON agent_external_effects(expires_at);
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(agent_actions)"
                ).fetchall()
            }
            if "host_delivery" not in columns:
                connection.execute(
                    """
                    ALTER TABLE agent_actions
                    ADD COLUMN host_delivery INTEGER NOT NULL DEFAULT 0
                    """
                )
            identity_columns = {
                "principal_kind": "TEXT",
                "executor_principal_id": "TEXT",
                "delegator_principal_id": "TEXT",
                "trigger_actor_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                "requester_principal_id": "TEXT",
                "policy_id": "TEXT",
            }
            for column, declaration in identity_columns.items():
                if column in columns:
                    continue
                connection.execute(
                    f"ALTER TABLE agent_actions ADD COLUMN {column} {declaration}"
                )
            effect_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(agent_external_effects)"
                ).fetchall()
            }
            effect_metadata_columns = {
                "arguments_fingerprint": "TEXT NOT NULL DEFAULT 'legacy'",
                "target_ids_json": "TEXT NOT NULL DEFAULT '{}'",
                "authorization_reference": "TEXT",
                "summary": "TEXT NOT NULL DEFAULT 'legacy external effect'",
            }
            for column, declaration in effect_metadata_columns.items():
                if column in effect_columns:
                    continue
                connection.execute(
                    "ALTER TABLE agent_external_effects "
                    f"ADD COLUMN {column} {declaration}"
                )
            connection.execute(
                """
                UPDATE agent_external_effects
                SET target_ids_json = '{}'
                WHERE target_ids_json = '[]'
                """
            )
            connection.execute(
                "DELETE FROM agent_actions WHERE expires_at <= ?",
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute(
                "DELETE FROM agent_external_effects WHERE expires_at <= ?",
                (datetime.now(UTC).isoformat(),),
            )
            connection.execute(
                """
                UPDATE agent_actions
                SET status = ?
                WHERE status = ?
                """,
                (
                    ActionStatus.SUCCEEDED.value,
                    ActionStatus.UNDOING.value,
                ),
            )
            if recover_interrupted:
                connection.execute(
                    """
                    UPDATE agent_external_effects
                    SET status = ?, updated_at = ?
                    WHERE status = ?
                    """,
                    (
                        ExternalEffectStatus.UNKNOWN.value,
                        datetime.now(UTC).isoformat(),
                        ExternalEffectStatus.DISPATCHED.value,
                    ),
                )
        os.chmod(self.path, 0o600)

    def _add(self, record: ActionRecord) -> ActionRecord:
        target_ids_json = json.dumps(
            dict(record.target_ids),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        undo_arguments_json = (
            json.dumps(
                record.undo_arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if record.undo_arguments is not None
            else None
        )
        if (
            undo_arguments_json is not None
            and len(undo_arguments_json) > _MAX_UNDO_ARGUMENT_CHARACTERS
        ):
            raise ValueError("undo arguments exceed the bounded action record size")
        trigger_actor_ids_json = json.dumps(
            record.trigger_actor_ids,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "DELETE FROM agent_actions WHERE expires_at <= ?",
                (now,),
            )
            connection.execute(
                """
                INSERT INTO agent_actions(
                    action_id, capability, actor_id, workspace_id, transport,
                    request_id, principal_kind, executor_principal_id,
                    delegator_principal_id, trigger_actor_ids_json,
                    requester_principal_id, policy_id, target_ids_json,
                    status, classification,
                    undo_capability, undo_arguments_json, host_delivery,
                    created_at, expires_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(action_id) DO NOTHING
                """,
                (
                    record.action_id,
                    record.capability,
                    record.actor_id,
                    record.workspace_id,
                    record.transport,
                    record.request_id,
                    record.principal_kind,
                    record.executor_principal_id,
                    record.delegator_principal_id,
                    trigger_actor_ids_json,
                    record.requester_principal_id,
                    record.policy_id,
                    target_ids_json,
                    record.status.value,
                    record.classification.value,
                    record.undo_capability,
                    undo_arguments_json,
                    int(record.host_delivery),
                    record.created_at.isoformat(),
                    record.expires_at.isoformat(),
                ),
            )
            persisted_row = connection.execute(
                "SELECT * FROM agent_actions WHERE action_id = ?",
                (record.action_id,),
            ).fetchone()
            assert persisted_row is not None
            persisted = _row_record(persisted_row)
            immutable_identity = (
                persisted.capability,
                persisted.actor_id,
                persisted.workspace_id,
                persisted.transport,
                persisted.request_id,
                persisted.principal_kind,
                persisted.executor_principal_id,
                persisted.delegator_principal_id,
                persisted.trigger_actor_ids,
                persisted.requester_principal_id,
                persisted.policy_id,
                persisted.target_ids,
                persisted.classification,
                persisted.undo_capability,
                persisted.undo_arguments,
                persisted.host_delivery,
            )
            requested_identity = (
                record.capability,
                record.actor_id,
                record.workspace_id,
                record.transport,
                record.request_id,
                record.principal_kind,
                record.executor_principal_id,
                record.delegator_principal_id,
                record.trigger_actor_ids,
                record.requester_principal_id,
                record.policy_id,
                record.target_ids,
                record.classification,
                record.undo_capability,
                record.undo_arguments,
                record.host_delivery,
            )
            if immutable_identity != requested_identity:
                raise ValueError("action ID already belongs to a different action")
            connection.execute(
                """
                DELETE FROM agent_actions
                WHERE action_id IN (
                    SELECT action_id FROM agent_actions
                    WHERE actor_id = ?
                    ORDER BY created_at DESC, action_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (record.actor_id, self.max_records_per_actor),
            )
            connection.execute(
                """
                DELETE FROM agent_actions
                WHERE action_id IN (
                    SELECT action_id FROM agent_actions
                    ORDER BY created_at DESC, action_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_records,),
            )
            connection.commit()
            return persisted

    def _plan_external_effect(
        self,
        record: ExternalEffectRecord,
    ) -> ExternalEffectRecord:
        target_ids_json = json.dumps(
            dict(record.target_ids),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "DELETE FROM agent_external_effects WHERE expires_at <= ?",
                (now,),
            )
            connection.execute(
                """
                INSERT INTO agent_external_effects(
                    effect_id, capability, actor_id, workspace_id, transport,
                    request_id, provider_thread_id, provider_turn_id,
                    tool_call_id, arguments_fingerprint, target_ids_json,
                    authorization_reference, summary, status, action_id,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(effect_id) DO NOTHING
                """,
                (
                    record.effect_id,
                    record.capability,
                    record.actor_id,
                    record.workspace_id,
                    record.transport,
                    record.request_id,
                    record.provider_thread_id,
                    record.provider_turn_id,
                    record.tool_call_id,
                    record.arguments_fingerprint,
                    target_ids_json,
                    record.authorization_reference,
                    record.summary,
                    record.status.value,
                    record.action_id,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.expires_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM agent_external_effects WHERE effect_id = ?",
                (record.effect_id,),
            ).fetchone()
            assert row is not None
            persisted = _row_external_effect(row)
            if _external_effect_identity(persisted) != _external_effect_identity(record):
                raise ValueError("external effect ID belongs to a different tool call")
            if persisted.status in {
                ExternalEffectStatus.REJECTED,
                ExternalEffectStatus.CANCELLED,
            }:
                connection.execute(
                    """
                    UPDATE agent_external_effects
                    SET status = ?, action_id = NULL, updated_at = ?, expires_at = ?
                    WHERE effect_id = ?
                    """,
                    (
                        ExternalEffectStatus.PLANNED.value,
                        record.updated_at.isoformat(),
                        record.expires_at.isoformat(),
                        record.effect_id,
                    ),
                )
                refreshed = connection.execute(
                    "SELECT * FROM agent_external_effects WHERE effect_id = ?",
                    (record.effect_id,),
                ).fetchone()
                assert refreshed is not None
                persisted = _row_external_effect(refreshed)
            if persisted.status is not ExternalEffectStatus.PLANNED:
                raise ValueError("external effect already crossed the replay barrier")
            self._enforce_external_effect_caps(connection, record.actor_id)
            connection.commit()
            return persisted

    def _enforce_external_effect_caps(
        self,
        connection: sqlite3.Connection,
        actor_id: str,
    ) -> None:
        terminal_statuses = (
            ExternalEffectStatus.REJECTED.value,
            ExternalEffectStatus.CANCELLED.value,
            ExternalEffectStatus.CONFIRMED.value,
            ExternalEffectStatus.RECONCILED.value,
        )
        actor_count_row = connection.execute(
            "SELECT COUNT(*) FROM agent_external_effects WHERE actor_id = ?",
            (actor_id,),
        ).fetchone()
        actor_overflow = (
            int(actor_count_row[0]) - self.max_records_per_actor
            if actor_count_row is not None
            else 0
        )
        if actor_overflow > 0:
            connection.execute(
                """
                DELETE FROM agent_external_effects
                WHERE effect_id IN (
                    SELECT effect_id FROM agent_external_effects
                    WHERE actor_id = ? AND status IN (?, ?, ?, ?)
                    ORDER BY created_at ASC, effect_id ASC
                    LIMIT ?
                )
                """,
                (actor_id, *terminal_statuses, actor_overflow),
            )
        total_count_row = connection.execute(
            "SELECT COUNT(*) FROM agent_external_effects"
        ).fetchone()
        total_overflow = (
            int(total_count_row[0]) - self.max_records
            if total_count_row is not None
            else 0
        )
        if total_overflow > 0:
            connection.execute(
                """
                DELETE FROM agent_external_effects
                WHERE effect_id IN (
                    SELECT effect_id FROM agent_external_effects
                    WHERE status IN (?, ?, ?, ?)
                    ORDER BY created_at ASC, effect_id ASC
                    LIMIT ?
                )
                """,
                (*terminal_statuses, total_overflow),
            )
        remaining_actor_row = connection.execute(
            "SELECT COUNT(*) FROM agent_external_effects WHERE actor_id = ?",
            (actor_id,),
        ).fetchone()
        remaining_total_row = connection.execute(
            "SELECT COUNT(*) FROM agent_external_effects"
        ).fetchone()
        remaining_actor = (
            int(remaining_actor_row[0]) if remaining_actor_row is not None else 0
        )
        remaining_total = (
            int(remaining_total_row[0]) if remaining_total_row is not None else 0
        )
        if (
            remaining_actor > self.max_records_per_actor
            or remaining_total > self.max_records
        ):
            raise ValueError(
                "external effect ledger capacity is exhausted by unresolved effects"
            )

    def _transition_external_effect(
        self,
        effect_id: str,
        expected: frozenset[ExternalEffectStatus],
        target: ExternalEffectStatus,
        action_id: str | None,
    ) -> ExternalEffectRecord:
        normalized_effect_id = effect_id.strip()
        if not normalized_effect_id:
            raise ValueError("external effect ID must not be empty")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_external_effects WHERE effect_id = ?",
                (normalized_effect_id,),
            ).fetchone()
            if row is None:
                raise ValueError("external effect does not exist")
            record = _row_external_effect(row)
            normalized_action_id: str | None = None
            if action_id is not None:
                normalized_action_id = action_id.strip()
                if not 1 <= len(normalized_action_id) <= 200:
                    raise ValueError("external effect action ID is invalid")
                action_row = connection.execute(
                    "SELECT 1 FROM agent_actions WHERE action_id = ?",
                    (normalized_action_id,),
                ).fetchone()
                if action_row is None:
                    raise ValueError("external effect action evidence does not exist")
            if (
                record.action_id is not None
                and normalized_action_id is not None
                and record.action_id != normalized_action_id
            ):
                raise ValueError("external effect action evidence conflicts")
            if record.status is target:
                connection.commit()
                return record
            if record.status not in expected:
                raise ValueError(
                    "external effect transition is invalid: "
                    f"{record.status.value} -> {target.value}"
                )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE agent_external_effects
                SET status = ?, action_id = COALESCE(?, action_id), updated_at = ?
                WHERE effect_id = ?
                """,
                (target.value, normalized_action_id, now, normalized_effect_id),
            )
            updated = connection.execute(
                "SELECT * FROM agent_external_effects WHERE effect_id = ?",
                (normalized_effect_id,),
            ).fetchone()
            assert updated is not None
            connection.commit()
            return _row_external_effect(updated)

    def _external_effect(
        self,
        effect_id: str,
    ) -> ExternalEffectRecord | None:
        normalized_effect_id = effect_id.strip()
        if not normalized_effect_id:
            raise ValueError("external effect ID must not be empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_external_effects WHERE effect_id = ?",
                (normalized_effect_id,),
            ).fetchone()
        return _row_external_effect(row) if row is not None else None

    def _external_effects(
        self,
        status: ExternalEffectStatus | None,
        limit: int,
    ) -> tuple[ExternalEffectRecord, ...]:
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT * FROM agent_external_effects
                    ORDER BY updated_at DESC, effect_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM agent_external_effects
                    WHERE status = ?
                    ORDER BY updated_at DESC, effect_id DESC
                    LIMIT ?
                    """,
                    (status.value, limit),
                ).fetchall()
        return tuple(_row_external_effect(row) for row in rows)

    def _is_confirmed_memory_evidence(
        self,
        action_id: str,
        actor_id: str,
        workspace_id: str | None,
        allow_any_actor: bool,
    ) -> bool:
        normalized_action_id = action_id.strip()
        if not normalized_action_id or workspace_id is None:
            return False
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM agent_actions AS action
                JOIN agent_external_effects AS effect
                  ON effect.action_id = action.action_id
                WHERE action.action_id = ?
                  AND action.workspace_id = ?
                  AND action.status = ?
                  AND action.capability NOT LIKE 'memory.%'
                  AND action.expires_at > ?
                  AND effect.expires_at > ?
                  AND effect.status IN (?, ?)
                  AND (? = 1 OR action.actor_id = ?)
                LIMIT 1
                """,
                (
                    normalized_action_id,
                    workspace_id,
                    ActionStatus.SUCCEEDED.value,
                    now,
                    now,
                    ExternalEffectStatus.CONFIRMED.value,
                    ExternalEffectStatus.RECONCILED.value,
                    int(allow_any_actor),
                    actor_id,
                ),
            ).fetchone()
        return row is not None

    def _request_has_replay_barrier(self, request_id: str) -> bool:
        normalized_request_id = request_id.strip()
        if not normalized_request_id:
            raise ValueError("request ID must not be empty")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM agent_external_effects
                WHERE request_id = ? AND status IN (?, ?, ?, ?)
                LIMIT 1
                """,
                (
                    normalized_request_id,
                    ExternalEffectStatus.DISPATCHED.value,
                    ExternalEffectStatus.UNKNOWN.value,
                    ExternalEffectStatus.CONFIRMED.value,
                    ExternalEffectStatus.RECONCILED.value,
                ),
            ).fetchone()
        return row is not None

    def _claim_latest_or_id(
        self,
        action_id: str | None,
        actor_id: str,
        workspace_id: str | None,
        allow_any_actor: bool,
    ) -> ActionRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "DELETE FROM agent_actions WHERE expires_at <= ?",
                (now,),
            )
            where = ["undo_capability IS NOT NULL"]
            values: list[object] = []
            if action_id is not None:
                where.append("action_id = ?")
                values.append(action_id)
            if workspace_id is None:
                where.append("workspace_id IS NULL")
            else:
                where.append("workspace_id = ?")
                values.append(workspace_id)
            if not allow_any_actor or action_id is None:
                where.append("actor_id = ?")
                values.append(actor_id)
            query = "SELECT * FROM agent_actions WHERE " + " AND ".join(where)
            if action_id is None:
                query += " ORDER BY created_at DESC, action_id DESC LIMIT 1"
            row = connection.execute(query, values).fetchone()
            if (
                action_id is None
                and row is not None
                and bool(row["host_delivery"])
            ):
                # A final confirmation is posted after the action it describes.
                # Prefer a substantive write from that same turn so "undo" cancels
                # the timer/role/etc., not merely its confirmation. If the turn had
                # no such write, the host post itself remains naturally undoable.
                substantive_where = [
                    *where,
                    "request_id = ?",
                    "host_delivery = 0",
                    "capability <> ?",
                ]
                substantive_values = [
                    *values,
                    str(row["request_id"]),
                    "discord.send_message",
                ]
                substantive = connection.execute(
                    (
                        "SELECT * FROM agent_actions WHERE "
                        + " AND ".join(substantive_where)
                        + " ORDER BY created_at DESC, action_id DESC LIMIT 1"
                    ),
                    substantive_values,
                ).fetchone()
                if substantive is not None:
                    row = substantive
            if row is None:
                connection.commit()
                raise UserError("action.undo_not_found")
            record = _row_record(row)
            if record.status is ActionStatus.UNDONE:
                connection.commit()
                return record
            if record.status is ActionStatus.UNDOING:
                connection.commit()
                raise UserError("action.undo_in_progress")
            cursor = connection.execute(
                """
                UPDATE agent_actions
                SET status = ?
                WHERE action_id = ? AND status = ?
                """,
                (
                    ActionStatus.UNDOING.value,
                    record.action_id,
                    ActionStatus.SUCCEEDED.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise UserError("action.undo_in_progress")
            connection.commit()
        return dataclasses.replace(record, status=ActionStatus.UNDOING)

    def _release_failed(self, action_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE agent_actions
                SET status = ?
                WHERE action_id = ? AND status = ?
                """,
                (
                    ActionStatus.SUCCEEDED.value,
                    action_id,
                    ActionStatus.UNDOING.value,
                ),
            )

    def _complete(self, action_id: str, undo_action_id: str) -> ActionRecord:
        undone_at = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE agent_actions
                SET status = ?, undone_at = ?, undo_action_id = ?
                WHERE action_id = ? AND status = ?
                """,
                (
                    ActionStatus.UNDONE.value,
                    undone_at.isoformat(),
                    undo_action_id,
                    action_id,
                    ActionStatus.UNDOING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise UserError("action.undo_in_progress")
            row = connection.execute(
                "SELECT * FROM agent_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            raise UserError("action.undo_not_found")
        return _row_record(row)

    def _count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM agent_actions").fetchone()
        return int(row[0]) if row is not None else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


class ActionReceiptService:
    """Record bounded receipts and execute only statically defined inverse actions."""

    def __init__(
        self,
        *,
        store: ActionReceiptStore,
        registry: CapabilityRegistry,
        journal: ActionJournal | None = None,
        policies: Mapping[str, ActionPolicy] | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.journal = journal
        self._policies = dict(policies or {})
        for capability, policy in self._policies.items():
            if capability != policy.capability:
                raise ValueError(
                    "custom action policy key must match its capability"
                )
            if (
                capability in _ACTION_POLICIES
                or capability in NON_UNDOABLE_ACTION_CAPABILITIES
            ):
                raise ValueError(
                    f"custom action policy shadows built-in classification: {capability}"
                )

    def has_explicit_policy(self, capability: str) -> bool:
        """Return whether code explicitly classifies an agent write."""

        return (
            capability in self._policies
            or capability in _ACTION_POLICIES
            or capability in NON_UNDOABLE_ACTION_CAPABILITIES
        )

    async def plan_external_effect(
        self,
        *,
        capability: str,
        request: object,
        context: InvocationContext,
        authorization_reference: str | None,
    ) -> ExternalEffectDispatchHandle:
        """Persist typed intent without crossing the external replay barrier."""

        planned = await self.store.plan_external_effect(
            capability=capability,
            request=request,
            context=context,
            authorization_reference=authorization_reference,
        )
        await self._append_external_effect_journal(planned, context=context)
        return ExternalEffectDispatchHandle(
            effect_id=planned.effect_id,
            service=self,
            context=context,
        )

    async def dispatch_external_effect(
        self,
        effect_id: str,
        *,
        context: InvocationContext,
    ) -> None:
        dispatched = await self.store.dispatch_external_effect(effect_id)
        await self._append_external_effect_journal(dispatched, context=context)

    async def reject_external_effect(
        self,
        effect_id: str,
        *,
        context: InvocationContext,
    ) -> None:
        rejected = await self.store.reject_external_effect(effect_id)
        await self._append_external_effect_journal(rejected, context=context)

    async def cancel_external_effect(
        self,
        effect_id: str,
        *,
        context: InvocationContext,
    ) -> None:
        cancelled = await self.store.cancel_external_effect(effect_id)
        await self._append_external_effect_journal(cancelled, context=context)

    async def mark_external_effect_unknown(
        self,
        effect_id: str,
        *,
        context: InvocationContext,
    ) -> None:
        effect = await self.store.mark_external_effect_unknown(effect_id)
        await self._append_external_effect_journal(effect, context=context)

    async def confirm_external_effect(
        self,
        effect_id: str,
        *,
        context: InvocationContext,
        action_id: str | None,
    ) -> None:
        effect = await self.store.confirm_external_effect(
            effect_id,
            action_id=action_id,
        )
        await self._append_external_effect_journal(effect, context=context)

    async def request_has_replay_barrier(self, request_id: str) -> bool:
        return await self.store.request_has_replay_barrier(request_id)

    async def is_confirmed_memory_evidence(
        self,
        *,
        action_id: str,
        context: InvocationContext,
        allow_any_actor: bool,
    ) -> bool:
        """Expose only the body-free evidence predicate needed by Memory."""

        return await self.store.is_confirmed_memory_evidence(
            action_id=action_id,
            context=context,
            allow_any_actor=allow_any_actor,
        )

    async def record_posted_message(
        self,
        *,
        channel_id: str,
        message_id: str,
        context: InvocationContext,
    ) -> ActionReceipt | None:
        """Receipt a host-delivered agent post without retaining its message body."""

        normalized_channel_id = channel_id.strip()
        normalized_message_id = message_id.strip()
        if (
            not normalized_channel_id
            or not normalized_message_id
            or len(normalized_channel_id) > _MAX_TARGET_ID_CHARACTERS
            or len(normalized_message_id) > _MAX_TARGET_ID_CHARACTERS
        ):
            raise ValueError("posted Discord message IDs must be bounded and non-empty")
        action_id = "act_" + hashlib.sha256(
            (
                "host_delivery\0"
                f"{context.request_id}\0"
                f"{normalized_channel_id}\0"
                f"{normalized_message_id}"
            ).encode()
        ).hexdigest()[:32]
        return await self.record(
            capability="discord.send_message",
            request=_PostedMessageRequest(channel_id=normalized_channel_id),
            response=_PostedMessageResponse(
                channel_id=normalized_channel_id,
                message_id=normalized_message_id,
            ),
            context=context,
            host_delivery=True,
            action_id=action_id,
        )

    async def record_posted_messages(
        self,
        *,
        channel_id: str,
        message_ids: tuple[str, ...],
        context: InvocationContext,
    ) -> ActionReceipt | None:
        """Receipt one final response as a unit using bounded Discord IDs only."""

        normalized_channel_id = channel_id.strip()
        normalized_message_ids = tuple(message_id.strip() for message_id in message_ids)
        if (
            not normalized_channel_id
            or len(normalized_channel_id) > _MAX_TARGET_ID_CHARACTERS
            or not 1 <= len(normalized_message_ids) <= 100
            or len(set(normalized_message_ids)) != len(normalized_message_ids)
            or any(
                not message_id.isascii()
                or not message_id.isdigit()
                or int(message_id) <= 0
                or len(message_id) > _MAX_TARGET_ID_CHARACTERS
                for message_id in normalized_message_ids
            )
        ):
            raise ValueError("posted Discord message IDs must be bounded and unique")
        packed_message_ids = ",".join(normalized_message_ids)
        action_id = "act_" + hashlib.sha256(
            (
                "host_delivery_group\0"
                f"{context.request_id}\0"
                f"{normalized_channel_id}\0"
                f"{packed_message_ids}"
            ).encode()
        ).hexdigest()[:32]
        return await self.record(
            capability="discord.send_messages",
            request=_PostedMessageRequest(channel_id=normalized_channel_id),
            response=_PostedMessagesResponse(
                channel_id=normalized_channel_id,
                message_ids=packed_message_ids,
            ),
            context=context,
            host_delivery=True,
            action_id=action_id,
        )

    async def record(
        self,
        *,
        capability: str,
        request: object,
        response: object,
        context: InvocationContext,
        host_delivery: bool = False,
        action_id: str | None = None,
        effect_id: str | None = None,
    ) -> ActionReceipt | None:
        """Record one successful write without changing the write's own result."""

        if capability == "action.undo":
            if getattr(response, "status", None) == "already_undone":
                return None
            undo_action_id = getattr(response, "undo_action_id", None)
            if (
                not isinstance(undo_action_id, str)
                or not undo_action_id.startswith("act_")
                or len(undo_action_id) > 128
            ):
                raise ValueError("action undo result lacks a bounded receipt ID")
            action_id = undo_action_id
        else:
            action_id = action_id or f"act_{uuid.uuid4().hex}"
        policy = self._policies.get(capability, action_policy(capability))
        undo_capability = policy.undo_capability
        undo_arguments: Mapping[str, object] | None = None
        if undo_capability is not None and policy.undo_arguments is not None:
            try:
                captured = policy.undo_arguments(request, response)
                if captured is None:
                    undo_capability = None
                else:
                    candidate = dict(captured)
                    _validate_undo_arguments(candidate)
                    inverse = self.registry.endpoint(undo_capability)
                    if inverse.descriptor.idempotency != "idempotent_write":
                        raise ValueError("Undo inverse must be idempotent")
                    undo_arguments = candidate
            except (CapabilityError, TypeError, ValueError):
                log.exception(
                    "Action inverse could not be captured capability=%s request_id=%s",
                    capability,
                    context.request_id,
                )
                undo_capability = None
        target_ids = _target_ids(request, response)
        try:
            await self.store.add(
                action_id=action_id,
                capability=capability,
                context=context,
                target_ids=target_ids,
                classification=policy.classification,
                undo_capability=undo_capability,
                undo_arguments=undo_arguments,
                host_delivery=host_delivery,
            )
            if effect_id is not None:
                await self.confirm_external_effect(
                    effect_id,
                    context=context,
                    action_id=action_id,
                )
            tracked = True
        except Exception:
            tracked = False
            undo_capability = None
            log.exception(
                "Action receipt persistence failed capability=%s request_id=%s",
                capability,
                context.request_id,
            )
            if effect_id is not None:
                try:
                    await self.mark_external_effect_unknown(
                        effect_id,
                        context=context,
                    )
                except Exception:
                    log.critical(
                        "External effect could not be marked unknown effect=%s request=%s",
                        effect_id,
                        context.request_id,
                        exc_info=True,
                    )
        receipt = ActionReceipt(
            action_id=action_id if tracked else None,
            capability=capability,
            status="succeeded",
            tracked=tracked,
            undo_available=tracked and undo_capability is not None,
            undo_capability=undo_capability,
            classification=policy.classification,
            principal_kind=context.principal_kind,
            executor_principal_id=context.executor_principal_id,
            delegator_principal_id=context.delegator_principal_id,
            trigger_actor_ids=context.trigger_actor_ids,
            requester_principal_id=context.requester_principal_id,
            policy_id=context.policy_id,
        )
        await self._append_journal(
            kind="agent.action.recorded",
            context=context,
            payload={
                "action_id": receipt.action_id,
                "capability": capability,
                "status": receipt.status,
                "result": "succeeded",
                "classification": receipt.classification.value,
                "undo_available": receipt.undo_available,
                "undo_capability": receipt.undo_capability,
                "target_ids": dict(target_ids),
                "evidence_event_ids": [context.request_id],
                "tracked": receipt.tracked,
                "host_delivery": host_delivery,
            },
        )
        return receipt

    async def undo(
        self,
        request: ActionUndoRequest,
        context: InvocationContext,
    ) -> ActionUndoResponse:
        """Undo one persisted action once; a repeated request returns the same result."""

        record = await self.store.claim_latest_or_id(
            action_id=request.action_id,
            context=context,
        )
        if record.status is ActionStatus.UNDONE:
            await context.complete_external_effect_without_dispatch()
            assert record.undo_capability is not None
            assert record.undo_action_id is not None
            return ActionUndoResponse(
                action_id=record.action_id,
                capability=record.capability,
                status="already_undone",
                undo_capability=record.undo_capability,
                undo_action_id=record.undo_action_id,
            )
        assert record.undo_capability is not None
        assert record.undo_arguments is not None
        undo_action_id = f"act_{uuid.uuid4().hex}"
        try:
            selected = self.registry.endpoint(record.undo_capability)
            inverse_request = _build_inverse_request(
                selected.request_type,
                record.undo_arguments,
            )
            inverse_context = dataclasses.replace(
                context,
                # Undo-any authorizes reversing the recorded action, but
                # ownership and live Discord permission checks still run as
                # the actor who performed that action. This is required for
                # owner-scoped resources such as timers.
                actor_id=record.actor_id,
                request_id=f"{context.request_id}:undo:{record.action_id}",
            )
            await self.registry.invoke(
                record.undo_capability,
                inverse_request,
                inverse_context,
            )
        except asyncio.CancelledError:
            # A shutdown between claim and inverse completion must not leave the
            # receipt permanently stuck in ``undoing``.
            await self.store.release_failed(record.action_id)
            raise
        except Exception as exc:
            await self.store.release_failed(record.action_id)
            await self._append_journal(
                kind="agent.action.undo_failed",
                context=context,
                payload={
                    "action_id": record.action_id,
                    "capability": record.capability,
                    "undo_capability": record.undo_capability,
                    "result": "failed",
                    "error_type": type(exc).__name__,
                    "evidence_event_ids": [context.request_id],
                },
            )
            raise
        completed = await self.store.complete(record.action_id, undo_action_id)
        await self._append_journal(
            kind="agent.action.undone",
            context=context,
            payload={
                "action_id": completed.action_id,
                "capability": completed.capability,
                "undo_capability": completed.undo_capability,
                "undo_action_id": undo_action_id,
                "result": "succeeded",
                "target_ids": dict(completed.target_ids),
                "evidence_event_ids": [context.request_id],
            },
        )
        return ActionUndoResponse(
            action_id=completed.action_id,
            capability=completed.capability,
            status="succeeded",
            undo_capability=record.undo_capability,
            undo_action_id=undo_action_id,
        )

    async def _append_journal(
        self,
        *,
        kind: str,
        context: InvocationContext,
        payload: dict[str, object],
    ) -> None:
        if self.journal is None:
            return
        try:
            await self.journal.append(
                kind=kind,
                payload={**payload, **_action_identity_payload(context)},
                actor_id=context.actor_id,
                workspace_id=context.workspace_id,
                transport=context.transport,
                request_id=context.request_id,
            )
        except Exception:
            log.exception(
                "Action journal append failed kind=%s request_id=%s",
                kind,
                context.request_id,
            )

    async def _append_external_effect_journal(
        self,
        effect: ExternalEffectRecord,
        *,
        context: InvocationContext,
    ) -> None:
        await self._append_journal(
            kind=f"agent.external_effect.{effect.status.value}",
            context=context,
            payload={
                "schema_version": 2,
                "public_reference_id": context.public_reference_id,
                "effect_id": effect.effect_id,
                "capability": effect.capability,
                "status": effect.status.value,
                "provider_thread_id": effect.provider_thread_id,
                "provider_turn_id": effect.provider_turn_id,
                "tool_call_id": effect.tool_call_id,
                "arguments_fingerprint": effect.arguments_fingerprint,
                "target_ids": dict(effect.target_ids),
                "authorization_reference": effect.authorization_reference,
                "summary": effect.summary,
                "action_id": effect.action_id,
            },
        )


class ExternalEffectDispatchHandle:
    """Idempotent turn-local bridge from an adapter to the durable effect ledger."""

    def __init__(
        self,
        *,
        effect_id: str,
        service: ActionReceiptService,
        context: InvocationContext,
    ) -> None:
        self.effect_id = effect_id
        self._service = service
        self._context = context
        self._dispatched = False
        self._completed_without_dispatch = False
        self._lock = asyncio.Lock()

    @property
    def dispatched(self) -> bool:
        return self._dispatched

    @property
    def completed_without_dispatch(self) -> bool:
        return self._completed_without_dispatch

    async def dispatch(self) -> None:
        async with self._lock:
            if self._dispatched:
                return
            if self._completed_without_dispatch:
                raise RuntimeError("external effect already completed without dispatch")
            await self._service.dispatch_external_effect(
                self.effect_id,
                context=self._context,
            )
            self._dispatched = True

    async def complete_without_dispatch(self) -> None:
        async with self._lock:
            if self._completed_without_dispatch:
                return
            if self._dispatched:
                raise RuntimeError("dispatched external effect cannot become a no-op")
            await self._service.cancel_external_effect(
                self.effect_id,
                context=self._context,
            )
            self._completed_without_dispatch = True


def _action_identity_payload(context: InvocationContext) -> dict[str, object]:
    """Keep execution, delegation, trigger, and policy identities distinct."""

    return {
        "executor_principal_id": context.executor_principal_id,
        "delegator_principal_id": context.delegator_principal_id,
        "trigger_actor_ids": list(context.trigger_actor_ids),
        "requester_principal_id": context.requester_principal_id,
        "principal_kind": context.principal_kind,
        "policy_id": context.policy_id,
    }


def _validate_action_identities(context: InvocationContext) -> None:
    if context.principal_kind not in {
        "requester",
        "service",
        "system",
        "legacy_unknown",
    }:
        raise ValueError("action principal kind is invalid")
    values = (
        context.executor_principal_id,
        context.delegator_principal_id,
        context.requester_principal_id,
        context.policy_id,
    )
    if any(value is not None and len(value) > 200 for value in values):
        raise ValueError("action principal identities must be bounded")
    if len(context.trigger_actor_ids) > 32 or any(
        not value or len(value) > 200 for value in context.trigger_actor_ids
    ):
        raise ValueError("action trigger actor identities must be bounded")


def build_action_undo_endpoint(service: ActionReceiptService) -> CapabilityEndpoint:
    """Expose recent-action Undo without exposing the inverse endpoints themselves."""

    return endpoint(
        CapabilityDescriptor(
            name="action.undo",
            summary=(
                "Undo an action by receipt ID, or undo the current actor's most recent "
                "undoable action when no ID is supplied."
            ),
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.NEVER,
            keywords=("undo", "revert", "restore", "戻す", "取り消し"),
            side_effects=("Executes one statically defined inverse capability at most once.",),
            requires_workspace=True,
            idempotency="idempotent_write",
            expected_errors=(
                "action.undo_not_found",
                "action.undo_in_progress",
                "action.undo_conflict",
                "action.undo_target_in_use",
                "action.undo_target_state_uncertain",
            ),
            timeout_seconds=30,
            user_visible_effect="Restores the most recent supported bot action.",
        ),
        ActionUndoRequest,
        ActionUndoResponse,
        service.undo,
    )


def _validate_undo_arguments(arguments: Mapping[str, object]) -> None:
    if not arguments:
        raise ValueError("undo arguments must not be empty")
    encoded = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded) > _MAX_UNDO_ARGUMENT_CHARACTERS:
        raise ValueError("undo arguments are too large")
    for value in arguments.values():
        if (
            value is not None
            and (
                not isinstance(value, (str, int, float, bool))
                or isinstance(value, bytes)
            )
        ):
            raise TypeError("undo arguments may contain only bounded scalar values")


def _target_ids(request: object, response: object) -> tuple[tuple[str, str], ...]:
    names = {
        "action_id",
        "audio_destination_id",
        "channel_id",
        "connector_id",
        "delivery_target_id",
        "destination_channel_id",
        "destination_guild_id",
        "file_ref",
        "guild_id",
        "job_id",
        "memory_id",
        "message_id",
        "path",
        "publication_id",
        "forum_id",
        "reply_to_event_id",
        "resource_id",
        "role_id",
        "source_channel_id",
        "source_guild_id",
        "source_message_id",
        "thread_id",
        "timer_id",
        "tool",
        "undo_action_id",
        "user_id",
    }
    selected: dict[str, str] = {}
    visited: set[int] = set()

    def visit(value: object) -> None:
        if len(selected) >= _MAX_TARGET_IDS:
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)
            items = tuple(
                (field.name, getattr(value, field.name))
                for field in dataclasses.fields(value)
            )
        elif isinstance(value, Mapping):
            identity = id(value)
            if identity in visited:
                return
            visited.add(identity)
            items = tuple(
                (key, item)
                for key, item in value.items()
                if isinstance(key, str)
            )
        else:
            return
        for name, item in items:
            if (
                name in names
                and isinstance(item, (str, int))
                and not isinstance(item, bool)
            ):
                text = str(item)
                if 0 < len(text) <= _MAX_TARGET_ID_CHARACTERS:
                    selected.setdefault(name, text)
            elif dataclasses.is_dataclass(item) or isinstance(item, Mapping):
                visit(item)

    visit(request)
    visit(response)
    return tuple(sorted(selected.items()))[:_MAX_TARGET_IDS]


def _build_inverse_request(
    request_type: type[Any],
    arguments: Mapping[str, object],
) -> object:
    if not dataclasses.is_dataclass(request_type):
        raise CapabilityError(f"{request_type.__name__} must be a dataclass.")
    names = {field.name for field in dataclasses.fields(request_type)}
    unknown = set(arguments) - names
    if unknown:
        raise CapabilityError(
            "Undo policy contains unknown request fields: "
            + ", ".join(sorted(unknown))
        )
    try:
        return request_type(**dict(arguments))
    except (TypeError, ValueError) as exc:
        raise CapabilityError("Undo policy produced invalid inverse arguments.") from exc


def _external_effect_arguments_fingerprint(request: object) -> str:
    encoded = json.dumps(
        _canonical_effect_value(request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_effect_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_effect_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("external effect mappings require text keys")
        return {
            str(key): _canonical_effect_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_effect_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_effect_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "external effect request contains an unsupported value: "
        f"{type(value).__name__}"
    )


def _external_effect_summary(
    capability: str,
    target_ids: tuple[tuple[str, str], ...],
    *,
    authorization_reference: str | None,
) -> str:
    target_fields = ",".join(name for name, _ in target_ids) or "none"
    authorization = "bound" if authorization_reference is not None else "none"
    summary = (
        f"{capability}; target_fields={target_fields}; authorization={authorization}"
    )
    if len(summary) > _MAX_EXTERNAL_EFFECT_SUMMARY_CHARACTERS:
        raise ValueError("external effect summary is too long")
    return summary


def _external_effect_record(
    *,
    capability: str,
    request: object,
    context: InvocationContext,
    authorization_reference: str | None,
    ttl: timedelta,
) -> ExternalEffectRecord:
    normalized_capability = capability.strip()
    tool_call_id = context.tool_call_id.strip() if context.tool_call_id else ""
    if not normalized_capability or not tool_call_id:
        raise ValueError("external effects require a capability and provider tool call ID")
    normalized_authorization_reference = (
        authorization_reference.strip()
        if authorization_reference is not None
        else None
    )
    if normalized_authorization_reference == "":
        normalized_authorization_reference = None
    if (
        normalized_authorization_reference is not None
        and len(normalized_authorization_reference)
        > _MAX_AUTHORIZATION_REFERENCE_CHARACTERS
    ):
        raise ValueError("external effect authorization reference is too long")
    target_ids = _target_ids(request, None)
    arguments_fingerprint = _external_effect_arguments_fingerprint(request)
    summary = _external_effect_summary(
        normalized_capability,
        target_ids,
        authorization_reference=normalized_authorization_reference,
    )
    identity = "\0".join(
        (
            context.request_id,
            context.actor_id,
            context.workspace_id or "",
            context.provider_thread_id or "",
            context.provider_turn_id or "",
            tool_call_id,
            normalized_capability,
        )
    )
    now = datetime.now(UTC)
    return ExternalEffectRecord(
        effect_id="eff_" + hashlib.sha256(identity.encode()).hexdigest()[:32],
        capability=normalized_capability,
        actor_id=context.actor_id,
        workspace_id=context.workspace_id,
        transport=context.transport,
        request_id=context.request_id,
        provider_thread_id=context.provider_thread_id,
        provider_turn_id=context.provider_turn_id,
        tool_call_id=tool_call_id,
        arguments_fingerprint=arguments_fingerprint,
        target_ids=target_ids,
        authorization_reference=normalized_authorization_reference,
        summary=summary,
        status=ExternalEffectStatus.PLANNED,
        action_id=None,
        created_at=now,
        updated_at=now,
        expires_at=now + ttl,
    )


def _external_effect_identity(record: ExternalEffectRecord) -> tuple[object, ...]:
    return (
        record.effect_id,
        record.capability,
        record.actor_id,
        record.workspace_id,
        record.transport,
        record.request_id,
        record.provider_thread_id,
        record.provider_turn_id,
        record.tool_call_id,
        record.arguments_fingerprint,
        record.target_ids,
        record.authorization_reference,
        record.summary,
    )


def _external_effect_target_ids_from_json(
    value: str,
) -> tuple[tuple[str, str], ...]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Invalid external effect target IDs") from exc
    if not isinstance(decoded, dict) or len(decoded) > _MAX_TARGET_IDS:
        raise RuntimeError("Invalid external effect target IDs")
    targets: list[tuple[str, str]] = []
    for key, item in decoded.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise RuntimeError("Invalid external effect target IDs")
        if (
            not key
            or len(key) > _MAX_TARGET_ID_CHARACTERS
            or not item
            or len(item) > _MAX_TARGET_ID_CHARACTERS
        ):
            raise RuntimeError("Invalid external effect target IDs")
        targets.append((key, item))
    return tuple(sorted(targets))


def _row_external_effect(row: sqlite3.Row) -> ExternalEffectRecord:
    arguments_fingerprint = str(row["arguments_fingerprint"])
    authorization_reference = (
        str(row["authorization_reference"])
        if row["authorization_reference"] is not None
        else None
    )
    summary = str(row["summary"])
    if not arguments_fingerprint or len(arguments_fingerprint) > 128:
        raise RuntimeError("Invalid external effect arguments fingerprint")
    if (
        authorization_reference is not None
        and len(authorization_reference) > _MAX_AUTHORIZATION_REFERENCE_CHARACTERS
    ):
        raise RuntimeError("Invalid external effect authorization reference")
    if not summary or len(summary) > _MAX_EXTERNAL_EFFECT_SUMMARY_CHARACTERS:
        raise RuntimeError("Invalid external effect summary")
    return ExternalEffectRecord(
        effect_id=str(row["effect_id"]),
        capability=str(row["capability"]),
        actor_id=str(row["actor_id"]),
        workspace_id=(
            str(row["workspace_id"]) if row["workspace_id"] is not None else None
        ),
        transport=str(row["transport"]),
        request_id=str(row["request_id"]),
        provider_thread_id=(
            str(row["provider_thread_id"])
            if row["provider_thread_id"] is not None
            else None
        ),
        provider_turn_id=(
            str(row["provider_turn_id"])
            if row["provider_turn_id"] is not None
            else None
        ),
        tool_call_id=str(row["tool_call_id"]),
        arguments_fingerprint=arguments_fingerprint,
        target_ids=_external_effect_target_ids_from_json(
            str(row["target_ids_json"])
        ),
        authorization_reference=authorization_reference,
        summary=summary,
        status=ExternalEffectStatus(str(row["status"])),
        action_id=(str(row["action_id"]) if row["action_id"] is not None else None),
        created_at=datetime.fromisoformat(str(row["created_at"])).astimezone(UTC),
        updated_at=datetime.fromisoformat(str(row["updated_at"])).astimezone(UTC),
        expires_at=datetime.fromisoformat(str(row["expires_at"])).astimezone(UTC),
    )


def _action_principal_kind(value: object) -> AgentPrincipalKind | None:
    if value is None:
        return None
    normalized = str(value)
    if normalized not in {
        "requester",
        "service",
        "system",
        "legacy_unknown",
    }:
        raise RuntimeError("Invalid action principal kind")
    return cast(AgentPrincipalKind, normalized)


def _row_record(row: sqlite3.Row) -> ActionRecord:
    target_ids_value = json.loads(str(row["target_ids_json"]))
    trigger_actor_ids_value = json.loads(str(row["trigger_actor_ids_json"]))
    undo_arguments_value = (
        json.loads(str(row["undo_arguments_json"]))
        if row["undo_arguments_json"] is not None
        else None
    )
    if not isinstance(target_ids_value, dict):
        raise RuntimeError("Invalid target ID record")
    if not isinstance(trigger_actor_ids_value, list) or any(
        not isinstance(value, str) for value in trigger_actor_ids_value
    ):
        raise RuntimeError("Invalid trigger actor ID record")
    if undo_arguments_value is not None and not isinstance(
        undo_arguments_value,
        dict,
    ):
        raise RuntimeError("Invalid undo argument record")
    return ActionRecord(
        action_id=str(row["action_id"]),
        capability=str(row["capability"]),
        actor_id=str(row["actor_id"]),
        workspace_id=(
            str(row["workspace_id"]) if row["workspace_id"] is not None else None
        ),
        transport=str(row["transport"]),
        request_id=str(row["request_id"]),
        principal_kind=_action_principal_kind(row["principal_kind"]),
        executor_principal_id=(
            str(row["executor_principal_id"])
            if row["executor_principal_id"] is not None
            else None
        ),
        delegator_principal_id=(
            str(row["delegator_principal_id"])
            if row["delegator_principal_id"] is not None
            else None
        ),
        trigger_actor_ids=tuple(trigger_actor_ids_value),
        requester_principal_id=(
            str(row["requester_principal_id"])
            if row["requester_principal_id"] is not None
            else None
        ),
        policy_id=(str(row["policy_id"]) if row["policy_id"] is not None else None),
        target_ids=tuple(
            sorted(
                (str(name), str(value))
                for name, value in target_ids_value.items()
            )
        ),
        status=ActionStatus(str(row["status"])),
        classification=ActionClassification(str(row["classification"])),
        undo_capability=(
            str(row["undo_capability"])
            if row["undo_capability"] is not None
            else None
        ),
        undo_arguments=undo_arguments_value,
        host_delivery=bool(row["host_delivery"]),
        created_at=datetime.fromisoformat(str(row["created_at"])).astimezone(UTC),
        expires_at=datetime.fromisoformat(str(row["expires_at"])).astimezone(UTC),
        undone_at=(
            datetime.fromisoformat(str(row["undone_at"])).astimezone(UTC)
            if row["undone_at"] is not None
            else None
        ),
        undo_action_id=(
            str(row["undo_action_id"])
            if row["undo_action_id"] is not None
            else None
        ),
    )
