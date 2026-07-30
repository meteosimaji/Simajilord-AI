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
from typing import Any, Protocol

from simajilord.core import (
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
        capability="discord.remove_thread_member",
        classification=ActionClassification.FULLY_REVERSIBLE,
        undo_capability="discord.add_thread_member",
        undo_arguments=_thread_member,
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
        "files.download_url",
        "files.write_text",
        "files.replace_text",
        "image.generate",
        "media.download",
        "media.save",
        "memory.forget",
        "memory.remember",
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
        self._initialize()

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
        now = datetime.now(UTC)
        record = ActionRecord(
            action_id=action_id,
            capability=capability,
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            transport=context.transport,
            request_id=context.request_id,
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

    def _initialize(self) -> None:
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
            connection.execute(
                "DELETE FROM agent_actions WHERE expires_at <= ?",
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
                    request_id, target_ids_json, status, classification,
                    undo_capability, undo_arguments_json, host_delivery,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO NOTHING
                """,
                (
                    record.action_id,
                    record.capability,
                    record.actor_id,
                    record.workspace_id,
                    record.transport,
                    record.request_id,
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
    ) -> ActionReceipt | None:
        """Record one successful write without changing the write's own result."""

        if capability == "action.undo":
            return None
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
            tracked = True
        except Exception:
            tracked = False
            undo_capability = None
            log.exception(
                "Action receipt persistence failed capability=%s request_id=%s",
                capability,
                context.request_id,
            )
        receipt = ActionReceipt(
            action_id=action_id if tracked else None,
            capability=capability,
            status="succeeded",
            tracked=tracked,
            undo_available=tracked and undo_capability is not None,
            undo_capability=undo_capability,
            classification=policy.classification,
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
                payload=payload,
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
        "audio_destination_id",
        "channel_id",
        "delivery_target_id",
        "destination_channel_id",
        "job_id",
        "memory_id",
        "message_id",
        "forum_id",
        "reply_to_event_id",
        "role_id",
        "source_channel_id",
        "source_message_id",
        "thread_id",
        "timer_id",
        "user_id",
    }
    selected: dict[str, str] = {}

    def visit(value: object) -> None:
        if len(selected) >= _MAX_TARGET_IDS or not dataclasses.is_dataclass(value):
            return
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if (
                field.name in names
                and isinstance(item, (str, int))
                and not isinstance(item, bool)
            ):
                text = str(item)
                if 0 < len(text) <= _MAX_TARGET_ID_CHARACTERS:
                    selected.setdefault(field.name, text)
            elif dataclasses.is_dataclass(item):
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


def _row_record(row: sqlite3.Row) -> ActionRecord:
    target_ids_value = json.loads(str(row["target_ids_json"]))
    undo_arguments_value = (
        json.loads(str(row["undo_arguments_json"]))
        if row["undo_arguments_json"] is not None
        else None
    )
    if not isinstance(target_ids_value, dict):
        raise RuntimeError("Invalid target ID record")
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
