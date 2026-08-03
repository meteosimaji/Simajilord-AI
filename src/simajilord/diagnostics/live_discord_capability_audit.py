"""Live, bounded audit of every registered Discord capability.

The audit has two deliberately separate phases:

* every endpoint is invoked once with a real connected client and an invalid
  workspace, proving typed dispatch and its first safety boundary without
  mutating Discord;
* safe reads and disposable writes are exercised in one explicitly selected
  guild. Created messages and threads stay in a temporary text channel; a
  temporary voice channel is used only to set and clear its status. Both
  channels are deleted in ``finally``.

Dangerous member actions, DMs, voice playback, and global presence changes are
never executed against a real target. Their live evidence is the structured,
safe rejection from the first phase.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import tempfile
from collections.abc import Callable
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import monotonic
from types import UnionType
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import discord

from simajilord.capabilities.file_scope import (
    file_provenance,
    file_workspace_id,
)
from simajilord.config import AgentFeatureAccess, load_settings
from simajilord.core import CapabilityEndpoint, InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.bot import _gateway_intents
from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.integrations.discord.platform_capabilities import (
    DiscordPlatformResourceKind,
)
from simajilord.runtime import SimajilordRuntime

EXPECTED_DISCORD_CAPABILITIES = 111
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
_PLATFORM_RESOURCE_KINDS: tuple[DiscordPlatformResourceKind, ...] = (
    "audit_log",
    "ban",
    "invite",
    "webhook",
    "scheduled_event",
    "scheduled_event_user",
    "emoji",
    "sticker",
    "soundboard",
    "application_emoji",
    "sku",
    "entitlement",
    "default_soundboard",
    "premium_sticker_pack",
    "automod_rule",
    "integration",
    "template",
    "stage_instance",
    "role_member_count",
    "onboarding",
    "welcome_screen",
    "widget",
    "vanity_invite",
    "active_thread",
    "guild_preview",
    "voice_region",
    "guild_voice_region",
    "prune_count",
    "subscription",
    "role_connection_metadata",
)


@dataclass(frozen=True, slots=True)
class LiveProbeRecord:
    """One secret-free invocation result."""

    capability: str
    case: str
    phase: Literal["guard", "live_read", "live_write", "cleanup"]
    outcome: Literal[
        "passed",
        "guarded",
        "environment_limited",
        "failed",
    ]
    duration_ms: float
    request_type: str
    response_type: str
    error_code: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    detail: str | None = None
    evidence: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class LiveAuditResult:
    """Serializable audit report."""

    started_at_iso: str
    completed_at_iso: str
    guild_id: str
    actor_id: str
    connected_guild_count: int
    capability_count: int
    guard_invocation_count: int
    live_invocation_count: int
    outcome_counts: dict[str, int]
    all_capabilities_invoked: bool
    required_cases_passed: bool
    cleanup_passed: bool
    passed: bool
    records: tuple[LiveProbeRecord, ...]


def minimal_request(request_type: type[Any]) -> Any:
    """Build the least-effect dataclass instance accepted by its constructor."""

    if not is_dataclass(request_type):
        raise TypeError(f"{request_type.__name__} is not a dataclass")
    hints = get_type_hints(request_type)
    values: dict[str, object] = {}
    for item in fields(request_type):
        if item.default is not MISSING or item.default_factory is not MISSING:
            continue
        values[item.name] = _minimal_value(hints.get(item.name, item.type))
    return request_type(**values)


def _minimal_value(annotation: object) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if not arguments:
            raise TypeError("Literal without values")
        return arguments[0]
    if origin in {Union, UnionType}:
        non_none = tuple(item for item in arguments if item is not type(None))
        if len(non_none) < len(arguments):
            return None
        if not non_none:
            raise TypeError("Union without values")
        return _minimal_value(non_none[0])
    if origin in {tuple, list, set, frozenset}:
        return origin()
    if origin is dict:
        return {}
    if annotation is str:
        return "0"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if annotation is bytes:
        return b""
    if annotation is Path:
        return Path("audit-probe")
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation))
    if isinstance(annotation, type) and is_dataclass(annotation):
        return minimal_request(annotation)
    raise TypeError(f"No minimal value for {annotation!r}")


class _AuditSession:
    def __init__(
        self,
        client: discord.Client,
        runtime: SimajilordRuntime,
        *,
        guild_id: str,
        actor_id: str,
    ) -> None:
        self.client = client
        self.runtime = runtime
        self.guild_id = guild_id
        self.actor_id = actor_id
        self.endpoints = {
            item.descriptor.name: item
            for item in build_discord_endpoints(client, runtime)
        }
        if len(self.endpoints) != EXPECTED_DISCORD_CAPABILITIES:
            raise RuntimeError(
                "Discord capability count drifted: "
                f"{len(self.endpoints)} != {EXPECTED_DISCORD_CAPABILITIES}"
            )
        for endpoint in self.endpoints.values():
            self.runtime.registry.register(endpoint)
        self.records: list[LiveProbeRecord] = []
        self._sequence = 0
        self._required_failed = False

    def context(
        self,
        *,
        valid: bool,
        origin_resource_id: str | None = None,
    ) -> InvocationContext:
        self._sequence += 1
        return InvocationContext(
            actor_id=self.actor_id if valid else "0",
            workspace_id=self.guild_id if valid else "0",
            transport="agent",
            request_id=f"discord-live-audit:{self._sequence}",
            origin_resource_id=origin_resource_id,
            resource_ids=(
                (origin_resource_id,) if origin_resource_id is not None else ()
            ),
            grants=frozenset(
                {
                    "audio",
                    "files",
                    "hive",
                    "image",
                    "message",
                    "moderation",
                    "quote",
                    "reaction",
                    "repost",
                }
            ),
            approvals=frozenset(self.endpoints),
            agent_task_id="tsk_live_discord_capability_audit",
        )

    async def invoke(
        self,
        capability: str,
        *,
        case: str,
        phase: Literal["guard", "live_read", "live_write", "cleanup"],
        context: InvocationContext,
        request: object | None = None,
        overrides: dict[str, object] | None = None,
        required: bool = False,
    ) -> object | None:
        selected = self.endpoints[capability]
        try:
            actual_request = (
                request
                if request is not None
                else selected.request_type(**(overrides or {}))
            )
        except Exception as exc:
            self._append_failure(
                selected,
                case=case,
                phase=phase,
                duration_ms=0.0,
                exc=exc,
                detail="request construction failed",
            )
            if required:
                self._required_failed = True
            return None
        started = monotonic()
        try:
            response = await self.runtime.registry.invoke(
                capability,
                actual_request,
                context,
            )
        except UserError as exc:
            duration_ms = (monotonic() - started) * 1_000
            outcome: Literal["guarded", "environment_limited", "failed"]
            if phase == "guard":
                outcome = "guarded"
            elif required:
                outcome = "failed"
                self._required_failed = True
            else:
                outcome = "environment_limited"
            self.records.append(
                LiveProbeRecord(
                    capability=capability,
                    case=case,
                    phase=phase,
                    outcome=outcome,
                    duration_ms=round(duration_ms, 3),
                    request_type=selected.request_type.__name__,
                    response_type=selected.response_type.__name__,
                    error_code=exc.code,
                    error_type=type(exc).__name__,
                    detail=_bounded_detail(str(exc)),
                )
            )
            return None
        except Exception as exc:
            duration_ms = (monotonic() - started) * 1_000
            self._append_failure(
                selected,
                case=case,
                phase=phase,
                duration_ms=duration_ms,
                exc=exc,
            )
            if required:
                self._required_failed = True
            return None
        self.records.append(
            LiveProbeRecord(
                capability=capability,
                case=case,
                phase=phase,
                outcome="passed",
                duration_ms=round((monotonic() - started) * 1_000, 3),
                request_type=selected.request_type.__name__,
                response_type=selected.response_type.__name__,
                evidence=_response_evidence(response),
            )
        )
        return response

    def _append_failure(
        self,
        selected: CapabilityEndpoint,
        *,
        case: str,
        phase: Literal["guard", "live_read", "live_write", "cleanup"],
        duration_ms: float,
        exc: Exception,
        detail: str | None = None,
    ) -> None:
        status = getattr(exc, "status", None)
        self.records.append(
            LiveProbeRecord(
                capability=selected.descriptor.name,
                case=case,
                phase=phase,
                outcome="failed",
                duration_ms=round(duration_ms, 3),
                request_type=selected.request_type.__name__,
                response_type=selected.response_type.__name__,
                error_type=type(exc).__name__,
                http_status=status if isinstance(status, int) else None,
                detail=_bounded_detail(detail or str(exc)),
            )
        )

    async def invoke_all_guards(self) -> None:
        """Invoke every endpoint against a non-existent Discord workspace."""

        for capability, selected in sorted(self.endpoints.items()):
            await self.invoke(
                capability,
                case="invalid-workspace-safety-boundary",
                phase="guard",
                context=self.context(valid=False),
                request=minimal_request(selected.request_type),
            )

    async def live_reads(
        self,
        *,
        channel_id: str | None = None,
        message_id: str | None = None,
        attachment_message_id: str | None = None,
        thread_id: str | None = None,
        poll_message_id: str | None = None,
        poll_answer_id: str | None = None,
    ) -> None:
        origin = channel_id or await self._readable_origin_channel_id()

        def context() -> InvocationContext:
            return self.context(valid=True, origin_resource_id=origin)

        core: tuple[tuple[str, dict[str, object]], ...] = (
            ("discord.list_servers", {}),
            ("discord.inspect_server", {}),
            ("discord.inspect_user", {"user_id": self.actor_id}),
            ("discord.list_voice_states", {}),
            ("discord.list_roles", {}),
            ("discord.list_channels", {"include_threads": True}),
            ("discord.list_members", {"limit": 5}),
            ("discord.inspect_application", {}),
            ("discord.read_aloud_status", {}),
            ("discord.read_aloud_policy_status", {}),
            ("discord.read_aloud_dictionary_list", {}),
        )
        for capability, overrides in core:
            await self.invoke(
                capability,
                case="required-live-read",
                phase="live_read",
                context=context(),
                overrides=overrides,
                required=True,
            )
        if channel_id is not None:
            message_cases: list[tuple[str, dict[str, object], bool]] = [
                (
                    "discord.inspect_channel",
                    {"channel_id": channel_id},
                    True,
                ),
                (
                    "discord.read_messages",
                    {"channel_id": channel_id, "limit": 5},
                    True,
                ),
                (
                    "discord.search_messages",
                    {
                        "channel_ids": (channel_id,),
                        "content": "Simajilord live audit",
                        "limit": 5,
                    },
                    True,
                ),
                (
                    "discord.list_pins",
                    {"channel_id": channel_id, "limit": 5},
                    True,
                ),
            ]
            if message_id is not None:
                message_cases.extend(
                    (
                        (
                            "discord.get_message",
                            {
                                "channel_id": channel_id,
                                "message_id": message_id,
                            },
                            True,
                        ),
                        (
                            "discord.expand_message",
                            {
                                "guild_id": self.guild_id,
                                "channel_id": channel_id,
                                "message_id": message_id,
                            },
                            True,
                        ),
                        (
                            "discord.list_reaction_users",
                            {
                                "channel_id": channel_id,
                                "message_id": message_id,
                                "emoji": "✅",
                            },
                            True,
                        ),
                        (
                            "discord.translate_message",
                            {
                                "channel_id": channel_id,
                                "message_id": message_id,
                                "target_language": "en",
                            },
                            False,
                        ),
                    )
                )
            if attachment_message_id is not None:
                message_cases.extend(
                    (
                        (
                            "discord.view_image_attachment",
                            {
                                "channel_id": channel_id,
                                "message_id": attachment_message_id,
                            },
                            True,
                        ),
                        (
                            "discord.import_attachment",
                            {
                                "channel_id": channel_id,
                                "message_id": attachment_message_id,
                                "destination_path": "audit/imported.png",
                            },
                            True,
                        ),
                        (
                            "discord.analyze_attachment",
                            {
                                "channel_id": channel_id,
                                "message_id": attachment_message_id,
                            },
                            False,
                        ),
                    )
                )
            if thread_id is not None:
                message_cases.append(
                    (
                        "discord.list_thread_members",
                        {"thread_id": thread_id, "limit": 5},
                        True,
                    )
                )
            if poll_message_id is not None:
                poll_response = await self.invoke(
                    "discord.get_message",
                    case="poll-results-live-read",
                    phase="live_read",
                    context=context(),
                    overrides={
                        "channel_id": channel_id,
                        "message_id": poll_message_id,
                    },
                    required=True,
                )
                poll_summary = getattr(poll_response, "poll", None)
                answer_ids = {
                    getattr(answer, "answer_id", None)
                    for answer in getattr(poll_summary, "answers", ())
                }
                if (
                    poll_summary is None
                    or not answer_ids
                    or (
                        poll_answer_id is not None
                        and poll_answer_id not in answer_ids
                    )
                ):
                    selected = self.endpoints["discord.get_message"]
                    self._append_failure(
                        selected,
                        case="poll-results-shape",
                        phase="live_read",
                        duration_ms=0.0,
                        exc=RuntimeError("poll result summary is incomplete"),
                    )
                    self._required_failed = True
            if poll_message_id is not None and poll_answer_id is not None:
                message_cases.append(
                    (
                        "discord.list_poll_voters",
                        {
                            "channel_id": channel_id,
                            "message_id": poll_message_id,
                            "answer_id": poll_answer_id,
                            "limit": 5,
                        },
                        True,
                    )
                )
            for capability, overrides, required in message_cases:
                await self.invoke(
                    capability,
                    case="fixture-live-read",
                    phase="live_read",
                    context=context(),
                    overrides=overrides,
                    required=required,
                )
        await self._platform_resource_reads(context)

    async def _platform_resource_reads(
        self,
        context_factory: Callable[[], InvocationContext],
    ) -> None:
        scheduled_event_id: str | None = None
        sku_id: str | None = None
        for kind in _PLATFORM_RESOURCE_KINDS:
            overrides: dict[str, object] = {"kind": kind, "limit": 5}
            if kind == "scheduled_event_user":
                overrides["resource_id"] = scheduled_event_id
            elif kind == "subscription":
                overrides.update(
                    {
                        "resource_id": sku_id,
                        "user_id": self.actor_id,
                    }
                )
            response = await self.invoke(
                "discord.list_platform_resources",
                case=f"platform-resource:{kind}",
                phase="live_read",
                context=context_factory(),
                overrides=overrides,
                required=False,
            )
            resources = getattr(response, "resources", ())
            if kind == "scheduled_event" and resources:
                scheduled_event_id = str(resources[0].resource_id)
            elif kind == "sku" and resources:
                sku_id = str(resources[0].resource_id)

    async def disposable_writes(self) -> None:
        """Run reversible writes only inside one temporary audit channel."""

        guild = self._guild()
        origin_id = await self._readable_origin_channel_id()
        create_context = self.context(valid=True, origin_resource_id=origin_id)
        channel_response = await self.invoke(
            "discord.create_channel",
            case="create-disposable-audit-channel",
            phase="live_write",
            context=create_context,
            overrides={
                "name": f"simajilord-api-audit-{datetime.now(UTC):%H%M%S}",
                "topic": "Temporary Simajilord capability audit; automatically removed.",
                "reason": "Automated disposable Discord capability audit",
            },
            required=True,
        )
        channel_id = getattr(channel_response, "channel_id", None)
        if not isinstance(channel_id, str):
            return
        channel = guild.get_channel(int(channel_id))
        if not isinstance(channel, discord.TextChannel):
            self._required_failed = True
            return
        cleanup_ok = False
        voice_channel_id: str | None = None
        voice_cleanup_ok = True
        try:
            def context() -> InvocationContext:
                return self.context(
                    valid=True,
                    origin_resource_id=channel_id,
                )

            if self.runtime.files is None:
                raise RuntimeError("Temporary audit runtime has no file sandbox")
            file_context = context()
            file_workspace = file_workspace_id(file_context)
            self.runtime.files.import_bytes(
                file_workspace,
                "audit/probe.png",
                _PNG_1X1,
                provenance=file_provenance(file_context),
            )
            self.runtime.files.import_bytes(
                file_workspace,
                "audit/probe.txt",
                b"Simajilord live audit attachment.\n",
                provenance=file_provenance(file_context),
            )
            voice_channel = await self.invoke(
                "discord.create_guild_resource",
                case="create-disposable-voice-channel",
                phase="live_write",
                context=context(),
                overrides={
                    "kind": "voice_channel",
                    "name": f"simajilord-voice-audit-{datetime.now(UTC):%H%M%S}",
                    "reason": "Disposable live audit voice channel",
                },
                required=True,
            )
            created_voice_channel_id = getattr(
                voice_channel,
                "resource_id",
                None,
            )
            if isinstance(created_voice_channel_id, str):
                voice_channel_id = created_voice_channel_id
                await self.invoke(
                    "discord.channel_operation",
                    case="set-disposable-voice-status",
                    phase="live_write",
                    context=context(),
                    overrides={
                        "operation": "set_voice_status",
                        "channel_id": voice_channel_id,
                        "name": "Simajilord API audit",
                        "reason": "Disposable live audit voice status",
                    },
                    required=True,
                )
                await self.invoke(
                    "discord.channel_operation",
                    case="clear-disposable-voice-status",
                    phase="cleanup",
                    context=context(),
                    overrides={
                        "operation": "set_voice_status",
                        "channel_id": voice_channel_id,
                        "name": "",
                        "reason": "Disposable live audit voice status cleanup",
                    },
                    required=True,
                )
            else:
                self._required_failed = True
            base = await self.invoke(
                "discord.send_message",
                case="send-suppressed-link-message",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "content": (
                        "Simajilord live audit https://example.invalid/ "
                        "(preview must be suppressed)"
                    ),
                },
                required=True,
            )
            message_id = getattr(base, "message_id", None)
            if not isinstance(message_id, str):
                return
            await self.invoke(
                "discord.edit_own_message",
                case="edit-owned-fixture-message",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "content": "Simajilord live audit message (edited)",
                },
                required=True,
            )
            await self.invoke(
                "discord.add_reaction",
                case="add-fixture-reaction",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "emoji": "✅",
                },
                required=True,
            )
            await self.invoke(
                "discord.pin_message",
                case="pin-fixture-message",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "reason": "Disposable live audit pin",
                },
                required=True,
            )
            reply = await self.invoke(
                "discord.reply_message",
                case="reply-to-selected-message",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "content": "Live audit reply",
                    "silent": True,
                },
                required=True,
            )
            embed = await self.invoke(
                "discord.send_embed",
                case="send-autonomous-embed",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "title": "Simajilord API audit",
                    "description": "Disposable embed delivery check.",
                    "reply_to_message_id": message_id,
                    "silent": True,
                },
                required=True,
            )
            attachment = await self.invoke(
                "discord.send_file",
                case="send-image-attachment",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "path": "audit/probe.png",
                    "caption": "Disposable image attachment",
                    "reply_to_message_id": message_id,
                    "silent": True,
                },
                required=True,
            )
            files = await self.invoke(
                "discord.send_files",
                case="send-multiple-attachments",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "attachments": (
                        _attachment_request(
                            self.endpoints["discord.send_files"],
                            "audit/probe.png",
                        ),
                        _attachment_request(
                            self.endpoints["discord.send_files"],
                            "audit/probe.txt",
                        ),
                    ),
                    "caption": "Disposable multi-file check",
                    "silent": True,
                },
                required=True,
            )
            poll = await self.invoke(
                "discord.create_poll",
                case="create-disposable-poll",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "question": "Did the capability audit reach Discord?",
                    "options": ("Yes", "No"),
                    "duration_hours": 1,
                },
                required=True,
            )
            thread = await self.invoke(
                "discord.create_thread",
                case="create-disposable-thread",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "name": "simajilord-live-audit-thread",
                    "message_id": message_id,
                    "reason": "Disposable live audit thread",
                },
                required=True,
            )
            thread_id = getattr(thread, "thread_id", None)
            if isinstance(thread_id, str):
                await self.invoke(
                    "discord.update_thread",
                    case="rename-disposable-thread",
                    phase="live_write",
                    context=context(),
                    overrides={
                        "thread_id": thread_id,
                        "name": "simajilord-live-audit-thread-updated",
                        "reason": "Disposable live audit thread update",
                    },
                    required=True,
                )
                await self.invoke(
                    "discord.add_thread_member",
                    case="add-requester-to-disposable-thread",
                    phase="live_write",
                    context=context(),
                    overrides={
                        "thread_id": thread_id,
                        "user_id": self.actor_id,
                        "reason": "Disposable live audit thread member",
                    },
                    required=False,
                )
            await self.invoke(
                "discord.update_channel_settings",
                case="update-disposable-channel",
                phase="live_write",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "topic": "Temporary Simajilord capability audit (updated).",
                    "slowmode_seconds": 0,
                    "reason": "Disposable live audit channel update",
                },
                required=True,
            )
            await self.invoke(
                "discord.forward_message",
                case="forward-fixture-message",
                phase="live_write",
                context=context(),
                overrides={
                    "source_channel_id": channel_id,
                    "source_message_id": message_id,
                    "destination_channel_id": channel_id,
                },
                required=False,
            )
            await self.invoke(
                "discord.post_expanded_message",
                case="post-expanded-fixture-message",
                phase="live_write",
                context=context(),
                overrides={
                    "source_guild_id": self.guild_id,
                    "source_channel_id": channel_id,
                    "source_message_id": message_id,
                    "destination_channel_id": channel_id,
                },
                required=True,
            )
            poll_message_id = getattr(poll, "message_id", None)
            poll_answer_id = await _first_poll_answer_id(
                channel,
                poll_message_id,
            )
            await self.live_reads(
                channel_id=channel_id,
                message_id=message_id,
                attachment_message_id=getattr(attachment, "message_id", None),
                thread_id=thread_id if isinstance(thread_id, str) else None,
                poll_message_id=(
                    poll_message_id if isinstance(poll_message_id, str) else None
                ),
                poll_answer_id=poll_answer_id,
            )
            await self.invoke(
                "discord.unpin_message",
                case="unpin-fixture-message",
                phase="cleanup",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "reason": "Disposable live audit cleanup",
                },
                required=True,
            )
            await self.invoke(
                "discord.remove_own_reaction",
                case="remove-fixture-reaction",
                phase="cleanup",
                context=context(),
                overrides={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "emoji": "✅",
                },
                required=True,
            )
            deletable_ids = tuple(
                str(value)
                for value in (
                    getattr(reply, "message_id", None),
                    getattr(embed, "message_id", None),
                    getattr(files, "message_id", None),
                )
                if value is not None
            )
            if deletable_ids:
                await self.invoke(
                    "discord.delete_own_messages",
                    case="delete-owned-fixture-messages",
                    phase="cleanup",
                    context=context(),
                    overrides={
                        "channel_id": channel_id,
                        "message_ids": ",".join(deletable_ids),
                    },
                    required=True,
                )
        finally:
            if voice_channel_id is not None:
                deleted_voice_channel = await self.invoke(
                    "discord.delete_guild_resource",
                    case="delete-disposable-voice-channel",
                    phase="cleanup",
                    context=self.context(
                        valid=True,
                        origin_resource_id=channel_id,
                    ),
                    overrides={
                        "kind": "channel",
                        "resource_id": voice_channel_id,
                        "reason": "Completed disposable live audit voice checks",
                    },
                    required=True,
                )
                voice_cleanup_ok = deleted_voice_channel is not None
            try:
                await channel.delete(
                    reason="Completed disposable Simajilord capability audit"
                )
            except discord.NotFound:
                cleanup_ok = True
            except Exception as exc:
                selected = self.endpoints["discord.delete_created_channel"]
                self._append_failure(
                    selected,
                    case="delete-disposable-audit-channel",
                    phase="cleanup",
                    duration_ms=0.0,
                    exc=exc,
                )
                self._required_failed = True
            else:
                cleanup_ok = True
                selected = self.endpoints["discord.delete_created_channel"]
                self.records.append(
                    LiveProbeRecord(
                        capability=selected.descriptor.name,
                        case="delete-disposable-audit-channel",
                        phase="cleanup",
                        outcome="passed",
                        duration_ms=0.0,
                        request_type=selected.request_type.__name__,
                        response_type=selected.response_type.__name__,
                    )
                )
        if not cleanup_ok or not voice_cleanup_ok:
            self._required_failed = True

    async def _readable_origin_channel_id(self) -> str:
        guild = self._guild()
        actor = guild.get_member(int(self.actor_id))
        if actor is None:
            actor = await guild.fetch_member(int(self.actor_id))
        bot = guild.me
        if bot is None:
            raise RuntimeError("Bot member is unavailable in the selected guild")
        for channel in guild.text_channels:
            actor_permissions = channel.permissions_for(actor)
            bot_permissions = channel.permissions_for(bot)
            if (
                actor_permissions.view_channel
                and bot_permissions.view_channel
                and actor_permissions.read_message_history
                and bot_permissions.read_message_history
            ):
                return str(channel.id)
        raise RuntimeError("No mutually readable text channel exists in the selected guild")

    def _guild(self) -> discord.Guild:
        guild = self.client.get_guild(int(self.guild_id))
        if guild is None:
            raise RuntimeError(
                f"Bot is not connected to selected guild {self.guild_id}"
            )
        return guild


def _attachment_request(
    endpoint: CapabilityEndpoint,
    path: str,
) -> object:
    hints = get_type_hints(endpoint.request_type)
    attachment_type = get_args(hints["attachments"])[0]
    return attachment_type(path=path)


async def _first_poll_answer_id(
    channel: discord.TextChannel,
    message_id: object,
) -> str | None:
    if not isinstance(message_id, str):
        return None
    try:
        message = await channel.fetch_message(int(message_id))
    except discord.DiscordException:
        return None
    poll = message.poll
    if poll is None or not poll.answers:
        return None
    return str(poll.answers[0].id)


def _bounded_detail(value: str) -> str | None:
    normalized = " ".join(value.split())
    return normalized[:300] if normalized else None


def _response_evidence(response: object) -> dict[str, object] | None:
    """Retain useful counts/IDs without copying message bodies or credentials."""

    evidence: dict[str, object] = {}
    for name in (
        "source_guild_id",
        "source_channel_id",
        "channel_id",
        "message_id",
        "thread_id",
        "answer_id",
        "answer_text",
        "kind",
        "complete",
        "next_offset",
        "presence_intent_enabled",
        "member_cache_complete",
    ):
        value = getattr(response, name, None)
        if (
            isinstance(value, (str, int, float, bool)) or value is None
        ) and hasattr(response, name):
            evidence[name] = value
    for name in (
        "servers",
        "channels",
        "roles",
        "members",
        "messages",
        "users",
        "voters",
        "resources",
        "voice_states",
        "attachments",
    ):
        value = getattr(response, name, None)
        if isinstance(value, (tuple, list)):
            evidence[f"{name}_count"] = len(value)
    poll = getattr(response, "poll", None)
    poll_answers = getattr(poll, "answers", ())
    if poll is not None and isinstance(poll_answers, (tuple, list)):
        evidence["poll_answer_ids"] = [
            str(answer.answer_id)
            for answer in poll_answers
            if getattr(answer, "answer_id", None) is not None
        ]
        evidence["poll_vote_counts"] = [
            int(answer.vote_count)
            for answer in poll_answers
            if isinstance(getattr(answer, "vote_count", None), int)
        ]
        evidence["poll_total_vote_count"] = getattr(
            poll,
            "total_vote_count",
            None,
        )
        evidence["poll_finalized"] = getattr(poll, "finalized", None)
    return evidence or None


async def run_live_audit(
    *,
    guild_id: str,
    actor_id: str,
    allow_safe_writes: bool,
    dotenv_path: Path = Path(".env"),
) -> LiveAuditResult:
    """Connect a temporary client and produce one complete audit report."""

    if not guild_id.isdigit() or int(guild_id) <= 0:
        raise ValueError("guild_id must be a positive Discord snowflake")
    if not actor_id.isdigit() or int(actor_id) <= 0:
        raise ValueError("actor_id must be a positive Discord snowflake")
    started_at = datetime.now(UTC)
    settings = load_settings(dotenv_path=dotenv_path)
    with tempfile.TemporaryDirectory(
        prefix="simajilord-live-discord-audit-"
    ) as temporary:
        isolated_settings = replace(
            settings,
            data_dir=Path(temporary),
            agent_enabled=False,
            agent_autonomy_enabled=False,
            agent_file_sandbox_enabled=True,
            image_generation_access=AgentFeatureAccess.DISABLED,
            hive_api_key=None,
        )
        runtime = SimajilordRuntime.build(isolated_settings)
        client = discord.Client(intents=_gateway_intents(settings))
        await client.login(settings.token)
        client_task = asyncio.create_task(
            client.connect(reconnect=True),
            name="discord-live-capability-audit-client",
        )
        try:
            await asyncio.wait_for(client.wait_until_ready(), timeout=45)
            session = _AuditSession(
                client,
                runtime,
                guild_id=guild_id,
                actor_id=actor_id,
            )
            await session.invoke_all_guards()
            await session.live_reads()
            if allow_safe_writes:
                await session.disposable_writes()
            guard_names = {
                item.capability for item in session.records if item.phase == "guard"
            }
            outcomes: dict[str, int] = {}
            for record in session.records:
                outcomes[record.outcome] = outcomes.get(record.outcome, 0) + 1
            cleanup_records = [
                item for item in session.records if item.phase == "cleanup"
            ]
            cleanup_passed = not allow_safe_writes or (
                bool(cleanup_records)
                and all(item.outcome == "passed" for item in cleanup_records)
            )
            all_capabilities_invoked = guard_names == set(session.endpoints)
            required_cases_passed = not session._required_failed
            passed = (
                all_capabilities_invoked
                and required_cases_passed
                and cleanup_passed
                and not any(item.outcome == "failed" for item in session.records)
            )
            return LiveAuditResult(
                started_at_iso=started_at.isoformat(),
                completed_at_iso=datetime.now(UTC).isoformat(),
                guild_id=guild_id,
                actor_id=actor_id,
                connected_guild_count=len(client.guilds),
                capability_count=len(session.endpoints),
                guard_invocation_count=len(
                    [item for item in session.records if item.phase == "guard"]
                ),
                live_invocation_count=len(
                    [item for item in session.records if item.phase != "guard"]
                ),
                outcome_counts=outcomes,
                all_capabilities_invoked=all_capabilities_invoked,
                required_cases_passed=required_cases_passed,
                cleanup_passed=cleanup_passed,
                passed=passed,
                records=tuple(session.records),
            )
        finally:
            await client.close()
            try:
                await asyncio.wait_for(client_task, timeout=15)
            except (asyncio.CancelledError, TimeoutError):
                client_task.cancel()
                await asyncio.gather(client_task, return_exceptions=True)
            await runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live-audit all typed Simajilord Discord capabilities.",
    )
    parser.add_argument("--guild-id", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument(
        "--allow-safe-writes",
        action="store_true",
        help="Create and automatically remove one disposable audit channel.",
    )
    parser.add_argument("--dotenv-path", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = asyncio.run(
        run_live_audit(
            guild_id=arguments.guild_id,
            actor_id=arguments.actor_id,
            allow_safe_writes=arguments.allow_safe_writes,
            dotenv_path=arguments.dotenv_path,
        )
    )
    payload = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result.passed:
        raise SystemExit("Live Discord capability audit failed.")


if __name__ == "__main__":
    main()
