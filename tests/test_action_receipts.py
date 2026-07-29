from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from simajilord.agent.actions import (
    ACTION_UNDO_ANY_GRANT,
    NON_UNDOABLE_ACTION_CAPABILITIES,
    ActionClassification,
    ActionReceiptService,
    ActionReceiptStore,
    ActionUndoRequest,
    action_policy,
    build_action_undo_endpoint,
)
from simajilord.agent.errors import AgentToolError
from simajilord.agent.tools import AgentToolCatalog
from simajilord.capabilities.audio import AudioControlResponse, AudioVolumeRequest
from simajilord.capabilities.focus_timer import (
    FocusTimerCancelRequest,
    FocusTimerCreateRequest,
    build_focus_timer_endpoints,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAnnouncementsSetRequest,
    ReadAloudContentModeSetRequest,
    ReadAloudPolicyResponse,
    ReadAloudSemanticsSetRequest,
    build_read_aloud_policy_endpoints,
)
from simajilord.core import (
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.focus_timer import FocusTimerService, FocusTimerStatus
from simajilord.services.read_aloud import ReadAloudContentMode, ReadAloudService


@dataclass(frozen=True, slots=True)
class ReactionRequest:
    channel_id: str
    message_id: str
    emoji: str


@dataclass(frozen=True, slots=True)
class ReactionResponse:
    channel_id: str
    message_id: str
    emoji: str
    reacted: bool
    changed: bool = True


@dataclass(frozen=True, slots=True)
class FileWriteRequest:
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class FileWriteResponse:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ChannelSettingRequest:
    channel_id: str
    topic: str | None = None
    slowmode_seconds: int | None = None
    expected_topic: str | None = None
    expected_slowmode_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ChannelSettingResponse:
    channel_id: str
    topic: str | None
    slowmode_seconds: int
    old_topic: str | None
    old_slowmode_seconds: int
    changed: bool = True


@dataclass(frozen=True, slots=True)
class MessageDeleteRequest:
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class MessageDeleteResponse:
    channel_id: str
    message_id: str
    changed: bool = True


@dataclass(frozen=True, slots=True)
class MessagesDeleteRequest:
    channel_id: str
    message_ids: str


@dataclass(frozen=True, slots=True)
class MessagesDeleteResponse:
    channel_id: str
    deleted_message_ids: tuple[str, ...]


class RecordingJournal:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, object]]] = []

    async def append(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None = None,
        workspace_id: str | None = None,
        transport: str | None = None,
        request_id: str | None = None,
    ) -> int:
        self.entries.append((kind, payload))
        return len(self.entries)


def _context(
    actor_id: str = "actor",
    *,
    request_id: str = "event",
    grants: frozenset[str] = frozenset({"messages"}),
) -> InvocationContext:
    return InvocationContext(
        actor_id=actor_id,
        workspace_id="guild",
        transport="agent",
        request_id=request_id,
        resource_ids=("channel",),
        grants=grants,
    )


def _reaction_registry(
    calls: list[tuple[str, ReactionRequest]],
) -> CapabilityRegistry:
    registry = CapabilityRegistry()

    async def add(
        request: ReactionRequest,
        _context: InvocationContext,
    ) -> ReactionResponse:
        calls.append(("add", request))
        return ReactionResponse(
            request.channel_id,
            request.message_id,
            request.emoji,
            True,
        )

    async def remove(
        request: ReactionRequest,
        _context: InvocationContext,
    ) -> ReactionResponse:
        calls.append(("remove", request))
        return ReactionResponse(
            request.channel_id,
            request.message_id,
            request.emoji,
            False,
        )

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.add_reaction",
                "Add reaction.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            ReactionRequest,
            ReactionResponse,
            add,
        )
    )
    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.remove_own_reaction",
                "Remove own reaction.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            ReactionRequest,
            ReactionResponse,
            remove,
        )
    )
    return registry


async def test_catalog_preserves_result_fields_and_adds_action_receipt(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ReactionRequest]] = []
    registry = _reaction_registry(calls)
    journal = RecordingJournal()
    service = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
        journal=journal,
    )
    registry.register(build_action_undo_endpoint(service))
    catalog = AgentToolCatalog(
        registry,
        (
            "discord.add_reaction",
            "discord.remove_own_reaction",
            "action.undo",
        ),
        required_grants={
            "discord.add_reaction": "messages",
            "discord.remove_own_reaction": "messages",
            "action.undo": "messages",
        },
        write_capabilities=(
            "discord.add_reaction",
            "discord.remove_own_reaction",
            "action.undo",
        ),
        action_receipts=service,
    )
    context = _context()
    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="discord_add_reaction",
        arguments={
            "channel_id": "channel",
            "message_id": "message",
            "emoji": "✅",
            "authorization_event_id": "event",
        },
        context=context,
        max_output_characters=4_000,
    )
    payload = json.loads(output.text)

    assert payload["channel_id"] == "channel"
    assert payload["message_id"] == "message"
    assert payload["reacted"] is True
    assert payload["action_receipt"] == {
        "action_id": payload["action_receipt"]["action_id"],
        "capability": "discord.add_reaction",
        "classification": "fully_reversible",
        "status": "succeeded",
        "undo_available": True,
        "undo_capability": "discord.remove_own_reaction",
    }
    assert calls == [
        (
            "add",
            ReactionRequest(
                channel_id="channel",
                message_id="message",
                emoji="✅",
            ),
        )
    ]
    assert journal.entries[0][0] == "agent.action.recorded"


async def test_host_post_receipt_keeps_only_ids_and_deletes_the_bot_post(
    tmp_path: Path,
) -> None:
    deleted: list[MessageDeleteRequest] = []
    registry = CapabilityRegistry()

    async def delete_own_message(
        request: MessageDeleteRequest,
        _context: InvocationContext,
    ) -> MessageDeleteResponse:
        deleted.append(request)
        return MessageDeleteResponse(request.channel_id, request.message_id)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.delete_own_message",
                "Delete one bot post.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            MessageDeleteRequest,
            MessageDeleteResponse,
            delete_own_message,
        )
    )
    path = tmp_path / "actions.sqlite3"
    service = ActionReceiptService(
        store=ActionReceiptStore(path),
        registry=registry,
    )

    receipt = await service.record_posted_message(
        channel_id="channel-20",
        message_id="message-30",
        context=_context("owner"),
    )

    assert receipt is not None
    assert receipt.capability == "discord.send_message"
    assert receipt.undo_available is True
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            """
            SELECT target_ids_json, undo_arguments_json
            FROM agent_actions
            WHERE action_id = ?
            """,
            (receipt.action_id,),
        ).fetchone()
    assert stored == (
        '{"channel_id":"channel-20","message_id":"message-30"}',
        '{"channel_id":"channel-20","message_id":"message-30"}',
    )
    retried = await service.record_posted_message(
        channel_id="channel-20",
        message_id="message-30",
        context=_context("owner"),
    )
    assert retried is not None
    assert retried.action_id == receipt.action_id
    assert await service.store.count() == 1

    await service.undo(
        ActionUndoRequest(),
        _context("owner", request_id="undo-host-post"),
    )

    assert deleted == [MessageDeleteRequest("channel-20", "message-30")]


async def test_multi_post_receipt_undoes_the_complete_response_as_one_action(
    tmp_path: Path,
) -> None:
    deleted: list[MessagesDeleteRequest] = []
    registry = CapabilityRegistry()

    async def delete_own_messages(
        request: MessagesDeleteRequest,
        _context: InvocationContext,
    ) -> MessagesDeleteResponse:
        deleted.append(request)
        return MessagesDeleteResponse(
            channel_id=request.channel_id,
            deleted_message_ids=tuple(request.message_ids.split(",")),
        )

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.delete_own_messages",
                "Delete one bot response.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            MessagesDeleteRequest,
            MessagesDeleteResponse,
            delete_own_messages,
        )
    )
    path = tmp_path / "actions.sqlite3"
    service = ActionReceiptService(
        store=ActionReceiptStore(path),
        registry=registry,
    )

    receipt = await service.record_posted_messages(
        channel_id="20",
        message_ids=("301", "302", "303"),
        context=_context("owner", request_id="multi-post"),
    )

    assert receipt is not None
    assert receipt.capability == "discord.send_messages"
    assert receipt.undo_available is True
    assert await service.store.count() == 1
    with sqlite3.connect(path) as connection:
        stored = connection.execute(
            """
            SELECT undo_arguments_json
            FROM agent_actions
            WHERE action_id = ?
            """,
            (receipt.action_id,),
        ).fetchone()
    assert stored == (
        '{"channel_id":"20","message_ids":"301,302,303"}',
    )

    undone = await service.undo(
        ActionUndoRequest(),
        _context("owner", request_id="undo-multi-post"),
    )

    assert undone.action_id == receipt.action_id
    assert deleted == [MessagesDeleteRequest("20", "301,302,303")]


async def test_cancelled_undo_releases_receipt_for_retry(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    registry = CapabilityRegistry()

    async def delete_own_message(
        request: MessageDeleteRequest,
        _context: InvocationContext,
    ) -> MessageDeleteResponse:
        started.set()
        await release.wait()
        return MessageDeleteResponse(request.channel_id, request.message_id)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.delete_own_message",
                "Delete one bot post.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            MessageDeleteRequest,
            MessageDeleteResponse,
            delete_own_message,
        )
    )
    service = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    receipt = await service.record_posted_message(
        channel_id="channel-20",
        message_id="message-30",
        context=_context("owner"),
    )
    assert receipt is not None

    first = asyncio.create_task(
        service.undo(
            ActionUndoRequest(receipt.action_id),
            _context("owner", request_id="cancelled-undo"),
        )
    )
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    release.set()
    retried = await service.undo(
        ActionUndoRequest(receipt.action_id),
        _context("owner", request_id="retried-undo"),
    )
    assert retried.status == "succeeded"


async def test_idless_undo_prefers_same_turn_action_over_final_confirmation(
    tmp_path: Path,
) -> None:
    deleted: list[MessageDeleteRequest] = []
    registry = CapabilityRegistry()
    timers = FocusTimerService(tmp_path / "timers.sqlite3")
    for capability in build_focus_timer_endpoints(timers):
        registry.register(capability)

    async def delete_own_message(
        request: MessageDeleteRequest,
        _context: InvocationContext,
    ) -> MessageDeleteResponse:
        deleted.append(request)
        return MessageDeleteResponse(request.channel_id, request.message_id)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.delete_own_message",
                "Delete one bot post.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            MessageDeleteRequest,
            MessageDeleteResponse,
            delete_own_message,
        )
    )
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    context = _context(request_id="same-turn")
    request = FocusTimerCreateRequest(
        duration_seconds=60,
        delivery_target_id="channel-20",
    )
    response = await registry.invoke("timer.create", request, context)
    timer_receipt = await receipts.record(
        capability="timer.create",
        request=request,
        response=response,
        context=context,
    )
    final_receipt = await receipts.record_posted_message(
        channel_id="channel-20",
        message_id="message-30",
        context=context,
    )
    assert timer_receipt is not None
    assert final_receipt is not None

    undone = await receipts.undo(
        ActionUndoRequest(),
        _context(request_id="undo-same-turn"),
    )

    assert undone.action_id == timer_receipt.action_id
    assert await timers.active(workspace_id="guild") == ()
    assert deleted == []


def test_catalog_rejects_any_write_without_an_explicit_action_policy(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry()

    async def write(
        request: FileWriteRequest,
        _context: InvocationContext,
    ) -> FileWriteResponse:
        return FileWriteResponse(request.path, "sha")

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "example.unclassified_write",
                "Unclassified test write.",
                RiskLevel.WRITE,
            ),
            FileWriteRequest,
            FileWriteResponse,
            write,
        )
    )
    service = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )

    with pytest.raises(
        AgentToolError,
        match=r"require an explicit Action policy: example\.unclassified_write",
    ):
        AgentToolCatalog(
            registry,
            ("example.unclassified_write",),
            required_grants={"example.unclassified_write": "write"},
            write_capabilities=("example.unclassified_write",),
            action_receipts=service,
        )


async def test_recent_undo_survives_restart_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.sqlite3"
    calls: list[tuple[str, ReactionRequest]] = []
    registry = _reaction_registry(calls)
    first = ActionReceiptService(
        store=ActionReceiptStore(path),
        registry=registry,
    )
    response = await registry.invoke(
        "discord.add_reaction",
        ReactionRequest("channel", "message", "👍"),
        _context(),
    )
    receipt = await first.record(
        capability="discord.add_reaction",
        request=ReactionRequest("channel", "message", "👍"),
        response=response,
        context=_context(),
    )
    assert receipt is not None

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE agent_actions SET status = 'undoing' WHERE action_id = ?",
            (receipt.action_id,),
        )
    restarted = ActionReceiptService(
        store=ActionReceiptStore(path),
        registry=registry,
    )
    undone = await restarted.undo(ActionUndoRequest(), _context(request_id="undo"))
    repeated = await restarted.undo(
        ActionUndoRequest(action_id=receipt.action_id),
        _context(request_id="undo-again"),
    )

    assert undone.action_id == receipt.action_id
    assert undone.status == "succeeded"
    assert repeated.status == "already_undone"
    assert repeated.undo_action_id == undone.undo_action_id
    assert [name for name, _ in calls] == ["add", "remove"]


async def test_idempotent_noop_does_not_offer_an_inverse_that_changes_state(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ReactionRequest]] = []
    registry = _reaction_registry(calls)
    service = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    receipt = await service.record(
        capability="discord.remove_own_reaction",
        request=ReactionRequest("channel", "message", "✅"),
        response=ReactionResponse(
            "channel",
            "message",
            "✅",
            False,
            changed=False,
        ),
        context=_context(),
    )

    assert receipt is not None
    assert receipt.undo_available is False
    with pytest.raises(UserError, match=r"action\.undo_not_found"):
        await service.undo(ActionUndoRequest(receipt.action_id), _context())


async def test_undo_is_actor_scoped_unless_host_grants_cross_actor_access(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ReactionRequest]] = []
    registry = _reaction_registry(calls)
    service = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    response = await registry.invoke(
        "discord.add_reaction",
        ReactionRequest("channel", "message", "🎉"),
        _context("owner"),
    )
    receipt = await service.record(
        capability="discord.add_reaction",
        request=ReactionRequest("channel", "message", "🎉"),
        response=response,
        context=_context("owner"),
    )
    assert receipt is not None

    with pytest.raises(UserError) as denied:
        await service.undo(
            ActionUndoRequest(receipt.action_id),
            _context("other", request_id="denied"),
        )
    assert denied.value.code == "action.undo_not_found"

    with pytest.raises(UserError, match=r"action\.undo_not_found"):
        await service.undo(
            ActionUndoRequest(),
            _context(
                "moderator",
                request_id="ambiguous",
                grants=frozenset({ACTION_UNDO_ANY_GRANT}),
            ),
        )
    allowed = await service.undo(
        ActionUndoRequest(receipt.action_id),
        _context(
            "moderator",
            request_id="allowed",
            grants=frozenset({ACTION_UNDO_ANY_GRANT}),
        ),
    )
    assert allowed.status == "succeeded"


async def test_timer_create_and_cancel_both_have_restart_safe_compensation(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry()
    timer_service = FocusTimerService(tmp_path / "timers.sqlite3")
    for capability in build_focus_timer_endpoints(timer_service):
        registry.register(capability)
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    context = _context()

    create_request = FocusTimerCreateRequest(
        duration_seconds=60,
        delivery_target_id="channel",
    )
    created = await registry.invoke("timer.create", create_request, context)
    create_receipt = await receipts.record(
        capability="timer.create",
        request=create_request,
        response=created,
        context=context,
    )
    assert create_receipt is not None
    await receipts.undo(
        ActionUndoRequest(create_receipt.action_id),
        _context(request_id="undo-create"),
    )
    assert await timer_service.active(workspace_id="guild") == ()

    second = await registry.invoke("timer.create", create_request, context)
    timer_id = second.timer.timer_id
    cancel_request = FocusTimerCancelRequest(timer_id)
    cancelled = await registry.invoke("timer.cancel", cancel_request, context)
    cancel_receipt = await receipts.record(
        capability="timer.cancel",
        request=cancel_request,
        response=cancelled,
        context=context,
    )
    assert cancel_receipt is not None
    await receipts.undo(
        ActionUndoRequest(cancel_receipt.action_id),
        _context(request_id="undo-cancel"),
    )
    active = await timer_service.active(workspace_id="guild")
    assert tuple(timer.timer_id for timer in active) == (timer_id,)
    assert active[0].status is FocusTimerStatus.SCHEDULED


async def test_cross_actor_admin_undo_runs_inverse_as_original_actor(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry()
    timer_service = FocusTimerService(tmp_path / "timers.sqlite3")
    for capability in build_focus_timer_endpoints(timer_service):
        registry.register(capability)
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    owner = _context("owner")
    request = FocusTimerCreateRequest(
        duration_seconds=60,
        delivery_target_id="channel",
    )
    created = await registry.invoke("timer.create", request, owner)
    receipt = await receipts.record(
        capability="timer.create",
        request=request,
        response=created,
        context=owner,
    )
    assert receipt is not None

    undone = await receipts.undo(
        ActionUndoRequest(receipt.action_id),
        _context(
            "moderator",
            request_id="admin-undo",
            grants=frozenset({ACTION_UNDO_ANY_GRANT}),
        ),
    )

    assert undone.status == "succeeded"
    assert await timer_service.active(workspace_id="guild") == ()


async def test_undo_capture_preserves_optional_null_scalar_state(
    tmp_path: Path,
) -> None:
    calls: list[ChannelSettingRequest] = []
    registry = CapabilityRegistry()

    async def update_settings(
        request: ChannelSettingRequest,
        _context: InvocationContext,
    ) -> ChannelSettingResponse:
        calls.append(request)
        return ChannelSettingResponse(
            channel_id=request.channel_id,
            topic=request.topic,
            slowmode_seconds=request.slowmode_seconds or 0,
            old_topic="new topic",
            old_slowmode_seconds=10,
        )

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.update_channel_settings",
                "Update channel settings.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            ChannelSettingRequest,
            ChannelSettingResponse,
            update_settings,
        )
    )
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    request = ChannelSettingRequest(
        channel_id="channel",
        topic="new topic",
        slowmode_seconds=10,
    )
    response = ChannelSettingResponse(
        channel_id="channel",
        topic="new topic",
        slowmode_seconds=10,
        old_topic=None,
        old_slowmode_seconds=0,
    )
    receipt = await receipts.record(
        capability="discord.update_channel_settings",
        request=request,
        response=response,
        context=_context(),
    )
    assert receipt is not None
    assert receipt.undo_available is True

    await receipts.undo(
        ActionUndoRequest(receipt.action_id),
        _context(request_id="undo-settings"),
    )

    assert calls == [
        ChannelSettingRequest(
            channel_id="channel",
            topic=None,
            slowmode_seconds=0,
            expected_topic="new topic",
            expected_slowmode_seconds=10,
        )
    ]


async def test_nonundoable_file_write_never_persists_body_and_store_is_bounded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actions.sqlite3"
    registry = CapabilityRegistry()
    store = ActionReceiptStore(
        path,
        ttl=timedelta(days=1),
        max_records=3,
        max_records_per_actor=2,
    )
    service = ActionReceiptService(store=store, registry=registry)
    secret_body = "PRIVATE-FILE-BODY-" * 200
    for index in range(4):
        receipt = await service.record(
            capability="files.write_text",
            request=FileWriteRequest(f"note-{index}.txt", secret_body),
            response=FileWriteResponse(f"note-{index}.txt", f"sha-{index}"),
            context=_context(request_id=f"write-{index}"),
        )
        assert receipt is not None
        assert receipt.undo_available is False
        assert receipt.classification is ActionClassification.NON_UNDOABLE

    assert await store.count() == 2
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT target_ids_json, undo_arguments_json
            FROM agent_actions
            """
        ).fetchall()
    serialized = json.dumps(rows, ensure_ascii=False)
    assert secret_body not in serialized
    assert all(row[1] is None for row in rows)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE agent_actions SET expires_at = '2000-01-01T00:00:00+00:00'"
        )
    restarted = ActionReceiptStore(
        path,
        ttl=timedelta(days=1),
        max_records=3,
        max_records_per_actor=2,
    )
    assert await restarted.count() == 0


@pytest.mark.parametrize(
    "capability",
    ("audio.set_volume", "discord.set_audio_volume"),
)
def test_volume_undo_captures_only_changed_previous_scalars(capability: str) -> None:
    policy = action_policy(capability)
    assert policy.undo_arguments is not None

    arguments = policy.undo_arguments(
        AudioVolumeRequest(music_percent=70),
        AudioControlResponse(
            action="volume",
            loop_mode=None,
            music_volume_percent=70,
            speech_volume_percent=120,
            previous_music_volume_percent=95,
            previous_speech_volume_percent=120,
        ),
    )

    assert policy.classification is ActionClassification.FULLY_REVERSIBLE
    assert policy.undo_capability == capability
    assert arguments == {
        "music_percent": 95,
        "expected_music_percent": 70,
    }


def test_scalar_noop_does_not_displace_the_latest_meaningful_undo() -> None:
    volume = action_policy("audio.set_volume")
    announcements = action_policy("speech.read_aloud_announcements_set")
    assert volume.undo_arguments is not None
    assert announcements.undo_arguments is not None

    assert (
        volume.undo_arguments(
            AudioVolumeRequest(music_percent=70),
            AudioControlResponse(
                action="volume",
                loop_mode=None,
                music_volume_percent=70,
                speech_volume_percent=100,
                previous_music_volume_percent=70,
                previous_speech_volume_percent=100,
            ),
        )
        is None
    )
    assert (
        announcements.undo_arguments(
            ReadAloudAnnouncementsSetRequest(join=True),
            ReadAloudPolicyResponse(
                dictionary=(),
                ignored_user_ids=(),
                ignored_role_ids=(),
                announce_join=True,
                announce_leave=False,
                announce_move=False,
                read_author_names=True,
                read_replies=True,
                read_attachments=True,
                previous_announce_join=True,
            ),
        )
        is None
    )


async def test_volume_action_undo_restores_previous_level(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    state = {"music": 100, "speech": 100}

    async def set_volume(
        request: AudioVolumeRequest,
        _context: InvocationContext,
    ) -> AudioControlResponse:
        previous_music = state["music"]
        previous_speech = state["speech"]
        if request.music_percent is not None:
            state["music"] = request.music_percent
        if request.speech_percent is not None:
            state["speech"] = request.speech_percent
        return AudioControlResponse(
            action="volume",
            loop_mode=None,
            music_volume_percent=state["music"],
            speech_volume_percent=state["speech"],
            previous_music_volume_percent=previous_music,
            previous_speech_volume_percent=previous_speech,
        )

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "audio.set_volume",
                "Set volume.",
                RiskLevel.WRITE,
                idempotency="idempotent_write",
            ),
            AudioVolumeRequest,
            AudioControlResponse,
            set_volume,
        )
    )
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    request = AudioVolumeRequest(music_percent=65)
    response = await registry.invoke("audio.set_volume", request, _context())
    receipt = await receipts.record(
        capability="audio.set_volume",
        request=request,
        response=response,
        context=_context(),
    )
    assert receipt is not None
    assert receipt.undo_available is True

    await receipts.undo(
        ActionUndoRequest(receipt.action_id),
        _context(request_id="undo-volume"),
    )

    assert state == {"music": 100, "speech": 100}


@pytest.mark.parametrize(
    ("capability", "action_request", "expected"),
    (
        (
            "speech.read_aloud_announcements_set",
            ReadAloudAnnouncementsSetRequest(join=True),
            {"join": False, "expected_join": True},
        ),
        (
            "discord.read_aloud_announcements_set",
            ReadAloudAnnouncementsSetRequest(move=True),
            {"move": False, "expected_move": True},
        ),
        (
            "speech.read_aloud_semantics_set",
            ReadAloudSemanticsSetRequest(author_names=True),
            {"author_names": False, "expected_author_names": True},
        ),
        (
            "discord.read_aloud_semantics_set",
            ReadAloudSemanticsSetRequest(replies=False),
            {"replies": True, "expected_replies": False},
        ),
    ),
)
def test_read_aloud_scalar_setters_have_static_undo(
    capability: str,
    action_request: object,
    expected: dict[str, object],
) -> None:
    policy = action_policy(capability)
    assert policy.undo_arguments is not None
    response = ReadAloudPolicyResponse(
        dictionary=(),
        ignored_user_ids=(),
        ignored_role_ids=(),
        announce_join=True,
        announce_leave=True,
        announce_move=True,
        read_author_names=True,
        read_replies=False,
        read_attachments=True,
        previous_announce_join=False,
        previous_announce_leave=True,
        previous_announce_move=False,
        previous_read_author_names=False,
        previous_read_replies=True,
        previous_read_attachments=False,
        previous_vc_members_only=True,
    )

    assert policy.classification is ActionClassification.FULLY_REVERSIBLE
    assert policy.undo_capability == capability
    assert policy.undo_arguments(action_request, response) == expected


async def test_read_aloud_content_mode_scalar_undo_restores_previous_mode(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry()
    read_aloud = ReadAloudService(tmp_path / "read-aloud.json")
    for capability in build_read_aloud_policy_endpoints(read_aloud):
        registry.register(capability)
    receipts = ActionReceiptService(
        store=ActionReceiptStore(tmp_path / "actions.sqlite3"),
        registry=registry,
    )
    await read_aloud.set_announcements(
        workspace_id="guild",
        join=True,
        leave=False,
        move=False,
    )
    request = ReadAloudContentModeSetRequest(ReadAloudContentMode.OFF)
    response = await registry.invoke(
        "speech.read_aloud_content_mode_set",
        request,
        _context(),
    )
    assert response.content_mode == ReadAloudContentMode.OFF.value
    assert response.previous_content_mode == ReadAloudContentMode.ALL.value
    assert response.previous_read_messages is True
    receipt = await receipts.record(
        capability="speech.read_aloud_content_mode_set",
        request=request,
        response=response,
        context=_context(),
    )
    assert receipt is not None
    assert receipt.undo_available is True

    await receipts.undo(
        ActionUndoRequest(receipt.action_id),
        _context(request_id="undo-content-mode"),
    )

    policy = read_aloud.policy("guild")
    assert policy.read_messages is True
    assert policy.announce_join is True
    assert policy.announce_leave is False
    assert policy.announce_move is False


def test_every_reversible_policy_has_one_static_inverse_and_no_collision() -> None:
    reversible = {
        "discord.add_reaction": "discord.remove_own_reaction",
        "discord.remove_own_reaction": "discord.add_reaction",
        "discord.send_message": "discord.delete_own_message",
        "discord.send_file": "discord.delete_own_message",
        "discord.post_expanded_message": "discord.delete_own_message",
        "discord.create_quote_image": "discord.delete_own_message",
        "discord.create_poll": "discord.delete_own_message",
        "timer.create": "timer.cancel",
        "timer.cancel": "timer.restore",
        "timer.restore": "timer.cancel",
        "audio.set_volume": "audio.set_volume",
        "discord.set_audio_volume": "discord.set_audio_volume",
        "speech.read_aloud_announcements_set": (
            "speech.read_aloud_announcements_set"
        ),
        "discord.read_aloud_announcements_set": (
            "discord.read_aloud_announcements_set"
        ),
        "speech.read_aloud_semantics_set": "speech.read_aloud_semantics_set",
        "discord.read_aloud_semantics_set": "discord.read_aloud_semantics_set",
        "speech.read_aloud_content_mode_set": (
            "speech.read_aloud_content_state_restore"
        ),
        "discord.read_aloud_content_mode_set": (
            "speech.read_aloud_content_state_restore"
        ),
    }
    for capability, inverse in reversible.items():
        policy = action_policy(capability)
        assert policy.undo_capability == inverse
        assert policy.classification is not ActionClassification.NON_UNDOABLE
        assert capability not in NON_UNDOABLE_ACTION_CAPABILITIES

    for capability in NON_UNDOABLE_ACTION_CAPABILITIES:
        policy = action_policy(capability)
        assert policy.classification is ActionClassification.NON_UNDOABLE
        assert policy.undo_capability is None
