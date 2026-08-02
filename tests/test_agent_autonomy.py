from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.agent import (
    AGENT_NO_ACTION_CONTENT,
    AGENT_REQUESTED_WRITE_CAPABILITIES,
    AgentAutonomyMode,
    AutonomyEnqueueResult,
    AutonomyEventBatch,
    AutonomyEventKind,
    AutonomyEventQueue,
    AutonomyLeaseLostError,
    AutonomyQueuedEvent,
)
from simajilord.agent.autonomy import (
    AutonomyDeliveryConflictError,
    AutonomyDeliveryReceiptState,
    AutonomyDeliveryRecord,
    AutonomyDeliverySpec,
)
from simajilord.agent.store import (
    AgentHostDeliveryRecord,
    AgentInterruptedMention,
    AgentPendingHostDelivery,
    AgentUnroutedTaskCandidate,
)
from simajilord.config import AgentFeatureAccess
from simajilord.core import ApprovalMode, InvocationContext
from simajilord.integrations.discord.cogs import (
    AgentAutonomyCog,
    AgentCog,
    ObservationCog,
    _agent_delivery_nonce,
    _agent_request_replay_barrier_reason,
)


def _queue(
    path: Path,
    *,
    maximum: int = 100,
    per_channel: int = 50,
    per_actor: int = 50,
) -> AutonomyEventQueue:
    return AutonomyEventQueue(
        path,
        max_pending_events=maximum,
        max_pending_events_per_channel=per_channel,
        max_pending_events_per_actor=min(per_actor, maximum),
    )


def _post_permissions(*, can_send: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        view_channel=True,
        read_message_history=True,
        send_messages=can_send,
        send_messages_in_threads=can_send,
        connect=True,
    )


async def _message_history(
    messages: tuple[discord.Message, ...],
) -> AsyncIterator[discord.Message]:
    for message in messages:
        yield message


async def _enqueue_messages(
    queue: AutonomyEventQueue,
    *,
    count: int,
    occurred_at: datetime,
    channel_id: str = "20",
) -> None:
    for index in range(count):
        result = await queue.enqueue(
            kind=AutonomyEventKind.MESSAGE_CREATE,
            deduplication_key=f"message:{channel_id}:{index}",
            workspace_id="10",
            channel_id=channel_id,
            actor_id=str(100 + index),
            message_id=str(1_000 + index),
            occurred_at=occurred_at + timedelta(seconds=index),
            enqueued_at=occurred_at + timedelta(seconds=index),
            payload={"content_length": index + 1},
        )
        assert result is AutonomyEnqueueResult.INSERTED


@pytest.mark.asyncio
async def test_same_channel_candidates_survive_restart_and_share_one_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    base = datetime(2026, 7, 29, 1, 0, tzinfo=UTC)
    queue = _queue(path)
    await _enqueue_messages(queue, count=5, occurred_at=base)

    assert (
        await queue.next_batch(
            debounce_seconds=10,
            candidate_limit=5,
            now=base + timedelta(seconds=9),
        )
        is None
    )

    restarted = _queue(path)
    batch = await restarted.next_batch(
        debounce_seconds=10,
        candidate_limit=5,
        now=base + timedelta(seconds=10),
    )

    assert batch is not None
    assert [event.message_id for event in batch.events] == [
        "1000",
        "1001",
        "1002",
        "1003",
        "1004",
    ]
    assert len({event.actor_id for event in batch.events}) == 5
    await restarted.mark_processed(batch, processed_at=base + timedelta(seconds=15))
    assert await restarted.pending_count() == 0


@pytest.mark.asyncio
async def test_candidate_limit_never_advances_past_uninspected_events(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 7, 29, 2, 0, tzinfo=UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=7, occurred_at=base)

    first = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=20),
    )
    assert first is not None
    assert len(first.events) == 5
    await queue.mark_processed(first, processed_at=base + timedelta(seconds=20))

    second = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=20),
    )
    assert second is not None
    assert [event.message_id for event in second.events] == ["1005", "1006"]
    await queue.mark_processed(second, processed_at=base + timedelta(seconds=20))
    assert await queue.pending_count() == 0


@pytest.mark.asyncio
async def test_queue_deduplicates_and_enforces_per_channel_admission(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path / "autonomy.sqlite3", maximum=3, per_channel=2)
    now = datetime.now(UTC)
    common = {
        "kind": AutonomyEventKind.REACTION_ADD,
        "workspace_id": "10",
        "channel_id": "20",
        "actor_id": "30",
        "message_id": "40",
        "occurred_at": now,
    }

    assert (
        await queue.enqueue(deduplication_key="reaction:1", **common)
        is AutonomyEnqueueResult.INSERTED
    )
    assert (
        await queue.enqueue(deduplication_key="reaction:1", **common)
        is AutonomyEnqueueResult.DUPLICATE
    )
    assert (
        await queue.enqueue(deduplication_key="reaction:2", **common)
        is AutonomyEnqueueResult.INSERTED
    )
    assert (
        await queue.enqueue(deduplication_key="reaction:3", **common)
        is AutonomyEnqueueResult.CHANNEL_QUEUE_FULL
    )
    assert (
        await queue.enqueue(
            deduplication_key="reaction:other-channel",
            **{**common, "channel_id": "21"},
        )
        is AutonomyEnqueueResult.INSERTED
    )
    assert (
        await queue.enqueue(
            deduplication_key="reaction:global-full",
            **{**common, "channel_id": "22"},
        )
        is AutonomyEnqueueResult.QUEUE_FULL
    )


@pytest.mark.asyncio
async def test_human_actor_cap_spans_channels_without_blocking_other_sources(
    tmp_path: Path,
) -> None:
    queue = _queue(
        tmp_path / "autonomy.sqlite3",
        maximum=6,
        per_channel=6,
        per_actor=2,
    )
    now = datetime.now(UTC)
    for index in range(2):
        assert (
            await queue.enqueue(
                kind=AutonomyEventKind.MESSAGE_CREATE,
                deduplication_key=f"actor-one:{index}",
                workspace_id="10",
                channel_id=str(20 + index),
                actor_id="30",
                message_id=str(40 + index),
                occurred_at=now,
            )
            is AutonomyEnqueueResult.INSERTED
        )
    assert (
        await queue.enqueue(
            kind=AutonomyEventKind.REACTION_ADD,
            deduplication_key="actor-one:third",
            workspace_id="10",
            channel_id="22",
            actor_id="30",
            message_id="42",
            occurred_at=now,
        )
        is AutonomyEnqueueResult.ACTOR_QUEUE_FULL
    )
    assert (
        await queue.enqueue(
            kind=AutonomyEventKind.MESSAGE_CREATE,
            deduplication_key="actor-two:first",
            workspace_id="10",
            channel_id="22",
            actor_id="31",
            message_id="43",
            occurred_at=now,
        )
        is AutonomyEnqueueResult.INSERTED
    )
    assert (
        await queue.enqueue(
            kind=AutonomyEventKind.TIMER_DUE,
            deduplication_key="actor-one:timer",
            workspace_id="10",
            channel_id="22",
            actor_id="30",
            message_id="44",
            occurred_at=now,
        )
        is AutonomyEnqueueResult.INSERTED
    )


@pytest.mark.asyncio
async def test_rate_limited_batch_is_retained_until_retry_time(tmp_path: Path) -> None:
    base = datetime(2026, 7, 29, 3, 0, tzinfo=UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=1, occurred_at=base)
    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=5),
    )
    assert batch is not None

    await queue.defer(
        batch,
        retry_after_seconds=60,
        now=base + timedelta(seconds=5),
    )

    assert await queue.pending_count() == 1
    assert (
        await queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            now=base + timedelta(seconds=64),
        )
        is None
    )
    assert (
        await queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            now=base + timedelta(seconds=65),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_batch_window_is_bounded_during_continuous_channel_traffic(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 7, 29, 3, 30, tzinfo=UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=12, occurred_at=base)

    assert (
        await queue.next_batch(
            debounce_seconds=5,
            candidate_limit=20,
            now=base + timedelta(seconds=4),
        )
        is None
    )
    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=20,
        now=base + timedelta(seconds=5),
    )

    assert batch is not None
    assert [event.message_id for event in batch.events] == [
        "1000",
        "1001",
        "1002",
        "1003",
        "1004",
        "1005",
    ]


@pytest.mark.asyncio
async def test_source_timestamp_does_not_bypass_local_batch_window(
    tmp_path: Path,
) -> None:
    admitted_at = datetime(2026, 7, 29, 4, 0, tzinfo=UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await queue.enqueue(
        kind=AutonomyEventKind.TIMER_DUE,
        deduplication_key="timer:overdue",
        workspace_id="10",
        channel_id="20",
        occurred_at=admitted_at - timedelta(hours=1),
        enqueued_at=admitted_at,
    )

    assert (
        await queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            now=admitted_at + timedelta(seconds=4),
        )
        is None
    )
    assert (
        await queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            now=admitted_at + timedelta(seconds=5),
        )
        is not None
    )


@pytest.mark.asyncio
async def test_batch_claim_is_exclusive_and_stale_worker_cannot_ack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    base = datetime(2026, 7, 29, 4, 30, tzinfo=UTC)
    first_queue = _queue(path)
    second_queue = _queue(path)
    await _enqueue_messages(first_queue, count=1, occurred_at=base)
    first = await first_queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=5),
    )
    assert first is not None

    assert (
        await second_queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            lease_seconds=60,
            now=base + timedelta(seconds=64),
        )
        is None
    )
    reclaimed = await second_queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=65),
    )
    assert reclaimed is not None
    assert reclaimed.batch_id == first.batch_id
    assert reclaimed.claim_token != first.claim_token

    with pytest.raises(AutonomyLeaseLostError):
        await first_queue.mark_processed(first)
    await second_queue.mark_processed(
        reclaimed,
        processed_at=base + timedelta(seconds=66),
    )
    assert await second_queue.pending_count() == 0


@pytest.mark.asyncio
async def test_retry_keeps_fixed_membership_when_new_channel_event_arrives(
    tmp_path: Path,
) -> None:
    base = datetime(2026, 7, 29, 5, 0, tzinfo=UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=1, occurred_at=base)
    first = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=5),
    )
    assert first is not None
    await queue.reschedule(
        first,
        retry_after_seconds=60,
        now=base + timedelta(seconds=5),
    )
    await queue.enqueue(
        kind=AutonomyEventKind.MESSAGE_CREATE,
        deduplication_key="message:late",
        workspace_id="10",
        channel_id="20",
        actor_id="30",
        message_id="2000",
        occurred_at=base + timedelta(seconds=6),
        enqueued_at=base + timedelta(seconds=6),
    )

    retried = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=65),
    )

    assert retried is not None
    assert retried.batch_id == first.batch_id
    assert [event.message_id for event in retried.events] == ["1000"]
    assert retried.attempt_count == 1
    await queue.mark_processed(
        retried,
        processed_at=base + timedelta(seconds=66),
    )
    late = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=65),
    )
    assert late is not None
    assert [event.message_id for event in late.events] == ["2000"]


@pytest.mark.asyncio
async def test_enqueue_prunes_processed_events_during_long_uptime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    queue = _queue(path)
    old = datetime.now(UTC) - timedelta(days=31)
    await _enqueue_messages(queue, count=1, occurred_at=old)
    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=old + timedelta(seconds=5),
    )
    assert batch is not None
    await queue.mark_processed(batch, processed_at=old + timedelta(seconds=5))

    await queue.enqueue(
        kind=AutonomyEventKind.MESSAGE_CREATE,
        deduplication_key="message:new",
        workspace_id="10",
        channel_id="20",
        actor_id="30",
        message_id="40",
        occurred_at=datetime.now(UTC),
    )

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT deduplication_key FROM autonomy_events ORDER BY sequence"
        ).fetchall()
    assert rows == [("message:new",)]


@pytest.mark.asyncio
async def test_queue_redacts_sensitive_external_event_metadata(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await queue.enqueue(
        kind=AutonomyEventKind.GITHUB_UPDATE,
        deduplication_key="github:delivery:1",
        workspace_id="10",
        channel_id="20",
        occurred_at=now,
        enqueued_at=now,
        payload={
            "repository": "owner/repo",
            "access_token": "private-token",
            "nested": {"client_secret": "private-secret"},
        },
    )

    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=now + timedelta(seconds=5),
    )

    assert batch is not None
    assert batch.events[0].payload == {
        "access_token": "[redacted]",
        "nested": {"client_secret": "[redacted]"},
        "repository": "owner/repo",
    }


@pytest.mark.asyncio
async def test_delivery_ledger_is_body_free_durable_and_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    base = datetime.now(UTC)
    queue = _queue(path)
    await _enqueue_messages(queue, count=1, occurred_at=base)
    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=5),
    )
    assert batch is not None
    contents = ("first chunk", "second chunk")
    specs = tuple(
        AutonomyDeliverySpec(
            purpose="response",
            chunk_index=index,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            nonce=_agent_delivery_nonce(batch.batch_id, index),
        )
        for index, content in enumerate(contents)
    )

    prepared = await queue.prepare_deliveries(
        batch,
        specs,
        now=base + timedelta(seconds=6),
    )
    sent = await queue.mark_delivery_sent(
        batch,
        purpose="response",
        chunk_index=0,
        message_id="301",
        sent_at=base + timedelta(seconds=7),
    )
    receipted = await queue.mark_delivery_receipted(
        batch,
        purpose="response",
        chunk_index=0,
        receipted_at=base + timedelta(seconds=8),
    )
    restarted = _queue(path)
    recovered = await restarted.prepare_deliveries(
        batch,
        specs,
        now=base + timedelta(seconds=9),
    )

    assert prepared[0].message_id is None
    assert sent.message_id == "301"
    assert receipted.receipt_state is AutonomyDeliveryReceiptState.RECORDED
    assert recovered[0].message_id == "301"
    assert recovered[1].message_id is None
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(autonomy_deliveries)"
            ).fetchall()
        }
        stored = connection.execute(
            """
            SELECT content_sha256, message_id, receipt_state
            FROM autonomy_deliveries
            ORDER BY chunk_index
            """
        ).fetchall()
    assert "content" not in columns
    assert "body" not in columns
    assert stored == [
        (specs[0].content_sha256, "301", "recorded"),
        (specs[1].content_sha256, None, "pending"),
    ]

    changed = (
        AutonomyDeliverySpec(
            purpose="response",
            chunk_index=0,
            content_sha256=hashlib.sha256(b"changed").hexdigest(),
            nonce=specs[0].nonce,
        ),
        specs[1],
    )
    with pytest.raises(AutonomyDeliveryConflictError):
        await restarted.prepare_deliveries(
            batch,
            changed,
            now=base + timedelta(seconds=10),
        )


def test_queue_migrates_pre_ledger_database_without_rewriting_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE autonomy_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                deduplication_key TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                enqueued_at TEXT NOT NULL,
                not_before TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                workspace_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                actor_id TEXT,
                message_id TEXT,
                payload_json TEXT NOT NULL,
                batch_id TEXT,
                claim_token TEXT,
                lease_until TEXT,
                processed_at TEXT,
                dead_lettered_at TEXT,
                dead_letter_reason TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO autonomy_events(
                deduplication_key, kind, occurred_at, enqueued_at, not_before,
                attempt_count, workspace_id, channel_id, payload_json
            ) VALUES ('legacy:1', 'timer.due', ?, ?, ?, 7, '10', '20', '{}')
            """,
            (datetime.now(UTC).isoformat(),) * 3,
        )

    queue = _queue(path)

    assert queue.path == path
    with sqlite3.connect(path) as connection:
        event_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(autonomy_events)"
            ).fetchall()
        }
        delivery_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'autonomy_deliveries'
            """
        ).fetchone()
        event_count = connection.execute(
            "SELECT COUNT(*) FROM autonomy_events"
        ).fetchone()[0]
        migrated_counts = connection.execute(
            "SELECT attempt_count, failure_count FROM autonomy_events"
        ).fetchone()
    assert {"failure_count", "deferral_count"} <= event_columns
    assert delivery_table == ("autonomy_deliveries",)
    assert event_count == 1
    assert migrated_counts == (7, 0)


@pytest.mark.asyncio
async def test_heartbeat_renewal_extends_whole_batch_and_rejects_stale_worker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "autonomy.sqlite3"
    base = datetime.now(UTC)
    first_queue = _queue(path)
    second_queue = _queue(path)
    await _enqueue_messages(first_queue, count=2, occurred_at=base)
    first = await first_queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=5),
    )
    assert first is not None

    renewed_until = await first_queue.renew_lease(
        first,
        lease_seconds=60,
        now=base + timedelta(seconds=50),
    )
    assert renewed_until == base + timedelta(seconds=110)
    assert (
        await second_queue.next_batch(
            debounce_seconds=5,
            candidate_limit=5,
            lease_seconds=60,
            now=base + timedelta(seconds=65),
        )
        is None
    )
    reclaimed = await second_queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=110),
    )
    assert reclaimed is not None
    assert reclaimed.claim_token != first.claim_token
    with pytest.raises(AutonomyLeaseLostError):
        await first_queue.renew_lease(
            first,
            lease_seconds=60,
            now=base + timedelta(seconds=111),
        )
    await second_queue.mark_processed(
        reclaimed,
        processed_at=base + timedelta(seconds=111),
    )


@pytest.mark.asyncio
async def test_expired_claim_cannot_settle_even_before_another_claim(
    tmp_path: Path,
) -> None:
    base = datetime.now(UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=1, occurred_at=base)
    batch = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        lease_seconds=60,
        now=base + timedelta(seconds=5),
    )
    assert batch is not None
    expired_at = base + timedelta(seconds=65)

    with pytest.raises(AutonomyLeaseLostError):
        await queue.renew_lease(
            batch,
            lease_seconds=60,
            now=expired_at,
        )
    with pytest.raises(AutonomyLeaseLostError):
        await queue.mark_processed(batch, processed_at=expired_at)
    with pytest.raises(AutonomyLeaseLostError):
        await queue.reschedule(
            batch,
            retry_after_seconds=60,
            now=expired_at,
        )
    with pytest.raises(AutonomyLeaseLostError):
        await queue.defer(
            batch,
            retry_after_seconds=60,
            now=expired_at,
        )
    with pytest.raises(AutonomyLeaseLostError):
        await queue.dead_letter(
            batch,
            reason="expired",
            failed_at=expired_at,
        )


@pytest.mark.asyncio
async def test_admission_deferral_does_not_consume_failure_budget(
    tmp_path: Path,
) -> None:
    base = datetime.now(UTC)
    queue = _queue(tmp_path / "autonomy.sqlite3")
    await _enqueue_messages(queue, count=1, occurred_at=base)
    first = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=5),
    )
    assert first is not None
    await queue.defer(
        first,
        retry_after_seconds=60,
        now=base + timedelta(seconds=6),
    )
    deferred = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=66),
    )
    assert deferred is not None
    assert deferred.attempt_count == 0
    assert deferred.deferral_count == 1
    await queue.reschedule(
        deferred,
        retry_after_seconds=60,
        now=base + timedelta(seconds=67),
    )
    failed = await queue.next_batch(
        debounce_seconds=5,
        candidate_limit=5,
        now=base + timedelta(seconds=127),
    )
    assert failed is not None
    assert failed.attempt_count == 1
    assert failed.deferral_count == 1


@pytest.mark.asyncio
async def test_observation_queues_human_message_but_not_mention_or_bot_message() -> None:
    bot_user = SimpleNamespace(id=999)
    bot = SimpleNamespace(user=bot_user)
    queue = SimpleNamespace(
        enqueue=AsyncMock(return_value=AutonomyEnqueueResult.INSERTED)
    )
    runtime = SimpleNamespace(
        agent=object(),
        settings=SimpleNamespace(
            agent_autonomy_enabled=True,
            agent_autonomy_mode=AgentAutonomyMode.ACT,
            agent_autonomy_guild_ids=frozenset({"10"}),
        ),
        journal=SimpleNamespace(append=AsyncMock()),
        autonomy_events=queue,
    )
    message = SimpleNamespace(
        id=40,
        guild=SimpleNamespace(id=10),
        channel=SimpleNamespace(id=20),
        author=SimpleNamespace(
            id=30,
            bot=False,
            display_name="person",
        ),
        webhook_id=None,
        mentions=[],
        attachments=[],
        content="hello",
        created_at=datetime.now(UTC),
    )
    cog = ObservationCog(bot, runtime)

    await cog.on_message(message)
    assert queue.enqueue.await_count == 1

    message.mentions = [bot_user]
    await cog.on_message(message)
    assert queue.enqueue.await_count == 1

    message.mentions = []
    message.author.bot = True
    await cog.on_message(message)
    assert queue.enqueue.await_count == 1


@pytest.mark.asyncio
async def test_observation_does_not_replay_prefix_commands_through_autonomy() -> None:
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_context=AsyncMock(return_value=SimpleNamespace(valid=True)),
    )
    queue = SimpleNamespace(
        enqueue=AsyncMock(return_value=AutonomyEnqueueResult.INSERTED)
    )
    runtime = SimpleNamespace(
        agent=object(),
        settings=SimpleNamespace(
            agent_autonomy_enabled=True,
            agent_autonomy_mode=AgentAutonomyMode.ACT,
            agent_autonomy_guild_ids=frozenset({"10"}),
        ),
        journal=SimpleNamespace(append=AsyncMock()),
        autonomy_events=queue,
    )
    message = SimpleNamespace(
        id=40,
        guild=SimpleNamespace(id=10),
        channel=SimpleNamespace(id=20),
        author=SimpleNamespace(id=30, bot=False, display_name="person"),
        webhook_id=None,
        mentions=[],
        attachments=[],
        content="!play example",
        created_at=datetime.now(UTC),
    )

    await ObservationCog(bot, runtime).on_message(message)

    bot.get_context.assert_awaited_once_with(message)
    queue.enqueue.assert_not_awaited()


@pytest.mark.asyncio
async def test_observation_uses_current_edit_and_excludes_mention_related_edits() -> None:
    bot_user = SimpleNamespace(id=999)
    bot = SimpleNamespace(user=bot_user)
    queue = SimpleNamespace(
        enqueue=AsyncMock(return_value=AutonomyEnqueueResult.INSERTED)
    )
    runtime = SimpleNamespace(
        agent=object(),
        settings=SimpleNamespace(
            agent_autonomy_enabled=True,
            agent_autonomy_mode=AgentAutonomyMode.ACT,
            agent_autonomy_guild_ids=frozenset({"10"}),
        ),
        journal=SimpleNamespace(append=AsyncMock()),
        autonomy_events=queue,
    )
    edited_at = "2026-07-29T10:20:30+00:00"
    current = SimpleNamespace(
        author=SimpleNamespace(id=30, bot=False),
        webhook_id=None,
        mentions=[],
    )
    cached = SimpleNamespace(
        author=SimpleNamespace(id=30, bot=False),
        webhook_id=None,
        mentions=[],
    )
    payload = SimpleNamespace(
        guild_id=10,
        channel_id=20,
        message_id=40,
        message=current,
        cached_message=cached,
        data={
            "content": "edited",
            "edited_timestamp": edited_at,
        },
    )
    cog = ObservationCog(bot, runtime)

    await cog.on_raw_message_edit(payload)

    call = queue.enqueue.await_args
    assert call.kwargs["occurred_at"] == datetime.fromisoformat(edited_at)
    assert call.kwargs["deduplication_key"].endswith(edited_at)
    assert call.kwargs["payload"]["content_changed"] is True

    current.mentions = [bot_user]
    payload.data["edited_timestamp"] = "2026-07-29T10:20:31+00:00"
    await cog.on_raw_message_edit(payload)
    assert queue.enqueue.await_count == 1

    current.mentions = []
    cached.mentions = [bot_user]
    payload.data["edited_timestamp"] = "2026-07-29T10:20:32+00:00"
    await cog.on_raw_message_edit(payload)
    assert queue.enqueue.await_count == 1

    cached.mentions = []
    payload.data = {"embeds": []}
    await cog.on_raw_message_edit(payload)
    assert queue.enqueue.await_count == 1


@pytest.mark.asyncio
async def test_agent_routes_edited_bot_mention_as_explicit_turn() -> None:
    cog = AgentCog(SimpleNamespace(), SimpleNamespace())
    cog._handle_mention = AsyncMock()  # type: ignore[method-assign]
    edited_at = "2026-07-29T10:20:30+00:00"
    current = SimpleNamespace(id=40)
    payload = SimpleNamespace(
        message_id=40,
        message=current,
        data={
            "content": "<@999> updated question",
            "edited_timestamp": edited_at,
        },
    )

    await cog.on_raw_message_edit(payload)

    cog._handle_mention.assert_awaited_once_with(
        current,
        event_id=f"discord:message-edit:40:{edited_at}",
        occurred_at=datetime.fromisoformat(edited_at),
        message_edited_at=datetime.fromisoformat(edited_at),
    )


@pytest.mark.asyncio
async def test_agent_recovers_prior_process_mention_as_a_fresh_turn() -> None:
    now = datetime.now(UTC)
    interrupted = AgentInterruptedMention(
        event_id="discord:message:40",
        public_reference_id="agt_0123456789abcdef0123",
        task_id="tsk_0123456789abcdef0123",
        channel_id="20",
        source_message_id="40",
        occurred_at=now - timedelta(minutes=1),
        started_at=now - timedelta(seconds=30),
    )
    source = Mock(spec=discord.Message)
    source.edited_at = None
    store = SimpleNamespace(
        unrouted_task_candidates=AsyncMock(return_value=()),
        interrupted_mentions=AsyncMock(return_value=(interrupted,)),
        fail_interrupted_mention=AsyncMock(return_value=False),
    )
    bot = SimpleNamespace(wait_until_ready=AsyncMock())
    cog = AgentCog(bot, SimpleNamespace(agent_store=store))
    cog._started_at = now
    cog._agent_host_channel = AsyncMock(return_value=object())  # type: ignore[method-assign]
    cog._agent_source_message = AsyncMock(return_value=source)  # type: ignore[method-assign]
    cog._handle_mention = AsyncMock()  # type: ignore[method-assign]

    await cog._recover_interrupted_mentions()

    bot.wait_until_ready.assert_awaited_once()
    cog._handle_mention.assert_awaited_once_with(
        source,
        event_id=interrupted.event_id,
        occurred_at=interrupted.occurred_at,
        message_edited_at=None,
        allow_routing=False,
    )
    store.fail_interrupted_mention.assert_awaited_once_with(
        interrupted.event_id,
        error_type="RecoverySkipped",
    )


@pytest.mark.asyncio
async def test_agent_does_not_replay_interrupted_mention_after_external_write() -> None:
    now = datetime.now(UTC)
    interrupted = AgentInterruptedMention(
        event_id="discord:message:41",
        public_reference_id="agt_1123456789abcdef0123",
        task_id="tsk_1123456789abcdef0123",
        channel_id="20",
        source_message_id="41",
        occurred_at=now - timedelta(minutes=1),
        started_at=now - timedelta(seconds=30),
    )
    store = SimpleNamespace(
        unrouted_task_candidates=AsyncMock(return_value=()),
        interrupted_mentions=AsyncMock(return_value=(interrupted,)),
        fail_interrupted_mention=AsyncMock(return_value=True),
    )
    receipts = SimpleNamespace(
        request_has_replay_barrier=AsyncMock(return_value=True),
    )
    journal = SimpleNamespace(append=AsyncMock(return_value=1))
    bot = SimpleNamespace(wait_until_ready=AsyncMock())
    cog = AgentCog(
        bot,
        SimpleNamespace(
            action_receipts=receipts,
            agent_store=store,
            journal=journal,
        ),
    )
    cog._started_at = now
    cog._handle_mention = AsyncMock()  # type: ignore[method-assign]

    await cog._recover_interrupted_mentions()

    receipts.request_has_replay_barrier.assert_awaited_once_with(interrupted.event_id)
    cog._handle_mention.assert_not_awaited()
    store.fail_interrupted_mention.assert_awaited_once_with(
        interrupted.event_id,
        error_type="RecoveryBlockedByExternalEffect",
    )
    assert journal.append.await_args.kwargs["kind"] == "agent.turn.recovery_blocked"


@pytest.mark.asyncio
async def test_interrupted_replay_barrier_scans_the_whole_task_trace() -> None:
    task_id = "tsk_2123456789abcdef0123"
    receipts = SimpleNamespace(
        request_has_replay_barrier=AsyncMock(return_value=False),
    )
    journal = SimpleNamespace(
        agent_trace=AsyncMock(
            return_value=(
                SimpleNamespace(
                    kind="agent.tool.started",
                    payload={"write": True},
                ),
            )
        )
    )
    runtime = SimpleNamespace(action_receipts=receipts, journal=journal)

    reason = await _agent_request_replay_barrier_reason(
        runtime,
        "discord:message:root",
        task_id=task_id,
    )

    assert reason == "legacy_write_trace"
    journal.agent_trace.assert_awaited_once_with(
        request_id=None,
        task_id=task_id,
        limit=1_000,
    )


@pytest.mark.asyncio
async def test_agent_recovers_unrouted_candidate_as_task_isolated_separate_work() -> None:
    now = datetime.now(UTC)
    candidate = AgentUnroutedTaskCandidate(
        event_id="discord:message:42",
        task_id="tsk_0123456789abcdef0123",
        public_reference_id="agt_0123456789abcdef0123",
        channel_id="20",
        source_message_id="42",
        occurred_at=now - timedelta(minutes=1),
        created_at=now - timedelta(seconds=30),
    )
    source = Mock(spec=discord.Message)
    source.edited_at = now - timedelta(seconds=20)
    store = SimpleNamespace(
        unrouted_task_candidates=AsyncMock(return_value=(candidate,)),
        default_task_candidate_to_separate=AsyncMock(return_value=True),
        request_by_public_reference_id=AsyncMock(return_value=SimpleNamespace()),
        fail_unrouted_task_candidate=AsyncMock(return_value=False),
    )
    cog = AgentCog(SimpleNamespace(), SimpleNamespace(agent_store=store))
    cog._started_at = now
    cog._agent_host_channel = AsyncMock(return_value=object())  # type: ignore[method-assign]
    cog._agent_source_message = AsyncMock(return_value=source)  # type: ignore[method-assign]
    cog._handle_mention = AsyncMock()  # type: ignore[method-assign]

    await cog._recover_unrouted_task_candidates()

    store.default_task_candidate_to_separate.assert_awaited_once_with(
        candidate.event_id,
        reason="startup_default_separate",
    )
    cog._handle_mention.assert_awaited_once_with(
        source,
        event_id=candidate.event_id,
        occurred_at=candidate.occurred_at,
        message_edited_at=source.edited_at,
        allow_routing=False,
    )
    store.fail_unrouted_task_candidate.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_terminalizes_unrecoverable_unrouted_candidate() -> None:
    now = datetime.now(UTC)
    candidate = AgentUnroutedTaskCandidate(
        event_id="discord:message:43",
        task_id="tsk_1123456789abcdef0123",
        public_reference_id="agt_1123456789abcdef0123",
        channel_id="20",
        source_message_id="43",
        occurred_at=now - timedelta(minutes=1),
        created_at=now - timedelta(seconds=30),
    )
    store = SimpleNamespace(
        unrouted_task_candidates=AsyncMock(return_value=(candidate,)),
        default_task_candidate_to_separate=AsyncMock(return_value=True),
        fail_unrouted_task_candidate=AsyncMock(return_value=True),
    )
    cog = AgentCog(SimpleNamespace(), SimpleNamespace(agent_store=store))
    cog._started_at = now
    cog._agent_host_channel = AsyncMock(return_value=object())  # type: ignore[method-assign]
    cog._agent_source_message = AsyncMock(return_value=None)  # type: ignore[method-assign]
    cog._handle_mention = AsyncMock()  # type: ignore[method-assign]

    await cog._recover_unrouted_task_candidates()

    store.fail_unrouted_task_candidate.assert_awaited_once_with(
        candidate.event_id,
        error_type="RecoverySourceUnavailable",
    )
    cog._handle_mention.assert_not_awaited()


@pytest.mark.asyncio
async def test_autonomy_cog_passes_whole_batch_under_bot_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    events = tuple(
        AutonomyQueuedEvent(
            sequence=index + 1,
            deduplication_key=f"message:{index}",
            kind=AutonomyEventKind.MESSAGE_CREATE,
            occurred_at=now + timedelta(milliseconds=index),
            enqueued_at=now + timedelta(milliseconds=index),
            not_before=now,
            attempt_count=0,
            workspace_id="10",
            channel_id="20",
            actor_id=str(100 + index),
            message_id=str(200 + index),
            payload={"content_length": index + 1},
        )
        for index in range(5)
    )
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:test",
        claim_token="claim",
        lease_until=now + timedelta(minutes=30),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=events,
    )
    agent = SimpleNamespace(
        respond=AsyncMock(
            return_value=SimpleNamespace(content=AGENT_NO_ACTION_CONTENT)
        )
    )
    runtime = SimpleNamespace(
        agent=agent,
        agent_store=SimpleNamespace(
            public_reference_id_for_event=AsyncMock(return_value=None),
            task_id_for_event=AsyncMock(return_value=None),
        ),
        settings=SimpleNamespace(
            agent_autonomy_mode=AgentAutonomyMode.ACT,
            agent_autonomy_guild_ids=frozenset({"10"}),
            agent_file_sandbox_enabled=False,
            agent_web_search_access=AgentFeatureAccess.DISABLED,
            agent_safe_compute_access=AgentFeatureAccess.DISABLED,
                agent_admin_user_ids=frozenset(),
                agent_conversation_compatibility_epoch=4,
                image_generation_access=AgentFeatureAccess.DISABLED,
        ),
        files=None,
        compute=None,
        moderation=SimpleNamespace(provider=None),
        image=SimpleNamespace(provider=None),
        registry=SimpleNamespace(
            all=lambda: tuple(
                SimpleNamespace(
                    descriptor=SimpleNamespace(
                        name=name,
                        approval=ApprovalMode.WHEN_REQUESTED,
                    )
                )
                for name in AGENT_REQUESTED_WRITE_CAPABILITIES
            )
        ),
    )
    bot_member = SimpleNamespace(id=999)
    guild = SimpleNamespace(id=10, me=bot_member)
    channel = Mock(spec=discord.TextChannel)
    channel.permissions_for = Mock(return_value=_post_permissions())
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_guild=lambda guild_id: guild if guild_id == 10 else None,
        get_channel=lambda channel_id: channel if channel_id == 20 else None,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.agent_readable_channel_ids",
        lambda *args, **kwargs: ("20",),
    )
    cog = AgentAutonomyCog(bot, runtime)

    await cog._inspect(batch)

    request = agent.respond.await_args.args[0]
    assert request.actor_id == "999"
    assert f":task:{request.task_id}" in request.conversation_id
    assert request.conversation_id.index(f":task:{request.task_id}") < (
        request.conversation_id.index(":profile:")
    )
    assert len(request.events) == 5
    assert {event.payload["source_actor_id"] for event in request.events} == {
        "100",
        "101",
        "102",
        "103",
        "104",
    }
    assert set(AGENT_REQUESTED_WRITE_CAPABILITIES) <= request.approvals


@pytest.mark.asyncio
async def test_autonomy_host_reply_receipts_only_posted_ids_for_source_actor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:host-post",
        claim_token="claim",
        lease_until=now + timedelta(minutes=30),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(
            AutonomyQueuedEvent(
                sequence=1,
                deduplication_key="message:201",
                kind=AutonomyEventKind.MESSAGE_CREATE,
                occurred_at=now,
                enqueued_at=now,
                not_before=now,
                attempt_count=0,
                workspace_id="10",
                channel_id="20",
                actor_id="101",
                message_id="201",
                payload={"content_length": 5},
            ),
        ),
    )
    agent = SimpleNamespace(
        respond=AsyncMock(return_value=SimpleNamespace(content="自主返信"))
    )
    receipts = SimpleNamespace(record_posted_messages=AsyncMock())
    delivery_records: dict[int, AutonomyDeliveryRecord] = {}

    async def prepare_deliveries(
        _: AutonomyEventBatch,
        specs: tuple[AutonomyDeliverySpec, ...],
    ) -> tuple[AutonomyDeliveryRecord, ...]:
        for spec in specs:
            delivery_records[spec.chunk_index] = AutonomyDeliveryRecord(
                batch_id=batch.batch_id,
                chunk_index=spec.chunk_index,
                purpose=spec.purpose,
                channel_id=batch.channel_id,
                content_sha256=spec.content_sha256,
                nonce=spec.nonce,
                message_id=None,
                receipt_state=AutonomyDeliveryReceiptState.PENDING,
                prepared_at=now,
                sent_at=None,
                receipted_at=None,
            )
        return tuple(delivery_records[index] for index in range(len(specs)))

    async def mark_delivery_sent(
        _: AutonomyEventBatch,
        *,
        purpose: str,
        chunk_index: int,
        message_id: str,
    ) -> AutonomyDeliveryRecord:
        previous = delivery_records[chunk_index]
        current = AutonomyDeliveryRecord(
            batch_id=previous.batch_id,
            chunk_index=chunk_index,
            purpose=purpose,
            channel_id=previous.channel_id,
            content_sha256=previous.content_sha256,
            nonce=previous.nonce,
            message_id=message_id,
            receipt_state=previous.receipt_state,
            prepared_at=previous.prepared_at,
            sent_at=now,
            receipted_at=None,
        )
        delivery_records[chunk_index] = current
        return current

    async def mark_delivery_receipted(
        _: AutonomyEventBatch,
        *,
        purpose: str,
        chunk_index: int,
    ) -> AutonomyDeliveryRecord:
        previous = delivery_records[chunk_index]
        current = AutonomyDeliveryRecord(
            batch_id=previous.batch_id,
            chunk_index=chunk_index,
            purpose=purpose,
            channel_id=previous.channel_id,
            content_sha256=previous.content_sha256,
            nonce=previous.nonce,
            message_id=previous.message_id,
            receipt_state=AutonomyDeliveryReceiptState.RECORDED,
            prepared_at=previous.prepared_at,
            sent_at=previous.sent_at,
            receipted_at=now,
        )
        delivery_records[chunk_index] = current
        return current

    autonomy_events = SimpleNamespace(
        prepare_deliveries=AsyncMock(side_effect=prepare_deliveries),
        mark_delivery_sent=AsyncMock(side_effect=mark_delivery_sent),
        mark_delivery_receipted=AsyncMock(
            side_effect=mark_delivery_receipted
        ),
    )
    runtime = SimpleNamespace(
        agent=agent,
        agent_store=SimpleNamespace(
            public_reference_id_for_event=AsyncMock(return_value=None),
            task_id_for_event=AsyncMock(return_value=None),
        ),
        action_receipts=receipts,
        autonomy_events=autonomy_events,
        settings=SimpleNamespace(
            agent_autonomy_mode=AgentAutonomyMode.ACT,
            agent_autonomy_guild_ids=frozenset({"10"}),
            agent_file_sandbox_enabled=False,
            agent_web_search_access=AgentFeatureAccess.DISABLED,
            agent_safe_compute_access=AgentFeatureAccess.DISABLED,
                agent_admin_user_ids=frozenset(),
                agent_conversation_compatibility_epoch=4,
                image_generation_access=AgentFeatureAccess.DISABLED,
        ),
        files=None,
        compute=None,
        moderation=SimpleNamespace(provider=None),
        image=SimpleNamespace(provider=None),
        registry=SimpleNamespace(
            all=lambda: tuple(
                SimpleNamespace(
                    descriptor=SimpleNamespace(
                        name=name,
                        approval=ApprovalMode.WHEN_REQUESTED,
                    )
                )
                for name in AGENT_REQUESTED_WRITE_CAPABILITIES
            )
        ),
        journal=SimpleNamespace(append=AsyncMock()),
    )
    posted = Mock(spec=discord.Message)
    posted.id = 301
    target = Mock(spec=discord.Message)
    target.author = SimpleNamespace(id=101, bot=False)
    target.reply = AsyncMock(return_value=posted)
    channel = Mock(spec=discord.TextChannel)
    channel.fetch_message = AsyncMock(return_value=target)
    channel.send = AsyncMock()
    channel.history = Mock(return_value=_message_history(()))
    bot_member = SimpleNamespace(id=999)
    channel.permissions_for = Mock(return_value=_post_permissions())
    guild = SimpleNamespace(id=10, me=bot_member)
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_guild=lambda guild_id: guild if guild_id == 10 else None,
        get_channel=lambda channel_id: channel if channel_id == 20 else None,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.agent_readable_channel_ids",
        lambda *args, **kwargs: ("20",),
    )

    acted = await AgentAutonomyCog(bot, runtime)._inspect(batch)

    assert acted is True
    target.reply.assert_awaited_once()
    assert target.reply.await_args.kwargs["nonce"] == _agent_delivery_nonce(
        batch.batch_id,
        0,
    )
    assert target.reply.await_args.kwargs["suppress_embeds"] is True
    channel.send.assert_not_awaited()
    receipt_call = receipts.record_posted_messages.await_args
    assert receipt_call.kwargs["channel_id"] == "20"
    assert receipt_call.kwargs["message_ids"] == ("301",)
    context = receipt_call.kwargs["context"]
    assert context.actor_id == "101"
    assert context.workspace_id == "10"
    assert context.origin_resource_id == "20"


@pytest.mark.asyncio
async def test_autonomy_delivery_reconciles_saved_id_then_nonce_without_resend() -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:reconcile",
        claim_token="claim",
        lease_until=now + timedelta(minutes=1),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(),
    )
    # Identical bodies are intentional: only the event-owned nonce may recover
    # the missing second chunk.
    contents = ("same", "same")
    records = tuple(
        AutonomyDeliveryRecord(
            batch_id=batch.batch_id,
            chunk_index=index,
            purpose="response",
            channel_id="20",
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            nonce=_agent_delivery_nonce(batch.batch_id, index),
            message_id="301" if index == 0 else None,
            receipt_state=(
                AutonomyDeliveryReceiptState.RECORDED
                if index == 0
                else AutonomyDeliveryReceiptState.PENDING
            ),
            prepared_at=now,
            sent_at=now if index == 0 else None,
            receipted_at=now if index == 0 else None,
        )
        for index, content in enumerate(contents)
    )
    saved = Mock(spec=discord.Message)
    saved.id = 301
    saved.author = SimpleNamespace(id=999)
    saved.nonce = records[0].nonce
    saved.content = contents[0]
    recovered = Mock(spec=discord.Message)
    recovered.id = 302
    recovered.author = SimpleNamespace(id=999)
    recovered.nonce = records[1].nonce
    recovered.content = contents[1]
    unrelated = Mock(spec=discord.Message)
    unrelated.id = 399
    unrelated.author = SimpleNamespace(id=999)
    unrelated.nonce = "another-event"
    unrelated.content = contents[1]
    channel = Mock(spec=discord.TextChannel)
    channel.fetch_message = AsyncMock(return_value=saved)
    channel.history = Mock(
        return_value=_message_history((saved, unrelated, recovered))
    )
    channel.send = AsyncMock()
    updated = AutonomyDeliveryRecord(
        batch_id=records[1].batch_id,
        chunk_index=1,
        purpose="response",
        channel_id="20",
        content_sha256=records[1].content_sha256,
        nonce=records[1].nonce,
        message_id="302",
        receipt_state=AutonomyDeliveryReceiptState.PENDING,
        prepared_at=now,
        sent_at=now,
        receipted_at=None,
    )
    receipted = AutonomyDeliveryRecord(
        batch_id=updated.batch_id,
        chunk_index=1,
        purpose="response",
        channel_id="20",
        content_sha256=updated.content_sha256,
        nonce=updated.nonce,
        message_id="302",
        receipt_state=AutonomyDeliveryReceiptState.RECORDED,
        prepared_at=now,
        sent_at=now,
        receipted_at=now,
    )
    queue = SimpleNamespace(
        prepare_deliveries=AsyncMock(return_value=records),
        mark_delivery_sent=AsyncMock(return_value=updated),
        mark_delivery_receipted=AsyncMock(return_value=receipted),
    )
    receipts = SimpleNamespace(record_posted_messages=AsyncMock())
    runtime = SimpleNamespace(
        autonomy_events=queue,
        action_receipts=receipts,
    )
    context = InvocationContext(
        actor_id="101",
        workspace_id="10",
        transport="agent",
        request_id=batch.batch_id,
        resource_ids=("20",),
        origin_resource_id="20",
    )

    await AgentAutonomyCog(SimpleNamespace(), runtime)._deliver_response(
        batch,
        channel=channel,
        target=None,
        messages=contents,
        context=context,
        bot_user_id=999,
    )

    channel.fetch_message.assert_awaited_once_with(301)
    channel.send.assert_not_awaited()
    queue.mark_delivery_sent.assert_awaited_once()
    assert queue.mark_delivery_sent.await_args.kwargs["chunk_index"] == 1
    assert queue.mark_delivery_sent.await_args.kwargs["message_id"] == "302"
    receipts.record_posted_messages.assert_awaited_once()
    assert receipts.record_posted_messages.await_args.kwargs["message_ids"] == (
        "301",
        "302",
    )
    queue.mark_delivery_receipted.assert_awaited_once()


@pytest.mark.asyncio
async def test_mention_recovery_does_not_reuse_saved_identical_chunk() -> None:
    now = datetime.now(UTC)
    pending = AgentPendingHostDelivery(
        event_id="discord:message:201",
        public_reference_id="agt_0123456789abcdef0123",
        actor_id="101",
        workspace_id="10",
        channel_id="20",
        source_message_id="201",
        response_content="same",
        occurred_at=now,
        completed_at=now,
    )
    body_hash = hashlib.sha256(b"same").hexdigest()
    records = tuple(
        AgentHostDeliveryRecord(
            event_id=pending.event_id,
            purpose="response",
            chunk_index=index,
            content_sha256=body_hash,
            channel_id="20",
            message_id="301" if index == 0 else None,
            receipted_at=now if index == 0 else None,
            created_at=now,
            updated_at=now,
        )
        for index in range(2)
    )
    saved = Mock(spec=discord.Message)
    saved.id = 301
    saved.author = SimpleNamespace(id=999)
    saved.nonce = _agent_delivery_nonce(pending.event_id, 0)
    saved.content = "same"
    recovered = Mock(spec=discord.Message)
    recovered.id = 302
    recovered.author = SimpleNamespace(id=999)
    recovered.nonce = _agent_delivery_nonce(pending.event_id, 1)
    recovered.content = "same"
    unrelated = Mock(spec=discord.Message)
    unrelated.id = 399
    unrelated.author = SimpleNamespace(id=999)
    unrelated.nonce = "another-event"
    unrelated.content = "same"
    channel = Mock(spec=discord.TextChannel)
    channel.history = Mock(
        return_value=_message_history((saved, unrelated, recovered))
    )

    async def record_message(
        *,
        event_id: str,
        purpose: str,
        chunk_index: int,
        message_id: str,
    ) -> AgentHostDeliveryRecord:
        original = records[chunk_index]
        return AgentHostDeliveryRecord(
            event_id=event_id,
            purpose=purpose,
            chunk_index=chunk_index,
            content_sha256=original.content_sha256,
            channel_id=original.channel_id,
            message_id=message_id,
            receipted_at=original.receipted_at,
            created_at=original.created_at,
            updated_at=now,
        )

    store = SimpleNamespace(
        record_host_delivery_message=AsyncMock(side_effect=record_message)
    )
    cog = AgentCog(
        SimpleNamespace(user=SimpleNamespace(id=999)),
        SimpleNamespace(agent_store=store),
    )

    recovery_candidates: dict[str, tuple[discord.Message, ...]] = {}
    reconciled = await cog._reconcile_host_messages(
        channel,
        pending,
        records,
        recovery_candidates=recovery_candidates,
    )
    await cog._reconcile_host_messages(
        channel,
        pending,
        records,
        recovery_candidates=recovery_candidates,
    )

    assert tuple(record.message_id for record in reconciled) == ("301", "302")
    assert channel.history.call_count == 1
    assert store.record_host_delivery_message.await_count == 2
    assert (
        store.record_host_delivery_message.await_args.kwargs["message_id"]
        == "302"
    )


@pytest.mark.asyncio
async def test_ack_retries_same_claim_without_redelivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:ack",
        claim_token="claim",
        lease_until=now + timedelta(minutes=1),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(),
    )
    queue = SimpleNamespace(
        mark_processed=AsyncMock(
            side_effect=(sqlite3.OperationalError("busy"), None)
        )
    )
    sleep = AsyncMock()
    monkeypatch.setattr("simajilord.integrations.discord.cogs.asyncio.sleep", sleep)
    cog = AgentAutonomyCog(
        SimpleNamespace(),
        SimpleNamespace(autonomy_events=queue),
    )

    await cog._ack_with_retry(batch)

    assert queue.mark_processed.await_count == 2
    sleep.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_heartbeat_lease_loss_cancels_inflight_inspection() -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:heartbeat-loss",
        claim_token="claim",
        lease_until=now + timedelta(minutes=1),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(),
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def inspect(_: AutonomyEventBatch) -> bool:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def heartbeat(_: AutonomyEventBatch) -> None:
        await started.wait()
        raise AutonomyLeaseLostError("reclaimed")

    cog = AgentAutonomyCog(
        SimpleNamespace(),
        SimpleNamespace(),
    )
    cog._inspect = inspect  # type: ignore[method-assign]
    cog._heartbeat = heartbeat  # type: ignore[method-assign]

    result = await cog._consume_batch(batch)

    assert result is None
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_settlement_storage_failure_does_not_escape_consumer() -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:settle",
        claim_token="claim",
        lease_until=now + timedelta(minutes=1),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(),
    )
    runtime = SimpleNamespace(
        autonomy_events=SimpleNamespace(
            reschedule=AsyncMock(
                side_effect=sqlite3.OperationalError("unavailable")
            )
        ),
        journal=SimpleNamespace(append=AsyncMock()),
    )

    settled = await AgentAutonomyCog(
        SimpleNamespace(),
        runtime,
    )._safe_retry_failure(
        batch,
        reason="RuntimeError",
        retry_after_seconds=60,
    )

    assert settled is False


@pytest.mark.asyncio
async def test_missing_post_permission_is_terminal_before_model_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    batch = AutonomyEventBatch(
        batch_id="autonomy:10:20:no-send",
        claim_token="claim",
        lease_until=now + timedelta(minutes=1),
        attempt_count=0,
        workspace_id="10",
        channel_id="20",
        events=(
            AutonomyQueuedEvent(
                sequence=1,
                deduplication_key="message:1",
                kind=AutonomyEventKind.MESSAGE_CREATE,
                occurred_at=now,
                enqueued_at=now,
                not_before=now,
                attempt_count=0,
                workspace_id="10",
                channel_id="20",
                actor_id="101",
                message_id="201",
                payload={},
            ),
        ),
    )
    agent = SimpleNamespace(respond=AsyncMock())
    runtime = SimpleNamespace(
        agent=agent,
        settings=SimpleNamespace(agent_autonomy_guild_ids=frozenset({"10"})),
    )
    bot_member = SimpleNamespace(id=999)
    guild = SimpleNamespace(id=10, me=bot_member)
    channel = Mock(spec=discord.TextChannel)
    channel.permissions_for = Mock(
        return_value=_post_permissions(can_send=False)
    )
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_guild=lambda _: guild,
        get_channel=lambda _: channel,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs.agent_readable_channel_ids",
        lambda *args, **kwargs: ("20",),
    )

    with pytest.raises(RuntimeError, match="channel_not_postable"):
        await AgentAutonomyCog(bot, runtime)._inspect(batch)

    agent.respond.assert_not_awaited()
