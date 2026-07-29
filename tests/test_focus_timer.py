from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

import simajilord.integrations.discord.cogs as discord_cogs
from simajilord.capabilities.focus_timer import (
    FocusTimerCancelRequest,
    FocusTimerCreateRequest,
    FocusTimerListRequest,
    FocusTimerResponse,
    build_focus_timer_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.cogs import (
    FocusTimerCog,
    _focus_timer_delivery_marker,
)
from simajilord.services.focus_timer import FocusTimerService, FocusTimerStatus
from simajilord.services.read_aloud import (
    ReadAloudContentMode,
    ReadAloudService,
)


def _context(actor_id: str = "actor") -> InvocationContext:
    return InvocationContext(
        actor_id=actor_id,
        workspace_id="guild",
        transport="test",
        request_id="request",
        origin_resource_id="channel",
    )


def test_focus_timer_is_claimed_once_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "timers.sqlite3"

    async def run() -> None:
        service = FocusTimerService(path)
        created = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="  Finished   studying. ",
        )
        assert created.message == "Finished studying."
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
                (
                    (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                    created.timer_id,
                ),
            )
        claimed = await service.claim_due()
        assert tuple(item.timer_id for item in claimed) == (created.timer_id,)
        assert claimed[0].status is FocusTimerStatus.DELIVERING
        assert await service.claim_due() == ()

        restarted = FocusTimerService(path)
        reclaimed = await restarted.claim_due()
        assert tuple(item.timer_id for item in reclaimed) == (created.timer_id,)
        completed = await restarted.complete(created.timer_id)
        assert completed.status is FocusTimerStatus.COMPLETED
        assert await restarted.active(workspace_id="guild") == ()

    asyncio.run(run())


def test_focus_timer_migrates_delivery_message_column(tmp_path: Path) -> None:
    path = tmp_path / "legacy-timers.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE focus_timers (
                timer_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                delivery_target_id TEXT NOT NULL,
                due_at TEXT NOT NULL,
                message TEXT NOT NULL,
                voice_notify INTEGER NOT NULL,
                focus_session INTEGER NOT NULL,
                restore_content_mode TEXT,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    FocusTimerService(path)

    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(focus_timers)")
        }
    assert {"delivery_message_id", "finished_at"} <= columns


def test_focus_timer_owner_boundary_and_retry(tmp_path: Path) -> None:
    async def run() -> None:
        service = FocusTimerService(tmp_path / "timers.sqlite3")
        created = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="Done",
        )
        with pytest.raises(UserError, match=r"timer\.not_owner"):
            await service.cancel(
                timer_id=created.timer_id,
                workspace_id="guild",
                actor_id="other",
            )
        cancelled = await service.cancel(
            timer_id=created.timer_id,
            workspace_id="guild",
            actor_id="actor",
        )
        assert cancelled.status is FocusTimerStatus.CANCELLED

    asyncio.run(run())


def test_focus_timer_restore_clears_prior_delivery_message_id(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = FocusTimerService(tmp_path / "timers.sqlite3")
        created = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="Done",
        )
        await service.set_delivery_message(created.timer_id, "old-message")
        await service.cancel(
            timer_id=created.timer_id,
            workspace_id="guild",
            actor_id="actor",
        )

        restored = await service.restore(
            timer_id=created.timer_id,
            workspace_id="guild",
            actor_id="actor",
        )

        assert restored.status is FocusTimerStatus.SCHEDULED
        assert restored.delivery_message_id is None

    asyncio.run(run())


def test_focus_timer_cancel_wins_over_in_flight_completion(tmp_path: Path) -> None:
    path = tmp_path / "timers.sqlite3"

    async def run() -> None:
        service = FocusTimerService(path)
        created = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="Done",
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
                (
                    (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                    created.timer_id,
                ),
            )
        claimed = (await service.claim_due())[0]
        assert claimed.status is FocusTimerStatus.DELIVERING
        cancelled = await service.cancel(
            timer_id=created.timer_id,
            workspace_id="guild",
            actor_id="actor",
        )
        assert cancelled.status is FocusTimerStatus.CANCELLED

        completed = await service.complete(created.timer_id)
        assert completed.status is FocusTimerStatus.CANCELLED

    asyncio.run(run())


def test_focus_timer_retention_never_removes_active_timers(tmp_path: Path) -> None:
    path = tmp_path / "timers.sqlite3"

    async def run() -> None:
        service = FocusTimerService(path)
        active = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="Still active",
        )
        terminal = await service.create(
            workspace_id="guild",
            actor_id="actor",
            delivery_target_id="channel",
            duration_seconds=5,
            message="Finished",
        )
        await service.cancel(
            timer_id=terminal.timer_id,
            workspace_id="guild",
            actor_id="actor",
        )
        old = datetime.now(UTC) - timedelta(days=31)
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
                (old.isoformat(), active.timer_id),
            )
            connection.execute(
                "UPDATE focus_timers SET finished_at = ? WHERE timer_id = ?",
                (old.isoformat(), terminal.timer_id),
            )

        removed = await service.prune_terminal(
            before=datetime.now(UTC) - timedelta(days=30)
        )
        assert removed == 1
        assert tuple(
            timer.timer_id for timer in await service.active(workspace_id="guild")
        ) == (active.timer_id,)

    asyncio.run(run())


def test_focus_timer_capability_enters_and_restores_focus_policy(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = FocusTimerService(tmp_path / "timers.sqlite3")
        read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
        endpoints = {
            item.descriptor.name: item
            for item in build_focus_timer_endpoints(service, read_aloud)
        }
        created = await endpoints["timer.create"].invoke(
            FocusTimerCreateRequest(
                duration_seconds=60,
                message="Done",
                focus_session=True,
            ),
            _context(),
        )
        assert isinstance(created, FocusTimerResponse)
        assert read_aloud.policy("guild").read_messages is False
        active = await service.active(workspace_id="guild")
        assert active[0].restore_content_mode == ReadAloudContentMode.MESSAGES.value

        listed = await endpoints["timer.list"].invoke(
            FocusTimerListRequest(),
            _context(),
        )
        assert tuple(item.timer_id for item in listed.timers) == (
            created.timer.timer_id,
        )
        cancelled = await endpoints["timer.cancel"].invoke(
            FocusTimerCancelRequest(created.timer.timer_id),
            _context(),
        )
        assert cancelled.timer.status == FocusTimerStatus.CANCELLED.value
        assert cancelled.changed is True
        assert read_aloud.policy("guild").read_messages is True
        cancelled_again = await endpoints["timer.cancel"].invoke(
            FocusTimerCancelRequest(created.timer.timer_id),
            _context(),
        )
        assert cancelled_again.timer.status == FocusTimerStatus.CANCELLED.value
        assert cancelled_again.changed is False
        restored = await endpoints["timer.restore"].invoke(
            FocusTimerCancelRequest(created.timer.timer_id),
            _context(),
        )
        assert restored.timer.status == FocusTimerStatus.SCHEDULED.value
        assert restored.changed is True
        assert read_aloud.policy("guild").read_messages is False
        restored_again = await endpoints["timer.restore"].invoke(
            FocusTimerCancelRequest(created.timer.timer_id),
            _context(),
        )
        assert restored_again.timer.status == FocusTimerStatus.SCHEDULED.value
        assert restored_again.changed is False

    asyncio.run(run())


def test_overlapping_focus_sessions_restore_only_after_the_last_timer(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = FocusTimerService(tmp_path / "timers.sqlite3")
        read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
        endpoints = {
            item.descriptor.name: item
            for item in build_focus_timer_endpoints(service, read_aloud)
        }
        first = await endpoints["timer.create"].invoke(
            FocusTimerCreateRequest(duration_seconds=60, focus_session=True),
            _context(),
        )
        second = await endpoints["timer.create"].invoke(
            FocusTimerCreateRequest(duration_seconds=90, focus_session=True),
            _context(),
        )
        active = await service.active(workspace_id="guild")
        assert {
            timer.restore_content_mode
            for timer in active
        } == {ReadAloudContentMode.MESSAGES.value}

        await endpoints["timer.cancel"].invoke(
            FocusTimerCancelRequest(first.timer.timer_id),
            _context(),
        )
        assert read_aloud.policy("guild").read_messages is False
        await endpoints["timer.cancel"].invoke(
            FocusTimerCancelRequest(second.timer.timer_id),
            _context(),
        )
        assert read_aloud.policy("guild").read_messages is True

    asyncio.run(run())


def test_focus_cancel_does_not_overwrite_a_later_human_policy_change(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        service = FocusTimerService(tmp_path / "timers.sqlite3")
        read_aloud = ReadAloudService(tmp_path / "read_aloud.json")
        endpoints = {
            item.descriptor.name: item
            for item in build_focus_timer_endpoints(service, read_aloud)
        }
        created = await endpoints["timer.create"].invoke(
            FocusTimerCreateRequest(duration_seconds=60, focus_session=True),
            _context(),
        )
        await read_aloud.set_content_mode(
            workspace_id="guild",
            mode=ReadAloudContentMode.ALL,
        )

        await endpoints["timer.cancel"].invoke(
            FocusTimerCancelRequest(created.timer.timer_id),
            _context(),
        )
        policy = read_aloud.policy("guild")
        assert policy.read_messages is True
        assert policy.announce_join is True

    asyncio.run(run())


@pytest.mark.asyncio
async def test_focus_timer_delivery_lock_prevents_parallel_duplicate_posts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timers.sqlite3"
    service = FocusTimerService(path)
    created = await service.create(
        workspace_id="1",
        actor_id="2",
        delivery_target_id="3",
        duration_seconds=5,
        message="Done",
        voice_notify=False,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
            (
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                created.timer_id,
            ),
        )
    claimed = (await service.claim_due())[0]

    send_started = asyncio.Event()
    release_send = asyncio.Event()
    message = Mock(spec=discord.Message)
    message.id = 123
    message.author = SimpleNamespace(id=999)
    message.embeds = []

    async def send(**kwargs: object) -> discord.Message:
        message.embeds = [kwargs["embed"]]
        send_started.set()
        await release_send.wait()
        return message

    async def empty_history(
        *,
        limit: int,
        after: datetime,
        oldest_first: bool,
    ):
        assert limit == 1_000
        assert after.tzinfo is not None
        assert oldest_first is True
        if False:
            yield message

    channel = Mock(spec=discord.TextChannel)
    channel.send = AsyncMock(side_effect=send)
    channel.fetch_message = AsyncMock()
    channel.history = empty_history
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_channel=lambda _: channel,
        fetch_channel=AsyncMock(),
    )
    cog = object.__new__(FocusTimerCog)
    cog.bot = bot
    cog.runtime = SimpleNamespace(focus_timer=service)
    monkeypatch.setattr(
        discord_cogs,
        "_publish_autonomy_event",
        AsyncMock(return_value=None),
    )

    first = asyncio.create_task(cog._deliver(claimed))
    await asyncio.wait_for(send_started.wait(), timeout=1)
    second = asyncio.create_task(cog._deliver(claimed))
    await asyncio.sleep(0)
    release_send.set()
    await asyncio.gather(first, second)

    assert channel.send.await_count == 1
    current = await service.current(created.timer_id)
    assert current.status is FocusTimerStatus.COMPLETED
    assert current.delivery_message_id == "123"
    assert service._delivery_locks == {}


@pytest.mark.asyncio
async def test_focus_timer_recovers_send_before_message_id_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timers.sqlite3"
    before_crash = FocusTimerService(path)
    created = await before_crash.create(
        workspace_id="1",
        actor_id="2",
        delivery_target_id="3",
        duration_seconds=5,
        message="Done",
        voice_notify=False,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
            (
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                created.timer_id,
            ),
        )
    claimed = (await before_crash.claim_due())[0]
    assert claimed.delivery_message_id is None

    already_sent = Mock(spec=discord.Message)
    already_sent.id = 456
    already_sent.author = SimpleNamespace(id=999)
    recovered_embed = discord.Embed(title="Focus Timer complete")
    recovered_embed.set_footer(
        text=_focus_timer_delivery_marker(created.timer_id)
    )
    already_sent.embeds = [recovered_embed]

    after_restart = FocusTimerService(path)
    reclaimed = (await after_restart.claim_due())[0]
    assert reclaimed.delivery_message_id is None

    async def history(
        *,
        limit: int,
        after: datetime,
        oldest_first: bool,
    ):
        assert limit == 1_000
        assert after.tzinfo is not None
        assert oldest_first is True
        yield already_sent

    channel = Mock(spec=discord.TextChannel)
    channel.send = AsyncMock()
    channel.fetch_message = AsyncMock()
    channel.history = history
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_channel=lambda _: channel,
        fetch_channel=AsyncMock(),
    )
    cog = object.__new__(FocusTimerCog)
    cog.bot = bot
    cog.runtime = SimpleNamespace(focus_timer=after_restart)
    monkeypatch.setattr(
        discord_cogs,
        "_publish_autonomy_event",
        AsyncMock(return_value=None),
    )

    await cog._deliver(reclaimed)

    assert channel.send.await_count == 0
    current = await after_restart.current(created.timer_id)
    assert current.status is FocusTimerStatus.COMPLETED
    assert current.delivery_message_id == "456"


@pytest.mark.asyncio
async def test_focus_timer_transient_message_fetch_failure_never_reposts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "timers.sqlite3"
    service = FocusTimerService(path)
    created = await service.create(
        workspace_id="1",
        actor_id="2",
        delivery_target_id="3",
        duration_seconds=5,
        message="Done",
        voice_notify=False,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE focus_timers SET due_at = ? WHERE timer_id = ?",
            (
                (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                created.timer_id,
            ),
        )
    claimed = (await service.claim_due())[0]
    await service.set_delivery_message(created.timer_id, "789")
    claimed = await service.current(created.timer_id)

    response = SimpleNamespace(status=503, reason="Service Unavailable")
    channel = Mock(spec=discord.TextChannel)
    channel.fetch_message = AsyncMock(
        side_effect=discord.HTTPException(response, "temporary outage")
    )
    channel.send = AsyncMock()
    bot = SimpleNamespace(
        user=SimpleNamespace(id=999),
        get_channel=lambda _: channel,
        fetch_channel=AsyncMock(),
    )
    cog = object.__new__(FocusTimerCog)
    cog.bot = bot
    cog.runtime = SimpleNamespace(focus_timer=service)
    monkeypatch.setattr(
        discord_cogs,
        "_publish_autonomy_event",
        AsyncMock(return_value=None),
    )

    await cog._deliver(claimed)

    assert channel.send.await_count == 0
    current = await service.current(created.timer_id)
    assert current.status is FocusTimerStatus.SCHEDULED
    assert current.delivery_message_id == "789"
