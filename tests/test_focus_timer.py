from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simajilord.capabilities.focus_timer import (
    FocusTimerCancelRequest,
    FocusTimerCreateRequest,
    FocusTimerListRequest,
    FocusTimerResponse,
    build_focus_timer_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
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
        assert read_aloud.policy("guild").read_messages is True

    asyncio.run(run())
