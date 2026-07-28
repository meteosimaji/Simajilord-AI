from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from simajilord.services.media import FairMediaScheduler, MediaPriority
from simajilord.services.speech import FairSpeechScheduler


@pytest.mark.asyncio
@pytest.mark.parametrize("guild_count", (2, 4, 8))
async def test_media_and_tts_schedulers_sustain_multi_guild_load(
    guild_count: int,
) -> None:
    """Exercise both shared providers long enough to expose starvation or leakage."""

    speech = FairSpeechScheduler(2)
    media = FairMediaScheduler(max_concurrent=3, max_per_workspace=1)
    speech_active = 0
    media_active = 0
    speech_max = 0
    media_max = 0
    speech_by_guild: dict[str, int] = defaultdict(int)
    media_by_guild: dict[str, int] = defaultdict(int)
    speech_order: dict[str, list[int]] = defaultdict(list)
    media_order: dict[str, list[int]] = defaultdict(list)

    async def speech_operation(workspace_id: str, sequence: int) -> tuple[str, int]:
        nonlocal speech_active, speech_max
        speech_active += 1
        speech_by_guild[workspace_id] += 1
        speech_max = max(speech_max, speech_active)
        assert speech_by_guild[workspace_id] == 1
        speech_order[workspace_id].append(sequence)
        await asyncio.sleep(0.002)
        speech_by_guild[workspace_id] -= 1
        speech_active -= 1
        return workspace_id, sequence

    async def media_operation(workspace_id: str, sequence: int) -> tuple[str, int]:
        nonlocal media_active, media_max
        media_active += 1
        media_by_guild[workspace_id] += 1
        media_max = max(media_max, media_active)
        assert media_by_guild[workspace_id] == 1
        media_order[workspace_id].append(sequence)
        await asyncio.sleep(0.002)
        media_by_guild[workspace_id] -= 1
        media_active -= 1
        return workspace_id, sequence

    async def run_speech(workspace_id: str, sequence: int) -> tuple[str, int]:
        async def operation() -> tuple[str, int]:
            return await speech_operation(workspace_id, sequence)

        return await speech.run(workspace_id, operation)

    async def run_media(workspace_id: str, sequence: int) -> tuple[str, int]:
        async def operation() -> tuple[str, int]:
            return await media_operation(workspace_id, sequence)

        return await media.run(
            workspace_id=workspace_id,
            priority=(
                MediaPriority.INTERACTIVE
                if sequence % 3 == 0
                else MediaPriority.NORMAL
            ),
            operation_name="soak",
            operation=operation,
        )

    try:
        speech_tasks = [
            asyncio.create_task(run_speech(f"guild-{guild}", sequence))
            for sequence in range(8)
            for guild in range(guild_count)
        ]
        media_tasks = [
            asyncio.create_task(run_media(f"guild-{guild}", sequence))
            for sequence in range(8)
            for guild in range(guild_count)
        ]
        speech_results, media_results = await asyncio.gather(
            asyncio.gather(*speech_tasks),
            asyncio.gather(*media_tasks),
        )
    finally:
        await speech.close()
        await media.close()

    expected_count = guild_count * 8
    assert len(speech_results) == expected_count
    assert len(media_results) == expected_count
    assert speech_max == min(2, guild_count)
    assert media_max == min(3, guild_count)
    assert all(order == list(range(8)) for order in speech_order.values())
    assert all(sorted(order) == list(range(8)) for order in media_order.values())
    assert set(speech_order) == {f"guild-{guild}" for guild in range(guild_count)}
    assert set(media_order) == set(speech_order)


@pytest.mark.asyncio
async def test_eight_guild_soak_keeps_manual_media_ahead_of_radio_failures() -> None:
    """Model blocked connections, four refills, interactive play, timeout, and 429."""

    media = FairMediaScheduler(max_concurrent=4, max_per_workspace=1)
    release_connections = asyncio.Event()
    started: list[tuple[str, str]] = []
    completed: list[tuple[str, str]] = []

    async def run(
        workspace_id: str,
        label: str,
        priority: MediaPriority,
        *,
        blocked: bool = False,
        failure: str | None = None,
    ) -> str:
        async def operation() -> str:
            started.append((workspace_id, label))
            if blocked:
                await release_connections.wait()
            else:
                await asyncio.sleep(0.001)
            if failure is not None:
                raise RuntimeError(failure)
            completed.append((workspace_id, label))
            return label

        return await media.run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name=label,
            operation=operation,
        )

    blockers = [
        asyncio.create_task(
            run(
                f"guild-{index}",
                "connect",
                MediaPriority.INTERACTIVE,
                blocked=True,
            )
        )
        for index in range(4)
    ]
    while len(started) < 4:
        await asyncio.sleep(0)

    radio = [
        asyncio.create_task(
            run(
                f"guild-{index}",
                "radio",
                MediaPriority.BACKGROUND,
                failure=("timeout" if index % 2 == 0 else "HTTP 429"),
            )
        )
        for index in range(4, 8)
    ]
    manual = [
        asyncio.create_task(
            run(
                f"guild-{index}",
                "manual",
                MediaPriority.INTERACTIVE,
            )
        )
        for index in range(4, 8)
    ]
    release_connections.set()
    try:
        results = await asyncio.gather(
            *blockers,
            *manual,
            *radio,
            return_exceptions=True,
        )
    finally:
        await media.close()

    assert results[:8] == ["connect"] * 4 + ["manual"] * 4
    assert all(isinstance(result, RuntimeError) for result in results[8:])
    for workspace_id in (f"guild-{index}" for index in range(4, 8)):
        guild_starts = [label for guild, label in started if guild == workspace_id]
        assert guild_starts == ["manual", "radio"]


@pytest.mark.asyncio
async def test_one_slow_guild_does_not_block_other_guilds() -> None:
    media = FairMediaScheduler(max_concurrent=3, max_per_workspace=1)
    release_slow = asyncio.Event()
    completed: list[str] = []

    async def run(workspace_id: str, *, slow: bool) -> str:
        async def operation() -> str:
            if slow:
                await release_slow.wait()
            else:
                await asyncio.sleep(0.001)
            completed.append(workspace_id)
            return workspace_id

        return await media.run(
            workspace_id=workspace_id,
            priority=MediaPriority.INTERACTIVE,
            operation_name="connect",
            operation=operation,
        )

    slow = asyncio.create_task(run("guild-slow", slow=True))
    fast = [
        asyncio.create_task(run(f"guild-{index}", slow=False))
        for index in range(7)
    ]
    try:
        fast_results = await asyncio.wait_for(asyncio.gather(*fast), timeout=1)
        assert fast_results == [f"guild-{index}" for index in range(7)]
        assert "guild-slow" not in completed
        release_slow.set()
        assert await asyncio.wait_for(slow, timeout=1) == "guild-slow"
    finally:
        release_slow.set()
        await media.close()
