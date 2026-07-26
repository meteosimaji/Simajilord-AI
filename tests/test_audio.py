from __future__ import annotations

import asyncio
from typing import cast

import pytest

from simajilord.capabilities.audio import (
    AudioQueueRequest,
    AudioQueueResponse,
    build_audio_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind, LoopMode
from simajilord.services.audio import AudioSession, AudioSessionManager
from simajilord.services.media import MediaService


class FakeOutput:
    connected = True
    paused = False

    def __init__(self) -> None:
        self.played: list[str] = []
        self.release = asyncio.Event()

    async def connect(self, destination_id: str) -> None:
        self.connected = True

    async def play(self, item: AudioItem) -> None:
        self.played.append(item.title)
        await self.release.wait()
        self.release.clear()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.release.set()

    async def disconnect(self) -> None:
        self.connected = False
        self.release.set()


@pytest.mark.asyncio
async def test_speech_is_prioritized_ahead_of_waiting_music() -> None:
    output = FakeOutput()
    session = AudioSession("one", output, max_pending_speech=3)
    await session.enqueue(AudioItem("a", "first", "a"))
    await asyncio.sleep(0)
    await session.enqueue(AudioItem("b", "second", "b"))
    await session.enqueue(
        AudioItem("speech", "speech", "local://speech", kind=AudioKind.SPEECH)
    )
    output.release.set()
    await asyncio.sleep(0.02)
    assert output.played[:2] == ["first", "speech"]
    await session.close()


@pytest.mark.asyncio
async def test_snapshot_is_transport_neutral() -> None:
    output = FakeOutput()
    session = AudioSession("one", output, max_pending_speech=3)
    await session.set_loop(LoopMode.QUEUE)
    await session.enqueue(AudioItem("a", "track", "a"))
    snapshot = await session.snapshot()
    assert snapshot.loop is LoopMode.QUEUE
    assert snapshot.current is None or snapshot.current.title == "track"
    await session.close()


@pytest.mark.asyncio
async def test_queue_capability_returns_transport_neutral_state() -> None:
    output = FakeOutput()
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    session = manager.get_or_create("guild", lambda: output)
    await session.enqueue(AudioItem("one", "Now playing", "https://example.com/one"))
    await asyncio.sleep(0)
    await session.enqueue(AudioItem("two", "Next track", "https://example.com/two"))
    endpoints = build_audio_endpoints(cast(MediaService, object()), manager)
    queue_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.queue"
    )
    response = await queue_endpoint.invoke(
        AudioQueueRequest(),
        InvocationContext("actor", "guild", "test", "request"),
    )
    assert isinstance(response, AudioQueueResponse)
    assert response.current is not None
    assert response.current.title == "Now playing"
    assert response.pending[0].title == "Next track"
    assert response.loop_mode == LoopMode.NONE.value
    await manager.close()


@pytest.mark.asyncio
async def test_multiple_guilds_have_independent_voice_sessions() -> None:
    first_output = FakeOutput()
    second_output = FakeOutput()
    first_output.connected = False
    second_output.connected = False
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    first = manager.get_or_create("guild-one", lambda: first_output)
    second = manager.get_or_create("guild-two", lambda: second_output)
    assert first is not second
    await first.connect("voice-one")
    await second.connect("voice-two")
    assert manager.active_session_count == 2
    await first.enqueue(AudioItem("one", "First guild", "https://example.com/one"))
    await asyncio.sleep(0)
    assert first.current is not None
    assert second.current is None
    await manager.close()


@pytest.mark.asyncio
async def test_voice_capacity_counts_other_guilds_only() -> None:
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=1, max_pending_speech=3)
    session = manager.get_or_create("guild-one", lambda: output)
    await session.connect("voice-one")
    manager.assert_connection_capacity("guild-one")
    with pytest.raises(UserError, match=r"audio\.capacity_reached"):
        manager.assert_connection_capacity("guild-two")
    await manager.close()
