from __future__ import annotations

import asyncio
from typing import cast

import pytest

from simajilord.capabilities.audio import (
    AudioHistoryRequest,
    AudioHistoryResponse,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueRequest,
    AudioQueueResponse,
    build_audio_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind, LoopMode
from simajilord.services.audio import AudioSession, AudioSessionManager
from simajilord.services.audio_state import AudioStateStore
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
async def test_play_api_queues_without_voice_then_starts_and_records_history(
    tmp_path,
) -> None:
    class FakeMedia:
        async def resolve_audio(self, reference: str) -> AudioItem:
            return AudioItem(
                "resolved-stream",
                "Requested track",
                reference,
                duration_seconds=123,
                resolver_reference=reference,
            )

    output = FakeOutput()
    output.connected = False
    store = AudioStateStore(tmp_path / "audio_sessions.json")
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=store,
    )
    manager.get_or_create("guild", lambda: output)
    endpoints = build_audio_endpoints(cast(MediaService, FakeMedia()), manager)
    play_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.play"
    )
    history_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.history"
    )
    context = InvocationContext("requester-id", "guild", "test", "request")

    response = await play_endpoint.invoke(
        AudioPlayRequest(
            reference="https://example.com/watch",
            requested_by_name="Requester",
        ),
        context,
    )
    assert isinstance(response, AudioPlayResponse)
    assert response.playback_state == "waiting_for_voice"
    assert response.requested_by_name == "Requester"
    assert output.played == []
    stored = store.all()[0]
    assert stored.destination_id is None
    assert stored.waiting_actor_ids == ("requester-id",)
    assert stored.items[0].requested_by_name == "Requester"

    session = manager.require("guild")
    await session.connect("voice")
    await asyncio.sleep(0.02)
    assert output.played == ["Requested track"]
    output.release.set()
    await asyncio.sleep(0.02)

    history = await history_endpoint.invoke(AudioHistoryRequest(), context)
    assert isinstance(history, AudioHistoryResponse)
    assert history.items[0].title == "Requested track"
    assert history.items[0].requested_by_name == "Requester"
    assert history.items[0].played_at_epoch is not None
    stored_history = store.all()[0].history
    assert stored_history[0].title == "Requested track"
    assert stored_history[0].requested_by_name == "Requester"
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


@pytest.mark.asyncio
async def test_failed_stream_is_reresolved_without_another_command(monkeypatch) -> None:
    class FlakyOutput(FakeOutput):
        def __init__(self) -> None:
            super().__init__()
            self.sources: list[str] = []

        async def play(self, item: AudioItem) -> None:
            self.sources.append(item.source)
            if len(self.sources) == 1:
                raise RuntimeError("expired stream")

    async def resolve(reference: str) -> AudioItem:
        assert reference == "https://example.com/watch"
        return AudioItem(
            "fresh-stream",
            "Recovered",
            reference,
            resolver_reference=reference,
        )

    monkeypatch.setattr(
        "simajilord.services.audio._IMMEDIATE_RETRY_DELAYS",
        (0.0, 0.0, 0.0),
    )
    output = FlakyOutput()
    session = AudioSession(
        "one",
        output,
        max_pending_speech=3,
        resolver=resolve,
    )
    await session.enqueue(
        AudioItem(
            "expired-stream",
            "Track",
            "https://example.com/watch",
            resolver_reference="https://example.com/watch",
        )
    )
    await asyncio.sleep(0.02)
    assert output.sources == ["expired-stream", "fresh-stream"]
    await session.close()


@pytest.mark.asyncio
async def test_music_queue_survives_manager_restart_without_signed_url(tmp_path) -> None:
    state_path = tmp_path / "audio_sessions.json"
    store = AudioStateStore(state_path)
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=store,
    )
    session = manager.get_or_create("guild", lambda: output)
    await session.connect("voice")
    await session.enqueue(
        AudioItem(
            "https://signed.invalid/expires-soon",
            "Durable track",
            "https://example.com/watch",
            resolver_reference="https://example.com/watch",
        )
    )
    await asyncio.sleep(0)
    await manager.close()

    saved = AudioStateStore(state_path).all()
    assert len(saved) == 1
    assert saved[0].destination_id == "voice"
    assert saved[0].items[0].reference == "https://example.com/watch"
    assert "signed.invalid" not in state_path.read_text(encoding="utf-8")
    assert state_path.stat().st_mode & 0o077 == 0

    restored_output = FakeOutput()
    restored_output.connected = False
    restored_manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(state_path),
    )
    sessions = restored_manager.restore(lambda _: restored_output)
    assert len(sessions) == 1
    snapshot = await sessions[0].snapshot()
    assert snapshot.destination_id == "voice"
    assert snapshot.pending[0].source == ""
    await restored_manager.close()


@pytest.mark.asyncio
async def test_auto_leave_suspends_voice_without_losing_current_track(tmp_path) -> None:
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(tmp_path / "audio_sessions.json"),
    )
    session = manager.get_or_create("guild", lambda: output)
    await session.connect("voice")
    await session.enqueue(
        AudioItem(
            "signed-stream",
            "Keep me",
            "https://example.com/watch",
            resolver_reference="https://example.com/watch",
        )
    )
    await asyncio.sleep(0)
    assert session.current is not None

    await session.suspend()
    snapshot = await session.snapshot()
    assert not output.connected
    assert snapshot.current is None
    assert [item.title for item in snapshot.pending] == ["Keep me"]
    assert snapshot.pending[0].start_seconds > 0
    assert snapshot.destination_id == "voice"
    await manager.close()
