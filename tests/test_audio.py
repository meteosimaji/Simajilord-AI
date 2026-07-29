from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

from simajilord.capabilities.audio import (
    AudioHistoryRequest,
    AudioHistoryResponse,
    AudioMixRequest,
    AudioMixResponse,
    AudioMoveRequest,
    AudioNoArgsRequest,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueRequest,
    AudioQueueResponse,
    AudioSearchReason,
    AudioSearchRequest,
    AudioSearchResponse,
    AudioVolumeRequest,
    build_audio_endpoints,
)
from simajilord.core import ApprovalMode, InvocationContext
from simajilord.core.errors import EarlyPlaybackEnd, MediaError, UserError
from simajilord.domain.audio import AudioItem, AudioKind, AudioQueueLane, LoopMode
from simajilord.domain.media import MediaCandidate
from simajilord.services.audio import AudioSession, AudioSessionManager
from simajilord.services.audio_state import AudioStateStore, StoredAudioSession
from simajilord.services.media import MediaPriority, MediaService


class FakeOutput:
    connected = True
    paused = False

    def __init__(self) -> None:
        self.played: list[str] = []
        self.played_items: list[AudioItem] = []
        self.overlays: list[str] = []
        self.music_updates: list[float] = []
        self.fade_outs: list[tuple[str, float, float]] = []
        self.overlay_error: Exception | None = None
        self.overlay_attempts = 0
        self.stop_calls = 0
        self.release = asyncio.Event()

    async def connect(self, destination_id: str) -> None:
        self.connected = True

    async def play(self, item: AudioItem) -> None:
        self.played.append(item.title)
        self.played_items.append(item)
        await self.release.wait()
        self.release.clear()

    async def overlay_speech(
        self,
        music: AudioItem,
        speech: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        del music, position_seconds
        self.overlay_attempts += 1
        if self.overlay_error is not None:
            raise self.overlay_error
        self.overlays.append(speech.source)

    async def update_music(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        del music
        self.music_updates.append(position_seconds)

    async def fade_out(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
        duration_seconds: float,
    ) -> None:
        self.fade_outs.append((music.title, position_seconds, duration_seconds))

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stop_calls += 1
        self.release.set()

    async def disconnect(self) -> None:
        self.connected = False
        self.release.set()


@pytest.mark.asyncio
async def test_speech_overlays_current_music_before_waiting_music() -> None:
    output = FakeOutput()
    session = AudioSession("one", output, max_pending_speech=3)
    await session.enqueue(AudioItem("a", "first", "a"))
    await asyncio.sleep(0)
    await session.enqueue(AudioItem("b", "second", "b"))
    await session.enqueue(
        AudioItem("speech", "speech", "local://speech", kind=AudioKind.SPEECH)
    )
    for _ in range(20):
        if output.overlays == ["speech"]:
            break
        await asyncio.sleep(0)
    assert output.played == ["first"]
    assert output.overlays == ["speech"]
    assert output.fade_outs == []
    assert output.music_updates
    assert output.stop_calls == 0

    output.release.set()
    for _ in range(20):
        if output.played[:2] == ["first", "second"]:
            break
        await asyncio.sleep(0)
    assert output.played[:2] == ["first", "second"]
    await session.close()


@pytest.mark.asyncio
async def test_speech_ducks_current_music_without_restarting_the_player(
    tmp_path: Path,
) -> None:
    resolved: list[str] = []

    async def resolve(reference: str) -> AudioItem:
        resolved.append(reference)
        return AudioItem(
            "fresh-music-stream",
            "music",
            reference,
            resolver_reference=reference,
        )

    output = FakeOutput()
    session = AudioSession(
        "one",
        output,
        max_pending_speech=3,
        resolver=resolve,
    )
    await session.enqueue(
        AudioItem(
            "music-stream",
            "music",
            "https://example.com/music",
            resolver_reference="https://example.com/music",
        )
    )
    await asyncio.sleep(0)
    speech_file = tmp_path / "speech.wav"
    speech_file.write_bytes(b"speech")
    await session.enqueue(
        AudioItem(
            str(speech_file),
            "speech",
            "local://speech",
            duration_seconds=0.01,
            kind=AudioKind.SPEECH,
            owned_file=speech_file,
        )
    )
    for _ in range(50):
        if output.overlays == [str(speech_file)] and output.music_updates:
            break
        await asyncio.sleep(0)
    assert output.played == ["music"]
    assert output.fade_outs == []
    assert output.overlays == [str(speech_file)]
    assert output.stop_calls == 0
    assert resolved == []
    assert not speech_file.exists()
    snapshot = await session.snapshot()
    assert snapshot.current is not None
    assert snapshot.current.kind is AudioKind.MUSIC
    assert snapshot.current.speech_overlay_source is None
    assert snapshot.speech_active is False
    assert snapshot.position_seconds >= 0
    assert snapshot.history == ()
    await session.close()


@pytest.mark.asyncio
async def test_three_speech_overlays_keep_fifo_order_without_stopping_music() -> None:
    output = FakeOutput()
    session = AudioSession("one", output, max_pending_speech=4)
    await session.enqueue(AudioItem("music", "music", "music"))
    await asyncio.sleep(0)

    for index in range(3):
        await session.enqueue(
            AudioItem(
                f"speech-{index}",
                f"speech-{index}",
                f"local://speech-{index}",
                kind=AudioKind.SPEECH,
            )
        )
    for _ in range(50):
        if output.overlays == ["speech-0", "speech-1", "speech-2"]:
            break
        await asyncio.sleep(0)

    assert output.overlays == ["speech-0", "speech-1", "speech-2"]
    assert output.played == ["music"]
    assert output.stop_calls == 0
    await session.close()


@pytest.mark.asyncio
async def test_failed_speech_overlay_falls_back_then_resumes_music() -> None:
    output = FakeOutput()
    output.overlay_error = RuntimeError("overlay unavailable")
    session = AudioSession("one", output, max_pending_speech=3)
    await session.enqueue(AudioItem("music", "music", "music"))
    await asyncio.sleep(0)
    await session.enqueue(
        AudioItem("speech", "speech", "local://speech", kind=AudioKind.SPEECH)
    )

    for _ in range(50):
        if output.played == ["music", "speech"]:
            break
        await asyncio.sleep(0)
    assert output.played == ["music", "speech"]
    assert output.overlays == []
    assert output.overlay_attempts == 3
    assert output.stop_calls == 1

    output.release.set()
    for _ in range(50):
        if output.played == ["music", "speech", "music"]:
            break
        await asyncio.sleep(0)
    assert output.played == ["music", "speech", "music"]
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lane", "initial_failures"),
    (
        (AudioQueueLane.REQUEST, 2),
        (AudioQueueLane.AUTOPLAY, 1),
    ),
)
async def test_failed_music_is_dropped_at_lane_retry_cap(
    monkeypatch: pytest.MonkeyPatch,
    lane: AudioQueueLane,
    initial_failures: int,
) -> None:
    class FailedOutput(FakeOutput):
        async def play(self, item: AudioItem) -> None:
            self.played.append(item.title)
            raise RuntimeError("transport failed")

    monkeypatch.setattr(
        "simajilord.services.audio._IMMEDIATE_RETRY_DELAYS",
        (0.0,),
    )
    output = FailedOutput()
    session = AudioSession("bounded", output, max_pending_speech=3)
    await session.enqueue(
        AudioItem(
            "stream",
            "bounded failure",
            "https://example.com/watch",
            failure_count=initial_failures,
            queue_lane=lane,
        )
    )

    for _ in range(50):
        snapshot = await session.snapshot()
        if output.played and snapshot.current is None and not snapshot.pending:
            break
        await asyncio.sleep(0)

    snapshot = await session.snapshot()
    assert output.played == ["bounded failure"]
    assert snapshot.current is None
    assert snapshot.pending == ()
    await session.close()


@pytest.mark.asyncio
async def test_disconnected_output_preserves_entire_queue_until_reconnected() -> None:
    output = FakeOutput()
    output.connected = False
    session = AudioSession("disconnected", output, max_pending_speech=3)
    await session.enqueue(AudioItem("one", "first", "https://example.com/one"))
    await session.enqueue(AudioItem("two", "second", "https://example.com/two"))

    for _ in range(10):
        await asyncio.sleep(0)

    snapshot = await session.snapshot()
    assert output.played == []
    assert tuple(item.title for item in snapshot.pending) == ("first", "second")
    assert tuple(item.failure_count for item in snapshot.pending) == (0, 0)

    await session.connect("voice")
    for _ in range(20):
        if output.played == ["first"]:
            break
        await asyncio.sleep(0)

    assert output.played == ["first"]
    await session.close()


@pytest.mark.asyncio
async def test_permanent_media_failure_is_not_requeued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnavailableOutput(FakeOutput):
        async def play(self, item: AudioItem) -> None:
            self.played.append(item.title)
            raise MediaError("unavailable", "private or deleted")

    monkeypatch.setattr(
        "simajilord.services.audio._IMMEDIATE_RETRY_DELAYS",
        (0.0,),
    )
    output = UnavailableOutput()
    session = AudioSession("permanent", output, max_pending_speech=3)
    await session.enqueue(
        AudioItem(
            "stream",
            "unavailable",
            "https://example.com/watch",
        )
    )

    for _ in range(50):
        snapshot = await session.snapshot()
        if snapshot.current is None and not snapshot.pending:
            break
        await asyncio.sleep(0)

    snapshot = await session.snapshot()
    assert output.played == ["unavailable"]
    assert snapshot.current is None
    assert snapshot.pending == ()
    await session.close()


@pytest.mark.asyncio
async def test_early_playback_end_reresolves_once_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EarlyEndOutput(FakeOutput):
        async def play(self, item: AudioItem) -> None:
            self.played.append(item.source)
            if len(self.played) == 1:
                raise EarlyPlaybackEnd(
                    elapsed_seconds=10.0,
                    expected_seconds=120.0,
                )

    resolved: list[str] = []

    async def resolve(reference: str) -> AudioItem:
        resolved.append(reference)
        return AudioItem(
            "fresh-stream",
            "recovered",
            reference,
            duration_seconds=120.0,
            resolver_reference=reference,
        )

    monkeypatch.setattr(
        "simajilord.services.audio._IMMEDIATE_RETRY_DELAYS",
        (0.0, 0.0),
    )
    output = EarlyEndOutput()
    session = AudioSession(
        "early-eof",
        output,
        max_pending_speech=3,
        resolver=resolve,
    )
    await session.enqueue(
        AudioItem(
            "old-stream",
            "track",
            "https://example.com/watch",
            duration_seconds=120.0,
            resolver_reference="https://example.com/watch",
        )
    )

    for _ in range(50):
        if output.played == ["old-stream", "fresh-stream"]:
            break
        await asyncio.sleep(0)

    assert output.played == ["old-stream", "fresh-stream"]
    assert resolved == ["https://example.com/watch"]
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


def test_loop_clone_restarts_from_the_beginning() -> None:
    item = AudioItem(
        "stream",
        "track",
        "https://example.com/track",
        duration_seconds=249,
        start_seconds=245.6,
        retry_after=123.0,
        played_at_epoch=456,
    )

    looped = item.clone_for_loop()

    assert looped.start_seconds == 0
    assert looped.retry_after == 0
    assert looped.played_at_epoch is None


@pytest.mark.asyncio
async def test_mix_and_loop_require_explicit_conflict_replacement() -> None:
    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        del seeds, limit
        return ()

    output = FakeOutput()
    output.connected = False
    session = AudioSession(
        "conflict",
        output,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    seed = ("https://www.youtube.com/watch?v=seed",)
    await session.set_loop(LoopMode.TRACK)

    with pytest.raises(UserError, match=r"audio\.mix_loop_conflict"):
        await session.enable_autoplay(seed)
    snapshot = await session.snapshot()
    assert snapshot.loop is LoopMode.TRACK
    assert snapshot.autoplay_enabled is False

    await session.enable_autoplay(seed, replace_loop=True)
    snapshot = await session.snapshot()
    assert snapshot.loop is LoopMode.NONE
    assert snapshot.autoplay_enabled is True

    with pytest.raises(UserError, match=r"audio\.loop_mix_conflict"):
        await session.set_loop(LoopMode.QUEUE)
    snapshot = await session.snapshot()
    assert snapshot.loop is LoopMode.NONE
    assert snapshot.autoplay_enabled is True

    await session.set_loop(LoopMode.QUEUE, replace_autoplay=True)
    snapshot = await session.snapshot()
    assert snapshot.loop is LoopMode.QUEUE
    assert snapshot.autoplay_enabled is False
    await session.close()


@pytest.mark.asyncio
async def test_music_queue_limits_reject_and_clean_up_resolved_files(tmp_path) -> None:
    output = FakeOutput()
    output.connected = False
    session = AudioSession(
        "one",
        output,
        max_pending_speech=3,
        max_pending_music=2,
        max_pending_music_per_actor=2,
    )
    await session.wait_for_listener("actor")
    for index in range(2):
        await session.enqueue(
            AudioItem(
                f"stream-{index}",
                f"track-{index}",
                f"https://example.com/{index}",
                resolver_reference=f"https://example.com/{index}",
                requested_by_id=f"actor-{index}",
            )
        )
    owned_file = tmp_path / "rejected.webm"
    owned_file.write_bytes(b"temporary")
    with pytest.raises(UserError, match=r"audio\.queue_full"):
        await session.enqueue(
            AudioItem(
                "stream-rejected",
                "rejected",
                "https://example.com/rejected",
                resolver_reference="https://example.com/rejected",
                requested_by_id="third",
                owned_file=owned_file,
            )
        )
    assert not owned_file.exists()
    await session.close()


@pytest.mark.asyncio
async def test_music_queue_limits_each_actor_and_duplicate_reference() -> None:
    output = FakeOutput()
    output.connected = False
    per_actor = AudioSession(
        "actor-limit",
        output,
        max_pending_speech=3,
        max_pending_music=10,
        max_pending_music_per_actor=1,
    )
    await per_actor.wait_for_listener("actor")
    await per_actor.enqueue(
        AudioItem(
            "one",
            "one",
            "https://example.com/one",
            resolver_reference="https://example.com/one",
            requested_by_id="actor",
        )
    )
    with pytest.raises(UserError, match=r"audio\.user_queue_full"):
        await per_actor.enqueue(
            AudioItem(
                "two",
                "two",
                "https://example.com/two",
                resolver_reference="https://example.com/two",
                requested_by_id="actor",
            )
        )
    await per_actor.close()

    duplicate = AudioSession(
        "duplicate-limit",
        output,
        max_pending_speech=3,
        max_pending_music=10,
        max_pending_music_per_actor=10,
    )
    await duplicate.wait_for_listener("actor")
    for actor_id in ("one", "two"):
        await duplicate.enqueue(
            AudioItem(
                actor_id,
                actor_id,
                "https://example.com/same",
                resolver_reference="https://example.com/same",
                requested_by_id=actor_id,
            )
        )
    with pytest.raises(UserError, match=r"audio\.duplicate_limit"):
        await duplicate.enqueue(
            AudioItem(
                "three",
                "three",
                "https://example.com/same",
                resolver_reference="https://example.com/same",
                requested_by_id="three",
            )
        )
    await duplicate.close()


@pytest.mark.asyncio
async def test_music_move_clear_mine_and_volume_are_transport_neutral() -> None:
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    session = manager.get_or_create("guild", lambda: output)
    await session.wait_for_listener("alice")
    for source, actor in (("first", "alice"), ("second", "bob"), ("third", "alice")):
        await session.enqueue(
            AudioItem(
                source,
                source,
                f"https://example.com/{source}",
                resolver_reference=f"https://example.com/{source}",
                requested_by_id=actor,
            )
        )
    endpoints = {
        item.descriptor.name: item
        for item in build_audio_endpoints(cast(MediaService, object()), manager)
    }
    context = InvocationContext("alice", "guild", "test", "request")

    moved = await endpoints["audio.move"].invoke(
        AudioMoveRequest(from_position=3, to_position=1),
        context,
    )
    assert moved.affected_title == "third"
    volume = await endpoints["audio.set_volume"].invoke(
        AudioVolumeRequest(music_percent=75, speech_percent=125),
        context,
    )
    assert volume.music_volume_percent == 75
    assert volume.speech_volume_percent == 125
    assert volume.previous_music_volume_percent == 100
    assert volume.previous_speech_volume_percent == 100
    cleared = await endpoints["audio.clear_mine"].invoke(AudioNoArgsRequest(), context)
    assert cleared.removed_count == 2

    queue_endpoint = endpoints["audio.queue"]
    queue = await queue_endpoint.invoke(AudioQueueRequest(), context)
    assert isinstance(queue, AudioQueueResponse)
    assert [item.title for item in queue.pending] == ["second"]
    assert queue.music_volume_percent == 75
    assert queue.speech_volume_percent == 125
    await manager.close()


@pytest.mark.asyncio
async def test_volume_compare_and_set_rejects_stale_undo() -> None:
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=1, max_pending_speech=1)
    manager.get_or_create("guild", lambda: output)
    endpoints = {
        item.descriptor.name: item
        for item in build_audio_endpoints(cast(MediaService, object()), manager)
    }
    context = InvocationContext("alice", "guild", "test", "request")

    await endpoints["audio.set_volume"].invoke(
        AudioVolumeRequest(music_percent=75),
        context,
    )
    await endpoints["audio.set_volume"].invoke(
        AudioVolumeRequest(music_percent=80),
        context,
    )

    with pytest.raises(UserError, match=r"action\.undo_conflict"):
        await endpoints["audio.set_volume"].invoke(
            AudioVolumeRequest(
                music_percent=100,
                expected_music_percent=75,
            ),
            context,
        )

    queue = await endpoints["audio.queue"].invoke(AudioQueueRequest(), context)
    assert queue.music_volume_percent == 80
    await manager.close()


@pytest.mark.asyncio
async def test_volume_undo_retry_accepts_already_restored_target() -> None:
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=1, max_pending_speech=1)
    manager.get_or_create("guild", lambda: output)
    endpoints = {
        item.descriptor.name: item
        for item in build_audio_endpoints(cast(MediaService, object()), manager)
    }
    context = InvocationContext("alice", "guild", "test", "request")

    await endpoints["audio.set_volume"].invoke(
        AudioVolumeRequest(music_percent=75),
        context,
    )
    undo_request = AudioVolumeRequest(
        music_percent=100,
        expected_music_percent=75,
    )
    first = await endpoints["audio.set_volume"].invoke(undo_request, context)
    retried = await endpoints["audio.set_volume"].invoke(undo_request, context)

    assert first.music_volume_percent == 100
    assert retried.music_volume_percent == 100
    assert retried.previous_music_volume_percent == 100
    await manager.close()


def test_audio_exact_write_endpoints_have_small_typed_requests() -> None:
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    endpoints = {
        item.descriptor.name: item
        for item in build_audio_endpoints(cast(MediaService, object()), manager)
    }
    exact_names = {
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
        "audio.set_volume",
        "audio.move",
        "audio.clear_mine",
    }
    assert exact_names <= endpoints.keys()
    assert all(
        endpoints[name].descriptor.approval is ApprovalMode.WHEN_REQUESTED
        for name in exact_names
    )
    assert endpoints["audio.pause"].request_type is AudioNoArgsRequest
    assert endpoints["audio.move"].request_type is AudioMoveRequest
    assert endpoints["audio.set_volume"].request_type is AudioVolumeRequest


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
        async def resolve_audio(
            self,
            reference: str,
            *,
            workspace_id: str,
            priority: MediaPriority,
        ) -> AudioItem:
            assert workspace_id == "guild"
            assert priority is MediaPriority.INTERACTIVE
            return AudioItem(
                "resolved-stream",
                "Requested track",
                reference,
                duration_seconds=123,
                resolver_reference=reference,
                uploader="Example Artist",
                thumbnail_url="https://img.example.com/track.jpg",
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
    assert response.uploader == "Example Artist"
    assert response.thumbnail_url == "https://img.example.com/track.jpg"
    assert output.played == []
    stored = store.all()[0]
    assert stored.destination_id is None
    assert stored.waiting_actor_ids == ("requester-id",)
    assert stored.items[0].requested_by_name == "Requester"
    assert stored.items[0].uploader == "Example Artist"
    assert stored.items[0].thumbnail_url == "https://img.example.com/track.jpg"

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
async def test_search_requires_one_click_for_same_title_from_different_artists() -> None:
    class FakeMedia:
        async def search_audio(
            self,
            query: str,
            *,
            limit: int,
            workspace_id: str,
            priority: MediaPriority,
        ) -> tuple[MediaCandidate, ...]:
            assert query == "Hello"
            assert limit == 5
            assert workspace_id == "guild"
            assert priority is MediaPriority.INTERACTIVE
            return (
                MediaCandidate(
                    "https://example.com/adele",
                    "Adele - Hello (Official Video)",
                    295,
                    uploader="Adele",
                ),
                MediaCandidate(
                    "https://example.com/lionel",
                    "Lionel Richie - Hello (Official Video)",
                    327,
                    uploader="Lionel Richie",
                ),
            )

    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    endpoints = build_audio_endpoints(cast(MediaService, FakeMedia()), manager)
    search_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.search"
    )
    response = await search_endpoint.invoke(
        AudioSearchRequest(query="Hello"),
        InvocationContext("actor", "guild", "test", "request"),
    )
    assert isinstance(response, AudioSearchResponse)
    assert response.selection_required
    assert response.selected_index is None
    assert response.reason is AudioSearchReason.AMBIGUOUS_TITLE


@pytest.mark.asyncio
async def test_search_uses_explicit_artist_without_an_extra_click() -> None:
    class FakeMedia:
        async def search_audio(
            self,
            query: str,
            *,
            limit: int,
            workspace_id: str,
            priority: MediaPriority,
        ) -> tuple[MediaCandidate, ...]:
            assert workspace_id == "guild"
            assert priority is MediaPriority.INTERACTIVE
            return (
                MediaCandidate(
                    "https://example.com/wrong",
                    "Lionel Richie - Hello",
                    327,
                    uploader="Lionel Richie",
                ),
                MediaCandidate(
                    "https://example.com/right",
                    "Adele - Hello",
                    295,
                    uploader="Adele",
                ),
            )

    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    endpoints = build_audio_endpoints(cast(MediaService, FakeMedia()), manager)
    search_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.search"
    )
    response = await search_endpoint.invoke(
        AudioSearchRequest(query="Adele Hello"),
        InvocationContext("actor", "guild", "test", "request"),
    )
    assert isinstance(response, AudioSearchResponse)
    assert not response.selection_required
    assert response.selected_index == 1
    assert response.reason is AudioSearchReason.UPLOADER


@pytest.mark.asyncio
async def test_search_reuses_requesters_durable_choice_without_an_extra_click() -> None:
    candidates = (
        MediaCandidate(
            "https://example.com/first",
            "Artist One - Same",
            100,
            uploader="Artist One",
        ),
        MediaCandidate(
            "https://example.com/chosen",
            "Artist Two - Same",
            101,
            uploader="Artist Two",
        ),
    )

    class FakeMedia:
        async def search_audio(
            self,
            query: str,
            *,
            limit: int,
            workspace_id: str,
            priority: MediaPriority,
        ) -> tuple[MediaCandidate, ...]:
            assert workspace_id == "guild"
            assert priority is MediaPriority.INTERACTIVE
            return candidates

    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    session = manager.get_or_create("guild", lambda: output)
    await session.wait_for_listener("actor")
    await session.enqueue(
        AudioItem(
            "",
            "Artist Two - Same",
            candidates[1].reference,
            resolver_reference=candidates[1].reference,
            requested_by_id="actor",
        )
    )
    endpoints = build_audio_endpoints(cast(MediaService, FakeMedia()), manager)
    search_endpoint = next(
        item for item in endpoints if item.descriptor.name == "audio.search"
    )
    response = await search_endpoint.invoke(
        AudioSearchRequest(query="Same"),
        InvocationContext("actor", "guild", "test", "request"),
    )
    assert isinstance(response, AudioSearchResponse)
    assert response.selected_index == 1
    assert not response.selection_required
    assert response.reason is AudioSearchReason.HISTORY
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
    manager.get_or_create("guild-one", lambda: output)
    await manager.connect("guild-one", "voice-one")
    manager.assert_connection_capacity("guild-one")
    with pytest.raises(UserError, match=r"audio\.capacity_reached"):
        manager.assert_connection_capacity("guild-two")
    await manager.close()


@pytest.mark.asyncio
async def test_concurrent_voice_connections_cannot_exceed_process_limit() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowConnectOutput(FakeOutput):
        async def connect(self, destination_id: str) -> None:
            del destination_id
            entered.set()
            await release.wait()
            self.connected = True

    first_output = SlowConnectOutput()
    second_output = FakeOutput()
    first_output.connected = False
    second_output.connected = False
    manager = AudioSessionManager(max_active=1, max_pending_speech=3)
    manager.get_or_create("guild-one", lambda: first_output)
    manager.get_or_create("guild-two", lambda: second_output)

    first = asyncio.create_task(manager.connect("guild-one", "voice-one"))
    await entered.wait()
    second = asyncio.create_task(manager.connect("guild-two", "voice-two"))
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert results[0] is None
    assert isinstance(results[1], UserError)
    assert str(results[1]) == "audio.capacity_reached"
    assert manager.active_session_count == 1
    await manager.close()


@pytest.mark.asyncio
async def test_slow_voice_connection_does_not_block_another_reserved_guild() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowConnectOutput(FakeOutput):
        async def connect(self, destination_id: str) -> None:
            del destination_id
            entered.set()
            await release.wait()
            self.connected = True

    first_output = SlowConnectOutput()
    second_output = FakeOutput()
    first_output.connected = False
    second_output.connected = False
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    manager.get_or_create("guild-one", lambda: first_output)
    manager.get_or_create("guild-two", lambda: second_output)

    first = asyncio.create_task(manager.connect("guild-one", "voice-one"))
    await entered.wait()
    await asyncio.wait_for(
        manager.connect("guild-two", "voice-two"),
        timeout=0.5,
    )
    assert second_output.connected
    release.set()
    await first
    assert manager.active_session_count == 2
    await manager.close()


@pytest.mark.asyncio
async def test_audio_state_store_coalesces_burst_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AudioStateStore(
        tmp_path / "audio_sessions.json",
        debounce_seconds=60.0,
    )
    writes = 0
    original_write = store._write

    def counted_write() -> None:
        nonlocal writes
        writes += 1
        original_write()

    monkeypatch.setattr(store, "_write", counted_write)
    first = StoredAudioSession(
        workspace_id="guild",
        destination_id=None,
        waiting_actor_ids=(),
        loop_mode=LoopMode.NONE,
        auto_leave=True,
        speed=1.0,
        pitch=1.0,
        items=(),
        history=(),
    )
    second = StoredAudioSession(
        workspace_id="guild",
        destination_id="voice",
        waiting_actor_ids=(),
        loop_mode=LoopMode.NONE,
        auto_leave=True,
        speed=1.0,
        pitch=1.0,
        items=(),
        history=(),
    )

    await store.put(first)
    await store.put(second)
    assert writes == 0
    await store.flush()

    assert writes == 1
    assert AudioStateStore(store.path).all()[0].destination_id == "voice"
    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert "voice_activation_required" in persisted["sessions"][0]
    assert "resume_confirmation_required" not in persisted["sessions"][0]


def test_audio_state_store_migrates_legacy_resume_confirmation(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "audio_sessions.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [
                    {
                        "workspace_id": "guild",
                        "destination_id": "voice",
                        "waiting_actor_ids": [],
                        "loop_mode": "none",
                        "auto_leave": True,
                        "speed": 1.0,
                        "pitch": 1.0,
                        "items": [],
                        "history": [],
                        "resume_confirmation_required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    restored = AudioStateStore(state_path).all()

    assert len(restored) == 1
    assert restored[0].voice_activation_required is True


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
    await session.set_volume(music=0.7, speech=1.2)
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
    assert saved[0].music_volume == pytest.approx(0.7)
    assert saved[0].speech_volume == pytest.approx(1.2)
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
    assert snapshot.music_volume == pytest.approx(0.7)
    assert snapshot.speech_volume == pytest.approx(1.2)
    assert snapshot.voice_activation_required is True
    await restored_manager.close()


@pytest.mark.asyncio
async def test_audio_preferences_survive_restart_without_a_queue(tmp_path: Path) -> None:
    state_path = tmp_path / "audio_sessions.json"
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(state_path),
    )
    session = manager.get_or_create("guild", lambda: output)
    await session.set_volume(music=0.65, speech=1.15)
    await session.set_auto_leave(False)
    await session.set_loop(LoopMode.QUEUE)
    await manager.close()

    stored = AudioStateStore(state_path).all()
    assert len(stored) == 1
    assert stored[0].items == ()
    assert stored[0].history == ()

    restored_output = FakeOutput()
    restored_output.connected = False
    restored_manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(state_path),
    )
    (restored_session,) = restored_manager.restore(lambda _: restored_output)
    snapshot = await restored_session.snapshot()
    assert snapshot.music_volume == pytest.approx(0.65)
    assert snapshot.speech_volume == pytest.approx(1.15)
    assert snapshot.auto_leave is False
    assert snapshot.loop is LoopMode.QUEUE
    await restored_manager.close()


@pytest.mark.asyncio
async def test_mix_uses_multiple_seeds_and_manual_requests_keep_priority() -> None:
    supplied: list[tuple[str, ...]] = []

    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        supplied.append(seeds)
        assert limit == 30
        return tuple(
            AudioItem(
                "",
                f"automatic-{index}",
                f"https://www.youtube.com/watch?v=auto{index}",
                resolver_reference=f"https://www.youtube.com/watch?v=auto{index}",
                queue_lane=AudioQueueLane.AUTOPLAY,
            )
            for index in range(3)
        )

    output = FakeOutput()
    output.connected = False
    session = AudioSession(
        "guild",
        output,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    await session.wait_for_listener("listener")
    seeds = (
        "https://www.youtube.com/watch?v=seed1",
        "https://www.youtube.com/watch?v=seed2",
    )
    assert await session.enable_autoplay(seeds) == seeds
    for _ in range(20):
        if (await session.snapshot()).autoplay_next is not None:
            break
        await asyncio.sleep(0)
    assert supplied == [seeds]
    assert (await session.snapshot()).autoplay_next is not None

    await session.enqueue(
        AudioItem(
            "",
            "manual",
            "https://www.youtube.com/watch?v=manual",
            resolver_reference="https://www.youtube.com/watch?v=manual",
            requested_by_id="listener",
        )
    )
    waiting = await session.snapshot()
    assert [item.title for item in waiting.pending] == ["manual"]
    assert waiting.autoplay_next is None
    await session.close()


def test_radio_reservoir_is_bounded_deduplicated_and_deterministic() -> None:
    session = AudioSession(
        "guild",
        FakeOutput(),
        max_pending_speech=3,
    )
    session._history.extend(
        (
            AudioItem(
                "",
                "Recently played song",
                "https://www.youtube.com/watch?v=recent",
                uploader="Repeated Artist",
            ),
        )
    )
    candidates = (
        AudioItem(
            "",
            "Recently played song",
            "https://www.youtube.com/watch?v=recent-again",
            uploader="Repeated Artist",
        ),
        AudioItem(
            "",
            "Fresh Song",
            "https://www.youtube.com/watch?v=fresh",
            uploader="Fresh Artist",
        ),
        AudioItem(
            "",
            "Fresh Song (Official Video)",
            "https://www.youtube.com/watch?v=fresh-duplicate",
            uploader="Fresh Artist",
        ),
        AudioItem(
            "",
            "Another Song (Nightcore)",
            "https://www.youtube.com/watch?v=variant",
            uploader="Another Artist",
        ),
        AudioItem(
            "",
            "Third Song",
            "https://www.youtube.com/watch?v=third",
            uploader="Third Artist",
        ),
        AudioItem(
            "",
            "Two Hour Compilation",
            "https://www.youtube.com/watch?v=long",
            duration_seconds=7_200,
            uploader="Long Artist",
        ),
    )

    first = session._select_radio_candidates(
        candidates,
        generation=4,
        recent_references={"https://www.youtube.com/watch?v=recent"},
    )
    second = session._select_radio_candidates(
        candidates,
        generation=4,
        recent_references={"https://www.youtube.com/watch?v=recent"},
    )

    assert [item.page_url for item in first] == [item.page_url for item in second]
    assert len(first) == 3
    assert first[0].title == "Fresh Song"
    assert len({item.title for item in first}) == 3
    assert all(item.title != "Recently played song" for item in first)


@pytest.mark.asyncio
async def test_mix_refills_after_automatic_lane_is_consumed() -> None:
    supplied: list[tuple[str, ...]] = []

    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        supplied.append(seeds)
        index = len(supplied)
        assert limit == 30
        return (
            AudioItem(
                f"stream-{index}",
                f"automatic-{index}",
                f"https://www.youtube.com/watch?v=auto{index}",
                resolver_reference=f"https://www.youtube.com/watch?v=auto{index}",
                queue_lane=AudioQueueLane.AUTOPLAY,
            ),
        )

    output = FakeOutput()
    session = AudioSession(
        "guild",
        output,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    await session.connect("voice")
    seed = "https://www.youtube.com/watch?v=seed"
    await session.enable_autoplay((seed,))
    for _ in range(50):
        if output.played == ["automatic-1"]:
            break
        await asyncio.sleep(0)
    assert output.played == ["automatic-1"]

    output.release.set()
    for _ in range(50):
        if output.played[:2] == ["automatic-1", "automatic-2"]:
            break
        await asyncio.sleep(0)
    assert output.played[:2] == ["automatic-1", "automatic-2"]
    assert len(supplied) >= 2
    assert supplied[0] == (seed,)
    # Generated Radio tracks must not silently replace the listener's station
    # intent. Every refill remains anchored to the explicit seed.
    assert supplied[1] == (seed,)
    await session.close()


@pytest.mark.asyncio
async def test_idle_manual_start_reservation_wins_an_inflight_radio_refill() -> None:
    supplier_started = asyncio.Event()
    supplier_cancelled = asyncio.Event()

    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        del seeds, limit
        supplier_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            supplier_cancelled.set()
            raise
        return ()

    output = FakeOutput()
    session = AudioSession(
        "guild",
        output,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    await session.enable_autoplay(("https://www.youtube.com/watch?v=seed",))
    await supplier_started.wait()

    reservation = await session.reserve_manual_music_start()
    await supplier_cancelled.wait()
    await session.enqueue(
        AudioItem(
            "manual-stream",
            "manual-request",
            "https://www.youtube.com/watch?v=manual",
            resolver_reference="https://www.youtube.com/watch?v=manual",
            requested_by_id="listener",
        )
    )
    await reservation.release()

    for _ in range(50):
        if output.played:
            break
        await asyncio.sleep(0)
    assert output.played == ["manual-request"]
    await session.close()


@pytest.mark.asyncio
async def test_explicit_radio_seeds_replace_an_older_station_intent() -> None:
    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        del seeds, limit
        return ()

    output = FakeOutput()
    output.connected = False
    session = AudioSession(
        "guild",
        output,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    await session.enable_autoplay(("https://www.youtube.com/watch?v=old",))
    await session.enable_autoplay(
        (
            "https://www.youtube.com/watch?v=new-one",
            "https://www.youtube.com/watch?v=new-two",
        )
    )

    assert (await session.snapshot()).mix_seed_references == (
        "https://www.youtube.com/watch?v=new-one",
        "https://www.youtube.com/watch?v=new-two",
    )
    await session.close()


@pytest.mark.asyncio
async def test_audio_mix_capability_reports_station_state() -> None:
    async def supply(
        seeds: tuple[str, ...],
        limit: int,
    ) -> tuple[AudioItem, ...]:
        del seeds, limit
        return (
            AudioItem(
                "",
                "automatic",
                "https://www.youtube.com/watch?v=automatic",
                resolver_reference="https://www.youtube.com/watch?v=automatic",
                queue_lane=AudioQueueLane.AUTOPLAY,
            ),
        )

    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        autoplay_supplier=supply,
    )
    manager.get_or_create("guild", lambda: output)
    media = cast(MediaService, object())
    endpoints = {
        endpoint.descriptor.name: endpoint
        for endpoint in build_audio_endpoints(media, manager)
    }
    context = InvocationContext(
        actor_id="listener",
        workspace_id="guild",
        transport="discord",
        request_id="mix",
    )
    response = await endpoints["audio.mix"].invoke(
        AudioMixRequest(
            enabled=True,
            seed_references=("https://www.youtube.com/watch?v=seed",),
        ),
        context,
    )
    assert isinstance(response, AudioMixResponse)
    assert response.enabled is True
    assert response.seed_references == ("https://www.youtube.com/watch?v=seed",)
    await manager.close()


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
    assert snapshot.voice_activation_required is True
    await manager.close()


@pytest.mark.asyncio
async def test_auto_leave_holds_read_aloud_only_route_for_explicit_resume() -> None:
    output = FakeOutput()
    session = AudioSession("read-aloud-only", output, max_pending_speech=3)
    await session.connect("voice")

    await session.suspend()

    snapshot = await session.snapshot()
    assert output.connected is False
    assert snapshot.current is None
    assert snapshot.pending == ()
    assert snapshot.destination_id == "voice"
    assert snapshot.voice_activation_required is True
    await session.close()


@pytest.mark.asyncio
async def test_read_aloud_only_resume_hold_survives_manager_restart(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "audio_sessions.json"
    output = FakeOutput()
    output.connected = False
    manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(state_path),
    )
    session = manager.get_or_create("guild", lambda: output)
    await session.connect("voice")
    await session.suspend()
    await manager.close()

    restored_output = FakeOutput()
    restored_output.connected = False
    restored_manager = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(state_path),
    )
    restored = restored_manager.restore(lambda _: restored_output)

    assert len(restored) == 1
    snapshot = await restored[0].snapshot()
    assert snapshot.pending == ()
    assert snapshot.destination_id == "voice"
    assert snapshot.voice_activation_required is True
    await restored_manager.close()
