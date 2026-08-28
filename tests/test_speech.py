from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from aiohttp import web

from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
    build_speech_endpoint,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import ProviderError, UserError
from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.providers.speech import MacOSSayProvider, VoicevoxSpeechProvider
from simajilord.services.audio import AudioSessionManager
from simajilord.services.audio_state import AudioStateStore
from simajilord.services.speech import (
    FairSpeechScheduler,
    SpeechSegment,
    SpeechSegmentKind,
    SpeechService,
    normalize_speech,
    speech_chunks,
)


class WaveSpeechProvider:
    async def synthesize(self, text: str, destination: Path) -> None:
        assert text == "hello"
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0" * (48_000 // 5))

    async def close(self) -> None:
        pass


class SelectableWaveSpeechProvider(WaveSpeechProvider):
    def __init__(self) -> None:
        self.voice_ids: list[int] = []

    async def synthesize_voice(
        self,
        text: str,
        destination: Path,
        voice_id: int,
    ) -> None:
        self.voice_ids.append(voice_id)
        await self.synthesize(text, destination)


class TunableWaveSpeechProvider(SelectableWaveSpeechProvider):
    def __init__(self) -> None:
        super().__init__()
        self.tunings: list[tuple[int | None, float, float]] = []

    async def synthesize_tuned(
        self,
        text: str,
        destination: Path,
        *,
        voice_id: int | None,
        speed_scale: float,
        pitch_scale: float,
    ) -> None:
        self.tunings.append((voice_id, speed_scale, pitch_scale))
        await self.synthesize(text, destination)


@pytest.mark.asyncio
async def test_speech_service_probes_duration_for_music_ducking(tmp_path: Path) -> None:
    service = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
    )

    item = await service.synthesize("hello")

    assert item.kind is AudioKind.SPEECH
    assert item.duration_seconds == pytest.approx(0.1, abs=0.02)
    assert item.owned_file is not None and item.owned_file.is_file()
    item.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_warms_optional_provider(tmp_path: Path) -> None:
    warmups = 0

    class WarmableWaveProvider(WaveSpeechProvider):
        async def warm_up(self) -> None:
            nonlocal warmups
            warmups += 1

    service = SpeechService(
        WarmableWaveProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
    )

    assert await service.warm_up() is True
    assert warmups == 1
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_skips_warmup_for_basic_provider(tmp_path: Path) -> None:
    service = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
    )

    assert await service.warm_up() is False
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_resolves_named_voice_preset(tmp_path: Path) -> None:
    provider = SelectableWaveSpeechProvider()
    service = SpeechService(
        provider,
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        voice_presets={"clear": 2, "cute": 3},
        file_suffix=".wav",
    )

    item = await service.synthesize("hello", voice_preset="cute")

    assert provider.voice_ids == [3]
    item.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_applies_tuning_with_voice_and_cache_identity(
    tmp_path: Path,
) -> None:
    provider = TunableWaveSpeechProvider()
    service = SpeechService(
        provider,
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        voice_presets={"cute": 3},
        file_suffix=".wav",
    )
    segment = SpeechSegment(
        SpeechSegmentKind.AUTHOR,
        "hello",
        cache_key="author:1",
    )

    first = await service.synthesize_segments(
        (segment,),
        workspace_id="guild",
        voice_preset="cute",
        speed_scale=1.25,
        pitch_scale=0.05,
    )
    second = await service.synthesize_segments(
        (segment,),
        workspace_id="guild",
        voice_preset="cute",
        speed_scale=1.0,
        pitch_scale=0.0,
    )

    assert provider.tunings == [(3, 1.25, 0.05)]
    assert provider.voice_ids == [3]
    assert len(tuple((tmp_path / "speech" / "cache").glob("*.wav"))) == 2
    first.cleanup()
    second.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_rejects_unknown_voice_preset(tmp_path: Path) -> None:
    service = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        voice_presets={"clear": 2},
    )

    with pytest.raises(UserError, match=r"speech\.voice_preset_invalid"):
        await service.synthesize("hello", voice_preset="missing")

    await service.close()


@pytest.mark.asyncio
async def test_speech_validation_happens_before_effect_dispatch(tmp_path: Path) -> None:
    service = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        voice_presets={"clear": 2},
    )
    dispatches = 0

    async def before_synthesis() -> None:
        nonlocal dispatches
        dispatches += 1

    with pytest.raises(UserError, match=r"speech\.no_readable_text"):
        await service.synthesize(
            "   ",
            before_synthesis=before_synthesis,
        )
    with pytest.raises(UserError, match=r"speech\.voice_preset_invalid"):
        await service.synthesize(
            "hello",
            voice_preset="missing",
            before_synthesis=before_synthesis,
        )
    with pytest.raises(UserError, match=r"speech\.voice_tuning_unavailable"):
        await service.synthesize(
            "hello",
            speed_scale=1.2,
            before_synthesis=before_synthesis,
        )
    with pytest.raises(UserError, match=r"speech\.voice_tuning_invalid"):
        await service.synthesize(
            "hello",
            pitch_scale=float("nan"),
            before_synthesis=before_synthesis,
        )

    assert dispatches == 0
    await service.close()


@pytest.mark.asyncio
async def test_speech_capability_uses_shared_audio_session(tmp_path: Path) -> None:
    class HoldingOutput:
        connected = True
        paused = False

        async def connect(self, destination_id: str) -> None:
            del destination_id

        async def play(
            self,
            item: AudioItem,
            *,
            on_started: Callable[[], Awaitable[None]] | None = None,
        ) -> None:
            del item
            if on_started is not None:
                await on_started()
            await release.wait()

        async def overlay_speech(
            self,
            music: AudioItem,
            speech: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, speech, position_seconds

        async def update_music(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, position_seconds

        async def fade_out(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
            duration_seconds: float,
        ) -> None:
            del music, position_seconds, duration_seconds

        def pause(self) -> None:
            self.paused = True

        def resume(self) -> None:
            self.paused = False

        def stop(self) -> None:
            release.set()

        async def disconnect(self) -> None:
            release.set()

    release = asyncio.Event()
    speech = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
    )
    sessions = AudioSessionManager(
        max_active=2,
        max_pending_speech=3,
        state_store=AudioStateStore(tmp_path / "audio.json"),
    )
    sessions.get_or_create("1", HoldingOutput)
    endpoint = build_speech_endpoint(speech, sessions)

    response = await endpoint.invoke(
        SpeechSpeakRequest(text="hello", title="Greeting"),
        InvocationContext(
            actor_id="7",
            workspace_id="1",
            transport="test",
            request_id="speak-1",
        ),
    )

    assert isinstance(response, SpeechSpeakResponse)
    assert response.title == "Greeting"
    assert response.playback_state == "playing"
    assert response.duration_seconds == pytest.approx(0.1, abs=0.02)
    snapshot = await sessions.require("1").snapshot()
    queued_item = snapshot.current or snapshot.pending[0]
    assert queued_item.request_id == "speak-1"
    assert queued_item.request_source == "test"
    release.set()
    await sessions.close()
    await speech.close()


@pytest.mark.asyncio
async def test_blank_speech_releases_fifo_reservation_for_the_next_request(
    tmp_path: Path,
) -> None:
    class ImmediateOutput:
        connected = True
        paused = False

        async def connect(self, destination_id: str) -> None:
            del destination_id

        async def play(
            self,
            item: AudioItem,
            *,
            on_started: Callable[[], Awaitable[None]] | None = None,
        ) -> None:
            del item
            if on_started is not None:
                await on_started()

        async def overlay_speech(
            self,
            music: AudioItem,
            speech: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, speech, position_seconds

        async def update_music(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, position_seconds

        async def fade_out(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
            duration_seconds: float,
        ) -> None:
            del music, position_seconds, duration_seconds

        def pause(self) -> None:
            self.paused = True

        def resume(self) -> None:
            self.paused = False

        def stop(self) -> None:
            pass

        async def disconnect(self) -> None:
            self.connected = False

    speech = SpeechService(
        WaveSpeechProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
    )
    sessions = AudioSessionManager(max_active=1, max_pending_speech=2)
    sessions.get_or_create("guild", ImmediateOutput)
    endpoint = build_speech_endpoint(speech, sessions)

    with pytest.raises(UserError, match=r"speech\.no_readable_text"):
        await endpoint.invoke(
            SpeechSpeakRequest(text="  \n "),
            InvocationContext("first", "guild", "test", "blank-request"),
        )

    response = await asyncio.wait_for(
        endpoint.invoke(
            SpeechSpeakRequest(text="hello"),
            InvocationContext("second", "guild", "test", "valid-request"),
        ),
        timeout=1,
    )

    assert response.queue_position == 1
    await sessions.close()
    await speech.close()


@pytest.mark.asyncio
async def test_single_speech_item_keeps_fifo_while_synthesis_runs(
    tmp_path: Path,
) -> None:
    long_synthesis_started = asyncio.Event()
    release_long_synthesis = asyncio.Event()
    later_synthesis_completed = asyncio.Event()
    first_play_started = asyncio.Event()
    release_playback = asyncio.Event()

    def write_wave(destination: Path) -> None:
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0" * 9_600)

    class ControlledProvider:
        async def synthesize(self, text: str, destination: Path) -> None:
            if text == "hello":
                write_wave(destination)
                later_synthesis_completed.set()
                return
            long_synthesis_started.set()
            await release_long_synthesis.wait()
            write_wave(destination)

        async def close(self) -> None:
            pass

    class HoldingOutput:
        connected = True
        paused = False

        async def connect(self, destination_id: str) -> None:
            del destination_id

        async def play(
            self,
            item: AudioItem,
            *,
            on_started: Callable[[], Awaitable[None]] | None = None,
        ) -> None:
            del item
            first_play_started.set()
            if on_started is not None:
                await on_started()
            await release_playback.wait()

        async def overlay_speech(
            self,
            music: AudioItem,
            speech: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, speech, position_seconds

        async def update_music(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, position_seconds

        async def fade_out(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
            duration_seconds: float,
        ) -> None:
            del music, position_seconds, duration_seconds

        def pause(self) -> None:
            self.paused = True

        def resume(self) -> None:
            self.paused = False

        def stop(self) -> None:
            release_playback.set()

        async def disconnect(self) -> None:
            self.connected = False
            release_playback.set()

    speech = SpeechService(
        ControlledProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=400,
        max_concurrent=2,
        max_provider_calls=2,
        file_suffix=".wav",
    )
    sessions = AudioSessionManager(max_active=1, max_pending_speech=3)
    sessions.get_or_create("guild", HoldingOutput)
    endpoint = build_speech_endpoint(speech, sessions)
    long_text = "長文" * 35
    long_task = asyncio.create_task(
        endpoint.invoke(
            SpeechSpeakRequest(text=long_text, title="Long"),
            InvocationContext("first", "guild", "test", "long-request"),
        )
    )

    await asyncio.wait_for(long_synthesis_started.wait(), timeout=1)
    later_task = asyncio.create_task(
        endpoint.invoke(
            SpeechSpeakRequest(text="hello", title="Later"),
            InvocationContext("second", "guild", "test", "later-request"),
        )
    )
    await asyncio.sleep(0)

    assert not long_task.done()
    assert not later_task.done()
    assert not later_synthesis_completed.is_set()
    snapshot = await sessions.require("guild").snapshot()
    assert snapshot.current is None
    assert snapshot.pending == ()

    release_long_synthesis.set()
    await asyncio.wait_for(first_play_started.wait(), timeout=1)
    await asyncio.wait_for(later_synthesis_completed.wait(), timeout=1)
    long_response = await asyncio.wait_for(long_task, timeout=1)
    later_response = await asyncio.wait_for(later_task, timeout=1)

    assert long_response.queue_position == 1
    assert later_response.queue_position == 2
    snapshot = await sessions.require("guild").snapshot()
    assert snapshot.current is not None
    assert snapshot.current.request_id == "long-request"
    assert [item.request_id for item in snapshot.pending] == ["later-request"]
    release_playback.set()
    await sessions.close()
    await speech.close()


def test_speech_chunks_preserve_all_text_at_natural_boundaries() -> None:
    chunks = speech_chunks("今日は晴れです。明日も晴れるでしょう。終わり", 10)

    assert all(len(chunk) <= 10 for chunk in chunks)
    assert "".join(chunks) == "今日は晴れです。明日も晴れるでしょう。終わり"


def test_speech_normalization_replaces_discord_markup_and_urls() -> None:
    normalized = normalize_speech(
        "See https://example.com/a <@123456789> <#987654321> "
        "<:dragon:456789123> <a:dance:987123456>"
    )

    assert normalized == "See link mention channel emoji emoji"


def test_speech_normalization_keeps_visual_line_breaks_in_one_request() -> None:
    normalized = normalize_speech("投稿者\n一行目  です\n\n二行目です")

    assert normalized == "投稿者\n一行目 です\n二行目です"
    assert speech_chunks(normalized, 100) == (normalized,)


@pytest.mark.asyncio
async def test_fair_speech_scheduler_round_robins_waiting_guilds() -> None:
    scheduler = FairSpeechScheduler(1)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def operation(label: str, *, block: bool = False) -> str:
        order.append(label)
        if block:
            first_started.set()
            await release_first.wait()
        return label

    first = asyncio.create_task(scheduler.run("guild-a", lambda: operation("a1", block=True)))
    await first_started.wait()
    a2 = asyncio.create_task(scheduler.run("guild-a", lambda: operation("a2")))
    a3 = asyncio.create_task(scheduler.run("guild-a", lambda: operation("a3")))
    b1 = asyncio.create_task(scheduler.run("guild-b", lambda: operation("b1")))
    await asyncio.sleep(0)
    release_first.set()
    assert await asyncio.gather(first, a2, a3, b1) == ["a1", "a2", "a3", "b1"]
    assert order == ["a1", "b1", "a2", "a3"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_fair_speech_scheduler_serializes_each_guild_but_runs_other_guilds(
) -> None:
    scheduler = FairSpeechScheduler(2)
    first_started = asyncio.Event()
    other_started = asyncio.Event()
    release_first = asyncio.Event()
    same_guild_second_started = asyncio.Event()

    async def first_operation() -> None:
        first_started.set()
        await release_first.wait()

    async def same_guild_operation() -> None:
        same_guild_second_started.set()

    async def other_guild_operation() -> None:
        other_started.set()

    first = asyncio.create_task(scheduler.run("guild-a", first_operation))
    await first_started.wait()
    same_guild = asyncio.create_task(
        scheduler.run("guild-a", same_guild_operation)
    )
    other_guild = asyncio.create_task(
        scheduler.run("guild-b", other_guild_operation)
    )

    await asyncio.wait_for(other_started.wait(), timeout=1.0)
    assert not same_guild_second_started.is_set()
    release_first.set()
    await asyncio.gather(first, same_guild, other_guild)
    assert same_guild_second_started.is_set()
    await scheduler.close()


@pytest.mark.asyncio
async def test_speech_service_synthesizes_at_most_two_parts_in_parallel(
    tmp_path: Path,
) -> None:
    active = 0
    maximum_active = 0
    calls: list[str] = []

    class ParallelWaveProvider:
        async def synthesize(self, text: str, destination: Path) -> None:
            nonlocal active, maximum_active
            calls.append(text)
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.02)
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(48_000)
                output.writeframes(b"\0" * 9_600)
            active -= 1

        async def close(self) -> None:
            pass

    service = SpeechService(
        ParallelWaveProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=7,
        max_concurrent=1,
        max_parallel_parts=2,
        file_suffix=".wav",
    )

    item = await service.synthesize("first second third", workspace_id="guild")

    assert calls == ["first", "second", "third"]
    assert maximum_active == 2
    assert item.duration_seconds == pytest.approx(0.3, abs=0.04)
    item.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_speech_queue_is_reserved_before_provider_work_starts(
    tmp_path: Path,
) -> None:
    provider_started = asyncio.Event()
    release_provider = asyncio.Event()
    calls = 0

    class BlockingWaveProvider:
        async def synthesize(self, text: str, destination: Path) -> None:
            nonlocal calls
            del text
            calls += 1
            provider_started.set()
            await release_provider.wait()
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(48_000)
                output.writeframes(b"\0" * 9_600)

        async def close(self) -> None:
            pass

    class DisconnectedOutput:
        connected = False
        paused = False

        async def connect(self, destination_id: str) -> None:
            del destination_id
            self.connected = True

        async def play(
            self,
            item: AudioItem,
            *,
            on_started: Callable[[], Awaitable[None]] | None = None,
        ) -> None:
            del item
            if on_started is not None:
                await on_started()

        async def overlay_speech(
            self,
            music: AudioItem,
            speech: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, speech, position_seconds

        async def update_music(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
        ) -> None:
            del music, position_seconds

        async def fade_out(
            self,
            music: AudioItem,
            *,
            position_seconds: float,
            duration_seconds: float,
        ) -> None:
            del music, position_seconds, duration_seconds

        def pause(self) -> None:
            pass

        def resume(self) -> None:
            pass

        def stop(self) -> None:
            pass

        async def disconnect(self) -> None:
            self.connected = False

    speech = SpeechService(
        BlockingWaveProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        file_suffix=".wav",
    )
    sessions = AudioSessionManager(max_active=1, max_pending_speech=1)
    sessions.get_or_create("guild", DisconnectedOutput)
    endpoint = build_speech_endpoint(speech, sessions)
    context = InvocationContext("actor", "guild", "test", "request")

    first = asyncio.create_task(
        endpoint.invoke(SpeechSpeakRequest(text="first"), context)
    )
    await provider_started.wait()
    with pytest.raises(UserError, match=r"speech\.queue_full"):
        await endpoint.invoke(SpeechSpeakRequest(text="second"), context)
    assert calls == 1

    release_provider.set()
    await first
    await sessions.close()
    await speech.close()


@pytest.mark.asyncio
async def test_semantic_author_segment_cache_is_reused_but_outputs_are_owned(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class RecordingWaveProvider:
        cache_identity = "recording:speaker=1"

        async def synthesize(self, text: str, destination: Path) -> None:
            assert destination.suffix == ".wav"
            calls.append(text)
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(48_000)
                output.writeframes(b"\0" * 9_600)

        async def close(self) -> None:
            pass

    service = SpeechService(
        RecordingWaveProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=100,
        max_concurrent=1,
        file_suffix=".wav",
    )
    segment = SpeechSegment(
        SpeechSegmentKind.AUTHOR,
        "めておさん",
        cache_key="author:1:めてお",
    )

    first = await service.synthesize_segments(
        (segment,),
        workspace_id="guild-one",
    )
    second = await service.synthesize_segments(
        (segment,),
        workspace_id="guild-two",
    )

    assert calls == ["めておさん"]
    assert first.owned_file != second.owned_file
    assert first.owned_file is not None and first.owned_file.is_file()
    assert second.owned_file is not None and second.owned_file.is_file()
    first.cleanup()
    second.cleanup()
    assert tuple((tmp_path / "speech" / "cache").glob("*.wav"))
    await service.close()


@pytest.mark.asyncio
async def test_speech_service_joins_long_input_without_truncating(tmp_path: Path) -> None:
    spoken: list[str] = []

    class RecordingProvider:
        async def synthesize(self, text: str, destination: Path) -> None:
            spoken.append(text)
            with wave.open(str(destination), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(48_000)
                output.writeframes(b"\0" * 9_600)

        async def close(self) -> None:
            pass

    service = SpeechService(
        RecordingProvider(),
        output_dir=tmp_path / "speech",
        chunk_characters=6,
        max_concurrent=1,
        file_suffix=".wav",
    )

    item = await service.synthesize("hello wonderful world")

    assert "".join(spoken) == "hellowonderfulworld"
    assert len(spoken) > 1
    assert item.owned_file is not None and item.owned_file.is_file()
    assert item.duration_seconds > 0.2
    item.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_voicevox_provider_uses_two_stage_api_and_writes_valid_wave(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []
    synthesis_payloads: list[dict[str, object]] = []
    version_calls = 0

    async def version(_: web.Request) -> web.Response:
        nonlocal version_calls
        version_calls += 1
        return web.json_response("0.25.1")

    async def audio_query(request: web.Request) -> web.Response:
        calls.append(("query", request.query["text"]))
        expected_speaker = "8" if request.query["text"] == "調整" else "3"
        assert request.query["speaker"] == expected_speaker
        return web.json_response(
            {"accent_phrases": [], "speedScale": 1.0, "pitchScale": 0.0}
        )

    async def synthesis(request: web.Request) -> web.Response:
        calls.append(("synthesis", request.query["speaker"]))
        synthesis_payloads.append(await request.json())
        return web.Response(body=_wave_bytes(), content_type="audio/wav")

    async def initialize_speaker(request: web.Request) -> web.Response:
        calls.append(("initialize", request.query["speaker"]))
        assert request.query["skip_reinit"] == "true"
        return web.Response(status=204)

    application = web.Application()
    application.router.add_get("/version", version)
    application.router.add_post("/audio_query", audio_query)
    application.router.add_post("/synthesis", synthesis)
    application.router.add_post("/initialize_speaker", initialize_speaker)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]

    provider = VoicevoxSpeechProvider(
        base_url=f"http://127.0.0.1:{port}",
        speaker_id=3,
        timeout_seconds=5.0,
        engine_path=None,
        auto_start=False,
        preload_voice_ids=(2,),
    )
    destination = tmp_path / "speech.wav"
    try:
        await provider.warm_up()
        await provider.synthesize("こんにちは", destination)
        await provider.synthesize("さようなら", tmp_path / "speech-2.wav")
        await provider.synthesize_tuned(
            "調整",
            tmp_path / "speech-3.wav",
            voice_id=8,
            speed_scale=1.3,
            pitch_scale=0.07,
        )
    finally:
        await provider.close()
        await runner.cleanup()

    assert calls == [
        ("initialize", "2"),
        ("query", "こんにちは"),
        ("synthesis", "3"),
        ("query", "さようなら"),
        ("synthesis", "3"),
        ("query", "調整"),
        ("synthesis", "8"),
    ]
    assert synthesis_payloads == [
        {"accent_phrases": [], "speedScale": 1.0, "pitchScale": 0.0},
        {"accent_phrases": [], "speedScale": 1.0, "pitchScale": 0.0},
        {"accent_phrases": [], "speedScale": 1.3, "pitchScale": 0.07},
    ]
    assert version_calls == 1
    assert destination.read_bytes() == _wave_bytes()
    assert destination.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_voicevox_provider_reports_unavailable_engine(tmp_path: Path) -> None:
    provider = VoicevoxSpeechProvider(
        base_url="http://127.0.0.1:59999",
        speaker_id=3,
        timeout_seconds=0.2,
        engine_path=None,
        auto_start=False,
    )
    try:
        with pytest.raises(ProviderError, match="not responding"):
            await provider.synthesize("hello", tmp_path / "speech.wav")
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_macos_provider_defers_platform_check_until_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("simajilord.providers.speech.macos.shutil.which", lambda _: None)

    provider = MacOSSayProvider("Samantha")

    with pytest.raises(ProviderError, match="unavailable"):
        await provider.synthesize("hello", tmp_path / "speech.aiff")


def _wave_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\0" * 2_400)
    return output.getvalue()
