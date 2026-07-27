from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import pytest
from aiohttp import web

from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
    build_speech_endpoint,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import ProviderError
from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.providers.speech import MacOSSayProvider, VoicevoxSpeechProvider
from simajilord.services.audio import AudioSessionManager
from simajilord.services.audio_state import AudioStateStore
from simajilord.services.speech import SpeechService, normalize_speech, speech_chunks


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
async def test_speech_capability_uses_shared_audio_session(tmp_path: Path) -> None:
    class HoldingOutput:
        connected = True
        paused = False

        async def connect(self, destination_id: str) -> None:
            del destination_id

        async def play(self, item: AudioItem) -> None:
            del item
            await release.wait()

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
    assert response.playback_state == "queued"
    assert response.duration_seconds == pytest.approx(0.1, abs=0.02)
    release.set()
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

    async def version(_: web.Request) -> web.Response:
        return web.json_response("0.25.1")

    async def audio_query(request: web.Request) -> web.Response:
        calls.append(("query", request.query["text"]))
        assert request.query["speaker"] == "3"
        return web.json_response({"accent_phrases": [], "speedScale": 1.0})

    async def synthesis(request: web.Request) -> web.Response:
        calls.append(("synthesis", request.query["speaker"]))
        assert await request.json() == {"accent_phrases": [], "speedScale": 1.0}
        return web.Response(body=_wave_bytes(), content_type="audio/wav")

    application = web.Application()
    application.router.add_get("/version", version)
    application.router.add_post("/audio_query", audio_query)
    application.router.add_post("/synthesis", synthesis)
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
    )
    destination = tmp_path / "speech.wav"
    try:
        await provider.synthesize("こんにちは", destination)
    finally:
        await provider.close()
        await runner.cleanup()

    assert calls == [("query", "こんにちは"), ("synthesis", "3")]
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
