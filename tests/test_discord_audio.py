from __future__ import annotations

import asyncio
import sys
import threading
import wave
from array import array
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.core.errors import ProviderError
from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.integrations.discord.audio import (
    DiscordAudioOutput,
    _ensure_discord_opus_loaded,
    _LiveSpeechMixer,
    _mix_stereo_s16le,
    _PrefetchedAudioSource,
    build_discord_audio_source,
    verify_ffmpeg_opus,
)


@pytest.mark.asyncio
async def test_host_ffmpeg_can_generate_discord_opus() -> None:
    await verify_ffmpeg_opus()


def test_discord_source_is_preencoded_opus(tmp_path) -> None:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0" * (48_000 // 10))

    source = build_discord_audio_source(AudioItem(str(path), "Silence", path.as_uri(), volume=0.75))
    stdout = source._stdout
    try:
        assert source.is_opus()
        arguments = " ".join(str(value) for value in source._process.args)
        assert "volume=0.750000" in arguments
    finally:
        source.cleanup()
    assert stdout.closed


def test_managed_discord_source_cleanup_is_idempotent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0" * (48_000 // 10))

    cleanup_calls = 0
    original_cleanup = discord.FFmpegOpusAudio.cleanup

    def counted_cleanup(source: discord.FFmpegOpusAudio) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        original_cleanup(source)

    monkeypatch.setattr(discord.FFmpegOpusAudio, "cleanup", counted_cleanup)
    source = build_discord_audio_source(AudioItem(str(path), "Silence", path.as_uri()))

    source.cleanup()
    source.cleanup()

    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_discord_output_reports_playback_after_voice_accepts_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Source(discord.AudioSource):
        def read(self) -> bytes:
            return b"packet"

        def is_opus(self) -> bool:
            return True

        def cleanup(self) -> None:
            events.append("cleanup")

    class Voice:
        def __init__(self) -> None:
            self.playing = False

        def is_connected(self) -> bool:
            return True

        def is_playing(self) -> bool:
            return self.playing

        def is_paused(self) -> bool:
            return False

        def play(self, source: discord.AudioSource, *, after) -> None:
            assert source.is_opus()
            events.append("voice.play")
            self.playing = True
            after(None)

        def stop(self) -> None:
            self.playing = False

        async def disconnect(self, *, force: bool) -> None:
            assert force is True

    source = Source()
    voice = Voice()
    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    output._voice = voice  # type: ignore[assignment]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio.build_discord_audio_source",
        lambda _item: source,
    )

    async def on_started() -> None:
        events.append("started")

    await output.play(
        AudioItem("music", "Music", "https://example.test/music"),
        on_started=on_started,
    )

    assert events == ["voice.play", "started", "cleanup"]
    await output.disconnect()


def test_live_speech_pcm_mix_ducks_music_and_saturates() -> None:
    music_samples = array("h", [4_000, -4_000] * 960)
    speech_samples = array("h", [1_000, -1_000] * 960)

    mixed = array(
        "h",
        _mix_stereo_s16le(
            music_samples.tobytes(),
            speech_samples.tobytes(),
        ),
    )

    assert mixed[:4] == array("h", [2_000, -2_000, 2_000, -2_000])

    loud = array("h", [32_000, -32_000] * 960)
    saturated = array("h", _mix_stereo_s16le(loud.tobytes(), loud.tobytes()))
    assert saturated[:2] == array("h", [32_767, -32_768])


@pytest.mark.asyncio
async def test_live_speech_mixer_keeps_existing_music_packets(tmp_path) -> None:
    if not _ensure_discord_opus_loaded():
        pytest.skip("The host has no loadable system Opus codec.")
    music_path = tmp_path / "music.wav"
    speech_path = tmp_path / "speech.wav"
    for path, frame_count in (
        (music_path, 48_000),
        (speech_path, 4_800),
    ):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0" * (frame_count * 4))

    music_source = build_discord_audio_source(
        AudioItem(
            str(music_path),
            "Music",
            music_path.as_uri(),
            kind=AudioKind.MUSIC,
        )
    )
    raw_speech_source = build_discord_audio_source(
        AudioItem(
            str(speech_path),
            "Speech",
            speech_path.as_uri(),
            kind=AudioKind.SPEECH,
        )
    )
    first_speech_packet = await asyncio.to_thread(raw_speech_source.read)
    assert first_speech_packet
    mixer = _LiveSpeechMixer(
        music_source,
        _PrefetchedAudioSource(raw_speech_source, first_speech_packet),
        loop=asyncio.get_running_loop(),
    )
    packets: list[bytes] = []
    try:
        for _ in range(8):
            packet = await asyncio.to_thread(mixer.read)
            assert packet
            packets.append(packet)
            await asyncio.sleep(0)
        await asyncio.wait_for(mixer.wait_finished(), timeout=1.0)
        assert len(packets) == 8
        assert mixer.read()
    finally:
        mixer.cleanup()


@pytest.mark.asyncio
async def test_live_overlay_restores_same_music_source_without_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Source(discord.AudioSource):
        def __init__(self) -> None:
            self.cleanup_calls = 0

        def read(self) -> bytes:
            return b"packet"

        def is_opus(self) -> bool:
            return True

        def cleanup(self) -> None:
            self.cleanup_calls += 1

    class Voice:
        def __init__(self, source: discord.AudioSource) -> None:
            self._source = source
            self.assignments: list[discord.AudioSource] = []

        @property
        def source(self) -> discord.AudioSource:
            return self._source

        @source.setter
        def source(self, value: discord.AudioSource) -> None:
            self._source = value
            self.assignments.append(value)

        def is_connected(self) -> bool:
            return True

        def is_playing(self) -> bool:
            return True

        async def disconnect(self, *, force: bool) -> None:
            assert force is True

    class Mixer(discord.AudioSource):
        instances: ClassVar[list[Mixer]] = []

        def __init__(
            self,
            music_source: discord.AudioSource,
            speech_source: discord.AudioSource,
            *,
            loop: asyncio.AbstractEventLoop,
        ) -> None:
            del loop
            self.music_source = music_source
            self.speech_source = speech_source
            self.detached = False
            self.cleaned = False
            self.instances.append(self)

        def read(self) -> bytes:
            return self.music_source.read()

        def is_opus(self) -> bool:
            return True

        async def wait_finished(self) -> None:
            return

        def detach_music(self) -> None:
            self.detached = True

        def cleanup(self) -> None:
            self.cleaned = True
            self.speech_source.cleanup()

    music_source = Source()
    speech_source = Source()
    voice = Voice(music_source)
    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    output._voice = voice  # type: ignore[assignment]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._ensure_discord_opus_loaded",
        lambda: True,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._LiveSpeechMixer",
        Mixer,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio.build_discord_audio_source",
        lambda _item: speech_source,
    )

    await output.overlay_speech(
        AudioItem(
            "music",
            "Music",
            "https://example.test/music",
            kind=AudioKind.MUSIC,
        ),
        AudioItem(
            "speech",
            "Speech",
            "local://speech",
            kind=AudioKind.SPEECH,
            request_id="request:p1of1",
        ),
        position_seconds=12.0,
    )
    await output.update_music(
        AudioItem(
            "music",
            "Music",
            "https://example.test/music",
            kind=AudioKind.MUSIC,
        ),
        position_seconds=12.1,
    )

    assert len(Mixer.instances) == 1
    assert voice.assignments == [Mixer.instances[0], music_source]
    assert Mixer.instances[0].detached is True
    assert Mixer.instances[0].cleaned is True
    assert music_source.cleanup_calls == 0
    assert speech_source.cleanup_calls == 1
    await output.disconnect()


@pytest.mark.asyncio
async def test_live_overlay_keeps_reconnect_compatible_fallback_without_libopus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    compatibility_swap = AsyncMock()
    output._swap_music_source = compatibility_swap  # type: ignore[method-assign]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._ensure_discord_opus_loaded",
        lambda: False,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("simajilord.integrations.discord.audio.asyncio.sleep", sleep)
    music = AudioItem(
        "https://example.test/music",
        "Music",
        "https://example.test/music",
        kind=AudioKind.MUSIC,
    )
    speech = AudioItem(
        "local-speech.wav",
        "Speech",
        "local://speech",
        kind=AudioKind.SPEECH,
        duration_seconds=1.25,
        volume=1.1,
    )

    await output.overlay_speech(music, speech, position_seconds=12.0)

    compatibility_swap.assert_awaited_once()
    overlay = compatibility_swap.await_args.args[0]
    assert overlay.source == music.source
    assert overlay.start_seconds == 12.0
    assert overlay.speech_overlay_source == speech.source
    assert overlay.speech_overlay_duration_seconds == 1.25
    assert overlay.speech_overlay_volume == 1.1
    sleep.assert_awaited_once_with(1.4)
    assert output._live_mixing_disabled is True
    await output.disconnect()


def test_discord_source_uses_bounded_fades(tmp_path) -> None:
    path = tmp_path / "fade.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0" * (48_000 // 10))
    source = build_discord_audio_source(
        AudioItem(
            str(path),
            "Fade",
            path.as_uri(),
            fade_in_seconds=0.4,
            fade_out_seconds=0.4,
        )
    )
    try:
        arguments = " ".join(str(value) for value in source._process.args)
        assert "afade=t=in:st=0:d=0.400" in arguments
        assert "afade=t=out:st=0:d=0.400" in arguments
    finally:
        source.cleanup()


def test_discord_source_keeps_music_at_a_stable_duck_level_during_speech(
    tmp_path,
) -> None:
    music = tmp_path / "music.wav"
    speech = tmp_path / "speech.wav"
    for path, frame_count in (
        (music, 48_000),
        (speech, 4_800),
    ):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0" * (frame_count * 4))

    source = build_discord_audio_source(
        AudioItem(
            str(music),
            "Music with speech",
            music.as_uri(),
            kind=AudioKind.MUSIC,
            volume=0.6,
            speech_overlay_source=str(speech),
            speech_overlay_duration_seconds=0.1,
            speech_overlay_volume=1.25,
        )
    )
    stdout = source._stdout
    try:
        arguments = " ".join(str(value) for value in source._process.args)
        assert "sidechaincompress=" not in arguments
        assert "amix=" in arguments
        assert "duration=longest" in arguments
        assert "[1:a]volume=0.600000,volume=0.250000[ducked]" in arguments
        assert ("aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11,volume=1.250000") in arguments
        assert "[mixed]" in arguments
        assert str(speech) in arguments
        packets = 0
        while source.read():
            packets += 1
        # Discord Opus packets are 20 ms. The old duration=longest graph ended
        # after the 100 ms speech input; the music source must remain near 1 s.
        assert packets >= 40
    finally:
        source.cleanup()
    assert stdout.closed


def test_standalone_speech_is_loudness_normalized_before_user_volume(
    tmp_path,
) -> None:
    speech = tmp_path / "speech.wav"
    with wave.open(str(speech), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\0" * 9_600)

    source = build_discord_audio_source(
        AudioItem(
            str(speech),
            "Read aloud",
            speech.as_uri(),
            kind=AudioKind.SPEECH,
            volume=1.25,
        )
    )
    try:
        arguments = " ".join(str(value) for value in source._process.args)
        assert "loudnorm=I=-16:TP=-1.5:LRA=11,volume=1.250000" in arguments
    finally:
        source.cleanup()


@pytest.mark.asyncio
async def test_discord_playback_watchdog_stops_missing_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SimpleNamespace(cleanup=Mock())

    class Voice:
        def __init__(self) -> None:
            self.started = False
            self.stopped = 0

        def is_connected(self) -> bool:
            return True

        def is_playing(self) -> bool:
            return self.started and self.stopped == 0

        def is_paused(self) -> bool:
            return False

        def play(self, _source: object, *, after: object) -> None:
            del _source, after
            self.started = True

        def stop(self) -> None:
            self.stopped += 1

        async def disconnect(self, *, force: bool) -> None:
            assert force is True

    voice = Voice()
    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    output._voice = voice  # type: ignore[assignment]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio.build_discord_audio_source",
        lambda _item: source,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._PLAYBACK_WATCHDOG_INTERVAL_SECONDS",
        0.005,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._PLAYBACK_COMPLETION_GRACE_SECONDS",
        0.01,
    )

    with pytest.raises(ProviderError, match="bounded completion"):
        await asyncio.wait_for(
            output.play(
                AudioItem(
                    "source",
                    "Missing callback",
                    "https://example.test/audio",
                    duration_seconds=0.001,
                )
            ),
            timeout=0.2,
        )

    assert voice.stopped == 1
    source.cleanup.assert_called_once_with()
    await output.disconnect()


@pytest.mark.asyncio
async def test_audio_source_preflight_timeout_cleans_and_joins_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released = threading.Event()

    class Replacement:
        def __init__(self) -> None:
            self.cleaned = 0

        def read(self) -> bytes:
            released.wait(timeout=1)
            return b""

        def cleanup(self) -> None:
            self.cleaned += 1
            released.set()

    replacement = Replacement()

    class Voice:
        source = SimpleNamespace(cleanup=Mock())

        def is_connected(self) -> bool:
            return True

        def is_playing(self) -> bool:
            return True

        async def disconnect(self, *, force: bool) -> None:
            assert force is True

    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    output._voice = Voice()  # type: ignore[assignment]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio.build_discord_audio_source",
        lambda _item: replacement,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio._SOURCE_PREFLIGHT_TIMEOUT_SECONDS",
        0.01,
    )

    with pytest.raises(ProviderError, match="preflight timed out"):
        await output._swap_music_source(
            AudioItem("source", "Replacement", "https://example.test/audio")
        )

    assert released.is_set()
    assert replacement.cleaned >= 1
    assert output._preflight_poisoned is False
    await output.disconnect()


@pytest.mark.asyncio
async def test_audio_source_preflight_cancellation_joins_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_started = threading.Event()
    released = threading.Event()

    class Replacement:
        def __init__(self) -> None:
            self.cleaned = 0

        def read(self) -> bytes:
            read_started.set()
            released.wait(timeout=1)
            return b""

        def cleanup(self) -> None:
            self.cleaned += 1
            released.set()

    replacement = Replacement()

    class Voice:
        source = SimpleNamespace(cleanup=Mock())

        def is_connected(self) -> bool:
            return True

        def is_playing(self) -> bool:
            return True

        async def disconnect(self, *, force: bool) -> None:
            assert force is True

    output = DiscordAudioOutput(SimpleNamespace(get_guild=lambda _guild: None), 1)
    output._voice = Voice()  # type: ignore[assignment]
    monkeypatch.setattr(
        "simajilord.integrations.discord.audio.build_discord_audio_source",
        lambda _item: replacement,
    )
    task = asyncio.create_task(
        output._swap_music_source(AudioItem("source", "Replacement", "https://example.test/audio"))
    )
    assert await asyncio.to_thread(read_started.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()
    assert replacement.cleaned >= 1
    assert output._preflight_poisoned is False
    await output.disconnect()


@pytest.mark.asyncio
async def test_streaming_opus_starts_before_tail_without_changing_audio(tmp_path):
    import io
    import math

    from simajilord.domain.speech_stream import SpeechStream
    from simajilord.integrations.discord.audio import _read_opus_audio_packet

    samples = array(
        "h", (int(4000 * math.sin(i * math.tau * 220 / 24000)) for i in range(24000 * 8))
    )
    if sys.byteorder != "little":
        samples.byteswap()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(samples.tobytes())
    data = buffer.getvalue()
    stream = SpeechStream(tmp_path / "growing.wav")
    split = 44 + 24000 * 2 * 4
    stream.append(data[:split])
    item = AudioItem(
        str(stream.path), "Speech", "local://speech", kind=AudioKind.SPEECH, speech_stream=stream
    )
    source = build_discord_audio_source(item)

    def drain(audio):
        packets = []
        while packet := _read_opus_audio_packet(audio):
            packets.append(packet)
        return packets

    try:
        first = await asyncio.wait_for(asyncio.to_thread(_read_opus_audio_packet, source), 2)
        assert first  # Four seconds are available, but the eight-second tail is not.
        assert stream.path.stat().st_size < len(data)
        stream.append(data[split:])
        stream.finish()
        streamed_packets = [first, *await asyncio.to_thread(drain, source)]
    finally:
        source.cleanup()
        stream.close()
    complete = tmp_path / "complete.wav"
    complete.write_bytes(data)
    ordinary = build_discord_audio_source(
        AudioItem(str(complete), "Speech", "local://speech", kind=AudioKind.SPEECH)
    )
    try:
        assert streamed_packets == await asyncio.to_thread(drain, ordinary)
        assert len(streamed_packets) == 401
    finally:
        ordinary.cleanup()
