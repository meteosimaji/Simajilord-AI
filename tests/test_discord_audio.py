from __future__ import annotations

import asyncio
import threading
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simajilord.core.errors import ProviderError
from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.integrations.discord.audio import (
    DiscordAudioOutput,
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

    source = build_discord_audio_source(
        AudioItem(str(path), "Silence", path.as_uri(), volume=0.75)
    )
    stdout = source._stdout
    try:
        assert source.is_opus()
        arguments = " ".join(str(value) for value in source._process.args)
        assert "volume=0.750000" in arguments
    finally:
        source.cleanup()
    assert stdout.closed


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
        assert (
            "aresample=48000,loudnorm=I=-16:TP=-1.5:LRA=11,"
            "volume=1.250000"
        ) in arguments
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
        assert (
            "loudnorm=I=-16:TP=-1.5:LRA=11,volume=1.250000"
            in arguments
        )
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
        output._swap_music_source(
            AudioItem("source", "Replacement", "https://example.test/audio")
        )
    )
    assert await asyncio.to_thread(read_started.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released.is_set()
    assert replacement.cleaned >= 1
    assert output._preflight_poisoned is False
    await output.disconnect()
