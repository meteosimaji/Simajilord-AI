from __future__ import annotations

import wave

import pytest

from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.integrations.discord.audio import (
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
    try:
        assert source.is_opus()
        arguments = " ".join(str(value) for value in source._process.args)
        assert "volume=0.750000" in arguments
    finally:
        source.cleanup()


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
