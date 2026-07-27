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


def test_discord_source_mixes_speech_with_sidechain_music_ducking(tmp_path) -> None:
    music = tmp_path / "music.wav"
    speech = tmp_path / "speech.wav"
    for path in (music, speech):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48_000)
            output.writeframes(b"\0" * (48_000 // 10))

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
        assert "sidechaincompress=" in arguments
        assert "amix=" in arguments
        assert "[1:a]volume=0.600000[music]" in arguments
        assert "aresample=48000,volume=1.250000" in arguments
        assert "[mixed]" in arguments
        assert str(speech) in arguments
        assert source.read()
    finally:
        source.cleanup()
