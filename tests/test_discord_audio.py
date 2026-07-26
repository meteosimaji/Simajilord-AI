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
        AudioItem(str(path), "Silence", path.as_uri())
    )
    try:
        assert source.is_opus()
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
            speech_overlay_source=str(speech),
            speech_overlay_duration_seconds=0.1,
        )
    )
    try:
        arguments = " ".join(str(value) for value in source._process.args)
        assert "sidechaincompress=" in arguments
        assert "amix=" in arguments
        assert "[mixed]" in arguments
        assert str(speech) in arguments
        assert source.read()
    finally:
        source.cleanup()
