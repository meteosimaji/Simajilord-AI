from __future__ import annotations

import wave

import pytest

from simajilord.domain.audio import AudioItem
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
