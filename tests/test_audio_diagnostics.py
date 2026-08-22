from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from simajilord.core.errors import MediaError
from simajilord.diagnostics.audio import run_audio_doctor
from simajilord.domain.audio import AudioItem, AudioKind


@pytest.mark.asyncio
async def test_audio_doctor_keeps_playable_video_success_when_mix_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = AudioItem(
        source="https://media.example.test/audio",
        title="Playable video",
        page_url="https://www.youtube.com/watch?v=video",
        kind=AudioKind.MUSIC,
    )
    provider = SimpleNamespace(
        resolve_audio=AsyncMock(return_value=item),
        mix_audio=AsyncMock(
            side_effect=MediaError(
                "unavailable",
                "YouTube Mix returned no new tracks.",
            )
        ),
    )
    source = SimpleNamespace(
        read=lambda: b"opus-packet",
        is_opus=lambda: True,
        cleanup=lambda: None,
    )
    monkeypatch.setattr(
        "simajilord.diagnostics.audio.verify_ffmpeg_opus",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "simajilord.diagnostics.audio._executable_version",
        AsyncMock(return_value="available"),
    )
    monkeypatch.setattr(
        "simajilord.diagnostics.audio.YtDlpProvider",
        lambda **_kwargs: provider,
    )
    monkeypatch.setattr(
        "simajilord.diagnostics.audio.build_discord_audio_source",
        lambda _item: source,
    )

    results = await run_audio_doctor(item.page_url)

    assert "Media resolver: OK (Playable video)" in results
    assert "Opus packets: OK (5 packets, 55 bytes)" in results
    assert "YouTube Mix: unavailable (no new candidates)" in results
    provider.resolve_audio.assert_awaited_once_with(item.page_url)
    provider.mix_audio.assert_awaited_once_with((item.page_url,), limit=3)
