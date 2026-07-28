from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import discord
import pytest
from aiohttp import web

from simajilord.activity import build_activity_snapshot
from simajilord.activity.server import ActivityServer, _public_url
from simajilord.domain.audio import (
    AudioItem,
    AudioKind,
    LoopMode,
    QueueSnapshot,
)
from simajilord.runtime import SimajilordRuntime


def test_activity_snapshot_is_read_only_and_excludes_transport_secrets() -> None:
    current = AudioItem(
        source="https://signed.example/audio?secret=transport",
        title="Primary Colors",
        page_url="https://example.com/watch/1",
        duration_seconds=274,
        kind=AudioKind.MUSIC,
        http_headers={"Authorization": "private"},
        requested_by_id="123",
        requested_by_name="Meteo",
        uploader="PELICAN FANCLUB",
        thumbnail_url="https://images.example/cover.jpg",
        owned_file=Path("/private/cache/audio.opus"),
    )
    speech = AudioItem(
        source="/private/cache/speech.wav",
        title="read aloud",
        page_url="local://speech",
        kind=AudioKind.SPEECH,
    )
    next_track = AudioItem(
        source="https://signed.example/next",
        title="Good Morning World!",
        page_url="https://example.com/watch/2",
        duration_seconds=249,
        requested_by_name="Alice",
    )
    snapshot = QueueSnapshot(
        current=current,
        pending=(speech, next_track),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.NONE,
        destination_id="456",
        position_seconds=132.5,
        music_volume=0.82,
        speech_volume=1.1,
        autoplay_enabled=True,
        connected=True,
    )

    result = build_activity_snapshot(snapshot, read_aloud_enabled=True)

    assert result["current"] == {
        "title": "Primary Colors",
        "page_url": "https://example.com/watch/1",
        "thumbnail_url": "https://images.example/cover.jpg",
        "uploader": "PELICAN FANCLUB",
        "requested_by": "Meteo",
        "duration_seconds": 274.0,
    }
    assert result["up_next"] == [
        {
            "title": "Good Morning World!",
            "page_url": "https://example.com/watch/2",
            "thumbnail_url": None,
            "uploader": None,
            "requested_by": "Alice",
            "duration_seconds": 249.0,
        }
    ]
    assert result["read_aloud"] is True
    assert result["levels"] == {
        "music_percent": 82,
        "read_aloud_percent": 110,
    }
    serialized = repr(result)
    assert "transport" not in serialized
    assert "Authorization" not in serialized
    assert "/private/" not in serialized


def test_activity_does_not_present_speech_as_the_current_music_track() -> None:
    speech = AudioItem(
        source="/private/speech.wav",
        title="Hello",
        page_url="local://speech",
        kind=AudioKind.SPEECH,
    )
    snapshot = QueueSnapshot(
        current=speech,
        pending=(),
        history=(),
        paused=False,
        speech_active=True,
        loop=LoopMode.NONE,
        connected=True,
    )

    result = build_activity_snapshot(snapshot, read_aloud_enabled=True)

    assert result["current"] is None
    assert result["speech_active"] is True


def test_built_activity_contains_no_client_secret_name_or_source_maps() -> None:
    static_dir = (
        Path(__file__).parents[1] / "src" / "simajilord" / "activity" / "static"
    )
    files = tuple(path for path in static_dir.rglob("*") if path.is_file())

    assert any(path.name == "index.html" for path in files)
    assert not any(path.suffix == ".map" for path in files)
    bundled = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in files
    )
    assert "DISCORD_CLIENT_SECRET" not in bundled


@pytest.mark.parametrize(
    "value",
    (
        "http://example.com/cover.jpg",
        "https://user:secret@example.com/cover.jpg",
        "javascript:alert(1)",
        "https://example.com/\ncover.jpg",
        "local-media://private",
    ),
)
def test_activity_rejects_non_https_or_credentialed_public_urls(value: str) -> None:
    assert _public_url(value) is None


@pytest.mark.asyncio
async def test_activity_voice_authorization_is_rechecked_after_membership_changes() -> None:
    voice_channel = SimpleNamespace(id=10)
    member = SimpleNamespace(voice=SimpleNamespace(channel=voice_channel))

    class FakeGuild:
        def get_member(self, _: int) -> object:
            return member

    class FakeBot:
        def get_guild(self, _: int) -> object:
            return FakeGuild()

    session = SimpleNamespace(
        destination_id="10",
        output=SimpleNamespace(connected=True),
    )
    runtime = SimpleNamespace(
        audio=SimpleNamespace(find=lambda _: session),
    )
    server = ActivityServer(
        cast(discord.Client, FakeBot()),
        cast(SimajilordRuntime, runtime),
    )

    authorized = await server._authorized_session("1", "10", "7")
    assert authorized is session

    member.voice = None
    with pytest.raises(web.HTTPForbidden) as denied:
        await server._authorized_session("1", "10", "7")
    assert denied.value.text == "Join this voice channel to view its player."
