from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from time import time

import pytest

from simajilord.core.errors import UserError
from simajilord.services.local_media import LocalMediaStore


def _tone(path: Path, *, frequency: int = 440, duration: float = 0.4) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for the local-media integration test")
    subprocess.run(
        (
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={frequency}:duration={duration}",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(path),
        ),
        check=True,
        timeout=20,
    )


@pytest.mark.asyncio
async def test_local_media_is_content_addressed_and_restart_safe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recording.wav"
    _tone(source)
    state_path = tmp_path / "audio_sessions.json"
    store = LocalMediaStore(
        tmp_path / "store",
        max_file_bytes=1_000_000,
        max_cache_bytes=2_000_000,
        max_duration_seconds=60,
        audio_state_path=state_path,
    )

    first = await store.import_file(
        source,
        original_filename="Discord recording.wav",
        content_type="audio/wav",
        source_jump_url=(
            "https://discord.com/channels/1415260807494766627/"
            "1415260808103067670/1531170172465971200"
        ),
        uploaded_by_id="1307345055924617317",
        uploaded_by_name="Meteo",
    )
    duplicate = await store.import_file(
        source,
        original_filename="renamed.wav",
        content_type="audio/wav",
        source_jump_url=None,
        uploaded_by_id=None,
        uploaded_by_name=None,
    )

    assert first.reference == duplicate.reference
    assert first.reference.startswith("local-media://")
    assert first.path.is_file()
    assert first.duration_seconds == pytest.approx(0.4, abs=0.05)
    assert first.source_jump_url is not None

    reopened = LocalMediaStore(
        tmp_path / "store",
        max_file_bytes=1_000_000,
        max_cache_bytes=2_000_000,
        max_duration_seconds=60,
        audio_state_path=state_path,
    )
    playable = await reopened.resolve_audio(first.reference)
    assert playable.source == str(first.path)
    assert playable.resolver_reference == first.reference
    assert playable.title == "Discord recording.wav"
    assert playable.uploader == "Meteo"


@pytest.mark.asyncio
async def test_local_media_rejects_non_media_content_type(tmp_path: Path) -> None:
    source = tmp_path / "payload.txt"
    source.write_text("not media", encoding="utf-8")
    store = LocalMediaStore(
        tmp_path / "store",
        max_file_bytes=1_000,
        max_cache_bytes=2_000,
        max_duration_seconds=60,
    )

    with pytest.raises(UserError, match=r"local_media\.content_type_unsupported"):
        await store.import_file(
            source,
            original_filename="payload.txt",
            content_type="text/plain",
            source_jump_url=None,
            uploaded_by_id=None,
            uploaded_by_name=None,
        )


@pytest.mark.asyncio
async def test_local_media_lru_does_not_evict_a_persisted_queue_item(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first.wav"
    second_source = tmp_path / "second.wav"
    _tone(first_source, frequency=440, duration=0.4)
    _tone(second_source, frequency=880, duration=0.4)
    state_path = tmp_path / "audio_sessions.json"
    store = LocalMediaStore(
        tmp_path / "store",
        max_file_bytes=50_000,
        max_cache_bytes=50_000,
        max_duration_seconds=60,
        audio_state_path=state_path,
    )
    first = await store.import_file(
        first_source,
        original_filename="first.wav",
        content_type="audio/wav",
        source_jump_url=None,
        uploaded_by_id=None,
        uploaded_by_name=None,
    )
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [
                    {
                        "items": [{"reference": first.reference}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserError, match=r"local_media\.cache_full"):
        await store.import_file(
            second_source,
            original_filename="second.wav",
            content_type="audio/wav",
            source_jump_url=None,
            uploaded_by_id=None,
            uploaded_by_name=None,
        )

    assert first.path.is_file()
    record = await store.record(first.reference)
    assert record is not None
    assert record.reference_count == 1


@pytest.mark.asyncio
async def test_local_media_retention_preserves_persisted_queue_item(
    tmp_path: Path,
) -> None:
    queued_source = tmp_path / "queued.wav"
    expired_source = tmp_path / "expired.wav"
    _tone(queued_source, frequency=440, duration=0.4)
    _tone(expired_source, frequency=880, duration=0.4)
    state_path = tmp_path / "audio_sessions.json"
    store = LocalMediaStore(
        tmp_path / "store",
        max_file_bytes=1_000_000,
        max_cache_bytes=2_000_000,
        max_duration_seconds=60,
        audio_state_path=state_path,
    )
    queued = await store.import_file(
        queued_source,
        original_filename="queued.wav",
        content_type="audio/wav",
        source_jump_url=None,
        uploaded_by_id=None,
        uploaded_by_name=None,
    )
    expired = await store.import_file(
        expired_source,
        original_filename="expired.wav",
        content_type="audio/wav",
        source_jump_url=None,
        uploaded_by_id=None,
        uploaded_by_name=None,
    )
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "sessions": [{"items": [{"reference": queued.reference}]}],
            }
        ),
        encoding="utf-8",
    )
    old_epoch = int(time()) - 31 * 24 * 60 * 60
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE local_media SET last_used_epoch = ?",
            (old_epoch,),
        )

    removed = await store.cleanup_expired(
        before_epoch=int(time()) - 30 * 24 * 60 * 60
    )

    assert removed == 1
    assert await store.record(queued.reference) is not None
    assert await store.record(expired.reference) is None
    assert queued.path.is_file()
    assert not expired.path.exists()
