from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path

import pytest

from simajilord.domain.speech_stream import SpeechStream
from simajilord.services.speech import (
    SpeechSegment,
    SpeechSegmentKind,
    SpeechService,
    _stream_wave_format,
)


def wave_bytes(seconds: int = 4) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\x00\x10" * 24_000 * seconds)
    return buffer.getvalue()


class ControlledStreamingProvider:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()
        self.fail = False
        self.texts: list[str] = []

    async def supports_streaming(self, voice_id):
        return True

    async def stream_voice(self, text, **kwargs):
        self.texts.append(text)
        data = wave_bytes()
        try:
            # Exercise headers split across unrelated network packet boundaries.
            yield data[:17]
            yield data[17 : 44 + 96_000]
            await self.release.wait()
            if self.fail:
                raise OSError("network disconnected")
            yield data[44 + 96_000 :]
        finally:
            self.stopped.set()

    async def synthesize(self, text, destination):
        raise AssertionError("Streaming must not buffer a separate complete synthesis")

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_multiline_stream_keeps_author_and_long_body_in_one_synthesis(tmp_path: Path) -> None:
    provider = ControlledStreamingProvider()
    provider.release.set()
    service = SpeechService(provider, output_dir=tmp_path, chunk_characters=8, max_concurrent=1)
    body = "最初の段落です。\n" + "長い文章の途中で合成を分けません。" * 30 + "\n最後の段落です。"
    item = await service.synthesize_segments(
        (
            SpeechSegment(SpeechSegmentKind.AUTHOR, "めておさん"),
            SpeechSegment(SpeechSegmentKind.BODY, body),
        ),
        workspace_id="1",
    )
    assert provider.texts == ["めておさん。" + body]
    assert len(provider.texts[0]) > 400
    assert item.speech_stream is not None
    await asyncio.to_thread(item.speech_stream.wait_finished)
    item.cleanup()
    await service.close()


@pytest.mark.asyncio
async def test_stream_returns_early_then_feeds_one_continuous_replayable_wave(
    tmp_path: Path,
) -> None:
    provider = ControlledStreamingProvider()
    service = SpeechService(provider, output_dir=tmp_path, chunk_characters=8, max_concurrent=1)
    item = await asyncio.wait_for(service.synthesize("文章を区切らず自然に読み上げます。"), 1)
    assert item.speech_stream is not None
    assert item.duration_seconds == 4.0
    assert not provider.stopped.is_set()
    assert provider.texts == ["文章を区切らず自然に読み上げます。"]
    reader = item.speech_stream.open_reader()
    initial = reader.read()
    next_read = asyncio.create_task(asyncio.to_thread(reader.read))
    await asyncio.sleep(0.02)
    assert not next_read.done()  # Empty buffer is not EOF.
    provider.release.set()
    tail = await asyncio.wait_for(next_read, 1)
    await asyncio.to_thread(item.speech_stream.wait_finished)
    assert initial + tail + reader.read() == wave_bytes()
    replay = item.speech_stream.open_reader()
    assert replay.read() == wave_bytes()
    reader.close()
    replay.close()
    item.cleanup()
    await service.close()
    assert not list(tmp_path.glob("speech-*"))


@pytest.mark.asyncio
async def test_cleanup_cancels_inflight_synthesis_without_killing_scheduler(tmp_path: Path) -> None:
    provider = ControlledStreamingProvider()
    service = SpeechService(provider, output_dir=tmp_path, chunk_characters=400, max_concurrent=1)
    item = await service.synthesize("途中で停止")
    assert item.speech_stream is not None
    reader = item.speech_stream.open_reader()
    reader.read()
    waiting = asyncio.create_task(asyncio.to_thread(reader.read))
    item.cleanup()
    assert await asyncio.wait_for(waiting, 1) == b""
    await asyncio.wait_for(provider.stopped.wait(), 1)
    provider.release.set()
    following = await asyncio.wait_for(service.synthesize("次の発話"), 1)
    following.cleanup()
    reader.close()
    await service.close()


@pytest.mark.asyncio
async def test_midstream_failure_is_not_reported_as_completed_audio(tmp_path: Path) -> None:
    provider = ControlledStreamingProvider()
    provider.fail = True
    service = SpeechService(provider, output_dir=tmp_path, chunk_characters=400, max_concurrent=1)
    item = await service.synthesize("失敗の確認")
    assert item.speech_stream is not None
    provider.release.set()
    with pytest.raises(OSError, match="failed"):
        await asyncio.to_thread(item.speech_stream.wait_finished)
    with pytest.raises(OSError, match="failed"):
        item.speech_stream.check_error()
    item.cleanup()
    await service.close()


def test_stream_header_rejects_invalid_size_and_format() -> None:
    data = wave_bytes()
    assert _stream_wave_format(data[:44]) == (len(data), 48_000)
    for offset in (0, 4, 20, 22, 24, 28, 32, 34, 36, 40):
        broken = bytearray(data[:44])
        broken[offset] ^= 0xFF
        with pytest.raises(RuntimeError):
            _stream_wave_format(bytes(broken))


def test_stream_spool_is_bounded_and_reader_cleanup_unblocks(tmp_path: Path) -> None:
    spool = SpeechStream(tmp_path / "wave", maximum_bytes=4)
    spool.append(b"1234")
    with pytest.raises(OSError, match="size limit"):
        spool.append(b"5")
    spool.close()
