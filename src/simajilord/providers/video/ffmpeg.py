"""Hardware-friendly FFmpeg H.264/VP8 encoded-frame source."""

from __future__ import annotations

import asyncio
import functools
import logging
import shutil
import struct
import subprocess
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, suppress

from simajilord.core.errors import ProviderError
from simajilord.domain.video import EncodedVideoFrame, VideoCodec, VideoProfile
from simajilord.media.video.h264 import h264_frame_is_keyframe

log = logging.getLogger(__name__)
_READ_SIZE = 64 * 1024


async def verify_ffmpeg_video() -> None:
    """Encode one local frame with every supported codec without network access."""

    for codec in VideoCodec:
        profile = VideoProfile(codec=codec, width=320, height=180, frame_rate=5)
        source = FfmpegEncodedVideoSource(
            "testsrc2=size=320x180:rate=5",
            profile,
            input_format="lavfi",
        )
        try:
            async with aclosing(source.frames()) as frames:
                frame = await anext(frames)
        except StopAsyncIteration as exc:
            raise ProviderError(f"FFmpeg produced no {codec.value} video frame.") from exc
        finally:
            await source.close()
        if frame.codec is not codec or not frame.data:
            raise ProviderError(f"FFmpeg returned an invalid {codec.value} frame.")
        log.info("FFmpeg %s video self-test passed", codec.value)


class FfmpegEncodedVideoSource:
    """Yield complete encoded frames and restart to satisfy keyframe feedback."""

    def __init__(
        self,
        source: str,
        profile: VideoProfile,
        *,
        input_format: str | None = None,
        realtime: bool = True,
    ) -> None:
        self.source = source
        self.profile = profile
        self.input_format = input_format
        self.realtime = realtime
        self._process: asyncio.subprocess.Process | None = None
        self._closed = False
        self._restart_requested = False

    async def frames(self) -> AsyncGenerator[EncodedVideoFrame]:
        """Run FFmpeg and yield frames until EOF or explicit closure."""

        timestamp = 0
        while not self._closed:
            self._restart_requested = False
            process = await self._spawn()
            self._process = process
            assert process.stdout is not None
            try:
                iterator = (
                    _read_h264_frames(process.stdout)
                    if self.profile.codec is VideoCodec.H264
                    else _read_vp8_frames(process.stdout)
                )
                async for data, keyframe in iterator:
                    yield EncodedVideoFrame(
                        data=data,
                        timestamp=timestamp,
                        keyframe=keyframe,
                        codec=self.profile.codec,
                    )
                    timestamp = (timestamp + self.profile.timestamp_step) & 0xFFFFFFFF
            except GeneratorExit:
                self._closed = True
                raise
            finally:
                await self._finish_process(process)
                if self._process is process:
                    self._process = None
            if not self._restart_requested:
                break

    async def request_keyframe(self) -> None:
        """Restart the bounded-GOP encoder so its next output is an intra frame."""

        if self._closed:
            return
        self._restart_requested = True
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()

    async def close(self) -> None:
        self._closed = True
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            await self._finish_process(process)

    async def _spawn(self) -> asyncio.subprocess.Process:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ProviderError("FFmpeg is not installed or is not available on PATH.")
        command = [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if self.realtime and self.input_format != "lavfi":
            command.append("-re")
        if self.input_format is not None:
            command.extend(("-f", self.input_format))
        command.extend(
            (
                "-i",
                self.source,
                "-an",
                "-vf",
                (
                    f"scale={self.profile.width}:{self.profile.height}:"
                    "force_original_aspect_ratio=decrease,"
                    f"pad={self.profile.width}:{self.profile.height}:(ow-iw)/2:(oh-ih)/2,"
                    "format=yuv420p"
                ),
                "-r",
                str(self.profile.frame_rate),
            )
        )
        if self.profile.codec is VideoCodec.H264:
            command.extend(self._h264_arguments())
        else:
            command.extend(self._vp8_arguments())
        command.append("pipe:1")
        return await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def _h264_arguments(self) -> tuple[str, ...]:
        keyframe_frames = self.profile.frame_rate * self.profile.keyframe_interval_seconds
        encoder = "h264_videotoolbox" if _has_encoder("h264_videotoolbox") else "libx264"
        arguments = [
            "-c:v",
            encoder,
            "-b:v",
            f"{self.profile.bitrate_kbps}k",
            "-maxrate",
            f"{self.profile.bitrate_kbps}k",
            "-bufsize",
            f"{self.profile.bitrate_kbps * 2}k",
            "-g",
            str(keyframe_frames),
        ]
        if encoder == "libx264":
            arguments.extend(
                (
                    "-preset",
                    "veryfast",
                    "-tune",
                    "zerolatency",
                    "-x264-params",
                    f"keyint={keyframe_frames}:min-keyint={keyframe_frames}:scenecut=0",
                )
            )
        arguments.extend(("-f", "h264", "-bsf:v", "h264_metadata=aud=insert"))
        return tuple(arguments)

    def _vp8_arguments(self) -> tuple[str, ...]:
        keyframe_frames = self.profile.frame_rate * self.profile.keyframe_interval_seconds
        return (
            "-c:v",
            "libvpx",
            "-deadline",
            "realtime",
            "-cpu-used",
            "6",
            "-b:v",
            f"{self.profile.bitrate_kbps}k",
            "-maxrate",
            f"{self.profile.bitrate_kbps}k",
            "-bufsize",
            f"{self.profile.bitrate_kbps * 2}k",
            "-g",
            str(keyframe_frames),
            "-f",
            "ivf",
        )

    async def _finish_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            process.kill()
            await process.wait()
        if process.stderr is not None:
            error = await process.stderr.read()
            if (
                process.returncode not in {0, -15}
                and not self._restart_requested
                and not self._closed
            ):
                detail = error.decode(errors="replace")[-1_000:]
                raise ProviderError(
                    f"FFmpeg video encoder failed: {detail or process.returncode}"
                )


@functools.lru_cache(maxsize=16)
def _has_encoder(name: str) -> bool:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return False
    result = subprocess.run(
        (executable, "-nostdin", "-hide_banner", "-encoders"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and name in result.stdout


async def _read_h264_frames(
    reader: asyncio.StreamReader,
) -> AsyncIterator[tuple[bytes, bool]]:
    """Group an Annex-B stream by Access Unit Delimiter NAL units."""

    current: list[bytes] = []
    async for unit in _read_annex_b_units(reader):
        nal_type = unit[_annex_b_prefix_size(unit)] & 0x1F
        if nal_type == 9 and current:
            frame = b"".join(current)
            yield frame, h264_frame_is_keyframe(frame)
            current = []
        current.append(unit)
    if current:
        frame = b"".join(current)
        yield frame, h264_frame_is_keyframe(frame)


async def _read_annex_b_units(reader: asyncio.StreamReader) -> AsyncIterator[bytes]:
    buffer = bytearray()
    while chunk := await reader.read(_READ_SIZE):
        buffer.extend(chunk)
        while True:
            first = _find_start_code(buffer, 0)
            if first is None:
                if len(buffer) > 4:
                    del buffer[:-4]
                break
            second = _find_start_code(buffer, first[0] + first[1])
            if second is None:
                if first[0] > 0:
                    del buffer[: first[0]]
                break
            unit = bytes(buffer[first[0] : second[0]])
            del buffer[: second[0]]
            if len(unit) > first[1]:
                yield unit
    first = _find_start_code(buffer, 0)
    if first is not None:
        unit = bytes(buffer[first[0] :])
        if len(unit) > first[1]:
            yield unit


def _find_start_code(buffer: bytearray, start: int) -> tuple[int, int] | None:
    for index in range(start, len(buffer) - 2):
        if buffer[index : index + 3] == b"\x00\x00\x01":
            prefix_size = 4 if index > 0 and buffer[index - 1] == 0 else 3
            return (index - 1, 4) if prefix_size == 4 else (index, 3)
    return None


def _annex_b_prefix_size(unit: bytes) -> int:
    if unit.startswith(b"\x00\x00\x00\x01"):
        return 4
    if unit.startswith(b"\x00\x00\x01"):
        return 3
    raise ValueError("H.264 NAL unit is missing an Annex-B start code.")


async def _read_vp8_frames(
    reader: asyncio.StreamReader,
) -> AsyncIterator[tuple[bytes, bool]]:
    header = await reader.readexactly(32)
    if header[:4] != b"DKIF" or header[8:12] != b"VP80":
        raise ProviderError("FFmpeg returned an invalid VP8 IVF stream.")
    while True:
        try:
            frame_header = await reader.readexactly(12)
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return
            raise ProviderError("The VP8 IVF stream ended inside a frame header.") from exc
        frame_size = struct.unpack_from("<I", frame_header)[0]
        if frame_size <= 0 or frame_size > 64 * 1024 * 1024:
            raise ProviderError("The VP8 IVF frame size is invalid.")
        try:
            frame = await reader.readexactly(frame_size)
        except asyncio.IncompleteReadError as exc:
            raise ProviderError("The VP8 IVF stream ended inside a frame.") from exc
        yield frame, (frame[0] & 0x01) == 0
