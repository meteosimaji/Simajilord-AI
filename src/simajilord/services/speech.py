"""Speech synthesis use case and provider port."""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import struct
import uuid
import wave
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Protocol, TypeVar, runtime_checkable

from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind
from simajilord.services.metrics import ServiceMetricHook, ServiceOperationMetric


class SpeechProvider(Protocol):
    async def synthesize(self, text: str, destination: Path) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class SelectableSpeechProvider(Protocol):
    """Optional provider extension for choosing a voice per synthesis."""

    async def synthesize_voice(
        self,
        text: str,
        destination: Path,
        voice_id: int,
    ) -> None: ...


class SpeechSegmentKind(StrEnum):
    """Semantic role used for pacing, observability, and safe caching."""

    AUTHOR = "author"
    BODY = "body"
    ATTACHMENT = "attachment"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    kind: SpeechSegmentKind
    text: str
    cache_key: str | None = None


_T = TypeVar("_T")


@dataclass(slots=True)
class _SpeechWork:
    operation: Callable[[], Awaitable[object]]
    future: asyncio.Future[object]
    enqueued_at: float


class FairSpeechScheduler:
    """Bounded guild round-robin scheduler for a shared TTS provider."""

    def __init__(
        self,
        max_concurrent: int,
        *,
        metric_hook: ServiceMetricHook | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("TTS concurrency must be positive.")
        self.max_concurrent = max_concurrent
        self._condition = asyncio.Condition()
        self._queues: dict[str, deque[_SpeechWork]] = {}
        self._workspaces: deque[str] = deque()
        self._active_workspaces: set[str] = set()
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False
        self._metric_hook = metric_hook

    async def run(
        self,
        workspace_id: str,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        if not workspace_id.strip():
            raise ValueError("Speech workspace ID is required.")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()

        async def erased_operation() -> object:
            return await operation()

        async with self._condition:
            if self._closed:
                raise RuntimeError("Speech scheduler is closed.")
            queue = self._queues.get(workspace_id)
            if queue is None:
                queue = deque()
                self._queues[workspace_id] = queue
            if not queue and workspace_id not in self._active_workspaces:
                self._workspaces.append(workspace_id)
            queue.append(_SpeechWork(erased_operation, future, monotonic()))
            self._ensure_workers()
            self._condition.notify()
        return await future  # type: ignore[return-value]

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            workers = tuple(self._workers)
            self._condition.notify_all()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        async with self._condition:
            for queue in self._queues.values():
                for work in queue:
                    if not work.future.done():
                        work.future.cancel()
            self._queues.clear()
            self._workspaces.clear()
            self._active_workspaces.clear()
            self._workers.clear()

    def _ensure_workers(self) -> None:
        while len(self._workers) < self.max_concurrent:
            worker = asyncio.create_task(
                self._worker(),
                name=f"simajilord-tts-worker-{len(self._workers) + 1}",
            )
            self._workers.append(worker)

    async def _worker(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or bool(self._workspaces))
                if self._closed:
                    return
                workspace_id = self._workspaces.popleft()
                queue = self._queues[workspace_id]
                work = queue.popleft()
                self._active_workspaces.add(workspace_id)
            if work.future.cancelled():
                await self._finish_workspace(workspace_id)
                continue
            started_at = monotonic()
            outcome = "succeeded"
            try:
                result = await work.operation()
            except asyncio.CancelledError:
                outcome = "cancelled"
                if not work.future.done():
                    work.future.cancel()
                raise
            except Exception as exc:
                outcome = "failed"
                if not work.future.done():
                    work.future.set_exception(exc)
            else:
                if not work.future.done():
                    work.future.set_result(result)
            finally:
                finished_at = monotonic()
                await self._finish_workspace(workspace_id)
                await self._record_metric(
                    ServiceOperationMetric(
                        operation="tts.synthesize",
                        workspace_id=workspace_id,
                        wait_ms=max(0.0, (started_at - work.enqueued_at) * 1_000),
                        duration_ms=max(0.0, (finished_at - started_at) * 1_000),
                        outcome=outcome,
                    )
                )

    async def _finish_workspace(self, workspace_id: str) -> None:
        async with self._condition:
            self._active_workspaces.discard(workspace_id)
            queue = self._queues.get(workspace_id)
            if queue:
                self._workspaces.append(workspace_id)
                self._condition.notify()
            else:
                self._queues.pop(workspace_id, None)

    async def _record_metric(self, metric: ServiceOperationMetric) -> None:
        if self._metric_hook is None:
            return
        try:
            await self._metric_hook(metric)
        except Exception:
            # Observability must never break speech delivery.
            return


class SpeechService:
    def __init__(
        self,
        provider: SpeechProvider,
        *,
        output_dir: Path,
        chunk_characters: int,
        max_concurrent: int,
        max_parallel_parts: int = 2,
        max_provider_calls: int = 2,
        metric_hook: ServiceMetricHook | None = None,
        voice_presets: Mapping[str, int] | None = None,
        file_suffix: str = ".aiff",
    ) -> None:
        if not re.fullmatch(r"\.[a-z0-9]{2,5}", file_suffix):
            raise ValueError("Speech file suffix is invalid.")
        if chunk_characters < 1:
            raise ValueError("Speech chunk size must be positive.")
        if not 1 <= max_parallel_parts <= 2:
            raise ValueError("Speech part concurrency must be between 1 and 2.")
        if max_provider_calls < 1:
            raise ValueError("Speech provider concurrency must be positive.")
        self.provider = provider
        self.output_dir = output_dir
        self.chunk_characters = chunk_characters
        self.file_suffix = file_suffix
        self._scheduler = FairSpeechScheduler(
            max_concurrent,
            metric_hook=metric_hook,
        )
        self._max_parallel_parts = max_parallel_parts
        # This is intentionally separate from the fair job scheduler and the
        # per-job part limit. It is the final, process-wide pressure valve for
        # one SpeechService/provider instance shared by every workspace.
        self._provider_limit = asyncio.Semaphore(max_provider_calls)
        self._cache_dir = output_dir / "cache"
        self._cache_locks: dict[str, asyncio.Lock] = {}
        self._voice_presets = dict(voice_presets or {})
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def synthesize(
        self,
        text: str,
        *,
        title: str = "Read aloud",
        workspace_id: str = "default",
        voice_preset: str | None = None,
    ) -> AudioItem:
        return await self.synthesize_segments(
            (SpeechSegment(SpeechSegmentKind.BODY, text),),
            title=title,
            workspace_id=workspace_id,
            voice_preset=voice_preset,
        )

    async def synthesize_segments(
        self,
        segments: tuple[SpeechSegment, ...],
        *,
        title: str = "Read aloud",
        workspace_id: str,
        voice_preset: str | None = None,
    ) -> AudioItem:
        normalized_segments: list[SpeechSegment] = []
        for segment in segments:
            normalized_text = normalize_speech(segment.text)
            if normalized_text:
                normalized_segments.append(
                    SpeechSegment(
                        kind=segment.kind,
                        text=normalized_text,
                        cache_key=segment.cache_key,
                    )
                )
        prepared = tuple(normalized_segments)
        if not prepared:
            raise UserError("speech.no_readable_text")
        voice_id = self._resolve_voice_id(voice_preset)

        return await self._scheduler.run(
            workspace_id,
            lambda: self._synthesize_job(prepared, title=title, voice_id=voice_id),
        )

    async def _synthesize_job(
        self,
        prepared: tuple[SpeechSegment, ...],
        *,
        title: str,
        voice_id: int | None,
    ) -> AudioItem:
        destination = self.output_dir / f"speech-{uuid.uuid4().hex}{self.file_suffix}"
        parts: list[Path] = []
        planned_parts: list[tuple[SpeechSegment, str, Path, bool]] = []
        manifest = destination.with_suffix(".concat.txt")
        try:
            for segment_index, segment in enumerate(prepared, start=1):
                chunks = speech_chunks(segment.text, self.chunk_characters)
                for chunk_index, chunk in enumerate(chunks, start=1):
                    part = destination.with_name(
                        f"{destination.stem}-part-{segment_index:03d}-"
                        f"{chunk_index:03d}{self.file_suffix}"
                    )
                    parts.append(part)
                    planned_parts.append((segment, chunk, part, len(chunks) == 1))
            part_limit = asyncio.Semaphore(self._max_parallel_parts)

            async def synthesize_part(
                segment: SpeechSegment,
                chunk: str,
                part: Path,
                cacheable: bool,
            ) -> None:
                async with part_limit:
                    await self._synthesize_part(
                        segment,
                        chunk=chunk,
                        destination=part,
                        cacheable=cacheable,
                        voice_id=voice_id,
                    )

            tasks = tuple(
                asyncio.create_task(
                    synthesize_part(segment, chunk, part, cacheable),
                    name=f"simajilord-tts-part-{index}",
                )
                for index, (segment, chunk, part, cacheable) in enumerate(
                    planned_parts,
                    start=1,
                )
            )
            try:
                await asyncio.gather(*tasks)
            except Exception:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            if len(parts) == 1:
                parts[0].replace(destination)
                parts.clear()
            else:
                await _concatenate_audio(
                    parts,
                    manifest=manifest,
                    destination=destination,
                )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            manifest.unlink(missing_ok=True)
            for part in parts:
                part.unlink(missing_ok=True)
        duration_seconds = await _audio_duration_seconds(destination)
        return AudioItem(
            source=str(destination),
            title=title,
            page_url="local://speech",
            kind=AudioKind.SPEECH,
            owned_file=destination,
            duration_seconds=duration_seconds,
        )

    async def close(self) -> None:
        await self._scheduler.close()
        await self.provider.close()

    async def _synthesize_part(
        self,
        segment: SpeechSegment,
        *,
        chunk: str,
        destination: Path,
        cacheable: bool,
        voice_id: int | None,
    ) -> None:
        cache_path = (
            self._cache_path(segment, chunk, voice_id=voice_id)
            if cacheable and segment.cache_key is not None
            else None
        )
        if cache_path is not None:
            lock = self._cache_locks.setdefault(cache_path.name, asyncio.Lock())
            async with lock:
                if not cache_path.is_file():
                    temporary = cache_path.with_name(
                        f".{cache_path.stem}.{uuid.uuid4().hex}{cache_path.suffix}"
                    )
                    try:
                        await self._provider_synthesize(
                            chunk,
                            temporary,
                            voice_id=voice_id,
                        )
                        temporary.chmod(0o600)
                        temporary.replace(cache_path)
                    finally:
                        temporary.unlink(missing_ok=True)
                await asyncio.to_thread(shutil.copyfile, cache_path, destination)
                destination.chmod(0o600)
            return
        await self._provider_synthesize(chunk, destination, voice_id=voice_id)

    async def _provider_synthesize(
        self,
        text: str,
        destination: Path,
        *,
        voice_id: int | None,
    ) -> None:
        async with self._provider_limit:
            if voice_id is not None and isinstance(self.provider, SelectableSpeechProvider):
                await self.provider.synthesize_voice(text, destination, voice_id)
                return
            await self.provider.synthesize(text, destination)

    def _cache_path(
        self,
        segment: SpeechSegment,
        chunk: str,
        *,
        voice_id: int | None,
    ) -> Path:
        identity = str(
            getattr(
                self.provider,
                "cache_identity",
                type(self.provider).__qualname__,
            )
        )
        digest = hashlib.sha256(
            "\0".join(
                (
                    identity,
                    "" if voice_id is None else f"voice={voice_id}",
                    segment.kind.value,
                    segment.cache_key or "",
                    chunk,
                )
            ).encode()
        ).hexdigest()
        return self._cache_dir / f"{digest}{self.file_suffix}"

    def _resolve_voice_id(self, preset: str | None) -> int | None:
        if preset is None:
            return None
        voice_id = self._voice_presets.get(preset)
        if voice_id is None:
            raise UserError("speech.voice_preset_invalid")
        return voice_id


def normalize_speech(text: str) -> str:
    """Produce predictable speech without erasing meaningful line boundaries."""

    value = re.sub(r"https?://\S+", " link ", text)
    value = re.sub(r"<@!?\d+>", " mention ", value)
    value = re.sub(r"<#\d+>", " channel ", value)
    value = re.sub(r"<a?:[^:>]+:\d+>", " emoji ", value)
    lines = (" ".join(line.split()).strip() for line in value.splitlines())
    return "\n".join(line for line in lines if line)


def speech_chunks(text: str, maximum: int) -> tuple[str, ...]:
    """Split without dropping text, preferring natural sentence boundaries."""

    if maximum < 1:
        raise ValueError("Speech chunk size must be positive.")
    paragraphs = tuple(line.strip() for line in text.splitlines() if line.strip())
    if len(paragraphs) > 1:
        paragraph_chunks: list[str] = []
        for paragraph in paragraphs:
            paragraph_chunks.extend(speech_chunks(paragraph, maximum))
        return tuple(paragraph_chunks)

    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > maximum:
        window = remaining[:maximum]
        boundary = max(
            (
                window.rfind(separator)
                for separator in ("。", "\uff01", "\uff1f", "!", "?", "\uff1b", ";", "、", ",", " ")
            ),
            default=-1,
        )
        if boundary < max(1, maximum // 3):
            boundary = maximum
            chunk = remaining[:boundary]
        else:
            boundary += 1
            chunk = remaining[:boundary]
        chunks.append(chunk.strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


async def _concatenate_audio(
    parts: list[Path],
    *,
    manifest: Path,
    destination: Path,
) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is required to join speech chunks.")
    manifest.write_text(
        "".join(f"file '{part.name}'\n" for part in parts),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    codec = "pcm_s16be" if destination.suffix.lower() in {".aif", ".aiff"} else "pcm_s16le"
    process = await asyncio.create_subprocess_exec(
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c:a",
        codec,
        str(destination),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode(errors="replace")[-500:].strip()
        raise RuntimeError(f"FFmpeg could not join speech chunks: {detail}")
    destination.chmod(0o600)


async def _audio_duration_seconds(path: Path) -> float:
    """Probe a generated speech file so music ducking ends at the right moment."""

    executable = shutil.which("ffprobe")
    if executable is None:
        return await asyncio.to_thread(_header_duration_seconds, path)
    process = await asyncio.create_subprocess_exec(
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
    except TimeoutError:
        process.kill()
        await process.wait()
        return await asyncio.to_thread(_header_duration_seconds, path)
    if process.returncode != 0:
        return await asyncio.to_thread(_header_duration_seconds, path)
    try:
        duration = float(stdout.decode().strip())
    except ValueError:
        duration = 0.0
    if duration > 0:
        return duration
    return await asyncio.to_thread(_header_duration_seconds, path)


def _header_duration_seconds(path: Path) -> float:
    """Read PCM WAV/AIFF headers without depending on ffprobe.

    Returning an invented tiny duration would truncate read-aloud playback. A
    generated file whose duration cannot be established is therefore rejected
    before it reaches the audio queue.
    """

    suffix = path.suffix.lower()
    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as stream:
                rate = stream.getframerate()
                if rate > 0:
                    duration = stream.getnframes() / rate
                    if duration > 0:
                        return duration
        except (EOFError, OSError, wave.Error):
            pass
    elif suffix in {".aif", ".aiff", ".aifc"}:
        duration = _aiff_duration_seconds(path)
        if duration > 0:
            return duration
    raise RuntimeError(f"Could not determine generated speech duration: {path.name}")


def _aiff_duration_seconds(path: Path) -> float:
    """Parse the AIFF COMM chunk, including its 80-bit sample rate."""

    try:
        with path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12 or header[:4] != b"FORM" or header[8:12] not in {
                b"AIFF",
                b"AIFC",
            }:
                return 0.0
            while True:
                chunk_header = stream.read(8)
                if len(chunk_header) != 8:
                    return 0.0
                chunk_id, size = struct.unpack(">4sI", chunk_header)
                data = stream.read(size)
                if len(data) != size:
                    return 0.0
                if size % 2:
                    stream.read(1)
                if chunk_id != b"COMM" or len(data) < 18:
                    continue
                frame_count = int(struct.unpack(">I", data[2:6])[0])
                sample_rate = _decode_ieee_extended(data[8:18])
                if frame_count > 0 and sample_rate > 0:
                    return float(frame_count / sample_rate)
                return 0.0
    except OSError:
        return 0.0


def _decode_ieee_extended(value: bytes) -> float:
    """Decode the positive 80-bit extended float used by AIFF headers."""

    if len(value) != 10:
        return 0.0
    sign_exponent, mantissa = (
        int(part) for part in struct.unpack(">HQ", value)
    )
    sign = -1.0 if sign_exponent & 0x8000 else 1.0
    exponent = sign_exponent & 0x7FFF
    if exponent == 0 and mantissa == 0:
        return 0.0
    if exponent == 0x7FFF:
        return 0.0
    return float(sign * mantissa * (2.0 ** (exponent - 16383 - 63)))
