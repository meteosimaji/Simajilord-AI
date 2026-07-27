"""Speech synthesis use case and provider port."""

from __future__ import annotations

import asyncio
import re
import shutil
import uuid
from pathlib import Path
from typing import Protocol

from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind


class SpeechProvider(Protocol):
    async def synthesize(self, text: str, destination: Path) -> None: ...

    async def close(self) -> None: ...


class SpeechService:
    def __init__(
        self,
        provider: SpeechProvider,
        *,
        output_dir: Path,
        chunk_characters: int,
        max_concurrent: int,
        file_suffix: str = ".aiff",
    ) -> None:
        if not re.fullmatch(r"\.[a-z0-9]{2,5}", file_suffix):
            raise ValueError("Speech file suffix is invalid.")
        if chunk_characters < 1:
            raise ValueError("Speech chunk size must be positive.")
        self.provider = provider
        self.output_dir = output_dir
        self.chunk_characters = chunk_characters
        self.file_suffix = file_suffix
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def synthesize(self, text: str, *, title: str = "Read aloud") -> AudioItem:
        normalized = normalize_speech(text)
        if not normalized:
            raise UserError("speech.no_readable_text")

        destination = self.output_dir / f"speech-{uuid.uuid4().hex}{self.file_suffix}"
        chunks = speech_chunks(normalized, self.chunk_characters)
        async with self._semaphore:
            try:
                if len(chunks) == 1:
                    await self.provider.synthesize(chunks[0], destination)
                else:
                    await self._synthesize_chunks(chunks, destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
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
        await self.provider.close()

    async def _synthesize_chunks(
        self,
        chunks: tuple[str, ...],
        destination: Path,
    ) -> None:
        parts: list[Path] = []
        manifest = destination.with_suffix(".concat.txt")
        try:
            for index, chunk in enumerate(chunks, start=1):
                part = destination.with_name(
                    f"{destination.stem}-part-{index:04d}{self.file_suffix}"
                )
                await self.provider.synthesize(chunk, part)
                parts.append(part)
            await _concatenate_audio(parts, manifest=manifest, destination=destination)
        finally:
            manifest.unlink(missing_ok=True)
            for part in parts:
                part.unlink(missing_ok=True)


def normalize_speech(text: str) -> str:
    """Produce short, predictable speech without reading raw URLs."""

    value = re.sub(r"https?://\S+", " link ", text)
    value = re.sub(r"<@!?\d+>", " mention ", value)
    value = re.sub(r"<#\d+>", " channel ", value)
    value = re.sub(r"<a?:[^:>]+:\d+>", " emoji ", value)
    return " ".join(value.split()).strip()


def speech_chunks(text: str, maximum: int) -> tuple[str, ...]:
    """Split without dropping text, preferring natural sentence boundaries."""

    if maximum < 1:
        raise ValueError("Speech chunk size must be positive.")
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
        return 0.0
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
        return 0.0
    if process.returncode != 0:
        return 0.0
    try:
        return max(0.0, float(stdout.decode().strip()))
    except ValueError:
        return 0.0
