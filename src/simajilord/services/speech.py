"""Speech synthesis use case and provider port."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Protocol

from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind


class SpeechProvider(Protocol):
    async def synthesize(self, text: str, destination: Path) -> None: ...


class SpeechService:
    def __init__(
        self,
        provider: SpeechProvider,
        *,
        output_dir: Path,
        max_characters: int,
        max_concurrent: int,
    ) -> None:
        self.provider = provider
        self.output_dir = output_dir
        self.max_characters = max_characters
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    async def synthesize(self, text: str, *, title: str = "Read aloud") -> AudioItem:
        normalized = normalize_speech(text)
        if not normalized:
            raise UserError("speech.no_readable_text")
        if len(normalized) > self.max_characters:
            normalized = normalized[: self.max_characters].rstrip() + "…"

        destination = self.output_dir / f"speech-{uuid.uuid4().hex}.aiff"
        async with self._semaphore:
            try:
                await self.provider.synthesize(normalized, destination)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        return AudioItem(
            source=str(destination),
            title=title,
            page_url="local://speech",
            kind=AudioKind.SPEECH,
            owned_file=destination,
        )


def normalize_speech(text: str) -> str:
    """Produce short, predictable speech without reading raw URLs."""

    value = re.sub(r"https?://\\S+", " link ", text)
    value = re.sub(r"<@!?\\d+>", " mention ", value)
    value = re.sub(r"<#\\d+>", " channel ", value)
    value = re.sub(r"<a?:[^:>]+:\\d+>", " emoji ", value)
    return " ".join(value.split()).strip()
