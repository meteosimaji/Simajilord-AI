"""Provider boundary for local image generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

ImageProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ImageProviderResult:
    generation_seconds: float
    model: str


class ImageGenerationProvider(Protocol):
    async def generate(
        self,
        *,
        caption_json: str,
        destination: Path,
        width: int,
        height: int,
        seed: int,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageProviderResult: ...
