"""Provider boundary for image generation that yields a local file."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from simajilord.domain.image import ImageGenerationModel

ImageProgressCallback = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ImageProviderResult:
    generation_seconds: float
    model: str
    width: int | None = None
    height: int | None = None


class ImageGenerationProvider(Protocol):
    async def generate(
        self,
        *,
        brief_json: str,
        destination: Path,
        width: int,
        height: int,
        seed: int,
        model: ImageGenerationModel,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageProviderResult: ...

    async def close(self) -> None: ...
