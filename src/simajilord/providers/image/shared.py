"""Bind the durable image queue to the primary Codex app-server."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from simajilord.core.errors import ProviderError
from simajilord.domain.image import ImageGenerationModel

from .base import ImageProgressCallback, ImageProviderResult


class CodexImageGenerationPort(Protocol):
    async def generate_image(
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


class SharedCodexImageProvider:
    """Late-bound image provider owned by the primary conversation runtime."""

    def __init__(self) -> None:
        self._provider: CodexImageGenerationPort | None = None

    def bind(self, provider: CodexImageGenerationPort) -> None:
        if self._provider is not None and self._provider is not provider:
            raise RuntimeError("The shared Codex image provider is already bound.")
        self._provider = provider

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
    ) -> ImageProviderResult:
        provider = self._provider
        if provider is None:
            raise ProviderError("The primary Codex app-server is not available.")
        return await provider.generate_image(
            brief_json=brief_json,
            destination=destination,
            width=width,
            height=height,
            seed=seed,
            model=model,
            on_progress=on_progress,
        )

    async def close(self) -> None:
        """The primary agent provider owns and closes the shared process."""
