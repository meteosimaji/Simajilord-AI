"""Media use cases independent of chat transports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from simajilord.domain.audio import AudioItem
from simajilord.domain.media import DownloadArtifact, DownloadFormat


class MediaProvider(Protocol):
    async def resolve_audio(self, reference: str) -> AudioItem: ...

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact: ...


class MediaService:
    def __init__(self, provider: MediaProvider) -> None:
        self.provider = provider

    async def resolve_audio(self, reference: str) -> AudioItem:
        return await self.provider.resolve_audio(reference)

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact:
        return await self.provider.download(
            url,
            media_type,
            destination,
            max_bytes=max_bytes,
        )
