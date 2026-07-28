"""Media use cases independent of chat transports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from simajilord.domain.audio import AudioItem
from simajilord.domain.media import DownloadArtifact, DownloadFormat, MediaCandidate


class MediaProvider(Protocol):
    async def resolve_audio(self, reference: str) -> AudioItem: ...

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]: ...

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]: ...

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

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        return await self.provider.search_audio(query, limit=limit)

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        return await self.provider.mix_audio(seed_references, limit=limit)

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
