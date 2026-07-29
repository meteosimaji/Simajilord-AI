"""Route durable local references while retaining the remote media provider."""

from __future__ import annotations

from pathlib import Path

from simajilord.domain.audio import AudioItem
from simajilord.domain.media import (
    DownloadArtifact,
    DownloadBatch,
    DownloadFormat,
    MediaCandidate,
)
from simajilord.services.local_media import LOCAL_MEDIA_SCHEME, LocalMediaStore
from simajilord.services.media import MediaProvider


class RoutingMediaProvider:
    """Delegate local-media references locally and everything else remotely."""

    def __init__(
        self,
        remote: MediaProvider,
        local: LocalMediaStore,
    ) -> None:
        self.remote = remote
        self.local = local

    async def resolve_audio(self, reference: str) -> AudioItem:
        if reference.startswith(f"{LOCAL_MEDIA_SCHEME}://"):
            return await self.local.resolve_audio(reference)
        return await self.remote.resolve_audio(reference)

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        return await self.remote.search_audio(query, limit=limit)

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        return await self.remote.mix_audio(seed_references, limit=limit)

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact:
        return await self.remote.download(
            url,
            media_type,
            destination,
            max_bytes=max_bytes,
        )

    async def download_many(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
        max_items: int,
    ) -> DownloadBatch:
        return await self.remote.download_many(
            url,
            media_type,
            destination,
            max_bytes=max_bytes,
            max_items=max_items,
        )
