"""Media download capability returning an artifact to the calling adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.domain.media import DownloadFormat
from simajilord.services.media import MediaService


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    url: str
    media_type: DownloadFormat
    destination: Path
    max_bytes: int


@dataclass(frozen=True, slots=True)
class DownloadResponse:
    path: Path
    title: str
    size_bytes: int
    source_url: str


def build_download_endpoint(media: MediaService) -> CapabilityEndpoint:
    async def download(
        request: DownloadRequest,
        _: InvocationContext,
    ) -> DownloadResponse:
        artifact = await media.download(
            request.url,
            request.media_type,
            request.destination,
            max_bytes=request.max_bytes,
        )
        return DownloadResponse(
            path=artifact.path,
            title=artifact.title,
            size_bytes=artifact.size_bytes,
            source_url=artifact.source_url,
        )

    return endpoint(
        CapabilityDescriptor(
            name="media.download",
            summary="Download one supported public media URL into a bounded directory.",
            risk=RiskLevel.EXTERNAL,
            keywords=("youtube", "tiktok", "video", "audio", "save"),
            side_effects=("Uses a media site.", "Creates a temporary local file."),
        ),
        DownloadRequest,
        DownloadResponse,
        download,
    )
