"""Media models shared by capabilities and provider adapters."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DownloadFormat(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"


@dataclass(frozen=True, slots=True)
class MediaReference:
    value: str


@dataclass(frozen=True, slots=True)
class MediaCandidate:
    """Stable, lightweight media metadata safe to show before resolving a stream."""

    reference: str
    title: str
    duration_seconds: float
    uploader: str | None = None
    thumbnail_url: str | None = None


@dataclass(frozen=True, slots=True)
class DownloadArtifact:
    path: Path
    title: str
    media_type: DownloadFormat
    source_url: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DownloadBatch:
    """One public post may contain several independently usable media files."""

    artifacts: tuple[DownloadArtifact, ...]
    skipped_items: int = 0
    partial: bool = False
