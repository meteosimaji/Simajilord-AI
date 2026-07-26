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
class DownloadArtifact:
    path: Path
    title: str
    media_type: DownloadFormat
    source_url: str
    size_bytes: int
