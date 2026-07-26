"""Audio models that do not know where playback occurs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from time import monotonic

log = logging.getLogger(__name__)


class AudioKind(StrEnum):
    MUSIC = "music"
    SPEECH = "speech"


class LoopMode(StrEnum):
    NONE = "none"
    TRACK = "track"
    QUEUE = "queue"


@dataclass(slots=True)
class AudioItem:
    """Resolved audio accepted by any compatible output adapter."""

    source: str
    title: str
    page_url: str
    duration_seconds: float = 0.0
    kind: AudioKind = AudioKind.MUSIC
    http_headers: dict[str, str] | None = None
    resolver_reference: str | None = None
    resolved_at: float = field(default_factory=monotonic)
    owned_file: Path | None = None
    failure_count: int = 0
    retry_after: float = 0.0
    start_seconds: float = 0.0
    speed: float = 1.0
    pitch: float = 1.0
    requested_by_id: str | None = None
    requested_by_name: str | None = None
    played_at_epoch: int | None = None

    def clone_for_loop(self) -> AudioItem:
        """Return a loopable item without duplicating temporary-file ownership."""

        return replace(self, owned_file=None)

    def unresolved_copy(self, *, failure_count: int | None = None) -> AudioItem:
        """Keep durable metadata but discard an expiring provider stream URL."""

        return replace(
            self,
            source="",
            http_headers=None,
            resolved_at=0.0,
            owned_file=None,
            failure_count=self.failure_count if failure_count is None else failure_count,
            retry_after=0.0,
        )

    def cleanup(self) -> None:
        """Remove a temporary source owned by this item."""

        if self.owned_file is None:
            return
        try:
            self.owned_file.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary audio file: %s", self.owned_file)
        finally:
            self.owned_file = None


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    current: AudioItem | None
    pending: tuple[AudioItem, ...]
    history: tuple[AudioItem, ...]
    paused: bool
    loop: LoopMode
    destination_id: str | None = None
    waiting_actor_ids: tuple[str, ...] = ()
    auto_leave: bool = True
    position_seconds: float = 0.0
    speed: float = 1.0
    pitch: float = 1.0
