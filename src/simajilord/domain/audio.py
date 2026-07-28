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


class AudioQueueLane(StrEnum):
    """Whether a track was explicitly requested or supplied automatically."""

    REQUEST = "request"
    AUTOPLAY = "autoplay"


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
    volume: float = 1.0
    speech_overlay_volume: float = 1.0
    requested_by_id: str | None = None
    requested_by_name: str | None = None
    queue_lane: AudioQueueLane = AudioQueueLane.REQUEST
    request_source: str | None = None
    request_id: str | None = None
    requested_at_epoch: int | None = None
    played_at_epoch: int | None = None
    uploader: str | None = None
    thumbnail_url: str | None = None
    speech_overlay_source: str | None = None
    speech_overlay_owned_file: Path | None = None
    speech_overlay_duration_seconds: float = 0.0
    resume_after_overlay: bool = False
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.0

    def clone_for_loop(self) -> AudioItem:
        """Return a loopable item without duplicating temporary-file ownership."""

        return replace(
            self,
            owned_file=None,
            start_seconds=0.0,
            retry_after=0.0,
            played_at_epoch=None,
            fade_in_seconds=0.0,
            fade_out_seconds=0.0,
            speech_overlay_source=None,
            speech_overlay_owned_file=None,
            speech_overlay_duration_seconds=0.0,
            resume_after_overlay=False,
        )

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
            speech_overlay_source=None,
            speech_overlay_owned_file=None,
            speech_overlay_duration_seconds=0.0,
            resume_after_overlay=False,
        )

    def cleanup(self) -> None:
        """Remove a temporary source owned by this item."""

        self.cleanup_speech_overlay()
        owned_file = self.owned_file
        if owned_file is None:
            return
        try:
            owned_file.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary audio file: %s", owned_file)
        finally:
            self.owned_file = None

    def cleanup_speech_overlay(self) -> None:
        """Remove only the temporary read-aloud overlay after FFmpeg opens it."""

        owned_file = self.speech_overlay_owned_file
        if owned_file is None:
            return
        try:
            owned_file.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove temporary speech overlay: %s", owned_file)
        finally:
            self.speech_overlay_owned_file = None


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    current: AudioItem | None
    pending: tuple[AudioItem, ...]
    history: tuple[AudioItem, ...]
    paused: bool
    speech_active: bool
    loop: LoopMode
    destination_id: str | None = None
    waiting_actor_ids: tuple[str, ...] = ()
    auto_leave: bool = True
    position_seconds: float = 0.0
    speed: float = 1.0
    pitch: float = 1.0
    music_volume: float = 1.0
    speech_volume: float = 1.0
    autoplay_enabled: bool = False
    autoplay_next: AudioItem | None = None
    mix_seed_references: tuple[str, ...] = ()
    resume_confirmation_required: bool = False
