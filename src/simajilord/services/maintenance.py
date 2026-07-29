"""Bounded retention and storage maintenance for local runtime data."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from simajilord.agent.memory import AgentMemoryStore
from simajilord.agent.store import AgentConversationStore
from simajilord.observability import EventJournal

from .focus_timer import FocusTimerService
from .image import ImageGenerationStore
from .local_media import LocalMediaStore
from .moderation import ModerationStore
from .speech import SpeechService

_SPEECH_CACHE_LIMIT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    """One completed maintenance pass, suitable for status presentation."""

    completed_at: datetime | None = None
    storage_used_bytes: int = 0
    storage_limit_bytes: int = 0
    over_capacity: bool = False
    event_records_removed: int = 0
    agent_requests_removed: int = 0
    agent_conversations_removed: int = 0
    agent_memories_removed: int = 0
    focus_timers_removed: int = 0
    image_jobs_removed: int = 0
    image_files_removed: int = 0
    speech_orphans_removed: int = 0
    speech_cache_files_removed: int = 0
    moderation_cache_rows_removed: int = 0
    moderation_quota_rows_removed: int = 0
    local_media_removed: int = 0

    @property
    def orphan_cleanup_removed(self) -> int:
        return (
            self.image_files_removed
            + self.speech_orphans_removed
            + self.speech_cache_files_removed
            + self.local_media_removed
        )


class DataMaintenanceService:
    """Coordinate owner-aware cleanup without touching active runtime objects."""

    def __init__(
        self,
        *,
        data_dir: Path,
        retention_days: int,
        max_data_bytes: int,
        journal: EventJournal,
        agent_store: AgentConversationStore,
        memory_store: AgentMemoryStore,
        focus_timers: FocusTimerService,
        image_store: ImageGenerationStore,
        image_output_dir: Path,
        moderation_store: ModerationStore,
        speech: SpeechService,
        local_media: LocalMediaStore,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be positive.")
        if max_data_bytes < 1:
            raise ValueError("max_data_bytes must be positive.")
        self.data_dir = data_dir.resolve()
        self.retention_days = retention_days
        self.max_data_bytes = max_data_bytes
        self.journal = journal
        self.agent_store = agent_store
        self.memory_store = memory_store
        self.focus_timers = focus_timers
        self.image_store = image_store
        self.image_output_dir = image_output_dir.resolve()
        self.moderation_store = moderation_store
        self.speech = speech
        self.local_media = local_media
        self._lock = asyncio.Lock()
        self._last_report = MaintenanceReport(
            storage_used_bytes=_directory_size(self.data_dir),
            storage_limit_bytes=max_data_bytes,
        )

    @property
    def last_report(self) -> MaintenanceReport:
        return self._last_report

    async def run(self, *, now: datetime | None = None) -> MaintenanceReport:
        """Run one explicit pass; callers decide when maintenance occurs."""

        current = (now or datetime.now(UTC)).astimezone(UTC)
        cutoff = current - timedelta(days=self.retention_days)
        async with self._lock:
            events = await self.journal.prune(before=cutoff)
            agent_requests, agent_conversations = await self.agent_store.prune(
                before=cutoff
            )
            agent_memories = await self.memory_store.cleanup(now=current)
            focus_timers = await self.focus_timers.prune_terminal(before=cutoff)
            image_jobs, image_paths = await asyncio.to_thread(
                self.image_store.prune_delivered_terminal,
                before=cutoff,
            )
            image_files = _remove_image_outputs(
                image_paths,
                root=self.image_output_dir,
            )
            moderation_cache, moderation_quota = await self.moderation_store.prune(
                before=cutoff
            )
            speech_orphans, speech_cache = await self.speech.cleanup_orphans(
                before=cutoff,
                max_cache_bytes=min(
                    _SPEECH_CACHE_LIMIT_BYTES,
                    max(0, self.max_data_bytes // 10),
                ),
            )
            local_media = await self.local_media.cleanup_expired(
                before_epoch=int(cutoff.timestamp())
            )

            storage_used = await asyncio.to_thread(
                _directory_size,
                self.data_dir,
            )
            while storage_used > self.max_data_bytes:
                removed_job, pressure_paths = await asyncio.to_thread(
                    self.image_store.prune_oldest_delivered_terminal
                )
                if not removed_job:
                    break
                image_jobs += 1
                image_files += _remove_image_outputs(
                    pressure_paths,
                    root=self.image_output_dir,
                )
                storage_used = await asyncio.to_thread(
                    _directory_size,
                    self.data_dir,
                )

            self._last_report = MaintenanceReport(
                completed_at=current,
                storage_used_bytes=storage_used,
                storage_limit_bytes=self.max_data_bytes,
                over_capacity=storage_used > self.max_data_bytes,
                event_records_removed=events,
                agent_requests_removed=agent_requests,
                agent_conversations_removed=agent_conversations,
                agent_memories_removed=agent_memories,
                focus_timers_removed=focus_timers,
                image_jobs_removed=image_jobs,
                image_files_removed=image_files,
                speech_orphans_removed=speech_orphans,
                speech_cache_files_removed=speech_cache,
                moderation_cache_rows_removed=moderation_cache,
                moderation_quota_rows_removed=moderation_quota,
                local_media_removed=local_media,
            )
            return self._last_report


def _directory_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _remove_image_outputs(paths: tuple[Path, ...], *, root: Path) -> int:
    removed = 0
    for original in paths:
        try:
            path = original.resolve()
        except OSError:
            continue
        if not path.is_relative_to(root):
            continue
        if path.is_file() and not path.is_symlink():
            path.unlink(missing_ok=True)
            removed += 1
        metadata = path.with_suffix(".metadata.json")
        if (
            metadata.is_relative_to(root)
            and metadata.is_file()
            and not metadata.is_symlink()
        ):
            metadata.unlink(missing_ok=True)
            removed += 1
    return removed
