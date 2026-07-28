"""Durable, transport-neutral music session state."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simajilord.domain.audio import AudioQueueLane, LoopMode


@dataclass(frozen=True, slots=True)
class StoredAudioItem:
    """A media reference that can be resolved again without a signed stream URL."""

    reference: str
    title: str
    page_url: str
    duration_seconds: float
    failure_count: int = 0
    start_seconds: float = 0.0
    requested_by_id: str | None = None
    requested_by_name: str | None = None
    queue_lane: AudioQueueLane = AudioQueueLane.REQUEST
    request_source: str | None = None
    request_id: str | None = None
    requested_at_epoch: int | None = None
    played_at_epoch: int | None = None
    uploader: str | None = None
    thumbnail_url: str | None = None


@dataclass(frozen=True, slots=True)
class StoredAudioSession:
    """The recoverable part of one workspace's audio session."""

    workspace_id: str
    destination_id: str | None
    waiting_actor_ids: tuple[str, ...]
    loop_mode: LoopMode
    auto_leave: bool
    speed: float
    pitch: float
    items: tuple[StoredAudioItem, ...]
    history: tuple[StoredAudioItem, ...]
    music_volume: float = 1.0
    speech_volume: float = 1.0
    autoplay_enabled: bool = False
    mix_seed_references: tuple[str, ...] = ()
    voice_activation_required: bool = False


class AudioStateStore:
    """Atomically persist queues while excluding expiring provider stream URLs."""

    def __init__(self, path: Path, *, debounce_seconds: float = 0.1) -> None:
        if debounce_seconds < 0:
            raise ValueError("Audio state debounce cannot be negative.")
        self.path = path
        self.debounce_seconds = debounce_seconds
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._sessions = self._load()
        self._dirty = False
        self._write_task: asyncio.Task[None] | None = None
        self._flush_event = asyncio.Event()

    def all(self) -> tuple[StoredAudioSession, ...]:
        return tuple(self._sessions.values())

    async def put(self, state: StoredAudioSession) -> None:
        async with self._lock:
            self._sessions[state.workspace_id] = state
            self._dirty = True
            self._ensure_writer_locked()

    async def remove(self, workspace_id: str) -> None:
        async with self._lock:
            if self._sessions.pop(workspace_id, None) is not None:
                self._dirty = True
                self._ensure_writer_locked()

    async def flush(self) -> None:
        """Force pending coalesced changes to durable storage."""

        async with self._lock:
            if self._dirty:
                self._ensure_writer_locked()
            task = self._write_task
            if task is None:
                return
            self._flush_event.set()
        await task

    def _ensure_writer_locked(self) -> None:
        if self._write_task is None or self._write_task.done():
            self._write_task = asyncio.create_task(
                self._debounced_writer(),
                name="simajilord-audio-state-writer",
            )

    async def _debounced_writer(self) -> None:
        while True:
            forced = False
            try:
                await asyncio.wait_for(
                    self._flush_event.wait(),
                    timeout=self.debounce_seconds,
                )
                forced = True
            except TimeoutError:
                pass
            async with self._lock:
                if forced:
                    self._flush_event.clear()
                if not self._dirty:
                    self._write_task = None
                    return
                self._dirty = False
                await asyncio.to_thread(self._write)
                if forced:
                    self._write_task = None
                    return

    def _load(self) -> dict[str, StoredAudioSession]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("version") != 1:
                return {}
            sessions = raw.get("sessions")
            if not isinstance(sessions, list):
                return {}
            restored: dict[str, StoredAudioSession] = {}
            for value in sessions:
                state = _decode_session(value)
                if state is not None:
                    restored[state.workspace_id] = state
            return restored
        except (OSError, ValueError, TypeError):
            return {}

    def _write(self) -> None:
        payload = {
            "version": 1,
            "sessions": [
                {
                    **asdict(state),
                    "loop_mode": state.loop_mode.value,
                    "items": [asdict(item) for item in state.items],
                }
                for state in self._sessions.values()
            ],
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        os.chmod(self.path, 0o600)


def _decode_session(value: object) -> StoredAudioSession | None:
    if not isinstance(value, dict):
        return None
    try:
        workspace_id = _required_text(value, "workspace_id")
        raw_destination_id = value.get("destination_id")
        destination_id = (
            None
            if raw_destination_id is None
            else _required_text(value, "destination_id")
        )
        raw_waiting_actor_ids = value.get("waiting_actor_ids", ())
        if not isinstance(raw_waiting_actor_ids, (list, tuple)):
            return None
        waiting_actor_ids = tuple(
            actor_id
            for actor_id in raw_waiting_actor_ids
            if isinstance(actor_id, str) and actor_id
        )
        loop_mode = LoopMode(_required_text(value, "loop_mode"))
        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            return None
        items = tuple(
            _decode_item(item)
            for item in raw_items
            if isinstance(item, dict)
        )
        raw_history = value.get("history", ())
        if not isinstance(raw_history, (list, tuple)):
            return None
        history = tuple(
            _decode_item(item)
            for item in raw_history
            if isinstance(item, dict)
        )
        raw_mix_seed_references = value.get("mix_seed_references", ())
        if not isinstance(raw_mix_seed_references, (list, tuple)):
            return None
        mix_seed_references = tuple(
            reference
            for reference in raw_mix_seed_references
            if isinstance(reference, str) and reference.startswith("https://")
        )[:8]
        return StoredAudioSession(
            workspace_id=workspace_id,
            destination_id=destination_id,
            waiting_actor_ids=waiting_actor_ids,
            loop_mode=loop_mode,
            auto_leave=bool(value.get("auto_leave", True)),
            speed=_bounded_factor(value.get("speed", 1.0)),
            pitch=_bounded_factor(value.get("pitch", 1.0)),
            items=items,
            history=history,
            music_volume=_bounded_volume(value.get("music_volume", 1.0)),
            speech_volume=_bounded_volume(value.get("speech_volume", 1.0)),
            autoplay_enabled=bool(value.get("autoplay_enabled", False)),
            mix_seed_references=mix_seed_references,
            voice_activation_required=bool(
                value.get(
                    "voice_activation_required",
                    value.get("resume_confirmation_required", False),
                )
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _required_text(value: dict[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _bounded_factor(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 1.0
    factor = float(value)
    return factor if 0.5 <= factor <= 2.0 else 1.0


def _bounded_volume(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 1.0
    volume = float(value)
    return volume if 0.0 <= volume <= 2.0 else 1.0


def _decode_item(value: dict[str, Any]) -> StoredAudioItem:
    requested_by_id = value.get("requested_by_id")
    requested_by_name = value.get("requested_by_name")
    request_source = value.get("request_source")
    request_id = value.get("request_id")
    requested_at_epoch = value.get("requested_at_epoch")
    played_at_epoch = value.get("played_at_epoch")
    uploader = value.get("uploader")
    thumbnail_url = value.get("thumbnail_url")
    return StoredAudioItem(
        reference=_required_text(value, "reference"),
        title=_required_text(value, "title"),
        page_url=_required_text(value, "page_url"),
        duration_seconds=max(0.0, float(value.get("duration_seconds", 0.0))),
        failure_count=max(0, int(value.get("failure_count", 0))),
        start_seconds=max(0.0, float(value.get("start_seconds", 0.0))),
        requested_by_id=(
            requested_by_id
            if isinstance(requested_by_id, str) and requested_by_id
            else None
        ),
        requested_by_name=(
            requested_by_name
            if isinstance(requested_by_name, str) and requested_by_name
            else None
        ),
        queue_lane=AudioQueueLane(value.get("queue_lane", AudioQueueLane.REQUEST)),
        request_source=(
            request_source
            if isinstance(request_source, str) and request_source
            else None
        ),
        request_id=request_id if isinstance(request_id, str) and request_id else None,
        requested_at_epoch=(
            max(0, int(requested_at_epoch))
            if isinstance(requested_at_epoch, (int, float, str))
            else None
        ),
        played_at_epoch=(
            max(0, int(played_at_epoch))
            if isinstance(played_at_epoch, (int, float, str))
            else None
        ),
        uploader=uploader if isinstance(uploader, str) and uploader else None,
        thumbnail_url=(
            thumbnail_url
            if isinstance(thumbnail_url, str) and thumbnail_url.startswith("https://")
            else None
        ),
    )
