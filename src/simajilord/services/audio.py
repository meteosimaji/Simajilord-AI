"""Platform-owned audio queue orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from typing import Protocol

from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind, LoopMode, QueueSnapshot

log = logging.getLogger(__name__)


class AudioOutput(Protocol):
    """The small surface required from Discord or another audio transport."""

    @property
    def connected(self) -> bool: ...

    @property
    def paused(self) -> bool: ...

    async def connect(self, destination_id: str) -> None: ...

    async def play(self, item: AudioItem) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    async def disconnect(self) -> None: ...


class AudioSession:
    """A transport-neutral, serialized music and speech queue."""

    def __init__(self, workspace_id: str, output: AudioOutput, *, max_pending_speech: int) -> None:
        self.workspace_id = workspace_id
        self.output = output
        self.max_pending_speech = max_pending_speech
        self._music: deque[AudioItem] = deque()
        self._speech: deque[AudioItem] = deque()
        self._current: AudioItem | None = None
        self._loop_mode = LoopMode.NONE
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._skip_requested = False
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def current(self) -> AudioItem | None:
        return self._current

    async def connect(self, destination_id: str) -> None:
        await self.output.connect(destination_id)
        self._ensure_worker()

    async def enqueue(self, item: AudioItem) -> int:
        """Queue an item, prioritizing speech ahead of waiting music."""

        async with self._lock:
            if self._closed:
                item.cleanup()
                raise UserError("audio.session_closed")
            if item.kind is AudioKind.SPEECH:
                pending_speech = len(self._speech)
                if self._current and self._current.kind is AudioKind.SPEECH:
                    pending_speech += 1
                if pending_speech >= self.max_pending_speech:
                    item.cleanup()
                    raise UserError("speech.queue_full")
                self._speech.append(item)
                position = pending_speech + 1
            else:
                self._music.append(item)
                position = len(self._music)
            self._wake.set()
            self._ensure_worker()
            return position

    async def set_loop(self, mode: LoopMode) -> None:
        async with self._lock:
            self._loop_mode = mode

    async def skip(self) -> None:
        if self._current is None:
            raise UserError("audio.nothing_playing")
        self._skip_requested = True
        self.output.stop()

    async def clear(self) -> None:
        async with self._lock:
            self._skip_requested = True
            for item in (*self._speech, *self._music):
                item.cleanup()
            self._speech.clear()
            self._music.clear()
            if self._current is not None:
                self.output.stop()

    def pause(self) -> None:
        if self._current is None or self.output.paused:
            raise UserError("audio.nothing_playing")
        self.output.pause()

    def resume(self) -> None:
        if not self.output.paused:
            raise UserError("audio.not_paused")
        self.output.resume()

    async def disconnect(self) -> None:
        await self.clear()
        await self.output.disconnect()

    async def close(self) -> None:
        self._closed = True
        await self.disconnect()
        self._wake.set()
        if self._worker is not None:
            await self._worker

    async def snapshot(self) -> QueueSnapshot:
        async with self._lock:
            return QueueSnapshot(
                current=self._current,
                pending=tuple((*self._speech, *self._music)),
                paused=self.output.paused,
                loop=self._loop_mode,
            )

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run(),
                name=f"simajilord-audio-{self.workspace_id}",
            )

    async def _next_item(self) -> AudioItem | None:
        async with self._lock:
            if self._speech:
                return self._speech.popleft()
            if self._music:
                return self._music.popleft()
            return None

    async def _has_pending(self) -> bool:
        async with self._lock:
            return bool(self._speech or self._music)

    async def _run(self) -> None:
        while not self._closed:
            item = await self._next_item()
            if item is None:
                self._wake.clear()
                if await self._has_pending():
                    self._wake.set()
                    continue
                await self._wake.wait()
                continue

            self._current = item
            self._skip_requested = False
            completed = False
            try:
                if not self.output.connected:
                    raise UserError("audio.output_disconnected")
                await self.output.play(item)
                completed = True
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "Audio playback failed for workspace=%s item=%s",
                    self.workspace_id,
                    item.title,
                )
            finally:
                skipped = self._skip_requested
                self._current = None
                self._skip_requested = False

            if (
                completed
                and not skipped
                and item.kind is AudioKind.MUSIC
                and self._loop_mode is not LoopMode.NONE
            ):
                looped = item.clone_for_loop()
                async with self._lock:
                    if self._loop_mode is LoopMode.TRACK:
                        self._music.appendleft(looped)
                    elif self._loop_mode is LoopMode.QUEUE:
                        self._music.append(looped)
                    self._wake.set()
            item.cleanup()


class AudioSessionManager:
    """Own one independent audio session per workspace."""

    def __init__(self, *, max_active: int, max_pending_speech: int) -> None:
        self.max_active = max_active
        self.max_pending_speech = max_pending_speech
        self._sessions: dict[str, AudioSession] = {}

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def active_session_count(self) -> int:
        return sum(session.output.connected for session in self._sessions.values())

    def get_or_create(
        self,
        workspace_id: str,
        output_factory: Callable[[], AudioOutput],
    ) -> AudioSession:
        existing = self._sessions.get(workspace_id)
        if existing is not None:
            return existing
        session = AudioSession(
            workspace_id,
            output_factory(),
            max_pending_speech=self.max_pending_speech,
        )
        self._sessions[workspace_id] = session
        return session

    def require(self, workspace_id: str) -> AudioSession:
        try:
            return self._sessions[workspace_id]
        except KeyError as exc:
            raise UserError("audio.session_missing") from exc

    def assert_connection_capacity(self, workspace_id: str) -> None:
        active_other_sessions = sum(
            session.output.connected
            for session_id, session in self._sessions.items()
            if session_id != workspace_id
        )
        if active_other_sessions >= self.max_active:
            raise UserError("audio.capacity_reached")

    async def close(self) -> None:
        await asyncio.gather(*(session.close() for session in self._sessions.values()))
        self._sessions.clear()
