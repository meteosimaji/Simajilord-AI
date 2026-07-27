"""Platform-owned, durable audio queue orchestration."""

from __future__ import annotations

import asyncio
import logging
import random
from collections import deque
from collections.abc import Awaitable, Callable
from time import monotonic, time
from typing import Protocol

from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind, LoopMode, QueueSnapshot

from .audio_state import AudioStateStore, StoredAudioItem, StoredAudioSession

log = logging.getLogger(__name__)

_MAX_STREAM_AGE_SECONDS = 10 * 60
_IMMEDIATE_RETRY_DELAYS = (0.0, 1.0, 3.0)
_MAX_HISTORY_ITEMS = 25
_MAX_IDENTICAL_PENDING_REFERENCES = 2


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


AudioResolver = Callable[[str], Awaitable[AudioItem]]
StateHook = Callable[["AudioSession"], Awaitable[None]]


class AudioSession:
    """A transport-neutral queue with recovery, persistence hooks, and one worker."""

    def __init__(
        self,
        workspace_id: str,
        output: AudioOutput,
        *,
        max_pending_speech: int,
        max_pending_music: int = 100,
        max_pending_music_per_actor: int = 20,
        resolver: AudioResolver | None = None,
        state_hook: StateHook | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.output = output
        self.max_pending_speech = max_pending_speech
        self.max_pending_music = max_pending_music
        self.max_pending_music_per_actor = max_pending_music_per_actor
        self.destination_id: str | None = None
        self.auto_leave = True
        self._resolver = resolver
        self._state_hook = state_hook
        self._music: deque[AudioItem] = deque()
        self._speech: deque[AudioItem] = deque()
        self._history: deque[AudioItem] = deque(maxlen=_MAX_HISTORY_ITEMS)
        self._current: AudioItem | None = None
        self._waiting_actor_ids: set[str] = set()
        self._loop_mode = LoopMode.NONE
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._skip_requested = False
        self._discard_requested = False
        self._suspend_requested = False
        self._restart_requested = False
        self._suspended = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._started_at: float | None = None
        self._paused_at: float | None = None
        self._paused_seconds = 0.0
        self._speed = 1.0
        self._pitch = 1.0
        self._music_volume = 1.0
        self._speech_volume = 1.0

    @property
    def current(self) -> AudioItem | None:
        return self._current

    @property
    def loop_mode(self) -> LoopMode:
        return self._loop_mode

    @property
    def has_music(self) -> bool:
        return bool(
            (self._current and self._current.kind is AudioKind.MUSIC) or self._music
        )

    @property
    def waiting_for_voice(self) -> bool:
        return bool(self._waiting_actor_ids)

    async def connect(self, destination_id: str) -> None:
        await self.output.connect(destination_id)
        self.destination_id = destination_id
        self._waiting_actor_ids.clear()
        self._suspended = False
        self._wake.set()
        self._ensure_worker()
        await self._state_changed()

    async def enqueue(self, item: AudioItem) -> int:
        """Queue an item, prioritizing speech ahead of waiting music."""

        interrupt_music = False
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
                current = self._current
                if (
                    current is not None
                    and current.kind is AudioKind.MUSIC
                    and current.speech_overlay_source is None
                    and not self._restart_requested
                ):
                    current.start_seconds = self._position_seconds()
                    self._restart_requested = True
                    interrupt_music = True
            else:
                try:
                    self._assert_music_queue_capacity(item)
                except UserError:
                    item.cleanup()
                    raise
                self._music.append(item)
                position = len(self._music)
            self._wake.set()
            self._ensure_worker()
        if interrupt_music:
            self.output.stop()
        await self._state_changed()
        return position

    async def wait_for_listener(self, actor_id: str) -> None:
        """Keep playback dormant until one of the requesting actors joins voice."""

        if not actor_id:
            raise ValueError("actor_id must not be empty")
        if self.output.connected:
            return
        self._waiting_actor_ids.add(actor_id)
        self._suspended = True
        await self._state_changed()

    def can_start_for(self, actor_id: str) -> bool:
        """Return whether an actor may activate a queue waiting without a destination."""

        return not self._waiting_actor_ids or actor_id in self._waiting_actor_ids

    def can_control_while_waiting(self, actor_id: str) -> bool:
        return actor_id in self._waiting_actor_ids

    async def set_loop(self, mode: LoopMode) -> None:
        async with self._lock:
            self._loop_mode = mode
        await self._state_changed()

    async def set_auto_leave(self, enabled: bool) -> None:
        self.auto_leave = enabled
        await self._state_changed()

    async def shuffle(self) -> None:
        async with self._lock:
            shuffled = list(self._music)
            random.shuffle(shuffled)
            self._music = deque(shuffled)
        await self._state_changed()

    async def seek(self, position_seconds: float) -> float:
        current = self._current
        if current is None or current.kind is not AudioKind.MUSIC:
            raise UserError("audio.nothing_playing")
        upper = current.duration_seconds if current.duration_seconds > 0 else position_seconds
        bounded = max(0.0, min(position_seconds, upper))
        current.start_seconds = bounded
        self._restart_requested = True
        self.output.stop()
        await self._wait_for_current()
        return bounded

    async def tune(self, speed: float, pitch: float) -> None:
        if not 0.5 <= speed <= 2.0 or not 0.5 <= pitch <= 2.0:
            raise UserError("audio.tune_range_invalid")
        current = self._current
        if current is not None and current.kind is AudioKind.MUSIC:
            current.start_seconds = self._position_seconds()
            self._restart_requested = True
        self._speed = speed
        self._pitch = pitch
        if self._restart_requested:
            self.output.stop()
            await self._wait_for_current()
        await self._state_changed()

    async def set_volume(
        self,
        *,
        music: float | None = None,
        speech: float | None = None,
    ) -> tuple[float, float]:
        """Set durable music/read-aloud gain without coupling to Discord."""

        if music is None and speech is None:
            raise UserError("audio.volume_value_required")
        if music is not None and not 0.0 <= music <= 2.0:
            raise UserError("audio.volume_range_invalid")
        if speech is not None and not 0.0 <= speech <= 2.0:
            raise UserError("audio.volume_range_invalid")
        restart_music = (
            music is not None
            and self._current is not None
            and self._current.kind is AudioKind.MUSIC
        )
        if restart_music and self._current is not None:
            self._current.start_seconds = self._position_seconds()
            self._restart_requested = True
        if music is not None:
            self._music_volume = music
        if speech is not None:
            self._speech_volume = speech
        if restart_music:
            self.output.stop()
            await self._wait_for_current()
        await self._state_changed()
        return self._music_volume, self._speech_volume

    async def remove(self, position: int) -> AudioItem:
        if position < 1:
            raise UserError("audio.queue_position_invalid")
        async with self._lock:
            try:
                item = self._music[position - 1]
            except IndexError as exc:
                raise UserError("audio.queue_position_invalid") from exc
            del self._music[position - 1]
        item.cleanup()
        await self._state_changed()
        return item

    async def move(self, from_position: int, to_position: int) -> AudioItem:
        """Move one pending music item using the one-based positions shown to users."""

        if from_position < 1 or to_position < 1:
            raise UserError("audio.queue_position_invalid")
        async with self._lock:
            if from_position > len(self._music) or to_position > len(self._music):
                raise UserError("audio.queue_position_invalid")
            item = self._music[from_position - 1]
            del self._music[from_position - 1]
            self._music.insert(to_position - 1, item)
        await self._state_changed()
        return item

    async def clear_for_actor(self, actor_id: str) -> tuple[AudioItem, ...]:
        """Remove only pending music requested by one actor."""

        if not actor_id:
            raise ValueError("actor_id must not be empty")
        async with self._lock:
            removed = tuple(
                item for item in self._music if item.requested_by_id == actor_id
            )
            self._music = deque(
                item for item in self._music if item.requested_by_id != actor_id
            )
        for item in removed:
            item.cleanup()
        if removed:
            await self._state_changed()
        return removed

    async def skip(self) -> None:
        if self._current is None:
            raise UserError("audio.nothing_playing")
        self._skip_requested = True
        self.output.stop()

    async def clear(self) -> None:
        async with self._lock:
            self._discard_requested = True
            self._waiting_actor_ids.clear()
            for item in (*self._speech, *self._music):
                item.cleanup()
            self._speech.clear()
            self._music.clear()
            if self._current is not None:
                self.output.stop()
        await self._wait_for_current()
        await self._state_changed()

    def pause(self) -> None:
        if self._current is None or self.output.paused:
            raise UserError("audio.nothing_playing")
        self.output.pause()
        self._paused_at = monotonic()

    def resume(self) -> None:
        if not self.output.paused:
            raise UserError("audio.not_paused")
        if self._paused_at is not None:
            self._paused_seconds += monotonic() - self._paused_at
        self._paused_at = None
        self.output.resume()

    async def disconnect(self) -> None:
        """Explicitly leave and forget the queue."""

        await self.clear()
        await self.output.disconnect()
        self.destination_id = None
        await self._state_changed()

    async def suspend(self) -> None:
        """Leave voice while preserving music for automatic reconnection."""

        self._suspended = True
        self._suspend_requested = True
        if self._current is not None:
            if self._current.kind is AudioKind.MUSIC:
                self._current.start_seconds = self._position_seconds()
            self.output.stop()
            await self._wait_for_current()
        await self.output.disconnect()
        await self._state_changed()

    async def shutdown(self) -> None:
        """Stop the process without deleting the last durable queue snapshot."""

        await self._state_changed()
        self._closed = True
        if self._current is not None:
            self.output.stop()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        for item in (*self._speech, *self._music):
            item.cleanup()
        if self._current is not None:
            self._current.cleanup()
        self._speech.clear()
        self._music.clear()
        self._current = None
        await self.output.disconnect()

    async def close(self) -> None:
        """Close a standalone session and discard its queue."""

        self._closed = True
        await self.clear()
        await self.output.disconnect()
        self._wake.set()
        if self._worker is not None:
            await self._worker

    async def snapshot(self) -> QueueSnapshot:
        async with self._lock:
            return QueueSnapshot(
                current=self._current,
                pending=tuple((*self._speech, *self._music)),
                history=tuple(reversed(self._history)),
                paused=self.output.paused,
                loop=self._loop_mode,
                destination_id=self.destination_id,
                waiting_actor_ids=tuple(sorted(self._waiting_actor_ids)),
                auto_leave=self.auto_leave,
                position_seconds=self._position_seconds(),
                speed=self._speed,
                pitch=self._pitch,
                music_volume=self._music_volume,
                speech_volume=self._speech_volume,
            )

    def restore(self, state: StoredAudioSession) -> None:
        """Load placeholders whose direct URLs will be resolved immediately before play."""

        if state.workspace_id != self.workspace_id:
            raise ValueError("Stored audio workspace does not match the session.")
        self.destination_id = state.destination_id
        self._waiting_actor_ids = set(state.waiting_actor_ids)
        self._loop_mode = state.loop_mode
        self.auto_leave = state.auto_leave
        self._speed = state.speed
        self._pitch = state.pitch
        self._music_volume = state.music_volume
        self._speech_volume = state.speech_volume
        self._music.extend(
            AudioItem(
                source="",
                title=item.title,
                page_url=item.page_url,
                duration_seconds=item.duration_seconds,
                resolver_reference=item.reference,
                failure_count=item.failure_count,
                start_seconds=item.start_seconds,
                requested_by_id=item.requested_by_id,
                requested_by_name=item.requested_by_name,
                played_at_epoch=item.played_at_epoch,
                uploader=item.uploader,
                thumbnail_url=item.thumbnail_url,
            )
            for item in state.items
        )
        self._history.extend(
            AudioItem(
                source="",
                title=item.title,
                page_url=item.page_url,
                duration_seconds=item.duration_seconds,
                resolver_reference=item.reference,
                requested_by_id=item.requested_by_id,
                requested_by_name=item.requested_by_name,
                played_at_epoch=item.played_at_epoch,
                uploader=item.uploader,
                thumbnail_url=item.thumbnail_url,
            )
            for item in state.history[-_MAX_HISTORY_ITEMS:]
        )
        if self._music and not self.output.connected:
            self._suspended = True

    async def persisted_state(self) -> StoredAudioSession | None:
        async with self._lock:
            items = [
                item
                for item in ((self._current,) if self._current else ())
                if item.kind is AudioKind.MUSIC
            ]
            items.extend(self._music)
            destination_id = self.destination_id
            stored_items_list: list[StoredAudioItem] = []
            for item in items:
                reference = item.resolver_reference or item.page_url
                if not reference:
                    continue
                start_seconds = item.start_seconds
                if item is self._current:
                    start_seconds = self._position_seconds()
                stored_items_list.append(
                    StoredAudioItem(
                        reference=reference,
                        title=item.title,
                        page_url=item.page_url,
                        duration_seconds=item.duration_seconds,
                        failure_count=item.failure_count,
                        start_seconds=start_seconds,
                        requested_by_id=item.requested_by_id,
                        requested_by_name=item.requested_by_name,
                        played_at_epoch=item.played_at_epoch,
                        uploader=item.uploader,
                        thumbnail_url=item.thumbnail_url,
                    )
                )
            stored_items = tuple(stored_items_list)
            stored_history = tuple(
                StoredAudioItem(
                    reference=item.resolver_reference or item.page_url,
                    title=item.title,
                    page_url=item.page_url,
                    duration_seconds=item.duration_seconds,
                    requested_by_id=item.requested_by_id,
                    requested_by_name=item.requested_by_name,
                    played_at_epoch=item.played_at_epoch,
                    uploader=item.uploader,
                    thumbnail_url=item.thumbnail_url,
                )
                for item in self._history
                if item.resolver_reference or item.page_url
            )
            if not stored_items and not stored_history:
                return None
            return StoredAudioSession(
                workspace_id=self.workspace_id,
                destination_id=destination_id,
                waiting_actor_ids=tuple(sorted(self._waiting_actor_ids)),
                loop_mode=self._loop_mode,
                auto_leave=self.auto_leave,
                speed=self._speed,
                pitch=self._pitch,
                items=stored_items,
                history=stored_history,
                music_volume=self._music_volume,
                speech_volume=self._speech_volume,
            )

    def _assert_music_queue_capacity(self, item: AudioItem) -> None:
        if len(self._music) >= self.max_pending_music:
            raise UserError("audio.queue_full")
        actor_id = item.requested_by_id
        if actor_id is not None:
            actor_pending = sum(
                queued.requested_by_id == actor_id for queued in self._music
            )
            if actor_pending >= self.max_pending_music_per_actor:
                raise UserError("audio.user_queue_full")
        reference = item.resolver_reference or item.page_url
        if reference:
            identical = sum(
                (queued.resolver_reference or queued.page_url) == reference
                for queued in self._music
            )
            if identical >= _MAX_IDENTICAL_PENDING_REFERENCES:
                raise UserError("audio.duplicate_limit")

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run(),
                name=f"simajilord-audio-{self.workspace_id}",
            )

    async def _next_item(self) -> AudioItem | None:
        async with self._lock:
            if self._speech:
                speech = self._speech.popleft()
                music = self._pop_ready_music()
                if music is not None and speech.duration_seconds > 0:
                    music.speech_overlay_source = speech.source
                    music.speech_overlay_owned_file = speech.owned_file
                    music.speech_overlay_duration_seconds = speech.duration_seconds
                    speech.owned_file = None
                    speech.cleanup()
                    return music
                if music is not None:
                    self._music.appendleft(music)
                return speech
            return self._pop_ready_music()

    def _pop_ready_music(self) -> AudioItem | None:
        """Pop one playable music item while the caller owns ``self._lock``."""

        now = monotonic()
        for _ in range(len(self._music)):
            item = self._music.popleft()
            if item.retry_after <= now:
                return item
            self._music.append(item)
        return None

    async def _next_retry_delay(self) -> float | None:
        async with self._lock:
            if self._speech:
                return 0.0
            if not self._music:
                return None
            return max(0.0, min(item.retry_after for item in self._music) - monotonic())

    async def _run(self) -> None:
        while not self._closed:
            if self._suspended:
                self._wake.clear()
                if self._suspended:
                    await self._wake.wait()
                continue
            item = await self._next_item()
            if item is None:
                self._wake.clear()
                delay = await self._next_retry_delay()
                if delay is None:
                    await self._wake.wait()
                elif delay == 0:
                    self._wake.set()
                else:
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=delay)
                    except TimeoutError:
                        self._wake.set()
                continue

            self._current = item
            self._skip_requested = False
            self._discard_requested = False
            self._suspend_requested = False
            self._restart_requested = False
            item.speed = self._speed
            item.pitch = self._pitch
            item.volume = (
                self._speech_volume
                if item.kind is AudioKind.SPEECH
                else self._music_volume
            )
            item.speech_overlay_volume = self._speech_volume
            self._started_at = monotonic()
            self._paused_at = None
            self._paused_seconds = 0.0
            await self._state_changed()
            completed = False
            playable = item
            try:
                if not self.output.connected:
                    raise UserError("audio.output_disconnected")
                playable = await self._play_with_recovery(item)
                completed = True
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "Audio playback exhausted immediate retries workspace=%s item=%s",
                    self.workspace_id,
                    item.title,
                )
            finally:
                skipped = self._skip_requested
                discarded = self._discard_requested
                suspended = self._suspend_requested
                restarted = self._restart_requested or playable.resume_after_overlay
                self._current = None
                self._skip_requested = False
                self._discard_requested = False
                self._suspend_requested = False
                self._restart_requested = False
                self._started_at = None
                self._paused_at = None
                self._paused_seconds = 0.0

            keep_item = False
            if completed and playable.kind is AudioKind.MUSIC and not (suspended or restarted):
                history_item = playable.unresolved_copy()
                history_item.played_at_epoch = int(time())
                self._history.append(history_item)
            if (suspended or restarted) and playable.kind is AudioKind.MUSIC:
                async with self._lock:
                    self._music.appendleft(playable.unresolved_copy())
                    if suspended:
                        self._wake.clear()
                    else:
                        self._wake.set()
                keep_item = True
            elif (
                completed
                and not skipped
                and not discarded
                and playable.kind is AudioKind.MUSIC
                and self._loop_mode is not LoopMode.NONE
            ):
                looped = playable.clone_for_loop()
                async with self._lock:
                    if self._loop_mode is LoopMode.TRACK:
                        self._music.appendleft(looped)
                    elif self._loop_mode is LoopMode.QUEUE:
                        self._music.append(looped)
                    self._wake.set()
                keep_item = True
            elif (
                not completed
                and not skipped
                and not discarded
                and playable.kind is AudioKind.MUSIC
            ):
                retry = playable.unresolved_copy(
                    failure_count=playable.failure_count + 1
                )
                delay = min(300.0, 15.0 * (2 ** min(retry.failure_count - 1, 4)))
                retry.retry_after = monotonic() + delay
                async with self._lock:
                    self._music.append(retry)
                keep_item = True
                log.warning(
                    "Keeping failed track queued for retry in %.0fs workspace=%s item=%s",
                    delay,
                    self.workspace_id,
                    retry.title,
                )
            if not keep_item:
                playable.cleanup()
            await self._state_changed()

    async def _play_with_recovery(self, item: AudioItem) -> AudioItem:
        playable = item
        last_error: Exception | None = None
        for attempt, delay in enumerate(_IMMEDIATE_RETRY_DELAYS, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                if self._must_resolve(playable, attempt):
                    playable = await self._resolve(playable)
                    self._current = playable
                    await self._state_changed()
                await self.output.play(playable)
                return playable
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Audio attempt %s/%s failed workspace=%s item=%s error=%s",
                    attempt,
                    len(_IMMEDIATE_RETRY_DELAYS),
                    self.workspace_id,
                    item.title,
                    type(exc).__name__,
                )
                unresolved = playable.unresolved_copy(
                    failure_count=playable.failure_count
                )
                _move_speech_overlay(playable, unresolved)
                playable = unresolved
        if last_error is None:
            raise RuntimeError("Audio playback failed without an error.")
        raise last_error

    def _must_resolve(self, item: AudioItem, attempt: int) -> bool:
        return (
            item.kind is AudioKind.MUSIC
            and (
                not item.source
                or attempt > 1
                or monotonic() - item.resolved_at > _MAX_STREAM_AGE_SECONDS
            )
        )

    async def _resolve(self, item: AudioItem) -> AudioItem:
        reference = item.resolver_reference or item.page_url
        if self._resolver is None or not reference:
            raise RuntimeError("No media resolver is available for audio recovery.")
        resolved = await self._resolver(reference)
        resolved.failure_count = item.failure_count
        resolved.start_seconds = item.start_seconds
        resolved.speed = item.speed
        resolved.pitch = item.pitch
        resolved.volume = item.volume
        resolved.speech_overlay_volume = item.speech_overlay_volume
        resolved.requested_by_id = item.requested_by_id
        resolved.requested_by_name = item.requested_by_name
        resolved.uploader = resolved.uploader or item.uploader
        resolved.thumbnail_url = resolved.thumbnail_url or item.thumbnail_url
        _move_speech_overlay(item, resolved)
        return resolved

    def _position_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._paused_at or monotonic()
        current = self._current
        offset = current.start_seconds if current is not None else 0.0
        speed = current.speed if current is not None else self._speed
        return max(
            0.0,
            offset + (end - self._started_at - self._paused_seconds) * speed,
        )

    async def _wait_for_current(self) -> None:
        try:
            async with asyncio.timeout(2.0):
                while self._current is not None:
                    await asyncio.sleep(0)
        except TimeoutError:
            log.warning("Timed out waiting for audio playback callback in %s", self.workspace_id)

    async def _state_changed(self) -> None:
        if self._state_hook is None:
            return
        try:
            await self._state_hook(self)
        except Exception:
            log.exception("Could not persist audio state for %s", self.workspace_id)


def _move_speech_overlay(source: AudioItem, destination: AudioItem) -> None:
    """Transfer ephemeral overlay ownership without persisting or duplicating it."""

    destination.speech_overlay_source = source.speech_overlay_source
    destination.speech_overlay_owned_file = source.speech_overlay_owned_file
    destination.speech_overlay_duration_seconds = source.speech_overlay_duration_seconds
    source.speech_overlay_owned_file = None


class AudioSessionManager:
    """Own one independent, recoverable audio session per workspace."""

    def __init__(
        self,
        *,
        max_active: int,
        max_pending_speech: int,
        max_pending_music: int = 100,
        max_pending_music_per_actor: int = 20,
        resolver: AudioResolver | None = None,
        state_store: AudioStateStore | None = None,
    ) -> None:
        self.max_active = max_active
        self.max_pending_speech = max_pending_speech
        self.max_pending_music = max_pending_music
        self.max_pending_music_per_actor = max_pending_music_per_actor
        self._resolver = resolver
        self._state_store = state_store
        self._sessions: dict[str, AudioSession] = {}
        self._connection_lock = asyncio.Lock()

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
            max_pending_music=self.max_pending_music,
            max_pending_music_per_actor=self.max_pending_music_per_actor,
            resolver=self._resolver,
            state_hook=self._persist,
        )
        self._sessions[workspace_id] = session
        return session

    def restore(
        self,
        output_factory: Callable[[str], AudioOutput],
    ) -> tuple[AudioSession, ...]:
        if self._state_store is None:
            return ()
        restored: list[AudioSession] = []
        for state in self._state_store.all():
            workspace_id = state.workspace_id

            def create_output(workspace_id: str = workspace_id) -> AudioOutput:
                return output_factory(workspace_id)

            session = self.get_or_create(
                workspace_id,
                create_output,
            )
            if not session.has_music:
                session.restore(state)
            restored.append(session)
        return tuple(restored)

    def require(self, workspace_id: str) -> AudioSession:
        try:
            return self._sessions[workspace_id]
        except KeyError as exc:
            raise UserError("audio.session_missing") from exc

    def find(self, workspace_id: str) -> AudioSession | None:
        return self._sessions.get(workspace_id)

    def assert_connection_capacity(self, workspace_id: str) -> None:
        active_other_sessions = sum(
            session.output.connected
            for session_id, session in self._sessions.items()
            if session_id != workspace_id
        )
        if active_other_sessions >= self.max_active:
            raise UserError("audio.capacity_reached")

    async def connect(self, workspace_id: str, destination_id: str) -> None:
        """Atomically enforce the process voice limit and connect one session."""

        async with self._connection_lock:
            session = self.require(workspace_id)
            if not session.output.connected:
                self.assert_connection_capacity(workspace_id)
            await session.connect(destination_id)

    async def close(self) -> None:
        await asyncio.gather(*(session.shutdown() for session in self._sessions.values()))
        self._sessions.clear()

    async def _persist(self, session: AudioSession) -> None:
        if self._state_store is None:
            return
        state = await session.persisted_state()
        if state is None:
            await self._state_store.remove(session.workspace_id)
        else:
            await self._state_store.put(state)
