"""Platform-owned, durable audio queue orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import re
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import monotonic, time
from typing import Protocol

from simajilord.core.errors import EarlyPlaybackEnd, MediaError, UserError
from simajilord.domain.audio import (
    AudioItem,
    AudioKind,
    AudioQueueLane,
    LoopMode,
    QueueSnapshot,
)

from .audio_state import AudioStateStore, StoredAudioItem, StoredAudioSession
from .metrics import ServiceMetricHook, ServiceOperationMetric

log = logging.getLogger(__name__)

_MAX_STREAM_AGE_SECONDS = 10 * 60
_IMMEDIATE_RETRY_DELAYS = (0.0, 1.0, 3.0)
_MAX_HISTORY_ITEMS = 25
_MAX_IDENTICAL_PENDING_REFERENCES = 2
_MAX_MIX_SEEDS = 8
_AUTOPLAY_REFILL_ITEMS = 30
_AUTOPLAY_QUEUE_TARGET = 3
_AUTOPLAY_RETRY_SECONDS = 60.0
MAX_MANUAL_PLAYBACK_FAILURES = 3
MAX_RADIO_PLAYBACK_FAILURES = 2
_PERMANENT_MEDIA_FAILURES = frozenset(
    {
        "geo_restricted",
        "too_large",
        "unavailable",
        "unsupported",
        "unsafe_path",
    }
)
_OVERLAY_ATTEMPTS = 3
_RADIO_VARIANT_PATTERN = re.compile(
    r"(?i)(?:\bcover\b|\blive\b|\bremix\b|\bnightcore\b|"
    r"\bslowed\b|\bsped[ -]?up\b|\bkaraoke\b|\binstrumental\b|"
    r"\bofficial\b|\bvideo\b|\blyrics?\b|\bremaster(?:ed)?\b)"
)


class AudioOutput(Protocol):
    """The small surface required from Discord or another audio transport."""

    @property
    def connected(self) -> bool: ...

    @property
    def paused(self) -> bool: ...

    async def connect(self, destination_id: str) -> None: ...

    async def play(self, item: AudioItem) -> None: ...

    async def overlay_speech(
        self,
        music: AudioItem,
        speech: AudioItem,
        *,
        position_seconds: float,
    ) -> None: ...

    async def update_music(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
    ) -> None: ...

    async def fade_out(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
        duration_seconds: float,
    ) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    async def disconnect(self) -> None: ...


AudioResolver = Callable[[str], Awaitable[AudioItem]]
AutoplaySupplier = Callable[[tuple[str, ...], int], Awaitable[tuple[AudioItem, ...]]]
WorkspaceAudioResolver = Callable[[str, str], Awaitable[AudioItem]]
WorkspaceAutoplaySupplier = Callable[
    [str, tuple[str, ...], int],
    Awaitable[tuple[AudioItem, ...]],
]
StateHook = Callable[["AudioSession"], Awaitable[None]]
StateListener = Callable[["AudioSession"], Awaitable[None]]


def _append_mix_seed_reference(
    references: deque[str],
    reference: str,
) -> None:
    normalized = reference.strip()
    if not normalized.startswith("https://"):
        return
    with suppress(ValueError):
        references.remove(normalized)
    references.append(normalized)


class SpeechQueueReservation:
    """One pre-synthesis slot in a workspace speech queue."""

    def __init__(self, session: AudioSession, token: str) -> None:
        self._session = session
        self._token = token
        self._active = True

    async def commit(self, item: AudioItem) -> int:
        """Transfer a synthesized speech item into the reserved queue slot."""

        if not self._active:
            item.cleanup()
            raise UserError("speech.reservation_inactive")
        try:
            position = await self._session._commit_speech_reservation(
                self._token,
                item,
            )
        except Exception:
            item.cleanup()
            raise
        self._active = False
        return position

    async def release(self) -> None:
        """Release an unused slot; safe to call after commit or cancellation."""

        if not self._active:
            return
        self._active = False
        await self._session._release_speech_reservation(self._token)


class ManualMusicStartReservation:
    """Temporarily keep Radio idle while an explicit request is being resolved."""

    def __init__(self, session: AudioSession, token: str | None) -> None:
        self._session = session
        self._token = token
        self._active = token is not None

    async def release(self) -> None:
        """Release the hold; safe after enqueue, failure, or cancellation."""

        if not self._active or self._token is None:
            return
        self._active = False
        await self._session._release_manual_music_start(self._token)


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
        autoplay_supplier: AutoplaySupplier | None = None,
        state_hook: StateHook | None = None,
        metric_hook: ServiceMetricHook | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.output = output
        self.max_pending_speech = max_pending_speech
        self.max_pending_music = max_pending_music
        self.max_pending_music_per_actor = max_pending_music_per_actor
        self.destination_id: str | None = None
        self.auto_leave = True
        self._resolver = resolver
        self._autoplay_supplier = autoplay_supplier
        self._state_hook = state_hook
        self._metric_hook = metric_hook
        self._music: deque[AudioItem] = deque()
        self._autoplay: deque[AudioItem] = deque()
        self._speech: deque[AudioItem] = deque()
        self._speech_reservations: set[str] = set()
        self._manual_music_start_reservations: set[str] = set()
        self._history: deque[AudioItem] = deque(maxlen=_MAX_HISTORY_ITEMS)
        self._current: AudioItem | None = None
        self._waiting_actor_ids: set[str] = set()
        self._loop_mode = LoopMode.NONE
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._overlay_task: asyncio.Task[None] | None = None
        self._autoplay_refill_task: asyncio.Task[None] | None = None
        self._autoplay_enabled = False
        self._autoplay_generation = 0
        self._autoplay_retry_at = 0.0
        self._mix_seed_references: deque[str] = deque(maxlen=_MAX_MIX_SEEDS)
        self._speech_active = False
        self._skip_requested = False
        self._discard_requested = False
        self._suspend_requested = False
        self._restart_requested = False
        self._suspended = False
        self._voice_activation_required = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._transport_lock = asyncio.Lock()
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
            (self._current and self._current.kind is AudioKind.MUSIC)
            or self._music
            or self._autoplay
        )

    @property
    def waiting_for_voice(self) -> bool:
        return bool(self._waiting_actor_ids)

    @property
    def voice_activation_required(self) -> bool:
        """Whether the voice route requires an explicit activation action."""

        return self._voice_activation_required

    async def connect(self, destination_id: str) -> None:
        await self.output.connect(destination_id)
        self.destination_id = destination_id
        self._waiting_actor_ids.clear()
        self._suspended = False
        self._voice_activation_required = False
        self._wake.set()
        self._ensure_worker()
        await self._state_changed()

    async def enqueue(
        self,
        item: AudioItem,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> int:
        """Queue an item, prioritizing speech ahead of waiting music."""

        async with self._lock:
            if self._closed:
                item.cleanup()
                raise UserError("audio.session_closed")
            if item.kind is AudioKind.SPEECH:
                if self._speech_load_locked() >= self.max_pending_speech:
                    item.cleanup()
                    raise UserError("speech.queue_full")
            elif item.queue_lane is not AudioQueueLane.AUTOPLAY:
                try:
                    self._assert_music_queue_capacity(item)
                except UserError:
                    item.cleanup()
                    raise
            if before_mutation is not None:
                try:
                    await before_mutation()
                except BaseException:
                    item.cleanup()
                    raise
            if item.kind is AudioKind.SPEECH:
                position = self._enqueue_speech_locked(item)
            elif item.queue_lane is AudioQueueLane.AUTOPLAY:
                self._autoplay.append(item)
                position = len(self._autoplay)
            else:
                self._remember_mix_seed(item)
                self._invalidate_autoplay_locked()
                self._music.append(item)
                position = len(self._music)
            self._wake.set()
            self._ensure_worker()
        await self._state_changed()
        return position

    async def reserve_speech(self) -> SpeechQueueReservation:
        """Reserve queue capacity before an expensive speech synthesis starts."""

        async with self._lock:
            if self._closed:
                raise UserError("audio.session_closed")
            if self._speech_load_locked() >= self.max_pending_speech:
                raise UserError("speech.queue_full")
            token = uuid.uuid4().hex
            self._speech_reservations.add(token)
        return SpeechQueueReservation(self, token)

    async def reserve_manual_music_start(self) -> ManualMusicStartReservation:
        """Hold an idle session so a slow explicit lookup starts before Radio.

        Radio is not interrupted when a track is already playing. The hold only
        closes the race between an idle session's background refill and a
        user's interactive search/resolve operation.
        """

        async with self._lock:
            if self._closed:
                raise UserError("audio.session_closed")
            if self._current is not None or self._music:
                return ManualMusicStartReservation(self, None)
            token = uuid.uuid4().hex
            self._manual_music_start_reservations.add(token)
            self._invalidate_autoplay_locked()
            self._wake.set()
        await self._state_changed()
        return ManualMusicStartReservation(self, token)

    async def _release_manual_music_start(self, token: str) -> None:
        async with self._lock:
            self._manual_music_start_reservations.discard(token)
            self._wake.set()
        self._ensure_autoplay_refill()
        await self._state_changed()

    async def _commit_speech_reservation(
        self,
        token: str,
        item: AudioItem,
    ) -> int:
        if item.kind is not AudioKind.SPEECH:
            raise ValueError("A speech reservation only accepts speech audio.")
        async with self._lock:
            if token not in self._speech_reservations:
                raise UserError("speech.reservation_cancelled")
            self._speech_reservations.remove(token)
            if self._closed:
                raise UserError("audio.session_closed")
            position = self._enqueue_speech_locked(item)
            self._wake.set()
            self._ensure_worker()
        await self._state_changed()
        return position

    async def _release_speech_reservation(self, token: str) -> None:
        async with self._lock:
            self._speech_reservations.discard(token)

    def _speech_load_locked(self) -> int:
        pending = len(self._speech) + len(self._speech_reservations)
        if self._speech_active or (
            self._current is not None and self._current.kind is AudioKind.SPEECH
        ):
            pending += 1
        return pending

    def _enqueue_speech_locked(self, item: AudioItem) -> int:
        item.volume = self._speech_volume
        pending = len(self._speech)
        if self._speech_active or (
            self._current is not None and self._current.kind is AudioKind.SPEECH
        ):
            pending += 1
        position = pending + 1
        current = self._current
        if (
            current is not None
            and current.kind is AudioKind.MUSIC
            and self._overlay_task is None
        ):
            self._speech_active = True
            self._overlay_task = asyncio.create_task(
                self._run_speech_overlays(current, item),
                name=f"simajilord-speech-overlay-{self.workspace_id}",
            )
        else:
            self._speech.append(item)
        return position

    async def enqueue_many(
        self,
        items: tuple[AudioItem, ...],
        *,
        wait_for_actor_id: str | None = None,
    ) -> tuple[int, int]:
        """Atomically validate and append one human-request queue batch.

        ``wait_for_actor_id`` is committed with the batch only when the output is
        disconnected. A rejected batch therefore cannot leave a phantom voice
        waiter behind.
        """

        if not items:
            raise ValueError("audio.batch_empty")
        if any(item.kind is not AudioKind.MUSIC for item in items):
            raise ValueError("audio.batch_music_only")
        if wait_for_actor_id is not None and not wait_for_actor_id:
            raise ValueError("wait_for_actor_id must not be empty")
        async with self._lock:
            if self._closed:
                for item in items:
                    item.cleanup()
                raise UserError("audio.session_closed")
            if len(self._music) + len(items) > self.max_pending_music:
                for item in items:
                    item.cleanup()
                raise UserError("audio.queue_full")

            actor_counts: dict[str, int] = {}
            reference_counts: dict[str, int] = {}
            for queued in self._music:
                if queued.requested_by_id is not None:
                    actor_counts[queued.requested_by_id] = (
                        actor_counts.get(queued.requested_by_id, 0) + 1
                    )
                reference = queued.resolver_reference or queued.page_url
                if reference:
                    reference_counts[reference] = reference_counts.get(reference, 0) + 1
            for item in items:
                actor_id = item.requested_by_id
                if actor_id is not None:
                    actor_counts[actor_id] = actor_counts.get(actor_id, 0) + 1
                    if actor_counts[actor_id] > self.max_pending_music_per_actor:
                        for pending in items:
                            pending.cleanup()
                        raise UserError("audio.user_queue_full")
                reference = item.resolver_reference or item.page_url
                if reference:
                    reference_counts[reference] = reference_counts.get(reference, 0) + 1
                    if reference_counts[reference] > _MAX_IDENTICAL_PENDING_REFERENCES:
                        for pending in items:
                            pending.cleanup()
                        raise UserError("audio.duplicate_limit")

            first_position = len(self._music) + 1
            for item in items:
                item.queue_lane = AudioQueueLane.REQUEST
                self._remember_mix_seed(item)
                self._music.append(item)
            if wait_for_actor_id is not None and not self.output.connected:
                self._waiting_actor_ids.add(wait_for_actor_id)
                self._suspended = True
            self._invalidate_autoplay_locked()
            last_position = len(self._music)
            self._wake.set()
            self._ensure_worker()
        await self._state_changed()
        return first_position, last_position

    async def wait_for_listener(
        self,
        actor_id: str,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Keep playback dormant until one of the requesting actors joins voice."""

        if not actor_id:
            raise ValueError("actor_id must not be empty")
        if self.output.connected or actor_id in self._waiting_actor_ids:
            return
        if before_mutation is not None:
            await before_mutation()
        self._waiting_actor_ids.add(actor_id)
        self._suspended = True
        await self._state_changed()

    def can_start_for(self, actor_id: str) -> bool:
        """Return whether an actor may activate a queue waiting without a destination."""

        return not self._waiting_actor_ids or actor_id in self._waiting_actor_ids

    def can_control_while_waiting(self, actor_id: str) -> bool:
        return actor_id in self._waiting_actor_ids

    async def set_loop(
        self,
        mode: LoopMode,
        *,
        replace_autoplay: bool = False,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        autoplay_task: asyncio.Task[None] | None = None
        async with self._lock:
            if (
                mode is not LoopMode.NONE
                and self._autoplay_enabled
                and not replace_autoplay
            ):
                raise UserError("audio.loop_mix_conflict")
            disables_autoplay = (
                mode is not LoopMode.NONE and self._autoplay_enabled
            )
            if self._loop_mode is mode and not disables_autoplay:
                if on_noop is not None:
                    await on_noop()
                return
            if before_mutation is not None:
                await before_mutation()
            if mode is not LoopMode.NONE and self._autoplay_enabled:
                self._autoplay_enabled = False
                self._autoplay_generation += 1
                autoplay_task = self._autoplay_refill_task
                self._autoplay_refill_task = None
                for item in self._autoplay:
                    item.cleanup()
                self._autoplay.clear()
            self._loop_mode = mode
            self._wake.set()
        if autoplay_task is not None:
            autoplay_task.cancel()
            await asyncio.gather(autoplay_task, return_exceptions=True)
        await self._state_changed()

    async def set_auto_leave(
        self,
        enabled: bool,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if self.auto_leave == enabled:
            if on_noop is not None:
                await on_noop()
            return
        if before_mutation is not None:
            await before_mutation()
        self.auto_leave = enabled
        await self._state_changed()

    async def enable_autoplay(
        self,
        seed_references: tuple[str, ...] = (),
        *,
        replace_loop: bool = False,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[str, ...]:
        """Enable a request-priority station seeded by explicit music references."""

        async with self._lock:
            if self._loop_mode is not LoopMode.NONE and not replace_loop:
                raise UserError(
                    "audio.mix_loop_conflict",
                    loop_mode=self._loop_mode.value,
                )
            prospective_seeds = deque(
                self._mix_seed_references,
                maxlen=_MAX_MIX_SEEDS,
            )
            if seed_references:
                # Explicit seeds describe the listener's current intent. Do not
                # blend them with unrelated tracks left by an older station.
                prospective_seeds.clear()
                for reference in seed_references:
                    _append_mix_seed_reference(prospective_seeds, reference)
            if not prospective_seeds:
                candidates = (
                    *((self._current,) if self._current is not None else ()),
                    *self._music,
                    *reversed(self._history),
                )
                for item in candidates:
                    if item.kind is AudioKind.MUSIC:
                        _append_mix_seed_reference(
                            prospective_seeds,
                            item.resolver_reference or item.page_url,
                        )
                    if len(prospective_seeds) >= _MAX_MIX_SEEDS:
                        break
            if not prospective_seeds:
                raise UserError("audio.mix_seed_required")
            if self._autoplay_supplier is None:
                raise UserError("audio.mix_unavailable")
            if before_mutation is not None:
                await before_mutation()
            self._mix_seed_references.clear()
            self._mix_seed_references.extend(prospective_seeds)
            if replace_loop:
                self._loop_mode = LoopMode.NONE
            self._autoplay_enabled = True
            self._invalidate_autoplay_locked()
            self._autoplay_retry_at = 0.0
            seeds = tuple(self._mix_seed_references)
            self._wake.set()
        self._ensure_autoplay_refill()
        await self._state_changed()
        return seeds

    async def disable_autoplay(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Stop automatic supply without changing explicit user requests."""

        task: asyncio.Task[None] | None
        async with self._lock:
            if (
                not self._autoplay_enabled
                and self._autoplay_refill_task is None
                and not self._autoplay
            ):
                if on_noop is not None:
                    await on_noop()
                return
            if before_mutation is not None:
                await before_mutation()
            self._autoplay_enabled = False
            self._autoplay_generation += 1
            task = self._autoplay_refill_task
            self._autoplay_refill_task = None
            for item in self._autoplay:
                item.cleanup()
            self._autoplay.clear()
            self._wake.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._state_changed()

    async def shuffle(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        async with self._lock:
            if len(self._music) < 2:
                if on_noop is not None:
                    await on_noop()
                return
            if before_mutation is not None:
                await before_mutation()
            shuffled = list(self._music)
            random.shuffle(shuffled)
            self._music = deque(shuffled)
        await self._state_changed()

    async def seek(
        self,
        position_seconds: float,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> float:
        async with self._transport_lock:
            current = self._current
            if current is None or current.kind is not AudioKind.MUSIC:
                raise UserError("audio.nothing_playing")
            upper = current.duration_seconds if current.duration_seconds > 0 else position_seconds
            bounded = max(0.0, min(position_seconds, upper))
            if before_mutation is not None:
                await before_mutation()
            current.start_seconds = bounded
            self._restart_requested = True
            self.output.stop()
            await self._wait_for_current()
        return bounded

    async def tune(
        self,
        speed: float,
        pitch: float,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if not 0.5 <= speed <= 2.0 or not 0.5 <= pitch <= 2.0:
            raise UserError("audio.tune_range_invalid")
        async with self._transport_lock:
            current = self._current
            if (
                current is None
                and math.isclose(self._speed, speed)
                and math.isclose(self._pitch, pitch)
            ):
                if on_noop is not None:
                    await on_noop()
                return
            if before_mutation is not None:
                await before_mutation()
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

        current_music, current_speech, _, _ = await self.set_volume_with_previous(
            music=music,
            speech=speech,
        )
        return current_music, current_speech

    async def set_volume_with_previous(
        self,
        *,
        music: float | None = None,
        speech: float | None = None,
        expected_music: float | None = None,
        expected_speech: float | None = None,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[float, float, float, float]:
        """Set durable gains and return current then previous music/speech values."""

        if music is None and speech is None:
            raise UserError("audio.volume_value_required")
        if music is not None and not 0.0 <= music <= 2.0:
            raise UserError("audio.volume_range_invalid")
        if speech is not None and not 0.0 <= speech <= 2.0:
            raise UserError("audio.volume_range_invalid")
        async with self._transport_lock:
            previous_music = self._music_volume
            previous_speech = self._speech_volume
            target_matches = (
                music is None
                or math.isclose(
                    previous_music,
                    music,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ) and (
                speech is None
                or math.isclose(
                    previous_speech,
                    speech,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            if target_matches:
                if on_noop is not None:
                    await on_noop()
                return (
                    previous_music,
                    previous_speech,
                    previous_music,
                    previous_speech,
                )
            if (
                expected_music is not None
                and not math.isclose(
                    previous_music,
                    expected_music,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ) or (
                expected_speech is not None
                and not math.isclose(
                    previous_speech,
                    expected_speech,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise UserError("action.undo_conflict")
            if before_mutation is not None:
                await before_mutation()
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
                if restart_music and self._current is not None:
                    self._current.volume = music
            if speech is not None:
                self._speech_volume = speech
            if restart_music:
                self.output.stop()
                await self._wait_for_current()
        await self._state_changed()
        return (
            self._music_volume,
            self._speech_volume,
            previous_music,
            previous_speech,
        )

    async def remove(
        self,
        position: int,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> AudioItem:
        if position < 1:
            raise UserError("audio.queue_position_invalid")
        async with self._lock:
            try:
                item = self._music[position - 1]
            except IndexError as exc:
                raise UserError("audio.queue_position_invalid") from exc
            if before_mutation is not None:
                await before_mutation()
            del self._music[position - 1]
        item.cleanup()
        await self._state_changed()
        return item

    async def move(
        self,
        from_position: int,
        to_position: int,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> AudioItem:
        """Move one pending music item using the one-based positions shown to users."""

        if from_position < 1 or to_position < 1:
            raise UserError("audio.queue_position_invalid")
        async with self._lock:
            if from_position > len(self._music) or to_position > len(self._music):
                raise UserError("audio.queue_position_invalid")
            item = self._music[from_position - 1]
            if from_position == to_position:
                if on_noop is not None:
                    await on_noop()
                return item
            if before_mutation is not None:
                await before_mutation()
            del self._music[from_position - 1]
            self._music.insert(to_position - 1, item)
        await self._state_changed()
        return item

    async def clear_for_actor(
        self,
        actor_id: str,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[AudioItem, ...]:
        """Remove only pending music requested by one actor."""

        if not actor_id:
            raise ValueError("actor_id must not be empty")
        async with self._lock:
            removed = tuple(item for item in self._music if item.requested_by_id == actor_id)
            if not removed:
                if on_noop is not None:
                    await on_noop()
                return ()
            if before_mutation is not None:
                await before_mutation()
            self._music = deque(item for item in self._music if item.requested_by_id != actor_id)
        for item in removed:
            item.cleanup()
        await self._state_changed()
        return removed

    async def skip(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        async with self._transport_lock:
            if self._current is None:
                raise UserError("audio.nothing_playing")
            if before_mutation is not None:
                await before_mutation()
            self._skip_requested = True
            self.output.stop()

    async def clear(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        autoplay_task: asyncio.Task[None] | None
        async with self._transport_lock, self._lock:
            changed = bool(
                self._current is not None
                or self._speech
                or self._speech_reservations
                or self._music
                or self._autoplay
                or self._waiting_actor_ids
                or self._voice_activation_required
                or self._autoplay_enabled
                or self._autoplay_refill_task is not None
            )
            if not changed:
                if on_noop is not None:
                    await on_noop()
                return False
            if before_mutation is not None:
                await before_mutation()
            self._discard_requested = True
            self._waiting_actor_ids.clear()
            self._voice_activation_required = False
            self._autoplay_enabled = False
            self._autoplay_generation += 1
            autoplay_task = self._autoplay_refill_task
            self._autoplay_refill_task = None
            for item in (*self._speech, *self._music, *self._autoplay):
                item.cleanup()
            self._speech.clear()
            self._speech_reservations.clear()
            self._music.clear()
            self._autoplay.clear()
            if self._current is not None:
                self.output.stop()
        if autoplay_task is not None:
            autoplay_task.cancel()
            await asyncio.gather(autoplay_task, return_exceptions=True)
        await self._wait_for_current()
        await self._state_changed()
        return True

    async def pause(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        async with self._transport_lock:
            if self._current is None or self.output.paused:
                raise UserError("audio.nothing_playing")
            if before_mutation is not None:
                await before_mutation()
            self.output.pause()
            self._paused_at = monotonic()
        await self._state_changed()

    async def resume(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        async with self._transport_lock:
            if not self.output.paused:
                raise UserError("audio.not_paused")
            if before_mutation is not None:
                await before_mutation()
            if self._paused_at is not None:
                self._paused_seconds += monotonic() - self._paused_at
            self._paused_at = None
            self.output.resume()
        await self._state_changed()

    async def disconnect(
        self,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Explicitly leave and forget the queue."""

        cleared = await self.clear(before_mutation=before_mutation)
        if self.output.connected or self.destination_id is not None:
            if before_mutation is not None:
                await before_mutation()
            if self.output.connected:
                await self.output.disconnect()
            self.destination_id = None
            await self._state_changed()
        elif not cleared and on_noop is not None:
            await on_noop()

    async def suspend(self) -> None:
        """Leave voice and hold the audio route until a listener explicitly resumes it."""

        async with self._transport_lock:
            self._suspended = True
            self._voice_activation_required = True
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
        if self._autoplay_refill_task is not None:
            self._autoplay_refill_task.cancel()
            await asyncio.gather(self._autoplay_refill_task, return_exceptions=True)
            self._autoplay_refill_task = None
        for item in (*self._speech, *self._music, *self._autoplay):
            item.cleanup()
        if self._current is not None:
            self._current.cleanup()
        self._speech.clear()
        self._speech_reservations.clear()
        self._music.clear()
        self._autoplay.clear()
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
                speech_active=self._speech_active,
                loop=self._loop_mode,
                destination_id=self.destination_id,
                waiting_actor_ids=tuple(sorted(self._waiting_actor_ids)),
                auto_leave=self.auto_leave,
                position_seconds=self._position_seconds(),
                speed=self._speed,
                pitch=self._pitch,
                music_volume=self._music_volume,
                speech_volume=self._speech_volume,
                autoplay_enabled=self._autoplay_enabled,
                autoplay_next=self._autoplay[0] if self._autoplay else None,
                mix_seed_references=tuple(self._mix_seed_references),
                voice_activation_required=self._voice_activation_required,
                connected=self.output.connected,
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
        # Older state may contain the invalid combination that used to let a
        # loop starve Mix forever. Preserve the explicit loop and require the
        # user to confirm a switch to Mix after restart.
        self._autoplay_enabled = (
            state.autoplay_enabled and state.loop_mode is LoopMode.NONE
        )
        self._mix_seed_references.extend(state.mix_seed_references[-_MAX_MIX_SEEDS:])
        restored_items = tuple(
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
                queue_lane=item.queue_lane,
                request_source=item.request_source,
                request_id=item.request_id,
                requested_at_epoch=item.requested_at_epoch,
                played_at_epoch=item.played_at_epoch,
                uploader=item.uploader,
                thumbnail_url=item.thumbnail_url,
            )
            for item in state.items
        )
        self._music.extend(
            item for item in restored_items if item.queue_lane is AudioQueueLane.REQUEST
        )
        self._autoplay.extend(
            item for item in restored_items if item.queue_lane is AudioQueueLane.AUTOPLAY
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
                queue_lane=item.queue_lane,
                request_source=item.request_source,
                request_id=item.request_id,
                requested_at_epoch=item.requested_at_epoch,
                played_at_epoch=item.played_at_epoch,
                uploader=item.uploader,
                thumbnail_url=item.thumbnail_url,
            )
            for item in state.history[-_MAX_HISTORY_ITEMS:]
        )
        self._voice_activation_required = state.voice_activation_required
        if (self._music or self._autoplay) and not self.output.connected:
            self._suspended = True
            self._voice_activation_required = True
        elif self._voice_activation_required and not self.output.connected:
            self._suspended = True

    async def persisted_state(self) -> StoredAudioSession:
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
                        queue_lane=item.queue_lane,
                        request_source=item.request_source,
                        request_id=item.request_id,
                        requested_at_epoch=item.requested_at_epoch,
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
                    queue_lane=item.queue_lane,
                    request_source=item.request_source,
                    request_id=item.request_id,
                    requested_at_epoch=item.requested_at_epoch,
                    played_at_epoch=item.played_at_epoch,
                    uploader=item.uploader,
                    thumbnail_url=item.thumbnail_url,
                )
                for item in self._history
                if item.resolver_reference or item.page_url
            )
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
                autoplay_enabled=self._autoplay_enabled,
                mix_seed_references=tuple(self._mix_seed_references),
                voice_activation_required=self._voice_activation_required,
            )

    def _assert_music_queue_capacity(self, item: AudioItem) -> None:
        if len(self._music) >= self.max_pending_music:
            raise UserError("audio.queue_full")
        actor_id = item.requested_by_id
        if actor_id is not None:
            actor_pending = sum(queued.requested_by_id == actor_id for queued in self._music)
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

    def _remember_mix_seed(self, item: AudioItem) -> None:
        if (
            item.kind is not AudioKind.MUSIC
            or item.queue_lane is AudioQueueLane.AUTOPLAY
        ):
            return
        self._remember_mix_seed_reference(item.resolver_reference or item.page_url)

    def _remember_mix_seed_reference(self, reference: str) -> None:
        _append_mix_seed_reference(self._mix_seed_references, reference)

    def _invalidate_autoplay_locked(self) -> None:
        """Discard only generated candidates; explicit requests stay untouched."""

        self._autoplay_generation += 1
        task = self._autoplay_refill_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._autoplay_refill_task = None
        for item in self._autoplay:
            item.cleanup()
        self._autoplay.clear()

    def _ensure_autoplay_refill(self) -> None:
        """Start one bounded background refill when the automatic lane is empty."""

        if (
            not self._autoplay_enabled
            or self._autoplay_supplier is None
            or not self._mix_seed_references
            or self._manual_music_start_reservations
            or self._autoplay
            or (self._autoplay_refill_task is not None and not self._autoplay_refill_task.done())
            or monotonic() < self._autoplay_retry_at
        ):
            return
        generation = self._autoplay_generation
        seeds = self._radio_seed_sample(generation)
        self._autoplay_refill_task = asyncio.create_task(
            self._run_autoplay_refill(generation, seeds),
            name=f"simajilord-audio-autoplay-{self.workspace_id}",
        )

    async def _run_autoplay_refill(
        self,
        generation: int,
        seeds: tuple[str, ...],
    ) -> None:
        """Fetch flat metadata off the playback path and publish it atomically."""

        current_task = asyncio.current_task()
        candidates: tuple[AudioItem, ...] = ()
        try:
            supplier = self._autoplay_supplier
            if supplier is None:
                return
            candidates = await supplier(seeds, _AUTOPLAY_REFILL_ITEMS)
            async with self._lock:
                if not self._autoplay_enabled or generation != self._autoplay_generation:
                    for item in candidates:
                        item.cleanup()
                    return
                recent_references = {
                    reference
                    for queued in (
                        *((self._current,) if self._current is not None else ()),
                        *self._music,
                        *self._autoplay,
                        *tuple(self._history)[-10:],
                    )
                    if (reference := queued.resolver_reference or queued.page_url)
                }
                selected = self._select_radio_candidates(
                    candidates,
                    generation=generation,
                    recent_references=recent_references,
                )
                added = 0
                selected_ids = {id(item) for item in selected}
                for item in candidates:
                    if id(item) not in selected_ids:
                        item.cleanup()
                for item in selected:
                    reference = item.resolver_reference or item.page_url
                    if (
                        item.kind is not AudioKind.MUSIC
                        or not reference
                        or reference in recent_references
                    ):
                        item.cleanup()
                        continue
                    item.queue_lane = AudioQueueLane.AUTOPLAY
                    item.request_source = item.request_source or "youtube_mix"
                    item.requested_at_epoch = item.requested_at_epoch or int(time())
                    self._autoplay.append(item)
                    recent_references.add(reference)
                    added += 1
                self._autoplay_retry_at = 0.0 if added else monotonic() + _AUTOPLAY_RETRY_SECONDS
                self._wake.set()
        except asyncio.CancelledError:
            for item in candidates:
                item.cleanup()
            raise
        except Exception:
            log.exception(
                "Automatic mix refill failed workspace=%s seeds=%s",
                self.workspace_id,
                len(seeds),
            )
            await self._record_metric(
                ServiceOperationMetric(
                    operation="audio.autoplay_refill",
                    workspace_id=self.workspace_id,
                    wait_ms=0.0,
                    duration_ms=0.0,
                    outcome="failed",
                )
            )
            async with self._lock:
                if generation == self._autoplay_generation:
                    self._autoplay_retry_at = monotonic() + _AUTOPLAY_RETRY_SECONDS
                    self._wake.set()
        finally:
            if self._autoplay_refill_task is current_task:
                self._autoplay_refill_task = None
            await self._state_changed()

    def _radio_seed_sample(self, generation: int) -> tuple[str, ...]:
        """Prefer active manual intent instead of blending unrelated old history."""

        del generation  # Candidate ordering still varies by generation.
        active_requests = (
            *((self._current,) if self._current is not None else ()),
            *self._music,
        )
        active_references = tuple(
            dict.fromkeys(
                reference
                for item in active_requests
                if item.kind is AudioKind.MUSIC
                and item.queue_lane is AudioQueueLane.REQUEST
                and (reference := item.resolver_reference or item.page_url)
            )
        )
        if active_references:
            return active_references[-3:]
        references = tuple(self._mix_seed_references)
        return references[-3:] if references else ()

    def _select_radio_candidates(
        self,
        candidates: tuple[AudioItem, ...],
        *,
        generation: int,
        recent_references: set[str],
    ) -> tuple[AudioItem, ...]:
        """Score a bounded reservoir and queue only the next three radio items."""

        recent_items = tuple(self._history)[-25:]
        recent_titles = {
            _normalize_radio_title(item.title)
            for item in recent_items
            if item.title
        }
        recent_artists = {
            _normalize_radio_text(item.uploader)
            for item in recent_items[-5:]
            if item.uploader
        }
        randomizer = random.Random(
            _stable_radio_seed(self.workspace_id, generation, "candidates")
        )
        ranked: list[tuple[float, int, AudioItem]] = []
        seen_references: set[str] = set()
        seen_titles: set[str] = set()
        for rank, item in enumerate(candidates):
            reference = item.resolver_reference or item.page_url
            normalized_title = _normalize_radio_title(item.title)
            if (
                not reference
                or reference in seen_references
                or (normalized_title and normalized_title in seen_titles)
            ):
                continue
            seen_references.add(reference)
            if normalized_title:
                seen_titles.add(normalized_title)
            score = 100.0 - (rank * 1.5)
            if reference not in recent_references:
                score += 12.0
            if normalized_title in recent_titles:
                score -= 80.0
            artist = _normalize_radio_text(item.uploader)
            if artist and artist in recent_artists:
                score -= 20.0
            if _RADIO_VARIANT_PATTERN.search(item.title):
                score -= 8.0
            if item.duration_seconds > 12 * 60:
                score -= min(25.0, (item.duration_seconds - 12 * 60) / 120)
            score += randomizer.uniform(-2.0, 2.0)
            ranked.append((score, -rank, item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return tuple(item for _, _, item in ranked[:_AUTOPLAY_QUEUE_TARGET])

    async def _next_item(self) -> AudioItem | None:
        async with self._lock:
            if self._speech:
                return self._speech.popleft()
            item = self._pop_ready_music(self._music)
            if item is not None:
                return item
            if self._manual_music_start_reservations:
                return None
            return self._pop_ready_music(self._autoplay)

    def _pop_ready_music(self, queue: deque[AudioItem]) -> AudioItem | None:
        """Pop one playable music item while the caller owns ``self._lock``."""

        now = monotonic()
        for _ in range(len(queue)):
            item = queue.popleft()
            if item.retry_after <= now:
                return item
            queue.append(item)
        return None

    async def _next_retry_delay(self) -> float | None:
        async with self._lock:
            if self._speech:
                return 0.0
            retry_times = tuple(item.retry_after for item in (*self._music, *self._autoplay))
            if retry_times:
                return max(0.0, min(retry_times) - monotonic())
            if self._autoplay_enabled and self._autoplay_retry_at:
                return max(0.0, self._autoplay_retry_at - monotonic())
            return None

    async def _run(self) -> None:
        while not self._closed:
            if self._suspended:
                self._wake.clear()
                if self._suspended:
                    await self._wake.wait()
                continue
            if not self.output.connected:
                # A dropped voice transport is a session-level condition, not
                # one failure per queued track. Preserve the queue unchanged
                # until ``connect()`` wakes the worker after voice is ready.
                self._wake.clear()
                if not self.output.connected:
                    await self._wake.wait()
                continue
            item = await self._next_item()
            if item is None:
                self._ensure_autoplay_refill()
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
            self._speech_active = item.kind is AudioKind.SPEECH
            if item.kind is AudioKind.MUSIC:
                self._remember_mix_seed(item)
                self._ensure_autoplay_refill()
            self._skip_requested = False
            self._discard_requested = False
            self._suspend_requested = False
            self._restart_requested = False
            item.speed = self._speed
            item.pitch = self._pitch
            item.volume = (
                self._speech_volume if item.kind is AudioKind.SPEECH else self._music_volume
            )
            item.speech_overlay_volume = self._speech_volume
            self._started_at = monotonic()
            self._paused_at = None
            self._paused_seconds = 0.0
            await self._state_changed()
            completed = False
            playable = item
            playback_error: Exception | None = None
            try:
                if not self.output.connected:
                    raise UserError("audio.output_disconnected")
                playable = await self._play_with_recovery(item)
                completed = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                playback_error = exc
                log.exception(
                    "Audio playback exhausted immediate retries workspace=%s item=%s",
                    self.workspace_id,
                    item.title,
                )
            finally:
                await self._cancel_overlay_for(playable)
                skipped = self._skip_requested
                discarded = self._discard_requested
                suspended = self._suspend_requested
                restarted = self._restart_requested
                self._current = None
                self._skip_requested = False
                self._discard_requested = False
                self._suspend_requested = False
                self._restart_requested = False
                self._speech_active = False
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
                    resumed = _resume_copy(playable)
                    target = (
                        self._autoplay
                        if resumed.queue_lane is AudioQueueLane.AUTOPLAY
                        else self._music
                    )
                    target.appendleft(resumed)
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
                    target = (
                        self._autoplay
                        if looped.queue_lane is AudioQueueLane.AUTOPLAY
                        else self._music
                    )
                    if self._loop_mode is LoopMode.TRACK:
                        target.appendleft(looped)
                    elif self._loop_mode is LoopMode.QUEUE:
                        target.append(looped)
                    self._wake.set()
                keep_item = True
            elif (
                not completed and not skipped and not discarded and playable.kind is AudioKind.MUSIC
            ):
                retry = playable.unresolved_copy(failure_count=playable.failure_count + 1)
                failure_limit = (
                    MAX_RADIO_PLAYBACK_FAILURES
                    if retry.queue_lane is AudioQueueLane.AUTOPLAY
                    else MAX_MANUAL_PLAYBACK_FAILURES
                )
                permanent = _is_permanent_playback_error(playback_error)
                if not permanent and retry.failure_count < failure_limit:
                    delay = min(300.0, 15.0 * (2 ** min(retry.failure_count - 1, 4)))
                    retry.retry_after = monotonic() + delay
                    async with self._lock:
                        target = (
                            self._autoplay
                            if retry.queue_lane is AudioQueueLane.AUTOPLAY
                            else self._music
                        )
                        target.append(retry)
                    keep_item = True
                    log.warning(
                        "Keeping failed track queued for retry %s/%s in %.0fs "
                        "workspace=%s item=%s",
                        retry.failure_count,
                        failure_limit,
                        delay,
                        self.workspace_id,
                        retry.title,
                    )
                else:
                    log.error(
                        "Dropping failed track after bounded retries workspace=%s "
                        "item=%s lane=%s failures=%s permanent=%s error=%s",
                        self.workspace_id,
                        retry.title,
                        retry.queue_lane.value,
                        retry.failure_count,
                        permanent,
                        type(playback_error).__name__ if playback_error is not None else "unknown",
                    )
                    await self._record_metric(
                        ServiceOperationMetric(
                            operation="audio.playback_dropped",
                            workspace_id=self.workspace_id,
                            wait_ms=0.0,
                            duration_ms=0.0,
                            outcome=(
                                "permanent"
                                if permanent
                                else f"{retry.queue_lane.value}_retry_exhausted"
                            ),
                        )
                    )
            if not keep_item:
                playable.cleanup()
            self._ensure_autoplay_refill()
            await self._state_changed()

    async def _play_with_recovery(self, item: AudioItem) -> AudioItem:
        playable = item
        last_error: Exception | None = None
        early_eof_retried = False
        retry_delays = iter(_IMMEDIATE_RETRY_DELAYS)
        retry_early_eof_now = False
        attempt = 0
        while True:
            if retry_early_eof_now:
                delay = 0.0
                retry_early_eof_now = False
            else:
                try:
                    delay = next(retry_delays)
                except StopIteration:
                    break
            attempt += 1
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
            except EarlyPlaybackEnd as exc:
                await self._record_metric(
                    ServiceOperationMetric(
                        operation="audio.early_eof_count",
                        workspace_id=self.workspace_id,
                        wait_ms=0.0,
                        duration_ms=0.0,
                        outcome="detected",
                    )
                )
                if early_eof_retried:
                    raise
                early_eof_retried = True
                retry_early_eof_now = True
                last_error = exc
                unresolved = playable.unresolved_copy(
                    failure_count=playable.failure_count
                )
                unresolved.start_seconds = self._position_seconds()
                _move_speech_overlay(playable, unresolved)
                playable = unresolved
                continue
            except Exception as exc:
                last_error = exc
                log.warning(
                    "Audio attempt %s failed workspace=%s item=%s error=%s",
                    attempt,
                    self.workspace_id,
                    item.title,
                    type(exc).__name__,
                )
                unresolved = playable.unresolved_copy(failure_count=playable.failure_count)
                _move_speech_overlay(playable, unresolved)
                playable = unresolved
        if last_error is None:
            raise RuntimeError("Audio playback failed without an error.")
        raise last_error

    async def _run_speech_overlays(
        self,
        music: AudioItem,
        first_speech: AudioItem,
    ) -> None:
        """Overlay queued speech without ending the Discord music player."""

        speech: AudioItem | None = first_speech
        try:
            while speech is not None:
                if self._current is not music or music.kind is not AudioKind.MUSIC:
                    async with self._lock:
                        self._speech.appendleft(speech)
                        self._speech_active = False
                        self._wake.set()
                    return
                speech.volume = self._speech_volume
                preparation_started = monotonic()
                try:
                    overlay_error: Exception | None = None
                    for attempt in range(1, _OVERLAY_ATTEMPTS + 1):
                        if self._resolver is not None and (
                            self._must_resolve(music, 1) or attempt > 1
                        ):
                            resolved = await self._resolve(music.unresolved_copy())
                            _adopt_resolved_stream(music, resolved)
                        try:
                            async with self._transport_lock:
                                await self.output.overlay_speech(
                                    music,
                                    speech,
                                    position_seconds=self._position_seconds(),
                                )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            overlay_error = exc
                            log.warning(
                                "Speech overlay attempt %s/%s failed workspace=%s "
                                "item=%s error=%s",
                                attempt,
                                _OVERLAY_ATTEMPTS,
                                self.workspace_id,
                                speech.title,
                                type(exc).__name__,
                            )
                            continue
                        overlay_error = None
                        break
                    if overlay_error is not None:
                        raise overlay_error
                    await self._record_metric(
                        ServiceOperationMetric(
                            operation="audio.overlay_prepare_ms",
                            workspace_id=self.workspace_id,
                            wait_ms=0.0,
                            duration_ms=max(
                                0.0,
                                (monotonic() - preparation_started) * 1_000,
                            ),
                            outcome="succeeded",
                        )
                    )
                except asyncio.CancelledError:
                    async with self._lock:
                        self._speech.appendleft(speech)
                        self._wake.set()
                    speech = None
                    raise
                except Exception:
                    log.exception(
                        "Speech overlay failed after bounded retries; switching to "
                        "standalone speech and then resuming music "
                        "workspace=%s item=%s",
                        self.workspace_id,
                        speech.title,
                    )
                    await self._record_metric(
                        ServiceOperationMetric(
                            operation="audio.overlay_failed",
                            workspace_id=self.workspace_id,
                            wait_ms=0.0,
                            duration_ms=max(
                                0.0,
                                (monotonic() - preparation_started) * 1_000,
                            ),
                            outcome="fallback_standalone",
                        )
                    )
                    await self._fallback_to_standalone_speech(music, speech)
                    # Ownership moved back to the main queue. Stopping the
                    # current transport wakes the worker, which prioritizes
                    # this speech item and then resumes the saved music.
                    speech = None
                    return
                else:
                    speech.cleanup()
                async with self._lock:
                    speech = self._speech.popleft() if self._speech else None
                    self._speech_active = speech is not None

            if self._current is music and self.output.connected:
                music.start_seconds = self._position_seconds()
                swap_started = monotonic()
                async with self._transport_lock:
                    await self.output.update_music(
                        music,
                        position_seconds=music.start_seconds,
                    )
                await self._record_metric(
                    ServiceOperationMetric(
                        operation="audio.source_swap_ms",
                        workspace_id=self.workspace_id,
                        wait_ms=0.0,
                        duration_ms=max(0.0, (monotonic() - swap_started) * 1_000),
                        outcome="succeeded",
                    )
                )
                self._started_at = monotonic()
                self._paused_seconds = 0.0
        finally:
            if speech is not None and speech.owned_file is not None:
                speech.cleanup()
            self._speech_active = False
            if self._overlay_task is asyncio.current_task():
                self._overlay_task = None
            await self._state_changed()

    async def _fallback_to_standalone_speech(
        self,
        music: AudioItem,
        speech: AudioItem,
    ) -> None:
        """Hand a failed overlay back to the normal speech playback path."""

        async with self._transport_lock, self._lock:
            if self._overlay_task is asyncio.current_task():
                # The main worker must not cancel this task after the intentional
                # stop below; ownership has already moved back to its queue.
                self._overlay_task = None
            self._speech.appendleft(speech)
            self._speech_active = False
            self._wake.set()
            if self._current is music and self.output.connected:
                music.start_seconds = self._position_seconds()
                self._restart_requested = True
                self.output.stop()

    async def _cancel_overlay_for(self, music: AudioItem) -> None:
        task = self._overlay_task
        if task is None or task is asyncio.current_task() or self._current is not music:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._overlay_task is task:
            self._overlay_task = None
        self._speech_active = False

    def _must_resolve(self, item: AudioItem, attempt: int) -> bool:
        return item.kind is AudioKind.MUSIC and (
            not item.source
            or attempt > 1
            or monotonic() - item.resolved_at > _MAX_STREAM_AGE_SECONDS
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
        resolved.fade_in_seconds = item.fade_in_seconds
        resolved.fade_out_seconds = item.fade_out_seconds
        resolved.requested_by_id = item.requested_by_id
        resolved.requested_by_name = item.requested_by_name
        resolved.queue_lane = item.queue_lane
        resolved.request_source = item.request_source
        resolved.request_id = item.request_id
        resolved.requested_at_epoch = item.requested_at_epoch
        resolved.played_at_epoch = item.played_at_epoch
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
        position = max(
            0.0,
            offset + (end - self._started_at - self._paused_seconds) * speed,
        )
        if current is not None and current.duration_seconds > 0:
            return min(position, current.duration_seconds)
        return position

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

    async def _record_metric(self, metric: ServiceOperationMetric) -> None:
        if self._metric_hook is None:
            return
        try:
            await self._metric_hook(metric)
        except Exception:
            log.exception(
                "Audio metric hook failed operation=%s workspace=%s",
                metric.operation,
                self.workspace_id,
            )


def _stable_radio_seed(workspace_id: str, generation: int, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{workspace_id}\0{generation}\0{purpose}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _normalize_radio_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())


def _normalize_radio_title(value: str) -> str:
    without_variants = _RADIO_VARIANT_PATTERN.sub(" ", value)
    return _normalize_radio_text(without_variants)


def _move_speech_overlay(source: AudioItem, destination: AudioItem) -> None:
    """Transfer ephemeral overlay ownership without persisting or duplicating it."""

    destination.speech_overlay_source = source.speech_overlay_source
    destination.speech_overlay_owned_file = source.speech_overlay_owned_file
    destination.speech_overlay_duration_seconds = source.speech_overlay_duration_seconds
    source.speech_overlay_owned_file = None


def _is_permanent_playback_error(error: Exception | None) -> bool:
    return isinstance(error, MediaError) and error.category in _PERMANENT_MEDIA_FAILURES


def _adopt_resolved_stream(destination: AudioItem, resolved: AudioItem) -> None:
    """Refresh only transport fields while preserving the worker's item identity."""

    destination.source = resolved.source
    destination.http_headers = resolved.http_headers
    destination.resolved_at = resolved.resolved_at
    destination.duration_seconds = resolved.duration_seconds or destination.duration_seconds
    destination.page_url = resolved.page_url or destination.page_url
    destination.resolver_reference = (
        resolved.resolver_reference or destination.resolver_reference
    )
    destination.uploader = resolved.uploader or destination.uploader
    destination.thumbnail_url = resolved.thumbnail_url or destination.thumbnail_url


def _resume_copy(item: AudioItem) -> AudioItem:
    """Transfer a playable stream back to the queue without forcing re-resolution."""

    resumed = item.clone_for_loop()
    resumed.start_seconds = item.start_seconds
    resumed.failure_count = item.failure_count
    resumed.fade_in_seconds = item.fade_in_seconds
    resumed.fade_out_seconds = 0.0
    resumed.owned_file = item.owned_file
    item.owned_file = None
    return resumed


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
        autoplay_supplier: AutoplaySupplier | None = None,
        workspace_resolver: WorkspaceAudioResolver | None = None,
        workspace_autoplay_supplier: WorkspaceAutoplaySupplier | None = None,
        state_store: AudioStateStore | None = None,
        metric_hook: ServiceMetricHook | None = None,
    ) -> None:
        self.max_active = max_active
        self.max_pending_speech = max_pending_speech
        self.max_pending_music = max_pending_music
        self.max_pending_music_per_actor = max_pending_music_per_actor
        self._resolver = resolver
        self._autoplay_supplier = autoplay_supplier
        self._workspace_resolver = workspace_resolver
        self._workspace_autoplay_supplier = workspace_autoplay_supplier
        self._state_store = state_store
        self._metric_hook = metric_hook
        self._sessions: dict[str, AudioSession] = {}
        self._connection_lock = asyncio.Lock()
        self._connection_locks: dict[str, asyncio.Lock] = {}
        self._connection_reservations: set[str] = set()
        self._state_listeners: list[StateListener] = []

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def active_session_count(self) -> int:
        return sum(session.output.connected for session in self._sessions.values())

    async def queued_audio_count(self) -> int:
        """Return pending music and speech across all isolated guild sessions."""

        snapshots = await asyncio.gather(
            *(session.snapshot() for session in self._sessions.values())
        )
        return sum(len(snapshot.pending) for snapshot in snapshots)

    def get_or_create(
        self,
        workspace_id: str,
        output_factory: Callable[[], AudioOutput],
    ) -> AudioSession:
        existing = self._sessions.get(workspace_id)
        if existing is not None:
            return existing

        session_resolver = self._resolver
        if self._workspace_resolver is not None:

            async def resolve_for_workspace(reference: str) -> AudioItem:
                workspace_resolver = self._workspace_resolver
                if workspace_resolver is None:
                    raise RuntimeError("Workspace media resolver is unavailable.")
                return await workspace_resolver(workspace_id, reference)

            session_resolver = resolve_for_workspace

        session_autoplay_supplier = self._autoplay_supplier
        if self._workspace_autoplay_supplier is not None:

            async def supply_for_workspace(
                seeds: tuple[str, ...],
                limit: int,
            ) -> tuple[AudioItem, ...]:
                workspace_supplier = self._workspace_autoplay_supplier
                if workspace_supplier is None:
                    raise RuntimeError("Workspace autoplay supplier is unavailable.")
                return await workspace_supplier(workspace_id, seeds, limit)

            session_autoplay_supplier = supply_for_workspace

        session = AudioSession(
            workspace_id,
            output_factory(),
            max_pending_speech=self.max_pending_speech,
            max_pending_music=self.max_pending_music,
            max_pending_music_per_actor=self.max_pending_music_per_actor,
            resolver=session_resolver,
            autoplay_supplier=session_autoplay_supplier,
            state_hook=self._state_changed,
            metric_hook=self._metric_hook,
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
        """Reserve global capacity briefly, then connect without blocking other guilds."""

        requested_at = monotonic()
        workspace_lock = self._connection_locks.setdefault(
            workspace_id,
            asyncio.Lock(),
        )
        outcome = "succeeded"
        async with workspace_lock:
            session = self.require(workspace_id)
            reserved = False
            started_at = monotonic()
            try:
                if not session.output.connected:
                    async with self._connection_lock:
                        active_or_reserved = sum(
                            other.output.connected
                            for other_id, other in self._sessions.items()
                            if other_id != workspace_id
                        ) + sum(
                            reservation != workspace_id
                            for reservation in self._connection_reservations
                        )
                        if active_or_reserved >= self.max_active:
                            raise UserError("audio.capacity_reached")
                        self._connection_reservations.add(workspace_id)
                        reserved = True
                await session.connect(destination_id)
            except Exception:
                outcome = "failed"
                raise
            finally:
                if reserved:
                    async with self._connection_lock:
                        self._connection_reservations.discard(workspace_id)
                finished_at = monotonic()
                await self._record_metric(
                    ServiceOperationMetric(
                        operation="voice.connect",
                        workspace_id=workspace_id,
                        wait_ms=max(0.0, (started_at - requested_at) * 1_000),
                        duration_ms=max(0.0, (finished_at - started_at) * 1_000),
                        outcome=outcome,
                        resource_id=destination_id,
                    )
                )

    async def close(self) -> None:
        await asyncio.gather(*(session.shutdown() for session in self._sessions.values()))
        if self._state_store is not None:
            await self._state_store.flush()
        self._sessions.clear()
        self._connection_locks.clear()
        self._connection_reservations.clear()

    def add_state_listener(self, listener: StateListener) -> None:
        """Observe durable session changes without replacing persistence."""

        if listener in self._state_listeners:
            return
        self._state_listeners.append(listener)

    def remove_state_listener(self, listener: StateListener) -> None:
        with suppress(ValueError):
            self._state_listeners.remove(listener)

    async def _state_changed(self, session: AudioSession) -> None:
        await self._persist(session)
        for listener in tuple(self._state_listeners):
            try:
                await listener(session)
            except Exception:
                log.exception(
                    "Audio state listener failed workspace=%s listener=%r",
                    session.workspace_id,
                    listener,
                )

    async def _persist(self, session: AudioSession) -> None:
        if self._state_store is None:
            return
        state = await session.persisted_state()
        await self._state_store.put(state)

    async def _record_metric(self, metric: ServiceOperationMetric) -> None:
        if self._metric_hook is None:
            return
        try:
            await self._metric_hook(metric)
        except Exception:
            log.exception(
                "Audio metric hook failed operation=%s workspace=%s",
                metric.operation,
                metric.workspace_id,
            )
