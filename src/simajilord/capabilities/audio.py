"""Audio capabilities using platform services rather than Discord objects."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, AudioKind, LoopMode
from simajilord.domain.media import MediaCandidate
from simajilord.services.audio import AudioSessionManager
from simajilord.services.media import MediaService


class AudioAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    SKIP = "skip"
    STOP = "stop"
    LEAVE = "leave"
    LOOP = "loop"
    REMOVE = "remove"
    AUTO_LEAVE = "auto_leave"
    SHUFFLE = "shuffle"
    SEEK = "seek"
    TUNE = "tune"


class AudioSearchReason(StrEnum):
    """Why a candidate was selected or why one human choice is still useful."""

    HISTORY = "history"
    UPLOADER = "uploader"
    SINGLE = "single"
    TOP_RESULT = "top_result"
    AMBIGUOUS_TITLE = "ambiguous_title"


@dataclass(frozen=True, slots=True)
class AudioSearchRequest:
    query: str
    limit: int = 5


@dataclass(frozen=True, slots=True)
class AudioSearchItem:
    reference: str
    title: str
    duration_seconds: float
    uploader: str | None = None
    thumbnail_url: str | None = None


@dataclass(frozen=True, slots=True)
class AudioSearchResponse:
    query: str
    candidates: tuple[AudioSearchItem, ...]
    selected_index: int | None
    selection_required: bool
    reason: AudioSearchReason


@dataclass(frozen=True, slots=True)
class AudioPlayRequest:
    reference: str
    requested_by_name: str | None = None


@dataclass(frozen=True, slots=True)
class AudioPlayResponse:
    title: str
    page_url: str
    queue_position: int
    duration_seconds: float
    destination_id: str | None
    playback_state: str
    requested_by_name: str | None
    uploader: str | None = None
    thumbnail_url: str | None = None


@dataclass(frozen=True, slots=True)
class AudioQueueRequest:
    pass


@dataclass(frozen=True, slots=True)
class AudioQueueItem:
    title: str
    page_url: str
    kind: str
    duration_seconds: float
    requested_by_name: str | None
    uploader: str | None = None
    thumbnail_url: str | None = None


@dataclass(frozen=True, slots=True)
class AudioQueueResponse:
    current: AudioQueueItem | None
    pending: tuple[AudioQueueItem, ...]
    paused: bool
    loop_mode: str
    destination_id: str | None
    auto_leave: bool
    position_seconds: float
    speed: float
    pitch: float
    waiting_for_voice: bool


@dataclass(frozen=True, slots=True)
class AudioHistoryRequest:
    limit: int = 10


@dataclass(frozen=True, slots=True)
class AudioHistoryItem:
    title: str
    page_url: str
    duration_seconds: float
    requested_by_name: str | None
    played_at_epoch: int | None


@dataclass(frozen=True, slots=True)
class AudioHistoryResponse:
    items: tuple[AudioHistoryItem, ...]


@dataclass(frozen=True, slots=True)
class AudioControlRequest:
    action: AudioAction
    loop_mode: LoopMode | None = None
    position: int | None = None
    enabled: bool | None = None
    position_seconds: float | None = None
    speed: float | None = None
    pitch: float | None = None


@dataclass(frozen=True, slots=True)
class AudioControlResponse:
    action: str
    loop_mode: str | None
    affected_title: str | None = None
    enabled: bool | None = None
    position_seconds: float | None = None
    speed: float | None = None
    pitch: float | None = None


def build_audio_endpoints(
    media: MediaService,
    sessions: AudioSessionManager,
) -> tuple[CapabilityEndpoint, ...]:
    async def search(
        request: AudioSearchRequest,
        context: InvocationContext,
    ) -> AudioSearchResponse:
        if not 1 <= request.limit <= 10:
            raise UserError("audio.search_limit_invalid")
        workspace_id = _workspace(context)
        candidates = await media.search_audio(request.query, limit=request.limit)
        session = sessions.find(workspace_id)
        snapshot = await session.snapshot() if session is not None else None
        selected_index, reason = _select_candidate(
            request.query,
            candidates,
            actor_id=context.actor_id,
            known_items=(
                ()
                if snapshot is None
                else tuple(
                    item
                    for item in (
                        *((snapshot.current,) if snapshot.current is not None else ()),
                        *snapshot.pending,
                        *snapshot.history,
                    )
                    if item.kind is AudioKind.MUSIC
                )
            ),
        )
        return AudioSearchResponse(
            query=request.query,
            candidates=tuple(_search_item(candidate) for candidate in candidates),
            selected_index=selected_index,
            selection_required=selected_index is None,
            reason=reason,
        )

    async def play(
        request: AudioPlayRequest,
        context: InvocationContext,
    ) -> AudioPlayResponse:
        workspace_id = _workspace(context)
        session = sessions.require(workspace_id)
        item = await media.resolve_audio(request.reference)
        item.requested_by_id = context.actor_id
        item.requested_by_name = request.requested_by_name
        if not session.output.connected:
            await session.wait_for_listener(context.actor_id)
        position = await session.enqueue(item)
        snapshot = await session.snapshot()
        if snapshot.current is not None:
            playback_state = "playing"
        elif snapshot.waiting_actor_ids:
            playback_state = "waiting_for_voice"
        else:
            playback_state = "queued"
        return AudioPlayResponse(
            title=item.title,
            page_url=item.page_url,
            queue_position=position,
            duration_seconds=item.duration_seconds,
            destination_id=snapshot.destination_id,
            playback_state=playback_state,
            requested_by_name=item.requested_by_name,
            uploader=item.uploader,
            thumbnail_url=item.thumbnail_url,
        )

    async def queue(
        _: AudioQueueRequest,
        context: InvocationContext,
    ) -> AudioQueueResponse:
        snapshot = await sessions.require(_workspace(context)).snapshot()
        return AudioQueueResponse(
            current=_queue_item(snapshot.current) if snapshot.current else None,
            pending=tuple(_queue_item(item) for item in snapshot.pending),
            paused=snapshot.paused,
            loop_mode=snapshot.loop.value,
            destination_id=snapshot.destination_id,
            auto_leave=snapshot.auto_leave,
            position_seconds=snapshot.position_seconds,
            speed=snapshot.speed,
            pitch=snapshot.pitch,
            waiting_for_voice=bool(snapshot.waiting_actor_ids),
        )

    async def history(
        request: AudioHistoryRequest,
        context: InvocationContext,
    ) -> AudioHistoryResponse:
        if not 1 <= request.limit <= 25:
            raise UserError("audio.history_limit_invalid")
        snapshot = await sessions.require(_workspace(context)).snapshot()
        return AudioHistoryResponse(
            items=tuple(
                AudioHistoryItem(
                    title=item.title,
                    page_url=item.page_url,
                    duration_seconds=item.duration_seconds,
                    requested_by_name=item.requested_by_name,
                    played_at_epoch=item.played_at_epoch,
                )
                for item in snapshot.history[: request.limit]
                if item.kind is AudioKind.MUSIC
            )
        )

    async def control(
        request: AudioControlRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        session = sessions.require(_workspace(context))
        affected_title: str | None = None
        enabled: bool | None = None
        position_seconds: float | None = None
        speed: float | None = None
        pitch: float | None = None
        if request.action is AudioAction.PAUSE:
            session.pause()
        elif request.action is AudioAction.RESUME:
            session.resume()
        elif request.action is AudioAction.SKIP:
            await session.skip()
        elif request.action is AudioAction.STOP:
            await session.clear()
        elif request.action is AudioAction.LEAVE:
            await session.disconnect()
        elif request.action is AudioAction.REMOVE:
            if request.position is None:
                raise UserError("audio.queue_position_invalid")
            affected_title = (await session.remove(request.position)).title
        elif request.action is AudioAction.AUTO_LEAVE:
            if request.enabled is None:
                raise UserError("audio.auto_leave_value_required")
            await session.set_auto_leave(request.enabled)
            enabled = request.enabled
        elif request.action is AudioAction.SHUFFLE:
            await session.shuffle()
        elif request.action is AudioAction.SEEK:
            if request.position_seconds is None:
                raise UserError("audio.seek_position_required")
            position_seconds = await session.seek(request.position_seconds)
        elif request.action is AudioAction.TUNE:
            if request.speed is None or request.pitch is None:
                raise UserError("audio.tune_values_required")
            await session.tune(request.speed, request.pitch)
            speed = request.speed
            pitch = request.pitch
        else:
            if request.loop_mode is None:
                raise UserError("audio.loop_mode_required")
            await session.set_loop(request.loop_mode)
        return AudioControlResponse(
            action=request.action.value,
            loop_mode=request.loop_mode.value if request.loop_mode else None,
            affected_title=affected_title,
            enabled=enabled,
            position_seconds=position_seconds,
            speed=speed,
            pitch=pitch,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="audio.search",
                summary="Search music and identify whether one candidate needs confirmation.",
                risk=RiskLevel.EXTERNAL,
                keywords=(
                    "music",
                    "search",
                    "song",
                    "track",
                    "candidate",
                    "disambiguate",
                ),
                side_effects=("Uses a media search provider.",),
            ),
            AudioSearchRequest,
            AudioSearchResponse,
            search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.history",
                summary="Inspect recently played music for one workspace.",
                risk=RiskLevel.READ,
                keywords=("music", "history", "recent", "played", "karaoke"),
            ),
            AudioHistoryRequest,
            AudioHistoryResponse,
            history,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.queue",
                summary="Inspect current and pending audio for one workspace.",
                risk=RiskLevel.READ,
                keywords=("music", "speech", "queue", "playing", "now"),
            ),
            AudioQueueRequest,
            AudioQueueResponse,
            queue,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.play",
                summary="Resolve a public media reference and enqueue its audio.",
                risk=RiskLevel.EXTERNAL,
                keywords=("music", "media", "song", "track", "stream", "queue"),
                side_effects=("Uses a media site.", "Plays audio in the active output."),
            ),
            AudioPlayRequest,
            AudioPlayResponse,
            play,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.control",
                summary="Pause, resume, skip, stop, leave, or change looping.",
                risk=RiskLevel.WRITE,
                keywords=("music", "voice", "pause", "skip", "loop"),
                side_effects=("Changes playback state.",),
            ),
            AudioControlRequest,
            AudioControlResponse,
            control,
        ),
    )


def _queue_item(item: AudioItem) -> AudioQueueItem:
    return AudioQueueItem(
        title=item.title,
        page_url=item.page_url,
        kind=item.kind.value,
        duration_seconds=item.duration_seconds,
        requested_by_name=item.requested_by_name,
        uploader=item.uploader,
        thumbnail_url=item.thumbnail_url,
    )


def _workspace(context: InvocationContext) -> str:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    return context.workspace_id


def _search_item(candidate: MediaCandidate) -> AudioSearchItem:
    return AudioSearchItem(
        reference=candidate.reference,
        title=candidate.title,
        duration_seconds=candidate.duration_seconds,
        uploader=candidate.uploader,
        thumbnail_url=candidate.thumbnail_url,
    )


def _select_candidate(
    query: str,
    candidates: tuple[MediaCandidate, ...],
    *,
    actor_id: str,
    known_items: tuple[AudioItem, ...],
) -> tuple[int | None, AudioSearchReason]:
    if not candidates:
        raise UserError("audio.search_empty")

    candidate_indexes = {
        candidate.reference: index for index, candidate in enumerate(candidates)
    }
    for item in known_items:
        if item.requested_by_id != actor_id:
            continue
        reference = item.resolver_reference or item.page_url
        if reference in candidate_indexes:
            return candidate_indexes[reference], AudioSearchReason.HISTORY

    query_key = _text_key(query)
    uploader_matches = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.uploader
        and _contains_phrase(query_key, _text_key(candidate.uploader))
    ]
    if len(uploader_matches) == 1:
        return uploader_matches[0], AudioSearchReason.UPLOADER

    if len(candidates) == 1:
        return 0, AudioSearchReason.SINGLE

    exact_title_matches = [
        index
        for index, candidate in enumerate(candidates)
        if _candidate_title_key(candidate) == query_key
    ]
    exact_uploaders = {
        _text_key(candidates[index].uploader or "")
        for index in exact_title_matches
        if candidates[index].uploader
    }
    if len(exact_title_matches) > 1 and len(exact_uploaders) > 1:
        return None, AudioSearchReason.AMBIGUOUS_TITLE
    if len(exact_title_matches) == 1:
        return exact_title_matches[0], AudioSearchReason.TOP_RESULT
    return 0, AudioSearchReason.TOP_RESULT


def _candidate_title_key(candidate: MediaCandidate) -> str:
    title_key = _text_key(candidate.title)
    uploader_key = _text_key(candidate.uploader or "")
    if uploader_key and title_key.startswith(f"{uploader_key} "):
        title_key = title_key[len(uploader_key) + 1 :]
    ignored = {
        "4k",
        "audio",
        "hd",
        "hq",
        "lyrics",
        "lyric",
        "mv",
        "official",
        "remaster",
        "remastered",
        "video",
        "visualizer",
    }
    return " ".join(token for token in title_key.split() if token not in ignored)


def _text_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "
