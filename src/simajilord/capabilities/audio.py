"""Audio capabilities using platform services rather than Discord objects."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from time import time
from typing import Any

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.domain.audio import (
    AudioItem,
    AudioKind,
    LoopMode,
    QueueSnapshot,
)
from simajilord.domain.media import MediaCandidate
from simajilord.services.audio import AudioSessionManager
from simajilord.services.media import MediaPriority, MediaService


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
    VOLUME = "volume"
    MOVE = "move"
    CLEAR_MINE = "clear_mine"


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
    queue_lane: str = "request"
    request_source: str | None = None
    request_id: str | None = None
    requested_at_epoch: int | None = None


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
    music_volume_percent: int = 100
    speech_volume_percent: int = 100
    autoplay_enabled: bool = False
    autoplay_next: AudioQueueItem | None = None
    mix_seed_references: tuple[str, ...] = ()
    resume_confirmation_required: bool = False
    connected: bool = False


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
class AudioMixRequest:
    enabled: bool
    seed_references: tuple[str, ...] = ()
    replace_loop: bool = False


@dataclass(frozen=True, slots=True)
class AudioMixResponse:
    enabled: bool
    seed_references: tuple[str, ...]
    next_item: AudioQueueItem | None


@dataclass(frozen=True, slots=True)
class AudioControlRequest:
    action: AudioAction
    loop_mode: LoopMode | None = None
    position: int | None = None
    enabled: bool | None = None
    position_seconds: float | None = None
    speed: float | None = None
    pitch: float | None = None
    music_percent: int | None = None
    speech_percent: int | None = None
    to_position: int | None = None
    replace_mix: bool = False


@dataclass(frozen=True, slots=True)
class AudioControlResponse:
    action: str
    loop_mode: str | None
    affected_title: str | None = None
    enabled: bool | None = None
    position_seconds: float | None = None
    speed: float | None = None
    pitch: float | None = None
    music_volume_percent: int | None = None
    speech_volume_percent: int | None = None
    removed_count: int | None = None


@dataclass(frozen=True, slots=True)
class AudioNoArgsRequest:
    pass


@dataclass(frozen=True, slots=True)
class AudioLoopRequest:
    mode: LoopMode
    replace_mix: bool = False


@dataclass(frozen=True, slots=True)
class AudioQueuePositionRequest:
    position: int


@dataclass(frozen=True, slots=True)
class AudioMoveRequest:
    from_position: int
    to_position: int


@dataclass(frozen=True, slots=True)
class AudioAutoLeaveRequest:
    enabled: bool


@dataclass(frozen=True, slots=True)
class AudioSeekRequest:
    position_seconds: float


@dataclass(frozen=True, slots=True)
class AudioTuneRequest:
    speed: float
    pitch: float


@dataclass(frozen=True, slots=True)
class AudioVolumeRequest:
    music_percent: int | None = None
    speech_percent: int | None = None


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
        candidates = await media.search_audio(
            request.query,
            limit=request.limit,
            workspace_id=workspace_id,
            priority=MediaPriority.INTERACTIVE,
        )
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
        item = await media.resolve_audio(
            request.reference,
            workspace_id=workspace_id,
            priority=MediaPriority.INTERACTIVE,
        )
        item.requested_by_id = context.actor_id
        item.requested_by_name = request.requested_by_name
        item.request_source = context.transport
        item.request_id = context.request_id
        item.requested_at_epoch = int(time())
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
        return audio_queue_response(snapshot)

    async def mix_station(
        request: AudioMixRequest,
        context: InvocationContext,
    ) -> AudioMixResponse:
        session = sessions.require(_workspace(context))
        if request.enabled:
            if len(request.seed_references) > 8:
                raise UserError("audio.mix_seed_limit")
            seeds = await session.enable_autoplay(
                request.seed_references,
                replace_loop=request.replace_loop,
            )
        else:
            await session.disable_autoplay()
            seeds = ()
        snapshot = await session.snapshot()
        return AudioMixResponse(
            enabled=snapshot.autoplay_enabled,
            seed_references=seeds or snapshot.mix_seed_references,
            next_item=(
                _queue_item(snapshot.autoplay_next) if snapshot.autoplay_next is not None else None
            ),
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
        music_volume_percent: int | None = None
        speech_volume_percent: int | None = None
        removed_count: int | None = None
        if request.action is AudioAction.PAUSE:
            await session.pause()
        elif request.action is AudioAction.RESUME:
            await session.resume()
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
        elif request.action is AudioAction.VOLUME:
            if request.music_percent is None and request.speech_percent is None:
                raise UserError("audio.volume_value_required")
            music_volume, speech_volume = await session.set_volume(
                music=(None if request.music_percent is None else request.music_percent / 100),
                speech=(None if request.speech_percent is None else request.speech_percent / 100),
            )
            music_volume_percent = round(music_volume * 100)
            speech_volume_percent = round(speech_volume * 100)
        elif request.action is AudioAction.MOVE:
            if request.position is None or request.to_position is None:
                raise UserError("audio.queue_position_invalid")
            affected_title = (await session.move(request.position, request.to_position)).title
        elif request.action is AudioAction.CLEAR_MINE:
            removed_count = len(await session.clear_for_actor(context.actor_id))
        else:
            if request.loop_mode is None:
                raise UserError("audio.loop_mode_required")
            await session.set_loop(
                request.loop_mode,
                replace_autoplay=request.replace_mix,
            )
        return AudioControlResponse(
            action=request.action.value,
            loop_mode=request.loop_mode.value if request.loop_mode else None,
            affected_title=affected_title,
            enabled=enabled,
            position_seconds=position_seconds,
            speed=speed,
            pitch=pitch,
            music_volume_percent=music_volume_percent,
            speech_volume_percent=speech_volume_percent,
            removed_count=removed_count,
        )

    async def pause(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.PAUSE), context)

    async def resume(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.RESUME), context)

    async def skip(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.SKIP), context)

    async def stop(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.STOP), context)

    async def leave(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.LEAVE), context)

    async def set_loop(
        request: AudioLoopRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(
                action=AudioAction.LOOP,
                loop_mode=request.mode,
                replace_mix=request.replace_mix,
            ),
            context,
        )

    async def remove(
        request: AudioQueuePositionRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(action=AudioAction.REMOVE, position=request.position),
            context,
        )

    async def set_auto_leave(
        request: AudioAutoLeaveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(action=AudioAction.AUTO_LEAVE, enabled=request.enabled),
            context,
        )

    async def shuffle(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.SHUFFLE), context)

    async def seek(
        request: AudioSeekRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(
                action=AudioAction.SEEK,
                position_seconds=request.position_seconds,
            ),
            context,
        )

    async def tune(
        request: AudioTuneRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(
                action=AudioAction.TUNE,
                speed=request.speed,
                pitch=request.pitch,
            ),
            context,
        )

    async def set_volume(
        request: AudioVolumeRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(
                action=AudioAction.VOLUME,
                music_percent=request.music_percent,
                speech_percent=request.speech_percent,
            ),
            context,
        )

    async def move(
        request: AudioMoveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(
            AudioControlRequest(
                action=AudioAction.MOVE,
                position=request.from_position,
                to_position=request.to_position,
            ),
            context,
        )

    async def clear_mine(
        _request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await control(AudioControlRequest(action=AudioAction.CLEAR_MINE), context)

    def exact_control_endpoint(
        name: str,
        summary: str,
        keywords: tuple[str, ...],
        request_type: type[Any],
        handler: Callable[..., Awaitable[AudioControlResponse]],
    ) -> CapabilityEndpoint:
        return endpoint(
            CapabilityDescriptor(
                name=name,
                summary=summary,
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("music", "audio", *keywords),
                side_effects=("サーバーの永続音声セッションを変更します。",),
            ),
            request_type,
            AudioControlResponse,
            handler,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="audio.search",
                summary="音楽を検索し、候補を自動選択できるか確認します。",
                risk=RiskLevel.EXTERNAL,
                keywords=(
                    "music",
                    "search",
                    "song",
                    "track",
                    "candidate",
                    "disambiguate",
                ),
                side_effects=("メディア検索サービスへ接続します。",),
            ),
            AudioSearchRequest,
            AudioSearchResponse,
            search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.history",
                summary="このワークスペースで最近再生した音楽を確認します。",
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
                summary="このワークスペースで再生中・待機中の音声を確認します。",
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
                summary="公開メディアを解決し、音声を再生キューへ追加します。",
                risk=RiskLevel.EXTERNAL,
                keywords=("music", "media", "song", "track", "stream", "queue"),
                side_effects=("メディアサイトへ接続します。", "現在の出力先で音声を再生します。"),
            ),
            AudioPlayRequest,
            AudioPlayResponse,
            play,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.control",
                summary="一時停止・再開・スキップ・停止・退出・ループ変更などを行います。",
                risk=RiskLevel.WRITE,
                keywords=("music", "voice", "pause", "skip", "loop"),
                side_effects=("再生状態を変更します。",),
            ),
            AudioControlRequest,
            AudioControlResponse,
            control,
        ),
        endpoint(
            CapabilityDescriptor(
                name="audio.mix",
                summary=(
                    "最大8曲を起点にYouTube Mixを有効化または停止します。"
                    "人が追加した曲を常に優先し、空いたときだけ次の1曲を自動供給します。"
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("music", "mix", "autoplay", "station", "radio"),
                side_effects=(
                    "YouTube Mixの候補メタデータを取得します。",
                    "サーバーの永続音声セッションを変更します。",
                ),
            ),
            AudioMixRequest,
            AudioMixResponse,
            mix_station,
        ),
        exact_control_endpoint(
            "audio.pause",
            "再生中の音楽を一時停止します。",
            ("pause",),
            AudioNoArgsRequest,
            pause,
        ),
        exact_control_endpoint(
            "audio.resume",
            "一時停止した音楽を再開します。",
            ("resume",),
            AudioNoArgsRequest,
            resume,
        ),
        exact_control_endpoint(
            "audio.skip",
            "再生中の曲をスキップします。",
            ("skip",),
            AudioNoArgsRequest,
            skip,
        ),
        exact_control_endpoint(
            "audio.stop",
            "再生を止め、音楽キューを空にします。",
            ("stop", "clear"),
            AudioNoArgsRequest,
            stop,
        ),
        exact_control_endpoint(
            "audio.leave",
            "音楽キューを空にして音声接続を終了します。",
            ("leave", "disconnect"),
            AudioNoArgsRequest,
            leave,
        ),
        exact_control_endpoint(
            "audio.set_loop",
            "1曲またはキュー全体のループ方法を設定します。",
            ("loop", "repeat"),
            AudioLoopRequest,
            set_loop,
        ),
        exact_control_endpoint(
            "audio.remove",
            "指定位置の待機曲をキューから削除します。",
            ("remove", "queue"),
            AudioQueuePositionRequest,
            remove,
        ),
        exact_control_endpoint(
            "audio.set_auto_leave",
            "人がいなくなったときの自動退出を設定します。",
            ("auto", "leave"),
            AudioAutoLeaveRequest,
            set_auto_leave,
        ),
        exact_control_endpoint(
            "audio.shuffle",
            "待機中の音楽キューをシャッフルします。",
            ("shuffle", "queue"),
            AudioNoArgsRequest,
            shuffle,
        ),
        exact_control_endpoint(
            "audio.seek",
            "再生中の曲を指定秒へ移動します。",
            ("seek", "position"),
            AudioSeekRequest,
            seek,
        ),
        exact_control_endpoint(
            "audio.tune",
            "音楽の速度とピッチを設定します。",
            ("speed", "pitch", "tune"),
            AudioTuneRequest,
            tune,
        ),
        exact_control_endpoint(
            "audio.set_volume",
            "音楽と読み上げの音量を百分率で設定します。",
            ("volume", "speech"),
            AudioVolumeRequest,
            set_volume,
        ),
        exact_control_endpoint(
            "audio.move",
            "待機曲をキュー内の別の位置へ移動します。",
            ("move", "queue", "reorder"),
            AudioMoveRequest,
            move,
        ),
        exact_control_endpoint(
            "audio.clear_mine",
            "依頼者自身が追加した待機曲だけを削除します。",
            ("clear", "mine", "requester"),
            AudioNoArgsRequest,
            clear_mine,
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
        queue_lane=item.queue_lane.value,
        request_source=item.request_source,
        request_id=item.request_id,
        requested_at_epoch=item.requested_at_epoch,
    )


def audio_queue_response(snapshot: QueueSnapshot) -> AudioQueueResponse:
    """Project one transport-neutral snapshot into the public capability model."""

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
        music_volume_percent=round(snapshot.music_volume * 100),
        speech_volume_percent=round(snapshot.speech_volume * 100),
        autoplay_enabled=snapshot.autoplay_enabled,
        autoplay_next=(
            _queue_item(snapshot.autoplay_next) if snapshot.autoplay_next is not None else None
        ),
        mix_seed_references=snapshot.mix_seed_references,
        resume_confirmation_required=snapshot.resume_confirmation_required,
        connected=snapshot.connected,
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

    candidate_indexes = {candidate.reference: index for index, candidate in enumerate(candidates)}
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
        if candidate.uploader and _contains_phrase(query_key, _text_key(candidate.uploader))
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
