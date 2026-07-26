"""Audio capabilities using platform services rather than Discord objects."""

from __future__ import annotations

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
from simajilord.domain.audio import AudioItem, LoopMode
from simajilord.services.audio import AudioSessionManager
from simajilord.services.media import MediaService


class AudioAction(StrEnum):
    PAUSE = "pause"
    RESUME = "resume"
    SKIP = "skip"
    STOP = "stop"
    LEAVE = "leave"
    LOOP = "loop"


@dataclass(frozen=True, slots=True)
class AudioPlayRequest:
    reference: str


@dataclass(frozen=True, slots=True)
class AudioPlayResponse:
    title: str
    page_url: str
    queue_position: int


@dataclass(frozen=True, slots=True)
class AudioQueueRequest:
    pass


@dataclass(frozen=True, slots=True)
class AudioQueueItem:
    title: str
    page_url: str
    kind: str


@dataclass(frozen=True, slots=True)
class AudioQueueResponse:
    current: AudioQueueItem | None
    pending: tuple[AudioQueueItem, ...]
    paused: bool
    loop_mode: str


@dataclass(frozen=True, slots=True)
class AudioControlRequest:
    action: AudioAction
    loop_mode: LoopMode | None = None


@dataclass(frozen=True, slots=True)
class AudioControlResponse:
    action: str
    loop_mode: str | None


def build_audio_endpoints(
    media: MediaService,
    sessions: AudioSessionManager,
) -> tuple[CapabilityEndpoint, ...]:
    async def play(
        request: AudioPlayRequest,
        context: InvocationContext,
    ) -> AudioPlayResponse:
        workspace_id = _workspace(context)
        session = sessions.require(workspace_id)
        item = await media.resolve_audio(request.reference)
        position = await session.enqueue(item)
        return AudioPlayResponse(
            title=item.title,
            page_url=item.page_url,
            queue_position=position,
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
        )

    async def control(
        request: AudioControlRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        session = sessions.require(_workspace(context))
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
        else:
            if request.loop_mode is None:
                raise UserError("audio.loop_mode_required")
            await session.set_loop(request.loop_mode)
        return AudioControlResponse(
            action=request.action.value,
            loop_mode=request.loop_mode.value if request.loop_mode else None,
        )

    return (
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
                summary="Resolve a supported media reference and enqueue its audio.",
                risk=RiskLevel.EXTERNAL,
                keywords=("music", "youtube", "tiktok", "song", "queue"),
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
    )


def _workspace(context: InvocationContext) -> str:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    return context.workspace_id
