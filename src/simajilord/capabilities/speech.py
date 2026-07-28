"""Speech playback capability shared by transport adapters and agents."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.audio import AudioSessionManager
from simajilord.services.speech import SpeechSegment, SpeechService


@dataclass(frozen=True, slots=True)
class SpeechSpeakRequest:
    text: str = ""
    title: str | None = None
    segments: tuple[SpeechSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeechSpeakResponse:
    title: str
    queue_position: int
    duration_seconds: float
    destination_id: str | None
    playback_state: str


def build_speech_endpoint(
    speech: SpeechService,
    sessions: AudioSessionManager,
) -> CapabilityEndpoint:
    async def speak(
        request: SpeechSpeakRequest,
        context: InvocationContext,
    ) -> SpeechSpeakResponse:
        if context.workspace_id is None:
            raise UserError("workspace.required")
        session = sessions.require(context.workspace_id)
        title = (request.title or "Spoken message").strip() or "Spoken message"
        if request.segments:
            item = await speech.synthesize_segments(
                request.segments,
                title=title,
                workspace_id=context.workspace_id,
            )
        else:
            item = await speech.synthesize(
                request.text,
                title=title,
                workspace_id=context.workspace_id,
            )
        item.requested_by_id = context.actor_id
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
        return SpeechSpeakResponse(
            title=item.title,
            queue_position=position,
            duration_seconds=item.duration_seconds,
            destination_id=snapshot.destination_id,
            playback_state=playback_state,
        )

    return endpoint(
        CapabilityDescriptor(
            name="speech.speak",
            summary=(
                "文章を音声合成して共通音声セッションのキューへ追加します。"
                "音楽の再生中は聞き取りやすいよう自動調整します。"
            ),
            risk=RiskLevel.WRITE,
            keywords=("speech", "voice", "tts", "say", "speak", "voicevox"),
            side_effects=("合成音声を生成して再生します。",),
        ),
        SpeechSpeakRequest,
        SpeechSpeakResponse,
        speak,
    )
