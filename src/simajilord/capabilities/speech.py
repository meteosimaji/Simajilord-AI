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
    voice_preset: str | None = None


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
        reservation = await session.reserve_speech()
        title = (request.title or "Spoken message").strip() or "Spoken message"
        try:
            if request.segments:
                item = await speech.synthesize_segments(
                    request.segments,
                    title=title,
                    workspace_id=context.workspace_id,
                    voice_preset=request.voice_preset,
                )
            else:
                item = await speech.synthesize(
                    request.text,
                    title=title,
                    workspace_id=context.workspace_id,
                    voice_preset=request.voice_preset,
                )
            item.requested_by_id = context.actor_id
            if not session.output.connected:
                await session.wait_for_listener(context.actor_id)
            position = await reservation.commit(item)
        finally:
            await reservation.release()
        snapshot = await session.snapshot()
        if snapshot.current is not None or (
            position == 1 and session.output.connected
        ):
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
                "Synthesize text and add it to the shared audio session. "
                "Speech is mixed for intelligibility while music is playing."
            ),
            risk=RiskLevel.WRITE,
            keywords=("speech", "voice", "tts", "say", "speak", "voicevox"),
            side_effects=("Generates and plays synthesized speech.",),
            requires_workspace=True,
            idempotency="non_idempotent_write",
            expected_errors=("workspace.required",),
            timeout_seconds=90,
            user_visible_effect="Plays synthesized speech in the shared audio session.",
        ),
        SpeechSpeakRequest,
        SpeechSpeakResponse,
        speak,
    )
