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
from simajilord.services.speech import SpeechService


@dataclass(frozen=True, slots=True)
class SpeechSpeakRequest:
    text: str
    title: str | None = None


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
        item = await speech.synthesize(
            request.text,
            title=(request.title or "Spoken message").strip() or "Spoken message",
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
                "Synthesize a spoken message and enqueue it in the shared audio "
                "session, ducking active music automatically."
            ),
            risk=RiskLevel.WRITE,
            keywords=("speech", "voice", "tts", "say", "speak", "voicevox"),
            side_effects=("Synthesizes and plays audible speech.",),
        ),
        SpeechSpeakRequest,
        SpeechSpeakResponse,
        speak,
    )
