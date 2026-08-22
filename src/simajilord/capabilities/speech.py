"""Speech playback capability shared by transport adapters and agents."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.audio import AudioSessionManager
from simajilord.services.speech import (
    SpeechSegment,
    SpeechSegmentKind,
    SpeechService,
    progressive_speech_parts,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SpeechSpeakRequest:
    text: str = ""
    title: str | None = None
    segments: tuple[SpeechSegment, ...] = ()
    voice_preset: str | None = None
    purpose: Literal["progress", "requested_action", "final"] = "requested_action"


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
        requested_segments = request.segments or (
            SpeechSegment(SpeechSegmentKind.BODY, request.text),
        )
        parts = progressive_speech_parts(
            requested_segments,
            maximum_parts=min(8, session.max_pending_speech),
        )
        reservation = await session.reserve_speech(slots=len(parts))
        first_position: int | None = None
        total_duration_seconds = 0.0
        first_started_at = monotonic()
        try:
            for part_index, part in enumerate(parts, start=1):
                item = await speech.synthesize_segments(
                    part,
                    title=title,
                    workspace_id=context.workspace_id,
                    voice_preset=request.voice_preset,
                    before_synthesis=(
                        context.dispatch_external_effect if part_index == 1 else None
                    ),
                )
                item.requested_by_id = context.actor_id
                item.request_id = (
                    context.request_id
                    if len(parts) == 1
                    else f"{context.request_id}:p{part_index}of{len(parts)}"
                )
                item.request_source = context.transport
                if part_index == 1 and not session.output.connected:
                    await session.wait_for_listener(context.actor_id)
                position = await reservation.commit_part(
                    item,
                    final=part_index == len(parts),
                )
                total_duration_seconds += item.duration_seconds or 0.0
                if first_position is None:
                    first_position = position
                    if len(parts) > 1:
                        log.info(
                            "Progressive speech first part queued workspace=%s "
                            "request=%s parts=%s preparation_ms=%.1f",
                            context.workspace_id,
                            context.request_id,
                            len(parts),
                            max(0.0, (monotonic() - first_started_at) * 1_000),
                        )
        finally:
            await reservation.release()
        if first_position is None:
            raise RuntimeError("Speech planning produced no audio parts")
        snapshot = await session.snapshot()
        if snapshot.current is not None or (
            first_position == 1 and session.output.connected
        ):
            playback_state = "playing"
        elif snapshot.waiting_actor_ids:
            playback_state = "waiting_for_voice"
        else:
            playback_state = "queued"
        return SpeechSpeakResponse(
            title=title,
            queue_position=first_position,
            duration_seconds=total_duration_seconds,
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
