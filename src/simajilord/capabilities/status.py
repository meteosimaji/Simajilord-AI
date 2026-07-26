"""Structured platform status for every presenter and future agent."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.observability import EventJournal
from simajilord.services.audio import AudioSessionManager


@dataclass(frozen=True, slots=True)
class StatusRequest:
    pass


@dataclass(frozen=True, slots=True)
class StatusResponse:
    status: str
    capability_count: int
    event_cursor: int
    audio_session_count: int
    active_audio_session_count: int
    model_runtime: str


def build_status_endpoint(
    registry: CapabilityRegistry,
    journal: EventJournal,
    audio: AudioSessionManager,
) -> CapabilityEndpoint:
    async def status(_: StatusRequest, __: InvocationContext) -> StatusResponse:
        return StatusResponse(
            status="ok",
            capability_count=len(registry.all()),
            event_cursor=await journal.latest_sequence(),
            audio_session_count=audio.session_count,
            active_audio_session_count=audio.active_session_count,
            model_runtime="disabled",
        )

    return endpoint(
        CapabilityDescriptor(
            name="system.status",
            summary="Return structured platform, journal, audio, and model-runtime status.",
            risk=RiskLevel.READ,
            keywords=("status", "health", "events", "model", "sessions"),
        ),
        StatusRequest,
        StatusResponse,
        status,
    )
