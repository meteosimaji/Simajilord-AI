"""Structured platform status for every presenter and agent."""

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
from simajilord.services.web import WebService


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
    speech_provider: str
    speech_voice: str
    web_search_backend: str
    web_search_ready: bool


def build_status_endpoint(
    registry: CapabilityRegistry,
    journal: EventJournal,
    audio: AudioSessionManager,
    web: WebService,
    *,
    agent_enabled: bool,
    speech_provider: str,
    speech_voice: str,
) -> CapabilityEndpoint:
    async def status(_: StatusRequest, __: InvocationContext) -> StatusResponse:
        web_ready, web_backend, _web_warning = await web.status()
        return StatusResponse(
            status="ok",
            capability_count=len(registry.all()),
            event_cursor=await journal.latest_sequence(),
            audio_session_count=audio.session_count,
            active_audio_session_count=audio.active_session_count,
            model_runtime="enabled" if agent_enabled else "disabled",
            speech_provider=speech_provider,
            speech_voice=speech_voice,
            web_search_backend=web_backend,
            web_search_ready=web_ready,
        )

    return endpoint(
        CapabilityDescriptor(
            name="system.status",
            summary="基盤・イベント記録・音声・AI実行環境の状態をまとめて返します。",
            risk=RiskLevel.READ,
            keywords=("status", "health", "events", "model", "sessions"),
        ),
        StatusRequest,
        StatusResponse,
        status,
    )
