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
from simajilord.services.maintenance import DataMaintenanceService
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
    storage_used_bytes: int
    storage_limit_bytes: int
    storage_over_capacity: bool
    queued_audio_count: int
    orphan_cleanup_removed: int
    cleanup_completed_at_epoch: int | None
    last_radio_failure_at_epoch: int | None
    overlay_failure_count: int
    dashboard_429_count: int


def build_status_endpoint(
    registry: CapabilityRegistry,
    journal: EventJournal,
    audio: AudioSessionManager,
    web: WebService,
    maintenance: DataMaintenanceService,
    *,
    agent_enabled: bool,
    speech_provider: str,
    speech_voice: str,
) -> CapabilityEndpoint:
    async def status(_: StatusRequest, __: InvocationContext) -> StatusResponse:
        web_ready, web_backend, _web_warning = await web.status()
        maintenance_report = maintenance.last_report
        diagnostics = await journal.operation_diagnostics()
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
            storage_used_bytes=maintenance_report.storage_used_bytes,
            storage_limit_bytes=maintenance_report.storage_limit_bytes,
            storage_over_capacity=maintenance_report.over_capacity,
            queued_audio_count=await audio.queued_audio_count(),
            orphan_cleanup_removed=maintenance_report.orphan_cleanup_removed,
            cleanup_completed_at_epoch=(
                int(maintenance_report.completed_at.timestamp())
                if maintenance_report.completed_at is not None
                else None
            ),
            last_radio_failure_at_epoch=(
                int(diagnostics.last_radio_failure_at.timestamp())
                if diagnostics.last_radio_failure_at is not None
                else None
            ),
            overlay_failure_count=diagnostics.overlay_failure_count,
            dashboard_429_count=diagnostics.dashboard_429_count,
        )

    return endpoint(
        CapabilityDescriptor(
            name="system.status",
            summary="Summarise platform, event journal, audio, and AI readiness.",
            risk=RiskLevel.READ,
            keywords=(
                "status",
                "health",
                "events",
                "model",
                "sessions",
                "storage",
                "cleanup",
                "radio",
            ),
        ),
        StatusRequest,
        StatusResponse,
        status,
    )
