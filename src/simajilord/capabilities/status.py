"""Structured platform status for every presenter and agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from simajilord.config import EffectiveSecurityPolicy
from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    DisclosureClass,
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
    audit_pending_event_count: int
    audit_retried_event_count: int
    audit_outbox_event_count: int
    audit_lost_event_count: int
    audit_last_failure_at_epoch: int | None
    audit_last_failure_type: str | None
    audit_writer_state: str
    agent_active_turn_count: int
    agent_pending_turn_count: int
    agent_ready_pending_turn_count: int
    agent_workspace_slot_registry_size: int
    agent_conversation_lock_registry_size: int
    security_configured_preset: str
    security_effective_preset: str
    security_preset_expires_at_epoch: int | None
    security_preset_expired: bool
    security_override_names: tuple[str, ...]
    security_agent_enabled: bool
    security_file_sandbox_enabled: bool
    security_file_workspace_mode: str
    security_information_flow_mode: str
    security_read_aloud_audience_mode: str
    security_high_risk_authorization_mode: str
    security_autonomy_enabled: bool
    security_autonomy_mode: str
    security_autonomy_policy_mode: str
    security_autonomy_max_runs: int
    security_web_access: str
    security_compute_access: str
    security_image_access: str
    security_shell_access: str
    security_connector_access: str
    security_warnings: tuple[str, ...]


def build_status_endpoint(
    registry: CapabilityRegistry,
    journal: EventJournal,
    audio: AudioSessionManager,
    web: WebService,
    maintenance: DataMaintenanceService,
    *,
    agent_enabled: bool,
    security_policy: EffectiveSecurityPolicy,
    agent_metrics: Callable[[], Mapping[str, int]] | None = None,
    speech_provider: str,
    speech_voice: str,
) -> CapabilityEndpoint:
    async def status(_: StatusRequest, __: InvocationContext) -> StatusResponse:
        web_ready, web_backend, _web_warning = await web.status()
        maintenance_report = maintenance.last_report
        diagnostics = await journal.operation_diagnostics()
        audit = await journal.audit_health()
        current_agent_metrics = dict(agent_metrics()) if agent_metrics else {}
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
            audit_pending_event_count=audit.pending_events,
            audit_retried_event_count=audit.retried_event_count,
            audit_outbox_event_count=audit.outbox_event_count,
            audit_lost_event_count=audit.lost_event_count,
            audit_last_failure_at_epoch=(
                int(audit.last_failure_at.timestamp())
                if audit.last_failure_at is not None
                else None
            ),
            audit_last_failure_type=audit.last_failure_type,
            audit_writer_state=audit.writer_state,
            agent_active_turn_count=current_agent_metrics.get("active_turns", 0),
            agent_pending_turn_count=current_agent_metrics.get("pending_turns", 0),
            agent_ready_pending_turn_count=current_agent_metrics.get(
                "ready_pending_turns",
                0,
            ),
            agent_workspace_slot_registry_size=current_agent_metrics.get(
                "workspace_slot_registry_size",
                0,
            ),
            agent_conversation_lock_registry_size=current_agent_metrics.get(
                "conversation_lock_registry_size",
                0,
            ),
            security_configured_preset=security_policy.configured_preset,
            security_effective_preset=security_policy.effective_preset,
            security_preset_expires_at_epoch=security_policy.preset_expires_at_epoch,
            security_preset_expired=security_policy.preset_expired,
            security_override_names=security_policy.override_names,
            security_agent_enabled=security_policy.agent_enabled,
            security_file_sandbox_enabled=security_policy.file_sandbox_enabled,
            security_file_workspace_mode=security_policy.file_workspace_mode,
            security_information_flow_mode=security_policy.information_flow_mode,
            security_read_aloud_audience_mode=security_policy.read_aloud_audience_mode,
            security_high_risk_authorization_mode=(
                security_policy.high_risk_authorization_mode
            ),
            security_autonomy_enabled=security_policy.autonomy_enabled,
            security_autonomy_mode=security_policy.autonomy_mode,
            security_autonomy_policy_mode=security_policy.autonomy_policy_mode,
            security_autonomy_max_runs=security_policy.autonomy_max_runs,
            security_web_access=security_policy.web_access,
            security_compute_access=security_policy.compute_access,
            security_image_access=security_policy.image_access,
            security_shell_access=security_policy.shell_access,
            security_connector_access=security_policy.connector_access,
            security_warnings=security_policy.warnings,
        )

    return endpoint(
        CapabilityDescriptor(
            name="system.status",
            summary=(
                "Summarise platform, event journal, audio, AI readiness, and the "
                "effective secret-free security boundary."
            ),
            risk=RiskLevel.READ,
            disclosure_class=DisclosureClass.NO_USER_CONTENT,
            keywords=(
                "status",
                "health",
                "events",
                "model",
                "sessions",
                "storage",
                "cleanup",
                "radio",
                "security",
                "policy",
                "grants",
            ),
        ),
        StatusRequest,
        StatusResponse,
        status,
    )
