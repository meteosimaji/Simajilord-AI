"""Typed endpoint for explicitly requested local feedback submissions."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.services.feedback import FeedbackService


@dataclass(frozen=True, slots=True)
class FeedbackCreateRequest:
    """Only report content may come from a caller; provenance is host-owned."""

    details: str
    title: str | None = None
    expected: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackCreateResponse:
    report_id: str
    status: str
    created: bool


def build_feedback_endpoint(service: FeedbackService) -> CapabilityEndpoint:
    async def create(
        request: FeedbackCreateRequest,
        context: InvocationContext,
    ) -> FeedbackCreateResponse:
        result = await service.create(
            title=request.title,
            details=request.details,
            expected=request.expected,
            reporter_actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            source_transport=context.transport,
            source_event_id=context.request_id,
            source_channel_id=context.origin_resource_id,
            public_reference_id=context.public_reference_id,
            before_mutation=context.dispatch_external_effect,
            on_noop=context.complete_external_effect_without_dispatch,
        )
        return FeedbackCreateResponse(
            report_id=result.report.report_id,
            status=result.report.status.value,
            created=result.created,
        )

    return endpoint(
        CapabilityDescriptor(
            name="feedback.create",
            summary="Save explicitly requested feedback to the local administrator inbox.",
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.WHEN_REQUESTED,
            keywords=(
                "feedback",
                "report bug",
                "feature request",
                "フィードバック",
                "不具合報告",
                "要望",
            ),
            side_effects=("Persists one report in the local feedback database.",),
            idempotency="idempotent_write",
            expected_errors=(
                "feedback.details_required",
                "feedback.details_too_long",
                "feedback.title_too_long",
                "feedback.expected_too_long",
                "feedback.reference_invalid",
            ),
            timeout_seconds=10,
            user_visible_effect="Adds a report to the local administrator inbox.",
        ),
        FeedbackCreateRequest,
        FeedbackCreateResponse,
        create,
    )
