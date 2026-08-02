"""Typed host/model handshake for routing Discord events to active tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError


@dataclass(frozen=True, slots=True)
class TaskRouteRequest:
    candidate_event_id: str = field(
        metadata={
            "description": "Copy the pending candidate_event_id from the host pointer exactly."
        }
    )
    decision: Literal["attach", "separate", "finish", "cancel"] = field(
        metadata={
            "description": (
                "attach adds a genuine instruction to the active task; separate preserves it "
                "as independent work; finish resolves a correction, resend, or no-new-work "
                "event and asks the active task to conclude; cancel interrupts unfinished "
                "work when the user is withdrawing it."
            )
        }
    )
    reason: str


@dataclass(frozen=True, slots=True)
class TaskRouteResponse:
    candidate_event_id: str
    decision: Literal["attach", "separate", "finish", "cancel"]
    reason: str
    validated: bool


def build_task_route_endpoint() -> CapabilityEndpoint:
    """Build the typed decision surface; provider state applies the decision."""

    async def route(
        request: TaskRouteRequest,
        _: InvocationContext,
    ) -> TaskRouteResponse:
        event_id = request.candidate_event_id.strip()
        reason = " ".join(request.reason.split())
        if not event_id or len(event_id) > 500:
            raise UserError("agent.task_candidate_invalid")
        if not reason or len(reason) > 400:
            raise UserError("agent.task_route_reason_invalid")
        return TaskRouteResponse(
            candidate_event_id=event_id,
            decision=request.decision,
            reason=reason,
            validated=True,
        )

    return endpoint(
        CapabilityDescriptor(
            name="turn.route_task_event",
            summary=(
                "After reading the exact pending Discord candidate, choose the typed "
                "relationship to the active task: attach, separate, finish, or cancel. This is "
                "semantic model judgment; the host does not classify message text."
            ),
            risk=RiskLevel.READ,
            keywords=(
                "task routing",
                "follow-up decision",
                "attach separate finish cancel",
                "タスク振り分け",
            ),
            idempotency="read",
            expected_errors=(
                "agent.task_candidate_invalid",
                "agent.task_candidate_unknown",
                "agent.task_candidate_message_not_read",
                "agent.task_candidate_revision_changed",
                "agent.task_route_reason_invalid",
            ),
            timeout_seconds=5,
            audit_payload="metadata",
        ),
        TaskRouteRequest,
        TaskRouteResponse,
        route,
    )
