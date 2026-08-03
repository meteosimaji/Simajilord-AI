"""Typed declaration of an ordered, bounded high-risk action plan."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError

MAX_HIGH_RISK_PLAN_ACTIONS = 8
MIN_HIGH_RISK_PLAN_ACTIONS = 2
MIN_HIGH_RISK_PLAN_EXPIRY_SECONDS = 30
MAX_HIGH_RISK_PLAN_EXPIRY_SECONDS = 300


@dataclass(frozen=True, slots=True)
class HighRiskPlanActionRequest:
    capability: str = field(
        metadata={
            "description": ("Exact high-risk capability name loaded with capability_describe.")
        }
    )
    contract_id: str = field(
        metadata={
            "description": (
                "Opaque current contract_id returned for this capability by capability_describe."
            )
        }
    )
    arguments: Any = field(
        metadata={
            "description": (
                "Complete arguments object for this action, excluding authorization_event_id."
            )
        }
    )


@dataclass(frozen=True, slots=True)
class HighRiskPlanRequest:
    authorization_event_id: str = field(
        metadata={
            "description": (
                "Exact active event whose requester will privately confirm this whole plan."
            )
        }
    )
    actions: tuple[HighRiskPlanActionRequest, ...] = field(
        metadata={
            "description": (
                "Every action in immutable execution order. Load each distinct capability "
                "contract first; repeated actions may reuse that capability's contract."
            )
        }
    )
    max_actions: int = field(
        metadata={
            "description": (
                "Hard action ceiling. It must exactly equal the fixed action count so no "
                "unreviewed slots remain."
            )
        }
    )
    expires_in_seconds: int = 120


@dataclass(frozen=True, slots=True)
class HighRiskPlanResponse:
    authorization_event_id: str
    action_count: int
    max_actions: int
    expires_in_seconds: int
    validated: bool


def high_risk_plan_request_from_arguments(arguments: object) -> HighRiskPlanRequest:
    """Parse provider-side raw JSON into the same typed request as the endpoint."""

    if not isinstance(arguments, dict):
        raise UserError("agent.high_risk_plan_arguments_invalid")
    if set(arguments) - {
        "authorization_event_id",
        "actions",
        "max_actions",
        "expires_in_seconds",
    }:
        raise UserError("agent.high_risk_plan_arguments_invalid")
    authorization_event_id = arguments.get("authorization_event_id")
    raw_actions = arguments.get("actions")
    max_actions = arguments.get("max_actions")
    expires_in_seconds = arguments.get("expires_in_seconds", 120)
    if (
        not isinstance(authorization_event_id, str)
        or not isinstance(raw_actions, list)
        or not isinstance(max_actions, int)
        or isinstance(max_actions, bool)
        or not isinstance(expires_in_seconds, int)
        or isinstance(expires_in_seconds, bool)
    ):
        raise UserError("agent.high_risk_plan_arguments_invalid")
    actions: list[HighRiskPlanActionRequest] = []
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict) or set(raw_action) != {
            "capability",
            "contract_id",
            "arguments",
        }:
            raise UserError("agent.high_risk_plan_arguments_invalid")
        capability = raw_action.get("capability")
        contract_id = raw_action.get("contract_id")
        action_arguments = raw_action.get("arguments")
        if not isinstance(capability, str) or not isinstance(contract_id, str):
            raise UserError("agent.high_risk_plan_arguments_invalid")
        actions.append(
            HighRiskPlanActionRequest(
                capability=capability,
                contract_id=contract_id,
                arguments=action_arguments,
            )
        )
    request = HighRiskPlanRequest(
        authorization_event_id=authorization_event_id,
        actions=tuple(actions),
        max_actions=max_actions,
        expires_in_seconds=expires_in_seconds,
    )
    validate_high_risk_plan_request(request)
    return request


def validate_high_risk_plan_request(request: HighRiskPlanRequest) -> None:
    """Validate only the shape; the provider binds contracts and current authority."""

    event_id = request.authorization_event_id.strip()
    if not event_id or len(event_id) > 500:
        raise UserError("agent.high_risk_plan_authorization_invalid")
    if not MIN_HIGH_RISK_PLAN_ACTIONS <= len(request.actions) <= MAX_HIGH_RISK_PLAN_ACTIONS:
        raise UserError("agent.high_risk_plan_size_invalid")
    if request.max_actions != len(request.actions):
        raise UserError("agent.high_risk_plan_limit_mismatch")
    if not (
        MIN_HIGH_RISK_PLAN_EXPIRY_SECONDS
        <= request.expires_in_seconds
        <= MAX_HIGH_RISK_PLAN_EXPIRY_SECONDS
    ):
        raise UserError("agent.high_risk_plan_expiry_invalid")
    for action in request.actions:
        capability = action.capability.strip()
        contract_id = action.contract_id.strip()
        if not capability or len(capability) > 200:
            raise UserError("agent.high_risk_plan_capability_invalid")
        if not contract_id or len(contract_id) > 500:
            raise UserError("agent.high_risk_plan_contract_invalid")
        if not isinstance(action.arguments, dict) or not all(
            isinstance(key, str) for key in action.arguments
        ):
            raise UserError("agent.high_risk_plan_arguments_invalid")
        if "authorization_event_id" in action.arguments:
            raise UserError("agent.high_risk_plan_nested_authorization")
        try:
            json.dumps(
                action.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise UserError("agent.high_risk_plan_arguments_invalid") from exc


def build_high_risk_plan_endpoint() -> CapabilityEndpoint:
    """Build the model declaration surface; provider state performs confirmation."""

    async def declare(
        request: HighRiskPlanRequest,
        _: InvocationContext,
    ) -> HighRiskPlanResponse:
        validate_high_risk_plan_request(request)
        return HighRiskPlanResponse(
            authorization_event_id=request.authorization_event_id.strip(),
            action_count=len(request.actions),
            max_actions=request.max_actions,
            expires_in_seconds=request.expires_in_seconds,
            validated=True,
        )

    return endpoint(
        CapabilityDescriptor(
            name="turn.high_risk_plan",
            summary=(
                "Before two or more high-risk writes share one authorization event, declare "
                "the complete ordered action list, each current contract and exact arguments, "
                "the hard action ceiling, and expiry. The host privately confirms one immutable "
                "binding, stops after the first failure, and never rolls back or retries it "
                "automatically. A single direct high-risk action remains supported without this."
            ),
            risk=RiskLevel.READ,
            disclosure_class=DisclosureClass.NO_USER_CONTENT,
            keywords=(
                "bounded high risk plan",
                "multi action confirmation",
                "ordered write plan",
                "複数操作確認",
            ),
            idempotency="read",
            expected_errors=(
                "agent.high_risk_plan_authorization_invalid",
                "agent.high_risk_plan_size_invalid",
                "agent.high_risk_plan_limit_mismatch",
                "agent.high_risk_plan_expiry_invalid",
                "agent.high_risk_plan_capability_invalid",
                "agent.high_risk_plan_contract_invalid",
                "agent.high_risk_plan_arguments_invalid",
                "agent.high_risk_plan_nested_authorization",
            ),
            timeout_seconds=5,
            audit_payload="metadata",
        ),
        HighRiskPlanRequest,
        HighRiskPlanResponse,
        declare,
    )
