from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.agent import (
    AGENT_HIGH_RISK_CAPABILITIES,
    AgentHighRiskConfirmation,
    AgentHighRiskPlanActionStatus,
    AgentHighRiskPlanStatus,
    AgentHighRiskPlanStatusUpdate,
)
from simajilord.agent.providers.codex import (
    CodexAppServerProvider,
    _bind_high_risk_authorization,
    _capability_discovery_tool_failure,
    _confirm_high_risk_plan,
    _dispatch_high_risk_plan_action,
    _high_risk_plan_status_update,
    _record_high_risk_plan_action_outcome,
    _stop_high_risk_plan,
    _ToolTurnBudget,
)
from simajilord.agent.tools import AgentToolCatalog, AgentToolOutput
from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.integrations.discord.cogs import (
    AgentCog,
    _high_risk_plan_status_embed,
    _high_risk_private_embeds,
    _high_risk_public_embed,
)


@dataclass(frozen=True, slots=True)
class _CreateRoleRequest:
    name: str


@dataclass(frozen=True, slots=True)
class _CreateRoleResponse:
    role_id: str


@dataclass(frozen=True, slots=True)
class _AssignRoleRequest:
    user_id: str
    role_id: str


@dataclass(frozen=True, slots=True)
class _AssignRoleResponse:
    user_id: str
    role_id: str
    assigned: bool


def _context(*, request_id: str = "event") -> InvocationContext:
    capabilities = frozenset({"discord.create_role", "discord.assign_role"})
    return InvocationContext(
        actor_id="requester",
        workspace_id="guild",
        transport="agent",
        request_id=request_id,
        grants=frozenset({"role-write"}),
        approvals=capabilities,
        origin_resource_id="channel",
        active_message_id="message",
        active_message_edited_at="2026-08-03T00:00:00+00:00",
        requester_principal_id="requester",
        high_risk_authorization_mode="bound_once",
    )


def _catalog() -> AgentToolCatalog:
    registry = CapabilityRegistry()

    async def create_role(
        request: _CreateRoleRequest,
        _: InvocationContext,
    ) -> _CreateRoleResponse:
        return _CreateRoleResponse(role_id=f"role:{request.name}")

    async def assign_role(
        request: _AssignRoleRequest,
        _: InvocationContext,
    ) -> _AssignRoleResponse:
        return _AssignRoleResponse(
            user_id=request.user_id,
            role_id=request.role_id,
            assigned=True,
        )

    for name, request_type, response_type, invoke in (
        (
            "discord.create_role",
            _CreateRoleRequest,
            _CreateRoleResponse,
            create_role,
        ),
        (
            "discord.assign_role",
            _AssignRoleRequest,
            _AssignRoleResponse,
            assign_role,
        ),
    ):
        registry.register(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=f"Test {name}.",
                    risk=RiskLevel.WRITE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                ),
                request_type,
                response_type,
                invoke,
            )
        )
    names = ("discord.create_role", "discord.assign_role")
    return AgentToolCatalog(
        registry,
        names,
        required_grants={name: "role-write" for name in names},
        eager_capabilities=(),
        write_capabilities=names,
    )


async def _contracts(
    catalog: AgentToolCatalog,
    context: InvocationContext,
) -> dict[str, str]:
    search = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "create one role and assign it to members"},
        context=context,
        max_output_characters=8_000,
    )
    catalog_id = json.loads(search.text)["catalog_id"]
    contracts: dict[str, str] = {}
    for name in ("discord.create_role", "discord.assign_role"):
        described = await catalog.invoke(
            namespace="simajilord",
            tool_name="capability_describe",
            arguments={"catalog_id": catalog_id, "name": name},
            context=context,
            max_output_characters=8_000,
        )
        contracts[name] = json.loads(described.text)["contract_id"]
    return contracts


def _plan_arguments(
    contracts: dict[str, str],
    *,
    authorization_event_id: str,
    role_name: str = "operators",
) -> dict[str, object]:
    role_reference = {"$plan_result": {"action": 1, "field": "role_id"}}
    return {
        "authorization_event_id": authorization_event_id,
        "actions": [
            {
                "capability": "discord.create_role",
                "contract_id": contracts["discord.create_role"],
                "arguments": {"name": role_name},
            },
            {
                "capability": "discord.assign_role",
                "contract_id": contracts["discord.assign_role"],
                "arguments": {"user_id": "A", "role_id": role_reference},
            },
            {
                "capability": "discord.assign_role",
                "contract_id": contracts["discord.assign_role"],
                "arguments": {"user_id": "B", "role_id": role_reference},
            },
        ],
        "max_actions": 3,
        "expires_in_seconds": 120,
    }


def _budget(
    context: InvocationContext,
    contracts: dict[str, str],
    *,
    authorization_event_id: str,
    confirmations: list[AgentHighRiskConfirmation],
    statuses: list[AgentHighRiskPlanStatusUpdate],
) -> _ToolTurnBudget:
    async def confirm(proposal: AgentHighRiskConfirmation) -> bool:
        confirmations.append(proposal)
        return True

    async def status(update: AgentHighRiskPlanStatusUpdate) -> None:
        statuses.append(update)

    return _ToolTurnBudget(
        context=context,
        calls_remaining=20,
        output_characters_remaining=20_000,
        on_progress=None,
        required_message_id=None,
        on_high_risk_confirmation=confirm,
        on_high_risk_plan_status=status,
        authorization_contexts={authorization_event_id: context},
        authorization_message_ids={authorization_event_id: "message"},
        read_authorization_event_ids={authorization_event_id},
        capability_discovery_contracts=dict(contracts),
        capability_discovery_name="discord.assign_role",
        capability_discovery_contract_id=contracts["discord.assign_role"],
    )


@pytest.mark.asyncio
async def test_bounded_role_plan_binds_order_target_limit_and_partial_failure() -> None:
    catalog = _catalog()
    context = _context()
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth",
        confirmations=confirmations,
        statuses=statuses,
    )
    plan, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(contracts, authorization_event_id="auth"),
        tools=catalog,
    )

    assert failure is None
    assert plan is not None
    assert plan.max_actions == 3
    assert len(confirmations) == 1
    assert confirmations[0].confirmation_kind == "high_risk_plan"
    assert [action.capability for action in confirmations[0].plan_actions] == [
        "discord.create_role",
        "discord.assign_role",
        "discord.assign_role",
    ]
    private_review = json.dumps(
        [
            [field.value for field in action.presentation.review_fields]
            for action in confirmations[0].plan_actions
        ]
    )
    assert "$plan_result" in private_review
    assert confirmations[0].expires_at is not None
    public_embed = _high_risk_public_embed(
        confirmations[0],
        expires_at=confirmations[0].expires_at,
    )
    private_embeds = _high_risk_private_embeds(
        confirmations[0],
        expires_at=confirmations[0].expires_at,
    )
    assert len(private_embeds) == 4
    public_payload = json.dumps(public_embed.to_dict(), ensure_ascii=False)
    private_payload = json.dumps(
        [embed.to_dict() for embed in private_embeds],
        ensure_ascii=False,
    )
    assert "operators" not in public_payload
    assert 'user_id: "A"' not in public_payload
    assert "operators" in private_payload
    private_field_text = "\n".join(
        field.value
        for action in confirmations[0].plan_actions
        for field in action.presentation.review_fields
    )
    assert 'user_id: "A"' in private_field_text
    assert 'user_id: "B"' in private_field_text

    out_of_order = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth",
        capability_name="discord.assign_role",
        arguments={"user_id": "A", "role_id": "role-1"},
        context=context,
        tool_call_id="wrong-order",
    )
    assert out_of_order is not None
    assert out_of_order[0] == "agent.high_risk_plan_order_changed"

    changed_target = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth",
        capability_name="discord.create_role",
        arguments={"name": "different"},
        context=context,
        tool_call_id="changed-target",
    )
    assert changed_target is not None
    assert changed_target[0] == "agent.high_risk_plan_action_changed"

    changed_later_target = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth",
        capability_name="discord.assign_role",
        arguments={"user_id": "C", "role_id": "role-1"},
        context=context,
        tool_call_id="changed-later-target",
    )
    assert changed_later_target is not None
    assert changed_later_target[0] == "agent.high_risk_plan_action_changed"

    changed_plan, changed_failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(
            contracts,
            authorization_event_id="auth",
            role_name="different",
        ),
        tools=catalog,
    )
    assert changed_plan is None
    assert changed_failure is not None
    assert changed_failure[0] == "agent.high_risk_plan_changed"
    assert len(confirmations) == 1

    assert (
        _bind_high_risk_authorization(
            budget,
            authorization_event_id="auth",
            capability_name="discord.create_role",
            arguments={"name": "operators"},
            context=context,
            tool_call_id="call-1",
        )
        is None
    )
    _dispatch_high_risk_plan_action(budget, "call-1")
    first = await _record_high_risk_plan_action_outcome(
        budget,
        tool_call_id="call-1",
        succeeded=True,
        error_code=None,
        action_receipt_id="act_1",
        external_effect_id="eff_1",
        tool_output='{"role_id":"role-1"}',
    )
    assert first is not None
    assert first.actions[0].status is AgentHighRiskPlanActionStatus.SUCCEEDED
    assert first.actions[0].action_receipt_id == "act_1"
    assert first.actions[0].external_effect_id == "eff_1"

    assert (
        _bind_high_risk_authorization(
            budget,
            authorization_event_id="auth",
            capability_name="discord.assign_role",
            arguments={"user_id": "A", "role_id": "role-1"},
            context=context,
            tool_call_id="call-2",
        )
        is None
    )
    _dispatch_high_risk_plan_action(budget, "call-2")
    stopped = await _record_high_risk_plan_action_outcome(
        budget,
        tool_call_id="call-2",
        succeeded=False,
        error_code="discord.member_not_found",
        external_effect_id="eff_2",
    )
    assert stopped is not None
    assert stopped.status is AgentHighRiskPlanStatus.STOPPED
    assert [action.status for action in stopped.actions] == [
        AgentHighRiskPlanActionStatus.SUCCEEDED,
        AgentHighRiskPlanActionStatus.FAILED,
        AgentHighRiskPlanActionStatus.NOT_RUN,
    ]
    assert stopped.actions[1].error_code == "discord.member_not_found"
    assert stopped.actions[2].error_code == ("agent.high_risk_plan_not_run_after_failure")
    status_payload = json.dumps(
        _high_risk_plan_status_embed(stopped).to_dict(),
        ensure_ascii=False,
    )
    assert "succeeded" in status_payload
    assert "failed" in status_payload
    assert "not_run" in status_payload
    message = Mock(spec=discord.Message)
    message.guild = SimpleNamespace(id=10)
    message.edit = AsyncMock()
    runtime = SimpleNamespace(journal=SimpleNamespace(append=AsyncMock()))
    cog = AgentCog(SimpleNamespace(), runtime)
    cog._high_risk_plan_messages[stopped.plan_id] = message
    await cog._update_high_risk_plan_status(stopped)
    runtime.journal.append.assert_awaited_once()
    journal_actions = runtime.journal.append.await_args.kwargs["payload"]["actions"]
    assert [action["status"] for action in journal_actions] == [
        "succeeded",
        "failed",
        "not_run",
    ]
    message.edit.assert_awaited_once()
    assert stopped.plan_id not in cog._high_risk_plan_messages
    blocked_third = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth",
        capability_name="discord.assign_role",
        arguments={"user_id": "B", "role_id": "role-1"},
        context=context,
        tool_call_id="call-3",
    )
    assert blocked_third is not None
    assert blocked_third[0] == "agent.high_risk_plan_stopped"
    assert statuses


@pytest.mark.asyncio
async def test_completed_three_action_plan_rejects_unreviewed_fourth_action() -> None:
    catalog = _catalog()
    context = _context(request_id="event-2")
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth-2",
        confirmations=confirmations,
        statuses=statuses,
    )
    plan, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(contracts, authorization_event_id="auth-2"),
        tools=catalog,
    )
    assert failure is None
    assert plan is not None

    calls = (
        ("discord.create_role", {"name": "operators"}, '{"role_id":"role-2"}'),
        (
            "discord.assign_role",
            {"user_id": "A", "role_id": "role-2"},
            '{"assigned":true}',
        ),
        (
            "discord.assign_role",
            {"user_id": "B", "role_id": "role-2"},
            '{"assigned":true}',
        ),
    )
    completed = None
    for position, (capability, arguments, output) in enumerate(calls, start=1):
        call_id = f"success-{position}"
        assert (
            _bind_high_risk_authorization(
                budget,
                authorization_event_id="auth-2",
                capability_name=capability,
                arguments=arguments,
                context=context,
                tool_call_id=call_id,
            )
            is None
        )
        _dispatch_high_risk_plan_action(budget, call_id)
        completed = await _record_high_risk_plan_action_outcome(
            budget,
            tool_call_id=call_id,
            succeeded=True,
            error_code=None,
            action_receipt_id=f"act_{position}",
            external_effect_id=f"eff_{position}",
            tool_output=output,
        )
    assert completed is not None
    assert completed.status is AgentHighRiskPlanStatus.COMPLETED
    assert all(
        action.status is AgentHighRiskPlanActionStatus.SUCCEEDED for action in completed.actions
    )

    fourth = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth-2",
        capability_name="discord.assign_role",
        arguments={"user_id": "C", "role_id": "role-2"},
        context=context,
        tool_call_id="success-4",
    )
    assert fourth is not None
    assert fourth[0] == "agent.high_risk_plan_exhausted"


@pytest.mark.asyncio
async def test_revision_stop_keeps_inflight_action_until_its_receipt_is_recorded() -> None:
    catalog = _catalog()
    context = _context(request_id="event-revision-race")
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth-revision-race",
        confirmations=confirmations,
        statuses=statuses,
    )
    plan, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(
            contracts,
            authorization_event_id="auth-revision-race",
        ),
        tools=catalog,
    )
    assert failure is None
    assert plan is not None
    assert (
        _bind_high_risk_authorization(
            budget,
            authorization_event_id="auth-revision-race",
            capability_name="discord.create_role",
            arguments={"name": "operators"},
            context=context,
            tool_call_id="inflight-call",
        )
        is None
    )
    _dispatch_high_risk_plan_action(budget, "inflight-call")
    _stop_high_risk_plan(
        plan,
        status=AgentHighRiskPlanStatus.STOPPED,
        error_code="agent.high_risk_plan_authorization_revision_changed",
    )

    message = Mock(spec=discord.Message)
    message.guild = SimpleNamespace(id=10)
    message.edit = AsyncMock()
    runtime = SimpleNamespace(journal=SimpleNamespace(append=AsyncMock()))
    cog = AgentCog(SimpleNamespace(), runtime)
    cog._high_risk_plan_messages[plan.plan_id] = message
    await cog._update_high_risk_plan_status(_high_risk_plan_status_update(plan))
    assert plan.plan_id in cog._high_risk_plan_messages

    settled = await _record_high_risk_plan_action_outcome(
        budget,
        tool_call_id="inflight-call",
        succeeded=True,
        error_code=None,
        action_receipt_id="act_after_edit",
        external_effect_id="eff_after_edit",
        tool_output='{"role_id":"role-after-edit"}',
    )
    assert settled is not None
    assert settled.status is AgentHighRiskPlanStatus.STOPPED
    assert [action.status for action in settled.actions] == [
        AgentHighRiskPlanActionStatus.SUCCEEDED,
        AgentHighRiskPlanActionStatus.NOT_RUN,
        AgentHighRiskPlanActionStatus.NOT_RUN,
    ]
    assert settled.actions[0].action_receipt_id == "act_after_edit"
    assert settled.actions[0].external_effect_id == "eff_after_edit"
    await cog._update_high_risk_plan_status(settled)
    assert plan.plan_id not in cog._high_risk_plan_messages


@pytest.mark.asyncio
async def test_predispatch_failure_consumes_slot_and_stops_without_retry() -> None:
    catalog = _catalog()
    context = _context(request_id="event-predispatch-failure")
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth-predispatch-failure",
        confirmations=confirmations,
        statuses=statuses,
    )
    plan, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(
            contracts,
            authorization_event_id="auth-predispatch-failure",
        ),
        tools=catalog,
    )
    assert failure is None
    assert plan is not None
    assert (
        _bind_high_risk_authorization(
            budget,
            authorization_event_id="auth-predispatch-failure",
            capability_name="discord.create_role",
            arguments={"name": "operators"},
            context=context,
            tool_call_id="predispatch-call",
        )
        is None
    )

    stopped = await _record_high_risk_plan_action_outcome(
        budget,
        tool_call_id="predispatch-call",
        succeeded=False,
        error_code="discord.manage_roles_required",
    )
    assert stopped is not None
    assert stopped.status is AgentHighRiskPlanStatus.STOPPED
    assert [action.status for action in stopped.actions] == [
        AgentHighRiskPlanActionStatus.FAILED,
        AgentHighRiskPlanActionStatus.NOT_RUN,
        AgentHighRiskPlanActionStatus.NOT_RUN,
    ]
    assert stopped.actions[0].tool_call_id == "predispatch-call"
    assert stopped.actions[0].action_receipt_id is None
    assert stopped.actions[0].external_effect_id is None
    retry = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth-predispatch-failure",
        capability_name="discord.create_role",
        arguments={"name": "operators"},
        context=context,
        tool_call_id="forbidden-retry",
    )
    assert retry is not None
    assert retry[0] == "agent.high_risk_plan_stopped"


@pytest.mark.asyncio
async def test_provider_keeps_host_receipt_and_effect_ids_outside_bounded_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    context = _context(request_id="event-metadata")
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth-metadata",
        confirmations=confirmations,
        statuses=statuses,
    )
    plan, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(
            contracts,
            authorization_event_id="auth-metadata",
        ),
        tools=catalog,
    )
    assert failure is None
    assert plan is not None
    budget.evidence_plan_recorded = True

    provider = CodexAppServerProvider(
        executable="codex",
        model="test",
        workspace_dir=tmp_path / "agent",
        idle_timeout_seconds=10,
        reasoning_effort="low",
        tools=catalog,
    )
    provider._active_tool_budgets["thread"] = budget
    response = AsyncMock()
    monkeypatch.setattr(provider, "_tool_response", response)

    async def invoke_with_body_free_ids(**kwargs: object) -> AgentToolOutput:
        before_invoke = kwargs.get("before_invoke")
        assert callable(before_invoke)
        before_invoke()
        return AgentToolOutput(
            text='{"role_id":"role-actual"}',
            action_receipt_id="act_host_only",
            external_effect_id="eff_host_only",
        )

    monkeypatch.setattr(catalog, "invoke", invoke_with_body_free_ids)
    await provider._handle_dynamic_tool(
        1,
        {
            "namespace": "simajilord",
            "tool": "capability_invoke",
            "callId": "plan-action-1",
            "threadId": "thread",
            "arguments": {
                "name": "discord.create_role",
                "contract_id": contracts["discord.create_role"],
                "authorization_event_id": "auth-metadata",
                "arguments": {"name": "operators"},
            },
        },
    )

    assert response.await_args.kwargs["success"] is True
    terminal = statuses[-1]
    assert terminal.actions[0].status is AgentHighRiskPlanActionStatus.SUCCEEDED
    assert terminal.actions[0].action_receipt_id == "act_host_only"
    assert terminal.actions[0].external_effect_id == "eff_host_only"


@pytest.mark.asyncio
async def test_plan_rejects_forward_result_reference_and_expires_fail_closed() -> None:
    catalog = _catalog()
    context = _context(request_id="event-invalid-reference")
    contracts = await _contracts(catalog, context)
    confirmations: list[AgentHighRiskConfirmation] = []
    statuses: list[AgentHighRiskPlanStatusUpdate] = []
    budget = _budget(
        context,
        contracts,
        authorization_event_id="auth-invalid-reference",
        confirmations=confirmations,
        statuses=statuses,
    )
    single = _plan_arguments(
        contracts,
        authorization_event_id="auth-invalid-reference",
    )
    single_actions = single["actions"]
    assert isinstance(single_actions, list)
    single["actions"] = single_actions[:1]
    single["max_actions"] = 1
    rejected_single, single_rejection = await _confirm_high_risk_plan(
        budget,
        arguments=single,
        tools=catalog,
    )
    assert rejected_single is None
    assert single_rejection is not None
    assert single_rejection[0] == "agent.high_risk_plan_size_invalid"

    invalid = _plan_arguments(
        contracts,
        authorization_event_id="auth-invalid-reference",
    )
    actions = invalid["actions"]
    assert isinstance(actions, list)
    second = actions[1]
    assert isinstance(second, dict)
    second["arguments"] = {
        "user_id": "A",
        "role_id": {"$plan_result": {"action": 3, "field": "role_id"}},
    }
    rejected, rejection = await _confirm_high_risk_plan(
        budget,
        arguments=invalid,
        tools=catalog,
    )
    assert rejected is None
    assert rejection is not None
    assert rejection[0] == "agent.high_risk_confirmation_unreviewable"
    assert confirmations == []

    valid, failure = await _confirm_high_risk_plan(
        budget,
        arguments=_plan_arguments(
            contracts,
            authorization_event_id="auth-invalid-reference",
        ),
        tools=catalog,
    )
    assert failure is None
    assert valid is not None
    valid.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    expired = _bind_high_risk_authorization(
        budget,
        authorization_event_id="auth-invalid-reference",
        capability_name="discord.create_role",
        arguments={"name": "operators"},
        context=context,
        tool_call_id="expired-action",
    )
    assert expired is not None
    assert expired[0] == "agent.high_risk_plan_expired"
    assert valid.status is AgentHighRiskPlanStatus.EXPIRED
    assert all(action.status is AgentHighRiskPlanActionStatus.NOT_RUN for action in valid.actions)


def test_discovery_allows_only_bounded_distinct_high_risk_plan_contracts() -> None:
    context = _context()
    budget = _ToolTurnBudget(
        context=context,
        calls_remaining=5,
        output_characters_remaining=5_000,
        on_progress=None,
        required_message_id=None,
        evidence_plan_recorded=True,
        capability_discovery_catalog_id="catalog",
        capability_discovery_name="discord.create_role",
        capability_discovery_contract_id="contract-create",
        capability_discovery_contracts={"discord.create_role": "contract-create"},
    )

    assert (
        _capability_discovery_tool_failure(
            budget,
            tool_name="capability_describe",
            arguments={
                "catalog_id": "catalog",
                "name": "discord.assign_role",
            },
            capability_name=None,
        )
        is None
    )
    assert (
        _capability_discovery_tool_failure(
            budget,
            tool_name="capability_describe",
            arguments={
                "catalog_id": "catalog",
                "name": "turn.high_risk_plan",
            },
            capability_name=None,
        )
        is None
    )
    duplicate = _capability_discovery_tool_failure(
        budget,
        tool_name="capability_describe",
        arguments={
            "catalog_id": "catalog",
            "name": "discord.create_role",
        },
        capability_name=None,
    )
    assert duplicate is not None
    assert duplicate[0] == "agent.capability_contract_pending"

    budget.capability_discovery_name = "turn.high_risk_plan"
    budget.capability_discovery_contract_id = "contract-plan"
    assert (
        _capability_discovery_tool_failure(
            budget,
            tool_name="capability_describe",
            arguments={
                "catalog_id": "catalog",
                "name": "discord.assign_role",
            },
            capability_name=None,
        )
        is None
    )

    eight_capabilities = tuple(sorted(AGENT_HIGH_RISK_CAPABILITIES))[:8]
    budget.capability_discovery_name = eight_capabilities[-1]
    budget.capability_discovery_contracts = {
        name: f"contract-{index}" for index, name in enumerate(eight_capabilities)
    }
    assert (
        _capability_discovery_tool_failure(
            budget,
            tool_name="capability_describe",
            arguments={
                "catalog_id": "catalog",
                "name": "turn.high_risk_plan",
            },
            capability_name=None,
        )
        is None
    )
