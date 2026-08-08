from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from simajilord.agent.actions import (
    ActionClassification,
    ActionPolicy,
    ActionReceiptService,
    ActionReceiptStore,
    ExternalEffectStatus,
)
from simajilord.agent.human import HumanCapabilityExecutor
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import (
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError


@dataclass(frozen=True, slots=True)
class AuditWriteRequest:
    channel_id: str
    outcome: str


@dataclass(frozen=True, slots=True)
class AuditWriteResponse:
    channel_id: str
    changed: bool


class RecordingInvocationJournal:
    def __init__(self) -> None:
        self.invocations: list[dict[str, object]] = []

    async def record_invocation(self, **values: object) -> None:
        self.invocations.append(dict(values))


def _context(request_id: str) -> InvocationContext:
    return InvocationContext(
        actor_id="actor",
        workspace_id="guild",
        transport="discord",
        request_id=request_id,
        tool_call_id=request_id,
        origin_resource_id="channel",
        grants=frozenset({"test.write"}),
        principal_kind="requester",
    )


def _executor(
    tmp_path: Path,
) -> tuple[
    HumanCapabilityExecutor,
    AgentToolCatalog,
    ActionReceiptStore,
    RecordingInvocationJournal,
]:
    journal = RecordingInvocationJournal()
    registry = CapabilityRegistry(journal)

    async def write(
        request: AuditWriteRequest,
        context: InvocationContext,
    ) -> AuditWriteResponse:
        if request.outcome == "reject":
            raise UserError("test.pre_dispatch_rejected")
        await context.dispatch_external_effect()
        if request.outcome == "timeout":
            raise TimeoutError("dispatch result unknown")
        return AuditWriteResponse(channel_id=request.channel_id, changed=True)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                name="test.audit_write",
                summary="Write once through the shared typed invocation path.",
                risk=RiskLevel.WRITE,
                idempotency="non_idempotent_write",
                audit_payload="metadata",
            ),
            AuditWriteRequest,
            AuditWriteResponse,
            write,
        )
    )
    store = ActionReceiptStore(tmp_path / "actions.sqlite3")
    receipts = ActionReceiptService(
        store=store,
        registry=registry,
        policies={
            "test.audit_write": ActionPolicy(
                capability="test.audit_write",
                classification=ActionClassification.NON_UNDOABLE,
            )
        },
    )
    human = HumanCapabilityExecutor(
        registry=registry,
        action_receipts=receipts,
        allowed_capabilities=("test.audit_write",),
        write_capabilities=("test.audit_write",),
    )
    catalog = AgentToolCatalog(
        registry,
        ("test.audit_write",),
        required_grants={"test.audit_write": "test.write"},
        write_capabilities=("test.audit_write",),
        action_receipts=receipts,
    )
    return human, catalog, store, journal


@pytest.mark.asyncio
async def test_human_and_agent_paths_share_one_invocation_effect_and_receipt(
    tmp_path: Path,
) -> None:
    human, catalog, store, journal = _executor(tmp_path)
    human_response = await human.invoke(
        "test.audit_write",
        AuditWriteRequest(channel_id="channel", outcome="success"),
        _context("interaction-1"),
    )
    agent_response = await catalog.invoke(
        namespace="simajilord",
        tool_name="test_audit_write",
        arguments={"channel_id": "channel", "outcome": "success"},
        context=replace(_context("agent-1"), transport="agent"),
        max_output_characters=4_000,
    )

    assert human_response == AuditWriteResponse(channel_id="channel", changed=True)
    assert '"changed":true' in agent_response.text
    assert [item["capability_name"] for item in journal.invocations] == [
        "test.audit_write",
        "test.audit_write",
    ]
    effects = await store.external_effects(limit=10)
    assert len(effects) == 2
    assert {item.status for item in effects} == {ExternalEffectStatus.CONFIRMED}
    with sqlite3.connect(store.path) as connection:
        actions = connection.execute(
            "SELECT request_id, capability FROM agent_actions ORDER BY created_at"
        ).fetchall()
    assert actions == [
        ("interaction-1", "test.audit_write"),
        ("agent-1", "test.audit_write"),
    ]


@pytest.mark.asyncio
async def test_human_pre_dispatch_failure_is_rejected_not_unknown(
    tmp_path: Path,
) -> None:
    human, _catalog, store, journal = _executor(tmp_path)
    with pytest.raises(UserError, match=r"test\.pre_dispatch_rejected"):
        await human.invoke(
            "test.audit_write",
            AuditWriteRequest(channel_id="channel", outcome="reject"),
            _context("interaction-reject"),
        )

    effects = await store.external_effects(limit=10)
    assert [item.status for item in effects] == [ExternalEffectStatus.REJECTED]
    assert len(journal.invocations) == 1
    assert isinstance(journal.invocations[0]["error"], UserError)


@pytest.mark.asyncio
async def test_human_post_dispatch_timeout_is_unknown_and_not_replayed(
    tmp_path: Path,
) -> None:
    human, _catalog, store, journal = _executor(tmp_path)
    with pytest.raises(TimeoutError, match="unknown"):
        await human.invoke(
            "test.audit_write",
            AuditWriteRequest(channel_id="channel", outcome="timeout"),
            _context("interaction-timeout"),
        )

    effects = await store.external_effects(limit=10)
    assert [item.status for item in effects] == [ExternalEffectStatus.UNKNOWN]
    assert len(journal.invocations) == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agent_actions").fetchone() == (0,)
