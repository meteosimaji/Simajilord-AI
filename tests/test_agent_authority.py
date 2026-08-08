from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simajilord.agent import (
    AGENT_AUTHORITY_MANAGE_GRANT,
    AgentToolError,
    AuthorityRequestCreateRequest,
    CapabilityLeaseCreateRequest,
    CapabilityLeaseRevokeRequest,
    CapabilityLeaseStore,
)
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityRegistry,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError


@dataclass(frozen=True, slots=True)
class ScopedRequest:
    channel_id: str = ""
    repository: str = ""


@dataclass(frozen=True, slots=True)
class ScopedResponse:
    ok: bool


def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()

    async def read(_: ScopedRequest, __: InvocationContext) -> ScopedResponse:
        return ScopedResponse(ok=True)

    for name in ("demo.channel", "demo.repo", "demo.write"):
        registry.register(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=f"Use {name} within an exact lease scope.",
                    risk=RiskLevel.WRITE if name == "demo.write" else RiskLevel.READ,
                    approval=(
                        ApprovalMode.WHEN_REQUESTED if name == "demo.write" else ApprovalMode.NEVER
                    ),
                    disclosure_class=(
                        None if name == "demo.write" else DisclosureClass.NO_USER_CONTENT
                    ),
                    requires_workspace=True,
                ),
                ScopedRequest,
                ScopedResponse,
                read,
            )
        )
    return registry


def _context(
    *,
    actor_id: str = "actor-1",
    workspace_id: str = "guild-1",
    task_id: str | None = "task-1",
    grants: frozenset[str] = frozenset(
        {AGENT_AUTHORITY_MANAGE_GRANT, "demo.channel.use", "demo.repo.use"}
    ),
) -> InvocationContext:
    return InvocationContext(
        actor_id=actor_id,
        workspace_id=workspace_id,
        transport="agent",
        request_id="request-1",
        grants=grants,
        agent_task_id=task_id,
        principal_kind="requester",
        agent_trigger="mention",
        allowed_capabilities=frozenset({"demo.channel", "demo.repo"}),
    )


def _store(path: Path, registry: CapabilityRegistry | None = None) -> CapabilityLeaseStore:
    return CapabilityLeaseStore(
        path,
        registry=registry or _registry(),
        configured_capabilities=("demo.channel", "demo.repo", "demo.write"),
        required_grants={
            "demo.channel": "demo.channel.use",
            "demo.repo": "demo.repo.use",
            "demo.write": "demo.write.use",
        },
    )


def _expiry(*, minutes: int = 10) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def test_lease_actor_task_workspace_and_channel_scope_matrix(tmp_path: Path) -> None:
    store = _store(tmp_path / "leases.sqlite3")
    manager = _context()
    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-2",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            max_uses=1,
            reason="Allow one operation in the reviewed channel.",
            agent_task_id="task-2",
            target_kind="channel",
            target_id="channel-7",
        ),
        manager,
    )
    grantee = replace(
        _context(actor_id="actor-2", task_id="task-2", grants=frozenset()),
        request_id="grantee-request",
    )
    bindings = store.resolve_bindings(grantee, static_grants=frozenset())

    assert [(item.capability, item.lease_id) for item in bindings] == [
        ("demo.channel", created.lease_id)
    ]
    assert not store.resolve_bindings(
        replace(grantee, actor_id="actor-3"), static_grants=frozenset()
    )
    assert not store.resolve_bindings(
        replace(grantee, agent_task_id="task-other"), static_grants=frozenset()
    )
    assert not store.resolve_bindings(
        replace(grantee, workspace_id="guild-other"), static_grants=frozenset()
    )
    with pytest.raises(UserError, match=r"authority\.target_scope_mismatch"):
        store.consume_binding(bindings[0], ScopedRequest(channel_id="channel-8"), grantee)

    store.consume_binding(bindings[0], ScopedRequest(channel_id="channel-7"), grantee)
    with pytest.raises(UserError, match=r"authority\.lease_unavailable"):
        store.consume_binding(bindings[0], ScopedRequest(channel_id="channel-7"), grantee)


def test_role_and_repository_target_lease(tmp_path: Path) -> None:
    store = _store(tmp_path / "leases.sqlite3")
    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="role",
            grantee_id="role-9",
            capabilities=("demo.repo",),
            expires_at_iso=_expiry(),
            max_uses=2,
            reason="Permit this role to inspect one repository.",
            target_kind="repository",
            target_id="owner/repo",
        ),
        _context(),
    )
    role_context = replace(
        _context(actor_id="actor-9", grants=frozenset()),
        principal_role_ids=("role-9",),
    )
    bindings = store.resolve_bindings(role_context, static_grants=frozenset())
    assert bindings[0].lease_id == created.lease_id
    store.consume_binding(bindings[0], ScopedRequest(repository="owner/repo"), role_context)
    with pytest.raises(UserError, match=r"authority\.target_scope_mismatch"):
        store.consume_binding(bindings[0], ScopedRequest(repository="owner/other"), role_context)

    service_created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="service",
            grantee_id="service-1",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            reason="Permit one bounded service principal.",
        ),
        _context(),
    )
    service_context = replace(
        _context(actor_id="service-1", grants=frozenset()),
        principal_kind="service",
    )
    service_bindings = store.resolve_bindings(service_context, static_grants=frozenset())
    assert service_bindings[0].lease_id == service_created.lease_id


def test_lease_ceiling_autonomy_expiry_revoke_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    registry = _registry()
    store = _store(path, registry)
    manager = _context()
    with pytest.raises(UserError, match=r"authority\.delegator_ceiling_exceeded"):
        store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="actor",
                grantee_id="actor-2",
                capabilities=("demo.repo",),
                expires_at_iso=_expiry(),
                reason="Missing delegator authority.",
            ),
            replace(manager, grants=frozenset({AGENT_AUTHORITY_MANAGE_GRANT})),
        )
    with pytest.raises(UserError, match=r"authority\.global_ceiling_exceeded"):
        store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="actor",
                grantee_id="actor-2",
                capabilities=("demo.repo",),
                expires_at_iso=_expiry(),
                reason="Disabled by the configured turn ceiling.",
            ),
            replace(manager, allowed_capabilities=frozenset({"demo.channel"})),
        )
    disabled_store = CapabilityLeaseStore(
        tmp_path / "disabled.sqlite3",
        registry=registry,
        configured_capabilities=("demo.channel",),
        required_grants={"demo.channel": "demo.channel.use"},
    )
    with pytest.raises(UserError, match=r"authority\.global_ceiling_exceeded"):
        disabled_store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="actor",
                grantee_id="actor-2",
                capabilities=("demo.repo",),
                expires_at_iso=_expiry(),
                reason="The feature is globally disabled.",
            ),
            manager,
        )
    with pytest.raises(UserError, match=r"authority\.global_ceiling_exceeded"):
        store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="actor",
                grantee_id="actor-2",
                grants=(AGENT_AUTHORITY_MANAGE_GRANT,),
                expires_at_iso=_expiry(),
                reason="Management authority is not transitive.",
            ),
            manager,
        )
    with pytest.raises(UserError, match=r"authority\.expiry_invalid"):
        store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="actor",
                grantee_id="actor-2",
                capabilities=("demo.channel",),
                expires_at_iso=_expiry(minutes=31 * 24 * 60),
                reason="Too long.",
            ),
            manager,
        )
    autonomous = replace(manager, agent_trigger="autonomous", principal_kind="service")
    with pytest.raises(UserError, match=r"authority\.autonomous_management_forbidden"):
        store.create_lease(
            CapabilityLeaseCreateRequest(
                grantee_kind="service",
                grantee_id="service-1",
                capabilities=("demo.channel",),
                expires_at_iso=_expiry(),
                reason="Autonomy cannot grant itself authority.",
            ),
            autonomous,
        )
    with pytest.raises(UserError, match=r"authority\.autonomous_request_forbidden"):
        store.create_authority_request(
            AuthorityRequestCreateRequest(
                capability="demo.channel",
                reason="Autonomy cannot request an undisclosed schema.",
            ),
            autonomous,
        )

    expiring = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-expired",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            reason="Persist an expiry decision.",
        ),
        manager,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE capability_leases SET expires_at = ? WHERE lease_id = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), expiring.lease_id),
        )
    expired_context = _context(actor_id="actor-expired", grants=frozenset())
    assert store.resolve_bindings(expired_context, static_grants=frozenset()) == ()

    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-2",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            max_uses=2,
            reason="Revoke and persist this decision.",
        ),
        manager,
    )
    revoked = store.revoke_lease(
        CapabilityLeaseRevokeRequest(lease_id=created.lease_id, expected_revision=1),
        manager,
    )
    assert revoked.changed and revoked.revision == 2
    restarted = _store(path, registry)
    grantee = _context(actor_id="actor-2", grants=frozenset())
    assert restarted.resolve_bindings(grantee, static_grants=frozenset()) == ()


def test_concurrent_lease_use_is_atomic_and_schema_is_body_free(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    store = _store(path)
    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-2",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            max_uses=1,
            reason="Exactly one concurrent use.",
        ),
        _context(),
    )
    grantee = _context(actor_id="actor-2", grants=frozenset())
    binding = store.resolve_bindings(grantee, static_grants=frozenset())[0]

    def consume() -> str:
        try:
            store.consume_binding(binding, ScopedRequest(), grantee)
        except UserError as exc:
            return exc.code
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(lambda _: consume(), range(2)))
    assert results == ["authority.lease_unavailable", "ok"]

    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for table in ("capability_leases", "authority_requests")
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        uses = connection.execute(
            "SELECT uses FROM capability_leases WHERE lease_id = ?", (created.lease_id,)
        ).fetchone()
    assert uses == (1,)
    assert not {"body", "message", "arguments", "secret", "token"} & columns


@pytest.mark.asyncio
async def test_lease_changes_catalog_binding_and_stales_old_discovery(tmp_path: Path) -> None:
    registry = _registry()
    store = _store(tmp_path / "leases.sqlite3", registry)
    catalog = AgentToolCatalog(
        registry,
        ("demo.channel",),
        required_grants={"demo.channel": "demo.channel.use"},
        eager_capabilities=(),
    )
    manager = _context()
    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-2",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            max_uses=2,
            reason="Reissue the catalog for this actor.",
        ),
        manager,
    )
    bare = replace(
        _context(actor_id="actor-2", grants=frozenset()),
        allowed_capabilities=frozenset({"demo.channel"}),
    )
    bare_search = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "use channel lease"},
        context=bare,
        max_output_characters=4_000,
    )
    assert json.loads(bare_search.text)["matches"] == []
    bindings = store.resolve_bindings(bare, static_grants=frozenset())
    leased = replace(
        bare,
        capability_lease_bindings=tuple(
            (item.capability, item.lease_id, item.revision) for item in bindings
        ),
    )
    assert catalog.dynamic_specs(leased)
    search = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "use channel lease"},
        context=leased,
        max_output_characters=4_000,
    )
    old_catalog_id = json.loads(search.text)["catalog_id"]
    store.revoke_lease(
        CapabilityLeaseRevokeRequest(lease_id=created.lease_id, expected_revision=1),
        manager,
    )
    refreshed = replace(leased, capability_lease_bindings=())
    with pytest.raises(AgentToolError, match=r"changed|stale"):
        await catalog.invoke(
            namespace="simajilord",
            tool_name="capability_describe",
            arguments={"catalog_id": old_catalog_id, "name": "demo.channel"},
            context=refreshed,
            max_output_characters=4_000,
        )


@pytest.mark.asyncio
async def test_lease_does_not_bypass_write_approval() -> None:
    registry = _registry()
    catalog = AgentToolCatalog(
        registry,
        ("demo.write",),
        required_grants={"demo.write": "demo.write.use"},
        eager_capabilities=(),
        write_capabilities=("demo.write",),
    )
    binding = (("demo.write", "lease_" + "a" * 32, 1),)
    unapproved = replace(
        _context(grants=frozenset()),
        allowed_capabilities=frozenset({"demo.write"}),
        capability_lease_bindings=binding,
    )
    approved = replace(unapproved, approvals=frozenset({"demo.write"}))

    denied = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "write with approval"},
        context=unapproved,
        max_output_characters=4_000,
    )
    allowed = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "write with approval"},
        context=approved,
        max_output_characters=4_000,
    )
    assert json.loads(denied.text)["matches"] == []
    assert [item["name"] for item in json.loads(allowed.text)["matches"]] == ["demo.write"]


def test_active_lease_status_metadata_is_bounded(tmp_path: Path) -> None:
    store = _store(tmp_path / "leases.sqlite3")
    created = store.create_lease(
        CapabilityLeaseCreateRequest(
            grantee_kind="actor",
            grantee_id="actor-2",
            capabilities=("demo.channel",),
            expires_at_iso=_expiry(),
            max_uses=3,
            reason="Status metadata.",
            target_kind="channel",
            target_id="channel-7",
        ),
        _context(),
    )
    grantee = _context(actor_id="actor-2", grants=frozenset())
    binding = store.resolve_bindings(grantee, static_grants=frozenset())[0]
    bound = replace(
        grantee,
        capability_lease_bindings=((binding.capability, created.lease_id, 1),),
    )
    metadata = store.active_metadata(bound)
    assert store.active_count(bound) == 1
    assert len(metadata) == 1
    assert "demo.channel" in metadata[0]
    assert "target=channel" in metadata[0]
    assert "channel-7" not in metadata[0]
    assert "remaining=3" in metadata[0]
    assert len(metadata[0]) <= 500
