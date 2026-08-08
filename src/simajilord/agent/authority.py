"""Durable, bounded capability leases within the configured runtime ceiling."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError

from .contracts import expand_agent_grants

CapabilityLeaseGranteeKind = Literal["actor", "role", "service"]
CapabilityLeaseTargetKind = Literal[
    "channel",
    "repository",
    "connector",
    "file_publication",
    "guild_resource",
]

AGENT_AUTHORITY_REQUEST_GRANT = "authority.request"
AGENT_AUTHORITY_MANAGE_GRANT = "authority.manage"
_MAX_LEASE_TTL = timedelta(days=30)
_MAX_REQUEST_TTL = timedelta(hours=24)
_NON_DELEGABLE_CAPABILITIES = frozenset({"authority.lease_create", "authority.lease_revoke"})
_NON_DELEGABLE_GRANTS = frozenset({AGENT_AUTHORITY_MANAGE_GRANT})


@dataclass(frozen=True, slots=True)
class CapabilityLeaseRecord:
    lease_id: str
    delegator_actor_id: str
    grantee_kind: CapabilityLeaseGranteeKind
    grantee_id: str
    workspace_id: str
    agent_task_id: str | None
    target_kind: CapabilityLeaseTargetKind | None
    target_id: str | None
    capabilities: tuple[str, ...]
    grants: tuple[str, ...]
    starts_at: str
    expires_at: str
    max_uses: int
    uses: int
    reason: str
    revision: int
    revoked_at: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class CapabilityLeaseBinding:
    capability: str
    required_grant: str
    lease_id: str
    revision: int


@dataclass(frozen=True, slots=True)
class AuthorityRequestCreateRequest:
    capability: str
    reason: str
    target_kind: CapabilityLeaseTargetKind | None = None
    target_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuthorityRequestCreateResponse:
    request_id: str
    capability: str
    expires_at_iso: str
    recorded: bool


@dataclass(frozen=True, slots=True)
class CapabilityLeaseCreateRequest:
    grantee_kind: CapabilityLeaseGranteeKind
    grantee_id: str
    capabilities: tuple[str, ...] = ()
    grants: tuple[str, ...] = ()
    expires_at_iso: str = ""
    max_uses: int = 1
    reason: str = ""
    agent_task_id: str | None = None
    target_kind: CapabilityLeaseTargetKind | None = None
    target_id: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityLeaseCreateResponse:
    lease_id: str
    revision: int
    expires_at_iso: str
    max_uses: int


@dataclass(frozen=True, slots=True)
class CapabilityLeaseRevokeRequest:
    lease_id: str
    expected_revision: int


@dataclass(frozen=True, slots=True)
class CapabilityLeaseRevokeResponse:
    lease_id: str
    revision: int
    revoked: bool
    changed: bool


@dataclass(frozen=True, slots=True)
class CapabilityLeaseListRequest:
    active_only: bool = True
    limit: int = 50


@dataclass(frozen=True, slots=True)
class CapabilityLeaseListResponse:
    leases: tuple[CapabilityLeaseRecord, ...]


@dataclass(frozen=True, slots=True)
class _PreparedLease:
    grantee_id: str
    capabilities: tuple[str, ...]
    grants: tuple[str, ...]
    expiry: datetime
    reason: str
    target_kind: CapabilityLeaseTargetKind | None
    target_id: str | None
    task_id: str | None


class CapabilityLeaseStore:
    """SQLite lease authority with atomic use consumption and no user bodies."""

    def __init__(
        self,
        path: Path,
        *,
        registry: CapabilityRegistry,
        configured_capabilities: Sequence[str],
        required_grants: Mapping[str, str],
    ) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._registry = registry
        self._configured_capabilities = frozenset(configured_capabilities)
        self._required_grants = dict(required_grants)
        self._lock = threading.RLock()
        unknown = set(self._required_grants) - self._configured_capabilities
        if unknown:
            raise ValueError(
                "lease grant policy references unconfigured capabilities: "
                + ", ".join(sorted(unknown))
            )
        self._initialize()

    def create_authority_request(
        self,
        request: AuthorityRequestCreateRequest,
        context: InvocationContext,
    ) -> AuthorityRequestCreateResponse:
        reason, target_kind, target_id = self.validate_authority_request(request, context)
        request_id = f"areq_{uuid.uuid4().hex}"
        created_at = datetime.now(UTC)
        expires_at = created_at + _MAX_REQUEST_TTL
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO authority_requests (
                    request_id, requester_actor_id, workspace_id, agent_task_id,
                    capability, target_kind, target_id, reason, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    context.actor_id,
                    context.workspace_id,
                    context.agent_task_id,
                    request.capability,
                    target_kind,
                    target_id,
                    reason,
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return AuthorityRequestCreateResponse(
            request_id=request_id,
            capability=request.capability,
            expires_at_iso=expires_at.isoformat(),
            recorded=True,
        )

    def create_lease(
        self,
        request: CapabilityLeaseCreateRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseCreateResponse:
        prepared = self.validate_lease(request, context)
        now = datetime.now(UTC)
        lease_id = f"lease_{uuid.uuid4().hex}"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO capability_leases (
                    lease_id, delegator_actor_id, grantee_kind, grantee_id,
                    workspace_id, agent_task_id, target_kind, target_id,
                    capabilities_json, grants_json, starts_at, expires_at,
                    max_uses, uses, reason, revision, revoked_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1, NULL, ?)
                """,
                (
                    lease_id,
                    context.actor_id,
                    request.grantee_kind,
                    prepared.grantee_id,
                    context.workspace_id,
                    prepared.task_id,
                    prepared.target_kind,
                    prepared.target_id,
                    _json_tuple(prepared.capabilities),
                    _json_tuple(prepared.grants),
                    now.isoformat(),
                    prepared.expiry.isoformat(),
                    request.max_uses,
                    prepared.reason,
                    now.isoformat(),
                ),
            )
        return CapabilityLeaseCreateResponse(
            lease_id=lease_id,
            revision=1,
            expires_at_iso=prepared.expiry.isoformat(),
            max_uses=request.max_uses,
        )

    def revoke_lease(
        self,
        request: CapabilityLeaseRevokeRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseRevokeResponse:
        self.validate_revoke(request, context)
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM capability_leases WHERE lease_id = ?",
                (request.lease_id,),
            ).fetchone()
            if row is None or str(row["workspace_id"]) != context.workspace_id:
                raise UserError("authority.lease_not_found")
            if int(row["revision"]) != request.expected_revision:
                raise UserError("authority.revision_conflict")
            if row["revoked_at"] is not None:
                return CapabilityLeaseRevokeResponse(
                    lease_id=request.lease_id,
                    revision=int(row["revision"]),
                    revoked=True,
                    changed=False,
                )
            revoked_at = datetime.now(UTC).isoformat()
            revision = int(row["revision"]) + 1
            connection.execute(
                """
                UPDATE capability_leases
                SET revoked_at = ?, revision = ?
                WHERE lease_id = ? AND revision = ? AND revoked_at IS NULL
                """,
                (revoked_at, revision, request.lease_id, request.expected_revision),
            )
        return CapabilityLeaseRevokeResponse(
            lease_id=request.lease_id,
            revision=revision,
            revoked=True,
            changed=True,
        )

    def validate_authority_request(
        self,
        request: AuthorityRequestCreateRequest,
        context: InvocationContext,
    ) -> tuple[str, CapabilityLeaseTargetKind | None, str | None]:
        """Validate an authority request without recording or dispatching it."""

        if context.agent_trigger == "autonomous" or context.principal_kind != "requester":
            raise UserError("authority.autonomous_request_forbidden")
        if context.workspace_id is None:
            raise UserError("authority.workspace_required")
        self._validate_capability(request.capability)
        reason = _bounded_reason(request.reason)
        target_kind, target_id = _target_pair(request.target_kind, request.target_id)
        return reason, target_kind, target_id

    def validate_lease(
        self,
        request: CapabilityLeaseCreateRequest,
        context: InvocationContext,
    ) -> _PreparedLease:
        """Validate the complete lease ceiling before any effect is dispatched."""

        self._require_manager(context)
        if context.workspace_id is None:
            raise UserError("authority.workspace_required")
        if request.grantee_kind not in {"actor", "role", "service"}:
            raise UserError("authority.grantee_invalid")
        grantee_id = _bounded_identifier(request.grantee_id, "authority.grantee_invalid")
        capabilities = _unique_bounded(request.capabilities, "authority.capability_invalid")
        grants = _unique_bounded(request.grants, "authority.grant_invalid")
        if not capabilities and not grants:
            raise UserError("authority.scope_required")
        if _NON_DELEGABLE_CAPABILITIES.intersection(capabilities) or (
            _NON_DELEGABLE_GRANTS.intersection(grants)
        ):
            raise UserError("authority.global_ceiling_exceeded")
        effective_delegator_grants = expand_agent_grants(context.grants)
        if any(grant not in effective_delegator_grants for grant in grants):
            raise UserError("authority.delegator_ceiling_exceeded")
        for capability in capabilities:
            self._validate_capability(capability)
            if (
                context.allowed_capabilities is not None
                and capability not in context.allowed_capabilities
            ):
                raise UserError("authority.global_ceiling_exceeded")
            required = self._required_grants.get(capability)
            if required is not None and required not in effective_delegator_grants:
                raise UserError("authority.delegator_ceiling_exceeded")
        delegated_capabilities = {
            capability
            for capability, required in self._required_grants.items()
            if required in grants
        }
        if context.allowed_capabilities is not None and not delegated_capabilities <= set(
            context.allowed_capabilities
        ):
            raise UserError("authority.global_ceiling_exceeded")
        expiry = _future_expiry(request.expires_at_iso)
        if not 1 <= request.max_uses <= 1_000:
            raise UserError("authority.max_uses_invalid")
        reason = _bounded_reason(request.reason)
        target_kind, target_id = _target_pair(request.target_kind, request.target_id)
        task_id = (
            _bounded_identifier(request.agent_task_id, "authority.task_invalid")
            if request.agent_task_id is not None
            else None
        )
        return _PreparedLease(
            grantee_id=grantee_id,
            capabilities=capabilities,
            grants=grants,
            expiry=expiry,
            reason=reason,
            target_kind=target_kind,
            target_id=target_id,
            task_id=task_id,
        )

    def validate_revoke(
        self,
        request: CapabilityLeaseRevokeRequest,
        context: InvocationContext,
    ) -> bool:
        """Return whether a validated revoke still needs a state change."""

        self._require_manager(context)
        _validate_lease_id(request.lease_id)
        if not isinstance(request.expected_revision, int) or isinstance(
            request.expected_revision, bool
        ):
            raise UserError("authority.revision_conflict")
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM capability_leases WHERE lease_id = ?",
                (request.lease_id,),
            ).fetchone()
        if row is None or str(row["workspace_id"]) != context.workspace_id:
            raise UserError("authority.lease_not_found")
        if int(row["revision"]) != request.expected_revision:
            raise UserError("authority.revision_conflict")
        return row["revoked_at"] is None

    def list_leases(
        self,
        request: CapabilityLeaseListRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseListResponse:
        if context.workspace_id is None:
            raise UserError("authority.workspace_required")
        if not 1 <= request.limit <= 100:
            raise UserError("authority.limit_invalid")
        effective = expand_agent_grants(context.grants)
        manager = AGENT_AUTHORITY_MANAGE_GRANT in effective
        now = datetime.now(UTC).isoformat()
        clauses = ["workspace_id = ?"]
        parameters: list[object] = [context.workspace_id]
        if not manager:
            clauses.append("grantee_kind = 'actor' AND grantee_id = ?")
            parameters.append(context.actor_id)
        if request.active_only:
            clauses.append("revoked_at IS NULL AND starts_at <= ? AND expires_at > ?")
            clauses.append("uses < max_uses")
            parameters.extend((now, now))
        query = (
            "SELECT * FROM capability_leases WHERE "
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, lease_id DESC LIMIT ?"
        )
        parameters.append(request.limit)
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return CapabilityLeaseListResponse(leases=tuple(_lease_from_row(row) for row in rows))

    def resolve_bindings(
        self,
        context: InvocationContext,
        *,
        static_grants: frozenset[str],
    ) -> tuple[CapabilityLeaseBinding, ...]:
        """Resolve catalog additions without consuming use count."""

        if context.workspace_id is None or context.agent_trigger == "autonomous":
            return ()
        effective_static = expand_agent_grants(static_grants)
        principal_ids = {("actor", context.actor_id)}
        principal_ids.update(("role", role_id) for role_id in context.principal_role_ids)
        if context.principal_kind == "service":
            principal_ids.add(("service", context.actor_id))
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT * FROM capability_leases
                WHERE workspace_id = ? AND revoked_at IS NULL
                  AND starts_at <= ? AND expires_at > ? AND uses < max_uses
                ORDER BY expires_at ASC, lease_id ASC
                """,
                (context.workspace_id, now, now),
            ).fetchall()
        selected: dict[str, CapabilityLeaseBinding] = {}
        for row in rows:
            record = _lease_from_row(row)
            if (record.grantee_kind, record.grantee_id) not in principal_ids:
                continue
            if record.agent_task_id is not None and record.agent_task_id != context.agent_task_id:
                continue
            capabilities = set(record.capabilities)
            capabilities.update(
                capability
                for capability, required in self._required_grants.items()
                if required in record.grants
            )
            capabilities.intersection_update(self._configured_capabilities)
            if context.allowed_capabilities is not None:
                capabilities.intersection_update(context.allowed_capabilities)
            for capability in sorted(capabilities):
                required = self._required_grants.get(capability)
                if required is None or required in effective_static or capability in selected:
                    continue
                selected[capability] = CapabilityLeaseBinding(
                    capability=capability,
                    required_grant=required,
                    lease_id=record.lease_id,
                    revision=record.revision,
                )
        return tuple(selected[name] for name in sorted(selected))

    def consume_binding(
        self,
        binding: CapabilityLeaseBinding,
        request: object,
        context: InvocationContext,
    ) -> None:
        """Atomically consume one use after revalidating identity, scope, and target."""

        if context.agent_trigger == "autonomous":
            raise UserError("authority.autonomous_use_forbidden")
        _validate_lease_id(binding.lease_id)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM capability_leases WHERE lease_id = ?",
                (binding.lease_id,),
            ).fetchone()
            if row is None:
                raise UserError("authority.lease_unavailable")
            record = _lease_from_row(row)
            if (
                record.revision != binding.revision
                or record.workspace_id != context.workspace_id
                or record.revoked_at is not None
                or record.starts_at > now
                or record.expires_at <= now
                or record.uses >= record.max_uses
                or binding.capability not in self._configured_capabilities
            ):
                raise UserError("authority.lease_unavailable")
            principals = {("actor", context.actor_id)}
            principals.update(("role", role_id) for role_id in context.principal_role_ids)
            if context.principal_kind == "service":
                principals.add(("service", context.actor_id))
            if (record.grantee_kind, record.grantee_id) not in principals:
                raise UserError("authority.lease_unavailable")
            if record.agent_task_id is not None and record.agent_task_id != context.agent_task_id:
                raise UserError("authority.lease_unavailable")
            capabilities = set(record.capabilities)
            capabilities.update(
                capability
                for capability, required in self._required_grants.items()
                if required in record.grants
            )
            if binding.capability not in capabilities:
                raise UserError("authority.lease_unavailable")
            if record.target_kind is not None:
                actual_target = _request_target_id(record.target_kind, request, context)
                if actual_target != record.target_id:
                    raise UserError("authority.target_scope_mismatch")
            cursor = connection.execute(
                """
                UPDATE capability_leases
                SET uses = uses + 1
                WHERE lease_id = ? AND revision = ? AND revoked_at IS NULL
                  AND starts_at <= ? AND expires_at > ? AND uses < max_uses
                """,
                (binding.lease_id, binding.revision, now, now),
            )
            if cursor.rowcount != 1:
                raise UserError("authority.lease_unavailable")

    def active_count(self, context: InvocationContext) -> int:
        return len(context.capability_lease_bindings)

    def active_metadata(self, context: InvocationContext) -> tuple[str, ...]:
        """Return bounded body-free summaries for leases bound to this turn."""

        if context.workspace_id is None:
            return ()
        binding_by_lease: dict[str, list[str]] = {}
        for capability, lease_id, _revision in context.capability_lease_bindings[:64]:
            binding_by_lease.setdefault(lease_id, []).append(capability)
        if not binding_by_lease:
            return ()
        with self._lock, self._connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = tuple(
                row
                for lease_id in binding_by_lease
                if (
                    row := connection.execute(
                        "SELECT * FROM capability_leases WHERE lease_id = ?",
                        (lease_id,),
                    ).fetchone()
                )
                is not None
            )
        summaries: list[str] = []
        for row in rows:
            record = _lease_from_row(row)
            if record.workspace_id != context.workspace_id:
                continue
            target = record.target_kind or "workspace"
            capabilities = ",".join(sorted(binding_by_lease.get(record.lease_id, ())))
            summaries.append(
                f"{capabilities}; target={target}; expires={record.expires_at}; "
                f"remaining={max(0, record.max_uses - record.uses)}"
            )
        return tuple(sorted(summaries)[:20])

    def _validate_capability(self, capability: str) -> None:
        if capability not in self._configured_capabilities:
            raise UserError("authority.global_ceiling_exceeded")
        try:
            self._registry.endpoint(capability)
        except Exception as exc:
            raise UserError("authority.global_ceiling_exceeded") from exc

    @staticmethod
    def _require_manager(context: InvocationContext) -> None:
        if context.agent_trigger == "autonomous" or context.principal_kind != "requester":
            raise UserError("authority.autonomous_management_forbidden")
        if AGENT_AUTHORITY_MANAGE_GRANT not in expand_agent_grants(context.grants):
            raise UserError("authority.manage_forbidden")

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_leases (
                    lease_id TEXT PRIMARY KEY,
                    delegator_actor_id TEXT NOT NULL,
                    grantee_kind TEXT NOT NULL,
                    grantee_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    agent_task_id TEXT,
                    target_kind TEXT,
                    target_id TEXT,
                    capabilities_json TEXT NOT NULL,
                    grants_json TEXT NOT NULL,
                    starts_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    max_uses INTEGER NOT NULL,
                    uses INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    revoked_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS capability_leases_active_scope
                ON capability_leases (
                    workspace_id, grantee_kind, grantee_id, expires_at, revoked_at
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS authority_requests (
                    request_id TEXT PRIMARY KEY,
                    requester_actor_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    agent_task_id TEXT,
                    capability TEXT NOT NULL,
                    target_kind TEXT,
                    target_id TEXT,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
        self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def build_authority_endpoints(store: CapabilityLeaseStore) -> tuple[CapabilityEndpoint, ...]:
    async def create_request(
        request: AuthorityRequestCreateRequest,
        context: InvocationContext,
    ) -> AuthorityRequestCreateResponse:
        store.validate_authority_request(request, context)
        await context.dispatch_external_effect()
        return store.create_authority_request(request, context)

    async def create_lease(
        request: CapabilityLeaseCreateRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseCreateResponse:
        store.validate_lease(request, context)
        await context.dispatch_external_effect()
        return store.create_lease(request, context)

    async def revoke_lease(
        request: CapabilityLeaseRevokeRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseRevokeResponse:
        if store.validate_revoke(request, context):
            await context.dispatch_external_effect()
        else:
            await context.complete_external_effect_without_dispatch()
        return store.revoke_lease(request, context)

    async def list_leases(
        request: CapabilityLeaseListRequest,
        context: InvocationContext,
    ) -> CapabilityLeaseListResponse:
        return store.list_leases(request, context)

    return (
        endpoint(
            CapabilityDescriptor(
                name="authority.request",
                summary=(
                    "Record one body-free request for a concrete configured capability "
                    "without revealing unavailable schemas or granting authority."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                requires_workspace=True,
                idempotency="non_idempotent_write",
                side_effects=("Records one private host authority request.",),
                user_visible_effect="Creates a private, expiring authority request.",
                audit_payload="metadata",
                expected_errors=(
                    "authority.autonomous_request_forbidden",
                    "authority.global_ceiling_exceeded",
                ),
            ),
            AuthorityRequestCreateRequest,
            AuthorityRequestCreateResponse,
            create_request,
        ),
        endpoint(
            CapabilityDescriptor(
                name="authority.lease_create",
                summary=(
                    "Create one requester-reviewed actor, role, task, workspace, and "
                    "optional target-scoped capability lease within the delegator ceiling."
                ),
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                requires_workspace=True,
                idempotency="non_idempotent_write",
                side_effects=("Temporarily expands one grantee catalog within a fixed ceiling.",),
                user_visible_effect="Creates one expiring, use-limited authority lease.",
                audit_payload="metadata",
                expected_errors=(
                    "authority.manage_forbidden",
                    "authority.delegator_ceiling_exceeded",
                    "authority.global_ceiling_exceeded",
                ),
            ),
            CapabilityLeaseCreateRequest,
            CapabilityLeaseCreateResponse,
            create_lease,
        ),
        endpoint(
            CapabilityDescriptor(
                name="authority.lease_revoke",
                summary="Revoke one scoped capability lease by exact revision.",
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                requires_workspace=True,
                idempotency="idempotent_write",
                side_effects=("Prevents future use of one capability lease.",),
                user_visible_effect="Revokes one temporary authority lease.",
                audit_payload="metadata",
                expected_errors=(
                    "authority.manage_forbidden",
                    "authority.lease_not_found",
                    "authority.revision_conflict",
                ),
            ),
            CapabilityLeaseRevokeRequest,
            CapabilityLeaseRevokeResponse,
            revoke_lease,
        ),
        endpoint(
            CapabilityDescriptor(
                name="authority.lease_list",
                summary="List bounded, body-free capability lease metadata visible to this actor.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                requires_workspace=True,
                expected_errors=("authority.limit_invalid",),
            ),
            CapabilityLeaseListRequest,
            CapabilityLeaseListResponse,
            list_leases,
        ),
    )


def _lease_from_row(row: sqlite3.Row) -> CapabilityLeaseRecord:
    capabilities = _json_values(str(row["capabilities_json"]))
    grants = _json_values(str(row["grants_json"]))
    grantee_kind = str(row["grantee_kind"])
    target_kind = str(row["target_kind"]) if row["target_kind"] is not None else None
    if grantee_kind not in {"actor", "role", "service"}:
        raise UserError("authority.lease_invalid")
    if target_kind not in {
        None,
        "channel",
        "repository",
        "connector",
        "file_publication",
        "guild_resource",
    }:
        raise UserError("authority.lease_invalid")
    return CapabilityLeaseRecord(
        lease_id=str(row["lease_id"]),
        delegator_actor_id=str(row["delegator_actor_id"]),
        grantee_kind=cast(CapabilityLeaseGranteeKind, grantee_kind),
        grantee_id=str(row["grantee_id"]),
        workspace_id=str(row["workspace_id"]),
        agent_task_id=(str(row["agent_task_id"]) if row["agent_task_id"] is not None else None),
        target_kind=cast(CapabilityLeaseTargetKind | None, target_kind),
        target_id=str(row["target_id"]) if row["target_id"] is not None else None,
        capabilities=capabilities,
        grants=grants,
        starts_at=str(row["starts_at"]),
        expires_at=str(row["expires_at"]),
        max_uses=int(row["max_uses"]),
        uses=int(row["uses"]),
        reason=str(row["reason"]),
        revision=int(row["revision"]),
        revoked_at=str(row["revoked_at"]) if row["revoked_at"] is not None else None,
        created_at=str(row["created_at"]),
    )


def _request_target_id(
    kind: CapabilityLeaseTargetKind,
    request: object,
    context: InvocationContext,
) -> str | None:
    fields = {
        "channel": ("channel_id", "thread_id"),
        "repository": ("repository", "repository_full_name", "repo"),
        "connector": ("connector", "connector_name", "app_name"),
        "file_publication": ("publication_id", "file_ref"),
        "guild_resource": ("resource_id", "guild_id", "role_id"),
    }[kind]
    for name in fields:
        value = getattr(request, name, None)
        if isinstance(value, str) and value:
            return value
    if kind == "channel":
        return context.origin_resource_id
    if kind == "guild_resource":
        return context.workspace_id
    return None


def _target_pair(
    kind: CapabilityLeaseTargetKind | None,
    target_id: str | None,
) -> tuple[CapabilityLeaseTargetKind | None, str | None]:
    if (kind is None) != (target_id is None):
        raise UserError("authority.target_scope_invalid")
    if kind is None:
        return None, None
    if kind not in {"channel", "repository", "connector", "file_publication", "guild_resource"}:
        raise UserError("authority.target_scope_invalid")
    assert target_id is not None
    return kind, _bounded_identifier(target_id, "authority.target_scope_invalid")


def _future_expiry(value: str) -> datetime:
    try:
        expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError("authority.expiry_invalid") from exc
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        raise UserError("authority.expiry_invalid")
    expiry = expiry.astimezone(UTC)
    now = datetime.now(UTC)
    if not now < expiry <= now + _MAX_LEASE_TTL:
        raise UserError("authority.expiry_invalid")
    return expiry


def _bounded_reason(value: str) -> str:
    reason = " ".join(value.split())
    if not 1 <= len(reason) <= 500:
        raise UserError("authority.reason_invalid")
    return reason


def _bounded_identifier(value: str, code: str) -> str:
    normalized = value.strip()
    if not 1 <= len(normalized) <= 200 or "\x00" in normalized:
        raise UserError(code)
    return normalized


def _unique_bounded(values: tuple[str, ...], code: str) -> tuple[str, ...]:
    if len(values) > 64 or len(set(values)) != len(values):
        raise UserError(code)
    return tuple(_bounded_identifier(value, code) for value in values)


def _json_tuple(values: tuple[str, ...]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def _json_values(value: str) -> tuple[str, ...]:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise UserError("authority.lease_invalid") from exc
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise UserError("authority.lease_invalid")
    return _unique_bounded(tuple(raw), "authority.lease_invalid")


def _validate_lease_id(value: str) -> None:
    if not re.fullmatch(r"lease_[0-9a-f]{32}", value):
        raise UserError("authority.lease_not_found")
