"""Transport-neutral capability discovery and invocation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from time import monotonic
from typing import Any, Literal, Protocol, TypeVar

from .errors import CapabilityError, UserError
from .search import (
    normalize_search_text,
    normalized_substring,
    phrase_match_score,
    search_overlap_score,
)

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
CapabilityIdempotency = Literal[
    "read",
    "idempotent_write",
    "non_idempotent_write",
]
CapabilityAuditPayload = Literal["full", "metadata"]
AgentPrincipalKind = Literal[
    "requester",
    "service",
    "system",
    "legacy_unknown",
]
AgentReadScopeMode = Literal["resource_ids", "requester_live", "service_live"]
InformationFlowMode = Literal["enforce", "audit", "disabled"]
FileWorkspaceMode = Literal["actor_task", "actor", "guild_shared"]
HighRiskAuthorizationMode = Literal["bound_once", "legacy_event"]
log = logging.getLogger(__name__)


class RiskLevel(StrEnum):
    """Operational risk visible to an agent before invocation."""

    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class ApprovalMode(StrEnum):
    """Whether an agent must obtain approval before invocation."""

    NEVER = "never"
    WHEN_REQUESTED = "when_requested"
    ALWAYS = "always"


class DisclosureClass(StrEnum):
    """Kind of information a capability can add to the active model turn."""

    NO_USER_CONTENT = "no_user_content"
    GUILD_MEMBER_METADATA = "guild_member_metadata"
    GUILD_PUBLIC_METADATA = "guild_public_metadata"
    CHANNEL_SCOPED_CONTENT = "channel_scoped_content"
    ACTOR_PRIVATE = "actor_private"
    EXTERNAL_PUBLIC = "external_public"
    EXTERNAL_PRIVATE = "external_private"
    UNKNOWN = "unknown"


class EgressFieldKind(StrEnum):
    """Body-free category of data sent beyond the local capability platform."""

    QUERY = "query"
    URL = "url"
    PROMPT = "prompt"
    MEDIA = "media"
    CONNECTOR_ARGUMENTS = "connector_arguments"


class EgressSinkAudience(StrEnum):
    """Host knowledge about the audience receiving one external transfer."""

    EXTERNAL_PUBLIC = "external_public"
    EXTERNAL_PRIVATE = "external_private"
    UNKNOWN = "unknown"


class EgressConsentRequirement(StrEnum):
    """When source labels require a host-verifiable requester decision."""

    NONE = "none"
    RESTRICTED_OR_UNCERTAIN = "restricted_or_uncertain"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class EgressDescriptor:
    """Typed declaration of model- or host-supplied data leaving the platform."""

    provider: str
    field_kinds: tuple[EgressFieldKind, ...]
    sink_audience: EgressSinkAudience
    consent: EgressConsentRequirement = (
        EgressConsentRequirement.RESTRICTED_OR_UNCERTAIN
    )
    request_fields: tuple[str, ...] = ()
    source_resource_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider.strip() or len(self.provider) > 100:
            raise ValueError("egress provider must contain 1 to 100 characters")
        if not self.field_kinds or len(set(self.field_kinds)) != len(
            self.field_kinds
        ):
            raise ValueError("egress field kinds must be non-empty and unique")
        if any(not isinstance(item, EgressFieldKind) for item in self.field_kinds):
            raise ValueError("egress field kinds must be EgressFieldKind values")
        if len(set(self.request_fields)) != len(self.request_fields) or any(
            not item.strip() for item in self.request_fields
        ):
            raise ValueError("egress request fields must be non-empty and unique")
        if len(set(self.source_resource_fields)) != len(
            self.source_resource_fields
        ) or any(not item.strip() for item in self.source_resource_fields):
            raise ValueError(
                "egress source resource fields must be non-empty and unique"
            )
        if not isinstance(self.sink_audience, EgressSinkAudience):
            raise ValueError("egress sink audience must be an EgressSinkAudience")
        if not isinstance(self.consent, EgressConsentRequirement):
            raise ValueError("egress consent must be an EgressConsentRequirement")


@dataclass(frozen=True, slots=True)
class DisclosureObservation:
    """One source audience observed by an active model turn."""

    source_workspace_id: str
    source_resource_id: str
    visibility: Literal["guild_public", "restricted", "uncertain"]
    relation_to_origin: Literal["same_or_narrower", "broader", "uncertain"]


class ExternalEffectDispatch(Protocol):
    """Turn-scoped replay barrier advanced by an adapter at its effect boundary."""

    @property
    def dispatched(self) -> bool: ...

    @property
    def completed_without_dispatch(self) -> bool: ...

    async def dispatch(self) -> None: ...

    async def complete_without_dispatch(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InvocationContext:
    """Identity and scope supplied by a transport adapter."""

    actor_id: str
    workspace_id: str | None
    transport: str
    request_id: str
    resource_ids: tuple[str, ...] = ()
    grants: frozenset[str] = frozenset()
    origin_resource_id: str | None = None
    approvals: frozenset[str] = frozenset()
    public_reference_id: str | None = None
    agent_task_id: str | None = None
    agent_conversation_id: str | None = None
    provider_thread_id: str | None = None
    provider_turn_id: str | None = None
    tool_call_id: str | None = None
    active_message_id: str | None = None
    active_message_edited_at: str | None = None
    batched_message_ids: tuple[str, ...] = ()
    agent_trigger: Literal["mention", "autonomous"] | None = None
    principal_kind: AgentPrincipalKind = "requester"
    read_scope_mode: AgentReadScopeMode = "resource_ids"
    information_flow_mode: InformationFlowMode = "enforce"
    file_workspace_mode: FileWorkspaceMode = "actor_task"
    high_risk_authorization_mode: HighRiskAuthorizationMode = "bound_once"
    disclosure_observations: tuple[DisclosureObservation, ...] = ()
    executor_principal_id: str | None = None
    delegator_principal_id: str | None = None
    trigger_actor_ids: tuple[str, ...] = ()
    requester_principal_id: str | None = None
    policy_id: str | None = None
    principal_role_ids: tuple[str, ...] = ()
    capability_lease_bindings: tuple[tuple[str, str, int], ...] = ()
    allowed_capabilities: frozenset[str] | None = None
    external_effect_dispatch: ExternalEffectDispatch | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    async def dispatch_external_effect(self) -> None:
        """Advance a tracked write immediately before its first real side effect."""

        if self.external_effect_dispatch is not None:
            await self.external_effect_dispatch.dispatch()

    async def complete_external_effect_without_dispatch(self) -> None:
        """Close a validated idempotent no-op without creating a replay barrier."""

        if self.external_effect_dispatch is not None:
            await self.external_effect_dispatch.complete_without_dispatch()


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Small metadata document loaded before an implementation schema."""

    name: str
    summary: str
    risk: RiskLevel
    approval: ApprovalMode = ApprovalMode.NEVER
    keywords: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    requires_workspace: bool = False
    requires_voice: bool = False
    requires_same_voice: bool = False
    idempotency: CapabilityIdempotency = "read"
    expected_errors: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    user_visible_effect: str | None = None
    audit_payload: CapabilityAuditPayload = "full"
    disclosure_class: DisclosureClass | None = None
    egress: EgressDescriptor | None = None

    def __post_init__(self) -> None:
        """Reject metadata that would teach an agent contradictory behavior."""

        if self.requires_same_voice and not self.requires_voice:
            raise ValueError("requires_same_voice requires requires_voice")
        if self.requires_voice and not self.requires_workspace:
            raise ValueError("requires_voice requires requires_workspace")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if len(set(self.expected_errors)) != len(self.expected_errors):
            raise ValueError("expected_errors must be unique")
        if any(not item.strip() for item in self.expected_errors):
            raise ValueError("expected_errors must not contain empty values")
        if self.risk is RiskLevel.READ and self.idempotency != "read":
            raise ValueError("read capabilities must use read idempotency")
        if self.risk is RiskLevel.READ and self.disclosure_class is None:
            raise ValueError("read capabilities require an explicit disclosure_class")
        if self.disclosure_class is not None and not isinstance(
            self.disclosure_class,
            DisclosureClass,
        ):
            raise ValueError("disclosure_class must be a DisclosureClass")
        if self.egress is not None and not isinstance(
            self.egress,
            EgressDescriptor,
        ):
            raise ValueError("egress must be an EgressDescriptor")
        if self.egress is not None and self.audit_payload != "metadata":
            raise ValueError("egress capabilities require metadata-only audit payloads")
        if (
            self.risk in {RiskLevel.WRITE, RiskLevel.DESTRUCTIVE}
            and self.idempotency == "read"
        ):
            object.__setattr__(self, "idempotency", "non_idempotent_write")


@dataclass(frozen=True, slots=True)
class CapabilitySchema:
    """Introspectable request and response field names."""

    request_fields: tuple[str, ...]
    response_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityEndpoint:
    """A typed application endpoint erased only at the registry boundary."""

    descriptor: CapabilityDescriptor
    request_type: type[Any]
    response_type: type[Any]
    invoke: Callable[[Any, InvocationContext], Awaitable[Any]]

    @property
    def schema(self) -> CapabilitySchema:
        return CapabilitySchema(
            request_fields=_field_names(self.request_type),
            response_fields=_field_names(self.response_type),
        )


def endpoint(
    descriptor: CapabilityDescriptor,
    request_type: type[RequestT],
    response_type: type[ResponseT],
    handler: Callable[[RequestT, InvocationContext], Awaitable[ResponseT]],
) -> CapabilityEndpoint:
    """Create a runtime-checked endpoint from a typed handler."""

    request_fields = _field_names(request_type)
    if descriptor.egress is not None:
        declared_egress_fields = set(descriptor.egress.request_fields) | set(
            descriptor.egress.source_resource_fields
        )
        unknown_egress_fields = declared_egress_fields - set(request_fields)
        if unknown_egress_fields:
            raise CapabilityError(
                f"{descriptor.name} egress fields are absent from "
                f"{request_type.__name__}: {', '.join(sorted(unknown_egress_fields))}."
            )

    async def checked(request: Any, context: InvocationContext) -> Any:
        if not isinstance(request, request_type):
            raise CapabilityError(
                f"{descriptor.name} expects {request_type.__name__}, "
                f"received {type(request).__name__}."
            )
        response = await handler(request, context)
        if not isinstance(response, response_type):
            raise CapabilityError(
                f"{descriptor.name} returned {type(response).__name__}; "
                f"expected {response_type.__name__}."
            )
        return response

    return CapabilityEndpoint(
        descriptor=descriptor,
        request_type=request_type,
        response_type=response_type,
        invoke=checked,
    )


def _field_names(model: type[Any]) -> tuple[str, ...]:
    if not is_dataclass(model):
        raise CapabilityError(f"{model.__name__} must be a dataclass.")
    return tuple(field.name for field in fields(model))


class CapabilityRegistry:
    """Single authority for discoverable, invokable platform capabilities."""

    def __init__(self, journal: InvocationJournal | None = None) -> None:
        self._endpoints: dict[str, CapabilityEndpoint] = {}
        self._journal = journal

    def register(self, capability: CapabilityEndpoint) -> None:
        name = capability.descriptor.name
        if name in self._endpoints:
            raise CapabilityError(f"Duplicate capability name: {name}")
        self._endpoints[name] = capability

    def endpoint(self, name: str) -> CapabilityEndpoint:
        try:
            return self._endpoints[name]
        except KeyError as exc:
            raise CapabilityError(f"Unknown capability: {name}") from exc

    def all(self) -> tuple[CapabilityEndpoint, ...]:
        return tuple(sorted(self._endpoints.values(), key=lambda item: item.descriptor.name))

    def search(self, query: str, *, limit: int = 5) -> tuple[CapabilityEndpoint, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        normalized_query = normalize_search_text(query)
        if not normalized_query:
            return self.all()[:limit]

        scored: list[tuple[int, CapabilityEndpoint]] = []
        for item in self._endpoints.values():
            descriptor = item.descriptor
            searchable_text = " ".join(
                (
                    descriptor.summary,
                    *descriptor.keywords,
                    *descriptor.side_effects,
                    descriptor.user_visible_effect or "",
                )
            )
            schema_text = " ".join(
                (
                    *item.schema.request_fields,
                    *item.schema.response_fields,
                )
            )
            score = (
                search_overlap_score(query, searchable_text)
                + 3 * search_overlap_score(query, descriptor.name)
                + search_overlap_score(query, schema_text)
                + 3 * phrase_match_score(query, descriptor.keywords)
                + 2 * int(normalized_substring(query, searchable_text))
            )
            if score:
                scored.append((score, item))
        scored.sort(key=lambda entry: (-entry[0], entry[1].descriptor.name))
        return tuple(item for _, item in scored[:limit])

    async def invoke(
        self,
        name: str,
        request: object,
        context: InvocationContext,
    ) -> object:
        """Invoke a capability identically from Discord, MCP, or a future adapter."""

        selected = self.endpoint(name)
        started = monotonic()
        try:
            timeout_seconds = selected.descriptor.timeout_seconds
            if timeout_seconds is None:
                response = await selected.invoke(request, context)
            else:
                try:
                    async with asyncio.timeout(timeout_seconds):
                        response = await selected.invoke(request, context)
                except TimeoutError as exc:
                    raise UserError(
                        "capability.timeout",
                        capability=name,
                        timeout_seconds=timeout_seconds,
                    ) from exc
        except Exception as exc:
            await self._record_invocation_safely(
                capability_name=name,
                audit_payload=selected.descriptor.audit_payload,
                context=context,
                request=request,
                response=None,
                error=exc,
                duration_ms=(monotonic() - started) * 1_000,
            )
            raise
        await self._record_invocation_safely(
            capability_name=name,
            audit_payload=selected.descriptor.audit_payload,
            context=context,
            request=request,
            response=response,
            error=None,
            duration_ms=(monotonic() - started) * 1_000,
        )
        return response

    async def _record_invocation_safely(
        self,
        *,
        capability_name: str,
        audit_payload: CapabilityAuditPayload,
        context: InvocationContext,
        request: object,
        response: object | None,
        error: Exception | None,
        duration_ms: float,
    ) -> None:
        """Keep observability failures from changing the capability outcome."""

        if self._journal is None:
            return
        try:
            await self._journal.record_invocation(
                capability_name=capability_name,
                audit_payload=audit_payload,
                context=context,
                request=request,
                response=response,
                error=error,
                duration_ms=duration_ms,
            )
        except Exception:
            log.exception(
                "Capability journal record failed capability=%s request_id=%s",
                capability_name,
                context.request_id,
            )

    def manifest(self) -> tuple[Mapping[str, object], ...]:
        """Return a compact manifest suitable for MCP-style discovery."""

        return tuple(
            {
                "name": item.descriptor.name,
                "summary": item.descriptor.summary,
                "risk": item.descriptor.risk.value,
                "approval": item.descriptor.approval.value,
                "requires_workspace": item.descriptor.requires_workspace,
                "requires_voice": item.descriptor.requires_voice,
                "requires_same_voice": item.descriptor.requires_same_voice,
                "idempotency": item.descriptor.idempotency,
                "expected_errors": item.descriptor.expected_errors,
                "timeout_seconds": item.descriptor.timeout_seconds,
                "user_visible_effect": item.descriptor.user_visible_effect,
                "request_fields": item.schema.request_fields,
                "response_fields": item.schema.response_fields,
                "side_effects": item.descriptor.side_effects,
                "disclosure_class": (
                    item.descriptor.disclosure_class.value
                    if item.descriptor.disclosure_class is not None
                    else None
                ),
                "egress": (
                    {
                        "provider": item.descriptor.egress.provider,
                        "field_kinds": tuple(
                            field_kind.value
                            for field_kind in item.descriptor.egress.field_kinds
                        ),
                        "request_fields": item.descriptor.egress.request_fields,
                        "source_resource_fields": (
                            item.descriptor.egress.source_resource_fields
                        ),
                        "sink_audience": (
                            item.descriptor.egress.sink_audience.value
                        ),
                        "consent": item.descriptor.egress.consent.value,
                    }
                    if item.descriptor.egress is not None
                    else None
                ),
            }
            for item in self.all()
        )


class InvocationJournal(Protocol):
    """Write-only journal port kept outside the capability kernel."""

    async def record_invocation(
        self,
        *,
        capability_name: str,
        audit_payload: CapabilityAuditPayload = "full",
        context: InvocationContext,
        request: object,
        response: object | None,
        error: Exception | None,
        duration_ms: float,
    ) -> None: ...
