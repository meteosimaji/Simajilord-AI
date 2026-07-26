"""Transport-neutral capability discovery and invocation."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from time import monotonic
from typing import Any, Protocol, TypeVar

from .errors import CapabilityError

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


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


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Small metadata document loaded before an implementation schema."""

    name: str
    summary: str
    risk: RiskLevel
    approval: ApprovalMode = ApprovalMode.NEVER
    keywords: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()


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
        terms = set(re.findall(r"[a-z0-9_.-]+", query.lower()))
        if not terms:
            return self.all()[:limit]

        scored: list[tuple[int, CapabilityEndpoint]] = []
        for item in self._endpoints.values():
            descriptor = item.descriptor
            name_terms = set(re.findall(r"[a-z0-9_.-]+", descriptor.name.lower()))
            text_terms = set(
                re.findall(
                    r"[a-z0-9_.-]+",
                    " ".join(
                        (
                            descriptor.summary,
                            *descriptor.keywords,
                            *descriptor.side_effects,
                        )
                    ).lower(),
                )
            )
            score = len(terms & text_terms) + 3 * len(terms & name_terms)
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
            response = await selected.invoke(request, context)
        except Exception as exc:
            if self._journal is not None:
                await self._journal.record_invocation(
                    capability_name=name,
                    context=context,
                    request=request,
                    response=None,
                    error=exc,
                    duration_ms=(monotonic() - started) * 1_000,
                )
            raise
        if self._journal is not None:
            await self._journal.record_invocation(
                capability_name=name,
                context=context,
                request=request,
                response=response,
                error=None,
                duration_ms=(monotonic() - started) * 1_000,
            )
        return response

    def manifest(self) -> tuple[Mapping[str, object], ...]:
        """Return a compact manifest suitable for MCP-style discovery."""

        return tuple(
            {
                "name": item.descriptor.name,
                "summary": item.descriptor.summary,
                "risk": item.descriptor.risk.value,
                "approval": item.descriptor.approval.value,
                "request_fields": item.schema.request_fields,
                "response_fields": item.schema.response_fields,
                "side_effects": item.descriptor.side_effects,
            }
            for item in self.all()
        )


class InvocationJournal(Protocol):
    """Write-only journal port kept outside the capability kernel."""

    async def record_invocation(
        self,
        *,
        capability_name: str,
        context: InvocationContext,
        request: object,
        response: object | None,
        error: Exception | None,
        duration_ms: float,
    ) -> None: ...
