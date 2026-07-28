"""Transport-neutral capability discovery and invocation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from time import monotonic
from typing import Any, Literal, Protocol, TypeVar

from .errors import CapabilityError

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
CapabilityIdempotency = Literal[
    "read",
    "idempotent_write",
    "non_idempotent_write",
]
_SEARCH_TOKEN_PATTERN = re.compile(r"[\w.-]+", re.UNICODE)


def _normalize_search_text(value: str) -> str:
    """Normalize user and descriptor text without assuming an ASCII language."""

    return " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    )


def _search_tokens(value: str) -> set[str]:
    return set(_SEARCH_TOKEN_PATTERN.findall(_normalize_search_text(value)))


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
    approvals: frozenset[str] = frozenset()


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
        if self.risk is RiskLevel.WRITE and self.idempotency == "read":
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
        normalized_query = _normalize_search_text(query)
        if not normalized_query:
            return self.all()[:limit]
        terms = _search_tokens(normalized_query)

        scored: list[tuple[int, CapabilityEndpoint]] = []
        for item in self._endpoints.values():
            descriptor = item.descriptor
            name_terms = _search_tokens(descriptor.name)
            searchable_text = _normalize_search_text(
                " ".join(
                    (
                        descriptor.summary,
                        *descriptor.keywords,
                        *descriptor.side_effects,
                    )
                )
            )
            text_terms = _search_tokens(searchable_text)
            keyword_phrases = tuple(
                normalized
                for keyword in descriptor.keywords
                if (normalized := _normalize_search_text(keyword))
            )
            keyword_substring_score = 2 * sum(
                phrase in normalized_query
                for phrase in keyword_phrases
            )
            exact_text_score = int(
                len(normalized_query) >= 3
                and normalized_query in searchable_text
            )
            score = (
                len(terms & text_terms)
                + 3 * len(terms & name_terms)
                + keyword_substring_score
                + exact_text_score
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
