"""System capabilities shared by Discord and future adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)


@dataclass(frozen=True, slots=True)
class PingRequest:
    transport_latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class PingResponse:
    status: str
    checked_at: datetime
    transport_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class UptimeRequest:
    pass


@dataclass(frozen=True, slots=True)
class UptimeResponse:
    started_at: datetime
    uptime_seconds: float


@dataclass(frozen=True, slots=True)
class CapabilitySummary:
    name: str
    summary: str
    risk: str
    approval: str
    request_fields: tuple[str, ...]
    response_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilitySearchRequest:
    query: str = ""
    limit: int = 5


@dataclass(frozen=True, slots=True)
class CapabilitySearchResponse:
    capabilities: tuple[CapabilitySummary, ...]


def build_system_endpoints(
    registry: CapabilityRegistry,
    *,
    started_at: datetime,
    started_monotonic: float,
) -> tuple[CapabilityEndpoint, CapabilityEndpoint, CapabilityEndpoint]:
    async def ping(request: PingRequest, _: InvocationContext) -> PingResponse:
        return PingResponse(
            status="ok",
            checked_at=datetime.now(UTC),
            transport_latency_ms=request.transport_latency_ms,
        )

    async def discover(
        request: CapabilitySearchRequest,
        _: InvocationContext,
    ) -> CapabilitySearchResponse:
        items = registry.search(request.query, limit=min(max(request.limit, 1), 20))
        return CapabilitySearchResponse(
            capabilities=tuple(
                CapabilitySummary(
                    name=item.descriptor.name,
                    summary=item.descriptor.summary,
                    risk=item.descriptor.risk.value,
                    approval=item.descriptor.approval.value,
                    request_fields=item.schema.request_fields,
                    response_fields=item.schema.response_fields,
                )
                for item in items
            )
        )

    async def uptime(_: UptimeRequest, __: InvocationContext) -> UptimeResponse:
        return UptimeResponse(
            started_at=started_at,
            uptime_seconds=max(0.0, monotonic() - started_monotonic),
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="system.ping",
                summary="Check platform health and optional transport latency.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=("health", "online", "latency"),
            ),
            PingRequest,
            PingResponse,
            ping,
        ),
        endpoint(
            CapabilityDescriptor(
                name="system.discover_capabilities",
                summary="Find a small set of capabilities for an intended task.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=("help", "features", "tools", "what can you do"),
            ),
            CapabilitySearchRequest,
            CapabilitySearchResponse,
            discover,
        ),
        endpoint(
            CapabilityDescriptor(
                name="system.uptime",
                summary="Show when the current platform process started.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=("uptime", "started", "runtime"),
            ),
            UptimeRequest,
            UptimeResponse,
            uptime,
        ),
    )
