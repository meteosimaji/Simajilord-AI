"""Platform kernel shared by every transport adapter."""

from .capabilities import (
    AgentPrincipalKind,
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityIdempotency,
    CapabilityRegistry,
    DisclosureClass,
    DisclosureObservation,
    EgressConsentRequirement,
    EgressDescriptor,
    EgressFieldKind,
    EgressSinkAudience,
    InvocationContext,
    RiskLevel,
    endpoint,
)

__all__ = [
    "AgentPrincipalKind",
    "ApprovalMode",
    "CapabilityDescriptor",
    "CapabilityEndpoint",
    "CapabilityIdempotency",
    "CapabilityRegistry",
    "DisclosureClass",
    "DisclosureObservation",
    "EgressConsentRequirement",
    "EgressDescriptor",
    "EgressFieldKind",
    "EgressSinkAudience",
    "InvocationContext",
    "RiskLevel",
    "endpoint",
]
