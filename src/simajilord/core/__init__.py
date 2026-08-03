"""Platform kernel shared by every transport adapter."""

from .capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityIdempotency,
    CapabilityRegistry,
    DisclosureClass,
    DisclosureObservation,
    InvocationContext,
    RiskLevel,
    endpoint,
)

__all__ = [
    "ApprovalMode",
    "CapabilityDescriptor",
    "CapabilityEndpoint",
    "CapabilityIdempotency",
    "CapabilityRegistry",
    "DisclosureClass",
    "DisclosureObservation",
    "InvocationContext",
    "RiskLevel",
    "endpoint",
]
