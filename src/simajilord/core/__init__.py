"""Platform kernel shared by every transport adapter."""

from .capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)

__all__ = [
    "ApprovalMode",
    "CapabilityDescriptor",
    "CapabilityEndpoint",
    "CapabilityRegistry",
    "InvocationContext",
    "RiskLevel",
    "endpoint",
]
