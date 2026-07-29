"""Transport-neutral wait and duration metrics for shared service schedulers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServiceOperationMetric:
    """One bounded service operation attributed to a workspace."""

    operation: str
    workspace_id: str
    wait_ms: float
    duration_ms: float
    outcome: str
    resource_id: str | None = None


ServiceMetricHook = Callable[[ServiceOperationMetric], Awaitable[None]]
