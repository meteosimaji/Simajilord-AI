"""Provider port implemented by Codex now and a local model later."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from simajilord.core import InvocationContext

from ..contracts import (
    AgentHighRiskConfirmation,
    AgentProgressUpdate,
    AgentTaskRouteDecision,
    AgentTokenUsage,
)

AgentProgressCallback = Callable[[AgentProgressUpdate], Awaitable[None]]
AgentHighRiskConfirmationCallback = Callable[
    [AgentHighRiskConfirmation], Awaitable[bool]
]

@dataclass(frozen=True, slots=True)
class ProviderTurnResult:
    """Final provider result with no reasoning stream."""

    thread_id: str
    model: str
    content: str
    usage: AgentTokenUsage


class AgentProvider(Protocol):
    """Minimal model-provider boundary used by the orchestration service."""

    model: str

    async def respond(
        self,
        *,
        provider_thread_id: str | None,
        event_prompt: str,
        context: InvocationContext,
        on_progress: AgentProgressCallback | None = None,
        on_high_risk_confirmation: AgentHighRiskConfirmationCallback | None = None,
    ) -> ProviderTurnResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class SemanticRoutingAgentProvider(Protocol):
    """Optional provider extension for typed same-turn task routing."""

    async def route_candidate(
        self,
        *,
        event_prompt: str,
        context: InvocationContext,
    ) -> AgentTaskRouteDecision | None: ...

    async def confirm_candidate_route(
        self,
        *,
        event_id: str,
        decision: AgentTaskRouteDecision,
        committed: bool,
        context: InvocationContext,
    ) -> bool:
        """Release the model tool call only after the host commits its route."""

        ...


class AgentToolTraceSink(Protocol):
    """Minimal structured event port used by provider-specific tool tracing."""

    async def append(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None = None,
        workspace_id: str | None = None,
        transport: str | None = None,
        request_id: str | None = None,
    ) -> int: ...


class AgentProviderThreadBindingSink(Protocol):
    """Durable task/thread binding written before a provider turn begins."""

    async def bind_provider_thread(
        self,
        *,
        event_id: str,
        task_id: str,
        conversation_id: str,
        provider_thread_id: str,
        model: str,
    ) -> bool: ...
