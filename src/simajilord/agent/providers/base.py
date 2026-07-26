"""Provider port implemented by Codex now and a local model later."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from simajilord.core import InvocationContext

from ..contracts import AgentProgressStage, AgentTokenUsage

AgentProgressCallback = Callable[[AgentProgressStage], Awaitable[None]]

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
    ) -> ProviderTurnResult: ...

    async def close(self) -> None: ...
