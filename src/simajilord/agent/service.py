"""Cost-bounded event-driven agent orchestration."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from simajilord.core import InvocationContext
from simajilord.observability import EventJournal

from .contracts import (
    AgentProgressStage,
    AgentRequest,
    AgentResponse,
    AgentResponseStatus,
)
from .errors import (
    AgentBusyError,
    AgentRateLimitError,
    AgentThreadError,
)
from .providers import AgentProgressCallback, AgentProvider
from .store import AgentConversationRecord, AgentConversationStore


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Hard local limits applied before any provider request."""

    per_user_requests: int
    per_user_window_seconds: int
    per_workspace_requests: int
    per_workspace_window_seconds: int
    max_tokens_per_24_hours: int
    max_conversation_turns: int
    max_context_ratio: float
    max_response_characters: int
    max_pending_turns: int
    rate_limit_exempt_actor_ids: frozenset[str] = frozenset()


class AgentService:
    """Share one provider path across explicit and autonomous event origins."""

    def __init__(
        self,
        *,
        provider: AgentProvider,
        store: AgentConversationStore,
        journal: EventJournal,
        limits: AgentLimits,
    ) -> None:
        self.provider = provider
        self.store = store
        self.journal = journal
        self.limits = limits
        self._admission_lock = asyncio.Lock()
        self._turn_slot = asyncio.Semaphore(1)
        self._admitted_turns = 0
        self._conversation_locks: dict[str, asyncio.Lock] = {}

    @property
    def model(self) -> str:
        return self.provider.model

    async def respond(
        self,
        request: AgentRequest,
        *,
        on_progress: AgentProgressCallback | None = None,
    ) -> AgentResponse:
        cached = await self.store.completed_response(request.event_id)
        if cached is not None:
            return cached
        await self._admit(request, on_progress=on_progress)
        try:
            lock = self._conversation_locks.setdefault(
                request.conversation_id,
                asyncio.Lock(),
            )
            async with lock:
                cached = await self.store.completed_response(request.event_id)
                if cached is not None:
                    return cached
                await self._check_budgets(request)
                promoted_from = await self.store.promote_compatible_conversation(
                    request.conversation_id
                )
                if promoted_from is not None:
                    await self.journal.append(
                        kind="agent.conversation.promoted",
                        actor_id=request.actor_id,
                        workspace_id=request.workspace_id,
                        transport="agent",
                        request_id=request.event_id,
                        payload={
                            "conversation_id": request.conversation_id,
                            "promoted_from": promoted_from,
                            "reason": "capability_grant_expansion",
                        },
                    )
                await self.store.begin(request, model=self.model)
                conversation = await self.store.conversation(request.conversation_id)
                provider_thread_id = (
                    conversation.provider_thread_id if conversation is not None else None
                )
                if conversation is not None and self._must_rotate(conversation):
                    await self.store.rotate(request.conversation_id, model=self.model)
                    provider_thread_id = None
                    await self.journal.append(
                        kind="agent.conversation.rotated",
                        actor_id=request.actor_id,
                        workspace_id=request.workspace_id,
                        transport="agent",
                        request_id=request.event_id,
                        payload={
                            "conversation_id": request.conversation_id,
                            "reason": "context_budget",
                        },
                    )
                context = InvocationContext(
                    actor_id=request.actor_id,
                    workspace_id=request.workspace_id,
                    transport="agent",
                    request_id=request.event_id,
                    resource_ids=request.resource_ids,
                    grants=request.grants,
                    origin_resource_id=request.channel_id,
                    approvals=request.approvals,
                )
                if on_progress is not None:
                    await on_progress(AgentProgressStage.STARTING)
                try:
                    result = await self.provider.respond(
                        provider_thread_id=provider_thread_id,
                        event_prompt=_event_prompt(request),
                        context=context,
                        on_progress=on_progress,
                    )
                except AgentThreadError:
                    await self.store.rotate(request.conversation_id, model=self.model)
                    result = await self.provider.respond(
                        provider_thread_id=None,
                        event_prompt=_event_prompt(request),
                        context=context,
                        on_progress=on_progress,
                    )
                content = _bounded_text(
                    result.content.strip(),
                    self.limits.max_response_characters,
                )
                response = AgentResponse(
                    status=AgentResponseStatus.COMPLETED,
                    conversation_id=request.conversation_id,
                    provider_thread_id=result.thread_id,
                    model=result.model,
                    content=content,
                    usage=result.usage,
                )
                await self.store.complete(request, response)
                await self.journal.append(
                    kind="agent.turn.completed",
                    actor_id=request.actor_id,
                    workspace_id=request.workspace_id,
                    transport="agent",
                    request_id=request.event_id,
                    payload={
                        "conversation_id": request.conversation_id,
                        "trigger": request.trigger.value,
                        "model": result.model,
                        "response_characters": len(content),
                        "usage": {
                            "input_tokens": result.usage.input_tokens,
                            "cached_input_tokens": result.usage.cached_input_tokens,
                            "output_tokens": result.usage.output_tokens,
                            "reasoning_output_tokens": result.usage.reasoning_output_tokens,
                            "total_tokens": result.usage.total_tokens,
                            "model_context_window": result.usage.model_context_window,
                        },
                    },
                )
                return response
        except Exception as exc:
            await self.store.fail(
                request,
                model=self.model,
                error_type=type(exc).__name__,
            )
            await self.journal.append(
                kind="agent.turn.failed",
                actor_id=request.actor_id,
                workspace_id=request.workspace_id,
                transport="agent",
                request_id=request.event_id,
                payload={
                    "conversation_id": request.conversation_id,
                    "trigger": request.trigger.value,
                    "model": self.model,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        finally:
            await self._release()

    async def close(self) -> None:
        await self.provider.close()

    async def _admit(
        self,
        request: AgentRequest,
        *,
        on_progress: AgentProgressCallback | None,
    ) -> None:
        async with self._admission_lock:
            if self._admitted_turns >= self.limits.max_pending_turns:
                await self.journal.append(
                    kind="agent.turn.rejected",
                    actor_id=request.actor_id,
                    workspace_id=request.workspace_id,
                    transport="agent",
                    request_id=request.event_id,
                    payload={"reason": "queue_full", "trigger": request.trigger.value},
                )
                raise AgentBusyError("The bounded agent turn queue is full.")
            self._admitted_turns += 1
            queued = self._turn_slot.locked()
        try:
            if queued and on_progress is not None:
                await on_progress(AgentProgressStage.QUEUED)
            await self._turn_slot.acquire()
        except BaseException:
            async with self._admission_lock:
                self._admitted_turns -= 1
            raise

    async def _release(self) -> None:
        self._turn_slot.release()
        async with self._admission_lock:
            self._admitted_turns -= 1

    async def _check_budgets(self, request: AgentRequest) -> None:
        if request.actor_id in self.limits.rate_limit_exempt_actor_ids:
            return
        now = datetime.now(UTC)
        user_count, user_oldest = await self.store.request_window(
            actor_id=request.actor_id,
            workspace_id=None,
            since=now - timedelta(seconds=self.limits.per_user_window_seconds),
        )
        if user_count >= self.limits.per_user_requests:
            raise AgentRateLimitError(
                "The per-user agent budget is exhausted.",
                retry_after_seconds=_retry_after_seconds(
                    now,
                    user_oldest,
                    self.limits.per_user_window_seconds,
                ),
            )
        if request.workspace_id is not None:
            workspace_count, workspace_oldest = await self.store.request_window(
                actor_id=None,
                workspace_id=request.workspace_id,
                since=now - timedelta(seconds=self.limits.per_workspace_window_seconds),
                excluded_actor_ids=self.limits.rate_limit_exempt_actor_ids,
            )
            if workspace_count >= self.limits.per_workspace_requests:
                raise AgentRateLimitError(
                    "The server agent budget is exhausted.",
                    retry_after_seconds=_retry_after_seconds(
                        now,
                        workspace_oldest,
                        self.limits.per_workspace_window_seconds,
                    ),
                )
        usage = await self.store.token_usage_since(
            now - timedelta(hours=24),
            excluded_actor_ids=self.limits.rate_limit_exempt_actor_ids,
        )
        if usage >= self.limits.max_tokens_per_24_hours:
            raise AgentRateLimitError("The rolling agent token budget is exhausted.")

    def _must_rotate(self, conversation: AgentConversationRecord) -> bool:
        if conversation.turn_count >= self.limits.max_conversation_turns:
            return True
        input_tokens = conversation.last_input_tokens
        context_window = conversation.model_context_window
        return (
            isinstance(context_window, int)
            and context_window > 0
            and input_tokens / context_window >= self.limits.max_context_ratio
        )


def _event_prompt(request: AgentRequest) -> str:
    message_id = request.message_id or "none"
    workspace_id = request.workspace_id or "direct-message"
    return "\n".join(
        (
            "SIMAJILORD_EVENT_V1",
            f"trigger={request.trigger.value}",
            f"event_id={request.event_id}",
            f"conversation_id={request.conversation_id}",
            f"workspace_id={workspace_id}",
            f"channel_id={request.channel_id}",
            f"message_id={message_id}",
            f"actor_id={request.actor_id}",
            f"actor_name={request.actor_name}",
            f"occurred_at={request.occurred_at.isoformat()}",
            (
                "No message body is included. Inspect only what you need through bounded "
                "Simajilord tools, then produce one user-facing response."
            ),
        )
    )


def _bounded_text(value: str, maximum: int) -> str:
    if not value:
        return "I could not produce a response."
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


def _retry_after_seconds(
    now: datetime,
    oldest: datetime | None,
    window_seconds: int,
) -> int | None:
    if oldest is None:
        return None
    remaining = (oldest + timedelta(seconds=window_seconds) - now).total_seconds()
    return max(1, math.ceil(remaining))
