"""Cost-bounded event-driven agent orchestration."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from simajilord.core import InvocationContext
from simajilord.observability import EventJournal

from .contracts import (
    AGENT_MEMORY_GRANT,
    AgentProgressStage,
    AgentProgressUpdate,
    AgentRequest,
    AgentResponse,
    AgentResponseStatus,
    AgentTrigger,
)
from .errors import (
    AgentBusyError,
    AgentProviderError,
    AgentRateLimitError,
    AgentThreadError,
    AgentTimeoutError,
)
from .memory import AgentMemoryRecord, AgentMemoryService
from .providers import (
    AgentProgressCallback,
    AgentProvider,
    SteerableAgentProvider,
)
from .store import AgentConversationStore


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Hard local limits applied before any provider request."""

    per_user_requests: int
    per_user_window_seconds: int
    per_workspace_requests: int
    per_workspace_window_seconds: int
    max_tokens_per_24_hours: int
    max_response_characters: int
    max_active_turns: int
    max_pending_turns: int
    max_pending_turns_per_user: int
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
        memory: AgentMemoryService | None = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.journal = journal
        self.limits = limits
        self.memory = memory
        self._admission_lock = asyncio.Lock()
        self._admission_condition = asyncio.Condition(self._admission_lock)
        self._budget_lock = asyncio.Lock()
        self._workspace_turn_slots: dict[str, asyncio.Semaphore] = {}
        self._active_turn_slots = asyncio.Semaphore(self.limits.max_active_turns)
        self._workspace_waiters: dict[str, int] = {}
        self._active_turns = 0
        self._pending_turns = 0
        self._pending_turns_by_actor: dict[str, int] = {}
        self._ready_pending_turns = 0
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._active_origins: dict[
            tuple[str | None, str],
            tuple[AgentRequest, int],
        ] = {}
        self._active_follow_ups_by_actor: dict[
            tuple[str | None, str, str, str],
            int,
        ] = {}

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
        turn_slots = await self._admit(request, on_progress=on_progress)
        try:
            lock = self._conversation_locks.setdefault(
                request.conversation_id,
                asyncio.Lock(),
            )
            async with lock:
                cached = await self.store.completed_response(request.event_id)
                if cached is not None:
                    return cached
                conversation = await self.store.conversation(request.conversation_id)
                provider_thread_id = (
                    conversation.provider_thread_id if conversation is not None else None
                )
                continuity_reset_reason: str | None = None
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
                memory_context: tuple[AgentMemoryRecord, ...] = ()
                if self.memory is not None and AGENT_MEMORY_GRANT in request.grants:
                    try:
                        memory_context = await self.memory.context_for_turn(context)
                    except Exception as exc:
                        await self.journal.append(
                            kind="agent.memory.context_failed",
                            actor_id=request.actor_id,
                            workspace_id=request.workspace_id,
                            transport="agent",
                            request_id=request.event_id,
                            payload={"error_type": type(exc).__name__},
                        )
                if on_progress is not None:
                    await on_progress(AgentProgressUpdate(AgentProgressStage.STARTING))
                origin_key = (request.workspace_id, request.channel_id)
                async with self._admission_lock:
                    self._active_origins[origin_key] = (request, 0)
                try:
                    result = await self.provider.respond(
                        provider_thread_id=provider_thread_id,
                        event_prompt=_event_prompt(
                            request,
                            continuity_reset_reason=continuity_reset_reason,
                            memory_context=memory_context,
                        ),
                        context=context,
                        on_progress=on_progress,
                    )
                except AgentThreadError:
                    await self.store.rotate(request.conversation_id, model=self.model)
                    result = await self.provider.respond(
                        provider_thread_id=None,
                        event_prompt=_event_prompt(
                            request,
                            continuity_reset_reason="saved_thread_unavailable",
                            memory_context=memory_context,
                        ),
                        context=context,
                        on_progress=on_progress,
                    )
                finally:
                    async with self._admission_lock:
                        active = self._active_origins.get(origin_key)
                        if active is not None and active[0].event_id == request.event_id:
                            self._active_origins.pop(origin_key, None)
                            self._clear_follow_up_counts(
                                origin_key,
                                original_event_id=request.event_id,
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
            failure_payload: dict[str, object] = {
                "conversation_id": request.conversation_id,
                "trigger": request.trigger.value,
                "model": self.model,
                "error_type": type(exc).__name__,
            }
            if isinstance(exc, AgentTimeoutError):
                failure_payload["inactivity_timeout"] = {
                    "seconds": exc.timeout_seconds,
                    "automatic_retry_attempted": exc.auto_retry_attempted,
                    "runtime_restarted": exc.runtime_restarted,
                    "non_idempotent_write_attempted": exc.write_attempted,
                    "diagnostic": exc.diagnostic,
                }
            elif isinstance(exc, AgentProviderError) and exc.diagnostic:
                failure_payload["provider_diagnostic"] = exc.diagnostic
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
                payload=failure_payload,
            )
            raise
        finally:
            await self._release(turn_slots)

    async def close(self) -> None:
        await self.provider.close()

    async def try_follow_up(self, request: AgentRequest) -> str | None:
        """Steer a turn and return its original event ID when accepted."""

        if not isinstance(self.provider, SteerableAgentProvider):
            return None
        origin_key = (request.workspace_id, request.channel_id)
        async with self._admission_lock:
            active = self._active_origins.get(origin_key)
            if active is None:
                return None
            original, follow_up_count = active
            if original.trigger is not AgentTrigger.MENTION:
                # An explicit user request must never disappear into an
                # autonomous observation that is allowed to return NO_ACTION.
                return None
            if request.grants != original.grants or request.approvals != original.approvals:
                # The active provider thread exposes the original capability
                # profile. Queue a separate turn so a stronger or different
                # contributor receives exactly their own tool surface.
                return None
            if follow_up_count >= self.limits.max_pending_turns:
                raise AgentBusyError("The bounded agent follow-up queue is full.")
            actor_key = (
                *origin_key,
                original.event_id,
                request.actor_id,
            )
            actor_follow_ups = self._active_follow_ups_by_actor.get(actor_key, 0)
            if actor_follow_ups >= self.limits.max_pending_turns_per_user:
                raise AgentBusyError("The bounded per-user follow-up queue is full.")
            self._active_origins[origin_key] = (original, follow_up_count + 1)
            self._active_follow_ups_by_actor[actor_key] = actor_follow_ups + 1
        accepted = False
        try:
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
            accepted = await self.provider.steer(
                event_prompt=_follow_up_prompt(
                    request,
                    original_actor_id=original.actor_id,
                ),
                context=context,
            )
            if accepted:
                await self.journal.append(
                    kind="agent.turn.steered",
                    actor_id=request.actor_id,
                    workspace_id=request.workspace_id,
                    transport="agent",
                    request_id=request.event_id,
                    payload={
                        "conversation_id": original.conversation_id,
                        "original_actor_id": original.actor_id,
                        "follow_up_actor_id": request.actor_id,
                        "same_actor": request.actor_id == original.actor_id,
                    },
                )
            return original.event_id if accepted else None
        finally:
            if not accepted:
                async with self._admission_lock:
                    active = self._active_origins.get(origin_key)
                    if active is not None and active[0].event_id == original.event_id:
                        self._active_origins[origin_key] = (
                            original,
                            max(0, active[1] - 1),
                        )
                        self._decrement_follow_up_actor(actor_key)

    async def _admit(
        self,
        request: AgentRequest,
        *,
        on_progress: AgentProgressCallback | None,
    ) -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        slot_key = request.workspace_id or f"dm:{request.channel_id}"
        turn_slot: asyncio.Semaphore | None = None
        turn_begun = False
        pending_reserved = False
        workspace_waiter_counted = False
        workspace_acquired = False
        active_acquired = False
        active_counted = False
        ready_pending_counted = False
        queued = False
        queue_position: int | None = None
        try:
            # Reject rate-limited work before it can consume either an active
            # or pending slot. Budget check, durable begin, and the selected
            # reservation stay atomic across arriving workspaces.
            async with self._admission_lock, self._budget_lock:
                await self._check_budgets(request)
                turn_slot = self._workspace_turn_slots.setdefault(
                    slot_key,
                    asyncio.Semaphore(1),
                )
                activate_immediately = (
                    not turn_slot.locked()
                    and self._active_turns + self._ready_pending_turns
                    < self.limits.max_active_turns
                )
                actor_pending = self._pending_turns_by_actor.get(
                    request.actor_id,
                    0,
                )
                rejection_reason: str | None = None
                if not activate_immediately:
                    if self._pending_turns >= self.limits.max_pending_turns:
                        rejection_reason = "queue_full"
                    elif actor_pending >= self.limits.max_pending_turns_per_user:
                        rejection_reason = "user_queue_full"
                if rejection_reason is not None:
                    await self.journal.append(
                        kind="agent.turn.rejected",
                        actor_id=request.actor_id,
                        workspace_id=request.workspace_id,
                        transport="agent",
                        request_id=request.event_id,
                        payload={
                            "reason": rejection_reason,
                            "trigger": request.trigger.value,
                        },
                    )
                    raise AgentBusyError("The bounded agent turn queue is full.")
                if activate_immediately:
                    # Every active-slot claim and release is serialized by
                    # admission_lock. Both acquires are therefore immediately
                    # available after the counter and workspace checks.
                    await turn_slot.acquire()
                    workspace_acquired = True
                    await self._active_turn_slots.acquire()
                    active_acquired = True
                    self._active_turns += 1
                    active_counted = True
                else:
                    self._pending_turns += 1
                    self._pending_turns_by_actor[request.actor_id] = actor_pending + 1
                    pending_reserved = True
                    queued = True
                    queue_position = self._workspace_waiters.get(slot_key, 0) + 1
                    self._workspace_waiters[slot_key] = queue_position
                    workspace_waiter_counted = True
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
                turn_begun = True

            if not active_counted:
                if queued and on_progress is not None:
                    await on_progress(
                        AgentProgressUpdate(
                            AgentProgressStage.QUEUED,
                            queue_position=queue_position,
                        )
                    )
                await turn_slot.acquire()
                workspace_acquired = True
                async with self._admission_condition:
                    if workspace_waiter_counted:
                        self._decrement_waiter(slot_key)
                        workspace_waiter_counted = False
                    self._ready_pending_turns += 1
                    ready_pending_counted = True
                    while self._active_turns >= self.limits.max_active_turns:
                        await self._admission_condition.wait()
                    self._decrement_pending(request.actor_id)
                    pending_reserved = False
                    self._ready_pending_turns -= 1
                    ready_pending_counted = False
                    self._active_turns += 1
                    active_counted = True
                    await self._active_turn_slots.acquire()
                    active_acquired = True
        except BaseException:

            async def cleanup_failed_admission() -> None:
                async with self._admission_condition:
                    if active_counted:
                        self._decrement_active()
                        if active_acquired:
                            self._active_turn_slots.release()
                        self._admission_condition.notify_all()
                    elif pending_reserved:
                        self._decrement_pending(request.actor_id)
                    if ready_pending_counted:
                        self._ready_pending_turns -= 1
                    if workspace_waiter_counted:
                        self._decrement_waiter(slot_key)
                if workspace_acquired and turn_slot is not None:
                    turn_slot.release()
                if turn_begun:
                    await self.store.fail(
                        request,
                        model=self.model,
                        error_type="AdmissionCancelled",
                    )

            await _finish_cleanup(cleanup_failed_admission())
            raise
        if turn_slot is None:
            raise RuntimeError("Agent turn admission did not initialize a workspace slot.")
        return turn_slot, self._active_turn_slots

    async def _release(
        self,
        turn_slots: tuple[asyncio.Semaphore, asyncio.Semaphore],
    ) -> None:
        await _finish_cleanup(self._release_turn_slots(turn_slots))

    async def _release_turn_slots(
        self,
        turn_slots: tuple[asyncio.Semaphore, asyncio.Semaphore],
    ) -> None:
        workspace_slot, active_slot = turn_slots
        async with self._admission_condition:
            self._decrement_active()
            active_slot.release()
            workspace_slot.release()
            self._admission_condition.notify_all()

    def _decrement_active(self) -> None:
        self._active_turns -= 1

    def _decrement_pending(self, actor_id: str) -> None:
        self._pending_turns -= 1
        remaining = self._pending_turns_by_actor.get(actor_id, 0) - 1
        if remaining > 0:
            self._pending_turns_by_actor[actor_id] = remaining
        else:
            self._pending_turns_by_actor.pop(actor_id, None)

    def _decrement_waiter(self, slot_key: str) -> None:
        remaining = self._workspace_waiters.get(slot_key, 0) - 1
        if remaining > 0:
            self._workspace_waiters[slot_key] = remaining
        else:
            self._workspace_waiters.pop(slot_key, None)

    def _decrement_follow_up_actor(
        self,
        key: tuple[str | None, str, str, str],
    ) -> None:
        remaining = self._active_follow_ups_by_actor.get(key, 0) - 1
        if remaining > 0:
            self._active_follow_ups_by_actor[key] = remaining
        else:
            self._active_follow_ups_by_actor.pop(key, None)

    def _clear_follow_up_counts(
        self,
        origin_key: tuple[str | None, str],
        *,
        original_event_id: str,
    ) -> None:
        for key in tuple(self._active_follow_ups_by_actor):
            if key[:2] == origin_key and key[2] == original_event_id:
                self._active_follow_ups_by_actor.pop(key, None)

    async def _check_budgets(self, request: AgentRequest) -> None:
        if request.actor_id in self.limits.rate_limit_exempt_actor_ids:
            return
        now = datetime.now(UTC)
        if request.trigger is not AgentTrigger.AUTONOMOUS:
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
        if request.workspace_id is not None and request.trigger in {
            AgentTrigger.MENTION,
            AgentTrigger.AUTONOMOUS,
        }:
            workspace_count, workspace_oldest = await self.store.request_window(
                actor_id=None,
                workspace_id=request.workspace_id,
                since=now - timedelta(seconds=self.limits.per_workspace_window_seconds),
                excluded_actor_ids=self.limits.rate_limit_exempt_actor_ids,
                included_triggers=frozenset({request.trigger}),
            )
            if workspace_count >= self.limits.per_workspace_requests:
                raise AgentRateLimitError(
                    (
                        "The autonomous server agent budget is exhausted."
                        if request.trigger is AgentTrigger.AUTONOMOUS
                        else "The server agent budget is exhausted."
                    ),
                    retry_after_seconds=_retry_after_seconds(
                        now,
                        workspace_oldest,
                        self.limits.per_workspace_window_seconds,
                    ),
                )
        usage, token_release_anchor = await self.store.token_budget_window(
            now - timedelta(hours=24),
            limit=self.limits.max_tokens_per_24_hours,
            excluded_actor_ids=self.limits.rate_limit_exempt_actor_ids,
        )
        if usage >= self.limits.max_tokens_per_24_hours:
            raise AgentRateLimitError(
                "The rolling agent token budget is exhausted.",
                retry_after_seconds=_retry_after_seconds(
                    now,
                    token_release_anchor,
                    24 * 60 * 60,
                ),
            )


async def _finish_cleanup(awaitable: Awaitable[None]) -> None:
    """Finish mandatory cleanup before preserving an arriving cancellation."""

    cleanup_task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    cleanup_task.result()
    if cancellation is not None:
        raise cancellation


def _event_prompt(
    request: AgentRequest,
    *,
    continuity_reset_reason: str | None = None,
    memory_context: tuple[AgentMemoryRecord, ...] = (),
) -> str:
    message_id = request.message_id or "none"
    workspace_id = request.workspace_id or "direct-message"
    event_pointers = tuple(
        json.dumps(
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "occurred_at": event.occurred_at.isoformat(),
                "workspace_id": event.workspace_id,
                "payload": event.payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        for event in request.events
    )
    memory_pointers = tuple(
        json.dumps(
            {
                "basis": memory.basis.value,
                "confidence": memory.confidence,
                "key": memory.key,
                "memory_id": memory.memory_id,
                "source_message_locators": [
                    {
                        "channel_id": locator.channel_id,
                        "guild_id": locator.guild_id,
                        "message_id": locator.message_id,
                    }
                    for locator in memory.source_message_locators
                ],
                "summary": memory.summary,
                "updated_at": memory.updated_at.isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for memory in memory_context
    )
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
            f"batched_event_count={len(event_pointers)}",
            *(f"batched_event={pointer}" for pointer in event_pointers),
            f"requester_memory_count={len(memory_pointers)}",
            *(f"requester_memory={pointer}" for pointer in memory_pointers),
            *(
                (
                    (
                        "Requester memories are host-scoped durable context for this "
                        "actor only. Use a memory only when relevant. Its basis and "
                        "source locators are provenance, not current authorization; "
                        "describe user_stated evidence honestly and verify facts that "
                        "may have changed."
                    ),
                )
                if memory_pointers
                else ()
            ),
            (
                "No message body is included. Inspect only what you need through bounded "
                "Simajilord tools, then produce one user-facing response."
            ),
            (
                "Each batched event keeps its own source_actor_id in the pointer. These "
                "contributors are context only: never borrow a source user's identity, "
                "permissions, grants, or approvals for an autonomous write."
            ),
            *(
                (
                    f"continuity_reset_reason={continuity_reset_reason}",
                    (
                        "The provider conversation was reset. Do not pretend to retain "
                        "unseen context. Recover only the minimum needed context through "
                        "bounded Discord reads and sourced memory before continuing."
                    ),
                )
                if continuity_reset_reason is not None
                else ()
            ),
        )
    )


def _follow_up_prompt(
    request: AgentRequest,
    *,
    original_actor_id: str,
) -> str:
    message_id = request.message_id or "none"
    same_actor = request.actor_id == original_actor_id
    return "\n".join(
        (
            "SIMAJILORD_FOLLOW_UP_V1",
            f"event_id={request.event_id}",
            f"workspace_id={request.workspace_id or 'direct-message'}",
            f"channel_id={request.channel_id}",
            f"message_id={message_id}",
            f"actor_id={request.actor_id}",
            f"actor_name={request.actor_name}",
            f"same_actor_as_original={'true' if same_actor else 'false'}",
            (
                "A user sent this while the current task was running. Read the exact "
                "Discord message through the bounded message tool and incorporate it "
                "before finishing."
            ),
            (
                "This pointer contains no message body. If same_actor_as_original is "
                "false, identify the contributor separately. They may authorize actions "
                "only through this accepted follow-up's host-issued authorization handle "
                "and their own Discord permissions; never borrow the original actor's "
                "authority."
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
