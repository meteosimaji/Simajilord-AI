"""Durable agent conversation identity, idempotency, and usage accounting."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .contracts import (
    AGENT_NO_ACTION_CONTENT,
    AgentRequest,
    AgentResponse,
    AgentResponseStatus,
    AgentTaskRouteDecision,
    AgentTaskRouteResult,
    AgentTokenUsage,
    AgentTrigger,
    is_agent_public_reference_id,
    is_agent_task_id,
    new_agent_public_reference_id,
    new_agent_task_id,
)

_IN_PROGRESS_STATUS = "in_progress"
log = logging.getLogger(__name__)


class AgentTaskRouteUnavailableError(RuntimeError):
    """The selected active task became terminal before route commit."""


@dataclass(frozen=True, slots=True)
class AgentConversationRecord:
    """Provider state for one transport-neutral conversation."""

    conversation_id: str
    provider_thread_id: str | None
    model: str
    generation: int
    turn_count: int
    last_input_tokens: int
    model_context_window: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentPendingHostDelivery:
    """One completed turn whose Discord response is not durably delivered."""

    event_id: str
    public_reference_id: str
    actor_id: str
    workspace_id: str | None
    channel_id: str
    source_message_id: str | None
    response_content: str
    occurred_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class AgentInterruptedMention:
    """One explicit mention left active when an earlier process stopped."""

    event_id: str
    public_reference_id: str
    task_id: str
    channel_id: str
    source_message_id: str
    occurred_at: datetime
    started_at: datetime


@dataclass(frozen=True, slots=True)
class AgentHostDeliveryRecord:
    """Body-free delivery evidence for one Discord response chunk."""

    event_id: str
    purpose: str
    chunk_index: int
    content_sha256: str
    channel_id: str
    message_id: str | None
    receipted_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentRequestRecord:
    """Body-free request metadata addressable by a public reference ID."""

    public_reference_id: str
    task_id: str
    event_id: str
    conversation_id: str
    trigger: AgentTrigger
    actor_id: str
    workspace_id: str | None
    channel_id: str
    source_message_id: str | None
    model: str
    status: str
    provider_thread_id: str | None
    error_type: str | None
    occurred_at: datetime
    started_at: datetime
    completed_at: datetime | None
    host_delivered_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentTaskSnapshot:
    """Bounded task, route, execution, and delivery metadata for operations UI."""

    task_id: str
    public_reference_id: str
    root_event_id: str
    execution_event_id: str
    actor_id: str
    workspace_id: str | None
    channel_id: str
    state: str
    completion_reason: str | None
    routed_task_id: str | None
    route_decision: str
    route_reason: str | None
    event_count: int
    attached_event_count: int
    request_status: str | None
    model: str | None
    provider_thread_id: str | None
    error_type: str | None
    delivery_count: int
    receipted_delivery_count: int
    started_at: datetime | None
    completed_at: datetime | None
    host_delivered_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentUnroutedTaskCandidate:
    """Pointer-only candidate whose route was not safely applied before restart."""

    event_id: str
    task_id: str
    public_reference_id: str
    channel_id: str
    source_message_id: str
    occurred_at: datetime
    created_at: datetime


class AgentConversationStore:
    """SQLite authority for restart-safe agent state without user message bodies."""

    def __init__(self, path: Path, *, compatibility_epoch: int = 4) -> None:
        if compatibility_epoch < 1 or compatibility_epoch > 10_000:
            raise ValueError("compatibility epoch must be between 1 and 10000")
        self.path = path
        self.compatibility_epoch = compatibility_epoch
        self.compatibility_reset_count = 0
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def conversation(self, conversation_id: str) -> AgentConversationRecord | None:
        async with self._lock:
            return await asyncio.to_thread(self._select_conversation, conversation_id)

    async def promote_compatible_conversation(
        self,
        conversation_id: str,
    ) -> str | None:
        """Move a lower-grant thread into a compatible expanded grant profile."""

        async with self._lock:
            return await asyncio.to_thread(
                self._promote_compatible_conversation,
                conversation_id,
            )

    async def completed_response(self, event_id: str) -> AgentResponse | None:
        """Return a durable terminal response so completed/cancelled events never rerun."""

        async with self._lock:
            return await asyncio.to_thread(self._select_completed_response, event_id)

    async def public_reference_id_for_event(self, event_id: str) -> str | None:
        """Return the persisted public reference without exposing request bodies."""

        async with self._lock:
            return await asyncio.to_thread(
                self._select_public_reference_id_for_event,
                event_id,
            )

    async def task_id_for_event(self, event_id: str) -> str | None:
        """Return the candidate/root task identity persisted for one event."""

        async with self._lock:
            return await asyncio.to_thread(self._select_task_id_for_event, event_id)

    async def task_snapshot_by_public_reference_id(
        self,
        public_reference_id: str,
    ) -> AgentTaskSnapshot | None:
        if not is_agent_public_reference_id(public_reference_id):
            raise ValueError("invalid agent public reference ID")
        async with self._lock:
            return await asyncio.to_thread(
                self._task_snapshot_by_public_reference_id,
                public_reference_id,
            )

    async def route_for_event(
        self,
        event_id: str,
    ) -> AgentTaskRouteResult | None:
        """Return every committed route once its required provider apply completed."""

        normalized_event_id = event_id.strip()
        if not normalized_event_id or len(normalized_event_id) > 500:
            raise ValueError("agent event ID must be bounded and non-empty")
        async with self._lock:
            return await asyncio.to_thread(
                self._route_for_event,
                normalized_event_id,
            )

    async def bind_provider_thread(
        self,
        *,
        event_id: str,
        task_id: str,
        conversation_id: str,
        provider_thread_id: str,
        model: str,
    ) -> bool:
        """Persist an in-progress task/thread binding before provider input starts."""

        async with self._lock:
            return await asyncio.to_thread(
                self._bind_provider_thread,
                event_id,
                task_id,
                conversation_id,
                provider_thread_id,
                model,
            )

    async def record_task_candidate(
        self,
        original: AgentRequest,
        candidate: AgentRequest,
    ) -> bool:
        """Commit a candidate before asking the active model to classify it."""

        async with self._lock:
            return await asyncio.to_thread(
                self._record_task_candidate,
                original,
                candidate,
            )

    async def route_task_candidate(
        self,
        candidate_event_id: str,
        *,
        decision: AgentTaskRouteDecision,
        active_task_id: str,
        reason: str,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._route_task_candidate,
                candidate_event_id,
                decision,
                active_task_id,
                reason,
            )

    async def mark_task_candidate_provider_applied(
        self,
        candidate_event_id: str,
        *,
        decision: AgentTaskRouteDecision,
        active_task_id: str,
    ) -> bool:
        """Mark a selected route only after the active provider applied it."""

        async with self._lock:
            return await asyncio.to_thread(
                self._mark_task_candidate_provider_applied,
                candidate_event_id,
                decision,
                active_task_id,
            )

    async def cancel_routed_task(
        self,
        candidate_event_id: str,
        *,
        active_request: AgentRequest,
        model: str,
    ) -> bool:
        """Atomically apply a typed cancel route and close its active request."""

        async with self._lock:
            return await asyncio.to_thread(
                self._cancel_routed_task,
                candidate_event_id,
                active_request,
                model,
            )

    async def default_task_candidate_to_separate(
        self,
        candidate_event_id: str,
        *,
        reason: str,
    ) -> bool:
        """Recover a candidate that never reached confirmed provider state."""

        async with self._lock:
            return await asyncio.to_thread(
                self._default_task_candidate_to_separate,
                candidate_event_id,
                reason,
            )

    async def unrouted_task_candidates(
        self,
        *,
        created_before: datetime,
        limit: int = 100,
    ) -> tuple[AgentUnroutedTaskCandidate, ...]:
        if created_before.tzinfo is None:
            raise ValueError("candidate cutoff must be timezone-aware")
        if limit < 1 or limit > 1_000:
            raise ValueError("candidate limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(
                self._unrouted_task_candidates,
                created_before.astimezone(UTC).isoformat(),
                limit,
            )

    async def fail_unrouted_task_candidate(
        self,
        event_id: str,
        *,
        error_type: str,
    ) -> bool:
        """Terminalize a separate candidate that cannot be recovered from Discord."""

        async with self._lock:
            return await asyncio.to_thread(
                self._fail_unrouted_task_candidate,
                event_id,
                error_type,
            )

    async def request_by_public_reference_id(
        self,
        public_reference_id: str,
    ) -> AgentRequestRecord | None:
        """Resolve one opaque support identifier to bounded request metadata."""

        if not is_agent_public_reference_id(public_reference_id):
            raise ValueError("invalid agent public reference ID")
        async with self._lock:
            return await asyncio.to_thread(
                self._select_request_by_public_reference_id,
                public_reference_id,
            )

    async def pending_host_deliveries(
        self,
        *,
        limit: int = 100,
    ) -> tuple[AgentPendingHostDelivery, ...]:
        """Return completed turns whose host delivery still needs reconciliation."""

        if limit < 1 or limit > 1_000:
            raise ValueError("host delivery query limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(self._pending_host_deliveries, limit)

    async def pending_host_delivery(
        self,
        event_id: str,
    ) -> AgentPendingHostDelivery | None:
        async with self._lock:
            return await asyncio.to_thread(
                self._pending_host_delivery,
                event_id,
            )

    async def interrupted_mentions(
        self,
        *,
        started_after: datetime,
        started_before: datetime,
        limit: int = 100,
    ) -> tuple[AgentInterruptedMention, ...]:
        """Return prior-process mentions that never reached a terminal state."""

        if started_after.tzinfo is None or started_before.tzinfo is None:
            raise ValueError("interrupted mention cutoffs must be timezone-aware")
        if started_after >= started_before:
            raise ValueError("started_after must be earlier than started_before")
        if limit < 1 or limit > 1_000:
            raise ValueError("interrupted mention limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(
                self._interrupted_mentions,
                started_after.astimezone(UTC).isoformat(),
                started_before.astimezone(UTC).isoformat(),
                limit,
            )

    async def fail_interrupted_mention(
        self,
        event_id: str,
        *,
        error_type: str,
    ) -> bool:
        """Close an unrecoverable interrupted mention without touching terminal rows."""

        async with self._lock:
            return await asyncio.to_thread(
                self._fail_interrupted_mention,
                event_id,
                error_type,
            )

    async def plan_host_delivery(
        self,
        *,
        event_id: str,
        purpose: str,
        channel_id: str,
        contents: tuple[str, ...],
    ) -> tuple[AgentHostDeliveryRecord, ...]:
        """Persist body-free chunk intents before any Discord send."""

        async with self._lock:
            return await asyncio.to_thread(
                self._plan_host_delivery,
                event_id,
                purpose,
                channel_id,
                contents,
            )

    async def host_delivery_records(
        self,
        *,
        event_id: str,
        purpose: str,
    ) -> tuple[AgentHostDeliveryRecord, ...]:
        async with self._lock:
            return await asyncio.to_thread(
                self._host_delivery_records,
                event_id,
                purpose,
            )

    async def record_host_delivery_message(
        self,
        *,
        event_id: str,
        purpose: str,
        chunk_index: int,
        message_id: str,
    ) -> AgentHostDeliveryRecord:
        """Persist the Discord message ID without replacing prior evidence."""

        async with self._lock:
            return await asyncio.to_thread(
                self._record_host_delivery_message,
                event_id,
                purpose,
                chunk_index,
                message_id,
            )

    async def mark_host_delivery_receipted(
        self,
        *,
        event_id: str,
        purpose: str,
        chunk_index: int,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._mark_host_delivery_receipted,
                event_id,
                purpose,
                chunk_index,
            )

    async def complete_host_delivery(
        self,
        event_id: str,
        *,
        allow_empty: bool = False,
    ) -> bool:
        """Mark a response delivered only after every planned chunk is receipted."""

        async with self._lock:
            return await asyncio.to_thread(
                self._complete_host_delivery,
                event_id,
                allow_empty,
            )

    async def begin(self, request: AgentRequest, *, model: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._begin, request, model)

    async def complete(
        self,
        request: AgentRequest,
        response: AgentResponse,
    ) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._complete, request, response)

    async def fail(
        self,
        request: AgentRequest,
        *,
        model: str,
        error_type: str,
    ) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._fail, request, model, error_type)

    async def cancel(
        self,
        request: AgentRequest,
        *,
        model: str,
        reason: str = "user_requested",
    ) -> bool:
        async with self._lock:
            return await asyncio.to_thread(self._cancel, request, model, reason)

    async def rotate(self, conversation_id: str, *, model: str) -> None:
        """Forget only the provider thread while preserving conversation identity."""

        async with self._lock:
            await asyncio.to_thread(self._rotate, conversation_id, model)

    async def reset_provider_continuity(
        self,
        conversation_ids: tuple[str, ...] | None = None,
    ) -> int:
        """Clear provider bindings only, preserving requests and delivery evidence."""

        normalized_ids: tuple[str, ...] | None = None
        if conversation_ids is not None:
            normalized_ids = tuple(
                dict.fromkeys(
                    conversation_id.strip()
                    for conversation_id in conversation_ids
                    if conversation_id.strip()
                )
            )
            if not normalized_ids:
                raise ValueError("at least one conversation ID is required")
            if any(len(conversation_id) > 500 for conversation_id in normalized_ids):
                raise ValueError("conversation IDs must be bounded")
        async with self._lock:
            return await asyncio.to_thread(
                self._reset_provider_continuity,
                normalized_ids,
            )

    async def prune(self, *, before: datetime) -> tuple[int, int]:
        """Remove old request accounting and then unreferenced old conversations."""

        if before.tzinfo is None:
            raise ValueError("Retention cutoffs must be timezone-aware.")
        async with self._lock:
            return await asyncio.to_thread(
                self._prune,
                before.astimezone(UTC).isoformat(),
            )

    async def request_window(
        self,
        *,
        actor_id: str | None,
        workspace_id: str | None,
        since: datetime,
        excluded_actor_ids: frozenset[str] = frozenset(),
        included_triggers: frozenset[AgentTrigger] = frozenset(),
    ) -> tuple[int, datetime | None]:
        async with self._lock:
            return await asyncio.to_thread(
                self._request_window,
                actor_id,
                workspace_id,
                since,
                excluded_actor_ids,
                included_triggers,
            )

    async def token_budget_window(
        self,
        since: datetime,
        *,
        limit: int,
        excluded_actor_ids: frozenset[str] = frozenset(),
    ) -> tuple[int, datetime | None]:
        """Return non-cached usage and the oldest expiry that drops it below limit."""

        if since.tzinfo is None:
            raise ValueError("token budget cutoff must be timezone-aware")
        if limit < 1:
            raise ValueError("token budget limit must be positive")
        async with self._lock:
            return await asyncio.to_thread(
                self._token_budget_window,
                since,
                limit,
                excluded_actor_ids,
            )

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    provider_thread_id TEXT,
                    model TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    last_input_tokens INTEGER NOT NULL DEFAULT 0,
                    model_context_window INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_requests (
                    event_id TEXT PRIMARY KEY,
                    public_reference_id TEXT NOT NULL UNIQUE,
                    task_id TEXT,
                    conversation_id TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    channel_id TEXT NOT NULL,
                    message_id TEXT,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider_thread_id TEXT,
                    response_content TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    model_context_window INTEGER,
                    error_type TEXT,
                    occurred_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    host_delivered_at TEXT,
                    FOREIGN KEY(conversation_id)
                        REFERENCES agent_conversations(conversation_id)
                );

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    public_reference_id TEXT NOT NULL UNIQUE,
                    root_event_id TEXT NOT NULL UNIQUE,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    channel_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    completion_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_task_events (
                    event_id TEXT PRIMARY KEY,
                    public_reference_id TEXT NOT NULL UNIQUE,
                    candidate_task_id TEXT NOT NULL,
                    routed_task_id TEXT,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    channel_id TEXT NOT NULL,
                    message_id TEXT,
                    route_decision TEXT NOT NULL,
                    route_reason TEXT,
                    occurred_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    routed_at TEXT,
                    provider_applied_at TEXT,
                    FOREIGN KEY(candidate_task_id)
                        REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    FOREIGN KEY(routed_task_id)
                        REFERENCES agent_tasks(task_id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runtime_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_host_deliveries (
                    event_id TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    message_id TEXT,
                    receipted_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(event_id, purpose, chunk_index),
                    FOREIGN KEY(event_id)
                        REFERENCES agent_requests(event_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS agent_requests_actor_started
                    ON agent_requests(actor_id, started_at);
                CREATE INDEX IF NOT EXISTS agent_requests_workspace_started
                    ON agent_requests(workspace_id, started_at);
                CREATE INDEX IF NOT EXISTS agent_requests_completed
                    ON agent_requests(completed_at);
                CREATE INDEX IF NOT EXISTS agent_task_events_routed
                    ON agent_task_events(routed_task_id, occurred_at);
                CREATE INDEX IF NOT EXISTS agent_task_events_pending
                    ON agent_task_events(route_decision, created_at);
                CREATE INDEX IF NOT EXISTS agent_tasks_state_updated
                    ON agent_tasks(state, updated_at);
                """
            )
            request_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(agent_requests)")
            }
            if "host_delivered_at" not in request_columns:
                connection.execute(
                    "ALTER TABLE agent_requests ADD COLUMN host_delivered_at TEXT"
                )
                # Pre-outbox responses were delivered synchronously. Treat them
                # as terminal so the first upgraded start never reposts history.
                connection.execute(
                    """
                    UPDATE agent_requests
                    SET host_delivered_at = completed_at
                    WHERE status = ? AND completed_at IS NOT NULL
                    """,
                    (AgentResponseStatus.COMPLETED.value,),
                )
            if "public_reference_id" not in request_columns:
                connection.execute(
                    "ALTER TABLE agent_requests ADD COLUMN public_reference_id TEXT"
                )
            if "task_id" not in request_columns:
                connection.execute(
                    "ALTER TABLE agent_requests ADD COLUMN task_id TEXT"
                )
            task_event_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(agent_task_events)")
            }
            if "provider_applied_at" not in task_event_columns:
                connection.execute(
                    "ALTER TABLE agent_task_events ADD COLUMN provider_applied_at TEXT"
                )
                # Routes written by a pre-handshake build had no pending/apply
                # distinction. Preserve them as already applied during migration.
                connection.execute(
                    """
                    UPDATE agent_task_events
                    SET provider_applied_at = routed_at
                    WHERE route_decision != 'candidate'
                      AND provider_applied_at IS NULL
                    """
                )
            existing_references = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT public_reference_id
                    FROM agent_requests
                    WHERE public_reference_id IS NOT NULL
                      AND public_reference_id != ''
                    """
                )
            }
            missing_reference_rows = connection.execute(
                """
                SELECT event_id
                FROM agent_requests
                WHERE public_reference_id IS NULL OR public_reference_id = ''
                ORDER BY event_id
                """
            ).fetchall()
            for (event_id,) in missing_reference_rows:
                reference_id = new_agent_public_reference_id()
                while reference_id in existing_references:
                    reference_id = new_agent_public_reference_id()
                connection.execute(
                    """
                    UPDATE agent_requests
                    SET public_reference_id = ?
                    WHERE event_id = ?
                    """,
                    (reference_id, event_id),
                )
                existing_references.add(reference_id)
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_requests_public_reference
                ON agent_requests(public_reference_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_requests_host_pending
                ON agent_requests(status, host_delivered_at, completed_at)
                """
            )
            existing_task_ids = {
                str(row[0])
                for row in connection.execute("SELECT task_id FROM agent_tasks")
            }
            missing_task_rows = connection.execute(
                """
                SELECT event_id, public_reference_id, actor_id, workspace_id,
                       channel_id, status, error_type, occurred_at, started_at,
                       completed_at
                FROM agent_requests
                WHERE task_id IS NULL OR task_id = ''
                ORDER BY started_at, event_id
                """
            ).fetchall()
            for row in missing_task_rows:
                task_id = new_agent_task_id()
                while task_id in existing_task_ids:
                    task_id = new_agent_task_id()
                existing_task_ids.add(task_id)
                status = str(row[5])
                state = _task_state_for_request_status(status)
                updated_at = str(row[9] or row[8])
                connection.execute(
                    """
                    INSERT INTO agent_tasks(
                        task_id, public_reference_id, root_event_id, actor_id,
                        workspace_id, channel_id, state, created_at, updated_at,
                        completed_at, completion_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        row[1],
                        row[0],
                        row[2],
                        row[3],
                        row[4],
                        state,
                        row[8],
                        updated_at,
                        row[9],
                        row[6],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO agent_task_events(
                        event_id, public_reference_id, candidate_task_id,
                        routed_task_id, actor_id, workspace_id, channel_id,
                        message_id, route_decision, route_reason, occurred_at,
                        created_at, routed_at, provider_applied_at
                    )
                    SELECT event_id, public_reference_id, ?, ?, actor_id,
                           workspace_id, channel_id, message_id, 'root', NULL,
                           occurred_at, started_at, started_at, started_at
                    FROM agent_requests WHERE event_id = ?
                    """,
                    (task_id, task_id, row[0]),
                )
                connection.execute(
                    "UPDATE agent_requests SET task_id = ? WHERE event_id = ?",
                    (task_id, row[0]),
                )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS agent_requests_task
                ON agent_requests(task_id)
                """
            )
            metadata_now = datetime.now(UTC).isoformat()
            epoch_row = connection.execute(
                "SELECT value FROM agent_runtime_metadata WHERE key = ?",
                ("conversation_compatibility_epoch",),
            ).fetchone()
            if epoch_row is None:
                connection.execute(
                    """
                    INSERT INTO agent_runtime_metadata(key, value, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (
                        "conversation_compatibility_epoch",
                        str(self.compatibility_epoch),
                        metadata_now,
                    ),
                )
            elif str(epoch_row[0]) != str(self.compatibility_epoch):
                cursor = connection.execute(
                    """
                    UPDATE agent_conversations
                    SET provider_thread_id = NULL, generation = generation + 1,
                        turn_count = 0, last_input_tokens = 0,
                        model_context_window = NULL, updated_at = ?
                    WHERE provider_thread_id IS NOT NULL OR turn_count != 0
                       OR last_input_tokens != 0 OR model_context_window IS NOT NULL
                    """,
                    (metadata_now,),
                )
                self.compatibility_reset_count = cursor.rowcount
                connection.execute(
                    """
                    UPDATE agent_runtime_metadata
                    SET value = ?, updated_at = ? WHERE key = ?
                    """,
                    (
                        str(self.compatibility_epoch),
                        metadata_now,
                        "conversation_compatibility_epoch",
                    ),
                )
                log.warning(
                    "Agent conversation compatibility epoch changed old=%s new=%s "
                    "provider_bindings_reset=%d",
                    epoch_row[0],
                    self.compatibility_epoch,
                    self.compatibility_reset_count,
                )
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _select_conversation(
        self,
        conversation_id: str,
    ) -> AgentConversationRecord | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                "SELECT * FROM agent_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return _conversation_from_row(row) if row is not None else None
        finally:
            connection.close()

    def _select_public_reference_id_for_event(self, event_id: str) -> str | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT public_reference_id FROM agent_task_events WHERE event_id = ?
                UNION ALL
                SELECT public_reference_id FROM agent_requests WHERE event_id = ?
                LIMIT 1
                """,
                (event_id, event_id),
            ).fetchone()
            return (
                str(row["public_reference_id"])
                if row is not None and row["public_reference_id"] is not None
                else None
            )
        finally:
            connection.close()

    def _select_task_id_for_event(self, event_id: str) -> str | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT candidate_task_id AS task_id
                FROM agent_task_events WHERE event_id = ?
                UNION ALL
                SELECT task_id FROM agent_requests WHERE event_id = ?
                LIMIT 1
                """,
                (event_id, event_id),
            ).fetchone()
            return (
                str(row["task_id"])
                if row is not None and row["task_id"] is not None
                else None
            )
        finally:
            connection.close()

    def _route_for_event(
        self,
        event_id: str,
    ) -> AgentTaskRouteResult | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT events.route_decision, routed.task_id,
                       routed.root_event_id, routed.public_reference_id
                FROM agent_task_events events
                JOIN agent_tasks routed ON routed.task_id = events.routed_task_id
                WHERE events.event_id = ?
                  AND events.route_decision IN ('attach', 'separate', 'finish', 'cancel')
                  AND events.provider_applied_at IS NOT NULL
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            return AgentTaskRouteResult(
                decision=AgentTaskRouteDecision(str(row["route_decision"])),
                active_event_id=str(row["root_event_id"]),
                active_task_id=str(row["task_id"]),
                active_public_reference_id=str(row["public_reference_id"]),
            )
        finally:
            connection.close()

    def _bind_provider_thread(
        self,
        event_id: str,
        task_id: str,
        conversation_id: str,
        provider_thread_id: str,
        model: str,
    ) -> bool:
        if not is_agent_task_id(task_id):
            raise ValueError("invalid agent task ID")
        normalized_thread_id = provider_thread_id.strip()
        if not normalized_thread_id or len(normalized_thread_id) > 500:
            raise ValueError("provider thread ID must contain 1 to 500 characters")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_cursor = connection.execute(
                """
                UPDATE agent_requests
                SET provider_thread_id = ?, model = ?
                WHERE event_id = ? AND task_id = ? AND conversation_id = ?
                  AND status = ?
                """,
                (
                    normalized_thread_id,
                    model,
                    event_id,
                    task_id,
                    conversation_id,
                    _IN_PROGRESS_STATUS,
                ),
            )
            if request_cursor.rowcount != 1:
                connection.rollback()
                return False
            conversation_cursor = connection.execute(
                """
                UPDATE agent_conversations
                SET provider_thread_id = ?, model = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (normalized_thread_id, model, now, conversation_id),
            )
            if conversation_cursor.rowcount != 1:
                raise RuntimeError("agent conversation disappeared during thread binding")
            connection.commit()
            return True
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _task_snapshot_by_public_reference_id(
        self,
        public_reference_id: str,
    ) -> AgentTaskSnapshot | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT t.task_id, t.public_reference_id, t.root_event_id,
                       execution.root_event_id AS execution_event_id,
                       t.actor_id, t.workspace_id, t.channel_id, t.state,
                       t.completion_reason, root.routed_task_id,
                       root.route_decision, root.route_reason,
                       (
                           SELECT COUNT(*) FROM agent_task_events events
                           WHERE events.routed_task_id = COALESCE(
                               root.routed_task_id, t.task_id
                           )
                       ) AS event_count,
                       (
                           SELECT COUNT(*) FROM agent_task_events events
                           WHERE events.routed_task_id = COALESCE(
                               root.routed_task_id, t.task_id
                           )
                             AND events.route_decision IN ('attach', 'finish')
                       ) AS attached_event_count,
                       request.status AS request_status, request.model,
                       request.provider_thread_id, request.error_type,
                       (
                           SELECT COUNT(*) FROM agent_host_deliveries delivery
                           WHERE delivery.event_id = execution.root_event_id
                       ) AS delivery_count,
                       (
                           SELECT COUNT(*) FROM agent_host_deliveries delivery
                           WHERE delivery.event_id = execution.root_event_id
                             AND delivery.receipted_at IS NOT NULL
                       ) AS receipted_delivery_count,
                       request.started_at, request.completed_at,
                       request.host_delivered_at
                FROM agent_tasks t
                JOIN agent_task_events root ON root.event_id = t.root_event_id
                JOIN agent_tasks execution
                  ON execution.task_id = COALESCE(root.routed_task_id, t.task_id)
                LEFT JOIN agent_requests request
                  ON request.task_id = COALESCE(root.routed_task_id, t.task_id)
                WHERE t.public_reference_id = ?
                """,
                (public_reference_id,),
            ).fetchone()
            return _task_snapshot_from_row(row) if row is not None else None
        finally:
            connection.close()

    def _unrouted_task_candidates(
        self,
        created_before: str,
        limit: int,
    ) -> tuple[AgentUnroutedTaskCandidate, ...]:
        connection = _connection(self.path)
        try:
            rows = connection.execute(
                """
                SELECT events.event_id, events.candidate_task_id,
                       events.public_reference_id, events.channel_id,
                       events.message_id, events.occurred_at, events.created_at
                FROM agent_task_events events
                JOIN agent_tasks task ON task.task_id = events.candidate_task_id
                WHERE (
                        events.route_decision = 'candidate'
                        OR (
                            events.route_decision IN ('attach', 'finish', 'cancel')
                            AND events.provider_applied_at IS NULL
                        )
                        OR (
                            events.route_decision = 'separate'
                            AND NOT EXISTS (
                                SELECT 1 FROM agent_requests requests
                                WHERE requests.event_id = events.event_id
                            )
                        )
                      )
                  AND task.state IN ('candidate', 'pending', 'routed')
                  AND events.message_id IS NOT NULL
                  AND events.created_at < ?
                ORDER BY events.created_at, events.event_id
                LIMIT ?
                """,
                (created_before, limit),
            ).fetchall()
            return tuple(_unrouted_candidate_from_row(row) for row in rows)
        finally:
            connection.close()

    def _select_request_by_public_reference_id(
        self,
        public_reference_id: str,
    ) -> AgentRequestRecord | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT event_id, public_reference_id, conversation_id, trigger,
                       task_id, actor_id, workspace_id, channel_id, message_id, model,
                       status, provider_thread_id, error_type, occurred_at,
                       started_at, completed_at, host_delivered_at
                FROM agent_requests
                WHERE public_reference_id = ?
                """,
                (public_reference_id,),
            ).fetchone()
            return _request_record_from_row(row) if row is not None else None
        finally:
            connection.close()

    def _prune(self, cutoff: str) -> tuple[int, int]:
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_cursor = connection.execute(
                """
                DELETE FROM agent_requests
                WHERE COALESCE(completed_at, started_at) < ?
                  AND status != 'in_progress'
                """,
                (cutoff,),
            )
            connection.execute(
                """
                DELETE FROM agent_tasks
                WHERE completed_at IS NOT NULL AND completed_at < ?
                  AND state NOT IN ('active', 'finishing', 'pending', 'candidate')
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_requests
                      WHERE agent_requests.task_id = agent_tasks.task_id
                  )
                """,
                (cutoff,),
            )
            conversation_cursor = connection.execute(
                """
                DELETE FROM agent_conversations
                WHERE updated_at < ?
                  AND NOT EXISTS (
                      SELECT 1
                      FROM agent_requests
                      WHERE agent_requests.conversation_id =
                            agent_conversations.conversation_id
                  )
                """,
                (cutoff,),
            )
            connection.commit()
            return request_cursor.rowcount, conversation_cursor.rowcount
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _select_completed_response(self, event_id: str) -> AgentResponse | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT * FROM agent_requests
                WHERE event_id = ? AND status IN (?, ?)
                """,
                (
                    event_id,
                    AgentResponseStatus.COMPLETED.value,
                    AgentResponseStatus.CANCELLED.value,
                ),
            ).fetchone()
            if row is None:
                return None
            status = AgentResponseStatus(str(row["status"]))
            return AgentResponse(
                status=status,
                conversation_id=str(row["conversation_id"]),
                provider_thread_id=_optional_text(row["provider_thread_id"]),
                model=str(row["model"]),
                content=(
                    str(row["response_content"] or "")
                    if status is AgentResponseStatus.COMPLETED
                    else AGENT_NO_ACTION_CONTENT
                ),
                usage=_usage_from_row(row),
            )
        finally:
            connection.close()

    def _pending_host_deliveries(
        self,
        limit: int,
    ) -> tuple[AgentPendingHostDelivery, ...]:
        connection = _connection(self.path)
        try:
            rows = connection.execute(
                """
                SELECT event_id, public_reference_id, actor_id, workspace_id,
                       channel_id, message_id, response_content, occurred_at,
                       completed_at
                FROM agent_requests
                WHERE status = ?
                  AND trigger = ?
                  AND host_delivered_at IS NULL
                  AND completed_at IS NOT NULL
                ORDER BY completed_at, event_id
                LIMIT ?
                """,
                (
                    AgentResponseStatus.COMPLETED.value,
                    AgentTrigger.MENTION.value,
                    limit,
                ),
            ).fetchall()
            return tuple(_pending_host_delivery_from_row(row) for row in rows)
        finally:
            connection.close()

    def _pending_host_delivery(
        self,
        event_id: str,
    ) -> AgentPendingHostDelivery | None:
        connection = _connection(self.path)
        try:
            row = connection.execute(
                """
                SELECT event_id, public_reference_id, actor_id, workspace_id,
                       channel_id, message_id, response_content, occurred_at,
                       completed_at
                FROM agent_requests
                WHERE event_id = ?
                  AND status = ?
                  AND trigger = ?
                  AND host_delivered_at IS NULL
                  AND completed_at IS NOT NULL
                """,
                (
                    event_id,
                    AgentResponseStatus.COMPLETED.value,
                    AgentTrigger.MENTION.value,
                ),
            ).fetchone()
            return (
                _pending_host_delivery_from_row(row)
                if row is not None
                else None
            )
        finally:
            connection.close()

    def _interrupted_mentions(
        self,
        started_after: str,
        started_before: str,
        limit: int,
    ) -> tuple[AgentInterruptedMention, ...]:
        connection = _connection(self.path)
        try:
            rows = connection.execute(
                """
                SELECT event_id, public_reference_id, task_id, channel_id,
                       message_id, occurred_at, started_at
                FROM agent_requests
                WHERE status = ?
                  AND trigger = ?
                  AND message_id IS NOT NULL
                  AND event_id LIKE 'discord:message:%'
                  AND event_id NOT LIKE 'discord:message-edit:%'
                  AND started_at >= ?
                  AND started_at < ?
                ORDER BY started_at, event_id
                LIMIT ?
                """,
                (
                    _IN_PROGRESS_STATUS,
                    AgentTrigger.MENTION.value,
                    started_after,
                    started_before,
                    limit,
                ),
            ).fetchall()
            return tuple(_interrupted_mention_from_row(row) for row in rows)
        finally:
            connection.close()

    def _plan_host_delivery(
        self,
        event_id: str,
        purpose: str,
        channel_id: str,
        contents: tuple[str, ...],
    ) -> tuple[AgentHostDeliveryRecord, ...]:
        normalized_event_id = event_id.strip()
        normalized_purpose = purpose.strip()
        normalized_channel_id = channel_id.strip()
        if (
            not normalized_event_id
            or not normalized_purpose
            or len(normalized_purpose) > 40
            or not normalized_channel_id
        ):
            raise ValueError("host delivery identifiers must be bounded and non-empty")
        if not contents:
            return ()
        if any(not content for content in contents):
            raise ValueError("host delivery chunks must be non-empty")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_row = connection.execute(
                "SELECT status FROM agent_requests WHERE event_id = ?",
                (normalized_event_id,),
            ).fetchone()
            if (
                request_row is None
                or str(request_row["status"]) != AgentResponseStatus.COMPLETED.value
            ):
                raise ValueError("host delivery requires a completed agent request")
            for index, content in enumerate(contents):
                content_sha256 = hashlib.sha256(content.encode()).hexdigest()
                connection.execute(
                    """
                    INSERT INTO agent_host_deliveries(
                        event_id, purpose, chunk_index, content_sha256,
                        channel_id, message_id, receipted_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)
                    ON CONFLICT(event_id, purpose, chunk_index) DO NOTHING
                    """,
                    (
                        normalized_event_id,
                        normalized_purpose,
                        index,
                        content_sha256,
                        normalized_channel_id,
                        now,
                        now,
                    ),
                )
            rows = connection.execute(
                """
                SELECT * FROM agent_host_deliveries
                WHERE event_id = ? AND purpose = ?
                ORDER BY chunk_index
                """,
                (normalized_event_id, normalized_purpose),
            ).fetchall()
            records = tuple(_host_delivery_from_row(row) for row in rows)
            expected = tuple(
                (
                    index,
                    hashlib.sha256(content.encode()).hexdigest(),
                    normalized_channel_id,
                )
                for index, content in enumerate(contents)
            )
            actual = tuple(
                (record.chunk_index, record.content_sha256, record.channel_id)
                for record in records
            )
            if actual != expected:
                raise ValueError("host delivery plan conflicts with persisted intent")
            connection.commit()
            return records
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _host_delivery_records(
        self,
        event_id: str,
        purpose: str,
    ) -> tuple[AgentHostDeliveryRecord, ...]:
        connection = _connection(self.path)
        try:
            rows = connection.execute(
                """
                SELECT * FROM agent_host_deliveries
                WHERE event_id = ? AND purpose = ?
                ORDER BY chunk_index
                """,
                (event_id, purpose),
            ).fetchall()
            return tuple(_host_delivery_from_row(row) for row in rows)
        finally:
            connection.close()

    def _record_host_delivery_message(
        self,
        event_id: str,
        purpose: str,
        chunk_index: int,
        message_id: str,
    ) -> AgentHostDeliveryRecord:
        normalized_message_id = message_id.strip()
        if chunk_index < 0 or not normalized_message_id:
            raise ValueError("host delivery message evidence is invalid")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM agent_host_deliveries
                WHERE event_id = ? AND purpose = ? AND chunk_index = ?
                """,
                (event_id, purpose, chunk_index),
            ).fetchone()
            if row is None:
                raise ValueError("host delivery intent does not exist")
            existing_message_id = _optional_text(row["message_id"])
            if (
                existing_message_id is not None
                and existing_message_id != normalized_message_id
            ):
                raise ValueError("host delivery already has different message evidence")
            connection.execute(
                """
                UPDATE agent_host_deliveries
                SET message_id = ?, updated_at = ?
                WHERE event_id = ? AND purpose = ? AND chunk_index = ?
                """,
                (
                    normalized_message_id,
                    now,
                    event_id,
                    purpose,
                    chunk_index,
                ),
            )
            updated = connection.execute(
                """
                SELECT * FROM agent_host_deliveries
                WHERE event_id = ? AND purpose = ? AND chunk_index = ?
                """,
                (event_id, purpose, chunk_index),
            ).fetchone()
            assert updated is not None
            connection.commit()
            return _host_delivery_from_row(updated)
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _mark_host_delivery_receipted(
        self,
        event_id: str,
        purpose: str,
        chunk_index: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            cursor = connection.execute(
                """
                UPDATE agent_host_deliveries
                SET receipted_at = COALESCE(receipted_at, ?), updated_at = ?
                WHERE event_id = ? AND purpose = ? AND chunk_index = ?
                  AND message_id IS NOT NULL
                """,
                (now, now, event_id, purpose, chunk_index),
            )
            if cursor.rowcount != 1:
                raise ValueError("delivered message must exist before it is receipted")
            connection.commit()
        finally:
            connection.close()

    def _complete_host_delivery(self, event_id: str, allow_empty: bool) -> bool:
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(
                           CASE
                               WHEN message_id IS NOT NULL
                                AND receipted_at IS NOT NULL
                               THEN 1 ELSE 0
                           END
                       ) AS complete
                FROM agent_host_deliveries
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            assert row is not None
            total = int(row["total"])
            completed = int(row["complete"] or 0)
            if (total == 0 and not allow_empty) or completed != total:
                connection.rollback()
                return False
            cursor = connection.execute(
                """
                UPDATE agent_requests
                SET host_delivered_at = COALESCE(host_delivered_at, ?)
                WHERE event_id = ? AND status = ?
                """,
                (now, event_id, AgentResponseStatus.COMPLETED.value),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _promote_compatible_conversation(
        self,
        conversation_id: str,
    ) -> str | None:
        base, grants = _conversation_profile(conversation_id)
        if not grants:
            return None
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM agent_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone():
                connection.rollback()
                return None
            rows = connection.execute(
                """
                SELECT * FROM agent_conversations
                WHERE conversation_id = ? OR conversation_id LIKE ?
                ORDER BY updated_at DESC
                """,
                (base, f"{base}:profile:%"),
            ).fetchall()
            source = next(
                (
                    row
                    for row in rows
                    if row["conversation_id"] != conversation_id
                    and _conversation_profile(str(row["conversation_id"]))[1] < grants
                    and row["provider_thread_id"]
                ),
                None,
            )
            if source is None:
                connection.rollback()
                return None
            now = datetime.now(UTC).isoformat()
            source_id = str(source["conversation_id"])
            connection.execute(
                """
                INSERT INTO agent_conversations(
                    conversation_id, provider_thread_id, model, generation,
                    turn_count, last_input_tokens, model_context_window,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    source["provider_thread_id"],
                    source["model"],
                    source["generation"],
                    source["turn_count"],
                    source["last_input_tokens"],
                    source["model_context_window"],
                    source["created_at"],
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE agent_conversations
                SET provider_thread_id = NULL, generation = generation + 1,
                    turn_count = 0, last_input_tokens = 0,
                    model_context_window = NULL, updated_at = ?
                WHERE conversation_id = ?
                """,
                (now, source_id),
            )
            connection.commit()
            return source_id
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _record_task_candidate(
        self,
        original: AgentRequest,
        candidate: AgentRequest,
    ) -> bool:
        if not is_agent_task_id(original.task_id) or not is_agent_task_id(
            candidate.task_id
        ):
            raise ValueError("invalid agent task ID")
        if not is_agent_public_reference_id(candidate.public_reference_id):
            raise ValueError("invalid agent public reference ID")
        if original.task_id == candidate.task_id:
            raise ValueError("candidate task must be independent before routing")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            original_row = connection.execute(
                """
                SELECT state, public_reference_id, root_event_id
                FROM agent_tasks WHERE task_id = ?
                """,
                (original.task_id,),
            ).fetchone()
            if original_row is None or str(original_row["state"]) not in {
                "active",
                "finishing",
            }:
                connection.rollback()
                return False
            if (
                str(original_row["public_reference_id"])
                != original.public_reference_id
                or str(original_row["root_event_id"]) != original.event_id
            ):
                raise ValueError("active task identity conflicts with persisted routing")
            connection.execute(
                """
                INSERT INTO agent_tasks(
                    task_id, public_reference_id, root_event_id, actor_id,
                    workspace_id, channel_id, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'candidate', ?, ?)
                ON CONFLICT(task_id) DO NOTHING
                """,
                (
                    candidate.task_id,
                    candidate.public_reference_id,
                    candidate.event_id,
                    candidate.actor_id,
                    candidate.workspace_id,
                    candidate.channel_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO agent_task_events(
                    event_id, public_reference_id, candidate_task_id,
                    routed_task_id, actor_id, workspace_id, channel_id,
                    message_id, route_decision, route_reason, occurred_at,
                    created_at, routed_at, provider_applied_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, 'candidate', NULL, ?, ?, NULL, NULL
                )
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    candidate.event_id,
                    candidate.public_reference_id,
                    candidate.task_id,
                    original.task_id,
                    candidate.actor_id,
                    candidate.workspace_id,
                    candidate.channel_id,
                    candidate.message_id,
                    candidate.occurred_at.isoformat(),
                    now,
                ),
            )
            row = connection.execute(
                """
                SELECT public_reference_id, candidate_task_id
                FROM agent_task_events WHERE event_id = ?
                """,
                (candidate.event_id,),
            ).fetchone()
            if (
                row is None
                or str(row["public_reference_id"]) != candidate.public_reference_id
                or str(row["candidate_task_id"]) != candidate.task_id
            ):
                raise ValueError("candidate event identity conflicts with persisted routing")
            task_row = connection.execute(
                """
                SELECT public_reference_id, root_event_id
                FROM agent_tasks WHERE task_id = ?
                """,
                (candidate.task_id,),
            ).fetchone()
            if (
                task_row is None
                or str(task_row["public_reference_id"])
                != candidate.public_reference_id
                or str(task_row["root_event_id"]) != candidate.event_id
            ):
                raise ValueError("candidate task identity conflicts with persisted routing")
            connection.commit()
            return True
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _fail_unrouted_task_candidate(
        self,
        event_id: str,
        error_type: str,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        normalized_error = " ".join(error_type.split())[:200] or "RecoverySkipped"
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET state = 'failed', updated_at = ?, completed_at = ?,
                    completion_reason = ?
                WHERE task_id = (
                    SELECT candidate_task_id FROM agent_task_events
                    WHERE event_id = ? AND route_decision = 'separate'
                )
                  AND state = 'pending'
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_requests WHERE event_id = ?
                  )
                """,
                (now, now, normalized_error, event_id, event_id),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _route_task_candidate(
        self,
        candidate_event_id: str,
        decision: AgentTaskRouteDecision,
        active_task_id: str,
        reason: str,
    ) -> None:
        normalized_reason = " ".join(reason.split())
        if not normalized_reason or len(normalized_reason) > 400:
            raise ValueError("task route reason must contain 1 to 400 characters")
        if not is_agent_task_id(active_task_id):
            raise ValueError("invalid active task ID")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT candidate_task_id, route_decision, routed_task_id
                FROM agent_task_events WHERE event_id = ?
                """,
                (candidate_event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown task candidate event")
            candidate_task_id = str(row["candidate_task_id"])
            routed_task_id = (
                candidate_task_id
                if decision is AgentTaskRouteDecision.SEPARATE
                else active_task_id
            )
            previous_decision = str(row["route_decision"])
            previous_routed_task_id = _optional_text(row["routed_task_id"])
            if previous_decision != "candidate":
                if (
                    previous_decision == decision.value
                    and previous_routed_task_id == routed_task_id
                ):
                    connection.rollback()
                    return
                raise ValueError("task candidate already has a conflicting route")
            if decision is not AgentTaskRouteDecision.SEPARATE:
                active = connection.execute(
                    "SELECT state FROM agent_tasks WHERE task_id = ?",
                    (active_task_id,),
                ).fetchone()
                if active is None or str(active["state"]) not in {
                    "active",
                    "finishing",
                }:
                    raise AgentTaskRouteUnavailableError(
                        "active routed task is no longer available"
                    )
            connection.execute(
                """
                UPDATE agent_task_events
                SET routed_task_id = ?, route_decision = ?, route_reason = ?,
                    routed_at = ?, provider_applied_at = ?
                WHERE event_id = ? AND route_decision = 'candidate'
                """,
                (
                    routed_task_id,
                    decision.value,
                    normalized_reason,
                    now,
                    now if decision is AgentTaskRouteDecision.SEPARATE else None,
                    candidate_event_id,
                ),
            )
            if decision is AgentTaskRouteDecision.SEPARATE:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'pending', updated_at = ?, completed_at = NULL,
                        completion_reason = NULL
                    WHERE task_id = ?
                    """,
                    (now, candidate_task_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'routed', updated_at = ?, completed_at = ?,
                        completion_reason = ?
                    WHERE task_id = ?
                    """,
                    (
                        now,
                        now,
                        f"routed_{decision.value}",
                        candidate_task_id,
                    ),
                )
                if decision is AgentTaskRouteDecision.FINISH:
                    connection.execute(
                        """
                        UPDATE agent_tasks
                        SET state = 'finishing', updated_at = ?
                        WHERE task_id = ? AND state = 'active'
                        """,
                        (now, active_task_id),
                    )
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _mark_task_candidate_provider_applied(
        self,
        candidate_event_id: str,
        decision: AgentTaskRouteDecision,
        active_task_id: str,
    ) -> bool:
        if decision not in {
            AgentTaskRouteDecision.ATTACH,
            AgentTaskRouteDecision.FINISH,
            AgentTaskRouteDecision.CANCEL,
        }:
            raise ValueError("only active-task routes require provider apply state")
        if not is_agent_task_id(active_task_id):
            raise ValueError("invalid active task ID")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT route_decision, routed_task_id, provider_applied_at
                FROM agent_task_events WHERE event_id = ?
                """,
                (candidate_event_id,),
            ).fetchone()
            if (
                row is None
                or str(row["route_decision"]) != decision.value
                or _optional_text(row["routed_task_id"]) != active_task_id
            ):
                connection.rollback()
                return False
            if row["provider_applied_at"] is not None:
                connection.rollback()
                return True
            cursor = connection.execute(
                """
                UPDATE agent_task_events
                SET provider_applied_at = ?
                WHERE event_id = ? AND route_decision = ?
                  AND routed_task_id = ? AND provider_applied_at IS NULL
                """,
                (now, candidate_event_id, decision.value, active_task_id),
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _cancel_routed_task(
        self,
        candidate_event_id: str,
        active_request: AgentRequest,
        model: str,
    ) -> bool:
        """Commit provider-applied semantic cancellation as one transaction."""

        if not is_agent_task_id(active_request.task_id):
            raise ValueError("invalid active task ID")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            route = connection.execute(
                """
                SELECT route_decision, routed_task_id, provider_applied_at
                FROM agent_task_events
                WHERE event_id = ?
                """,
                (candidate_event_id,),
            ).fetchone()
            if (
                route is None
                or str(route["route_decision"])
                != AgentTaskRouteDecision.CANCEL.value
                or _optional_text(route["routed_task_id"])
                != active_request.task_id
            ):
                connection.rollback()
                return False

            request = connection.execute(
                """
                SELECT status, task_id
                FROM agent_requests
                WHERE event_id = ?
                """,
                (active_request.event_id,),
            ).fetchone()
            task = connection.execute(
                """
                SELECT state, root_event_id
                FROM agent_tasks
                WHERE task_id = ?
                """,
                (active_request.task_id,),
            ).fetchone()
            if (
                request is None
                or str(request["task_id"]) != active_request.task_id
                or task is None
                or str(task["root_event_id"]) != active_request.event_id
            ):
                connection.rollback()
                return False
            if (
                route["provider_applied_at"] is not None
                and str(request["status"])
                == AgentResponseStatus.CANCELLED.value
                and str(task["state"]) == "cancelled"
            ):
                connection.rollback()
                return True
            if (
                route["provider_applied_at"] is not None
                or str(request["status"]) != _IN_PROGRESS_STATUS
                or str(task["state"]) not in {"active", "finishing"}
            ):
                connection.rollback()
                return False

            request_cursor = connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, model = ?, error_type = ?, completed_at = ?
                WHERE event_id = ? AND task_id = ? AND status = ?
                """,
                (
                    AgentResponseStatus.CANCELLED.value,
                    model,
                    "AgentTaskCancelled",
                    now,
                    active_request.event_id,
                    active_request.task_id,
                    _IN_PROGRESS_STATUS,
                ),
            )
            task_cursor = connection.execute(
                """
                UPDATE agent_tasks
                SET state = 'cancelled', updated_at = ?, completed_at = ?,
                    completion_reason = 'follow_up_cancelled'
                WHERE task_id = ? AND root_event_id = ?
                  AND state IN ('active', 'finishing')
                """,
                (
                    now,
                    now,
                    active_request.task_id,
                    active_request.event_id,
                ),
            )
            route_cursor = connection.execute(
                """
                UPDATE agent_task_events
                SET provider_applied_at = ?
                WHERE event_id = ? AND route_decision = 'cancel'
                  AND routed_task_id = ? AND provider_applied_at IS NULL
                """,
                (now, candidate_event_id, active_request.task_id),
            )
            if (
                request_cursor.rowcount != 1
                or task_cursor.rowcount != 1
                or route_cursor.rowcount != 1
            ):
                connection.rollback()
                return False
            connection.commit()
            return True
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _default_task_candidate_to_separate(
        self,
        candidate_event_id: str,
        reason: str,
    ) -> bool:
        normalized_reason = " ".join(reason.split())
        if not normalized_reason or len(normalized_reason) > 400:
            raise ValueError("task route reason must contain 1 to 400 characters")
        now = datetime.now(UTC).isoformat()
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT candidate_task_id, route_decision, routed_task_id,
                       provider_applied_at
                FROM agent_task_events WHERE event_id = ?
                """,
                (candidate_event_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            candidate_task_id = str(row["candidate_task_id"])
            previous_decision = str(row["route_decision"])
            previous_routed_task_id = _optional_text(row["routed_task_id"])
            if previous_decision == AgentTaskRouteDecision.SEPARATE.value:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'pending', updated_at = ?, completed_at = NULL,
                        completion_reason = NULL
                    WHERE task_id = ? AND state NOT IN ('completed', 'cancelled')
                    """,
                    (now, candidate_task_id),
                )
                connection.commit()
                return True
            if previous_decision not in {
                "candidate",
                AgentTaskRouteDecision.ATTACH.value,
                AgentTaskRouteDecision.FINISH.value,
                AgentTaskRouteDecision.CANCEL.value,
            } or (
                previous_decision != "candidate"
                and row["provider_applied_at"] is not None
            ):
                connection.rollback()
                return False
            connection.execute(
                """
                UPDATE agent_task_events
                SET routed_task_id = candidate_task_id,
                    route_decision = 'separate', route_reason = ?,
                    routed_at = ?, provider_applied_at = ?
                WHERE event_id = ?
                """,
                (normalized_reason, now, now, candidate_event_id),
            )
            connection.execute(
                """
                UPDATE agent_tasks
                SET state = 'pending', updated_at = ?, completed_at = NULL,
                    completion_reason = NULL
                WHERE task_id = ? AND state NOT IN ('completed', 'cancelled')
                """,
                (now, candidate_task_id),
            )
            if (
                previous_decision == AgentTaskRouteDecision.FINISH.value
                and previous_routed_task_id is not None
            ):
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'active', updated_at = ?
                    WHERE task_id = ? AND state = 'finishing'
                      AND NOT EXISTS (
                          SELECT 1 FROM agent_task_events
                          WHERE routed_task_id = ? AND route_decision = 'finish'
                      )
                    """,
                    (now, previous_routed_task_id, previous_routed_task_id),
                )
            connection.commit()
            return True
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _begin(self, request: AgentRequest, model: str) -> None:
        if not is_agent_public_reference_id(request.public_reference_id):
            raise ValueError("invalid agent public reference ID")
        if not is_agent_task_id(request.task_id):
            raise ValueError("invalid agent task ID")
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO agent_conversations(
                    conversation_id, provider_thread_id, model, generation,
                    turn_count, last_input_tokens, model_context_window,
                    created_at, updated_at
                ) VALUES (?, NULL, ?, 0, 0, 0, NULL, ?, ?)
                ON CONFLICT(conversation_id) DO NOTHING
                """,
                (request.conversation_id, model, now, now),
            )
            connection.execute(
                """
                INSERT INTO agent_tasks(
                    task_id, public_reference_id, root_event_id, actor_id,
                    workspace_id, channel_id, state, created_at, updated_at,
                    completed_at, completion_reason
                ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL, NULL)
                ON CONFLICT(task_id) DO UPDATE SET
                    state = CASE
                        WHEN agent_tasks.state IN (
                            'candidate', 'routed', 'completed', 'cancelled'
                        )
                            THEN agent_tasks.state
                        ELSE 'active'
                    END,
                    updated_at = excluded.updated_at,
                    completed_at = CASE
                        WHEN agent_tasks.state IN (
                            'candidate', 'routed', 'completed', 'cancelled'
                        )
                            THEN agent_tasks.completed_at
                        ELSE NULL
                    END,
                    completion_reason = CASE
                        WHEN agent_tasks.state IN (
                            'candidate', 'routed', 'completed', 'cancelled'
                        )
                            THEN agent_tasks.completion_reason
                        ELSE NULL
                    END
                """,
                (
                    request.task_id,
                    request.public_reference_id,
                    request.event_id,
                    request.actor_id,
                    request.workspace_id,
                    request.channel_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO agent_task_events(
                    event_id, public_reference_id, candidate_task_id,
                    routed_task_id, actor_id, workspace_id, channel_id,
                    message_id, route_decision, route_reason, occurred_at,
                    created_at, routed_at, provider_applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'root', NULL, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    request.event_id,
                    request.public_reference_id,
                    request.task_id,
                    request.task_id,
                    request.actor_id,
                    request.workspace_id,
                    request.channel_id,
                    request.message_id,
                    request.occurred_at.isoformat(),
                    now,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO agent_requests(
                    event_id, public_reference_id, task_id, conversation_id, trigger,
                    actor_id, workspace_id, channel_id, message_id, model,
                    status, occurred_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    status = CASE
                        WHEN agent_requests.status IN ('completed', 'cancelled')
                            THEN agent_requests.status
                        ELSE 'in_progress'
                    END,
                    started_at = CASE
                        WHEN agent_requests.status IN ('completed', 'cancelled')
                            THEN agent_requests.started_at
                        ELSE excluded.started_at
                    END,
                    error_type = CASE
                        WHEN agent_requests.status IN ('completed', 'cancelled')
                            THEN agent_requests.error_type
                        ELSE NULL
                    END
                """,
                (
                    request.event_id,
                    request.public_reference_id,
                    request.task_id,
                    request.conversation_id,
                    request.trigger.value,
                    request.actor_id,
                    request.workspace_id,
                    request.channel_id,
                    request.message_id,
                    model,
                    request.occurred_at.isoformat(),
                    now,
                ),
            )
            stored_reference = connection.execute(
                """
                SELECT public_reference_id, task_id, conversation_id
                FROM agent_requests
                WHERE event_id = ?
                """,
                (request.event_id,),
            ).fetchone()
            if (
                stored_reference is None
                or str(stored_reference[0]) != request.public_reference_id
                or str(stored_reference[1]) != request.task_id
                or str(stored_reference[2]) != request.conversation_id
            ):
                raise ValueError(
                    "agent event ID is bound to a different reference, task, or conversation"
                )
            stored_task = connection.execute(
                """
                SELECT public_reference_id, root_event_id, state
                FROM agent_tasks WHERE task_id = ?
                """,
                (request.task_id,),
            ).fetchone()
            if (
                stored_task is None
                or str(stored_task[0]) != request.public_reference_id
                or str(stored_task[1]) != request.event_id
            ):
                raise ValueError("agent task ID is bound to different root identity")
            if str(stored_task[2]) in {"candidate", "routed"}:
                raise ValueError(
                    "an attached or unrouted task candidate cannot execute independently"
                )
            connection.commit()
        finally:
            connection.close()

    def _complete(self, request: AgentRequest, response: AgentResponse) -> bool:
        now = datetime.now(UTC).isoformat()
        usage = response.usage
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_cursor = connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, provider_thread_id = ?, model = ?,
                    response_content = ?, input_tokens = ?,
                    cached_input_tokens = ?, output_tokens = ?,
                    reasoning_output_tokens = ?, total_tokens = ?,
                    model_context_window = ?, error_type = NULL, completed_at = ?
                WHERE event_id = ? AND status = ?
                """,
                (
                    response.status.value,
                    response.provider_thread_id,
                    response.model,
                    response.content,
                    usage.input_tokens,
                    usage.cached_input_tokens,
                    usage.output_tokens,
                    usage.reasoning_output_tokens,
                    usage.total_tokens,
                    usage.model_context_window,
                    now,
                    request.event_id,
                    _IN_PROGRESS_STATUS,
                ),
            )
            if request_cursor.rowcount != 1:
                connection.rollback()
                return False
            connection.execute(
                """
                UPDATE agent_conversations
                SET provider_thread_id = ?, model = ?, turn_count = turn_count + 1,
                    last_input_tokens = ?, model_context_window = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    response.provider_thread_id,
                    response.model,
                    usage.input_tokens,
                    usage.model_context_window,
                    now,
                    request.conversation_id,
                ),
            )
            connection.execute(
                """
                UPDATE agent_tasks
                SET state = 'completed', updated_at = ?, completed_at = ?,
                    completion_reason = 'provider_completed'
                WHERE task_id = ?
                """,
                (now, now, request.task_id),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def _fail(
        self,
        request: AgentRequest,
        model: str,
        error_type: str,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_cursor = connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, model = ?, error_type = ?, completed_at = ?
                WHERE event_id = ? AND status = ?
                """,
                (
                    AgentResponseStatus.FAILED.value,
                    model,
                    error_type[:200],
                    now,
                    request.event_id,
                    _IN_PROGRESS_STATUS,
                ),
            )
            if request_cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'failed', updated_at = ?, completed_at = ?,
                        completion_reason = ?
                    WHERE task_id = ?
                    """,
                    (now, now, error_type[:200], request.task_id),
                )
            connection.commit()
            return request_cursor.rowcount == 1
        finally:
            connection.close()

    def _cancel(
        self,
        request: AgentRequest,
        model: str,
        reason: str,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        normalized_reason = " ".join(reason.split())[:200] or "user_requested"
        connection = _connection(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_cursor = connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, model = ?, error_type = ?, completed_at = ?
                WHERE event_id = ? AND status = ?
                """,
                (
                    AgentResponseStatus.CANCELLED.value,
                    model,
                    "AgentTaskCancelled",
                    now,
                    request.event_id,
                    _IN_PROGRESS_STATUS,
                ),
            )
            if request_cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'cancelled', updated_at = ?, completed_at = ?,
                        completion_reason = ?
                    WHERE task_id = ? AND state IN ('pending', 'active', 'finishing')
                    """,
                    (now, now, normalized_reason, request.task_id),
                )
            connection.commit()
            return request_cursor.rowcount == 1
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _fail_interrupted_mention(
        self,
        event_id: str,
        error_type: str,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            cursor = connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, error_type = ?, completed_at = ?
                WHERE event_id = ?
                  AND status = ?
                  AND trigger = ?
                """,
                (
                    AgentResponseStatus.FAILED.value,
                    error_type[:200],
                    now,
                    event_id,
                    _IN_PROGRESS_STATUS,
                    AgentTrigger.MENTION.value,
                ),
            )
            if cursor.rowcount == 1:
                connection.execute(
                    """
                    UPDATE agent_tasks
                    SET state = 'failed', updated_at = ?, completed_at = ?,
                        completion_reason = ?
                    WHERE task_id = (
                        SELECT task_id FROM agent_requests WHERE event_id = ?
                    )
                    """,
                    (now, now, error_type[:200], event_id),
                )
            connection.commit()
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _rotate(self, conversation_id: str, model: str) -> None:
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                """
                UPDATE agent_conversations
                SET provider_thread_id = NULL, model = ?, generation = generation + 1,
                    turn_count = 0, last_input_tokens = 0,
                    model_context_window = NULL, updated_at = ?
                WHERE conversation_id = ?
                """,
                (model, now, conversation_id),
            )
            connection.commit()
        finally:
            connection.close()

    def _reset_provider_continuity(
        self,
        conversation_ids: tuple[str, ...] | None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            where = (
                "(provider_thread_id IS NOT NULL OR turn_count != 0 "
                "OR last_input_tokens != 0 OR model_context_window IS NOT NULL)"
            )
            values: tuple[object, ...] = ()
            if conversation_ids is not None:
                placeholders = ", ".join("?" for _ in conversation_ids)
                where += f" AND conversation_id IN ({placeholders})"
                values = tuple(conversation_ids)
            cursor = connection.execute(
                f"""
                UPDATE agent_conversations
                SET provider_thread_id = NULL, generation = generation + 1,
                    turn_count = 0, last_input_tokens = 0,
                    model_context_window = NULL, updated_at = ?
                WHERE {where}
                """,
                (now, *values),
            )
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _request_window(
        self,
        actor_id: str | None,
        workspace_id: str | None,
        since: datetime,
        excluded_actor_ids: frozenset[str],
        included_triggers: frozenset[AgentTrigger],
    ) -> tuple[int, datetime | None]:
        connection = sqlite3.connect(self.path)
        try:
            trigger_sql, trigger_values = _trigger_filter(included_triggers)
            if actor_id is not None:
                row = connection.execute(
                    (
                        "SELECT COUNT(*), MIN(started_at) FROM agent_requests "
                        "WHERE actor_id = ? AND started_at >= ?"
                        f"{trigger_sql}"
                    ),
                    (actor_id, since.isoformat(), *trigger_values),
                ).fetchone()
            elif workspace_id is not None:
                exclusion_sql, exclusion_values = _actor_exclusion(excluded_actor_ids)
                row = connection.execute(
                    (
                        "SELECT COUNT(*), MIN(started_at) FROM agent_requests "
                        "WHERE workspace_id = ? AND started_at >= ?"
                        f"{exclusion_sql}"
                        f"{trigger_sql}"
                    ),
                    (
                        workspace_id,
                        since.isoformat(),
                        *exclusion_values,
                        *trigger_values,
                    ),
                ).fetchone()
            else:
                raise ValueError("actor_id or workspace_id is required")
            oldest = datetime.fromisoformat(str(row[1])) if row[1] is not None else None
            return int(row[0]), oldest
        finally:
            connection.close()

    def _token_budget_window(
        self,
        since: datetime,
        limit: int,
        excluded_actor_ids: frozenset[str],
    ) -> tuple[int, datetime | None]:
        connection = sqlite3.connect(self.path)
        try:
            exclusion_sql, exclusion_values = _actor_exclusion(excluded_actor_ids)
            rows = connection.execute(
                (
                    "SELECT COALESCE(total_tokens, 0), "
                    "COALESCE(cached_input_tokens, 0), completed_at "
                    "FROM agent_requests "
                    f"WHERE completed_at >= ?{exclusion_sql} "
                    "ORDER BY completed_at, event_id"
                ),
                (since.isoformat(), *exclusion_values),
            ).fetchall()
            contributions = tuple(
                max(0, int(row[0]) - int(row[1]))
                for row in rows
            )
            usage = sum(contributions)
            if usage < limit:
                return usage, None
            remaining = usage
            release_anchor: datetime | None = None
            for row, contribution in zip(rows, contributions, strict=True):
                remaining -= contribution
                release_anchor = datetime.fromisoformat(str(row[2]))
                if remaining < limit:
                    break
            return usage, release_anchor
        finally:
            connection.close()


def _connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _conversation_from_row(row: sqlite3.Row) -> AgentConversationRecord:
    context_window = row["model_context_window"]
    return AgentConversationRecord(
        conversation_id=str(row["conversation_id"]),
        provider_thread_id=_optional_text(row["provider_thread_id"]),
        model=str(row["model"]),
        generation=int(row["generation"]),
        turn_count=int(row["turn_count"]),
        last_input_tokens=int(row["last_input_tokens"]),
        model_context_window=int(context_window) if context_window is not None else None,
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _pending_host_delivery_from_row(
    row: sqlite3.Row,
) -> AgentPendingHostDelivery:
    completed_at = row["completed_at"]
    assert completed_at is not None
    return AgentPendingHostDelivery(
        event_id=str(row["event_id"]),
        public_reference_id=str(row["public_reference_id"]),
        actor_id=str(row["actor_id"]),
        workspace_id=_optional_text(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        source_message_id=_optional_text(row["message_id"]),
        response_content=str(row["response_content"] or ""),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        completed_at=datetime.fromisoformat(str(completed_at)),
    )


def _interrupted_mention_from_row(
    row: sqlite3.Row,
) -> AgentInterruptedMention:
    message_id = row["message_id"]
    assert message_id is not None
    return AgentInterruptedMention(
        event_id=str(row["event_id"]),
        public_reference_id=str(row["public_reference_id"]),
        task_id=str(row["task_id"]),
        channel_id=str(row["channel_id"]),
        source_message_id=str(message_id),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        started_at=datetime.fromisoformat(str(row["started_at"])),
    )


def _host_delivery_from_row(row: sqlite3.Row) -> AgentHostDeliveryRecord:
    receipted_at = row["receipted_at"]
    return AgentHostDeliveryRecord(
        event_id=str(row["event_id"]),
        purpose=str(row["purpose"]),
        chunk_index=int(row["chunk_index"]),
        content_sha256=str(row["content_sha256"]),
        channel_id=str(row["channel_id"]),
        message_id=_optional_text(row["message_id"]),
        receipted_at=(
            datetime.fromisoformat(str(receipted_at))
            if receipted_at is not None
            else None
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


def _request_record_from_row(row: sqlite3.Row) -> AgentRequestRecord:
    completed_at = row["completed_at"]
    host_delivered_at = row["host_delivered_at"]
    return AgentRequestRecord(
        public_reference_id=str(row["public_reference_id"]),
        task_id=str(row["task_id"]),
        event_id=str(row["event_id"]),
        conversation_id=str(row["conversation_id"]),
        trigger=AgentTrigger(str(row["trigger"])),
        actor_id=str(row["actor_id"]),
        workspace_id=_optional_text(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        source_message_id=_optional_text(row["message_id"]),
        model=str(row["model"]),
        status=str(row["status"]),
        provider_thread_id=_optional_text(row["provider_thread_id"]),
        error_type=_optional_text(row["error_type"]),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        started_at=datetime.fromisoformat(str(row["started_at"])),
        completed_at=(
            datetime.fromisoformat(str(completed_at))
            if completed_at is not None
            else None
        ),
        host_delivered_at=(
            datetime.fromisoformat(str(host_delivered_at))
            if host_delivered_at is not None
            else None
        ),
    )


def _task_snapshot_from_row(row: sqlite3.Row) -> AgentTaskSnapshot:
    started_at = row["started_at"]
    completed_at = row["completed_at"]
    host_delivered_at = row["host_delivered_at"]
    return AgentTaskSnapshot(
        task_id=str(row["task_id"]),
        public_reference_id=str(row["public_reference_id"]),
        root_event_id=str(row["root_event_id"]),
        execution_event_id=str(row["execution_event_id"]),
        actor_id=str(row["actor_id"]),
        workspace_id=_optional_text(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        state=str(row["state"]),
        completion_reason=_optional_text(row["completion_reason"]),
        routed_task_id=_optional_text(row["routed_task_id"]),
        route_decision=str(row["route_decision"]),
        route_reason=_optional_text(row["route_reason"]),
        event_count=int(row["event_count"]),
        attached_event_count=int(row["attached_event_count"]),
        request_status=_optional_text(row["request_status"]),
        model=_optional_text(row["model"]),
        provider_thread_id=_optional_text(row["provider_thread_id"]),
        error_type=_optional_text(row["error_type"]),
        delivery_count=int(row["delivery_count"]),
        receipted_delivery_count=int(row["receipted_delivery_count"]),
        started_at=(
            datetime.fromisoformat(str(started_at)) if started_at is not None else None
        ),
        completed_at=(
            datetime.fromisoformat(str(completed_at))
            if completed_at is not None
            else None
        ),
        host_delivered_at=(
            datetime.fromisoformat(str(host_delivered_at))
            if host_delivered_at is not None
            else None
        ),
    )


def _unrouted_candidate_from_row(
    row: sqlite3.Row,
) -> AgentUnroutedTaskCandidate:
    message_id = row["message_id"]
    assert message_id is not None
    return AgentUnroutedTaskCandidate(
        event_id=str(row["event_id"]),
        task_id=str(row["candidate_task_id"]),
        public_reference_id=str(row["public_reference_id"]),
        channel_id=str(row["channel_id"]),
        source_message_id=str(message_id),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def _usage_from_row(row: sqlite3.Row) -> AgentTokenUsage:
    context_window = row["model_context_window"]
    return AgentTokenUsage(
        input_tokens=int(row["input_tokens"]),
        cached_input_tokens=int(row["cached_input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        reasoning_output_tokens=int(row["reasoning_output_tokens"]),
        total_tokens=int(row["total_tokens"]),
        model_context_window=int(context_window) if context_window is not None else None,
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _actor_exclusion(actor_ids: frozenset[str]) -> tuple[str, tuple[str, ...]]:
    values = tuple(sorted(actor_ids))
    if not values:
        return "", ()
    placeholders = ",".join("?" for _ in values)
    return f" AND actor_id NOT IN ({placeholders})", values


def _trigger_filter(
    triggers: frozenset[AgentTrigger],
) -> tuple[str, tuple[str, ...]]:
    values = tuple(sorted(trigger.value for trigger in triggers))
    if not values:
        return "", ()
    placeholders = ",".join("?" for _ in values)
    return f" AND trigger IN ({placeholders})", values


def _task_state_for_request_status(status: str) -> str:
    if status == _IN_PROGRESS_STATUS:
        return "active"
    if status == AgentResponseStatus.COMPLETED.value:
        return "completed"
    if status == AgentResponseStatus.CANCELLED.value:
        return "cancelled"
    return "failed"


def _conversation_profile(conversation_id: str) -> tuple[str, frozenset[str]]:
    marker = ":profile:"
    if marker not in conversation_id:
        return conversation_id, frozenset()
    base, profile = conversation_id.rsplit(marker, 1)
    grants = frozenset(value for value in profile.split("+") if value)
    return base, grants
