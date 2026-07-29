"""Durable agent conversation identity, idempotency, and usage accounting."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .contracts import (
    AgentRequest,
    AgentResponse,
    AgentResponseStatus,
    AgentTokenUsage,
    AgentTrigger,
)

_IN_PROGRESS_STATUS = "in_progress"


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


class AgentConversationStore:
    """SQLite authority for restart-safe agent state without user message bodies."""

    def __init__(self, path: Path) -> None:
        self.path = path
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
        async with self._lock:
            return await asyncio.to_thread(self._select_completed_response, event_id)

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
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._complete, request, response)

    async def fail(
        self,
        request: AgentRequest,
        *,
        model: str,
        error_type: str,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(self._fail, request, model, error_type)

    async def rotate(self, conversation_id: str, *, model: str) -> None:
        """Forget only the provider thread while preserving conversation identity."""

        async with self._lock:
            await asyncio.to_thread(self._rotate, conversation_id, model)

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
        """Return current usage and the oldest expiry that drops it below limit."""

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
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS agent_requests_host_pending
                ON agent_requests(status, host_delivered_at, completed_at)
                """
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
                WHERE event_id = ? AND status = ?
                """,
                (event_id, AgentResponseStatus.COMPLETED.value),
            ).fetchone()
            if row is None:
                return None
            return AgentResponse(
                status=AgentResponseStatus.COMPLETED,
                conversation_id=str(row["conversation_id"]),
                provider_thread_id=_optional_text(row["provider_thread_id"]),
                model=str(row["model"]),
                content=str(row["response_content"] or ""),
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
                SELECT event_id, actor_id, workspace_id, channel_id, message_id,
                       response_content, occurred_at, completed_at
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
                SELECT event_id, actor_id, workspace_id, channel_id, message_id,
                       response_content, occurred_at, completed_at
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
                SELECT event_id, channel_id, message_id, occurred_at, started_at
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

    def _begin(self, request: AgentRequest, model: str) -> None:
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
                INSERT INTO agent_requests(
                    event_id, conversation_id, trigger, actor_id, workspace_id,
                    channel_id, message_id, model, status, occurred_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    status = CASE
                        WHEN agent_requests.status = 'completed'
                            THEN agent_requests.status
                        ELSE 'in_progress'
                    END,
                    started_at = CASE
                        WHEN agent_requests.status = 'completed'
                            THEN agent_requests.started_at
                        ELSE excluded.started_at
                    END,
                    error_type = CASE
                        WHEN agent_requests.status = 'completed'
                            THEN agent_requests.error_type
                        ELSE NULL
                    END
                """,
                (
                    request.event_id,
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
            connection.commit()
        finally:
            connection.close()

    def _complete(self, request: AgentRequest, response: AgentResponse) -> None:
        now = datetime.now(UTC).isoformat()
        usage = response.usage
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
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
                UPDATE agent_requests
                SET status = ?, provider_thread_id = ?, model = ?,
                    response_content = ?, input_tokens = ?,
                    cached_input_tokens = ?, output_tokens = ?,
                    reasoning_output_tokens = ?, total_tokens = ?,
                    model_context_window = ?, error_type = NULL, completed_at = ?
                WHERE event_id = ?
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
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _fail(
        self,
        request: AgentRequest,
        model: str,
        error_type: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                """
                UPDATE agent_requests
                SET status = ?, model = ?, error_type = ?, completed_at = ?
                WHERE event_id = ?
                """,
                (
                    AgentResponseStatus.FAILED.value,
                    model,
                    error_type[:200],
                    now,
                    request.event_id,
                ),
            )
            connection.commit()
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
                    "SELECT COALESCE(total_tokens, 0), completed_at "
                    "FROM agent_requests "
                    f"WHERE completed_at >= ?{exclusion_sql} "
                    "ORDER BY completed_at, event_id"
                ),
                (since.isoformat(), *exclusion_values),
            ).fetchall()
            usage = sum(int(row[0]) for row in rows)
            if usage < limit:
                return usage, None
            remaining = usage
            release_anchor: datetime | None = None
            for row in rows:
                remaining -= int(row[0])
                release_anchor = datetime.fromisoformat(str(row[1]))
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


def _conversation_profile(conversation_id: str) -> tuple[str, frozenset[str]]:
    marker = ":profile:"
    if marker not in conversation_id:
        return conversation_id, frozenset()
    base, profile = conversation_id.rsplit(marker, 1)
    grants = frozenset(value for value in profile.split("+") if value)
    return base, grants
