"""Durable agent conversation identity, idempotency, and usage accounting."""

from __future__ import annotations

import asyncio
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
)


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

    async def request_window(
        self,
        *,
        actor_id: str | None,
        workspace_id: str | None,
        since: datetime,
        excluded_actor_ids: frozenset[str] = frozenset(),
    ) -> tuple[int, datetime | None]:
        async with self._lock:
            return await asyncio.to_thread(
                self._request_window,
                actor_id,
                workspace_id,
                since,
                excluded_actor_ids,
            )

    async def token_usage_since(
        self,
        since: datetime,
        *,
        excluded_actor_ids: frozenset[str] = frozenset(),
    ) -> int:
        async with self._lock:
            return await asyncio.to_thread(
                self._token_usage_since,
                since,
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
                    FOREIGN KEY(conversation_id)
                        REFERENCES agent_conversations(conversation_id)
                );

                CREATE INDEX IF NOT EXISTS agent_requests_actor_started
                    ON agent_requests(actor_id, started_at);
                CREATE INDEX IF NOT EXISTS agent_requests_workspace_started
                    ON agent_requests(workspace_id, started_at);
                CREATE INDEX IF NOT EXISTS agent_requests_completed
                    ON agent_requests(completed_at);
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
    ) -> tuple[int, datetime | None]:
        connection = sqlite3.connect(self.path)
        try:
            if actor_id is not None:
                row = connection.execute(
                    """
                    SELECT COUNT(*), MIN(started_at) FROM agent_requests
                    WHERE actor_id = ? AND started_at >= ?
                    """,
                    (actor_id, since.isoformat()),
                ).fetchone()
            elif workspace_id is not None:
                exclusion_sql, exclusion_values = _actor_exclusion(excluded_actor_ids)
                row = connection.execute(
                    (
                        "SELECT COUNT(*), MIN(started_at) FROM agent_requests "
                        "WHERE workspace_id = ? AND started_at >= ?"
                        f"{exclusion_sql}"
                    ),
                    (workspace_id, since.isoformat(), *exclusion_values),
                ).fetchone()
            else:
                raise ValueError("actor_id or workspace_id is required")
            oldest = datetime.fromisoformat(str(row[1])) if row[1] is not None else None
            return int(row[0]), oldest
        finally:
            connection.close()

    def _token_usage_since(
        self,
        since: datetime,
        excluded_actor_ids: frozenset[str],
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            exclusion_sql, exclusion_values = _actor_exclusion(excluded_actor_ids)
            row = connection.execute(
                (
                    "SELECT COALESCE(SUM(total_tokens), 0) FROM agent_requests "
                    f"WHERE completed_at >= ?{exclusion_sql}"
                ),
                (since.isoformat(), *exclusion_values),
            ).fetchone()
            return int(row[0])
        finally:
            connection.close()


def _connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
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


def _conversation_profile(conversation_id: str) -> tuple[str, frozenset[str]]:
    marker = ":profile:"
    if marker not in conversation_id:
        return conversation_id, frozenset()
    base, profile = conversation_id.rsplit(marker, 1)
    grants = frozenset(value for value in profile.split("+") if value)
    return base, grants
