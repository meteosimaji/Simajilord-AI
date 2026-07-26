"""Append-only local SQLite event journal."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from simajilord.core.capabilities import InvocationContext

_SENSITIVE_PARTS = (
    "token",
    "secret",
    "password",
    "cookie",
    "authorization",
    "data_url",
)


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    occurred_at: datetime
    kind: str
    actor_id: str | None
    workspace_id: str | None
    transport: str | None
    request_id: str | None
    payload: dict[str, object]


class EventJournal:
    """Store capability and transport events for audit and agent reconciliation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def record_invocation(
        self,
        *,
        capability_name: str,
        context: InvocationContext,
        request: object,
        response: object | None,
        error: Exception | None,
        duration_ms: float,
    ) -> None:
        payload: dict[str, object] = {
            "capability": capability_name,
            "outcome": "failed" if error else "succeeded",
            "duration_ms": round(duration_ms, 3),
            "request": _safe_value(request),
        }
        if error is None:
            payload["response"] = _safe_value(response)
        else:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = _bounded(str(error), 1_000)
        await self.append(
            kind="capability.invocation",
            payload=payload,
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            transport=context.transport,
            request_id=context.request_id,
        )

    async def append(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None = None,
        workspace_id: str | None = None,
        transport: str | None = None,
        request_id: str | None = None,
    ) -> int:
        safe_payload = cast_payload(_safe_value(payload))
        async with self._lock:
            return await asyncio.to_thread(
                self._insert,
                kind,
                safe_payload,
                actor_id,
                workspace_id,
                transport,
                request_id,
            )

    async def recent(
        self,
        *,
        after_sequence: int = 0,
        workspace_id: str | None = None,
        limit: int = 100,
    ) -> tuple[EventRecord, ...]:
        bounded_limit = min(max(limit, 1), 1_000)
        async with self._lock:
            return await asyncio.to_thread(
                self._select,
                after_sequence,
                workspace_id,
                bounded_limit,
            )

    async def latest_sequence(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._latest_sequence)

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    occurred_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    actor_id TEXT,
                    workspace_id TEXT,
                    transport TEXT,
                    request_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS events_workspace_sequence "
                "ON events(workspace_id, sequence)"
            )
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _insert(
        self,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None,
        workspace_id: str | None,
        transport: str | None,
        request_id: str | None,
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            cursor = connection.execute(
                """
                INSERT INTO events(
                    occurred_at, kind, actor_id, workspace_id, transport,
                    request_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    kind,
                    actor_id,
                    workspace_id,
                    transport,
                    request_id,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an event sequence.")
            return cursor.lastrowid
        finally:
            connection.close()

    def _select(
        self,
        after_sequence: int,
        workspace_id: str | None,
        limit: int,
    ) -> tuple[EventRecord, ...]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            if workspace_id is None:
                rows = connection.execute(
                    "SELECT * FROM events WHERE sequence > ? ORDER BY sequence LIMIT ?",
                    (after_sequence, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM events
                    WHERE sequence > ? AND workspace_id = ?
                    ORDER BY sequence LIMIT ?
                    """,
                    (after_sequence, workspace_id, limit),
                ).fetchall()
            return tuple(
                EventRecord(
                    sequence=int(row["sequence"]),
                    occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
                    kind=str(row["kind"]),
                    actor_id=row["actor_id"],
                    workspace_id=row["workspace_id"],
                    transport=row["transport"],
                    request_id=row["request_id"],
                    payload=json.loads(str(row["payload_json"])),
                )
                for row in rows
            )
        finally:
            connection.close()

    def _latest_sequence(self) -> int:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events").fetchone()
            return int(row[0])
        finally:
            connection.close()


def _safe_value(value: object, *, key: str = "") -> object:
    if any(part in key.lower() for part in _SENSITIVE_PARTS):
        return "[REDACTED]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_bytes": len(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _safe_value(getattr(value, field.name), key=field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            _bounded(str(item_key), 200): _safe_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in list(value)[:100]]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return _bounded(value, 4_000)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded(repr(value), 1_000)


def cast_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Event payload must be an object.")
    return value


def _bounded(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"
