"""Append-only local SQLite event journal."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
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
_AUDIT_QUEUE_MAX_EVENTS = 4_096
_AUDIT_BATCH_MAX_EVENTS = 64

log = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class OperationDiagnostics:
    """Bounded operational counters derived from the retained event journal."""

    last_radio_failure_at: datetime | None
    overlay_failure_count: int
    dashboard_429_count: int


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    occurred_at: str
    kind: str
    payload: dict[str, object]
    actor_id: str | None
    workspace_id: str | None
    transport: str | None
    request_id: str | None


@dataclass(frozen=True, slots=True)
class _FlushAuditQueue:
    completed: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class _CloseAuditQueue:
    completed: asyncio.Future[None]


class EventJournal:
    """Store capability and transport events for audit and agent reconciliation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._audit_queue: asyncio.Queue[
            _PendingEvent | _FlushAuditQueue | _CloseAuditQueue
        ] = asyncio.Queue(maxsize=_AUDIT_QUEUE_MAX_EVENTS)
        self._audit_writer_task: asyncio.Task[None] | None = None
        self._closed = False
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
        if self._closed:
            raise RuntimeError("Event journal is closed.")
        self._ensure_audit_writer()
        await self._audit_queue.put(
            _PendingEvent(
                occurred_at=datetime.now(UTC).isoformat(),
                kind="capability.invocation",
                payload=cast_payload(_safe_value(payload)),
                actor_id=context.actor_id,
                workspace_id=context.workspace_id,
                transport=context.transport,
                request_id=context.request_id,
            )
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
        """Durably append an explicit event before returning its sequence."""

        if self._closed:
            raise RuntimeError("Event journal is closed.")
        pending = _PendingEvent(
            occurred_at=datetime.now(UTC).isoformat(),
            kind=kind,
            payload=cast_payload(_safe_value(payload)),
            actor_id=actor_id,
            workspace_id=workspace_id,
            transport=transport,
            request_id=request_id,
        )
        async with self._write_lock:
            return await asyncio.to_thread(self._insert, pending)

    async def recent(
        self,
        *,
        after_sequence: int = 0,
        workspace_id: str | None = None,
        limit: int = 100,
    ) -> tuple[EventRecord, ...]:
        bounded_limit = min(max(limit, 1), 1_000)
        await self._flush_audit_queue()
        return await asyncio.to_thread(
            self._select,
            after_sequence,
            workspace_id,
            bounded_limit,
        )

    async def latest_sequence(self) -> int:
        """Return the latest committed cursor without waiting for queued audits."""

        return await asyncio.to_thread(self._latest_sequence)

    async def prune(self, *, before: datetime) -> int:
        """Delete retained events older than an explicit UTC cutoff."""

        cutoff = _utc_iso(before)
        await self._flush_audit_queue()
        async with self._write_lock:
            return await asyncio.to_thread(self._prune, cutoff)

    async def operation_diagnostics(self) -> OperationDiagnostics:
        """Read the committed O(1) status projection without waiting for audits."""

        return await asyncio.to_thread(self._operation_diagnostics)

    async def close(self) -> None:
        """Flush capability audits and stop the lazy writer."""

        if self._closed:
            return
        self._closed = True
        await self._flush_audit_queue()
        task = self._audit_writer_task
        if task is None or task.done():
            if task is not None:
                self._consume_writer_result(task)
            self._audit_writer_task = None
            return
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        await self._audit_queue.put(_CloseAuditQueue(completed))
        await completed
        await task
        self._audit_writer_task = None

    def _ensure_audit_writer(self) -> None:
        task = self._audit_writer_task
        if task is not None and task.done():
            self._consume_writer_result(task)
            task = None
            self._audit_writer_task = None
        if task is None:
            self._audit_writer_task = asyncio.create_task(
                self._audit_writer(),
                name="simajilord-event-journal-audit-writer",
            )

    @staticmethod
    def _consume_writer_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            log.warning("Capability audit writer was cancelled.")
        except Exception:
            log.exception("Capability audit writer stopped unexpectedly.")

    async def _flush_audit_queue(self) -> None:
        task = self._audit_writer_task
        if self._audit_queue.empty() and (task is None or task.done()):
            if task is not None:
                self._consume_writer_result(task)
                self._audit_writer_task = None
            return
        self._ensure_audit_writer()
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        await self._audit_queue.put(_FlushAuditQueue(completed))
        await completed

    async def _audit_writer(self) -> None:
        while True:
            item = await self._audit_queue.get()
            if isinstance(item, _PendingEvent):
                batch = [item]
                control: _FlushAuditQueue | _CloseAuditQueue | None = None
                while len(batch) < _AUDIT_BATCH_MAX_EVENTS:
                    try:
                        queued = self._audit_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if isinstance(queued, _PendingEvent):
                        batch.append(queued)
                    else:
                        control = queued
                        break
                await self._write_audit_batch_safely(tuple(batch))
                for _ in batch:
                    self._audit_queue.task_done()
                if control is not None:
                    should_close = isinstance(control, _CloseAuditQueue)
                    if not control.completed.done():
                        control.completed.set_result(None)
                    self._audit_queue.task_done()
                    if should_close:
                        return
                continue
            should_close = isinstance(item, _CloseAuditQueue)
            if not item.completed.done():
                item.completed.set_result(None)
            self._audit_queue.task_done()
            if should_close:
                return

    async def _write_audit_batch_safely(
        self,
        batch: tuple[_PendingEvent, ...],
    ) -> None:
        try:
            async with self._write_lock:
                await asyncio.to_thread(self._insert_batch, batch)
        except Exception:
            log.exception(
                "Capability audit batch persistence failed events=%s",
                len(batch),
            )

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
            connection.execute(
                "CREATE INDEX IF NOT EXISTS events_kind_sequence "
                "ON events(kind, sequence)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_diagnostics (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    last_radio_failure_at TEXT,
                    overlay_failure_count INTEGER NOT NULL,
                    dashboard_429_count INTEGER NOT NULL,
                    projected_through_sequence INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO operation_diagnostics(
                    singleton,
                    last_radio_failure_at,
                    overlay_failure_count,
                    dashboard_429_count,
                    projected_through_sequence
                ) VALUES (1, NULL, 0, 0, 0)
                """
            )
            projected = connection.execute(
                """
                SELECT projected_through_sequence
                FROM operation_diagnostics
                WHERE singleton = 1
                """
            ).fetchone()
            retained = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0)
                FROM events
                WHERE kind = 'service.operation'
                """
            ).fetchone()
            if (
                projected is None
                or retained is None
                or int(projected[0]) != int(retained[0])
            ):
                self._rebuild_operation_projection(connection)
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _insert(self, event: _PendingEvent) -> int:
        connection = sqlite3.connect(self.path)
        try:
            sequence = self._insert_one(connection, event)
            connection.commit()
            return sequence
        finally:
            connection.close()

    def _insert_batch(self, events: tuple[_PendingEvent, ...]) -> None:
        connection = sqlite3.connect(self.path)
        try:
            for event in events:
                self._insert_one(connection, event)
            connection.commit()
        finally:
            connection.close()

    def _insert_one(
        self,
        connection: sqlite3.Connection,
        event: _PendingEvent,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO events(
                occurred_at, kind, actor_id, workspace_id, transport,
                request_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.occurred_at,
                event.kind,
                event.actor_id,
                event.workspace_id,
                event.transport,
                event.request_id,
                json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an event sequence.")
        sequence = int(cursor.lastrowid)
        self._project_operation_event(connection, sequence, event)
        return sequence

    @staticmethod
    def _project_operation_event(
        connection: sqlite3.Connection,
        sequence: int,
        event: _PendingEvent,
    ) -> None:
        if event.kind != "service.operation":
            return
        operation = event.payload.get("operation")
        outcome = event.payload.get("outcome")
        radio_failed = operation == "audio.autoplay_refill" and outcome == "failed"
        connection.execute(
            """
            UPDATE operation_diagnostics
            SET
                last_radio_failure_at = CASE
                    WHEN ? THEN ?
                    ELSE last_radio_failure_at
                END,
                overlay_failure_count = overlay_failure_count + ?,
                dashboard_429_count = dashboard_429_count + ?,
                projected_through_sequence = MAX(projected_through_sequence, ?)
            WHERE singleton = 1
            """,
            (
                radio_failed,
                event.occurred_at,
                int(operation == "audio.overlay_failed"),
                int(operation == "discord.dashboard_429"),
                sequence,
            ),
        )

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

    def _prune(self, cutoff: str) -> int:
        connection = sqlite3.connect(self.path)
        try:
            cursor = connection.execute(
                "DELETE FROM events WHERE occurred_at < ?",
                (cutoff,),
            )
            self._rebuild_operation_projection(connection)
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _operation_diagnostics(self) -> OperationDiagnostics:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """
                SELECT
                    last_radio_failure_at,
                    overlay_failure_count,
                    dashboard_429_count
                FROM operation_diagnostics
                WHERE singleton = 1
                """
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return OperationDiagnostics(None, 0, 0)
        raw_radio_failure = row["last_radio_failure_at"]
        return OperationDiagnostics(
            last_radio_failure_at=(
                datetime.fromisoformat(str(raw_radio_failure))
                if raw_radio_failure is not None
                else None
            ),
            overlay_failure_count=int(row["overlay_failure_count"]),
            dashboard_429_count=int(row["dashboard_429_count"]),
        )

    @staticmethod
    def _rebuild_operation_projection(connection: sqlite3.Connection) -> None:
        """Rebuild the retained-event projection during migration or pruning."""

        summary = connection.execute(
            """
            WITH service_events AS (
                SELECT
                    sequence,
                    CASE
                        WHEN json_valid(payload_json)
                        THEN json_extract(payload_json, '$.operation')
                    END AS operation
                FROM events
                WHERE kind = 'service.operation'
            )
            SELECT
                COALESCE(SUM(operation = 'audio.overlay_failed'), 0),
                COALESCE(SUM(operation = 'discord.dashboard_429'), 0),
                COALESCE(MAX(sequence), 0)
            FROM service_events
            """
        ).fetchone()
        radio = connection.execute(
            """
            SELECT occurred_at
            FROM events
            WHERE kind = 'service.operation'
              AND json_valid(payload_json)
              AND json_extract(payload_json, '$.operation') = 'audio.autoplay_refill'
              AND json_extract(payload_json, '$.outcome') = 'failed'
            ORDER BY sequence DESC
            LIMIT 1
            """
        ).fetchone()
        assert summary is not None
        connection.execute(
            """
            UPDATE operation_diagnostics
            SET
                last_radio_failure_at = ?,
                overlay_failure_count = ?,
                dashboard_429_count = ?,
                projected_through_sequence = ?
            WHERE singleton = 1
            """,
            (
                str(radio[0]) if radio is not None else None,
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            ),
        )

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


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Retention cutoffs must be timezone-aware.")
    return value.astimezone(UTC).isoformat()
