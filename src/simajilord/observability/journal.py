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
from uuid import uuid4

from simajilord.core.capabilities import CapabilityAuditPayload, InvocationContext

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
_AUDIT_OUTBOX_MAX_EVENTS = 16_384
_AUDIT_OVERFLOW_TASK_MAX = 128
_AUDIT_RETRY_DELAYS_SECONDS = (0.05, 0.2)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    event_id: str
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
class AuditHealth:
    """Current asynchronous audit durability state."""

    pending_events: int
    retried_event_count: int
    outbox_event_count: int
    lost_event_count: int
    last_failure_at: datetime | None
    last_failure_type: str | None
    writer_state: str


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    event_id: str
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
        self.outbox_path = path.with_name("audit_outbox.sqlite3")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._outbox_lock = asyncio.Lock()
        self._audit_queue: asyncio.Queue[
            _PendingEvent | _FlushAuditQueue | _CloseAuditQueue
        ] = asyncio.Queue(maxsize=_AUDIT_QUEUE_MAX_EVENTS)
        self._audit_writer_task: asyncio.Task[None] | None = None
        self._audit_inflight_event_count = 0
        self._overflow_tasks: set[asyncio.Task[None]] = set()
        self._retried_event_count = 0
        self._lost_event_count = 0
        self._last_audit_failure_at: datetime | None = None
        self._last_audit_failure_type: str | None = None
        self._closed = False
        self._initialize()
        self._initialize_outbox()

    async def record_invocation(
        self,
        *,
        capability_name: str,
        audit_payload: CapabilityAuditPayload = "full",
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
            "public_reference_id": context.public_reference_id,
            "task_id": context.agent_task_id,
            "provider_thread_id": context.provider_thread_id,
            "provider_turn_id": context.provider_turn_id,
            "tool_call_id": context.tool_call_id,
            "executor_principal_id": context.executor_principal_id,
            "delegator_principal_id": context.delegator_principal_id,
            "trigger_actor_ids": list(context.trigger_actor_ids),
            "requester_principal_id": context.requester_principal_id,
            "principal_kind": context.principal_kind,
            "policy_id": context.policy_id,
        }
        if audit_payload == "full":
            payload["request"] = _safe_value(request)
            if error is None:
                payload["response"] = _safe_value(response)
        else:
            payload["request_fields"] = _value_field_names(request)
            payload["response_fields"] = _value_field_names(response)
        if error is not None:
            payload["error_type"] = type(error).__name__
            if audit_payload == "full":
                payload["error_message"] = _bounded(str(error), 1_000)
        if self._closed:
            raise RuntimeError("Event journal is closed.")
        self._ensure_audit_writer()
        pending = _PendingEvent(
            event_id=_new_event_id(),
            occurred_at=datetime.now(UTC).isoformat(),
            kind="capability.invocation",
            payload=cast_payload(_safe_value(payload)),
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            transport=context.transport,
            request_id=context.request_id,
        )
        try:
            self._audit_queue.put_nowait(pending)
        except asyncio.QueueFull:
            self._schedule_overflow_spool(pending)

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
        self._ensure_audit_writer()
        pending = _PendingEvent(
            event_id=_new_event_id(),
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

    async def agent_trace(
        self,
        *,
        request_id: str | None = None,
        public_reference_id: str | None = None,
        task_id: str | None = None,
        limit: int = 500,
    ) -> tuple[EventRecord, ...]:
        """Return one bounded, chronological agent trace without body search."""

        if request_id is None and public_reference_id is None and task_id is None:
            raise ValueError("request_id, public_reference_id, or task_id is required")
        if request_id is not None and not request_id.strip():
            raise ValueError("request_id must not be empty")
        if public_reference_id is not None and not public_reference_id.strip():
            raise ValueError("public_reference_id must not be empty")
        if task_id is not None and not task_id.strip():
            raise ValueError("task_id must not be empty")
        bounded_limit = min(max(limit, 1), 1_000)
        await self._flush_audit_queue()
        return await asyncio.to_thread(
            self._select_agent_trace,
            request_id,
            public_reference_id,
            task_id,
            bounded_limit,
        )

    async def prune(self, *, before: datetime) -> int:
        """Delete retained events older than an explicit UTC cutoff."""

        cutoff = _utc_iso(before)
        await self._flush_audit_queue()
        async with self._write_lock:
            return await asyncio.to_thread(self._prune, cutoff)

    async def operation_diagnostics(self) -> OperationDiagnostics:
        """Read the committed O(1) status projection without waiting for audits."""

        return await asyncio.to_thread(self._operation_diagnostics)

    async def audit_health(self) -> AuditHealth:
        """Return bounded audit queue, retry, outbox, and loss health."""

        async with self._outbox_lock:
            outbox_count = await asyncio.to_thread(self._outbox_count)
        task = self._audit_writer_task
        if self._closed and (task is None or task.done()):
            writer_state = "closed"
        elif task is None:
            writer_state = "idle"
        elif task.done():
            writer_state = "failed"
        else:
            writer_state = "running"
        return AuditHealth(
            pending_events=(
                self._audit_queue.qsize() + self._audit_inflight_event_count
            ),
            retried_event_count=self._retried_event_count,
            outbox_event_count=outbox_count,
            lost_event_count=self._lost_event_count,
            last_failure_at=self._last_audit_failure_at,
            last_failure_type=self._last_audit_failure_type,
            writer_state=writer_state,
        )

    async def close(self) -> None:
        """Flush capability audits and stop the lazy writer."""

        if self._closed:
            return
        self._closed = True
        await self._await_overflow_tasks()
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
        await self._await_audit_control(completed)
        task = self._audit_writer_task
        if task is not None:
            await task
        self._audit_writer_task = None

    def _schedule_overflow_spool(self, event: _PendingEvent) -> None:
        if len(self._overflow_tasks) >= _AUDIT_OVERFLOW_TASK_MAX:
            self._lost_event_count += 1
            log.critical(
                "Capability audit overflow task limit reached; event lost event_id=%s",
                event.event_id,
            )
            return
        task = asyncio.create_task(
            self._spool_overflow_event(event),
            name=f"simajilord-event-journal-overflow-{event.event_id}",
        )
        self._overflow_tasks.add(task)
        task.add_done_callback(self._overflow_tasks.discard)

    async def _spool_overflow_event(self, event: _PendingEvent) -> None:
        if await self._spool_batch_safely((event,)):
            return
        self._lost_event_count += 1
        log.critical(
            "Capability audit queue overflow could not be spooled; event lost event_id=%s",
            event.event_id,
        )

    async def _await_overflow_tasks(self) -> None:
        tasks = tuple(self._overflow_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

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
        await self._await_overflow_tasks()
        task = self._audit_writer_task
        async with self._outbox_lock:
            outbox_pending = await asyncio.to_thread(self._outbox_count)
        if (
            self._audit_queue.empty()
            and outbox_pending == 0
            and (task is None or task.done())
        ):
            if task is not None:
                self._consume_writer_result(task)
                self._audit_writer_task = None
            return
        self._ensure_audit_writer()
        task = self._audit_writer_task
        assert task is not None
        loop = asyncio.get_running_loop()
        completed = loop.create_future()
        await self._audit_queue.put(_FlushAuditQueue(completed))
        await self._await_audit_control(completed)

    async def _await_audit_control(
        self,
        completed: asyncio.Future[None],
    ) -> None:
        """Wait without hanging when a recovered writer exits before control."""

        while not completed.done():
            task = self._audit_writer_task
            if task is None or task.done():
                if task is not None:
                    self._consume_writer_result(task)
                    if self._audit_writer_task is task:
                        self._audit_writer_task = None
                if self._audit_queue.empty():
                    raise RuntimeError(
                        "Capability audit writer lost a flush control message."
                    )
                self._ensure_audit_writer()
                task = self._audit_writer_task
                assert task is not None
            done, _ = await asyncio.wait(
                {completed, task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if completed in done:
                break
        await completed

    async def _audit_writer(self) -> None:
        batch: tuple[_PendingEvent, ...] = ()
        control: _FlushAuditQueue | _CloseAuditQueue | None = None
        try:
            await self._replay_outbox_safely()
            while True:
                item = await self._audit_queue.get()
                if isinstance(item, _PendingEvent):
                    pending = [item]
                    while len(pending) < _AUDIT_BATCH_MAX_EVENTS:
                        try:
                            queued = self._audit_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if isinstance(queued, _PendingEvent):
                            pending.append(queued)
                        else:
                            control = queued
                            break
                    batch = tuple(pending)
                    self._audit_inflight_event_count = len(batch)
                    await self._write_audit_batch_safely(batch)
                    for _ in batch:
                        self._audit_queue.task_done()
                    batch = ()
                    self._audit_inflight_event_count = 0
                    if control is not None:
                        await self._replay_outbox_safely(drain=True)
                        should_close = isinstance(control, _CloseAuditQueue)
                        if not control.completed.done():
                            control.completed.set_result(None)
                        self._audit_queue.task_done()
                        control = None
                        if should_close:
                            return
                    else:
                        await self._replay_outbox_safely()
                    continue
                control = item
                await self._replay_outbox_safely(drain=True)
                should_close = isinstance(control, _CloseAuditQueue)
                if not control.completed.done():
                    control.completed.set_result(None)
                self._audit_queue.task_done()
                control = None
                if should_close:
                    return
        except BaseException as exc:
            await asyncio.shield(
                self._recover_failed_audit_writer(
                    batch=batch,
                    control=control,
                    error=exc,
                )
            )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _recover_failed_audit_writer(
        self,
        *,
        batch: tuple[_PendingEvent, ...],
        control: _FlushAuditQueue | _CloseAuditQueue | None,
        error: BaseException,
    ) -> None:
        """Spool every dequeued or queued event and release control waiters."""

        if (
            isinstance(error, asyncio.CancelledError)
            and not batch
            and control is None
            and self._audit_queue.empty()
        ):
            self._audit_inflight_event_count = 0
            return
        if isinstance(error, Exception):
            self._record_audit_failure(error)
        else:
            self._last_audit_failure_at = datetime.now(UTC)
            self._last_audit_failure_type = type(error).__name__
        if isinstance(error, asyncio.CancelledError):
            log.warning(
                "Capability audit writer was cancelled with pending work; "
                "recovering to outbox"
            )
        else:
            log.error(
                "Capability audit writer stopped unexpectedly; recovering to outbox "
                "error=%s",
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )
        durable = True
        recovered_controls: list[_FlushAuditQueue | _CloseAuditQueue] = []

        async def spool(events: tuple[_PendingEvent, ...]) -> None:
            nonlocal durable
            if not events:
                return
            persisted = await self._spool_batch_safely(events)
            if not persisted:
                durable = False
                self._lost_event_count += len(events)

        await spool(batch)
        for _ in batch:
            self._audit_queue.task_done()
        self._audit_inflight_event_count = 0

        if control is not None:
            recovered_controls.append(control)
            self._audit_queue.task_done()

        pending: list[_PendingEvent] = []
        while True:
            try:
                queued = self._audit_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(queued, _PendingEvent):
                pending.append(queued)
                if len(pending) >= _AUDIT_BATCH_MAX_EVENTS:
                    await spool(tuple(pending))
                    for _ in pending:
                        self._audit_queue.task_done()
                    pending.clear()
                continue
            await spool(tuple(pending))
            for _ in pending:
                self._audit_queue.task_done()
            pending.clear()
            recovered_controls.append(queued)
            self._audit_queue.task_done()
        await spool(tuple(pending))
        for _ in pending:
            self._audit_queue.task_done()
        if durable:
            await self._replay_outbox_safely(drain=True)
        for recovered_control in recovered_controls:
            self._complete_recovered_control(
                recovered_control,
                durable=durable,
            )

    @staticmethod
    def _complete_recovered_control(
        control: _FlushAuditQueue | _CloseAuditQueue,
        *,
        durable: bool,
    ) -> None:
        if control.completed.done():
            return
        if durable:
            control.completed.set_result(None)
        else:
            control.completed.set_exception(
                RuntimeError("Capability audit events could not be recovered.")
            )

    async def _write_audit_batch_safely(
        self,
        batch: tuple[_PendingEvent, ...],
    ) -> None:
        for attempt in range(len(_AUDIT_RETRY_DELAYS_SECONDS) + 1):
            try:
                async with self._write_lock:
                    await asyncio.to_thread(self._insert_batch, batch)
                return
            except Exception as exc:
                self._record_audit_failure(exc)
                if attempt >= len(_AUDIT_RETRY_DELAYS_SECONDS):
                    break
                self._retried_event_count += len(batch)
                await asyncio.sleep(_AUDIT_RETRY_DELAYS_SECONDS[attempt])
        log.error(
            "Capability audit batch exhausted retries; moving to outbox events=%s",
            len(batch),
            exc_info=True,
        )
        if await self._spool_batch_safely(batch):
            return
        self._lost_event_count += len(batch)
        log.critical(
            "Capability audit batch could not be persisted or spooled; events lost=%s",
            len(batch),
        )

    def _record_audit_failure(self, error: Exception) -> None:
        self._last_audit_failure_at = datetime.now(UTC)
        self._last_audit_failure_type = type(error).__name__

    async def _spool_batch_safely(
        self,
        batch: tuple[_PendingEvent, ...],
    ) -> bool:
        try:
            async with self._outbox_lock:
                await asyncio.to_thread(self._outbox_put, batch)
            return True
        except Exception as exc:
            self._record_audit_failure(exc)
            log.exception(
                "Capability audit outbox persistence failed events=%s",
                len(batch),
            )
            return False

    async def _replay_outbox_safely(self, *, drain: bool = False) -> None:
        max_batches = _AUDIT_OUTBOX_MAX_EVENTS if drain else 4
        for _ in range(max_batches):
            async with self._outbox_lock:
                batch = await asyncio.to_thread(
                    self._outbox_select,
                    _AUDIT_BATCH_MAX_EVENTS,
                )
            if not batch:
                return
            self._retried_event_count += len(batch)
            try:
                async with self._write_lock:
                    await asyncio.to_thread(self._insert_batch, batch)
            except Exception as exc:
                self._record_audit_failure(exc)
                log.warning(
                    "Capability audit outbox replay deferred events=%s error=%s",
                    len(batch),
                    type(exc).__name__,
                )
                return
            try:
                async with self._outbox_lock:
                    await asyncio.to_thread(
                        self._outbox_delete,
                        tuple(item.event_id for item in batch),
                    )
            except Exception as exc:
                self._record_audit_failure(exc)
                log.exception(
                    "Capability audit outbox cleanup failed events=%s",
                    len(batch),
                )
                return

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
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
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            if "event_id" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN event_id TEXT")
            connection.execute(
                """
                UPDATE events
                SET event_id = 'legacy:' || sequence
                WHERE event_id IS NULL OR event_id = ''
                """
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS events_event_id "
                "ON events(event_id)"
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
                "CREATE INDEX IF NOT EXISTS events_request_sequence "
                "ON events(request_id, sequence)"
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS events_public_reference_sequence
                ON events(
                    json_extract(payload_json, '$.public_reference_id'),
                    sequence
                )
                WHERE json_valid(payload_json)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS events_agent_task_sequence
                ON events(
                    json_extract(payload_json, '$.task_id'),
                    sequence
                )
                WHERE json_valid(payload_json)
                """
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

    def _initialize_outbox(self) -> None:
        connection = sqlite3.connect(self.outbox_path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_outbox (
                    event_id TEXT PRIMARY KEY,
                    occurred_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    actor_id TEXT,
                    workspace_id TEXT,
                    transport TEXT,
                    request_id TEXT,
                    payload_json TEXT NOT NULL,
                    spooled_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS audit_outbox_spooled "
                "ON audit_outbox(spooled_at, event_id)"
            )
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.outbox_path, 0o600)

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
                event_id, occurred_at, kind, actor_id, workspace_id, transport,
                request_id, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO NOTHING
            """,
            (
                event.event_id,
                event.occurred_at,
                event.kind,
                event.actor_id,
                event.workspace_id,
                event.transport,
                event.request_id,
                json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        inserted = cursor.rowcount > 0
        if inserted:
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an event sequence.")
            sequence = int(cursor.lastrowid)
        else:
            row = connection.execute(
                "SELECT sequence FROM events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("SQLite ignored an event without retaining its ID.")
            sequence = int(row[0])
        if not inserted:
            return sequence
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
                _event_record_from_row(row)
                for row in rows
            )
        finally:
            connection.close()

    def _select_agent_trace(
        self,
        request_id: str | None,
        public_reference_id: str | None,
        task_id: str | None,
        limit: int,
    ) -> tuple[EventRecord, ...]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            clauses: list[str] = []
            values: list[object] = []
            if request_id is not None:
                clauses.append("request_id = ?")
                values.append(request_id)
            if public_reference_id is not None:
                clauses.append(
                    """
                    (
                        json_valid(payload_json)
                        AND json_extract(
                            payload_json,
                            '$.public_reference_id'
                        ) = ?
                    )
                    """
                )
                values.append(public_reference_id)
            if task_id is not None:
                clauses.append(
                    """
                    (
                        json_valid(payload_json)
                        AND json_extract(payload_json, '$.task_id') = ?
                    )
                    """
                )
                values.append(task_id)
            rows = connection.execute(
                f"""
                SELECT *
                FROM events
                WHERE {" OR ".join(clauses)}
                ORDER BY sequence
                LIMIT ?
                """,
                (*values, limit),
            ).fetchall()
            return tuple(_event_record_from_row(row) for row in rows)
        finally:
            connection.close()

    def _outbox_put(self, events: tuple[_PendingEvent, ...]) -> None:
        connection = sqlite3.connect(self.outbox_path, timeout=10)
        try:
            existing_ids = {
                str(row[0])
                for row in connection.execute(
                    "SELECT event_id FROM audit_outbox WHERE event_id IN "
                    f"({','.join('?' for _ in events)})",
                    tuple(event.event_id for event in events),
                ).fetchall()
            }
            current_count = int(
                connection.execute("SELECT COUNT(*) FROM audit_outbox").fetchone()[0]
            )
            new_count = sum(event.event_id not in existing_ids for event in events)
            if current_count + new_count > _AUDIT_OUTBOX_MAX_EVENTS:
                raise RuntimeError("Capability audit outbox is full.")
            spooled_at = datetime.now(UTC).isoformat()
            connection.executemany(
                """
                INSERT OR IGNORE INTO audit_outbox(
                    event_id, occurred_at, kind, actor_id, workspace_id,
                    transport, request_id, payload_json, spooled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        event.event_id,
                        event.occurred_at,
                        event.kind,
                        event.actor_id,
                        event.workspace_id,
                        event.transport,
                        event.request_id,
                        json.dumps(
                            event.payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        spooled_at,
                    )
                    for event in events
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _outbox_select(self, limit: int) -> tuple[_PendingEvent, ...]:
        connection = sqlite3.connect(self.outbox_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT
                    event_id, occurred_at, kind, actor_id, workspace_id,
                    transport, request_id, payload_json
                FROM audit_outbox
                ORDER BY spooled_at, event_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return tuple(
                _PendingEvent(
                    event_id=str(row["event_id"]),
                    occurred_at=str(row["occurred_at"]),
                    kind=str(row["kind"]),
                    actor_id=row["actor_id"],
                    workspace_id=row["workspace_id"],
                    transport=row["transport"],
                    request_id=row["request_id"],
                    payload=cast_payload(json.loads(str(row["payload_json"]))),
                )
                for row in rows
            )
        finally:
            connection.close()

    def _outbox_delete(self, event_ids: tuple[str, ...]) -> None:
        if not event_ids:
            return
        connection = sqlite3.connect(self.outbox_path, timeout=10)
        try:
            connection.execute(
                "DELETE FROM audit_outbox WHERE event_id IN "
                f"({','.join('?' for _ in event_ids)})",
                event_ids,
            )
            connection.commit()
        finally:
            connection.close()

    def _outbox_count(self) -> int:
        connection = sqlite3.connect(self.outbox_path, timeout=10)
        try:
            row = connection.execute("SELECT COUNT(*) FROM audit_outbox").fetchone()
            return int(row[0])
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


def _event_record_from_row(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        sequence=int(row["sequence"]),
        event_id=str(row["event_id"]),
        occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
        kind=str(row["kind"]),
        actor_id=row["actor_id"],
        workspace_id=row["workspace_id"],
        transport=row["transport"],
        request_id=row["request_id"],
        payload=cast_payload(json.loads(str(row["payload_json"]))),
    )


def _safe_value(value: object, *, key: str = "") -> object:
    if (
        key == "authorization_reference_id"
        and isinstance(value, str)
        and value.startswith("authref_")
        and len(value) == 28
        and all(character in "0123456789abcdef" for character in value[8:])
    ):
        return value
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


def _value_field_names(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return tuple(field.name for field in dataclasses.fields(value))
    if isinstance(value, dict):
        return tuple(sorted(_bounded(str(key), 200) for key in value))
    return (type(value).__name__,)


def _new_event_id() -> str:
    return f"evt_{uuid4().hex}"


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
