"""Durable, event-driven batching for autonomous agent observations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

_SENSITIVE_PARTS = (
    "token",
    "secret",
    "password",
    "cookie",
    "authorization",
    "data_url",
)
_PROCESSED_EVENT_RETENTION = timedelta(days=30)
_DELIVERY_PURPOSE_MAX_CHARACTERS = 50
_DELIVERY_NONCE_MAX_CHARACTERS = 25
_CONTENT_SHA256_CHARACTERS = 64
_ACTOR_LIMITED_EVENT_KINDS = frozenset(
    {
        "discord.message.created",
        "discord.message.edited",
        "discord.reaction.added",
        "discord.thread.created",
        "discord.voice_state.updated",
    }
)
_EVENT_COLUMNS = """
    sequence,
    deduplication_key,
    kind,
    occurred_at,
    enqueued_at,
    not_before,
    failure_count,
    deferral_count,
    workspace_id,
    channel_id,
    actor_id,
    message_id,
    payload_json
"""


class AutonomyEventKind(StrEnum):
    """Event kinds accepted by the shared autonomous-agent queue."""

    MESSAGE_CREATE = "discord.message.created"
    MESSAGE_EDIT = "discord.message.edited"
    REACTION_ADD = "discord.reaction.added"
    THREAD_CREATE = "discord.thread.created"
    VOICE_STATE_UPDATE = "discord.voice_state.updated"
    TIMER_DUE = "timer.due"
    AUDIO_ERROR = "audio.error"
    GITHUB_UPDATE = "github.update"
    RSS_UPDATE = "rss.update"


class AutonomyEnqueueResult(StrEnum):
    """Why an event was or was not added to the durable queue."""

    INSERTED = "inserted"
    DUPLICATE = "duplicate"
    QUEUE_FULL = "queue_full"
    CHANNEL_QUEUE_FULL = "channel_queue_full"
    ACTOR_QUEUE_FULL = "actor_queue_full"


class AutonomyLeaseLostError(RuntimeError):
    """A stale worker tried to acknowledge a batch it no longer owns."""


class AutonomyDeliveryConflictError(RuntimeError):
    """A stable delivery key was reused for different immutable output."""


class AutonomyDeliveryReceiptState(StrEnum):
    """Receipt progress retained independently from Discord delivery."""

    PENDING = "pending"
    RECORDED = "recorded"


@dataclass(frozen=True, slots=True)
class AutonomyDeliverySpec:
    """Body-free immutable identity for one rendered Discord chunk."""

    purpose: str
    chunk_index: int
    content_sha256: str
    nonce: str


@dataclass(frozen=True, slots=True)
class AutonomyDeliveryRecord:
    """Durable reconciliation state without retaining response content."""

    batch_id: str
    chunk_index: int
    purpose: str
    channel_id: str
    content_sha256: str
    nonce: str
    message_id: str | None
    receipt_state: AutonomyDeliveryReceiptState
    prepared_at: datetime
    sent_at: datetime | None
    receipted_at: datetime | None


@dataclass(frozen=True, slots=True)
class AutonomyQueuedEvent:
    """One content-free event pointer retained until a completed AI turn."""

    sequence: int
    deduplication_key: str
    kind: AutonomyEventKind
    occurred_at: datetime
    enqueued_at: datetime
    not_before: datetime
    attempt_count: int
    workspace_id: str
    channel_id: str
    actor_id: str | None
    message_id: str | None
    payload: dict[str, object]
    deferral_count: int = 0


@dataclass(frozen=True, slots=True)
class AutonomyEventBatch:
    """A same-channel batch presented to one autonomous agent turn."""

    batch_id: str
    claim_token: str
    lease_until: datetime
    attempt_count: int
    workspace_id: str
    channel_id: str
    events: tuple[AutonomyQueuedEvent, ...]
    deferral_count: int = 0

    @property
    def message_id(self) -> str | None:
        for event in reversed(self.events):
            if event.message_id is not None:
                return event.message_id
        return None


class AutonomyEventQueue:
    """SQLite-backed queue with debounce, deduplication, and bounded admission."""

    def __init__(
        self,
        path: Path,
        *,
        max_pending_events: int,
        max_pending_events_per_channel: int,
        max_pending_events_per_actor: int,
    ) -> None:
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be positive.")
        if not 1 <= max_pending_events_per_channel <= max_pending_events:
            raise ValueError(
                "max_pending_events_per_channel must be positive and no greater "
                "than max_pending_events."
            )
        if not 1 <= max_pending_events_per_actor <= max_pending_events:
            raise ValueError(
                "max_pending_events_per_actor must be positive and no greater "
                "than max_pending_events."
            )
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.max_pending_events = max_pending_events
        self.max_pending_events_per_channel = max_pending_events_per_channel
        self.max_pending_events_per_actor = max_pending_events_per_actor
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._initialize()

    async def enqueue(
        self,
        *,
        kind: AutonomyEventKind,
        deduplication_key: str,
        workspace_id: str,
        channel_id: str,
        occurred_at: datetime,
        actor_id: str | None = None,
        message_id: str | None = None,
        payload: dict[str, object] | None = None,
        enqueued_at: datetime | None = None,
    ) -> AutonomyEnqueueResult:
        """Persist a bounded pointer and wake the consumer after commit.

        ``occurred_at`` is source evidence, while the locally controlled
        ``enqueued_at`` starts the batching window. Normal adapters must omit
        ``enqueued_at``; it exists only to make clock-sensitive tests deterministic.
        """

        normalized_key = deduplication_key.strip()
        if not normalized_key or len(normalized_key) > 500:
            raise ValueError("deduplication_key must contain 1 to 500 characters.")
        if not workspace_id.isdigit() or int(workspace_id) <= 0:
            raise ValueError("workspace_id must be a positive Discord ID.")
        if not channel_id.isdigit() or int(channel_id) <= 0:
            raise ValueError("channel_id must be a positive Discord ID.")
        normalized_payload = _safe_payload(payload or {})
        payload_json = json.dumps(
            normalized_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(payload_json) > 4_000:
            raise ValueError("autonomy event payload exceeds 4,000 characters.")
        normalized_time = _as_utc(occurred_at)
        admitted_at = _as_utc(enqueued_at or datetime.now(UTC))
        async with self._lock:
            result = await asyncio.to_thread(
                self._insert,
                kind.value,
                normalized_key,
                workspace_id,
                channel_id,
                actor_id,
                message_id,
                normalized_time.isoformat(),
                admitted_at.isoformat(),
                payload_json,
            )
        if result is AutonomyEnqueueResult.INSERTED:
            self._wake.set()
        return result

    async def next_batch(
        self,
        *,
        debounce_seconds: int,
        candidate_limit: int,
        lease_seconds: int = 1_860,
        now: datetime | None = None,
    ) -> AutonomyEventBatch | None:
        """Atomically claim the oldest eligible, fixed-membership channel batch."""

        _validate_batch_options(debounce_seconds, candidate_limit)
        if not 60 <= lease_seconds <= 7_200:
            raise ValueError("lease_seconds must be between 60 and 7,200.")
        current = _as_utc(now or datetime.now(UTC))
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_next_batch,
                debounce_seconds,
                candidate_limit,
                current.isoformat(),
                lease_seconds,
            )

    async def seconds_until_ready(
        self,
        *,
        debounce_seconds: int,
        candidate_limit: int,
        now: datetime | None = None,
    ) -> float | None:
        """Return a bounded timer only for already-persisted deferred work."""

        _validate_batch_options(debounce_seconds, candidate_limit)
        current = _as_utc(now or datetime.now(UTC))
        async with self._lock:
            ready = await asyncio.to_thread(
                self._next_ready_at,
                debounce_seconds,
                candidate_limit,
                current.isoformat(),
            )
        if ready is None:
            return None
        return max(0.0, (ready - current).total_seconds())

    async def renew_lease(
        self,
        batch: AutonomyEventBatch,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> datetime:
        """Extend a live claim using a whole-batch compare-and-swap."""

        if not 30 <= lease_seconds <= 7_200:
            raise ValueError("lease_seconds must be between 30 and 7,200.")
        current = _as_utc(now or datetime.now(UTC))
        lease_until = current + timedelta(seconds=lease_seconds)
        identifiers = tuple(event.sequence for event in batch.events)
        if not identifiers:
            raise ValueError("cannot renew an empty autonomy batch")
        async with self._lock:
            updated = await asyncio.to_thread(
                self._renew_lease,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                current.isoformat(),
                lease_until.isoformat(),
            )
        if updated != len(identifiers):
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before renewal: {batch.batch_id}"
            )
        return lease_until

    async def prepare_deliveries(
        self,
        batch: AutonomyEventBatch,
        specs: tuple[AutonomyDeliverySpec, ...],
        *,
        now: datetime | None = None,
    ) -> tuple[AutonomyDeliveryRecord, ...]:
        """Atomically persist an immutable, body-free delivery plan."""

        if not specs:
            return ()
        normalized_specs = tuple(_validated_delivery_spec(spec) for spec in specs)
        keys = {(spec.chunk_index, spec.purpose) for spec in normalized_specs}
        if len(keys) != len(normalized_specs):
            raise ValueError("autonomy delivery specs contain duplicate keys")
        timestamp = _as_utc(now or datetime.now(UTC)).isoformat()
        identifiers = tuple(event.sequence for event in batch.events)
        async with self._lock:
            records = await asyncio.to_thread(
                self._prepare_deliveries,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                batch.channel_id,
                normalized_specs,
                timestamp,
            )
        if records is None:
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before delivery planning: "
                f"{batch.batch_id}"
            )
        return records

    async def mark_delivery_sent(
        self,
        batch: AutonomyEventBatch,
        *,
        purpose: str,
        chunk_index: int,
        message_id: str,
        sent_at: datetime | None = None,
    ) -> AutonomyDeliveryRecord:
        """Persist the Discord message ID before receipt or queue ACK."""

        normalized_purpose = _validated_delivery_purpose(purpose)
        if chunk_index < 0:
            raise ValueError("delivery chunk_index must be non-negative")
        normalized_message_id = message_id.strip()
        if not normalized_message_id.isdigit() or int(normalized_message_id) <= 0:
            raise ValueError("delivery message_id must be a positive Discord ID")
        timestamp = _as_utc(sent_at or datetime.now(UTC)).isoformat()
        identifiers = tuple(event.sequence for event in batch.events)
        async with self._lock:
            record = await asyncio.to_thread(
                self._mark_delivery_sent,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                normalized_purpose,
                chunk_index,
                normalized_message_id,
                timestamp,
            )
        if record is None:
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before delivery persistence: "
                f"{batch.batch_id}"
            )
        return record

    async def mark_delivery_receipted(
        self,
        batch: AutonomyEventBatch,
        *,
        purpose: str,
        chunk_index: int,
        receipted_at: datetime | None = None,
    ) -> AutonomyDeliveryRecord:
        """Mark receipt persistence separately from Discord send success."""

        normalized_purpose = _validated_delivery_purpose(purpose)
        if chunk_index < 0:
            raise ValueError("delivery chunk_index must be non-negative")
        timestamp = _as_utc(receipted_at or datetime.now(UTC)).isoformat()
        identifiers = tuple(event.sequence for event in batch.events)
        async with self._lock:
            record = await asyncio.to_thread(
                self._mark_delivery_receipted,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                normalized_purpose,
                chunk_index,
                timestamp,
            )
        if record is None:
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before receipt persistence: "
                f"{batch.batch_id}"
            )
        return record

    async def mark_processed(
        self,
        batch: AutonomyEventBatch,
        *,
        processed_at: datetime | None = None,
    ) -> None:
        """Acknowledge every pointer only after its turn has completed."""

        identifiers = tuple(event.sequence for event in batch.events)
        if not identifiers:
            return
        timestamp = _as_utc(processed_at or datetime.now(UTC)).isoformat()
        async with self._lock:
            updated = await asyncio.to_thread(
                self._mark_processed,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                timestamp,
                timestamp,
            )
        if updated != len(identifiers):
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before ACK: {batch.batch_id}"
            )

    async def reschedule(
        self,
        batch: AutonomyEventBatch,
        *,
        retry_after_seconds: int,
        now: datetime | None = None,
    ) -> None:
        """Retain a failed batch, increment its execution failures, and defer it."""

        await self._reschedule_public(
            batch,
            retry_after_seconds=retry_after_seconds,
            now=now,
            count_failure=True,
        )

    async def defer(
        self,
        batch: AutonomyEventBatch,
        *,
        retry_after_seconds: int,
        now: datetime | None = None,
    ) -> None:
        """Defer admission-limited work without consuming its failure budget."""

        await self._reschedule_public(
            batch,
            retry_after_seconds=retry_after_seconds,
            now=now,
            count_failure=False,
        )

    async def _reschedule_public(
        self,
        batch: AutonomyEventBatch,
        *,
        retry_after_seconds: int,
        now: datetime | None,
        count_failure: bool,
    ) -> None:
        """Set one fixed batch's next eligible time while it is still owned."""

        identifiers = tuple(event.sequence for event in batch.events)
        if not identifiers:
            return
        delay = min(max(retry_after_seconds, 1), 86_400)
        current = _as_utc(now or datetime.now(UTC))
        not_before = (current + timedelta(seconds=delay)).isoformat()
        async with self._lock:
            updated = await asyncio.to_thread(
                self._reschedule,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                current.isoformat(),
                not_before,
                count_failure,
            )
        if updated != len(identifiers):
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before retry: {batch.batch_id}"
            )
        self._wake.set()

    async def dead_letter(
        self,
        batch: AutonomyEventBatch,
        *,
        reason: str,
        failed_at: datetime | None = None,
    ) -> None:
        """Move a poison batch out of the live FIFO with an auditable reason."""

        identifiers = tuple(event.sequence for event in batch.events)
        if not identifiers:
            return
        normalized_reason = reason.strip()[:500] or "unknown"
        timestamp = _as_utc(failed_at or datetime.now(UTC)).isoformat()
        async with self._lock:
            updated = await asyncio.to_thread(
                self._dead_letter,
                batch.batch_id,
                batch.claim_token,
                identifiers,
                timestamp,
                normalized_reason,
                timestamp,
            )
        if updated != len(identifiers):
            raise AutonomyLeaseLostError(
                f"Autonomy batch lease was lost before dead-letter: {batch.batch_id}"
            )

    async def pending_count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._pending_count)

    async def dead_letter_count(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._dead_letter_count)

    def clear_wake(self) -> None:
        self._wake.clear()

    async def wait(self, timeout_seconds: float | None) -> None:
        if timeout_seconds is not None and timeout_seconds <= 0:
            return
        try:
            if timeout_seconds is None:
                await self._wake.wait()
            else:
                async with asyncio.timeout(timeout_seconds):
                    await self._wake.wait()
        except TimeoutError:
            return

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomy_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    deduplication_key TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    enqueued_at TEXT NOT NULL,
                    not_before TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    deferral_count INTEGER NOT NULL DEFAULT 0,
                    workspace_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    actor_id TEXT,
                    message_id TEXT,
                    payload_json TEXT NOT NULL,
                    batch_id TEXT,
                    claim_token TEXT,
                    lease_until TEXT,
                    processed_at TEXT,
                    dead_lettered_at TEXT,
                    dead_letter_reason TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS autonomy_deliveries (
                    batch_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    purpose TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    nonce TEXT NOT NULL,
                    message_id TEXT,
                    receipt_state TEXT NOT NULL DEFAULT 'pending',
                    prepared_at TEXT NOT NULL,
                    sent_at TEXT,
                    receipted_at TEXT,
                    PRIMARY KEY (batch_id, chunk_index, purpose)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(autonomy_events)"
                ).fetchall()
            }
            for name, declaration in (
                ("enqueued_at", "TEXT"),
                ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
                ("failure_count", "INTEGER NOT NULL DEFAULT 0"),
                ("deferral_count", "INTEGER NOT NULL DEFAULT 0"),
                ("batch_id", "TEXT"),
                ("claim_token", "TEXT"),
                ("lease_until", "TEXT"),
                ("dead_lettered_at", "TEXT"),
                ("dead_letter_reason", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE autonomy_events ADD COLUMN {name} {declaration}"
                    )
            connection.execute(
                """
                UPDATE autonomy_events
                SET enqueued_at = occurred_at
                WHERE enqueued_at IS NULL
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS autonomy_events_pending "
                "ON autonomy_events("
                "processed_at, dead_lettered_at, workspace_id, channel_id, sequence"
                ")"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS autonomy_events_batch "
                "ON autonomy_events(batch_id, claim_token, lease_until)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS autonomy_deliveries_message "
                "ON autonomy_deliveries(channel_id, message_id)"
            )
            retention_cutoff = (
                datetime.now(UTC) - _PROCESSED_EVENT_RETENTION
            ).isoformat()
            connection.execute(
                "DELETE FROM autonomy_events "
                "WHERE (processed_at IS NOT NULL AND processed_at < ?) "
                "OR (dead_lettered_at IS NOT NULL AND dead_lettered_at < ?)",
                (retention_cutoff, retention_cutoff),
            )
            connection.execute(
                """
                DELETE FROM autonomy_deliveries
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM autonomy_events
                    WHERE autonomy_events.batch_id =
                        autonomy_deliveries.batch_id
                )
                """
            )
            connection.commit()
        finally:
            connection.close()
        os.chmod(self.path, 0o600)

    def _insert(
        self,
        kind: str,
        deduplication_key: str,
        workspace_id: str,
        channel_id: str,
        actor_id: str | None,
        message_id: str | None,
        occurred_at: str,
        enqueued_at: str,
        payload_json: str,
    ) -> AutonomyEnqueueResult:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            retention_cutoff = (
                datetime.now(UTC) - _PROCESSED_EVENT_RETENTION
            ).isoformat()
            connection.execute(
                "DELETE FROM autonomy_events "
                "WHERE (processed_at IS NOT NULL AND processed_at < ?) "
                "OR (dead_lettered_at IS NOT NULL AND dead_lettered_at < ?)",
                (retention_cutoff, retention_cutoff),
            )
            connection.execute(
                """
                DELETE FROM autonomy_deliveries
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM autonomy_events
                    WHERE autonomy_events.batch_id =
                        autonomy_deliveries.batch_id
                )
                """
            )
            duplicate = connection.execute(
                "SELECT 1 FROM autonomy_events WHERE deduplication_key = ?",
                (deduplication_key,),
            ).fetchone()
            if duplicate is not None:
                connection.rollback()
                return AutonomyEnqueueResult.DUPLICATE
            pending = int(
                connection.execute(
                    "SELECT COUNT(*) FROM autonomy_events "
                    "WHERE processed_at IS NULL AND dead_lettered_at IS NULL"
                ).fetchone()[0]
            )
            if pending >= self.max_pending_events:
                connection.rollback()
                return AutonomyEnqueueResult.QUEUE_FULL
            if actor_id is not None and kind in _ACTOR_LIMITED_EVENT_KINDS:
                actor_pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM autonomy_events "
                        "WHERE processed_at IS NULL AND dead_lettered_at IS NULL "
                        "AND workspace_id = ? AND actor_id = ? "
                        f"AND kind IN ({','.join('?' for _ in _ACTOR_LIMITED_EVENT_KINDS)})",
                        (
                            workspace_id,
                            actor_id,
                            *sorted(_ACTOR_LIMITED_EVENT_KINDS),
                        ),
                    ).fetchone()[0]
                )
                if actor_pending >= self.max_pending_events_per_actor:
                    connection.rollback()
                    return AutonomyEnqueueResult.ACTOR_QUEUE_FULL
            channel_pending = int(
                connection.execute(
                    "SELECT COUNT(*) FROM autonomy_events "
                    "WHERE processed_at IS NULL AND dead_lettered_at IS NULL "
                    "AND workspace_id = ? AND channel_id = ?",
                    (workspace_id, channel_id),
                ).fetchone()[0]
            )
            if channel_pending >= self.max_pending_events_per_channel:
                connection.rollback()
                return AutonomyEnqueueResult.CHANNEL_QUEUE_FULL
            connection.execute(
                """
                INSERT INTO autonomy_events (
                    deduplication_key,
                    kind,
                    occurred_at,
                    enqueued_at,
                    not_before,
                    workspace_id,
                    channel_id,
                    actor_id,
                    message_id,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deduplication_key,
                    kind,
                    occurred_at,
                    enqueued_at,
                    enqueued_at,
                    workspace_id,
                    channel_id,
                    actor_id,
                    message_id,
                    payload_json,
                ),
            )
            connection.commit()
            return AutonomyEnqueueResult.INSERTED
        finally:
            connection.close()

    def _claim_next_batch(
        self,
        debounce_seconds: int,
        candidate_limit: int,
        now_iso: str,
        lease_seconds: int,
    ) -> AutonomyEventBatch | None:
        now = datetime.fromisoformat(now_iso)
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE autonomy_events
                SET claim_token = NULL, lease_until = NULL
                WHERE processed_at IS NULL
                  AND dead_lettered_at IS NULL
                  AND claim_token IS NOT NULL
                  AND (lease_until IS NULL OR lease_until <= ?)
                """,
                (now_iso,),
            )
            fixed = connection.execute(
                """
                SELECT batch_id, MIN(sequence)
                FROM autonomy_events
                WHERE processed_at IS NULL
                  AND dead_lettered_at IS NULL
                  AND batch_id IS NOT NULL
                GROUP BY batch_id
                HAVING
                    SUM(
                        CASE WHEN claim_token IS NOT NULL THEN 1 ELSE 0 END
                    ) = 0
                    AND MAX(not_before) <= ?
                ORDER BY MIN(sequence)
                LIMIT 1
                """,
                (now_iso,),
            ).fetchone()
            batch_id: str | None = str(fixed[0]) if fixed is not None else None
            selected_rows: tuple[tuple[object, ...], ...]
            if batch_id is not None:
                selected_rows = tuple(
                    connection.execute(
                        f"""
                        SELECT {_EVENT_COLUMNS}
                        FROM autonomy_events
                        WHERE processed_at IS NULL
                          AND dead_lettered_at IS NULL
                          AND batch_id = ?
                        ORDER BY sequence
                        """,
                        (batch_id,),
                    ).fetchall()
                )
            else:
                blocked_channels = {
                    (str(row[0]), str(row[1]))
                    for row in connection.execute(
                        """
                        SELECT DISTINCT workspace_id, channel_id
                        FROM autonomy_events
                        WHERE processed_at IS NULL
                          AND dead_lettered_at IS NULL
                          AND batch_id IS NOT NULL
                        """
                    ).fetchall()
                }
                unbatched_rows = tuple(
                    connection.execute(
                        f"""
                        SELECT {_EVENT_COLUMNS}
                        FROM autonomy_events
                        WHERE processed_at IS NULL
                          AND dead_lettered_at IS NULL
                          AND batch_id IS NULL
                          AND claim_token IS NULL
                        ORDER BY sequence
                        LIMIT ?
                        """,
                        (self.max_pending_events,),
                    ).fetchall()
                )
                grouped: dict[
                    tuple[str, str],
                    list[tuple[object, ...]],
                ] = {}
                for row in unbatched_rows:
                    key = (str(row[8]), str(row[9]))
                    if key not in blocked_channels:
                        grouped.setdefault(key, []).append(row)
                eligible: list[
                    tuple[int, tuple[tuple[object, ...], ...]]
                ] = []
                window = timedelta(seconds=debounce_seconds)
                for rows in grouped.values():
                    oldest = _event_from_row(rows[0])
                    deadline = oldest.enqueued_at + window
                    if deadline > now:
                        continue
                    candidates = tuple(
                        row
                        for row in rows
                        if _event_from_row(row).enqueued_at <= deadline
                    )[:candidate_limit]
                    if candidates:
                        eligible.append((oldest.sequence, candidates))
                if not eligible:
                    connection.commit()
                    return None
                _, selected_rows = min(eligible, key=lambda item: item[0])
                selected_events = tuple(_event_from_row(row) for row in selected_rows)
                workspace_id = selected_events[0].workspace_id
                channel_id = selected_events[0].channel_id
                digest = hashlib.sha256(
                    ",".join(
                        str(event.sequence) for event in selected_events
                    ).encode()
                ).hexdigest()[:24]
                batch_id = f"autonomy:{workspace_id}:{channel_id}:{digest}"
            if not selected_rows or batch_id is None:
                connection.commit()
                return None
            identifiers = tuple(int(str(row[0])) for row in selected_rows)
            placeholders = ",".join("?" for _ in identifiers)
            claim_token = uuid.uuid4().hex
            lease_until = now + timedelta(seconds=lease_seconds)
            cursor = connection.execute(
                f"""
                UPDATE autonomy_events
                SET batch_id = ?, claim_token = ?, lease_until = ?
                WHERE sequence IN ({placeholders})
                  AND processed_at IS NULL
                  AND dead_lettered_at IS NULL
                  AND claim_token IS NULL
                """,
                (
                    batch_id,
                    claim_token,
                    lease_until.isoformat(),
                    *identifiers,
                ),
            )
            if cursor.rowcount != len(identifiers):
                connection.rollback()
                return None
            connection.commit()
        finally:
            connection.close()
        events = tuple(_event_from_row(row) for row in selected_rows)
        return AutonomyEventBatch(
            batch_id=batch_id,
            claim_token=claim_token,
            lease_until=lease_until,
            attempt_count=max(event.attempt_count for event in events),
            workspace_id=events[0].workspace_id,
            channel_id=events[0].channel_id,
            events=events,
            deferral_count=max(event.deferral_count for event in events),
        )

    def _next_ready_at(
        self,
        debounce_seconds: int,
        _candidate_limit: int,
        now_iso: str,
    ) -> datetime | None:
        now = datetime.fromisoformat(now_iso)
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute(
                """
                SELECT
                    sequence, enqueued_at, not_before, workspace_id, channel_id,
                    batch_id, claim_token, lease_until
                FROM autonomy_events
                WHERE processed_at IS NULL AND dead_lettered_at IS NULL
                ORDER BY sequence
                LIMIT ?
                """,
                (self.max_pending_events,),
            ).fetchall()
        finally:
            connection.close()
        fixed: dict[str, list[tuple[object, ...]]] = {}
        blocked_channels: set[tuple[str, str]] = set()
        unbatched: dict[tuple[str, str], list[tuple[object, ...]]] = {}
        for row in rows:
            batch_id = row[5]
            key = (str(row[3]), str(row[4]))
            if batch_id is not None:
                fixed.setdefault(str(batch_id), []).append(row)
                blocked_channels.add(key)
            else:
                unbatched.setdefault(key, []).append(row)
        ready_at: list[datetime] = []
        for batch_rows in fixed.values():
            claimed = any(row[6] is not None for row in batch_rows)
            if claimed:
                leases = [
                    datetime.fromisoformat(str(row[7]))
                    for row in batch_rows
                    if row[7] is not None
                ]
                ready_at.append(max(leases) if leases else now)
            else:
                ready_at.append(
                    max(datetime.fromisoformat(str(row[2])) for row in batch_rows)
                )
        window = timedelta(seconds=debounce_seconds)
        for key, channel_rows in unbatched.items():
            if key in blocked_channels:
                continue
            ready_at.append(
                datetime.fromisoformat(str(channel_rows[0][1])) + window
            )
        return min(ready_at) if ready_at else None

    def _renew_lease(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        now_iso: str,
        lease_until: str,
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                now_iso,
            ):
                connection.rollback()
                return 0
            placeholders = ",".join("?" for _ in identifiers)
            cursor = connection.execute(
                f"""
                UPDATE autonomy_events
                SET lease_until = ?
                WHERE sequence IN ({placeholders})
                  AND batch_id = ?
                  AND claim_token = ?
                  AND lease_until > ?
                  AND processed_at IS NULL
                  AND dead_lettered_at IS NULL
                """,
                (
                    lease_until,
                    *identifiers,
                    batch_id,
                    claim_token,
                    now_iso,
                ),
            )
            if cursor.rowcount != len(identifiers):
                connection.rollback()
                return 0
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _prepare_deliveries(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        channel_id: str,
        specs: tuple[AutonomyDeliverySpec, ...],
        timestamp: str,
    ) -> tuple[AutonomyDeliveryRecord, ...] | None:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                timestamp,
            ):
                connection.rollback()
                return None
            for spec in specs:
                connection.execute(
                    """
                    INSERT INTO autonomy_deliveries(
                        batch_id, chunk_index, purpose, channel_id,
                        content_sha256, nonce, prepared_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(batch_id, chunk_index, purpose) DO NOTHING
                    """,
                    (
                        batch_id,
                        spec.chunk_index,
                        spec.purpose,
                        channel_id,
                        spec.content_sha256,
                        spec.nonce,
                        timestamp,
                    ),
                )
            purposes = tuple(sorted({spec.purpose for spec in specs}))
            placeholders = ",".join("?" for _ in purposes)
            rows = connection.execute(
                f"""
                SELECT *
                FROM autonomy_deliveries
                WHERE batch_id = ? AND purpose IN ({placeholders})
                ORDER BY purpose, chunk_index
                """,
                (batch_id, *purposes),
            ).fetchall()
            expected = {
                (spec.chunk_index, spec.purpose): spec
                for spec in specs
            }
            if len(rows) != len(expected):
                connection.rollback()
                raise AutonomyDeliveryConflictError(
                    f"Autonomy delivery plan size changed for {batch_id}"
                )
            records: list[AutonomyDeliveryRecord] = []
            for row in rows:
                record = _delivery_from_row(row)
                expected_spec = expected.get(
                    (record.chunk_index, record.purpose)
                )
                if (
                    expected_spec is None
                    or record.channel_id != channel_id
                    or record.content_sha256 != expected_spec.content_sha256
                    or record.nonce != expected_spec.nonce
                ):
                    connection.rollback()
                    raise AutonomyDeliveryConflictError(
                        f"Autonomy delivery content changed for {batch_id}:"
                        f"{record.purpose}:{record.chunk_index}"
                    )
                records.append(record)
            connection.commit()
            by_key = {
                (record.chunk_index, record.purpose): record
                for record in records
            }
            return tuple(
                by_key[(spec.chunk_index, spec.purpose)]
                for spec in specs
            )
        finally:
            connection.close()

    def _mark_delivery_sent(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        purpose: str,
        chunk_index: int,
        message_id: str,
        timestamp: str,
    ) -> AutonomyDeliveryRecord | None:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                timestamp,
            ):
                connection.rollback()
                return None
            row = connection.execute(
                """
                SELECT *
                FROM autonomy_deliveries
                WHERE batch_id = ? AND chunk_index = ? AND purpose = ?
                """,
                (batch_id, chunk_index, purpose),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise AutonomyDeliveryConflictError(
                    f"Autonomy delivery was not prepared: "
                    f"{batch_id}:{purpose}:{chunk_index}"
                )
            existing = _delivery_from_row(row)
            if (
                existing.message_id is not None
                and existing.message_id != message_id
            ):
                connection.rollback()
                raise AutonomyDeliveryConflictError(
                    f"Autonomy delivery message ID changed for "
                    f"{batch_id}:{purpose}:{chunk_index}"
                )
            connection.execute(
                """
                UPDATE autonomy_deliveries
                SET message_id = COALESCE(message_id, ?),
                    sent_at = COALESCE(sent_at, ?)
                WHERE batch_id = ? AND chunk_index = ? AND purpose = ?
                """,
                (message_id, timestamp, batch_id, chunk_index, purpose),
            )
            updated = connection.execute(
                """
                SELECT *
                FROM autonomy_deliveries
                WHERE batch_id = ? AND chunk_index = ? AND purpose = ?
                """,
                (batch_id, chunk_index, purpose),
            ).fetchone()
            connection.commit()
            if updated is None:
                raise RuntimeError("Autonomy delivery disappeared after send")
            return _delivery_from_row(updated)
        finally:
            connection.close()

    def _mark_delivery_receipted(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        purpose: str,
        chunk_index: int,
        timestamp: str,
    ) -> AutonomyDeliveryRecord | None:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                timestamp,
            ):
                connection.rollback()
                return None
            cursor = connection.execute(
                """
                UPDATE autonomy_deliveries
                SET receipt_state = ?, receipted_at = COALESCE(receipted_at, ?)
                WHERE batch_id = ? AND chunk_index = ? AND purpose = ?
                  AND message_id IS NOT NULL
                """,
                (
                    AutonomyDeliveryReceiptState.RECORDED.value,
                    timestamp,
                    batch_id,
                    chunk_index,
                    purpose,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise AutonomyDeliveryConflictError(
                    f"Autonomy delivery cannot be receipted before send: "
                    f"{batch_id}:{purpose}:{chunk_index}"
                )
            updated = connection.execute(
                """
                SELECT *
                FROM autonomy_deliveries
                WHERE batch_id = ? AND chunk_index = ? AND purpose = ?
                """,
                (batch_id, chunk_index, purpose),
            ).fetchone()
            connection.commit()
            if updated is None:
                raise RuntimeError("Autonomy delivery disappeared after receipt")
            return _delivery_from_row(updated)
        finally:
            connection.close()

    def _mark_processed(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        processed_at: str,
        now_iso: str,
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                now_iso,
            ):
                connection.rollback()
                return 0
            placeholders = ",".join("?" for _ in identifiers)
            cursor = connection.execute(
                f"""
                UPDATE autonomy_events
                SET processed_at = ?, claim_token = NULL, lease_until = NULL
                WHERE sequence IN ({placeholders})
                  AND batch_id = ?
                  AND claim_token = ?
                  AND lease_until > ?
                  AND processed_at IS NULL
                  AND dead_lettered_at IS NULL
                """,
                (
                    processed_at,
                    *identifiers,
                    batch_id,
                    claim_token,
                    now_iso,
                ),
            )
            if cursor.rowcount != len(identifiers):
                connection.rollback()
                return 0
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _reschedule(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        now_iso: str,
        not_before: str,
        count_failure: bool,
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                now_iso,
            ):
                connection.rollback()
                return 0
            placeholders = ",".join("?" for _ in identifiers)
            cursor = connection.execute(
                f"""
                UPDATE autonomy_events
                SET not_before = ?,
                    attempt_count = attempt_count + ?,
                    failure_count = failure_count + ?,
                    deferral_count = deferral_count + ?,
                    claim_token = NULL, lease_until = NULL
                WHERE sequence IN ({placeholders})
                  AND batch_id = ?
                  AND claim_token = ?
                  AND lease_until > ?
                  AND processed_at IS NULL
                  AND dead_lettered_at IS NULL
                """,
                (
                    not_before,
                    int(count_failure),
                    int(count_failure),
                    int(not count_failure),
                    *identifiers,
                    batch_id,
                    claim_token,
                    now_iso,
                ),
            )
            if cursor.rowcount != len(identifiers):
                connection.rollback()
                return 0
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _dead_letter(
        self,
        batch_id: str,
        claim_token: str,
        identifiers: tuple[int, ...],
        failed_at: str,
        reason: str,
        now_iso: str,
    ) -> int:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not _batch_owned(
                connection,
                batch_id,
                claim_token,
                identifiers,
                now_iso,
            ):
                connection.rollback()
                return 0
            placeholders = ",".join("?" for _ in identifiers)
            cursor = connection.execute(
                f"""
                UPDATE autonomy_events
                SET dead_lettered_at = ?, dead_letter_reason = ?,
                    attempt_count = attempt_count + 1,
                    failure_count = failure_count + 1,
                    claim_token = NULL, lease_until = NULL
                WHERE sequence IN ({placeholders})
                  AND batch_id = ?
                  AND claim_token = ?
                  AND lease_until > ?
                  AND processed_at IS NULL
                  AND dead_lettered_at IS NULL
                """,
                (
                    failed_at,
                    reason,
                    *identifiers,
                    batch_id,
                    claim_token,
                    now_iso,
                ),
            )
            if cursor.rowcount != len(identifiers):
                connection.rollback()
                return 0
            connection.commit()
            return cursor.rowcount
        finally:
            connection.close()

    def _pending_count(self) -> int:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM autonomy_events "
                "WHERE processed_at IS NULL AND dead_lettered_at IS NULL"
            ).fetchone()
            return int(row[0])
        finally:
            connection.close()

    def _dead_letter_count(self) -> int:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(
                "SELECT COUNT(*) FROM autonomy_events "
                "WHERE dead_lettered_at IS NOT NULL"
            ).fetchone()
            return int(row[0])
        finally:
            connection.close()


def _batch_owned(
    connection: sqlite3.Connection,
    batch_id: str,
    claim_token: str,
    identifiers: tuple[int, ...],
    now_iso: str,
) -> bool:
    """Require every live row in one fixed batch to share an unexpired claim."""

    if not identifiers:
        return False
    total = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM autonomy_events
            WHERE batch_id = ?
              AND processed_at IS NULL
              AND dead_lettered_at IS NULL
            """,
            (batch_id,),
        ).fetchone()[0]
    )
    if total != len(identifiers):
        return False
    placeholders = ",".join("?" for _ in identifiers)
    owned = int(
        connection.execute(
            f"""
            SELECT COUNT(*)
            FROM autonomy_events
            WHERE sequence IN ({placeholders})
              AND batch_id = ?
              AND claim_token = ?
              AND lease_until > ?
              AND processed_at IS NULL
              AND dead_lettered_at IS NULL
            """,
            (*identifiers, batch_id, claim_token, now_iso),
        ).fetchone()[0]
    )
    return owned == len(identifiers)


def _validated_delivery_purpose(value: str) -> str:
    purpose = value.strip()
    if not purpose or len(purpose) > _DELIVERY_PURPOSE_MAX_CHARACTERS:
        raise ValueError(
            "delivery purpose must contain 1 to "
            f"{_DELIVERY_PURPOSE_MAX_CHARACTERS} characters"
        )
    return purpose


def _validated_delivery_spec(spec: AutonomyDeliverySpec) -> AutonomyDeliverySpec:
    purpose = _validated_delivery_purpose(spec.purpose)
    if spec.chunk_index < 0:
        raise ValueError("delivery chunk_index must be non-negative")
    content_sha256 = spec.content_sha256.strip().lower()
    if (
        len(content_sha256) != _CONTENT_SHA256_CHARACTERS
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValueError("delivery content_sha256 must be a SHA-256 hex digest")
    nonce = spec.nonce.strip()
    if not nonce or len(nonce) > _DELIVERY_NONCE_MAX_CHARACTERS:
        raise ValueError(
            "delivery nonce must contain 1 to "
            f"{_DELIVERY_NONCE_MAX_CHARACTERS} characters"
        )
    return AutonomyDeliverySpec(
        purpose=purpose,
        chunk_index=spec.chunk_index,
        content_sha256=content_sha256,
        nonce=nonce,
    )


def _delivery_from_row(row: sqlite3.Row) -> AutonomyDeliveryRecord:
    return AutonomyDeliveryRecord(
        batch_id=str(row["batch_id"]),
        chunk_index=int(row["chunk_index"]),
        purpose=str(row["purpose"]),
        channel_id=str(row["channel_id"]),
        content_sha256=str(row["content_sha256"]),
        nonce=str(row["nonce"]),
        message_id=(
            str(row["message_id"]) if row["message_id"] is not None else None
        ),
        receipt_state=AutonomyDeliveryReceiptState(str(row["receipt_state"])),
        prepared_at=datetime.fromisoformat(str(row["prepared_at"])),
        sent_at=(
            datetime.fromisoformat(str(row["sent_at"]))
            if row["sent_at"] is not None
            else None
        ),
        receipted_at=(
            datetime.fromisoformat(str(row["receipted_at"]))
            if row["receipted_at"] is not None
            else None
        ),
    )


def _event_from_row(row: tuple[object, ...]) -> AutonomyQueuedEvent:
    payload = json.loads(str(row[12]))
    if not isinstance(payload, dict):
        payload = {}
    return AutonomyQueuedEvent(
        sequence=int(str(row[0])),
        deduplication_key=str(row[1]),
        kind=AutonomyEventKind(str(row[2])),
        occurred_at=datetime.fromisoformat(str(row[3])),
        enqueued_at=datetime.fromisoformat(str(row[4])),
        not_before=datetime.fromisoformat(str(row[5])),
        attempt_count=int(str(row[6])),
        workspace_id=str(row[8]),
        channel_id=str(row[9]),
        actor_id=str(row[10]) if row[10] is not None else None,
        message_id=str(row[11]) if row[11] is not None else None,
        payload={str(key): value for key, value in payload.items()},
        deferral_count=int(str(row[7])),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_batch_options(debounce_seconds: int, candidate_limit: int) -> None:
    if not 5 <= debounce_seconds <= 15:
        raise ValueError("debounce_seconds must be between 5 and 15.")
    if not 1 <= candidate_limit <= 100:
        raise ValueError("candidate_limit must be between 1 and 100.")


def _safe_payload(
    value: dict[str, object],
    *,
    depth: int = 0,
) -> dict[str, object]:
    if depth >= 6:
        return {"truncated": True}
    safe: dict[str, object] = {}
    for raw_key, item in tuple(value.items())[:50]:
        key = str(raw_key)[:100]
        if any(part in key.casefold() for part in _SENSITIVE_PARTS):
            safe[key] = "[redacted]"
        else:
            safe[key] = _safe_payload_value(item, depth=depth + 1)
    return safe


def _safe_payload_value(value: object, *, depth: int) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:2_000]
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, dict):
        return _safe_payload(
            {str(key): item for key, item in value.items()},
            depth=depth,
        )
    if isinstance(value, (list, tuple)):
        if depth >= 6:
            return ["[truncated]"]
        return [
            _safe_payload_value(item, depth=depth + 1)
            for item in value[:50]
        ]
    return str(value)[:2_000]
