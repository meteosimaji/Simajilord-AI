"""Restart-safe focus timers shared by every transport adapter."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from simajilord.core.errors import UserError


class FocusTimerStatus(StrEnum):
    SCHEDULED = "scheduled"
    DELIVERING = "delivering"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FocusTimer:
    timer_id: str
    workspace_id: str
    actor_id: str
    delivery_target_id: str
    due_at: datetime
    message: str
    voice_notify: bool
    focus_session: bool
    restore_content_mode: str | None
    status: FocusTimerStatus
    attempts: int = 0


class FocusTimerService:
    """SQLite authority for timers and their at-most-one active delivery lease."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()
        self._requeue_interrupted()

    async def create(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        delivery_target_id: str,
        duration_seconds: int,
        message: str,
        voice_notify: bool = True,
        focus_session: bool = False,
        restore_content_mode: str | None = None,
    ) -> FocusTimer:
        if not 5 <= duration_seconds <= 7 * 24 * 60 * 60:
            raise UserError("timer.duration_invalid")
        normalized = " ".join(message.split())
        if not normalized:
            raise UserError("timer.message_required")
        if len(normalized) > 500:
            raise UserError("timer.message_too_long")
        timer = FocusTimer(
            timer_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            actor_id=actor_id,
            delivery_target_id=delivery_target_id,
            due_at=datetime.now(UTC) + timedelta(seconds=duration_seconds),
            message=normalized,
            voice_notify=voice_notify,
            focus_session=focus_session,
            restore_content_mode=restore_content_mode,
            status=FocusTimerStatus.SCHEDULED,
        )
        async with self._lock:
            await asyncio.to_thread(self._insert, timer)
        return timer

    async def active(
        self,
        *,
        workspace_id: str,
        actor_id: str | None = None,
    ) -> tuple[FocusTimer, ...]:
        async with self._lock:
            return await asyncio.to_thread(self._active, workspace_id, actor_id)

    async def claim_due(self, *, limit: int = 10) -> tuple[FocusTimer, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        async with self._lock:
            return await asyncio.to_thread(self._claim_due, limit)

    async def complete(self, timer_id: str) -> FocusTimer:
        async with self._lock:
            return await asyncio.to_thread(
                self._set_status,
                timer_id,
                FocusTimerStatus.COMPLETED,
            )

    async def retry(self, timer_id: str, *, delay_seconds: int = 30) -> FocusTimer:
        async with self._lock:
            return await asyncio.to_thread(self._retry, timer_id, delay_seconds)

    async def cancel(
        self,
        *,
        timer_id: str,
        workspace_id: str,
        actor_id: str | None = None,
    ) -> FocusTimer:
        async with self._lock:
            timer = await asyncio.to_thread(self._require, timer_id)
            if timer.workspace_id != workspace_id:
                raise UserError("timer.not_found")
            if actor_id is not None and timer.actor_id != actor_id:
                raise UserError("timer.not_owner")
            if timer.status not in {
                FocusTimerStatus.SCHEDULED,
                FocusTimerStatus.DELIVERING,
            }:
                raise UserError("timer.not_active")
            return await asyncio.to_thread(
                self._set_status,
                timer_id,
                FocusTimerStatus.CANCELLED,
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS focus_timers (
                    timer_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    delivery_target_id TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    voice_notify INTEGER NOT NULL,
                    focus_session INTEGER NOT NULL,
                    restore_content_mode TEXT,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS focus_timers_due
                ON focus_timers (status, due_at);
                """
            )
        self.path.chmod(0o600)

    def _requeue_interrupted(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE focus_timers
                SET status = ?, due_at = ?
                WHERE status = ?
                """,
                (
                    FocusTimerStatus.SCHEDULED.value,
                    datetime.now(UTC).isoformat(),
                    FocusTimerStatus.DELIVERING.value,
                ),
            )

    def _insert(self, timer: FocusTimer) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO focus_timers (
                    timer_id, workspace_id, actor_id, delivery_target_id,
                    due_at, message, voice_notify, focus_session,
                    restore_content_mode, status, attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _timer_values(timer),
            )

    def _active(
        self,
        workspace_id: str,
        actor_id: str | None,
    ) -> tuple[FocusTimer, ...]:
        query = (
            "SELECT * FROM focus_timers "
            "WHERE workspace_id = ? AND status IN (?, ?)"
        )
        values: list[str] = [
            workspace_id,
            FocusTimerStatus.SCHEDULED.value,
            FocusTimerStatus.DELIVERING.value,
        ]
        if actor_id is not None:
            query += " AND actor_id = ?"
            values.append(actor_id)
        query += " ORDER BY due_at, timer_id"
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return tuple(_row_to_timer(row) for row in rows)

    def _claim_due(self, limit: int) -> tuple[FocusTimer, ...]:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT * FROM focus_timers
                WHERE status = ? AND due_at <= ?
                ORDER BY due_at, timer_id
                LIMIT ?
                """,
                (FocusTimerStatus.SCHEDULED.value, now, limit),
            ).fetchall()
            ids = tuple(str(row["timer_id"]) for row in rows)
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"""
                    UPDATE focus_timers
                    SET status = ?, attempts = attempts + 1
                    WHERE timer_id IN ({placeholders}) AND status = ?
                    """,
                    (
                        FocusTimerStatus.DELIVERING.value,
                        *ids,
                        FocusTimerStatus.SCHEDULED.value,
                    ),
                )
            connection.commit()
            claimed = tuple(self._require(timer_id) for timer_id in ids)
        return claimed

    def _set_status(
        self,
        timer_id: str,
        status: FocusTimerStatus,
    ) -> FocusTimer:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE focus_timers SET status = ? WHERE timer_id = ?",
                (status.value, timer_id),
            )
            if cursor.rowcount != 1:
                raise UserError("timer.not_found")
        return self._require(timer_id)

    def _retry(self, timer_id: str, delay_seconds: int) -> FocusTimer:
        due_at = datetime.now(UTC) + timedelta(seconds=max(5, delay_seconds))
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE focus_timers
                SET status = ?, due_at = ?
                WHERE timer_id = ? AND status = ?
                """,
                (
                    FocusTimerStatus.SCHEDULED.value,
                    due_at.isoformat(),
                    timer_id,
                    FocusTimerStatus.DELIVERING.value,
                ),
            )
            if cursor.rowcount != 1:
                raise UserError("timer.not_active")
        return self._require(timer_id)

    def _require(self, timer_id: str) -> FocusTimer:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM focus_timers WHERE timer_id = ?",
                (timer_id,),
            ).fetchone()
        if row is None:
            raise UserError("timer.not_found")
        return _row_to_timer(row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def _timer_values(timer: FocusTimer) -> tuple[object, ...]:
    return (
        timer.timer_id,
        timer.workspace_id,
        timer.actor_id,
        timer.delivery_target_id,
        timer.due_at.isoformat(),
        timer.message,
        int(timer.voice_notify),
        int(timer.focus_session),
        timer.restore_content_mode,
        timer.status.value,
        timer.attempts,
    )


def _row_to_timer(row: sqlite3.Row) -> FocusTimer:
    return FocusTimer(
        timer_id=str(row["timer_id"]),
        workspace_id=str(row["workspace_id"]),
        actor_id=str(row["actor_id"]),
        delivery_target_id=str(row["delivery_target_id"]),
        due_at=datetime.fromisoformat(str(row["due_at"])).astimezone(UTC),
        message=str(row["message"]),
        voice_notify=bool(row["voice_notify"]),
        focus_session=bool(row["focus_session"]),
        restore_content_mode=(
            str(row["restore_content_mode"])
            if row["restore_content_mode"] is not None
            else None
        ),
        status=FocusTimerStatus(str(row["status"])),
        attempts=int(row["attempts"]),
    )
