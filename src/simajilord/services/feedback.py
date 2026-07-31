"""Restart-safe local feedback inbox with transport-derived attribution."""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from simajilord.agent import is_agent_public_reference_id
from simajilord.core.errors import UserError

_MAX_TITLE_CHARACTERS = 160
_MAX_DETAILS_CHARACTERS = 4_000
_MAX_EXPECTED_CHARACTERS = 2_000
_MAX_ID_CHARACTERS = 256
_REPORT_ID_HEX_CHARACTERS = 24


class FeedbackStatus(StrEnum):
    """Local triage state, never selected by a feedback submitter."""

    NEW = "new"
    TRIAGED = "triaged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DUPLICATE = "duplicate"


class FeedbackKind(StrEnum):
    """Local classification, initially untriaged for every submission."""

    UNTRIAGED = "untriaged"
    BUG = "bug"
    FEATURE = "feature"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class FeedbackReport:
    """One report in the local inbox."""

    report_id: str
    status: FeedbackStatus
    kind: FeedbackKind
    title: str
    details: str
    expected: str | None
    reporter_actor_id: str
    workspace_id: str | None
    source_transport: str
    source_event_id: str
    source_channel_id: str | None
    public_reference_id: str | None
    duplicate_of: str | None
    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


@dataclass(frozen=True, slots=True)
class FeedbackCreateResult:
    """A new report or the existing result of an idempotent retry."""

    report: FeedbackReport
    created: bool


class FeedbackService:
    """SQLite authority for feedback submitted by commands or the agent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def create(
        self,
        *,
        details: str,
        reporter_actor_id: str,
        source_transport: str,
        source_event_id: str,
        title: str | None = None,
        expected: str | None = None,
        workspace_id: str | None = None,
        source_channel_id: str | None = None,
        public_reference_id: str | None = None,
    ) -> FeedbackCreateResult:
        """Persist only caller-supplied feedback and host-supplied provenance."""

        normalized_details = _normalize_multiline(
            details,
            required_code="feedback.details_required",
            too_long_code="feedback.details_too_long",
            maximum=_MAX_DETAILS_CHARACTERS,
        )
        normalized_title = _normalize_title(title, normalized_details)
        normalized_expected = _normalize_optional_multiline(
            expected,
            too_long_code="feedback.expected_too_long",
            maximum=_MAX_EXPECTED_CHARACTERS,
        )
        normalized_actor = _bounded_id(reporter_actor_id)
        normalized_transport = _bounded_id(source_transport)
        normalized_event = _bounded_id(source_event_id)
        normalized_workspace = _optional_bounded_id(workspace_id)
        normalized_channel = _optional_bounded_id(source_channel_id)
        normalized_reference = _public_reference(public_reference_id)
        content_hash = _content_hash(
            title=normalized_title,
            details=normalized_details,
            expected=normalized_expected,
        )
        idempotency_key = _idempotency_key(
            source_transport=normalized_transport,
            source_event_id=normalized_event,
            content_hash=content_hash,
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._insert_or_existing,
                title=normalized_title,
                details=normalized_details,
                expected=normalized_expected,
                reporter_actor_id=normalized_actor,
                workspace_id=normalized_workspace,
                source_transport=normalized_transport,
                source_event_id=normalized_event,
                source_channel_id=normalized_channel,
                public_reference_id=normalized_reference,
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    async def get(self, report_id: str) -> FeedbackReport:
        normalized_id = _bounded_id(report_id)
        async with self._lock:
            report = await asyncio.to_thread(self._get, normalized_id)
        if report is None:
            raise UserError("feedback.not_found")
        return report

    async def list(
        self,
        *,
        status: FeedbackStatus | None = None,
        limit: int = 20,
    ) -> tuple[FeedbackReport, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        async with self._lock:
            return await asyncio.to_thread(self._list, status, limit)

    async def set_status(
        self,
        report_id: str,
        status: FeedbackStatus,
        *,
        duplicate_of: str | None = None,
    ) -> FeedbackReport:
        normalized_id = _bounded_id(report_id)
        normalized_duplicate = _optional_bounded_id(duplicate_of)
        if status is FeedbackStatus.DUPLICATE:
            if normalized_duplicate is None:
                raise UserError("feedback.duplicate_target_required")
            if normalized_duplicate == normalized_id:
                raise UserError("feedback.duplicate_target_invalid")
        elif normalized_duplicate is not None:
            raise UserError("feedback.duplicate_target_unexpected")
        async with self._lock:
            report = await asyncio.to_thread(
                self._set_status,
                normalized_id,
                status,
                normalized_duplicate,
            )
        if report is None:
            raise UserError("feedback.not_found")
        return report

    async def set_kind(
        self,
        report_id: str,
        kind: FeedbackKind,
    ) -> FeedbackReport:
        normalized_id = _bounded_id(report_id)
        async with self._lock:
            report = await asyncio.to_thread(self._set_kind, normalized_id, kind)
        if report is None:
            raise UserError("feedback.not_found")
        return report

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_reports (
                    report_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    details TEXT NOT NULL,
                    expected TEXT,
                    reporter_actor_id TEXT NOT NULL,
                    workspace_id TEXT,
                    source_transport TEXT NOT NULL,
                    source_event_id TEXT NOT NULL,
                    source_channel_id TEXT,
                    public_reference_id TEXT,
                    duplicate_of TEXT,
                    content_hash TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolved_at TEXT,
                    FOREIGN KEY (duplicate_of) REFERENCES feedback_reports(report_id)
                );

                CREATE INDEX IF NOT EXISTS feedback_reports_status_created_idx
                ON feedback_reports(status, created_at DESC);

                CREATE INDEX IF NOT EXISTS feedback_reports_reporter_created_idx
                ON feedback_reports(reporter_actor_id, created_at DESC);

                CREATE INDEX IF NOT EXISTS feedback_reports_public_reference_idx
                ON feedback_reports(public_reference_id)
                WHERE public_reference_id IS NOT NULL;
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(feedback_reports)"
                ).fetchall()
            }
            if "resolved_at" not in columns:
                connection.execute(
                    "ALTER TABLE feedback_reports ADD COLUMN resolved_at TEXT"
                )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _insert_or_existing(
        self,
        *,
        title: str,
        details: str,
        expected: str | None,
        reporter_actor_id: str,
        workspace_id: str | None,
        source_transport: str,
        source_event_id: str,
        source_channel_id: str | None,
        public_reference_id: str | None,
        content_hash: str,
        idempotency_key: str,
    ) -> FeedbackCreateResult:
        now = datetime.now(UTC)
        for _attempt in range(3):
            report_id = f"fdb_{secrets.token_hex(_REPORT_ID_HEX_CHARACTERS // 2)}"
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO feedback_reports (
                        report_id,
                        status,
                        kind,
                        title,
                        details,
                        expected,
                        reporter_actor_id,
                        workspace_id,
                        source_transport,
                        source_event_id,
                        source_channel_id,
                        public_reference_id,
                        duplicate_of,
                        content_hash,
                        idempotency_key,
                        created_at,
                        updated_at,
                        resolved_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, NULL)
                    """,
                    (
                        report_id,
                        FeedbackStatus.NEW.value,
                        FeedbackKind.UNTRIAGED.value,
                        title,
                        details,
                        expected,
                        reporter_actor_id,
                        workspace_id,
                        source_transport,
                        source_event_id,
                        source_channel_id,
                        public_reference_id,
                        content_hash,
                        idempotency_key,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
                row = connection.execute(
                    """
                    SELECT *
                    FROM feedback_reports
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    report = _report_from_row(row)
                    return FeedbackCreateResult(
                        report=report,
                        created=cursor.rowcount == 1 and report.report_id == report_id,
                    )
        raise RuntimeError("Could not allocate a unique feedback report ID")

    def _get(self, report_id: str) -> FeedbackReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return _report_from_row(row) if row is not None else None

    def _list(
        self,
        status: FeedbackStatus | None,
        limit: int,
    ) -> tuple[FeedbackReport, ...]:
        with self._connect() as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM feedback_reports
                    ORDER BY created_at DESC, report_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM feedback_reports
                    WHERE status = ?
                    ORDER BY created_at DESC, report_id DESC
                    LIMIT ?
                    """,
                    (status.value, limit),
                ).fetchall()
        return tuple(_report_from_row(row) for row in rows)

    def _set_status(
        self,
        report_id: str,
        status: FeedbackStatus,
        duplicate_of: str | None,
    ) -> FeedbackReport | None:
        with self._connect() as connection:
            if duplicate_of is not None:
                duplicate_exists = connection.execute(
                    "SELECT 1 FROM feedback_reports WHERE report_id = ?",
                    (duplicate_of,),
                ).fetchone()
                if duplicate_exists is None:
                    raise UserError("feedback.duplicate_target_not_found")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE feedback_reports
                SET status = ?,
                    duplicate_of = ?,
                    updated_at = ?,
                    resolved_at = CASE
                        WHEN ? = ? THEN ?
                        ELSE NULL
                    END
                WHERE report_id = ?
                """,
                (
                    status.value,
                    duplicate_of,
                    now,
                    status.value,
                    FeedbackStatus.RESOLVED.value,
                    now,
                    report_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM feedback_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return _report_from_row(row) if row is not None else None

    def _set_kind(
        self,
        report_id: str,
        kind: FeedbackKind,
    ) -> FeedbackReport | None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE feedback_reports
                SET kind = ?, updated_at = ?
                WHERE report_id = ?
                """,
                (kind.value, datetime.now(UTC).isoformat(), report_id),
            )
            row = connection.execute(
                "SELECT * FROM feedback_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return _report_from_row(row) if row is not None else None


def _report_from_row(row: sqlite3.Row) -> FeedbackReport:
    return FeedbackReport(
        report_id=str(row["report_id"]),
        status=FeedbackStatus(str(row["status"])),
        kind=FeedbackKind(str(row["kind"])),
        title=str(row["title"]),
        details=str(row["details"]),
        expected=str(row["expected"]) if row["expected"] is not None else None,
        reporter_actor_id=str(row["reporter_actor_id"]),
        workspace_id=(
            str(row["workspace_id"]) if row["workspace_id"] is not None else None
        ),
        source_transport=str(row["source_transport"]),
        source_event_id=str(row["source_event_id"]),
        source_channel_id=(
            str(row["source_channel_id"])
            if row["source_channel_id"] is not None
            else None
        ),
        public_reference_id=(
            str(row["public_reference_id"])
            if row["public_reference_id"] is not None
            else None
        ),
        duplicate_of=(
            str(row["duplicate_of"]) if row["duplicate_of"] is not None else None
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        resolved_at=(
            datetime.fromisoformat(str(row["resolved_at"]))
            if row["resolved_at"] is not None
            else None
        ),
    )


def _normalize_title(value: str | None, details: str) -> str:
    if value is None or not value.strip():
        first_line = next(
            (line.strip() for line in details.splitlines() if line.strip()),
            details,
        )
        normalized = " ".join(first_line.split())
        return normalized[:_MAX_TITLE_CHARACTERS]
    normalized = " ".join(value.split())
    if len(normalized) > _MAX_TITLE_CHARACTERS:
        raise UserError("feedback.title_too_long")
    return normalized


def _normalize_multiline(
    value: str,
    *,
    required_code: str,
    too_long_code: str,
    maximum: int,
) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise UserError(required_code)
    if len(normalized) > maximum:
        raise UserError(too_long_code)
    return normalized


def _normalize_optional_multiline(
    value: str | None,
    *,
    too_long_code: str,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return None
    if len(normalized) > maximum:
        raise UserError(too_long_code)
    return normalized


def _bounded_id(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_ID_CHARACTERS:
        raise UserError("feedback.source_invalid")
    return normalized


def _optional_bounded_id(value: str | None) -> str | None:
    return _bounded_id(value) if value is not None else None


def _public_reference(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not is_agent_public_reference_id(normalized):
        raise UserError("feedback.reference_invalid")
    return normalized


def _content_hash(*, title: str, details: str, expected: str | None) -> str:
    canonical = "\x1f".join((title, details, expected or ""))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _idempotency_key(
    *,
    source_transport: str,
    source_event_id: str,
    content_hash: str,
) -> str:
    canonical = "\x1f".join((source_transport, source_event_id, content_hash))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
