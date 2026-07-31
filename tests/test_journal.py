from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.observability import EventJournal


@dataclass(frozen=True)
class SecretRequest:
    token: str
    content: str
    binary: bytes = b""


@dataclass(frozen=True)
class AuditResponse:
    value: str


@pytest.mark.asyncio
async def test_journal_records_cursor_and_redacts_secrets(tmp_path) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    await journal.record_invocation(
        capability_name="test.secret",
        context=InvocationContext("actor", "workspace", "test", "request"),
        request=SecretRequest(
            token="do-not-store",
            content="visible",
            binary=b"never-store-these-bytes",
        ),
        response={"ok": True},
        error=None,
        duration_ms=1.5,
    )
    records = await journal.recent(after_sequence=0, workspace_id="workspace")
    assert len(records) == 1
    assert records[0].payload["capability"] == "test.secret"
    request = records[0].payload["request"]
    assert isinstance(request, dict)
    assert request["token"] == "[REDACTED]"
    assert request["content"] == "visible"
    assert request["binary"] == {"binary_bytes": 23}
    assert "never-store" not in str(records[0].payload)
    assert (tmp_path / "events.sqlite3").stat().st_mode & 0o077 == 0
    await journal.close()


@pytest.mark.asyncio
async def test_journal_prunes_old_events_and_reports_audio_diagnostics(tmp_path) -> None:
    path = tmp_path / "events.sqlite3"
    journal = EventJournal(path)
    await journal.append(
        kind="service.operation",
        payload={"operation": "audio.autoplay_refill", "outcome": "failed"},
    )
    await journal.append(
        kind="service.operation",
        payload={"operation": "audio.overlay_failed", "outcome": "failed"},
    )
    await journal.append(
        kind="service.operation",
        payload={"operation": "discord.dashboard_429", "outcome": "failed"},
    )
    old = datetime.now(UTC) - timedelta(days=31)
    await journal.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE events SET occurred_at = ? WHERE sequence = 1",
            (old.isoformat(),),
        )
        # Simulate migrating an existing event database to the new projection.
        connection.execute("DELETE FROM operation_diagnostics")
    journal = EventJournal(path)

    diagnostics = await journal.operation_diagnostics()
    assert diagnostics.last_radio_failure_at == old
    assert diagnostics.overlay_failure_count == 1
    assert diagnostics.dashboard_429_count == 1

    removed = await journal.prune(before=datetime.now(UTC) - timedelta(days=30))
    assert removed == 1
    retained = await journal.recent()
    assert tuple(record.sequence for record in retained) == (2, 3)
    diagnostics = await journal.operation_diagnostics()
    assert diagnostics.last_radio_failure_at is None
    assert diagnostics.overlay_failure_count == 1
    assert diagnostics.dashboard_429_count == 1
    with sqlite3.connect(path) as connection:
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(events)").fetchall()
        }
        projection = connection.execute(
            """
            SELECT
                overlay_failure_count,
                dashboard_429_count,
                projected_through_sequence
            FROM operation_diagnostics
            WHERE singleton = 1
            """
        ).fetchone()
    assert "events_kind_sequence" in indexes
    assert projection == (1, 1, 3)
    await journal.close()


@pytest.mark.asyncio
async def test_capability_results_do_not_wait_for_slow_audit_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    original_insert_batch = journal._insert_batch
    audit_started = threading.Event()
    release_audit = threading.Event()

    def slow_insert_batch(events: tuple[object, ...]) -> None:
        audit_started.set()
        if not release_audit.wait(timeout=5):
            raise RuntimeError("test audit release timed out")
        original_insert_batch(events)  # type: ignore[arg-type]

    monkeypatch.setattr(journal, "_insert_batch", slow_insert_batch)
    registry = CapabilityRegistry(journal=journal)

    async def read(
        request: SecretRequest,
        _context: InvocationContext,
    ) -> AuditResponse:
        return AuditResponse(request.content)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.audit_independent",
                "Return before the audit disk commit.",
                RiskLevel.READ,
            ),
            SecretRequest,
            AuditResponse,
            read,
        )
    )
    context = InvocationContext("actor", "workspace", "test", "request")
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    registry.invoke(
                        "test.audit_independent",
                        SecretRequest(token="secret", content=str(index)),
                        context,
                    )
                    for index in range(20)
                )
            ),
            timeout=1,
        )
        assert [result.value for result in results] == [
            str(index) for index in range(20)
        ]
        assert await asyncio.to_thread(audit_started.wait, 1)
        assert release_audit.is_set() is False
        diagnostics = await asyncio.wait_for(
            journal.operation_diagnostics(),
            timeout=1,
        )
        committed_cursor = await asyncio.wait_for(
            journal.latest_sequence(),
            timeout=1,
        )
        assert diagnostics.overlay_failure_count == 0
        assert committed_cursor == 0
    finally:
        release_audit.set()

    await journal.close()
    records = await journal.recent(limit=100)
    assert len(records) == 20
    assert {record.kind for record in records} == {"capability.invocation"}


@pytest.mark.asyncio
async def test_audit_writer_retries_transient_failure_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    original_insert_batch = journal._insert_batch
    attempts = 0

    def flaky_insert_batch(events: tuple[object, ...]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        original_insert_batch(events)  # type: ignore[arg-type]

    monkeypatch.setattr(journal, "_insert_batch", flaky_insert_batch)
    await journal.record_invocation(
        capability_name="test.retry",
        context=InvocationContext("actor", "workspace", "test", "request"),
        request={"value": "safe"},
        response={"ok": True},
        error=None,
        duration_ms=1,
    )

    records = await journal.recent()
    health = await journal.audit_health()
    assert attempts == 2
    assert len(records) == 1
    assert records[0].payload["capability"] == "test.retry"
    assert health.retried_event_count == 1
    assert health.outbox_event_count == 0
    assert health.lost_event_count == 0
    assert health.last_failure_type == "OperationalError"
    await journal.close()


@pytest.mark.asyncio
async def test_audit_writer_spools_redacted_event_and_replays_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.sqlite3"
    journal = EventJournal(path)

    def failed_insert_batch(_events: tuple[object, ...]) -> None:
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(journal, "_insert_batch", failed_insert_batch)
    await journal.record_invocation(
        capability_name="test.outbox",
        context=InvocationContext("actor", "workspace", "test", "request"),
        request=SecretRequest(token="never-persist", content="visible"),
        response={"ok": True},
        error=None,
        duration_ms=1,
    )
    await journal.close()

    health = await journal.audit_health()
    assert health.outbox_event_count == 1
    assert health.lost_event_count == 0
    assert journal.outbox_path.stat().st_mode & 0o077 == 0
    with sqlite3.connect(journal.outbox_path) as connection:
        payload_json = str(
            connection.execute(
                "SELECT payload_json FROM audit_outbox"
            ).fetchone()[0]
        )
    payload = json.loads(payload_json)
    assert payload["request"]["token"] == "[REDACTED]"
    assert "never-persist" not in payload_json

    replayed = EventJournal(path)
    records = await replayed.recent()
    replay_health = await replayed.audit_health()
    assert len(records) == 1
    assert records[0].payload["capability"] == "test.outbox"
    assert replay_health.outbox_event_count == 0
    assert replay_health.lost_event_count == 0
    await replayed.close()


@pytest.mark.asyncio
async def test_audit_retry_is_idempotent_after_uncertain_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    original_insert_batch = journal._insert_batch
    attempts = 0

    def committed_then_failed(events: tuple[object, ...]) -> None:
        nonlocal attempts
        attempts += 1
        original_insert_batch(events)  # type: ignore[arg-type]
        if attempts == 1:
            raise sqlite3.OperationalError("connection closed after commit")

    monkeypatch.setattr(journal, "_insert_batch", committed_then_failed)
    await journal.record_invocation(
        capability_name="test.uncertain_commit",
        context=InvocationContext("actor", "workspace", "test", "request"),
        request={},
        response={},
        error=None,
        duration_ms=1,
    )

    records = await journal.recent()
    assert attempts == 2
    assert len(records) == 1
    assert len({record.event_id for record in records}) == 1
    await journal.close()


@pytest.mark.asyncio
async def test_full_audit_queue_spools_without_blocking_capability_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    journal._audit_queue = asyncio.Queue(maxsize=1)
    original_insert_batch = journal._insert_batch
    audit_started = threading.Event()
    release_audit = threading.Event()

    def slow_insert_batch(events: tuple[object, ...]) -> None:
        audit_started.set()
        if not release_audit.wait(timeout=5):
            raise RuntimeError("test audit release timed out")
        original_insert_batch(events)  # type: ignore[arg-type]

    monkeypatch.setattr(journal, "_insert_batch", slow_insert_batch)
    context = InvocationContext("actor", "workspace", "test", "request")
    await journal.record_invocation(
        capability_name="test.first",
        context=context,
        request={},
        response={},
        error=None,
        duration_ms=1,
    )
    assert await asyncio.to_thread(audit_started.wait, 1)
    await journal.record_invocation(
        capability_name="test.queued",
        context=context,
        request={},
        response={},
        error=None,
        duration_ms=1,
    )
    await asyncio.wait_for(
        journal.record_invocation(
            capability_name="test.overflow",
            context=context,
            request={},
            response={},
            error=None,
            duration_ms=1,
        ),
        timeout=0.1,
    )
    try:
        await journal._await_overflow_tasks()
        assert (await journal.audit_health()).outbox_event_count == 1
    finally:
        release_audit.set()
    await journal.close()
    records = await journal.recent()
    assert {record.payload["capability"] for record in records} == {
        "test.first",
        "test.queued",
        "test.overflow",
    }
