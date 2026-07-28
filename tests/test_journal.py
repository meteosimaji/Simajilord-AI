from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from simajilord.core import InvocationContext
from simajilord.observability import EventJournal


@dataclass(frozen=True)
class SecretRequest:
    token: str
    content: str
    binary: bytes = b""


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
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE events SET occurred_at = ? WHERE sequence = 1",
            (old.isoformat(),),
        )

    diagnostics = await journal.operation_diagnostics()
    assert diagnostics.last_radio_failure_at == old
    assert diagnostics.overlay_failure_count == 1
    assert diagnostics.dashboard_429_count == 1

    removed = await journal.prune(before=datetime.now(UTC) - timedelta(days=30))
    assert removed == 1
    retained = await journal.recent()
    assert tuple(record.sequence for record in retained) == (2, 3)
