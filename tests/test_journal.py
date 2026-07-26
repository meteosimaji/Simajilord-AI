from __future__ import annotations

from dataclasses import dataclass

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
