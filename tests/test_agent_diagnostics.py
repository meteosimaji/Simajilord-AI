from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from simajilord.agent import (
    AgentRequest,
    AgentTrigger,
)
from simajilord.agent.store import AgentConversationStore
from simajilord.core import InvocationContext
from simajilord.diagnostics.agent_reference import inspect_agent_reference
from simajilord.observability import EventJournal


@pytest.mark.asyncio
async def test_agent_reference_diagnostic_correlates_body_free_trace(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "agent.sqlite3"
    events_path = tmp_path / "events.sqlite3"
    reference_id = "agt_00000000000000000006"
    request = AgentRequest(
        conversation_id="conversation",
        event_id="event",
        trigger=AgentTrigger.MENTION,
        actor_id="actor",
        actor_name="name",
        workspace_id="workspace",
        channel_id="channel",
        message_id="message",
        occurred_at=datetime.now(UTC),
        resource_ids=("channel",),
        public_reference_id=reference_id,
    )
    store = AgentConversationStore(requests_path)
    await store.begin(request, model="model")
    journal = EventJournal(events_path)
    await journal.append(
        kind="agent.tool.finished",
        actor_id=request.actor_id,
        workspace_id=request.workspace_id,
        transport="agent",
        request_id=request.event_id,
        payload={
            "public_reference_id": reference_id,
            "tool_call_id": "call",
            "requested_tool": "test_read",
            "resolved_capability": "test.read",
            "outcome": "succeeded",
        },
    )
    await journal.record_invocation(
        capability_name="test.read",
        context=InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            public_reference_id=reference_id,
            tool_call_id="call",
        ),
        request={"content": "body-must-not-be-returned"},
        response={"content": "response-must-not-be-returned"},
        error=None,
        duration_ms=1,
    )
    await journal.close()

    result = await inspect_agent_reference(
        reference_id=reference_id,
        requests_database=requests_path,
        events_database=events_path,
    )

    assert result is not None
    assert result["event_count"] == 2
    assert "body-must-not-be-returned" not in str(result)
    assert "response-must-not-be-returned" not in str(result)
