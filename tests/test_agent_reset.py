from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from simajilord.agent import (
    AgentRequest,
    AgentResponse,
    AgentResponseStatus,
    AgentTokenUsage,
    AgentTrigger,
    new_agent_public_reference_id,
)
from simajilord.agent.store import AgentConversationStore
from simajilord.diagnostics.agent_reset import _run


def _request(event_id: str, conversation_id: str) -> AgentRequest:
    from datetime import UTC, datetime

    return AgentRequest(
        conversation_id=conversation_id,
        event_id=event_id,
        trigger=AgentTrigger.MENTION,
        actor_id="actor",
        actor_name="Actor",
        workspace_id="guild",
        channel_id="channel",
        message_id="message",
        occurred_at=datetime.now(UTC),
        resource_ids=("channel",),
        public_reference_id=new_agent_public_reference_id(),
    )


async def _complete(
    store: AgentConversationStore,
    request: AgentRequest,
    thread_id: str,
) -> None:
    await store.begin(request, model="model")
    await store.complete(
        request,
        AgentResponse(
            status=AgentResponseStatus.COMPLETED,
            conversation_id=request.conversation_id,
            provider_thread_id=thread_id,
            model="model",
            content="preserved response",
            usage=AgentTokenUsage(input_tokens=10, total_tokens=12),
        ),
    )


@pytest.mark.asyncio
async def test_provider_continuity_reset_preserves_request_and_delivery_rows(
    tmp_path: Path,
) -> None:
    store = AgentConversationStore(tmp_path / "agent.sqlite3")
    first = _request("event-1", "conversation-1")
    second = _request("event-2", "conversation-2")
    await _complete(store, first, "thread-1")
    await _complete(store, second, "thread-2")

    reset = await store.reset_provider_continuity((first.conversation_id,))

    assert reset == 1
    first_conversation = await store.conversation(first.conversation_id)
    second_conversation = await store.conversation(second.conversation_id)
    assert first_conversation is not None
    assert first_conversation.provider_thread_id is None
    assert first_conversation.generation == 1
    assert second_conversation is not None
    assert second_conversation.provider_thread_id == "thread-2"
    assert await store.completed_response(first.event_id) is not None
    assert await store.pending_host_delivery(first.event_id) is not None


@pytest.mark.asyncio
async def test_agent_reset_cli_creates_sqlite_backup_before_reset(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite3"
    backup = tmp_path / "agent.backup.sqlite3"
    store = AgentConversationStore(database)
    request = _request("event-1", "conversation-1")
    await _complete(store, request, "thread-1")

    result = await _run(
        argparse.Namespace(
            database=database,
            all=True,
            conversation=None,
            backup_path=backup,
            yes=True,
        )
    )

    assert result["reset_conversations"] == 1
    assert backup.is_file()
    backup_store = AgentConversationStore(backup)
    backed_up = await backup_store.conversation(request.conversation_id)
    assert backed_up is not None
    assert backed_up.provider_thread_id == "thread-1"
    reset_conversation = await store.conversation(request.conversation_id)
    assert reset_conversation is not None
    assert reset_conversation.provider_thread_id is None
