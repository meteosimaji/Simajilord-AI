"""One-shot, non-CI AI/Discord-adapter QA using the real Codex app-server.

This deliberately avoids the Discord gateway and records typed message sends in
memory, so it consumes one model turn but no Discord API rate limit.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from simajilord.agent import (
    AGENT_MESSAGE_GRANT,
    AGENT_WEB_GRANT,
    AgentProgressUpdate,
    AgentRequest,
    AgentTrigger,
)
from simajilord.agent.providers.codex import CodexAppServerProvider
from simajilord.agent.service import AgentLimits, AgentService
from simajilord.agent.store import AgentConversationStore
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.observability import EventJournal


@dataclass(frozen=True, slots=True)
class GetMessageRequest:
    channel_id: str
    message_id: str
    offset: int = 0
    max_characters: int = 1_000
    include_reply_context: bool = True
    max_reply_depth: int = 2


@dataclass(frozen=True, slots=True)
class GetMessageResponse:
    message_id: str
    channel_id: str
    guild_id: str
    author_id: str
    author_name: str
    content_chunk: str
    content_length: int
    offset: int
    next_offset: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class SendMessageRequest:
    channel_id: str
    content: str


@dataclass(frozen=True, slots=True)
class SendMessageResponse:
    message_id: str
    channel_id: str


async def run() -> dict[str, object]:
    """Run one production-shaped turn and return an assertion-friendly summary."""

    source_message = (
        "Simajilordの手動QAです。Codexの組み込みWeb検索がChatGPT OAuthで"
        "利用できるかを最新のOpenAI公式情報で確認し、検索モードと安全上の注意を"
        "自然な日本語で詳しくまとめてください。途中経過では、確認対象と次の検証、"
        "確認できた内容と残る作業が具体的に分かるようにしてください。"
    )
    sent_messages: list[str] = []
    read_message_ids: list[str] = []
    progress: list[dict[str, object]] = []
    registry = CapabilityRegistry()

    async def get_message(
        request: GetMessageRequest,
        _context: InvocationContext,
    ) -> GetMessageResponse:
        if request.channel_id != "channel-qa" or request.message_id != "message-qa":
            raise RuntimeError("The manual QA requested an unexpected Discord pointer.")
        read_message_ids.append(request.message_id)
        end = min(len(source_message), request.offset + request.max_characters)
        return GetMessageResponse(
            message_id=request.message_id,
            channel_id=request.channel_id,
            guild_id="guild-qa",
            author_id="user-qa",
            author_name="QA User",
            content_chunk=source_message[request.offset:end],
            content_length=len(source_message),
            offset=request.offset,
            next_offset=end if end < len(source_message) else None,
            complete=end == len(source_message),
        )

    async def send_message(
        request: SendMessageRequest,
        _context: InvocationContext,
    ) -> SendMessageResponse:
        sent_messages.append(request.content)
        return SendMessageResponse(
            message_id=f"recorded-{len(sent_messages)}",
            channel_id=request.channel_id,
        )

    registry.register(
        endpoint(
            CapabilityDescriptor(
                name="discord.get_message",
                summary="Read one exact Discord message by channel and message ID.",
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "message", "read", "exact"),
            ),
            GetMessageRequest,
            GetMessageResponse,
            get_message,
        )
    )
    registry.register(
        endpoint(
            CapabilityDescriptor(
                name="discord.send_message",
                summary="Send one user-requested message to the active Discord channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "message", "send", "progress"),
            ),
            SendMessageRequest,
            SendMessageResponse,
            send_message,
        )
    )
    tools = AgentToolCatalog(
        registry,
        ("discord.get_message", "discord.send_message"),
        required_grants={"discord.send_message": AGENT_MESSAGE_GRANT},
        write_capabilities=("discord.send_message",),
    )

    async def on_progress(update: AgentProgressUpdate) -> None:
        progress.append(
            {
                "stage": update.stage.value,
                "queue_position": update.queue_position,
            }
        )

    with tempfile.TemporaryDirectory(prefix="simajilord-agent-qa-") as temporary:
        root = Path(temporary)
        provider = CodexAppServerProvider(
            executable="codex",
            model="gpt-5.6-sol",
            workspace_dir=root / "workspace",
            idle_timeout_seconds=600,
            reasoning_effort="medium",
            tools=tools,
            max_tool_calls=32,
            max_tool_output_characters=24_000,
        )
        service = AgentService(
            provider=provider,
            store=AgentConversationStore(root / "agent.sqlite3"),
            journal=EventJournal(root / "events.sqlite3"),
            limits=AgentLimits(
                per_user_requests=3,
                per_user_window_seconds=600,
                per_workspace_requests=10,
                per_workspace_window_seconds=3_600,
                max_tokens_per_24_hours=150_000,
                max_response_characters=3_800,
                max_active_turns=4,
                max_pending_turns=20,
                max_pending_turns_per_user=2,
            ),
        )
        try:
            response = await service.respond(
                AgentRequest(
                    conversation_id="discord:guild:guild-qa:channel:channel-qa",
                    event_id="discord:message:message-qa",
                    trigger=AgentTrigger.MENTION,
                    actor_id="user-qa",
                    actor_name="QA User",
                    workspace_id="guild-qa",
                    channel_id="channel-qa",
                    message_id="message-qa",
                    occurred_at=datetime.now(UTC),
                    resource_ids=("channel-qa",),
                    grants=frozenset({AGENT_MESSAGE_GRANT, AGENT_WEB_GRANT}),
                    approvals=frozenset({"discord.send_message"}),
                ),
                on_progress=on_progress,
            )
        finally:
            await service.close()

    result = {
        "model": response.model,
        "read_message_ids": read_message_ids,
        "progress": progress,
        "intermediate_messages": sent_messages,
        "final_response": response.content,
        "response_characters": len(response.content),
        "passed": (
            read_message_ids == ["message-qa"]
            and all(len(message) >= 25 for message in sent_messages)
            and {
                "starting",
                "reading_discord",
                "searching_web",
                "preparing_response",
            }.issubset({str(item["stage"]) for item in progress})
            and "https://" in response.content
            and len(response.content) >= 200
        ),
    }
    return result


def main() -> None:
    result = asyncio.run(run())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["passed"] is not True:
        raise SystemExit("Manual agent Discord QA failed.")


if __name__ == "__main__":
    main()
