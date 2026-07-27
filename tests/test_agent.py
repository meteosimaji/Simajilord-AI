from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from simajilord.agent import (
    AGENT_IMAGE_GRANT,
    AGENT_WEB_GRANT,
    AgentBusyError,
    AgentProgressStage,
    AgentRateLimitError,
    AgentRequest,
    AgentTokenUsage,
    AgentToolError,
    AgentTrigger,
)
from simajilord.agent.providers import ProviderTurnResult
from simajilord.agent.providers.codex import (
    CodexAppServerProvider,
    _base_instructions,
    _last_write_failure,
    _tool_read_exact_event,
    _ToolTurnBudget,
)
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
class ReadRequest:
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ReadResponse:
    content: str
    next_offset: int | None


@dataclass(frozen=True, slots=True)
class WriteRequest:
    subject: str


@dataclass(frozen=True, slots=True)
class WriteResponse:
    job_id: str


@dataclass(frozen=True, slots=True)
class LiteralRequest:
    mode: Literal["preview", "animation", "frame"] = "preview"


@dataclass(frozen=True, slots=True)
class LiteralResponse:
    mode: str


class FakeProvider:
    model = "test-luna"

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str, InvocationContext]] = []
        self.closed = False

    async def respond(
        self,
        *,
        provider_thread_id: str | None,
        event_prompt: str,
        context: InvocationContext,
        on_progress: object = None,
    ) -> ProviderTurnResult:
        del on_progress
        self.calls.append((provider_thread_id, event_prompt, context))
        return ProviderTurnResult(
            thread_id=provider_thread_id or "thread-1",
            model=self.model,
            content="Hello from the agent.",
            usage=AgentTokenUsage(
                input_tokens=100,
                cached_input_tokens=50,
                output_tokens=10,
                total_tokens=110,
                model_context_window=1_000,
            ),
        )

    async def close(self) -> None:
        self.closed = True


def _request(
    event_id: str = "event-1",
    *,
    actor_id: str = "3",
    conversation_id: str = "discord:guild:1:channel:2",
) -> AgentRequest:
    return AgentRequest(
        conversation_id=conversation_id,
        event_id=event_id,
        trigger=AgentTrigger.MENTION,
        actor_id=actor_id,
        actor_name="person",
        workspace_id="1",
        channel_id="2",
        message_id="4",
        occurred_at=datetime.now(UTC),
        resource_ids=("2",),
    )


def _limits(**overrides: object) -> AgentLimits:
    values: dict[str, object] = {
        "per_user_requests": 3,
        "per_user_window_seconds": 600,
        "per_workspace_requests": 10,
        "per_workspace_window_seconds": 3_600,
        "max_tokens_per_24_hours": 100_000,
        "max_conversation_turns": 24,
        "max_context_ratio": 0.5,
        "max_response_characters": 3_800,
        "max_pending_turns": 20,
        "rate_limit_exempt_actor_ids": frozenset(),
    }
    values.update(overrides)
    return AgentLimits(**values)  # type: ignore[arg-type]


def test_provider_accepts_only_complete_exact_event_from_message_index() -> None:
    output = (
        '{"messages":[{"message_id":"4","content_preview":"full request",'
        '"preview_truncated":false}]}'
    )
    assert _tool_read_exact_event(
        tool_name="discord_read_messages",
        arguments={"channel_id": "2"},
        output=output,
        required_message_id="4",
    )
    assert not _tool_read_exact_event(
        tool_name="discord_read_messages",
        arguments={"channel_id": "2"},
        output=output.replace('"preview_truncated":false', '"preview_truncated":true'),
        required_message_id="4",
    )


def test_later_failed_write_is_not_hidden_by_an_earlier_success() -> None:
    budget = _ToolTurnBudget(
        context=InvocationContext(
            actor_id="3",
            workspace_id="1",
            transport="agent",
            request_id="event",
        ),
        calls_remaining=3,
        output_characters_remaining=1_000,
        on_progress=None,
        required_message_id=None,
        write_successes={"discord.send_message"},
        write_failures=[("discord.play_audio", "audio.same_voice_required")],
    )
    assert _last_write_failure(budget) == (
        "discord.play_audio",
        "audio.same_voice_required",
    )
    assert not _tool_read_exact_event(
        tool_name="discord_read_messages",
        arguments={"channel_id": "2"},
        output='{"truncated":true,"preview":"partial"}',
        required_message_id="4",
    )


@pytest.mark.asyncio
async def test_agent_event_uses_pointer_only_and_reuses_thread(tmp_path) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    first = await service.respond(_request("event-1"))
    second = await service.respond(_request("event-2"))

    assert first.provider_thread_id == "thread-1"
    assert second.provider_thread_id == "thread-1"
    assert provider.calls[0][0] is None
    assert provider.calls[1][0] == "thread-1"
    assert "message_id=4" in provider.calls[0][1]
    assert "No message body is included" in provider.calls[0][1]


@pytest.mark.asyncio
async def test_agent_request_is_idempotent_without_second_model_turn(tmp_path) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    first = await service.respond(_request())
    second = await service.respond(_request())

    assert second == first
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_expanded_capability_profile_safely_inherits_prior_thread(
    tmp_path,
) -> None:
    provider = FakeProvider()
    store = AgentConversationStore(tmp_path / "agent.sqlite3")
    service = AgentService(
        provider=provider,
        store=store,
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    base = "discord:guild:1:channel:2"
    old_id = f"{base}:profile:discord_message+web"
    expanded_id = f"{base}:profile:audio+discord_message+web"

    await service.respond(_request("old", conversation_id=old_id))
    await service.respond(_request("expanded", conversation_id=expanded_id))

    assert provider.calls[0][0] is None
    assert provider.calls[1][0] == "thread-1"
    old = await store.conversation(old_id)
    expanded = await store.conversation(expanded_id)
    assert old is not None and old.provider_thread_id is None
    assert expanded is not None and expanded.provider_thread_id == "thread-1"
    assert expanded.turn_count == 2


@pytest.mark.asyncio
async def test_reduced_capability_profile_never_inherits_privileged_thread(
    tmp_path,
) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    base = "discord:guild:1:channel:2"
    expanded_id = f"{base}:profile:audio+discord_message+web"
    reduced_id = f"{base}:profile:discord_message+web"

    await service.respond(_request("expanded", conversation_id=expanded_id))
    await service.respond(_request("reduced", conversation_id=reduced_id))

    assert provider.calls[0][0] is None
    assert provider.calls[1][0] is None


@pytest.mark.asyncio
async def test_agent_emits_only_structured_progress_stages(tmp_path) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    stages: list[AgentProgressStage] = []

    async def record(stage: AgentProgressStage) -> None:
        stages.append(stage)

    await service.respond(_request(), on_progress=record)
    assert stages == [AgentProgressStage.STARTING]


@pytest.mark.asyncio
async def test_agent_queues_parallel_server_turns_instead_of_dropping_them(
    tmp_path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def respond(
            self,
            *,
            provider_thread_id: str | None,
            event_prompt: str,
            context: InvocationContext,
            on_progress: object = None,
        ) -> ProviderTurnResult:
            entered.set()
            await release.wait()
            return await super().respond(
                provider_thread_id=provider_thread_id,
                event_prompt=event_prompt,
                context=context,
                on_progress=on_progress,
            )

    provider = BlockingProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(),
    )
    first = asyncio.create_task(service.respond(_request("one")))
    await entered.wait()
    queued_stages: list[AgentProgressStage] = []
    queued_notified = asyncio.Event()

    async def record_queued(stage: AgentProgressStage) -> None:
        queued_stages.append(stage)
        if stage is AgentProgressStage.QUEUED:
            queued_notified.set()

    second = asyncio.create_task(
        service.respond(
            _request(
                "two",
                actor_id="4",
                conversation_id="discord:guild:9:channel:8",
            ),
            on_progress=record_queued,
        )
    )
    await asyncio.wait_for(queued_notified.wait(), timeout=1)
    assert not second.done()
    assert queued_stages == [AgentProgressStage.QUEUED]

    release.set()
    await asyncio.gather(first, second)
    assert len(provider.calls) == 2
    assert queued_stages == [
        AgentProgressStage.QUEUED,
        AgentProgressStage.STARTING,
    ]


@pytest.mark.asyncio
async def test_agent_queue_has_a_hard_admission_bound(tmp_path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def respond(
            self,
            *,
            provider_thread_id: str | None,
            event_prompt: str,
            context: InvocationContext,
            on_progress: object = None,
        ) -> ProviderTurnResult:
            entered.set()
            await release.wait()
            return await super().respond(
                provider_thread_id=provider_thread_id,
                event_prompt=event_prompt,
                context=context,
                on_progress=on_progress,
            )

    provider = BlockingProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(max_pending_turns=1),
    )
    first = asyncio.create_task(service.respond(_request("one")))
    await entered.wait()
    with pytest.raises(AgentBusyError, match="queue is full"):
        await service.respond(
            _request(
                "two",
                actor_id="4",
                conversation_id="discord:guild:9:channel:8",
            )
        )
    release.set()
    await first


@pytest.mark.asyncio
async def test_agent_local_rate_limit_blocks_before_provider(tmp_path) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(per_user_requests=1),
    )
    await service.respond(_request("event-1"))
    with pytest.raises(AgentRateLimitError) as raised:
        await service.respond(_request("event-2"))
    assert raised.value.retry_after_seconds is not None
    assert 1 <= raised.value.retry_after_seconds <= 600
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_fixed_exempt_actor_bypasses_every_local_agent_budget(tmp_path) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(
            per_user_requests=1,
            per_workspace_requests=1,
            max_tokens_per_24_hours=1,
            rate_limit_exempt_actor_ids=frozenset({"admin"}),
        ),
    )
    await service.respond(_request("event-1", actor_id="admin"))
    await service.respond(_request("event-2", actor_id="admin"))
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_exempt_actor_does_not_consume_other_users_workspace_budget(
    tmp_path,
) -> None:
    provider = FakeProvider()
    service = AgentService(
        provider=provider,
        store=AgentConversationStore(tmp_path / "agent.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(
            per_workspace_requests=1,
            rate_limit_exempt_actor_ids=frozenset({"admin"}),
        ),
    )
    await service.respond(_request("admin-event", actor_id="admin"))
    await service.respond(_request("user-event", actor_id="regular"))
    with pytest.raises(AgentRateLimitError):
        await service.respond(_request("user-event-2", actor_id="regular"))
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_agent_rotates_provider_thread_at_context_budget(tmp_path) -> None:
    provider = FakeProvider()
    store = AgentConversationStore(tmp_path / "agent.sqlite3")
    service = AgentService(
        provider=provider,
        store=store,
        journal=EventJournal(tmp_path / "events.sqlite3"),
        limits=_limits(max_context_ratio=0.1),
    )
    await service.respond(_request("event-1"))
    await service.respond(_request("event-2"))

    assert provider.calls[0][0] is None
    assert provider.calls[1][0] is None
    conversation = await store.conversation("discord:guild:1:channel:2")
    assert conversation is not None
    assert conversation.generation == 1


@pytest.mark.asyncio
async def test_dynamic_tool_catalog_builds_typed_schema_and_invokes(tmp_path) -> None:
    del tmp_path
    registry = CapabilityRegistry()

    async def read(
        request: ReadRequest,
        _: InvocationContext,
    ) -> ReadResponse:
        return ReadResponse(content="abcdef"[request.offset : request.offset + 2], next_offset=2)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.read",
                "Read a bounded test chunk.",
                RiskLevel.READ,
            ),
            ReadRequest,
            ReadResponse,
            read,
        )
    )
    catalog = AgentToolCatalog(registry, ("test.read",))
    specs = catalog.dynamic_specs()
    namespace = specs[0]
    tools = namespace["tools"]
    assert isinstance(tools, list)
    schema = tools[0]["inputSchema"]
    assert schema["properties"]["offset"]["type"] == "integer"
    assert "required" not in schema

    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="test_read",
        arguments={"offset": 1},
        context=InvocationContext("actor", "workspace", "agent", "request"),
        max_output_characters=1_000,
    )
    assert '"content":"bc"' in output


@pytest.mark.asyncio
async def test_dynamic_tool_catalog_validates_literal_choices() -> None:
    registry = CapabilityRegistry()

    async def select(
        request: LiteralRequest,
        _: InvocationContext,
    ) -> LiteralResponse:
        return LiteralResponse(mode=request.mode)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.select",
                "Select one test mode.",
                RiskLevel.READ,
            ),
            LiteralRequest,
            LiteralResponse,
            select,
        )
    )
    catalog = AgentToolCatalog(registry, ("test.select",))
    tools = catalog.dynamic_specs()[0]["tools"]
    assert isinstance(tools, list)
    schema = tools[0]["inputSchema"]
    assert schema["properties"]["mode"] == {
        "type": "string",
        "enum": ["preview", "animation", "frame"],
    }
    context = InvocationContext("actor", "workspace", "agent", "request")
    result = await catalog.invoke(
        namespace="simajilord",
        tool_name="test_select",
        arguments={"mode": "frame"},
        context=context,
        max_output_characters=1_000,
    )
    assert '"mode":"frame"' in result
    with pytest.raises(AgentToolError, match="arguments are invalid"):
        await catalog.invoke(
            namespace="simajilord",
            tool_name="test_select",
            arguments={"mode": "unknown"},
            context=context,
            max_output_characters=1_000,
        )


@pytest.mark.asyncio
async def test_external_agent_tool_requires_runtime_grant() -> None:
    registry = CapabilityRegistry()

    async def read(
        request: ReadRequest,
        _: InvocationContext,
    ) -> ReadResponse:
        return ReadResponse(content=str(request.offset), next_offset=None)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "web.test",
                "Read one bounded public test value.",
                RiskLevel.EXTERNAL,
            ),
            ReadRequest,
            ReadResponse,
            read,
        )
    )
    catalog = AgentToolCatalog(
        registry,
        ("web.test",),
        required_grants={"web.test": AGENT_WEB_GRANT},
    )
    denied_context = InvocationContext("actor", "workspace", "agent", "denied")
    granted_context = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "granted",
        grants=frozenset({AGENT_WEB_GRANT}),
    )

    assert catalog.dynamic_specs(denied_context) == ()
    assert catalog.dynamic_specs(granted_context)
    with pytest.raises(AgentToolError, match="grant"):
        await catalog.invoke(
            namespace="simajilord",
            tool_name="web_test",
            arguments={"offset": 1},
            context=denied_context,
            max_output_characters=1_000,
        )
    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="web_test",
        arguments={"offset": 1},
        context=granted_context,
        max_output_characters=1_000,
    )
    assert '"content":"1"' in output


@pytest.mark.asyncio
async def test_progressive_catalog_hides_schema_until_search_and_granted_invoke() -> None:
    registry = CapabilityRegistry()

    async def read(
        request: ReadRequest,
        _: InvocationContext,
    ) -> ReadResponse:
        return ReadResponse(content=str(request.offset), next_offset=None)

    async def write(
        request: WriteRequest,
        _: InvocationContext,
    ) -> WriteResponse:
        return WriteResponse(job_id=f"image:{request.subject}")

    registry.register(
        endpoint(
            CapabilityDescriptor("test.read", "Read a message.", RiskLevel.READ),
            ReadRequest,
            ReadResponse,
            read,
        )
    )
    registry.register(
        endpoint(
            CapabilityDescriptor(
                "image.generate",
                "Generate a local image.",
                RiskLevel.WRITE,
                keywords=("image", "picture"),
            ),
            WriteRequest,
            WriteResponse,
            write,
        )
    )
    catalog = AgentToolCatalog(
        registry,
        ("test.read", "image.generate"),
        required_grants={"image.generate": AGENT_IMAGE_GRANT},
        eager_capabilities=("test.read",),
        write_capabilities=("image.generate",),
    )
    assert (
        catalog.write_capability_for_call(
            tool_name="capability_invoke",
            arguments={"name": "image.generate", "arguments": {"subject": "cat"}},
        )
        == "image.generate"
    )
    assert catalog.capability_for_call(
        tool_name="capability_invoke",
        arguments={"name": "image.generate", "arguments": {"subject": "cat"}},
    ) == "image.generate"
    assert catalog.capability_for_call(
        tool_name="test_read",
        arguments={"offset": 0},
    ) == "test.read"
    granted = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "request",
        grants=frozenset({AGENT_IMAGE_GRANT}),
    )
    tools = catalog.dynamic_specs(granted)[0]["tools"]
    assert isinstance(tools, list)
    assert [tool["name"] for tool in tools] == [
        "test_read",
        "capability_search",
        "capability_invoke",
    ]
    search = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "generate image"},
        context=granted,
        max_output_characters=2_000,
    )
    assert '"name":"image.generate"' in search
    result = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_invoke",
        arguments={"name": "image.generate", "arguments": {"subject": "cat"}},
        context=granted,
        max_output_characters=2_000,
    )
    assert '"job_id":"image:cat"' in result


@pytest.mark.asyncio
async def test_when_requested_tool_requires_capability_specific_turn_approval() -> None:
    registry = CapabilityRegistry()

    async def write(
        request: WriteRequest,
        _: InvocationContext,
    ) -> WriteResponse:
        return WriteResponse(job_id=f"write:{request.subject}")

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.write",
                "Perform a requested test write.",
                RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
            ),
            WriteRequest,
            WriteResponse,
            write,
        )
    )
    catalog = AgentToolCatalog(
        registry,
        ("test.write",),
        required_grants={"test.write": "write-scope"},
        write_capabilities=("test.write",),
    )
    scope_only = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "scope-only",
        grants=frozenset({"write-scope"}),
    )
    approved = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "approved",
        grants=frozenset({"write-scope"}),
        approvals=frozenset({"test.write"}),
    )

    assert catalog.dynamic_specs(scope_only) == ()
    assert catalog.dynamic_specs(approved)
    with pytest.raises(AgentToolError, match="grant"):
        await catalog.invoke(
            namespace="simajilord",
            tool_name="test_write",
            arguments={"subject": "blocked"},
            context=scope_only,
            max_output_characters=1_000,
        )
    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="test_write",
        arguments={"subject": "allowed"},
        context=approved,
        max_output_characters=1_000,
    )
    assert '"job_id":"write:allowed"' in output


@pytest.mark.asyncio
async def test_provider_rejects_write_before_exact_event_is_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invoked: list[str] = []
    registry = CapabilityRegistry()

    async def write(
        request: WriteRequest,
        _: InvocationContext,
    ) -> WriteResponse:
        invoked.append(request.subject)
        return WriteResponse(job_id="done")

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.write",
                "Perform one requested write.",
                RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
            ),
            WriteRequest,
            WriteResponse,
            write,
        )
    )
    catalog = AgentToolCatalog(
        registry,
        ("test.write",),
        required_grants={"test.write": "write-scope"},
        write_capabilities=("test.write",),
    )
    provider = CodexAppServerProvider(
        executable="codex",
        model="test",
        workspace_dir=tmp_path / "agent",
        timeout_seconds=10,
        reasoning_effort="low",
        tools=catalog,
        max_tool_calls=4,
        max_tool_output_characters=4_000,
    )
    context = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "event",
        grants=frozenset({"write-scope"}),
        approvals=frozenset({"test.write"}),
    )
    provider._active_tool_budget = _ToolTurnBudget(
        context=context,
        calls_remaining=4,
        output_characters_remaining=4_000,
        on_progress=None,
        required_message_id="123",
    )
    response = AsyncMock()
    monkeypatch.setattr(provider, "_tool_response", response)
    request = {
        "namespace": "simajilord",
        "tool": "test_write",
        "arguments": {"subject": "requested"},
    }

    await provider._handle_dynamic_tool(1, request)

    assert invoked == []
    first_response = response.await_args
    assert first_response is not None
    assert first_response.kwargs["success"] is False
    assert provider._active_tool_budget.write_failures == [
        ("test.write", "agent.event_message_not_read")
    ]
    provider._active_tool_budget.event_message_read = True
    await provider._handle_dynamic_tool(2, request)
    assert invoked == ["requested"]
    second_response = response.await_args
    assert second_response is not None
    assert second_response.kwargs["success"] is True


def test_base_instructions_are_short_and_use_runtime_identity() -> None:
    instructions = _base_instructions("gpt-5.6-luna")
    assert len(instructions) < 2_000
    assert "Simajilord AI" in instructions
    assert "gpt-5.6-luna" in instructions
    assert "generic Codex/OpenAI Assistant" in instructions
    assert "capability_search" in instructions
    assert "natural, concise Japanese by default" in instructions
