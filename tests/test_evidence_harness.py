from __future__ import annotations

import json
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

import pytest

from simajilord.agent.providers import codex
from simajilord.agent.providers.codex import (
    _evidence_plan_gap,
    _tool_read_anchored_conversation_context,
    _ToolTurnBudget,
    _write_readiness_failure,
)
from simajilord.agent.service import _conversation_rotation_reason
from simajilord.agent.store import AgentConversationRecord
from simajilord.capabilities.source_inspection import (
    EvidencePlanRequest,
    EvidencePlanResponse,
    SourceReadRequest,
    SourceReadResponse,
    SourceSearchRequest,
    SourceSearchResponse,
    build_source_inspection_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.capabilities import DiscordGetMessageResponse
from simajilord.services.source_inspection import SourceInspectionService


def _budget() -> _ToolTurnBudget:
    return _ToolTurnBudget(
        context=InvocationContext(
            actor_id="actor",
            workspace_id="guild",
            transport="agent",
            request_id="discord:message:200",
            origin_resource_id="channel",
        ),
        calls_remaining=8,
        output_characters_remaining=8_000,
        on_progress=None,
        required_message_id="200",
        evidence_anchor_message_id="200",
        event_message_read=True,
        read_authorization_event_ids={"auth"},
        last_write_authorization_event_id="auth",
    )


def test_evidence_plan_is_semantic_and_not_a_host_keyword_audit() -> None:
    assert not hasattr(codex, "_source_inspection_requested")
    assert not hasattr(codex, "_text_requests_source_inspection")

    budget = _budget()
    assert _evidence_plan_gap(budget) == (
        "agent.evidence_plan_required",
        (
            "Record turn.evidence_plan after semantically assessing the exact "
            "active request. The host does not infer this decision from keywords."
        ),
    )

    budget.evidence_plan_recorded = True
    budget.conversation_context_required = True
    assert _evidence_plan_gap(budget)[0] == "agent.conversation_context_required"
    budget.conversation_context_satisfied = True
    budget.source_inspection_required = True
    assert _evidence_plan_gap(budget)[0] == "agent.source_inspection_required"
    budget.source_inspection_satisfied = True
    assert _evidence_plan_gap(budget) is None


def test_write_cannot_bypass_the_ai_authored_evidence_plan() -> None:
    budget = _budget()

    assert _write_readiness_failure(budget)[0] == "agent.evidence_plan_required"

    budget.evidence_plan_recorded = True
    assert _write_readiness_failure(budget) is None


def test_context_is_retrieved_only_when_ai_requests_it_and_is_anchored() -> None:
    assert "recent_context" not in {
        item.name for item in fields(DiscordGetMessageResponse)
    }
    budget = _budget()
    output = json.dumps(
        {
            "messages": [],
            "source_channel_id": "channel",
            "truncated": False,
        }
    )

    assert _tool_read_anchored_conversation_context(
        capability_name="discord.read_messages",
        arguments={
            "channel_id": "channel",
            "before_message_id": "200",
            "limit": 5,
        },
        output=output,
        budget=budget,
    )
    assert not _tool_read_anchored_conversation_context(
        capability_name="discord.read_messages",
        arguments={
            "channel_id": "channel",
            "before_message_id": "199",
            "limit": 5,
        },
        output=output,
        budget=budget,
    )


def test_provider_thread_rotation_bounds_stale_self_history() -> None:
    now = datetime.now(UTC)
    base = AgentConversationRecord(
        conversation_id="conversation",
        provider_thread_id="thread",
        model="model",
        generation=1,
        turn_count=11,
        last_input_tokens=34_999,
        model_context_window=100_000,
        created_at=now,
        updated_at=now,
    )

    assert _conversation_rotation_reason(base) is None
    assert (
        _conversation_rotation_reason(
            AgentConversationRecord(
                conversation_id=base.conversation_id,
                provider_thread_id=base.provider_thread_id,
                model=base.model,
                generation=base.generation,
                turn_count=12,
                last_input_tokens=base.last_input_tokens,
                model_context_window=base.model_context_window,
                created_at=base.created_at,
                updated_at=base.updated_at,
            )
        )
        == "bounded_thread_turn_limit"
    )
    assert (
        _conversation_rotation_reason(
            AgentConversationRecord(
                conversation_id=base.conversation_id,
                provider_thread_id=base.provider_thread_id,
                model=base.model,
                generation=base.generation,
                turn_count=1,
                last_input_tokens=35_000,
                model_context_window=100_000,
                created_at=base.created_at,
                updated_at=base.updated_at,
            )
        )
        == "bounded_thread_context_pressure"
    )


@pytest.mark.asyncio
async def test_ai_evidence_plan_supports_context_and_source_independently(
    tmp_path: Path,
) -> None:
    endpoints = {
        item.descriptor.name: item
        for item in build_source_inspection_endpoints(
            SourceInspectionService(tmp_path)
        )
    }
    response = await endpoints["turn.evidence_plan"].invoke(
        EvidencePlanRequest(
            conversation_context="required",
            source_inspection="not_required",
            reason="The current message refers implicitly to the preceding discussion.",
        ),
        InvocationContext("actor", "guild", "agent", "event"),
    )

    assert response == EvidencePlanResponse(
        conversation_context="required",
        source_inspection="not_required",
        recorded=True,
    )
    with pytest.raises(UserError, match=r"agent\.evidence_plan_reason_invalid"):
        await endpoints["turn.evidence_plan"].invoke(
            EvidencePlanRequest(
                conversation_context="not_required",
                source_inspection="not_required",
                reason=" ",
            ),
            InvocationContext("actor", "guild", "agent", "event"),
        )


@pytest.mark.asyncio
async def test_source_inspection_is_bounded_and_excludes_runtime_data(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "src" / "simajilord"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "feature.py"
    source_file.write_text(
        "def current_implementation() -> str:\n"
        "    return 'verified source evidence'\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_feature.py").write_text(
        "def test_current_implementation():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    swift_dir = (
        tmp_path
        / "native"
        / "macos"
        / "TranslationHelper"
        / "Sources"
        / "TranslationHelper"
    )
    swift_dir.mkdir(parents=True)
    (swift_dir / "main.swift").write_text(
        "let nativeEvidence = \"verified native source evidence\"\n",
        encoding="utf-8",
    )
    activity_dir = tmp_path / "activity" / "src"
    activity_dir.mkdir(parents=True)
    (activity_dir / "main.js").write_text(
        "const activityEvidence = 'verified activity source evidence';\n",
        encoding="utf-8",
    )
    generated_dir = tmp_path / "src" / "simajilord" / "activity" / "static"
    generated_dir.mkdir(parents=True)
    (generated_dir / "bundle.js").write_text(
        "const generatedEvidence = 'never search generated static';\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("SECRET=do-not-read\n", encoding="utf-8")
    data_dir = tmp_path / ".data"
    data_dir.mkdir()
    (data_dir / "private.py").write_text("SECRET = 'hidden'\n", encoding="utf-8")
    service = SourceInspectionService(tmp_path)
    endpoints = {
        item.descriptor.name: item
        for item in build_source_inspection_endpoints(service)
    }

    search = await endpoints["source.search"].invoke(
        SourceSearchRequest(query="verified source"),
        InvocationContext("actor", "guild", "agent", "event"),
    )
    assert isinstance(search, SourceSearchResponse)
    assert tuple(item.path for item in search.matches) == (
        "src/simajilord/feature.py",
    )
    native_search = await service.search("verified native source evidence")
    activity_search = await service.search("verified activity source evidence")
    generated_search = await service.search("never search generated static")
    assert tuple(item.path for item in native_search.matches) == (
        "native/macos/TranslationHelper/Sources/TranslationHelper/main.swift",
    )
    assert tuple(item.path for item in activity_search.matches) == (
        "activity/src/main.js",
    )
    assert generated_search.matches == ()

    read = await endpoints["source.read"].invoke(
        SourceReadRequest(
            path="src/simajilord/feature.py",
            start_line=1,
            max_lines=20,
        ),
        InvocationContext("actor", "guild", "agent", "event"),
    )
    assert isinstance(read, SourceReadResponse)
    assert "verified source evidence" in read.content
    assert len(read.sha256) == 64

    with pytest.raises(UserError, match=r"source\.path_forbidden"):
        await service.read(".env")
    with pytest.raises(UserError, match=r"source\.path_forbidden"):
        await service.read(".data/private.py")
    with pytest.raises(UserError, match=r"source\.path_forbidden"):
        await service.read("../outside.py")
