from __future__ import annotations

import json
import sqlite3
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta

import pytest

from simajilord.agent import (
    AGENT_MEMORY_CURATOR_GRANT,
    AGENT_MEMORY_GRANT,
    AGENT_MEMORY_WRITE_CAPABILITIES,
    NON_UNDOABLE_ACTION_CAPABILITIES,
    ActionClassification,
    ActionReceiptStore,
    AgentMemoryBasis,
    AgentMemoryForgetRequest,
    AgentMemoryRememberRequest,
    AgentMemoryReviewDecision,
    AgentMemoryReviewRequest,
    AgentMemoryReviewState,
    AgentMemoryScope,
    AgentMemorySearchRequest,
    AgentMemoryService,
    AgentMemorySourceLocator,
    AgentMemoryStore,
    AgentMemoryUpdateRequest,
    AgentMemoryVisibility,
    action_policy,
    build_memory_endpoints,
)
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import (
    ApprovalMode,
    CapabilityRegistry,
    DisclosureObservation,
    InvocationContext,
    RiskLevel,
)
from simajilord.core.errors import UserError

_CONFIRMED_ACTION_ID = "act_" + "a" * 32
_MEMORY_ACTION_ID = "act_" + "c" * 32


class _ConfirmedActionEvidence:
    async def is_confirmed_memory_evidence(
        self,
        *,
        action_id: str,
        context: InvocationContext,
        allow_any_actor: bool,
    ) -> bool:
        del context, allow_any_actor
        return action_id == _CONFIRMED_ACTION_ID


def _service(path) -> AgentMemoryService:
    return AgentMemoryService(
        AgentMemoryStore(path),
        _ConfirmedActionEvidence(),
    )


def _context(
    *,
    actor_id: str = "100",
    workspace_id: str = "200",
    channel_id: str = "300",
    approvals: frozenset[str] = frozenset(),
    resource_ids: tuple[str, ...] | None = None,
    curator: bool = False,
) -> InvocationContext:
    grants = {AGENT_MEMORY_GRANT}
    if curator:
        grants.add(AGENT_MEMORY_CURATOR_GRANT)
    return InvocationContext(
        actor_id=actor_id,
        workspace_id=workspace_id,
        transport="agent",
        request_id="discord:message:400",
        resource_ids=resource_ids or (channel_id,),
        grants=frozenset(grants),
        origin_resource_id=channel_id,
        approvals=approvals,
    )


async def _remember(
    service: AgentMemoryService,
    *,
    context: InvocationContext,
    scope: AgentMemoryScope,
    key: str,
    summary: str,
    source_id: str,
    basis: AgentMemoryBasis = AgentMemoryBasis.USER_STATED,
    confidence: float = 1.0,
    ttl_days: int | None = None,
):
    return await service.remember(
        AgentMemoryRememberRequest(
            scope=scope,
            key=key,
            summary=summary,
            source_message_ids=(source_id,),
            basis=basis,
            confidence=confidence,
            ttl_days=ttl_days,
            verified_action_id=(
                _CONFIRMED_ACTION_ID
                if scope is AgentMemoryScope.PROCEDURE
                else None
            ),
        ),
        context,
    )


@pytest.mark.asyncio
async def test_memory_upserts_normalized_key_and_survives_store_restart(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    service = _service(path)
    context = _context()

    first = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key=" Response.Language ",
        summary="Respond in Japanese.",
        source_id="111",
    )
    second = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="response.language",
        summary="日本語で回答する。",
        source_id="222",
    )

    assert first.created is True
    assert second.created is False
    assert second.memory.memory_id == first.memory.memory_id
    assert second.memory.created_at == first.memory.created_at
    assert second.memory.updated_at >= first.memory.updated_at
    assert second.memory.source_message_ids == ("222",)
    assert second.memory.source_message_locators == (
        AgentMemorySourceLocator(
            message_id="222",
            channel_id="300",
            guild_id="200",
        ),
    )

    restarted = _service(path)
    found = await restarted.search(
        AgentMemorySearchRequest(
            query="日本語",
            scopes=(AgentMemoryScope.USER,),
        ),
        context,
    )
    assert [item.memory_id for item in found.memories] == [
        first.memory.memory_id
    ]
    assert found.memories[0].summary == "日本語で回答する。"
    assert found.memories[0].last_used_at >= second.memory.last_used_at


@pytest.mark.asyncio
async def test_memory_migrates_and_records_verified_failure_procedure(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    context = _context()
    service = _service(path)
    existing = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="response.language",
        summary="Respond in Japanese.",
        source_id="111",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            ALTER TABLE agent_memories RENAME TO agent_memories_before_basis_migration;
            CREATE TABLE agent_memories AS
                SELECT * FROM agent_memories_before_basis_migration;
            DROP TABLE agent_memories_before_basis_migration;
            """
        )

    restarted = _service(path)
    failure = await _remember(
        restarted,
        context=context,
        scope=AgentMemoryScope.PROCEDURE,
        key="procedure.audio.callback_timeout",
        summary=(
            "Dropping the queued item after a playback callback timeout lost "
            "recovery state; preserve it and retry with a fresh stream."
        ),
        source_id="112",
        basis=AgentMemoryBasis.VERIFIED_FAILURE,
    )
    found = await restarted.search(
        AgentMemorySearchRequest(
            query="",
            scopes=(AgentMemoryScope.USER, AgentMemoryScope.PROCEDURE),
        ),
        context,
    )

    assert {item.memory_id for item in found.memories} == {
        existing.memory.memory_id,
        failure.memory.memory_id,
    }
    assert failure.memory.basis is AgentMemoryBasis.VERIFIED_FAILURE
    with sqlite3.connect(path) as connection:
        table_sql = str(
            connection.execute(
                """
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'agent_memories'
                """
            ).fetchone()[0]
        )
    assert "verified_failure" in table_sql


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query, expected_key",
    (
        ("短い日本語", "response.style"),
        ("brief Japanese replies", "response.style"),
        ("response style", "response.style"),
        ("ＰＤＦレビュー手順", "procedure.pdf_review"),
        ("ファイルの読み方", "procedure.pdf_review"),
        ("前に成功したやり方", "procedure.pdf_review"),
        ("私の好みを思い出す", "response.style"),
        ("メモリー", "memory.behavior"),
        ("このAIの管理者は誰", "assistant.creator"),
    ),
)
async def test_memory_search_handles_japanese_english_and_spelling_variants(
    tmp_path,
    query: str,
    expected_key: str,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    context = _context()
    await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="response.style",
        summary="ユーザーは簡潔な日本語の返答を好む。",
        source_id="111",
    )
    await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.PROCEDURE,
        key="procedure.pdf_review",
        summary="添付ファイルを分割して読み、出力ハッシュを検証してから送る。",
        source_id="112",
        basis=AgentMemoryBasis.VERIFIED_SUCCESS,
    )
    await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="memory.behavior",
        summary="必要な時だけメモリを検索する。",
        source_id="113",
    )
    await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="assistant.creator",
        summary="The requester created this assistant.",
        source_id="114",
    )

    result = await service.search(
        AgentMemorySearchRequest(query=query, limit=1),
        context,
    )

    assert [memory.key for memory in result.memories] == [expected_key]


@pytest.mark.asyncio
async def test_turn_context_is_bounded_requester_private_and_not_counted_as_used(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    service = _service(path)
    owner = _context(actor_id="100", workspace_id="200", channel_id="300")
    other_user = _context(actor_id="101", workspace_id="200", channel_id="300")
    owner_memory = await _remember(
        service,
        context=owner,
        scope=AgentMemoryScope.USER,
        key="assistant.creator",
        summary="The requester created this assistant.",
        source_id="111",
    )
    await _remember(
        service,
        context=owner,
        scope=AgentMemoryScope.WORKSPACE,
        key="workspace.public",
        summary="Visible to the workspace.",
        source_id="112",
    )
    before = owner_memory.memory.last_used_at

    owner_context = await service.context_for_turn(owner, limit=1)
    other_context = await service.context_for_turn(other_user, limit=1)
    after = (
        await service.search(
            AgentMemorySearchRequest(
                query="creator",
                scopes=(AgentMemoryScope.USER,),
                limit=1,
            ),
            owner,
        )
    ).memories[0]

    assert tuple(item.key for item in owner_context) == ("assistant.creator",)
    assert other_context == ()
    assert owner_context[0].scope is AgentMemoryScope.USER
    assert owner_context[0].last_used_at == before
    assert after.last_used_at >= before


@pytest.mark.asyncio
async def test_memory_scope_is_strict_across_users_channels_and_workspaces(
    tmp_path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    original = _context(
        actor_id="100",
        workspace_id="200",
        channel_id="300",
        resource_ids=("300", "301"),
        curator=True,
    )
    other_channel = _context(
        actor_id="100",
        workspace_id="200",
        channel_id="301",
        resource_ids=("300", "301"),
    )
    other_user = _context(
        actor_id="101",
        workspace_id="200",
        channel_id="300",
    )
    other_workspace = _context(
        actor_id="100",
        workspace_id="201",
        channel_id="300",
    )
    no_source_access = _context(
        actor_id="102",
        workspace_id="200",
        channel_id="301",
        resource_ids=("301",),
    )

    await _remember(
        service,
        context=original,
        scope=AgentMemoryScope.USER,
        key="user.language",
        summary="Respond in Japanese.",
        source_id="111",
    )
    await _remember(
        service,
        context=original,
        scope=AgentMemoryScope.CHANNEL,
        key="channel.style",
        summary="Use concise status updates in this channel.",
        source_id="112",
    )
    await _remember(
        service,
        context=original,
        scope=AgentMemoryScope.WORKSPACE,
        key="workspace.term",
        summary="Call the project Simajilord.",
        source_id="113",
    )
    await _remember(
        service,
        context=original,
        scope=AgentMemoryScope.PROCEDURE,
        key="procedure.pdf_review",
        summary="Import, read in chunks, verify the output hash, then send.",
        source_id="114",
        basis=AgentMemoryBasis.VERIFIED_SUCCESS,
        confidence=0.95,
    )

    async def visible(context: InvocationContext) -> set[AgentMemoryScope]:
        result = await service.search(
            AgentMemorySearchRequest(query="", limit=10),
            context,
        )
        return {item.scope for item in result.memories}

    assert await visible(original) == set(AgentMemoryScope)
    assert await visible(other_channel) == set(AgentMemoryScope)
    assert await visible(other_user) == {
        AgentMemoryScope.CHANNEL,
        AgentMemoryScope.WORKSPACE,
        AgentMemoryScope.PROCEDURE,
    }
    assert await visible(no_source_access) == set()
    assert await visible(other_workspace) == set()


@pytest.mark.asyncio
async def test_shared_memory_requires_creator_or_curator_and_explicit_review(
    tmp_path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    owner = _context(actor_id="100", resource_ids=("300",))
    intruder = _context(actor_id="101", resource_ids=("300",))
    curator = _context(
        actor_id="900",
        resource_ids=("300",),
        curator=True,
    )
    created = await _remember(
        service,
        context=owner,
        scope=AgentMemoryScope.WORKSPACE,
        key="workspace.release_rule",
        summary="Use the release checklist before deployment.",
        source_id="111",
    )
    assert created.memory.created_by_actor_id == "100"
    assert created.memory.review_state is AgentMemoryReviewState.PENDING
    assert (
        await service.search(
            AgentMemorySearchRequest(
                query="release checklist",
                scopes=(AgentMemoryScope.WORKSPACE,),
            ),
            intruder,
        )
    ).memories == ()
    assert (
        await service.search(
            AgentMemorySearchRequest(
                query="release checklist",
                scopes=(AgentMemoryScope.WORKSPACE,),
            ),
            owner,
        )
    ).memories[0].memory_id == created.memory.memory_id

    with pytest.raises(UserError) as denied_update:
        await service.update(
            AgentMemoryUpdateRequest(
                memory_id=created.memory.memory_id,
                summary="Skip the release checklist.",
                source_message_ids=("222",),
                confidence=1.0,
            ),
            intruder,
        )
    assert denied_update.value.code == "memory.not_found"
    with pytest.raises(UserError) as denied_delete:
        await service.forget(
            AgentMemoryForgetRequest(created.memory.memory_id),
            intruder,
        )
    assert denied_delete.value.code == "memory.not_found"

    no_source_curator = _context(
        actor_id="900",
        channel_id="301",
        resource_ids=("301",),
        curator=True,
    )
    with pytest.raises(UserError) as denied_review:
        await service.review(
            AgentMemoryReviewRequest(
                created.memory.memory_id,
                AgentMemoryReviewDecision.APPROVE,
            ),
            no_source_curator,
        )
    assert denied_review.value.code == "memory.not_found"
    with pytest.raises(UserError) as denied_curator_update:
        await service.update(
            AgentMemoryUpdateRequest(
                memory_id=created.memory.memory_id,
                summary="Replace a hidden source through curator authority.",
                source_message_ids=("222",),
                confidence=1.0,
            ),
            no_source_curator,
        )
    assert denied_curator_update.value.code == "memory.not_found"
    with pytest.raises(UserError) as denied_curator_upsert:
        await _remember(
            service,
            context=no_source_curator,
            scope=AgentMemoryScope.WORKSPACE,
            key="workspace.release_rule",
            summary="Replace a hidden record through a matching shared key.",
            source_id="223",
        )
    assert denied_curator_upsert.value.code == "memory.not_found"
    with pytest.raises(UserError) as denied_curator_delete:
        await service.forget(
            AgentMemoryForgetRequest(created.memory.memory_id),
            no_source_curator,
        )
    assert denied_curator_delete.value.code == "memory.not_found"

    reviewed = await service.review(
        AgentMemoryReviewRequest(
            created.memory.memory_id,
            AgentMemoryReviewDecision.APPROVE,
        ),
        curator,
    )
    assert reviewed.memory.review_state is AgentMemoryReviewState.APPROVED
    assert reviewed.memory.reviewed_by_actor_id == "900"
    assert (
        await service.search(
            AgentMemorySearchRequest(
                query="release checklist",
                scopes=(AgentMemoryScope.WORKSPACE,),
            ),
            intruder,
        )
    ).memories[0].memory_id == created.memory.memory_id

    updated = await service.update(
        AgentMemoryUpdateRequest(
            memory_id=created.memory.memory_id,
            summary="Use the current release checklist before deployment.",
            source_message_ids=("333",),
            confidence=1.0,
        ),
        owner,
    )
    assert updated.memory.review_state is AgentMemoryReviewState.PENDING
    assert updated.memory.reviewed_by_actor_id is None
    forgotten = await service.forget(
        AgentMemoryForgetRequest(created.memory.memory_id),
        owner,
    )
    assert forgotten.forgotten is True


@pytest.mark.asyncio
async def test_memory_persists_source_audience_and_revision(tmp_path) -> None:
    context = replace(
        _context(curator=True),
        active_message_id="111",
        active_message_edited_at="2026-08-03T01:02:03+00:00",
        disclosure_observations=(
            DisclosureObservation(
                source_workspace_id="200",
                source_resource_id="300",
                visibility="restricted",
                relation_to_origin="same_or_narrower",
            ),
        ),
    )
    service = _service(tmp_path / "memory.sqlite3")
    created = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.WORKSPACE,
        key="workspace.restricted",
        summary="Use the restricted release channel as the source of truth.",
        source_id="111",
    )

    assert created.memory.source_message_locators[0].message_edited_at == (
        "2026-08-03T01:02:03+00:00"
    )
    assert created.memory.provenance.origin_visibility is (
        AgentMemoryVisibility.RESTRICTED
    )
    assert created.memory.provenance.source_resources == (
        ("200", "300", AgentMemoryVisibility.RESTRICTED),
    )
    assert created.memory.provenance.unlabelled_input is False


@pytest.mark.asyncio
async def test_procedure_memory_requires_confirmed_non_memory_action(
    tmp_path,
) -> None:
    memory_path = tmp_path / "memory.sqlite3"
    without_ledger = AgentMemoryService(AgentMemoryStore(memory_path))
    request = AgentMemoryRememberRequest(
        scope=AgentMemoryScope.PROCEDURE,
        key="procedure.publish",
        summary="Send the verified result, then retain its Action Receipt.",
        source_message_ids=("111",),
        basis=AgentMemoryBasis.VERIFIED_SUCCESS,
        confidence=1.0,
    )
    with pytest.raises(UserError) as missing:
        await without_ledger.remember(request, _context())
    assert missing.value.code == "memory.verified_action_required"

    action_store = ActionReceiptStore(tmp_path / "actions.sqlite3")
    action_context = replace(
        _context(),
        request_id="discord:message:action-source",
        provider_thread_id="thread",
        provider_turn_id="turn",
        tool_call_id="tool-call",
    )
    await action_store.add(
        action_id=_CONFIRMED_ACTION_ID,
        capability="discord.send_message",
        context=action_context,
        target_ids=(("channel_id", "300"),),
        classification=ActionClassification.NON_UNDOABLE,
        undo_capability=None,
        undo_arguments=None,
    )
    effect = await action_store.plan_external_effect(
        capability="discord.send_message",
        context=action_context,
        request={"channel_id": "300"},
        authorization_reference=None,
    )
    await action_store.dispatch_external_effect(effect.effect_id)
    await action_store.confirm_external_effect(
        effect.effect_id,
        action_id=_CONFIRMED_ACTION_ID,
    )
    service = AgentMemoryService(AgentMemoryStore(memory_path), action_store)
    remembered = await service.remember(
        replace(request, verified_action_id=_CONFIRMED_ACTION_ID),
        _context(),
    )
    assert remembered.memory.verified_action_id == _CONFIRMED_ACTION_ID

    with pytest.raises(UserError) as wrong_actor:
        await service.remember(
            replace(
                request,
                key="procedure.other_actor",
                verified_action_id=_CONFIRMED_ACTION_ID,
            ),
            _context(actor_id="101"),
        )
    assert wrong_actor.value.code == "memory.verified_action_invalid"

    memory_action_context = replace(
        action_context,
        request_id="discord:message:memory-action",
        tool_call_id="memory-tool-call",
    )
    await action_store.add(
        action_id=_MEMORY_ACTION_ID,
        capability="memory.remember",
        context=memory_action_context,
        target_ids=(),
        classification=ActionClassification.NON_UNDOABLE,
        undo_capability=None,
        undo_arguments=None,
    )
    memory_effect = await action_store.plan_external_effect(
        capability="memory.remember",
        context=memory_action_context,
        request={"key": "procedure.self_referential"},
        authorization_reference=None,
    )
    await action_store.dispatch_external_effect(memory_effect.effect_id)
    await action_store.confirm_external_effect(
        memory_effect.effect_id,
        action_id=_MEMORY_ACTION_ID,
    )
    with pytest.raises(UserError) as self_referential:
        await service.remember(
            replace(
                request,
                key="procedure.self_referential",
                verified_action_id=_MEMORY_ACTION_ID,
            ),
            _context(),
        )
    assert self_referential.value.code == "memory.verified_action_invalid"


@pytest.mark.asyncio
async def test_memory_search_filters_basis_confidence_time_and_paginates(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    service = _service(path)
    context = _context()
    for index, confidence in enumerate((0.81, 0.9, 1.0), start=1):
        await _remember(
            service,
            context=context,
            scope=AgentMemoryScope.USER,
            key=f"preference.{index}",
            summary=f"Use preference {index}.",
            source_id=str(100 + index),
            confidence=confidence,
        )
    procedure = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.PROCEDURE,
        key="procedure.pdf",
        summary="Read chunks, verify the output hash, then send.",
        source_id="200",
        basis=AgentMemoryBasis.VERIFIED_SUCCESS,
        confidence=0.95,
    )
    with sqlite3.connect(path) as connection:
        last_used_before = {
            str(memory_id): str(last_used_at)
            for memory_id, last_used_at in connection.execute(
                """
                SELECT memory_id, last_used_at FROM agent_memories
                WHERE memory_key IN ('preference.2', 'preference.3')
                """
            )
        }

    first_page = await service.search(
        AgentMemorySearchRequest(
            query="",
            basis=AgentMemoryBasis.USER_STATED,
            min_confidence=0.85,
            limit=1,
        ),
        context,
    )
    with sqlite3.connect(path) as connection:
        last_used_after_first_page = {
            str(memory_id): str(last_used_at)
            for memory_id, last_used_at in connection.execute(
                """
                SELECT memory_id, last_used_at FROM agent_memories
                WHERE memory_key IN ('preference.2', 'preference.3')
                """
            )
        }
    second_page = await service.search(
        AgentMemorySearchRequest(
            query="",
            basis=AgentMemoryBasis.USER_STATED,
            min_confidence=0.85,
            offset=first_page.next_offset or 0,
            limit=1,
        ),
        context,
    )
    procedures = await service.search(
        AgentMemorySearchRequest(
            query="hash",
            scopes=(AgentMemoryScope.PROCEDURE,),
            basis=AgentMemoryBasis.VERIFIED_SUCCESS,
            min_confidence=0.9,
        ),
        context,
    )
    future = await service.search(
        AgentMemorySearchRequest(
            query="",
            updated_after=(datetime.now(UTC) + timedelta(days=1)).isoformat(),
        ),
        context,
    )

    assert first_page.next_offset == 1
    first_memory_id = first_page.memories[0].memory_id
    lookahead_id = next(
        memory_id
        for memory_id in last_used_before
        if memory_id != first_memory_id
    )
    assert (
        last_used_after_first_page[first_memory_id]
        > last_used_before[first_memory_id]
    )
    assert (
        last_used_after_first_page[lookahead_id]
        == last_used_before[lookahead_id]
    )
    assert second_page.next_offset is None
    assert {
        first_page.memories[0].key,
        second_page.memories[0].key,
    } == {"preference.2", "preference.3"}
    assert tuple(item.memory_id for item in procedures.memories) == (
        procedure.memory.memory_id,
    )
    assert procedures.memories[0].memory_id
    assert procedures.memories[0].source_message_ids == ("200",)
    assert procedures.memories[0].confidence == 0.95
    assert procedures.memories[0].updated_at.tzinfo is not None
    assert future.memories == ()
    assert future.next_offset is None


@pytest.mark.asyncio
async def test_memory_update_and_forget_enforce_original_scope(
    tmp_path,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    owner = _context(actor_id="100", channel_id="300")
    intruder = _context(actor_id="101", channel_id="300")
    created = await _remember(
        service,
        context=owner,
        scope=AgentMemoryScope.USER,
        key="response.depth",
        summary="Give substantive answers.",
        source_id="111",
    )
    memory_id = created.memory.memory_id

    with pytest.raises(UserError) as denied:
        await service.update(
            AgentMemoryUpdateRequest(
                memory_id=memory_id,
                summary="Give one-word answers.",
                source_message_ids=("222",),
                confidence=1.0,
            ),
            intruder,
        )
    assert denied.value.code == "memory.not_found"

    updated = await service.update(
        AgentMemoryUpdateRequest(
            memory_id=memory_id,
            summary="Give direct but substantive answers.",
            source_message_ids=("333",),
            confidence=0.9,
            ttl_days=30,
        ),
        owner,
    )
    assert updated.created is False
    assert updated.memory.summary == "Give direct but substantive answers."
    assert updated.memory.expires_at is not None

    with pytest.raises(UserError) as denied_forget:
        await service.forget(AgentMemoryForgetRequest(memory_id), intruder)
    assert denied_forget.value.code == "memory.not_found"

    forgotten = await service.forget(AgentMemoryForgetRequest(memory_id), owner)
    assert forgotten.forgotten is True
    with pytest.raises(UserError) as repeated:
        await service.forget(AgentMemoryForgetRequest(memory_id), owner)
    assert repeated.value.code == "memory.not_found"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("memory_request", "error_code"),
    (
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.USER,
                key="credential",
                summary="API key = sk-abcdefghijklmnop",
                source_message_ids=("111",),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=1.0,
            ),
            "memory.secret_forbidden",
        ),
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.USER,
                key="profile.language",
                summary="Probably prefers English.",
                source_message_ids=("111",),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=0.95,
            ),
            "memory.inference_forbidden",
        ),
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.USER,
                key="profile.language",
                summary="Prefers English.",
                source_message_ids=("111",),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=0.79,
            ),
            "memory.confidence_too_low",
        ),
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.PROCEDURE,
                key="procedure.review",
                summary="Read the source and verify the output.",
                source_message_ids=("111",),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=1.0,
            ),
            "memory.basis_invalid",
        ),
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.USER,
                key="procedure.failed",
                summary="This method failed under the verified test condition.",
                source_message_ids=("111",),
                basis=AgentMemoryBasis.VERIFIED_FAILURE,
                confidence=1.0,
            ),
            "memory.basis_invalid",
        ),
        (
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.USER,
                key="profile.language",
                summary="Prefers English.",
                source_message_ids=("not-a-discord-id",),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=1.0,
            ),
            "memory.source_message_ids_invalid",
        ),
    ),
)
async def test_memory_rejects_secrets_inference_low_confidence_and_bad_evidence(
    tmp_path,
    memory_request: AgentMemoryRememberRequest,
    error_code: str,
) -> None:
    service = _service(tmp_path / "memory.sqlite3")

    with pytest.raises(UserError) as error:
        await service.remember(memory_request, _context())

    assert error.value.code == error_code


@pytest.mark.asyncio
async def test_memory_ttl_cleanup_and_per_scope_cap_are_bounded(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = AgentMemoryStore(
        path,
        max_records=4,
        max_records_per_workspace=3,
        max_records_per_user=2,
        max_records_per_channel=2,
        max_workspace_records=2,
        max_procedure_records=2,
    )
    started = datetime(2026, 7, 1, tzinfo=UTC)
    ids: list[str] = []
    for index in range(3):
        record, _ = await store.remember(
            scope=AgentMemoryScope.USER,
            workspace_id="200",
            owner_user_id="100",
            channel_id=None,
            key=f"preference.{index}",
            summary=f"Preference {index}.",
            source_message_ids=(str(1000 + index),),
            basis=AgentMemoryBasis.USER_STATED,
            confidence=1.0,
            expires_at=(
                started + timedelta(days=1)
                if index == 2
                else None
            ),
            now=started + timedelta(minutes=index),
        )
        ids.append(record.memory_id)

    visible = await store.search(
        query="",
        scopes=(AgentMemoryScope.USER,),
        limit=10,
        workspace_id="200",
        actor_id="100",
        channel_id="300",
        now=started + timedelta(hours=1),
    )
    assert {item.memory_id for item in visible} == {ids[1], ids[2]}
    assert ids[0] not in {item.memory_id for item in visible}

    removed = await store.cleanup(now=started + timedelta(days=2))
    assert removed == 1
    remaining = await store.search(
        query="",
        scopes=(AgentMemoryScope.USER,),
        limit=10,
        workspace_id="200",
        actor_id="100",
        channel_id="300",
        now=started + timedelta(days=2),
    )
    assert [item.memory_id for item in remaining] == [ids[1]]


@pytest.mark.asyncio
async def test_memory_narrow_cap_eviction_preserves_unrelated_shared_memory(
    tmp_path,
) -> None:
    store = AgentMemoryStore(
        tmp_path / "memory.sqlite3",
        max_records=3,
        max_records_per_workspace=3,
        max_records_per_user=2,
        max_records_per_channel=2,
        max_workspace_records=2,
        max_procedure_records=2,
    )
    started = datetime(2026, 7, 1, tzinfo=UTC)
    shared, _ = await store.remember(
        scope=AgentMemoryScope.WORKSPACE,
        workspace_id="200",
        owner_user_id=None,
        channel_id=None,
        key="shared.rule",
        summary="Shared server rule.",
        source_message_ids=("1000",),
        source_message_locators=(
            AgentMemorySourceLocator("1000", "300", "200"),
        ),
        source_resources=(
            ("200", "300", AgentMemoryVisibility.RESTRICTED),
        ),
        origin_channel_id="300",
        created_by_actor_id="100",
        is_curator=True,
        basis=AgentMemoryBasis.USER_STATED,
        confidence=1.0,
        expires_at=None,
        now=started,
    )
    user_ids: list[str] = []
    for index in range(3):
        record, _ = await store.remember(
            scope=AgentMemoryScope.USER,
            workspace_id="200",
            owner_user_id="100",
            channel_id=None,
            key=f"user.preference.{index}",
            summary=f"User preference {index}.",
            source_message_ids=(str(1001 + index),),
            basis=AgentMemoryBasis.USER_STATED,
            confidence=1.0,
            expires_at=None,
            now=started + timedelta(minutes=index + 1),
        )
        user_ids.append(record.memory_id)

    visible = await store.search(
        query="",
        scopes=(AgentMemoryScope.USER, AgentMemoryScope.WORKSPACE),
        limit=10,
        workspace_id="200",
        actor_id="100",
        channel_id="300",
        resource_ids=("300",),
        now=started + timedelta(hours=1),
    )
    visible_ids = {memory.memory_id for memory in visible}
    assert visible_ids == {shared.memory_id, user_ids[1], user_ids[2]}


def test_memory_schema_has_only_bounded_summary_and_source_pointers(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    AgentMemoryStore(path)
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(agent_memories)"
            ).fetchall()
        }

    assert "summary" in columns
    assert "source_message_ids_json" in columns
    assert "source_message_locators_json" in columns
    assert "message_content" not in columns
    assert "attachment" not in columns
    request_fields = {item.name for item in fields(AgentMemoryRememberRequest)}
    assert "target_user_id" not in request_fields
    assert "message_content" not in request_fields
    assert "attachment" not in request_fields


@pytest.mark.asyncio
async def test_memory_policy_migration_marks_legacy_shared_rows_pending(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    timestamp = "2026-08-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_memories (
                memory_id TEXT PRIMARY KEY,
                locator TEXT NOT NULL UNIQUE,
                scope TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                owner_user_id TEXT,
                channel_id TEXT,
                memory_key TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_message_ids_json TEXT NOT NULL,
                source_message_locators_json TEXT NOT NULL DEFAULT '[]',
                basis TEXT NOT NULL CHECK (
                    basis IN ('user_stated', 'verified_success', 'verified_failure')
                ),
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                expires_at TEXT
            );
            """
        )
        connection.execute(
            """
            INSERT INTO agent_memories(
                memory_id, locator, scope, workspace_id, owner_user_id,
                channel_id, memory_key, summary, source_message_ids_json,
                source_message_locators_json, basis, confidence, created_at,
                updated_at, last_used_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "mem_" + "b" * 32,
                "workspace\x1f200\x1f-\x1fworkspace.legacy",
                "workspace",
                "200",
                None,
                None,
                "workspace.legacy",
                "Legacy shared memory requires review.",
                '["111"]',
                '[{"message_id":"111","channel_id":"300","guild_id":"200"}]',
                "user_stated",
                1.0,
                timestamp,
                timestamp,
                timestamp,
                None,
            ),
        )

    AgentMemoryStore(path)
    with sqlite3.connect(path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(agent_memories)"
            ).fetchall()
        }
    assert {
        "created_by_actor_id",
        "source_audience_json",
        "origin_channel_id",
        "origin_visibility",
        "review_state",
        "reviewed_by_actor_id",
        "reviewed_at",
        "verified_action_id",
    } <= columns

    service = _service(path)
    ordinary = await service.search(
        AgentMemorySearchRequest(
            query="legacy",
            scopes=(AgentMemoryScope.WORKSPACE,),
        ),
        _context(actor_id="100"),
    )
    curator = await service.search(
        AgentMemorySearchRequest(
            query="legacy",
            scopes=(AgentMemoryScope.WORKSPACE,),
        ),
        _context(actor_id="900", curator=True),
    )
    assert ordinary.memories == ()
    assert curator.memories[0].created_by_actor_id == "simajilord:legacy"
    assert curator.memories[0].review_state is AgentMemoryReviewState.PENDING
    assert curator.memories[0].provenance.source_resources == (
        ("200", "300", AgentMemoryVisibility.UNCERTAIN),
    )


@pytest.mark.asyncio
async def test_memory_locator_migration_backfills_channel_provenance(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    context = _context(workspace_id="200", channel_id="300")
    service = _service(path)
    remembered = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.CHANNEL,
        key="channel.rule",
        summary="Use the release checklist in this channel.",
        source_id="444",
    )
    assert remembered.memory.source_message_locators[0].channel_id == "300"

    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE agent_memories DROP COLUMN source_message_locators_json"
        )
    restarted = _service(path)
    found = await restarted.search(
        AgentMemorySearchRequest(
            query="release checklist",
            scopes=(AgentMemoryScope.CHANNEL,),
        ),
        context,
    )

    assert found.memories[0].source_message_locators == (
        AgentMemorySourceLocator(
            message_id="444",
            channel_id="300",
            guild_id="200",
        ),
    )


@pytest.mark.asyncio
async def test_memory_accepts_explicit_cross_channel_source_locator(tmp_path) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    context = _context(resource_ids=("300", "301"))
    response = await service.remember(
        AgentMemoryRememberRequest(
            scope=AgentMemoryScope.PROCEDURE,
            key="procedure.cross_channel",
            summary="Use the verified cross-channel procedure.",
            source_message_ids=("555",),
            source_message_locators=(
                AgentMemorySourceLocator(
                    message_id="555",
                    channel_id="301",
                    guild_id="200",
                ),
            ),
            basis=AgentMemoryBasis.VERIFIED_SUCCESS,
            verified_action_id=_CONFIRMED_ACTION_ID,
            confidence=1.0,
        ),
        context,
    )

    assert response.memory.source_message_locators[0].guild_id == "200"
    assert response.memory.source_message_locators[0].channel_id == "301"


@pytest.mark.asyncio
async def test_memory_rejects_cross_workspace_source_locator(tmp_path) -> None:
    service = _service(tmp_path / "memory.sqlite3")
    with pytest.raises(UserError) as error:
        await service.remember(
            AgentMemoryRememberRequest(
                scope=AgentMemoryScope.WORKSPACE,
                key="workspace.cross_server",
                summary="Do not launder a source from another server.",
                source_message_ids=("555",),
                source_message_locators=(
                    AgentMemorySourceLocator("555", "301", "201"),
                ),
                basis=AgentMemoryBasis.USER_STATED,
                confidence=1.0,
            ),
            _context(resource_ids=("300", "301")),
        )
    assert error.value.code == "memory.source_workspace_mismatch"


@pytest.mark.asyncio
async def test_memory_capabilities_are_discoverable_authorized_and_receipted(
    tmp_path,
) -> None:
    registry = CapabilityRegistry()
    service = _service(tmp_path / "memory.sqlite3")
    for item in build_memory_endpoints(service):
        registry.register(item)
    required_grants = {
        name: AGENT_MEMORY_GRANT
        for name in (
            "memory.search",
            *AGENT_MEMORY_WRITE_CAPABILITIES,
        )
        if name != "memory.review"
    }
    required_grants["memory.review"] = AGENT_MEMORY_CURATOR_GRANT
    catalog = AgentToolCatalog(
        registry,
        ("memory.search", *AGENT_MEMORY_WRITE_CAPABILITIES),
        required_grants=required_grants,
        eager_capabilities=(
            "memory.search",
            "memory.remember",
            "memory.update",
        ),
        write_capabilities=AGENT_MEMORY_WRITE_CAPABILITIES,
    )
    context = _context(
        approvals=frozenset(AGENT_MEMORY_WRITE_CAPABILITIES)
    )

    tool_names = {
        str(tool["name"])
        for namespace in catalog.dynamic_specs(context)
        for tool in namespace["tools"]
    }
    assert {
        "memory_search",
        "memory_remember",
        "memory_update",
        "capability_search",
    } <= tool_names
    ordinary_list = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_list",
        arguments={"limit": 25},
        context=context,
        max_output_characters=10_000,
    )
    curator_list = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_list",
        arguments={"limit": 25},
        context=replace(
            context,
            grants=context.grants | {AGENT_MEMORY_CURATOR_GRANT},
        ),
        max_output_characters=10_000,
    )
    assert "memory.review" not in {
        item["name"] for item in json.loads(ordinary_list.text)["tools"]
    }
    assert "memory.review" in {
        item["name"] for item in json.loads(curator_list.text)["tools"]
    }
    memory_tool = next(
        tool
        for namespace in catalog.dynamic_specs(context)
        for tool in namespace["tools"]
        if tool["name"] == "memory_search"
    )
    assert {
        "query",
        "scopes",
        "basis",
        "min_confidence",
        "updated_after",
        "offset",
        "limit",
    } <= set(memory_tool["inputSchema"]["properties"])
    properties = memory_tool["inputSchema"]["properties"]
    assert "Japanese or English" in properties["query"]["description"]
    assert "current requester's private memory" in properties["scopes"]["description"]
    assert "next_offset" in properties["limit"]["description"]
    remembered = await _remember(
        service,
        context=context,
        scope=AgentMemoryScope.USER,
        key="response.language",
        summary="Respond in Japanese.",
        source_id="400",
    )
    search_output = await catalog.invoke(
        namespace="simajilord",
        tool_name="memory_search",
        arguments={
            "query": "Japanese",
            "scopes": ["user"],
            "basis": "user_stated",
            "min_confidence": 0.9,
            "offset": 0,
            "limit": 1,
        },
        context=context,
        max_output_characters=10_000,
    )
    payload = json.loads(search_output.text)
    assert payload["memories"][0]["memory_id"] == remembered.memory.memory_id
    broker_discovery = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "search the requester's selective memory"},
        context=context,
        max_output_characters=10_000,
    )
    broker_catalog_id = json.loads(broker_discovery.text)["catalog_id"]
    broker_contract = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_describe",
        arguments={
            "catalog_id": broker_catalog_id,
            "name": "memory.search",
        },
        context=context,
        max_output_characters=10_000,
    )
    broker_contract_id = json.loads(broker_contract.text)["contract_id"]
    brokered_search_output = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_invoke",
        arguments={
            "name": "memory.search",
            "contract_id": broker_contract_id,
            "arguments": {
                "query": "Japanese",
                "scopes": ["user"],
                "basis": "user_stated",
                "min_confidence": 0.9,
                "offset": 0,
                "limit": 1,
            },
        },
        context=context,
        max_output_characters=10_000,
    )
    assert (
        json.loads(brokered_search_output.text)["memories"][0]["memory_id"]
        == remembered.memory.memory_id
    )
    assert payload["memories"][0]["source_message_ids"] == ["400"]
    assert payload["memories"][0]["source_message_locators"] == [
        {
            "channel_id": "300",
            "guild_id": "200",
            "message_edited_at": None,
            "message_id": "400",
        }
    ]
    assert payload["memories"][0]["basis"] == "user_stated"
    assert payload["memories"][0]["confidence"] == 1.0
    assert payload["memories"][0]["updated_at"]
    assert payload["next_offset"] is None
    discovery = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={
            "query": "update or forget an existing memory by result ID",
            "limit": 5,
        },
        context=context,
        max_output_characters=10_000,
    )
    assert '"name":"memory.update"' in discovery.text
    assert '"name":"memory.forget"' in discovery.text
    invoke_arguments = {
        "name": "memory.remember",
        "arguments": {},
        "authorization_event_id": "discord:message:400",
    }
    assert catalog.write_capability_for_call(
        tool_name="capability_invoke",
        arguments=invoke_arguments,
    ) == "memory.remember"
    assert catalog.authorization_event_id_for_call(
        tool_name="capability_invoke",
        arguments=invoke_arguments,
    ) == "discord:message:400"
    assert registry.search("remember preference")[0].descriptor.name == (
        "memory.remember"
    )
    assert registry.endpoint("memory.search").descriptor.risk is RiskLevel.READ
    for capability in AGENT_MEMORY_WRITE_CAPABILITIES:
        descriptor = registry.endpoint(capability).descriptor
        assert descriptor.risk is RiskLevel.WRITE
        assert descriptor.approval is ApprovalMode.WHEN_REQUESTED
        assert capability in NON_UNDOABLE_ACTION_CAPABILITIES
        assert (
            action_policy(capability).classification
            is ActionClassification.NON_UNDOABLE
        )
    assert (
        registry.endpoint("memory.forget").descriptor.idempotency
        == "non_idempotent_write"
    )
