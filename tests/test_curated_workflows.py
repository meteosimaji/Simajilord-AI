from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from simajilord.agent import (
    CuratedWorkflowSearchRequest,
    build_curated_workflow_endpoint,
)
from simajilord.agent.providers import CodexAppServerProvider
from simajilord.config import load_settings
from simajilord.core import InvocationContext
from simajilord.runtime import SimajilordRuntime


async def test_curated_workflow_search_finds_goal_without_command_trigger() -> None:
    endpoint = build_curated_workflow_endpoint(
        frozenset(
            {
                "discord.search_messages",
                "discord.read_messages",
                "web.search",
                "web.fetch",
                "web.find",
            }
        )
    )
    response = await endpoint.invoke(
        CuratedWorkflowSearchRequest(
            query="PDFの特定箇所を探して長文を根拠付きで調査",
        ),
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            grants=frozenset({"web"}),
        ),
    )

    assert response.workflows[0].workflow_id == "web.document_research"
    assert tuple(
        step.capability for step in response.workflows[0].steps
    ) == ("web.search", "web.fetch", "web.find")


async def test_curated_workflow_search_handles_natural_japanese_and_nfkc() -> None:
    endpoint = build_curated_workflow_endpoint(
        frozenset(
            {
                "web.search",
                "web.fetch",
                "web.find",
                "discord.import_attachment",
                "files.read",
                "files.write_text",
                "compute.run",
                "discord.send_file",
                "media.save",
                "files.list",
                "action.undo",
            }
        )
    )
    context = InvocationContext(
        "actor",
        "workspace",
        "agent",
        "event",
        grants=frozenset(
            {
                "web",
                "files",
                "safe_compute",
                "media_download",
                "discord_message",
            }
        ),
    )
    expected_by_query = {
        "添付ＰＤＦの内容を修正して返して": "file.safe_compute_transform",
        "この動画を保存してファイルで送って": "media.save_and_deliver",
        "この議論をスレッドに分けて後で元に戻したい": (
            "action.execute_with_receipt"
        ),
        "長いＰＤＦから根拠を探して": "web.document_research",
    }

    for query, expected in expected_by_query.items():
        response = await endpoint.invoke(
            CuratedWorkflowSearchRequest(query=query, limit=5),
            context,
        )
        assert response.workflows
        assert response.workflows[0].workflow_id == expected


async def test_curated_workflows_hide_missing_capabilities_and_grants() -> None:
    endpoint = build_curated_workflow_endpoint(
        frozenset(
            {
                "web.search",
                "web.fetch",
                "web.find",
                "media.save",
                "files.list",
                "discord.send_file",
            }
        )
    )
    no_grants = await endpoint.invoke(
        CuratedWorkflowSearchRequest(query="web PDF media save", limit=5),
        InvocationContext("actor", "workspace", "agent", "event"),
    )
    web_only = await endpoint.invoke(
        CuratedWorkflowSearchRequest(query="web PDF media save", limit=5),
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            grants=frozenset({"web"}),
        ),
    )

    assert no_grants.workflows == ()
    assert tuple(item.workflow_id for item in web_only.workflows) == (
        "web.document_research",
    )


async def test_curated_workflow_catalog_contains_no_host_execution_steps() -> None:
    capabilities = frozenset(
        {
            "discord.search_messages",
            "discord.read_messages",
            "web.search",
            "web.fetch",
            "web.find",
            "discord.import_attachment",
            "files.read",
            "files.write_text",
            "compute.run",
            "discord.send_file",
            "media.save",
            "files.list",
            "action.undo",
            "memory.search",
            "memory.remember",
            "memory.update",
            "memory.forget",
        }
    )
    endpoint = build_curated_workflow_endpoint(capabilities)
    response = await endpoint.invoke(
        CuratedWorkflowSearchRequest(
            query="research file media action memory PDF 保存 調査",
            limit=5,
        ),
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            grants=frozenset(
                {
                    "web",
                    "files",
                    "safe_compute",
                    "media_download",
                    "discord_message",
                    "memory",
                }
            ),
        ),
    )

    assert response.workflows
    forbidden = {"shell", "exec", "mcp", "plugin", "host.file"}
    step_capabilities = {
        step.capability
        for workflow in response.workflows
        for step in workflow.steps
    }
    assert not step_capabilities & forbidden
    assert all(
        capability.startswith(
            (
                "action.",
                "capability_",
                "compute.",
                "discord.",
                "files.",
                "media.",
                "memory.",
                "web.",
            )
        )
        for capability in step_capabilities
    )


async def test_curated_memory_workflow_connects_search_update_and_forget() -> None:
    endpoint = build_curated_workflow_endpoint(
        frozenset(
            {
                "memory.search",
                "memory.remember",
                "memory.update",
                "memory.forget",
            }
        )
    )
    response = await endpoint.invoke(
        CuratedWorkflowSearchRequest(
            query="好みや成功した手順を必要な時だけ覚えて後で修正したい",
        ),
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            grants=frozenset({"memory"}),
        ),
    )

    assert response.workflows[0].workflow_id == "memory.selective_capture"
    assert tuple(
        step.capability for step in response.workflows[0].steps
    ) == (
        "memory.search",
        "memory.remember",
        "memory.update",
        "memory.forget",
    )
    assert "explicitly asks" in response.workflows[0].steps[-1].instruction


@pytest.mark.parametrize(
    "query",
    (
        "前に成功したやり方をもう一度使いたい",
        "私の好みを思い出して",
    ),
)
async def test_curated_memory_workflow_is_found_from_natural_japanese(
    query: str,
) -> None:
    endpoint = build_curated_workflow_endpoint(
        frozenset(
            {
                "memory.search",
                "memory.remember",
                "memory.update",
                "memory.forget",
            }
        )
    )

    response = await endpoint.invoke(
        CuratedWorkflowSearchRequest(query=query),
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "event",
            grants=frozenset({"memory"}),
        ),
    )

    assert response.workflows[0].workflow_id == "memory.selective_capture"


async def test_curated_workflow_hides_unapproved_step_capabilities() -> None:
    memory_capabilities = frozenset(
        {
            "memory.search",
            "memory.remember",
            "memory.update",
            "memory.forget",
        }
    )
    endpoint = build_curated_workflow_endpoint(
        memory_capabilities,
        capability_grants={
            capability: "memory" for capability in memory_capabilities
        },
        approval_capabilities=frozenset(
            {
                "memory.remember",
                "memory.update",
                "memory.forget",
            }
        ),
    )
    query = CuratedWorkflowSearchRequest(
        query="好みや成功手順を必要な時だけ覚えて後で修正したい"
    )
    grant_only = await endpoint.invoke(
        query,
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "grant-only",
            grants=frozenset({"memory"}),
        ),
    )
    approved = await endpoint.invoke(
        query,
        InvocationContext(
            "actor",
            "workspace",
            "agent",
            "approved",
            grants=frozenset({"memory"}),
            approvals=frozenset(
                {
                    "memory.remember",
                    "memory.update",
                    "memory.forget",
                }
            ),
        ),
    )

    assert grant_only.workflows == ()
    assert approved.workflows[0].workflow_id == "memory.selective_capture"


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_curated_workflow_switch_controls_registry_and_agent_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: bool,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv(
        "AGENT_CURATED_SKILLS_ENABLED",
        "true" if enabled else "false",
    )
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)

    try:
        capability_names = {
            endpoint.descriptor.name for endpoint in runtime.registry.all()
        }
        assert ("workflow.search" in capability_names) is enabled
        assert runtime.agent is not None
        provider = cast(CodexAppServerProvider, runtime.agent.provider)
        specs = provider.tools.dynamic_specs(
            InvocationContext("actor", "workspace", "agent", "event")
        )
        aliases = {
            str(tool["name"])
            for namespace in specs
            for tool in namespace["tools"]
            if isinstance(tool, dict)
        }
        assert ("workflow_search" in aliases) is enabled
        assert "capability_search" in aliases
    finally:
        asyncio.run(runtime.close())
