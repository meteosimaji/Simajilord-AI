from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, call

import pytest

from simajilord.agent import AgentProviderError, AgentUnavailableError
from simajilord.agent.providers.codex import (
    CodexAppServerProvider,
    _codex_app_server_environment,
    _ToolTurnBudget,
    _verify_codex_version,
)
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.observability.journal import EventJournal
from simajilord.providers.codex_features import codex_feature_arguments
from simajilord.providers.discord_codex_policy import (
    DISCORD_CODEX_APP_POLICIES,
    DISCORD_CODEX_BROKER_CONNECTORS,
    DISCORD_CODEX_DISABLED_PLUGINS,
    DISCORD_CODEX_ENABLED_PLUGINS,
    DISCORD_CODEX_PERMISSION_PROFILE,
    DiscordCodexAppActionClass,
    discord_codex_app_tool_action_class,
    discord_codex_app_tool_is_write,
    discord_codex_policy_arguments,
)


def _provider(tmp_path: Path, *, trace_sink: EventJournal | None = None) -> CodexAppServerProvider:
    return CodexAppServerProvider(
        executable="codex",
        model="test",
        workspace_dir=tmp_path / "discord-workspaces",
        idle_timeout_seconds=10,
        reasoning_effort="low",
        tools=AgentToolCatalog(CapabilityRegistry(), ()),
        max_tool_calls=4,
        max_tool_output_characters=4_000,
        trace_sink=trace_sink,
    )


def _config_overrides(arguments: tuple[str, ...]) -> dict[str, str]:
    pairs = tuple(zip(arguments[::2], arguments[1::2], strict=True))
    assert all(flag == "-c" for flag, _setting in pairs)
    settings = tuple(setting.split("=", 1) for _flag, setting in pairs)
    assert len(settings) == len({key for key, _value in settings})
    return dict(settings)


def test_discord_app_and_plugin_policy_has_no_collisions() -> None:
    app_ids = [policy.app_id for policy in DISCORD_CODEX_APP_POLICIES]
    app_names = [policy.name for policy in DISCORD_CODEX_APP_POLICIES]

    assert len(app_ids) == len(set(app_ids)) == 12
    assert len(app_names) == len(set(app_names)) == 12
    assert set(DISCORD_CODEX_ENABLED_PLUGINS).isdisjoint(DISCORD_CODEX_DISABLED_PLUGINS)
    assert len(DISCORD_CODEX_ENABLED_PLUGINS) == len(set(DISCORD_CODEX_ENABLED_PLUGINS))
    assert len(DISCORD_CODEX_DISABLED_PLUGINS) == len(set(DISCORD_CODEX_DISABLED_PLUGINS))
    for policy in DISCORD_CODEX_APP_POLICIES:
        if policy.enabled_tools is not None:
            assert len(policy.enabled_tools) == len(set(policy.enabled_tools))

    policies = {policy.name: policy for policy in DISCORD_CODEX_APP_POLICIES}
    assert len(policies["GitHub"].enabled_tools or ()) == 50
    assert len(policies["Hugging Face"].enabled_tools or ()) == 8
    assert set(policies["Plugin Management"].enabled_tools or ()) == {
        "get_app_permissions",
        "get_plugin_dependencies",
    }


def test_discord_policy_is_fail_closed_and_process_local(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        '[mcp_servers.personal_gateway]\nurl = "https://example.invalid/mcp"\n',
        encoding="utf-8",
    )
    disabled_skill = codex_home / "skills" / "playwright" / "SKILL.md"
    enabled_skill = codex_home / "skills" / "imagegen" / "SKILL.md"
    disabled_skill.parent.mkdir(parents=True)
    enabled_skill.parent.mkdir(parents=True)
    disabled_skill.write_text("disabled", encoding="utf-8")
    enabled_skill.write_text("enabled", encoding="utf-8")
    disabled_plugin_skill = (
        codex_home
        / "plugins"
        / "cache"
        / "openai-curated-remote"
        / "hugging-face"
        / "1.0.0"
        / "skills"
        / "jobs"
        / "SKILL.md"
    )
    enabled_plugin_skill = disabled_plugin_skill.parents[1] / "papers" / "SKILL.md"
    disabled_plugin_skill.parent.mkdir(parents=True)
    enabled_plugin_skill.parent.mkdir(parents=True)
    disabled_plugin_skill.write_text("disabled", encoding="utf-8")
    enabled_plugin_skill.write_text("enabled", encoding="utf-8")
    plugin_manifest = enabled_plugin_skill.parents[2] / ".mcp.json"
    plugin_manifest.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "design_plugin_server": {
                        "command": "unsafe-plugin-command",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    overrides = _config_overrides(discord_codex_policy_arguments(codex_home=codex_home))

    assert overrides["apps._default.enabled"] == "false"
    assert overrides["apps._default.destructive_enabled"] == "false"
    assert overrides["apps._default.open_world_enabled"] == "false"
    assert overrides["mcp_servers.node_repl.enabled"] == "false"
    assert overrides["mcp_servers.playwright.enabled"] == "false"
    assert overrides["mcp_servers.computer-use.enabled"] == "false"
    for server in (
        "creative_production_mcp",
        "dataAnalyticsWidgets",
        "event-stream",
        "openai-api-key-local-confirmation",
    ):
        assert overrides[f"mcp_servers.{server}.command"] == '"/usr/bin/false"'
        assert overrides[f"mcp_servers.{server}.enabled"] == "false"
    assert overrides["mcp_servers.openaiDeveloperDocs.enabled"] == "false"
    assert overrides["mcp_servers.design_plugin_server.command"] == '"/usr/bin/false"'
    assert overrides["mcp_servers.design_plugin_server.enabled"] == "false"
    assert overrides["mcp_servers.personal_gateway.enabled"] == "false"
    assert "mcp_servers.personal_gateway.command" not in overrides
    assert overrides["shell_environment_policy.inherit"] == '"none"'
    assert overrides["shell_environment_policy.set.HOME"] == '"/nonexistent"'
    assert (
        overrides[f"permissions.{DISCORD_CODEX_PERMISSION_PROFILE}"]
        == '{description="Discord-only isolated workspace",extends=":workspace",'
        'filesystem={":minimal"="read",":workspace_roots"={"."="write"}},'
        "network={enabled=false}}"
    )
    assert '"github@openai-curated"={enabled=true}' in overrides["plugins"]
    assert '"gmail@openai-curated"={enabled=false}' in overrides["plugins"]
    assert (
        '"record-and-replay@openai-bundled"='
        '{enabled=false,mcp_servers={"event-stream"={enabled=false}}}'
        in overrides["plugins"]
    )
    assert str(disabled_skill.resolve()) in overrides["skills.config"]
    assert str(enabled_skill.resolve()) not in overrides["skills.config"]
    assert str(disabled_plugin_skill.resolve()) in overrides["skills.config"]
    assert str(enabled_plugin_skill.resolve()) not in overrides["skills.config"]
    for policy in DISCORD_CODEX_APP_POLICIES:
        prefix = f"apps.{policy.app_id}"
        assert overrides[f"{prefix}.enabled"] == "false"
        assert overrides[f"{prefix}.destructive_enabled"] == "false"
        assert overrides[f"{prefix}.open_world_enabled"] == "false"
        assert overrides[f"{prefix}.default_tools_enabled"] == "false"
    assert set(DISCORD_CODEX_BROKER_CONNECTORS.values()) == {
        "Adobe",
        "Adobe Express",
        "BioRender",
        "Canva",
        "Figma",
    }


def test_discord_policy_rejects_an_uninspectable_plugin_mcp_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "plugins" / "cache" / "vendor" / "plugin" / "1" / ".mcp.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Plugin MCP manifest is invalid"):
        discord_codex_policy_arguments(codex_home=tmp_path)


def test_discord_extensions_enable_only_reviewed_feature_families() -> None:
    arguments = codex_feature_arguments(allow_discord_extensions=True)
    pairs = set(zip(arguments[::2], arguments[1::2], strict=True))

    for feature in (
        "apps",
        "plugins",
        "skill_search",
    ):
        assert ("--enable", feature) in pairs
        assert ("--disable", feature) not in pairs
    for feature in (
        "browser_use",
        "code_mode_host",
        "computer_use",
        "goals",
        "multi_agent",
        "remote_plugin",
        "shell_tool",
        "tool_suggest",
        "unified_exec",
        "workspace_dependencies",
    ):
        assert ("--disable", feature) in pairs


def test_app_write_classification_is_exact() -> None:
    assert discord_codex_app_tool_is_write(
        "connector_68df038e0ba48191908c8434991bbac2",
        "generate_figma_design",
    )
    assert not discord_codex_app_tool_is_write(
        "connector_76869538009648d5b282a4bb21c3d157",
        "fetch_file",
    )
    assert discord_codex_app_tool_is_write("unknown", "generate_figma_design")
    assert (
        discord_codex_app_tool_action_class("unknown", "unknown")
        is DiscordCodexAppActionClass.UNKNOWN
    )


def test_provider_uses_one_isolated_workspace_per_actor_and_task(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    guild_a = InvocationContext(
        "actor-a",
        "guild-a",
        "discord",
        "request-a",
        agent_task_id="tsk_aaaaaaaaaaaaaaaaaaaaaaaa",
    )
    guild_a_other_user = InvocationContext(
        "actor-b",
        "guild-a",
        "discord",
        "request-b",
        agent_task_id="tsk_bbbbbbbbbbbbbbbbbbbbbbbb",
    )
    guild_a_other_task = InvocationContext(
        "actor-a",
        "guild-a",
        "discord",
        "request-c",
        agent_task_id="tsk_cccccccccccccccccccccccc",
    )
    guild_b = InvocationContext("actor-a", "guild-b", "discord", "request-d")
    direct = InvocationContext("actor-a", None, "discord", "request-e")

    workspace_a = provider._workspace_for_context(guild_a)
    assert workspace_a == provider._workspace_for_context(guild_a)
    workspaces = {
        workspace_a,
        provider._workspace_for_context(guild_a_other_user),
        provider._workspace_for_context(guild_a_other_task),
        provider._workspace_for_context(guild_b),
        provider._workspace_for_context(direct),
    }
    assert len(workspaces) == 5
    assert workspace_a.parent == provider.workspace_dir.resolve()
    assert workspace_a.stat().st_mode & 0o777 == 0o700


def test_codex_app_server_environment_does_not_inherit_service_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "never-child")
    monkeypatch.setenv("HIVE_API_KEY", "never-child")
    monkeypatch.setenv("WEB_SEARCH_SHARED_SECRET", "never-child")
    monkeypatch.setenv("OPENAI_API_KEY", "never-child")

    environment = _codex_app_server_environment()

    assert environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert "CODEX_HOME" in environment
    assert "HOME" in environment
    assert "DISCORD_TOKEN" not in environment
    assert "HIVE_API_KEY" not in environment
    assert "WEB_SEARCH_SHARED_SECRET" not in environment
    assert "OPENAI_API_KEY" not in environment


@pytest.mark.asyncio
async def test_codex_version_guard_accepts_only_configured_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"codex-cli 0.147.0-alpha.6.5\n", b""

    async def create_process(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = {"PATH": "/usr/bin:/bin"}

    assert await _verify_codex_version(
        "/resolved/codex",
        expected_prefix="0.147.",
        environment=environment,
    ) == "0.147.0-alpha.6.5"
    with pytest.raises(AgentUnavailableError, match="supported prefix"):
        await _verify_codex_version(
            "/resolved/codex",
            expected_prefix="0.146.",
            environment=environment,
        )


@pytest.mark.asyncio
async def test_connector_broker_protocol_is_active_thread_scoped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    provider._active_threads.add("thread")
    request = AsyncMock(
        side_effect=(
            {
                "data": [
                    {
                        "name": "codex_apps",
                        "tools": {
                            "first": {
                                "inputSchema": {"type": "object"},
                            }
                        },
                    },
                    {"name": "disabled", "tools": {}},
                ],
                "nextCursor": "next-page",
            },
            {
                "data": [
                    {
                        "name": "codex_apps",
                        "tools": {
                            "inventory-key": {
                                "name": "second",
                                "inputSchema": {"type": "object"},
                            }
                        },
                    }
                ],
                "nextCursor": None,
            },
            {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            },
        )
    )
    monkeypatch.setattr(provider, "_request", request)

    inventory = await provider.connector_tool_inventory(thread_id="thread")
    result = await provider.call_connector_tool(
        thread_id="thread",
        server="codex_apps",
        tool="second",
        arguments={"id": "design"},
    )

    assert tuple(item["name"] for item in inventory) == ("first", "second")
    assert result["isError"] is False
    assert request.await_args_list == [
        call(
            "mcpServerStatus/list",
            {"detail": "full", "limit": 100, "threadId": "thread"},
        ),
        call(
            "mcpServerStatus/list",
            {
                "detail": "full",
                "limit": 100,
                "threadId": "thread",
                "cursor": "next-page",
            },
        ),
        call(
            "mcpServer/tool/call",
            {
                "server": "codex_apps",
                "threadId": "thread",
                "tool": "second",
                "arguments": {"id": "design"},
            },
        ),
    ]

    with pytest.raises(AgentProviderError, match="not active"):
        await provider.connector_tool_inventory(thread_id="inactive")
    with pytest.raises(AgentProviderError, match="not active"):
        await provider.call_connector_tool(
            thread_id="inactive",
            server="codex_apps",
            tool="second",
            arguments={},
        )


@pytest.mark.asyncio
async def test_app_tool_audit_omits_arguments_results_and_resource_uri(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    provider = _provider(tmp_path, trace_sink=journal)
    reference_id = "agt_0000000000000000000a"
    context = InvocationContext(
        "actor",
        "guild",
        "discord",
        "request",
        public_reference_id=reference_id,
    )
    budget = _ToolTurnBudget(
        context=context,
        calls_remaining=4,
        output_characters_remaining=4_000,
        on_progress=None,
        required_message_id=None,
    )
    provider._active_tool_budgets["thread"] = budget
    provider._thread_by_turn["turn"] = "thread"
    started_item: dict[str, object] = {
        "type": "mcpToolCall",
        "id": "call",
        "server": "codex_apps",
        "tool": "generate_figma_design",
        "arguments": {
            "prompt": "never-persist-this-secret",
            "access_token": "never-persist-this-token",
        },
        "appContext": {
            "connectorId": "connector_68df038e0ba48191908c8434991bbac2",
            "appName": "Figma",
            "actionName": "generate_figma_design",
            "resourceUri": "figma://private/never-persist-this-uri",
        },
        "status": "inProgress",
    }
    params: dict[str, object] = {
        "threadId": "thread",
        "turnId": "turn",
        "item": started_item,
    }

    await provider._handle_notification("item/started", params)
    completed_item = {
        **started_item,
        "status": "completed",
        "result": {"content": "never-persist-this-result"},
    }
    await provider._handle_notification(
        "item/completed",
        {**params, "item": completed_item},
    )

    trace = await journal.agent_trace(public_reference_id=reference_id)
    app_records = [record for record in trace if record.kind.startswith("agent.app_tool.")]
    assert [record.kind for record in app_records] == [
        "agent.app_tool.started",
        "agent.app_tool.finished",
    ]
    serialized = json.dumps([record.payload for record in app_records], ensure_ascii=False)
    assert "never-persist-this-secret" not in serialized
    assert "never-persist-this-token" not in serialized
    assert "never-persist-this-uri" not in serialized
    assert "never-persist-this-result" not in serialized
    argument_names = app_records[0].payload["argument_names"]
    assert isinstance(argument_names, list)
    assert set(argument_names) == {"access_token", "prompt"}
    assert budget.write_attempts == {
        "app:connector_68df038e0ba48191908c8434991bbac2:generate_figma_design"
    }
    assert budget.write_successes == budget.write_attempts
    await journal.close()
