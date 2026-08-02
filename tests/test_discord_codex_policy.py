from __future__ import annotations

import json
from pathlib import Path

import pytest

from simajilord.agent.providers.codex import (
    CodexAppServerProvider,
    _ToolTurnBudget,
)
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.observability.journal import EventJournal
from simajilord.providers.codex_features import codex_feature_arguments
from simajilord.providers.discord_codex_policy import (
    DISCORD_CODEX_APP_POLICIES,
    DISCORD_CODEX_DISABLED_PLUGINS,
    DISCORD_CODEX_ENABLED_PLUGINS,
    DISCORD_CODEX_PERMISSION_PROFILE,
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
    return dict(setting.split("=", 1) for _flag, setting in pairs)


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

    overrides = _config_overrides(discord_codex_policy_arguments(codex_home=codex_home))

    assert overrides["apps._default.enabled"] == "false"
    assert overrides["apps._default.destructive_enabled"] == "false"
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
    assert overrides["mcp_servers.openaiDeveloperDocs.enabled"] == "true"
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


def test_discord_extensions_enable_only_reviewed_feature_families() -> None:
    arguments = codex_feature_arguments(allow_discord_extensions=True)
    pairs = set(zip(arguments[::2], arguments[1::2], strict=True))

    for feature in (
        "apps",
        "plugins",
        "remote_plugin",
        "skill_search",
        "tool_suggest",
    ):
        assert ("--enable", feature) in pairs
        assert ("--disable", feature) not in pairs
    for feature in (
        "browser_use",
        "code_mode_host",
        "computer_use",
        "goals",
        "multi_agent",
        "shell_tool",
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
    assert not discord_codex_app_tool_is_write("unknown", "generate_figma_design")


def test_provider_uses_one_isolated_workspace_per_discord_scope(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    guild_a = InvocationContext("actor-a", "guild-a", "discord", "request-a")
    guild_a_other_user = InvocationContext("actor-b", "guild-a", "discord", "request-b")
    guild_b = InvocationContext("actor-a", "guild-b", "discord", "request-c")
    direct = InvocationContext("actor-a", None, "discord", "request-d")

    workspace_a = provider._workspace_for_context(guild_a)
    assert workspace_a == provider._workspace_for_context(guild_a_other_user)
    workspaces = {
        workspace_a,
        provider._workspace_for_context(guild_b),
        provider._workspace_for_context(direct),
    }
    assert len(workspaces) == 3
    assert workspace_a.parent == provider.workspace_dir.resolve()
    assert workspace_a.stat().st_mode & 0o777 == 0o700


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
