from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from simajilord.agent import AgentProviderError, AgentUnavailableError
from simajilord.agent.providers.codex import (
    _CODEX_PROTOCOL_ENVELOPE_CONTRACTS,
    _CODEX_PROTOCOL_METHOD_CONTRACTS,
    _CODEX_PROTOCOL_RESPONSE_CONTRACTS,
    CodexAppServerProvider,
    _codex_app_server_environment,
    _codex_runtime_arguments,
    _ToolTurnBudget,
    _validate_codex_protocol_schema,
    _verify_codex_runtime_configuration,
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


def _write_minimal_codex_protocol_schema(schema_dir: Path) -> None:
    string_parameter_values = {
        ("thread/start", "approvalPolicy"): ["never"],
        ("thread/start", "sandbox"): ["read-only"],
        ("thread/start", "historyMode"): ["legacy"],
        ("thread/start", "sessionStartSource"): ["startup"],
        ("thread/resume", "approvalPolicy"): ["never"],
        ("turn/start", "approvalPolicy"): ["never"],
        ("turn/start", "effort"): ["high", "low"],
        ("mcpServerStatus/list", "detail"): ["full"],
    }
    for schema_file, (envelope_fields, _sent_envelope_fields) in (
        _CODEX_PROTOCOL_ENVELOPE_CONTRACTS.items()
    ):
        branches: list[dict[str, object]] = []
        for contract in _CODEX_PROTOCOL_METHOD_CONTRACTS:
            if contract.schema_file != schema_file:
                continue
            properties: dict[str, object] = {field: {} for field in envelope_fields}
            properties["method"] = {"enum": [contract.method]}
            if "params" in envelope_fields:
                parameter_properties: dict[str, object] = {
                    field: {} for field in contract.parameter_fields
                }
                for (method, parameter), values in string_parameter_values.items():
                    if method == contract.method and parameter in parameter_properties:
                        parameter_properties[parameter] = {"enum": values}
                if contract.method == "turn/start" and "sandboxPolicy" in parameter_properties:
                    parameter_properties["sandboxPolicy"] = {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"type": {"enum": ["readOnly"]}},
                                "required": ["type"],
                            }
                        ]
                    }
                properties["params"] = {
                    "type": "object",
                    "properties": parameter_properties,
                    "required": [],
                }
            branches.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(envelope_fields),
                }
            )
        (schema_dir / schema_file).write_text(
            json.dumps({"oneOf": branches}),
            encoding="utf-8",
        )

    for contract in _CODEX_PROTOCOL_RESPONSE_CONTRACTS:
        path = schema_dir / contract.schema_file
        path.parent.mkdir(parents=True, exist_ok=True)
        properties: dict[str, object] = {field: {} for field in contract.property_fields}
        if contract.nested_property is not None:
            properties[contract.nested_property] = {
                "type": "object",
                "properties": {
                    field: {} for field in contract.nested_property_fields
                },
                "required": sorted(contract.nested_required_fields),
            }
        if contract.schema_file == "DynamicToolCallResponse.json":
            properties["contentItems"] = {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"enum": ["inputText"]},
                            },
                            "required": ["text", "type"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "imageUrl": {"type": "string"},
                                "type": {"enum": ["inputImage"]},
                            },
                            "required": ["imageUrl", "type"],
                        },
                    ]
                },
            }
        if contract.schema_file in {
            "CommandExecutionRequestApprovalResponse.json",
            "FileChangeRequestApprovalResponse.json",
        }:
            properties["decision"] = {"enum": ["decline"]}
        path.write_text(
            json.dumps(
                {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(contract.required_fields),
                }
            ),
            encoding="utf-8",
        )


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
        (
            '[mcp_servers.personal_gateway]\nurl = "https://example.invalid/mcp"\n'
            '[mcp_servers.cua_repl]\ncommand = "/configured/cua-repl"\n'
        ),
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
    assert overrides["mcp_servers.node_repl.command"] == '"/usr/bin/false"'
    assert overrides["mcp_servers.playwright.enabled"] == "false"
    assert overrides["mcp_servers.playwright.command"] == '"/usr/bin/false"'
    assert overrides["mcp_servers.computer-use.enabled"] == "false"
    assert overrides["mcp_servers.computer-use.command"] == '"/usr/bin/false"'
    for server in (
        "creative_production_mcp",
        "dataAnalyticsWidgets",
        "event-stream",
        "openai-api-key-local-confirmation",
    ):
        assert overrides[f"mcp_servers.{server}.command"] == '"/usr/bin/false"'
        assert overrides[f"mcp_servers.{server}.enabled"] == "false"
    assert overrides["mcp_servers.openaiDeveloperDocs.enabled"] == "false"
    assert overrides["mcp_servers.openaiDeveloperDocs.command"] == '"/usr/bin/false"'
    assert overrides["mcp_servers.design_plugin_server.command"] == '"/usr/bin/false"'
    assert overrides["mcp_servers.design_plugin_server.enabled"] == "false"
    assert overrides["mcp_servers.personal_gateway.enabled"] == "false"
    assert "mcp_servers.personal_gateway.command" not in overrides
    assert overrides["mcp_servers.personal_gateway.url"] == '"http://127.0.0.1:9"'
    assert overrides["mcp_servers.cua_repl.enabled"] == "false"
    assert overrides["mcp_servers.cua_repl.command"] == '"/usr/bin/false"'
    assert "example.invalid" not in " ".join(overrides.values())
    assert "/configured/cua-repl" not in " ".join(overrides.values())
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


def test_discord_policy_rejects_configured_mcp_without_transport(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "[mcp_servers.cua_repl]\nenabled = false\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="invalid transport"):
        discord_codex_policy_arguments(codex_home=tmp_path)


def test_discord_extensions_enable_only_reviewed_feature_families() -> None:
    arguments = codex_feature_arguments(allow_discord_extensions=True)
    ordered_pairs = tuple(zip(arguments[::2], arguments[1::2], strict=True))
    pairs = set(ordered_pairs)

    assert len(ordered_pairs) == len(pairs)

    for feature in (
        "apps",
        "code_mode_host",
        "plugins",
        "skill_search",
    ):
        assert ("--enable", feature) in pairs
        assert ("--disable", feature) not in pairs
    for feature in (
        "browser_use",
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
            return b"codex-cli 0.150.0-alpha.8\n", b""

    async def create_process(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    environment = {"PATH": "/usr/bin:/bin"}

    assert await _verify_codex_version(
        "/resolved/codex",
        expected_prefix="0.150.",
        environment=environment,
    ) == "0.150.0-alpha.8"
    with pytest.raises(AgentUnavailableError, match="supported prefix"):
        await _verify_codex_version(
            "/resolved/codex",
            expected_prefix="0.148.",
            environment=environment,
        )


@pytest.mark.asyncio
async def test_codex_auto_version_policy_requires_protocol_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"codex-cli 0.151.0\n", b""

    async def create_process(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return Process()

    schema_check = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._verify_codex_protocol_schema",
        schema_check,
    )
    environment = {"PATH": "/usr/bin:/bin"}

    assert await _verify_codex_version(
        "/resolved/codex",
        expected_prefix="auto",
        environment=environment,
    ) == "0.151.0"
    schema_check.assert_awaited_once_with(
        "/resolved/codex",
        environment=environment,
    )


@pytest.mark.asyncio
async def test_codex_runtime_configuration_probe_uses_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Process:
        returncode: int | None = None
        stderr = SimpleNamespace(read=AsyncMock(return_value=b""))

        async def wait(self) -> int:
            if self.returncode is None:
                await asyncio.Future()
            assert self.returncode is not None
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    async def create_process(*args: object, **kwargs: object) -> Process:
        created.append((args, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._CODEX_RUNTIME_CONFIG_PROBE_SECONDS",
        0.01,
    )
    runtime_arguments = ("-c", "mcp_servers.cua_repl.enabled=false")

    await _verify_codex_runtime_configuration(
        "/resolved/codex",
        environment={"PATH": "/usr/bin:/bin"},
        runtime_arguments=runtime_arguments,
    )

    assert created[0][0] == (
        "/resolved/codex",
        "app-server",
        "--strict-config",
        "--listen",
        "stdio://",
        *runtime_arguments,
    )


@pytest.mark.asyncio
async def test_codex_runtime_configuration_probe_rejects_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode = 1
        stderr = SimpleNamespace(read=AsyncMock(return_value=b"invalid transport"))

        def kill(self) -> None:
            raise AssertionError("completed process must not be killed")

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*args: object, **kwargs: object) -> Process:
        del args, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(AgentUnavailableError, match="configuration is incompatible"):
        await _verify_codex_runtime_configuration(
            "/resolved/codex",
            environment={"PATH": "/usr/bin:/bin"},
            runtime_arguments=("-c", "mcp_servers.cua_repl.enabled=false"),
        )


def test_codex_runtime_arguments_have_no_duplicate_overrides(tmp_path: Path) -> None:
    arguments = _codex_runtime_arguments(
        codex_home=tmp_path,
        allow_image_generation=False,
    )
    config_arguments = arguments[arguments.index("-c") :]

    _config_overrides(config_arguments)


@pytest.mark.asyncio
async def test_provider_restarts_idle_app_server_after_runtime_config_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _provider(tmp_path)
    old_process = SimpleNamespace(returncode=None, pid=10)
    new_process = SimpleNamespace(
        returncode=None,
        pid=20,
        stdin=None,
        stdout=None,
        stderr=None,
    )
    provider._process = old_process
    provider._runtime_arguments = ("old",)

    async def close_unlocked() -> None:
        provider._process = None
        provider._runtime_arguments = None

    close = AsyncMock(side_effect=close_unlocked)
    verify_version = AsyncMock(return_value="0.152.1")
    verify_runtime = AsyncMock()
    request = AsyncMock(return_value={})
    notify = AsyncMock()
    create_process = AsyncMock(return_value=new_process)
    monkeypatch.setattr(provider, "_close_unlocked", close)
    monkeypatch.setattr(provider, "_request", request)
    monkeypatch.setattr(provider, "_notify", notify)
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._codex_app_server_environment",
        lambda: {"PATH": "/usr/bin:/bin", "CODEX_HOME": str(tmp_path)},
    )
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._codex_runtime_arguments",
        lambda **_kwargs: ("new",),
    )
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._resolve_executable",
        lambda _value: "/resolved/codex",
    )
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._verify_codex_version",
        verify_version,
    )
    monkeypatch.setattr(
        "simajilord.agent.providers.codex._verify_codex_runtime_configuration",
        verify_runtime,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    await provider._ensure_started()

    close.assert_awaited_once()
    verify_runtime.assert_awaited_once_with(
        "/resolved/codex",
        environment={
            "PATH": "/usr/bin:/bin",
            "CODEX_HOME": str(tmp_path),
            "RUST_LOG": "info",
        },
        runtime_arguments=("new",),
    )
    assert provider._process is new_process
    assert provider._runtime_arguments == ("new",)
    request.assert_awaited_once()
    notify.assert_awaited_once_with("initialized")


def test_codex_protocol_schema_contract_accepts_complete_surface(tmp_path: Path) -> None:
    _write_minimal_codex_protocol_schema(tmp_path)

    _validate_codex_protocol_schema(tmp_path)


def test_codex_protocol_schema_contract_rejects_missing_method(tmp_path: Path) -> None:
    _write_minimal_codex_protocol_schema(tmp_path)
    path = tmp_path / "ClientRequest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["oneOf"] = [
        branch
        for branch in document["oneOf"]
        if branch["properties"]["method"]["enum"] != ["turn/steer"]
    ]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AgentUnavailableError, match="protocol schema is incompatible"):
        _validate_codex_protocol_schema(tmp_path)


def test_codex_protocol_schema_contract_rejects_new_required_parameter(
    tmp_path: Path,
) -> None:
    _write_minimal_codex_protocol_schema(tmp_path)
    path = tmp_path / "ClientRequest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    initialize = next(
        branch
        for branch in document["oneOf"]
        if branch["properties"]["method"]["enum"] == ["initialize"]
    )
    params = initialize["properties"]["params"]
    params["properties"]["futureRequired"] = {}
    params["required"] = ["clientInfo", "futureRequired"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AgentUnavailableError, match="protocol schema is incompatible"):
        _validate_codex_protocol_schema(tmp_path)


def test_codex_protocol_schema_contract_rejects_removed_runtime_value(
    tmp_path: Path,
) -> None:
    _write_minimal_codex_protocol_schema(tmp_path)
    path = tmp_path / "ClientRequest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    thread_start = next(
        branch
        for branch in document["oneOf"]
        if branch["properties"]["method"]["enum"] == ["thread/start"]
    )
    thread_start["properties"]["params"]["properties"]["sandbox"] = {
        "enum": ["workspace-write"]
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AgentUnavailableError, match="protocol schema is incompatible"):
        _validate_codex_protocol_schema(tmp_path)


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
