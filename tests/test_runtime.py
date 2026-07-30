from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import cast

import discord
import pytest

from simajilord.agent import (
    AGENT_AUDIO_GRANT,
    AGENT_AUDIO_WRITE_CAPABILITIES,
    AGENT_COMPUTE_GRANT,
    AGENT_FILE_GRANT,
    AGENT_HIVE_GRANT,
    AGENT_MEDIA_GRANT,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_REQUESTED_WRITE_CAPABILITIES,
    AGENT_WEB_GRANT,
    NON_UNDOABLE_ACTION_CAPABILITIES,
    action_policy,
)
from simajilord.agent.providers import CodexAppServerProvider
from simajilord.config import load_settings
from simajilord.core import InvocationContext, RiskLevel
from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.providers.image import SharedCodexImageProvider
from simajilord.runtime import SimajilordRuntime


def test_runtime_composes_before_discord_starts_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "false")

    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)

    capability_names = tuple(
        endpoint.descriptor.name for endpoint in runtime.registry.all()
    )
    assert len(capability_names) == len(set(capability_names))
    assert {
        "web.search",
        "web.fetch",
        "web.find",
        "web.status",
        "moderation.detect_synthetic_media",
        "moderation.status",
        "image.generate",
        "image.status",
        "memory.search",
        "memory.remember",
        "memory.update",
        "memory.forget",
    } <= set(capability_names)
    asyncio.run(runtime.close())


def test_agent_and_image_queue_share_the_primary_codex_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "everyone")

    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)

    try:
        assert runtime.agent is not None
        assert isinstance(runtime.agent.provider, CodexAppServerProvider)
        assert isinstance(runtime.image.provider, SharedCodexImageProvider)
        assert runtime.image.provider._provider is runtime.agent.provider
        assert runtime.agent.provider.allow_image_generation is True
        assert runtime.agent.provider.image_timeout_seconds == (
            settings.image_timeout_seconds
        )
    finally:
        asyncio.run(runtime.close())


def test_complete_core_capability_catalog_uses_english_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "false")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    japanese = re.compile(r"[ぁ-んァ-ヶ一-龯]")

    try:
        for endpoint in runtime.registry.all():
            descriptor = endpoint.descriptor
            assert not japanese.search(descriptor.summary), descriptor.name
            for side_effect in descriptor.side_effects:
                assert not japanese.search(side_effect), descriptor.name
    finally:
        asyncio.run(runtime.close())


def test_every_current_mutation_has_an_explicit_undo_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "false")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    endpoints = (
        *runtime.registry.all(),
        *build_discord_endpoints(cast(discord.Client, object()), runtime),
    )
    mutations = {
        item.descriptor.name
        for item in endpoints
        if item.descriptor.risk is RiskLevel.WRITE
        or item.descriptor.idempotency != "read"
    }

    for capability in mutations:
        policy = action_policy(capability)
        assert (
            capability in NON_UNDOABLE_ACTION_CAPABILITIES
            or policy.undo_capability is not None
        ), capability
    asyncio.run(runtime.close())


def test_agent_discovers_only_permission_guarded_audio_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    autonomous_context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        grants=frozenset({AGENT_AUDIO_GRANT}),
    )
    requested_context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        grants=frozenset({AGENT_AUDIO_GRANT}),
        approvals=frozenset(AGENT_AUDIO_WRITE_CAPABILITIES),
    )

    async def run() -> None:
        assert provider.tools.dynamic_specs(autonomous_context)
        autonomous = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "music voice play speak", "limit": 5},
            context=autonomous_context,
            max_output_characters=10_000,
        )
        assert "discord.play_audio" not in autonomous.text
        assert "discord.speak" not in autonomous.text
        assert "audio.search" in autonomous.text
        output = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "music voice play speak", "limit": 5},
            context=requested_context,
            max_output_characters=10_000,
        )
        assert "discord.play_audio" in output.text
        assert "discord.speak" in output.text
        assert '"name":"audio.play"' not in output.text
        assert '"name":"speech.speak"' not in output.text
        playback_controls = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "pause move music queue", "limit": 5},
            context=requested_context,
            max_output_characters=10_000,
        )
        assert "discord.pause_audio" in playback_controls.text
        assert "discord.move_audio" in playback_controls.text
        assert "discord.control_audio" not in playback_controls.text
        queue_controls = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "clear my music volume", "limit": 5},
            context=requested_context,
            max_output_characters=10_000,
        )
        assert "discord.clear_my_audio" in queue_controls.text
        assert "discord.set_audio_volume" in queue_controls.text
        assert "discord.control_audio" not in queue_controls.text
        deprecated_mix = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "fresh mix draft preview", "limit": 5},
            context=requested_context,
            max_output_characters=10_000,
        )
        assert "fresh_mix" not in deprecated_mix.text
        autonomous_read_aloud = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "read aloud dictionary settings", "limit": 5},
            context=autonomous_context,
            max_output_characters=10_000,
        )
        assert "discord.read_aloud_dictionary_set" not in autonomous_read_aloud.text
        assert (
            "discord.read_aloud_dictionary_list" in autonomous_read_aloud.text
            or "discord.read_aloud_policy_status" in autonomous_read_aloud.text
        )
        requested_read_aloud = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "read aloud dictionary register", "limit": 5},
            context=requested_context,
            max_output_characters=10_000,
        )
        assert "discord.read_aloud_dictionary_set" in requested_read_aloud.text
        assert "discord.manage_read_aloud" not in requested_read_aloud.text
        media_output = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "Discord custom emoji sticker animation frame", "limit": 5},
            context=autonomous_context,
            max_output_characters=10_000,
        )
        assert "discord.view_custom_emoji" in media_output.text
        assert "discord.view_sticker" in media_output.text
        assert provider.tools.dynamic_specs(requested_context)
        await runtime.close()

    asyncio.run(run())


def test_agent_web_grant_exposes_local_search_fetch_and_find(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "everyone")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    denied = InvocationContext("7", "1", "agent", "denied")
    granted = InvocationContext(
        "7",
        "1",
        "agent",
        "granted",
        grants=frozenset({AGENT_WEB_GRANT}),
    )

    denied_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(denied)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }
    granted_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(granted)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }

    assert {"web_search", "web_fetch", "web_find"}.isdisjoint(denied_names)
    assert {"web_search", "web_fetch", "web_find"} <= granted_names
    asyncio.run(runtime.close())


def test_agent_file_grant_exposes_complete_attachment_read_and_delivery_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    denied = InvocationContext("7", "1", "agent", "denied")
    granted = InvocationContext(
        "7",
        "1",
        "agent",
        "granted",
        grants=frozenset({AGENT_FILE_GRANT}),
    )

    denied_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(denied)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }
    granted_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(granted)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }

    assert {
        "files_list",
        "files_read",
        "discord_import_attachment",
        "discord_send_file",
    }.isdisjoint(denied_names)
    assert {
        "files_list",
        "files_read",
        "discord_import_attachment",
        "discord_send_file",
    } <= granted_names
    asyncio.run(runtime.close())


def test_agent_safe_compute_access_exposes_only_isolated_workspace_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("AGENT_SAFE_COMPUTE_ACCESS", "everyone")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    assert runtime.compute is not None
    assert runtime.compute.web_fetcher is runtime.web.page_fetcher
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    denied = InvocationContext("7", "1", "agent", "denied")
    granted = InvocationContext(
        "7",
        "1",
        "agent",
        "granted",
        grants=frozenset(
            {
                AGENT_COMPUTE_GRANT,
                AGENT_FILE_GRANT,
            }
        ),
    )

    denied_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(denied)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }
    granted_names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(granted)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }

    assert {"compute_run", "files_download_url"}.isdisjoint(denied_names)
    assert {"compute_run", "files_download_url"} <= granted_names
    assert "shell_run" not in granted_names
    assert "host_files_read" not in granted_names
    asyncio.run(runtime.close())


def test_agent_can_view_an_image_attachment_without_enabling_file_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "false")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    context = InvocationContext(
        "7",
        "1",
        "agent",
        "image",
        grants=frozenset({AGENT_MESSAGE_GRANT}),
    )
    names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(context)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }

    assert "discord_view_image_attachment" in names
    assert "discord_import_attachment" not in names
    assert "files_read" not in names
    asyncio.run(runtime.close())


def test_agent_message_grant_exposes_restart_safe_undo_and_own_message_delete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    assert runtime.action_receipts is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    context = InvocationContext(
        "7",
        "1",
        "agent",
        "undo",
        grants=frozenset({AGENT_MESSAGE_GRANT}),
    )
    names = {
        str(tool["name"])
        for namespace in provider.tools.dynamic_specs(context)
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }

    assert {"action_undo", "discord_delete_own_message"} <= names
    assert runtime.registry.endpoint("timer.restore").descriptor.approval.value == "always"
    asyncio.run(runtime.close())


def test_agent_can_discover_existing_server_poll_voice_and_utility_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    context = InvocationContext(
        "7",
        "1",
        "agent",
        "discovery",
        grants=frozenset({AGENT_AUDIO_GRANT, AGENT_MESSAGE_GRANT}),
        approvals=frozenset(AGENT_REQUESTED_WRITE_CAPABILITIES),
    )

    async def run() -> None:
        server = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "サーバー メンバー 役職", "limit": 5},
            context=context,
            max_output_characters=10_000,
        )
        assert "discord.inspect_server" in server.text
        assert "discord.inspect_user" in server.text
        poll = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "投票 アンケート", "limit": 5},
            context=context,
            max_output_characters=10_000,
        )
        assert "discord.create_poll" in poll.text
        voice = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "ボイス 通話 参加", "limit": 5},
            context=context,
            max_output_characters=10_000,
        )
        assert "discord.connect_voice" in voice.text
        utility = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "random choose dice", "limit": 5},
            context=context,
            max_output_characters=10_000,
        )
        assert "utility.choose" in utility.text
        assert "utility.roll" in utility.text
        await runtime.close()

    asyncio.run(run())


def test_natural_japanese_search_discovers_non_eager_agent_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("AGENT_FILE_SANDBOX_ENABLED", "true")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)
    approved = InvocationContext(
        "7",
        "1",
        "agent",
        "approved",
        grants=frozenset(
            {
                AGENT_AUDIO_GRANT,
                AGENT_FILE_GRANT,
                AGENT_MEDIA_GRANT,
                AGENT_MESSAGE_GRANT,
                AGENT_MODERATION_GRANT,
            }
        ),
        approvals=frozenset(AGENT_REQUESTED_WRITE_CAPABILITIES),
    )

    async def search(query: str, context: InvocationContext = approved) -> str:
        return (
            await provider.tools.invoke(
                namespace="simajilord",
                tool_name="capability_search",
                arguments={"query": query, "limit": 5},
                context=context,
                max_output_characters=12_000,
            )
        ).text

    async def run() -> None:
        expected_by_query = {
            "２５分後に集中タイマーをかけて": "timer.create",
            "さっきのタイマーを取り消して": "timer.cancel",
            "この議論を新しいスレッドに分けて": "discord.create_thread",
            "バグ報告としてフォーラムへ整理して": "discord.create_forum_post",
            "新しい役職を作って田中さんに付けて": "discord.assign_role",
            "荒らしを10分間発言できなくして": "discord.set_timeout",
            "音楽を一時停止して": "discord.pause_audio",
            "添付テキストのこの箇所を直して返して": "files.replace_text",
            "この動画をファイルに保存して送って": "media.save",
            "replace this text in the attached file": "files.replace_text",
            "create a thread for this discussion": "discord.create_thread",
        }
        for query, expected in expected_by_query.items():
            assert expected in await search(query), query

        no_moderation = InvocationContext(
            "7",
            "1",
            "agent",
            "no-moderation",
            grants=frozenset({AGENT_MESSAGE_GRANT}),
            approvals=frozenset(AGENT_REQUESTED_WRITE_CAPABILITIES),
        )
        grant_without_approval = InvocationContext(
            "7",
            "1",
            "agent",
            "no-approval",
            grants=frozenset({AGENT_MODERATION_GRANT}),
        )
        for context in (no_moderation, grant_without_approval):
            output = await search("荒らしを10分間発言できなくして", context)
            assert "discord.set_timeout" not in output
            assert "discord.ban_member" not in output
        await runtime.close()

    asyncio.run(run())


def test_agent_hive_attachment_analysis_requires_per_message_write_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DISCORD_TOKEN", "test-token")
    monkeypatch.setenv("DISCORD_APPLICATION_ID", "123")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_ENABLED", "true")
    monkeypatch.setenv("HIVE_API_KEY", "test-key")
    monkeypatch.setenv("IMAGE_GENERATION_ACCESS", "disabled")
    monkeypatch.setenv("AGENT_WEB_SEARCH_ACCESS", "disabled")
    settings = load_settings(dotenv_path=tmp_path / "missing.env")
    runtime = SimajilordRuntime.build(settings)
    for item in build_discord_endpoints(
        cast(discord.Client, object()),
        runtime,
    ):
        runtime.registry.register(item)
    assert runtime.agent is not None
    provider = cast(CodexAppServerProvider, runtime.agent.provider)

    assert provider.tools.write_capability_for_call(
        tool_name="capability_invoke",
        arguments={
            "name": "discord.analyze_attachment",
            "arguments": {},
        },
    ) == "discord.analyze_attachment"
    hive_context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        grants=frozenset({AGENT_HIVE_GRANT}),
        approvals=frozenset({"discord.analyze_attachment"}),
    )
    moderation_context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        grants=frozenset({AGENT_MODERATION_GRANT}),
        approvals=frozenset({"discord.analyze_attachment"}),
    )

    async def search(context: InvocationContext) -> str:
        output = await provider.tools.invoke(
            namespace="simajilord",
            tool_name="capability_search",
            arguments={"query": "analyze synthetic attachment", "limit": 5},
            context=context,
            max_output_characters=10_000,
        )
        return output.text

    assert "discord.analyze_attachment" in asyncio.run(search(hive_context))
    assert "discord.analyze_attachment" not in asyncio.run(
        search(moderation_context)
    )
    asyncio.run(runtime.close())
