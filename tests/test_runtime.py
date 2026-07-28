from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import discord
import pytest

from simajilord.agent import AGENT_AUDIO_GRANT, AGENT_AUDIO_WRITE_CAPABILITIES
from simajilord.agent.providers import CodexAppServerProvider
from simajilord.config import load_settings
from simajilord.core import InvocationContext
from simajilord.integrations.discord.capabilities import build_discord_endpoints
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
    } <= set(capability_names)
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
