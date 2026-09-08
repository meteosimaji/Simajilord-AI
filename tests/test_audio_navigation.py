from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.audio_hub import AudioHubView, audio_hub_embed
from simajilord.integrations.discord.audio_navigation import (
    AudioNavigateRequest,
    build_audio_navigation_endpoints,
)
from simajilord.runtime import SimajilordRuntime
from simajilord.services.read_aloud import ReadAloudMode, ReadAloudRoute, ReadAloudService


@pytest.mark.asyncio
async def test_voice_profiles_survive_switch_disable_and_reload(tmp_path: Path) -> None:
    service = ReadAloudService(tmp_path / "routes.json")
    first = ReadAloudRoute(
        "1", "11", "10", ReadAloudMode.QUEUE, additional_text_channel_ids=("12",)
    )
    second = ReadAloudRoute("1", "22", "20", ReadAloudMode.QUEUE)
    await service.configure(first)
    await service.configure(second)
    await service.configure(replace(second, enabled=False))
    restored = ReadAloudService(service.state_file)
    assert restored.get("1") is None
    assert restored.saved_route("1", "10") == first
    assert restored.saved_route("1", "20") == second


def navigation_fixture(tmp_path: Path):
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud = ReadAloudService(tmp_path / "routes.json")
    runtime.settings.read_aloud_audience_mode = "disabled"
    runtime.audio.follow_actors = {}
    session = Mock()
    session.destination_id = "10"
    session.speech_only = False
    session.output.connected = True
    session.can_start_for.return_value = True
    session.snapshot = AsyncMock(return_value=SimpleNamespace(speech_active=False, pending=()))
    session.suspend = AsyncMock()
    session.discard_relocation_speech = AsyncMock()
    runtime.audio.get_or_create.return_value = session
    runtime.audio.find.return_value = session
    runtime.audio.connect = AsyncMock()
    member = Mock(spec=discord.Member)
    member.id = 7
    member.bot = False
    member.guild_permissions = discord.Permissions.none()
    destination = Mock(spec=discord.VoiceChannel)
    destination.id = 20
    destination.members = [member]
    origin = Mock(spec=discord.VoiceChannel)
    origin.id = 10
    origin.members = []
    source = Mock(spec=discord.TextChannel)
    source.id = 11
    source.permissions_for.return_value = discord.Permissions(
        view_channel=True, read_message_history=True
    )
    member.voice = SimpleNamespace(channel=destination)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_member.return_value = member
    guild.get_channel.side_effect = {10: origin, 20: destination, 11: source}.get
    guild.get_channel_or_thread.side_effect = guild.get_channel.side_effect
    client = Mock(spec=discord.Client)
    client.get_guild.return_value = guild
    endpoint = build_audio_navigation_endpoints(client, runtime)[0]
    context = InvocationContext("7", "1", "discord", "test")
    return runtime, session, member, origin, endpoint, context


@pytest.mark.asyncio
async def test_move_preserves_reading_sources_and_music_mode(tmp_path: Path) -> None:
    runtime, session, _, _, endpoint, context = navigation_fixture(tmp_path)
    route = ReadAloudRoute("1", "11", "10", ReadAloudMode.QUEUE)
    await runtime.read_aloud.configure(route)
    result = await endpoint.invoke(AudioNavigateRequest("20", "10"), context)
    session.suspend.assert_awaited_once()
    session.discard_relocation_speech.assert_awaited_once()
    runtime.audio.connect.assert_awaited_once_with("1", "20", speech_only=False)
    assert result.reading_sources == ("11",)
    assert runtime.read_aloud.get("1") == replace(route, audio_destination_id="20")
    assert runtime.read_aloud.saved_route("1", "10") == route


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmed, permitted, code",
    [
        (False, False, "audio.move_confirmation_required"),
        (True, False, "audio.move_occupied_forbidden"),
        (True, True, None),
    ],
)
async def test_occupied_move_requires_confirmation_and_permission(
    tmp_path: Path,
    confirmed: bool,
    permitted: bool,
    code: str | None,
) -> None:
    runtime, session, member, origin, endpoint, context = navigation_fixture(tmp_path)
    origin.members = [SimpleNamespace(id=8, bot=False)]
    member.guild_permissions = discord.Permissions(move_members=permitted)
    request = AudioNavigateRequest("20", "10", confirm_occupied=confirmed)
    if code:
        with pytest.raises(UserError, match=code):
            await endpoint.invoke(request, context)
        runtime.audio.connect.assert_not_awaited()
        session.suspend.assert_not_awaited()
    else:
        await endpoint.invoke(request, context)
        runtime.audio.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_panel_and_pending_speech_do_not_move(tmp_path: Path) -> None:
    runtime, session, _, _, endpoint, context = navigation_fixture(tmp_path)
    with pytest.raises(UserError, match=r"audio\.navigation_changed"):
        await endpoint.invoke(AudioNavigateRequest("20", "30"), context)
    session.snapshot.return_value.speech_active = True
    with pytest.raises(UserError, match=r"audio\.move_speech_busy"):
        await endpoint.invoke(AudioNavigateRequest("20", "10"), context)
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_move_restores_active_route(tmp_path: Path) -> None:
    runtime, _, _, _, endpoint, context = navigation_fixture(tmp_path)
    route = ReadAloudRoute("1", "11", "10", ReadAloudMode.QUEUE)
    await runtime.read_aloud.configure(route)
    runtime.audio.connect.side_effect = UserError("audio.connect_failed")
    with pytest.raises(UserError, match=r"audio\.connect_failed"):
        await endpoint.invoke(AudioNavigateRequest("20", "10"), context)
    assert runtime.read_aloud.get("1") == route


@pytest.mark.asyncio
async def test_follow_requires_live_same_destination_and_owner(tmp_path: Path) -> None:
    runtime, session, _, _, endpoint, context = navigation_fixture(tmp_path)
    with pytest.raises(UserError, match=r"audio\.follow_start_required"):
        await endpoint.invoke(AudioNavigateRequest("20", "10", "follow"), context)
    session.destination_id = "20"
    await endpoint.invoke(AudioNavigateRequest("20", "20", "follow"), context)
    assert runtime.audio.follow_actors == {"1": "7"}
    runtime.audio.follow_actors["1"] = "8"
    with pytest.raises(UserError, match=r"audio\.follow_owned"):
        await endpoint.invoke(AudioNavigateRequest("20", "20", "unfollow"), context)
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_hub_without_vc_exposes_settings_without_connecting(tmp_path: Path) -> None:
    runtime, _, _, _, _, _ = navigation_fixture(tmp_path)
    view = AudioHubView(
        cast(SimajilordRuntime, runtime),
        requester_id=7,
        workspace="1",
        destination=None,
        source_id="11",
    )
    assert view.speech.disabled and view.both.disabled and view.move.disabled
    assert view.follow.disabled and view.preview.disabled
    assert any(isinstance(item, discord.ui.Select) for item in view.children)
    assert "VCに参加" in (audio_hub_embed(runtime, "1", None).description or "")
    runtime.audio.connect.assert_not_awaited()
    view.to_components()


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["personal", "shared", "sources"])
async def test_separate_settings_fit_discord_rows(tmp_path: Path, section: str) -> None:
    from simajilord.integrations.discord.cogs import ReadAloudChannelSelectView

    runtime, _, _, _, _, _ = navigation_fixture(tmp_path)
    view = ReadAloudChannelSelectView(
        runtime, requester_id=7, destination_id=20, default_values=(), can_manage_semantics=False
    )
    view.configure_section(section)
    rows = view.to_components()
    assert len(rows) <= 5
    assert all(
        sum(1 if item["type"] == 2 else 5 for item in row["components"]) <= 5 for row in rows
    )
    assert view.setup_embed().title
    if section == "shared":
        assert view.length_selector.disabled and view.behavior_selector.disabled
    if section == "personal":
        assert not view.voice_selector.disabled
        assert view.selector not in view.children


@pytest.mark.asyncio
async def test_first_start_failure_does_not_enable_unconnected_route(tmp_path: Path) -> None:
    runtime, session, _, _, endpoint, context = navigation_fixture(tmp_path)
    session.destination_id = None
    session.output.connected = False
    runtime.audio.connect.side_effect = UserError("audio.connect_failed")
    with pytest.raises(UserError, match=r"audio\.connect_failed"):
        await endpoint.invoke(AudioNavigateRequest("20", None, "speech", "11"), context)
    assert runtime.read_aloud.get("1") is None
    assert runtime.read_aloud.saved_route("1", "20") is not None


@pytest.mark.asyncio
async def test_waiting_music_cannot_be_started_by_another_actor(tmp_path: Path) -> None:
    runtime, session, _, _, endpoint, context = navigation_fixture(tmp_path)
    session.output.connected = False
    session.can_start_for.return_value = False
    with pytest.raises(UserError, match=r"audio\.waiting_queue_restricted"):
        await endpoint.invoke(AudioNavigateRequest("20", "10", "both", "11"), context)
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_maps_vc_conversation_and_prefers_saved_target_profile(tmp_path: Path) -> None:
    service = ReadAloudService(tmp_path / "routes.json")
    first = ReadAloudRoute(
        "1", "10", "10", ReadAloudMode.QUEUE, additional_text_channel_ids=("11",)
    )
    await service.configure(first)
    preview = service.resume_route("1", "20")
    assert preview is not None
    assert preview.text_channel_ids == ("20", "11")
    assert preview.audio_destination_id == "20"
    assert service.get("1") == first
    second = ReadAloudRoute("1", "22", "20", ReadAloudMode.QUEUE)
    await service.configure(second)
    await service.configure(first)
    assert service.resume_route("1", "20") == second


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["preserve", "music"])
async def test_move_or_music_start_does_not_reenable_disabled_reading(tmp_path: Path, mode) -> None:
    runtime, _, _, _, endpoint, context = navigation_fixture(tmp_path)
    route = ReadAloudRoute("1", "11", "20", ReadAloudMode.QUEUE)
    await runtime.read_aloud.configure(route)
    await runtime.read_aloud.configure(replace(route, enabled=False))
    await endpoint.invoke(AudioNavigateRequest("20", "10", mode), context)
    assert runtime.read_aloud.get("1") is None
    runtime.audio.connect.assert_awaited_once_with("1", "20", speech_only=False)
