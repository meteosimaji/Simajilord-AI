from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord import app_commands
from discord.ext import commands

from simajilord.agent import AGENT_WEB_GRANT, AgentRateLimitError
from simajilord.capabilities.audio import (
    AudioPlayRequest,
    AudioPlayResponse,
    AudioSearchItem,
    AudioSearchReason,
    AudioSearchResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudRequest,
    ReadAloudResponse,
)
from simajilord.capabilities.web import WebFetchResponse
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.capabilities import (
    _assert_agent_channel_scope,
    _assert_agent_update_scope,
    _bounded_event_message,
    _discord_event_message_id,
    _message_preview,
    agent_readable_channel_ids,
    build_discord_endpoints,
)
from simajilord.integrations.discord.cogs import (
    ModerationCog,
    MusicCog,
    MusicControlsView,
    MusicSearchChoiceView,
    ReadAloudChannelSelect,
    ReadAloudCog,
    WebCog,
    WebFetchContinueView,
    _agent_error_text,
    _agent_message_groups,
    _discord_message_chunks,
    _retry_after_text,
    discord_conversation_id,
)
from simajilord.runtime import SimajilordRuntime


def test_common_music_actions_have_short_top_level_commands() -> None:
    commands = {
        command.name: command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert {"play", "queue", "history"} <= commands.keys()
    assert commands["play"].description == "Play a public media URL or search for a track."
    assert commands["queue"].description == "Show what is playing and what comes next."
    assert commands["history"].description == (
        "Show recently played tracks and who requested them."
    )


@pytest.mark.asyncio
async def test_agent_audio_adapter_rebinds_requester_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(
        return_value=AudioPlayResponse(
            title="Track",
            page_url="https://example.com/track",
            queue_position=1,
            duration_seconds=120,
            destination_id=None,
            playback_state="waiting_for_voice",
            requested_by_name="Real member",
        )
    )
    session = Mock()
    session.output.connected = False
    runtime.audio.get_or_create.return_value = session
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    member.display_name = "Real member"
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        lambda selected_guild, context: member,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._member_voice_channel",
        lambda selected_member: None,
    )
    endpoint_by_name = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
    )

    response = await endpoint_by_name["discord.play_audio"].invoke(
        AudioPlayRequest(
            reference="https://example.com/track",
            requested_by_name="Forged name",
        ),
        context,
    )

    assert response.requested_by_name == "Real member"
    delegated_request = runtime.registry.invoke.await_args.args[1]
    assert isinstance(delegated_request, AudioPlayRequest)
    assert delegated_request.requested_by_name == "Real member"


@pytest.mark.asyncio
async def test_agent_audio_adapter_rejects_remote_voice_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    session = Mock()
    session.output.connected = True
    session.destination_id = "55"
    runtime.audio.get_or_create.return_value = session
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        lambda selected_guild, context: member,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._member_voice_channel",
        lambda selected_member: None,
    )
    endpoint_by_name = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
    )

    with pytest.raises(UserError, match=r"audio\.same_voice_required"):
        await endpoint_by_name["discord.play_audio"].invoke(
            AudioPlayRequest(reference="https://example.com/track"),
            context,
        )
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_read_aloud_mutation_requires_manage_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    member.guild_permissions.manage_guild = False
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        lambda selected_guild, context: member,
    )
    endpoint_by_name = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }

    with pytest.raises(UserError, match=r"discord\.manage_guild_required"):
        await endpoint_by_name["discord.manage_read_aloud"].invoke(
            ReadAloudRequest(action=ReadAloudAction.DISABLE),
            InvocationContext(
                actor_id="7",
                workspace_id="1",
                transport="agent",
                request_id="event",
            ),
        )
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_join_self_service_is_limited_to_current_channel_and_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ReadAloudResponse(
        action=ReadAloudAction.ADD_SOURCE.value,
        enabled=True,
        text_channel_id="50",
        text_channel_ids=("50",),
        audio_destination_id="55",
        mode="queue",
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=response)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    source = Mock(spec=discord.TextChannel)
    source.id = 50
    voice = Mock(spec=discord.VoiceChannel)
    voice.id = 55
    guild.get_channel_or_thread.return_value = source
    guild.get_channel.return_value = voice
    member = Mock(spec=discord.Member)
    member.guild_permissions.manage_guild = False
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        lambda selected_guild, context: member,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._member_voice_channel",
        lambda selected_member: voice,
    )
    endpoint_by_name = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }
    request = ReadAloudRequest(
        action=ReadAloudAction.ADD_SOURCE,
        text_channel_id="50",
        audio_destination_id="55",
    )

    result = await endpoint_by_name["discord.manage_read_aloud"].invoke(
        request,
        InvocationContext(
            actor_id="7",
            workspace_id="1",
            transport="discord",
            request_id="event",
            origin_resource_id="50",
        ),
    )

    assert result == response
    runtime.registry.invoke.assert_awaited_once()
    delegated_name, delegated_request, delegated_context = (
        runtime.registry.invoke.await_args.args
    )
    assert delegated_name == "speech.manage_read_aloud"
    assert delegated_request == request
    assert delegated_context.origin_resource_id == "50"

    runtime.registry.invoke.reset_mock()
    with pytest.raises(UserError, match=r"discord\.manage_guild_required"):
        await endpoint_by_name["discord.manage_read_aloud"].invoke(
            request,
            InvocationContext(
                actor_id="7",
                workspace_id="1",
                transport="discord",
                request_id="other-event",
                origin_resource_id="51",
            ),
        )
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_multi_channel_join_is_atomic_and_limited_to_current_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ReadAloudResponse(
        action=ReadAloudAction.ADD_SOURCES.value,
        enabled=True,
        text_channel_id="50",
        text_channel_ids=("50", "51"),
        audio_destination_id="55",
        mode="queue",
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=response)
    runtime.read_aloud.get.return_value = None
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    source_one = Mock(spec=discord.TextChannel)
    source_one.id = 50
    source_two = Mock(spec=discord.VoiceChannel)
    source_two.id = 51
    voice = Mock(spec=discord.VoiceChannel)
    voice.id = 55
    bot_member = Mock(spec=discord.Member)
    member = Mock(spec=discord.Member)
    member.guild_permissions.manage_guild = False
    guild.me = bot_member
    guild.get_channel_or_thread.side_effect = lambda channel_id: {
        50: source_one,
        51: source_two,
    }.get(channel_id)
    guild.get_channel.return_value = voice
    readable_text = discord.Permissions(
        view_channel=True,
        read_message_history=True,
    )
    readable_voice = discord.Permissions(
        view_channel=True,
        read_message_history=True,
        connect=True,
    )
    source_one.permissions_for.return_value = readable_text
    source_two.permissions_for.return_value = readable_voice
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        lambda selected_guild, context: member,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._member_voice_channel",
        lambda selected_member: voice,
    )
    endpoint_by_name = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }
    request = ReadAloudRequest(
        action=ReadAloudAction.ADD_SOURCES,
        text_channel_ids=("50", "51"),
        audio_destination_id="55",
    )

    result = await endpoint_by_name["discord.manage_read_aloud"].invoke(
        request,
        InvocationContext(
            actor_id="7",
            workspace_id="1",
            transport="discord",
            request_id="event",
            origin_resource_id="50",
        ),
    )

    assert result == response
    delegated_request = runtime.registry.invoke.await_args.args[1]
    assert delegated_request == request


def test_advanced_music_group_keeps_compatible_and_power_commands() -> None:
    group = next(
        command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Group) and command.name == "music"
    )
    names = {command.name for command in group.commands}
    assert names == {
        "play",
        "queue",
        "history",
        "pause",
        "resume",
        "skip",
        "stop",
        "leave",
        "loop",
        "remove",
        "autoleave",
        "shuffle",
        "seek",
        "tune",
    }


@pytest.mark.asyncio
async def test_music_buttons_are_concise_grouped_and_uniquely_addressable() -> None:
    view = MusicControlsView(cast(SimajilordRuntime, object()))
    buttons = [
        child for child in view.children if isinstance(child, discord.ui.Button)
    ]
    assert [button.label for button in buttons] == [
        "Start in VC",
        "Pause",
        "Resume",
        "Skip",
        "Loop",
        "Leave",
    ]
    assert sum(button.row == 0 for button in buttons) == 5
    assert sum(button.row == 1 for button in buttons) == 1
    custom_ids = [button.custom_id for button in buttons]
    assert None not in custom_ids
    assert len(custom_ids) == len(set(custom_ids))


def test_ambiguous_results_are_direct_one_click_buttons() -> None:
    response = AudioSearchResponse(
        query="Same",
        candidates=tuple(
            AudioSearchItem(
                reference=f"https://example.com/{index}",
                title=f"Artist {index} - Same",
                duration_seconds=180,
                uploader=f"Artist {index}",
            )
            for index in range(1, 4)
        ),
        selected_index=None,
        selection_required=True,
        reason=AudioSearchReason.AMBIGUOUS_TITLE,
    )
    view = MusicSearchChoiceView(
        cast(commands.Bot, object()),
        cast(SimajilordRuntime, object()),
        response,
        requester_id=1,
        requester_name="Listener",
    )
    buttons = [
        child for child in view.children if isinstance(child, discord.ui.Button)
    ]
    assert [button.label for button in buttons] == [
        "1 · Artist 1",
        "2 · Artist 2",
        "3 · Artist 3",
    ]
    assert all(button.row == 0 for button in buttons)
    custom_ids = [button.custom_id for button in buttons]
    assert None not in custom_ids
    assert len(custom_ids) == len(set(custom_ids))


def test_agent_conversation_key_is_shared_per_channel() -> None:
    assert discord_conversation_id(guild_id=1, channel_id=2) == (
        "discord:guild:1:channel:2"
    )
    assert discord_conversation_id(guild_id=None, channel_id=2) == (
        "discord:direct:channel:2"
    )
    assert discord_conversation_id(
        guild_id=1,
        channel_id=2,
        grants=frozenset({AGENT_WEB_GRANT}),
    ) == "discord:guild:1:channel:2:profile:web"


def test_web_commands_are_short_direct_paths() -> None:
    commands = {
        command.name: command
        for command in WebCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"search", "fetch", "find"}
    assert commands["search"].description.startswith("Search the web")


def test_read_aloud_has_zero_argument_join_entrypoint() -> None:
    commands = {
        command.name: command
        for command in ReadAloudCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"join"}
    assert commands["join"].description == (
        "Read this channel aloud in the voice channel you joined."
    )
    assert commands["join"].parameters == []


def test_join_channel_selector_supports_one_to_twenty_five_conversations() -> None:
    selector = ReadAloudChannelSelect(
        cast(SimajilordRuntime, object()),
        requester_id=7,
        destination_id=55,
        default_values=(),
    )

    assert selector.custom_id == "simajilord:readaloud:channels"
    assert selector.min_values == 1
    assert selector.max_values == 25
    assert discord.ChannelType.text in selector.channel_types
    assert discord.ChannelType.voice in selector.channel_types
    assert discord.ChannelType.public_thread in selector.channel_types


def test_hive_analysis_is_one_direct_attachment_command() -> None:
    commands = {
        command.name: command
        for command in ModerationCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"detectai"}
    assert commands["detectai"].description == (
        "Analyze an image or video with HIVE AI-content detection."
    )


def test_web_fetch_continuation_is_one_click_and_uniquely_addressable() -> None:
    view = WebFetchContinueView(
        cast(SimajilordRuntime, object()),
        WebFetchResponse(
            title="Long page",
            url="https://example.com/page",
            content_type="text/html",
            text="first chunk",
            offset=0,
            total_characters=8_000,
            next_offset=3_500,
            links=(),
        ),
    )
    buttons = [
        child for child in view.children if isinstance(child, discord.ui.Button)
    ]
    assert [button.label for button in buttons] == ["Continue"]
    assert buttons[0].custom_id == "simajilord:web:fetch:continue"
    assert view.next_offset == 3_500


def test_message_index_preview_uses_full_short_or_25_plus_5() -> None:
    assert _message_preview("x" * 30) == ("x" * 30, False)
    preview, truncated = _message_preview("abcdefghijklmnopqrstuvwxyz123456789")
    assert preview == "abcdefghijklmnopqrstuvwxy…56789"
    assert truncated is True


def test_trigger_message_is_bounded_but_not_head_tail_summarized() -> None:
    content = "x" * 1_500
    preview, truncated = _bounded_event_message(content)
    assert preview == "x" * 1_000
    assert truncated is True
    assert _bounded_event_message("latest message") == ("latest message", False)


def test_trigger_message_id_comes_only_from_discord_request_id() -> None:
    context = InvocationContext(
        actor_id="actor",
        workspace_id="workspace",
        transport="agent",
        request_id="discord:message:1530953392980234250",
    )
    assert _discord_event_message_id(context) == "1530953392980234250"
    assert (
        _discord_event_message_id(
            InvocationContext(
                actor_id="actor",
                workspace_id="workspace",
                transport="agent",
                request_id="forged:1530953392980234250",
            )
        )
        is None
    )


def test_agent_response_chunks_prefer_readable_boundaries() -> None:
    chunks = _discord_message_chunks("alpha beta gamma", maximum=10)
    assert chunks == ("alpha beta", "gamma")
    assert "".join(chunks).replace(" ", "") == "alphabetagamma"


def test_agent_can_request_separate_discord_messages_without_another_turn() -> None:
    content = (
        "こんにちは\n"
        "<simajilord:message-break>\n"
        "こんばんは"
    )
    assert _agent_message_groups(content) == ("こんにちは", "こんばんは")
    assert _agent_message_groups("first\n\nsecond") == ("first\n\nsecond",)


def test_agent_message_breaks_have_no_artificial_post_count_limit() -> None:
    content = "<simajilord:message-break>".join(str(index) for index in range(8))
    messages = _agent_message_groups(content)
    assert messages == tuple(str(index) for index in range(8))


def test_agent_rate_limit_message_includes_exact_retry_time() -> None:
    error = AgentRateLimitError(
        "limited",
        retry_after_seconds=125,
    )
    assert _agent_error_text(error).endswith("Try again in 2m 5s.")
    assert _retry_after_text(3_661) == "1h 1m 1s"


def test_regular_guild_scope_requires_both_bot_and_actor_visibility() -> None:
    guild = Mock(spec=discord.Guild)
    bot_member = Mock(spec=discord.Member)
    actor = Mock(spec=discord.Member)
    guild.me = bot_member
    allowed = Mock(spec=discord.TextChannel)
    hidden = Mock(spec=discord.TextChannel)
    allowed.id = 10
    hidden.id = 20
    guild.text_channels = [allowed, hidden]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    readable = discord.Permissions(view_channel=True, read_message_history=True)
    denied = discord.Permissions.none()
    allowed.permissions_for.side_effect = lambda member: (
        readable if member in {bot_member, actor} else denied
    )
    hidden.permissions_for.side_effect = lambda member: (
        readable if member is bot_member else denied
    )

    assert agent_readable_channel_ids(
        guild,
        actor,
        trusted_guild=False,
        trigger_channel_id=10,
    ) == ("10",)


def test_trusted_guild_scope_uses_bot_visibility_without_model_self_report() -> None:
    guild = Mock(spec=discord.Guild)
    bot_member = Mock(spec=discord.Member)
    actor = Mock(spec=discord.Member)
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    guild.text_channels = [channel]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    readable = discord.Permissions(view_channel=True, read_message_history=True)
    denied = discord.Permissions.none()
    channel.permissions_for.side_effect = lambda member: (
        readable if member is bot_member else denied
    )

    assert agent_readable_channel_ids(
        guild,
        actor,
        trusted_guild=True,
        trigger_channel_id=10,
    ) == ("20",)


def test_voice_chat_requires_message_history_and_connect_permission() -> None:
    guild = Mock(spec=discord.Guild)
    bot_member = Mock(spec=discord.Member)
    actor = Mock(spec=discord.Member)
    guild.me = bot_member
    voice = Mock(spec=discord.VoiceChannel)
    voice.id = 30
    guild.text_channels = []
    guild.threads = []
    guild.voice_channels = [voice]
    guild.stage_channels = []
    readable = discord.Permissions(
        view_channel=True,
        read_message_history=True,
        connect=True,
    )
    voice.permissions_for.return_value = readable

    assert agent_readable_channel_ids(
        guild,
        actor,
        trusted_guild=False,
        trigger_channel_id=30,
    ) == ("30",)

    voice.permissions_for.return_value = discord.Permissions(
        view_channel=True,
        read_message_history=True,
        connect=False,
    )
    assert agent_readable_channel_ids(
        guild,
        actor,
        trusted_guild=False,
        trigger_channel_id=30,
    ) == ()


def test_agent_tool_cannot_expand_the_runtime_resource_scope() -> None:
    context = InvocationContext(
        actor_id="30",
        workspace_id="10",
        transport="agent",
        request_id="40",
        resource_ids=("50",),
    )
    _assert_agent_channel_scope(context, "50")
    with pytest.raises(UserError, match="not authorized"):
        _assert_agent_channel_scope(context, "60")


def test_agent_progress_update_stays_in_trigger_channel() -> None:
    context = InvocationContext(
        actor_id="30",
        workspace_id="10",
        transport="agent",
        request_id="discord:message:40",
        resource_ids=("50", "60"),
        origin_resource_id="50",
    )
    _assert_agent_update_scope(context, "50")
    with pytest.raises(UserError, match=r"discord\.agent_update_channel_forbidden"):
        _assert_agent_update_scope(context, "60")
