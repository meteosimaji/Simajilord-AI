from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord import app_commands
from discord.ext import commands
from PIL import Image

from simajilord.agent import (
    AGENT_AUDIO_GRANT,
    AGENT_FILE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_REPOST_GRANT,
    AGENT_WEB_GRANT,
    AgentRateLimitError,
)
from simajilord.capabilities.audio import (
    AudioControlResponse,
    AudioNoArgsRequest,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueItem,
    AudioQueueResponse,
    AudioSearchItem,
    AudioSearchReason,
    AudioSearchResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudDictionarySetRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudRequest,
    ReadAloudResponse,
)
from simajilord.capabilities.web import WebFetchResponse
from simajilord.config import AgentFeatureAccess
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, LoopMode, QueueSnapshot
from simajilord.integrations.discord.bot import SimajilordDiscordBot
from simajilord.integrations.discord.capabilities import (
    DiscordServerResponse,
    DiscordUserResponse,
    DiscordViewCustomEmojiRequest,
    DiscordViewCustomEmojiResponse,
    DiscordViewStickerRequest,
    DiscordViewStickerResponse,
    _actor_member,
    _assert_agent_channel_scope,
    _assert_agent_update_scope,
    _bounded_event_message,
    _can_post_expanded_message,
    _custom_emoji_records,
    _discord_event_message_id,
    _message_preview,
    _prepare_discord_animated_media,
    agent_readable_channel_ids,
    build_discord_endpoints,
    parse_discord_message_link,
)
from simajilord.integrations.discord.cogs import (
    _QUOTE_CONTEXT_MENU_NAME,
    DiscordActionCog,
    DiscordInfoCog,
    DownloadCog,
    FocusTimerCog,
    HelpCog,
    LoopMixConflictView,
    ModerationCog,
    MusicCog,
    MusicControlsView,
    MusicDashboardManager,
    MusicSearchChoiceView,
    PrefixCog,
    QuoteCog,
    QuoteComposerView,
    ReadAloudChannelSelect,
    ReadAloudChannelSelectView,
    ReadAloudCog,
    SystemCog,
    UtilityCog,
    VoiceLifecycleCog,
    WebCog,
    WebFetchContinueView,
    YouTubeLinkCardCog,
    YouTubeLinkCardView,
    _agent_error_text,
    _agent_grants,
    _agent_message_groups,
    _AgentProgressMessage,
    _discord_message_chunks,
    _retry_after_text,
    _youtube_card_reference,
    discord_conversation_id,
    error_message,
    server_info_embed,
    user_info_embed,
)
from simajilord.integrations.discord.help_catalog import (
    HELP_ENTRIES,
    HELP_ENTRIES_BY_TOPIC,
)
from simajilord.runtime import SimajilordRuntime


@pytest.mark.asyncio
async def test_actor_member_fetches_exact_member_when_cache_is_empty() -> None:
    guild = Mock(spec=discord.Guild)
    guild.get_member.return_value = None
    member = Mock(spec=discord.Member)
    member.id = 7
    guild.fetch_member = AsyncMock(return_value=member)
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="discord",
        request_id="message",
    )

    assert await _actor_member(guild, context) is member
    guild.fetch_member.assert_awaited_once_with(7)


@pytest.mark.asyncio
async def test_agent_no_action_sentinel_is_never_published() -> None:
    source = Mock(spec=discord.Message)
    source.id = 42
    published = Mock(spec=discord.Message)
    published.delete = AsyncMock()
    progress = _AgentProgressMessage(source)
    progress.message = published

    await progress.finish("<simajilord:no-action>")

    published.delete.assert_awaited_once_with()
    assert progress.message is None


def test_member_lookup_error_is_clear_and_english() -> None:
    message = error_message(UserError("discord.member_required"))
    assert message == "Could not resolve that member in this server."


def test_autonomous_agent_grants_keep_reads_but_remove_write_scopes() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.settings.agent_file_sandbox_enabled = True
    runtime.settings.agent_web_search_access = AgentFeatureAccess.EVERYONE
    runtime.settings.agent_admin_user_ids = frozenset()
    runtime.settings.image_generation_access = AgentFeatureAccess.EVERYONE
    runtime.files = object()
    runtime.moderation.provider = object()
    runtime.image.provider = object()

    requested = _agent_grants(runtime, actor_id="7")
    autonomous = _agent_grants(runtime, actor_id="simajilord:autonomy", autonomous=True)

    assert {
        AGENT_AUDIO_GRANT,
        AGENT_WEB_GRANT,
        AGENT_MODERATION_GRANT,
        AGENT_REPOST_GRANT,
    } <= autonomous
    assert {
        AGENT_MESSAGE_GRANT,
        AGENT_FILE_GRANT,
        AGENT_IMAGE_GRANT,
    } <= requested
    assert (
        not {
            AGENT_MESSAGE_GRANT,
            AGENT_FILE_GRANT,
            AGENT_IMAGE_GRANT,
        }
        & autonomous
    )


def test_common_music_actions_have_short_top_level_commands() -> None:
    commands = {
        command.name: command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"audio", "play", "radio"}
    assert commands["audio"].description == (
        "Open music controls and read-aloud setup in one panel."
    )
    assert commands["play"].description == (
        "Find a song or public URL and add it to the shared queue."
    )
    assert commands["radio"].description.startswith("Keep related music")


def test_every_public_slash_command_has_exactly_one_help_entry() -> None:
    cog_types = (
        HelpCog,
        SystemCog,
        FocusTimerCog,
        MusicCog,
        ReadAloudCog,
        WebCog,
        ModerationCog,
        DownloadCog,
        UtilityCog,
        DiscordInfoCog,
        DiscordActionCog,
    )
    public_topics: set[str] = set()
    command_by_topic: dict[str, app_commands.Command[object, ..., object]] = {}
    for cog_type in cog_types:
        for command in cog_type.__cog_app_commands__:
            if isinstance(command, app_commands.Group):
                for child in command.commands:
                    topic = f"{command.name} {child.name}"
                    public_topics.add(topic)
                    command_by_topic[topic] = child
            elif isinstance(command, app_commands.Command):
                public_topics.add(command.name)
                command_by_topic[command.name] = command
    public_topics.add(_QUOTE_CONTEXT_MENU_NAME)

    help_topics = {entry.topic for entry in HELP_ENTRIES}
    assert len(HELP_ENTRIES_BY_TOPIC) == len(HELP_ENTRIES)
    assert help_topics == public_topics

    for topic, command in command_by_topic.items():
        usage = HELP_ENTRIES_BY_TOPIC[topic.casefold()].usage
        for parameter in command.parameters:
            assert parameter.description and parameter.description != "…", (
                f"/{topic} option `{parameter.name}` has no Discord description"
            )
            assert parameter.name in usage, (
                f"/{topic} help omits the `{parameter.name}` option"
            )


def test_public_command_and_option_descriptions_use_the_official_english_surface() -> None:
    japanese = re.compile(r"[ぁ-んァ-ヶ一-龯]")
    cog_types = (
        HelpCog,
        SystemCog,
        FocusTimerCog,
        MusicCog,
        ReadAloudCog,
        WebCog,
        ModerationCog,
        DownloadCog,
        UtilityCog,
        DiscordInfoCog,
        DiscordActionCog,
    )
    for cog_type in cog_types:
        for command in cog_type.__cog_app_commands__:
            children = command.commands if isinstance(command, app_commands.Group) else (command,)
            for child in children:
                assert not japanese.search(child.description), child.qualified_name
                for parameter in child.parameters:
                    assert not japanese.search(parameter.description), (
                        child.qualified_name,
                        parameter.name,
                    )


def test_discord_capability_catalog_uses_the_official_english_surface() -> None:
    japanese = re.compile(r"[ぁ-んァ-ヶ一-龯]")
    endpoints = build_discord_endpoints(
        cast(discord.Client, object()),
        Mock(spec=SimajilordRuntime),
    )

    for item in endpoints:
        assert not japanese.search(item.descriptor.summary), item.descriptor.name
        for side_effect in item.descriptor.side_effects:
            assert not japanese.search(side_effect), item.descriptor.name


def test_help_categories_fit_discord_select_limits() -> None:
    categories = {entry.category for entry in HELP_ENTRIES}
    assert len(categories) <= 25
    for category in categories:
        entries = [entry for entry in HELP_ENTRIES if entry.category == category]
        assert 1 <= len(entries) <= 25
        assert all(len(entry.summary) <= 100 for entry in entries)


def test_prefix_help_is_not_shadowed_by_capability_search() -> None:
    assert PrefixCog.help.name == "help"
    assert "help" not in PrefixCog.capabilities.aliases
    assert PrefixCog.capabilities.name == "capabilities"


def test_server_info_embed_contains_identity_population_channels_and_safety() -> None:
    embed = server_info_embed(
        DiscordServerResponse(
            server_id="123",
            name="Example server",
            owner_id="7",
            member_count=0,
            text_channel_count=4,
            voice_channel_count=2,
            role_count=5,
            created_at_iso="2025-09-10T09:01:00+00:00",
            icon_url="https://cdn.example/icon.png",
            description="A useful server description",
            owner_name="Owner",
            human_count=0,
            bot_count=0,
            category_count=2,
            stage_channel_count=1,
            forum_channel_count=1,
            emoji_count=8,
            sticker_count=3,
            boost_level=2,
            boost_count=6,
            preferred_locale="ja",
            verification_level="medium",
            explicit_content_filter="all_members",
            features=("COMMUNITY",),
        )
    )
    fields = {field.name: field.value for field in embed.fields}
    assert embed.title == "Example server"
    assert "`123`" in (embed.description or "")
    assert "**0** total" in fields["Population"]
    assert "4 text" in fields["Channels"]
    assert "8 emoji" in fields["Community"]
    assert "Verification" in fields["Safety"]
    assert embed.thumbnail.url == "https://cdn.example/icon.png"


def test_user_info_embed_contains_account_membership_roles_and_permissions() -> None:
    embed = user_info_embed(
        DiscordUserResponse(
            user_id="7",
            display_name="Display",
            bot=False,
            created_at_iso="2024-01-02T03:04:00+00:00",
            joined_at_iso="2025-01-02T03:04:00+00:00",
            top_role="Moderator",
            avatar_url="https://cdn.example/avatar.png",
            username="account",
            nickname="Nick",
            role_names=("Member", "Moderator"),
            role_count=2,
            status="online",
            key_permissions=("Manage Messages",),
            colour_value=0x123456,
        ),
        mention="<@7>",
    )
    fields = {field.name: field.value for field in embed.fields}
    assert embed.title == "Display"
    assert "<@7>" in (embed.description or "")
    assert "Created" in fields["Account"]
    assert "Top role: **Moderator**" in fields["Server membership"]
    assert "Nickname: **Nick**" in fields["Status"]
    assert "Member" in fields["Roles · 2"]
    assert fields["Key server permissions"] == "Manage Messages"
    assert not next(field for field in embed.fields if field.name == "Account").inline
    assert not next(
        field for field in embed.fields if field.name == "Server membership"
    ).inline
    assert not next(field for field in embed.fields if field.name == "Status").inline
    assert embed.thumbnail.url == "https://cdn.example/avatar.png"


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
        AsyncMock(return_value=member),
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
async def test_agent_pause_adapter_invokes_exact_audio_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(
        return_value=AudioControlResponse(action="pause", loop_mode=None)
    )
    session = Mock()
    session.output.connected = False
    session.waiting_for_voice = False
    runtime.audio.require.return_value = session
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        AsyncMock(return_value=member),
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

    response = await endpoint_by_name["discord.pause_audio"].invoke(
        AudioNoArgsRequest(),
        context,
    )

    assert response.action == "pause"
    runtime.registry.invoke.assert_awaited_once_with(
        "audio.pause",
        AudioNoArgsRequest(),
        context,
    )


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
        AsyncMock(return_value=member),
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
        AsyncMock(return_value=member),
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
async def test_read_aloud_self_mute_is_allowed_but_other_user_requires_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ReadAloudPolicyResponse(
        dictionary=(),
        ignored_user_ids=("7",),
        ignored_role_ids=(),
        announce_join=False,
        announce_leave=False,
        announce_move=False,
        read_author_names=True,
        read_replies=True,
        read_attachments=True,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=response)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    member.id = 7
    member.bot = False
    member.guild_permissions.manage_guild = False
    guild.get_member.return_value = member
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        AsyncMock(return_value=member),
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
        transport="discord",
        request_id="event",
    )

    own = await endpoint_by_name["discord.read_aloud_exclusion_set"].invoke(
        ReadAloudExclusionSetRequest(
            target=ReadAloudExclusionTarget.USER,
            target_id="7",
            ignored=True,
        ),
        context,
    )

    assert own == response
    assert runtime.registry.invoke.await_args.args[0] == ("speech.read_aloud_exclusion_set")
    runtime.registry.invoke.reset_mock()
    with pytest.raises(UserError, match=r"discord\.manage_guild_required"):
        await endpoint_by_name["discord.read_aloud_exclusion_set"].invoke(
            ReadAloudExclusionSetRequest(
                target=ReadAloudExclusionTarget.USER,
                target_id="8",
                ignored=True,
            ),
            context,
        )
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_aloud_dictionary_write_requires_manage_guild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock()
    guild = Mock(spec=discord.Guild)
    member = Mock(spec=discord.Member)
    member.guild_permissions.manage_guild = False
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        AsyncMock(return_value=member),
    )
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }

    with pytest.raises(UserError, match=r"discord\.manage_guild_required"):
        await endpoints["discord.read_aloud_dictionary_set"].invoke(
            ReadAloudDictionarySetRequest("IUT", "あいゆーてぃー"),
            InvocationContext(
                actor_id="7",
                workspace_id="1",
                transport="discord",
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
        AsyncMock(return_value=member),
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
    delegated_name, delegated_request, delegated_context = runtime.registry.invoke.await_args.args
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
        AsyncMock(return_value=member),
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
        "volume",
        "move",
        "clear-mine",
    }


@pytest.mark.asyncio
async def test_music_buttons_are_concise_grouped_and_uniquely_addressable() -> None:
    response = AudioQueueResponse(
        current=AudioQueueItem(
            title="Track",
            page_url="https://example.com/track",
            kind="music",
            duration_seconds=240,
            requested_by_name="Requester",
        ),
        pending=(),
        paused=False,
        loop_mode="none",
        destination_id="10",
        auto_leave=True,
        position_seconds=30,
        speed=1,
        pitch=1,
        waiting_for_voice=False,
        connected=True,
    )
    view = MusicControlsView(
        cast(SimajilordRuntime, object()),
        response=response,
    )
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
    assert [button.label for button in buttons] == [
        "Pause",
        "Skip",
        "Stop",
        "Add music",
    ]
    assert all(button.row == 0 for button in buttons)
    selects = [child for child in view.children if isinstance(child, discord.ui.Select)]
    assert len(selects) == 1
    assert selects[0].placeholder == "More actions"
    assert {option.value for option in selects[0].options} == {
        "radio",
        "loop",
        "queue",
        "levels",
        "read_aloud",
        "history",
        "clear_mine",
        "details",
        "leave",
    }
    custom_ids = [
        child.custom_id
        for child in view.children
        if isinstance(child, (discord.ui.Button, discord.ui.Select))
    ]
    assert None not in custom_ids
    assert len(custom_ids) == len(set(custom_ids))


@pytest.mark.asyncio
async def test_music_pause_button_changes_to_resume_without_duplicate_control() -> None:
    response = AudioQueueResponse(
        current=AudioQueueItem(
            title="Track",
            page_url="https://example.com/track",
            kind="music",
            duration_seconds=240,
            requested_by_name="Requester",
        ),
        pending=(),
        paused=True,
        loop_mode="none",
        destination_id="10",
        auto_leave=True,
        position_seconds=30,
        speed=1,
        pitch=1,
        waiting_for_voice=False,
        autoplay_enabled=True,
        connected=True,
    )
    view = MusicControlsView(
        cast(SimajilordRuntime, object()),
        response=response,
    )
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
    assert [button.label for button in buttons].count("Resume") == 1
    assert "Pause" not in [button.label for button in buttons]
    assert any(
        isinstance(child, discord.ui.Select)
        and child.placeholder == "More actions"
        for child in view.children
    )


@pytest.mark.asyncio
async def test_music_resume_confirmation_uses_start_without_pause_resume_controls() -> None:
    response = AudioQueueResponse(
        current=None,
        pending=(
            AudioQueueItem(
                title="Held track",
                page_url="https://example.com/held",
                kind="music",
                duration_seconds=240,
                requested_by_name="Requester",
            ),
        ),
        paused=False,
        loop_mode="none",
        destination_id="10",
        auto_leave=True,
        position_seconds=30,
        speed=1,
        pitch=1,
        waiting_for_voice=False,
        resume_confirmation_required=True,
        connected=False,
    )
    view = MusicControlsView(
        cast(SimajilordRuntime, object()),
        response=response,
    )
    labels = [
        child.label for child in view.children if isinstance(child, discord.ui.Button)
    ]
    assert labels == ["Start", "Add music"]
    assert any(
        isinstance(child, discord.ui.Select)
        and child.placeholder == "More actions"
        for child in view.children
    )


@pytest.mark.asyncio
async def test_disconnected_idle_radio_panel_only_shows_relevant_entry_points() -> None:
    response = AudioQueueResponse(
        current=None,
        pending=(),
        paused=False,
        loop_mode="none",
        destination_id="10",
        auto_leave=True,
        position_seconds=0,
        speed=1,
        pitch=1,
        waiting_for_voice=False,
        autoplay_enabled=True,
        connected=False,
    )

    view = MusicControlsView(
        cast(SimajilordRuntime, object()),
        response=response,
    )
    labels = [
        child.label for child in view.children if isinstance(child, discord.ui.Button)
    ]

    assert labels == ["Add music"]
    assert any(
        isinstance(child, discord.ui.Select)
        and child.placeholder == "More actions"
        for child in view.children
    )


@pytest.mark.asyncio
async def test_music_dashboard_edits_existing_panel_without_reposting() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    first = Mock(spec=discord.Message)
    first.id = 101
    first.delete = AsyncMock()
    first.edit = AsyncMock(return_value=first)
    channel.send = AsyncMock(return_value=first)
    bot.get_channel.return_value = channel
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    await manager._repost(session)
    first_kwargs = channel.send.await_args_list[0].kwargs
    assert first_kwargs["silent"] is True
    allowed_mentions = first_kwargs["allowed_mentions"]
    assert allowed_mentions.everyone is False
    assert allowed_mentions.users is False
    assert allowed_mentions.roles is False

    session.snapshot.return_value = QueueSnapshot(
        current=None,
        pending=(),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.TRACK,
        destination_id="55",
    )
    await manager._repost(session)
    channel.send.assert_awaited_once()
    first.edit.assert_awaited_once()
    first.delete.assert_not_awaited()
    await manager.close()


@pytest.mark.asyncio
async def test_music_dashboard_retries_the_same_message_once_after_429() -> None:
    class DashboardError(discord.DiscordException):
        def __init__(self) -> None:
            super().__init__("rate limited")
            self.status = 429
            self.code = 0
            self.retry_after = 0.0

    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    runtime.journal.append = AsyncMock()
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    panel = Mock(spec=discord.Message)
    panel.id = 121
    panel.delete = AsyncMock()
    panel.edit = AsyncMock(side_effect=[DashboardError(), panel])
    channel.send = AsyncMock(return_value=panel)
    bot.get_channel.return_value = channel
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    await manager._repost(session)
    session.snapshot.return_value = QueueSnapshot(
        current=None,
        pending=(),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.TRACK,
        destination_id="55",
    )
    await manager._repost(session)

    assert panel.edit.await_count == 2
    channel.send.assert_awaited_once()
    outcomes = [
        call.kwargs["payload"]["outcome"]
        for call in runtime.journal.append.await_args_list
        if call.kwargs["payload"]["operation"] == "discord.dashboard_429"
    ]
    assert outcomes == ["rate_limited", "retry_succeeded"]
    await manager.close()


@pytest.mark.asyncio
async def test_music_dashboard_stops_updates_after_403() -> None:
    class DashboardError(discord.DiscordException):
        def __init__(self) -> None:
            super().__init__("forbidden")
            self.status = 403
            self.code = 0

    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    runtime.journal.append = AsyncMock()
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    panel = Mock(spec=discord.Message)
    panel.id = 131
    panel.delete = AsyncMock()
    panel.edit = AsyncMock(side_effect=DashboardError())
    channel.send = AsyncMock(return_value=panel)
    bot.get_channel.return_value = channel
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    await manager._repost(session)
    session.snapshot.return_value = QueueSnapshot(
        current=None,
        pending=(),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.TRACK,
        destination_id="55",
    )
    await manager._repost(session)

    panel.edit.assert_awaited_once()
    channel.send.assert_awaited_once()
    assert "1" not in manager._channel_ids
    runtime.journal.append.assert_any_await(
        kind="service.operation",
        workspace_id=None,
        payload={
            "operation": "discord.dashboard_403",
            "wait_ms": 0.0,
            "duration_ms": 0.0,
            "outcome": "stopped",
        },
    )
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "metric"),
    (
        (404, 0, "discord.dashboard_404"),
        (400, 30046, "discord.dashboard_30046"),
    ),
)
async def test_music_dashboard_replaces_missing_or_expired_canonical_panel(
    status: int,
    code: int,
    metric: str,
) -> None:
    class DashboardError(discord.DiscordException):
        def __init__(self) -> None:
            super().__init__("panel cannot be edited")
            self.status = status
            self.code = code

    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    runtime.journal.append = AsyncMock()
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    old_panel = Mock(spec=discord.Message)
    old_panel.id = 141
    old_panel.delete = AsyncMock()
    old_panel.edit = AsyncMock(side_effect=DashboardError())
    replacement = Mock(spec=discord.Message)
    replacement.id = 142
    replacement.delete = AsyncMock()
    replacement.edit = AsyncMock(return_value=replacement)
    channel.send = AsyncMock(side_effect=[old_panel, replacement])
    bot.get_channel.return_value = channel
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    await manager._repost(session)
    session.snapshot.return_value = QueueSnapshot(
        current=None,
        pending=(),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.TRACK,
        destination_id="55",
    )
    await manager._repost(session)

    assert channel.send.await_count == 2
    assert manager._messages["1"] is replacement
    outcomes = [
        call.kwargs["payload"]["outcome"]
        for call in runtime.journal.append.await_args_list
        if call.kwargs["payload"]["operation"] == metric
    ]
    assert outcomes
    if code == 30046:
        old_panel.delete.assert_awaited_once()
    await manager.close()


@pytest.mark.asyncio
async def test_music_dashboard_expires_only_after_audio_becomes_idle() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    panel = Mock(spec=discord.Message)
    panel.id = 111
    panel.delete = AsyncMock()
    panel.edit = AsyncMock(return_value=panel)
    channel.send = AsyncMock(return_value=panel)
    bot.get_channel.return_value = channel
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=AudioItem(
                source="stream",
                title="Playing",
                page_url="https://example.com/playing",
                duration_seconds=120,
            ),
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    await manager._repost(session)
    assert "1" not in manager._expiry_tasks

    session.snapshot.return_value = QueueSnapshot(
        current=None,
        pending=(),
        history=(),
        paused=False,
        speech_active=False,
        loop=LoopMode.NONE,
        destination_id="55",
    )
    await manager._repost(session)
    assert "1" in manager._expiry_tasks
    panel.edit.assert_awaited_once()
    channel.send.assert_awaited_once()
    await manager.close()


@pytest.mark.asyncio
async def test_music_dashboard_replaces_persisted_panel_after_restart(
    tmp_path: Path,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.settings.data_dir = tmp_path
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    bot = Mock(spec=commands.Bot)
    channel = Mock(spec=discord.TextChannel)
    old = Mock(spec=discord.Message)
    old.id = 201
    old.delete = AsyncMock()
    old.edit = AsyncMock(return_value=old)
    channel.send = AsyncMock(return_value=old)
    channel.fetch_message = AsyncMock(return_value=old)
    bot.get_channel.return_value = channel
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id="55",
        )
    )

    first_manager = MusicDashboardManager(bot, runtime)
    first_manager.bind(1, 2)
    await first_manager._repost(session)
    await first_manager.close()

    second_manager = MusicDashboardManager(bot, runtime)
    second_manager.bind(1, 2)
    await second_manager._repost(session)

    channel.fetch_message.assert_awaited_once_with(201)
    old.edit.assert_awaited_once()
    old.delete.assert_not_awaited()
    channel.send.assert_awaited_once()
    state = json.loads((tmp_path / "discord_music_dashboards.json").read_text(encoding="utf-8"))
    assert state["messages"]["1"] == {"channel_id": 2, "message_id": 201}
    await second_manager.close()


@pytest.mark.asyncio
async def test_music_dashboard_removes_empty_panel_after_explicit_leave() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.audio.add_state_listener = Mock()
    runtime.audio.remove_state_listener = Mock()
    runtime.audio.find.return_value = None
    bot = Mock(spec=commands.Bot)
    panel = Mock(spec=discord.Message)
    panel.id = 301
    panel.delete = AsyncMock()
    manager = MusicDashboardManager(bot, runtime)
    manager.bind(1, 2)
    manager._messages["1"] = panel
    session = Mock()
    session.workspace_id = "1"
    session.snapshot = AsyncMock(
        return_value=QueueSnapshot(
            current=None,
            pending=(),
            history=(),
            paused=False,
            speech_active=False,
            loop=LoopMode.NONE,
            destination_id=None,
        )
    )

    await manager._repost(session)

    panel.delete.assert_awaited_once_with()
    assert "1" not in manager._messages
    await manager.close()


def test_loop_mix_conflict_view_offers_one_click_switch() -> None:
    mix_view = LoopMixConflictView(
        cast(SimajilordRuntime, object()),
        None,
        requester_id=7,
    )
    loop_view = LoopMixConflictView(
        cast(SimajilordRuntime, object()),
        None,
        requester_id=7,
        loop_mode=LoopMode.TRACK,
    )

    assert [
        child.label
        for child in mix_view.children
        if isinstance(child, discord.ui.Button)
    ] == ["Switch to Radio", "Keep current mode"]
    assert [
        child.label
        for child in loop_view.children
        if isinstance(child, discord.ui.Button)
    ] == ["Switch to Loop", "Keep current mode"]


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
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
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
    assert discord_conversation_id(guild_id=1, channel_id=2) == ("discord:guild:1:channel:2")
    assert discord_conversation_id(guild_id=None, channel_id=2) == ("discord:direct:channel:2")
    assert (
        discord_conversation_id(
            guild_id=1,
            channel_id=2,
            grants=frozenset({AGENT_WEB_GRANT}),
        )
        == "discord:guild:1:channel:2:profile:web"
    )


def test_web_commands_are_short_direct_paths() -> None:
    commands = {
        command.name: command
        for command in WebCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"search", "fetch", "find"}
    assert commands["search"].description == (
        "Search the web through Simajilord's local search service."
    )


def test_quote_context_menu_uses_a_short_native_label() -> None:
    assert _QUOTE_CONTEXT_MENU_NAME == "Quote"


def test_read_aloud_has_zero_argument_join_entrypoint() -> None:
    commands = {
        command.name: command
        for command in ReadAloudCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"join"}
    assert commands["join"].description == (
        "Choose conversation channels to read in your current VC."
    )
    assert commands["join"].parameters == []


def test_join_channel_selector_supports_one_to_twenty_five_conversations() -> None:
    view = ReadAloudChannelSelectView(
        cast(SimajilordRuntime, object()),
        requester_id=7,
        destination_id=55,
        default_values=(),
    )
    selector = view.selector

    assert selector.custom_id == "simajilord:readaloud:channels"
    assert selector.min_values == 1
    assert selector.max_values == 25
    assert discord.ChannelType.text in selector.channel_types
    assert discord.ChannelType.voice in selector.channel_types
    assert discord.ChannelType.public_thread in selector.channel_types
    assert any(
        isinstance(item, discord.ui.Button)
        and item.label == "Start"
        and item.style is discord.ButtonStyle.success
        for item in view.children
    )


@pytest.mark.asyncio
async def test_join_selection_is_staged_until_start_is_pressed() -> None:
    configured = ReadAloudResponse(
        action=ReadAloudAction.ADD_SOURCES.value,
        enabled=True,
        text_channel_id="50",
        text_channel_ids=("50", "51"),
        audio_destination_id="55",
        mode="queue",
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(side_effect=(configured, object()))
    selector = ReadAloudChannelSelect(
        runtime,
        requester_id=7,
        destination_id=55,
        default_values=(),
    )
    selected_one = Mock()
    selected_one.id = 50
    selected_two = Mock()
    selected_two.id = 51
    selector._values = [selected_one, selected_two]
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 90
    interaction.guild_id = 1
    interaction.channel_id = 50
    interaction.user.id = 7
    interaction.response.defer = AsyncMock()
    interaction.response.send_message = AsyncMock()
    interaction.edit_original_response = AsyncMock()

    await selector.callback(interaction)

    runtime.registry.invoke.assert_not_awaited()
    interaction.response.defer.assert_awaited_once()
    assert selector.selected_channel_ids == ("50", "51")

    interaction.response.defer.reset_mock()
    await selector.commit(interaction)

    assert [call.args[0] for call in runtime.registry.invoke.await_args_list] == [
        "discord.manage_read_aloud",
        "discord.connect_voice",
    ]
    interaction.response.defer.assert_awaited_once()
    interaction.edit_original_response.assert_awaited_once()
    embed = interaction.edit_original_response.await_args.kwargs["embed"]
    assert embed.title == "Read aloud is ready"
    assert any(
        field.name == "Connection" and field.value == "Ready"
        for field in embed.fields
    )


@pytest.mark.asyncio
async def test_quote_context_menu_opens_private_options_without_posting() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=object())
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 90
    interaction.guild_id = 1
    interaction.channel_id = 50
    interaction.user.id = 7
    interaction.response.send_message = AsyncMock()
    message = Mock(spec=discord.Message)
    message.id = 60
    message.channel.id = 50
    message.content = ""
    message.stickers = []

    await QuoteCog(runtime).create_quote(interaction, message)

    interaction.response.send_message.assert_awaited_once()
    assert interaction.response.send_message.await_args.kwargs["ephemeral"] is True
    assert isinstance(
        interaction.response.send_message.await_args.kwargs["view"],
        QuoteComposerView,
    )
    runtime.registry.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_quote_composer_preserves_combinable_styles_and_jump_choice() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=object())
    view = QuoteComposerView(
        runtime,
        requester_id=7,
        source_channel_id=50,
        source_message_id=60,
        destination_channel_id=50,
        has_animation=True,
    )
    view.color = True
    view.vertical = True
    view.bold = True
    view.flip = True
    view.animate = True
    view.include_jump = False

    request = view.request()
    assert request.color is True
    assert request.vertical is True
    assert request.bold is True
    assert request.flip is True
    assert request.animate is True
    assert request.include_jump is False


def test_quote_composer_uses_hierarchical_native_menu() -> None:
    view = QuoteComposerView(
        cast(SimajilordRuntime, object()),
        requester_id=7,
        source_channel_id=50,
        source_message_id=60,
        destination_channel_id=50,
        has_animation=True,
    )

    assert [
        item.label for item in view.children if isinstance(item, discord.ui.Button)
    ] == [
        "Layout · Landscape",
        "Style · B/W",
        "More · 1 On",
        "Generate",
        "Cancel",
    ]

    view._show_page("more")
    assert [
        item.label for item in view.children if isinstance(item, discord.ui.Button)
    ] == ["Flip Off", "Jump On", "Animation Off", "Back"]

    view._show_page("layout")
    assert [
        item.label for item in view.children if isinstance(item, discord.ui.Button)
    ] == ["Landscape", "Back"]


@pytest.mark.parametrize(
    ("content", "expected"),
    (
        (
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        (
            "Listen https://music.youtube.com/watch?v=dQw4w9WgXcQ&feature=share",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        (
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        ("https://youtube.com.example/watch?v=dQw4w9WgXcQ", None),
        (
            "https://youtu.be/dQw4w9WgXcQ https://youtu.be/9bZkp7q19f0",
            None,
        ),
    ),
)
def test_youtube_card_reference_is_conservative(
    content: str,
    expected: str | None,
) -> None:
    assert _youtube_card_reference(content) == expected


def test_youtube_card_has_three_direct_actions() -> None:
    view = YouTubeLinkCardView(
        cast(SimajilordRuntime, object()),
        reference="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )

    assert [
        item.label for item in view.children if isinstance(item, discord.ui.Button)
    ] == ["Play", "Add", "Radio"]


def test_fresh_mix_and_duplicate_music_aliases_are_hidden() -> None:
    top_level = {
        command.name
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    groups = [
        command
        for command in MusicCog.__cog_app_commands__
        if isinstance(command, app_commands.Group)
    ]

    assert "freshmix" not in top_level
    assert top_level == {"audio", "play", "radio"}
    assert len(groups) == 1
    grouped = {command.name for command in groups[0].commands}
    assert {"freshmix", "radio", "mix", "play", "queue", "history"} & grouped == set()


@pytest.mark.asyncio
async def test_youtube_message_listener_posts_silent_temporary_card() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    message = Mock(spec=discord.Message)
    message.guild = Mock(spec=discord.Guild)
    message.guild.id = 1
    message.channel = Mock(spec=discord.TextChannel)
    message.channel.id = 2
    message.id = 3
    message.author = Mock(spec=discord.Member)
    message.author.bot = False
    message.content = "https://youtu.be/dQw4w9WgXcQ"
    reply = Mock(spec=discord.Message)
    message.reply = AsyncMock(return_value=reply)

    await YouTubeLinkCardCog(runtime).on_message(message)

    message.reply.assert_awaited_once()
    kwargs = message.reply.await_args.kwargs
    assert kwargs["silent"] is True
    assert kwargs["mention_author"] is False
    assert isinstance(kwargs["view"], YouTubeLinkCardView)
    assert kwargs["view"].message is reply


@pytest.mark.asyncio
async def test_listener_join_keeps_persisted_read_aloud_route_in_standby() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    route = Mock()
    route.audio_destination_id = "55"
    runtime.read_aloud.get.return_value = route
    runtime.audio.find.return_value = None
    session = Mock()
    session.waiting_for_voice = False
    session.has_music = False
    session.resume_confirmation_required = False
    session.destination_id = None
    session.output.connected = False
    runtime.audio.get_or_create.return_value = session
    runtime.audio.connect = AsyncMock()
    dashboard = Mock(spec=MusicDashboardManager)
    member = Mock(spec=discord.Member)
    member.bot = False
    member.id = 7
    member.guild.id = 1
    before = Mock(spec=discord.VoiceState)
    before.channel = None
    after = Mock(spec=discord.VoiceState)
    after.channel = Mock(spec=discord.VoiceChannel)
    after.channel.id = 55
    cog = VoiceLifecycleCog(cast(commands.Bot, object()), runtime)
    cog.dashboard = dashboard

    await cog.on_voice_state_update(member, before, after)

    runtime.audio.get_or_create.assert_called_once()
    runtime.audio.connect.assert_not_awaited()
    dashboard.bind.assert_not_called()


@pytest.mark.asyncio
async def test_startup_keeps_read_aloud_passive_when_listener_is_already_present() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    route = Mock()
    route.audio_destination_id = "55"
    runtime.read_aloud.get.return_value = route
    session = Mock()
    session.output.connected = False
    session.has_music = False
    session.resume_confirmation_required = False
    session.destination_id = None
    runtime.audio.get_or_create.return_value = session
    runtime.audio.connect = AsyncMock()
    listener = Mock(spec=discord.Member)
    listener.bot = False
    channel = Mock(spec=discord.VoiceChannel)
    channel.members = [listener]
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel.return_value = channel
    bot = Mock(spec=SimajilordDiscordBot)
    bot.runtime = runtime
    bot.guilds = [guild]

    await SimajilordDiscordBot._prepare_read_aloud_presence(bot)

    runtime.audio.get_or_create.assert_not_called()
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_listener_join_reposts_panel_without_restarting_held_music() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.read_aloud.get.return_value = None
    session = Mock()
    session.waiting_for_voice = False
    session.has_music = True
    session.resume_confirmation_required = True
    session.destination_id = "55"
    session.output.connected = False
    runtime.audio.find.return_value = session
    runtime.audio.connect = AsyncMock()
    member = Mock(spec=discord.Member)
    member.bot = False
    member.id = 7
    member.guild.id = 1
    before = Mock(spec=discord.VoiceState)
    before.channel = None
    after = Mock(spec=discord.VoiceState)
    after.channel = Mock(spec=discord.VoiceChannel)
    after.channel.id = 55
    dashboard = Mock(spec=MusicDashboardManager)
    cog = VoiceLifecycleCog(cast(commands.Bot, object()), runtime)
    cog.dashboard = dashboard

    await cog.on_voice_state_update(member, before, after)

    dashboard.bind.assert_called_once_with(1, 55)
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_read_aloud_does_not_restart_held_read_aloud_route() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    route = Mock()
    route.audio_destination_id = "55"
    runtime.read_aloud.get.return_value = route
    session = Mock()
    session.output.connected = False
    session.has_music = False
    session.resume_confirmation_required = True
    session.destination_id = "55"
    runtime.audio.get_or_create.return_value = session
    runtime.audio.connect = AsyncMock()
    listener = Mock(spec=discord.Member)
    listener.bot = False
    channel = Mock(spec=discord.VoiceChannel)
    channel.members = [listener]
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel.return_value = channel
    bot = Mock(spec=SimajilordDiscordBot)
    bot.runtime = runtime
    bot.guilds = [guild]

    await SimajilordDiscordBot._prepare_read_aloud_presence(bot)

    runtime.audio.get_or_create.assert_not_called()
    runtime.audio.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_listener_join_does_not_restart_held_read_aloud_route() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    route = Mock()
    route.audio_destination_id = "55"
    runtime.read_aloud.get.return_value = route
    session = Mock()
    session.waiting_for_voice = False
    session.has_music = False
    session.resume_confirmation_required = True
    session.destination_id = "55"
    session.output.connected = False
    runtime.audio.find.return_value = session
    runtime.audio.connect = AsyncMock()
    member = Mock(spec=discord.Member)
    member.bot = False
    member.id = 7
    member.guild.id = 1
    before = Mock(spec=discord.VoiceState)
    before.channel = None
    after = Mock(spec=discord.VoiceState)
    after.channel = Mock(spec=discord.VoiceChannel)
    after.channel.id = 55
    cog = VoiceLifecycleCog(cast(commands.Bot, object()), runtime)

    await cog.on_voice_state_update(member, before, after)

    runtime.audio.connect.assert_not_awaited()


def test_hive_analysis_is_one_direct_attachment_command() -> None:
    commands = {
        command.name: command
        for command in ModerationCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"detectai"}
    assert commands["detectai"].description == (
        "Estimate AI-generation and deepfake likelihood with HIVE."
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
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
    assert [button.label for button in buttons] == ["Continue"]
    assert buttons[0].custom_id == "simajilord:web:fetch:continue"
    assert view.next_offset == 3_500


def test_message_index_preview_uses_full_short_or_25_plus_5() -> None:
    assert _message_preview("x" * 30) == ("x" * 30, False)
    preview, truncated = _message_preview("abcdefghijklmnopqrstuvwxyz123456789")
    assert preview == "abcdefghijklmnopqrstuvwxy…56789"
    assert truncated is True


def test_discord_message_link_parser_accepts_only_one_bare_safe_link() -> None:
    expected = ("1415260807494766627", "1415260808103067670", "1531170172465971200")
    for host in ("discord.com", "ptb.discord.com", "canary.discord.com"):
        parsed = parse_discord_message_link(
            f"https://{host}/channels/{expected[0]}/{expected[1]}/{expected[2]}/"
        )
        assert parsed is not None
        assert (parsed.guild_id, parsed.channel_id, parsed.message_id) == expected

    rejected = (
        f"text https://discord.com/channels/{expected[0]}/{expected[1]}/{expected[2]}",
        f"<https://discord.com/channels/{expected[0]}/{expected[1]}/{expected[2]}>",
        f"http://discord.com/channels/{expected[0]}/{expected[1]}/{expected[2]}",
        f"https://discord.com.evil.test/channels/{expected[0]}/{expected[1]}/{expected[2]}",
        f"https://discord.com:bad/channels/{expected[0]}/{expected[1]}/{expected[2]}",
        f"https://discord.com/channels/{expected[0]}/{expected[1]}/{expected[2]}?x=1",
        f"https://discord.com/channels/@me/{expected[1]}/{expected[2]}",
    )
    assert all(parse_discord_message_link(value) is None for value in rejected)


def test_custom_emoji_metadata_is_deduplicated_and_keeps_animation() -> None:
    content = (
        "a <:wave:111111111111111111> <a:dance:222222222222222222> <:wave_again:111111111111111111>"
    )
    records = _custom_emoji_records(content)
    assert [(item.index, item.emoji_id, item.name) for item in records] == [
        (0, "111111111111111111", "wave"),
        (1, "222222222222222222", "dance"),
    ]
    assert records[0].occurrences == 2
    assert records[0].animated is False
    assert records[1].occurrences == 1
    assert records[1].animated is True


def test_custom_emoji_metadata_is_bounded_to_25_unique_images() -> None:
    content = " ".join(f"<:e{index:02d}:{100000000000000000 + index}>" for index in range(30))
    records = _custom_emoji_records(content)
    assert len(records) == 25
    assert records[0].index == 0
    assert records[-1].index == 24


def test_animated_media_can_return_full_gif_or_an_exact_frame() -> None:
    frames = tuple(Image.new("RGBA", (4, 4), colour) for colour in ("red", "green", "blue"))
    source = io.BytesIO()
    frames[0].save(
        source,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=(40, 50, 60),
        loop=0,
    )
    content = source.getvalue()

    animation = _prepare_discord_animated_media(
        content,
        mode="animation",
        frame_index=0,
    )
    assert animation.content == content
    assert animation.content_type == "image/gif"
    assert animation.frame_count == 3
    assert animation.duration_ms == 150

    second = _prepare_discord_animated_media(
        content,
        mode="frame",
        frame_index=1,
    )
    assert second.content_type == "image/png"
    assert second.preview_kind == "selected_animation_frame"
    assert second.frame_index == 1
    assert second.frame_count == 3
    with Image.open(io.BytesIO(second.content)) as image:
        red, green, blue, _ = image.convert("RGBA").getpixel((0, 0))
    assert green > red
    assert green > blue

    with pytest.raises(ValueError, match=r"discord\.custom_emoji_frame_invalid"):
        _prepare_discord_animated_media(content, mode="frame", frame_index=3)


def test_animated_media_preserves_apng_when_full_animation_is_requested() -> None:
    frames = (
        Image.new("RGBA", (4, 4), "red"),
        Image.new("RGBA", (4, 4), "blue"),
    )
    source = io.BytesIO()
    frames[0].save(
        source,
        format="PNG",
        save_all=True,
        append_images=frames[1:],
        duration=(100, 120),
        loop=0,
    )
    animation = _prepare_discord_animated_media(
        source.getvalue(),
        mode="animation",
        frame_index=0,
    )
    assert animation.content_type == "image/apng"
    assert animation.frame_count == 2
    assert animation.duration_ms == 220


@pytest.mark.asyncio
async def test_custom_emoji_tool_fetches_only_selected_message_emoji(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = io.BytesIO()
    Image.new("RGBA", (4, 4), "purple").save(image, format="PNG")
    client = Mock(spec=discord.Client)
    client.http = Mock()
    client.http.get_from_cdn = AsyncMock(return_value=image.getvalue())
    guild = Mock(spec=discord.Guild)
    message = Mock(spec=discord.Message)
    message.content = "<:first:111111111111111111> <:second:222222222222222222>"
    message.stickers = []
    channel = Mock(spec=discord.TextChannel)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda selected_client, context: guild,
    )
    fetch = AsyncMock(return_value=(channel, message))
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._fetch_readable_message",
        fetch,
    )
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(client, Mock(spec=SimajilordRuntime))
    }
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        resource_ids=("10",),
    )

    response = cast(
        DiscordViewCustomEmojiResponse,
        await endpoints["discord.view_custom_emoji"].invoke(
            DiscordViewCustomEmojiRequest(
                channel_id="10",
                message_id="20",
                emoji_index=1,
            ),
            context,
        ),
    )

    assert response.emoji_id == "222222222222222222"
    assert response.name == "second"
    assert response.content_type == "image/png"
    assert response.frame_count == 1
    assert response.image_data_url.startswith("data:image/png;base64,")
    client.http.get_from_cdn.assert_awaited_once_with(
        "https://cdn.discordapp.com/emojis/222222222222222222.png?size=128&quality=lossless"
    )
    fetch.assert_awaited_once()


@pytest.mark.asyncio
async def test_sticker_tool_returns_full_gif_without_flattening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = (
        Image.new("RGBA", (4, 4), "red"),
        Image.new("RGBA", (4, 4), "blue"),
    )
    source = io.BytesIO()
    frames[0].save(
        source,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=(70, 80),
        loop=0,
    )
    client = Mock(spec=discord.Client)
    client.http = Mock()
    client.http.get_from_cdn = AsyncMock(return_value=source.getvalue())
    guild = Mock(spec=discord.Guild)
    sticker = Mock(spec=discord.StickerItem)
    sticker.id = 333333333333333333
    sticker.name = "dance"
    sticker.format = discord.StickerFormatType.gif
    sticker.url = "https://media.discordapp.net/stickers/333333333333333333.gif"
    message = Mock(spec=discord.Message)
    message.content = ""
    message.stickers = [sticker]
    channel = Mock(spec=discord.TextChannel)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda selected_client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._fetch_readable_message",
        AsyncMock(return_value=(channel, message)),
    )
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(client, Mock(spec=SimajilordRuntime))
    }
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        resource_ids=("10",),
    )

    response = cast(
        DiscordViewStickerResponse,
        await endpoints["discord.view_sticker"].invoke(
            DiscordViewStickerRequest(
                channel_id="10",
                message_id="20",
                mode="animation",
            ),
            context,
        ),
    )

    assert response.name == "dance"
    assert response.animated is True
    assert response.content_type == "image/gif"
    assert response.frame_count == 2
    assert response.duration_ms == 150
    assert response.image_data_url.startswith("data:image/gif;base64,")


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
    content = "こんにちは\n<simajilord:message-break>\nこんばんは"
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
    assert _agent_error_text(error).endswith("あと2分5秒ほどお待ちください。")
    assert _retry_after_text(3_661) == "1時間1分1秒"


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
    hidden.permissions_for.side_effect = lambda member: readable if member is bot_member else denied

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
    assert (
        agent_readable_channel_ids(
            guild,
            actor,
            trusted_guild=False,
            trigger_channel_id=30,
        )
        == ()
    )


def test_expanded_message_post_to_voice_chat_requires_connect() -> None:
    member = Mock(spec=discord.Member)
    voice = Mock(spec=discord.VoiceChannel)
    voice.permissions_for.return_value = discord.Permissions(
        view_channel=True,
        send_messages=True,
        embed_links=True,
        connect=False,
    )
    assert _can_post_expanded_message(voice, member) is False
    voice.permissions_for.return_value = discord.Permissions(
        view_channel=True,
        send_messages=True,
        embed_links=True,
        connect=True,
    )
    assert _can_post_expanded_message(voice, member) is True


def test_agent_tool_cannot_expand_the_runtime_resource_scope() -> None:
    context = InvocationContext(
        actor_id="30",
        workspace_id="10",
        transport="agent",
        request_id="40",
        resource_ids=("50",),
    )
    _assert_agent_channel_scope(context, "50")
    with pytest.raises(UserError, match="permission to view"):
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
