from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord import app_commands
from discord.ext import commands
from PIL import Image

from simajilord.agent import (
    ACTION_UNDO_ANY_GRANT,
    AGENT_AUDIO_GRANT,
    AGENT_COMPUTE_GRANT,
    AGENT_CONNECTOR_GRANT,
    AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES,
    AGENT_FILE_GRANT,
    AGENT_HIVE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_REACTION_GRANT,
    AGENT_REPOST_GRANT,
    AGENT_SHELL_GRANT,
    AGENT_WEB_GRANT,
    AgentAutonomyMode,
    AgentBusyError,
    AgentProgressStage,
    AgentProgressUpdate,
    AgentProviderLimitError,
    AgentRateLimitError,
    AgentTimeoutError,
)
from simajilord.capabilities.audio import (
    AudioAction,
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
from simajilord.capabilities.translation import (
    TranslationDetectResponse,
    TranslationLanguageItem,
    TranslationSegmentItem,
)
from simajilord.capabilities.web import WebFetchResponse
from simajilord.config import AgentFeatureAccess
from simajilord.core import ApprovalMode, CapabilityRegistry, InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem, LoopMode, QueueSnapshot
from simajilord.integrations.discord.bot import SimajilordDiscordBot
from simajilord.integrations.discord.capabilities import (
    DiscordImportAttachmentRequest,
    DiscordSendFileRequest,
    DiscordServerResponse,
    DiscordTranslatedSegmentRecord,
    DiscordTranslateMessageRequest,
    DiscordTranslateMessageResponse,
    DiscordUserResponse,
    DiscordViewCustomEmojiRequest,
    DiscordViewCustomEmojiResponse,
    DiscordViewStickerRequest,
    DiscordViewStickerResponse,
    _actor_member,
    _assert_agent_channel_scope,
    _assert_agent_update_scope,
    _attachment,
    _bounded_event_message,
    _can_post_expanded_message,
    _custom_emoji_records,
    _discord_event_message_id,
    _message_context_text,
    _message_preview,
    _prepare_discord_animated_media,
    _workspace_attachment_name,
    agent_readable_channel_ids,
    build_discord_endpoints,
    discord_translation_segments,
    parse_discord_message_link,
)
from simajilord.integrations.discord.cogs import (
    _QUOTE_CONTEXT_MENU_NAME,
    _TRANSLATE_CONTEXT_MENU_NAME,
    FocusTimerCog,
    HelpCog,
    InfoCog,
    LoopMixConflictView,
    MediaCog,
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
    SafeView,
    SystemCog,
    TranslationCog,
    TranslationLanguagePickerView,
    TranslationLanguageSelect,
    TranslationRegionSelect,
    UtilityCog,
    VoiceLifecycleCog,
    WebCog,
    WebFetchContinueView,
    _agent_delivery_nonce,
    _agent_error_text,
    _agent_grants,
    _agent_message_groups,
    _agent_progress_text,
    _AgentProgressMessage,
    _autonomy_approvals,
    _discord_message_chunks,
    _help_category_embed,
    _locale_target,
    _resolve_translation_target,
    _retry_after_text,
    _translation_detection_is_uncertain,
    _translation_detection_margin,
    _translation_result_embeds,
    _translation_target_autocomplete_choices,
    audio_control_capability_call,
    discord_conversation_id,
    edit_deferred_error,
    error_message,
    server_info_embed,
    translation_embed,
    user_info_embed,
)
from simajilord.integrations.discord.feedback import FeedbackCog
from simajilord.integrations.discord.help_catalog import (
    HELP_ENTRIES,
    HELP_ENTRIES_BY_TOPIC,
    PUBLIC_COMMAND_SPECS,
    PublicCommandSpec,
)
from simajilord.runtime import SimajilordRuntime
from simajilord.services.files import AgentFileSandbox
from simajilord.services.translation import TranslationPreference


def test_workspace_attachment_names_are_bounded_and_collision_free() -> None:
    first = Mock(spec=discord.Attachment)
    first.id = 101
    first.filename = f"{'a' * 250}.pdf"
    second = Mock(spec=discord.Attachment)
    second.id = 202
    second.filename = first.filename

    first_name = _workspace_attachment_name(first)
    second_name = _workspace_attachment_name(second)

    assert first_name != second_name
    assert len(first_name) <= 180
    assert first_name.endswith(".pdf")


@pytest.mark.asyncio
async def test_agent_imports_pdf_from_canonical_attachment_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"%PDF-1.7\ncanonical attachment"
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 1531959430940201000
    attachment.filename = "document.pdf"
    attachment.size = len(payload)

    async def read(*, use_cached: bool = False) -> bytes:
        if use_cached:
            raise AssertionError("the unsupported media proxy must not be primary")
        return payload

    attachment.read = AsyncMock(side_effect=read)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._attachment",
        AsyncMock(return_value=(Mock(spec=discord.Message), attachment)),
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "agent-files")
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }
    context = InvocationContext(
        actor_id="7",
        workspace_id="guild",
        transport="agent",
        request_id="discord:message:1531959431212961902",
    )

    response = await endpoints["discord.import_attachment"].invoke(
        DiscordImportAttachmentRequest(
            channel_id="1373866905357778984",
            message_id="1531959431212961902",
        ),
        context,
    )

    assert response.path == (
        "attachments/1531959431212961902/"
        "1531959430940201000-document.pdf"
    )
    assert runtime.files.path_for_delivery("guild", response.path).read_bytes() == payload
    attachment.read.assert_awaited_once_with(use_cached=False)


@pytest.mark.asyncio
async def test_agent_attachment_import_rechecks_downloaded_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 123
    attachment.filename = "payload.bin"
    attachment.size = 1
    attachment.read = AsyncMock(return_value=b"12345")
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._attachment",
        AsyncMock(return_value=(Mock(spec=discord.Message), attachment)),
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "agent-files", max_file_bytes=4)
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }

    with pytest.raises(UserError, match=r"files\.file_too_large"):
        await endpoints["discord.import_attachment"].invoke(
            DiscordImportAttachmentRequest(
                channel_id="1",
                message_id="2",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="guild",
                transport="agent",
                request_id="event",
            ),
        )
    assert runtime.files.list("guild") == ()


@pytest.mark.asyncio
async def test_agent_file_delivery_uses_the_guild_upload_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "agent-files")
    runtime.files.import_bytes("guild", "result.bin", b"12345")
    guild = Mock(spec=discord.Guild)
    guild.filesize_limit = 4
    channel = Mock(spec=discord.TextChannel)
    channel.send = AsyncMock()
    actor = Mock(spec=discord.Member)
    bot = Mock(spec=discord.Member)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._write_message_channel",
        AsyncMock(return_value=(guild, channel, actor, bot)),
    )
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }

    with pytest.raises(UserError, match=r"discord\.file_too_large"):
        await endpoints["discord.send_file"].invoke(
            DiscordSendFileRequest(channel_id="1", path="result.bin"),
            InvocationContext(
                actor_id="7",
                workspace_id="guild",
                transport="agent",
                request_id="event",
                resource_ids=("1",),
                origin_resource_id="1",
            ),
        )
    channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_file_delivery_sends_the_authorized_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "agent-files")
    runtime.files.import_bytes("guild", "result.bin", b"authorized")
    guild = Mock(spec=discord.Guild)
    guild.filesize_limit = 100
    channel = Mock(spec=discord.TextChannel)
    channel.id = 1
    actor = Mock(spec=discord.Member)
    bot = Mock(spec=discord.Member)
    sent = Mock(spec=discord.Message)
    sent.id = 99

    async def send(
        content: str | None,
        *,
        file: discord.File,
        nonce: str,
        allowed_mentions: discord.AllowedMentions,
        suppress_embeds: bool,
    ) -> discord.Message:
        del content, allowed_mentions
        assert suppress_embeds is True
        assert nonce.startswith("sla")
        assert len(nonce) == 25
        runtime.files.import_bytes("guild", "result.bin", b"newer")
        assert file.fp.read() == b"authorized"
        return sent

    channel.send = AsyncMock(side_effect=send)
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._write_message_channel",
        AsyncMock(return_value=(guild, channel, actor, bot)),
    )
    endpoints = {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            cast(discord.Client, object()),
            runtime,
        )
    }

    response = await endpoints["discord.send_file"].invoke(
        DiscordSendFileRequest(channel_id="1", path="result.bin"),
        InvocationContext(
            actor_id="7",
            workspace_id="guild",
            transport="agent",
            request_id="event",
            resource_ids=("1",),
            origin_resource_id="1",
        ),
    )

    assert response.message_id == "99"
    assert response.size_bytes == len(b"authorized")


@pytest.mark.asyncio
async def test_attachment_read_rechecks_live_actor_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild = Mock(spec=discord.Guild)
    channel = Mock(spec=discord.TextChannel)
    actor = Mock(spec=discord.Member)
    bot = Mock(spec=discord.Member)
    guild.me = bot
    guild.get_channel_or_thread.return_value = channel
    channel.permissions_for.return_value = discord.Permissions.none()
    channel.fetch_message = AsyncMock()
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._guild",
        lambda client, context: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.capabilities._actor_member",
        AsyncMock(return_value=actor),
    )

    with pytest.raises(
        UserError,
        match=r"discord\.agent_read_channel_forbidden",
    ):
        await _attachment(
            cast(discord.Client, object()),
            InvocationContext(
                actor_id="7",
                workspace_id="guild",
                transport="agent",
                request_id="event",
                resource_ids=("1",),
            ),
            "1",
            "2",
            0,
        )
    channel.fetch_message.assert_not_awaited()


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
    source.reply = AsyncMock()
    published = Mock(spec=discord.Message)
    published.id = 43
    published.delete = AsyncMock()
    progress = _AgentProgressMessage(source)
    progress.message = published

    await progress.finish("<simajilord:no-action>")

    published.delete.assert_awaited_once_with()
    source.reply.assert_not_awaited()
    assert progress.message is None


@pytest.mark.asyncio
async def test_agent_tool_final_sentinel_is_never_published() -> None:
    source = Mock(spec=discord.Message)
    source.id = 42
    source.reply = AsyncMock()
    published = Mock(spec=discord.Message)
    published.id = 43
    published.delete = AsyncMock()
    progress = _AgentProgressMessage(source)
    progress.message = published

    await progress.finish("<simajilord:final-delivered>")

    published.delete.assert_awaited_once_with()
    source.reply.assert_not_awaited()
    assert progress.message is None


@pytest.mark.asyncio
async def test_agent_working_panel_contains_only_live_progress() -> None:
    source = Mock(spec=discord.Message)
    source.id = 42
    source.reply = AsyncMock(return_value=Mock(spec=discord.Message))
    source.channel = Mock()
    source.channel.typing = AsyncMock()
    progress = _AgentProgressMessage(
        source,
        initial_delay_seconds=0,
    )

    await progress.update(AgentProgressUpdate(AgentProgressStage.STARTING))
    assert progress._task is not None
    await progress._task

    source.reply.assert_awaited_once()
    kwargs = source.reply.await_args.kwargs
    embed = kwargs["embed"]
    assert embed.title == "Working"
    assert "Checking your request" in embed.description
    assert embed.fields == []
    assert "view" not in kwargs
    await progress.prepare("done")


@pytest.mark.asyncio
async def test_agent_cancellation_replaces_working_and_removes_dead_control() -> None:
    source = Mock(spec=discord.Message)
    source.id = 42
    source.reply = AsyncMock()
    published = Mock(spec=discord.Message)
    published.id = 43
    published.edit = AsyncMock()
    progress = _AgentProgressMessage(
        source,
    )
    progress.message = published

    await progress.cancelled()

    published.edit.assert_awaited_once()
    assert "view" not in published.edit.await_args.kwargs
    embed = published.edit.await_args.kwargs["embed"]
    assert embed.title == "AI task cancelled"
    assert embed.fields == []
    source.reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_final_is_posted_after_temporary_progress_is_deleted() -> None:
    order: list[str] = []
    source = Mock(spec=discord.Message)
    source.id = 42
    source.channel = Mock()
    source.channel.send = AsyncMock()
    final_message = Mock(spec=discord.Message)
    final_message.id = 45

    async def reply(*args: object, **kwargs: object) -> Mock:
        del args, kwargs
        order.append("final")
        return final_message

    source.reply = AsyncMock(side_effect=reply)
    published = Mock(spec=discord.Message)
    published.id = 43

    async def delete() -> None:
        order.append("delete-progress")

    published.delete = AsyncMock(side_effect=delete)
    published.edit = AsyncMock()
    on_posted = AsyncMock()
    progress = _AgentProgressMessage(source, on_posted=on_posted)
    progress.message = published

    await progress.finish("Final answer")

    assert order == ["delete-progress", "final"]
    published.edit.assert_not_awaited()
    source.reply.assert_awaited_once()
    assert source.reply.await_args.kwargs["nonce"] == _agent_delivery_nonce(
        "discord:message:42",
        0,
    )
    assert source.reply.await_args.kwargs["suppress_embeds"] is True
    source.channel.send.assert_not_awaited()
    on_posted.assert_awaited_once_with(final_message)
    assert progress.message is None


@pytest.mark.asyncio
async def test_agent_receipt_failure_does_not_repost_a_delivered_final() -> None:
    source = Mock(spec=discord.Message)
    source.id = 42
    source.channel = Mock()
    source.channel.send = AsyncMock()
    final_message = Mock(spec=discord.Message)
    final_message.id = 45
    source.reply = AsyncMock(return_value=final_message)
    on_posted = AsyncMock(side_effect=RuntimeError("receipt unavailable"))
    progress = _AgentProgressMessage(source, on_posted=on_posted)

    await progress.finish("Final answer")

    source.reply.assert_awaited_once()
    source.channel.send.assert_not_awaited()
    on_posted.assert_awaited_once_with(final_message)


@pytest.mark.asyncio
async def test_agent_failure_replaces_working_with_a_new_reply() -> None:
    order: list[str] = []
    source = Mock(spec=discord.Message)
    source.id = 42

    async def reply(*args: object, **kwargs: object) -> Mock:
        del args, kwargs
        order.append("failure")
        return Mock(spec=discord.Message)

    source.reply = AsyncMock(side_effect=reply)
    published = Mock(spec=discord.Message)
    published.id = 43

    async def delete() -> None:
        order.append("delete-progress")

    published.delete = AsyncMock(side_effect=delete)
    published.edit = AsyncMock()
    on_posted = AsyncMock()
    progress = _AgentProgressMessage(source, on_posted=on_posted)
    progress.message = published

    await progress.fail("Could not finish")

    assert order == ["delete-progress", "failure"]
    published.edit.assert_not_awaited()
    source.reply.assert_awaited_once()
    assert source.reply.await_args.kwargs["nonce"] == _agent_delivery_nonce(
        "discord:message:42",
        0,
        purpose="error",
    )
    assert source.reply.await_args.kwargs["suppress_embeds"] is True
    on_posted.assert_not_awaited()
    assert progress.message is None


def test_agent_queue_progress_shows_same_server_wait_position() -> None:
    text = _agent_progress_text(
        AgentProgressUpdate(
            AgentProgressStage.QUEUED,
            queue_position=3,
        )
    )

    assert "Requests ahead of you" in text
    assert "**3**" in text


def test_member_lookup_error_is_clear_and_english() -> None:
    message = error_message(UserError("discord.member_required"))
    assert message == "Could not resolve that member in this server."


def test_unexpected_error_displays_the_logged_reference_id(caplog) -> None:
    message = error_message(
        RuntimeError("unexpected"),
        request_id="interaction-42",
    )

    assert "Reference ID: `interaction-42`" in message
    assert "Share this ID with the administrator." in message
    assert any(
        "request_id=interaction-42" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_view_error_boundary_returns_reference_id_to_user() -> None:
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 987
    interaction.response = Mock()
    interaction.response.is_done.return_value = False
    interaction.response.send_message = AsyncMock()

    await SafeView().on_error(
        interaction,
        RuntimeError("callback failed"),
        Mock(spec=discord.ui.Item),
    )

    interaction.response.send_message.assert_awaited_once()
    call = interaction.response.send_message.await_args
    assert call is not None
    assert call.kwargs["ephemeral"] is True
    assert "Reference ID: `987`" in (call.kwargs["embed"].description or "")


@pytest.mark.asyncio
async def test_expired_deferred_error_response_does_not_escape() -> None:
    response = SimpleNamespace(status=404, reason="Not Found")
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 988
    interaction.response = Mock()
    interaction.response.is_done.return_value = True
    interaction.edit_original_response = AsyncMock(
        side_effect=discord.NotFound(
            response,
            {"code": 10015, "message": "Unknown Webhook"},
        )
    )

    await edit_deferred_error(
        interaction,
        UserError("translation.message_text_required"),
    )

    interaction.edit_original_response.assert_awaited_once()


def test_autonomous_agent_grants_follow_typed_host_mode() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.settings.agent_file_sandbox_enabled = True
    runtime.settings.agent_web_search_access = AgentFeatureAccess.EVERYONE
    runtime.settings.agent_safe_compute_access = AgentFeatureAccess.EVERYONE
    runtime.settings.agent_isolated_shell_access = AgentFeatureAccess.ADMINS
    runtime.settings.agent_connector_access = AgentFeatureAccess.ADMINS
    runtime.settings.agent_admin_user_ids = frozenset({"7"})
    runtime.settings.image_generation_access = AgentFeatureAccess.EVERYONE
    runtime.files = object()
    runtime.compute = object()
    runtime.connectors = object()
    runtime.moderation.provider = object()
    runtime.image.provider = object()

    requested = _agent_grants(runtime, actor_id="7")
    runtime.settings.agent_autonomy_mode = AgentAutonomyMode.ASSIST
    assist = _agent_grants(runtime, actor_id="99", autonomous=True)
    runtime.settings.agent_autonomy_mode = AgentAutonomyMode.ACT
    act = _agent_grants(runtime, actor_id="99", autonomous=True)

    assert {
        AGENT_AUDIO_GRANT,
        AGENT_WEB_GRANT,
        AGENT_HIVE_GRANT,
        AGENT_MESSAGE_GRANT,
        AGENT_REACTION_GRANT,
    } <= assist
    assert AGENT_MODERATION_GRANT not in assist
    assert {
        AGENT_MESSAGE_GRANT,
        AGENT_REACTION_GRANT,
        AGENT_FILE_GRANT,
        AGENT_IMAGE_GRANT,
        AGENT_CONNECTOR_GRANT,
        AGENT_SHELL_GRANT,
    } <= requested
    assert not {AGENT_FILE_GRANT, AGENT_IMAGE_GRANT} & assist
    assert {
        AGENT_FILE_GRANT,
        AGENT_IMAGE_GRANT,
        AGENT_COMPUTE_GRANT,
        AGENT_MODERATION_GRANT,
        AGENT_REPOST_GRANT,
    } <= act
    assert AGENT_COMPUTE_GRANT in requested
    assert AGENT_COMPUTE_GRANT not in assist
    assert not {AGENT_CONNECTOR_GRANT, AGENT_SHELL_GRANT} & (assist | act)
    assert ACTION_UNDO_ANY_GRANT in requested
    assert ACTION_UNDO_ANY_GRANT not in assist


def test_discord_moderation_grant_does_not_depend_on_hive_provider() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.settings.agent_file_sandbox_enabled = False
    runtime.settings.agent_web_search_access = AgentFeatureAccess.DISABLED
    runtime.settings.agent_safe_compute_access = AgentFeatureAccess.DISABLED
    runtime.settings.agent_isolated_shell_access = AgentFeatureAccess.DISABLED
    runtime.settings.agent_connector_access = AgentFeatureAccess.DISABLED
    runtime.settings.agent_admin_user_ids = frozenset()
    runtime.settings.image_generation_access = AgentFeatureAccess.DISABLED
    runtime.settings.agent_autonomy_mode = AgentAutonomyMode.ACT
    runtime.files = None
    runtime.compute = None
    runtime.connectors = None
    runtime.moderation.provider = None
    runtime.image.provider = None

    assert AGENT_MODERATION_GRANT in _agent_grants(runtime, actor_id="7")
    assert AGENT_MODERATION_GRANT in _agent_grants(
        runtime,
        actor_id="99",
        autonomous=True,
    )


def test_destructive_approvals_are_exposed_only_to_act_autonomy() -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry.all.return_value = tuple(
        SimpleNamespace(
            descriptor=SimpleNamespace(
                name=name,
                approval=ApprovalMode.WHEN_REQUESTED,
            )
        )
        for name in AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES
    )

    assist = _autonomy_approvals(runtime, AgentAutonomyMode.ASSIST)
    act = _autonomy_approvals(runtime, AgentAutonomyMode.ACT)

    assert not set(AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES) & assist
    assert set(AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES) <= act


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
        "Add a song, public URL, or attached audio/video to the queue."
    )
    assert commands["radio"].description.startswith("Keep related music")


def test_every_public_slash_command_has_exactly_one_help_entry() -> None:
    cog_types = (
        HelpCog,
        FeedbackCog,
        SystemCog,
        FocusTimerCog,
        MusicCog,
        ReadAloudCog,
        WebCog,
        TranslationCog,
        MediaCog,
        UtilityCog,
        InfoCog,
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
    assert HELP_ENTRIES is PUBLIC_COMMAND_SPECS
    assert all(isinstance(entry, PublicCommandSpec) for entry in HELP_ENTRIES)

    for topic, command in command_by_topic.items():
        spec = HELP_ENTRIES_BY_TOPIC[topic.casefold()]
        usage = spec.usage
        assert spec.permissions
        assert spec.common_errors
        for parameter in command.parameters:
            assert parameter.description and parameter.description != "…", (
                f"/{topic} option `{parameter.name}` has no Discord description"
            )
            assert parameter.name in usage, f"/{topic} help omits the `{parameter.name}` option"


def test_public_command_and_option_descriptions_use_the_official_english_surface() -> None:
    japanese = re.compile(r"[ぁ-んァ-ヶ一-龯]")
    cog_types = (
        HelpCog,
        FeedbackCog,
        SystemCog,
        FocusTimerCog,
        MusicCog,
        ReadAloudCog,
        WebCog,
        TranslationCog,
        MediaCog,
        UtilityCog,
        InfoCog,
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
        embed = _help_category_embed(category)
        command_fields = tuple(
            field for field in embed.fields if field.name.startswith("Commands")
        )
        assert command_fields
        assert all(1 <= len(field.value) <= 1024 for field in command_fields)
        assert sum(field.value.count("\n") + 1 for field in command_fields) == len(entries)


def test_prefix_surface_is_derived_from_canonical_primary_commands() -> None:
    registered = {command.name for command in PrefixCog.__cog_commands__}
    canonical = {
        entry.prefix_name for entry in PUBLIC_COMMAND_SPECS if entry.prefix_name is not None
    }
    assert registered == canonical == {"help", "audio", "play"}


def test_translation_uses_short_canonical_names() -> None:
    commands = {
        command.name: command
        for command in TranslationCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    }
    assert set(commands) == {"translate"}
    assert _TRANSLATE_CONTEXT_MENU_NAME == "Translate"


def test_translation_target_accepts_code_english_and_native_name() -> None:
    languages = (
        TranslationLanguageItem("en-GB", "English", "English", "installed"),
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
    )

    assert _resolve_translation_target("EN_gb", languages) == "en-GB"
    assert _resolve_translation_target("English", languages) == "en"
    assert _resolve_translation_target("日本語", languages) == "ja"
    assert _resolve_translation_target("🇯🇵 Japanese (ja)", languages) == "ja"
    with pytest.raises(UserError, match=r"translation\.language_invalid"):
        _resolve_translation_target("not-a-language", languages)


def test_translation_context_menu_groups_every_language_by_region() -> None:
    codes = (
        "ar-AE",
        "zh-TW",
        "zh-HK",
        "zh",
        "da",
        "nl",
        "en-IN",
        "en-CA",
        "en-SG",
        "en-GB",
        "en-ZA",
        "en-AU",
        "en",
        "en-IE",
        "en-NZ",
        "fr-CA",
        "fr",
        "de-CH",
        "de",
        "hi",
        "id",
        "it-CH",
        "it",
        "ja",
        "ko",
        "nb",
        "pl",
        "pt",
        "pt-PT",
        "ru",
        "es-MX",
        "es",
        "es-US",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
    )
    languages = tuple(
        TranslationLanguageItem(
            code=code,
            english_name=f"Language {index:02d}",
            native_name=f"Native {index:02d}",
            availability="installed",
        )
        for index, code in enumerate(codes)
    )
    message = Mock(spec=discord.Message)
    message.jump_url = "https://discord.com/channels/1/2/3"
    view = TranslationLanguagePickerView(
        Mock(spec=TranslationCog),
        requester_id=7,
        message=message,
        languages=languages,
        source_language="en",
        target_language="ja",
        show_original=False,
        mode="target",
    )

    assert any(isinstance(item, TranslationRegionSelect) for item in view.children)
    selects = tuple(
        TranslationLanguageSelect(region, languages)
        for region in (
            "Asia & Pacific",
            "Europe",
            "Americas",
            "Middle East & Africa",
        )
    )
    assert all(1 <= len(select.options) <= 25 for select in selects)
    assert {option.value for select in selects for option in select.options} == set(codes)
    assert all(option.emoji is not None for select in selects for option in select.options)
    americas = next(select for select in selects if select.placeholder.endswith("Americas"))
    assert {option.value for option in americas.options} >= {"en", "pt"}
    assert next(option for option in americas.options if option.value == "en").emoji.name == "🇺🇸"
    assert next(option for option in americas.options if option.value == "pt").emoji.name == "🇧🇷"


def test_translation_region_picker_has_no_next_or_previous_controls() -> None:
    languages = (
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
        TranslationLanguageItem("en", "English", "English", "installed"),
    )
    message = Mock(spec=discord.Message)
    message.jump_url = "https://discord.com/channels/1/2/3"
    view = TranslationLanguagePickerView(
        Mock(spec=TranslationCog),
        requester_id=7,
        message=message,
        languages=languages,
        source_language="en",
        target_language="ja",
        show_original=False,
        mode="source",
    )
    labels = {item.label for item in view.children if isinstance(item, discord.ui.Button)}
    assert "Next" not in labels
    assert "Previous" not in labels


def test_translation_result_hides_backend_metadata() -> None:
    embed = translation_embed(
        original="Hello",
        translation="こんにちは",
        source_language="en",
        target_language="ja",
    )

    assert embed.title == "Translation · en → ja"
    assert embed.footer.text is None


@pytest.mark.asyncio
async def test_translation_target_matching_detection_requests_source_without_timeout() -> None:
    languages = (
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
    )
    cog = Mock(spec=TranslationCog)
    cog._translate_message_from_language_override = AsyncMock()
    message = Mock(spec=discord.Message)
    message.jump_url = "https://discord.com/channels/1/2/3"
    interaction = Mock(spec=discord.Interaction)
    interaction.response.edit_message = AsyncMock()
    view = TranslationLanguagePickerView(
        cog,
        requester_id=7,
        message=message,
        languages=languages,
        source_language="ja",
        target_language="en",
        show_original=False,
        mode="target",
    )

    await view.choose_language(interaction, "ja")

    interaction.response.edit_message.assert_awaited_once()
    response = interaction.response.edit_message.await_args.kwargs
    assert response["embed"].title == "Choose the source language"
    assert response["embed"].description
    assert isinstance(response["view"], TranslationLanguagePickerView)
    assert response["view"].mode == "source"
    assert response["view"].target_language == "ja"
    cog._translate_message_from_language_override.assert_not_awaited()


@pytest.mark.asyncio
async def test_translation_language_picker_acknowledges_nonempty_errors() -> None:
    languages = (
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
    )
    cog = Mock(spec=TranslationCog)
    cog._translate_message_from_language_override = AsyncMock(
        side_effect=UserError("translation.language_pair_unsupported")
    )
    message = Mock(spec=discord.Message)
    message.jump_url = "https://discord.com/channels/1/2/3"
    interaction = Mock(spec=discord.Interaction)
    interaction.response.is_done.return_value = False
    interaction.response.send_message = AsyncMock()
    view = TranslationLanguagePickerView(
        cog,
        requester_id=7,
        message=message,
        languages=languages,
        source_language="en",
        target_language="en",
        show_original=False,
        mode="target",
    )

    await view.choose_language(interaction, "ja")

    interaction.response.send_message.assert_awaited_once()
    response = interaction.response.send_message.await_args.kwargs
    assert response["ephemeral"] is True
    assert response["embed"].title == "Could not complete the request"
    assert response["embed"].description
    assert "does not support" in response["embed"].description
    assert response["embed"].description != "translation.language_pair_unsupported"


@pytest.mark.asyncio
async def test_automatic_message_translation_does_not_fix_detected_source() -> None:
    languages = (
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
    )
    registry = SimpleNamespace(
        invoke=AsyncMock(
            side_effect=(
                TranslationDetectResponse(
                    language="en",
                    confidence=0.999,
                    hypotheses=(("en", 0.999), ("nl", 0.001)),
                ),
                DiscordTranslateMessageResponse(
                    message_id="3",
                    channel_id="2",
                    jump_url="https://discord.com/channels/1/2/3",
                    author_name="Author",
                    original="This is a sufficiently long and unambiguous English sentence.",
                    translation="これは十分に長く、明確な英語の文です。",
                    source_language="en",
                    target_language="ja",
                    segments=(
                        DiscordTranslatedSegmentRecord(
                            identifier="content",
                            original=(
                                "This is a sufficiently long and unambiguous English sentence."
                            ),
                            translation="これは十分に長く、明確な英語の文です。",
                        ),
                    ),
                ),
            )
        )
    )
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(
            registry=registry,
            translation=SimpleNamespace(
                recent_targets=AsyncMock(return_value=()),
                record_recent_target=AsyncMock(),
            ),
            journal=SimpleNamespace(append=AsyncMock()),
        ),
    )
    cog = TranslationCog(runtime)
    cog._languages = AsyncMock(side_effect=(languages, languages))
    cog._default_translation_settings = AsyncMock(return_value=("ja", False))
    interaction = cast(
        discord.Interaction,
        SimpleNamespace(
            response=SimpleNamespace(send_message=AsyncMock()),
            edit_original_response=AsyncMock(),
            client=Mock(spec=discord.Client),
            user=SimpleNamespace(id=7),
            guild_id=1,
            channel_id=2,
            id=4,
        ),
    )
    message = cast(
        discord.Message,
        SimpleNamespace(
            id=3,
            channel=SimpleNamespace(id=2),
            content="This is a sufficiently long and unambiguous English sentence.",
            embeds=[],
            poll=None,
            components=[],
            attachments=[],
            jump_url="https://discord.com/channels/1/2/3",
        ),
    )

    await cog.translate_message(interaction, message)

    assert registry.invoke.await_count == 2
    capability_name, request, _ = registry.invoke.await_args_list[1].args
    assert capability_name == "discord.translate_message"
    assert isinstance(request, DiscordTranslateMessageRequest)
    assert request.source_language is None
    assert request.target_language == "ja"


def test_translation_slash_autocomplete_filters_all_languages_and_caps_results() -> None:
    languages = tuple(
        TranslationLanguageItem(
            code=f"x-{index}",
            english_name=f"Language {index:02d}",
            native_name=f"Native {index:02d}",
            availability="installed",
        )
        for index in range(38)
    )

    assert len(_translation_target_autocomplete_choices(languages, "")) == 25
    assert [
        choice.value
        for choice in _translation_target_autocomplete_choices(
            languages,
            "Language 37",
        )
    ] == ["x-37"]
    assert [
        choice.value
        for choice in _translation_target_autocomplete_choices(
            languages,
            "Native 36",
        )
    ] == ["x-36"]
    japanese = _translation_target_autocomplete_choices(
        (
            TranslationLanguageItem(
                "ja",
                "Japanese",
                "日本語",
                "installed",
            ),
        ),
        "",
    )
    assert japanese[0].name.startswith("🇯🇵 Japanese")


def test_translation_detection_treats_short_or_close_hypotheses_as_uncertain() -> None:
    short = TranslationDetectResponse(
        language="fr",
        confidence=0.127,
        hypotheses=(("fr", 0.127), ("nl", 0.113), ("en", 0.112)),
    )
    close = TranslationDetectResponse(
        language="ja",
        confidence=0.88,
        hypotheses=(("ja", 0.88), ("zh", 0.76)),
    )
    reliable = TranslationDetectResponse(
        language="en",
        confidence=0.999,
        hypotheses=(("en", 0.999), ("nl", 0.001)),
    )

    assert _translation_detection_margin(short) == pytest.approx(0.014)
    assert _translation_detection_is_uncertain("test", short) is True
    assert _translation_detection_is_uncertain("これは短いテストです", close) is True
    assert (
        _translation_detection_is_uncertain(
            "This is a sufficiently long and unambiguous English sentence.",
            reliable,
        )
        is False
    )


def test_translation_locale_resolution_supports_exact_base_and_chinese_script() -> None:
    languages = (
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem(
            "zh-Hans",
            "Chinese, Simplified",
            "简体中文",
            "installed",
        ),
        TranslationLanguageItem(
            "zh-Hant",
            "Chinese, Traditional",
            "繁體中文",
            "installed",
        ),
        TranslationLanguageItem("pt-BR", "Portuguese", "Português", "installed"),
    )

    assert _locale_target(("pt-BR",), languages) == "pt-BR"
    assert _locale_target(("en-US",), languages) == "en"
    assert _locale_target(("zh-TW",), languages) == "zh-Hant"
    assert _locale_target(("zh-CN",), languages) == "zh-Hans"


@pytest.mark.asyncio
async def test_translation_default_restores_target_and_original_preference() -> None:
    languages = (
        TranslationLanguageItem("en", "English", "English", "installed"),
        TranslationLanguageItem("fr", "French", "Français", "installed"),
        TranslationLanguageItem("ja", "Japanese", "日本語", "installed"),
    )
    translation = SimpleNamespace(
        preference=AsyncMock(
            return_value=TranslationPreference(
                target_language="fr",
                show_original=True,
            )
        )
    )
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(translation=translation),
    )
    cog = TranslationCog(runtime)
    interaction = cast(
        discord.Interaction,
        SimpleNamespace(
            user=SimpleNamespace(id=7),
            guild_id=11,
            locale=SimpleNamespace(value="ja"),
            guild_locale=SimpleNamespace(value="en-US"),
        ),
    )

    assert await cog._default_translation_settings(interaction, languages) == (
        "fr",
        True,
    )
    translation.preference.assert_awaited_once_with(
        actor_id="7",
        workspace_id="11",
    )


def test_structured_translation_extracts_and_rebuilds_discord_message() -> None:
    embed = discord.Embed(
        title="Title",
        description="Description",
        colour=discord.Colour.blurple(),
    )
    embed.set_author(name="Author")
    embed.add_field(name="Name", value="Value", inline=True)
    embed.set_footer(text="Footer")
    embed.set_image(url="https://example.com/image.png")
    component = SimpleNamespace(
        to_dict=lambda: {
            "type": 17,
            "components": [
                {"type": 10, "content": "Component text"},
            ],
        }
    )
    message = cast(
        discord.Message,
        SimpleNamespace(
            content="Content",
            embeds=[embed],
            poll=SimpleNamespace(
                question="Question",
                answers=[SimpleNamespace(text="Answer")],
            ),
            components=[component],
            attachments=[SimpleNamespace(description="Alt text")],
        ),
    )
    segments = discord_translation_segments(message)
    response = DiscordTranslateMessageResponse(
        message_id="1",
        channel_id="2",
        jump_url="https://discord.com/channels/1/2/3",
        author_name="Meteo",
        original="\n".join(item.text for item in segments),
        translation="\n".join(f"T:{item.text}" for item in segments),
        source_language="en",
        target_language="ja",
        segments=tuple(
            DiscordTranslatedSegmentRecord(
                identifier=item.identifier,
                original=item.text,
                translation=f"T:{item.text}",
            )
            for item in segments
        ),
    )

    assert tuple(item.identifier for item in segments) == (
        "content",
        "embed.0.author",
        "embed.0.title",
        "embed.0.description",
        "embed.0.field.0.name",
        "embed.0.field.0.value",
        "embed.0.footer",
        "poll.question",
        "poll.answer.0",
        "component.0.0.content",
        "attachment.0.description",
    )
    rendered = _translation_result_embeds(message, response, show_original=False)
    translated_embed = rendered[1]
    assert rendered[0].description == "T:Content"
    assert rendered[0].footer.text is None
    assert translated_embed.title == "T:Title"
    assert translated_embed.description == "T:Description"
    assert translated_embed.author.name == "T:Author"
    assert translated_embed.fields[0].name == "T:Name"
    assert translated_embed.fields[0].value == "T:Value"
    assert translated_embed.footer.text == "T:Footer"
    assert translated_embed.colour == discord.Colour.blurple()
    assert translated_embed.image.url == "https://example.com/image.png"
    assert all(item.title != "Original" for item in rendered)
    assert (
        _translation_result_embeds(
            message,
            response,
            show_original=True,
        )[-1].title
        == "Original"
    )


def test_structured_translation_uses_rendered_system_message_content() -> None:
    message = cast(
        discord.Message,
        SimpleNamespace(
            content="",
            is_system=lambda: True,
            system_content="Make it a Quote joined the server.",
            embeds=[],
            poll=None,
            components=[],
            attachments=[],
        ),
    )

    assert discord_translation_segments(message) == (
        TranslationSegmentItem(
            identifier="content",
            text="Make it a Quote joined the server.",
        ),
    )


def test_human_audio_controls_map_to_exact_agent_capabilities() -> None:
    calls = (
        (AudioAction.PAUSE, {}, "discord.pause_audio"),
        (AudioAction.RESUME, {}, "discord.resume_audio"),
        (AudioAction.SKIP, {}, "discord.skip_audio"),
        (AudioAction.STOP, {}, "discord.stop_audio"),
        (AudioAction.LEAVE, {}, "discord.leave_audio"),
        (
            AudioAction.LOOP,
            {"loop_mode": LoopMode.TRACK},
            "discord.set_audio_loop",
        ),
        (AudioAction.REMOVE, {"position": 1}, "discord.remove_audio"),
        (
            AudioAction.AUTO_LEAVE,
            {"enabled": True},
            "discord.set_audio_auto_leave",
        ),
        (AudioAction.SHUFFLE, {}, "discord.shuffle_audio"),
        (
            AudioAction.SEEK,
            {"position_seconds": 12.0},
            "discord.seek_audio",
        ),
        (
            AudioAction.TUNE,
            {"speed": 1.0, "pitch": 1.0},
            "discord.tune_audio",
        ),
        (
            AudioAction.VOLUME,
            {"music_percent": 80},
            "discord.set_audio_volume",
        ),
        (
            AudioAction.MOVE,
            {"position": 2, "to_position": 1},
            "discord.move_audio",
        ),
        (AudioAction.CLEAR_MINE, {}, "discord.clear_my_audio"),
    )

    assert {
        audio_control_capability_call(action, **arguments)[0] for action, arguments, _ in calls
    } == {expected for _, _, expected in calls}


def test_japanese_requests_find_the_intended_discord_capability() -> None:
    registry = CapabilityRegistry()
    for selected_endpoint in build_discord_endpoints(
        cast(discord.Client, object()),
        Mock(spec=SimajilordRuntime),
    ):
        registry.register(selected_endpoint)

    expected_by_query = {
        "この曲流して": "discord.play_audio",
        "読み上げを入退室だけにして": ("discord.read_aloud_announcements_set"),
        "音楽を少し下げて": "discord.set_audio_volume",
        "このメッセージをドイツ語にして": "discord.translate_message",
        "この添付音声を流して": "discord.play_attachment",
    }

    for query, expected in expected_by_query.items():
        matches = registry.search(query, limit=1)
        assert matches
        assert matches[0].descriptor.name == expected


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
    assert not next(field for field in embed.fields if field.name == "Server membership").inline
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
async def test_administrator_bypasses_individual_manage_server_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = ReadAloudResponse(
        action=ReadAloudAction.DISABLE.value,
        enabled=False,
        text_channel_id=None,
        text_channel_ids=(),
        audio_destination_id=None,
        mode=None,
    )
    runtime = Mock(spec=SimajilordRuntime)
    runtime.registry = Mock()
    runtime.registry.invoke = AsyncMock(return_value=response)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    member = Mock(spec=discord.Member)
    member.guild_permissions = discord.Permissions(administrator=True)
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

    result = await endpoint_by_name["discord.manage_read_aloud"].invoke(
        ReadAloudRequest(action=ReadAloudAction.DISABLE),
        InvocationContext(
            actor_id="7",
            workspace_id="1",
            transport="agent",
            request_id="event",
        ),
    )

    assert result == response
    runtime.registry.invoke.assert_awaited_once()


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
    bot = Mock(spec=discord.Member)
    guild.me = bot
    source.permissions_for.return_value = discord.Permissions(
        view_channel=True,
        read_message_history=True,
    )
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


def test_advanced_music_commands_are_not_duplicated_as_a_slash_group() -> None:
    assert not any(
        isinstance(command, app_commands.Group) and command.name == "music"
        for command in MusicCog.__cog_app_commands__
    )


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


def test_music_panel_exposes_official_activity_only_when_enabled() -> None:
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
    runtime = SimpleNamespace(settings=SimpleNamespace(activity_enabled=True))

    view = MusicControlsView(
        cast(SimajilordRuntime, runtime),
        response=response,
    )
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
    open_player = next(button for button in buttons if button.label == "Open Player")

    assert open_player.custom_id == "simajilord:audio:open-player"
    assert open_player.row == 2
    assert [button.label for button in buttons if button.row == 0] == [
        "Pause",
        "Skip",
        "Stop",
        "Add music",
    ]


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
        isinstance(child, discord.ui.Select) and child.placeholder == "More actions"
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
        voice_activation_required=True,
        connected=False,
    )
    view = MusicControlsView(
        cast(SimajilordRuntime, object()),
        response=response,
    )
    labels = [child.label for child in view.children if isinstance(child, discord.ui.Button)]
    assert labels == ["Start", "Add music"]
    assert any(
        isinstance(child, discord.ui.Select) and child.placeholder == "More actions"
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
    labels = [child.label for child in view.children if isinstance(child, discord.ui.Button)]

    assert labels == ["Add music"]
    assert any(
        isinstance(child, discord.ui.Select) and child.placeholder == "More actions"
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
async def test_music_dashboard_stops_updates_after_403(tmp_path: Path) -> None:
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
    runtime.settings = Mock()
    runtime.settings.data_dir = tmp_path
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
    assert "1" not in manager._stored_messages
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
    restored = MusicDashboardManager(bot, runtime)
    assert "1" not in restored._channel_ids
    await restored.close()


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

    assert [child.label for child in mix_view.children if isinstance(child, discord.ui.Button)] == [
        "Switch to Radio",
        "Keep current mode",
    ]
    assert [
        child.label for child in loop_view.children if isinstance(child, discord.ui.Button)
    ] == ["Switch to Loop", "Keep current mode"]


@pytest.mark.asyncio
async def test_loop_mix_conflict_controls_disable_after_one_minute() -> None:
    view = LoopMixConflictView(
        cast(SimajilordRuntime, object()),
        None,
        requester_id=7,
    )
    message = Mock(spec=discord.InteractionMessage)
    message.edit = AsyncMock()
    view.message = message

    await view.on_timeout()

    assert all(
        child.disabled
        for child in view.children
        if isinstance(child, (discord.ui.Button, discord.ui.Select))
    )
    message.edit.assert_awaited_once_with(view=view)


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


@pytest.mark.asyncio
async def test_music_search_selection_updates_the_component_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = AudioSearchResponse(
        query="Same",
        candidates=(
            AudioSearchItem(
                reference="https://example.com/track",
                title="Artist - Same",
                duration_seconds=180,
                uploader="Artist",
            ),
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
    interaction = Mock(spec=discord.Interaction)
    interaction.user = Mock()
    interaction.user.id = 1
    interaction.response = Mock()
    interaction.response.defer = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    monkeypatch.setattr(
        "simajilord.integrations.discord.cogs._enqueue_interaction_track",
        AsyncMock(
            return_value=AudioPlayResponse(
                title="Artist - Same",
                page_url="https://example.com/track",
                queue_position=1,
                duration_seconds=180,
                destination_id="voice",
                playback_state="playing",
                requested_by_name="Listener",
            )
        ),
    )

    await view.choose(interaction, 0)

    interaction.response.defer.assert_awaited_once_with()
    assert interaction.edit_original_response.await_count == 2
    first_edit = interaction.edit_original_response.await_args_list[0]
    assert first_edit.kwargs["view"] is view
    assert all(
        child.disabled
        for child in view.children
        if isinstance(child, discord.ui.Button)
    )
    final_edit = interaction.edit_original_response.await_args_list[1]
    assert final_edit.kwargs["view"] is None


def test_agent_conversation_key_is_private_per_actor_and_channel() -> None:
    assert discord_conversation_id(guild_id=1, channel_id=2, actor_id=3) == (
        "discord:v4:guild:1:channel:2:actor:3"
    )
    assert discord_conversation_id(guild_id=None, channel_id=2, actor_id=3) == (
        "discord:v4:direct:channel:2:actor:3"
    )
    assert (
        discord_conversation_id(
            guild_id=1,
            channel_id=2,
            actor_id=3,
            grants=frozenset({AGENT_WEB_GRANT}),
        )
        == "discord:v4:guild:1:channel:2:actor:3:profile:web"
    )
    assert discord_conversation_id(guild_id=1, channel_id=2, actor_id=4) != (
        discord_conversation_id(guild_id=1, channel_id=2, actor_id=3)
    )


def test_web_commands_use_one_discoverable_group() -> None:
    commands = {
        command.name: command
        for group in WebCog.__cog_app_commands__
        if isinstance(group, app_commands.Group)
        for command in group.commands
    }
    assert set(commands) == {"search", "fetch", "find"}
    assert commands["search"].description == (
        "Search the web through Simajilord's local search service."
    )


def test_public_command_tree_has_no_legacy_or_duplicate_top_level_names() -> None:
    cog_types = (
        HelpCog,
        FeedbackCog,
        SystemCog,
        FocusTimerCog,
        MusicCog,
        ReadAloudCog,
        WebCog,
        MediaCog,
        UtilityCog,
        InfoCog,
    )
    top_level = [
        command.name for cog_type in cog_types for command in cog_type.__cog_app_commands__
    ]
    assert len(top_level) == len(set(top_level))
    assert set(top_level) == {
        "help",
        "feedback",
        "status",
        "system",
        "timer",
        "audio",
        "play",
        "radio",
        "join",
        "readaloud",
        "web",
        "media",
        "utility",
        "info",
    }
    assert not {
        "ping",
        "capabilities",
        "about",
        "uptime",
        "search",
        "fetch",
        "find",
        "detectai",
        "download",
        "roll",
        "choose",
        "serverinfo",
        "userinfo",
        "avatar",
        "poll",
        "music",
    } & set(top_level)


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
    assert any(field.name == "Connection" and field.value == "Ready" for field in embed.fields)


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


@pytest.mark.asyncio
async def test_quote_composer_rolls_back_state_when_discord_rejects_edit() -> None:
    view = QuoteComposerView(
        cast(SimajilordRuntime, object()),
        requester_id=7,
        source_channel_id=50,
        source_message_id=60,
        destination_channel_id=50,
    )
    interaction = Mock(spec=discord.Interaction)
    interaction.response.edit_message = AsyncMock(side_effect=RuntimeError("expired"))

    with pytest.raises(RuntimeError, match="expired"):
        await view._toggle_setting(interaction, "flip")

    assert view.flip is False
    assert view.more_menu_button.label == "More · 1 On"

    with pytest.raises(RuntimeError, match="expired"):
        await view._open_page(interaction, "more")

    assert view._page == "main"
    assert [item.label for item in view.children if isinstance(item, discord.ui.Button)] == [
        "Layout · Landscape",
        "Style · B/W",
        "More · 1 On",
        "Generate",
        "Cancel",
    ]


def test_quote_composer_uses_hierarchical_native_menu() -> None:
    view = QuoteComposerView(
        cast(SimajilordRuntime, object()),
        requester_id=7,
        source_channel_id=50,
        source_message_id=60,
        destination_channel_id=50,
        has_animation=True,
    )

    assert [item.label for item in view.children if isinstance(item, discord.ui.Button)] == [
        "Layout · Landscape",
        "Style · B/W",
        "More · 1 On",
        "Generate",
        "Cancel",
    ]

    view._show_page("more")
    assert [item.label for item in view.children if isinstance(item, discord.ui.Button)] == [
        "Flip Off",
        "Jump On",
        "Animation Off",
        "Back",
    ]

    view._show_page("layout")
    assert [item.label for item in view.children if isinstance(item, discord.ui.Button)] == [
        "Landscape",
        "Back",
    ]


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
    assert groups == []


def test_youtube_url_card_feature_is_not_exposed() -> None:
    from simajilord.integrations.discord import cogs

    assert not hasattr(cogs, "YouTubeLinkCardCog")
    assert not hasattr(cogs, "YouTubeLinkCardView")
    assert not hasattr(cogs, "_youtube_card_reference")


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
    session.voice_activation_required = False
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
    session.voice_activation_required = False
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
    session.voice_activation_required = True
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
    session.voice_activation_required = True
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
    session.voice_activation_required = True
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
        for group in MediaCog.__cog_app_commands__
        if isinstance(group, app_commands.Group)
        for command in group.commands
    }
    assert set(commands) == {"detect-ai", "download"}
    assert commands["detect-ai"].description == (
        "Estimate AI-generation and deepfake likelihood."
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
            complete=False,
            source_truncated=False,
        ),
    )
    buttons = [child for child in view.children if isinstance(child, discord.ui.Button)]
    assert [button.label for button in buttons] == ["Continue"]
    assert buttons[0].custom_id == "simajilord:web:fetch:continue"
    assert view.next_offset == 3_500


def test_message_index_preview_keeps_bounded_context_from_both_ends() -> None:
    assert _message_preview("x" * 240) == ("x" * 240, False)
    preview, truncated = _message_preview("a" * 200 + "middle" + "z" * 39)
    assert preview == "a" * 200 + "…" + "z" * 39
    assert len(preview) == 240
    assert truncated is True


def test_agent_message_context_includes_embed_and_component_text() -> None:
    message = Mock(spec=discord.Message)
    message.content = "visible content"
    message.is_system.return_value = False
    message.embeds = [
        discord.Embed(
            title="Release title",
            description="Release description",
        ).add_field(name="Status", value="Ready")
    ]
    message.poll = None
    message.components = []
    message.attachments = []

    context = _message_context_text(message)

    assert context.splitlines() == [
        "visible content",
        "[embed.0.title] Release title",
        "[embed.0.description] Release description",
        "[embed.0.field.0.name] Status",
        "[embed.0.field.0.value] Ready",
    ]


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


def test_trigger_message_id_comes_only_from_typed_context() -> None:
    context = InvocationContext(
        actor_id="actor",
        workspace_id="workspace",
        transport="agent",
        request_id="forged:wrong",
        active_message_id="1530953392980234250",
    )
    assert _discord_event_message_id(context) == "1530953392980234250"
    assert (
        _discord_event_message_id(
            InvocationContext(
                actor_id="actor",
                workspace_id="workspace",
                transport="agent",
                request_id="discord:message:1530953392980234250",
            )
        )
        is None
    )


@pytest.mark.asyncio
async def test_bot_unloads_cogs_before_closing_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    async def close_activity() -> None:
        order.append("activity")

    async def close_discord(_bot: commands.Bot) -> None:
        order.append("discord")

    async def close_runtime() -> None:
        order.append("runtime")

    bot = object.__new__(SimajilordDiscordBot)
    bot.activity_server = SimpleNamespace(close=close_activity)
    bot.runtime = SimpleNamespace(close=close_runtime)
    monkeypatch.setattr(commands.Bot, "close", close_discord)

    await SimajilordDiscordBot.close(bot)

    assert order == ["activity", "discord", "runtime"]


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


def test_agent_delivery_nonce_is_stable_bounded_and_chunk_specific() -> None:
    first = _agent_delivery_nonce("discord:message-edit:42:2026-07-29", 0)

    assert first == _agent_delivery_nonce(
        "discord:message-edit:42:2026-07-29",
        0,
    )
    assert len(first) == 25
    assert first != _agent_delivery_nonce(
        "discord:message-edit:42:2026-07-29",
        1,
    )
    assert first != _agent_delivery_nonce(
        "discord:message-edit:42:2026-07-29",
        0,
        purpose="error",
    )


def test_agent_rate_limit_message_includes_exact_retry_time() -> None:
    error = AgentRateLimitError(
        "limited",
        retry_after_seconds=125,
    )
    assert _agent_error_text(error).endswith("2 minutes 5 seconds.")
    assert _retry_after_text(3_661) == "1時間1分1秒"


def test_agent_provider_limit_message_explains_the_actual_failure() -> None:
    message = _agent_error_text(AgentProviderLimitError("usage limit"))
    assert "AI provider usage limit" in message
    assert "Please try again" in message


def test_agent_timeout_message_explains_interruption_and_uncertain_results() -> None:
    message = _agent_error_text(
        AgentTimeoutError(
            "deadline reached",
            timeout_seconds=125,
            runtime_restarted=True,
            write_attempted=True,
        )
    )
    assert "no observable activity for 2 minutes 5 seconds" in message
    assert "stopped" in message
    assert "runtime was restarted automatically" in message
    assert "could create a duplicate" in message
    assert "partial response or operation result is unconfirmed" in message


def test_agent_execution_failure_displays_only_valid_public_reference() -> None:
    reference_id = "agt_0123456789abcdef0123"

    message = _agent_error_text(
        RuntimeError("failed"),
        reference_id=reference_id,
    )

    assert f"Reference ID: `{reference_id}`" in message
    assert _agent_error_text(
        RuntimeError("failed"),
        reference_id="discord:message:secret",
    ) == "The AI request could not be completed."


def test_agent_admission_rejections_never_display_public_reference() -> None:
    reference_id = "agt_0123456789abcdef0123"

    assert reference_id not in _agent_error_text(
        AgentBusyError("busy"),
        reference_id=reference_id,
    )
    assert reference_id not in _agent_error_text(
        AgentRateLimitError("limited"),
        reference_id=reference_id,
    )


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


def test_trusted_guild_scope_never_borrows_the_bots_wider_visibility() -> None:
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
    ) == ()


def test_trusted_autonomy_bot_principal_uses_only_bot_visibility() -> None:
    guild = Mock(spec=discord.Guild)
    bot_member = Mock(spec=discord.Member)
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    guild.text_channels = [channel]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    channel.permissions_for.return_value = discord.Permissions(
        view_channel=True,
        read_message_history=True,
    )

    assert agent_readable_channel_ids(
        guild,
        None,
        trusted_guild=True,
        trigger_channel_id=20,
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
    with pytest.raises(UserError, match=r"discord\.agent_read_channel_forbidden"):
        _assert_agent_channel_scope(context, "60")


def test_agent_write_scope_allows_authorized_channel_in_origin_guild() -> None:
    context = InvocationContext(
        actor_id="30",
        workspace_id="10",
        transport="agent",
        request_id="discord:message:40",
        resource_ids=("50", "60"),
        origin_resource_id="50",
    )
    _assert_agent_update_scope(context, "50")
    _assert_agent_update_scope(context, "60")
    with pytest.raises(UserError, match=r"discord\.agent_read_channel_forbidden"):
        _assert_agent_update_scope(context, "70")
