"""Discord-specific permission boundaries shared by commands and the agent."""

from __future__ import annotations

import asyncio
import base64
import io
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePath
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlsplit

import discord
from PIL import Image, UnidentifiedImageError

from simajilord.capabilities.audio import (
    AudioAutoLeaveRequest,
    AudioControlRequest,
    AudioControlResponse,
    AudioLoopRequest,
    AudioMoveRequest,
    AudioNoArgsRequest,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueuePositionRequest,
    AudioSeekRequest,
    AudioTuneRequest,
    AudioVolumeRequest,
)
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudAddSourcesRequest,
    ReadAloudAnnouncementsSetRequest,
    ReadAloudDictionaryListRequest,
    ReadAloudDictionaryRemoveRequest,
    ReadAloudDictionarySetRequest,
    ReadAloudDisableRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudRemoveSourceRequest,
    ReadAloudRequest,
    ReadAloudResponse,
    ReadAloudSemanticsSetRequest,
    ReadAloudStatusRequest,
)
from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
)
from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime
from simajilord.services.files import WorkspaceFileRecord
from simajilord.services.quote import (
    QuoteCustomEmojiAsset,
    QuoteRenderRequest,
    QuoteStickerAsset,
)

from .audio import DiscordAudioOutput
from .presenter import (
    expanded_message_embeds,
    expanded_message_view,
    quote_message_view,
)

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel
    | discord.Thread
    | discord.VoiceChannel
    | discord.StageChannel
)
_CUSTOM_EMOJI_PATTERN = re.compile(
    r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{2,32}):(?P<id>[0-9]{15,22})>"
)
_CUSTOM_EMOJI_PREVIEW_LIMIT = 25
_CUSTOM_EMOJI_MEDIA_MAX_BYTES = 5_000_000
_QUOTE_AVATAR_MAX_BYTES = 8_000_000


@dataclass(frozen=True, slots=True)
class DiscordServerRequest:
    pass


@dataclass(frozen=True, slots=True)
class DiscordServerResponse:
    server_id: str
    name: str
    owner_id: str | None
    member_count: int | None
    text_channel_count: int
    voice_channel_count: int
    role_count: int
    created_at_iso: str
    icon_url: str | None


@dataclass(frozen=True, slots=True)
class DiscordUserRequest:
    user_id: str


@dataclass(frozen=True, slots=True)
class DiscordUserResponse:
    user_id: str
    display_name: str
    bot: bool
    created_at_iso: str
    joined_at_iso: str | None
    top_role: str | None
    avatar_url: str


@dataclass(frozen=True, slots=True)
class DiscordListChannelsRequest:
    include_threads: bool = True


@dataclass(frozen=True, slots=True)
class DiscordChannelRecord:
    channel_id: str
    name: str
    kind: str
    category_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordListChannelsResponse:
    channels: tuple[DiscordChannelRecord, ...]


@dataclass(frozen=True, slots=True)
class DiscordReadMessagesRequest:
    channel_id: str
    limit: int = 10
    before_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordAttachmentRecord:
    attachment_id: str
    filename: str
    content_type: str | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DiscordCustomEmojiRecord:
    index: int
    emoji_id: str
    name: str
    animated: bool
    occurrences: int


@dataclass(frozen=True, slots=True)
class DiscordStickerRecord:
    index: int
    sticker_id: str
    name: str
    format: str
    animated: bool


@dataclass(frozen=True, slots=True)
class DiscordMessageRecord:
    message_id: str
    channel_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content_preview: str
    content_length: int
    preview_truncated: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    reference_message_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordReadMessagesResponse:
    messages: tuple[DiscordMessageRecord, ...]
    oldest_message_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordGetMessageRequest:
    channel_id: str
    message_id: str
    offset: int = 0
    max_characters: int = 600
    include_reply_context: bool = True
    max_reply_depth: int = 2


@dataclass(frozen=True, slots=True)
class DiscordReplyContextRecord:
    message_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content_chunk: str
    content_length: int
    complete: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    reference_message_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordGetMessageResponse:
    message_id: str
    channel_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content_chunk: str
    content_length: int
    offset: int
    next_offset: int | None
    complete: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    custom_emojis: tuple[DiscordCustomEmojiRecord, ...]
    stickers: tuple[DiscordStickerRecord, ...]
    reference_message_id: str | None
    reply_context: tuple[DiscordReplyContextRecord, ...]


@dataclass(frozen=True, slots=True)
class DiscordMessageLink:
    guild_id: str
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class DiscordExpandMessageRequest:
    guild_id: str
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class DiscordExpandedAttachmentRecord:
    filename: str
    content_type: str | None
    size_bytes: int
    url: str
    proxy_url: str
    spoiler: bool


@dataclass(frozen=True, slots=True)
class DiscordExpandedEmbedRecord:
    title: str | None
    description: str | None
    url: str | None
    image_url: str | None
    thumbnail_url: str | None


@dataclass(frozen=True, slots=True)
class DiscordExpandedPollRecord:
    question: str
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscordExpandMessageResponse:
    guild_id: str
    channel_id: str
    channel_name: str
    message_id: str
    jump_url: str
    author_id: str
    author_name: str
    author_avatar_url: str
    author_is_bot: bool
    content: str
    created_at_iso: str
    edited_at_iso: str | None
    attachments: tuple[DiscordExpandedAttachmentRecord, ...]
    embeds: tuple[DiscordExpandedEmbedRecord, ...]
    sticker_names: tuple[str, ...]
    poll: DiscordExpandedPollRecord | None
    reply_author_name: str | None
    reply_content_preview: str | None


@dataclass(frozen=True, slots=True)
class DiscordPostExpandedMessageRequest:
    source_guild_id: str
    source_channel_id: str
    source_message_id: str
    destination_channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordPostExpandedMessageResponse:
    message_id: str
    channel_id: str
    source_jump_url: str


@dataclass(frozen=True, slots=True)
class DiscordCreateQuoteImageRequest:
    source_channel_id: str
    source_message_id: str
    destination_channel_id: str
    color: bool = False
    light: bool = False
    flip: bool = False
    bold: bool = False
    vertical: bool = False
    animate: bool = False
    include_jump: bool = True


@dataclass(frozen=True, slots=True)
class DiscordCreateQuoteImageResponse:
    message_id: str
    channel_id: str
    source_message_id: str
    source_jump_url: str
    filename: str
    width: int
    height: int
    rendered_custom_emojis: int
    rendered_stickers: int
    text_truncated: bool
    animated: bool


@dataclass(frozen=True, slots=True)
class DiscordViewCustomEmojiRequest:
    channel_id: str
    message_id: str
    emoji_index: int = 0
    mode: Literal["preview", "animation", "frame"] = "preview"
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewCustomEmojiResponse:
    emoji_index: int
    emoji_id: str
    name: str
    animated: bool
    occurrences: int
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None
    content_type: str
    image_data_url: str


@dataclass(frozen=True, slots=True)
class DiscordViewStickerRequest:
    channel_id: str
    message_id: str
    sticker_index: int = 0
    mode: Literal["preview", "animation", "frame"] = "preview"
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewStickerResponse:
    sticker_index: int
    sticker_id: str
    name: str
    format: str
    animated: bool
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None
    content_type: str
    image_data_url: str


@dataclass(frozen=True, slots=True)
class _DiscordAnimatedMedia:
    content: bytes
    content_type: str
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None


@dataclass(frozen=True, slots=True)
class DiscordSendMessageRequest:
    channel_id: str
    content: str


@dataclass(frozen=True, slots=True)
class DiscordAnalyzeAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordImportAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0
    destination_path: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordViewImageAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewImageAttachmentResponse:
    filename: str
    content_type: str
    size_bytes: int
    image_data_url: str


@dataclass(frozen=True, slots=True)
class DiscordSendFileRequest:
    channel_id: str
    path: str
    caption: str = ""


@dataclass(frozen=True, slots=True)
class DiscordSendFileResponse:
    message_id: str
    channel_id: str
    filename: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DiscordSendMessageResponse:
    message_id: str
    channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordPollRequest:
    channel_id: str
    question: str
    options: tuple[str, ...]
    duration_hours: int = 24
    multiple: bool = False


@dataclass(frozen=True, slots=True)
class DiscordPollResponse:
    message_id: str
    channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordConnectVoiceRequest:
    channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordConnectVoiceResponse:
    channel_id: str
    connected: bool


def build_discord_endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> tuple[CapabilityEndpoint, ...]:
    async def inspect_server(
        _: DiscordServerRequest,
        context: InvocationContext,
    ) -> DiscordServerResponse:
        guild = _guild(client, context)
        return DiscordServerResponse(
            server_id=str(guild.id),
            name=guild.name,
            owner_id=str(guild.owner_id) if guild.owner_id else None,
            member_count=guild.member_count,
            text_channel_count=len(guild.text_channels),
            voice_channel_count=len(guild.voice_channels),
            role_count=len(guild.roles),
            created_at_iso=guild.created_at.isoformat(),
            icon_url=str(guild.icon.url) if guild.icon else None,
        )

    async def inspect_user(
        request: DiscordUserRequest,
        context: InvocationContext,
    ) -> DiscordUserResponse:
        guild = _guild(client, context)
        try:
            user_id = int(request.user_id)
        except ValueError as exc:
            raise UserError("ユーザーIDが正しくありません。") from exc
        member = guild.get_member(user_id)
        user = member or client.get_user(user_id)
        if user is None:
            try:
                user = await client.fetch_user(user_id)
            except discord.DiscordException as exc:
                raise UserError("Discordユーザーが見つかりませんでした。") from exc
        return DiscordUserResponse(
            user_id=str(user.id),
            display_name=member.display_name if member else user.display_name,
            bot=user.bot,
            created_at_iso=user.created_at.isoformat(),
            joined_at_iso=member.joined_at.isoformat() if member and member.joined_at else None,
            top_role=member.top_role.name if member else None,
            avatar_url=str(user.display_avatar.url),
        )

    async def list_channels(
        request: DiscordListChannelsRequest,
        context: InvocationContext,
    ) -> DiscordListChannelsResponse:
        guild = _guild(client, context)
        allowed_ids = set(context.resource_ids) if context.transport == "agent" else None
        records = [
            DiscordChannelRecord(
                channel_id=str(channel.id),
                name=channel.name,
                kind=str(channel.type),
                category_id=str(channel.category_id) if channel.category_id else None,
            )
            for channel in guild.channels
            if allowed_ids is None or str(channel.id) in allowed_ids
        ]
        if request.include_threads:
            records.extend(
                DiscordChannelRecord(
                    channel_id=str(thread.id),
                    name=thread.name,
                    kind="thread",
                    category_id=str(thread.parent_id) if thread.parent_id else None,
                )
                for thread in guild.threads
                if allowed_ids is None or str(thread.id) in allowed_ids
            )
        records.sort(key=lambda item: (item.kind, item.name, item.channel_id))
        return DiscordListChannelsResponse(channels=tuple(records))

    async def read_messages(
        request: DiscordReadMessagesRequest,
        context: InvocationContext,
    ) -> DiscordReadMessagesResponse:
        guild = _guild(client, context)
        _assert_agent_channel_scope(context, request.channel_id)
        channel = _message_channel(guild, request.channel_id)
        if not 1 <= request.limit <= 25:
            raise UserError("discord.message_limit_invalid")
        before = (
            discord.Object(_snowflake(request.before_message_id, "message"))
            if request.before_message_id
            else None
        )
        event_message_id = _discord_event_message_id(context)
        messages = [
            _message_record(message, event_message_id=event_message_id)
            async for message in channel.history(limit=request.limit, before=before)
        ]
        return DiscordReadMessagesResponse(
            messages=tuple(messages),
            oldest_message_id=messages[-1].message_id if messages else None,
        )

    async def get_message(
        request: DiscordGetMessageRequest,
        context: InvocationContext,
    ) -> DiscordGetMessageResponse:
        guild = _guild(client, context)
        _assert_agent_channel_scope(context, request.channel_id)
        channel = _message_channel(guild, request.channel_id)
        message_id = _snowflake(request.message_id, "message")
        if request.offset < 0:
            raise UserError("discord.message_offset_invalid")
        if request.max_characters < 1:
            raise UserError("discord.message_chunk_limit_invalid")
        if not 0 <= request.max_reply_depth <= 3:
            raise UserError("discord.reply_depth_invalid")
        # The model may request a larger window than the per-tool response budget.
        # Clamp instead of failing so it can always read the exact trigger message,
        # then follow next_offset only when the remaining text is actually needed.
        chunk_characters = min(request.max_characters, 1_000)
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound as exc:
            raise UserError("Discordメッセージが見つかりませんでした。") from exc
        except discord.DiscordException as exc:
            raise UserError("Discordメッセージを取得できませんでした。") from exc
        content_length = len(message.content)
        if request.offset > content_length:
            raise UserError("discord.message_offset_invalid")
        end = min(content_length, request.offset + chunk_characters)
        next_offset = end if end < content_length else None
        reply_context = (
            await _reply_context(
                channel,
                message,
                max_depth=request.max_reply_depth,
                chunk_characters=min(chunk_characters, 600),
            )
            if request.include_reply_context and request.max_reply_depth
            else ()
        )
        return DiscordGetMessageResponse(
            message_id=str(message.id),
            channel_id=str(message.channel.id),
            author_id=str(message.author.id),
            author_name=message.author.display_name,
            author_is_bot=message.author.bot,
            content_chunk=message.content[request.offset:end],
            content_length=content_length,
            offset=request.offset,
            next_offset=next_offset,
            complete=next_offset is None,
            created_at_iso=message.created_at.isoformat(),
            attachments=tuple(
                _attachment_record(attachment) for attachment in message.attachments
            ),
            custom_emojis=_custom_emoji_records(message.content),
            stickers=_sticker_records(message.stickers),
            reference_message_id=(
                str(message.reference.message_id)
                if message.reference and message.reference.message_id
                else None
            ),
            reply_context=reply_context,
        )

    async def expand_message(
        request: DiscordExpandMessageRequest,
        context: InvocationContext,
    ) -> DiscordExpandMessageResponse:
        guild = _guild(client, context)
        if request.guild_id != str(guild.id):
            raise UserError("discord.expand_cross_guild_forbidden")
        channel, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        reply_author_name, reply_content_preview = _expanded_reply(message)
        return DiscordExpandMessageResponse(
            guild_id=str(guild.id),
            channel_id=str(channel.id),
            channel_name=channel.name,
            message_id=str(message.id),
            jump_url=message.jump_url,
            author_id=str(message.author.id),
            author_name=message.author.display_name,
            author_avatar_url=str(message.author.display_avatar.url),
            author_is_bot=message.author.bot,
            content=message.content,
            created_at_iso=message.created_at.isoformat(),
            edited_at_iso=(
                message.edited_at.isoformat() if message.edited_at is not None else None
            ),
            attachments=tuple(
                _expanded_attachment(attachment)
                for attachment in message.attachments[:10]
            ),
            embeds=tuple(_expanded_embed(item) for item in message.embeds[:10]),
            sticker_names=tuple(sticker.name for sticker in message.stickers[:10]),
            poll=_expanded_poll(message.poll),
            reply_author_name=reply_author_name,
            reply_content_preview=reply_content_preview,
        )

    async def post_expanded_message(
        request: DiscordPostExpandedMessageRequest,
        context: InvocationContext,
    ) -> DiscordPostExpandedMessageResponse:
        response = await expand_message(
            DiscordExpandMessageRequest(
                guild_id=request.source_guild_id,
                channel_id=request.source_channel_id,
                message_id=request.source_message_id,
            ),
            context,
        )
        guild = _guild(client, context)
        destination = _message_channel(guild, request.destination_channel_id)
        if context.transport == "agent":
            _assert_agent_channel_scope(context, request.destination_channel_id)
        else:
            actor = await _actor_member(guild, context)
            if not _can_read_messages(
                destination, actor
            ) or not _can_read_private_thread(destination, actor):
                raise UserError("discord.expand_destination_unavailable")
        bot_member = guild.me
        if bot_member is None or not _can_post_expanded_message(destination, bot_member):
            raise UserError("discord.expand_destination_unavailable")
        try:
            posted = await destination.send(
                embeds=expanded_message_embeds(response),
                view=expanded_message_view(response.jump_url),
                allowed_mentions=discord.AllowedMentions.none(),
                silent=True,
            )
        except discord.Forbidden as exc:
            raise UserError("discord.expand_destination_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.expand_failed") from exc
        return DiscordPostExpandedMessageResponse(
            message_id=str(posted.id),
            channel_id=str(destination.id),
            source_jump_url=response.jump_url,
        )

    async def create_quote_image(
        request: DiscordCreateQuoteImageRequest,
        context: InvocationContext,
    ) -> DiscordCreateQuoteImageResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.source_channel_id,
            message_id=request.source_message_id,
            context=context,
        )
        destination = _message_channel(guild, request.destination_channel_id)
        if context.transport == "agent":
            _assert_agent_channel_scope(context, request.destination_channel_id)
        else:
            actor = await _actor_member(guild, context)
            if not _can_read_messages(
                destination, actor
            ) or not _can_read_private_thread(destination, actor):
                raise UserError("discord.quote_destination_unavailable")
        bot_member = guild.me
        if bot_member is None or not _can_post_quote_image(destination, bot_member):
            raise UserError("discord.quote_destination_unavailable")
        avatar, custom_emojis, stickers = await asyncio.gather(
            _quote_avatar(client, message),
            _quote_custom_emojis(client, message.content),
            _quote_stickers(client, message),
        )
        try:
            rendered = await asyncio.to_thread(
                runtime.quote.render,
                QuoteRenderRequest(
                    text=_quote_text(message),
                    display_name=message.author.display_name,
                    username=message.author.name,
                    avatar=avatar,
                    custom_emojis=custom_emojis,
                    stickers=stickers,
                    color=request.color,
                    light=request.light,
                    flip=request.flip,
                    bold=request.bold,
                    vertical=request.vertical,
                    animate=request.animate,
                ),
            )
        except (OSError, ValueError) as exc:
            raise UserError("discord.quote_render_failed") from exc
        extension = "gif" if rendered.animated else "png"
        filename = f"quote-{message.id}.{extension}"
        try:
            image_file = discord.File(
                io.BytesIO(rendered.content),
                filename=filename,
            )
            if request.include_jump:
                jump_view = quote_message_view(message.jump_url)
                assert jump_view is not None
                posted = await destination.send(
                    file=image_file,
                    view=jump_view,
                    allowed_mentions=discord.AllowedMentions.none(),
                    silent=True,
                )
            else:
                posted = await destination.send(
                    file=image_file,
                    allowed_mentions=discord.AllowedMentions.none(),
                    silent=True,
                )
        except discord.Forbidden as exc:
            raise UserError("discord.quote_destination_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.quote_failed") from exc
        return DiscordCreateQuoteImageResponse(
            message_id=str(posted.id),
            channel_id=str(destination.id),
            source_message_id=str(message.id),
            source_jump_url=message.jump_url,
            filename=filename,
            width=rendered.width,
            height=rendered.height,
            rendered_custom_emojis=rendered.rendered_custom_emojis,
            rendered_stickers=rendered.rendered_stickers,
            text_truncated=rendered.text_truncated,
            animated=rendered.animated,
        )

    async def view_custom_emoji(
        request: DiscordViewCustomEmojiRequest,
        context: InvocationContext,
    ) -> DiscordViewCustomEmojiResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        custom_emojis = _custom_emoji_records(message.content)
        if not 0 <= request.emoji_index < len(custom_emojis):
            raise UserError("discord.custom_emoji_index_invalid")
        selected = custom_emojis[request.emoji_index]
        if request.frame_index < 0:
            raise UserError("discord.custom_emoji_frame_invalid")
        if request.mode == "animation" and not selected.animated:
            raise UserError("discord.custom_emoji_not_animated")
        if request.mode != "frame" and request.frame_index != 0:
            raise UserError("discord.custom_emoji_frame_mode_required")
        extension = (
            "gif"
            if selected.animated and request.mode in {"animation", "frame"}
            else "png"
        )
        emoji_url = (
            f"https://cdn.discordapp.com/emojis/{selected.emoji_id}.{extension}"
            "?size=128&quality=lossless"
        )
        try:
            content = await client.http.get_from_cdn(emoji_url)
        except (discord.NotFound, discord.Forbidden) as exc:
            raise UserError("discord.custom_emoji_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.custom_emoji_failed") from exc
        if not content or len(content) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            raise UserError("discord.custom_emoji_invalid")
        try:
            media = await asyncio.to_thread(
                _prepare_discord_animated_media,
                content,
                mode=request.mode,
                frame_index=request.frame_index,
            )
        except ValueError as exc:
            raise UserError(str(exc)) from exc
        return DiscordViewCustomEmojiResponse(
            emoji_index=selected.index,
            emoji_id=selected.emoji_id,
            name=selected.name,
            animated=selected.animated,
            occurrences=selected.occurrences,
            preview_kind=media.preview_kind,
            frame_index=media.frame_index,
            frame_count=media.frame_count,
            duration_ms=media.duration_ms,
            content_type=media.content_type,
            image_data_url=(
                f"data:{media.content_type};base64,"
                + base64.b64encode(media.content).decode("ascii")
            ),
        )

    async def view_sticker(
        request: DiscordViewStickerRequest,
        context: InvocationContext,
    ) -> DiscordViewStickerResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        if not 0 <= request.sticker_index < len(message.stickers):
            raise UserError("discord.sticker_index_invalid")
        selected = message.stickers[request.sticker_index]
        if request.frame_index < 0:
            raise UserError("discord.sticker_frame_invalid")
        if request.mode != "frame" and request.frame_index != 0:
            raise UserError("discord.sticker_frame_mode_required")
        if request.mode == "animation" and selected.format is discord.StickerFormatType.png:
            raise UserError("discord.sticker_not_animated")
        if selected.format is discord.StickerFormatType.lottie:
            if request.mode != "preview":
                raise UserError("discord.sticker_lottie_animation_unavailable")
            sticker_url = (
                f"https://media.discordapp.net/stickers/{selected.id}.png"
                "?size=160&quality=lossless"
            )
        else:
            sticker_url = selected.url
        try:
            content = await client.http.get_from_cdn(str(sticker_url))
        except (discord.NotFound, discord.Forbidden) as exc:
            raise UserError("discord.sticker_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.sticker_failed") from exc
        if not content or len(content) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            raise UserError("discord.sticker_invalid")
        try:
            media = await asyncio.to_thread(
                _prepare_discord_animated_media,
                content,
                mode=request.mode,
                frame_index=request.frame_index,
            )
        except ValueError as exc:
            code = str(exc).replace("custom_emoji", "sticker")
            raise UserError(code) from exc
        return DiscordViewStickerResponse(
            sticker_index=request.sticker_index,
            sticker_id=str(selected.id),
            name=selected.name,
            format=selected.format.name,
            animated=selected.format is not discord.StickerFormatType.png,
            preview_kind=media.preview_kind,
            frame_index=media.frame_index,
            frame_count=media.frame_count,
            duration_ms=media.duration_ms,
            content_type=media.content_type,
            image_data_url=(
                f"data:{media.content_type};base64,"
                + base64.b64encode(media.content).decode("ascii")
            ),
        )

    async def analyze_attachment(
        request: DiscordAnalyzeAttachmentRequest,
        context: InvocationContext,
    ) -> SyntheticMediaAnalyzeResponse:
        guild = _guild(client, context)
        _assert_agent_channel_scope(context, request.channel_id)
        channel = _message_channel(guild, request.channel_id)
        if not 0 <= request.attachment_index <= 9:
            raise UserError("discord.attachment_index_invalid")
        try:
            message = await channel.fetch_message(
                _snowflake(request.message_id, "message")
            )
        except discord.NotFound as exc:
            raise UserError("Discordメッセージが見つかりませんでした。") from exc
        except discord.DiscordException as exc:
            raise UserError("Discordメッセージを取得できませんでした。") from exc
        try:
            attachment = message.attachments[request.attachment_index]
        except IndexError as exc:
            raise UserError("discord.attachment_missing") from exc
        if attachment.size > runtime.settings.hive_max_media_bytes:
            raise UserError("moderation.media_too_large")
        try:
            content = await attachment.read(use_cached=True)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        return cast(
            SyntheticMediaAnalyzeResponse,
            await runtime.registry.invoke(
                "moderation.detect_synthetic_media",
                SyntheticMediaAnalyzeRequest(
                    filename=attachment.filename,
                    content_type=attachment.content_type,
                    content=content,
                ),
                context,
            ),
        )

    async def import_attachment(
        request: DiscordImportAttachmentRequest,
        context: InvocationContext,
    ) -> WorkspaceFileRecord:
        if runtime.files is None:
            raise UserError("files.disabled")
        if context.workspace_id is None:
            raise UserError("files.workspace_required")
        message, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        del message
        if attachment.size > runtime.files.max_file_bytes:
            raise UserError("files.file_too_large")
        try:
            content = await attachment.read(use_cached=True)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        destination = request.destination_path or (
            f"attachments/{request.message_id}/{PurePath(attachment.filename).name}"
        )
        return await asyncio.to_thread(
            runtime.files.import_bytes,
            context.workspace_id,
            destination,
            content,
        )

    async def view_image_attachment(
        request: DiscordViewImageAttachmentRequest,
        context: InvocationContext,
    ) -> DiscordViewImageAttachmentResponse:
        _, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        if attachment.size > 8 * 1024 * 1024:
            raise UserError("discord.image_attachment_too_large")
        try:
            content = await attachment.read(use_cached=True)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        media_type = _image_media_type(content)
        if media_type is None:
            raise UserError("discord.attachment_not_supported_image")
        encoded = base64.b64encode(content).decode("ascii")
        return DiscordViewImageAttachmentResponse(
            filename=attachment.filename,
            content_type=media_type,
            size_bytes=len(content),
            image_data_url=f"data:{media_type};base64,{encoded}",
        )

    async def send_message(
        request: DiscordSendMessageRequest,
        context: InvocationContext,
    ) -> DiscordSendMessageResponse:
        guild = _guild(client, context)
        _assert_agent_update_scope(context, request.channel_id)
        if not 1 <= len(request.content) <= 2_000:
            raise UserError("Discordへ送るメッセージは1〜2000文字にしてください。")
        try:
            channel_id = int(request.channel_id)
        except ValueError as exc:
            raise UserError("チャンネルIDが正しくありません。") from exc
        channel = guild.get_channel_or_thread(channel_id)
        if not isinstance(
            channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            raise UserError("指定先はメッセージを送信できるDiscordチャンネルではありません。")
        message = await channel.send(
            request.content,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return DiscordSendMessageResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
        )

    async def send_file(
        request: DiscordSendFileRequest,
        context: InvocationContext,
    ) -> DiscordSendFileResponse:
        if runtime.files is None:
            raise UserError("files.disabled")
        if context.workspace_id is None:
            raise UserError("files.workspace_required")
        _assert_agent_update_scope(context, request.channel_id)
        if len(request.caption) > 2_000:
            raise UserError("discord.file_caption_too_long")
        guild = _guild(client, context)
        channel = _message_channel(guild, request.channel_id)
        path = runtime.files.path_for_delivery(context.workspace_id, request.path)
        if path.stat().st_size > 25 * 1024 * 1024:
            raise UserError("discord.file_too_large")
        message = await channel.send(
            request.caption or None,
            file=discord.File(path, filename=path.name),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return DiscordSendFileResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            filename=path.name,
            size_bytes=path.stat().st_size,
        )

    async def create_poll(
        request: DiscordPollRequest,
        context: InvocationContext,
    ) -> DiscordPollResponse:
        guild = _guild(client, context)
        channel = _text_channel(guild, request.channel_id)
        question = request.question.strip()
        options = tuple(option.strip() for option in request.options if option.strip())
        if not 1 <= len(question) <= 300:
            raise UserError("投票の質問は1〜300文字にしてください。")
        if not 2 <= len(options) <= 10:
            raise UserError("投票の選択肢は2〜10個にしてください。")
        if any(len(option) > 55 for option in options):
            raise UserError("投票の選択肢は1つ55文字以内にしてください。")
        if not 1 <= request.duration_hours <= 168:
            raise UserError("投票期間は1〜168時間で指定してください。")
        poll = discord.Poll(
            question,
            duration=timedelta(hours=request.duration_hours),
            multiple=request.multiple,
        )
        for option in options:
            poll.add_answer(text=option)
        message = await channel.send(poll=poll)
        return DiscordPollResponse(message_id=str(message.id), channel_id=str(channel.id))

    async def connect_voice(
        request: DiscordConnectVoiceRequest,
        context: InvocationContext,
    ) -> DiscordConnectVoiceResponse:
        guild = _guild(client, context)
        channel = guild.get_channel(_snowflake(request.channel_id, "voice channel"))
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise UserError("指定先はボイスチャンネルではありません。")
        workspace_id = str(guild.id)
        session = runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(client, guild.id),
        )
        if session.current is not None:
            output = session.output
            if isinstance(output, DiscordAudioOutput) and output.destination_id != channel.id:
                raise UserError("別のボイスチャンネルで音声を再生しています。")
        await runtime.audio.connect(workspace_id, str(channel.id))
        return DiscordConnectVoiceResponse(channel_id=str(channel.id), connected=True)

    async def play_audio(
        request: AudioPlayRequest,
        context: InvocationContext,
    ) -> AudioPlayResponse:
        """Enforce Discord voice scope before entering the shared audio API."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        await _prepare_actor_audio(client, runtime, guild, member)
        response = await runtime.registry.invoke(
            "audio.play",
            AudioPlayRequest(
                reference=request.reference,
                requested_by_name=member.display_name,
            ),
            context,
        )
        return cast(AudioPlayResponse, response)

    async def control_audio(
        request: AudioControlRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        """Enforce requester ownership before mutating a Discord audio session."""

        return await _invoke_audio_control("audio.control", request, context)

    async def _invoke_audio_control(
        capability_name: str,
        request: object,
        context: InvocationContext,
    ) -> AudioControlResponse:
        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        session = runtime.audio.require(str(guild.id))
        if session.output.connected:
            _assert_same_voice(
                session.destination_id,
                _member_voice_channel(member),
            )
        elif session.waiting_for_voice and not session.can_control_while_waiting(
            context.actor_id
        ):
            raise UserError("audio.waiting_queue_restricted")
        response = await runtime.registry.invoke(capability_name, request, context)
        return cast(AudioControlResponse, response)

    async def pause_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.pause", request, context)

    async def resume_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.resume", request, context)

    async def skip_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.skip", request, context)

    async def stop_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.stop", request, context)

    async def leave_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.leave", request, context)

    async def set_audio_loop(
        request: AudioLoopRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_loop", request, context)

    async def remove_audio(
        request: AudioQueuePositionRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.remove", request, context)

    async def set_audio_auto_leave(
        request: AudioAutoLeaveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_auto_leave", request, context)

    async def shuffle_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.shuffle", request, context)

    async def seek_audio(
        request: AudioSeekRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.seek", request, context)

    async def tune_audio(
        request: AudioTuneRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.tune", request, context)

    async def set_audio_volume(
        request: AudioVolumeRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_volume", request, context)

    async def move_audio(
        request: AudioMoveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.move", request, context)

    async def clear_my_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.clear_mine", request, context)

    def discord_audio_endpoint(
        name: str,
        summary: str,
        keywords: tuple[str, ...],
        request_type: type[Any],
        handler: Callable[..., Awaitable[AudioControlResponse]],
    ) -> CapabilityEndpoint:
        return endpoint(
            CapabilityDescriptor(
                name=name,
                summary=summary,
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", *keywords),
                side_effects=("サーバーの永続音声セッションを変更します。",),
            ),
            request_type,
            AudioControlResponse,
            handler,
        )

    async def speak(
        request: SpeechSpeakRequest,
        context: InvocationContext,
    ) -> SpeechSpeakResponse:
        """Prepare the requester's voice route before shared speech synthesis."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        await _prepare_actor_audio(client, runtime, guild, member)
        response = await runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=request.text,
                title=f"{member.display_name}さんの依頼",
            ),
            context,
        )
        return cast(SpeechSpeakResponse, response)

    async def manage_read_aloud(
        request: ReadAloudRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        """Protect persistent read-aloud mutations with Discord permissions."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        mutating = request.action is not ReadAloudAction.STATUS
        if mutating and not member.guild_permissions.manage_guild:
            member_voice = _member_voice_channel(member)
            if request.action is ReadAloudAction.ADD_SOURCES:
                current_route = runtime.read_aloud.get(str(guild.id))
                self_service_allowed = (
                    bool(request.text_channel_ids)
                    and request.audio_destination_id is not None
                    and member_voice is not None
                    and str(member_voice.id) == request.audio_destination_id
                    and (
                        current_route is None
                        or current_route.audio_destination_id
                        == request.audio_destination_id
                    )
                )
            elif request.action in {
                ReadAloudAction.CONFIGURE,
                ReadAloudAction.ADD_SOURCE,
            }:
                self_service_allowed = (
                    request.text_channel_id is not None
                    and request.audio_destination_id is not None
                    and context.origin_resource_id == request.text_channel_id
                    and member_voice is not None
                    and str(member_voice.id) == request.audio_destination_id
                )
            else:
                route = runtime.read_aloud.get(str(guild.id))
                self_service_allowed = (
                    route is not None
                    and member_voice is not None
                    and str(member_voice.id) == route.audio_destination_id
                    and (
                        request.action is ReadAloudAction.DISABLE
                        or (
                            request.action is ReadAloudAction.REMOVE_SOURCE
                            and request.text_channel_id is not None
                            and context.origin_resource_id == request.text_channel_id
                        )
                    )
                )
            if not self_service_allowed:
                raise UserError("discord.manage_guild_required")
        if request.action is ReadAloudAction.ADD_SOURCES:
            source_ids = tuple(dict.fromkeys(request.text_channel_ids))
            if not 1 <= len(source_ids) <= 25:
                raise UserError("read_aloud.source_channel_limit")
            if request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            for source_id in source_ids:
                source = _message_channel(guild, source_id)
                if not _can_read_messages(source, member):
                    raise UserError("discord.message_channel_unavailable")
                if guild.me is None or not _can_read_messages(source, guild.me):
                    raise UserError("discord.message_channel_unavailable")
            voice = guild.get_channel(
                _snowflake(request.audio_destination_id, "voice channel")
            )
            if not isinstance(voice, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("discord.voice_channel_required")
        elif request.action in {
            ReadAloudAction.CONFIGURE,
            ReadAloudAction.ADD_SOURCE,
        }:
            if request.text_channel_id is None or request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            source = _message_channel(guild, request.text_channel_id)
            if not _can_read_messages(source, member):
                raise UserError("discord.message_channel_unavailable")
            if guild.me is None or not _can_read_messages(source, guild.me):
                raise UserError("discord.message_channel_unavailable")
            voice = guild.get_channel(
                _snowflake(request.audio_destination_id, "voice channel")
            )
            if not isinstance(voice, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("discord.voice_channel_required")
        elif request.action is ReadAloudAction.REMOVE_SOURCE:
            if request.text_channel_id is None:
                raise UserError("read_aloud.source_channel_required")
            _message_channel(guild, request.text_channel_id)
        response = await runtime.registry.invoke(
            "speech.manage_read_aloud",
            request,
            context,
        )
        return cast(ReadAloudResponse, response)

    async def read_aloud_status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(action=ReadAloudAction.STATUS),
            context,
        )

    async def read_aloud_add_sources(
        request: ReadAloudAddSourcesRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(
                action=ReadAloudAction.ADD_SOURCES,
                text_channel_ids=request.text_channel_ids,
                audio_destination_id=request.audio_destination_id,
                mode=request.mode,
            ),
            context,
        )

    async def read_aloud_remove_source(
        request: ReadAloudRemoveSourceRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(
                action=ReadAloudAction.REMOVE_SOURCE,
                text_channel_id=request.text_channel_id,
            ),
            context,
        )

    async def read_aloud_disable(
        _request: ReadAloudDisableRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(action=ReadAloudAction.DISABLE),
            context,
        )

    async def read_aloud_policy_status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        _guild(client, context)
        response = await runtime.registry.invoke(
            "speech.read_aloud_policy_status",
            ReadAloudStatusRequest(),
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_list(
        _request: ReadAloudDictionaryListRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        _guild(client, context)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_list",
            ReadAloudDictionaryListRequest(),
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_set(
        request: ReadAloudDictionarySetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_remove(
        request: ReadAloudDictionaryRemoveRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_remove",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_exclusion_set(
        request: ReadAloudExclusionSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        target_id = _snowflake(request.target_id, request.target.value)
        if request.target is ReadAloudExclusionTarget.USER:
            if target_id != member.id:
                _require_manage_guild(member)
            target_member = guild.get_member(target_id)
            if target_member is None:
                try:
                    target_member = await guild.fetch_member(target_id)
                except discord.DiscordException as exc:
                    raise UserError("discord.member_lookup_failed") from exc
            if target_member.bot:
                raise UserError("read_aloud.ignore_bot_unnecessary")
        else:
            _require_manage_guild(member)
            if guild.get_role(target_id) is None:
                raise UserError("read_aloud.role_not_found")
        response = await runtime.registry.invoke(
            "speech.read_aloud_exclusion_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_announcements_set(
        request: ReadAloudAnnouncementsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_announcements_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_semantics_set(
        request: ReadAloudSemanticsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_semantics_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_server",
                summary="現在のDiscordサーバーの構成とIDを確認します。",
                risk=RiskLevel.READ,
                keywords=("server", "guild", "channels", "roles", "members"),
            ),
            DiscordServerRequest,
            DiscordServerResponse,
            inspect_server,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_user",
                summary="Discordユーザーの公開アカウント情報とサーバー参加情報を確認します。",
                risk=RiskLevel.READ,
                keywords=("user", "member", "avatar", "role"),
            ),
            DiscordUserRequest,
            DiscordUserResponse,
            inspect_user,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_channels",
                summary="現在のDiscordサーバーにあるチャンネルとIDを一覧表示します。",
                risk=RiskLevel.READ,
                keywords=("discord", "channels", "threads", "where"),
            ),
            DiscordListChannelsRequest,
            DiscordListChannelsResponse,
            list_channels,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_messages",
                summary="許可されたDiscordチャンネルから、一定件数のメッセージを読み取ります。",
                risk=RiskLevel.READ,
                keywords=("discord", "messages", "history", "conversation", "moderation"),
            ),
            DiscordReadMessagesRequest,
            DiscordReadMessagesResponse,
            read_messages,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.get_message",
                summary=(
                    "IDを指定してDiscordメッセージを一定文字数ずつ読み取ります。"
                    "必要に応じて同じチャンネル内の返信元も確認できます。"
                ),
                risk=RiskLevel.READ,
                keywords=("discord", "message", "chunk", "mention", "content"),
            ),
            DiscordGetMessageRequest,
            DiscordGetMessageResponse,
            get_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.expand_message",
                summary=(
                    "Discordメッセージリンクの内容を、閲覧権限を確認して引用表示用に取得します。"
                ),
                risk=RiskLevel.READ,
                keywords=("discord", "message", "link", "quote", "expand"),
            ),
            DiscordExpandMessageRequest,
            DiscordExpandMessageResponse,
            expand_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.post_expanded_message",
                summary=(
                    "閲覧可能なDiscordメッセージを、出典とJumpリンクを保った引用として"
                    "閲覧可能なチャンネルへ転載します。本文は変更できません。"
                ),
                risk=RiskLevel.WRITE,
                keywords=("discord", "message", "quote", "repost", "expand", "jump"),
                side_effects=(
                    "指定したDiscordチャンネルへ、元メッセージの引用を1件投稿します。",
                ),
            ),
            DiscordPostExpandedMessageRequest,
            DiscordPostExpandedMessageResponse,
            post_expanded_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_quote_image",
                summary=(
                    "閲覧可能なDiscordメッセージを、投稿者・アバター・絵文字を保った"
                    "引用画像としてローカル描画し、元投稿へのリンク付きで送信します。"
                ),
                risk=RiskLevel.WRITE,
                keywords=(
                    "discord",
                    "message",
                    "quote",
                    "quote image",
                    "引用画像",
                    "画像化",
                ),
                side_effects=(
                    "指定したDiscordチャンネルへ引用画像を1件投稿します。",
                ),
            ),
            DiscordCreateQuoteImageRequest,
            DiscordCreateQuoteImageResponse,
            create_quote_image,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_custom_emoji",
                summary=(
                    "指定したDiscordメッセージ内のカスタム絵文字を、必要な1件だけ"
                    "画像として確認します。アニメーション全体や任意フレームも選べます。"
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "message",
                    "emoji",
                    "custom emoji",
                    "animated emoji",
                    "image",
                ),
            ),
            DiscordViewCustomEmojiRequest,
            DiscordViewCustomEmojiResponse,
            view_custom_emoji,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_sticker",
                summary=(
                    "指定したDiscordメッセージのスタンプを、必要な1件だけ画像として"
                    "確認します。対応形式ではアニメーション全体や任意フレームも返します。"
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "message",
                    "sticker",
                    "stamp",
                    "animation",
                    "frame",
                    "image",
                ),
            ),
            DiscordViewStickerRequest,
            DiscordViewStickerResponse,
            view_sticker,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.analyze_attachment",
                summary=(
                    "許可されたDiscordメッセージの添付ファイルを、バイト列や署名付きURLを"
                    "モデルへ渡さずにHIVEで解析します。"
                ),
                risk=RiskLevel.EXTERNAL,
                keywords=(
                    "discord",
                    "attachment",
                    "image",
                    "video",
                    "ai-generated",
                    "deepfake",
                    "hive",
                ),
                side_effects=(
                    "許可されたDiscord添付ファイルを1件取得します。",
                    "保存済みの解析結果がない場合、HIVE APIを1回使用します。",
                ),
            ),
            DiscordAnalyzeAttachmentRequest,
            SyntheticMediaAnalyzeResponse,
            analyze_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.import_attachment",
                summary=(
                    "Discord添付ファイルを、このサーバー専用の隔離ワークスペースへ"
                    "取り込み、SHA-256を返します。"
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "attachment", "import", "file", "pdf", "zip"),
                side_effects=("隔離ワークスペース内のファイルを作成または置換します。",),
            ),
            DiscordImportAttachmentRequest,
            WorkspaceFileRecord,
            import_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_image_attachment",
                summary=(
                    "PNG・JPEG・GIF・WebP添付画像を、署名付きDiscord URLを公開せず"
                    "モデルの視覚入力として表示します。"
                ),
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "attachment", "image", "view", "vision"),
            ),
            DiscordViewImageAttachmentRequest,
            DiscordViewImageAttachmentResponse,
            view_image_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_message",
                summary="AIが判断した進捗または追加連絡をDiscordへ1件投稿します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "message",
                    "progress",
                    "update",
                    "follow-up",
                    "notify",
                ),
                side_effects=("Discordチャンネルにユーザーから見えるメッセージを作成します。",),
            ),
            DiscordSendMessageRequest,
            DiscordSendMessageResponse,
            send_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_file",
                summary="隔離ワークスペース内のファイルを現在のチャンネルへ送信します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "file", "attachment", "send", "deliver", "export"),
                side_effects=("添付ファイル付きDiscordメッセージを1件作成します。",),
            ),
            DiscordSendFileRequest,
            DiscordSendFileResponse,
            send_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_poll",
                summary="Discordテキストチャンネルに投票を作成します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "poll", "vote", "question"),
                side_effects=("Discordチャンネルにユーザーから見える投票を作成します。",),
            ),
            DiscordPollRequest,
            DiscordPollResponse,
            create_poll,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.connect_voice",
                summary="Simajilordの音声出力をDiscordボイスチャンネルへ接続します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "voice", "join", "connect", "vc"),
                side_effects=("BOTがボイスチャンネルへ参加または移動します。",),
            ),
            DiscordConnectVoiceRequest,
            DiscordConnectVoiceResponse,
            connect_voice,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.play_audio",
                summary=(
                    "公開メディアを依頼者のボイスチャンネルで再生します。"
                    "依頼者がVCにいない場合は、参加するまでキューに保持します。"
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", "audio", "play", "queue", "voice"),
                side_effects=(
                    "依頼者のボイスチャンネルへ参加する場合があります。",
                    "サーバーの永続キューへ曲を1件追加します。",
                ),
            ),
            AudioPlayRequest,
            AudioPlayResponse,
            play_audio,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.control_audio",
                summary=(
                    "依頼者がBOTと同じVCにいる場合、または待機中キューの所有者である場合に"
                    "サーバーの音声を操作します。"
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", "pause", "resume", "skip", "stop", "loop"),
                side_effects=("サーバーの永続音声セッションを変更します。",),
            ),
            AudioControlRequest,
            AudioControlResponse,
            control_audio,
        ),
        discord_audio_endpoint(
            "discord.pause_audio",
            "依頼者と同じVCで再生中の音楽を一時停止します。",
            ("pause",),
            AudioNoArgsRequest,
            pause_audio,
        ),
        discord_audio_endpoint(
            "discord.resume_audio",
            "依頼者と同じVCで一時停止中の音楽を再開します。",
            ("resume",),
            AudioNoArgsRequest,
            resume_audio,
        ),
        discord_audio_endpoint(
            "discord.skip_audio",
            "依頼者と同じVCで再生中の曲をスキップします。",
            ("skip",),
            AudioNoArgsRequest,
            skip_audio,
        ),
        discord_audio_endpoint(
            "discord.stop_audio",
            "依頼者と同じVCの再生を止め、音楽キューを空にします。",
            ("stop", "clear"),
            AudioNoArgsRequest,
            stop_audio,
        ),
        discord_audio_endpoint(
            "discord.leave_audio",
            "音楽キューを空にし、BOTを依頼者と同じVCから退出させます。",
            ("leave", "disconnect"),
            AudioNoArgsRequest,
            leave_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_loop",
            "音楽のループ方法を設定します。",
            ("loop", "repeat"),
            AudioLoopRequest,
            set_audio_loop,
        ),
        discord_audio_endpoint(
            "discord.remove_audio",
            "指定位置の待機曲をキューから削除します。",
            ("remove", "queue"),
            AudioQueuePositionRequest,
            remove_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_auto_leave",
            "人がいなくなったときのBOT自動退出を設定します。",
            ("auto", "leave"),
            AudioAutoLeaveRequest,
            set_audio_auto_leave,
        ),
        discord_audio_endpoint(
            "discord.shuffle_audio",
            "待機中の音楽キューをシャッフルします。",
            ("shuffle", "queue"),
            AudioNoArgsRequest,
            shuffle_audio,
        ),
        discord_audio_endpoint(
            "discord.seek_audio",
            "再生中の曲を指定秒へ移動します。",
            ("seek", "position"),
            AudioSeekRequest,
            seek_audio,
        ),
        discord_audio_endpoint(
            "discord.tune_audio",
            "音楽の速度とピッチを設定します。",
            ("speed", "pitch", "tune"),
            AudioTuneRequest,
            tune_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_volume",
            "音楽と読み上げの音量を百分率で設定します。",
            ("volume", "speech"),
            AudioVolumeRequest,
            set_audio_volume,
        ),
        discord_audio_endpoint(
            "discord.move_audio",
            "待機曲をキュー内の別の位置へ移動します。",
            ("move", "queue", "reorder"),
            AudioMoveRequest,
            move_audio,
        ),
        discord_audio_endpoint(
            "discord.clear_my_audio",
            "依頼者自身が追加した待機曲だけを削除します。",
            ("clear", "mine", "requester"),
            AudioNoArgsRequest,
            clear_my_audio,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.speak",
                summary=(
                    "VOICEVOXで短い文章を依頼者のVCへ読み上げます。"
                    "読み上げ中は再生中の音楽を自動調整します。"
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "voice", "tts", "speak", "say", "voicevox"),
                side_effects=(
                    "依頼者のボイスチャンネルへ参加する場合があります。",
                    "合成音声を再生します。",
                ),
            ),
            SpeechSpeakRequest,
            SpeechSpeakResponse,
            speak,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_status",
                summary="現在のDiscord読み上げ経路だけを確認します。",
                risk=RiskLevel.READ,
                keywords=("discord", "read aloud", "status", "route", "tts"),
            ),
            ReadAloudStatusRequest,
            ReadAloudResponse,
            read_aloud_status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_add_sources",
                summary="指定した会話チャンネルを参加中VCの読み上げ対象へ追加します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "add", "channel", "tts"),
                side_effects=("読み上げ対象チャンネルを永続設定へ追加します。",),
            ),
            ReadAloudAddSourcesRequest,
            ReadAloudResponse,
            read_aloud_add_sources,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_remove_source",
                summary="指定した会話チャンネルを読み上げ対象から外します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "remove", "channel", "tts"),
                side_effects=("読み上げ対象チャンネルを永続設定から外します。",),
            ),
            ReadAloudRemoveSourceRequest,
            ReadAloudResponse,
            read_aloud_remove_source,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_disable",
                summary="このDiscordサーバーの読み上げ経路を停止します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "disable", "stop", "tts"),
                side_effects=("このサーバーの読み上げ経路を削除します。",),
            ),
            ReadAloudDisableRequest,
            ReadAloudResponse,
            read_aloud_disable,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_policy_status",
                summary="読み上げ辞書・除外・入退室通知の現在値を確認します。",
                risk=RiskLevel.READ,
                keywords=("discord", "read aloud", "policy", "settings", "tts"),
            ),
            ReadAloudStatusRequest,
            ReadAloudPolicyResponse,
            read_aloud_policy_status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_list",
                summary="このDiscordサーバーの読み上げ辞書を一覧表示します。",
                risk=RiskLevel.READ,
                keywords=("discord", "read aloud", "dictionary", "pronunciation"),
            ),
            ReadAloudDictionaryListRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_list,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_set",
                summary="表記と読みをDiscordサーバー別読み上げ辞書へ登録します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "dictionary", "pronunciation"),
                side_effects=("サーバー別読み上げ辞書を更新します。",),
            ),
            ReadAloudDictionarySetRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_remove",
                summary="表記をDiscordサーバー別読み上げ辞書から削除します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "dictionary", "remove"),
                side_effects=("サーバー別読み上げ辞書を更新します。",),
            ),
            ReadAloudDictionaryRemoveRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_remove,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_exclusion_set",
                summary="ユーザーまたはロールの読み上げ除外を設定・解除します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "ignore", "mute", "role"),
                side_effects=("読み上げ除外設定を更新します。",),
            ),
            ReadAloudExclusionSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_exclusion_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_announcements_set",
                summary="VCへの参加・退出・移動の読み上げを設定します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "join", "leave", "move"),
                side_effects=("VC入退室通知の読み上げ設定を更新します。",),
            ),
            ReadAloudAnnouncementsSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_announcements_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_semantics_set",
                summary="投稿者名・返信先・添付の意味的な読み上げを設定します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "author", "reply", "attachment"),
                side_effects=("意味的な読み上げ設定を更新します。",),
            ),
            ReadAloudSemanticsSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_semantics_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.manage_read_aloud",
                summary=(
                    "読み上げ経路を確認します。サーバー管理権限があれば設定・無効化も行えます。"
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "tts", "route", "voicevox"),
                side_effects=("自動読み上げの永続設定を変更する場合があります。",),
            ),
            ReadAloudRequest,
            ReadAloudResponse,
            manage_read_aloud,
        ),
    )


def _guild(client: discord.Client, context: InvocationContext) -> discord.Guild:
    if context.workspace_id is None:
        raise UserError("このDiscord機能はサーバー内で使用してください。")
    try:
        guild_id = int(context.workspace_id)
    except ValueError as exc:
        raise UserError("DiscordサーバーIDが正しくありません。") from exc
    guild = client.get_guild(guild_id)
    if guild is None:
        raise UserError("Discordサーバーを利用できません。")
    return guild


async def _actor_member(
    guild: discord.Guild,
    context: InvocationContext,
) -> discord.Member:
    try:
        actor_id = int(context.actor_id)
    except ValueError as exc:
        raise UserError("discord.member_required") from exc
    member = guild.get_member(actor_id)
    if member is not None:
        return member
    try:
        member = await guild.fetch_member(actor_id)
    except (discord.NotFound, discord.Forbidden) as exc:
        raise UserError("discord.member_required") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.member_lookup_failed") from exc
    if member.id != actor_id:
        raise UserError("discord.member_required")
    return member


def _member_voice_channel(
    member: discord.Member,
) -> discord.VoiceChannel | discord.StageChannel | None:
    state = member.voice
    if state is None or not isinstance(
        state.channel,
        (discord.VoiceChannel, discord.StageChannel),
    ):
        return None
    return state.channel


def _assert_same_voice(
    destination_id: str | None,
    channel: discord.VoiceChannel | discord.StageChannel | None,
) -> None:
    if channel is None or destination_id is None or str(channel.id) != destination_id:
        raise UserError("audio.same_voice_required")


def _require_manage_guild(member: discord.Member) -> None:
    if not member.guild_permissions.manage_guild:
        raise UserError("discord.manage_guild_required")


async def _prepare_actor_audio(
    client: discord.Client,
    runtime: SimajilordRuntime,
    guild: discord.Guild,
    member: discord.Member,
) -> None:
    workspace_id = str(guild.id)
    session = runtime.audio.get_or_create(
        workspace_id,
        lambda: DiscordAudioOutput(client, guild.id),
    )
    channel = _member_voice_channel(member)
    if session.output.connected:
        _assert_same_voice(session.destination_id, channel)
        return
    if channel is not None:
        await runtime.audio.connect(workspace_id, str(channel.id))


def _snowflake(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        display_label = {
            "message": "メッセージ",
            "channel": "チャンネル",
            "voice channel": "ボイスチャンネル",
        }.get(label, label)
        raise UserError(f"{display_label}IDが正しくありません。") from exc


def _assert_agent_update_scope(
    context: InvocationContext,
    channel_id: str,
) -> None:
    _assert_agent_channel_scope(context, channel_id)
    if (
        context.transport == "agent"
        and context.origin_resource_id is not None
        and channel_id != context.origin_resource_id
    ):
        raise UserError("discord.agent_update_channel_forbidden")


def _text_channel(
    guild: discord.Guild,
    channel_id: str,
) -> discord.TextChannel | discord.Thread:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        raise UserError("指定先は書き込み可能なテキストチャンネルではありません。")
    return channel


def _message_channel(
    guild: discord.Guild,
    channel_id: str,
) -> DiscordMessageChannel:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if not isinstance(
        channel,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        raise UserError("指定先はDiscordのメッセージチャンネルではありません。")
    return channel


def _message_record(
    message: discord.Message,
    *,
    event_message_id: str | None = None,
) -> DiscordMessageRecord:
    if str(message.id) == event_message_id:
        preview, truncated = _bounded_event_message(message.content)
    else:
        preview, truncated = _message_preview(message.content)
    return DiscordMessageRecord(
        message_id=str(message.id),
        channel_id=str(message.channel.id),
        author_id=str(message.author.id),
        author_name=message.author.display_name,
        author_is_bot=message.author.bot,
        content_preview=preview,
        content_length=len(message.content),
        preview_truncated=truncated,
        created_at_iso=message.created_at.isoformat(),
        attachments=tuple(
            _attachment_record(attachment) for attachment in message.attachments
        ),
        reference_message_id=(
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        ),
    )


def _attachment_record(attachment: discord.Attachment) -> DiscordAttachmentRecord:
    return DiscordAttachmentRecord(
        attachment_id=str(attachment.id),
        filename=attachment.filename,
        content_type=attachment.content_type,
        size_bytes=attachment.size,
    )


def _custom_emoji_records(content: str) -> tuple[DiscordCustomEmojiRecord, ...]:
    """Return bounded, de-duplicated metadata in first-appearance order."""

    ordered: list[tuple[str, str, bool]] = []
    occurrence_counts: dict[str, int] = {}
    for match in _CUSTOM_EMOJI_PATTERN.finditer(content):
        emoji_id = match.group("id")
        occurrence_counts[emoji_id] = occurrence_counts.get(emoji_id, 0) + 1
        if occurrence_counts[emoji_id] == 1 and len(ordered) < _CUSTOM_EMOJI_PREVIEW_LIMIT:
            ordered.append(
                (
                    emoji_id,
                    match.group("name"),
                    match.group("animated") == "a",
                )
            )
    return tuple(
        DiscordCustomEmojiRecord(
            index=index,
            emoji_id=emoji_id,
            name=name,
            animated=animated,
            occurrences=occurrence_counts[emoji_id],
        )
        for index, (emoji_id, name, animated) in enumerate(ordered)
    )


def _sticker_records(
    stickers: list[discord.StickerItem],
) -> tuple[DiscordStickerRecord, ...]:
    return tuple(
        DiscordStickerRecord(
            index=index,
            sticker_id=str(sticker.id),
            name=sticker.name,
            format=sticker.format.name,
            animated=sticker.format is not discord.StickerFormatType.png,
        )
        for index, sticker in enumerate(stickers[:10])
    )


def _prepare_discord_animated_media(
    content: bytes,
    *,
    mode: Literal["preview", "animation", "frame"],
    frame_index: int,
) -> _DiscordAnimatedMedia:
    """Validate model media and optionally extract an exact animation frame."""

    try:
        with Image.open(io.BytesIO(content)) as image:
            image_format = image.format
            frame_count = getattr(image, "n_frames", 1)
            durations: list[int] = []
            for index in range(frame_count):
                image.seek(index)
                raw_duration = image.info.get("duration")
                durations.append(
                    int(raw_duration)
                    if isinstance(raw_duration, (int, float))
                    else 0
                )
            duration_ms = sum(durations) or None
            if mode == "animation":
                if image_format == "GIF":
                    content_type = "image/gif"
                elif image_format == "PNG" and frame_count > 1:
                    content_type = "image/apng"
                else:
                    raise ValueError("discord.custom_emoji_invalid")
                return _DiscordAnimatedMedia(
                    content=content,
                    content_type=content_type,
                    preview_kind="full_animation",
                    frame_index=None,
                    frame_count=frame_count,
                    duration_ms=duration_ms,
                )
            selected_frame = frame_index if mode == "frame" else 0
            if selected_frame >= frame_count:
                raise ValueError("discord.custom_emoji_frame_invalid")
            image.seek(selected_frame)
            output = io.BytesIO()
            image.convert("RGBA").save(output, format="PNG", optimize=True)
            return _DiscordAnimatedMedia(
                content=output.getvalue(),
                content_type="image/png",
                preview_kind=(
                    "selected_animation_frame"
                    if mode == "frame"
                    else "representative_static_frame"
                ),
                frame_index=selected_frame,
                frame_count=frame_count,
                duration_ms=duration_ms,
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("discord.custom_emoji_invalid") from exc


def parse_discord_message_link(content: str) -> DiscordMessageLink | None:
    """Parse one bare Discord message link; angle brackets are an explicit opt-out."""

    value = content.strip()
    if not value or value.startswith("<") or value.endswith(">"):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or (parsed.hostname or "").lower()
        not in {"discord.com", "ptb.discord.com", "canary.discord.com"}
    ):
        return None
    path = parsed.path.rstrip("/").split("/")
    if len(path) != 5 or path[1] != "channels":
        return None
    guild_id, channel_id, message_id = path[2:]
    if not all(value.isdigit() and int(value) > 0 for value in path[2:]):
        return None
    return DiscordMessageLink(
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
    )


def _expanded_attachment(
    attachment: discord.Attachment,
) -> DiscordExpandedAttachmentRecord:
    return DiscordExpandedAttachmentRecord(
        filename=attachment.filename,
        content_type=attachment.content_type,
        size_bytes=attachment.size,
        url=attachment.url,
        proxy_url=attachment.proxy_url,
        spoiler=attachment.is_spoiler(),
    )


def _expanded_embed(item: discord.Embed) -> DiscordExpandedEmbedRecord:
    return DiscordExpandedEmbedRecord(
        title=item.title,
        description=item.description,
        url=item.url,
        image_url=item.image.url if item.image.url else None,
        thumbnail_url=item.thumbnail.url if item.thumbnail.url else None,
    )


def _expanded_poll(poll: discord.Poll | None) -> DiscordExpandedPollRecord | None:
    if poll is None:
        return None
    return DiscordExpandedPollRecord(
        question=poll.question,
        answers=tuple(answer.text for answer in poll.answers),
    )


def _expanded_reply(message: discord.Message) -> tuple[str | None, str | None]:
    referenced = message.reference.resolved if message.reference is not None else None
    if not isinstance(referenced, discord.Message):
        return None, None
    content = referenced.content.strip()
    if not content and referenced.attachments:
        content = "添付ファイル"
    return referenced.author.display_name, content[:300] or None


async def _attachment(
    client: discord.Client,
    context: InvocationContext,
    channel_id: str,
    message_id: str,
    attachment_index: int,
) -> tuple[discord.Message, discord.Attachment]:
    guild = _guild(client, context)
    _assert_agent_channel_scope(context, channel_id)
    channel = _message_channel(guild, channel_id)
    if not 0 <= attachment_index <= 9:
        raise UserError("discord.attachment_index_invalid")
    try:
        message = await channel.fetch_message(_snowflake(message_id, "message"))
    except discord.NotFound as exc:
        raise UserError("Discordメッセージが見つかりませんでした。") from exc
    except discord.DiscordException as exc:
        raise UserError("Discordメッセージを取得できませんでした。") from exc
    try:
        attachment = message.attachments[attachment_index]
    except IndexError as exc:
        raise UserError("discord.attachment_missing") from exc
    return message, attachment


def _image_media_type(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if (
        len(content) >= 12
        and content.startswith(b"RIFF")
        and content[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


async def _reply_context(
    channel: DiscordMessageChannel,
    message: discord.Message,
    *,
    max_depth: int,
    chunk_characters: int,
) -> tuple[DiscordReplyContextRecord, ...]:
    records: list[DiscordReplyContextRecord] = []
    seen = {message.id}
    current = message
    for _ in range(max_depth):
        reference = current.reference
        if reference is None or reference.message_id is None:
            break
        if reference.channel_id is not None and reference.channel_id != channel.id:
            break
        if reference.message_id in seen:
            break
        seen.add(reference.message_id)
        resolved = reference.resolved
        if isinstance(resolved, discord.Message):
            parent = resolved
        else:
            try:
                parent = await channel.fetch_message(reference.message_id)
            except (discord.NotFound, discord.DiscordException):
                break
        chunk = parent.content[:chunk_characters]
        records.append(
            DiscordReplyContextRecord(
                message_id=str(parent.id),
                author_id=str(parent.author.id),
                author_name=parent.author.display_name,
                author_is_bot=parent.author.bot,
                content_chunk=chunk,
                content_length=len(parent.content),
                complete=len(chunk) == len(parent.content),
                created_at_iso=parent.created_at.isoformat(),
                attachments=tuple(
                    _attachment_record(attachment)
                    for attachment in parent.attachments
                ),
                reference_message_id=(
                    str(parent.reference.message_id)
                    if parent.reference and parent.reference.message_id
                    else None
                ),
            )
        )
        current = parent
    return tuple(records)


def _message_preview(content: str) -> tuple[str, bool]:
    """Show all short content, otherwise a 25-character head and 5-character tail."""

    if len(content) <= 30:
        return content, False
    return f"{content[:25]}…{content[-5:]}", True


def _bounded_event_message(content: str) -> tuple[str, bool]:
    """Return one bounded chunk of the event that caused this agent turn."""

    maximum = 1_000
    return content[:maximum], len(content) > maximum


def _discord_event_message_id(context: InvocationContext) -> str | None:
    prefix = "discord:message:"
    if not context.request_id.startswith(prefix):
        return None
    message_id = context.request_id.removeprefix(prefix)
    return message_id if message_id.isdecimal() else None


def _assert_agent_channel_scope(
    context: InvocationContext,
    channel_id: str,
) -> None:
    if context.transport == "agent" and channel_id not in context.resource_ids:
        raise UserError("AIにはこのDiscordチャンネルを閲覧する権限がありません。")


def agent_readable_channel_ids(
    guild: discord.Guild,
    actor: discord.Member | None,
    *,
    trusted_guild: bool,
    trigger_channel_id: int | None,
) -> tuple[str, ...]:
    """Resolve a non-forgeable read scope before an agent turn begins.

    A trusted private guild and an autonomous turn use the bot's own Discord
    visibility. Other user-triggered turns require both the bot and the actor
    to have View Channel and Read Message History. Private threads are only
    included for a regular user when they are the triggering thread.
    """

    bot_member = guild.me
    if bot_member is None:
        return ()
    use_bot_scope = trusted_guild or actor is None
    readable: list[str] = []
    channels: tuple[DiscordMessageChannel, ...] = (
        *guild.text_channels,
        *guild.threads,
        *guild.voice_channels,
        *guild.stage_channels,
    )
    for channel in channels:
        if not _can_read_messages(channel, bot_member):
            continue
        if not use_bot_scope:
            if actor is None or not _can_read_messages(channel, actor):
                continue
            if (
                isinstance(channel, discord.Thread)
                and channel.type is discord.ChannelType.private_thread
                and channel.id != trigger_channel_id
            ):
                continue
        readable.append(str(channel.id))
    return tuple(sorted(readable, key=int))


def _can_read_messages(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    can_read = permissions.view_channel and permissions.read_message_history
    if isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        return can_read and permissions.connect
    return can_read


async def _fetch_readable_message(
    guild: discord.Guild,
    *,
    channel_id: str,
    message_id: str,
    context: InvocationContext,
) -> tuple[DiscordMessageChannel, discord.Message]:
    """Fetch one exact message after actor, bot, thread, and voice checks."""

    channel = _message_channel(guild, channel_id)
    bot_member = guild.me
    if bot_member is None:
        raise UserError("discord.expand_unavailable")
    if context.transport == "agent":
        _assert_agent_channel_scope(context, channel_id)
    else:
        actor = await _actor_member(guild, context)
        if not _can_read_messages(channel, actor) or not _can_read_private_thread(
            channel, actor
        ):
            raise UserError("discord.expand_unavailable")
    if not _can_read_messages(channel, bot_member) or not _can_read_private_thread(
        channel, bot_member
    ):
        raise UserError("discord.expand_unavailable")
    try:
        message = await channel.fetch_message(_snowflake(message_id, "message"))
    except (discord.NotFound, discord.Forbidden) as exc:
        raise UserError("discord.expand_unavailable") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.expand_failed") from exc
    return channel, message


def _can_read_private_thread(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    if (
        not isinstance(channel, discord.Thread)
        or channel.type is not discord.ChannelType.private_thread
    ):
        return True
    permissions = channel.permissions_for(member)
    if permissions.administrator or permissions.manage_threads:
        return True
    return any(thread_member.id == member.id for thread_member in channel.members)


def _can_post_expanded_message(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    can_send = (
        permissions.send_messages_in_threads
        if isinstance(channel, discord.Thread)
        else permissions.send_messages
    )
    can_connect = (
        permissions.connect
        if isinstance(channel, (discord.VoiceChannel, discord.StageChannel))
        else True
    )
    return permissions.view_channel and can_send and permissions.embed_links and can_connect


def _can_post_quote_image(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    can_send = (
        permissions.send_messages_in_threads
        if isinstance(channel, discord.Thread)
        else permissions.send_messages
    )
    can_connect = (
        permissions.connect
        if isinstance(channel, (discord.VoiceChannel, discord.StageChannel))
        else True
    )
    return (
        permissions.view_channel
        and can_send
        and permissions.attach_files
        and can_connect
    )


def _quote_text(message: discord.Message) -> str:
    """Resolve Discord mentions while retaining custom emoji tokens for rendering."""

    text = message.content
    for member in message.mentions:
        replacement = f"@{member.display_name}"
        text = text.replace(f"<@{member.id}>", replacement)
        text = text.replace(f"<@!{member.id}>", replacement)
    for role in message.role_mentions:
        text = text.replace(f"<@&{role.id}>", f"@{role.name}")
    for channel in message.channel_mentions:
        text = text.replace(f"<#{channel.id}>", f"#{channel.name}")
    if text.strip():
        return text
    if message.stickers:
        return ""
    if message.attachments:
        image_count = sum(
            1
            for attachment in message.attachments
            if (attachment.content_type or "").startswith("image/")
        )
        if image_count == len(message.attachments):
            return f"画像を{image_count}件送信"
        return f"添付ファイルを{len(message.attachments)}件送信"
    if message.embeds:
        return "埋め込みメッセージ"
    return "…"


async def _quote_avatar(
    client: discord.Client,
    message: discord.Message,
) -> bytes | None:
    asset = message.author.display_avatar.with_static_format("png").with_size(512)
    try:
        content = await client.http.get_from_cdn(str(asset))
    except discord.DiscordException:
        return None
    if not content or len(content) > _QUOTE_AVATAR_MAX_BYTES:
        return None
    return content


async def _quote_custom_emojis(
    client: discord.Client,
    content: str,
) -> tuple[QuoteCustomEmojiAsset, ...]:
    async def fetch(record: DiscordCustomEmojiRecord) -> QuoteCustomEmojiAsset | None:
        extension = "gif" if record.animated else "png"
        url = (
            f"https://cdn.discordapp.com/emojis/{record.emoji_id}.{extension}"
            "?size=128&quality=lossless"
        )
        try:
            payload = await client.http.get_from_cdn(url)
        except discord.DiscordException:
            return None
        if not payload or len(payload) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            return None
        return QuoteCustomEmojiAsset(
            emoji_id=record.emoji_id,
            name=record.name,
            content=payload,
        )

    records = _custom_emoji_records(content)
    if not records:
        return ()
    fetched = await asyncio.gather(*(fetch(record) for record in records))
    return tuple(item for item in fetched if item is not None)


async def _quote_stickers(
    client: discord.Client,
    message: discord.Message,
) -> tuple[QuoteStickerAsset, ...]:
    async def fetch(sticker: discord.StickerItem) -> QuoteStickerAsset | None:
        url = (
            f"https://media.discordapp.net/stickers/{sticker.id}.png"
            if sticker.format is discord.StickerFormatType.lottie
            else str(sticker.url)
        )
        try:
            payload = await client.http.get_from_cdn(url)
        except discord.DiscordException:
            return None
        if not payload or len(payload) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            return None
        return QuoteStickerAsset(
            sticker_id=str(sticker.id),
            name=sticker.name,
            content=payload,
        )

    fetched = await asyncio.gather(
        *(fetch(sticker) for sticker in message.stickers[:3])
    )
    return tuple(item for item in fetched if item is not None)


def quote_message_has_animation(message: discord.Message) -> bool:
    """Return whether Quote can preserve at least one animated source asset."""

    if any(record.animated for record in _custom_emoji_records(message.content)):
        return True
    return any(
        sticker.format
        in {
            discord.StickerFormatType.apng,
            discord.StickerFormatType.gif,
        }
        for sticker in message.stickers
    )
