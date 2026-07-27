"""Discord-specific permission boundaries shared by commands and the agent."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePath
from typing import TypeAlias, cast

import discord

from simajilord.capabilities.audio import (
    AudioControlRequest,
    AudioControlResponse,
    AudioPlayRequest,
    AudioPlayResponse,
)
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudRequest,
    ReadAloudResponse,
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

from .audio import DiscordAudioOutput

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel
    | discord.Thread
    | discord.VoiceChannel
    | discord.StageChannel
)


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
    reference_message_id: str | None
    reply_context: tuple[DiscordReplyContextRecord, ...]


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
            reference_message_id=(
                str(message.reference.message_id)
                if message.reference and message.reference.message_id
                else None
            ),
            reply_context=reply_context,
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
        member = _actor_member(guild, context)
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

        guild = _guild(client, context)
        member = _actor_member(guild, context)
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
        response = await runtime.registry.invoke("audio.control", request, context)
        return cast(AudioControlResponse, response)

    async def speak(
        request: SpeechSpeakRequest,
        context: InvocationContext,
    ) -> SpeechSpeakResponse:
        """Prepare the requester's voice route before shared speech synthesis."""

        guild = _guild(client, context)
        member = _actor_member(guild, context)
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
        member = _actor_member(guild, context)
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
            _message_channel(guild, request.text_channel_id)
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


def _actor_member(
    guild: discord.Guild,
    context: InvocationContext,
) -> discord.Member:
    try:
        actor_id = int(context.actor_id)
    except ValueError as exc:
        raise UserError("discord.member_required") from exc
    member = guild.get_member(actor_id)
    if member is None:
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
