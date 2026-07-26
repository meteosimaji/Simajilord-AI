"""Discord-specific permission boundaries shared by commands and the agent."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePath
from typing import cast

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
            raise UserError("The user ID is invalid.") from exc
        member = guild.get_member(user_id)
        user = member or client.get_user(user_id)
        if user is None:
            try:
                user = await client.fetch_user(user_id)
            except discord.DiscordException as exc:
                raise UserError("The Discord user could not be found.") from exc
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
        channel = _text_channel(guild, request.channel_id)
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
        channel = _text_channel(guild, request.channel_id)
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
            raise UserError("The Discord message could not be found.") from exc
        except discord.DiscordException as exc:
            raise UserError("The Discord message could not be retrieved.") from exc
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
        channel = _text_channel(guild, request.channel_id)
        if not 0 <= request.attachment_index <= 9:
            raise UserError("discord.attachment_index_invalid")
        try:
            message = await channel.fetch_message(
                _snowflake(request.message_id, "message")
            )
        except discord.NotFound as exc:
            raise UserError("The Discord message could not be found.") from exc
        except discord.DiscordException as exc:
            raise UserError("The Discord message could not be retrieved.") from exc
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
            raise UserError("Discord messages must contain between 1 and 2000 characters.")
        try:
            channel_id = int(request.channel_id)
        except ValueError as exc:
            raise UserError("The channel ID is invalid.") from exc
        channel = guild.get_channel(channel_id)
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            raise UserError("The target is not a writable text channel.")
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
        channel = _text_channel(guild, request.channel_id)
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
            raise UserError("The poll question must contain between 1 and 300 characters.")
        if not 2 <= len(options) <= 10:
            raise UserError("A poll requires between 2 and 10 options.")
        if any(len(option) > 55 for option in options):
            raise UserError("Each poll option must be at most 55 characters.")
        if not 1 <= request.duration_hours <= 168:
            raise UserError("Poll duration must be between 1 and 168 hours.")
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
            raise UserError("The target is not a voice channel.")
        workspace_id = str(guild.id)
        session = runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(client, guild.id),
        )
        if session.current is not None:
            output = session.output
            if isinstance(output, DiscordAudioOutput) and output.destination_id != channel.id:
                raise UserError("Audio is active in another voice channel.")
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
                title=f"Spoken by request of {member.display_name}",
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
        if (
            request.action in {ReadAloudAction.CONFIGURE, ReadAloudAction.DISABLE}
            and not member.guild_permissions.manage_guild
        ):
            raise UserError("discord.manage_guild_required")
        if request.action is ReadAloudAction.CONFIGURE:
            if request.text_channel_id is None or request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            _text_channel(guild, request.text_channel_id)
            voice = guild.get_channel(
                _snowflake(request.audio_destination_id, "voice channel")
            )
            if not isinstance(voice, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("discord.voice_channel_required")
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
                summary="Inspect cached structure and identifiers for the current Discord server.",
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
                summary="Inspect one Discord user's public account and server membership data.",
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
                summary="List channels and identifiers in the current Discord server.",
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
                summary="Read a bounded page of messages from an allowed Discord channel.",
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
                    "Read one bounded contiguous chunk of a Discord message by ID. "
                    "By default, also reads a bounded same-channel reply chain. "
                    "Continue with next_offset only when more text is necessary."
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
                    "Analyze one attachment from an allowed Discord message with HIVE "
                    "without exposing its bytes or signed URL to the model."
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
                    "Downloads one allowed Discord attachment.",
                    "Consumes one HIVE request when no cached result exists.",
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
                    "Import one Discord attachment into this server's isolated agent "
                    "workspace and return its SHA-256."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "attachment", "import", "file", "pdf", "zip"),
                side_effects=("Creates or replaces one isolated workspace file.",),
            ),
            DiscordImportAttachmentRequest,
            WorkspaceFileRecord,
            import_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_image_attachment",
                summary=(
                    "View one bounded PNG, JPEG, GIF, or WebP attachment as model "
                    "visual input without exposing a signed Discord URL."
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
                summary="Post one agent-chosen progress or follow-up message.",
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
                side_effects=("Creates a visible message in a Discord channel.",),
            ),
            DiscordSendMessageRequest,
            DiscordSendMessageResponse,
            send_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_file",
                summary="Send one isolated workspace file back to the current channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("discord", "file", "attachment", "send", "deliver", "export"),
                side_effects=("Creates a Discord message with one file attachment.",),
            ),
            DiscordSendFileRequest,
            DiscordSendFileResponse,
            send_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_poll",
                summary="Create a bounded native poll in one Discord text channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "poll", "vote", "question"),
                side_effects=("Creates a visible poll in a Discord channel.",),
            ),
            DiscordPollRequest,
            DiscordPollResponse,
            create_poll,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.connect_voice",
                summary="Connect the platform audio output to one Discord voice channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "voice", "join", "connect", "vc"),
                side_effects=("Joins or moves the bot's server voice connection.",),
            ),
            DiscordConnectVoiceRequest,
            DiscordConnectVoiceResponse,
            connect_voice,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.play_audio",
                summary=(
                    "Play a resolved public media reference in the requester's voice "
                    "channel, or retain it until that requester joins voice."
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", "audio", "play", "queue", "voice"),
                side_effects=(
                    "May join the requester's voice channel.",
                    "Adds one track to the persistent server queue.",
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
                    "Control server audio only when the requester shares the bot's "
                    "voice channel or owns its waiting queue."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", "pause", "resume", "skip", "stop", "loop"),
                side_effects=("Changes the persistent server audio session.",),
            ),
            AudioControlRequest,
            AudioControlResponse,
            control_audio,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.speak",
                summary=(
                    "Speak a short message in the requester's voice channel through "
                    "VOICEVOX, automatically ducking active music."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "voice", "tts", "speak", "say", "voicevox"),
                side_effects=(
                    "May join the requester's voice channel.",
                    "Creates audible synthesized speech.",
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
                    "Inspect read-aloud routing, or configure and disable it when the "
                    "requester has Manage Server permission."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "read aloud", "tts", "route", "voicevox"),
                side_effects=("May change persistent automatic read-aloud routing.",),
            ),
            ReadAloudRequest,
            ReadAloudResponse,
            manage_read_aloud,
        ),
    )


def _guild(client: discord.Client, context: InvocationContext) -> discord.Guild:
    if context.workspace_id is None:
        raise UserError("This Discord capability requires a server.")
    try:
        guild_id = int(context.workspace_id)
    except ValueError as exc:
        raise UserError("The Discord server ID is invalid.") from exc
    guild = client.get_guild(guild_id)
    if guild is None:
        raise UserError("The Discord server is unavailable.")
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
        raise UserError(f"The {label} ID is invalid.") from exc


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
        raise UserError("The target is not a writable text channel.")
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
    channel = _text_channel(guild, channel_id)
    if not 0 <= attachment_index <= 9:
        raise UserError("discord.attachment_index_invalid")
    try:
        message = await channel.fetch_message(_snowflake(message_id, "message"))
    except discord.NotFound as exc:
        raise UserError("The Discord message could not be found.") from exc
    except discord.DiscordException as exc:
        raise UserError("The Discord message could not be retrieved.") from exc
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
    channel: discord.TextChannel | discord.Thread,
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
        raise UserError("The agent is not authorized to inspect that Discord channel.")


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
    channels: tuple[discord.TextChannel | discord.Thread, ...] = (
        *guild.text_channels,
        *guild.threads,
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
    channel: discord.TextChannel | discord.Thread,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    return permissions.view_channel and permissions.read_message_history
