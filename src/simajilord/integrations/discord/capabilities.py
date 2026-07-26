"""Discord-specific capabilities available to commands and a future agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import discord

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
    limit: int = 50
    before_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordMessageRecord:
    message_id: str
    channel_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content: str
    created_at_iso: str
    attachment_urls: tuple[str, ...]
    reference_message_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordReadMessagesResponse:
    messages: tuple[DiscordMessageRecord, ...]
    oldest_message_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordSendMessageRequest:
    channel_id: str
    content: str


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
        records = [
            DiscordChannelRecord(
                channel_id=str(channel.id),
                name=channel.name,
                kind=str(channel.type),
                category_id=str(channel.category_id) if channel.category_id else None,
            )
            for channel in guild.channels
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
            )
        records.sort(key=lambda item: (item.kind, item.name, item.channel_id))
        return DiscordListChannelsResponse(channels=tuple(records))

    async def read_messages(
        request: DiscordReadMessagesRequest,
        context: InvocationContext,
    ) -> DiscordReadMessagesResponse:
        guild = _guild(client, context)
        channel = _text_channel(guild, request.channel_id)
        if not 1 <= request.limit <= 100:
            raise UserError("discord.message_limit_invalid")
        before = (
            discord.Object(_snowflake(request.before_message_id, "message"))
            if request.before_message_id
            else None
        )
        messages = [
            DiscordMessageRecord(
                message_id=str(message.id),
                channel_id=str(message.channel.id),
                author_id=str(message.author.id),
                author_name=message.author.display_name,
                author_is_bot=message.author.bot,
                content=message.content,
                created_at_iso=message.created_at.isoformat(),
                attachment_urls=tuple(str(attachment.url) for attachment in message.attachments),
                reference_message_id=(
                    str(message.reference.message_id)
                    if message.reference and message.reference.message_id
                    else None
                ),
            )
            async for message in channel.history(limit=request.limit, before=before)
        ]
        return DiscordReadMessagesResponse(
            messages=tuple(messages),
            oldest_message_id=messages[-1].message_id if messages else None,
        )

    async def send_message(
        request: DiscordSendMessageRequest,
        context: InvocationContext,
    ) -> DiscordSendMessageResponse:
        guild = _guild(client, context)
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
        runtime.audio.assert_connection_capacity(workspace_id)
        session = runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(client, guild.id),
        )
        if session.current is not None:
            output = session.output
            if isinstance(output, DiscordAudioOutput) and output.destination_id != channel.id:
                raise UserError("Audio is active in another voice channel.")
        await session.connect(str(channel.id))
        return DiscordConnectVoiceResponse(channel_id=str(channel.id), connected=True)

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
                name="discord.send_message",
                summary="Send one plain Discord message without mentions.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "message", "speak", "notify", "greet"),
                side_effects=("Creates a visible message in a Discord channel.",),
            ),
            DiscordSendMessageRequest,
            DiscordSendMessageResponse,
            send_message,
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


def _snowflake(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise UserError(f"The {label} ID is invalid.") from exc


def _text_channel(
    guild: discord.Guild,
    channel_id: str,
) -> discord.TextChannel | discord.Thread:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        raise UserError("The target is not a writable text channel.")
    return channel
