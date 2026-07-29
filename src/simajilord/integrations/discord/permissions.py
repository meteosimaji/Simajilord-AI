"""Central Discord visibility and posting permission policy."""

from __future__ import annotations

from typing import Literal, TypeAlias

import discord

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel | discord.Thread | discord.VoiceChannel | discord.StageChannel
)
DiscordReadableChannel: TypeAlias = DiscordMessageChannel | discord.ForumChannel


def agent_readable_channel_ids(
    guild: discord.Guild,
    actor: discord.Member | None,
    *,
    trusted_guild: bool,
    trigger_channel_id: int | None,
) -> tuple[str, ...]:
    """Resolve channels readable by both the active requester and the bot.

    A trusted guild changes which agent capabilities may be exposed, but it must
    never let the agent borrow the bot's wider message visibility.
    """

    del trigger_channel_id
    bot_member = guild.me
    bot_principal = trusted_guild and actor is None
    if bot_member is None or (actor is None and not bot_principal):
        return ()
    readable: list[str] = []
    forums = guild.forums
    if not isinstance(forums, (list, tuple)):
        forums = []
    channels: tuple[DiscordReadableChannel, ...] = (
        *guild.text_channels,
        *forums,
        *guild.threads,
        *guild.voice_channels,
        *guild.stage_channels,
    )
    for channel in channels:
        if not can_read_messages(channel, bot_member):
            continue
        if not can_read_private_thread(channel, bot_member):
            continue
        if not bot_principal:
            if actor is None or not can_read_messages(channel, actor):
                continue
            if not can_read_private_thread(channel, actor):
                continue
        readable.append(str(channel.id))
    return tuple(sorted(readable, key=int))


DiscordAudienceRelation: TypeAlias = Literal[
    "same_or_narrower",
    "broader",
    "uncertain",
]
DiscordVisibility: TypeAlias = Literal["guild_public", "restricted", "uncertain"]


def channel_visibility(
    guild: discord.Guild,
    channel: DiscordReadableChannel,
) -> DiscordVisibility:
    """Classify a source from effective member permissions when complete."""

    readers, complete = _effective_reader_ids(guild, channel)
    if not complete:
        return "uncertain"
    member_ids = {member.id for member in guild.members if not member.bot}
    if member_ids.issubset(readers):
        return "guild_public"
    return "restricted"


def disclosure_audience_relation(
    source_guild: discord.Guild,
    source: DiscordReadableChannel,
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
) -> DiscordAudienceRelation:
    """Compare actual effective audiences; reject only a provable mismatch."""

    if source_guild.id == destination_guild.id and source.id == destination.id:
        return "same_or_narrower"
    source_readers, source_complete = _effective_reader_ids(source_guild, source)
    destination_readers, destination_complete = _effective_reader_ids(
        destination_guild,
        destination,
    )
    if destination_readers - source_readers:
        return "broader"
    if source_complete and destination_complete:
        return "same_or_narrower"
    return "uncertain"


def _effective_reader_ids(
    guild: discord.Guild,
    channel: DiscordReadableChannel,
) -> tuple[set[int], bool]:
    readers = {
        member.id
        for member in guild.members
        if not member.bot
        and can_read_messages(channel, member)
        and can_read_private_thread(channel, member)
    }
    member_count = guild.member_count
    complete = guild.chunked is True or (
        isinstance(member_count, int)
        and not isinstance(member_count, bool)
        and len(guild.members) >= member_count
    )
    return readers, complete


def can_read_messages(
    channel: DiscordReadableChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    can_read = permissions.view_channel and permissions.read_message_history
    if isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        return can_read and permissions.connect
    return can_read


def can_read_private_thread(
    channel: DiscordReadableChannel,
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


def can_post_expanded_message(
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


def can_post_quote_image(
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
    return permissions.view_channel and can_send and permissions.attach_files and can_connect
