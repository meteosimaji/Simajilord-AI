"""Central Discord visibility and posting permission policy."""

from __future__ import annotations

from typing import TypeAlias

import discord

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel | discord.Thread | discord.VoiceChannel | discord.StageChannel
)


def agent_readable_channel_ids(
    guild: discord.Guild,
    actor: discord.Member | None,
    *,
    trusted_guild: bool,
    trigger_channel_id: int | None,
) -> tuple[str, ...]:
    """Resolve the non-forgeable channel scope for one AI turn."""

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
        if not can_read_messages(channel, bot_member):
            continue
        if not use_bot_scope:
            if actor is None or not can_read_messages(channel, actor):
                continue
            if (
                isinstance(channel, discord.Thread)
                and channel.type is discord.ChannelType.private_thread
                and channel.id != trigger_channel_id
            ):
                continue
        readable.append(str(channel.id))
    return tuple(sorted(readable, key=int))


def can_read_messages(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    can_read = permissions.view_channel and permissions.read_message_history
    if isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        return can_read and permissions.connect
    return can_read


def can_read_private_thread(
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
