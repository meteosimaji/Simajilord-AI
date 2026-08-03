"""Central Discord visibility and posting permission policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import discord

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel | discord.Thread | discord.VoiceChannel | discord.StageChannel
)
DiscordReadableChannel: TypeAlias = DiscordMessageChannel | discord.ForumChannel


def permission_enabled(permissions: object, permission: str) -> bool:
    """Return only Discord's concrete boolean permission value."""

    return getattr(permissions, permission, False) is True


_permission_enabled = permission_enabled


@dataclass(frozen=True, slots=True)
class RequesterPrincipal:
    """A resolved human requester; absence is never a service identity."""

    member: discord.Member

    def __post_init__(self) -> None:
        if self.member.bot:
            raise ValueError("requester principal must be a human Discord member")


@dataclass(frozen=True, slots=True)
class ServicePrincipal:
    """The Discord application member acting under a host policy."""

    member: discord.Member

    def __post_init__(self) -> None:
        if not self.member.bot:
            raise ValueError("service principal must be a Discord bot member")


def readable_for_requester(
    guild: discord.Guild,
    principal: RequesterPrincipal,
) -> tuple[str, ...]:
    """Resolve the live intersection of requester and BOT visibility."""

    bot_member = guild.me
    if bot_member is None:
        return ()
    return _readable_channel_ids(
        guild,
        bot_member=bot_member,
        requester=principal.member,
    )


def readable_for_service(
    guild: discord.Guild,
    principal: ServicePrincipal,
) -> tuple[str, ...]:
    """Resolve BOT visibility only for an explicitly typed service principal."""

    bot_member = guild.me
    if bot_member is None or principal.member.id != bot_member.id:
        return ()
    return _readable_channel_ids(
        guild,
        bot_member=bot_member,
        requester=None,
    )


def _readable_channel_ids(
    guild: discord.Guild,
    *,
    bot_member: discord.Member,
    requester: discord.Member | None,
) -> tuple[str, ...]:
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
        if requester is not None:
            if not can_read_messages(channel, requester):
                continue
            if not can_read_private_thread(channel, requester):
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
    """Compare actual effective audiences without guessing incomplete membership."""

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


def read_aloud_audience_relation(
    guild: discord.Guild,
    source: DiscordReadableChannel,
    destination: discord.VoiceChannel | discord.StageChannel,
) -> DiscordAudienceRelation:
    """Prove every current human voice listener can read the text source."""

    member_count = guild.member_count
    cache_complete = guild.chunked is True or (
        isinstance(member_count, int)
        and not isinstance(member_count, bool)
        and len(guild.members) >= member_count
    )
    if not cache_complete:
        return "uncertain"
    listeners = tuple(member for member in destination.members if not member.bot)
    if any(
        not can_read_messages(source, listener)
        or not can_read_private_thread(source, listener)
        for listener in listeners
    ):
        return "broader"
    return "same_or_narrower"


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
    if _permission_enabled(permissions, "administrator"):
        return True
    can_read = _permission_enabled(
        permissions,
        "view_channel",
    ) and _permission_enabled(permissions, "read_message_history")
    if isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        return can_read and _permission_enabled(permissions, "connect")
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
    if _permission_enabled(
        permissions,
        "administrator",
    ) or _permission_enabled(permissions, "manage_threads"):
        return True
    return any(thread_member.id == member.id for thread_member in channel.members)


def can_post_expanded_message(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    if _permission_enabled(permissions, "administrator"):
        return True
    can_send = (
        _permission_enabled(permissions, "send_messages_in_threads")
        if isinstance(channel, discord.Thread)
        else _permission_enabled(permissions, "send_messages")
    )
    can_connect = (
        _permission_enabled(permissions, "connect")
        if isinstance(channel, (discord.VoiceChannel, discord.StageChannel))
        else True
    )
    return (
        _permission_enabled(permissions, "view_channel")
        and can_send
        and _permission_enabled(permissions, "embed_links")
        and can_connect
    )


def can_post_quote_image(
    channel: DiscordMessageChannel,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    if _permission_enabled(permissions, "administrator"):
        return True
    can_send = (
        _permission_enabled(permissions, "send_messages_in_threads")
        if isinstance(channel, discord.Thread)
        else _permission_enabled(permissions, "send_messages")
    )
    can_connect = (
        _permission_enabled(permissions, "connect")
        if isinstance(channel, (discord.VoiceChannel, discord.StageChannel))
        else True
    )
    return (
        _permission_enabled(permissions, "view_channel")
        and can_send
        and _permission_enabled(permissions, "attach_files")
        and can_connect
    )
