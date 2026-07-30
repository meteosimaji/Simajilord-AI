"""Broad, bounded Discord platform inspection capabilities.

The core Discord adapter keeps the high-frequency message and audio operations.
This module exposes the remaining bot-visible platform resources without
returning webhook tokens, credentials, or unbounded Discord payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import discord

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime

from .capabilities import (
    _activity_record,
    _can_view_channel,
    _enabled_flag_names,
    _readable_message_channel,
    _requested_guild,
    _require_common_guild,
    _require_guild_permission,
    _snowflake,
)
from .permissions import can_read_messages as _can_read_messages
from .permissions import can_read_private_thread as _can_read_private_thread

DiscordPlatformResourceKind = Literal[
    "audit_log",
    "ban",
    "invite",
    "webhook",
    "scheduled_event",
    "scheduled_event_user",
    "emoji",
    "sticker",
    "soundboard",
    "application_emoji",
    "sku",
    "entitlement",
    "default_soundboard",
    "premium_sticker_pack",
    "automod_rule",
    "integration",
    "template",
    "stage_instance",
    "role_member_count",
    "onboarding",
    "welcome_screen",
    "widget",
    "vanity_invite",
    "active_thread",
    "guild_preview",
]

_MAX_PLATFORM_OFFSET = 200
_MAX_PLATFORM_PAGE = 25


@dataclass(frozen=True, slots=True)
class DiscordListMembersRequest:
    query: str = field(
        default="",
        metadata={
            "description": (
                "Optional case-insensitive username, global-name, nickname, or display-name "
                "prefix. Empty pages through members by stable ID."
            )
        },
    )
    guild_id: str | None = field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )
    offset: int = field(
        default=0,
        metadata={"description": "Zero-based bounded result offset; copy next_offset."},
    )
    limit: int = field(
        default=15,
        metadata={"description": "Members returned per page, from 1 through 25."},
    )


@dataclass(frozen=True, slots=True)
class DiscordMemberRecord:
    user_id: str
    username: str
    display_name: str
    global_name: str | None
    nickname: str | None
    bot: bool
    system: bool
    joined_at_iso: str | None
    created_at_iso: str
    status: str | None
    presence_available: bool
    activities: tuple[str, ...]
    voice_channel_id: str | None
    voice_channel_name: str | None
    role_ids: tuple[str, ...]
    role_names: tuple[str, ...]
    enabled_guild_permissions: tuple[str, ...]
    pending: bool
    timed_out_until_iso: str | None
    premium_since_iso: str | None
    avatar_url: str


@dataclass(frozen=True, slots=True)
class DiscordListMembersResponse:
    source_guild_id: str
    members: tuple[DiscordMemberRecord, ...]
    next_offset: int | None
    complete: bool
    member_cache_complete: bool
    presence_intent_enabled: bool


@dataclass(frozen=True, slots=True)
class DiscordInspectChannelRequest:
    channel_id: str
    guild_id: str | None = field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordPermissionOverwriteRecord:
    target_id: str
    target_name: str
    target_kind: Literal["role", "member"]
    allowed: tuple[str, ...]
    denied: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscordForumTagRecord:
    tag_id: str
    name: str
    moderated: bool
    emoji: str | None


@dataclass(frozen=True, slots=True)
class DiscordChannelDetailsResponse:
    source_guild_id: str
    channel_id: str
    name: str
    kind: str
    created_at_iso: str
    category_id: str | None
    position: int | None
    topic: str | None
    nsfw: bool | None
    slowmode_seconds: int | None
    default_thread_slowmode_seconds: int | None
    default_auto_archive_minutes: int | None
    bitrate: int | None
    user_limit: int | None
    rtc_region: str | None
    video_quality_mode: str | None
    parent_id: str | None
    owner_id: str | None
    archived: bool | None
    locked: bool | None
    invitable: bool | None
    auto_archive_minutes: int | None
    message_count: int | None
    member_count: int | None
    last_message_id: str | None
    last_pin_at_iso: str | None
    flags: tuple[str, ...]
    requester_permissions: tuple[str, ...]
    bot_permissions: tuple[str, ...]
    overwrites: tuple[DiscordPermissionOverwriteRecord, ...]
    forum_tags: tuple[DiscordForumTagRecord, ...]
    applied_tag_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscordListPinsRequest:
    channel_id: str
    guild_id: str | None = None
    offset: int = 0
    limit: int = 15


@dataclass(frozen=True, slots=True)
class DiscordPinnedMessageRecord:
    message_id: str
    author_id: str
    author_name: str
    content_preview: str
    created_at_iso: str
    edited_at_iso: str | None
    pinned_at_iso: str | None
    attachment_count: int
    embed_count: int
    jump_url: str


@dataclass(frozen=True, slots=True)
class DiscordListPinsResponse:
    source_guild_id: str
    source_channel_id: str
    messages: tuple[DiscordPinnedMessageRecord, ...]
    next_offset: int | None
    complete: bool


@dataclass(frozen=True, slots=True)
class DiscordListReactionUsersRequest:
    channel_id: str
    message_id: str
    emoji: str
    guild_id: str | None = None
    offset: int = 0
    limit: int = 25


@dataclass(frozen=True, slots=True)
class DiscordReactionUserRecord:
    user_id: str
    username: str
    display_name: str
    bot: bool


@dataclass(frozen=True, slots=True)
class DiscordListReactionUsersResponse:
    source_guild_id: str
    source_channel_id: str
    message_id: str
    emoji: str
    users: tuple[DiscordReactionUserRecord, ...]
    next_offset: int | None
    complete: bool


@dataclass(frozen=True, slots=True)
class DiscordListPollVotersRequest:
    channel_id: str
    message_id: str
    answer_id: str
    guild_id: str | None = None
    offset: int = 0
    limit: int = 25


@dataclass(frozen=True, slots=True)
class DiscordListPollVotersResponse:
    source_guild_id: str
    source_channel_id: str
    message_id: str
    answer_id: str
    answer_text: str
    voters: tuple[DiscordReactionUserRecord, ...]
    next_offset: int | None
    complete: bool


@dataclass(frozen=True, slots=True)
class DiscordListThreadMembersRequest:
    thread_id: str
    guild_id: str | None = None
    offset: int = 0
    limit: int = 25


@dataclass(frozen=True, slots=True)
class DiscordThreadMemberRecord:
    user_id: str
    display_name: str | None
    joined_at_iso: str
    flags: int


@dataclass(frozen=True, slots=True)
class DiscordListThreadMembersResponse:
    source_guild_id: str
    thread_id: str
    members: tuple[DiscordThreadMemberRecord, ...]
    next_offset: int | None
    complete: bool


@dataclass(frozen=True, slots=True)
class DiscordListPlatformResourcesRequest:
    kind: DiscordPlatformResourceKind
    resource_id: str | None = field(
        default=None,
        metadata={
            "description": (
                "Parent or target resource ID when required: the scheduled event ID "
                "for scheduled_event_user, or one entitlement ID for entitlement."
            )
        },
    )
    guild_id: str | None = field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )
    offset: int = field(
        default=0,
        metadata={"description": "Zero-based bounded result offset; copy next_offset."},
    )
    limit: int = field(
        default=15,
        metadata={"description": "Resource records returned, from 1 through 25."},
    )


@dataclass(frozen=True, slots=True)
class DiscordResourceField:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class DiscordPlatformResourceRecord:
    resource_id: str
    kind: str
    name: str
    fields: tuple[DiscordResourceField, ...]


@dataclass(frozen=True, slots=True)
class DiscordListPlatformResourcesResponse:
    source_guild_id: str
    kind: DiscordPlatformResourceKind
    resources: tuple[DiscordPlatformResourceRecord, ...]
    next_offset: int | None
    complete: bool
    secrets_omitted: bool = True


@dataclass(frozen=True, slots=True)
class DiscordInspectApplicationRequest:
    pass


@dataclass(frozen=True, slots=True)
class DiscordApplicationResponse:
    application_id: str
    application_name: str
    description: str
    bot_user_id: str
    bot_username: str
    owner_id: str | None
    owner_name: str | None
    team_id: str | None
    guild_count: int
    cached_user_count: int
    latency_ms: float
    shard_count: int | None
    intents: tuple[str, ...]
    public: bool
    requires_code_grant: bool
    flags: tuple[str, ...]
    install_count: int | None


def build_discord_platform_endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> tuple[CapabilityEndpoint, ...]:
    """Build low-frequency Discord platform inspection endpoints."""

    del runtime

    async def list_members(
        request: DiscordListMembersRequest,
        context: InvocationContext,
    ) -> DiscordListMembersResponse:
        _validate_page(request.offset, request.limit, "member")
        query = " ".join(request.query.split())
        if len(query) > 100:
            raise UserError("discord.member_query_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        await _require_common_guild(guild, context)
        needed = request.offset + request.limit + 1
        members: list[discord.Member]
        if query:
            try:
                members = await guild.query_members(
                    query=query,
                    limit=min(100, needed),
                    presences=_presence_intent(client),
                    cache=True,
                )
            except discord.DiscordException as exc:
                raise UserError("discord.member_lookup_failed") from exc
            folded = query.casefold()
            members = [
                member
                for member in members
                if any(
                    folded in candidate.casefold()
                    for candidate in (
                        member.name,
                        member.display_name,
                        member.global_name or "",
                        member.nick or "",
                    )
                )
            ]
        elif _member_cache_complete(guild):
            members = list(guild.members)
        else:
            members = []
            try:
                async for member in guild.fetch_members(limit=needed):
                    members.append(member)
            except discord.Forbidden as exc:
                raise UserError("discord.member_list_forbidden") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.member_lookup_failed") from exc
        members.sort(key=lambda item: (item.display_name.casefold(), item.id))
        page = members[request.offset : request.offset + request.limit]
        has_more = len(members) > request.offset + len(page)
        return DiscordListMembersResponse(
            source_guild_id=str(guild.id),
            members=tuple(_member_record(member, client=client) for member in page),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
            member_cache_complete=_member_cache_complete(guild),
            presence_intent_enabled=_presence_intent(client),
        )

    async def inspect_channel(
        request: DiscordInspectChannelRequest,
        context: InvocationContext,
    ) -> DiscordChannelDetailsResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        if bot is None:
            raise UserError("discord.guild_unavailable")
        channel = guild.get_channel_or_thread(_snowflake(request.channel_id, "channel"))
        if not isinstance(channel, (discord.abc.GuildChannel, discord.Thread)):
            raise UserError("discord.channel_unavailable")
        if not _can_view_channel(channel, actor) or not _can_view_channel(channel, bot):
            raise UserError("discord.agent_read_channel_forbidden")
        if isinstance(
            channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
                discord.ForumChannel,
            ),
        ) and (
            not _can_read_messages(channel, actor)
            or not _can_read_private_thread(channel, actor)
            or not _can_read_messages(channel, bot)
            or not _can_read_private_thread(channel, bot)
        ):
            raise UserError("discord.agent_read_channel_forbidden")
        actor_permissions = channel.permissions_for(actor)
        bot_permissions = channel.permissions_for(bot)
        overwrites: list[DiscordPermissionOverwriteRecord] = []
        for target, overwrite in getattr(channel, "overwrites", {}).items():
            allow, deny = overwrite.pair()
            overwrites.append(
                DiscordPermissionOverwriteRecord(
                    target_id=str(target.id),
                    target_name=getattr(target, "display_name", getattr(target, "name", "Unknown")),
                    target_kind=("role" if isinstance(target, discord.Role) else "member"),
                    allowed=_enabled_flag_names(allow),
                    denied=_enabled_flag_names(deny),
                )
            )
        forum_tags = tuple(
            DiscordForumTagRecord(
                tag_id=str(tag.id),
                name=tag.name,
                moderated=tag.moderated,
                emoji=str(tag.emoji) if tag.emoji is not None else None,
            )
            for tag in getattr(channel, "available_tags", ())
        )
        return DiscordChannelDetailsResponse(
            source_guild_id=str(guild.id),
            channel_id=str(channel.id),
            name=channel.name,
            kind=str(channel.type),
            created_at_iso=(
                _optional_datetime(channel.created_at)
                or discord.utils.snowflake_time(channel.id).isoformat()
            ),
            category_id=_optional_id(getattr(channel, "category_id", None)),
            position=_optional_int(getattr(channel, "position", None)),
            topic=_optional_string(getattr(channel, "topic", None)),
            nsfw=_optional_bool(getattr(channel, "nsfw", None)),
            slowmode_seconds=_optional_int(getattr(channel, "slowmode_delay", None)),
            default_thread_slowmode_seconds=_optional_int(
                getattr(channel, "default_thread_slowmode_delay", None)
            ),
            default_auto_archive_minutes=_optional_int(
                getattr(channel, "default_auto_archive_duration", None)
            ),
            bitrate=_optional_int(getattr(channel, "bitrate", None)),
            user_limit=_optional_int(getattr(channel, "user_limit", None)),
            rtc_region=_optional_string(getattr(channel, "rtc_region", None)),
            video_quality_mode=_optional_string(
                getattr(channel, "video_quality_mode", None)
            ),
            parent_id=_optional_id(getattr(channel, "parent_id", None)),
            owner_id=_optional_id(getattr(channel, "owner_id", None)),
            archived=_optional_bool(getattr(channel, "archived", None)),
            locked=_optional_bool(getattr(channel, "locked", None)),
            invitable=_optional_bool(getattr(channel, "invitable", None)),
            auto_archive_minutes=_optional_int(
                getattr(channel, "auto_archive_duration", None)
            ),
            message_count=_optional_int(getattr(channel, "message_count", None)),
            member_count=_optional_int(getattr(channel, "member_count", None)),
            last_message_id=_optional_id(getattr(channel, "last_message_id", None)),
            last_pin_at_iso=_optional_datetime(getattr(channel, "last_pin_timestamp", None)),
            flags=_enabled_flag_names(getattr(channel, "flags", ())),
            requester_permissions=_enabled_flag_names(actor_permissions),
            bot_permissions=_enabled_flag_names(bot_permissions),
            overwrites=tuple(
                sorted(overwrites, key=lambda item: (item.target_kind, item.target_name))
            ),
            forum_tags=forum_tags,
            applied_tag_ids=tuple(
                str(tag.id) for tag in getattr(channel, "applied_tags", ())
            ),
        )

    async def list_pins(
        request: DiscordListPinsRequest,
        context: InvocationContext,
    ) -> DiscordListPinsResponse:
        _validate_page(request.offset, request.limit, "pin")
        guild, channel = await _readable_message_channel(
            client,
            context,
            guild_id=request.guild_id,
            channel_id=request.channel_id,
        )
        messages: list[discord.Message] = []
        try:
            async for message in channel.pins(limit=request.offset + request.limit + 1):
                messages.append(message)
        except discord.Forbidden as exc:
            raise UserError("discord.pins_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.pins_fetch_failed") from exc
        page = messages[request.offset : request.offset + request.limit]
        has_more = len(messages) > request.offset + len(page)
        return DiscordListPinsResponse(
            source_guild_id=str(guild.id),
            source_channel_id=str(channel.id),
            messages=tuple(_pinned_message_record(message) for message in page),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
        )

    async def list_reaction_users(
        request: DiscordListReactionUsersRequest,
        context: InvocationContext,
    ) -> DiscordListReactionUsersResponse:
        _validate_page(request.offset, request.limit, "reaction")
        guild, channel = await _readable_message_channel(
            client,
            context,
            guild_id=request.guild_id,
            channel_id=request.channel_id,
        )
        try:
            message = await channel.fetch_message(_snowflake(request.message_id, "message"))
        except discord.NotFound as exc:
            raise UserError("discord.message_not_found") from exc
        except discord.Forbidden as exc:
            raise UserError("discord.message_read_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_fetch_failed") from exc
        reaction = next(
            (item for item in message.reactions if str(item.emoji) == request.emoji),
            None,
        )
        if reaction is None:
            raise UserError("discord.reaction_not_found")
        users: list[discord.User | discord.Member] = []
        try:
            async for user in reaction.users(
                limit=request.offset + request.limit + 1
            ):
                users.append(user)
        except discord.DiscordException as exc:
            raise UserError("discord.reaction_users_fetch_failed") from exc
        page = users[request.offset : request.offset + request.limit]
        has_more = len(users) > request.offset + len(page)
        return DiscordListReactionUsersResponse(
            source_guild_id=str(guild.id),
            source_channel_id=str(channel.id),
            message_id=str(message.id),
            emoji=str(reaction.emoji),
            users=tuple(
                DiscordReactionUserRecord(
                    user_id=str(user.id),
                    username=user.name,
                    display_name=user.display_name,
                    bot=user.bot,
                )
                for user in page
            ),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
        )

    async def list_thread_members(
        request: DiscordListThreadMembersRequest,
        context: InvocationContext,
    ) -> DiscordListThreadMembersResponse:
        _validate_page(request.offset, request.limit, "thread_member")
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        thread = guild.get_thread(_snowflake(request.thread_id, "channel"))
        if thread is None or bot is None:
            raise UserError("discord.thread_unavailable")
        if (
            not _can_read_messages(thread, actor)
            or not _can_read_private_thread(thread, actor)
            or not _can_read_messages(thread, bot)
            or not _can_read_private_thread(thread, bot)
        ):
            raise UserError("discord.agent_read_channel_forbidden")
        try:
            members = await thread.fetch_members()
        except discord.Forbidden as exc:
            raise UserError("discord.thread_members_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.thread_members_fetch_failed") from exc
        members.sort(key=lambda item: item.id)
        page = members[request.offset : request.offset + request.limit]
        has_more = len(members) > request.offset + len(page)
        return DiscordListThreadMembersResponse(
            source_guild_id=str(guild.id),
            thread_id=str(thread.id),
            members=tuple(_thread_member_record(item, guild) for item in page),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
        )

    async def list_poll_voters(
        request: DiscordListPollVotersRequest,
        context: InvocationContext,
    ) -> DiscordListPollVotersResponse:
        _validate_page(request.offset, request.limit, "poll_voter")
        guild, channel = await _readable_message_channel(
            client,
            context,
            guild_id=request.guild_id,
            channel_id=request.channel_id,
        )
        try:
            message = await channel.fetch_message(
                _snowflake(request.message_id, "message"),
            )
        except discord.NotFound as exc:
            raise UserError("discord.message_not_found") from exc
        except discord.Forbidden as exc:
            raise UserError("discord.message_read_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_fetch_failed") from exc
        poll = message.poll
        if poll is None:
            raise UserError("discord.poll_not_found")
        answer_id = _snowflake(request.answer_id, "poll answer")
        answer = poll.get_answer(answer_id)
        if answer is None:
            raise UserError("discord.poll_answer_not_found")
        users: list[discord.User | discord.Member] = []
        try:
            async for user in answer.voters(
                limit=request.offset + request.limit + 1,
            ):
                users.append(user)
        except discord.Forbidden as exc:
            raise UserError("discord.poll_voters_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.poll_voters_fetch_failed") from exc
        page = users[request.offset : request.offset + request.limit]
        has_more = len(users) > request.offset + len(page)
        return DiscordListPollVotersResponse(
            source_guild_id=str(guild.id),
            source_channel_id=str(channel.id),
            message_id=str(message.id),
            answer_id=str(answer.id),
            answer_text=answer.text,
            voters=tuple(
                DiscordReactionUserRecord(
                    user_id=str(user.id),
                    username=user.name,
                    display_name=user.display_name,
                    bot=user.bot,
                )
                for user in page
            ),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
        )

    async def list_platform_resources(
        request: DiscordListPlatformResourcesRequest,
        context: InvocationContext,
    ) -> DiscordListPlatformResourcesResponse:
        _validate_page(request.offset, request.limit, "resource")
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        if bot is None:
            raise UserError("discord.guild_unavailable")
        resources = await _platform_resources(
            client,
            guild,
            actor=actor,
            bot=bot,
            kind=request.kind,
            resource_id=request.resource_id,
            needed=request.offset + request.limit + 1,
        )
        page = resources[request.offset : request.offset + request.limit]
        has_more = len(resources) > request.offset + len(page)
        return DiscordListPlatformResourcesResponse(
            source_guild_id=str(guild.id),
            kind=request.kind,
            resources=tuple(page),
            next_offset=(request.offset + len(page) if has_more else None),
            complete=not has_more,
        )

    async def inspect_application(
        _: DiscordInspectApplicationRequest,
        context: InvocationContext,
    ) -> DiscordApplicationResponse:
        guild = _requested_guild(client, context, None)
        await _require_common_guild(guild, context)
        user = client.user
        if user is None:
            raise UserError("discord.bot_unavailable")
        try:
            info = await client.application_info()
        except discord.DiscordException as exc:
            raise UserError("discord.application_fetch_failed") from exc
        intents = getattr(client, "intents", discord.Intents.none())
        owner = info.owner
        team = info.team
        return DiscordApplicationResponse(
            application_id=str(info.id),
            application_name=info.name,
            description=info.description,
            bot_user_id=str(user.id),
            bot_username=user.name,
            owner_id=str(owner.id) if owner is not None else None,
            owner_name=owner.name if owner is not None else None,
            team_id=str(team.id) if team is not None else None,
            guild_count=len(client.guilds),
            cached_user_count=len(client.users),
            latency_ms=round(client.latency * 1_000, 3),
            shard_count=client.shard_count,
            intents=_enabled_flag_names(intents),
            public=info.bot_public,
            requires_code_grant=info.bot_require_code_grant,
            flags=_enabled_flag_names(info.flags),
            install_count=getattr(info, "approximate_user_install_count", None),
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.list_members",
                summary=(
                    "Search or page through server members with roles, effective guild "
                    "permissions, cached presence/activity, and voice participation."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "members",
                    "users",
                    "presence",
                    "roles",
                    "voice",
                    "メンバー",
                    "ユーザー",
                    "オンライン",
                    "一覧",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_query_invalid",
                    "discord.member_list_forbidden",
                    "discord.member_lookup_failed",
                ),
                timeout_seconds=20,
            ),
            DiscordListMembersRequest,
            DiscordListMembersResponse,
            list_members,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_channel",
                summary=(
                    "Inspect one visible channel, category, forum, voice/stage channel, "
                    "or thread with settings, tags, overwrites, and both effective "
                    "requester and bot permissions."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "channel",
                    "permissions",
                    "overwrites",
                    "forum",
                    "thread",
                    "voice",
                    "チャンネル",
                    "権限",
                    "設定",
                    "上書き",
                ),
                requires_workspace=True,
                timeout_seconds=15,
            ),
            DiscordInspectChannelRequest,
            DiscordChannelDetailsResponse,
            inspect_channel,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_pins",
                summary="Page through pinned messages in a mutually visible channel.",
                risk=RiskLevel.READ,
                keywords=("discord", "pins", "pinned", "messages", "ピン", "固定"),
                requires_workspace=True,
                timeout_seconds=15,
            ),
            DiscordListPinsRequest,
            DiscordListPinsResponse,
            list_pins,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_reaction_users",
                summary=(
                    "Page through users represented by one reaction on a readable message."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "reaction",
                    "emoji",
                    "users",
                    "リアクション",
                    "絵文字",
                    "誰",
                ),
                requires_workspace=True,
                timeout_seconds=15,
            ),
            DiscordListReactionUsersRequest,
            DiscordListReactionUsersResponse,
            list_reaction_users,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_thread_members",
                summary="Page through the current member list of a mutually visible thread.",
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "thread",
                    "members",
                    "participants",
                    "スレッド",
                    "参加者",
                ),
                requires_workspace=True,
                timeout_seconds=15,
            ),
            DiscordListThreadMembersRequest,
            DiscordListThreadMembersResponse,
            list_thread_members,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_poll_voters",
                summary=(
                    "Page through users who selected one answer on a readable Discord poll."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "poll",
                    "answer",
                    "voters",
                    "投票",
                    "回答",
                    "投票者",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.poll_not_found",
                    "discord.poll_answer_not_found",
                    "discord.poll_voters_forbidden",
                    "discord.poll_voters_fetch_failed",
                ),
                timeout_seconds=15,
            ),
            DiscordListPollVotersRequest,
            DiscordListPollVotersResponse,
            list_poll_voters,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_platform_resources",
                summary=(
                    "Inspect one bounded Discord server resource family: audit log, bans, "
                    "invites, token-free webhooks, scheduled events, emojis, stickers, "
                    "soundboard and public application assets, scheduled-event users, "
                    "owner-only application SKUs/entitlements, AutoMod rules, "
                    "integrations, templates, stage instances, role counts, onboarding, "
                    "welcome screen, widget, vanity invite, active threads, or a "
                    "REST-fetched guild preview. "
                    "Permission-sensitive families require both requester and bot access."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "audit",
                    "bans",
                    "invites",
                    "webhooks",
                    "events",
                    "emoji",
                    "stickers",
                    "soundboard",
                    "sku",
                    "entitlements",
                    "automod",
                    "integrations",
                    "templates",
                    "stage",
                    "監査ログ",
                    "BAN",
                    "招待",
                    "イベント",
                ),
                requires_workspace=True,
                timeout_seconds=30,
            ),
            DiscordListPlatformResourcesRequest,
            DiscordListPlatformResourcesResponse,
            list_platform_resources,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_application",
                summary=(
                    "Inspect this bot application's public identity, gateway intents, "
                    "latency, shards, and cached reach without exposing credentials."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "discord",
                    "bot",
                    "application",
                    "intents",
                    "latency",
                    "shards",
                    "BOT",
                    "状態",
                    "機能",
                ),
                requires_workspace=True,
                timeout_seconds=15,
            ),
            DiscordInspectApplicationRequest,
            DiscordApplicationResponse,
            inspect_application,
        ),
    )


async def _platform_resources(
    client: discord.Client,
    guild: discord.Guild,
    *,
    actor: discord.Member,
    bot: discord.Member,
    kind: DiscordPlatformResourceKind,
    resource_id: str | None,
    needed: int,
) -> list[DiscordPlatformResourceRecord]:
    if kind == "audit_log":
        _require_both_guild_permission(actor, bot, "view_audit_log")
        records: list[DiscordPlatformResourceRecord] = []
        try:
            async for item in guild.audit_logs(limit=needed):
                records.append(
                    _resource(
                        item.id,
                        str(item.action),
                        kind,
                        created_at=item.created_at,
                        actor_id=getattr(item.user, "id", None),
                        actor_name=getattr(item.user, "name", None),
                        target_id=getattr(item.target, "id", None),
                        target_type=(
                            type(item.target).__name__
                            if item.target is not None
                            else None
                        ),
                        reason=item.reason,
                        category=item.category,
                    )
                )
        except discord.DiscordException as exc:
            raise UserError("discord.audit_log_fetch_failed") from exc
        return records
    if kind == "ban":
        _require_both_guild_permission(actor, bot, "ban_members")
        bans: list[DiscordPlatformResourceRecord] = []
        try:
            async for entry in guild.bans(limit=needed):
                bans.append(
                    _resource(
                        entry.user.id,
                        entry.user.name,
                        kind,
                        display_name=entry.user.display_name,
                        bot=entry.user.bot,
                        reason=entry.reason,
                    )
                )
        except discord.DiscordException as exc:
            raise UserError("discord.bans_fetch_failed") from exc
        return bans
    if kind == "invite":
        _require_both_guild_permission(actor, bot, "manage_guild")
        try:
            invites = await guild.invites()
        except discord.DiscordException as exc:
            raise UserError("discord.invites_fetch_failed") from exc
        return [
            _resource(
                invite.id,
                invite.code,
                kind,
                url=invite.url,
                channel_id=getattr(invite.channel, "id", None),
                channel_name=getattr(invite.channel, "name", None),
                inviter_id=getattr(invite.inviter, "id", None),
                inviter_name=getattr(invite.inviter, "name", None),
                uses=invite.uses,
                max_uses=invite.max_uses,
                max_age_seconds=invite.max_age,
                temporary=invite.temporary,
                created_at=invite.created_at,
                expires_at=invite.expires_at,
            )
            for invite in invites
        ]
    if kind == "webhook":
        _require_both_guild_permission(actor, bot, "manage_webhooks")
        try:
            webhooks = await guild.webhooks()
        except discord.DiscordException as exc:
            raise UserError("discord.webhooks_fetch_failed") from exc
        return [
            _resource(
                webhook.id,
                webhook.name or "Unnamed webhook",
                kind,
                channel_id=webhook.channel_id,
                type=webhook.type,
                user_id=getattr(webhook.user, "id", None),
                application_id=getattr(webhook, "application_id", None),
                token_omitted=True,
                url_omitted=True,
            )
            for webhook in webhooks
            if (
                webhook.channel_id is not None
                and (channel := guild.get_channel(webhook.channel_id))
                and _can_view_channel(channel, actor)
                and _can_view_channel(channel, bot)
            )
        ]
    if kind == "scheduled_event":
        try:
            events = await guild.fetch_scheduled_events(with_counts=True)
        except discord.DiscordException as exc:
            raise UserError("discord.events_fetch_failed") from exc
        records = []
        for event in events:
            channel = event.channel
            if channel is not None and (
                not _can_view_channel(channel, actor)
                or not _can_view_channel(channel, bot)
            ):
                continue
            records.append(
                _resource(
                    event.id,
                    event.name,
                    kind,
                    description=event.description,
                    status=event.status,
                    entity_type=event.entity_type,
                    channel_id=getattr(channel, "id", None),
                    location=event.location,
                    start_time=event.start_time,
                    end_time=event.end_time,
                    creator_id=getattr(event.creator, "id", None),
                    user_count=event.user_count,
                    cover_image_url=event.cover_image.url if event.cover_image else None,
                )
            )
        return records
    if kind == "scheduled_event_user":
        if resource_id is None:
            raise UserError("discord.event_id_required")
        try:
            event = await guild.fetch_scheduled_event(
                _snowflake(resource_id, "event"),
            )
        except discord.DiscordException as exc:
            raise UserError("discord.event_not_found") from exc
        event_channel = event.channel
        if event_channel is not None and (
            not _can_view_channel(event_channel, actor)
            or not _can_view_channel(event_channel, bot)
        ):
            raise UserError("discord.agent_read_channel_forbidden")
        users: list[DiscordPlatformResourceRecord] = []
        try:
            async for user in event.users(limit=needed):
                users.append(
                    _resource(
                        user.id,
                        user.display_name,
                        kind,
                        username=user.name,
                        global_name=user.global_name,
                        bot=user.bot,
                        event_id=event.id,
                        event_name=event.name,
                    )
                )
        except discord.DiscordException as exc:
            raise UserError("discord.event_users_fetch_failed") from exc
        return users
    if kind == "emoji":
        try:
            emojis = await guild.fetch_emojis()
        except discord.DiscordException as exc:
            raise UserError("discord.emojis_fetch_failed") from exc
        return [
            _resource(
                emoji.id,
                emoji.name,
                kind,
                url=emoji.url,
                animated=emoji.animated,
                available=emoji.available,
                managed=emoji.managed,
                require_colons=emoji.require_colons,
                role_ids=tuple(str(role.id) for role in emoji.roles),
                creator_id=getattr(emoji.user, "id", None),
            )
            for emoji in emojis
        ]
    if kind == "sticker":
        try:
            stickers = await guild.fetch_stickers()
        except discord.DiscordException as exc:
            raise UserError("discord.stickers_fetch_failed") from exc
        return [
            _resource(
                sticker.id,
                sticker.name,
                kind,
                description=sticker.description,
                emoji=sticker.emoji,
                format=sticker.format,
                url=sticker.url,
                available=sticker.available,
                creator_id=getattr(sticker.user, "id", None),
            )
            for sticker in stickers
        ]
    if kind == "soundboard":
        try:
            sounds = await guild.fetch_soundboard_sounds()
        except discord.DiscordException as exc:
            raise UserError("discord.soundboard_fetch_failed") from exc
        return [
            _resource(
                sound.id,
                sound.name,
                kind,
                volume=sound.volume,
                emoji=sound.emoji,
                available=sound.available,
                creator_id=getattr(sound.user, "id", None),
            )
            for sound in sounds
        ]
    if kind == "application_emoji":
        try:
            emojis = await client.fetch_application_emojis()
        except discord.DiscordException as exc:
            raise UserError("discord.application_emojis_fetch_failed") from exc
        return [
            _resource(
                emoji.id,
                emoji.name,
                kind,
                url=emoji.url,
                animated=emoji.animated,
                available=emoji.available,
                managed=emoji.managed,
            )
            for emoji in emojis
        ]
    if kind == "sku":
        await _require_application_owner(client, actor)
        try:
            skus = await client.fetch_skus()
        except discord.DiscordException as exc:
            raise UserError("discord.application_resources_fetch_failed") from exc
        return [
            _resource(
                sku.id,
                sku.name,
                kind,
                type=sku.type.name,
                application_id=sku.application_id,
                slug=sku.slug,
                flags=_enabled_flag_names(sku.flags),
            )
            for sku in sorted(skus, key=lambda item: item.id)[:needed]
        ]
    if kind == "entitlement":
        await _require_application_owner(client, actor)
        try:
            if resource_id is not None:
                entitlements = [
                    await client.fetch_entitlement(
                        _snowflake(resource_id, "entitlement"),
                    )
                ]
            else:
                entitlements = [
                    entitlement
                    async for entitlement in client.entitlements(
                        limit=needed,
                        guild=guild,
                        exclude_ended=False,
                        exclude_deleted=False,
                    )
                ]
        except discord.DiscordException as exc:
            raise UserError("discord.application_resources_fetch_failed") from exc
        return [
            _resource(
                entitlement.id,
                entitlement.id,
                kind,
                sku_id=entitlement.sku_id,
                application_id=entitlement.application_id,
                user_id=entitlement.user_id,
                guild_id=entitlement.guild_id,
                type=entitlement.type.name,
                deleted=entitlement.deleted,
                starts_at=entitlement.starts_at,
                ends_at=entitlement.ends_at,
                consumed=entitlement.consumed,
            )
            for entitlement in entitlements[:needed]
        ]
    if kind == "default_soundboard":
        try:
            default_sounds = await client.fetch_soundboard_default_sounds()
        except discord.DiscordException as exc:
            raise UserError("discord.default_soundboard_fetch_failed") from exc
        return [
            _resource(
                sound.id,
                sound.name,
                kind,
                volume=sound.volume,
                emoji=sound.emoji,
                url=sound.url,
            )
            for sound in default_sounds
        ]
    if kind == "premium_sticker_pack":
        try:
            packs = await client.fetch_premium_sticker_packs()
        except discord.DiscordException as exc:
            raise UserError("discord.sticker_packs_fetch_failed") from exc
        return [
            _resource(
                pack.id,
                pack.name,
                kind,
                description=pack.description,
                sku_id=pack.sku_id,
                cover_sticker_id=pack.cover_sticker_id,
                sticker_count=len(pack.stickers),
                banner_url=pack.banner.url if pack.banner is not None else None,
            )
            for pack in packs
        ]
    if kind == "automod_rule":
        _require_both_guild_permission(actor, bot, "manage_guild")
        try:
            rules = await guild.fetch_automod_rules()
        except discord.DiscordException as exc:
            raise UserError("discord.automod_fetch_failed") from exc
        return [
            _resource(
                rule.id,
                rule.name,
                kind,
                enabled=rule.enabled,
                event_type=rule.event_type,
                trigger_type=rule.trigger.type,
                actions=tuple(str(action.type) for action in rule.actions),
                exempt_role_ids=tuple(str(role.id) for role in rule.exempt_roles),
                exempt_channel_ids=tuple(
                    str(channel.id) for channel in rule.exempt_channels
                ),
                creator_id=rule.creator_id,
            )
            for rule in rules
        ]
    if kind == "integration":
        _require_both_guild_permission(actor, bot, "manage_guild")
        try:
            integrations = await guild.integrations()
        except discord.DiscordException as exc:
            raise UserError("discord.integrations_fetch_failed") from exc
        return [
            _resource(
                integration.id,
                integration.name,
                kind,
                type=integration.type,
                enabled=integration.enabled,
                account_id=integration.account.id,
                account_name=integration.account.name,
                application_id=getattr(
                    getattr(integration, "application", None),
                    "id",
                    None,
                ),
                user_id=getattr(integration.user, "id", None),
            )
            for integration in integrations
        ]
    if kind == "template":
        _require_both_guild_permission(actor, bot, "manage_guild")
        try:
            templates = await guild.templates()
        except discord.DiscordException as exc:
            raise UserError("discord.templates_fetch_failed") from exc
        return [
            _resource(
                template.code,
                template.name,
                kind,
                description=template.description,
                usage_count=template.uses,
                creator_id=getattr(template.creator, "id", None),
                creator_name=getattr(template.creator, "name", None),
                created_at=template.created_at,
                updated_at=template.updated_at,
                is_dirty=template.is_dirty,
            )
            for template in templates
        ]
    if kind == "stage_instance":
        records = []
        for channel in guild.stage_channels:
            if not _can_view_channel(channel, actor) or not _can_view_channel(channel, bot):
                continue
            try:
                instance = await channel.fetch_instance()
            except discord.NotFound:
                continue
            except discord.DiscordException as exc:
                raise UserError("discord.stage_instances_fetch_failed") from exc
            records.append(
                _resource(
                    instance.id,
                    instance.topic,
                    kind,
                    channel_id=channel.id,
                    channel_name=channel.name,
                    privacy_level=instance.privacy_level,
                    discoverable_disabled=instance.discoverable_disabled,
                    scheduled_event_id=instance.scheduled_event_id,
                )
            )
        return records
    if kind == "role_member_count":
        try:
            counts = await guild.role_member_counts()
        except discord.DiscordException as exc:
            raise UserError("discord.role_member_counts_fetch_failed") from exc
        records = []
        for role_or_id, count in counts.items():
            role_id = getattr(role_or_id, "id", None)
            if role_id is None:
                continue
            role = guild.get_role(role_id)
            records.append(
                _resource(
                    role_id,
                    role.name if role is not None else str(role_id),
                    kind,
                    count=count,
                    role_cached=role is not None,
                )
            )
        records.sort(key=lambda item: (item.name.casefold(), item.resource_id))
        return records
    if kind == "onboarding":
        try:
            onboarding = await guild.onboarding()
        except discord.DiscordException as exc:
            raise UserError("discord.onboarding_fetch_failed") from exc
        return [
            _resource(
                prompt.id,
                prompt.title,
                kind,
                enabled=onboarding.enabled,
                mode=onboarding.mode,
                type=prompt.type,
                single_select=prompt.single_select,
                required=prompt.required,
                in_onboarding=prompt.in_onboarding,
                option_count=len(prompt.options),
                option_titles=tuple(option.title for option in prompt.options),
                default_channel_ids=tuple(
                    str(channel_id)
                    for channel_id in onboarding.default_channel_ids
                ),
            )
            for prompt in onboarding.prompts
        ] or [
            _resource(
                guild.id,
                "Onboarding",
                kind,
                enabled=onboarding.enabled,
                mode=onboarding.mode,
                prompt_count=0,
                default_channel_ids=tuple(
                    str(channel_id)
                    for channel_id in onboarding.default_channel_ids
                ),
            )
        ]
    if kind == "welcome_screen":
        try:
            screen = await guild.welcome_screen()
        except discord.DiscordException as exc:
            raise UserError("discord.welcome_screen_fetch_failed") from exc
        channels = getattr(screen, "welcome_channels", ())
        return [
            _resource(
                getattr(item, "channel_id", index),
                getattr(item, "description", None) or f"Welcome channel {index + 1}",
                kind,
                enabled=screen.enabled,
                description=getattr(screen, "description", None),
                channel_id=getattr(item, "channel_id", None),
                emoji=getattr(item, "emoji", None),
            )
            for index, item in enumerate(channels)
        ] or [
            _resource(
                guild.id,
                "Welcome screen",
                kind,
                enabled=screen.enabled,
                description=getattr(screen, "description", None),
                channel_count=0,
            )
        ]
    if kind == "widget":
        try:
            widget = await guild.widget()
        except discord.DiscordException as exc:
            raise UserError("discord.widget_fetch_failed") from exc
        return [
            _resource(
                widget.id,
                widget.name,
                kind,
                presence_count=widget.presence_count,
                channel_count=len(widget.channels),
                member_count=len(widget.members),
                invite_url=widget.invite_url,
            )
        ]
    if kind == "vanity_invite":
        _require_both_guild_permission(actor, bot, "manage_guild")
        try:
            invite = await guild.vanity_invite()
        except discord.DiscordException as exc:
            raise UserError("discord.vanity_invite_fetch_failed") from exc
        if invite is None:
            return []
        return [
            _resource(
                invite.code,
                invite.code,
                kind,
                url=invite.url,
                uses=invite.uses,
                channel_id=getattr(invite.channel, "id", None),
            )
        ]
    if kind == "active_thread":
        try:
            threads = await guild.active_threads()
        except discord.DiscordException as exc:
            raise UserError("discord.active_threads_fetch_failed") from exc
        records = []
        for thread in threads:
            if (
                not _can_view_channel(thread, actor)
                or not _can_read_messages(thread, actor)
                or not _can_read_private_thread(thread, actor)
                or not _can_view_channel(thread, bot)
                or not _can_read_messages(thread, bot)
                or not _can_read_private_thread(thread, bot)
            ):
                continue
            records.append(
                _resource(
                    thread.id,
                    thread.name,
                    kind,
                    type=thread.type,
                    parent_id=thread.parent_id,
                    owner_id=thread.owner_id,
                    archived=thread.archived,
                    locked=thread.locked,
                    invitable=thread.invitable,
                    message_count=thread.message_count,
                    member_count=thread.member_count,
                    total_message_sent=thread.total_message_sent,
                    slowmode_seconds=thread.slowmode_delay,
                    auto_archive_minutes=thread.auto_archive_duration,
                    archive_timestamp=thread.archive_timestamp,
                    applied_tag_ids=tuple(str(tag.id) for tag in thread.applied_tags),
                )
            )
        records.sort(key=lambda item: (item.name.casefold(), item.resource_id))
        return records[:needed]
    if kind == "guild_preview":
        try:
            preview = await client.fetch_guild_preview(guild.id)
        except discord.DiscordException as exc:
            raise UserError("discord.guild_preview_fetch_failed") from exc
        return [
            _resource(
                preview.id,
                preview.name,
                kind,
                description=preview.description,
                approximate_member_count=preview.approximate_member_count,
                approximate_presence_count=preview.approximate_presence_count,
                features=tuple(preview.features),
                emoji_count=len(preview.emojis),
                sticker_count=len(preview.stickers),
                icon_url=(preview.icon.url if preview.icon is not None else None),
                splash_url=(
                    preview.splash.url if preview.splash is not None else None
                ),
                discovery_splash_url=(
                    preview.discovery_splash.url
                    if preview.discovery_splash is not None
                    else None
                ),
            )
        ]
    raise UserError("discord.resource_kind_invalid")


def _member_record(
    member: discord.Member,
    *,
    client: discord.Client,
) -> DiscordMemberRecord:
    voice = member.voice
    channel = voice.channel if voice is not None else None
    presence_available = _presence_intent(client)
    return DiscordMemberRecord(
        user_id=str(member.id),
        username=member.name,
        display_name=member.display_name,
        global_name=member.global_name,
        nickname=member.nick,
        bot=member.bot,
        system=member.system,
        joined_at_iso=(member.joined_at.isoformat() if member.joined_at else None),
        created_at_iso=member.created_at.isoformat(),
        status=str(member.status) if presence_available else None,
        presence_available=presence_available,
        activities=(
            tuple(
                f"{record.type}: {record.name}"
                for record in (
                    _activity_record(activity) for activity in member.activities
                )
            )
            if presence_available
            else ()
        ),
        voice_channel_id=str(channel.id) if channel is not None else None,
        voice_channel_name=getattr(channel, "name", None),
        role_ids=tuple(str(role.id) for role in member.roles[1:]),
        role_names=tuple(role.name for role in member.roles[1:]),
        enabled_guild_permissions=_enabled_flag_names(member.guild_permissions),
        pending=member.pending is True,
        timed_out_until_iso=(
            member.timed_out_until.isoformat()
            if member.timed_out_until is not None
            else None
        ),
        premium_since_iso=(
            member.premium_since.isoformat()
            if member.premium_since is not None
            else None
        ),
        avatar_url=str(member.display_avatar.url),
    )


def _pinned_message_record(message: discord.Message) -> DiscordPinnedMessageRecord:
    content = " ".join(message.content.split())
    pinned_at = getattr(message, "pinned_at", None)
    return DiscordPinnedMessageRecord(
        message_id=str(message.id),
        author_id=str(message.author.id),
        author_name=message.author.display_name,
        content_preview=content[:500],
        created_at_iso=message.created_at.isoformat(),
        edited_at_iso=message.edited_at.isoformat() if message.edited_at else None,
        pinned_at_iso=pinned_at.isoformat() if pinned_at is not None else None,
        attachment_count=len(message.attachments),
        embed_count=len(message.embeds),
        jump_url=message.jump_url,
    )


def _thread_member_record(
    item: discord.ThreadMember,
    guild: discord.Guild,
) -> DiscordThreadMemberRecord:
    member = guild.get_member(item.id)
    return DiscordThreadMemberRecord(
        user_id=str(item.id),
        display_name=member.display_name if member is not None else None,
        joined_at_iso=item.joined_at.isoformat(),
        flags=item.flags,
    )


def _resource(
    resource_id: object,
    name: object,
    kind: str,
    **values: object,
) -> DiscordPlatformResourceRecord:
    fields = tuple(
        DiscordResourceField(key=key, value=rendered)
        for key, value in values.items()
        if (rendered := _field_value(value)) is not None
    )
    return DiscordPlatformResourceRecord(
        resource_id=str(resource_id),
        kind=kind,
        name=str(name)[:200],
        fields=fields,
    )


def _field_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list, set, frozenset)):
        return ", ".join(str(item) for item in value)[:1_000]
    return str(value)[:1_000]


def _validate_page(offset: int, limit: int, label: str) -> None:
    if not 0 <= offset <= _MAX_PLATFORM_OFFSET:
        raise UserError(f"discord.{label}_offset_invalid")
    if not 1 <= limit <= _MAX_PLATFORM_PAGE:
        raise UserError(f"discord.{label}_limit_invalid")


def _member_cache_complete(guild: discord.Guild) -> bool:
    return guild.chunked is True or (
        isinstance(guild.member_count, int)
        and not isinstance(guild.member_count, bool)
        and len(guild.members) >= guild.member_count
    )


def _presence_intent(client: discord.Client) -> bool:
    intents = getattr(client, "intents", None)
    return isinstance(intents, discord.Intents) and intents.presences


def _require_both_guild_permission(
    actor: discord.Member,
    bot: discord.Member,
    permission: str,
) -> None:
    _require_guild_permission(actor, permission)
    _require_guild_permission(bot, permission)


async def _require_application_owner(
    client: discord.Client,
    actor: discord.Member,
) -> None:
    _require_guild_permission(actor, "administrator")
    try:
        info = await client.application_info()
    except discord.DiscordException as exc:
        raise UserError("discord.application_fetch_failed") from exc
    allowed_ids = {
        member.id
        for member in (info.team.members if info.team is not None else ())
    }
    if info.owner is not None:
        allowed_ids.add(info.owner.id)
    if actor.id not in allowed_ids:
        raise UserError("discord.application_owner_required")


def _optional_id(value: object) -> str | None:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_datetime(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None
