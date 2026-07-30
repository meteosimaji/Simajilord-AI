"""Typed, permission-checked Discord platform mutations.

These endpoints deliberately group low-frequency resource operations by
create/update/delete semantics. Every branch validates a closed resource-kind
enum and re-checks both the requester and bot's live effective permissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

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

from .capabilities import (
    _audit_reason,
    _bounded_name,
    _fetch_message_for_write,
    _requested_guild,
    _require_channel_permissions,
    _require_guild_permission,
    _require_member_below,
    _require_role_above,
    _snowflake,
    _write_members,
    _write_message_channel,
)
from .permissions import permission_enabled as _permission_enabled

DiscordCreateResourceKind = Literal[
    "text_channel",
    "voice_channel",
    "stage_channel",
    "forum_channel",
    "category",
    "invite",
    "scheduled_event",
    "stage_instance",
    "webhook",
    "template",
]
DiscordUpdateResourceKind = Literal[
    "channel",
    "member",
    "role",
    "scheduled_event",
    "stage_instance",
    "webhook",
    "template",
    "server",
]
DiscordDeleteResourceKind = Literal[
    "channel",
    "role",
    "invite",
    "scheduled_event",
    "stage_instance",
    "webhook",
    "template",
]
DiscordMessageAction = Literal[
    "publish",
    "clear_all_reactions",
    "clear_emoji_reactions",
    "end_poll",
]
EditableGuildChannel = (
    discord.TextChannel
    | discord.VoiceChannel
    | discord.StageChannel
    | discord.ForumChannel
    | discord.CategoryChannel
)


@dataclass(frozen=True, slots=True)
class DiscordCreateGuildResourceRequest:
    kind: DiscordCreateResourceKind
    name: str = ""
    guild_id: str | None = None
    channel_id: str | None = None
    category_id: str | None = None
    topic: str = ""
    description: str = ""
    location: str = ""
    start_time_iso: str | None = None
    end_time_iso: str | None = None
    nsfw: bool = False
    slowmode_seconds: int = 0
    bitrate: int | None = None
    user_limit: int = 0
    max_age_seconds: int = 86_400
    max_uses: int = 0
    temporary: bool = False
    unique: bool = True
    send_start_notification: bool = False
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = field(
        default=(),
        metadata={
            "description": (
                "Exact active Discord message IDs supporting this requested mutation."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordUpdateGuildResourceRequest:
    kind: DiscordUpdateResourceKind
    resource_id: str = ""
    guild_id: str | None = None
    name: str | None = None
    description: str | None = None
    topic: str | None = None
    category_id: str | None = None
    position: int | None = None
    nsfw: bool | None = None
    slowmode_seconds: int | None = None
    bitrate: int | None = None
    user_limit: int | None = None
    nickname: str | None = None
    mute: bool | None = None
    deafen: bool | None = None
    move_voice: bool = False
    voice_channel_id: str | None = None
    destination_channel_id: str | None = None
    colour_value: int | None = None
    hoist: bool | None = None
    mentionable: bool | None = None
    permissions_value: int | None = None
    start_time_iso: str | None = None
    end_time_iso: str | None = None
    location: str | None = None
    status: Literal["scheduled", "active", "completed", "cancelled"] | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordDeleteGuildResourceRequest:
    kind: DiscordDeleteResourceKind
    resource_id: str
    guild_id: str | None = None
    channel_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordGuildResourceMutationResponse:
    kind: str
    resource_id: str
    name: str
    source_guild_id: str
    channel_id: str | None = None
    url: str | None = None
    changed: bool = True
    secrets_omitted: bool = True


@dataclass(frozen=True, slots=True)
class DiscordMessageActionRequest:
    action: DiscordMessageAction
    channel_id: str
    message_id: str
    guild_id: str | None = None
    emoji: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordMessageActionResponse:
    action: DiscordMessageAction
    message_id: str
    channel_id: str
    source_guild_id: str
    changed: bool


@dataclass(frozen=True, slots=True)
class DiscordSetChannelOverwriteRequest:
    channel_id: str
    target_kind: Literal["role", "member"]
    target_id: str
    allowed_permissions: tuple[str, ...] = ()
    denied_permissions: tuple[str, ...] = ()
    delete: bool = False
    guild_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordSetChannelOverwriteResponse:
    channel_id: str
    target_kind: Literal["role", "member"]
    target_id: str
    source_guild_id: str
    deleted: bool
    changed: bool


def build_discord_platform_action_endpoints(
    client: discord.Client,
) -> tuple[CapabilityEndpoint, ...]:
    """Build the low-frequency Discord resource mutation endpoints."""

    async def create_resource(
        request: DiscordCreateGuildResourceRequest,
        context: InvocationContext,
    ) -> DiscordGuildResourceMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        reason = _audit_reason(request.reason or f"Create {request.kind}", context)
        if request.kind.endswith("_channel") or request.kind == "category":
            _require_both_guild_permission(actor, bot, "manage_channels")
            name = _bounded_name(request.name, "discord.channel_name_invalid")
            category = _optional_category(guild, request.category_id)
            if not 0 <= request.slowmode_seconds <= 21_600:
                raise UserError("discord.channel_slowmode_invalid")
            if request.kind == "text_channel":
                text_options: dict[str, object] = {
                    "category": category,
                    "nsfw": request.nsfw,
                    "slowmode_delay": request.slowmode_seconds,
                    "reason": reason,
                }
                if request.topic:
                    text_options["topic"] = request.topic
                text_channel = await cast(
                    Any,
                    guild.create_text_channel,
                )(name, **text_options)
                return _mutation_response(guild, request.kind, text_channel)
            elif request.kind == "voice_channel":
                voice_options: dict[str, object] = {
                    "category": category,
                    "user_limit": request.user_limit,
                    "reason": reason,
                }
                if request.bitrate is not None:
                    voice_options["bitrate"] = request.bitrate
                voice_channel = await cast(
                    Any,
                    guild.create_voice_channel,
                )(name, **voice_options)
                return _mutation_response(guild, request.kind, voice_channel)
            elif request.kind == "stage_channel":
                stage_options: dict[str, object] = {
                    "category": category,
                    "user_limit": request.user_limit,
                    "reason": reason,
                }
                if request.bitrate is not None:
                    stage_options["bitrate"] = request.bitrate
                stage_channel = await cast(
                    Any,
                    guild.create_stage_channel,
                )(name, **stage_options)
                return _mutation_response(guild, request.kind, stage_channel)
            elif request.kind == "forum_channel":
                forum_options: dict[str, object] = {
                    "category": category,
                    "nsfw": request.nsfw,
                    "slowmode_delay": request.slowmode_seconds,
                    "reason": reason,
                }
                if request.topic:
                    forum_options["topic"] = request.topic
                forum_channel = await cast(
                    Any,
                    guild.create_forum,
                )(name, **forum_options)
                return _mutation_response(guild, request.kind, forum_channel)
            category_channel = await guild.create_category(name, reason=reason)
            return _mutation_response(guild, request.kind, category_channel)
        if request.kind == "invite":
            if request.channel_id is None:
                raise UserError("discord.channel_id_invalid")
            invite_channel = _guild_channel(guild, request.channel_id)
            _require_both_channel_permission(
                invite_channel,
                actor,
                bot,
                "create_instant_invite",
            )
            if not 0 <= request.max_age_seconds <= 604_800:
                raise UserError("discord.invite_max_age_invalid")
            if not 0 <= request.max_uses <= 100:
                raise UserError("discord.invite_max_uses_invalid")
            invite = await invite_channel.create_invite(
                max_age=request.max_age_seconds,
                max_uses=request.max_uses,
                temporary=request.temporary,
                unique=request.unique,
                reason=reason,
            )
            return DiscordGuildResourceMutationResponse(
                kind=request.kind,
                resource_id=invite.code,
                name=invite.code,
                source_guild_id=str(guild.id),
                channel_id=str(invite_channel.id),
                url=invite.url,
            )
        if request.kind == "scheduled_event":
            _require_both_guild_permission(actor, bot, "create_events")
            name = _bounded_name(request.name, "discord.event_name_invalid")
            start = _required_time(request.start_time_iso, "discord.event_start_invalid")
            end = _optional_time(request.end_time_iso, "discord.event_end_invalid")
            location = request.location.strip()
            if location:
                if end is None:
                    raise UserError("discord.event_end_required")
                event_options: dict[str, object] = {
                    "name": name,
                    "start_time": start,
                    "end_time": end,
                    "entity_type": discord.EntityType.external,
                    "location": location,
                    "reason": reason,
                }
            else:
                if request.channel_id is None:
                    raise UserError("discord.event_channel_required")
                selected = _guild_channel(guild, request.channel_id)
                if not isinstance(selected, (discord.VoiceChannel, discord.StageChannel)):
                    raise UserError("discord.event_channel_invalid")
                for member in (actor, bot):
                    _require_channel_permissions(selected, member, "view_channel")
                    _require_channel_permissions(selected, member, "connect")
                entity_type = (
                    discord.EntityType.stage_instance
                    if isinstance(selected, discord.StageChannel)
                    else discord.EntityType.voice
                )
                event_options = {
                    "name": name,
                    "start_time": start,
                    "entity_type": entity_type,
                    "channel": selected,
                    "reason": reason,
                }
                if end is not None:
                    event_options["end_time"] = end
            if request.description:
                event_options["description"] = request.description
            event = await cast(
                Any,
                guild.create_scheduled_event,
            )(**event_options)
            return _mutation_response(guild, request.kind, event)
        if request.kind == "stage_instance":
            if request.channel_id is None:
                raise UserError("discord.stage_channel_required")
            channel = _guild_channel(guild, request.channel_id)
            if not isinstance(channel, discord.StageChannel):
                raise UserError("discord.stage_channel_required")
            for member in (actor, bot):
                for permission in ("manage_channels", "mute_members", "move_members"):
                    _require_channel_permissions(channel, member, permission)
            topic = request.topic.strip()
            if not 1 <= len(topic) <= 120:
                raise UserError("discord.stage_topic_invalid")
            instance = await channel.create_instance(
                topic=topic,
                send_start_notification=request.send_start_notification,
                reason=reason,
            )
            return _mutation_response(
                guild,
                request.kind,
                instance,
                channel_id=str(channel.id),
            )
        if request.kind == "webhook":
            if request.channel_id is None:
                raise UserError("discord.channel_id_invalid")
            channel = _guild_channel(guild, request.channel_id)
            if not isinstance(
                channel,
                (discord.TextChannel, discord.ForumChannel),
            ):
                raise UserError("discord.webhook_channel_invalid")
            _require_both_channel_permission(channel, actor, bot, "manage_webhooks")
            webhook = await channel.create_webhook(
                name=_bounded_name(request.name, "discord.webhook_name_invalid"),
                reason=reason,
            )
            return _mutation_response(
                guild,
                request.kind,
                webhook,
                channel_id=str(channel.id),
            )
        if request.kind == "template":
            _require_both_guild_permission(actor, bot, "manage_guild")
            template = await guild.create_template(
                name=_bounded_name(request.name, "discord.template_name_invalid"),
                description=request.description,
            )
            return DiscordGuildResourceMutationResponse(
                kind=request.kind,
                resource_id=template.code,
                name=template.name,
                source_guild_id=str(guild.id),
                url=template.url,
            )
        raise UserError("discord.resource_kind_invalid")

    async def update_resource(
        request: DiscordUpdateGuildResourceRequest,
        context: InvocationContext,
    ) -> DiscordGuildResourceMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        reason = _audit_reason(request.reason or f"Update {request.kind}", context)
        if request.kind == "channel":
            channel = _guild_channel(guild, request.resource_id)
            _require_both_channel_permission(channel, actor, bot, "manage_channels")
            kwargs: dict[str, object] = {}
            if request.name is not None:
                kwargs["name"] = _bounded_name(
                    request.name,
                    "discord.channel_name_invalid",
                )
            if request.position is not None:
                if request.position < 0:
                    raise UserError("discord.channel_position_invalid")
                kwargs["position"] = request.position
            if request.category_id is not None:
                kwargs["category"] = _optional_category(guild, request.category_id)
            if request.nsfw is not None:
                kwargs["nsfw"] = request.nsfw
            if request.slowmode_seconds is not None:
                if not 0 <= request.slowmode_seconds <= 21_600:
                    raise UserError("discord.channel_slowmode_invalid")
                kwargs["slowmode_delay"] = request.slowmode_seconds
            if request.topic is not None and isinstance(
                channel,
                (discord.TextChannel, discord.ForumChannel),
            ):
                kwargs["topic"] = request.topic or None
            if request.bitrate is not None and isinstance(
                channel,
                (discord.VoiceChannel, discord.StageChannel),
            ):
                kwargs["bitrate"] = request.bitrate
            if request.user_limit is not None and isinstance(
                channel,
                (discord.VoiceChannel, discord.StageChannel),
            ):
                kwargs["user_limit"] = request.user_limit
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated = await cast(Any, channel.edit)(reason=reason, **kwargs)
            return _mutation_response(guild, request.kind, updated or channel)
        if request.kind == "member":
            target = await _target_member(guild, request.resource_id)
            for member in (actor, bot):
                _require_member_below(member, target, guild)
            kwargs = {}
            if request.nickname is not None:
                for member in (actor, bot):
                    _require_guild_permission(member, "manage_nicknames")
                if len(request.nickname) > 32:
                    raise UserError("discord.nickname_invalid")
                kwargs["nick"] = request.nickname or None
            if request.mute is not None:
                for member in (actor, bot):
                    _require_guild_permission(member, "mute_members")
                kwargs["mute"] = request.mute
            if request.deafen is not None:
                for member in (actor, bot):
                    _require_guild_permission(member, "deafen_members")
                kwargs["deafen"] = request.deafen
            if request.move_voice:
                for member in (actor, bot):
                    _require_guild_permission(member, "move_members")
                destination = (
                    None
                    if request.voice_channel_id is None
                    else _vocal_channel(guild, request.voice_channel_id)
                )
                kwargs["voice_channel"] = destination
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated_member = await cast(Any, target.edit)(reason=reason, **kwargs)
            return _mutation_response(guild, request.kind, updated_member or target)
        if request.kind == "role":
            role = _guild_role(guild, request.resource_id)
            for member in (actor, bot):
                _require_guild_permission(member, "manage_roles")
                _require_role_above(member, role)
            kwargs = {}
            if request.name is not None:
                kwargs["name"] = _bounded_name(
                    request.name,
                    "discord.role_name_invalid",
                )
            if request.colour_value is not None:
                if not 0 <= request.colour_value <= 0xFFFFFF:
                    raise UserError("discord.role_colour_invalid")
                kwargs["colour"] = request.colour_value
            if request.hoist is not None:
                kwargs["hoist"] = request.hoist
            if request.mentionable is not None:
                kwargs["mentionable"] = request.mentionable
            if request.position is not None:
                kwargs["position"] = request.position
            if request.permissions_value is not None:
                if request.permissions_value < 0:
                    raise UserError("discord.role_permissions_invalid")
                permissions = discord.Permissions(request.permissions_value)
                for member in (actor, bot):
                    if (
                        not _permission_enabled(
                            member.guild_permissions,
                            "administrator",
                        )
                        and permissions.value & ~member.guild_permissions.value
                    ):
                        raise UserError("discord.role_permissions_forbidden")
                kwargs["permissions"] = permissions
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated_role = await cast(Any, role.edit)(reason=reason, **kwargs)
            return _mutation_response(guild, request.kind, updated_role or role)
        if request.kind == "scheduled_event":
            _require_both_guild_permission(actor, bot, "manage_events")
            event = await _scheduled_event(guild, request.resource_id)
            kwargs = {}
            if request.name is not None:
                kwargs["name"] = _bounded_name(
                    request.name,
                    "discord.event_name_invalid",
                )
            if request.description is not None:
                kwargs["description"] = request.description
            if request.start_time_iso is not None:
                kwargs["start_time"] = _required_time(
                    request.start_time_iso,
                    "discord.event_start_invalid",
                )
            if request.end_time_iso is not None:
                kwargs["end_time"] = _required_time(
                    request.end_time_iso,
                    "discord.event_end_invalid",
                )
            if request.location is not None:
                kwargs["location"] = request.location
            if request.status is not None:
                kwargs["status"] = _event_status(request.status)
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated_event = await cast(Any, event.edit)(reason=reason, **kwargs)
            return _mutation_response(guild, request.kind, updated_event)
        if request.kind == "stage_instance":
            channel = _stage_channel(guild, request.resource_id)
            _require_both_channel_permission(channel, actor, bot, "manage_channels")
            instance = await _stage_instance(channel)
            kwargs = {}
            if request.topic is not None:
                topic = request.topic.strip()
                if not 1 <= len(topic) <= 120:
                    raise UserError("discord.stage_topic_invalid")
                kwargs["topic"] = topic
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            await cast(Any, instance.edit)(reason=reason, **kwargs)
            return _mutation_response(
                guild,
                request.kind,
                instance,
                channel_id=str(channel.id),
            )
        if request.kind == "webhook":
            _require_both_guild_permission(actor, bot, "manage_webhooks")
            webhook = await _webhook(guild, request.resource_id)
            kwargs = {}
            if request.name is not None:
                kwargs["name"] = _bounded_name(
                    request.name,
                    "discord.webhook_name_invalid",
                )
            if request.destination_channel_id is not None:
                webhook_destination = _guild_channel(
                    guild,
                    request.destination_channel_id,
                )
                if not isinstance(
                    webhook_destination,
                    (discord.TextChannel, discord.ForumChannel),
                ):
                    raise UserError("discord.webhook_channel_invalid")
                kwargs["channel"] = webhook_destination
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated_webhook = await cast(Any, webhook.edit)(
                reason=reason,
                prefer_auth=True,
                **kwargs,
            )
            return _mutation_response(
                guild,
                request.kind,
                updated_webhook,
                channel_id=(
                    str(updated_webhook.channel_id)
                    if updated_webhook.channel_id is not None
                    else None
                ),
            )
        if request.kind == "template":
            _require_both_guild_permission(actor, bot, "manage_guild")
            template = await _template(guild, request.resource_id)
            if request.name is None and request.description is None:
                updated_template = await template.sync()
            else:
                kwargs = {}
                if request.name is not None:
                    kwargs["name"] = _bounded_name(
                        request.name,
                        "discord.template_name_invalid",
                    )
                if request.description is not None:
                    kwargs["description"] = request.description or None
                updated_template = await cast(Any, template.edit)(**kwargs)
            return DiscordGuildResourceMutationResponse(
                kind=request.kind,
                resource_id=updated_template.code,
                name=updated_template.name,
                source_guild_id=str(guild.id),
                url=updated_template.url,
            )
        if request.kind == "server":
            _require_both_guild_permission(actor, bot, "manage_guild")
            kwargs = {}
            if request.name is not None:
                kwargs["name"] = _bounded_name(
                    request.name,
                    "discord.guild_name_invalid",
                )
            if request.description is not None:
                kwargs["description"] = request.description or None
            if not kwargs:
                raise UserError("discord.resource_update_empty")
            updated_guild = await cast(Any, guild.edit)(reason=reason, **kwargs)
            return _mutation_response(guild, request.kind, updated_guild)
        raise UserError("discord.resource_kind_invalid")

    async def delete_resource(
        request: DiscordDeleteGuildResourceRequest,
        context: InvocationContext,
    ) -> DiscordGuildResourceMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        reason = _audit_reason(request.reason or f"Delete {request.kind}", context)
        if request.kind == "channel":
            channel = _guild_channel(guild, request.resource_id)
            _require_both_channel_permission(channel, actor, bot, "manage_channels")
            response = _mutation_response(guild, request.kind, channel)
            await channel.delete(reason=reason)
            return response
        if request.kind == "role":
            role = _guild_role(guild, request.resource_id)
            for member in (actor, bot):
                _require_guild_permission(member, "manage_roles")
                _require_role_above(member, role)
            response = _mutation_response(guild, request.kind, role)
            await role.delete(reason=reason)
            return response
        if request.kind == "invite":
            _require_both_guild_permission(actor, bot, "manage_guild")
            try:
                invite = await client.fetch_invite(request.resource_id)
            except discord.DiscordException as exc:
                raise UserError("discord.invite_not_found") from exc
            if invite.guild is None or invite.guild.id != guild.id:
                raise UserError("discord.invite_not_found")
            await client.delete_invite(invite, reason=reason)
            return DiscordGuildResourceMutationResponse(
                kind=request.kind,
                resource_id=invite.code,
                name=invite.code,
                source_guild_id=str(guild.id),
                channel_id=(
                    str(invite.channel.id)
                    if invite.channel is not None
                    else None
                ),
                url=invite.url,
            )
        if request.kind == "scheduled_event":
            _require_both_guild_permission(actor, bot, "manage_events")
            event = await _scheduled_event(guild, request.resource_id)
            response = _mutation_response(guild, request.kind, event)
            await event.delete(reason=reason)
            return response
        if request.kind == "stage_instance":
            channel_id = request.channel_id or request.resource_id
            channel = _stage_channel(guild, channel_id)
            _require_both_channel_permission(channel, actor, bot, "manage_channels")
            instance = await _stage_instance(channel)
            response = _mutation_response(
                guild,
                request.kind,
                instance,
                channel_id=str(channel.id),
            )
            await instance.delete(reason=reason)
            return response
        if request.kind == "webhook":
            _require_both_guild_permission(actor, bot, "manage_webhooks")
            webhook = await _webhook(guild, request.resource_id)
            response = _mutation_response(
                guild,
                request.kind,
                webhook,
                channel_id=(
                    str(webhook.channel_id)
                    if webhook.channel_id is not None
                    else None
                ),
            )
            await webhook.delete(reason=reason, prefer_auth=True)
            return response
        if request.kind == "template":
            _require_both_guild_permission(actor, bot, "manage_guild")
            template = await _template(guild, request.resource_id)
            response = DiscordGuildResourceMutationResponse(
                kind=request.kind,
                resource_id=template.code,
                name=template.name,
                source_guild_id=str(guild.id),
                url=template.url,
            )
            await template.delete()
            return response
        raise UserError("discord.resource_kind_invalid")

    async def message_action(
        request: DiscordMessageActionRequest,
        context: InvocationContext,
    ) -> DiscordMessageActionResponse:
        guild, channel, actor, bot = await _write_message_channel(
            client,
            context,
            request.channel_id,
            guild_id=request.guild_id,
        )
        message = await _fetch_message_for_write(channel, request.message_id)
        if request.action == "publish":
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "send_messages")
                _require_channel_permissions(channel, member, "manage_messages")
            await message.publish()
        elif request.action == "clear_all_reactions":
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "manage_messages")
            await message.clear_reactions()
        elif request.action == "clear_emoji_reactions":
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "manage_messages")
            if request.emoji is None:
                raise UserError("discord.reaction_emoji_invalid")
            reaction = next(
                (
                    candidate
                    for candidate in message.reactions
                    if str(candidate.emoji) == request.emoji
                ),
                None,
            )
            if reaction is None:
                raise UserError("discord.reaction_not_found")
            await reaction.clear()
        elif request.action == "end_poll":
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "manage_messages")
            if message.poll is None:
                raise UserError("discord.poll_not_found")
            if message.author.id != bot.id:
                raise UserError("discord.poll_not_owned")
            await message.end_poll()
        else:
            raise UserError("discord.message_action_invalid")
        return DiscordMessageActionResponse(
            action=request.action,
            message_id=str(message.id),
            channel_id=str(channel.id),
            source_guild_id=str(guild.id),
            changed=True,
        )

    async def set_channel_overwrite(
        request: DiscordSetChannelOverwriteRequest,
        context: InvocationContext,
    ) -> DiscordSetChannelOverwriteResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        channel = _guild_channel(guild, request.channel_id)
        _require_both_channel_permission(channel, actor, bot, "manage_roles")
        if request.target_kind == "role":
            role_target = _guild_role(
                guild,
                request.target_id,
            )
            for member in (actor, bot):
                _require_role_above(member, role_target)
            target: discord.Role | discord.Member = role_target
        else:
            target = await _target_member(guild, request.target_id)
        overlap = set(request.allowed_permissions) & set(request.denied_permissions)
        if overlap:
            raise UserError("discord.permission_overwrite_conflict")
        valid_permissions = set(discord.Permissions.VALID_FLAGS)
        requested_permissions = set(request.allowed_permissions) | set(
            request.denied_permissions
        )
        if not requested_permissions <= valid_permissions:
            raise UserError("discord.permission_name_invalid")
        if not request.delete:
            for member in (actor, bot):
                if _permission_enabled(member.guild_permissions, "administrator"):
                    continue
                missing = {
                    permission
                    for permission in request.allowed_permissions
                    if not _permission_enabled(member.guild_permissions, permission)
                }
                if missing:
                    raise UserError("discord.permission_overwrite_forbidden")
        overwrite = (
            None
            if request.delete
            else discord.PermissionOverwrite(
                **{
                    **{name: True for name in request.allowed_permissions},
                    **{name: False for name in request.denied_permissions},
                }
            )
        )
        await channel.set_permissions(
            target,
            overwrite=overwrite,
            reason=_audit_reason(
                request.reason or "Update channel permission overwrite",
                context,
            ),
        )
        return DiscordSetChannelOverwriteResponse(
            channel_id=str(channel.id),
            target_kind=request.target_kind,
            target_id=str(target.id),
            source_guild_id=str(guild.id),
            deleted=request.delete,
            changed=True,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.create_guild_resource",
                summary=(
                    "Create a Discord text/voice/stage/forum/category channel, invite, "
                    "scheduled event, stage instance, token-hidden webhook, or guild "
                    "template after checking both requester and bot permissions."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "create voice channel",
                    "create a new voice channel",
                    "create stage channel",
                    "create forum channel",
                    "create category",
                    "create invite",
                    "create scheduled event",
                    "start stage instance",
                    "create webhook",
                    "create guild template",
                    "ボイスチャンネルを作る",
                    "ステージチャンネルを作る",
                    "フォーラムを作る",
                    "フォーラムチャンネルを作って",
                    "カテゴリを作る",
                    "招待を作る",
                    "予定イベントを作る",
                    "ステージを開始",
                    "ウェブフックを作る",
                    "サーバーテンプレートを作る",
                ),
                side_effects=("Creates the selected Discord server resource.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=30,
            ),
            DiscordCreateGuildResourceRequest,
            DiscordGuildResourceMutationResponse,
            create_resource,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.update_guild_resource",
                summary=(
                    "Update a Discord channel, member, role, scheduled event, stage "
                    "instance, webhook, template, or server with live hierarchy and "
                    "effective-permission checks."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "update guild resource",
                    "edit channel",
                    "move member",
                    "edit role",
                    "update scheduled event",
                    "update this scheduled event",
                    "update stage instance",
                    "edit webhook",
                    "edit guild template",
                    "edit server",
                    "チャンネルを更新",
                    "メンバーを移動",
                    "サーバーミュート",
                    "ロールを更新",
                    "予定イベントを更新",
                    "ステージを更新",
                    "ウェブフックを更新",
                    "テンプレートを更新",
                    "サーバー設定を更新",
                ),
                side_effects=("Updates the selected Discord server resource.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=30,
            ),
            DiscordUpdateGuildResourceRequest,
            DiscordGuildResourceMutationResponse,
            update_resource,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_guild_resource",
                summary=(
                    "Delete a Discord channel, role, invite, scheduled event, stage "
                    "instance, webhook, or template after destructive approval and "
                    "live permission checks."
                ),
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "delete guild resource",
                    "delete channel",
                    "delete role",
                    "delete invite",
                    "delete scheduled event",
                    "delete stage instance",
                    "delete webhook",
                    "delete this webhook permanently",
                    "delete guild template",
                    "チャンネルを削除",
                    "ロールを削除",
                    "招待を削除",
                    "予定イベントを削除",
                    "ステージを終了",
                    "ウェブフックを削除",
                    "ウェブフックを完全に削除して",
                    "テンプレートを削除",
                ),
                side_effects=(
                    "Permanently deletes the selected Discord server resource.",
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=30,
            ),
            DiscordDeleteGuildResourceRequest,
            DiscordGuildResourceMutationResponse,
            delete_resource,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.message_action",
                summary=(
                    "Publish an announcement message, clear all or one emoji's reactions, "
                    "or end a bot-owned native poll."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "message",
                    "publish",
                    "crosspost",
                    "reactions",
                    "poll",
                    "公開",
                    "リアクション削除",
                    "投票終了",
                    "アナウンスメッセージを公開して",
                ),
                side_effects=("Changes one existing Discord message.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=20,
            ),
            DiscordMessageActionRequest,
            DiscordMessageActionResponse,
            message_action,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.set_channel_overwrite",
                summary=(
                    "Create, replace, or delete one role/member channel permission "
                    "overwrite, never granting a permission unavailable to the requester "
                    "or bot unless they are administrators."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "channel",
                    "permissions",
                    "overwrite",
                    "role",
                    "member",
                    "権限",
                    "上書き",
                    "ロール権限上書きを設定して",
                ),
                side_effects=("Changes one channel permission overwrite.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=20,
            ),
            DiscordSetChannelOverwriteRequest,
            DiscordSetChannelOverwriteResponse,
            set_channel_overwrite,
        ),
    )


def _require_both_guild_permission(
    actor: discord.Member,
    bot: discord.Member,
    permission: str,
) -> None:
    _require_guild_permission(actor, permission)
    _require_guild_permission(bot, permission)


def _require_both_channel_permission(
    channel: EditableGuildChannel | discord.Thread,
    actor: discord.Member,
    bot: discord.Member,
    permission: str,
) -> None:
    _require_channel_permissions(channel, actor, permission)
    _require_channel_permissions(channel, bot, permission)


def _guild_channel(
    guild: discord.Guild,
    channel_id: str,
) -> EditableGuildChannel:
    channel = guild.get_channel(_snowflake(channel_id, "channel"))
    if not isinstance(
        channel,
        (
            discord.TextChannel,
            discord.VoiceChannel,
            discord.StageChannel,
            discord.ForumChannel,
            discord.CategoryChannel,
        ),
    ):
        raise UserError("discord.channel_unavailable")
    return channel


def _optional_category(
    guild: discord.Guild,
    category_id: str | None,
) -> discord.CategoryChannel | None:
    if category_id is None:
        return None
    category = guild.get_channel(_snowflake(category_id, "channel"))
    if not isinstance(category, discord.CategoryChannel):
        raise UserError("discord.category_invalid")
    return category


def _vocal_channel(
    guild: discord.Guild,
    channel_id: str,
) -> discord.VoiceChannel | discord.StageChannel:
    channel = guild.get_channel(_snowflake(channel_id, "voice channel"))
    if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
        raise UserError("discord.voice_channel_id_invalid")
    return channel


def _stage_channel(guild: discord.Guild, channel_id: str) -> discord.StageChannel:
    channel = guild.get_channel(_snowflake(channel_id, "channel"))
    if not isinstance(channel, discord.StageChannel):
        raise UserError("discord.stage_channel_required")
    return channel


async def _stage_instance(channel: discord.StageChannel) -> discord.StageInstance:
    try:
        return await channel.fetch_instance()
    except discord.NotFound as exc:
        raise UserError("discord.stage_instance_not_found") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.stage_instance_fetch_failed") from exc


async def _target_member(guild: discord.Guild, user_id: str) -> discord.Member:
    member = guild.get_member(_snowflake(user_id, "user"))
    if member is not None:
        return member
    try:
        return await guild.fetch_member(_snowflake(user_id, "user"))
    except discord.DiscordException as exc:
        raise UserError("discord.member_required") from exc


def _guild_role(guild: discord.Guild, role_id: str) -> discord.Role:
    role = guild.get_role(_snowflake(role_id, "role"))
    if role is None or role.is_default() or role.managed:
        raise UserError("discord.role_unavailable")
    return role


async def _scheduled_event(
    guild: discord.Guild,
    event_id: str,
) -> discord.ScheduledEvent:
    try:
        return await guild.fetch_scheduled_event(_snowflake(event_id, "event"))
    except discord.DiscordException as exc:
        raise UserError("discord.event_not_found") from exc


async def _webhook(guild: discord.Guild, webhook_id: str) -> discord.Webhook:
    selected_id = _snowflake(webhook_id, "webhook")
    try:
        webhooks = await guild.webhooks()
    except discord.DiscordException as exc:
        raise UserError("discord.webhooks_fetch_failed") from exc
    webhook = next((item for item in webhooks if item.id == selected_id), None)
    if webhook is None:
        raise UserError("discord.webhook_not_found")
    return webhook


async def _template(guild: discord.Guild, code: str) -> discord.Template:
    try:
        templates = await guild.templates()
    except discord.DiscordException as exc:
        raise UserError("discord.templates_fetch_failed") from exc
    template = next((item for item in templates if item.code == code), None)
    if template is None:
        raise UserError("discord.template_not_found")
    return template


def _required_time(value: str | None, code: str) -> datetime:
    parsed = _optional_time(value, code)
    if parsed is None:
        raise UserError(code)
    return parsed


def _optional_time(value: str | None, code: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError(code) from exc
    if parsed.tzinfo is None:
        raise UserError(code)
    return parsed.astimezone(UTC)


def _event_status(value: str) -> discord.EventStatus:
    values = {
        "scheduled": discord.EventStatus.scheduled,
        "active": discord.EventStatus.active,
        "completed": discord.EventStatus.completed,
        "cancelled": discord.EventStatus.cancelled,
    }
    return values[value]


def _mutation_response(
    guild: discord.Guild,
    kind: str,
    resource: object,
    *,
    channel_id: str | None = None,
) -> DiscordGuildResourceMutationResponse:
    resource_id = getattr(resource, "id", None)
    if resource_id is None:
        raise UserError("discord.resource_response_invalid")
    return DiscordGuildResourceMutationResponse(
        kind=kind,
        resource_id=str(resource_id),
        name=str(
            getattr(resource, "name", None)
            or getattr(resource, "topic", None)
            or resource_id
        ),
        source_guild_id=str(guild.id),
        channel_id=channel_id,
    )
