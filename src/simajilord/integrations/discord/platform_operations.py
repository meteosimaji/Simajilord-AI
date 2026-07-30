"""Low-frequency Discord operations with explicit authorization boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import discord
from discord.http import Route

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
    DiscordDeliveryPurpose,
    _audit_reason,
    _bounded_name,
    _can_view_channel,
    _fetch_readable_message,
    _requested_guild,
    _require_channel_permissions,
    _require_guild_permission,
    _snowflake,
    _write_members,
    _write_message_channel,
)
from .permissions import can_read_private_thread as _can_read_private_thread
from .permissions import permission_enabled as _permission_enabled
from .platform_assets import _require_global_application_permission

DiscordChannelOperation = Literal[
    "clone",
    "follow",
    "join_thread",
    "leave_thread",
    "add_thread_tags",
    "remove_thread_tags",
    "create_forum_tag",
    "send_soundboard",
    "set_voice_status",
]
DiscordPresenceStatus = Literal["online", "idle", "dnd", "invisible"]
DiscordPresenceActivity = Literal[
    "none",
    "playing",
    "listening",
    "watching",
    "competing",
    "streaming",
    "custom",
]


@dataclass(frozen=True, slots=True)
class DiscordChannelOperationRequest:
    operation: DiscordChannelOperation
    channel_id: str
    guild_id: str | None = None
    destination_channel_id: str | None = None
    name: str = ""
    tag_ids: tuple[str, ...] = ()
    emoji: str | None = None
    moderated: bool = False
    sound_id: str | None = None
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
class DiscordChannelOperationResponse:
    operation: DiscordChannelOperation
    source_guild_id: str
    channel_id: str
    resource_id: str | None
    name: str | None
    changed: bool
    secrets_omitted: bool = True


@dataclass(frozen=True, slots=True)
class DiscordForwardMessageRequest:
    source_channel_id: str
    source_message_id: str
    destination_channel_id: str
    source_guild_id: str | None = None
    destination_guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordForwardMessageResponse:
    source_message_id: str
    destination_message_id: str
    destination_channel_id: str
    destination_guild_id: str


@dataclass(frozen=True, slots=True)
class DiscordSendDirectMessageRequest:
    user_id: str
    content: str
    guild_id: str | None = None
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordSendDirectMessageResponse:
    user_id: str
    message_id: str
    channel_id: str
    link_previews_suppressed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordSetBotPresenceRequest:
    status: DiscordPresenceStatus = "online"
    activity_kind: DiscordPresenceActivity = "none"
    activity_name: str = ""
    streaming_url: str | None = None
    guild_id: str | None = None
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordSetBotPresenceResponse:
    status: DiscordPresenceStatus
    activity_kind: DiscordPresenceActivity
    activity_name: str | None
    changed: bool


def build_discord_platform_operation_endpoints(
    client: discord.Client,
) -> tuple[CapabilityEndpoint, ...]:
    """Build bounded platform operations not covered by the main message adapter."""

    async def channel_operation(
        request: DiscordChannelOperationRequest,
        context: InvocationContext,
    ) -> DiscordChannelOperationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        reason = _audit_reason(
            request.reason or f"Discord channel operation: {request.operation}",
            context,
        )
        channel = guild.get_channel_or_thread(
            _snowflake(request.channel_id, "channel"),
        )
        if channel is None:
            raise UserError("discord.channel_unavailable")

        if request.operation == "clone":
            if not isinstance(channel, discord.abc.GuildChannel):
                raise UserError("discord.channel_clone_invalid")
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "manage_channels")
            clone_name = (
                _bounded_name(request.name, "discord.channel_name_invalid")
                if request.name
                else channel.name
            )
            cloned = await cast(Any, channel.clone)(name=clone_name, reason=reason)
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=cloned.id,
                resource_id=cloned.id,
                name=cloned.name,
            )

        if request.operation == "follow":
            if not isinstance(channel, discord.TextChannel) or not channel.is_news():
                raise UserError("discord.follow_source_invalid")
            if request.destination_channel_id is None:
                raise UserError("discord.follow_destination_required")
            destination = guild.get_channel(
                _snowflake(request.destination_channel_id, "channel"),
            )
            if not isinstance(destination, discord.TextChannel):
                raise UserError("discord.follow_destination_invalid")
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "read_message_history")
                _require_channel_permissions(destination, member, "manage_webhooks")
            webhook = await channel.follow(destination=destination, reason=reason)
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=destination.id,
                resource_id=webhook.id,
                name=webhook.name,
            )

        if request.operation == "create_forum_tag":
            if not isinstance(channel, discord.ForumChannel):
                raise UserError("discord.forum_channel_required")
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "manage_channels")
            tag = await channel.create_tag(
                name=_bounded_name(request.name, "discord.forum_tag_name_invalid"),
                emoji=(
                    discord.PartialEmoji.from_str(request.emoji)
                    if request.emoji
                    else None
                ),
                moderated=request.moderated,
                reason=reason,
            )
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=channel.id,
                resource_id=tag.id,
                name=tag.name,
            )

        if request.operation == "send_soundboard":
            if not isinstance(channel, discord.VoiceChannel):
                raise UserError("discord.voice_channel_invalid")
            if request.sound_id is None:
                raise UserError("discord.soundboard_sound_id_required")
            if actor.voice is None or actor.voice.channel != channel:
                raise UserError("discord.same_voice_required")
            for member in (actor, bot):
                _require_channel_permissions(channel, member, "use_soundboard")
            sound_id = _snowflake(request.sound_id, "soundboard sound")
            sound = guild.get_soundboard_sound(sound_id)
            if sound is None:
                try:
                    sound = await guild.fetch_soundboard_sound(sound_id)
                except discord.DiscordException as exc:
                    raise UserError("discord.soundboard_sound_not_found") from exc
            await channel.send_sound(sound)
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=channel.id,
                resource_id=sound.id,
                name=sound.name,
            )

        if request.operation == "set_voice_status":
            if not isinstance(
                channel,
                (discord.VoiceChannel, discord.StageChannel),
            ):
                raise UserError("discord.voice_channel_invalid")
            status = " ".join(request.name.split())
            if len(status) > 500:
                raise UserError("discord.voice_status_invalid")
            for member in (actor, bot):
                _require_channel_permissions(
                    channel,
                    member,
                    "set_voice_channel_status",
                )
            bot_voice = bot.voice
            if bot_voice is None or bot_voice.channel != channel:
                for member in (actor, bot):
                    _require_channel_permissions(
                        channel,
                        member,
                        "manage_channels",
                    )
            try:
                await client.http.request(
                    Route(
                        "PUT",
                        "/channels/{channel_id}/voice-status",
                        channel_id=channel.id,
                    ),
                    json={"status": status or None},
                    reason=reason,
                )
            except discord.Forbidden as exc:
                raise UserError("discord.voice_status_forbidden") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.voice_status_update_failed") from exc
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=channel.id,
                name=status or None,
            )

        if not isinstance(channel, discord.Thread):
            raise UserError("discord.thread_unavailable")
        if not _can_view_channel(channel, actor) or not _can_view_channel(channel, bot):
            raise UserError("discord.agent_write_channel_forbidden")
        if request.operation != "join_thread" and not _can_read_private_thread(
            channel,
            bot,
        ):
            raise UserError("discord.agent_write_channel_forbidden")
        if channel.type is discord.ChannelType.private_thread and not (
            _can_read_private_thread(channel, actor)
            or _permission_enabled(actor.guild_permissions, "manage_threads")
        ):
            raise UserError("discord.agent_write_channel_forbidden")

        if request.operation == "join_thread":
            _require_channel_permissions(channel, actor, "send_messages")
            await channel.join()
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=channel.id,
            )
        if request.operation == "leave_thread":
            await channel.leave()
            return _channel_operation_response(
                guild,
                request.operation,
                channel_id=channel.id,
            )
        if request.operation not in {"add_thread_tags", "remove_thread_tags"}:
            raise UserError("discord.channel_operation_invalid")
        for member in (actor, bot):
            _require_channel_permissions(channel, member, "manage_threads")
        tags = _forum_tags(channel, request.tag_ids)
        if request.operation == "add_thread_tags":
            await channel.add_tags(*tags, reason=reason)
        else:
            await channel.remove_tags(*tags, reason=reason)
        return _channel_operation_response(
            guild,
            request.operation,
            channel_id=channel.id,
            resource_id=",".join(str(tag.id) for tag in tags),
        )

    async def forward_message(
        request: DiscordForwardMessageRequest,
        context: InvocationContext,
    ) -> DiscordForwardMessageResponse:
        source_guild = _requested_guild(
            client,
            context,
            request.source_guild_id,
        )
        _, message = await _fetch_readable_message(
            source_guild,
            channel_id=request.source_channel_id,
            message_id=request.source_message_id,
            context=context,
        )
        destination_guild, destination, _, _ = await _write_message_channel(
            client,
            context,
            request.destination_channel_id,
            guild_id=request.destination_guild_id,
            required_permissions=("send_messages",),
        )
        try:
            posted = await message.forward(destination)
        except discord.Forbidden as exc:
            raise UserError("discord.message_forward_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_forward_failed") from exc
        return DiscordForwardMessageResponse(
            source_message_id=str(message.id),
            destination_message_id=str(posted.id),
            destination_channel_id=str(destination.id),
            destination_guild_id=str(destination_guild.id),
        )

    async def send_direct_message(
        request: DiscordSendDirectMessageRequest,
        context: InvocationContext,
    ) -> DiscordSendDirectMessageResponse:
        content = request.content.strip()
        if not 1 <= len(content) <= 2_000:
            raise UserError("discord.message_content_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        actor, _ = await _write_members(guild, context)
        target = await _target_member(guild, request.user_id)
        if target.id != actor.id:
            _require_guild_permission(actor, "administrator")
        try:
            posted = await target.send(
                content,
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
        except discord.Forbidden as exc:
            raise UserError("discord.direct_message_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.direct_message_failed") from exc
        return DiscordSendDirectMessageResponse(
            user_id=str(target.id),
            message_id=str(posted.id),
            channel_id=str(posted.channel.id),
        )

    async def set_bot_presence(
        request: DiscordSetBotPresenceRequest,
        context: InvocationContext,
    ) -> DiscordSetBotPresenceResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, _ = await _write_members(guild, context)
        await _require_global_application_permission(client, actor)
        activity_name = " ".join(request.activity_name.split())
        if request.activity_kind == "none":
            if activity_name or request.streaming_url is not None:
                raise UserError("discord.presence_activity_invalid")
            activity: discord.BaseActivity | None = None
        else:
            if not 1 <= len(activity_name) <= 128:
                raise UserError("discord.presence_activity_invalid")
            activity = _presence_activity(
                request.activity_kind,
                activity_name,
                request.streaming_url,
            )
        await client.change_presence(
            status=_presence_status(request.status),
            activity=activity,
        )
        return DiscordSetBotPresenceResponse(
            status=request.status,
            activity_kind=request.activity_kind,
            activity_name=activity_name or None,
            changed=True,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.channel_operation",
                summary=(
                    "Clone/follow a channel, join/leave/tag a thread, create a forum "
                    "tag, play a soundboard sound, or set/clear a voice-channel status "
                    "with live requester and bot permission checks."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "clone channel",
                    "follow announcement channel",
                    "follow this announcement channel",
                    "join thread",
                    "leave thread",
                    "add thread tag",
                    "remove thread tag",
                    "create forum tag",
                    "play soundboard sound",
                    "set voice channel status",
                    "チャンネルを複製",
                    "チャンネルをクローン",
                    "アナウンスチャンネルをフォロー",
                    "スレッドに参加",
                    "スレッドから退出",
                    "タグ追加",
                    "タグ解除",
                    "フォーラムタグを作る",
                    "サウンドボードを鳴らす",
                    "VCのステータスを変更",
                    "ボイスチャンネルの状態を変更",
                ),
                side_effects=("Mutates the selected Discord channel or thread.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=30,
            ),
            DiscordChannelOperationRequest,
            DiscordChannelOperationResponse,
            channel_operation,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.forward_message",
                summary=(
                    "Use Discord's native message forwarding after separately checking "
                    "source visibility and destination posting permission."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "forward", "message", "転送", "メッセージ"),
                side_effects=("Posts one forwarded message to the destination.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=20,
            ),
            DiscordForwardMessageRequest,
            DiscordForwardMessageResponse,
            forward_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_direct_message",
                summary=(
                    "Send a link-preview-suppressed DM to the requester, or to a shared "
                    "server member when the requester is an Administrator. Use purpose=final "
                    "when the DM is the complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "send DM",
                    "direct message",
                    "private message",
                    "DMを送る",
                    "DMを送って",
                    "メンバーへDMを送って",
                    "個別メッセージ",
                    "プライベートメッセージ",
                    "個チャ",
                    "ダイレクトメッセージ",
                ),
                side_effects=("Sends one direct message outside a server channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=15,
            ),
            DiscordSendDirectMessageRequest,
            DiscordSendDirectMessageResponse,
            send_direct_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.set_bot_presence",
                summary=(
                    "Set the global bot status and activity. This is restricted to an "
                    "Administrator who owns or belongs to the bot application team."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "set bot presence",
                    "bot status",
                    "bot activity",
                    "online idle dnd invisible",
                    "playing listening watching streaming custom status",
                    "ボットのプレゼンス",
                    "ボットをオンライン",
                    "ボットを離席中",
                    "ボットを取り込み中",
                    "ボットをオフライン表示",
                    "配信中にする",
                    "カスタムステータス",
                    "アクティビティを変更",
                ),
                side_effects=("Changes the bot's presence across every shared server.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=15,
            ),
            DiscordSetBotPresenceRequest,
            DiscordSetBotPresenceResponse,
            set_bot_presence,
        ),
    )


def _channel_operation_response(
    guild: discord.Guild,
    operation: DiscordChannelOperation,
    *,
    channel_id: int,
    resource_id: int | str | None = None,
    name: str | None = None,
) -> DiscordChannelOperationResponse:
    return DiscordChannelOperationResponse(
        operation=operation,
        source_guild_id=str(guild.id),
        channel_id=str(channel_id),
        resource_id=str(resource_id) if resource_id is not None else None,
        name=name,
        changed=True,
    )


def _forum_tags(
    thread: discord.Thread,
    tag_ids: tuple[str, ...],
) -> tuple[discord.ForumTag, ...]:
    if not 1 <= len(tag_ids) <= 5:
        raise UserError("discord.forum_tags_invalid")
    parent = thread.parent
    if not isinstance(parent, discord.ForumChannel):
        raise UserError("discord.forum_thread_required")
    available = {tag.id: tag for tag in parent.available_tags}
    tags: list[discord.ForumTag] = []
    for tag_id in dict.fromkeys(tag_ids):
        tag = available.get(_snowflake(tag_id, "forum tag"))
        if tag is None:
            raise UserError("discord.forum_tag_unavailable")
        tags.append(tag)
    return tuple(tags)


async def _target_member(
    guild: discord.Guild,
    user_id: str,
) -> discord.Member:
    target_id = _snowflake(user_id, "user")
    member = guild.get_member(target_id)
    if member is not None:
        return member
    try:
        return await guild.fetch_member(target_id)
    except (discord.NotFound, discord.Forbidden) as exc:
        raise UserError("discord.member_required") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.member_lookup_failed") from exc


def _presence_status(status: DiscordPresenceStatus) -> discord.Status:
    return {
        "online": discord.Status.online,
        "idle": discord.Status.idle,
        "dnd": discord.Status.dnd,
        "invisible": discord.Status.invisible,
    }[status]


def _presence_activity(
    kind: DiscordPresenceActivity,
    name: str,
    streaming_url: str | None,
) -> discord.BaseActivity:
    if kind == "playing":
        return discord.Game(name=name)
    if kind == "custom":
        return discord.CustomActivity(name=name)
    if kind == "streaming":
        if streaming_url is None or not streaming_url.startswith(("https://", "http://")):
            raise UserError("discord.presence_streaming_url_invalid")
        return discord.Streaming(name=name, url=streaming_url)
    activity_type = {
        "listening": discord.ActivityType.listening,
        "watching": discord.ActivityType.watching,
        "competing": discord.ActivityType.competing,
    }.get(kind)
    if activity_type is None:
        raise UserError("discord.presence_activity_invalid")
    return discord.Activity(type=activity_type, name=name)
