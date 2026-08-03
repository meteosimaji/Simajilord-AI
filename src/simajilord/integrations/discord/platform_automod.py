"""Typed Discord AutoMod mutations with live permission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

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
    _enforce_information_flow_to_guild,
    _requested_guild,
    _require_channel_permissions,
    _require_guild_permission,
    _snowflake,
    _write_members,
)

DiscordAutoModTriggerKind = Literal[
    "keyword",
    "spam",
    "keyword_preset",
    "mention_spam",
    "member_profile",
]
DiscordAutoModActionKind = Literal[
    "block_message",
    "send_alert_message",
    "timeout",
    "block_member_interactions",
]

_MAX_ACTIONS = 3
_MAX_EXEMPT_CHANNELS = 50
_MAX_EXEMPT_ROLES = 20
_MAX_KEYWORDS = 1_000
_MAX_REGEX_PATTERNS = 10
_MAX_TIMEOUT_SECONDS = 2_419_200


@dataclass(frozen=True, slots=True)
class DiscordAutoModActionInput:
    kind: DiscordAutoModActionKind
    channel_id: str | None = None
    duration_seconds: int | None = None
    custom_message: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordAutoModRuleInput:
    name: str
    trigger_kind: DiscordAutoModTriggerKind
    actions: tuple[DiscordAutoModActionInput, ...]
    enabled: bool = False
    keyword_filter: tuple[str, ...] = ()
    regex_patterns: tuple[str, ...] = ()
    allow_list: tuple[str, ...] = ()
    preset_profanity: bool = False
    preset_sexual_content: bool = False
    preset_slurs: bool = False
    mention_limit: int | None = None
    mention_raid_protection: bool = False
    exempt_role_ids: tuple[str, ...] = ()
    exempt_channel_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordCreateAutoModRuleRequest:
    rule: DiscordAutoModRuleInput
    guild_id: str | None = None
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
class DiscordUpdateAutoModRuleRequest:
    rule_id: str
    rule: DiscordAutoModRuleInput
    guild_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordDeleteAutoModRuleRequest:
    rule_id: str
    guild_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordAutoModRuleMutationResponse:
    rule_id: str
    name: str
    source_guild_id: str
    enabled: bool
    trigger_kind: str
    action_kinds: tuple[str, ...]
    changed: bool = True


def build_discord_automod_endpoints(
    client: discord.Client,
) -> tuple[CapabilityEndpoint, ...]:
    """Build complete create/update/delete AutoMod operations."""

    async def create_rule(
        request: DiscordCreateAutoModRuleRequest,
        context: InvocationContext,
    ) -> DiscordAutoModRuleMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _automod_members(guild, context)
        trigger, actions, roles, channels = _rule_parts(
            guild, actor, bot, request.rule
        )
        rule = await guild.create_automod_rule(
            name=_bounded_name(request.rule.name, "discord.automod_name_invalid"),
            event_type=_event_type(request.rule.trigger_kind),
            trigger=trigger,
            actions=actions,
            enabled=request.rule.enabled,
            exempt_roles=roles,
            exempt_channels=channels,
            reason=_audit_reason(request.reason or "Create AutoMod rule", context),
        )
        return _response(guild, rule)

    async def update_rule(
        request: DiscordUpdateAutoModRuleRequest,
        context: InvocationContext,
    ) -> DiscordAutoModRuleMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _automod_members(guild, context)
        existing = await _fetch_rule(guild, request.rule_id)
        trigger, actions, roles, channels = _rule_parts(
            guild, actor, bot, request.rule
        )
        updated = await existing.edit(
            name=_bounded_name(request.rule.name, "discord.automod_name_invalid"),
            event_type=_event_type(request.rule.trigger_kind),
            trigger=trigger,
            actions=actions,
            enabled=request.rule.enabled,
            exempt_roles=roles,
            exempt_channels=channels,
            reason=_audit_reason(request.reason or "Update AutoMod rule", context),
        )
        return _response(guild, updated)

    async def delete_rule(
        request: DiscordDeleteAutoModRuleRequest,
        context: InvocationContext,
    ) -> DiscordAutoModRuleMutationResponse:
        guild = _requested_guild(client, context, request.guild_id)
        await _automod_members(guild, context)
        rule = await _fetch_rule(guild, request.rule_id)
        response = _response(guild, rule)
        await rule.delete(
            reason=_audit_reason(request.reason or "Delete AutoMod rule", context)
        )
        return response

    shared_keywords = (
        "discord",
        "automod",
        "moderation",
        "filter",
        "spam",
        "keyword",
        "自動モデレーション",
        "スパム",
        "フィルター",
    )
    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.create_automod_rule",
                summary=(
                    "Create a Discord AutoMod keyword, spam, preset, mention-spam, "
                    "or member-profile rule with typed actions and live Manage Server "
                    "permission checks."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(*shared_keywords, "create", "作成"),
                side_effects=("Creates one server AutoMod rule.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=20,
            ),
            DiscordCreateAutoModRuleRequest,
            DiscordAutoModRuleMutationResponse,
            create_rule,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.update_automod_rule",
                summary=(
                    "Replace one Discord AutoMod rule after live requester and bot "
                    "permission checks."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(*shared_keywords, "update", "edit", "編集"),
                side_effects=("Updates one server AutoMod rule.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=20,
            ),
            DiscordUpdateAutoModRuleRequest,
            DiscordAutoModRuleMutationResponse,
            update_rule,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_automod_rule",
                summary=(
                    "Permanently delete one Discord AutoMod rule after destructive "
                    "approval and live permission checks."
                ),
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(*shared_keywords, "delete", "削除"),
                side_effects=("Permanently deletes one server AutoMod rule.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=20,
            ),
            DiscordDeleteAutoModRuleRequest,
            DiscordAutoModRuleMutationResponse,
            delete_rule,
        ),
    )


async def _automod_members(
    guild: discord.Guild,
    context: InvocationContext,
) -> tuple[discord.Member, discord.Member]:
    actor, bot = await _write_members(guild, context)
    for member in (actor, bot):
        _require_guild_permission(member, "manage_guild")
    return actor, bot


def _rule_parts(
    guild: discord.Guild,
    actor: discord.Member,
    bot: discord.Member,
    rule: DiscordAutoModRuleInput,
) -> tuple[
    discord.AutoModTrigger,
    list[discord.AutoModRuleAction],
    list[discord.Role],
    list[discord.abc.GuildChannel],
]:
    return (
        _trigger(rule),
        _actions(guild, actor, bot, rule),
        _roles(guild, rule.exempt_role_ids),
        _channels(guild, actor, bot, rule.exempt_channel_ids),
    )


def _trigger(rule: DiscordAutoModRuleInput) -> discord.AutoModTrigger:
    keywords = _bounded_unique_text(
        rule.keyword_filter, _MAX_KEYWORDS, "discord.automod_keywords_invalid"
    )
    regex = _bounded_unique_text(
        rule.regex_patterns, _MAX_REGEX_PATTERNS, "discord.automod_regex_invalid"
    )
    allow_list = _bounded_unique_text(
        rule.allow_list, _MAX_KEYWORDS, "discord.automod_allow_list_invalid"
    )
    if rule.trigger_kind == "keyword":
        if not keywords and not regex:
            raise UserError("discord.automod_keywords_required")
        return discord.AutoModTrigger(
            type=discord.AutoModRuleTriggerType.keyword,
            keyword_filter=keywords,
            regex_patterns=regex,
            allow_list=allow_list,
        )
    if rule.trigger_kind == "spam":
        return discord.AutoModTrigger(type=discord.AutoModRuleTriggerType.spam)
    if rule.trigger_kind == "keyword_preset":
        presets = discord.AutoModPresets(
            profanity=rule.preset_profanity,
            sexual_content=rule.preset_sexual_content,
            slurs=rule.preset_slurs,
        )
        if presets.value == 0:
            raise UserError("discord.automod_preset_required")
        return discord.AutoModTrigger(
            type=discord.AutoModRuleTriggerType.keyword_preset,
            presets=presets,
            allow_list=allow_list,
        )
    if rule.trigger_kind == "mention_spam":
        if rule.mention_limit is None or not 1 <= rule.mention_limit <= 50:
            raise UserError("discord.automod_mention_limit_invalid")
        return discord.AutoModTrigger(
            type=discord.AutoModRuleTriggerType.mention_spam,
            mention_limit=rule.mention_limit,
            mention_raid_protection=rule.mention_raid_protection,
        )
    if not keywords and not regex:
        raise UserError("discord.automod_keywords_required")
    return discord.AutoModTrigger(
        type=discord.AutoModRuleTriggerType.member_profile,
        keyword_filter=keywords,
        regex_patterns=regex,
        allow_list=allow_list,
    )


def _actions(
    guild: discord.Guild,
    actor: discord.Member,
    bot: discord.Member,
    rule: DiscordAutoModRuleInput,
) -> list[discord.AutoModRuleAction]:
    if not 1 <= len(rule.actions) <= _MAX_ACTIONS:
        raise UserError("discord.automod_actions_invalid")
    if len({action.kind for action in rule.actions}) != len(rule.actions):
        raise UserError("discord.automod_action_duplicate")
    values: list[discord.AutoModRuleAction] = []
    for action in rule.actions:
        if action.kind == "block_message":
            if action.custom_message is not None and len(action.custom_message) > 150:
                raise UserError("discord.automod_custom_message_invalid")
            values.append(
                discord.AutoModRuleAction(
                    type=discord.AutoModRuleActionType.block_message,
                    custom_message=action.custom_message,
                )
            )
        elif action.kind == "send_alert_message":
            if action.channel_id is None:
                raise UserError("discord.automod_alert_channel_required")
            channel = _visible_channel(guild, actor, bot, action.channel_id)
            if not isinstance(channel, discord.TextChannel):
                raise UserError("discord.automod_alert_channel_invalid")
            values.append(
                discord.AutoModRuleAction(
                    type=discord.AutoModRuleActionType.send_alert_message,
                    channel_id=channel.id,
                )
            )
        elif action.kind == "timeout":
            if (
                action.duration_seconds is None
                or not 1 <= action.duration_seconds <= _MAX_TIMEOUT_SECONDS
            ):
                raise UserError("discord.automod_timeout_invalid")
            if rule.trigger_kind == "member_profile":
                raise UserError("discord.automod_action_trigger_invalid")
            values.append(
                discord.AutoModRuleAction(
                    type=discord.AutoModRuleActionType.timeout,
                    duration=timedelta(seconds=action.duration_seconds),
                )
            )
        else:
            if rule.trigger_kind != "member_profile":
                raise UserError("discord.automod_action_trigger_invalid")
            values.append(
                discord.AutoModRuleAction(
                    type=discord.AutoModRuleActionType.block_member_interactions,
                )
            )
    return values


def _roles(guild: discord.Guild, role_ids: tuple[str, ...]) -> list[discord.Role]:
    if len(role_ids) > _MAX_EXEMPT_ROLES:
        raise UserError("discord.automod_exempt_roles_invalid")
    roles: list[discord.Role] = []
    for role_id in dict.fromkeys(role_ids):
        role = guild.get_role(_snowflake(role_id, "role"))
        if role is None:
            raise UserError("discord.role_unavailable")
        roles.append(role)
    return roles


def _channels(
    guild: discord.Guild,
    actor: discord.Member,
    bot: discord.Member,
    channel_ids: tuple[str, ...],
) -> list[discord.abc.GuildChannel]:
    if len(channel_ids) > _MAX_EXEMPT_CHANNELS:
        raise UserError("discord.automod_exempt_channels_invalid")
    return [
        _visible_channel(guild, actor, bot, channel_id)
        for channel_id in dict.fromkeys(channel_ids)
    ]


def _visible_channel(
    guild: discord.Guild,
    actor: discord.Member,
    bot: discord.Member,
    channel_id: str,
) -> discord.abc.GuildChannel:
    channel = guild.get_channel(_snowflake(channel_id, "channel"))
    if channel is None:
        raise UserError("discord.channel_unavailable")
    for member in (actor, bot):
        _require_channel_permissions(channel, member, "view_channel")
    return channel


def _bounded_unique_text(
    values: tuple[str, ...],
    maximum: int,
    code: str,
) -> list[str]:
    if len(values) > maximum:
        raise UserError(code)
    normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
    if any(len(value) > 260 for value in normalized):
        raise UserError(code)
    return normalized


def _event_type(kind: DiscordAutoModTriggerKind) -> discord.AutoModRuleEventType:
    if kind == "member_profile":
        return discord.AutoModRuleEventType.member_update
    return discord.AutoModRuleEventType.message_send


async def _fetch_rule(
    guild: discord.Guild,
    rule_id: str,
) -> discord.AutoModRule:
    try:
        return await guild.fetch_automod_rule(_snowflake(rule_id, "AutoMod rule"))
    except discord.DiscordException as exc:
        raise UserError("discord.automod_rule_not_found") from exc


def _response(
    guild: discord.Guild,
    rule: discord.AutoModRule,
) -> DiscordAutoModRuleMutationResponse:
    return DiscordAutoModRuleMutationResponse(
        rule_id=str(rule.id),
        name=rule.name,
        source_guild_id=str(guild.id),
        enabled=rule.enabled,
        trigger_kind=rule.trigger.type.name,
        action_kinds=tuple(action.type.name for action in rule.actions),
    )
