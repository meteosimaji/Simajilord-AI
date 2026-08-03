"""Permission-checked Discord expression and soundboard mutations."""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import discord

from simajilord.capabilities.file_scope import file_workspace_id
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
from simajilord.services.files import WorkspaceFileProvenance

from .capabilities import (
    _audit_reason,
    _bounded_name,
    _enforce_file_provenance_to_guild,
    _enforce_file_provenance_to_unknown_audience,
    _enforce_information_flow_to_guild,
    _enforce_unknown_audience,
    _requested_guild,
    _require_guild_permission,
    _snowflake,
    _write_members,
)

DiscordPlatformAssetKind = Literal[
    "guild_emoji",
    "application_emoji",
    "guild_sticker",
    "soundboard",
]

_ASSET_SIZE_LIMITS = {
    "guild_emoji": 256 * 1024,
    "application_emoji": 256 * 1024,
    "guild_sticker": 512 * 1024,
    "soundboard": 512 * 1024,
}


@dataclass(frozen=True, slots=True)
class DiscordCreatePlatformAssetRequest:
    kind: DiscordPlatformAssetKind
    name: str
    path: str
    guild_id: str | None = None
    description: str = ""
    emoji: str = ""
    volume: float = 1.0
    role_ids: tuple[str, ...] = ()
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
class DiscordUpdatePlatformAssetRequest:
    kind: DiscordPlatformAssetKind
    resource_id: str
    guild_id: str | None = None
    name: str | None = None
    description: str | None = None
    emoji: str | None = None
    clear_emoji: bool = False
    volume: float | None = None
    role_ids: tuple[str, ...] | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordDeletePlatformAssetRequest:
    kind: DiscordPlatformAssetKind
    resource_id: str
    guild_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordPlatformAssetResponse:
    kind: DiscordPlatformAssetKind
    resource_id: str
    name: str
    source_guild_id: str
    url: str | None
    changed: bool = True


def build_discord_platform_asset_endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> tuple[CapabilityEndpoint, ...]:
    """Build expression mutations backed only by the agent file sandbox."""

    async def create_asset(
        request: DiscordCreatePlatformAssetRequest,
        context: InvocationContext,
    ) -> DiscordPlatformAssetResponse:
        guild = _requested_guild(client, context, request.guild_id)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        _require_asset_create_permission(request.kind, actor, bot)
        name = _bounded_name(request.name, "discord.expression_name_invalid")
        content, filename, provenance = await _asset_bytes(
            runtime,
            context,
            request.path,
            kind=request.kind,
        )
        _enforce_file_provenance_to_guild(context, guild, provenance)
        if request.kind == "application_emoji":
            _enforce_unknown_audience(
                context,
                sink="global_application_emoji",
            )
            _enforce_file_provenance_to_unknown_audience(
                context,
                provenance,
                sink="global_application_emoji",
            )
        reason = _audit_reason(
            request.reason or f"Create {request.kind}",
            context,
        )
        if request.kind == "guild_emoji":
            roles = _asset_roles(guild, request.role_ids)
            guild_emoji = await guild.create_custom_emoji(
                name=name,
                image=content,
                roles=roles,
                reason=reason,
            )
            return _asset_response(guild, request.kind, guild_emoji)
        if request.kind == "application_emoji":
            await _require_global_application_permission(client, actor)
            application_emoji = await client.create_application_emoji(
                name=name,
                image=content,
            )
            return _asset_response(guild, request.kind, application_emoji)
        if request.kind == "guild_sticker":
            emoji = request.emoji.strip()
            if not emoji:
                raise UserError("discord.sticker_emoji_required")
            file = discord.File(io.BytesIO(content), filename=filename)
            try:
                sticker = await guild.create_sticker(
                    name=name,
                    description=request.description,
                    emoji=emoji,
                    file=file,
                    reason=reason,
                )
            finally:
                file.close()
            return _asset_response(guild, request.kind, sticker)
        if not 0 <= request.volume <= 1:
            raise UserError("discord.soundboard_volume_invalid")
        sound = await guild.create_soundboard_sound(
            name=name,
            sound=content,
            volume=request.volume,
            emoji=request.emoji.strip() or None,
            reason=reason,
        )
        return _asset_response(guild, request.kind, sound)

    async def update_asset(
        request: DiscordUpdatePlatformAssetRequest,
        context: InvocationContext,
    ) -> DiscordPlatformAssetResponse:
        guild = _requested_guild(client, context, request.guild_id)
        _enforce_information_flow_to_guild(context, guild)
        if request.kind == "application_emoji":
            _enforce_unknown_audience(
                context,
                sink="global_application_emoji",
            )
        actor, bot = await _write_members(guild, context)
        _require_asset_manage_permission(request.kind, actor, bot)
        name = (
            _bounded_name(request.name, "discord.expression_name_invalid")
            if request.name is not None
            else None
        )
        reason = _audit_reason(
            request.reason or f"Update {request.kind}",
            context,
        )
        if request.kind in {"guild_emoji", "application_emoji"}:
            asset = await _fetch_emoji(
                client,
                guild,
                request.kind,
                request.resource_id,
            )
            if name is None and request.role_ids is None:
                raise UserError("discord.resource_update_empty")
            roles = (
                _asset_roles(guild, request.role_ids)
                if request.role_ids is not None
                else None
            )
            if request.kind == "application_emoji":
                await _require_global_application_permission(client, actor)
                if request.role_ids is not None:
                    raise UserError("discord.application_emoji_roles_invalid")
                updated_emoji = await asset.edit(name=name or asset.name)
            elif roles is None:
                updated_emoji = await asset.edit(
                    name=name or asset.name,
                    reason=reason,
                )
            elif name is None:
                updated_emoji = await asset.edit(roles=roles, reason=reason)
            else:
                updated_emoji = await asset.edit(
                    name=name,
                    roles=roles,
                    reason=reason,
                )
            return _asset_response(guild, request.kind, updated_emoji)
        if request.kind == "guild_sticker":
            sticker = await _fetch_sticker(guild, request.resource_id)
            if (
                name is None
                and request.description is None
                and request.emoji is None
            ):
                raise UserError("discord.resource_update_empty")
            options: dict[str, object] = {"reason": reason}
            if name is not None:
                options["name"] = name
            if request.description is not None:
                options["description"] = request.description
            if request.emoji is not None:
                emoji = request.emoji.strip()
                if not emoji:
                    raise UserError("discord.sticker_emoji_required")
                options["emoji"] = emoji
            updated_sticker = await cast(Any, sticker.edit)(**options)
            return _asset_response(guild, request.kind, updated_sticker)
        sound = await _fetch_sound(guild, request.resource_id)
        if (
            name is None
            and request.volume is None
            and request.emoji is None
            and not request.clear_emoji
        ):
            raise UserError("discord.resource_update_empty")
        if request.volume is not None and not 0 <= request.volume <= 1:
            raise UserError("discord.soundboard_volume_invalid")
        sound_options: dict[str, object] = {"reason": reason}
        if name is not None:
            sound_options["name"] = name
        if request.volume is not None:
            sound_options["volume"] = request.volume
        if request.emoji is not None or request.clear_emoji:
            sound_options["emoji"] = (
                None if request.clear_emoji else request.emoji
            )
        updated_sound = await cast(Any, sound.edit)(**sound_options)
        return _asset_response(guild, request.kind, updated_sound or sound)

    async def delete_asset(
        request: DiscordDeletePlatformAssetRequest,
        context: InvocationContext,
    ) -> DiscordPlatformAssetResponse:
        guild = _requested_guild(client, context, request.guild_id)
        actor, bot = await _write_members(guild, context)
        _require_asset_manage_permission(request.kind, actor, bot)
        reason = _audit_reason(
            request.reason or f"Delete {request.kind}",
            context,
        )
        if request.kind in {"guild_emoji", "application_emoji"}:
            asset = await _fetch_emoji(
                client,
                guild,
                request.kind,
                request.resource_id,
            )
            if request.kind == "application_emoji":
                await _require_global_application_permission(client, actor)
                response = _asset_response(guild, request.kind, asset)
                await asset.delete()
                return response
            response = _asset_response(guild, request.kind, asset)
            await asset.delete(reason=reason)
            return response
        if request.kind == "guild_sticker":
            sticker = await _fetch_sticker(guild, request.resource_id)
            response = _asset_response(guild, request.kind, sticker)
            await sticker.delete(reason=reason)
            return response
        sound = await _fetch_sound(guild, request.resource_id)
        response = _asset_response(guild, request.kind, sound)
        await sound.delete(reason=reason)
        return response

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.create_platform_asset",
                summary=(
                    "Create a guild/application emoji, guild sticker, or soundboard "
                    "sound from a workspace file with live expression permissions."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "emoji",
                    "sticker",
                    "soundboard",
                    "expression",
                    "絵文字",
                    "スタンプ",
                    "サウンド",
                    "作成",
                ),
                side_effects=("Creates one Discord expression or sound.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=30,
            ),
            DiscordCreatePlatformAssetRequest,
            DiscordPlatformAssetResponse,
            create_asset,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.update_platform_asset",
                summary=(
                    "Edit a guild/application emoji, guild sticker, or soundboard "
                    "sound with live expression permissions."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "emoji",
                    "sticker",
                    "soundboard",
                    "edit",
                    "絵文字",
                    "スタンプ",
                    "編集",
                    "サーバースタンプの名前を編集して",
                ),
                side_effects=("Updates one Discord expression or sound.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                timeout_seconds=30,
            ),
            DiscordUpdatePlatformAssetRequest,
            DiscordPlatformAssetResponse,
            update_asset,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_platform_asset",
                summary=(
                    "Permanently delete a guild/application emoji, guild sticker, "
                    "or soundboard sound with live expression permissions."
                ),
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "emoji",
                    "sticker",
                    "soundboard",
                    "delete",
                    "絵文字",
                    "スタンプ",
                    "削除",
                    "サウンドボード音源を削除して",
                ),
                side_effects=("Permanently deletes one Discord expression or sound.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                timeout_seconds=30,
            ),
            DiscordDeletePlatformAssetRequest,
            DiscordPlatformAssetResponse,
            delete_asset,
        ),
    )


async def _asset_bytes(
    runtime: SimajilordRuntime,
    context: InvocationContext,
    path: str,
    *,
    kind: DiscordPlatformAssetKind,
) -> tuple[bytes, str, WorkspaceFileProvenance | None]:
    if "files" not in context.grants:
        raise UserError("files.grant_required")
    if runtime.files is None:
        raise UserError("files.disabled")
    if context.workspace_id is None:
        raise UserError("files.workspace_required")
    filename, content, provenance = await asyncio.to_thread(
        runtime.files.snapshot_for_delivery_with_provenance,
        file_workspace_id(context),
        path,
    )
    if len(content) > _ASSET_SIZE_LIMITS[kind]:
        raise UserError("discord.expression_file_too_large")
    return content, filename, provenance


def _require_asset_create_permission(
    kind: DiscordPlatformAssetKind,
    actor: discord.Member,
    bot: discord.Member,
) -> None:
    permission = (
        "administrator" if kind == "application_emoji" else "create_expressions"
    )
    members = (actor,) if kind == "application_emoji" else (actor, bot)
    for member in members:
        _require_guild_permission(member, permission)


def _require_asset_manage_permission(
    kind: DiscordPlatformAssetKind,
    actor: discord.Member,
    bot: discord.Member,
) -> None:
    permission = (
        "administrator" if kind == "application_emoji" else "manage_expressions"
    )
    members = (actor,) if kind == "application_emoji" else (actor, bot)
    for member in members:
        _require_guild_permission(member, permission)


async def _require_global_application_permission(
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


def _asset_roles(
    guild: discord.Guild,
    role_ids: tuple[str, ...],
) -> tuple[discord.Role, ...]:
    if len(role_ids) > 100:
        raise UserError("discord.expression_roles_invalid")
    roles: list[discord.Role] = []
    for role_id in dict.fromkeys(role_ids):
        role = guild.get_role(_snowflake(role_id, "role"))
        if role is None:
            raise UserError("discord.role_unavailable")
        roles.append(role)
    return tuple(roles)


async def _fetch_emoji(
    client: discord.Client,
    guild: discord.Guild,
    kind: DiscordPlatformAssetKind,
    resource_id: str,
) -> discord.Emoji:
    emoji_id = _snowflake(resource_id, "emoji")
    try:
        if kind == "application_emoji":
            return await client.fetch_application_emoji(emoji_id)
        return await guild.fetch_emoji(emoji_id)
    except discord.DiscordException as exc:
        raise UserError("discord.emoji_not_found") from exc


async def _fetch_sticker(
    guild: discord.Guild,
    resource_id: str,
) -> discord.GuildSticker:
    try:
        return await guild.fetch_sticker(_snowflake(resource_id, "sticker"))
    except discord.DiscordException as exc:
        raise UserError("discord.sticker_not_found") from exc


async def _fetch_sound(
    guild: discord.Guild,
    resource_id: str,
) -> discord.SoundboardSound:
    try:
        return await guild.fetch_soundboard_sound(
            _snowflake(resource_id, "soundboard sound"),
        )
    except discord.DiscordException as exc:
        raise UserError("discord.soundboard_sound_not_found") from exc


def _asset_response(
    guild: discord.Guild,
    kind: DiscordPlatformAssetKind,
    asset: object,
) -> DiscordPlatformAssetResponse:
    resource_id = getattr(asset, "id", None)
    name = getattr(asset, "name", None)
    if resource_id is None or not isinstance(name, str):
        raise UserError("discord.expression_response_invalid")
    url = getattr(asset, "url", None)
    return DiscordPlatformAssetResponse(
        kind=kind,
        resource_id=str(resource_id),
        name=name,
        source_guild_id=str(guild.id),
        url=str(url) if url is not None else None,
    )
