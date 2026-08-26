"""Restart-safe, join-created temporary voice rooms and private controls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.async_locks import KeyedAsyncLockPool
from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime
from simajilord.services import (
    DEFAULT_TEMP_VOICE_GRACE_SECONDS,
    TempVoiceConfig,
    TempVoiceCreator,
    TempVoiceRoom,
    TempVoiceRoomEndReason,
    normalize_temp_voice_room_name,
    render_temp_voice_room_name,
    validate_temp_voice_user_limit,
)

from .permissions import permission_enabled
from .presenter import EmbedField, EmbedTone, command_embed

log = logging.getLogger(__name__)

_TEMP_VOICE_CATEGORY_NAME = "TEMP VC"
_TEMP_VOICE_CREATOR_NAME = "Join to create"
_TEMP_VOICE_DELETE_RETRY_SECONDS = 60
_TEMP_VOICE_RECONCILE_SECONDS = 5 * 60
_TEMP_VOICE_VIEW_TIMEOUT_SECONDS = 10 * 60
_TEMP_VOICE_AUDIT_REASON = "Simajilord TempVC"


class TempVoiceSafeView(discord.ui.View):
    """Final error boundary for TempVC component interactions."""

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item[discord.ui.View],
    ) -> None:
        del item
        await _send_temp_voice_error(interaction, error)


class TempVoiceSafeModal(discord.ui.Modal):
    """Final error boundary for TempVC modal submissions."""

    async def on_error(  # type: ignore[override]
        self,
        interaction: discord.Interaction,
        error: Exception,
    ) -> None:
        await _send_temp_voice_error(interaction, error)


class TempVoiceAdminSettingsModal(TempVoiceSafeModal, title="TempVC settings"):
    room_template: discord.ui.TextInput[TempVoiceAdminSettingsModal] = discord.ui.TextInput(
        label="Room name template",
        placeholder="{display_name}'s room",
        min_length=1,
        max_length=100,
    )
    grace_minutes: discord.ui.TextInput[TempVoiceAdminSettingsModal] = discord.ui.TextInput(
        label="Empty-room recovery (minutes)",
        placeholder="10",
        min_length=1,
        max_length=2,
    )

    def __init__(
        self,
        parent: TempVoiceAdminView,
        config: TempVoiceConfig,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.parent = parent
        self.room_template.default = config.room_name_template
        self.grace_minutes.default = str(config.empty_grace_seconds // 60)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            try:
                minutes = int(str(self.grace_minutes).strip())
            except ValueError as exc:
                raise UserError("temp_voice.grace_invalid") from exc
            await interaction.response.defer(ephemeral=True)
            await self.parent.cog.runtime.temp_voice.update_settings(
                str(self.parent.guild_id),
                room_name_template=str(self.room_template),
                empty_grace_seconds=minutes * 60,
            )
            await self.parent.cog.reschedule_empty_rooms(self.parent.guild_id)
            await self.parent.cog.edit_admin_panel(interaction, self.parent.guild_id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceCategorySelect(discord.ui.ChannelSelect[discord.ui.View]):
    def __init__(self) -> None:
        super().__init__(
            channel_types=[discord.ChannelType.category],
            placeholder="Add a creator in an existing category",
            min_values=1,
            max_values=1,
            row=2,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        parent = cast(TempVoiceAdminView, self.view)
        try:
            member = _require_temp_voice_manager(interaction)
            selected = self.values[0]
            category = member.guild.get_channel(selected.id)
            if not isinstance(category, discord.CategoryChannel):
                raise UserError("temp_voice.category_missing")
            await interaction.response.defer(ephemeral=True)
            await parent.cog.create_creator_in_category(
                member,
                category,
                request_id=str(interaction.id),
            )
            await parent.cog.edit_admin_panel(interaction, member.guild.id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceCreateView(TempVoiceSafeView):
    """Requester-only retry panel for a member still sitting in a creator lobby."""

    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        guild_id: int,
        creator_channel_id: int,
        creation_enabled: bool,
        can_manage: bool,
    ) -> None:
        super().__init__(timeout=_TEMP_VOICE_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.requester_id = requester_id
        self.guild_id = guild_id
        self.creator_channel_id = creator_channel_id
        self.create_button.disabled = not creation_enabled
        if not can_manage:
            self.remove_item(self.setup_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Run `/tempvc` to open your own creation panel.",
                ephemeral=True,
            )
            return False
        member = interaction.user
        return isinstance(member, discord.Member) and member.guild.id == self.guild_id

    @discord.ui.button(
        label="Retry room creation",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:tempvc:create-room",
        row=0,
    )
    async def create_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceCreateView],
    ) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            channel, recovered = await self.cog.create_room_from_creator(
                interaction,
                self.creator_channel_id,
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "TempVC recovered" if recovered else "TempVC created",
                    description=(
                        f"Moved you to {channel.mention}. Your previous room was still in its "
                        "recovery window, so no replacement was created."
                        if recovered
                        else f"Moved you to {channel.mention}. Run `/tempvc` there for controls."
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
            self.stop()
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Setup",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:creator-setup",
        row=0,
    )
    async def setup_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceCreateView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            await interaction.response.defer(ephemeral=True)
            config = await self.cog.runtime.temp_voice.config(str(member.guild.id))
            creators = await self.cog.runtime.temp_voice.creators(str(member.guild.id))
            embed, view = self.cog.admin_panel(member, config, creators)
            await interaction.edit_original_response(embed=embed, view=view)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceAdminView(TempVoiceSafeView):
    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        guild_id: int,
        config: TempVoiceConfig | None,
        creator_count: int,
    ) -> None:
        super().__init__(timeout=_TEMP_VOICE_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.requester_id = requester_id
        self.guild_id = guild_id
        self.config = config
        self.creator_count = creator_count
        self.create_hub_button.label = (
            "Create TempVC hub" if creator_count == 0 else "Add another hub"
        )
        self.pause_button.disabled = config is None
        self.settings_button.disabled = config is None
        self.permission_copy_button.disabled = creator_count == 0
        if config is not None:
            self.pause_button.label = "Pause creation" if config.enabled else "Resume creation"
            self.pause_button.style = (
                discord.ButtonStyle.secondary if config.enabled else discord.ButtonStyle.success
            )
        self.add_item(TempVoiceCategorySelect())

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Run `/tempvc` to open your own TempVC panel.",
                ephemeral=True,
            )
            return False
        try:
            member = _require_temp_voice_manager(interaction)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)
            return False
        return member.guild.id == self.guild_id

    @discord.ui.button(
        label="Create TempVC hub",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:tempvc:create-hub",
        row=0,
    )
    async def create_hub_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceAdminView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            await interaction.response.defer(ephemeral=True)
            await self.cog.create_hub(member, request_id=str(interaction.id))
            await self.cog.edit_admin_panel(interaction, member.guild.id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Pause creation",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:pause",
        row=0,
    )
    async def pause_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceAdminView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            config = await self.cog.runtime.temp_voice.config(str(member.guild.id))
            if config is None:
                raise UserError("temp_voice.not_configured")
            await interaction.response.defer(ephemeral=True)
            await self.cog.runtime.temp_voice.set_enabled(
                str(member.guild.id),
                not config.enabled,
            )
            await self.cog.record_event(
                "temp_voice.creation_toggled",
                workspace_id=str(member.guild.id),
                actor_id=str(member.id),
                request_id=str(interaction.id),
                payload={"enabled": not config.enabled},
            )
            await self.cog.edit_admin_panel(interaction, member.guild.id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Settings",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:settings",
        row=0,
    )
    async def settings_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceAdminView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            config = await self.cog.runtime.temp_voice.config(str(member.guild.id))
            if config is None:
                raise UserError("temp_voice.not_configured")
            await interaction.response.send_modal(TempVoiceAdminSettingsModal(self, config))
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Permission copy",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:permission-copy",
        row=1,
    )
    async def permission_copy_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceAdminView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            await interaction.response.defer(ephemeral=True)
            embed, view = await self.cog.permission_copy_panel(member)
            await interaction.edit_original_response(embed=embed, view=view)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceCreatorSelect(discord.ui.Select[discord.ui.View]):
    def __init__(
        self,
        guild: discord.Guild,
        creators: tuple[TempVoiceCreator, ...],
        selected_creator_id: int,
    ) -> None:
        options: list[discord.SelectOption] = []
        for creator in creators:
            channel = guild.get_channel(int(creator.channel_id))
            label = (
                channel.name
                if isinstance(channel, discord.VoiceChannel)
                else f"Missing creator {creator.channel_id}"
            )
            options.append(
                discord.SelectOption(
                    label=label[:100],
                    value=creator.channel_id,
                    default=int(creator.channel_id) == selected_creator_id,
                )
            )
        super().__init__(
            placeholder="Choose the creator lobby to configure",
            min_values=1,
            max_values=1,
            options=options,
            row=0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        parent = cast(TempVoicePermissionCopyView, self.view)
        try:
            await interaction.response.defer(ephemeral=True)
            await parent.cog.edit_permission_copy_panel(
                interaction,
                parent.guild_id,
                int(self.values[0]),
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoicePermissionSourceSelect(discord.ui.ChannelSelect[discord.ui.View]):
    def __init__(self) -> None:
        super().__init__(
            channel_types=[discord.ChannelType.voice],
            placeholder="Copy permissions from an existing voice channel",
            min_values=1,
            max_values=1,
            row=1,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        parent = cast(TempVoicePermissionCopyView, self.view)
        try:
            member = _require_temp_voice_manager(interaction)
            selected = self.values[0]
            source = member.guild.get_channel(selected.id)
            if not isinstance(source, discord.VoiceChannel):
                raise UserError("temp_voice.permission_source_missing")
            await interaction.response.defer(ephemeral=True)
            await parent.cog.set_creator_permission_source(
                interaction,
                parent.creator_channel_id,
                source.id,
            )
            await parent.cog.edit_permission_copy_panel(
                interaction,
                member.guild.id,
                parent.creator_channel_id,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoicePermissionCopyView(TempVoiceSafeView):
    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        guild: discord.Guild,
        creators: tuple[TempVoiceCreator, ...],
        creator_channel_id: int,
    ) -> None:
        super().__init__(timeout=_TEMP_VOICE_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.requester_id = requester_id
        self.guild_id = guild.id
        self.creator_channel_id = creator_channel_id
        self.add_item(TempVoiceCreatorSelect(guild, creators, creator_channel_id))
        self.add_item(TempVoicePermissionSourceSelect())

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Run `/tempvc` to open your own setup panel.",
                ephemeral=True,
            )
            return False
        try:
            member = _require_temp_voice_manager(interaction)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)
            return False
        return member.guild.id == self.guild_id

    @discord.ui.button(
        label="Use category permissions",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:permission-category",
        row=2,
    )
    async def use_category_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoicePermissionCopyView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            await interaction.response.defer(ephemeral=True)
            await self.cog.set_creator_permission_source(
                interaction,
                self.creator_channel_id,
                None,
            )
            await self.cog.edit_permission_copy_panel(
                interaction,
                member.guild.id,
                self.creator_channel_id,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Back to setup",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:permission-back",
        row=2,
    )
    async def back_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoicePermissionCopyView],
    ) -> None:
        try:
            member = _require_temp_voice_manager(interaction)
            await interaction.response.defer(ephemeral=True)
            config = await self.cog.runtime.temp_voice.config(str(member.guild.id))
            creators = await self.cog.runtime.temp_voice.creators(str(member.guild.id))
            embed, view = self.cog.admin_panel(member, config, creators)
            await interaction.edit_original_response(embed=embed, view=view)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceRenameModal(TempVoiceSafeModal, title="Rename TempVC"):
    room_name: discord.ui.TextInput[TempVoiceRenameModal] = discord.ui.TextInput(
        label="Room name",
        min_length=1,
        max_length=100,
    )

    def __init__(self, parent: TempVoiceRoomControlView, room: TempVoiceRoom) -> None:
        super().__init__(timeout=5 * 60)
        self.parent = parent
        self.room_name.default = room.name

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            await self.parent.cog.rename_room(
                interaction,
                self.parent.channel_id,
                str(self.room_name),
            )
            await self.parent.cog.edit_room_panel(interaction, self.parent.channel_id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceLimitModal(TempVoiceSafeModal, title="TempVC user limit"):
    user_limit: discord.ui.TextInput[TempVoiceLimitModal] = discord.ui.TextInput(
        label="Users (0 removes the limit)",
        min_length=1,
        max_length=2,
    )

    def __init__(self, parent: TempVoiceRoomControlView, room: TempVoiceRoom) -> None:
        super().__init__(timeout=5 * 60)
        self.parent = parent
        self.user_limit.default = str(room.user_limit)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            try:
                user_limit = int(str(self.user_limit).strip())
            except ValueError as exc:
                raise UserError("temp_voice.user_limit_invalid") from exc
            validate_temp_voice_user_limit(user_limit)
            await interaction.response.defer(ephemeral=True)
            await self.parent.cog.set_room_user_limit(
                interaction,
                self.parent.channel_id,
                user_limit,
            )
            await self.parent.cog.edit_room_panel(interaction, self.parent.channel_id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceRoomControlView(TempVoiceSafeView):
    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        channel_id: int,
        room: TempVoiceRoom,
        can_keep_permanent: bool,
    ) -> None:
        super().__init__(timeout=_TEMP_VOICE_VIEW_TIMEOUT_SECONDS)
        self.cog = cog
        self.requester_id = requester_id
        self.channel_id = channel_id
        self.lock_button.label = "Unlock" if room.locked else "Lock"
        self.lock_button.style = (
            discord.ButtonStyle.success if room.locked else discord.ButtonStyle.secondary
        )
        if not can_keep_permanent:
            self.remove_item(self.keep_permanent_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Run `/tempvc` inside your room to open your own controls.",
            ephemeral=True,
        )
        return False

    @discord.ui.button(
        label="Rename",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:rename",
        row=0,
    )
    async def rename_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            room, _channel, _member = await self.cog.require_room_controller(
                interaction,
                self.channel_id,
            )
            await interaction.response.send_modal(TempVoiceRenameModal(self, room))
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Limit",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:limit",
        row=0,
    )
    async def limit_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            room, _channel, _member = await self.cog.require_room_controller(
                interaction,
                self.channel_id,
            )
            await interaction.response.send_modal(TempVoiceLimitModal(self, room))
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Lock",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:lock",
        row=0,
    )
    async def lock_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            room, _channel, _member = await self.cog.require_room_controller(
                interaction,
                self.channel_id,
            )
            await interaction.response.defer(ephemeral=True)
            await self.cog.set_room_lock(
                interaction,
                self.channel_id,
                not room.locked,
            )
            await self.cog.edit_room_panel(interaction, self.channel_id)
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Invite",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:tempvc:invite",
        row=1,
    )
    async def invite_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            await self.cog.require_room_controller(interaction, self.channel_id)
            await interaction.response.send_message(
                embed=command_embed(
                    "Invite to TempVC",
                    description=(
                        "Choose one member. They will be allowed to enter this room, "
                        "including while it is locked."
                    ),
                ),
                view=TempVoiceMemberSelectView(
                    self.cog,
                    requester_id=self.requester_id,
                    channel_id=self.channel_id,
                    action="invite",
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Remove member",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:remove-member",
        row=1,
    )
    async def remove_member_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            await self.cog.require_room_controller(interaction, self.channel_id)
            await interaction.response.send_message(
                embed=command_embed(
                    "Remove from TempVC",
                    description="Choose a current room member, then choose remove once or ban.",
                ),
                view=TempVoiceMemberSelectView(
                    self.cog,
                    requester_id=self.requester_id,
                    channel_id=self.channel_id,
                    action="remove",
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Transfer owner",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:transfer",
        row=1,
    )
    async def transfer_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            await self.cog.require_room_controller(interaction, self.channel_id)
            await interaction.response.send_message(
                embed=command_embed(
                    "Transfer TempVC ownership",
                    description="Choose a current room member to become the owner.",
                    tone=EmbedTone.WARNING,
                ),
                view=TempVoiceMemberSelectView(
                    self.cog,
                    requester_id=self.requester_id,
                    channel_id=self.channel_id,
                    action="transfer",
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @discord.ui.button(
        label="Keep room",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:tempvc:keep",
        row=2,
    )
    async def keep_permanent_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRoomControlView],
    ) -> None:
        try:
            await self.cog.require_room_controller(
                interaction,
                self.channel_id,
                manager_required=True,
            )
            await interaction.response.defer(ephemeral=True)
            channel = await self.cog.keep_room_permanent(interaction, self.channel_id)
            await interaction.edit_original_response(
                embed=command_embed(
                    "Room kept permanently",
                    description=(
                        f"{channel.mention} is now an ordinary voice channel. "
                        "TempVC will no longer delete it."
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceUserSelect(discord.ui.UserSelect[discord.ui.View]):
    def __init__(self, action: str) -> None:
        placeholders = {
            "invite": "Choose a member to invite",
            "remove": "Choose a current member",
            "transfer": "Choose the new owner",
        }
        super().__init__(
            placeholder=placeholders[action],
            min_values=1,
            max_values=1,
        )
        self.action = action

    async def callback(self, interaction: discord.Interaction) -> None:
        parent = cast(TempVoiceMemberSelectView, self.view)
        try:
            target = self.values[0]
            if not isinstance(target, discord.Member) or target.bot:
                raise UserError("temp_voice.member_invalid")
            if self.action == "remove":
                _room, channel, _member = await parent.cog.require_room_controller(
                    interaction,
                    parent.channel_id,
                )
                if target not in channel.members:
                    raise UserError("temp_voice.member_not_present")
                await interaction.response.edit_message(
                    embed=command_embed(
                        "Confirm member removal",
                        description=(
                            f"Remove **{discord.utils.escape_markdown(target.display_name)}** "
                            "once, or deny them access to this room?"
                        ),
                        tone=EmbedTone.WARNING,
                    ),
                    view=TempVoiceRemovalConfirmView(
                        parent.cog,
                        requester_id=parent.requester_id,
                        channel_id=parent.channel_id,
                        target_id=target.id,
                    ),
                )
                return
            await interaction.response.defer(ephemeral=True)
            if self.action == "invite":
                await parent.cog.invite_member(
                    interaction,
                    parent.channel_id,
                    target.id,
                )
                title = "Member invited"
                description = f"{target.mention} can now enter this TempVC."
            else:
                await parent.cog.transfer_owner(
                    interaction,
                    parent.channel_id,
                    target.id,
                )
                title = "Ownership transferred"
                description = f"{target.mention} now controls this TempVC."
            await interaction.edit_original_response(
                embed=command_embed(
                    title,
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceMemberSelectView(TempVoiceSafeView):
    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        channel_id: int,
        action: str,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.channel_id = channel_id
        self.add_item(TempVoiceUserSelect(action))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the person who opened these controls can use them.",
            ephemeral=True,
        )
        return False


class TempVoiceRemovalConfirmView(TempVoiceSafeView):
    def __init__(
        self,
        cog: TempVoiceCog,
        *,
        requester_id: int,
        channel_id: int,
        target_id: int,
    ) -> None:
        super().__init__(timeout=60)
        self.cog = cog
        self.requester_id = requester_id
        self.channel_id = channel_id
        self.target_id = target_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the person who opened these controls can use them.",
            ephemeral=True,
        )
        return False

    @discord.ui.button(
        label="Remove once",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:tempvc:kick",
    )
    async def remove_once_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRemovalConfirmView],
    ) -> None:
        await self._finish(interaction, ban=False)

    @discord.ui.button(
        label="Ban from room",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:tempvc:ban",
    )
    async def ban_button(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[TempVoiceRemovalConfirmView],
    ) -> None:
        await self._finish(interaction, ban=True)

    async def _finish(self, interaction: discord.Interaction, *, ban: bool) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            target = await self.cog.remove_member(
                interaction,
                self.channel_id,
                self.target_id,
                ban=ban,
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Member banned" if ban else "Member removed",
                    description=(
                        f"{target.mention} cannot re-enter until invited."
                        if ban
                        else f"{target.mention} was disconnected from this room."
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
            self.stop()
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)


class TempVoiceCog(commands.Cog):
    """Create, recover, control, and retire BOT-owned temporary voice rooms."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._guild_locks = KeyedAsyncLockPool()
        self._member_locks = KeyedAsyncLockPool()
        self._room_locks = KeyedAsyncLockPool()
        self._deletion_tasks: dict[int, asyncio.Task[None]] = {}
        self._room_deletions_in_flight: set[int] = set()
        self._reconciliation_task: asyncio.Task[None] | None = None
        self._restored = False

    async def cog_unload(self) -> None:
        tasks = list(self._deletion_tasks.values())
        if self._reconciliation_task is not None:
            tasks.append(self._reconciliation_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._deletion_tasks.clear()
        self._room_deletions_in_flight.clear()
        self._reconciliation_task = None

    @app_commands.command(
        name="tempvc",
        description="Set up TempVC or manage the temporary room you are in.",
    )
    async def tempvc(self, interaction: discord.Interaction) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member):
                raise UserError("workspace.required")
            voice_channel = member.voice.channel if member.voice is not None else None
            room = (
                await self.runtime.temp_voice.room(str(voice_channel.id))
                if isinstance(voice_channel, discord.VoiceChannel)
                else None
            )
            if room is not None and isinstance(voice_channel, discord.VoiceChannel):
                if room.owner_id != str(member.id) and not _can_manage_temp_voice(member):
                    await interaction.response.send_message(
                        embed=command_embed(
                            "TempVC room",
                            description=(
                                f"{voice_channel.mention} is controlled by <@{room.owner_id}>. "
                                "The owner can use `/tempvc` here to change it."
                            ),
                            fields=(
                                EmbedField("User limit", _user_limit_label(room.user_limit)),
                                EmbedField("Access", "Locked" if room.locked else "Open"),
                            ),
                        ),
                        ephemeral=True,
                    )
                    return
                config = await self.runtime.temp_voice.config(str(member.guild.id))
                if config is None:
                    raise UserError("temp_voice.not_configured")
                room_embed, room_view = self.room_panel(
                    member,
                    voice_channel,
                    room,
                    config,
                )
                await interaction.response.send_message(
                    embed=room_embed,
                    view=room_view,
                    ephemeral=True,
                )
                return

            creator = (
                await self.runtime.temp_voice.creator(str(voice_channel.id))
                if isinstance(voice_channel, discord.VoiceChannel)
                else None
            )
            if (
                creator is not None
                and isinstance(voice_channel, discord.VoiceChannel)
                and creator.workspace_id == str(member.guild.id)
            ):
                config = await self.runtime.temp_voice.config(str(member.guild.id))
                if config is None:
                    raise UserError("temp_voice.not_configured")
                creator_embed, creator_view = self.creation_panel(
                    member,
                    voice_channel,
                    creator,
                    config,
                )
                await interaction.response.send_message(
                    embed=creator_embed,
                    view=creator_view,
                    ephemeral=True,
                )
                return

            config = await self.runtime.temp_voice.config(str(member.guild.id))
            creators = await self.runtime.temp_voice.creators(str(member.guild.id))
            if _can_manage_temp_voice(member):
                admin_embed, admin_view = self.admin_panel(member, config, creators)
                await interaction.response.send_message(
                    embed=admin_embed,
                    view=admin_view,
                    ephemeral=True,
                )
                return
            if config is None or not creators:
                description = "TempVC has not been set up by a server administrator."
                tone = EmbedTone.WARNING
            elif not config.enabled:
                description = "New TempVC rooms are temporarily paused. Existing rooms are kept."
                tone = EmbedTone.WARNING
            else:
                description = (
                    "Join one of these permanent creator lobbies. METEOBOT creates your room "
                    "and moves you automatically. If you remain in the lobby after a temporary "
                    "failure, run `/tempvc` and press **Retry room creation**:\n"
                    + "\n".join(f"• <#{creator.channel_id}>" for creator in creators)
                )
                tone = EmbedTone.SUCCESS
            await interaction.response.send_message(
                embed=command_embed(
                    "TempVC",
                    description=description,
                    fields=(
                        EmbedField(
                            "Recovery",
                            (
                                "Empty rooms wait **10 minutes by default** before deletion. "
                                "Returning cancels deletion, and your name, limit, and lock "
                                "setting are restored next time."
                            ),
                            inline=False,
                        ),
                    ),
                    tone=tone,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await _send_temp_voice_error(interaction, exc)

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot or before.channel == after.channel:
            return
        try:
            if isinstance(after.channel, discord.VoiceChannel):
                joined_room = await self.runtime.temp_voice.room(str(after.channel.id))
                if joined_room is not None:
                    await self._handle_room_join(member, after.channel, joined_room)
                else:
                    creator = await self.runtime.temp_voice.creator(str(after.channel.id))
                    if (
                        creator is not None
                        and creator.workspace_id == str(member.guild.id)
                    ):
                        request_id = f"voice:{member.id}:{after.channel.id}"
                        try:
                            await self._create_or_recover_room_from_creator(
                                member,
                                after.channel.id,
                                request_id=request_id,
                            )
                        except UserError as exc:
                            log.warning(
                                "TempVC automatic creation rejected guild=%s member=%s "
                                "creator=%s error=%s",
                                member.guild.id,
                                member.id,
                                after.channel.id,
                                exc.code,
                            )
                            await self.record_event(
                                "temp_voice.auto_create_rejected",
                                workspace_id=str(member.guild.id),
                                actor_id=str(member.id),
                                request_id=request_id,
                                payload={
                                    "creator_channel_id": str(after.channel.id),
                                    "error": exc.code,
                                },
                            )
            if isinstance(before.channel, discord.VoiceChannel):
                left_room = await self.runtime.temp_voice.room(str(before.channel.id))
                if left_room is not None:
                    await self._handle_room_departure(member, before.channel, left_room)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "TempVC voice transition failed guild=%s member=%s before=%s after=%s",
                member.guild.id,
                member.id,
                getattr(before.channel, "id", None),
                getattr(after.channel, "id", None),
            )

    @commands.Cog.listener()
    async def on_guild_channel_delete(self, channel: discord.abc.GuildChannel) -> None:
        if not isinstance(channel, discord.VoiceChannel):
            return
        try:
            creator = await self.runtime.temp_voice.creator(str(channel.id))
            if creator is not None:
                await self.runtime.temp_voice.remove_creator(creator.channel_id)
            room = await self.runtime.temp_voice.room(str(channel.id))
            if room is None or channel.id in self._room_deletions_in_flight:
                return
            self.cancel_deletion(channel.id)
            await self._prepare_audio_for_room_deletion(room, channel)
            await self.runtime.temp_voice.finish_room(
                room.channel_id,
                reason=TempVoiceRoomEndReason.MISSING,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "TempVC channel-deletion reconciliation failed guild=%s channel=%s",
                channel.guild.id,
                channel.id,
            )

    async def restore_tracked_rooms(self) -> None:
        """Reconcile only rooms in the durable BOT-owned registry after restart."""

        if self._restored:
            return
        await self.reconcile_tracked_rooms()
        self._restored = True
        self._reconciliation_task = asyncio.create_task(
            self._reconciliation_loop(),
            name="simajilord-tempvc-reconcile",
        )

    async def reconcile_tracked_rooms(self) -> None:
        """Repair missed voice events without discovering or deleting untracked channels."""

        for guild in self.bot.guilds:
            creators = await self.runtime.temp_voice.creators(str(guild.id))
            for creator in creators:
                channel = guild.get_channel(int(creator.channel_id))
                if isinstance(channel, discord.VoiceChannel):
                    continue
                await self.runtime.temp_voice.remove_creator(creator.channel_id)
                log.warning(
                    "Removed stale TempVC creator guild=%s channel=%s",
                    guild.id,
                    creator.channel_id,
                )
        for room in await self.runtime.temp_voice.rooms():
            room_guild = self.bot.get_guild(int(room.workspace_id))
            if room_guild is None:
                continue
            async with self._room_locks.hold(room.channel_id):
                current = await self.runtime.temp_voice.room(room.channel_id)
                if current is None:
                    continue
                channel = room_guild.get_channel(int(current.channel_id))
                if not isinstance(channel, discord.VoiceChannel):
                    self.cancel_deletion(int(current.channel_id))
                    await self.runtime.temp_voice.finish_room(
                        current.channel_id,
                        reason=TempVoiceRoomEndReason.MISSING,
                    )
                    continue
                humans = _human_voice_members(channel)
                if humans:
                    self.cancel_deletion(channel.id)
                    current = await self.runtime.temp_voice.mark_room_occupied(current.channel_id)
                    if all(str(item.id) != current.owner_id for item in humans):
                        await self._transfer_owner_without_interaction(
                            channel,
                            current,
                            humans[0],
                        )
                else:
                    marked = (
                        current
                        if current.empty_since is not None
                        else await self.runtime.temp_voice.mark_room_empty(current.channel_id)
                    )
                    self.schedule_deletion(marked)

    async def _reconciliation_loop(self) -> None:
        try:
            while not self.bot.is_closed():
                await asyncio.sleep(_TEMP_VOICE_RECONCILE_SECONDS)
                try:
                    await self.reconcile_tracked_rooms()
                except Exception:
                    log.exception("Periodic TempVC reconciliation failed")
        except asyncio.CancelledError:
            raise

    async def create_hub(
        self,
        member: discord.Member,
        *,
        request_id: str,
    ) -> TempVoiceCreator:
        _require_member_manage_channels(member)
        workspace_id = str(member.guild.id)
        async with self._guild_locks.hold(workspace_id):
            await self.runtime.temp_voice.ensure_config(workspace_id)
            category_name = _unique_channel_name(
                _TEMP_VOICE_CATEGORY_NAME,
                (category.name for category in member.guild.categories),
            )
            category = await member.guild.create_category(
                category_name,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} setup by {member}",
            )
            try:
                creator = await self._create_creator_channel(member, category)
            except BaseException:
                with suppress(discord.DiscordException):
                    if not category.channels:
                        await category.delete(reason=f"{_TEMP_VOICE_AUDIT_REASON} setup rollback")
                raise
        await self.record_event(
            "temp_voice.hub_created",
            workspace_id=workspace_id,
            actor_id=str(member.id),
            request_id=request_id,
            payload={
                "creator_channel_id": creator.channel_id,
                "category_id": creator.category_id,
            },
        )
        return creator

    async def create_creator_in_category(
        self,
        member: discord.Member,
        category: discord.CategoryChannel,
        *,
        request_id: str,
    ) -> TempVoiceCreator:
        _require_member_manage_channels(member)
        if category.guild.id != member.guild.id:
            raise UserError("temp_voice.category_missing")
        workspace_id = str(member.guild.id)
        async with self._guild_locks.hold(workspace_id):
            await self.runtime.temp_voice.ensure_config(workspace_id)
            creator = await self._create_creator_channel(member, category)
        await self.record_event(
            "temp_voice.creator_created",
            workspace_id=workspace_id,
            actor_id=str(member.id),
            request_id=request_id,
            payload={
                "creator_channel_id": creator.channel_id,
                "category_id": creator.category_id,
            },
        )
        return creator

    async def _create_creator_channel(
        self,
        member: discord.Member,
        category: discord.CategoryChannel,
    ) -> TempVoiceCreator:
        if len(category.channels) >= 50:
            raise UserError("temp_voice.category_full")
        _require_bot_channel_permissions(category, ("view_channel", "manage_channels"))
        name = _unique_channel_name(
            _TEMP_VOICE_CREATOR_NAME,
            (channel.name for channel in category.voice_channels),
        )
        channel = await member.guild.create_voice_channel(
            name,
            category=category,
            user_limit=0,
            reason=f"{_TEMP_VOICE_AUDIT_REASON} creator added by {member}",
        )
        try:
            return await self.runtime.temp_voice.add_creator(
                workspace_id=str(member.guild.id),
                channel_id=str(channel.id),
                category_id=str(category.id),
                permission_source_channel_id=str(channel.id),
            )
        except BaseException:
            with suppress(discord.DiscordException):
                await channel.delete(reason=f"{_TEMP_VOICE_AUDIT_REASON} creator rollback")
            raise

    async def create_room_from_creator(
        self,
        interaction: discord.Interaction,
        creator_channel_id: int,
    ) -> tuple[discord.VoiceChannel, bool]:
        member = interaction.user
        if not isinstance(member, discord.Member):
            raise UserError("workspace.required")
        return await self._create_or_recover_room_from_creator(
            member,
            creator_channel_id,
            request_id=str(interaction.id),
        )

    async def _create_or_recover_room_from_creator(
        self,
        member: discord.Member,
        creator_channel_id: int,
        *,
        request_id: str,
    ) -> tuple[discord.VoiceChannel, bool]:
        member_key = f"{member.guild.id}:{member.id}"
        async with self._member_locks.hold(member_key):
            current_voice = member.voice.channel if member.voice is not None else None
            if (
                not isinstance(current_voice, discord.VoiceChannel)
                or current_voice.id != creator_channel_id
            ):
                raise UserError("temp_voice.creator_join_required")
            creator = await self.runtime.temp_voice.creator(str(creator_channel_id))
            if creator is None or creator.workspace_id != str(member.guild.id):
                raise UserError("temp_voice.creator_missing")
            config = await self.runtime.temp_voice.config(str(member.guild.id))
            if config is None:
                raise UserError("temp_voice.not_configured")
            if not config.enabled:
                raise UserError("temp_voice.creation_paused")
            category = member.guild.get_channel(int(creator.category_id))
            if not isinstance(category, discord.CategoryChannel):
                raise UserError("temp_voice.category_missing")
            if len(category.channels) >= 50:
                raise UserError("temp_voice.category_full")
            _require_bot_channel_permissions(
                current_voice,
                ("view_channel", "connect", "move_members"),
            )
            _require_bot_channel_permissions(
                category,
                ("view_channel", "manage_channels", "move_members"),
            )
            recovered = await self._recover_owned_empty_room(member)
            if recovered is not None:
                try:
                    await member.move_to(
                        recovered,
                        reason=f"{_TEMP_VOICE_AUDIT_REASON} room recovery",
                    )
                except BaseException:
                    async with self._room_locks.hold(str(recovered.id)):
                        current = await self.runtime.temp_voice.room(str(recovered.id))
                        if current is not None:
                            if _human_voice_members(recovered):
                                await self.runtime.temp_voice.mark_room_occupied(current.channel_id)
                            else:
                                marked = await self.runtime.temp_voice.mark_room_empty(
                                    current.channel_id
                                )
                                self.schedule_deletion(marked)
                    raise
                async with self._room_locks.hold(str(recovered.id)):
                    current = await self.runtime.temp_voice.room(str(recovered.id))
                    if current is not None:
                        await self.runtime.temp_voice.mark_room_occupied(current.channel_id)
                await self.record_event(
                    "temp_voice.room_recovered",
                    workspace_id=str(member.guild.id),
                    actor_id=str(member.id),
                    request_id=request_id,
                    payload={"channel_id": str(recovered.id)},
                )
                return recovered, True
            created = await self._create_room_for_member(
                member,
                current_voice,
                creator,
                category,
                config,
            )
            return created, False

    async def _recover_owned_empty_room(
        self,
        member: discord.Member,
    ) -> discord.VoiceChannel | None:
        owned = reversed(
            await self.runtime.temp_voice.rooms_owned_by(
                workspace_id=str(member.guild.id),
                owner_id=str(member.id),
            )
        )
        empty_rooms: list[tuple[TempVoiceRoom, discord.VoiceChannel]] = []
        for room in owned:
            channel = member.guild.get_channel(int(room.channel_id))
            if not isinstance(channel, discord.VoiceChannel):
                await self.runtime.temp_voice.finish_room(
                    room.channel_id,
                    reason=TempVoiceRoomEndReason.MISSING,
                )
                continue
            if _human_voice_members(channel):
                raise UserError(
                    "temp_voice.room_already_owned",
                    channel_id=room.channel_id,
                )
            empty_rooms.append((room, channel))
        for room, channel in empty_rooms:
            async with self._room_locks.hold(room.channel_id):
                current = await self.runtime.temp_voice.room(room.channel_id)
                if current is None or _human_voice_members(channel):
                    continue
                self.cancel_deletion(channel.id)
            return channel
        return None

    async def _create_room_for_member(
        self,
        member: discord.Member,
        creator_channel: discord.VoiceChannel,
        creator: TempVoiceCreator,
        category: discord.CategoryChannel,
        config: TempVoiceConfig,
    ) -> discord.VoiceChannel:
        profile_before = await self.runtime.temp_voice.profile(
            workspace_id=str(member.guild.id),
            owner_id=str(member.id),
        )
        base_name = (
            profile_before.room_name
            if profile_before is not None
            else render_temp_voice_room_name(
                config.room_name_template,
                display_name=member.display_name,
                username=member.name,
            )
        )
        room_name = _unique_channel_name(
            base_name,
            (channel.name for channel in category.voice_channels),
        )
        user_limit = (
            profile_before.user_limit if profile_before is not None else creator_channel.user_limit
        )
        locked = profile_before.locked if profile_before is not None else False
        if creator.permission_source_channel_id is None:
            permission_source: discord.abc.GuildChannel = category
        else:
            configured_source = member.guild.get_channel(int(creator.permission_source_channel_id))
            if not isinstance(configured_source, discord.VoiceChannel):
                raise UserError("temp_voice.permission_source_missing")
            _require_bot_channel_permissions(configured_source, ("view_channel",))
            permission_source = configured_source
        overwrites = dict(permission_source.overwrites)
        owner_overwrite = overwrites.get(member, discord.PermissionOverwrite())
        owner_overwrite.update(view_channel=True, connect=True, send_messages=True)
        overwrites[member] = owner_overwrite
        bot_member = member.guild.me
        if bot_member is None:
            raise UserError("temp_voice.bot_member_missing")
        bot_overwrite = overwrites.get(bot_member, discord.PermissionOverwrite())
        bot_overwrite.update(
            view_channel=True,
            connect=True,
            send_messages=True,
            manage_channels=True,
            move_members=True,
        )
        overwrites[bot_member] = bot_overwrite
        everyone_overwrite = overwrites.get(
            member.guild.default_role,
            discord.PermissionOverwrite(),
        )
        base_everyone_connect = everyone_overwrite.connect
        if locked:
            everyone_overwrite.update(connect=False)
            overwrites[member.guild.default_role] = everyone_overwrite
        channel = await member.guild.create_voice_channel(
            room_name,
            category=category,
            user_limit=user_limit,
            overwrites=overwrites,
            reason=f"{_TEMP_VOICE_AUDIT_REASON} room for {member}",
        )
        registered = False
        try:
            await self.runtime.temp_voice.register_room(
                workspace_id=str(member.guild.id),
                channel_id=str(channel.id),
                creator_channel_id=creator.channel_id,
                owner_id=str(member.id),
                name=room_name,
                user_limit=user_limit,
                locked=locked,
                base_everyone_connect=base_everyone_connect,
            )
            registered = True
            await member.move_to(
                channel,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} join-to-create",
            )
        except BaseException:
            with suppress(discord.DiscordException):
                await channel.delete(reason=f"{_TEMP_VOICE_AUDIT_REASON} room rollback")
            if registered:
                await self.runtime.temp_voice.finish_room(
                    str(channel.id),
                    reason=TempVoiceRoomEndReason.MOVE_FAILED,
                    remember_for_recreation=False,
                    replacement_last_channel_id=(
                        profile_before.last_room_channel_id if profile_before is not None else None
                    ),
                )
            raise
        await self._remap_saved_voice_destinations(
            member.guild.id,
            previous_channel_id=(
                profile_before.last_room_channel_id if profile_before is not None else None
            ),
            replacement_channel_id=str(channel.id),
        )
        await self._send_room_welcome(channel, member, config)
        await self.record_event(
            "temp_voice.room_created",
            workspace_id=str(member.guild.id),
            actor_id=str(member.id),
            request_id=f"voice:{member.id}:{channel.id}",
            payload={
                "channel_id": str(channel.id),
                "creator_channel_id": creator.channel_id,
                "category_id": creator.category_id,
                "user_limit": user_limit,
                "locked": locked,
                "restored_profile": profile_before is not None,
            },
        )
        return channel

    async def _handle_room_join(
        self,
        member: discord.Member,
        channel: discord.VoiceChannel,
        room: TempVoiceRoom,
    ) -> None:
        async with self._room_locks.hold(room.channel_id):
            current = await self.runtime.temp_voice.room(room.channel_id)
            if current is None:
                return
            self.cancel_deletion(channel.id)
            current = await self.runtime.temp_voice.mark_room_occupied(room.channel_id)
            humans = _human_voice_members(channel)
            if humans and all(str(item.id) != current.owner_id for item in humans):
                await self._transfer_owner_without_interaction(channel, current, humans[0])

    async def _handle_room_departure(
        self,
        member: discord.Member,
        channel: discord.VoiceChannel,
        room: TempVoiceRoom,
    ) -> None:
        async with self._room_locks.hold(room.channel_id):
            current = await self.runtime.temp_voice.room(room.channel_id)
            if current is None:
                return
            humans = _human_voice_members(channel)
            if humans:
                self.cancel_deletion(channel.id)
                current = await self.runtime.temp_voice.mark_room_occupied(room.channel_id)
                if all(str(item.id) != current.owner_id for item in humans):
                    await self._transfer_owner_without_interaction(
                        channel,
                        current,
                        humans[0],
                    )
                return
            marked = await self.runtime.temp_voice.mark_room_empty(room.channel_id)
            self.schedule_deletion(marked)
            await self.record_event(
                "temp_voice.room_empty",
                workspace_id=marked.workspace_id,
                actor_id=str(member.id),
                request_id=f"voice:{member.id}:{channel.id}:empty",
                payload={"channel_id": marked.channel_id},
            )

    def schedule_deletion(self, room: TempVoiceRoom) -> None:
        if room.empty_since is None:
            return
        channel_id = int(room.channel_id)
        current = self._deletion_tasks.get(channel_id)
        if current is not None and not current.done():
            return
        self._deletion_tasks[channel_id] = asyncio.create_task(
            self._delete_empty_room(room.channel_id),
            name=f"simajilord-tempvc-delete-{room.channel_id}",
        )

    def cancel_deletion(self, channel_id: int) -> None:
        task = self._deletion_tasks.pop(channel_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def reschedule_empty_rooms(self, guild_id: int) -> None:
        rooms = await self.runtime.temp_voice.rooms(str(guild_id))
        for room in rooms:
            if room.empty_since is None:
                continue
            self.cancel_deletion(int(room.channel_id))
            self.schedule_deletion(room)

    async def _delete_empty_room(self, channel_id: str) -> None:
        numeric_channel_id = int(channel_id)
        try:
            while not self.bot.is_closed():
                room = await self.runtime.temp_voice.room(channel_id)
                if room is None or room.empty_since is None:
                    return
                config = await self.runtime.temp_voice.config(room.workspace_id)
                grace_seconds = (
                    config.empty_grace_seconds
                    if config is not None
                    else DEFAULT_TEMP_VOICE_GRACE_SECONDS
                )
                due_at = room.empty_since + timedelta(seconds=grace_seconds)
                delay = (due_at - datetime.now(UTC)).total_seconds()
                if delay > 0:
                    await asyncio.sleep(delay)
                    continue
                async with self._room_locks.hold(channel_id):
                    current = await self.runtime.temp_voice.room(channel_id)
                    if current is None or current.empty_since is None:
                        return
                    guild = self.bot.get_guild(int(current.workspace_id))
                    if guild is None:
                        return
                    channel = guild.get_channel(numeric_channel_id)
                    if not isinstance(channel, discord.VoiceChannel):
                        await self.runtime.temp_voice.finish_room(
                            channel_id,
                            reason=TempVoiceRoomEndReason.MISSING,
                        )
                        return
                    if _human_voice_members(channel):
                        await self.runtime.temp_voice.mark_room_occupied(channel_id)
                        return
                    try:
                        await self._prepare_audio_for_room_deletion(current, channel)
                        if _human_voice_members(channel):
                            await self.runtime.temp_voice.mark_room_occupied(channel_id)
                            return
                        self._room_deletions_in_flight.add(numeric_channel_id)
                        await channel.delete(reason=f"{_TEMP_VOICE_AUDIT_REASON} empty-room expiry")
                    except (discord.Forbidden, discord.HTTPException):
                        self._room_deletions_in_flight.discard(numeric_channel_id)
                        log.warning(
                            "TempVC deletion will retry guild=%s channel=%s",
                            current.workspace_id,
                            channel_id,
                            exc_info=True,
                        )
                    else:
                        await self.runtime.temp_voice.finish_room(
                            channel_id,
                            reason=TempVoiceRoomEndReason.EMPTY,
                        )
                        await self.record_event(
                            "temp_voice.room_deleted",
                            workspace_id=current.workspace_id,
                            actor_id=current.owner_id,
                            request_id=f"tempvc-delete:{channel_id}",
                            payload={
                                "channel_id": channel_id,
                                "empty_since": current.empty_since.isoformat(),
                                "audio_queue_preserved": True,
                            },
                        )
                        return
                await asyncio.sleep(_TEMP_VOICE_DELETE_RETRY_SECONDS)
        except asyncio.CancelledError:
            raise
        finally:
            self._room_deletions_in_flight.discard(numeric_channel_id)
            if self._deletion_tasks.get(numeric_channel_id) is asyncio.current_task():
                self._deletion_tasks.pop(numeric_channel_id, None)

    async def _prepare_audio_for_room_deletion(
        self,
        room: TempVoiceRoom,
        channel: discord.VoiceChannel,
    ) -> None:
        session = self.runtime.audio.find(room.workspace_id)
        if (
            session is not None
            and session.destination_id == room.channel_id
            and session.output.connected
        ):
            await session.suspend()
            log.info(
                "Suspended audio before TempVC deletion guild=%s channel=%s auto_leave=%s",
                room.workspace_id,
                room.channel_id,
                session.auto_leave,
            )
        dashboard = getattr(self.bot, "_simajilord_music_dashboard", None)
        forget_channel = getattr(dashboard, "forget_channel", None)
        if callable(forget_channel):
            await forget_channel(room.workspace_id, channel.id)

    async def _remap_saved_voice_destinations(
        self,
        guild_id: int,
        *,
        previous_channel_id: str | None,
        replacement_channel_id: str,
    ) -> None:
        if previous_channel_id is None or previous_channel_id == replacement_channel_id:
            return
        workspace_id = str(guild_id)
        session = self.runtime.audio.find(workspace_id)
        audio_remapped = False
        if session is not None:
            audio_remapped = await session.remap_suspended_destination(
                expected_destination_id=previous_channel_id,
                replacement_destination_id=replacement_channel_id,
            )
        route = self.runtime.read_aloud.get(workspace_id)
        read_aloud_remapped = False
        if route is not None and route.audio_destination_id == previous_channel_id:
            await self.runtime.read_aloud.configure(
                replace(route, audio_destination_id=replacement_channel_id)
            )
            read_aloud_remapped = True
        if audio_remapped or read_aloud_remapped:
            await self.record_event(
                "temp_voice.destinations_remapped",
                workspace_id=workspace_id,
                actor_id=None,
                request_id=f"tempvc-remap:{previous_channel_id}:{replacement_channel_id}",
                payload={
                    "previous_channel_id": previous_channel_id,
                    "replacement_channel_id": replacement_channel_id,
                    "audio_remapped": audio_remapped,
                    "read_aloud_remapped": read_aloud_remapped,
                    "auto_connected": False,
                },
            )

    async def rename_room(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        requested_name: str,
    ) -> TempVoiceRoom:
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        normalized = normalize_temp_voice_room_name(requested_name)
        name = _unique_channel_name(
            normalized,
            (
                candidate.name
                for candidate in channel.category.voice_channels
                if candidate.id != channel.id
            )
            if channel.category is not None
            else (),
        )
        async with self._room_locks.hold(room.channel_id):
            await channel.edit(
                name=name,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} rename by {member}",
            )
            try:
                updated = await self.runtime.temp_voice.rename_room(room.channel_id, name)
            except BaseException:
                with suppress(discord.DiscordException):
                    await channel.edit(
                        name=room.name,
                        reason=f"{_TEMP_VOICE_AUDIT_REASON} rename rollback",
                    )
                raise
        return updated

    async def set_room_user_limit(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        user_limit: int,
    ) -> TempVoiceRoom:
        validate_temp_voice_user_limit(user_limit)
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        async with self._room_locks.hold(room.channel_id):
            await channel.edit(
                user_limit=user_limit,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} limit by {member}",
            )
            try:
                updated = await self.runtime.temp_voice.set_room_user_limit(
                    room.channel_id,
                    user_limit,
                )
            except BaseException:
                with suppress(discord.DiscordException):
                    await channel.edit(
                        user_limit=room.user_limit,
                        reason=f"{_TEMP_VOICE_AUDIT_REASON} limit rollback",
                    )
                raise
        return updated

    async def set_room_lock(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        locked: bool,
    ) -> TempVoiceRoom:
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        target = channel.guild.default_role
        original = channel.overwrites_for(target)
        changed = discord.PermissionOverwrite.from_pair(*original.pair())
        changed.update(
            connect=False if locked else room.base_everyone_connect,
        )
        async with self._room_locks.hold(room.channel_id):
            await channel.set_permissions(
                target,
                overwrite=None if changed.is_empty() else changed,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} access by {member}",
            )
            try:
                updated = await self.runtime.temp_voice.set_room_locked(
                    room.channel_id,
                    locked,
                )
            except BaseException:
                with suppress(discord.DiscordException):
                    await channel.set_permissions(
                        target,
                        overwrite=None if original.is_empty() else original,
                        reason=f"{_TEMP_VOICE_AUDIT_REASON} access rollback",
                    )
                raise
        return updated

    async def invite_member(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        target_id: int,
    ) -> discord.Member:
        _room, channel, member = await self.require_room_controller(interaction, channel_id)
        target = channel.guild.get_member(target_id)
        if target is None or target.bot:
            raise UserError("temp_voice.member_invalid")
        overwrite = channel.overwrites_for(target)
        overwrite.update(view_channel=True, connect=True)
        await channel.set_permissions(
            target,
            overwrite=overwrite,
            reason=f"{_TEMP_VOICE_AUDIT_REASON} invite by {member}",
        )
        return target

    async def remove_member(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        target_id: int,
        *,
        ban: bool,
    ) -> discord.Member:
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        target = channel.guild.get_member(target_id)
        if target is None or target.bot:
            raise UserError("temp_voice.member_invalid")
        if str(target.id) == room.owner_id:
            raise UserError("temp_voice.owner_removal_forbidden")
        if target not in channel.members:
            raise UserError("temp_voice.member_not_present")
        if ban:
            overwrite = channel.overwrites_for(target)
            overwrite.update(connect=False)
            await channel.set_permissions(
                target,
                overwrite=overwrite,
                reason=f"{_TEMP_VOICE_AUDIT_REASON} ban by {member}",
            )
        await target.move_to(
            None,
            reason=(
                f"{_TEMP_VOICE_AUDIT_REASON} ban by {member}"
                if ban
                else f"{_TEMP_VOICE_AUDIT_REASON} remove by {member}"
            ),
        )
        return target

    async def transfer_owner(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        target_id: int,
    ) -> TempVoiceRoom:
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        target = channel.guild.get_member(target_id)
        if target is None or target.bot:
            raise UserError("temp_voice.member_invalid")
        if target not in channel.members:
            raise UserError("temp_voice.member_not_present")
        if str(target.id) == room.owner_id:
            return room
        async with self._room_locks.hold(room.channel_id):
            await _allow_room_owner(channel, target, reason=f"ownership transfer by {member}")
            return await self.runtime.temp_voice.transfer_room(
                room.channel_id,
                str(target.id),
            )

    async def _transfer_owner_without_interaction(
        self,
        channel: discord.VoiceChannel,
        room: TempVoiceRoom,
        target: discord.Member,
    ) -> TempVoiceRoom:
        await _allow_room_owner(channel, target, reason="automatic owner transfer")
        updated = await self.runtime.temp_voice.transfer_room(
            room.channel_id,
            str(target.id),
        )
        await self.record_event(
            "temp_voice.owner_transferred",
            workspace_id=room.workspace_id,
            actor_id=str(target.id),
            request_id=f"tempvc-owner:{room.channel_id}:{target.id}",
            payload={
                "channel_id": room.channel_id,
                "previous_owner_id": room.owner_id,
                "owner_id": str(target.id),
                "automatic": True,
            },
        )
        return updated

    async def keep_room_permanent(
        self,
        interaction: discord.Interaction,
        channel_id: int,
    ) -> discord.VoiceChannel:
        room, channel, member = await self.require_room_controller(
            interaction,
            channel_id,
            manager_required=True,
        )
        async with self._room_locks.hold(room.channel_id):
            self.cancel_deletion(channel.id)
            await self.runtime.temp_voice.finish_room(
                room.channel_id,
                reason=TempVoiceRoomEndReason.KEPT_PERMANENT,
                remember_for_recreation=False,
            )
        await self.record_event(
            "temp_voice.room_kept",
            workspace_id=room.workspace_id,
            actor_id=str(member.id),
            request_id=str(interaction.id),
            payload={"channel_id": room.channel_id},
        )
        return channel

    async def require_room_controller(
        self,
        interaction: discord.Interaction,
        channel_id: int,
        *,
        manager_required: bool = False,
    ) -> tuple[TempVoiceRoom, discord.VoiceChannel, discord.Member]:
        member = interaction.user
        if not isinstance(member, discord.Member):
            raise UserError("workspace.required")
        room = await self.runtime.temp_voice.room(str(channel_id))
        channel = member.guild.get_channel(channel_id)
        if room is None or not isinstance(channel, discord.VoiceChannel):
            raise UserError("temp_voice.room_missing")
        current_voice = member.voice.channel if member.voice is not None else None
        if current_voice is None or current_voice.id != channel.id:
            raise UserError("temp_voice.room_join_required")
        is_manager = _can_manage_temp_voice(member)
        if manager_required and not is_manager:
            raise UserError("temp_voice.manager_required")
        if room.owner_id != str(member.id) and not is_manager:
            raise UserError("temp_voice.owner_required")
        _require_bot_channel_permissions(
            channel,
            ("view_channel", "manage_channels", "move_members"),
        )
        return room, channel, member

    async def set_creator_permission_source(
        self,
        interaction: discord.Interaction,
        creator_channel_id: int,
        permission_source_channel_id: int | None,
    ) -> TempVoiceCreator:
        member = _require_temp_voice_manager(interaction)
        creator = await self.runtime.temp_voice.creator(str(creator_channel_id))
        if creator is None or creator.workspace_id != str(member.guild.id):
            raise UserError("temp_voice.creator_missing")
        if permission_source_channel_id is not None:
            source = member.guild.get_channel(permission_source_channel_id)
            if not isinstance(source, discord.VoiceChannel):
                raise UserError("temp_voice.permission_source_missing")
            _require_bot_channel_permissions(source, ("view_channel",))
        updated = await self.runtime.temp_voice.set_creator_permission_source(
            creator.channel_id,
            (
                str(permission_source_channel_id)
                if permission_source_channel_id is not None
                else None
            ),
        )
        await self.record_event(
            "temp_voice.permission_source_updated",
            workspace_id=creator.workspace_id,
            actor_id=str(member.id),
            request_id=str(interaction.id),
            payload={
                "creator_channel_id": creator.channel_id,
                "permission_source_channel_id": updated.permission_source_channel_id,
            },
        )
        return updated

    async def permission_copy_panel(
        self,
        member: discord.Member,
        selected_creator_channel_id: int | None = None,
    ) -> tuple[discord.Embed, TempVoicePermissionCopyView]:
        _require_member_manage_channels(member)
        creators = await self.runtime.temp_voice.creators(str(member.guild.id))
        if not creators:
            raise UserError("temp_voice.creator_missing")
        selected = next(
            (
                creator
                for creator in creators
                if int(creator.channel_id) == selected_creator_channel_id
            ),
            creators[0],
        )
        source = (
            member.guild.get_channel(int(selected.permission_source_channel_id))
            if selected.permission_source_channel_id is not None
            else None
        )
        if selected.permission_source_channel_id is None:
            source_label = f"Category <#{selected.category_id}> (live synced defaults)"
            source_missing = False
        elif isinstance(source, discord.VoiceChannel):
            source_label = source.mention
            source_missing = False
        else:
            source_label = (
                f"Missing voice channel `{selected.permission_source_channel_id}`; "
                "room creation is blocked until this is changed."
            )
            source_missing = True
        embed = command_embed(
            "TempVC permission copy",
            description=(
                "Choose a creator lobby, then choose an existing voice channel. New rooms "
                "copy every role and member overwrite from that source. Name, bitrate, and "
                "user limit are not copied."
            ),
            fields=(
                EmbedField("Creator lobby", f"<#{selected.channel_id}>"),
                EmbedField("Destination category", f"<#{selected.category_id}>"),
                EmbedField("Current permission source", source_label, inline=False),
                EmbedField(
                    "Safety overrides",
                    (
                        "METEOBOT keeps only the permissions it needs to move users and clean "
                        "up tracked rooms. The room owner is always allowed to view and join."
                    ),
                    inline=False,
                ),
            ),
            tone=EmbedTone.WARNING if source_missing else EmbedTone.SUCCESS,
        )
        return (
            embed,
            TempVoicePermissionCopyView(
                self,
                requester_id=member.id,
                guild=member.guild,
                creators=creators,
                creator_channel_id=int(selected.channel_id),
            ),
        )

    async def edit_permission_copy_panel(
        self,
        interaction: discord.Interaction,
        guild_id: int,
        creator_channel_id: int,
    ) -> None:
        member = interaction.user
        if not isinstance(member, discord.Member) or member.guild.id != guild_id:
            raise UserError("workspace.required")
        embed, view = await self.permission_copy_panel(member, creator_channel_id)
        await interaction.edit_original_response(embed=embed, view=view)

    def creation_panel(
        self,
        member: discord.Member,
        creator_channel: discord.VoiceChannel,
        creator: TempVoiceCreator,
        config: TempVoiceConfig,
    ) -> tuple[discord.Embed, TempVoiceCreateView]:
        category = member.guild.get_channel(int(creator.category_id))
        source = (
            member.guild.get_channel(int(creator.permission_source_channel_id))
            if creator.permission_source_channel_id is not None
            else category
        )
        if isinstance(source, (discord.VoiceChannel, discord.CategoryChannel)):
            permission_label = source.mention
            source_valid = True
        else:
            permission_label = (
                "The configured permission source is missing; ask an administrator to fix it."
            )
            source_valid = False
        creation_available = (
            config.enabled
            and isinstance(category, discord.CategoryChannel)
            and source_valid
            and len(category.channels) < 50
        )
        fields = (
            EmbedField("Permission source", permission_label, inline=False),
            EmbedField(
                "Recovery",
                (
                    f"An empty room remains recoverable for {config.empty_grace_seconds // 60} "
                    "minutes. Creating again during that window returns you to it."
                ),
                inline=False,
            ),
            EmbedField(
                "Automatic creation",
                (
                    "Joining this lobby normally creates and moves you immediately. If an event "
                    "was missed or a temporary failure left you here, this private button retries "
                    "the same validated operation."
                ),
                inline=False,
            ),
        )
        return (
            command_embed(
                "TempVC creation retry",
                description=(
                    f"You are still in {creator_channel.mention}. METEOBOT normally creates a "
                    "room on join; press the retry button to create and move there."
                    if creation_available
                    else "Room creation is unavailable until an administrator fixes or resumes "
                    "this creator lobby. No channel was created."
                ),
                fields=fields,
                tone=EmbedTone.SUCCESS if creation_available else EmbedTone.WARNING,
            ),
            TempVoiceCreateView(
                self,
                requester_id=member.id,
                guild_id=member.guild.id,
                creator_channel_id=creator_channel.id,
                creation_enabled=creation_available,
                can_manage=_can_manage_temp_voice(member),
            ),
        )

    def admin_panel(
        self,
        member: discord.Member,
        config: TempVoiceConfig | None,
        creators: tuple[TempVoiceCreator, ...],
    ) -> tuple[discord.Embed, TempVoiceAdminView]:
        creator_lines_list: list[str] = []
        for creator in creators[:8]:
            permission_source = (
                f"<#{creator.permission_source_channel_id}>"
                if creator.permission_source_channel_id is not None
                else "category permissions"
            )
            creator_lines_list.append(
                f"• <#{creator.channel_id}> → <#{creator.category_id}>; copy {permission_source}"
            )
        if len(creators) > 8:
            creator_lines_list.append(f"…and {len(creators) - 8} more creator lobbies")
        creator_lines = (
            "\n".join(creator_lines_list) if creator_lines_list else "No creator channels yet."
        )
        description = (
            "Create a permanent lobby first. Joining it creates and moves the member into a "
            "private room automatically; `/tempvc` in the lobby remains a requester-only "
            "retry panel. Pausing only stops new rooms and never removes lobbies, settings, "
            "or active rooms."
        )
        fields: tuple[EmbedField, ...] = (
            EmbedField("Creator channels", creator_lines, inline=False),
            EmbedField(
                "Creation",
                (
                    "On"
                    if config is not None and config.enabled
                    else "Paused"
                    if config is not None
                    else "Not configured"
                ),
            ),
            EmbedField("Registered creators", str(len(creators))),
        )
        if config is not None:
            fields = (
                *fields,
                EmbedField("Room template", f"`{config.room_name_template}`"),
                EmbedField(
                    "Empty-room recovery",
                    f"{config.empty_grace_seconds // 60} minutes",
                ),
                EmbedField(
                    "Safety",
                    (
                        "Only BOT-tracked rooms can be deleted. Audio waits safely, queues stay "
                        "saved, and room owners retain name, limit, and lock preferences. "
                        "Administrators can keep an important room permanently."
                    ),
                    inline=False,
                ),
            )
        view = TempVoiceAdminView(
            self,
            requester_id=member.id,
            guild_id=member.guild.id,
            config=config,
            creator_count=len(creators),
        )
        return command_embed("TempVC setup", description=description, fields=fields), view

    def room_panel(
        self,
        member: discord.Member,
        channel: discord.VoiceChannel,
        room: TempVoiceRoom,
        config: TempVoiceConfig,
    ) -> tuple[discord.Embed, TempVoiceRoomControlView]:
        can_keep = _can_manage_temp_voice(member)
        return (
            command_embed(
                "TempVC controls",
                description=(
                    f"Manage {channel.mention}. Empty rooms remain recoverable for "
                    f"**{config.empty_grace_seconds // 60} minutes**. Returning cancels "
                    "deletion; if it expires, your room settings return on the next creation."
                ),
                fields=(
                    EmbedField("Owner", f"<@{room.owner_id}>"),
                    EmbedField("Access", "Locked" if room.locked else "Open"),
                    EmbedField("User limit", _user_limit_label(room.user_limit)),
                    EmbedField(
                        "Audio",
                        (
                            "Normal auto-leave waits 10 seconds and preserves the queue. "
                            "TempVC deletion also leaves audio in explicit-resume standby."
                        ),
                        inline=False,
                    ),
                    *(
                        ()
                        if can_keep
                        else (
                            EmbedField(
                                "Keep this room",
                                "A Manage Channels administrator can make it permanent.",
                                inline=False,
                            ),
                        )
                    ),
                ),
            ),
            TempVoiceRoomControlView(
                self,
                requester_id=member.id,
                channel_id=channel.id,
                room=room,
                can_keep_permanent=can_keep,
            ),
        )

    async def edit_admin_panel(
        self,
        interaction: discord.Interaction,
        guild_id: int,
    ) -> None:
        member = interaction.user
        if not isinstance(member, discord.Member) or member.guild.id != guild_id:
            raise UserError("workspace.required")
        config = await self.runtime.temp_voice.config(str(guild_id))
        creators = await self.runtime.temp_voice.creators(str(guild_id))
        embed, view = self.admin_panel(member, config, creators)
        await interaction.edit_original_response(embed=embed, view=view)

    async def edit_room_panel(
        self,
        interaction: discord.Interaction,
        channel_id: int,
    ) -> None:
        room, channel, member = await self.require_room_controller(interaction, channel_id)
        config = await self.runtime.temp_voice.config(str(member.guild.id))
        if config is None:
            raise UserError("temp_voice.not_configured")
        embed, view = self.room_panel(member, channel, room, config)
        await interaction.edit_original_response(embed=embed, view=view)

    async def _send_room_welcome(
        self,
        channel: discord.VoiceChannel,
        owner: discord.Member,
        config: TempVoiceConfig,
    ) -> None:
        with suppress(discord.DiscordException):
            await channel.send(
                embed=command_embed(
                    "Temporary room ready",
                    description=(
                        f"Owned by <@{owner.id}>. Use `/tempvc` in this room to rename, "
                        "lock, invite, remove members, set a limit, or transfer ownership."
                    ),
                    fields=(
                        EmbedField(
                            "Recovery",
                            (
                                f"The room waits {config.empty_grace_seconds // 60} minutes "
                                "after everyone leaves. Rejoin to cancel deletion. Name, limit, "
                                "and lock settings return even after deletion."
                            ),
                            inline=False,
                        ),
                        EmbedField(
                            "Audio",
                            (
                                "METEOBOT auto-leave preserves the music queue. Recreated room "
                                "destinations are restored in standby and never auto-play."
                            ),
                            inline=False,
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def record_event(
        self,
        kind: str,
        *,
        workspace_id: str,
        actor_id: str | None,
        request_id: str,
        payload: dict[str, object],
    ) -> None:
        try:
            await self.runtime.journal.append(
                kind=kind,
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport="discord",
                request_id=request_id,
                payload=payload,
            )
        except Exception:
            log.debug("Could not record TempVC event kind=%s", kind, exc_info=True)


async def _allow_room_owner(
    channel: discord.VoiceChannel,
    member: discord.Member,
    *,
    reason: str,
) -> None:
    overwrite = channel.overwrites_for(member)
    overwrite.update(view_channel=True, connect=True)
    await channel.set_permissions(
        member,
        overwrite=overwrite,
        reason=f"{_TEMP_VOICE_AUDIT_REASON} {reason}",
    )


def _require_temp_voice_manager(interaction: discord.Interaction) -> discord.Member:
    member = interaction.user
    if not isinstance(member, discord.Member):
        raise UserError("workspace.required")
    _require_member_manage_channels(member)
    return member


def _require_member_manage_channels(member: discord.Member) -> None:
    if not _can_manage_temp_voice(member):
        raise UserError("temp_voice.manager_required")


def _can_manage_temp_voice(member: discord.Member) -> bool:
    return permission_enabled(
        member.guild_permissions,
        "administrator",
    ) or permission_enabled(member.guild_permissions, "manage_channels")


def _require_bot_channel_permissions(
    channel: discord.abc.GuildChannel,
    required: tuple[str, ...],
) -> None:
    bot_member = channel.guild.me
    if bot_member is None:
        raise UserError("temp_voice.bot_member_missing")
    permissions = channel.permissions_for(bot_member)
    missing = tuple(
        permission for permission in required if not permission_enabled(permissions, permission)
    )
    if missing:
        raise UserError(
            "temp_voice.bot_permissions_missing",
            permissions=", ".join(missing),
        )


def _human_voice_members(channel: discord.VoiceChannel) -> tuple[discord.Member, ...]:
    return tuple(
        sorted(
            (member for member in channel.members if not member.bot),
            key=lambda member: member.id,
        )
    )


def _unique_channel_name(base_name: str, existing_names: Iterable[str]) -> str:
    names = {name.casefold() for name in existing_names}
    if base_name.casefold() not in names:
        return base_name
    for index in range(2, 10_000):
        suffix = f" · {index}"
        candidate = f"{base_name[: 100 - len(suffix)].rstrip()}{suffix}"
        if candidate.casefold() not in names:
            return candidate
    raise UserError("temp_voice.name_exhausted")


def _user_limit_label(user_limit: int) -> str:
    return "Unlimited" if user_limit == 0 else str(user_limit)


async def _send_temp_voice_error(
    interaction: discord.Interaction,
    error: Exception,
) -> None:
    from .cogs import send_error

    await send_error(interaction, error)
