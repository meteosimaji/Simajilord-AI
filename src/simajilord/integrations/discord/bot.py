"""Discord transport composition and command synchronization."""

from __future__ import annotations

import logging
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.config import CommandScope
from simajilord.runtime import SimajilordRuntime

from .capabilities import build_discord_endpoints
from .cogs import setup_cogs

log = logging.getLogger(__name__)


class SimajilordDiscordBot(commands.Bot):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.messages = True
        intents.message_content = True
        super().__init__(
            command_prefix=commands.when_mentioned_or(runtime.settings.command_prefix),
            intents=intents,
            help_command=None,
            application_id=runtime.settings.application_id,
        )
        self.runtime = runtime
        self._command_templates: tuple[
            app_commands.Command[Any, ..., Any] | app_commands.Group | app_commands.ContextMenu,
            ...,
        ] = ()
        self._commands_synchronized = False

    async def setup_hook(self) -> None:
        for item in build_discord_endpoints(self, self.runtime):
            self.runtime.registry.register(item)
        await setup_cogs(self, self.runtime)
        self._command_templates = tuple(self.tree.get_commands())

    async def on_interaction(self, interaction: discord.Interaction) -> None:
        if interaction.type is discord.InteractionType.application_command:
            data: object = interaction.data
            command_name = (
                str(data.get("name", "unknown"))
                if isinstance(data, dict)
                else "unknown"
            )
            await self.runtime.journal.append(
                kind="discord.command.received",
                actor_id=str(interaction.user.id),
                workspace_id=str(interaction.guild_id) if interaction.guild_id else None,
                transport="discord",
                request_id=str(interaction.id),
                payload={
                    "command": command_name,
                    "channel_id": str(interaction.channel_id) if interaction.channel_id else None,
                },
            )

    async def on_command(self, context: commands.Context[commands.Bot]) -> None:
        await self.runtime.journal.append(
            kind="discord.prefix_command.received",
            actor_id=str(context.author.id),
            workspace_id=str(context.guild.id) if context.guild else None,
            transport="discord",
            request_id=str(context.message.id),
            payload={
                "command": context.command.qualified_name if context.command else "unknown",
                "channel_id": str(context.channel.id),
            },
        )

    async def on_ready(self) -> None:
        if self.user is None:
            return
        log.info(
            "Connected as %s (%s) in %s guild(s)",
            self.user,
            self.user.id,
            len(self.guilds),
        )
        if self._commands_synchronized:
            return
        if self.runtime.settings.command_scope is CommandScope.GUILD:
            self._restore_global_templates()
            for guild in self.guilds:
                self.tree.clear_commands(guild=guild)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                log.info("Synchronized %s commands to guild %s", len(synced), guild.id)
            self.tree.clear_commands(guild=None)
            await self.tree.sync()
            self._restore_global_templates()
            log.info("Removed stale global commands while using guild command scope")
        else:
            synced = await self.tree.sync()
            log.info("Synchronized %s global commands", len(synced))
        self._commands_synchronized = True

    def _restore_global_templates(self) -> None:
        self.tree.clear_commands(guild=None)
        for command in self._command_templates:
            self.tree.add_command(command, override=True)

    async def close(self) -> None:
        await self.runtime.close()
        await super().close()
