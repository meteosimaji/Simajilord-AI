"""Discord transport composition and command synchronization."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.activity import ActivityServer
from simajilord.config import CommandScope
from simajilord.domain.image import ImageGenerationJob, ImageJobStatus
from simajilord.runtime import SimajilordRuntime

from .application_emojis import ApplicationEmojiCatalog
from .audio import DiscordAudioOutput, verify_ffmpeg_opus
from .capabilities import DiscordMessageChannel, build_discord_endpoints
from .cogs import error_message, handle_interaction_error, setup_cogs
from .presenter import EmbedTone, command_embed

log = logging.getLogger(__name__)


class SimajilordCommandTree(app_commands.CommandTree[commands.Bot]):
    """Global safety net for application commands not handled closer to source."""

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        await handle_interaction_error(interaction, error)


class SimajilordDiscordBot(commands.Bot):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.messages = True
        intents.message_content = True
        super().__init__(
            # Mentions are agent events. Prefix commands remain an explicit direct API path.
            command_prefix=runtime.settings.command_prefix,
            intents=intents,
            help_command=None,
            application_id=runtime.settings.application_id,
            tree_cls=SimajilordCommandTree,
        )
        self.runtime = runtime
        self.activity_server = ActivityServer(self, runtime)
        self.application_emojis = ApplicationEmojiCatalog.from_settings(
            runtime.settings
        )
        self._command_templates: tuple[
            app_commands.Command[Any, ..., Any] | app_commands.Group | app_commands.ContextMenu,
            ...,
        ] = ()
        self._commands_synchronized = False
        self._audio_restored = False
        self._command_sync_lock = asyncio.Lock()

    async def setup_hook(self) -> None:
        await verify_ffmpeg_opus()
        await self.application_emojis.refresh(self)
        for item in build_discord_endpoints(self, self.runtime):
            self.runtime.registry.register(item)
        await setup_cogs(self, self.runtime)
        await self.activity_server.start()
        report = await self.runtime.maintenance.run()
        log.info(
            "Data maintenance complete: used=%s limit=%s removed=%s over_capacity=%s",
            report.storage_used_bytes,
            report.storage_limit_bytes,
            report.orphan_cleanup_removed,
            report.over_capacity,
        )
        await self.runtime.image.start(self._deliver_image_job)
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

    async def on_command_error(
        self,
        context: commands.Context[Any],
        error: commands.CommandError,
    ) -> None:
        await context.send(
            embed=command_embed(
                "Could not complete the request",
                description=error_message(
                    error,
                    request_id=str(context.message.id),
                ),
                tone=EmbedTone.ERROR,
            ),
            allowed_mentions=discord.AllowedMentions.none(),
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
        if not self._audio_restored:
            await self._restore_audio_sessions()
            await self._prepare_read_aloud_presence()
            self._audio_restored = True
        dashboard = getattr(self, "_simajilord_music_dashboard", None)
        prune_stale = getattr(dashboard, "prune_stale_records", None)
        if callable(prune_stale):
            removed = await prune_stale(
                frozenset(str(guild.id) for guild in self.guilds)
            )
            if removed:
                log.info("Removed %s stale music dashboard record(s)", removed)
        await self._synchronize_initial_commands()

    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Publish private-development commands immediately in a newly joined guild."""

        if (
            self.runtime.settings.command_scope is not CommandScope.GUILD
            or not self._command_templates
        ):
            return
        async with self._command_sync_lock:
            self._restore_global_templates()
            self.tree.clear_commands(guild=guild)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            self.tree.clear_commands(guild=None)
            await self.tree.sync()
            self._restore_global_templates()
        log.info(
            "Synchronized %s commands to newly joined guild %s",
            len(synced),
            guild.id,
        )

    async def _synchronize_initial_commands(self) -> None:
        if self._commands_synchronized:
            return
        async with self._command_sync_lock:
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

    async def _restore_audio_sessions(self) -> None:
        """Restore durable state without turning process startup into a voice action."""

        sessions = self.runtime.audio.restore(
            lambda workspace_id: DiscordAudioOutput(self, int(workspace_id))
        )
        for session in sessions:
            guild = self.get_guild(int(session.workspace_id))
            if guild is None:
                continue
            log.info(
                "Audio session %s restored in standby; an explicit Start, play, "
                "join, or approved agent action is required",
                session.workspace_id,
            )

    async def _prepare_read_aloud_presence(self) -> None:
        """Keep persisted read-aloud routes passive until an explicit voice action."""

        for guild in self.guilds:
            workspace_id = str(guild.id)
            route = self.runtime.read_aloud.get(workspace_id)
            if route is None:
                continue
            channel = guild.get_channel(int(route.audio_destination_id))
            if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                log.warning(
                    "Cannot prepare read-aloud route %s: destination %s no longer exists",
                    workspace_id,
                    route.audio_destination_id,
                )
                continue
            log.info(
                "Read-aloud route %s restored in standby for voice channel %s",
                workspace_id,
                route.audio_destination_id,
            )

    def _restore_global_templates(self) -> None:
        self.tree.clear_commands(guild=None)
        for command in self._command_templates:
            self.tree.add_command(command, override=True)

    async def _deliver_image_job(self, job: ImageGenerationJob) -> None:
        """Publish durable job state without consuming another model turn."""

        if job.delivered:
            return
        try:
            channel_id = int(job.delivery_target_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Image job {job.job_id} has an invalid delivery target"
            ) from exc
        channel = self.get_channel(channel_id)
        if not isinstance(
            channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            raise RuntimeError(
                f"Image job {job.job_id} delivery channel is unavailable"
            )

        progress_message: discord.Message | None = None
        if job.delivery_message_id is not None:
            try:
                progress_message = await channel.fetch_message(
                    int(job.delivery_message_id)
                )
            except (ValueError, discord.DiscordException):
                progress_message = None

        if job.status in {ImageJobStatus.RUNNING, ImageJobStatus.QUEUED}:
            embed = _image_progress_embed(job)
            if progress_message is None:
                progress_message = await _send_image_progress(channel, job, embed)
                await self.runtime.image.set_delivery_message(
                    job.job_id,
                    str(progress_message.id),
                )
            else:
                await progress_message.edit(embed=embed)
            return

        if job.status is ImageJobStatus.COMPLETED:
            if progress_message is not None:
                await progress_message.edit(embed=_image_progress_embed(job))
            if job.output_path is None or not job.output_path.is_file():
                raise RuntimeError(f"Completed image job has no output: {job.job_id}")
            filename = f"simajilord-{job.job_id[:8]}.png"
            embed = _image_result_embed(job, filename=filename)
            await channel.send(
                embed=embed,
                file=discord.File(job.output_path, filename=filename),
                allowed_mentions=discord.AllowedMentions.none(),
            )
            await self.runtime.image.mark_delivered(job.job_id)
            return

        if job.status is ImageJobStatus.FAILED:
            embed = _image_progress_embed(job)
            if progress_message is None:
                await _send_image_progress(channel, job, embed)
            else:
                await progress_message.edit(embed=embed)
            await self.runtime.image.mark_delivered(job.job_id)

    async def close(self) -> None:
        await self.activity_server.close()
        await self.runtime.close()
        await super().close()


async def _send_image_progress(
    channel: DiscordMessageChannel,
    job: ImageGenerationJob,
    embed: discord.Embed,
) -> discord.Message:
    if job.reply_to_message_id is not None:
        try:
            source = await channel.fetch_message(int(job.reply_to_message_id))
        except (ValueError, discord.DiscordException):
            source = None
        if source is not None:
            return await source.reply(
                embed=embed,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
            )
    return await channel.send(
        embed=embed,
        allowed_mentions=discord.AllowedMentions.none(),
    )


def _image_progress_embed(job: ImageGenerationJob) -> discord.Embed:
    if job.status is ImageJobStatus.FAILED:
        embed = discord.Embed(
            title="画像を生成できませんでした",
            description=(
                "ローカル生成処理が途中で停止しました。ジョブと生成条件は保存されているため、"
                "実行環境を確認したあと同じ条件で再試行できます。"
            ),
            colour=discord.Colour.red(),
            timestamp=datetime.now(UTC),
        )
        if job.error_code:
            embed.add_field(name="エラーコード", value=f"`{job.error_code}`")
        return embed
    complete = job.status is ImageJobStatus.COMPLETED
    step = job.progress_total if complete else job.progress_step
    percentage = round(step / max(1, job.progress_total) * 100)
    return discord.Embed(
        title="画像が完成しました" if complete else "画像を生成しています",
        description=(
            "生成が終わりました。投稿の準備をしています…"
            if complete
            else "バックグラウンドで生成中です。そのまま会話を続けられます。"
        ),
        colour=discord.Colour.green() if complete else discord.Colour.blurple(),
        timestamp=datetime.now(UTC),
    ).add_field(
        name="進捗",
        value=f"{step}/{job.progress_total} · {percentage}%",
        inline=True,
    ).add_field(
        name="サイズ",
        value=f"幅 {job.width}・高さ {job.height}",
        inline=True,
    )


def _image_result_embed(
    job: ImageGenerationJob,
    *,
    filename: str,
) -> discord.Embed:
    duration = (
        f"{job.generation_seconds:.1f}秒"
        if job.generation_seconds is not None
        else "完了"
    )
    embed = discord.Embed(
        title="生成した画像",
        description=_image_prompt_preview(job),
        colour=discord.Colour.green(),
        timestamp=datetime.now(UTC),
    )
    embed.add_field(name="モデル", value="Ideogram 4・ローカルMLX", inline=True)
    embed.add_field(name="生成時間", value=duration, inline=True)
    embed.add_field(name="シード", value=str(job.seed), inline=True)
    embed.set_image(url=f"attachment://{filename}")
    return embed


def _image_prompt_preview(job: ImageGenerationJob) -> str:
    """Show the actual creative brief instead of hiding it behind one subject line."""

    parts = (
        job.prompt.subject.strip(),
        job.prompt.scene.strip(),
        f"{job.prompt.style.strip()} · {job.prompt.lighting.strip()}",
    )
    return "\n".join(part for part in parts if part)[:1_000]
