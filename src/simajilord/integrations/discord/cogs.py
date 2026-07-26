"""Discord slash commands as thin capability adapters."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal, TypeAlias, cast

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.capabilities.audio import (
    AudioAction,
    AudioControlRequest,
    AudioControlResponse,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueueRequest,
    AudioQueueResponse,
)
from simajilord.capabilities.media import DownloadRequest, DownloadResponse
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudRequest,
    ReadAloudResponse,
)
from simajilord.capabilities.status import StatusRequest, StatusResponse
from simajilord.capabilities.system import (
    CapabilitySearchRequest,
    CapabilitySearchResponse,
    PingRequest,
    PingResponse,
    UptimeRequest,
    UptimeResponse,
)
from simajilord.capabilities.utility import (
    ChooseRequest,
    ChooseResponse,
    RollRequest,
    RollResponse,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import MediaError, UserError
from simajilord.domain.audio import AudioKind, LoopMode
from simajilord.domain.media import DownloadFormat
from simajilord.runtime import SimajilordRuntime
from simajilord.services.read_aloud import ReadAloudMode

from .audio import DiscordAudioOutput
from .capabilities import (
    DiscordConnectVoiceRequest,
    DiscordConnectVoiceResponse,
    DiscordPollRequest,
    DiscordPollResponse,
    DiscordServerRequest,
    DiscordServerResponse,
    DiscordUserRequest,
    DiscordUserResponse,
)
from .presenter import EmbedField, EmbedTone, command_embed

log = logging.getLogger(__name__)
BotContext: TypeAlias = commands.Context[commands.Bot]

_ERROR_MESSAGES = {
    "audio.capacity_reached": "The active voice-server limit has been reached.",
    "audio.loop_mode_required": "Choose a loop mode.",
    "audio.not_paused": "Playback is not paused.",
    "audio.nothing_playing": "Nothing is currently playing.",
    "audio.output_disconnected": "The audio output is disconnected.",
    "audio.session_closed": "The audio session is closed.",
    "audio.session_missing": "No audio session exists in this server.",
    "media.reference_required": "Provide a media URL or search query.",
    "media.reference_too_long": "The media reference is too long.",
    "media.url_unsupported": (
        "Use a public HTTPS YouTube or TikTok URL without credentials or a custom port."
    ),
    "discord.message_limit_invalid": "Message history limit must be between 1 and 100.",
    "read_aloud.route_fields_required": "Choose both text and voice channels.",
    "speech.no_readable_text": "There is no readable text.",
    "speech.queue_full": "The read-aloud queue is full. Try again shortly.",
    "utility.dice_count_invalid": "Dice count must be between 1 and 20.",
    "utility.dice_sides_invalid": "Sides must be between 2 and 1000.",
    "utility.option_count_invalid": "Provide between 2 and 20 options.",
    "utility.option_too_long": "Each option must be at most 100 characters.",
    "workspace.required": "This action requires a Discord server.",
}

_MEDIA_ERROR_MESSAGES = {
    "cookie_required": (
        "This media requires authentication. Configure a private cookie file on the host."
    ),
    "geo_restricted": "This media is unavailable in the host region.",
    "rate_limited": "The media site is rate-limiting requests.",
    "timeout": "The media operation timed out.",
    "too_large": "The media exceeds this server's upload limit.",
    "unavailable": "This media is unavailable or private.",
    "unsafe_path": "The media provider returned an unsafe result.",
    "unsupported": "This media URL is not supported.",
    "unknown": "The media provider could not complete the request.",
}

_AUDIO_ACTION_MESSAGES = {
    AudioAction.PAUSE.value: "Playback paused.",
    AudioAction.RESUME.value: "Playback resumed.",
    AudioAction.SKIP.value: "Skipped the current item.",
    AudioAction.STOP.value: "Playback stopped and the queue was cleared.",
    AudioAction.LEAVE.value: "Disconnected from voice.",
}


def invocation_context(interaction: discord.Interaction) -> InvocationContext:
    return InvocationContext(
        actor_id=str(interaction.user.id),
        workspace_id=str(interaction.guild_id) if interaction.guild_id else None,
        transport="discord",
        request_id=str(interaction.id),
    )


def prefix_context(context: BotContext) -> InvocationContext:
    return InvocationContext(
        actor_id=str(context.author.id),
        workspace_id=str(context.guild.id) if context.guild else None,
        transport="discord",
        request_id=str(context.message.id),
    )


def error_message(error: Exception) -> str:
    if isinstance(error, MediaError):
        return _MEDIA_ERROR_MESSAGES.get(error.category, _MEDIA_ERROR_MESSAGES["unknown"])
    if isinstance(error, UserError):
        return _ERROR_MESSAGES.get(error.code, error.code)
    log.exception("Unhandled Discord command error", exc_info=error)
    return "The request failed unexpectedly. Check the host logs."


async def send_error(interaction: discord.Interaction, error: Exception) -> None:
    embed = command_embed(
        "Command failed",
        description=error_message(error),
        tone=EmbedTone.ERROR,
    )
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, ephemeral=True)


class SystemCog(commands.Cog):
    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @app_commands.command(name="ping", description="Check platform health and Discord latency.")
    async def ping(self, interaction: discord.Interaction) -> None:
        response = cast(
            PingResponse,
            await self.runtime.registry.invoke(
                "system.ping",
                PingRequest(transport_latency_ms=round(self.bot.latency * 1_000, 1)),
                invocation_context(interaction),
            ),
        )
        await interaction.response.send_message(
            embed=command_embed(
                "Platform health",
                fields=(
                    EmbedField("Status", response.status),
                    EmbedField(
                        "Discord latency",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @app_commands.command(
        name="capabilities",
        description="Find a small relevant set of available capabilities.",
    )
    @app_commands.describe(query="What do you want the platform to do?")
    async def capabilities(self, interaction: discord.Interaction, query: str = "") -> None:
        response = cast(
            CapabilitySearchResponse,
            await self.runtime.registry.invoke(
                "system.discover_capabilities",
                CapabilitySearchRequest(query=query, limit=8),
                invocation_context(interaction),
            ),
        )
        if not response.capabilities:
            await interaction.response.send_message(
                embed=command_embed(
                    "Capabilities",
                    description="No matching capabilities were found.",
                    tone=EmbedTone.WARNING,
                )
            )
            return
        lines = [
            f"• `{item.name}` — {item.summary} (`{item.risk}`, approval: `{item.approval}`)"
            for item in response.capabilities
        ]
        await interaction.response.send_message(
            embed=command_embed(
                "Capabilities",
                description="\n".join(lines),
                fields=(EmbedField("Query", query or "all", inline=False),),
            )
        )

    @app_commands.command(name="about", description="Explain this adapter's role.")
    async def about(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            embed=command_embed(
                "About Simajilord AI",
                description=(
                    "Simajilord AI is a capability platform. Discord is one transport "
                    "adapter; media, speech, audio policy, and future agent decisions live "
                    "outside this bot."
                ),
            )
        )

    @app_commands.command(name="uptime", description="Show process start time and uptime.")
    async def uptime(self, interaction: discord.Interaction) -> None:
        response = cast(
            UptimeResponse,
            await self.runtime.registry.invoke(
                "system.uptime",
                UptimeRequest(),
                invocation_context(interaction),
            ),
        )
        total_seconds = int(response.uptime_seconds)
        days, remainder = divmod(total_seconds, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, seconds = divmod(remainder, 60)
        await interaction.response.send_message(
            embed=command_embed(
                "Process uptime",
                fields=(
                    EmbedField(
                        "Started",
                        f"<t:{int(response.started_at.timestamp())}:F>",
                        inline=False,
                    ),
                    EmbedField(
                        "Uptime",
                        f"{days}d {hours}h {minutes}m {seconds}s",
                        inline=False,
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @app_commands.command(name="status", description="Show structured platform status.")
    async def status(self, interaction: discord.Interaction) -> None:
        response = cast(
            StatusResponse,
            await self.runtime.registry.invoke(
                "system.status",
                StatusRequest(),
                invocation_context(interaction),
            ),
        )
        await interaction.response.send_message(
            embed=command_embed(
                "Platform status",
                fields=(
                    EmbedField(
                        "Runtime",
                        f"Status: **{response.status.upper()}**\n"
                        f"AI: **{response.model_runtime.title()}**",
                    ),
                    EmbedField(
                        "System",
                        f"Capabilities: **{response.capability_count}**\n"
                        "Voice sessions: "
                        f"{response.active_audio_session_count}/"
                        f"{response.audio_session_count} active",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )


class MusicCog(commands.Cog):
    music = app_commands.Group(name="music", description="Play and control audio.")

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    async def _connect_to_member(self, interaction: discord.Interaction) -> None:
        if not isinstance(interaction.user, discord.Member):
            raise UserError("Your voice state is unavailable.")
        voice_state = interaction.user.voice
        if voice_state is None or voice_state.channel is None:
            raise UserError("Join a voice channel first.")
        if interaction.guild_id is None:
            raise UserError("Music commands require a Discord server.")
        cast(
            DiscordConnectVoiceResponse,
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(voice_state.channel.id)),
                invocation_context(interaction),
            ),
        )

    @music.command(name="play", description="Resolve and queue a URL or search query.")
    @app_commands.describe(reference="A YouTube/TikTok URL or a YouTube search query")
    async def play(self, interaction: discord.Interaction, reference: str) -> None:
        try:
            await interaction.response.defer(thinking=True)
            await self._connect_to_member(interaction)
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "audio.play",
                    AudioPlayRequest(reference=reference),
                    invocation_context(interaction),
                ),
            )
            await interaction.followup.send(
                embed=command_embed(
                    "Track queued",
                    description=f"[{response.title}]({response.page_url})",
                    fields=(
                        EmbedField("Queue position", str(response.queue_position)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(name="queue", description="Show current and pending audio.")
    async def queue(self, interaction: discord.Interaction) -> None:
        try:
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            lines = [
                f"Now: **{response.current.title}**"
                if response.current
                else "Now: nothing",
                f"State: {'paused' if response.paused else 'playing/idle'} · "
                f"Loop: `{response.loop_mode}`",
            ]
            lines.extend(
                f"{index}. [{item.kind}] {item.title}"
                for index, item in enumerate(response.pending[:10], start=1)
            )
            if len(response.pending) > 10:
                lines.append(f"…and {len(response.pending) - 10} more.")
            await interaction.response.send_message(
                embed=command_embed(
                    "Audio queue",
                    description="\n".join(lines),
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def _control(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        loop_mode: LoopMode | None = None,
    ) -> None:
        try:
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(action=action, loop_mode=loop_mode),
                    invocation_context(interaction),
                ),
            )
            if response.action == AudioAction.LOOP.value:
                message = f"Loop mode set to `{response.loop_mode}`."
            else:
                message = _AUDIO_ACTION_MESSAGES[response.action]
            await interaction.response.send_message(
                embed=command_embed(
                    "Audio control",
                    description=message,
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(name="pause", description="Pause playback.")
    async def pause(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.PAUSE)

    @music.command(name="resume", description="Resume playback.")
    async def resume(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.RESUME)

    @music.command(name="skip", description="Skip the current item.")
    async def skip(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SKIP)

    @music.command(name="stop", description="Stop playback and clear the queue.")
    async def stop(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.STOP)

    @music.command(name="leave", description="Disconnect from voice.")
    async def leave(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.LEAVE)

    @music.command(name="loop", description="Set music loop behavior.")
    async def loop(
        self,
        interaction: discord.Interaction,
        mode: Literal["none", "track", "queue"],
    ) -> None:
        await self._control(interaction, AudioAction.LOOP, LoopMode(mode))


class ReadAloudCog(commands.Cog):
    readaloud = app_commands.Group(
        name="readaloud",
        description="Configure automatic text-channel speech.",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @readaloud.command(name="setup", description="Route a text channel into a voice channel.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        text_channel="Messages from this channel are read automatically",
        voice_channel="Audio is played in this channel",
        mode="Queue speech, or skip it while music is active",
    )
    async def setup(
        self,
        interaction: discord.Interaction,
        text_channel: discord.TextChannel | None = None,
        voice_channel: discord.VoiceChannel | None = None,
        mode: Literal["queue", "skip_during_music"] = "queue",
    ) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not member.guild_permissions.manage_guild:
                raise UserError("Manage Server permission is required.")
            selected_text = text_channel
            if selected_text is None and isinstance(interaction.channel, discord.TextChannel):
                selected_text = interaction.channel
            if selected_text is None:
                raise UserError("Choose a server text channel.")
            selected_voice = voice_channel
            if selected_voice is None and member.voice is not None:
                candidate = member.voice.channel
                if isinstance(candidate, discord.VoiceChannel):
                    selected_voice = candidate
            if selected_voice is None:
                raise UserError("Choose a voice channel or join one first.")

            response = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "speech.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.CONFIGURE,
                        text_channel_id=str(selected_text.id),
                        audio_destination_id=str(selected_voice.id),
                        mode=ReadAloudMode(mode),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Read-aloud configured",
                    fields=(
                        EmbedField("Text channel", selected_text.mention),
                        EmbedField("Voice channel", selected_voice.mention),
                        EmbedField("Mode", response.mode or "queue"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(name="status", description="Show the configured route.")
    async def status(self, interaction: discord.Interaction) -> None:
        try:
            response = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "speech.manage_read_aloud",
                    ReadAloudRequest(action=ReadAloudAction.STATUS),
                    invocation_context(interaction),
                ),
            )
            if not response.enabled:
                await interaction.response.send_message(
                    embed=command_embed(
                        "Read-aloud status",
                        description="Read-aloud is disabled.",
                        tone=EmbedTone.WARNING,
                    )
                )
                return
            await interaction.response.send_message(
                embed=command_embed(
                    "Read-aloud status",
                    fields=(
                        EmbedField("Text channel", f"<#{response.text_channel_id}>"),
                        EmbedField(
                            "Voice channel",
                            f"<#{response.audio_destination_id}>",
                        ),
                        EmbedField("Mode", response.mode or "queue"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(name="disable", description="Disable automatic read-aloud.")
    @app_commands.default_permissions(manage_guild=True)
    async def disable(self, interaction: discord.Interaction) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not member.guild_permissions.manage_guild:
                raise UserError("Manage Server permission is required.")
            await self.runtime.registry.invoke(
                "speech.manage_read_aloud",
                ReadAloudRequest(action=ReadAloudAction.DISABLE),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Read-aloud disabled",
                    description="Automatic read-aloud routing has been disabled.",
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None or not message.content.strip():
            return
        workspace_id = str(message.guild.id)
        if not self.runtime.read_aloud.matches(workspace_id, str(message.channel.id)):
            return
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None:
            return
        guild_id = message.guild.id
        session = self.runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(self.bot, guild_id),
        )
        if (
            route.mode is ReadAloudMode.SKIP_DURING_MUSIC
            and session.current is not None
            and session.current.kind is AudioKind.MUSIC
        ):
            return
        output = cast(DiscordAudioOutput, session.output)
        try:
            if (
                output.connected
                and output.destination_id != int(route.audio_destination_id)
                and session.current is not None
            ):
                return
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=route.audio_destination_id),
                InvocationContext(
                    actor_id=str(message.author.id),
                    workspace_id=workspace_id,
                    transport="discord",
                    request_id=f"read-aloud:{message.id}",
                ),
            )
            item = await self.runtime.speech.synthesize(
                message.content,
                title=f"Message from {message.author.display_name}",
            )
            await session.enqueue(item)
        except Exception:
            log.exception(
                "Automatic read-aloud failed guild=%s channel=%s",
                message.guild.id,
                message.channel.id,
            )


class DownloadCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._last_request: dict[int, float] = {}

    @app_commands.command(
        name="download",
        description="Download one public YouTube or TikTok URL.",
    )
    async def download(
        self,
        interaction: discord.Interaction,
        url: str,
        media_type: Literal["video", "audio"] = "video",
    ) -> None:
        temporary: Path | None = None
        try:
            now = time.monotonic()
            previous = self._last_request.get(interaction.user.id, 0.0)
            if now - previous < 30:
                raise UserError("Wait 30 seconds between downloads.")
            self._last_request[interaction.user.id] = now
            await interaction.response.defer(ephemeral=True, thinking=True)
            download_root = self.runtime.settings.data_dir / "downloads"
            download_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix="request-", dir=download_root))
            guild_limit = interaction.guild.filesize_limit if interaction.guild else 10_000_000
            max_bytes = max(1_000_000, guild_limit - 1_000_000)
            response = cast(
                DownloadResponse,
                await self.runtime.registry.invoke(
                    "media.download",
                    DownloadRequest(
                        url=url,
                        media_type=DownloadFormat(media_type),
                        destination=temporary,
                        max_bytes=max_bytes,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.followup.send(
                embed=command_embed(
                    "Download ready",
                    description=response.title,
                    fields=(
                        EmbedField(
                            "Size",
                            f"{response.size_bytes / 1_000_000:.1f} MB",
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                file=discord.File(response.path),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)


class UtilityCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="roll", description="Roll bounded virtual dice.")
    async def roll(
        self,
        interaction: discord.Interaction,
        dice: app_commands.Range[int, 1, 20] = 1,
        sides: app_commands.Range[int, 2, 1_000] = 6,
    ) -> None:
        try:
            response = cast(
                RollResponse,
                await self.runtime.registry.invoke(
                    "utility.roll",
                    RollRequest(dice=dice, sides=sides),
                    invocation_context(interaction),
                ),
            )
            values = ", ".join(str(value) for value in response.rolls)
            await interaction.response.send_message(
                embed=command_embed(
                    "Dice roll",
                    fields=(
                        EmbedField("Rolls", values, inline=False),
                        EmbedField("Total", str(response.total)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="choose", description="Choose one comma-separated option.")
    async def choose(self, interaction: discord.Interaction, options: str) -> None:
        try:
            parsed = tuple(item.strip() for item in options.split(","))
            response = cast(
                ChooseResponse,
                await self.runtime.registry.invoke(
                    "utility.choose",
                    ChooseRequest(options=parsed),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Choice",
                    description=response.choice,
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)


class DiscordInfoCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="serverinfo", description="Show this server's public structure.")
    async def serverinfo(self, interaction: discord.Interaction) -> None:
        try:
            response = cast(
                DiscordServerResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_server",
                    DiscordServerRequest(),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.name,
                description=f"Server ID: `{response.server_id}`",
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.add_field(name="Members", value=response.member_count or "Unknown")
            embed.add_field(name="Text channels", value=response.text_channel_count)
            embed.add_field(name="Voice channels", value=response.voice_channel_count)
            embed.add_field(name="Roles", value=response.role_count)
            embed.add_field(
                name="Created",
                value=f"<t:{int(discord.utils.parse_time(response.created_at_iso).timestamp())}:F>",
            )
            if response.icon_url:
                embed.set_thumbnail(url=response.icon_url)
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="userinfo", description="Show public data for one Discord user.")
    async def userinfo(
        self,
        interaction: discord.Interaction,
        user: discord.User | None = None,
    ) -> None:
        try:
            target = user or interaction.user
            response = cast(
                DiscordUserResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_user",
                    DiscordUserRequest(user_id=str(target.id)),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.display_name,
                description=f"User ID: `{response.user_id}`",
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.set_thumbnail(url=response.avatar_url)
            embed.add_field(
                name="Account created",
                value=f"<t:{int(discord.utils.parse_time(response.created_at_iso).timestamp())}:F>",
                inline=False,
            )
            if response.joined_at_iso:
                joined = discord.utils.parse_time(response.joined_at_iso)
                embed.add_field(
                    name="Joined server",
                    value=f"<t:{int(joined.timestamp())}:F>",
                    inline=False,
                )
            if response.top_role:
                embed.add_field(name="Top role", value=response.top_role)
            embed.add_field(name="Account type", value="Bot" if response.bot else "User")
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)


class DiscordActionCog(commands.Cog):
    """Discord-native presentation actions."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(name="avatar", description="Show a Discord user's avatar.")
    async def avatar(
        self,
        interaction: discord.Interaction,
        user: discord.User | None = None,
    ) -> None:
        try:
            target = user or interaction.user
            response = cast(
                DiscordUserResponse,
                await self.runtime.registry.invoke(
                    "discord.inspect_user",
                    DiscordUserRequest(user_id=str(target.id)),
                    invocation_context(interaction),
                ),
            )
            embed = discord.Embed(
                title=response.display_name,
                colour=discord.Colour.blurple(),
                timestamp=discord.utils.utcnow(),
            )
            embed.set_image(url=response.avatar_url)
            await interaction.response.send_message(embed=embed)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(name="poll", description="Create a native Discord poll.")
    async def poll(
        self,
        interaction: discord.Interaction,
        question: str,
        options: str,
        hours: app_commands.Range[int, 1, 168] = 24,
        multiple: bool = False,
    ) -> None:
        try:
            if interaction.channel_id is None:
                raise UserError("This command requires a text channel.")
            response = cast(
                DiscordPollResponse,
                await self.runtime.registry.invoke(
                    "discord.create_poll",
                    DiscordPollRequest(
                        channel_id=str(interaction.channel_id),
                        question=question,
                        options=tuple(item.strip() for item in options.split(",")),
                        duration_hours=hours,
                        multiple=multiple,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Poll created",
                    description=(
                        f"[Open poll](https://discord.com/channels/{interaction.guild_id}/"
                        f"{response.channel_id}/{response.message_id})"
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)


class ObservationCog(commands.Cog):
    """Feed Discord changes into the platform event stream for agent reconciliation."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        await self.runtime.journal.append(
            kind="discord.message.created",
            actor_id=str(message.author.id),
            workspace_id=str(message.guild.id),
            transport="discord",
            request_id=str(message.id),
            payload={
                "message_id": str(message.id),
                "channel_id": str(message.channel.id),
                "author_name": message.author.display_name,
                "author_is_bot": message.author.bot,
                "content": message.content,
                "attachments": [
                    {
                        "id": str(attachment.id),
                        "filename": attachment.filename,
                        "size": attachment.size,
                    }
                    for attachment in message.attachments
                ],
            },
        )


class PrefixCog(commands.Cog):
    """Prefix presentation for the same APIs used by slash commands."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    async def _connect(self, context: BotContext) -> None:
        if context.guild is None or not isinstance(context.author, discord.Member):
            raise UserError("workspace.required")
        state = context.author.voice
        if state is None or state.channel is None:
            raise UserError("Join a voice channel first.")
        await self.runtime.registry.invoke(
            "discord.connect_voice",
            DiscordConnectVoiceRequest(channel_id=str(state.channel.id)),
            prefix_context(context),
        )

    @commands.command(name="ping")
    async def ping(self, context: BotContext) -> None:
        response = cast(
            PingResponse,
            await self.runtime.registry.invoke(
                "system.ping",
                PingRequest(transport_latency_ms=round(self.bot.latency * 1_000, 1)),
                prefix_context(context),
            ),
        )
        await context.send(
            embed=command_embed(
                "Platform health",
                fields=(
                    EmbedField("Status", response.status),
                    EmbedField(
                        "Discord latency",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @commands.command(name="capabilities", aliases=("help",))
    async def capabilities(self, context: BotContext, *, query: str = "") -> None:
        response = cast(
            CapabilitySearchResponse,
            await self.runtime.registry.invoke(
                "system.discover_capabilities",
                CapabilitySearchRequest(query=query, limit=8),
                prefix_context(context),
            ),
        )
        description = (
            "\n".join(
                f"• `{item.name}` — {item.summary} (`{item.risk}`)"
                for item in response.capabilities
            )
            or "No matching capabilities were found."
        )
        await context.send(
            embed=command_embed("Capabilities", description=description)
        )

    @commands.command(name="play")
    async def play(self, context: BotContext, *, reference: str) -> None:
        try:
            await self._connect(context)
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "audio.play",
                    AudioPlayRequest(reference=reference),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=command_embed(
                    "Track queued",
                    description=f"[{response.title}]({response.page_url})",
                    fields=(
                        EmbedField("Queue position", str(response.queue_position)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="queue")
    async def queue(self, context: BotContext) -> None:
        try:
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    prefix_context(context),
                ),
            )
            lines = [
                f"Now: **{response.current.title}**"
                if response.current
                else "Now: nothing"
            ]
            lines.extend(
                f"{index}. [{item.kind}] {item.title}"
                for index, item in enumerate(response.pending[:10], start=1)
            )
            await context.send(
                embed=command_embed("Audio queue", description="\n".join(lines))
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    async def _control(self, context: BotContext, action: AudioAction) -> None:
        try:
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(action=action),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=command_embed(
                    "Audio control",
                    description=_AUDIO_ACTION_MESSAGES[response.action],
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="pause")
    async def pause(self, context: BotContext) -> None:
        await self._control(context, AudioAction.PAUSE)

    @commands.command(name="resume")
    async def resume(self, context: BotContext) -> None:
        await self._control(context, AudioAction.RESUME)

    @commands.command(name="skip")
    async def skip(self, context: BotContext) -> None:
        await self._control(context, AudioAction.SKIP)

    @commands.command(name="stop")
    async def stop(self, context: BotContext) -> None:
        await self._control(context, AudioAction.STOP)

    @commands.command(name="leave")
    async def leave(self, context: BotContext) -> None:
        await self._control(context, AudioAction.LEAVE)


async def setup_cogs(bot: commands.Bot, runtime: SimajilordRuntime) -> None:
    await bot.add_cog(SystemCog(bot, runtime))
    await bot.add_cog(MusicCog(bot, runtime))
    await bot.add_cog(ReadAloudCog(bot, runtime))
    await bot.add_cog(DownloadCog(runtime))
    await bot.add_cog(UtilityCog(runtime))
    await bot.add_cog(DiscordInfoCog(runtime))
    await bot.add_cog(DiscordActionCog(runtime))
    await bot.add_cog(ObservationCog(runtime))
    await bot.add_cog(PrefixCog(bot, runtime))
