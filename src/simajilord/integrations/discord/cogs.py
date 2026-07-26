"""Discord slash commands as thin capability adapters."""

from __future__ import annotations

import asyncio
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
    AudioHistoryRequest,
    AudioHistoryResponse,
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
from simajilord.services.audio import AudioSession
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
    "audio.auto_leave_value_required": "Choose whether automatic voice leave is enabled.",
    "audio.capacity_reached": "The active voice-server limit has been reached.",
    "audio.history_limit_invalid": "History limit must be between 1 and 25.",
    "audio.loop_mode_required": "Choose a loop mode.",
    "audio.not_paused": "Playback is not paused.",
    "audio.nothing_playing": "Nothing is currently playing.",
    "audio.output_disconnected": "The audio output is disconnected.",
    "audio.queue_position_invalid": "Choose a valid upcoming queue position.",
    "audio.seek_position_required": "Provide a playback position.",
    "audio.session_closed": "The audio session is closed.",
    "audio.session_missing": "No audio session exists in this server.",
    "audio.same_voice_required": "Join the Bot's voice channel to control its music.",
    "audio.waiting_queue_restricted": (
        "This waiting queue can be started or changed by one of its requesters."
    ),
    "audio.tune_range_invalid": "Speed and pitch must each be between 0.5 and 2.0.",
    "audio.tune_values_required": "Provide both speed and pitch.",
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


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


def _parse_position(value: str) -> tuple[float, bool]:
    text = value.strip()
    relative = text.startswith(("+", "-"))
    sign = -1.0 if text.startswith("-") else 1.0
    unsigned = text[1:] if relative else text
    parts = unsigned.split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise UserError("Use a timestamp such as `1:23`, `+30`, or `-10`.")
    numbers = [int(part) for part in parts]
    seconds = 0
    for number in numbers:
        seconds = seconds * 60 + number
    return sign * float(seconds), relative


def _progress(position: float, duration: float, *, width: int = 16) -> str:
    if duration <= 0:
        return "─" * width
    ratio = max(0.0, min(1.0, position / duration))
    marker = min(width - 1, round(ratio * (width - 1)))
    return "━" * marker + "●" + "─" * (width - marker - 1)


def _requester(name: str | None) -> str:
    return discord.utils.escape_markdown(name) if name else "Unknown"


def music_added_embed(response: AudioPlayResponse) -> discord.Embed:
    if response.playback_state == "playing":
        playback = "Playing now"
    elif response.playback_state == "waiting_for_voice":
        playback = "Waiting for voice\nJoin a voice channel to start automatically."
    else:
        playback = f"Up next · #{response.queue_position}"
    return command_embed(
        "Added to queue",
        description=f"### [{response.title}]({response.page_url})",
        fields=(
            EmbedField("Playback", playback, inline=False),
            EmbedField("Duration", _duration(response.duration_seconds)),
            EmbedField("Requested by", _requester(response.requested_by_name)),
            EmbedField(
                "Voice",
                f"<#{response.destination_id}>"
                if response.destination_id
                else "Not connected yet",
            ),
        ),
        tone=EmbedTone.SUCCESS,
    )


def music_queue_embed(response: AudioQueueResponse) -> discord.Embed:
    fields: list[EmbedField] = []
    if response.current is None:
        description = "Nothing is playing."
    else:
        current = response.current
        description = f"### [{current.title}]({current.page_url})"
        elapsed = min(response.position_seconds, current.duration_seconds)
        fields.append(
            EmbedField(
                "Progress",
                f"`{_progress(elapsed, current.duration_seconds)}` "
                f"`{_duration(elapsed)} / {_duration(current.duration_seconds)}`",
                inline=False,
            )
        )
        fields.append(
            EmbedField("Requested by", _requester(current.requested_by_name))
        )

    upcoming = tuple(item for item in response.pending if item.kind == AudioKind.MUSIC.value)
    if upcoming:
        lines = [
            f"`{index:02d}` [{item.title}]({item.page_url}) · "
            f"`{_duration(item.duration_seconds)}` · {_requester(item.requested_by_name)}"
            for index, item in enumerate(upcoming[:10], start=1)
        ]
        if len(upcoming) > 10:
            lines.append(f"…and **{len(upcoming) - 10}** more")
        fields.append(EmbedField("Up next", "\n".join(lines), inline=False))
    else:
        fields.append(EmbedField("Up next", "Queue empty", inline=False))

    if response.waiting_for_voice:
        state = "Waiting for voice"
    elif response.paused:
        state = "Paused"
    elif response.current:
        state = "Playing"
    else:
        state = "Ready"
    fields.extend(
        (
            EmbedField("State", state),
            EmbedField("Loop", response.loop_mode.title()),
            EmbedField("Auto leave", "On" if response.auto_leave else "Off"),
        )
    )
    if response.speed != 1.0 or response.pitch != 1.0:
        fields.append(
            EmbedField("Tuning", f"{response.speed:.2f}x speed · {response.pitch:.2f}x pitch")
        )
    if response.destination_id:
        fields.append(EmbedField("Voice", f"<#{response.destination_id}>"))
    elif response.waiting_for_voice:
        fields.append(
            EmbedField(
                "Start",
                "Join a voice channel or press **Start in VC** after joining.",
                inline=False,
            )
        )
    return command_embed("Music", description=description, fields=tuple(fields))


def music_history_embed(response: AudioHistoryResponse) -> discord.Embed:
    if not response.items:
        return command_embed(
            "Recently played",
            description="No tracks have been played yet.",
        )
    lines = []
    for index, item in enumerate(response.items, start=1):
        when = f" · <t:{item.played_at_epoch}:R>" if item.played_at_epoch else ""
        lines.append(
            f"`{index:02d}` [{item.title}]({item.page_url}) · "
            f"`{_duration(item.duration_seconds)}` · {_requester(item.requested_by_name)}{when}"
        )
    return command_embed("Recently played", description="\n".join(lines))


def _discord_audio_session(
    bot: commands.Bot,
    runtime: SimajilordRuntime,
    guild_id: int | None,
) -> AudioSession:
    if guild_id is None:
        raise UserError("workspace.required")
    return runtime.audio.get_or_create(
        str(guild_id),
        lambda: DiscordAudioOutput(bot, guild_id),
    )


def _member_voice_channel(
    member: discord.abc.User,
) -> discord.VoiceChannel | discord.StageChannel | None:
    if not isinstance(member, discord.Member):
        return None
    state = member.voice
    if state is None or not isinstance(
        state.channel, (discord.VoiceChannel, discord.StageChannel)
    ):
        return None
    return state.channel


def _require_same_voice(session: AudioSession, member: discord.abc.User) -> None:
    if not session.output.connected:
        if session.waiting_for_voice and not session.can_control_while_waiting(
            str(member.id)
        ):
            raise UserError("audio.waiting_queue_restricted")
        return
    channel = _member_voice_channel(member)
    if (
        channel is None
        or session.destination_id is None
        or str(channel.id) != session.destination_id
    ):
        raise UserError("audio.same_voice_required")


class MusicControlsView(discord.ui.View):
    """Persistent controls backed by the same capability API as commands and agents."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        super().__init__(timeout=None)
        self.runtime = runtime

    async def _run(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        *,
        loop_mode: LoopMode | None = None,
        position_seconds: float | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "audio.control",
                AudioControlRequest(
                    action=action,
                    loop_mode=loop_mode,
                    position_seconds=position_seconds,
                    speed=speed,
                    pitch=pitch,
                ),
                invocation_context(interaction),
            )
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=music_queue_embed(response),
                view=self,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Start in VC",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:music:start",
        row=0,
    )
    async def start_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        try:
            channel = _member_voice_channel(interaction.user)
            if channel is None:
                raise UserError("Join a voice channel first.")
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            if session.output.connected:
                _require_same_voice(session, interaction.user)
            elif not session.can_start_for(str(interaction.user.id)):
                raise UserError("audio.waiting_queue_restricted")
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(channel.id)),
                invocation_context(interaction),
            )
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=music_queue_embed(response),
                view=self,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Pause",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:pause",
        row=0,
    )
    async def pause_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.PAUSE)

    @discord.ui.button(
        label="Resume",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:music:resume",
        row=0,
    )
    async def resume_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.RESUME)

    @discord.ui.button(
        label="Skip",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:music:skip",
        row=0,
    )
    async def skip_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.SKIP)

    @discord.ui.button(
        label="Loop",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:loop",
        row=0,
    )
    async def loop_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        response = cast(
            AudioQueueResponse,
            await self.runtime.registry.invoke(
                "audio.queue",
                AudioQueueRequest(),
                invocation_context(interaction),
            ),
        )
        modes = (LoopMode.NONE, LoopMode.TRACK, LoopMode.QUEUE)
        current = LoopMode(response.loop_mode)
        await self._run(
            interaction,
            AudioAction.LOOP,
            loop_mode=modes[(modes.index(current) + 1) % len(modes)],
        )

    @discord.ui.button(
        label="Leave",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:music:leave",
        row=1,
    )
    async def leave_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.LEAVE)


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
    music = app_commands.Group(
        name="music",
        description="Advanced music controls and grouped command aliases.",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    async def _prepare_play(self, interaction: discord.Interaction) -> None:
        session = _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
        channel = _member_voice_channel(interaction.user)
        if session.output.connected:
            _require_same_voice(session, interaction.user)
            return
        if channel is None:
            return
        cast(
            DiscordConnectVoiceResponse,
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(channel.id)),
                invocation_context(interaction),
            ),
        )

    async def _send_play(self, interaction: discord.Interaction, reference: str) -> None:
        try:
            await interaction.response.defer(thinking=True)
            await self._prepare_play(interaction)
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "audio.play",
                    AudioPlayRequest(
                        reference=reference,
                        requested_by_name=interaction.user.display_name,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.followup.send(
                embed=music_added_embed(response),
                view=MusicControlsView(self.runtime),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="play",
        description="Add a YouTube/TikTok URL or search to the music queue.",
    )
    @app_commands.describe(reference="Paste a supported URL or type a song/video name")
    async def quick_play(self, interaction: discord.Interaction, reference: str) -> None:
        await self._send_play(interaction, reference)

    @music.command(
        name="play",
        description="Add a URL or search to the music queue.",
    )
    @app_commands.describe(reference="Paste a supported URL or type a song/video name")
    async def play(self, interaction: discord.Interaction, reference: str) -> None:
        await self._send_play(interaction, reference)

    async def _send_queue(self, interaction: discord.Interaction) -> None:
        try:
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_queue_embed(response),
                view=MusicControlsView(self.runtime),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="queue",
        description="Show what is playing and what comes next.",
    )
    async def quick_queue(self, interaction: discord.Interaction) -> None:
        await self._send_queue(interaction)

    @music.command(
        name="queue",
        description="Show what is playing and what comes next.",
    )
    async def queue(self, interaction: discord.Interaction) -> None:
        await self._send_queue(interaction)

    async def _send_history(self, interaction: discord.Interaction, limit: int) -> None:
        try:
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=limit),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(embed=music_history_embed(response))
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="history",
        description="Show recently played tracks and who requested them.",
    )
    async def quick_history(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 25] = 10,
    ) -> None:
        await self._send_history(interaction, int(limit))

    @music.command(
        name="history",
        description="Show recently played tracks and who requested them.",
    )
    async def history(
        self,
        interaction: discord.Interaction,
        limit: app_commands.Range[int, 1, 25] = 10,
    ) -> None:
        await self._send_history(interaction, int(limit))

    async def _control(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        loop_mode: LoopMode | None = None,
        enabled: bool | None = None,
        position_seconds: float | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=action,
                        loop_mode=loop_mode,
                        enabled=enabled,
                        position_seconds=position_seconds,
                        speed=speed,
                        pitch=pitch,
                    ),
                    invocation_context(interaction),
                ),
            )
            if response.action == AudioAction.LOOP.value:
                message = f"Loop mode set to `{response.loop_mode}`."
            elif response.action == AudioAction.REMOVE.value:
                message = f"Removed **{response.affected_title}** from the queue."
            elif response.action == AudioAction.AUTO_LEAVE.value:
                message = f"Auto leave is now **{'on' if response.enabled else 'off'}**."
            elif response.action == AudioAction.SHUFFLE.value:
                message = "Upcoming tracks shuffled."
            elif response.action == AudioAction.SEEK.value:
                message = f"Moved to `{_duration(response.position_seconds or 0)}`."
            elif response.action == AudioAction.TUNE.value:
                message = (
                    f"Speed **{response.speed:.2f}x** · Pitch **{response.pitch:.2f}x**."
                    if response.speed is not None and response.pitch is not None
                    else "Playback tuning updated."
                )
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

    @music.command(name="remove", description="Remove one upcoming track by position.")
    @app_commands.describe(position="The number shown under Up next")
    async def remove(self, interaction: discord.Interaction, position: int) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    "audio.control",
                    AudioControlRequest(
                        action=AudioAction.REMOVE,
                        position=position,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Removed from queue",
                    description=f"**{response.affected_title}**",
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(
        name="autoleave",
        description="Leave when the last listener exits without losing the queue.",
    )
    async def autoleave(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._control(
            interaction,
            AudioAction.AUTO_LEAVE,
            enabled=enabled,
        )

    @music.command(name="shuffle", description="Shuffle upcoming music.")
    async def shuffle(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SHUFFLE)

    @music.command(name="seek", description="Move within the current track.")
    @app_commands.describe(position="Absolute 1:23 or relative +30 / -10")
    async def seek(self, interaction: discord.Interaction, position: str) -> None:
        try:
            parsed, relative = _parse_position(position)
            if relative:
                snapshot = cast(
                    AudioQueueResponse,
                    await self.runtime.registry.invoke(
                        "audio.queue",
                        AudioQueueRequest(),
                        invocation_context(interaction),
                    ),
                )
                parsed += snapshot.position_seconds
            await self._control(
                interaction,
                AudioAction.SEEK,
                position_seconds=max(0.0, parsed),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @music.command(name="tune", description="Adjust playback speed and pitch.")
    async def tune(
        self,
        interaction: discord.Interaction,
        speed: app_commands.Range[float, 0.5, 2.0] = 1.0,
        pitch: app_commands.Range[float, 0.5, 2.0] = 1.0,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.TUNE,
            speed=float(speed),
            pitch=float(pitch),
        )


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


class VoiceLifecycleCog(commands.Cog):
    """Keep voice presence aligned with listeners without losing the music queue."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._leave_tasks: dict[str, asyncio.Task[None]] = {}

    async def cog_unload(self) -> None:
        for task in self._leave_tasks.values():
            task.cancel()
        self._leave_tasks.clear()

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if member.bot:
            return
        workspace_id = str(member.guild.id)
        session = self.runtime.audio.find(workspace_id)
        if session is None:
            return

        destination_id = (
            int(session.destination_id) if session.destination_id is not None else None
        )
        joined_expected_channel = (
            after.channel is not None
            and (
                (session.waiting_for_voice and session.can_start_for(str(member.id)))
                or after.channel.id == destination_id
            )
        )
        if joined_expected_channel and after.channel is not None:
            task = self._leave_tasks.pop(workspace_id, None)
            if task is not None:
                task.cancel()
            if session.has_music and not session.output.connected:
                try:
                    self.runtime.audio.assert_connection_capacity(workspace_id)
                    await session.connect(str(after.channel.id))
                    log.info(
                        "Resumed preserved audio queue after a listener joined guild=%s",
                        workspace_id,
                    )
                except Exception:
                    log.exception(
                        "Could not resume preserved audio queue guild=%s",
                        workspace_id,
                    )
            return

        if destination_id is None:
            return
        if before.channel is None or before.channel.id != destination_id:
            return
        existing = self._leave_tasks.pop(workspace_id, None)
        if existing is not None:
            existing.cancel()

        async def leave_if_lonely() -> None:
            try:
                await asyncio.sleep(10)
                output = session.output
                if not output.connected or not session.auto_leave:
                    return
                guild = member.guild
                channel = guild.get_channel(destination_id)
                if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
                    await session.suspend()
                    return
                if any(not listener.bot for listener in channel.members):
                    return
                await session.suspend()
                log.info(
                    "Auto-left empty voice channel while preserving queue guild=%s channel=%s",
                    workspace_id,
                    destination_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Voice auto-leave failed guild=%s", workspace_id)
            finally:
                if self._leave_tasks.get(workspace_id) is asyncio.current_task():
                    self._leave_tasks.pop(workspace_id, None)

        self._leave_tasks[workspace_id] = asyncio.create_task(
            leave_if_lonely(),
            name=f"simajilord-auto-leave-{workspace_id}",
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

    async def _prepare_play(self, context: BotContext) -> None:
        if context.guild is None or not isinstance(context.author, discord.Member):
            raise UserError("workspace.required")
        session = _discord_audio_session(
            self.bot,
            self.runtime,
            context.guild.id,
        )
        channel = _member_voice_channel(context.author)
        if session.output.connected:
            _require_same_voice(session, context.author)
            return
        if channel is None:
            return
        await self.runtime.registry.invoke(
            "discord.connect_voice",
            DiscordConnectVoiceRequest(channel_id=str(channel.id)),
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
            await self._prepare_play(context)
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "audio.play",
                    AudioPlayRequest(
                        reference=reference,
                        requested_by_name=context.author.display_name,
                    ),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=music_added_embed(response),
                view=MusicControlsView(self.runtime),
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
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    prefix_context(context),
                ),
            )
            await context.send(
                embed=music_queue_embed(response),
                view=MusicControlsView(self.runtime),
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="history")
    async def history(self, context: BotContext, limit: int = 10) -> None:
        try:
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=limit),
                    prefix_context(context),
                ),
            )
            await context.send(embed=music_history_embed(response))
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
            if context.guild is None:
                raise UserError("workspace.required")
            session = self.runtime.audio.require(str(context.guild.id))
            _require_same_voice(session, context.author)
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
    bot.add_view(MusicControlsView(runtime))
    await bot.add_cog(SystemCog(bot, runtime))
    await bot.add_cog(MusicCog(bot, runtime))
    await bot.add_cog(ReadAloudCog(bot, runtime))
    await bot.add_cog(VoiceLifecycleCog(runtime))
    await bot.add_cog(DownloadCog(runtime))
    await bot.add_cog(UtilityCog(runtime))
    await bot.add_cog(DiscordInfoCog(runtime))
    await bot.add_cog(DiscordActionCog(runtime))
    await bot.add_cog(ObservationCog(runtime))
    await bot.add_cog(PrefixCog(bot, runtime))
