"""Discord slash commands as thin capability adapters."""

from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias, cast

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.agent import (
    AGENT_AUDIO_GRANT,
    AGENT_AUTONOMY_ACTOR_ID,
    AGENT_FILE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MESSAGE_BREAK,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_NO_ACTION_CONTENT,
    AGENT_WEB_GRANT,
    AgentBusyError,
    AgentProgressStage,
    AgentRateLimitError,
    AgentRequest,
    AgentTrigger,
    AgentUnavailableError,
)
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
    AudioSearchItem,
    AudioSearchRequest,
    AudioSearchResponse,
)
from simajilord.capabilities.media import DownloadRequest, DownloadResponse
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudRequest,
    ReadAloudResponse,
)
from simajilord.capabilities.speech import SpeechSpeakRequest
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
from simajilord.capabilities.web import (
    WebFetchRequest,
    WebFetchResponse,
    WebFindRequest,
    WebFindResponse,
    WebSearchRequest,
    WebSearchResponse,
)
from simajilord.config import AgentFeatureAccess
from simajilord.core import InvocationContext
from simajilord.core.errors import MediaError, ModerationError, UserError, WebError
from simajilord.domain.audio import AudioKind, LoopMode
from simajilord.domain.media import DownloadFormat
from simajilord.domain.web import SearchDepth, WebSource, WebTextMatch
from simajilord.observability import EventRecord
from simajilord.runtime import SimajilordRuntime
from simajilord.services.audio import AudioSession
from simajilord.services.read_aloud import ReadAloudMode

from .audio import DiscordAudioOutput
from .capabilities import (
    DiscordConnectVoiceRequest,
    DiscordPollRequest,
    DiscordPollResponse,
    DiscordServerRequest,
    DiscordServerResponse,
    DiscordUserRequest,
    DiscordUserResponse,
    agent_readable_channel_ids,
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
    "audio.search_empty": "No matching track was found.",
    "audio.search_limit_invalid": "Search limit must be between 1 and 10.",
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
    "media.query_url_not_allowed": "Search text must not contain a URL.",
    "media.url_private": "Private and local network addresses are not allowed.",
    "media.url_unresolvable": "The media host could not be resolved.",
    "media.url_unsupported": (
        "Use a public HTTPS media URL without credentials or a custom port."
    ),
    "moderation.daily_limit_reached": (
        "Today's HIVE API budget has been used. It resets at 00:00 UTC."
    ),
    "moderation.filename_invalid": "The attachment filename is invalid.",
    "moderation.media_empty": "The attachment is empty.",
    "moderation.media_too_large": "The attachment is too large for HIVE analysis.",
    "moderation.media_type_unsupported": (
        "HIVE analysis supports common image and video attachments."
    ),
    "moderation.not_configured": "HIVE Moderation is not configured on this host yet.",
    "discord.message_limit_invalid": "Message history limit must be between 1 and 100.",
    "discord.message_chunk_limit_invalid": (
        "Message chunks must contain between 1 and 1000 characters."
    ),
    "discord.message_offset_invalid": "Choose a valid message text offset.",
    "discord.manage_guild_required": (
        "Read-aloud is already assigned to another voice channel. "
        "A server manager can move it with `/readaloud setup`."
    ),
    "read_aloud.route_fields_required": "Choose both text and voice channels.",
    "read_aloud.destination_conflict": (
        "Read-aloud is already assigned to another voice channel."
    ),
    "read_aloud.source_channel_required": "Choose a conversation channel.",
    "read_aloud.source_channel_limit": "Choose between 1 and 25 conversation channels.",
    "read_aloud.source_channels_required": "Choose at least one conversation channel.",
    "discord.message_channel_unavailable": (
        "Every selected channel must be readable by both you and the Bot."
    ),
    "speech.no_readable_text": "There is no readable text.",
    "speech.queue_full": "The read-aloud queue is full. Try again shortly.",
    "utility.dice_count_invalid": "Dice count must be between 1 and 20.",
    "utility.dice_sides_invalid": "Sides must be between 2 and 1000.",
    "utility.option_count_invalid": "Provide between 2 and 20 options.",
    "utility.option_too_long": "Each option must be at most 100 characters.",
    "web.chunk_limit_invalid": "Page chunks must contain between 200 and 6000 characters.",
    "web.context_limit_invalid": "Find context must contain between 40 and 300 characters.",
    "web.match_limit_invalid": "Find can return between 1 and 10 matches.",
    "web.offset_invalid": "Choose a valid page-text offset.",
    "web.pattern_required": "Provide the phrase to find on the page.",
    "web.pattern_too_long": "The phrase to find must be at most 300 characters.",
    "web.query_required": "Provide a web search query.",
    "web.query_too_long": "The web search query must be at most 500 characters.",
    "web.safesearch_invalid": "Safe Search must be 0, 1, or 2.",
    "web.time_range_invalid": "Time range must be day, month, or year.",
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

_WEB_ERROR_MESSAGES = {
    "content_empty": "The page contains no extractable text.",
    "content_invalid": "The page could not be read.",
    "content_type_unsupported": "This page type cannot be read yet.",
    "fetch_failed": "The page could not be fetched.",
    "redirect_invalid": "The page returned an invalid redirect.",
    "request_too_broad": "The requested search is too broad.",
    "response_too_large": "The page is too large to read safely.",
    "search_backend_error": "The local search provider returned an error.",
    "search_invalid_response": "The local search provider returned invalid data.",
    "search_response_too_large": "The local search response was too large.",
    "search_unavailable": "The local search provider is temporarily unavailable.",
    "timeout": "The web operation timed out.",
    "too_many_redirects": "The page redirected too many times.",
    "upstream_unavailable": "The website is temporarily unavailable.",
    "url_invalid": "Provide a valid public HTTP or HTTPS URL.",
    "url_private": "Private and local network addresses cannot be opened.",
    "url_rejected": "The website rejected the request.",
    "url_unresolvable": "The web host could not be reached.",
}

_MODERATION_ERROR_MESSAGES = {
    "authentication_failed": "HIVE authentication failed. Check the host API key.",
    "invalid_response": "HIVE returned an invalid detection result.",
    "media_rejected": "HIVE could not analyze this attachment.",
    "provider_unavailable": "HIVE is temporarily unavailable.",
    "rate_limited": "HIVE is currently rate-limiting requests.",
    "response_too_large": "HIVE returned an unexpectedly large result.",
    "timeout": "HIVE analysis timed out.",
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


def _read_aloud_mode_label(mode: str | None) -> str:
    if mode == ReadAloudMode.SKIP_DURING_MUSIC.value:
        return "Skip messages while music is playing"
    return "Read every message · duck music automatically"


def _speech_voice_label(runtime: SimajilordRuntime) -> str:
    settings = runtime.settings
    if settings.tts_provider == "voicevox":
        return f"VOICEVOX · style {settings.voicevox_speaker_id}"
    return f"macOS · {settings.tts_voice}"


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


def _search_candidate_line(candidate: AudioSearchItem) -> str:
    title = discord.utils.escape_markdown(candidate.title)
    source = (
        f"\n{discord.utils.escape_markdown(candidate.uploader)}"
        if candidate.uploader
        else ""
    )
    return f"[{title}]({candidate.reference}){source}"


def music_added_embed(response: AudioPlayResponse) -> discord.Embed:
    if response.playback_state == "playing":
        title = "Now playing"
        playback = "Playing now"
    elif response.playback_state == "waiting_for_voice":
        title = "Queued · waiting for voice"
        playback = "Starts automatically when the requester joins voice."
    else:
        title = "Added to queue"
        playback = f"Up next · #{response.queue_position}"
    fields = [
        EmbedField("Status", playback, inline=False),
        EmbedField("Length", _duration(response.duration_seconds)),
        EmbedField("Requested by", _requester(response.requested_by_name)),
    ]
    if response.uploader:
        fields.append(EmbedField("Source", discord.utils.escape_markdown(response.uploader)))
    fields.append(
        EmbedField(
            "Voice",
            f"<#{response.destination_id}>"
            if response.destination_id
            else "Connects when the requester joins",
        )
    )
    embed = command_embed(
        title,
        description=f"### [{response.title}]({response.page_url})",
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )
    if response.thumbnail_url:
        embed.set_thumbnail(url=response.thumbnail_url)
    return embed


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
        if current.uploader:
            fields.append(
                EmbedField("Source", discord.utils.escape_markdown(current.uploader))
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
    embed = command_embed("Music", description=description, fields=tuple(fields))
    if response.current and response.current.thumbnail_url:
        embed.set_thumbnail(url=response.current.thumbnail_url)
    return embed


def music_search_embed(response: AudioSearchResponse) -> discord.Embed:
    fields = tuple(
        EmbedField(
            f"{index} · {_duration(candidate.duration_seconds)}",
            _search_candidate_line(candidate),
            inline=False,
        )
        for index, candidate in enumerate(response.candidates, start=1)
    )
    embed = command_embed(
        "Choose the right track",
        description=(
            "Several close matches were found for "
            f"**{discord.utils.escape_markdown(response.query)}**. "
            "Choose once; the durable listening history helps select it automatically next time."
        ),
        fields=fields,
        tone=EmbedTone.WARNING,
    )
    if response.candidates and response.candidates[0].thumbnail_url:
        embed.set_thumbnail(url=response.candidates[0].thumbnail_url)
    return embed


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


def _safe_markdown_url(value: str) -> str:
    return value.replace("(", "%28").replace(")", "%29")


def _web_source_text(index: int, source: WebSource) -> str:
    title = discord.utils.escape_markdown(source.title[:160])
    host = discord.utils.escape_markdown(source.host[:100])
    snippet = discord.utils.escape_markdown(source.snippet[:240])
    line = f"`{index:02d}` **[{title}]({_safe_markdown_url(source.url)})**\n{host}"
    return f"{line} · {snippet}" if snippet else line


def web_search_embed(response: WebSearchResponse) -> discord.Embed:
    if not response.sources:
        return command_embed(
            "No search results",
            description=(
                f"No sources were found for **{discord.utils.escape_markdown(response.query)}**. "
                "The local provider may be degraded; the request is already logged, so "
                "repeating it immediately is unlikely to help."
            ),
            fields=(
                EmbedField("Backend", response.backend),
                EmbedField("Provider warnings", str(len(response.warnings))),
            ),
            tone=EmbedTone.WARNING,
        )
    lines: list[str] = []
    used = 0
    for index, source in enumerate(response.sources, start=1):
        line = _web_source_text(index, source)
        if used + len(line) + 2 > 3_700:
            break
        lines.append(line)
        used += len(line) + 2
    coverage = (
        f"{response.candidate_count} candidates · {len(response.sources)} shown"
    )
    if response.maybe_more:
        coverage += " · broader set retained"
    fields = [EmbedField("Coverage", coverage, inline=False)]
    if response.warnings:
        fields.append(
            EmbedField(
                "Provider health",
                f"{len(response.warnings)} upstream source(s) reported a problem.",
                inline=False,
            )
        )
    return command_embed(
        "Search results",
        description="\n\n".join(lines),
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )


def web_fetch_embed(response: WebFetchResponse) -> discord.Embed:
    excerpt = discord.utils.escape_markdown(response.text[:3_500])
    if not excerpt:
        excerpt = "No readable text was found in this chunk."
    fields = [
        EmbedField("Source", f"[Open page]({_safe_markdown_url(response.url)})"),
        EmbedField("Type", response.content_type),
        EmbedField(
            "Text",
            f"{response.offset + len(response.text):,} / {response.total_characters:,} characters",
        ),
    ]
    if response.next_offset is not None:
        fields.append(EmbedField("Next offset", str(response.next_offset)))
    return command_embed(
        response.title[:256],
        description=excerpt,
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )


class WebFetchContinueView(discord.ui.View):
    """Continue a bounded Fetch result without making the user re-enter its URL."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        response: WebFetchResponse,
    ) -> None:
        super().__init__(timeout=600)
        self.runtime = runtime
        self.url = response.url
        self.next_offset = response.next_offset

    @discord.ui.button(
        label="Continue",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:web:fetch:continue",
    )
    async def continue_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[WebFetchContinueView],
    ) -> None:
        if self.next_offset is None:
            await interaction.response.edit_message(view=None)
            return
        await interaction.response.defer()
        try:
            response = cast(
                WebFetchResponse,
                await self.runtime.registry.invoke(
                    "web.fetch",
                    WebFetchRequest(
                        url=self.url,
                        offset=self.next_offset,
                        max_characters=3_500,
                    ),
                    invocation_context(interaction),
                ),
            )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            await interaction.edit_original_response(
                embed=web_fetch_embed(response),
                view=view,
            )
        except Exception as exc:
            await interaction.followup.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                ),
                ephemeral=True,
            )


def _web_match_text(index: int, match: WebTextMatch) -> str:
    before = discord.utils.escape_markdown(match.before)
    found = discord.utils.escape_markdown(match.match)
    after = discord.utils.escape_markdown(match.after)
    return f"`{index:02d}` …{before}**{found}**{after}…"


def web_find_embed(response: WebFindResponse) -> discord.Embed:
    if not response.matches:
        description = (
            f"No matches for **{discord.utils.escape_markdown(response.pattern)}** "
            f"were found in [{discord.utils.escape_markdown(response.title)}]"
            f"({_safe_markdown_url(response.url)})."
        )
        return command_embed(
            "No page matches",
            description=description,
            tone=EmbedTone.WARNING,
        )
    lines = tuple(
        _web_match_text(index, match)
        for index, match in enumerate(response.matches, start=1)
    )
    return command_embed(
        f"Found in {response.title}"[:256],
        description="\n\n".join(lines)[:3_800],
        fields=(
            EmbedField(
                "Matches",
                f"{response.total_matches} total · {len(response.matches)} shown",
            ),
            EmbedField("Source", f"[Open page]({_safe_markdown_url(response.url)})"),
        ),
        tone=EmbedTone.SUCCESS,
    )


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


async def _enqueue_interaction_track(
    runtime: SimajilordRuntime,
    interaction: discord.Interaction,
    *,
    reference: str,
    requested_by_name: str,
) -> AudioPlayResponse:
    return cast(
        AudioPlayResponse,
        await runtime.registry.invoke(
            "discord.play_audio",
            AudioPlayRequest(
                reference=reference,
                requested_by_name=requested_by_name,
            ),
            invocation_context(interaction),
        ),
    )


class MusicCandidateButton(discord.ui.Button["MusicSearchChoiceView"]):
    def __init__(self, index: int, candidate: AudioSearchItem, token: str) -> None:
        source = candidate.uploader or candidate.title
        label = f"{index + 1} · {source}"
        super().__init__(
            label=label[:80],
            style=(
                discord.ButtonStyle.primary
                if index == 0
                else discord.ButtonStyle.secondary
            ),
            custom_id=f"simajilord:music:choice:{token}:{index}",
            row=0,
        )
        self.index = index

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.view is not None:
            await self.view.choose(interaction, self.index)


class MusicSearchChoiceView(discord.ui.View):
    """One-click disambiguation used only when zero-click selection is unsafe."""

    def __init__(
        self,
        bot: commands.Bot,
        runtime: SimajilordRuntime,
        response: AudioSearchResponse,
        *,
        requester_id: int,
        requester_name: str,
    ) -> None:
        super().__init__(timeout=90)
        self.bot = bot
        self.runtime = runtime
        self.search = response
        self.requester_id = requester_id
        self.requester_name = requester_name
        self.message: discord.Message | None = None
        self._lock = asyncio.Lock()
        self._selected = False
        token = secrets.token_hex(5)
        for index, candidate in enumerate(response.candidates[:5]):
            self.add_item(MusicCandidateButton(index, candidate, token))

    async def choose(self, interaction: discord.Interaction, index: int) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                embed=command_embed(
                    "This choice belongs to another listener",
                    description="Run `/play` to start your own request.",
                    tone=EmbedTone.WARNING,
                ),
                ephemeral=True,
            )
            return
        async with self._lock:
            if self._selected:
                await interaction.response.send_message(
                    embed=command_embed(
                        "Track already selected",
                        description="The request is already being processed.",
                        tone=EmbedTone.WARNING,
                    ),
                    ephemeral=True,
                )
                return
            try:
                candidate = self.search.candidates[index]
            except IndexError:
                await interaction.response.send_message(
                    embed=command_embed(
                        "Selection expired",
                        description="Run `/play` again to refresh the results.",
                        tone=EmbedTone.ERROR,
                    ),
                    ephemeral=True,
                )
                return
            self._selected = True
            await interaction.response.defer(thinking=True)
            self._set_disabled(True)
            await interaction.edit_original_response(view=self)
            try:
                response = await _enqueue_interaction_track(
                    self.runtime,
                    interaction,
                    reference=candidate.reference,
                    requested_by_name=self.requester_name,
                )
                await interaction.edit_original_response(
                    embed=music_added_embed(response),
                    view=MusicControlsView(self.runtime),
                )
                self.stop()
            except Exception as exc:
                self._selected = False
                self._set_disabled(False)
                await interaction.edit_original_response(view=self)
                await send_error(interaction, exc)

    async def on_timeout(self) -> None:
        self._set_disabled(True)
        if self.message is not None:
            with suppress(discord.DiscordException):
                await self.message.edit(view=self)

    def _set_disabled(self, disabled: bool) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = disabled


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
        origin_resource_id=(
            str(interaction.channel_id) if interaction.channel_id is not None else None
        ),
    )


def prefix_context(context: BotContext) -> InvocationContext:
    return InvocationContext(
        actor_id=str(context.author.id),
        workspace_id=str(context.guild.id) if context.guild else None,
        transport="discord",
        request_id=str(context.message.id),
        origin_resource_id=str(context.channel.id),
    )


def error_message(error: Exception) -> str:
    if isinstance(error, MediaError):
        return _MEDIA_ERROR_MESSAGES.get(error.category, _MEDIA_ERROR_MESSAGES["unknown"])
    if isinstance(error, WebError):
        return _WEB_ERROR_MESSAGES.get(
            error.category,
            "The web operation could not be completed.",
        )
    if isinstance(error, ModerationError):
        return _MODERATION_ERROR_MESSAGES.get(
            error.category,
            "HIVE analysis could not be completed.",
        )
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


async def edit_deferred_error(
    interaction: discord.Interaction,
    error: Exception,
) -> None:
    await interaction.edit_original_response(
        embed=command_embed(
            "Command failed",
            description=error_message(error),
            tone=EmbedTone.ERROR,
        )
    )


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
                    "adapter; media, speech, audio policy, and agent decisions live "
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
                        f"{response.audio_session_count} active\n"
                        f"Speech: **{response.speech_provider.upper()} "
                        f"{response.speech_voice}**",
                    ),
                    EmbedField(
                        "Web",
                        "Search: "
                        f"**{'Ready' if response.web_search_ready else 'Degraded'}**\n"
                        f"Backend: **{response.web_search_backend}**",
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

    async def _send_play(self, interaction: discord.Interaction, reference: str) -> None:
        try:
            await interaction.response.defer(thinking=True)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            selected_reference = reference
            if "://" not in reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=reference, limit=5),
                        invocation_context(interaction),
                    ),
                )
                if search.selection_required:
                    view = MusicSearchChoiceView(
                        self.bot,
                        self.runtime,
                        search,
                        requester_id=interaction.user.id,
                        requester_name=interaction.user.display_name,
                    )
                    message = await interaction.followup.send(
                        embed=music_search_embed(search),
                        view=view,
                        wait=True,
                    )
                    view.message = message
                    return
                if search.selected_index is None:
                    raise UserError("audio.search_empty")
                selected_reference = search.candidates[search.selected_index].reference
            response = await _enqueue_interaction_track(
                self.runtime,
                interaction,
                reference=selected_reference,
                requested_by_name=interaction.user.display_name,
            )
            await interaction.followup.send(
                embed=music_added_embed(response),
                view=MusicControlsView(self.runtime),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="play",
        description="Play a public media URL or search for a track.",
    )
    @app_commands.describe(reference="Paste a public media URL or type a track name")
    async def quick_play(self, interaction: discord.Interaction, reference: str) -> None:
        await self._send_play(interaction, reference)

    @music.command(
        name="play",
        description="Play a public media URL or search for a track.",
    )
    @app_commands.describe(reference="Paste a public media URL or type a track name")
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


_READ_ALOUD_CHANNEL_TYPES = [
    discord.ChannelType.text,
    discord.ChannelType.news,
    discord.ChannelType.voice,
    discord.ChannelType.stage_voice,
    discord.ChannelType.news_thread,
    discord.ChannelType.public_thread,
    discord.ChannelType.private_thread,
]


class ReadAloudChannelSelect(discord.ui.ChannelSelect[discord.ui.View]):
    """Commit a bounded set of conversation channels in one interaction."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        destination_id: int,
        default_values: tuple[discord.abc.GuildChannel | discord.Thread, ...],
    ) -> None:
        self.runtime = runtime
        self.requester_id = requester_id
        self.destination_id = destination_id
        super().__init__(
            custom_id="simajilord:readaloud:channels",
            channel_types=_READ_ALOUD_CHANNEL_TYPES,
            placeholder="Select conversation channels",
            min_values=1,
            max_values=25,
            default_values=default_values,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the person who ran `/join` can change this selection.",
                ephemeral=True,
            )
            return
        configured: ReadAloudResponse | None = None
        try:
            await interaction.response.defer()
            source_ids = tuple(str(channel.id) for channel in self.values)
            configured = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.ADD_SOURCES,
                        text_channel_ids=source_ids,
                        audio_destination_id=str(self.destination_id),
                    ),
                    invocation_context(interaction),
                ),
            )
            await self.runtime.registry.invoke(
                "discord.connect_voice",
                DiscordConnectVoiceRequest(channel_id=str(self.destination_id)),
                invocation_context(interaction),
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Read-aloud ready",
                    description=(
                        "The selected conversations were added. New messages there "
                        "will be read automatically."
                    ),
                    fields=(
                        EmbedField(
                            "Conversation channels",
                            "\n".join(
                                f"<#{channel_id}>"
                                for channel_id in configured.text_channel_ids
                            ),
                            inline=False,
                        ),
                        EmbedField(
                            "Voice channel",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField("Connection", "Ready"),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
        except Exception as exc:
            if configured is None:
                await send_error(interaction, exc)
                return
            log.exception(
                "Read-aloud route was saved but eager voice connection failed "
                "guild=%s channel=%s",
                interaction.guild_id,
                self.destination_id,
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Read-aloud saved",
                    description=(
                        "The conversations are configured, but the voice connection "
                        "is not ready yet. The next message will retry automatically."
                    ),
                    fields=(
                        EmbedField(
                            "Voice channel",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField("Connection", error_message(exc)),
                    ),
                    tone=EmbedTone.WARNING,
                ),
                view=None,
            )


class ReadAloudChannelSelectView(discord.ui.View):
    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        destination_id: int,
        default_values: tuple[discord.abc.GuildChannel | discord.Thread, ...],
    ) -> None:
        super().__init__(timeout=300)
        self.add_item(
            ReadAloudChannelSelect(
                runtime,
                requester_id=requester_id,
                destination_id=destination_id,
                default_values=default_values,
            )
        )


class ReadAloudCog(commands.Cog):
    readaloud = app_commands.Group(
        name="readaloud",
        description="Configure automatic Discord message read-aloud.",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @app_commands.command(
        name="join",
        description="Read this channel aloud in the voice channel you joined.",
    )
    async def join(self, interaction: discord.Interaction) -> None:
        try:
            member = interaction.user
            source = interaction.channel
            if not isinstance(member, discord.Member):
                raise UserError("Use this command in a server.")
            if not isinstance(
                source,
                (
                    discord.TextChannel,
                    discord.Thread,
                    discord.VoiceChannel,
                    discord.StageChannel,
                ),
            ):
                raise UserError("Use this command in a server message channel.")
            destination = member.voice.channel if member.voice is not None else None
            if not isinstance(destination, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("Join a voice channel first.")
            default_channels: list[discord.abc.GuildChannel | discord.Thread] = []
            selected_source = member.guild.get_channel_or_thread(source.id)
            if selected_source is not None:
                default_channels.append(selected_source)
            view = ReadAloudChannelSelectView(
                self.runtime,
                requester_id=member.id,
                destination_id=destination.id,
                default_values=tuple(default_channels[:25]),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Add conversations",
                    description=(
                        "Select up to 25 text channels, threads, or voice-channel "
                        f"chats to add to {destination.mention}."
                    ),
                    fields=(
                        EmbedField("Current channel", source.mention),
                        EmbedField("Voice channel", destination.mention),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                ),
                view=view,
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="setup",
        description="Replace the route with one message channel and voice channel.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        text_channel="Messages from this text or voice channel are read automatically",
        voice_channel="Audio is played in this channel",
        mode="Queue speech, or skip it while music is active",
    )
    async def setup(
        self,
        interaction: discord.Interaction,
        text_channel: discord.TextChannel | discord.VoiceChannel | None = None,
        voice_channel: discord.VoiceChannel | None = None,
        mode: Literal["queue", "skip_during_music"] = "queue",
    ) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not member.guild_permissions.manage_guild:
                raise UserError("Manage Server permission is required.")
            selected_text = text_channel
            if selected_text is None and isinstance(
                interaction.channel,
                (discord.TextChannel, discord.VoiceChannel),
            ):
                selected_text = interaction.channel
            if selected_text is None:
                raise UserError("Choose a server text or voice channel.")
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
                    "discord.manage_read_aloud",
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
                        EmbedField("Conversation channel", selected_text.mention),
                        EmbedField("Voice channel", selected_voice.mention),
                        EmbedField("Mode", _read_aloud_mode_label(response.mode)),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
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
                    "discord.manage_read_aloud",
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
                        EmbedField(
                            "Conversation channels",
                            "\n".join(
                                f"<#{channel_id}>"
                                for channel_id in response.text_channel_ids
                            ),
                        ),
                        EmbedField(
                            "Voice channel",
                            f"<#{response.audio_destination_id}>",
                        ),
                        EmbedField("Mode", _read_aloud_mode_label(response.mode)),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="remove",
        description="Stop reading one conversation channel.",
    )
    @app_commands.describe(
        channel="Conversation channel to remove; defaults to this channel",
    )
    async def remove(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel | discord.VoiceChannel | None = None,
    ) -> None:
        try:
            selected = channel
            if selected is None and isinstance(
                interaction.channel,
                (discord.TextChannel, discord.VoiceChannel),
            ):
                selected = interaction.channel
            if selected is None:
                raise UserError("Choose a server text or voice channel.")
            response = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.REMOVE_SOURCE,
                        text_channel_id=str(selected.id),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Conversation removed",
                    description=(
                        "Read-aloud is now disabled."
                        if not response.enabled
                        else "Other configured conversation channels remain active."
                    ),
                    fields=(
                        EmbedField("Channel", selected.mention),
                        EmbedField(
                            "Remaining channels",
                            str(len(response.text_channel_ids)),
                        ),
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
                "discord.manage_read_aloud",
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
            await self.runtime.registry.invoke(
                "speech.speak",
                SpeechSpeakRequest(
                    text=message.content,
                    title=f"Message from {message.author.display_name}",
                ),
                InvocationContext(
                    actor_id=str(message.author.id),
                    workspace_id=workspace_id,
                    transport="discord",
                    request_id=f"read-aloud:{message.id}",
                ),
            )
        except Exception:
            log.exception(
                "Automatic read-aloud failed guild=%s channel=%s",
                message.guild.id,
                message.channel.id,
            )


class VoiceLifecycleCog(commands.Cog):
    """Keep voice presence aligned with listeners without losing the music queue."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
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
        route = self.runtime.read_aloud.get(workspace_id)
        joined_read_aloud_destination = (
            route is not None
            and after.channel is not None
            and str(after.channel.id) == route.audio_destination_id
        )
        session = self.runtime.audio.find(workspace_id)
        if session is None and joined_read_aloud_destination:
            session = self.runtime.audio.get_or_create(
                workspace_id,
                lambda: DiscordAudioOutput(self.bot, member.guild.id),
            )
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
                or joined_read_aloud_destination
            )
        )
        if joined_expected_channel and after.channel is not None:
            task = self._leave_tasks.pop(workspace_id, None)
            if task is not None:
                task.cancel()
            music_targets_another_channel = (
                session.has_music
                and session.destination_id is not None
                and str(after.channel.id) != session.destination_id
            )
            should_connect = (
                session.has_music or joined_read_aloud_destination
            ) and not music_targets_another_channel
            if should_connect and not session.output.connected:
                try:
                    await self.runtime.audio.connect(
                        workspace_id,
                        str(after.channel.id),
                    )
                    log.info(
                        "Prepared audio after a listener joined guild=%s "
                        "read_aloud=%s music=%s",
                        workspace_id,
                        joined_read_aloud_destination,
                        session.has_music,
                    )
                except Exception:
                    log.exception(
                        "Could not prepare audio after a listener joined guild=%s",
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


class WebCog(commands.Cog):
    """Discord presentation for the same web APIs available to the agent."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(
        name="search",
        description="Search the web through Simajilord's local metasearch provider.",
    )
    @app_commands.describe(
        query="Topic, question, or exact phrase to search for",
        depth="Quick, standard, or deep candidate collection",
    )
    async def search(
        self,
        interaction: discord.Interaction,
        query: str,
        depth: Literal["quick", "standard", "deep"] = "standard",
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebSearchResponse,
                await self.runtime.registry.invoke(
                    "web.search",
                    WebSearchRequest(
                        query=query,
                        depth=SearchDepth(depth),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(embed=web_search_embed(response))
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @app_commands.command(
        name="fetch",
        description="Open a public page and show a bounded readable excerpt.",
    )
    @app_commands.describe(
        url="Public HTTP or HTTPS page to open",
        offset="Optional text position to resume from",
    )
    async def fetch(
        self,
        interaction: discord.Interaction,
        url: str,
        offset: app_commands.Range[int, 0, 40_000] = 0,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebFetchResponse,
                await self.runtime.registry.invoke(
                    "web.fetch",
                    WebFetchRequest(
                        url=url,
                        offset=offset,
                        max_characters=3_500,
                    ),
                    invocation_context(interaction),
                ),
            )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            await interaction.edit_original_response(
                embed=web_fetch_embed(response),
                view=view,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @app_commands.command(
        name="find",
        description="Find a phrase inside one public web page.",
    )
    @app_commands.describe(
        url="Public HTTP or HTTPS page to inspect",
        phrase="Exact phrase to find, case-insensitively",
    )
    async def find(
        self,
        interaction: discord.Interaction,
        url: str,
        phrase: str,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = cast(
                WebFindResponse,
                await self.runtime.registry.invoke(
                    "web.find",
                    WebFindRequest(url=url, pattern=phrase),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(embed=web_find_embed(response))
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


def _percentage(score: float) -> str:
    return f"{score * 100:.1f}%"


_HIVE_SOURCE_LABELS = {
    "4o": "GPT-4o",
    "adobefirefly": "Adobe Firefly",
    "amused": "Amused",
    "aniportrait": "AniPortrait",
    "bagel": "Bagel",
    "bingimagecreator": "Bing Image Creator",
    "blip3o": "BLIP-3o",
    "bria": "Bria",
    "cogvideos": "CogVideo",
    "cogview": "CogView",
    "cosmos": "Cosmos",
    "dalle": "DALL-E",
    "deepfloyd": "DeepFloyd",
    "dmd2": "DMD2",
    "dreamid": "DreamID",
    "emu3": "Emu3",
    "flashvideo": "FlashVideo",
    "flux": "FLUX",
    "flux2": "FLUX.2",
    "gan": "GAN",
    "gemini": "Gemini",
    "gemini3": "Gemini 3",
    "glide": "GLIDE",
    "gptimage1_5": "GPT Image 1.5",
    "grok": "Grok",
    "grokimagine": "Grok Imagine",
    "haiper": "Haiper",
    "hailuo": "Hailuo",
    "hallo": "Hallo",
    "happyhorse": "HappyHorse",
    "hedra": "Hedra",
    "heygen": "HeyGen",
    "hidream": "HiDream",
    "higgsfield": "Higgsfield",
    "hunyuan": "Hunyuan",
    "ideogram": "Ideogram",
    "imagen": "Imagen",
    "imagen4": "Imagen 4",
    "imagineart": "ImagineArt",
    "infinity": "Infinity",
    "janus": "Janus",
    "kandinsky": "Kandinsky",
    "kling": "Kling",
    "krea": "Krea",
    "lcm": "LCM",
    "leonardo": "Leonardo",
    "liveportrait": "LivePortrait",
    "longcat": "LongCat",
    "ltx": "LTX",
    "lucid": "Lucid",
    "luma": "Luma",
    "luminagpt": "Lumina GPT",
    "mai": "MAI",
    "makeittalk": "MakeItTalk",
    "mcnet": "MCNet",
    "meta": "Meta",
    "midjourney": "Midjourney",
    "mochi": "Mochi",
    "moonvalley": "Moonvalley",
    "omnigen": "OmniGen",
    "other_image_generators": "Other image generator",
    "ovis": "Ovis",
    "personalive": "PersonaLive",
    "pika": "Pika",
    "pixart": "PixArt",
    "pixverse": "PixVerse",
    "pyramidflows": "Pyramid Flow",
    "qwen": "Qwen",
    "ray3": "Ray 3",
    "recraft": "Recraft",
    "reve": "Reve",
    "runway": "Runway",
    "sadtalker": "SadTalker",
    "sana": "Sana",
    "sanavideo": "Sana Video",
    "scail": "SCAIL",
    "sdxlinpaint": "SDXL Inpainting",
    "seedance": "Seedance",
    "seedance2": "Seedance 2",
    "seedream": "Seedream",
    "sora": "Sora",
    "sora2": "Sora 2",
    "stablecascade": "Stable Cascade",
    "stablediffusion": "Stable Diffusion",
    "stablediffusioninpaint": "Stable Diffusion Inpainting",
    "stablediffusionxl": "Stable Diffusion XL",
    "steadydancer": "SteadyDancer",
    "switti": "Switti",
    "titan": "Titan",
    "transpixar": "TransPixar",
    "var": "VAR",
    "veo3": "Veo 3",
    "vibe": "Vibe",
    "viduq2": "Vidu Q2",
    "vqdiffusion": "VQ Diffusion",
    "wan": "Wan",
    "wuerstchen": "Würstchen",
    "zimage": "Z-Image",
}


def _source_name(value: str) -> str:
    known = _HIVE_SOURCE_LABELS.get(value)
    if known is not None:
        return known
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())


def _likelihood(score: float, *, high_threshold: float) -> str:
    if score >= high_threshold:
        return "High"
    if score >= 0.5:
        return "Moderate"
    return "Low"


def synthetic_media_embed(
    response: SyntheticMediaAnalyzeResponse,
    *,
    attachment_url: str | None = None,
) -> discord.Embed:
    ai_likelihood = _likelihood(
        response.ai_generated_score,
        high_threshold=response.threshold,
    )
    deepfake_likelihood = (
        "High"
        if response.deepfake_likely
        else _likelihood(
            response.deepfake_score,
            high_threshold=response.threshold,
        )
    )
    severity_labels = {ai_likelihood, deepfake_likelihood}
    if "High" in severity_labels:
        tone = EmbedTone.ERROR
    elif "Moderate" in severity_labels:
        tone = EmbedTone.WARNING
    else:
        tone = EmbedTone.SUCCESS
    media_label = response.modality.value
    conclusion = (
        f"**{ai_likelihood} AI-generated {media_label} likelihood**\n"
        f"{deepfake_likelihood} deepfake likelihood"
    )
    fields = [
        EmbedField(
            "AI-generated visual",
            f"**{_percentage(response.ai_generated_score)}** · {ai_likelihood}",
        ),
        EmbedField(
            "Deepfake",
            f"**{_percentage(response.deepfake_score)}** · {deepfake_likelihood}",
        ),
    ]
    if (
        response.top_source is not None
        and response.top_source_score >= 0.5
    ):
        fields.append(
            EmbedField(
                "Likely generator",
                (
                    f"**{_source_name(response.top_source)}** · "
                    f"{_percentage(response.top_source_score)}"
                ),
                inline=False,
            )
        )
    cache_line = (
        "Reused cached result · no HIVE API request used"
        if response.cached
        else "New HIVE API analysis"
    )
    sample_name = {
        "image": "image",
        "video": "frames",
    }[response.modality.value]
    fields.extend(
        (
            EmbedField(
                "Analysis",
                (
                    f"{response.sample_count} {sample_name} · {cache_line}"
                ),
                inline=False,
            ),
            EmbedField(
                "Daily HIVE API budget",
                (
                    f"**{response.quota_used} / {response.quota_limit} used** · "
                    f"{response.quota_remaining} remaining · "
                    f"resets <t:{response.quota_reset_at_epoch}:R>"
                ),
                inline=False,
            ),
        )
    )
    embed = command_embed(
        "HIVE analysis",
        description=conclusion,
        fields=tuple(fields),
        tone=tone,
    )
    if attachment_url is not None and response.content_type.startswith("image/"):
        embed.set_thumbnail(url=attachment_url)
    model_label = response.model.removeprefix("hive/")
    embed.set_footer(text=f"Powered by HIVE Moderation · {model_label}")
    return embed


class ModerationCog(commands.Cog):
    """Discord upload adapter for the shared synthetic-media capability."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @app_commands.command(
        name="detectai",
        description="Analyze an image or video with HIVE AI-content detection.",
    )
    @app_commands.describe(media="Image or video attachment to analyze with HIVE")
    async def detectai(
        self,
        interaction: discord.Interaction,
        media: discord.Attachment,
    ) -> None:
        await interaction.response.defer(thinking=True)
        try:
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            content = await media.read(use_cached=True)
            response = cast(
                SyntheticMediaAnalyzeResponse,
                await self.runtime.registry.invoke(
                    "moderation.detect_synthetic_media",
                    SyntheticMediaAnalyzeRequest(
                        filename=media.filename,
                        content_type=media.content_type,
                        content=content,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=synthetic_media_embed(
                    response,
                    attachment_url=media.url,
                )
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class DownloadCog(commands.Cog):
    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._last_request: dict[int, float] = {}

    @app_commands.command(
        name="download",
        description="Save media from a provider-supported public HTTPS URL.",
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
            await interaction.response.defer(thinking=True)
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
                    "Media saved",
                    description=(
                        f"### {discord.utils.escape_markdown(response.title)}\n"
                        f"[Open original]({response.source_url})"
                    ),
                    fields=(
                        EmbedField(
                            "Size",
                            f"{response.size_bytes / 1_000_000:.1f} MB",
                        ),
                        EmbedField("Type", media_type.title()),
                        EmbedField("Format", response.path.suffix.lstrip(".").upper()),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                file=discord.File(response.path),
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


def discord_conversation_id(
    *,
    guild_id: int | None,
    channel_id: int,
    grants: frozenset[str] = frozenset(),
) -> str:
    """Map one Discord channel and capability profile to one agent conversation."""

    scope = f"guild:{guild_id}" if guild_id is not None else "direct"
    base = f"discord:{scope}:channel:{channel_id}"
    if not grants:
        return base
    profile = "+".join(sorted(grants))
    return f"{base}:profile:{profile}"


def _agent_grants(
    runtime: SimajilordRuntime,
    *,
    actor_id: str,
) -> frozenset[str]:
    settings = runtime.settings
    grants: set[str] = {AGENT_AUDIO_GRANT}
    grants.add(AGENT_MESSAGE_GRANT)
    if settings.agent_file_sandbox_enabled and runtime.files is not None:
        grants.add(AGENT_FILE_GRANT)
    web_access = settings.agent_web_search_access
    if web_access is AgentFeatureAccess.EVERYONE or (
        web_access is AgentFeatureAccess.ADMINS
        and actor_id in settings.agent_admin_user_ids
    ):
        grants.add(AGENT_WEB_GRANT)
    if runtime.moderation.provider is not None:
        grants.add(AGENT_MODERATION_GRANT)
    image_access = settings.image_generation_access
    if runtime.image.provider is not None and (
        image_access is AgentFeatureAccess.EVERYONE
        or (
            image_access is AgentFeatureAccess.ADMINS
            and actor_id in settings.agent_admin_user_ids
        )
    ):
        grants.add(AGENT_IMAGE_GRANT)
    return frozenset(grants)


def _discord_message_chunks(content: str, *, maximum: int = 1_900) -> tuple[str, ...]:
    """Bound Discord output without asking the model to repeat a long answer."""

    text = content.strip()
    if not text:
        return ()
    chunks: list[str] = []
    while text:
        if len(text) <= maximum:
            chunks.append(text)
            break
        boundary = text.rfind("\n", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = text.rfind(" ", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = maximum
        chunks.append(text[:boundary].rstrip())
        text = text[boundary:].lstrip()
    return tuple(chunk for chunk in chunks if chunk)


def _agent_message_groups(content: str) -> tuple[str, ...]:
    """Convert every explicit boundary into an independent Discord post."""

    groups = content.split(AGENT_MESSAGE_BREAK)
    messages: list[str] = []
    for group in groups:
        messages.extend(_discord_message_chunks(group))
    return tuple(messages)


def _agent_error_text(error: Exception) -> str:
    if isinstance(error, AgentBusyError):
        return "Simajilord's bounded AI turn queue is full. Try again shortly."
    if isinstance(error, AgentRateLimitError):
        if error.retry_after_seconds is not None:
            return (
                "The local AI usage guard is cooling down. "
                f"Try again in {_retry_after_text(error.retry_after_seconds)}."
            )
        return "The local AI usage guard is cooling down. Try again later."
    if isinstance(error, AgentUnavailableError):
        return "Simajilord's AI runtime is unavailable on this host."
    return "Simajilord could not complete that AI turn."


def _retry_after_text(total_seconds: int) -> str:
    seconds = max(1, total_seconds)
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return " ".join(parts)


_AGENT_PROGRESS_MESSAGES = {
    AgentProgressStage.QUEUED: "Waiting for the current AI turn to finish…",
    AgentProgressStage.STARTING: "Understanding the request…",
    AgentProgressStage.READING_DISCORD: "Checking the relevant Discord context…",
    AgentProgressStage.SEARCHING_WEB: "Searching the web…",
    AgentProgressStage.COMPUTING: "Running a bounded calculation…",
    AgentProgressStage.ANALYZING_MEDIA: "Analyzing the attachment with HIVE…",
    AgentProgressStage.GENERATING_IMAGE: "Preparing the local image generation job…",
    AgentProgressStage.USING_AUDIO: "Preparing the server audio session…",
    AgentProgressStage.PREPARING_RESPONSE: "Preparing the response…",
}


class _AgentProgressMessage:
    """Coalesce real execution stages into one low-frequency Discord message."""

    def __init__(
        self,
        source: discord.Message,
        *,
        initial_delay_seconds: float = 1.0,
        minimum_update_seconds: float = 2.5,
    ) -> None:
        self.source = source
        self.initial_delay_seconds = initial_delay_seconds
        self.minimum_update_seconds = minimum_update_seconds
        self.message: discord.Message | None = None
        self._latest: AgentProgressStage | None = None
        self._published: AgentProgressStage | None = None
        self._last_update = 0.0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def update(self, stage: AgentProgressStage) -> None:
        if self._closed:
            return
        self._latest = stage
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._flush_later(),
                name=f"simajilord-agent-progress-{self.source.id}",
            )

    async def finish(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        messages = _agent_message_groups(content)
        if not messages:
            return
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    messages[0],
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=messages[0],
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        for message in messages[1:]:
            await self.source.channel.send(
                message,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def fail(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    content,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=content,
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

    async def _flush_later(self) -> None:
        try:
            if self.message is None:
                delay = self.initial_delay_seconds
            else:
                elapsed = time.monotonic() - self._last_update
                delay = max(0.0, self.minimum_update_seconds - elapsed)
            if delay:
                await asyncio.sleep(delay)
            if self._closed or self._latest is None or self._latest is self._published:
                return
            stage = self._latest
            embed = command_embed(
                "Working",
                description=_AGENT_PROGRESS_MESSAGES[stage],
                tone=EmbedTone.INFO,
            )
            async with self._lock:
                if self._closed:
                    return
                if self.message is None:
                    self.message = await self.source.reply(
                        embed=embed,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await self.message.edit(content=None, embed=embed)
                self._published = stage
                self._last_update = time.monotonic()
        except asyncio.CancelledError:
            raise
        except discord.DiscordException:
            log.exception("Could not publish agent progress message.")
        finally:
            self._task = None
            if (
                not self._closed
                and self._latest is not None
                and self._latest is not self._published
            ):
                self._task = asyncio.create_task(
                    self._flush_later(),
                    name=f"simajilord-agent-progress-{self.source.id}",
                )

    async def _cancel_pending(self) -> None:
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class AgentCog(commands.Cog):
    """Wake one shared agent conversation only for explicit mentions."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        agent = self.runtime.agent
        bot_user = self.bot.user
        if (
            agent is None
            or bot_user is None
            or message.author.bot
            or message.webhook_id is not None
            or message.mention_everyone
            or bot_user not in message.mentions
        ):
            return
        if (
            message.guild is None
            or str(message.guild.id) not in self.runtime.settings.agent_allowed_guild_ids
        ):
            return
        member = (
            message.author
            if isinstance(message.author, discord.Member)
            else message.guild.get_member(message.author.id)
        )
        resource_ids = agent_readable_channel_ids(
            message.guild,
            member,
            trusted_guild=(
                str(message.guild.id)
                in self.runtime.settings.agent_trusted_guild_ids
            ),
            trigger_channel_id=message.channel.id,
        )
        if str(message.channel.id) not in resource_ids:
            log.info(
                "Mention agent turn rejected by channel scope guild=%s channel=%s "
                "channel_type=%s actor=%s",
                message.guild.id,
                message.channel.id,
                type(message.channel).__name__,
                message.author.id,
            )
            return
        actor_id = str(message.author.id)
        grants = _agent_grants(self.runtime, actor_id=actor_id)
        request = AgentRequest(
            conversation_id=discord_conversation_id(
                guild_id=message.guild.id if message.guild else None,
                channel_id=message.channel.id,
                grants=grants,
            ),
            event_id=f"discord:message:{message.id}",
            trigger=AgentTrigger.MENTION,
            actor_id=actor_id,
            actor_name=message.author.display_name,
            workspace_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            message_id=str(message.id),
            occurred_at=message.created_at,
            resource_ids=resource_ids,
            grants=grants,
        )
        progress = _AgentProgressMessage(message)
        try:
            async with message.channel.typing():
                response = await agent.respond(
                    request,
                    on_progress=progress.update,
                )
            await progress.finish(response.content)
        except Exception as exc:
            log.exception("Mention agent turn failed message=%s", message.id)
            await progress.fail(_agent_error_text(exc))


class ObservationCog(commands.Cog):
    """Feed Discord changes into the platform event stream for agent reconciliation."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        bot_user = self.bot.user
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
                "content_length": len(message.content),
                "mentions_bot": bot_user in message.mentions if bot_user else False,
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


class AgentAutonomyCog(commands.Cog):
    """Bounded default-off patrol over metadata-only Discord events."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._task: asyncio.Task[None] | None = None

    async def cog_load(self) -> None:
        settings = self.runtime.settings
        if (
            self.runtime.agent is not None
            and settings.agent_autonomy_enabled
            and bool(settings.agent_autonomy_guild_ids)
            and self._task is None
        ):
            self._task = asyncio.create_task(
                self._run(),
                name="simajilord-agent-autonomy",
            )

    async def cog_unload(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        settings = self.runtime.settings
        cursor = await self.runtime.journal.latest_sequence()
        await self.runtime.journal.append(
            kind="agent.autonomy.started",
            transport="agent",
            payload={
                "interval_seconds": settings.agent_autonomy_interval_seconds,
                "max_runs": settings.agent_autonomy_max_runs,
                "candidate_limit": settings.agent_autonomy_candidate_limit,
            },
        )
        completed_runs = 0
        try:
            while completed_runs < settings.agent_autonomy_max_runs:
                await asyncio.sleep(settings.agent_autonomy_interval_seconds)
                records = await self.runtime.journal.recent(
                    after_sequence=cursor,
                    limit=min(1_000, settings.agent_autonomy_candidate_limit * 20),
                )
                if records:
                    cursor = records[-1].sequence
                candidates = [
                    record
                    for record in records
                    if record.kind == "discord.message.created"
                    and record.actor_id is not None
                    and not bool(record.payload.get("author_is_bot"))
                    and not bool(record.payload.get("mentions_bot"))
                    and isinstance(record.payload.get("message_id"), str)
                    and isinstance(record.payload.get("channel_id"), str)
                    and record.workspace_id in settings.agent_autonomy_guild_ids
                ][-settings.agent_autonomy_candidate_limit :]
                completed_runs += 1
                await self.runtime.journal.append(
                    kind="agent.autonomy.checked",
                    transport="agent",
                    payload={
                        "run": completed_runs,
                        "candidate_count": len(candidates),
                        "model_called": bool(candidates),
                    },
                )
                if candidates:
                    await self._inspect(candidates[-1])
        except asyncio.CancelledError:
            raise
        finally:
            await self.runtime.journal.append(
                kind="agent.autonomy.stopped",
                transport="agent",
                payload={"completed_runs": completed_runs},
            )

    async def _inspect(self, record: EventRecord) -> None:
        agent = self.runtime.agent
        if agent is None:
            return
        payload = record.payload
        channel_id = str(payload["channel_id"])
        message_id = str(payload["message_id"])
        workspace_id = record.workspace_id
        occurred_at = record.occurred_at
        guild = self.bot.get_guild(int(workspace_id)) if workspace_id else None
        if guild is None:
            return
        resource_ids = agent_readable_channel_ids(
            guild,
            None,
            trusted_guild=True,
            trigger_channel_id=int(channel_id),
        )
        if channel_id not in resource_ids:
            return
        grants = _agent_grants(
            self.runtime,
            actor_id=AGENT_AUTONOMY_ACTOR_ID,
        )
        request = AgentRequest(
            conversation_id=discord_conversation_id(
                guild_id=int(workspace_id) if workspace_id else None,
                channel_id=int(channel_id),
                grants=grants,
            ),
            event_id=f"autonomy:event:{record.sequence}",
            trigger=AgentTrigger.AUTONOMOUS,
            actor_id=AGENT_AUTONOMY_ACTOR_ID,
            actor_name="Simajilord autonomy",
            workspace_id=workspace_id,
            channel_id=channel_id,
            message_id=message_id,
            occurred_at=occurred_at if isinstance(occurred_at, datetime) else datetime.now(UTC),
            resource_ids=resource_ids,
            grants=grants,
        )
        try:
            response = await agent.respond(request)
            if response.content.strip() == AGENT_NO_ACTION_CONTENT:
                return
            channel = self.bot.get_channel(int(channel_id))
            if not isinstance(
                channel,
                (
                    discord.TextChannel,
                    discord.Thread,
                    discord.VoiceChannel,
                    discord.StageChannel,
                ),
            ):
                return
            messages = _agent_message_groups(response.content)
            if not messages:
                return
            try:
                target = await channel.fetch_message(int(message_id))
            except discord.DiscordException:
                target = None
            if target is not None:
                await target.reply(
                    messages[0],
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await channel.send(
                    messages[0],
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            for message in messages[1:]:
                await channel.send(
                    message,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        except (AgentBusyError, AgentRateLimitError):
            log.info("Autonomous agent check skipped by local budget.")
        except Exception:
            log.exception("Autonomous agent check failed message=%s", message_id)


class PrefixCog(commands.Cog):
    """Prefix presentation for the same APIs used by slash commands."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

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

    @commands.command(name="search")
    async def search(self, context: BotContext, *, query: str) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebSearchResponse,
                    await self.runtime.registry.invoke(
                        "web.search",
                        WebSearchRequest(query=query),
                        prefix_context(context),
                    ),
                )
            await context.send(embed=web_search_embed(response))
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="fetch")
    async def fetch(self, context: BotContext, url: str, offset: int = 0) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebFetchResponse,
                    await self.runtime.registry.invoke(
                        "web.fetch",
                        WebFetchRequest(
                            url=url,
                            offset=offset,
                            max_characters=3_500,
                        ),
                        prefix_context(context),
                    ),
                )
            view = (
                WebFetchContinueView(self.runtime, response)
                if response.next_offset is not None
                else None
            )
            if view is None:
                await context.send(embed=web_fetch_embed(response))
            else:
                await context.send(
                    embed=web_fetch_embed(response),
                    view=view,
                )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="find")
    async def find(
        self,
        context: BotContext,
        url: str,
        *,
        phrase: str,
    ) -> None:
        try:
            async with context.typing():
                response = cast(
                    WebFindResponse,
                    await self.runtime.registry.invoke(
                        "web.find",
                        WebFindRequest(url=url, pattern=phrase),
                        prefix_context(context),
                    ),
                )
            await context.send(embed=web_find_embed(response))
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Command failed",
                    description=error_message(exc),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="detectai")
    async def detectai(self, context: BotContext) -> None:
        try:
            if not context.message.attachments:
                raise UserError("moderation.media_empty")
            media = context.message.attachments[0]
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            async with context.typing():
                response = cast(
                    SyntheticMediaAnalyzeResponse,
                    await self.runtime.registry.invoke(
                        "moderation.detect_synthetic_media",
                        SyntheticMediaAnalyzeRequest(
                            filename=media.filename,
                            content_type=media.content_type,
                            content=await media.read(use_cached=True),
                        ),
                        prefix_context(context),
                    ),
                )
            await context.send(
                embed=synthetic_media_embed(
                    response,
                    attachment_url=media.url,
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

    @commands.command(name="play")
    async def play(self, context: BotContext, *, reference: str) -> None:
        try:
            guild_id = context.guild.id if context.guild else None
            _discord_audio_session(self.bot, self.runtime, guild_id)
            selected_reference = reference
            if "://" not in reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=reference, limit=5),
                        prefix_context(context),
                    ),
                )
                if search.selection_required:
                    view = MusicSearchChoiceView(
                        self.bot,
                        self.runtime,
                        search,
                        requester_id=context.author.id,
                        requester_name=context.author.display_name,
                    )
                    message = await context.send(
                        embed=music_search_embed(search),
                        view=view,
                    )
                    view.message = message
                    return
                if search.selected_index is None:
                    raise UserError("audio.search_empty")
                selected_reference = search.candidates[search.selected_index].reference
            response = cast(
                AudioPlayResponse,
                await self.runtime.registry.invoke(
                    "discord.play_audio",
                    AudioPlayRequest(
                        reference=selected_reference,
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
    await bot.add_cog(VoiceLifecycleCog(bot, runtime))
    await bot.add_cog(WebCog(runtime))
    await bot.add_cog(ModerationCog(runtime))
    await bot.add_cog(DownloadCog(runtime))
    await bot.add_cog(UtilityCog(runtime))
    await bot.add_cog(DiscordInfoCog(runtime))
    await bot.add_cog(DiscordActionCog(runtime))
    await bot.add_cog(AgentCog(bot, runtime))
    await bot.add_cog(ObservationCog(bot, runtime))
    await bot.add_cog(AgentAutonomyCog(bot, runtime))
    await bot.add_cog(PrefixCog(bot, runtime))
