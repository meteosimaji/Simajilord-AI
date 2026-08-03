"""Discord slash commands as thin capability adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import time
from collections.abc import Awaitable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, cast
from urllib.parse import urlparse

import discord
from discord import app_commands
from discord.ext import commands

from simajilord.agent import (
    ACTION_UNDO_ANY_GRANT,
    AGENT_AUDIO_GRANT,
    AGENT_COMPUTE_GRANT,
    AGENT_CONNECTOR_GRANT,
    AGENT_FEEDBACK_GRANT,
    AGENT_FILE_GRANT,
    AGENT_FINAL_DELIVERED_CONTENT,
    AGENT_HIVE_GRANT,
    AGENT_IMAGE_GRANT,
    AGENT_MEDIA_GRANT,
    AGENT_MEMORY_CURATOR_GRANT,
    AGENT_MEMORY_GRANT,
    AGENT_MESSAGE_GRANT,
    AGENT_MODERATION_GRANT,
    AGENT_NO_ACTION_CONTENT,
    AGENT_QUOTE_GRANT,
    AGENT_REACTION_GRANT,
    AGENT_REPOST_GRANT,
    AGENT_REQUESTED_WRITE_CAPABILITIES,
    AGENT_SHELL_GRANT,
    AGENT_TIMER_WRITE_CAPABILITIES,
    AGENT_WEB_GRANT,
    AgentAutonomyMode,
    AgentAutonomyPolicyMode,
    AgentBusyError,
    AgentEvent,
    AgentHighRiskConfirmation,
    AgentRateLimitError,
    AgentRequest,
    AgentTaskRouteDecision,
    AgentTrigger,
    AutonomyEnqueueResult,
    AutonomyEventBatch,
    AutonomyEventKind,
    new_agent_public_reference_id,
    new_agent_task_id,
    task_scoped_conversation_id,
)
from simajilord.agent.autonomy import (
    AutonomyDeliveryConflictError,
    AutonomyDeliveryReceiptState,
    AutonomyDeliveryRecord,
    AutonomyDeliverySpec,
    AutonomyLeaseLostError,
)
from simajilord.agent.store import (
    AgentHostDeliveryRecord,
    AgentPendingHostDelivery,
)
from simajilord.async_locks import KeyedAsyncLockPool
from simajilord.capabilities.audio import (
    AudioAction,
    AudioAutoLeaveRequest,
    AudioControlResponse,
    AudioHistoryRequest,
    AudioHistoryResponse,
    AudioLoopRequest,
    AudioMixRequest,
    AudioMixResponse,
    AudioMoveRequest,
    AudioNoArgsRequest,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueuePositionRequest,
    AudioQueueRequest,
    AudioQueueResponse,
    AudioSearchItem,
    AudioSearchRequest,
    AudioSearchResponse,
    AudioSeekRequest,
    AudioTuneRequest,
    AudioVolumeRequest,
    audio_queue_response,
)
from simajilord.capabilities.focus_timer import (
    FocusTimerCancelRequest,
    FocusTimerCreateRequest,
    FocusTimerResponse,
)
from simajilord.capabilities.media import DownloadRequest, DownloadResponse
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudAnnouncementsSetRequest,
    ReadAloudContentModeSetRequest,
    ReadAloudDictionaryListRequest,
    ReadAloudDictionaryRemoveRequest,
    ReadAloudDictionarySetRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudRequest,
    ReadAloudResponse,
    ReadAloudSemanticsSetRequest,
    ReadAloudServerVoiceSetRequest,
    ReadAloudStatusRequest,
    ReadAloudUserVoiceSetRequest,
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
from simajilord.capabilities.translation import (
    TranslationDetectRequest,
    TranslationDetectResponse,
    TranslationLanguageItem,
    TranslationLanguagesRequest,
    TranslationLanguagesResponse,
    TranslationTranslateRequest,
    TranslationTranslateResponse,
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
from simajilord.core import AgentPrincipalKind, ApprovalMode, InvocationContext
from simajilord.core.errors import MediaError, ModerationError, UserError, WebError
from simajilord.domain.audio import AudioKind, LoopMode
from simajilord.domain.media import DownloadFormat
from simajilord.domain.web import SearchDepth, WebSource, WebTextMatch
from simajilord.runtime import SimajilordRuntime
from simajilord.services.audio import AudioSession
from simajilord.services.focus_timer import FocusTimer, FocusTimerStatus
from simajilord.services.read_aloud import (
    ReadAloudContentMode,
    ReadAloudMode,
    ReadAloudRoute,
    ReadAloudVoicePreset,
)
from simajilord.services.speech import SpeechSegment, SpeechSegmentKind

from .agent_ui import (
    AgentProgressMessage,
    agent_delivery_nonce,
    agent_error_text,
    agent_message_groups,
    agent_progress_text,
    discord_message_chunks,
    retry_after_text,
)
from .application_emojis import (
    ApplicationEmojiName,
    application_emoji,
)
from .attachment_io import read_attachment_bytes
from .audio import DiscordAudioOutput
from .capabilities import (
    DiscordConnectVoiceRequest,
    DiscordCreateQuoteImageRequest,
    DiscordPollRequest,
    DiscordPollResponse,
    DiscordPostExpandedMessageRequest,
    DiscordPostExpandedMessageResponse,
    DiscordServerRequest,
    DiscordServerResponse,
    DiscordTranslatedSegmentRecord,
    DiscordTranslateMessageRequest,
    DiscordTranslateMessageResponse,
    DiscordUserRequest,
    DiscordUserResponse,
    discord_translation_segments,
    parse_discord_message_link,
    quote_message_has_animation,
)
from .help_catalog import (
    HELP_CATEGORY_DESCRIPTIONS,
    HELP_ENTRIES,
    HELP_ENTRIES_BY_TOPIC,
    PublicCommandSpec,
)
from .local_media import attachment_can_play, import_discord_attachment
from .permissions import (
    RequesterPrincipal,
    ServicePrincipal,
    permission_enabled,
    read_aloud_audience_relation,
    readable_for_requester,
    readable_for_service,
)
from .presenter import (
    EmbedField,
    EmbedTone,
    command_embed,
)
from .read_aloud import (
    ReadAloudMessageFormatter,
    ReadAloudMessageText,
    merge_read_aloud_messages,
)

log = logging.getLogger(__name__)
BotContext: TypeAlias = commands.Context[commands.Bot]
_QUOTE_CONTEXT_MENU_NAME = "Quote"
_PLAY_AUDIO_CONTEXT_MENU_NAME = "Play Audio"
_TRANSLATE_CONTEXT_MENU_NAME = "Translate"
_MUSIC_DASHBOARD_ATTRIBUTE = "_simajilord_music_dashboard"
_MUSIC_DASHBOARD_STATE_FILE = "discord_music_dashboards.json"
_FOCUS_TIMER_DELIVERY_RECOVERY_LIMIT = 1_000
_AUTONOMY_LEASE_SECONDS = 60
_AUTONOMY_LEASE_HEARTBEAT_SECONDS = 20
_AUTONOMY_DELIVERY_RECOVERY_LIMIT = 1_000
_AutonomyResultT = TypeVar("_AutonomyResultT")


class SafeView(discord.ui.View):
    """Final interaction boundary shared by every Discord component view."""

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: Exception,
        item: discord.ui.Item[discord.ui.View],
    ) -> None:
        del item
        await handle_interaction_error(interaction, error)


class AgentHighRiskConfirmationView(SafeView):
    """One requester-only, expiring confirmation for an exact action binding."""

    def __init__(
        self,
        *,
        requester_id: int,
        binding_sha256: str,
        timeout: float,
    ) -> None:
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.binding_sha256 = binding_sha256
        self.message: discord.Message | None = None
        self._decision: asyncio.Future[bool] = (
            asyncio.get_running_loop().create_future()
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the requester can confirm this action.",
            ephemeral=True,
        )
        return False

    async def on_timeout(self) -> None:
        await self._finish(False, interaction=None)

    async def wait_for_decision(self) -> bool:
        return await self._decision

    async def _finish(
        self,
        decision: bool,
        *,
        interaction: discord.Interaction | None,
    ) -> None:
        if self._decision.done():
            if interaction is not None and not interaction.response.is_done():
                await interaction.response.defer()
            return
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True
        if interaction is not None and not interaction.response.is_done():
            await interaction.response.defer()
        self._decision.set_result(decision)
        self.stop()
        if self.message is not None:
            with suppress(discord.DiscordException):
                await self.message.edit(view=self)

    @discord.ui.button(
        label="Confirm exact action",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:agent:confirm-high-risk",
    )
    async def confirm_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[AgentHighRiskConfirmationView],
    ) -> None:
        await self._finish(True, interaction=interaction)

    @discord.ui.button(
        label="Do not run",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:agent:reject-high-risk",
    )
    async def reject_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[AgentHighRiskConfirmationView],
    ) -> None:
        await self._finish(False, interaction=interaction)


class SafeModal(discord.ui.Modal):
    """Final interaction boundary shared by every Discord modal."""

    async def on_error(  # type: ignore[override]
        self,
        interaction: discord.Interaction,
        error: Exception,
    ) -> None:
        await handle_interaction_error(interaction, error)


_ERROR_MESSAGES = {
    "audio.auto_leave_value_required": "Choose whether auto-leave is enabled.",
    "audio.capacity_reached": "The concurrent voice-server limit has been reached.",
    "audio.history_limit_invalid": "History limit must be between 1 and 25.",
    "audio.loop_mode_required": "Choose a loop mode.",
    "audio.loop_mix_conflict": ("Radio is on. Turn it off before enabling Loop."),
    "audio.mix_loop_conflict": ("Loop is on. Turn it off before enabling Radio."),
    "audio.mix_seed_limit": "Radio accepts at most eight seed tracks.",
    "audio.mix_seed_required": ("Provide a public seed track or add music before starting Radio."),
    "audio.mix_unavailable": "Radio is unavailable in this environment.",
    "audio.not_paused": "Playback is not paused.",
    "audio.nothing_playing": "No track is playing.",
    "audio.output_disconnected": "The BOT is not connected to voice.",
    "audio.other_voice_active": "Audio is already playing in another voice channel.",
    "audio.queue_position_invalid": "Choose a valid position shown in the queue.",
    "audio.queue_full": "This server's music queue is full.",
    "audio.user_queue_full": "You have reached your pending-request limit.",
    "audio.duplicate_limit": (
        "That track is already queued several times. Use Loop for repeated playback."
    ),
    "audio.seek_position_required": "Provide a playback position.",
    "audio.seek_position_invalid": "Enter a position such as `1:23`, `+30`, or `-10`.",
    "audio.search_empty": "No matching track was found.",
    "audio.search_limit_invalid": "Search limit must be between 1 and 10.",
    "audio.session_closed": "This audio session has ended.",
    "audio.session_missing": "This server has no audio session.",
    "audio.same_voice_required": ("Join the same voice channel as the BOT to control audio."),
    "audio.waiting_queue_restricted": (
        "Only the requester may start or alter a queue waiting for voice."
    ),
    "audio.tune_range_invalid": "Speed and pitch must each be between 0.5 and 2.0.",
    "audio.tune_values_required": "Provide both speed and pitch.",
    "audio.volume_range_invalid": "Volume must be between 0% and 200%.",
    "audio.volume_number_invalid": "Volume must be a number from 0 to 200.",
    "audio.volume_value_required": "Provide music or read-aloud volume.",
    "media.reference_required": "Provide a media URL or search query.",
    "media.reference_too_long": "The URL or search query is too long.",
    "media.query_url_not_allowed": "A search query cannot contain a URL.",
    "media.url_private": "Private and local-network addresses are not allowed.",
    "media.url_unresolvable": "Could not reach the media source.",
    "media.url_unsupported": "Provide a public HTTPS URL without credentials or a custom port.",
    "local_media.empty": "The attachment is empty.",
    "local_media.too_large": "That attachment exceeds the local playback limit.",
    "local_media.content_type_unsupported": (
        "Choose an audio file or a video file that contains audio."
    ),
    "local_media.duration_unknown": "Could not determine the attachment duration.",
    "local_media.duration_too_long": "That attachment is too long for local playback.",
    "local_media.audio_stream_missing": "That file does not contain a playable audio stream.",
    "local_media.invalid_media": "ffprobe could not validate that attachment as media.",
    "local_media.reference_invalid": "The saved local-media reference is invalid.",
    "feedback.details_required": "Describe what happened or what would help.",
    "feedback.details_too_long": "Feedback details must be at most 4,000 characters.",
    "feedback.title_too_long": "The feedback title must be at most 160 characters.",
    "feedback.expected_too_long": (
        "Expected behaviour must be at most 2,000 characters."
    ),
    "feedback.requester_mismatch": "Only the person who opened this form may submit it.",
    "local_media.cache_full": (
        "The local media store is full. Finish or remove queued local files first."
    ),
    "local_media.selection_required": "Provide either a search/URL or one attachment.",
    "local_media.multiple_inputs": "Provide a search/URL or an attachment, not both.",
    "translation.text_required": "Provide text to translate.",
    "translation.message_text_required": "That message has no text to translate.",
    "translation.text_too_long": "That text is too long for one translation.",
    "translation.target_required": "Choose a target language.",
    "translation.language_invalid": "Choose a valid BCP-47 language code.",
    "translation.language_not_detected": "The source language could not be detected.",
    "translation.same_language": "The source and target languages are the same.",
    "translation.language_pair_unsupported": (
        "The translation service does not support that language pair."
    ),
    "translation.language_pair_not_installed": (
        "Install that language pair in macOS Translation settings, then try again."
    ),
    "translation.helper_missing": "The local macOS translation helper is missing.",
    "translation.helper_unavailable": "The local macOS translation helper is unavailable.",
    "translation.helper_failed": "The local macOS translation helper could not complete.",
    "translation.timeout": "The local translation took too long.",
    "translation.failed": "The translation could not be completed.",
    "translation.unavailable": "Translation is unavailable right now.",
    "moderation.daily_limit_reached": "The daily analysis limit has been reached.",
    "moderation.filename_invalid": "Could not read the attachment filename.",
    "moderation.media_empty": "The attachment is empty.",
    "moderation.media_too_large": "The attachment is too large to analyze.",
    "moderation.media_type_unsupported": "Use a common image or video file.",
    "moderation.not_configured": "Synthetic-media analysis is not configured.",
    "discord.message_limit_invalid": "History limit must be between 1 and 100.",
    "discord.message_not_found": "The Discord message could not be found.",
    "discord.message_fetch_failed": "The Discord message could not be retrieved.",
    "discord.message_id_invalid": "The message ID is invalid.",
    "discord.message_length_invalid": "Messages must contain between 1 and 2,000 characters.",
    "discord.channel_id_invalid": "The channel ID is invalid.",
    "discord.voice_channel_id_invalid": "The voice-channel ID is invalid.",
    "discord.snowflake_invalid": "That Discord ID is invalid.",
    "discord.message_destination_invalid": (
        "The destination is not a Discord channel that accepts messages."
    ),
    "discord.text_destination_invalid": "The destination is not a writable text channel.",
    "discord.user_id_invalid": "The user ID is invalid.",
    "discord.user_not_found": "The Discord user could not be found.",
    "discord.guild_id_invalid": "The Discord server ID is invalid.",
    "discord.guild_unavailable": "The Discord server is unavailable.",
    "discord.voice_destination_invalid": "The configured voice destination is invalid.",
    "discord.voice_channel_unavailable": "That voice channel is unavailable.",
    "discord.voice_connect_failed": "Could not connect to the voice channel.",
    "discord.voice_join_required": "Join the intended voice channel first.",
    "discord.agent_read_channel_forbidden": (
        "The AI does not have permission to view this Discord channel."
    ),
    "discord.agent_principal_invalid": (
        "Could not safely resolve who is requesting this Discord action. Try again shortly."
    ),
    "discord.information_flow_forbidden": (
        "This action could reveal information to a wider or unverified audience."
    ),
    "discord.bulk_delete_message_too_old": (
        "Bulk delete cannot include messages that are 14 days old or older."
    ),
    "discord.custom_emoji_decode_failed": "Could not decode that custom emoji.",
    "discord.poll_question_invalid": "Poll questions must contain 1-300 characters.",
    "discord.poll_option_count_invalid": "Polls must contain 2-10 choices.",
    "discord.poll_option_too_long": "Each poll choice must be at most 55 characters.",
    "discord.poll_duration_invalid": "Poll duration must be between 1 and 168 hours.",
    "discord.message_chunk_limit_invalid": "Message chunk size must be 1-1,000 characters.",
    "discord.message_offset_invalid": "Provide a valid message offset.",
    "discord.member_lookup_failed": (
        "Could not retrieve the member from Discord. Try again shortly."
    ),
    "discord.member_required": ("Could not resolve that member in this server."),
    "discord.expand_cross_guild_forbidden": "Messages from another server cannot be expanded.",
    "discord.expand_unavailable": (
        "This message cannot be expanded. Check that you can view the source."
    ),
    "discord.expand_destination_unavailable": (
        "Cannot post the expanded message here. Check the BOT's Send Messages "
        "and Embed Links permissions."
    ),
    "discord.expand_failed": ("Could not retrieve the source message from Discord."),
    "discord.quote_destination_unavailable": (
        "Cannot post the quote here. Check the BOT's Send Messages and Attach Files permissions."
    ),
    "discord.quote_failed": ("Could not send the quote image to Discord."),
    "discord.quote_render_failed": ("Could not render the quote image. Check the source message."),
    "discord.manage_guild_required": (
        "Manage Server permission is required. Ask an administrator to run "
        "`/readaloud setup` to change managed routes."
    ),
    "read_aloud.route_fields_required": ("Choose a conversation source and a voice destination."),
    "read_aloud.destination_conflict": "Read aloud is configured for another VC.",
    "read_aloud.source_channel_required": "Choose a conversation source.",
    "read_aloud.source_channel_limit": "Choose between 1 and 25 source channels.",
    "read_aloud.source_channels_required": "Choose at least one source channel.",
    "read_aloud.dictionary_surface_required": "Provide the written form.",
    "read_aloud.dictionary_surface_too_long": "Written form must be at most 100 characters.",
    "read_aloud.dictionary_reading_required": "Provide its pronunciation.",
    "read_aloud.dictionary_reading_too_long": "Pronunciation must be at most 200 characters.",
    "read_aloud.dictionary_invalid": "The dictionary entry is invalid.",
    "read_aloud.exclusion_invalid": "The read-aloud exclusion is invalid.",
    "read_aloud.announcement_value_invalid": "The announcement settings are invalid.",
    "read_aloud.semantic_value_invalid": "The message-reading settings are invalid.",
    "read_aloud.announcement_value_required": "Choose at least one voice event to update.",
    "read_aloud.semantic_value_required": "Choose at least one message setting to update.",
    "read_aloud.ignore_bot_unnecessary": "BOT messages are already excluded.",
    "read_aloud.role_not_found": "That role was not found in this server.",
    "discord.message_channel_unavailable": (
        "You and the BOT must both be able to view every selected channel."
    ),
    "speech.no_readable_text": "There is no readable text.",
    "speech.queue_full": "Read aloud is busy. Try again shortly.",
    "utility.dice_count_invalid": "Dice count must be between 1 and 20.",
    "utility.dice_sides_invalid": "Dice sides must be between 2 and 1,000.",
    "utility.option_count_invalid": "Provide between 2 and 20 choices.",
    "utility.option_too_long": "Each choice must be at most 100 characters.",
    "web.chunk_limit_invalid": "Fetch size must be 200-6,000 characters.",
    "web.context_limit_invalid": "Match context must be 40-300 characters.",
    "web.match_limit_invalid": "Match limit must be between 1 and 10.",
    "web.offset_invalid": "Provide a valid page-text offset.",
    "web.pattern_required": "Provide a phrase to find.",
    "web.pattern_too_long": "The phrase must be at most 300 characters.",
    "web.query_required": "Provide a search query.",
    "web.query_too_long": "The query must be at most 500 characters.",
    "web.safesearch_invalid": "SafeSearch must be 0, 1, or 2.",
    "web.time_range_invalid": "Time range must be day, month, or year.",
    "media.download_cooldown": "Wait 30 seconds before starting another download.",
    "workspace.required": "Run this command inside a Discord server.",
    "files.workspace_mode_invalid": "The configured AI file-workspace policy is invalid.",
}

_MEDIA_ERROR_MESSAGES = {
    "cookie_required": "This media requires host-side login cookies.",
    "geo_restricted": "This media is unavailable from the current region.",
    "rate_limited": "The media source is rate-limiting requests.",
    "timeout": "Media processing timed out.",
    "too_large": "The file exceeds this server's upload limit.",
    "unavailable": "The media is private, deleted, or unavailable.",
    "unsafe_path": "The provider returned an unsafe output path.",
    "unsupported": "That media URL is not supported.",
    "unknown": "Could not process the media.",
}

_WEB_ERROR_MESSAGES = {
    "content_empty": "The page contains no readable text.",
    "content_invalid": "Could not parse the page text.",
    "content_type_unsupported": "That page type is not supported.",
    "fetch_failed": "Could not fetch the page.",
    "redirect_invalid": "The redirect destination is invalid.",
    "request_too_broad": "The search is too broad. Narrow the query.",
    "response_too_large": "The page is too large to read safely.",
    "search_backend_error": "The local search backend returned an error.",
    "search_invalid_response": "The local search backend returned an invalid response.",
    "search_response_too_large": "Search results are too large. Narrow the query.",
    "search_unavailable": "The local search service is temporarily unavailable.",
    "timeout": "Web processing timed out.",
    "too_many_redirects": "The page redirected too many times.",
    "upstream_unavailable": "The website is temporarily unavailable.",
    "url_invalid": "Provide a public HTTP or HTTPS URL.",
    "url_private": "Private and local-network addresses cannot be opened.",
    "url_rejected": "The website rejected the request.",
    "url_unresolvable": "Could not connect to the website.",
}

_MODERATION_ERROR_MESSAGES = {
    "authentication_failed": "Media analysis authentication failed.",
    "invalid_response": "The analysis service returned an invalid response.",
    "media_rejected": "The attachment could not be analyzed.",
    "provider_unavailable": "Media analysis is temporarily unavailable.",
    "rate_limited": "Media analysis is temporarily rate-limited.",
    "response_too_large": "The analysis response exceeded the safe limit.",
    "timeout": "Media analysis timed out.",
}

_AUDIO_ACTION_MESSAGES = {
    AudioAction.PAUSE.value: "Playback paused.",
    AudioAction.RESUME.value: "Playback resumed.",
    AudioAction.SKIP.value: "Skipped the current track.",
    AudioAction.STOP.value: "Playback stopped and the pending queue was cleared.",
    AudioAction.LEAVE.value: "Disconnected from voice.",
}

_AUDIO_CONTROL_CAPABILITIES = {
    AudioAction.PAUSE: "discord.pause_audio",
    AudioAction.RESUME: "discord.resume_audio",
    AudioAction.SKIP: "discord.skip_audio",
    AudioAction.STOP: "discord.stop_audio",
    AudioAction.LEAVE: "discord.leave_audio",
    AudioAction.LOOP: "discord.set_audio_loop",
    AudioAction.REMOVE: "discord.remove_audio",
    AudioAction.AUTO_LEAVE: "discord.set_audio_auto_leave",
    AudioAction.SHUFFLE: "discord.shuffle_audio",
    AudioAction.SEEK: "discord.seek_audio",
    AudioAction.TUNE: "discord.tune_audio",
    AudioAction.VOLUME: "discord.set_audio_volume",
    AudioAction.MOVE: "discord.move_audio",
    AudioAction.CLEAR_MINE: "discord.clear_my_audio",
}


def audio_control_capability_call(
    action: AudioAction,
    *,
    loop_mode: LoopMode | None = None,
    enabled: bool | None = None,
    position_seconds: float | None = None,
    speed: float | None = None,
    pitch: float | None = None,
    position: int | None = None,
    to_position: int | None = None,
    music_percent: int | None = None,
    speech_percent: int | None = None,
    replace_mix: bool = False,
) -> tuple[str, object]:
    """Map every human audio control to the same exact tool exposed to AI."""

    name = _AUDIO_CONTROL_CAPABILITIES[action]
    if action is AudioAction.LOOP:
        if loop_mode is None:
            raise ValueError("loop_mode is required")
        return name, AudioLoopRequest(mode=loop_mode, replace_mix=replace_mix)
    if action is AudioAction.REMOVE:
        if position is None:
            raise ValueError("position is required")
        return name, AudioQueuePositionRequest(position=position)
    if action is AudioAction.AUTO_LEAVE:
        if enabled is None:
            raise ValueError("enabled is required")
        return name, AudioAutoLeaveRequest(enabled=enabled)
    if action is AudioAction.SEEK:
        if position_seconds is None:
            raise ValueError("position_seconds is required")
        return name, AudioSeekRequest(position_seconds=position_seconds)
    if action is AudioAction.TUNE:
        if speed is None or pitch is None:
            raise ValueError("speed and pitch are required")
        return name, AudioTuneRequest(speed=speed, pitch=pitch)
    if action is AudioAction.VOLUME:
        return name, AudioVolumeRequest(
            music_percent=music_percent,
            speech_percent=speech_percent,
        )
    if action is AudioAction.MOVE:
        if position is None or to_position is None:
            raise ValueError("from and to positions are required")
        return name, AudioMoveRequest(
            from_position=position,
            to_position=to_position,
        )
    return name, AudioNoArgsRequest()


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes}:{remaining_seconds:02d}"


def _storage_size(value: int) -> str:
    amount = float(max(0, value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _read_aloud_mode_label(mode: str | None) -> str:
    if mode == ReadAloudMode.SKIP_DURING_MUSIC.value:
        return "Skip messages while music is playing"
    return "Read everything and automatically duck music"


def _on_off(enabled: bool) -> str:
    return "On" if enabled else "Off"


def _speech_voice_label(runtime: SimajilordRuntime) -> str:
    settings = runtime.settings
    if settings.tts_provider == "voicevox":
        return f"VOICEVOX · Style ID {settings.voicevox_speaker_id}"
    return f"macOS · {settings.tts_voice}"


def _parse_position(value: str) -> tuple[float, bool]:
    text = value.strip()
    relative = text.startswith(("+", "-"))
    sign = -1.0 if text.startswith("-") else 1.0
    unsigned = text[1:] if relative else text
    parts = unsigned.split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise UserError("audio.seek_position_invalid")
    numbers = [int(part) for part in parts]
    seconds = 0
    for number in numbers:
        seconds = seconds * 60 + number
    return sign * float(seconds), relative


def _requester(name: str | None, actor_id: str | None = None) -> str:
    if actor_id and actor_id.isdigit():
        return f"<@{actor_id}>"
    return discord.utils.escape_markdown(name) if name else "Unknown"


def _queue_requester(item: object) -> str:
    lane = getattr(item, "queue_lane", "request")
    if lane == "autoplay":
        return "Radio"
    return _requester(
        getattr(item, "requested_by_name", None),
        getattr(item, "requested_by_id", None),
    )


def _loop_mode_label(mode: str) -> str:
    return {
        LoopMode.NONE.value: "Off",
        LoopMode.TRACK.value: "Track",
        LoopMode.QUEUE.value: "Queue",
    }.get(mode, mode)


def _compact_panel_text(value: str, *, maximum: int) -> str:
    """Keep control-panel copy readable without altering the underlying value."""

    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= maximum else f"{cleaned[: maximum - 1].rstrip()}…"


def _read_aloud_status(route: ReadAloudRoute | None) -> str:
    if route is None:
        return "Off · Select **Read aloud** to choose channels."
    source_count = len(route.text_channel_ids)
    channel_label = "channel" if source_count == 1 else "channels"
    return (
        f"On · **{source_count} {channel_label}** → <#{route.audio_destination_id}>\n"
        f"Mode **{route.mode.value.replace('_', ' ').title()}**"
    )


def _active_read_aloud_route(
    runtime: SimajilordRuntime,
    workspace_id: str,
) -> ReadAloudRoute | None:
    route = runtime.read_aloud.get(workspace_id)
    return route if isinstance(route, ReadAloudRoute) else None


def _read_aloud_audience_allowed(
    runtime: SimajilordRuntime,
    message: discord.Message,
    destination: discord.VoiceChannel | discord.StageChannel,
) -> bool:
    """Apply the reversible listener-audience policy before speech is queued."""

    raw_mode = getattr(runtime.settings, "read_aloud_audience_mode", "enforce")
    mode = getattr(raw_mode, "value", raw_mode)
    if mode not in {"enforce", "audit", "disabled"}:
        mode = "enforce"
    if mode == "disabled":
        return True
    guild = message.guild
    source = message.channel
    if guild is None or not isinstance(
        source,
        (
            discord.TextChannel,
            discord.Thread,
            discord.ForumChannel,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        relation = "uncertain"
    else:
        relation = read_aloud_audience_relation(guild, source, destination)
    if relation == "same_or_narrower":
        return True
    log.warning(
        "Read-aloud audience check failed mode=%s guild=%s source=%s "
        "destination=%s relation=%s",
        mode,
        getattr(guild, "id", None),
        getattr(source, "id", None),
        destination.id,
        relation,
    )
    return mode == "audit"


def _risk_label(risk: str) -> str:
    return {
        "read": "Read",
        "write": "Write",
        "external": "External",
        "destructive": "Destructive",
    }.get(risk, risk)


def _approval_label(approval: str) -> str:
    return {
        "never": "None",
        "when_requested": "When requested",
        "always": "Always",
    }.get(approval, approval)


def _search_candidate_line(candidate: AudioSearchItem) -> str:
    title = discord.utils.escape_markdown(candidate.title)
    source = f"\n{discord.utils.escape_markdown(candidate.uploader)}" if candidate.uploader else ""
    return f"[{title}]({candidate.reference}){source}"


def _media_title_link(title: str, page_url: str, *, maximum: int) -> str:
    label = discord.utils.escape_markdown(_compact_panel_text(title, maximum=maximum))
    parsed = urlparse(page_url)
    if parsed.scheme == "https" and parsed.hostname:
        return f"[{label}]({_safe_markdown_url(page_url)})"
    return f"**{label}**"


def music_added_embed(response: AudioPlayResponse) -> discord.Embed:
    if response.playback_state == "playing":
        title = "Now playing"
        playback = "Playing now"
    elif response.playback_state == "waiting_for_voice":
        title = "Added to queue"
        playback = "Ready · Playback starts when you join a voice channel."
    else:
        title = "Added to queue"
        playback = f"Waiting · Position **{response.queue_position}**"
    fields = [
        EmbedField("Status", playback, inline=False),
        EmbedField("Duration", _duration(response.duration_seconds)),
        EmbedField(
            "Requested by",
            _requester(response.requested_by_name, response.requested_by_id),
        ),
    ]
    if response.uploader:
        fields.append(EmbedField("Source", discord.utils.escape_markdown(response.uploader)))
    fields.append(
        EmbedField(
            "Voice channel",
            f"<#{response.destination_id}>"
            if response.destination_id
            else "Connects when the requester joins",
        )
    )
    embed = command_embed(
        title,
        description=f"### {_media_title_link(response.title, response.page_url, maximum=100)}",
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )
    if response.thumbnail_url:
        embed.set_thumbnail(url=response.thumbnail_url)
    return embed


def music_queue_embed(
    response: AudioQueueResponse,
    *,
    page: int = 1,
    page_size: int = 10,
    read_aloud_route: ReadAloudRoute | None = None,
    loading_emoji: str = "⏳",
    audio_wave_emoji: str = "〰️",
    radio_emoji: str = "📻",
    now_epoch: float | None = None,
) -> discord.Embed:
    fields: list[EmbedField] = []
    upcoming = tuple(item for item in response.pending if item.kind == AudioKind.MUSIC.value)
    if response.current is None:
        title = "Audio"
        description_lines = ["No track is playing."]
    else:
        title = "Audio · Now playing"
        current = response.current
        elapsed = min(response.position_seconds, current.duration_seconds)
        uploader = discord.utils.escape_markdown(current.uploader or "Unknown uploader")
        if response.paused:
            timing = f"Paused at `{_duration(elapsed)}`"
        elif current.duration_seconds <= 0:
            timing = "Playing"
        elif LoopMode(response.loop_mode) is LoopMode.TRACK:
            timing = f"Looping track · `{_duration(current.duration_seconds)}` per loop"
        elif current.duration_seconds > elapsed:
            end_epoch = _playback_end_epoch(
                response,
                now_epoch=time.time() if now_epoch is None else now_epoch,
            )
            timing = (
                f"Playing · Ends <t:{end_epoch}:R> · "
                f"duration `{_duration(current.duration_seconds)}`"
                if end_epoch is not None
                else f"Playing · duration `{_duration(current.duration_seconds)}`"
            )
        else:
            timing = "Finishing"
        description_lines = [
            (
                f"### {audio_wave_emoji} "
                f"{_media_title_link(current.title, current.page_url, maximum=96)}"
                if current.kind == AudioKind.SPEECH.value
                else (f"### {_media_title_link(current.title, current.page_url, maximum=96)}")
            ),
            timing,
            (
                f"{_compact_panel_text(uploader, maximum=54)} · "
                f"requested\u00a0by\u00a0{_queue_requester(current)}"
            ),
        ]

    if response.voice_activation_required:
        description_lines.append("**Ready to resume** · Join the VC and press **Start**.")
    elif response.waiting_for_voice:
        description_lines.append("**Queued** · Playback starts when the requester joins the VC.")
    elif response.paused:
        description_lines.append("**Paused**")
    elif not response.connected:
        description_lines.append("**Disconnected** · Controls will adapt when you join a VC.")

    playback_parts: list[str] = []
    if upcoming:
        playback_parts.append(f"Queue **{len(upcoming)}**")
    if LoopMode(response.loop_mode) is not LoopMode.NONE:
        playback_parts.append(f"Loop **{_loop_mode_label(response.loop_mode)}**")
    if response.autoplay_enabled:
        playback_parts.append("Radio **On**")
    if playback_parts:
        fields.append(EmbedField("Playback", " · ".join(playback_parts), inline=False))
    if response.music_volume_percent != 100 or response.speech_volume_percent != 100:
        fields.append(
            EmbedField(
                "Mix levels",
                f"Music **{response.music_volume_percent}%** · "
                f"Read aloud **{response.speech_volume_percent}%**",
                inline=False,
            )
        )
    if read_aloud_route is not None:
        fields.append(
            EmbedField(
                "Read aloud",
                _read_aloud_status(read_aloud_route),
                inline=False,
            )
        )

    if upcoming:
        page_count = max(1, (len(upcoming) + page_size - 1) // page_size)
        selected_page = min(max(1, page), page_count)
        start = (selected_page - 1) * page_size
        visible = upcoming[start : start + page_size]
        lines = [
            f"`{index:02d}` {_media_title_link(item.title, item.page_url, maximum=80)} · "
            f"`{_duration(item.duration_seconds)}` · {_queue_requester(item)}"
            for index, item in enumerate(visible, start=start + 1)
        ]
        fields.append(
            EmbedField(
                f"Up Next · {selected_page}/{page_count}",
                "\n".join(lines),
                inline=False,
            )
        )
    if response.autoplay_enabled:
        if response.autoplay_next is not None:
            autoplay_text = _media_title_link(
                response.autoplay_next.title,
                response.autoplay_next.page_url,
                maximum=80,
            )
        elif upcoming:
            count = len(upcoming)
            autoplay_text = (
                f"Waiting behind **{count} manual {'request' if count == 1 else 'requests'}**."
            )
        else:
            autoplay_text = f"{loading_emoji} Finding the next related track…"
        fields.append(
            EmbedField(
                f"{radio_emoji} Radio",
                f"{autoplay_text}\nYour requests always play first.",
                inline=False,
            )
        )
    if response.speed != 1.0 or response.pitch != 1.0:
        fields.append(
            EmbedField(
                "Tuning",
                f"Speed {response.speed:.2f}x · Pitch {response.pitch:.2f}x",
            )
        )
    embed = command_embed(
        title,
        description="\n".join(description_lines),
        fields=tuple(fields),
    )
    # The permanent panel represents state, not the time of the last edit.
    embed.timestamp = None
    if response.current and response.current.thumbnail_url:
        embed.set_thumbnail(url=response.current.thumbnail_url)
    return embed


def _playback_end_epoch(
    response: AudioQueueResponse,
    *,
    now_epoch: float,
) -> int | None:
    """Project a live non-looping track end into Discord's relative-time syntax."""

    current = response.current
    if (
        current is None
        or response.paused
        or not response.connected
        or response.waiting_for_voice
        or response.voice_activation_required
        or LoopMode(response.loop_mode) is LoopMode.TRACK
        or current.duration_seconds <= 0
        or response.speed <= 0
    ):
        return None
    remaining_track_seconds = current.duration_seconds - max(
        0.0,
        response.position_seconds,
    )
    if remaining_track_seconds <= 0:
        return None
    return math.ceil(now_epoch + remaining_track_seconds / response.speed)


def music_now_playing_embed(response: AudioQueueResponse) -> discord.Embed:
    return music_queue_embed(response)


def music_details_embed(response: AudioQueueResponse) -> discord.Embed:
    """Render on-demand playback details without bloating the canonical panel."""

    current = response.current
    if current is None:
        return command_embed(
            "Audio details",
            description="No track is playing.",
        )
    fields = (
        EmbedField(
            "Playback",
            (
                f"Position `{_duration(response.position_seconds)}` · "
                f"Duration `{_duration(current.duration_seconds)}`"
            ),
            inline=False,
        ),
        EmbedField(
            "Source",
            (
                f"{discord.utils.escape_markdown(current.uploader or 'Unknown uploader')}\n"
                + (
                    f"[Open source]({_safe_markdown_url(current.page_url)})"
                    if urlparse(current.page_url).scheme == "https"
                    else "Saved local attachment"
                )
            ),
            inline=False,
        ),
        EmbedField("Requested by", _queue_requester(current)),
        EmbedField(
            "Settings",
            (
                f"Speed **{response.speed:.2f}x** · Pitch **{response.pitch:.2f}x**\n"
                f"Music **{response.music_volume_percent}%** · "
                f"Read aloud **{response.speech_volume_percent}%**"
            ),
            inline=False,
        ),
    )
    embed = command_embed(
        "Audio details",
        description=f"### {_media_title_link(current.title, current.page_url, maximum=120)}",
        fields=tuple(fields),
    )
    embed.timestamp = None
    if current.thumbnail_url:
        embed.set_image(url=current.thumbnail_url)
    return embed


class MusicDashboardManager:
    """Keep one current, silent music panel near the bottom of each bound channel."""

    _DEBOUNCE_SECONDS = 0.8
    _IDLE_SECONDS = 5 * 60

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._channel_ids: dict[str, int] = {}
        self._messages: dict[str, discord.Message] = {}
        self._fingerprints: dict[str, tuple[object, ...]] = {}
        self._repost_tasks: dict[str, asyncio.Task[None]] = {}
        self._expiry_tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._store_lock = asyncio.Lock()
        data_dir = getattr(getattr(runtime, "settings", None), "data_dir", None)
        self._state_path = (
            data_dir / _MUSIC_DASHBOARD_STATE_FILE if isinstance(data_dir, Path) else None
        )
        self._stored_messages = self._load_stored_messages()
        self._channel_ids.update(
            {
                workspace_id: channel_id
                for workspace_id, (channel_id, _) in self._stored_messages.items()
            }
        )
        runtime.audio.add_state_listener(self.on_audio_state_changed)
        for workspace_id in self._channel_ids:
            session = runtime.audio.find(workspace_id)
            if session is not None:
                self._schedule_repost(session)

    def bind(self, guild_id: int | None, channel_id: int | None) -> None:
        if guild_id is None or channel_id is None:
            return
        workspace_id = str(guild_id)
        changed = self._channel_ids.get(workspace_id) != channel_id
        if changed:
            self._channel_ids[workspace_id] = channel_id
            self._fingerprints.pop(workspace_id, None)
        if changed or workspace_id not in self._messages:
            session = self.runtime.audio.find(workspace_id)
            if session is not None:
                self._schedule_repost(session)

    async def on_audio_state_changed(self, session: AudioSession) -> None:
        self._schedule_repost(session)

    async def publish(
        self,
        session: AudioSession,
        *,
        obsolete_message: discord.Message | None = None,
        force: bool = False,
    ) -> None:
        """Publish one canonical panel immediately, cancelling a queued duplicate."""

        workspace_id = session.workspace_id
        previous = self._repost_tasks.pop(workspace_id, None)
        if previous is not None and previous is not asyncio.current_task():
            previous.cancel()
        async with self._locks.setdefault(workspace_id, asyncio.Lock()):
            await self._repost(
                session,
                obsolete_message=obsolete_message,
                force=force,
            )

    def _schedule_repost(self, session: AudioSession) -> None:
        workspace_id = session.workspace_id
        if workspace_id not in self._channel_ids:
            return
        previous = self._repost_tasks.pop(workspace_id, None)
        if previous is not None and previous is not asyncio.current_task():
            previous.cancel()
        self._repost_tasks[workspace_id] = asyncio.create_task(
            self._debounced_repost(session),
            name=f"simajilord-music-dashboard-{workspace_id}",
        )

    async def close(self) -> None:
        self.runtime.audio.remove_state_listener(self.on_audio_state_changed)
        tasks = (*self._repost_tasks.values(), *self._expiry_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._repost_tasks.clear()
        self._expiry_tasks.clear()

    async def dismiss(self, workspace_id: str) -> None:
        """Delete the visible panel while preserving the durable audio session."""

        for tasks in (self._repost_tasks, self._expiry_tasks):
            task = tasks.pop(workspace_id, None)
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        async with self._locks.setdefault(workspace_id, asyncio.Lock()):
            message = self._messages.get(workspace_id)
            if message is None:
                message = await self._fetch_stored_message(workspace_id)
            if message is not None:
                with suppress(discord.DiscordException):
                    await message.delete()
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
            if self._stored_messages.pop(workspace_id, None) is not None:
                await self._persist_stored_messages()

    async def prune_stale_records(self, valid_workspace_ids: frozenset[str]) -> int:
        """Drop dashboard pointers for guilds the BOT no longer belongs to."""

        stale = tuple(
            workspace_id
            for workspace_id in self._stored_messages
            if workspace_id not in valid_workspace_ids
        )
        if not stale:
            return 0
        for workspace_id in stale:
            self._stored_messages.pop(workspace_id, None)
            self._channel_ids.pop(workspace_id, None)
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
        await self._persist_stored_messages()
        return len(stale)

    async def _debounced_repost(self, session: AudioSession) -> None:
        workspace_id = session.workspace_id
        try:
            await asyncio.sleep(self._DEBOUNCE_SECONDS)
            async with self._locks.setdefault(workspace_id, asyncio.Lock()):
                await self._repost(session)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Could not repost music dashboard guild=%s", workspace_id)
        finally:
            if self._repost_tasks.get(workspace_id) is asyncio.current_task():
                self._repost_tasks.pop(workspace_id, None)

    async def _repost(
        self,
        session: AudioSession,
        *,
        obsolete_message: discord.Message | None = None,
        force: bool = False,
    ) -> None:
        workspace_id = session.workspace_id
        channel_id = self._channel_ids.get(workspace_id)
        if channel_id is None:
            return
        snapshot = await session.snapshot()
        response = audio_queue_response(snapshot)
        read_aloud_route = _active_read_aloud_route(self.runtime, workspace_id)
        fingerprint = _music_dashboard_fingerprint(response, read_aloud_route)
        current_message = self._messages.get(workspace_id)
        if current_message is None:
            current_message = await self._fetch_stored_message(workspace_id)
        if (
            not force
            and response.current is None
            and not any(item.kind == AudioKind.MUSIC.value for item in response.pending)
            and not response.autoplay_enabled
            and response.destination_id is None
            and not response.waiting_for_voice
            and not response.voice_activation_required
        ):
            for message in _unique_messages(current_message, obsolete_message):
                with suppress(discord.DiscordException):
                    await message.delete()
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
            if self._stored_messages.pop(workspace_id, None) is not None:
                await self._persist_stored_messages()
            return
        if self._fingerprints.get(workspace_id) == fingerprint and current_message:
            if obsolete_message is not None and obsolete_message.id != current_message.id:
                with suppress(discord.DiscordException):
                    await obsolete_message.delete()
            return
        embed = music_queue_embed(
            response,
            read_aloud_route=read_aloud_route,
            loading_emoji=application_emoji(
                self.bot,
                ApplicationEmojiName.LOADING,
            ),
            audio_wave_emoji=application_emoji(
                self.bot,
                ApplicationEmojiName.AUDIO_WAVE,
            ),
            radio_emoji=application_emoji(
                self.bot,
                ApplicationEmojiName.RADIO,
            ),
        )
        view = MusicControlsView(self.runtime, self, response=response)
        stored = self._stored_messages.get(workspace_id)
        current_is_in_bound_channel = stored is None or stored[0] == channel_id
        if current_message is not None and current_is_in_bound_channel and not force:
            edit_error: discord.DiscordException | None = None
            try:
                edited = await current_message.edit(embed=embed, view=view)
            except discord.DiscordException as exc:
                edit_error = exc
                if getattr(exc, "status", None) == 429:
                    await self._record_metric("discord.dashboard_429", "rate_limited")
                    retry_after = max(
                        0.0,
                        min(float(getattr(exc, "retry_after", 1.0) or 1.0), 30.0),
                    )
                    await asyncio.sleep(retry_after)
                    try:
                        edited = await current_message.edit(embed=embed, view=view)
                    except discord.DiscordException as retry_exc:
                        edit_error = retry_exc
                        await self._record_metric(
                            "discord.dashboard_429",
                            "retry_failed",
                        )
                    else:
                        edit_error = None
                        await self._record_metric(
                            "discord.dashboard_429",
                            "retry_succeeded",
                        )
            if edit_error is not None:
                status = getattr(edit_error, "status", None)
                code = getattr(edit_error, "code", None)
                if status == 403:
                    await self._record_metric("discord.dashboard_403", "stopped")
                    self._channel_ids.pop(workspace_id, None)
                    self._messages.pop(workspace_id, None)
                    self._fingerprints.pop(workspace_id, None)
                    if self._stored_messages.pop(workspace_id, None) is not None:
                        await self._persist_stored_messages()
                    log.warning(
                        "Stopping music dashboard updates after Discord denied access "
                        "guild=%s channel=%s",
                        workspace_id,
                        channel_id,
                    )
                    return
                if status == 429:
                    log.warning(
                        "Music dashboard edit remained rate-limited after one retry guild=%s",
                        workspace_id,
                    )
                    return
                if status == 404:
                    await self._record_metric("discord.dashboard_404", "reset")
                    self._messages.pop(workspace_id, None)
                    current_message = None
                    if self._stored_messages.pop(workspace_id, None) is not None:
                        await self._persist_stored_messages()
                if code == 30046:
                    await self._record_metric("discord.dashboard_30046", "rollover")
                log.warning(
                    "Could not edit music dashboard; posting a replacement guild=%s error=%s",
                    workspace_id,
                    edit_error,
                )
            else:
                await self._record_metric("discord.dashboard_edits", "succeeded")
                if isinstance(edited, discord.Message):
                    current_message = edited
                self._messages[workspace_id] = current_message
                self._fingerprints[workspace_id] = fingerprint
                stored_message = (channel_id, current_message.id)
                if self._stored_messages.get(workspace_id) != stored_message:
                    self._stored_messages[workspace_id] = stored_message
                    await self._persist_stored_messages()
                if obsolete_message is not None and obsolete_message.id != current_message.id:
                    with suppress(discord.DiscordException):
                        await obsolete_message.delete()
                self._refresh_expiry(
                    workspace_id,
                    current_message,
                    idle=_audio_dashboard_is_idle(response),
                )
                return

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            with suppress(discord.DiscordException):
                channel = await self.bot.fetch_channel(channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return

        # Sending first avoids losing the working panel when Discord rejects a post.
        new_message = await channel.send(
            embed=embed,
            view=view,
            silent=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self._messages[workspace_id] = new_message
        self._fingerprints[workspace_id] = fingerprint
        stored_message = (channel_id, new_message.id)
        if self._stored_messages.get(workspace_id) != stored_message:
            self._stored_messages[workspace_id] = stored_message
            await self._persist_stored_messages()
        for message in _unique_messages(current_message, obsolete_message):
            if message.id == new_message.id:
                continue
            with suppress(discord.DiscordException):
                await message.delete()

        self._refresh_expiry(
            workspace_id,
            new_message,
            idle=_audio_dashboard_is_idle(response),
        )

    def _refresh_expiry(
        self,
        workspace_id: str,
        message: discord.Message,
        *,
        idle: bool,
    ) -> None:
        expiry = self._expiry_tasks.pop(workspace_id, None)
        if expiry is not None and expiry is not asyncio.current_task():
            expiry.cancel()
        if not idle:
            return
        self._expiry_tasks[workspace_id] = asyncio.create_task(
            self._expire(workspace_id, message),
            name=f"simajilord-music-dashboard-expiry-{workspace_id}",
        )

    async def _expire(self, workspace_id: str, message: discord.Message) -> None:
        try:
            await asyncio.sleep(self._IDLE_SECONDS)
            if self._messages.get(workspace_id) is not message:
                return
            with suppress(discord.DiscordException):
                await message.delete()
            self._messages.pop(workspace_id, None)
            self._fingerprints.pop(workspace_id, None)
            stored = self._stored_messages.get(workspace_id)
            if stored is not None and stored[1] == message.id:
                self._stored_messages.pop(workspace_id, None)
                await self._persist_stored_messages()
        except asyncio.CancelledError:
            raise
        finally:
            if self._expiry_tasks.get(workspace_id) is asyncio.current_task():
                self._expiry_tasks.pop(workspace_id, None)

    async def _fetch_stored_message(
        self,
        workspace_id: str,
    ) -> discord.Message | None:
        stored = self._stored_messages.get(workspace_id)
        if stored is None:
            return None
        channel_id, message_id = stored
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            with suppress(discord.DiscordException):
                channel = await self.bot.fetch_channel(channel_id)
        fetch_message = getattr(channel, "fetch_message", None)
        if not callable(fetch_message):
            return None
        try:
            message = await fetch_message(message_id)
        except (discord.DiscordException, TypeError):
            return None
        return message if isinstance(message, discord.Message) else None

    def _load_stored_messages(self) -> dict[str, tuple[int, int]]:
        path = self._state_path
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return {}
            raw_messages = payload.get("messages")
            if not isinstance(raw_messages, dict):
                return {}
            restored: dict[str, tuple[int, int]] = {}
            for workspace_id, value in raw_messages.items():
                if (
                    isinstance(workspace_id, str)
                    and isinstance(value, dict)
                    and isinstance(value.get("channel_id"), int)
                    and isinstance(value.get("message_id"), int)
                ):
                    restored[workspace_id] = (
                        value["channel_id"],
                        value["message_id"],
                    )
            return restored
        except (OSError, TypeError, ValueError):
            return {}

    async def _persist_stored_messages(self) -> None:
        path = self._state_path
        if path is None:
            return
        async with self._store_lock:
            payload = {
                "version": 1,
                "messages": {
                    workspace_id: {
                        "channel_id": channel_id,
                        "message_id": message_id,
                    }
                    for workspace_id, (channel_id, message_id) in sorted(
                        self._stored_messages.items()
                    )
                },
            }
            await asyncio.to_thread(
                _write_private_json,
                path,
                payload,
            )

    async def _record_metric(self, operation: str, outcome: str) -> None:
        """Record dashboard transport events without coupling UI health to delivery."""

        try:
            await self.runtime.journal.append(
                kind="service.operation",
                workspace_id=None,
                payload={
                    "operation": operation,
                    "wait_ms": 0.0,
                    "duration_ms": 0.0,
                    "outcome": outcome,
                },
            )
        except Exception:
            log.debug("Could not record dashboard metric %s", operation, exc_info=True)


def _write_private_json(path: Path, payload: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _music_dashboard_fingerprint(
    response: AudioQueueResponse,
    read_aloud_route: ReadAloudRoute | None = None,
) -> tuple[object, ...]:
    """Retain visible state and timing anchors without polling every second."""

    current = response.current
    autoplay_next = response.autoplay_next
    return (
        None
        if current is None
        else (
            current.page_url,
            current.title,
            current.uploader,
            current.thumbnail_url,
            current.duration_seconds,
        ),
        tuple(
            (
                item.page_url,
                item.title,
                item.uploader,
                item.thumbnail_url,
                item.duration_seconds,
                item.requested_by_name,
                item.queue_lane,
            )
            for item in response.pending
            if item.kind == AudioKind.MUSIC.value
        ),
        response.paused,
        round(response.position_seconds, 3) if current is not None else None,
        response.loop_mode,
        response.destination_id,
        response.auto_leave,
        response.waiting_for_voice,
        response.voice_activation_required,
        response.connected,
        response.music_volume_percent,
        response.speech_volume_percent,
        response.speed,
        response.pitch,
        response.autoplay_enabled,
        None
        if autoplay_next is None
        else (
            autoplay_next.page_url,
            autoplay_next.title,
            autoplay_next.uploader,
            autoplay_next.thumbnail_url,
            autoplay_next.duration_seconds,
        ),
        None
        if read_aloud_route is None
        else (
            read_aloud_route.text_channel_ids,
            read_aloud_route.audio_destination_id,
            read_aloud_route.mode.value,
        ),
    )


def _audio_dashboard_is_idle(response: AudioQueueResponse) -> bool:
    """Expire an inactive shell, including Radio state held while disconnected."""

    return (
        response.current is None
        and not any(item.kind == AudioKind.MUSIC.value for item in response.pending)
        and not response.waiting_for_voice
        and not response.voice_activation_required
        and not response.connected
    )


def _unique_messages(
    *messages: discord.Message | None,
) -> tuple[discord.Message, ...]:
    unique: dict[int, discord.Message] = {}
    for message in messages:
        if message is not None:
            unique[message.id] = message
    return tuple(unique.values())


def _schedule_message_delete(message: discord.Message, *, delay: float) -> None:
    async def delete_later() -> None:
        await asyncio.sleep(delay)
        with suppress(discord.DiscordException):
            await message.delete()

    task = asyncio.create_task(
        delete_later(),
        name=f"simajilord-discord-delete-{message.id}",
    )
    task.add_done_callback(
        lambda completed: None if completed.cancelled() else completed.exception()
    )


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
            f"Several results closely match "
            f"**{discord.utils.escape_markdown(response.query)}**. "
            "Your selection is saved in playback history to improve future matches."
        ),
        fields=tuple(fields),
        tone=EmbedTone.WARNING,
    )
    if response.candidates and response.candidates[0].thumbnail_url:
        embed.set_thumbnail(url=response.candidates[0].thumbnail_url)
    return embed


def music_history_embed(response: AudioHistoryResponse) -> discord.Embed:
    if not response.items:
        return command_embed(
            "Playback history",
            description="No tracks have been played yet.",
        )
    lines = []
    for index, item in enumerate(response.items, start=1):
        when = f" · <t:{item.played_at_epoch}:R>" if item.played_at_epoch else ""
        lines.append(
            f"`{index:02d}` {_media_title_link(item.title, item.page_url, maximum=80)} · "
            f"`{_duration(item.duration_seconds)}` · "
            f"{_requester(item.requested_by_name, item.requested_by_id)}{when}"
        )
    return command_embed("Playback history", description="\n".join(lines))


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
                f"No useful sources matched "
                f"**{discord.utils.escape_markdown(response.query)}**. "
                "Try a more specific query or wait if the local search service is degraded."
            ),
            fields=(
                EmbedField("Backend", response.backend),
                EmbedField("Warnings", str(len(response.warnings))),
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
    coverage = f"{response.candidate_count} candidates · {len(response.sources)} shown"
    if response.maybe_more:
        coverage += " · more may be available"
    fields = [EmbedField("Coverage", coverage, inline=False)]
    if response.warnings:
        fields.append(
            EmbedField(
                "Search service",
                f"{len(response.warnings)} source warnings were reported.",
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
        excerpt = "No readable text was found in this range."
    fields = [
        EmbedField("Source", f"[Open original page]({_safe_markdown_url(response.url)})"),
        EmbedField("Format", response.content_type),
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


class WebFetchContinueView(SafeView):
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
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(interaction.id),
                    ),
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
            f"[{discord.utils.escape_markdown(response.title)}]"
            f"({_safe_markdown_url(response.url)}) does not contain "
            f"**{discord.utils.escape_markdown(response.pattern)}**."
        )
        return command_embed(
            "Phrase not found",
            description=description,
            tone=EmbedTone.WARNING,
        )
    lines = tuple(
        _web_match_text(index, match) for index, match in enumerate(response.matches, start=1)
    )
    return command_embed(
        f"Matches in {response.title}"[:256],
        description="\n\n".join(lines)[:3_800],
        fields=(
            EmbedField(
                "Matches",
                f"{len(response.matches)} shown · {response.total_matches} total",
            ),
            EmbedField("Source", f"[Open original page]({_safe_markdown_url(response.url)})"),
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
    if state is None or not isinstance(state.channel, (discord.VoiceChannel, discord.StageChannel)):
        return None
    return state.channel


def _require_same_voice(session: AudioSession, member: discord.abc.User) -> None:
    if not session.output.connected:
        if session.waiting_for_voice and not session.can_control_while_waiting(str(member.id)):
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
            style=(discord.ButtonStyle.primary if index == 0 else discord.ButtonStyle.secondary),
            custom_id=f"simajilord:music:choice:{token}:{index}",
            row=0,
        )
        self.index = index

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.view is not None:
            await self.view.choose(interaction, self.index)


class MusicSearchChoiceView(SafeView):
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
                    "These results belong to another user",
                    description="Run `/play` to search and add your own track.",
                    tone=EmbedTone.WARNING,
                ),
                ephemeral=True,
            )
            return
        async with self._lock:
            if self._selected:
                await interaction.response.send_message(
                    embed=command_embed(
                        "Selection already received",
                        description="The selected track is being added to the queue.",
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
                        "Search results expired",
                        description="Run `/play` again to refresh the search.",
                        tone=EmbedTone.ERROR,
                    ),
                    ephemeral=True,
                )
                return
            self._selected = True
            self._set_disabled(True)
            # Component deferral must target the existing search message. A
            # thinking response creates a separate interaction response and
            # leaves the visible selection controls stale.
            await interaction.response.defer()
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
                    view=None,
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


class MusicControlsView(SafeView):
    """Persistent controls backed by the same capability API as commands and agents."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None = None,
        *,
        response: AudioQueueResponse | None = None,
    ) -> None:
        super().__init__(timeout=None)
        self.runtime = runtime
        self.dashboard = dashboard
        if not getattr(getattr(runtime, "settings", None), "activity_enabled", False):
            self.remove_item(self.open_player_button)
        self._apply_state(response)

    def _apply_state(self, response: AudioQueueResponse | None) -> None:
        """Keep the panel concise while retaining one persistent callback set."""

        if response is None:
            return
        active = response.current is not None
        has_manual_queue = any(item.kind == AudioKind.MUSIC.value for item in response.pending)
        can_start = response.waiting_for_voice or response.voice_activation_required
        if not can_start:
            self.remove_item(self.start_button)
        if not active:
            self.remove_item(self.pause_button)
            self.remove_item(self.skip_button)
        elif response.paused or response.voice_activation_required:
            self.pause_button.label = "Resume"
            self.pause_button.style = discord.ButtonStyle.success
        else:
            self.pause_button.label = "Pause"
            self.pause_button.style = discord.ButtonStyle.secondary
        self.add_music_button.row = 0
        # Secondary actions live in one stateless menu. The decorated buttons remain
        # registered in the persistent callback view but are never rendered.
        self.remove_item(self.loop_button)
        self.remove_item(self.mix_button)
        self.remove_item(self.read_aloud_button)
        self.remove_item(self.leave_button)
        if not (active or has_manual_queue):
            self.remove_item(self.stop_button)
        if can_start:
            self.remove_item(self.pause_button)
            self.remove_item(self.skip_button)
            self.remove_item(self.stop_button)

    def _bind_dashboard(self, interaction: discord.Interaction) -> None:
        dashboard = self._dashboard_manager(interaction)
        if isinstance(dashboard, MusicDashboardManager):
            dashboard.bind(interaction.guild_id, interaction.channel_id)

    def _dashboard_manager(
        self,
        interaction: discord.Interaction,
    ) -> MusicDashboardManager | None:
        dashboard = self.dashboard
        if dashboard is None:
            dashboard = getattr(
                interaction.client,
                _MUSIC_DASHBOARD_ATTRIBUTE,
                None,
            )
        return dashboard if isinstance(dashboard, MusicDashboardManager) else None

    async def _publish_dashboard(
        self,
        interaction: discord.Interaction,
        session: AudioSession,
        response: AudioQueueResponse,
    ) -> None:
        dashboard = self._dashboard_manager(interaction)
        if dashboard is not None:
            await dashboard.publish(
                session,
                obsolete_message=interaction.message,
            )
            return
        await interaction.edit_original_response(
            embed=music_queue_embed(
                response,
                read_aloud_route=_active_read_aloud_route(
                    self.runtime,
                    session.workspace_id,
                ),
            ),
            view=MusicControlsView(
                self.runtime,
                self.dashboard,
                response=response,
            ),
        )

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
            self._bind_dashboard(interaction)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            capability_name, request = audio_control_capability_call(
                action,
                loop_mode=loop_mode,
                position_seconds=position_seconds,
                speed=speed,
                pitch=pitch,
            )
            await self.runtime.registry.invoke(
                capability_name,
                request,
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
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    async def _toggle_loop(self, interaction: discord.Interaction) -> None:
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
        next_mode = modes[(modes.index(current) + 1) % len(modes)]
        if response.autoplay_enabled and next_mode is not LoopMode.NONE:
            conflict_view = LoopMixConflictView(
                self.runtime,
                self.dashboard,
                requester_id=interaction.user.id,
                loop_mode=next_mode,
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Radio is on",
                    description=(
                        "Loop and Radio cannot run together.\n"
                        "Switch Radio off and use "
                        f"**{_loop_mode_label(next_mode.value)} loop**?\n"
                        "Confirm within **1 minute**."
                    ),
                    tone=EmbedTone.WARNING,
                ),
                view=conflict_view,
                ephemeral=True,
            )
            await conflict_view.bind_to_original_response(interaction)
            return
        await self._run(
            interaction,
            AudioAction.LOOP,
            loop_mode=next_mode,
        )

    async def _toggle_radio(self, interaction: discord.Interaction) -> None:
        try:
            self._bind_dashboard(interaction)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            current = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            if not current.autoplay_enabled and LoopMode(current.loop_mode) is not LoopMode.NONE:
                conflict_view = LoopMixConflictView(
                    self.runtime,
                    self.dashboard,
                    requester_id=interaction.user.id,
                )
                await interaction.response.send_message(
                    embed=command_embed(
                        "Loop is on",
                        description=(
                            "Loop and Radio cannot run together.\n"
                            "Turn Loop off and switch to **Radio**?\n"
                            "Confirm within **1 minute**."
                        ),
                        tone=EmbedTone.WARNING,
                    ),
                    view=conflict_view,
                    ephemeral=True,
                )
                await conflict_view.bind_to_original_response(interaction)
                return
            await interaction.response.defer()
            await self.runtime.registry.invoke(
                "discord.set_audio_radio",
                AudioMixRequest(enabled=not current.autoplay_enabled),
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
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.select(
        placeholder="More actions",
        custom_id="simajilord:audio:more",
        row=1,
        min_values=1,
        max_values=1,
        options=[
            discord.SelectOption(
                label="Radio",
                value="radio",
                description="Continue with related tracks",
                emoji="📻",
            ),
            discord.SelectOption(
                label="Loop",
                value="loop",
                description="Cycle off, track, and queue loop",
                emoji="🔁",
            ),
            discord.SelectOption(
                label="Queue",
                value="queue",
                description="Show upcoming requests",
                emoji="📋",
            ),
            discord.SelectOption(
                label="Mix levels",
                value="levels",
                description="Set music and read-aloud volume",
                emoji="🔊",
            ),
            discord.SelectOption(
                label="Read aloud",
                value="read_aloud",
                description="Choose conversation channels",
                emoji="🗣️",
            ),
            discord.SelectOption(
                label="History",
                value="history",
                description="Show recently played tracks",
                emoji="🕘",
            ),
            discord.SelectOption(
                label="Clear my queue",
                value="clear_mine",
                description="Remove only tracks you requested",
                emoji="🧹",
            ),
            discord.SelectOption(
                label="Details",
                value="details",
                description="Show track and playback details",
                emoji="🔎",
            ),
            discord.SelectOption(
                label="Leave",
                value="leave",
                description="Disconnect from voice",
                emoji="🚪",
            ),
        ],
    )
    async def more_actions(
        self,
        interaction: discord.Interaction,
        select: discord.ui.Select[MusicControlsView],
    ) -> None:
        action = select.values[0]
        if action == "radio":
            await self._toggle_radio(interaction)
            return
        if action == "loop":
            await self._toggle_loop(interaction)
            return
        if action == "read_aloud":
            await _send_read_aloud_setup(interaction, self.runtime)
            return
        if action == "levels":
            await interaction.response.send_modal(
                AudioLevelsModal(
                    self.runtime,
                    self._dashboard_manager(interaction),
                )
            )
            return
        if action == "queue":
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_queue_embed(
                    response,
                    read_aloud_route=_active_read_aloud_route(
                        self.runtime,
                        str(interaction.guild_id or ""),
                    ),
                ),
                ephemeral=True,
                silent=True,
            )
            return
        if action == "history":
            history_response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=10),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_history_embed(history_response),
                ephemeral=True,
                silent=True,
            )
            return
        if action == "clear_mine":
            await self._run(interaction, AudioAction.CLEAR_MINE)
            return
        if action == "leave":
            await self._run(interaction, AudioAction.LEAVE)
            return
        if action == "details":
            self._bind_dashboard(interaction)
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_details_embed(response),
                ephemeral=True,
                silent=True,
            )
            return
        await interaction.response.send_message(
            "That audio action is no longer available.",
            ephemeral=True,
        )

    @discord.ui.button(
        label="Open Player",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:audio:open-player",
        row=2,
    )
    async def open_player_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        """Launch the app's official, display-only Discord Activity."""

        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.launch_activity()
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Start",
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
            self._bind_dashboard(interaction)
            channel = _member_voice_channel(interaction.user)
            if channel is None:
                raise UserError("discord.voice_join_required")
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
            await self._publish_dashboard(interaction, session, response)
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(
        label="Pause",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:pause-resume",
        row=0,
    )
    async def pause_button(
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
        action = (
            AudioAction.RESUME
            if response.paused or response.voice_activation_required
            else AudioAction.PAUSE
        )
        await self._run(interaction, action)

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
        row=2,
    )
    async def loop_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._toggle_loop(interaction)

    @discord.ui.button(
        label="Radio",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:mix",
        row=2,
    )
    async def mix_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._toggle_radio(interaction)

    @discord.ui.button(
        label="Stop",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:music:stop",
        row=0,
    )
    async def stop_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.STOP)

    @discord.ui.button(
        label="Add music",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:audio:add-music",
        row=0,
    )
    async def add_music_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await interaction.response.send_modal(
            MusicAddModal(self.runtime, self._dashboard_manager(interaction))
        )

    @discord.ui.button(
        label="Read aloud",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:audio:read-aloud",
        row=2,
    )
    async def read_aloud_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await _send_read_aloud_setup(interaction, self.runtime)

    @discord.ui.button(
        label="Leave",
        style=discord.ButtonStyle.danger,
        custom_id="simajilord:music:leave",
        row=2,
    )
    async def leave_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[MusicControlsView],
    ) -> None:
        await self._run(interaction, AudioAction.LEAVE)


class MusicAddModal(SafeModal, title="Add music"):
    reference: discord.ui.TextInput[MusicAddModal] = discord.ui.TextInput(
        label="Song, artist, or public URL",
        placeholder="What would you like to hear?",
        min_length=1,
        max_length=500,
    )

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.runtime = runtime
        self.dashboard = dashboard

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            await interaction.response.defer(thinking=True, ephemeral=True)
            if self.dashboard is not None:
                self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            bot = cast(commands.Bot, interaction.client)
            session = _discord_audio_session(bot, self.runtime, interaction.guild_id)
            reservation = await session.reserve_manual_music_start()
            try:
                reference = str(self.reference).strip()
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
                        await interaction.edit_original_response(
                            embed=music_search_embed(search),
                            view=MusicSearchChoiceView(
                                bot,
                                self.runtime,
                                search,
                                requester_id=interaction.user.id,
                                requester_name=interaction.user.display_name,
                            ),
                        )
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
            finally:
                await reservation.release()
            await interaction.edit_original_response(
                embed=music_added_embed(response),
                view=None,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class AudioLevelsModal(SafeModal, title="Mix levels"):
    music: discord.ui.TextInput[AudioLevelsModal] = discord.ui.TextInput(
        label="Music volume (0-200%)",
        placeholder="100",
        min_length=1,
        max_length=3,
    )
    read_aloud: discord.ui.TextInput[AudioLevelsModal] = discord.ui.TextInput(
        label="Read-aloud volume (0-200%)",
        placeholder="100",
        min_length=1,
        max_length=3,
    )

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.runtime = runtime
        self.dashboard = dashboard

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            music_percent = _bounded_percent(str(self.music), label="Music")
            speech_percent = _bounded_percent(
                str(self.read_aloud),
                label="Read aloud",
            )
            workspace_id = str(interaction.guild_id or "")
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            capability_name, request = audio_control_capability_call(
                AudioAction.VOLUME,
                music_percent=music_percent,
                speech_percent=speech_percent,
            )
            await self.runtime.registry.invoke(
                capability_name,
                request,
                invocation_context(interaction),
            )
            if self.dashboard is not None:
                self.dashboard.bind(interaction.guild_id, interaction.channel_id)
                await self.dashboard.publish(
                    session,
                    obsolete_message=interaction.message,
                )
        except Exception as exc:
            await send_error(interaction, exc)


def _bounded_percent(value: str, *, label: str) -> int:
    try:
        percent = int(value.strip().removesuffix("%"))
    except ValueError as exc:
        raise UserError("audio.volume_number_invalid") from exc
    if not 0 <= percent <= 200:
        raise UserError("audio.volume_range_invalid")
    return percent


class LoopMixConflictView(SafeView):
    """Ask before replacing one mutually exclusive playback mode."""

    _TIMEOUT_SECONDS = 60

    def __init__(
        self,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None,
        *,
        requester_id: int,
        loop_mode: LoopMode | None = None,
        seed_references: tuple[str, ...] = (),
    ) -> None:
        super().__init__(timeout=self._TIMEOUT_SECONDS)
        self.runtime = runtime
        self.dashboard = dashboard
        self.requester_id = requester_id
        self.loop_mode = loop_mode
        self.seed_references = seed_references
        self.message: discord.InteractionMessage | None = None
        self.confirm_button.label = "Switch to Radio" if loop_mode is None else "Switch to Loop"

    async def bind_to_original_response(
        self,
        interaction: discord.Interaction,
    ) -> None:
        """Remember the ephemeral response so expired controls can be disabled."""

        with suppress(discord.DiscordException):
            self.message = await interaction.original_response()

    async def on_timeout(self) -> None:
        for item in self.children:
            if isinstance(item, (discord.ui.Button, discord.ui.Select)):
                item.disabled = True
        if self.message is not None:
            with suppress(discord.DiscordException):
                await self.message.edit(view=self)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the person who requested this change can confirm it.",
            ephemeral=True,
        )
        return False

    @discord.ui.button(
        label="Switch",
        style=discord.ButtonStyle.primary,
        custom_id="simajilord:music:replace-conflict",
    )
    async def confirm_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[LoopMixConflictView],
    ) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            await interaction.response.defer()
            if self.loop_mode is None:
                await self.runtime.registry.invoke(
                    "discord.set_audio_radio",
                    AudioMixRequest(
                        enabled=True,
                        seed_references=self.seed_references,
                        replace_loop=True,
                    ),
                    invocation_context(interaction),
                )
                title = "Switched to Radio"
                description = "Loop is off. Radio will supply related tracks continuously."
            else:
                capability_name, request = audio_control_capability_call(
                    AudioAction.LOOP,
                    loop_mode=self.loop_mode,
                    replace_mix=True,
                )
                await self.runtime.registry.invoke(
                    capability_name,
                    request,
                    invocation_context(interaction),
                )
                title = "Switched to Loop"
                description = (
                    f"Radio is off. Loop is now **{_loop_mode_label(self.loop_mode.value)}**."
                )
            await interaction.edit_original_response(
                embed=command_embed(
                    title,
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
            self.stop()
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @discord.ui.button(
        label="Keep current mode",
        style=discord.ButtonStyle.secondary,
        custom_id="simajilord:music:keep-conflict",
    )
    async def keep_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[LoopMixConflictView],
    ) -> None:
        await interaction.response.edit_message(
            embed=command_embed(
                "No changes made",
                description="The current playback mode is unchanged.",
            ),
            view=None,
        )
        self.stop()


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


def message_context(message: discord.Message) -> InvocationContext:
    return InvocationContext(
        actor_id=str(message.author.id),
        workspace_id=str(message.guild.id) if message.guild is not None else None,
        transport="discord",
        request_id=str(message.id),
        origin_resource_id=str(message.channel.id),
    )


def async_progress_embed(
    client: discord.Client,
    text: str,
) -> discord.Embed:
    """Show a real in-flight operation without inventing a completed result."""

    return command_embed(
        "Working",
        description=(f"{application_emoji(client, ApplicationEmojiName.LOADING)} {text}"),
    )


def error_message(
    error: Exception,
    *,
    request_id: str | None = None,
) -> str:
    error = _unwrap_discord_error(error)
    if isinstance(error, MediaError):
        return _MEDIA_ERROR_MESSAGES.get(error.category, _MEDIA_ERROR_MESSAGES["unknown"])
    if isinstance(error, WebError):
        return _WEB_ERROR_MESSAGES.get(
            error.category,
            "Could not complete web processing.",
        )
    if isinstance(error, ModerationError):
        return _MODERATION_ERROR_MESSAGES.get(
            error.category,
            "Could not complete the media analysis.",
        )
    if isinstance(error, UserError):
        return _ERROR_MESSAGES.get(error.code, error.code)
    reference_id = request_id or secrets.token_hex(8)
    log.error(
        "Unhandled Discord command error request_id=%s",
        reference_id,
        exc_info=(type(error), error, error.__traceback__),
    )
    return (
        "An unexpected error occurred. "
        f"Reference ID: `{reference_id}`. Share this ID with the administrator."
    )


def _unwrap_discord_error(error: Exception) -> Exception:
    if isinstance(
        error,
        (app_commands.CommandInvokeError, commands.CommandInvokeError),
    ):
        return error.original
    return error


async def handle_interaction_error(
    interaction: discord.Interaction,
    error: Exception,
) -> None:
    """Ensure every command, button, select, and modal has user-visible failure UX."""

    embed = command_embed(
        "Could not complete the request",
        description=error_message(error, request_id=str(interaction.id)),
        tone=EmbedTone.ERROR,
    )
    try:
        if interaction.response.is_done():
            await interaction.edit_original_response(embed=embed, view=None)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=True)
    except discord.NotFound:
        log.warning(
            "Could not publish interaction error because the response expired "
            "request_id=%s",
            interaction.id,
        )
    except discord.DiscordException:
        log.exception(
            "Could not publish interaction error request_id=%s",
            interaction.id,
        )


async def send_error(interaction: discord.Interaction, error: Exception) -> None:
    embed = command_embed(
        "Could not complete the request",
        description=error_message(error, request_id=str(interaction.id)),
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
    if not interaction.response.is_done():
        await send_error(interaction, error)
        return
    try:
        await interaction.edit_original_response(
            embed=command_embed(
                "Could not complete the request",
                description=error_message(error, request_id=str(interaction.id)),
                tone=EmbedTone.ERROR,
            )
        )
    except discord.NotFound:
        log.warning(
            "Could not edit an expired Discord interaction response",
            extra={
                "interaction_id": interaction.id,
                "error_type": type(error).__name__,
            },
        )


class FocusTimerCancelView(SafeView):
    """Short-lived convenience control; timer state itself remains persistent."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        timer: FocusTimer,
    ) -> None:
        super().__init__(timeout=300)
        self.runtime = runtime
        self.timer = timer

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[FocusTimerCancelView],
    ) -> None:
        if str(interaction.user.id) != self.timer.actor_id:
            await interaction.response.send_message(
                "Only the person who started this timer can cancel it.",
                ephemeral=True,
            )
            return
        try:
            await self.runtime.registry.invoke(
                "timer.cancel",
                FocusTimerCancelRequest(self.timer.timer_id),
                invocation_context(interaction),
            )
            await interaction.response.edit_message(
                embed=command_embed(
                    "Focus Timer cancelled",
                    description="The saved timer was cancelled.",
                    tone=EmbedTone.WARNING,
                ),
                view=None,
            )
            self.stop()
        except Exception as exc:
            await send_error(interaction, exc)


class FocusTimerCog(commands.Cog):
    """Discord delivery worker for the transport-neutral persistent timer API."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._worker = asyncio.create_task(
            self._run(),
            name="simajilord-focus-timer-delivery",
        )

    async def cog_unload(self) -> None:
        self._worker.cancel()

    @app_commands.command(
        name="timer",
        description="Set a persistent focus timer with text and optional VC delivery.",
    )
    @app_commands.describe(
        minutes="Duration in minutes, from 1 to 10,080",
        message="Text to post and optionally read when the timer ends",
        voice="Read the reminder in an occupied VC when the BOT is connected",
        focus_session="During focus, read VC events but not regular messages",
    )
    async def timer(
        self,
        interaction: discord.Interaction,
        minutes: app_commands.Range[int, 1, 10080],
        message: str = "Focus session complete.",
        voice: bool = True,
        focus_session: bool = False,
    ) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else None
            channel_id = str(interaction.channel_id) if interaction.channel_id else None
            if workspace_id is None or channel_id is None:
                raise UserError("workspace.required")
            response = cast(
                FocusTimerResponse,
                await self.runtime.registry.invoke(
                    "timer.create",
                    FocusTimerCreateRequest(
                        duration_seconds=minutes * 60,
                        message=message,
                        delivery_target_id=channel_id,
                        voice_notify=voice,
                        focus_session=focus_session,
                    ),
                    invocation_context(interaction),
                ),
            )
            timer = await self.runtime.focus_timer.active(
                workspace_id=workspace_id,
                actor_id=str(interaction.user.id),
            )
            created = next(item for item in timer if item.timer_id == response.timer.timer_id)
            await interaction.response.send_message(
                embed=command_embed(
                    "Focus Timer started",
                    description=(
                        f"Ends <t:{response.timer.due_at_epoch}:R> · "
                        f"<t:{response.timer.due_at_epoch}:t>\n"
                        f"{discord.utils.escape_markdown(response.timer.message)}"
                    ),
                    fields=(
                        EmbedField("Voice", "On" if voice else "Off"),
                        EmbedField(
                            "Focus session",
                            "Events only" if focus_session else "No setting changes",
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=FocusTimerCancelView(self.runtime, created),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def _run(self) -> None:
        await self.bot.wait_until_ready()
        while not self.bot.is_closed():
            try:
                for timer in await self.runtime.focus_timer.claim_due():
                    await self._deliver(timer)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Focus timer delivery scan failed")
            await asyncio.sleep(1)

    async def _deliver(self, timer: FocusTimer) -> None:
        async with self.runtime.focus_timer.delivery_lock(timer.timer_id):
            current = await self.runtime.focus_timer.current(timer.timer_id)
            if current.status is not FocusTimerStatus.DELIVERING:
                return
            await self._deliver_claimed_timer(current)

    async def _deliver_claimed_timer(self, timer: FocusTimer) -> None:
        try:
            try:
                channel_id = int(timer.delivery_target_id)
            except ValueError as exc:
                raise UserError("timer.delivery_target_invalid") from exc
            channel = self.bot.get_channel(channel_id)
            if channel is None:
                channel = await self.bot.fetch_channel(channel_id)
            if not isinstance(
                channel,
                (
                    discord.TextChannel,
                    discord.Thread,
                    discord.VoiceChannel,
                ),
            ):
                raise UserError("timer.delivery_target_invalid")
            completion_message = await self._focus_timer_delivery_message(
                channel,
                timer,
            )
            if completion_message is None:
                embed = command_embed(
                    "Focus Timer complete",
                    description=discord.utils.escape_markdown(timer.message),
                    fields=(EmbedField("Started by", f"<@{timer.actor_id}>"),),
                    tone=EmbedTone.SUCCESS,
                )
                embed.set_footer(text=_focus_timer_delivery_marker(timer.timer_id))
                completion_message = await channel.send(
                    embed=embed,
                    allowed_mentions=discord.AllowedMentions(users=True),
                )
                await self.runtime.focus_timer.set_delivery_message(
                    timer.timer_id,
                    str(completion_message.id),
                )
            await _publish_autonomy_event(
                self.runtime,
                kind=AutonomyEventKind.TIMER_DUE,
                deduplication_key=f"timer-due:{timer.timer_id}",
                workspace_id=timer.workspace_id,
                channel_id=timer.delivery_target_id,
                actor_id=timer.actor_id,
                message_id=str(completion_message.id),
                occurred_at=timer.due_at,
                payload={
                    "timer_id": timer.timer_id,
                    "message_length": len(timer.message),
                    "voice_notify": timer.voice_notify,
                },
            )
            if timer.voice_notify:
                with suppress(Exception):
                    await self._deliver_voice(timer)
            await _complete_focus_timer(self.runtime, timer)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Focus timer delivery failed timer=%s", timer.timer_id)
            try:
                await self.runtime.focus_timer.retry(timer.timer_id)
            except UserError as exc:
                if exc.code != "timer.not_active":
                    log.exception(
                        "Focus timer retry state update failed timer=%s",
                        timer.timer_id,
                    )
            except Exception:
                log.exception(
                    "Focus timer retry state update failed timer=%s",
                    timer.timer_id,
                )

    async def _focus_timer_delivery_message(
        self,
        channel: discord.TextChannel | discord.Thread | discord.VoiceChannel,
        timer: FocusTimer,
    ) -> discord.Message | None:
        """Resolve the one timer message across retry and restart boundaries."""

        if timer.delivery_message_id is not None:
            try:
                return await channel.fetch_message(int(timer.delivery_message_id))
            except (ValueError, discord.NotFound):
                pass

        bot_user = self.bot.user
        if bot_user is None:
            raise RuntimeError("Discord bot identity is unavailable")
        marker = _focus_timer_delivery_marker(timer.timer_id)
        async for candidate in channel.history(
            limit=_FOCUS_TIMER_DELIVERY_RECOVERY_LIMIT,
            after=timer.due_at - timedelta(minutes=5),
            oldest_first=True,
        ):
            if candidate.author.id != bot_user.id:
                continue
            if any(embed.footer.text == marker for embed in candidate.embeds):
                await self.runtime.focus_timer.set_delivery_message(
                    timer.timer_id,
                    str(candidate.id),
                )
                return candidate
        return None

    async def _deliver_voice(self, timer: FocusTimer) -> None:
        session = self.runtime.audio.find(timer.workspace_id)
        if session is None or not session.output.connected or session.destination_id is None:
            return
        guild = self.bot.get_guild(int(timer.workspace_id))
        if guild is None:
            return
        destination = guild.get_channel(int(session.destination_id))
        if not isinstance(
            destination,
            (discord.VoiceChannel, discord.StageChannel),
        ) or not any(not member.bot for member in destination.members):
            return
        await self.runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=timer.message,
                title="Focus Timer",
                voice_preset=self.runtime.read_aloud.policy(
                    timer.workspace_id
                ).default_voice_preset.value,
            ),
            InvocationContext(
                actor_id=timer.actor_id,
                workspace_id=timer.workspace_id,
                transport="discord",
                request_id=f"focus-timer:{timer.timer_id}",
            ),
        )


def _focus_timer_delivery_marker(timer_id: str) -> str:
    return f"Simajilord focus timer · {timer_id}"


async def _complete_focus_timer(
    runtime: SimajilordRuntime,
    timer: FocusTimer,
) -> None:
    async with runtime.focus_timer.focus_session_lock(timer.workspace_id):
        current = await runtime.focus_timer.current(timer.timer_id)
        if current.status is not FocusTimerStatus.DELIVERING:
            return
        if current.focus_session and current.restore_content_mode is not None:
            active_focus = tuple(
                active
                for active in await runtime.focus_timer.active(
                    workspace_id=current.workspace_id
                )
                if active.focus_session and active.timer_id != current.timer_id
            )
            if not active_focus:
                try:
                    mode = ReadAloudContentMode(current.restore_content_mode)
                except ValueError:
                    log.warning(
                        "Focus timer has invalid restore mode timer=%s mode=%s",
                        current.timer_id,
                        current.restore_content_mode,
                    )
                else:
                    await runtime.read_aloud.compare_and_set_content_mode(
                        workspace_id=current.workspace_id,
                        expected=ReadAloudContentMode.EVENTS,
                        mode=mode,
                    )
        await runtime.focus_timer.complete(current.timer_id)


def _help_overview_embed() -> discord.Embed:
    fields = tuple(
        EmbedField(
            (f"{category} · {sum(entry.category == category for entry in HELP_ENTRIES)} commands"),
            description,
            inline=False,
        )
        for category, description in HELP_CATEGORY_DESCRIPTIONS.items()
    )
    return command_embed(
        "Help",
        description=(
            "Choose a category below, or use `/help topic:<command>` for exact "
            "usage, examples, and requirements.\n\n"
            "**Quick start:** `/play` adds music, `/audio` opens the shared "
            "controls, `/join` starts read aloud, and mentioning the BOT starts "
            "an AI conversation."
        ),
        fields=fields,
    )


def _help_category_embed(category: str) -> discord.Embed:
    entries = tuple(entry for entry in HELP_ENTRIES if entry.category == category)
    lines = tuple(
        (
            f"**`/{entry.topic}`** — {entry.summary}"
            if entry.topic != "Quote"
            else f"**`Apps → Quote`** — {entry.summary}"
        )
        for entry in entries
    )
    command_groups: list[list[str]] = []
    for line in lines:
        if command_groups and len("\n".join((*command_groups[-1], line))) <= 1024:
            command_groups[-1].append(line)
        else:
            command_groups.append([line])
    command_fields = tuple(
        EmbedField(
            "Commands" if index == 0 else "Commands · continued",
            "\n".join(group),
            inline=False,
        )
        for index, group in enumerate(command_groups)
    )
    return command_embed(
        category,
        description=HELP_CATEGORY_DESCRIPTIONS[category],
        fields=(
            *command_fields,
            EmbedField(
                "Next",
                "Choose a command below to see its complete usage and examples.",
                inline=False,
            ),
        ),
    )


def _help_entry_embed(entry: PublicCommandSpec) -> discord.Embed:
    examples = "\n".join(f"`{example}`" for example in entry.examples)
    permissions = "\n".join(f"• {item}" for item in entry.permissions)
    side_effects = (
        "\n".join(f"• {item}" for item in entry.side_effects) if entry.side_effects else "None."
    )
    notes = "\n".join(f"• {note}" for note in entry.notes) if entry.notes else "None."
    common_errors = "\n".join(f"• {item}" for item in entry.common_errors)
    return command_embed(
        f"/{entry.topic}" if entry.topic != "Quote" else "Apps → Quote",
        description=f"**Purpose**\n{entry.summary}",
        fields=(
            EmbedField("Usage", f"`{entry.usage}`", inline=False),
            EmbedField("Examples", examples, inline=False),
            EmbedField("Required permissions", permissions, inline=False),
            EmbedField("Side effects", side_effects, inline=False),
            EmbedField("Behaviour notes", notes, inline=False),
            EmbedField("Common errors", common_errors, inline=False),
        ),
    )


class HelpView(SafeView):
    def __init__(self, requester_id: int) -> None:
        super().__init__(timeout=5 * 60)
        self.requester_id = requester_id

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Run `/help` to open your own command browser.",
            ephemeral=True,
        )
        return False

    @discord.ui.select(
        placeholder="Choose a category",
        min_values=1,
        max_values=1,
        options=[
            discord.SelectOption(
                label=category,
                value=category,
                description=description[:100],
            )
            for category, description in HELP_CATEGORY_DESCRIPTIONS.items()
        ],
    )
    async def category_select(
        self,
        interaction: discord.Interaction,
        select: discord.ui.Select[HelpView],
    ) -> None:
        category = select.values[0]
        entries = tuple(entry for entry in HELP_ENTRIES if entry.category == category)
        topic_control = next(
            child
            for child in self.children
            if isinstance(child, discord.ui.Select) and child.placeholder == "Choose a command"
        )
        topic_control.options = [
            discord.SelectOption(
                label=entry.topic,
                value=entry.topic,
                description=entry.summary[:100],
            )
            for entry in entries
        ]
        topic_control.disabled = False
        await interaction.response.edit_message(
            embed=_help_category_embed(category),
            view=self,
        )

    @discord.ui.select(
        placeholder="Choose a command",
        min_values=1,
        max_values=1,
        disabled=True,
        options=[
            discord.SelectOption(
                label="Choose a category first",
                value="_disabled",
            ),
        ],
    )
    async def topic_select(
        self,
        interaction: discord.Interaction,
        select: discord.ui.Select[HelpView],
    ) -> None:
        entry = HELP_ENTRIES_BY_TOPIC.get(select.values[0].casefold())
        if entry is None:
            await interaction.response.send_message(
                "That help topic is no longer available.",
                ephemeral=True,
            )
            return
        await interaction.response.edit_message(
            embed=_help_entry_embed(entry),
            view=self,
        )


class HelpCog(commands.Cog):
    @app_commands.command(
        name="help",
        description="Browse command usage, examples, and requirements.",
    )
    @app_commands.describe(topic="Command or feature to explain")
    async def help(
        self,
        interaction: discord.Interaction,
        topic: str | None = None,
    ) -> None:
        if topic is not None:
            normalized = " ".join(topic.strip().removeprefix("/").split()).casefold()
            entry = HELP_ENTRIES_BY_TOPIC.get(normalized)
            if entry is None:
                await interaction.response.send_message(
                    embed=command_embed(
                        "Unknown help topic",
                        description=(
                            f"No public command matches `{discord.utils.escape_markdown(topic)}`. "
                            "Choose a category below instead."
                        ),
                        tone=EmbedTone.WARNING,
                    ),
                    view=HelpView(interaction.user.id),
                    ephemeral=True,
                )
                return
            await interaction.response.send_message(
                embed=_help_entry_embed(entry),
                view=HelpView(interaction.user.id),
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            embed=_help_overview_embed(),
            view=HelpView(interaction.user.id),
            ephemeral=True,
        )

    @help.autocomplete("topic")
    async def help_topic_autocomplete(
        self,
        _interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        query = current.strip().removeprefix("/").casefold()
        matches = tuple(
            entry
            for entry in HELP_ENTRIES
            if not query or query in entry.topic.casefold() or query in entry.summary.casefold()
        )
        return [
            app_commands.Choice(
                name=(
                    f"/{entry.topic} — {entry.summary}"
                    if entry.topic != "Quote"
                    else f"Apps → Quote — {entry.summary}"
                )[:100],
                value=entry.topic,
            )
            for entry in matches[:25]
        ]


class SystemCog(commands.Cog):
    system = app_commands.Group(
        name="system",
        description="Inspect the BOT runtime and capability catalog.",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @system.command(
        name="ping",
        description="Check BOT health and Discord gateway latency.",
    )
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
                "Health check",
                fields=(
                    EmbedField(
                        "Status",
                        "Operational" if response.status == "ok" else response.status,
                    ),
                    EmbedField(
                        "Discord latency",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @system.command(
        name="capabilities",
        description="Search Simajilord capability APIs by intended task.",
    )
    @app_commands.describe(query="Task or capability to find")
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
                    description="No capability matches that task.",
                    tone=EmbedTone.WARNING,
                )
            )
            return
        lines = [
            f"• `{item.name}` — {item.summary} "
            f"— Risk: **{_risk_label(item.risk)}** / "
            f"Approval: **{_approval_label(item.approval)}**"
            for item in response.capabilities
        ]
        await interaction.response.send_message(
            embed=command_embed(
                "Capabilities",
                description="\n".join(lines),
                fields=(EmbedField("Query", query or "All", inline=False),),
            )
        )

    @system.command(
        name="about",
        description="Explain Simajilord AI and this Discord entrance.",
    )
    async def about(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message(
            embed=command_embed(
                "About Simajilord AI",
                description=(
                    "Simajilord AI connects agents and reusable capabilities through "
                    "one platform API. This BOT is its Discord entrance: music, read "
                    "aloud, media processing, and agent decisions remain independent "
                    "of Discord so other clients can reuse the same capabilities."
                ),
            )
        )

    @system.command(
        name="uptime",
        description="Show process start time and uninterrupted uptime.",
    )
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
                "Uptime",
                fields=(
                    EmbedField(
                        "Started",
                        (
                            f"<t:{int(response.started_at.timestamp())}:F>\n"
                            f"<t:{int(response.started_at.timestamp())}:R>"
                        ),
                        inline=False,
                    ),
                    EmbedField(
                        "Continuous uptime",
                        f"{days}d {hours}h {minutes}m {seconds}s",
                        inline=False,
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @app_commands.command(
        name="status",
        description="Show detailed platform, AI, audio, and web readiness.",
    )
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
                        (
                            "System: "
                            f"**{'Operational' if response.status == 'ok' else response.status}**\n"
                            "AI: "
                            "**"
                            f"{'Enabled' if response.model_runtime == 'enabled' else 'Disabled'}"
                            "**"
                        ),
                    ),
                    EmbedField(
                        "Capabilities",
                        f"Registered APIs: **{response.capability_count}**\n"
                        "Active audio sessions: "
                        f"{response.active_audio_session_count}/"
                        f"{response.audio_session_count}\n"
                        f"Read aloud: **{response.speech_provider.upper()} "
                        f"{response.speech_voice}**",
                    ),
                    EmbedField(
                        "Web search",
                        "Status: "
                        f"**{'Ready' if response.web_search_ready else 'Limited'}**\n"
                        f"Backend: **{response.web_search_backend}**",
                    ),
                    EmbedField(
                        "AI queues",
                        (
                            f"Active: **{response.agent_active_turn_count}**"
                            f" · Pending: **{response.agent_pending_turn_count}**"
                            f" · Ready: **{response.agent_ready_pending_turn_count}**"
                            "\nKeyed registries: "
                            f"workspace **{response.agent_workspace_slot_registry_size}**"
                            " · conversation "
                            f"**{response.agent_conversation_lock_registry_size}**"
                        ),
                    ),
                    EmbedField(
                        "Storage",
                        (
                            f"Used: **{_storage_size(response.storage_used_bytes)}** / "
                            f"{_storage_size(response.storage_limit_bytes)}"
                            + (" · **Limit exceeded**" if response.storage_over_capacity else "")
                            + f"\nQueued audio: **{response.queued_audio_count}**"
                            + (
                                "\nLast cleanup: "
                                f"<t:{response.cleanup_completed_at_epoch}:R> · "
                                f"removed **{response.orphan_cleanup_removed}** files"
                                if response.cleanup_completed_at_epoch is not None
                                else "\nLast cleanup: **Not run**"
                            )
                        ),
                        inline=False,
                    ),
                    EmbedField(
                        "Audio diagnostics",
                        (
                            "Last Radio failure: "
                            + (
                                f"<t:{response.last_radio_failure_at_epoch}:R>"
                                if response.last_radio_failure_at_epoch is not None
                                else "**None in retained logs**"
                            )
                            + f"\nOverlay failures: **{response.overlay_failure_count}**"
                            + f"\nDashboard 429s: **{response.dashboard_429_count}**"
                        ),
                        inline=False,
                    ),
                    EmbedField(
                        "Audit durability",
                        (
                            f"Writer: **{response.audit_writer_state}**"
                            f"\nPending: **{response.audit_pending_event_count}**"
                            f" · Outbox: **{response.audit_outbox_event_count}**"
                            f"\nRetried events: **{response.audit_retried_event_count}**"
                            f" · Lost: **{response.audit_lost_event_count}**"
                            + (
                                "\nLast failure: "
                                f"**{response.audit_last_failure_type or 'unknown'}** · "
                                f"<t:{response.audit_last_failure_at_epoch}:R>"
                                if response.audit_last_failure_at_epoch is not None
                                else "\nLast failure: **None since process start**"
                            )
                        ),
                        inline=False,
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )


class MusicCog(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        runtime: SimajilordRuntime,
        dashboard: MusicDashboardManager | None = None,
    ) -> None:
        self.bot = bot
        self.runtime = runtime
        existing = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard or (
            existing
            if isinstance(existing, MusicDashboardManager)
            else MusicDashboardManager(bot, runtime)
        )

    async def _send_play(
        self,
        interaction: discord.Interaction,
        *,
        reference: str | None = None,
        attachment: discord.Attachment | None = None,
        source_message: discord.Message | None = None,
    ) -> None:
        try:
            if reference is None and attachment is None:
                raise UserError("local_media.selection_required")
            if reference is not None and attachment is not None:
                raise UserError("local_media.multiple_inputs")
            await interaction.response.defer(thinking=True)
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            if attachment is not None:
                record = await import_discord_attachment(
                    self.runtime,
                    attachment,
                    source_message=source_message,
                    uploader=interaction.user,
                )
                selected_reference = record.reference
            else:
                assert reference is not None
                selected_reference = reference
            if attachment is None and "://" not in selected_reference:
                search = cast(
                    AudioSearchResponse,
                    await self.runtime.registry.invoke(
                        "audio.search",
                        AudioSearchRequest(query=selected_reference, limit=5),
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
                        silent=True,
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
            message = await interaction.followup.send(
                embed=music_added_embed(response),
                silent=True,
                wait=True,
            )
            _schedule_message_delete(message, delay=8)
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="play",
        description="Add a song, public URL, or attached audio/video to the queue.",
    )
    @app_commands.describe(
        reference="Song, artist, or public media URL",
        file="Audio or video attachment to keep and play locally",
    )
    async def quick_play(
        self,
        interaction: discord.Interaction,
        reference: str | None = None,
        file: discord.Attachment | None = None,
    ) -> None:
        await self._send_play(
            interaction,
            reference=reference,
            attachment=file,
        )

    async def play_attachment(
        self,
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        attachment = next(
            (item for item in message.attachments if attachment_can_play(item)),
            None,
        )
        if attachment is None:
            await send_error(
                interaction,
                UserError("local_media.content_type_unsupported"),
            )
            return
        await self._send_play(
            interaction,
            attachment=attachment,
            source_message=message,
        )

    async def _send_queue(self, interaction: discord.Interaction, page: int = 1) -> None:
        try:
            await interaction.response.defer(ephemeral=True)
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            session = _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            if page == 1:
                await self.dashboard.publish(session, force=True)
                await interaction.delete_original_response()
                return
            response = cast(
                AudioQueueResponse,
                await self.runtime.registry.invoke(
                    "audio.queue",
                    AudioQueueRequest(),
                    invocation_context(interaction),
                ),
            )
            await interaction.edit_original_response(
                embed=music_queue_embed(
                    response,
                    page=page,
                    read_aloud_route=_active_read_aloud_route(
                        self.runtime,
                        session.workspace_id,
                    ),
                ),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="audio",
        description="Open music controls and read-aloud setup in one panel.",
    )
    async def audio(self, interaction: discord.Interaction) -> None:
        await self._send_queue(interaction)

    async def _send_history(self, interaction: discord.Interaction, limit: int) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            _discord_audio_session(self.bot, self.runtime, interaction.guild_id)
            response = cast(
                AudioHistoryResponse,
                await self.runtime.registry.invoke(
                    "audio.history",
                    AudioHistoryRequest(limit=limit),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=music_history_embed(response),
                silent=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def _send_mix(
        self,
        interaction: discord.Interaction,
        *,
        enabled: bool,
        seeds: str | None,
    ) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            references = tuple(part for part in seeds.split() if part) if seeds is not None else ()
            if enabled:
                current = cast(
                    AudioQueueResponse,
                    await self.runtime.registry.invoke(
                        "audio.queue",
                        AudioQueueRequest(),
                        invocation_context(interaction),
                    ),
                )
                if LoopMode(current.loop_mode) is not LoopMode.NONE:
                    conflict_view = LoopMixConflictView(
                        self.runtime,
                        self.dashboard,
                        requester_id=interaction.user.id,
                        seed_references=references,
                    )
                    await interaction.response.send_message(
                        embed=command_embed(
                            "Loop is on",
                            description=(
                                "Loop and Radio cannot run together.\n"
                                "Would you like to turn off Loop and switch to **Radio**?\n"
                                "Confirm within **1 minute**."
                            ),
                            tone=EmbedTone.WARNING,
                        ),
                        view=conflict_view,
                        ephemeral=True,
                    )
                    await conflict_view.bind_to_original_response(interaction)
                    return
            response = cast(
                AudioMixResponse,
                await self.runtime.registry.invoke(
                    "discord.set_audio_radio",
                    AudioMixRequest(
                        enabled=enabled,
                        seed_references=references,
                    ),
                    invocation_context(interaction),
                ),
            )
            if response.enabled:
                description = (
                    f"Radio will use **{len(response.seed_references)} seed tracks** "
                    "to choose related music. Manual requests always play before "
                    "automatic selections."
                )
            else:
                description = "Automatic selection is off. Pending manual requests are unchanged."
            await interaction.response.send_message(
                embed=command_embed(
                    "Radio started" if response.enabled else "Radio stopped",
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @app_commands.command(
        name="radio",
        description="Keep related music playing while manual requests stay first.",
    )
    @app_commands.describe(
        enabled="Turn Radio on or off; omit to toggle",
        seeds="Up to eight public seed URLs separated by spaces",
    )
    async def quick_radio(
        self,
        interaction: discord.Interaction,
        enabled: bool = True,
        seeds: str | None = None,
    ) -> None:
        await self._send_mix(interaction, enabled=enabled, seeds=seeds)

    async def _control(
        self,
        interaction: discord.Interaction,
        action: AudioAction,
        loop_mode: LoopMode | None = None,
        enabled: bool | None = None,
        position_seconds: float | None = None,
        speed: float | None = None,
        pitch: float | None = None,
        position: int | None = None,
        to_position: int | None = None,
        music_percent: int | None = None,
        speech_percent: int | None = None,
    ) -> None:
        try:
            self.dashboard.bind(interaction.guild_id, interaction.channel_id)
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            capability_name, request = audio_control_capability_call(
                action,
                loop_mode=loop_mode,
                enabled=enabled,
                position_seconds=position_seconds,
                speed=speed,
                pitch=pitch,
                position=position,
                to_position=to_position,
                music_percent=music_percent,
                speech_percent=speech_percent,
            )
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    capability_name,
                    request,
                    invocation_context(interaction),
                ),
            )
            if response.action == AudioAction.LOOP.value:
                message = f"Loop is now **{_loop_mode_label(response.loop_mode or '')}**."
            elif response.action == AudioAction.REMOVE.value:
                message = f"Removed **{response.affected_title}** from the queue."
            elif response.action == AudioAction.AUTO_LEAVE.value:
                message = f"Auto-leave is **{'On' if response.enabled else 'Off'}**."
            elif response.action == AudioAction.SHUFFLE.value:
                message = "Shuffled the pending manual requests."
            elif response.action == AudioAction.SEEK.value:
                formatted_position = _duration(response.position_seconds or 0)
                message = f"Moved playback to `{formatted_position}`."
            elif response.action == AudioAction.TUNE.value:
                message = (
                    f"Speed **{response.speed:.2f}x** · Pitch **{response.pitch:.2f}x**"
                    if response.speed is not None and response.pitch is not None
                    else "Playback tuning updated."
                )
            elif response.action == AudioAction.VOLUME.value:
                message = (
                    f"Music **{response.music_volume_percent}%** · "
                    f"Read aloud **{response.speech_volume_percent}%**"
                )
            elif response.action == AudioAction.MOVE.value:
                message = f"Moved **{response.affected_title}** in the queue."
            elif response.action == AudioAction.CLEAR_MINE.value:
                removed_count = response.removed_count or 0
                message = f"Removed **{removed_count}** of your pending requests."
            else:
                message = _AUDIO_ACTION_MESSAGES[response.action]
            await interaction.response.send_message(
                embed=command_embed(
                    "Audio updated",
                    description=message,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def pause(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.PAUSE)

    async def resume(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.RESUME)

    async def skip(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SKIP)

    async def stop(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.STOP)

    async def leave(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.LEAVE)

    async def loop(
        self,
        interaction: discord.Interaction,
        mode: Literal["none", "track", "queue"],
    ) -> None:
        await self._control(interaction, AudioAction.LOOP, LoopMode(mode))

    async def remove(self, interaction: discord.Interaction, position: int) -> None:
        try:
            workspace_id = str(interaction.guild_id) if interaction.guild_id else ""
            session = self.runtime.audio.require(workspace_id)
            _require_same_voice(session, interaction.user)
            capability_name, request = audio_control_capability_call(
                AudioAction.REMOVE,
                position=position,
            )
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    capability_name,
                    request,
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Removed from queue",
                    description=f"**{response.affected_title}**",
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    async def autoleave(self, interaction: discord.Interaction, enabled: bool) -> None:
        await self._control(
            interaction,
            AudioAction.AUTO_LEAVE,
            enabled=enabled,
        )

    async def shuffle(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.SHUFFLE)

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

    async def volume(
        self,
        interaction: discord.Interaction,
        music: app_commands.Range[int, 0, 200] | None = None,
        read_aloud: app_commands.Range[int, 0, 200] | None = None,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.VOLUME,
            music_percent=None if music is None else int(music),
            speech_percent=None if read_aloud is None else int(read_aloud),
        )

    async def move(
        self,
        interaction: discord.Interaction,
        source: int,
        destination: int,
    ) -> None:
        await self._control(
            interaction,
            AudioAction.MOVE,
            position=source,
            to_position=destination,
        )

    async def clear_mine(self, interaction: discord.Interaction) -> None:
        await self._control(interaction, AudioAction.CLEAR_MINE)


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
    """Stage a bounded set of conversation channels for explicit confirmation."""

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
        self.selected_channel_ids = tuple(str(channel.id) for channel in default_values)
        super().__init__(
            custom_id="simajilord:readaloud:channels",
            channel_types=_READ_ALOUD_CHANNEL_TYPES,
            placeholder="Choose conversation channels",
            min_values=1,
            max_values=25,
            default_values=default_values,
            row=0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the person who opened this setup can change the selection.",
                ephemeral=True,
            )
            return
        self.selected_channel_ids = tuple(str(channel.id) for channel in self.values)
        await interaction.response.defer()

    async def commit(self, interaction: discord.Interaction) -> None:
        """Save the staged route and establish voice only after Start is pressed."""

        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Only the person who opened this setup can start read aloud.",
                ephemeral=True,
            )
            return
        configured: ReadAloudResponse | None = None
        try:
            await interaction.response.defer()
            if not self.selected_channel_ids:
                raise UserError("read_aloud.source_channels_required")
            configured = cast(
                ReadAloudResponse,
                await self.runtime.registry.invoke(
                    "discord.manage_read_aloud",
                    ReadAloudRequest(
                        action=ReadAloudAction.ADD_SOURCES,
                        text_channel_ids=self.selected_channel_ids,
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
                    "Read aloud is ready",
                    description=(
                        "New messages from the selected channels will be spoken automatically."
                    ),
                    fields=(
                        EmbedField(
                            "Reading from",
                            "\n".join(
                                f"<#{channel_id}>" for channel_id in configured.text_channel_ids
                            ),
                            inline=False,
                        ),
                        EmbedField(
                            "Speaking in",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField("Connection", "Ready"),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                view=None,
            )
            dashboard = getattr(
                interaction.client,
                _MUSIC_DASHBOARD_ATTRIBUTE,
                None,
            )
            session = self.runtime.audio.find(str(interaction.guild_id))
            if isinstance(dashboard, MusicDashboardManager) and session is not None:
                await dashboard.publish(session)
        except Exception as exc:
            if configured is None:
                await send_error(interaction, exc)
                return
            log.exception(
                "Read-aloud route was saved but eager voice connection failed guild=%s channel=%s",
                interaction.guild_id,
                self.destination_id,
            )
            await interaction.edit_original_response(
                embed=command_embed(
                    "Read aloud was saved",
                    description=(
                        "The channels are configured, but the voice connection is not ready yet. "
                        "Simajilord will retry when the next message arrives."
                    ),
                    fields=(
                        EmbedField(
                            "Speaking in",
                            f"<#{configured.audio_destination_id}>",
                        ),
                        EmbedField(
                            "Connection",
                            error_message(
                                exc,
                                request_id=str(interaction.id),
                            ),
                        ),
                    ),
                    tone=EmbedTone.WARNING,
                ),
                view=None,
            )


class ReadAloudChannelSelectView(SafeView):
    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        destination_id: int,
        default_values: tuple[discord.abc.GuildChannel | discord.Thread, ...],
    ) -> None:
        super().__init__(timeout=300)
        self.selector = ReadAloudChannelSelect(
            runtime,
            requester_id=requester_id,
            destination_id=destination_id,
            default_values=default_values,
        )
        self.add_item(self.selector)

    @discord.ui.button(
        label="Start",
        style=discord.ButtonStyle.success,
        custom_id="simajilord:readaloud:start",
        row=1,
    )
    async def start(
        self,
        interaction: discord.Interaction,
        _button: discord.ui.Button[ReadAloudChannelSelectView],
    ) -> None:
        await self.selector.commit(interaction)


def _read_aloud_setup(
    interaction: discord.Interaction,
    runtime: SimajilordRuntime,
) -> tuple[discord.Embed, ReadAloudChannelSelectView]:
    member = interaction.user
    source = interaction.channel
    if not isinstance(member, discord.Member):
        raise UserError("workspace.required")
    if not isinstance(
        source,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        raise UserError("discord.message_channel_unavailable")
    destination = member.voice.channel if member.voice is not None else None
    if not isinstance(destination, (discord.VoiceChannel, discord.StageChannel)):
        raise UserError("discord.voice_join_required")

    defaults: list[discord.abc.GuildChannel | discord.Thread] = []
    route = _active_read_aloud_route(runtime, str(member.guild.id))
    candidate_ids = (
        (*route.text_channel_ids, str(source.id))
        if route is not None and route.audio_destination_id == str(destination.id)
        else (str(source.id),)
    )
    for channel_id in dict.fromkeys(candidate_ids):
        selected = member.guild.get_channel_or_thread(int(channel_id))
        if selected is not None:
            defaults.append(selected)

    view = ReadAloudChannelSelectView(
        runtime,
        requester_id=member.id,
        destination_id=destination.id,
        default_values=tuple(defaults[:25]),
    )
    embed = command_embed(
        "Audio · Read aloud",
        description=(
            "Choose up to 25 text channels, threads, or voice-channel chats, then select **Start**."
        ),
        fields=(
            EmbedField("Current channel", source.mention),
            EmbedField("Speaking in", destination.mention),
            EmbedField("Voice", _speech_voice_label(runtime)),
        ),
    )
    return embed, view


async def _send_read_aloud_setup(
    interaction: discord.Interaction,
    runtime: SimajilordRuntime,
) -> None:
    try:
        embed, view = _read_aloud_setup(interaction, runtime)
        await interaction.response.send_message(
            embed=embed,
            view=view,
            ephemeral=True,
        )
    except Exception as exc:
        await send_error(interaction, exc)


class ReadAloudCog(commands.Cog):
    readaloud = app_commands.Group(
        name="readaloud",
        description="Configure automatic Discord message read aloud in voice.",
    )

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._message_formatter = ReadAloudMessageFormatter(runtime.read_aloud)
        self._voice_transitions: dict[
            tuple[int, int],
            tuple[
                discord.Member,
                discord.VoiceChannel | discord.StageChannel | None,
                discord.VoiceChannel | discord.StageChannel | None,
            ],
        ] = {}
        self._announcement_tasks: dict[
            tuple[int, int],
            asyncio.Task[None],
        ] = {}
        self._message_bursts: dict[
            tuple[int, int],
            list[tuple[discord.Message, ReadAloudMessageText]],
        ] = {}
        self._message_burst_tasks: dict[
            tuple[int, int],
            asyncio.Task[None],
        ] = {}

    async def cog_unload(self) -> None:
        for task in self._announcement_tasks.values():
            task.cancel()
        for task in self._message_burst_tasks.values():
            task.cancel()
        self._announcement_tasks.clear()
        self._message_burst_tasks.clear()
        self._message_bursts.clear()
        self._voice_transitions.clear()

    @app_commands.command(
        name="join",
        description="Choose conversation channels to read in your current VC.",
    )
    async def join(self, interaction: discord.Interaction) -> None:
        await _send_read_aloud_setup(interaction, self.runtime)

    @readaloud.command(
        name="setup",
        description="Create or replace one managed source-to-VC route.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        text_channel="Text channel or VC chat to read automatically",
        voice_channel="Voice channel where speech should play",
        mode="Queue speech during music or skip messages posted during music",
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
            if not isinstance(member, discord.Member) or not (
                permission_enabled(member.guild_permissions, "administrator")
                or permission_enabled(member.guild_permissions, "manage_guild")
            ):
                raise UserError("discord.manage_guild_required")
            selected_text = text_channel
            if selected_text is None and isinstance(
                interaction.channel,
                (discord.TextChannel, discord.VoiceChannel),
            ):
                selected_text = interaction.channel
            if selected_text is None:
                raise UserError("read_aloud.source_channel_required")
            selected_voice = voice_channel
            if selected_voice is None and member.voice is not None:
                candidate = member.voice.channel
                if isinstance(candidate, discord.VoiceChannel):
                    selected_voice = candidate
            if selected_voice is None:
                raise UserError("discord.voice_join_required")

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
                    "Read aloud configured",
                    fields=(
                        EmbedField("Source", selected_text.mention),
                        EmbedField("Speaking in", selected_voice.mention),
                        EmbedField("Music behaviour", _read_aloud_mode_label(response.mode)),
                        EmbedField("Voice", _speech_voice_label(self.runtime)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="status",
        description="Show routes, content mode, voices, dictionary, and exclusions.",
    )
    async def status(self, interaction: discord.Interaction) -> None:
        try:
            context = invocation_context(interaction)
            route_result, policy_result = await asyncio.gather(
                self.runtime.registry.invoke(
                    "discord.read_aloud_status",
                    ReadAloudStatusRequest(),
                    context,
                ),
                self.runtime.registry.invoke(
                    "discord.read_aloud_policy_status",
                    ReadAloudStatusRequest(),
                    context,
                ),
            )
            response = cast(ReadAloudResponse, route_result)
            policy = cast(ReadAloudPolicyResponse, policy_result)
            route_fields: tuple[EmbedField, ...] = ()
            if response.enabled:
                route_fields = (
                    EmbedField(
                        "Sources",
                        "\n".join(f"<#{channel_id}>" for channel_id in response.text_channel_ids),
                    ),
                    EmbedField(
                        "Speaking in",
                        f"<#{response.audio_destination_id}>",
                    ),
                    EmbedField("Music behaviour", _read_aloud_mode_label(response.mode)),
                )
            await interaction.response.send_message(
                embed=command_embed(
                    "Read-aloud status",
                    description=(
                        None
                        if response.enabled
                        else (
                            "Automatic read aloud is disabled. Pronunciations and "
                            "other server preferences are still saved."
                        )
                    ),
                    fields=(
                        *route_fields,
                        EmbedField(
                            "Content",
                            {
                                "all": "Messages and voice events",
                                "messages": "Messages",
                                "events": "Voice events",
                                "off": "Off",
                            }.get(policy.content_mode, policy.content_mode),
                        ),
                        EmbedField(
                            "Voice events",
                            (
                                f"Join {_on_off(policy.announce_join)} · "
                                f"Leave {_on_off(policy.announce_leave)} · "
                                f"Move {_on_off(policy.announce_move)}"
                            ),
                        ),
                        EmbedField(
                            "Message semantics",
                            (
                                f"Authors {_on_off(policy.read_author_names)} · "
                                f"Replies {_on_off(policy.read_replies)} · "
                                f"Attachments {_on_off(policy.read_attachments)}\n"
                                "Speakers: "
                                + (
                                    "VC members only"
                                    if policy.vc_members_only
                                    else "Everyone in selected channels"
                                )
                            ),
                        ),
                        EmbedField(
                            "Pronunciations & exclusions",
                            (
                                f"{len(policy.dictionary)} pronunciations · "
                                f"{len(policy.ignored_user_ids)} users · "
                                f"{len(policy.ignored_role_ids)} roles excluded"
                            ),
                        ),
                        EmbedField(
                            "Voices",
                            (
                                f"Server default: {policy.default_voice_preset.title()} · "
                                f"{len(policy.user_voice_presets)} personal presets"
                            ),
                        ),
                    ),
                    tone=(EmbedTone.SUCCESS if response.enabled else EmbedTone.WARNING),
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="mode",
        description="Read messages, voice events, both, or neither.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(mode="Content to read automatically")
    async def content_mode(
        self,
        interaction: discord.Interaction,
        mode: Literal["all", "messages", "events", "off"],
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_content_mode_set",
                    ReadAloudContentModeSetRequest(mode=ReadAloudContentMode(mode)),
                    invocation_context(interaction),
                ),
            )
            labels = {
                "all": "Messages and voice events",
                "messages": "Messages only",
                "events": "Voice events only",
                "off": "Off",
            }
            await interaction.response.send_message(
                embed=command_embed(
                    "Read-aloud content updated",
                    fields=(EmbedField("Content", labels[policy.content_mode]),),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary",
        description="List this server's pronunciation replacements.",
    )
    async def dictionary(self, interaction: discord.Interaction) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_list",
                    ReadAloudDictionaryListRequest(),
                    invocation_context(interaction),
                ),
            )
            entries = policy.dictionary[:20]
            await interaction.response.send_message(
                embed=command_embed(
                    "Pronunciation dictionary",
                    description=(
                        "\n".join(f"`{item.surface}` → {item.reading}" for item in entries)
                        if entries
                        else "No pronunciation replacements are registered."
                    ),
                    fields=(
                        EmbedField(
                            "Entries",
                            (
                                f"{len(policy.dictionary)}"
                                + (" · showing the first 20" if len(policy.dictionary) > 20 else "")
                            ),
                        ),
                    ),
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary-add",
        description="Add or replace one server pronunciation.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        word="Text as it appears in a message",
        reading="Kana or other reading to send to VOICEVOX",
    )
    async def dictionary_add(
        self,
        interaction: discord.Interaction,
        word: str,
        reading: str,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_set",
                    ReadAloudDictionarySetRequest(
                        surface=word,
                        reading=reading,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Pronunciation saved",
                    fields=(
                        EmbedField("Written form", word.strip()),
                        EmbedField("Reading", reading.strip()),
                        EmbedField("Dictionary", f"{len(policy.dictionary)} entries"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="dictionary-remove",
        description="Remove one server pronunciation.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(word="Text to remove from the pronunciation dictionary")
    async def dictionary_remove(
        self,
        interaction: discord.Interaction,
        word: str,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_dictionary_remove",
                    ReadAloudDictionaryRemoveRequest(surface=word),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Pronunciation removed",
                    fields=(
                        EmbedField("Written form", word.strip()),
                        EmbedField("Dictionary", f"{len(policy.dictionary)} entries"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="mute",
        description="Opt your own messages out of or back into read aloud.",
    )
    @app_commands.describe(ignored="True stops reading your messages; false restores them")
    async def mute(
        self,
        interaction: discord.Interaction,
        ignored: bool,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.USER,
                    target_id=str(interaction.user.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Your read-aloud preference was updated",
                    description=(
                        "Your messages will not be read aloud."
                        if ignored
                        else "Your messages may be read aloud again."
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="ignore-user",
        description="Exclude or restore another member's messages.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        user="Member whose read-aloud eligibility should change",
        ignored="True excludes this member; false restores them",
    )
    async def ignore_user(
        self,
        interaction: discord.Interaction,
        user: discord.Member,
        ignored: bool = True,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.USER,
                    target_id=str(user.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Member exclusion updated",
                    fields=(
                        EmbedField("Member", user.mention),
                        EmbedField("Read aloud", "Excluded" if ignored else "Allowed"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="ignore-role",
        description="Exclude or restore messages from members with a role.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        role="Role whose read-aloud eligibility should change",
        ignored="True excludes this role; false restores it",
    )
    async def ignore_role(
        self,
        interaction: discord.Interaction,
        role: discord.Role,
        ignored: bool = True,
    ) -> None:
        try:
            await self.runtime.registry.invoke(
                "discord.read_aloud_exclusion_set",
                ReadAloudExclusionSetRequest(
                    target=ReadAloudExclusionTarget.ROLE,
                    target_id=str(role.id),
                    ignored=ignored,
                ),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Role exclusion updated",
                    fields=(
                        EmbedField("Role", role.mention),
                        EmbedField("Read aloud", "Excluded" if ignored else "Allowed"),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="announcements",
        description="Configure voice join, leave, and move announcements.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        join="Announce members joining the destination VC",
        leave="Announce members leaving the destination VC",
        move="Announce members moving to or from the destination VC",
    )
    async def announcements(
        self,
        interaction: discord.Interaction,
        join: bool | None = None,
        leave: bool | None = None,
        move: bool | None = None,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_announcements_set",
                    ReadAloudAnnouncementsSetRequest(
                        join=join,
                        leave=leave,
                        move=move,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Voice announcements updated",
                    fields=(
                        EmbedField("Join", _on_off(policy.announce_join)),
                        EmbedField("Leave", _on_off(policy.announce_leave)),
                        EmbedField("Move", _on_off(policy.announce_move)),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="message-style",
        description="Configure author, reply, attachment, and VC-member semantics.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(
        author_names="Read an author's name when the speaker changes",
        replies="Describe the author of the replied-to message",
        attachments="Describe image and file attachments",
        vc_members_only="Read messages only from people currently in the VC",
    )
    async def message_style(
        self,
        interaction: discord.Interaction,
        author_names: bool | None = None,
        replies: bool | None = None,
        attachments: bool | None = None,
        vc_members_only: bool | None = None,
    ) -> None:
        try:
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "discord.read_aloud_semantics_set",
                    ReadAloudSemanticsSetRequest(
                        author_names=author_names,
                        replies=replies,
                        attachments=attachments,
                        vc_members_only=vc_members_only,
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Message semantics updated",
                    fields=(
                        EmbedField("Author names", _on_off(policy.read_author_names)),
                        EmbedField("Replies", _on_off(policy.read_replies)),
                        EmbedField("Attachments", _on_off(policy.read_attachments)),
                        EmbedField(
                            "Speakers",
                            (
                                "VC members only"
                                if policy.vc_members_only
                                else "Everyone in selected channels"
                            ),
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="server-voice",
        description="Set the default VOICEVOX voice preset for this server.",
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.describe(preset="Default VOICEVOX voice style for this server")
    @app_commands.choices(
        preset=[
            app_commands.Choice(name="Clear — balanced and intelligible", value="clear"),
            app_commands.Choice(name="Calm — relaxed delivery", value="calm"),
            app_commands.Choice(name="Energetic — bright delivery", value="energetic"),
            app_commands.Choice(name="Cute — friendly delivery", value="cute"),
            app_commands.Choice(name="Narrator — low guidance voice", value="narrator"),
        ]
    )
    async def server_voice(
        self,
        interaction: discord.Interaction,
        preset: app_commands.Choice[str],
    ) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not (
                permission_enabled(member.guild_permissions, "administrator")
                or permission_enabled(member.guild_permissions, "manage_guild")
            ):
                raise UserError("discord.manage_guild_required")
            policy = cast(
                ReadAloudPolicyResponse,
                await self.runtime.registry.invoke(
                    "speech.read_aloud_server_voice_set",
                    ReadAloudServerVoiceSetRequest(
                        preset=ReadAloudVoicePreset(preset.value),
                    ),
                    invocation_context(interaction),
                ),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Server voice updated",
                    fields=(
                        EmbedField(
                            "Voice",
                            policy.default_voice_preset.title(),
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="my-voice",
        description="Choose your voice preset or restore the server default.",
    )
    @app_commands.describe(preset="Your voice style, or Server default to reset it")
    @app_commands.choices(
        preset=[
            app_commands.Choice(name="Server default — clear personal setting", value="default"),
            app_commands.Choice(name="Clear — balanced and intelligible", value="clear"),
            app_commands.Choice(name="Calm — relaxed delivery", value="calm"),
            app_commands.Choice(name="Energetic — bright delivery", value="energetic"),
            app_commands.Choice(name="Cute — friendly delivery", value="cute"),
            app_commands.Choice(name="Narrator — low guidance voice", value="narrator"),
        ]
    )
    async def my_voice(
        self,
        interaction: discord.Interaction,
        preset: app_commands.Choice[str],
    ) -> None:
        try:
            selected = None if preset.value == "default" else ReadAloudVoicePreset(preset.value)
            await self.runtime.registry.invoke(
                "speech.read_aloud_user_voice_set",
                ReadAloudUserVoiceSetRequest(preset=selected),
                invocation_context(interaction),
            )
            await interaction.response.send_message(
                embed=command_embed(
                    "Your voice was updated",
                    fields=(
                        EmbedField(
                            "Voice",
                            "Server default" if selected is None else selected.value.title(),
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                ),
                ephemeral=True,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="remove",
        description="Remove one source channel from the active route.",
    )
    @app_commands.describe(
        channel="Source to remove; omit to remove the current channel",
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
                raise UserError("read_aloud.source_channel_required")
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
                    "Read-aloud source removed",
                    description=(
                        "Read aloud was disabled because no source channels remain."
                        if not response.enabled
                        else "Other configured source channels remain active."
                    ),
                    fields=(
                        EmbedField("Removed source", selected.mention),
                        EmbedField(
                            "Remaining",
                            f"{len(response.text_channel_ids)} channels",
                        ),
                    ),
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @readaloud.command(
        name="disable",
        description="Disable automatic read aloud for this server.",
    )
    @app_commands.default_permissions(manage_guild=True)
    async def disable(self, interaction: discord.Interaction) -> None:
        try:
            member = interaction.user
            if not isinstance(member, discord.Member) or not (
                permission_enabled(member.guild_permissions, "administrator")
                or permission_enabled(member.guild_permissions, "manage_guild")
            ):
                raise UserError("discord.manage_guild_required")
            await self.runtime.registry.invoke(
                "discord.manage_read_aloud",
                ReadAloudRequest(action=ReadAloudAction.DISABLE),
                invocation_context(interaction),
            )
            if interaction.guild_id is not None:
                self._message_formatter.forget_workspace(str(interaction.guild_id))
            await interaction.response.send_message(
                embed=command_embed(
                    "Read aloud disabled",
                    description="Automatic read aloud is now disabled for this server.",
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        workspace_id = str(message.guild.id)
        if not self.runtime.read_aloud.matches(workspace_id, str(message.channel.id)):
            return
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None:
            return
        role_ids = (
            tuple(str(role.id) for role in message.author.roles)
            if isinstance(message.author, discord.Member)
            else ()
        )
        if not self.runtime.read_aloud.allows_message(
            workspace_id=workspace_id,
            author_id=str(message.author.id),
            role_ids=role_ids,
            is_bot=message.author.bot,
            is_webhook=message.webhook_id is not None,
        ):
            return
        destination = message.guild.get_channel(int(route.audio_destination_id))
        if not isinstance(
            destination,
            (discord.VoiceChannel, discord.StageChannel),
        ) or not any(not listener.bot for listener in destination.members):
            return
        if not _read_aloud_audience_allowed(self.runtime, message, destination):
            return
        policy = self.runtime.read_aloud.policy(workspace_id)
        if policy.vc_members_only and (
            not isinstance(message.author, discord.Member)
            or message.author.voice is None
            or message.author.voice.channel is None
            or message.author.voice.channel.id != destination.id
        ):
            return
        prepared = await self._message_formatter.format(message)
        if prepared is None:
            return
        key = (message.guild.id, message.channel.id)
        burst = self._message_bursts.setdefault(key, [])
        if len(burst) < 8:
            burst.append((message, prepared))
        if key not in self._message_burst_tasks:
            self._message_burst_tasks[key] = asyncio.create_task(
                self._flush_message_burst(key),
                name=f"simajilord-read-aloud-burst-{key[0]}-{key[1]}",
            )

    async def _flush_message_burst(self, key: tuple[int, int]) -> None:
        try:
            await asyncio.sleep(0.65)
            burst = tuple(self._message_bursts.pop(key, ()))
            if not burst:
                return
            if self._message_burst_tasks.get(key) is asyncio.current_task():
                self._message_burst_tasks.pop(key, None)
            message = burst[0][0]
            prepared = merge_read_aloud_messages(
                tuple((str(item.author.id), item_prepared) for item, item_prepared in burst)
            )
            await self._deliver_read_aloud(message, prepared)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Automatic read-aloud burst failed guild=%s channel=%s",
                key[0],
                key[1],
            )
        finally:
            if self._message_burst_tasks.get(key) is asyncio.current_task():
                self._message_burst_tasks.pop(key, None)

    async def _deliver_read_aloud(
        self,
        message: discord.Message,
        prepared: ReadAloudMessageText,
    ) -> None:
        if message.guild is None:
            return
        workspace_id = str(message.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None or str(message.channel.id) not in route.text_channel_ids:
            return
        destination = message.guild.get_channel(int(route.audio_destination_id))
        if not isinstance(
            destination,
            (discord.VoiceChannel, discord.StageChannel),
        ) or not any(
            not listener.bot for listener in destination.members
        ) or not _read_aloud_audience_allowed(
            self.runtime,
            message,
            destination,
        ):
            return
        guild_id = message.guild.id
        session = self.runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(self.bot, guild_id),
        )
        if session.voice_activation_required and not session.output.connected:
            return
        if not session.output.connected:
            # A channel message is content to read, not consent to join voice.
            # The listener must use Start, /audio, /join, or an approved agent action.
            return
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
                "speech.speak",
                SpeechSpeakRequest(
                    title=prepared.title,
                    segments=prepared.segments,
                    voice_preset=self.runtime.read_aloud.voice_preset_for(
                        workspace_id=workspace_id,
                        user_id=str(message.author.id),
                    ).value,
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

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Debounce voice transitions before adding an announcement."""

        if member.bot or before.channel == after.channel:
            return
        workspace_id = str(member.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None:
            return
        destination_id = int(route.audio_destination_id)
        before_relevant = before.channel is not None and before.channel.id == destination_id
        after_relevant = after.channel is not None and after.channel.id == destination_id
        if not before_relevant and not after_relevant:
            return

        key = (member.guild.id, member.id)
        previous = self._voice_transitions.get(key)
        initial_channel = previous[1] if previous is not None else before.channel
        self._voice_transitions[key] = (member, initial_channel, after.channel)
        existing = self._announcement_tasks.get(key)
        if existing is not None:
            existing.cancel()
        self._announcement_tasks[key] = asyncio.create_task(
            self._flush_voice_transition(key),
            name=f"simajilord-read-aloud-voice-{member.guild.id}-{member.id}",
        )

    async def _flush_voice_transition(self, key: tuple[int, int]) -> None:
        try:
            await asyncio.sleep(0.7)
            transition = self._voice_transitions.get(key)
            if transition is None:
                return
            member, before_channel, after_channel = transition
            await self._announce_voice_transition(
                member,
                before_channel=before_channel,
                after_channel=after_channel,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Voice read-aloud announcement failed guild=%s member=%s",
                key[0],
                key[1],
            )
        finally:
            if self._announcement_tasks.get(key) is asyncio.current_task():
                self._announcement_tasks.pop(key, None)
                self._voice_transitions.pop(key, None)

    async def _announce_voice_transition(
        self,
        member: discord.Member,
        *,
        before_channel: discord.VoiceChannel | discord.StageChannel | None,
        after_channel: discord.VoiceChannel | discord.StageChannel | None,
    ) -> None:
        workspace_id = str(member.guild.id)
        route = self.runtime.read_aloud.get(workspace_id)
        if route is None or before_channel == after_channel:
            return
        policy = self.runtime.read_aloud.policy(workspace_id)
        destination_id = int(route.audio_destination_id)
        before_id = before_channel.id if before_channel is not None else None
        after_id = after_channel.id if after_channel is not None else None
        name = member.display_name.strip() or member.name

        if before_channel is not None and after_channel is not None:
            if not policy.announce_move or destination_id not in (before_id, after_id):
                return
            text = f"{name}さんが、{before_channel.name}から{after_channel.name}へ移動しました"
        elif after_id == destination_id:
            if not policy.announce_join:
                return
            text = f"{name}さんがボイスチャンネルに参加しました"
        elif before_id == destination_id:
            if not policy.announce_leave:
                return
            text = f"{name}さんがボイスチャンネルから退出しました"
        else:
            return

        destination = member.guild.get_channel(destination_id)
        if not isinstance(destination, (discord.VoiceChannel, discord.StageChannel)):
            return
        if not any(not listener.bot for listener in destination.members):
            return

        session = self.runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(self.bot, member.guild.id),
        )
        if session.voice_activation_required and not session.output.connected:
            return
        if (
            session.has_music
            and session.destination_id is not None
            and session.destination_id != route.audio_destination_id
        ):
            return
        if not session.output.connected:
            return
        spoken_text = self.runtime.read_aloud.apply_dictionary(workspace_id, text)
        await self.runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=spoken_text,
                title="VCの入退室通知",
                segments=(
                    SpeechSegment(
                        SpeechSegmentKind.EVENT,
                        spoken_text,
                    ),
                ),
                voice_preset=policy.default_voice_preset.value,
            ),
            InvocationContext(
                actor_id=str(member.id),
                workspace_id=workspace_id,
                transport="discord",
                request_id=f"read-aloud-voice:{member.id}:{time.time_ns()}",
            ),
        )


class VoiceLifecycleCog(commands.Cog):
    """Keep voice presence aligned with listeners without losing the music queue."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._leave_tasks: dict[str, asyncio.Task[None]] = {}
        dashboard = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard if isinstance(dashboard, MusicDashboardManager) else None

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

        destination_id = int(session.destination_id) if session.destination_id is not None else None
        joined_expected_channel = after.channel is not None and (
            (session.waiting_for_voice and session.can_start_for(str(member.id)))
            or after.channel.id == destination_id
            or joined_read_aloud_destination
        )
        if joined_expected_channel and after.channel is not None:
            if self.dashboard is not None and (
                session.has_music or session.waiting_for_voice or session.voice_activation_required
            ):
                self.dashboard.bind(member.guild.id, after.channel.id)
            task = self._leave_tasks.pop(workspace_id, None)
            if task is not None:
                task.cancel()
            music_targets_another_channel = (
                session.has_music
                and session.destination_id is not None
                and str(after.channel.id) != session.destination_id
            )
            if (
                (session.has_music or joined_read_aloud_destination)
                and not music_targets_another_channel
                and not session.output.connected
            ):
                log.info(
                    "Listener joined while audio is in standby guild=%s; "
                    "waiting for an explicit Start, play, join, or approved agent action",
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
                    if self.dashboard is not None:
                        await self.dashboard.dismiss(workspace_id)
                    return
                if any(not listener.bot for listener in channel.members):
                    return
                await session.suspend()
                if self.dashboard is not None:
                    await self.dashboard.dismiss(workspace_id)
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

    web = app_commands.Group(
        name="web",
        description="Search and inspect public web pages.",
    )

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @web.command(
        name="search",
        description="Search the web through Simajilord's local search service.",
    )
    @app_commands.describe(
        query="Topic, question, or exact phrase to search",
        depth="Search effort: quick, standard, or deep",
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

    @web.command(
        name="fetch",
        description="Fetch readable text and metadata from one public page.",
    )
    @app_commands.describe(
        url="Public HTTP or HTTPS URL to fetch",
        offset="Character offset when continuing a long page",
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

    @web.command(
        name="find",
        description="Find a phrase inside one public web page.",
    )
    @app_commands.describe(
        url="Public HTTP or HTTPS URL to inspect",
        phrase="Case-insensitive phrase to find",
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
    "gptimage2": "GPT Image 2",
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
        return "Medium"
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
    elif "Medium" in severity_labels:
        tone = EmbedTone.WARNING
    else:
        tone = EmbedTone.SUCCESS
    media_label = {"image": "image", "video": "video"}[response.modality.value]
    conclusion_lines = [f"**AI-generated {media_label} likelihood: {ai_likelihood}**"]
    # Avoid emphasizing a deepfake conclusion when one-decimal Discord display
    # rounds the score to 0.0%.
    if response.deepfake_score * 100 >= 0.05:
        conclusion_lines.append(f"Deepfake likelihood: {deepfake_likelihood}")
    conclusion = "\n".join(conclusion_lines)
    fields = [
        EmbedField(
            "AI-generated",
            f"**{_percentage(response.ai_generated_score)}** · {ai_likelihood}",
        ),
        EmbedField(
            "Deepfake",
            f"**{_percentage(response.deepfake_score)}** · {deepfake_likelihood}",
        ),
    ]
    if response.top_source is not None and response.top_source_score >= 0.5:
        fields.append(
            EmbedField(
                "Likely generation source",
                (
                    f"**{_source_name(response.top_source)}** · "
                    f"{_percentage(response.top_source_score)}"
                ),
                inline=False,
            )
        )
    cache_line = (
        "Cached result · no additional request"
        if response.cached
        else "New analysis"
    )
    sample_name = {
        "image": " image",
        "video": " frames",
    }[response.modality.value]
    fields.append(
        EmbedField(
            "Analysis",
            f"{response.sample_count}{sample_name} · {cache_line}",
            inline=False,
        )
    )
    embed = command_embed(
        "AI content analysis",
        description=conclusion,
        fields=tuple(fields),
        tone=tone,
    )
    if attachment_url is not None and response.content_type.startswith("image/"):
        embed.set_thumbnail(url=attachment_url)
    return embed


def _translation_text(value: str, *, maximum: int = 1_000) -> str:
    escaped = discord.utils.escape_markdown(value.strip())
    if len(escaped) <= maximum:
        return escaped
    return escaped[: maximum - 1].rstrip() + "…"


def translation_embed(
    *,
    original: str,
    translation: str,
    source_language: str,
    target_language: str,
    author_name: str | None = None,
    show_original: bool = False,
) -> discord.Embed:
    title = (
        f"Translation · {source_language} → {target_language}"
        if source_language and target_language
        else "Translation"
    )
    fields = [EmbedField("Translation", _translation_text(translation), inline=False)]
    if show_original:
        fields.append(EmbedField("Original", _translation_text(original), inline=False))
    if author_name:
        fields.append(EmbedField("Author", discord.utils.escape_markdown(author_name)))
    return command_embed(
        title,
        fields=tuple(fields),
        tone=EmbedTone.SUCCESS,
    )


def _translated_segment_map(
    response: DiscordTranslateMessageResponse,
) -> dict[str, DiscordTranslatedSegmentRecord]:
    return {item.identifier: item for item in response.segments}


def _translation_result_embeds(
    message: discord.Message,
    response: DiscordTranslateMessageResponse,
    *,
    show_original: bool,
) -> list[discord.Embed]:
    """Rebuild a translated message while retaining Discord embed presentation."""

    translated = _translated_segment_map(response)

    def value(identifier: str, fallback: str | None = None) -> str | None:
        segment = translated.get(identifier)
        return segment.translation if segment is not None else fallback

    description = value("content")
    summary = command_embed(
        f"Translation · {response.source_language} → {response.target_language}",
        description=_translation_text(description, maximum=4_000) if description else None,
        fields=(
            (EmbedField("Author", discord.utils.escape_markdown(response.author_name)),)
            if response.author_name
            else ()
        ),
        tone=EmbedTone.SUCCESS,
    )
    embeds: list[discord.Embed] = [summary]

    for embed_index, source_embed in enumerate(message.embeds[:8]):
        clone = discord.Embed.from_dict(source_embed.to_dict())
        clone.title = value(f"embed.{embed_index}.title", source_embed.title)
        clone.description = value(
            f"embed.{embed_index}.description",
            source_embed.description,
        )
        author_name = value(
            f"embed.{embed_index}.author",
            source_embed.author.name,
        )
        if author_name:
            clone.set_author(
                name=author_name,
                url=source_embed.author.url or None,
                icon_url=source_embed.author.icon_url or None,
            )
        for field_index, field in enumerate(source_embed.fields[:25]):
            clone.set_field_at(
                field_index,
                name=value(
                    f"embed.{embed_index}.field.{field_index}.name",
                    field.name,
                )
                or "\u200b",
                value=value(
                    f"embed.{embed_index}.field.{field_index}.value",
                    field.value,
                )
                or "\u200b",
                inline=field.inline,
            )
        footer_text = value(
            f"embed.{embed_index}.footer",
            source_embed.footer.text,
        )
        if footer_text:
            clone.set_footer(
                text=footer_text,
                icon_url=source_embed.footer.icon_url or None,
            )
        embeds.append(clone)

    supplemental = tuple(
        item
        for item in response.segments
        if item.identifier.startswith(("poll.", "component.", "attachment."))
    )
    if supplemental and len(embeds) < 10:
        fields: list[EmbedField] = []
        for item in supplemental[:25]:
            if item.identifier == "poll.question":
                label = "Poll"
            elif item.identifier.startswith("poll.answer."):
                label = "Option"
            elif item.identifier.startswith("attachment."):
                label = "Attachment"
            else:
                label = "Text"
            fields.append(
                EmbedField(
                    label,
                    _translation_text(item.translation),
                    inline=False,
                )
            )
        embeds.append(
            command_embed(
                "Translated message details",
                fields=tuple(fields),
            )
        )
    if show_original and len(embeds) < 10:
        embeds.append(
            command_embed(
                "Original",
                description=_translation_text(response.original, maximum=4_000),
            )
        )
    return embeds


def _translation_jump_view(jump_url: str | None) -> discord.ui.View | None:
    if jump_url is None:
        return None
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="Jump",
            style=discord.ButtonStyle.link,
            url=jump_url,
        )
    )
    return view


def _resolve_translation_target(
    value: str,
    languages: tuple[TranslationLanguageItem, ...],
) -> str:
    normalized = value.strip().replace("_", "-").casefold()
    if not normalized:
        raise UserError("translation.target_required")
    formatted_code = re.search(r"\(([A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)\)\s*$", value)
    if formatted_code is not None:
        normalized = formatted_code.group(1).casefold()
    for language in languages:
        if language.code.casefold() == normalized:
            return language.code
    named = tuple(
        language
        for language in languages
        if normalized
        in {
            language.english_name.casefold(),
            language.native_name.casefold(),
        }
    )
    if not named:
        raise UserError("translation.language_invalid")
    return min(
        named,
        key=lambda language: (
            "-" in language.code,
            len(language.code),
            language.code,
        ),
    ).code


def _locale_target(
    locales: tuple[str, ...],
    languages: tuple[TranslationLanguageItem, ...],
) -> str | None:
    by_code = {item.code.casefold(): item.code for item in languages}
    for locale in locales:
        normalized = locale.strip().replace("_", "-").casefold()
        if not normalized:
            continue
        if normalized in by_code:
            return by_code[normalized]
        base = normalized.split("-", 1)[0]
        if base in by_code:
            return by_code[base]
        variants = tuple(
            item.code for item in languages if item.code.casefold().split("-", 1)[0] == base
        )
        if len(variants) == 1:
            return variants[0]
        if base == "zh" and variants:
            preferred_token = (
                "hant"
                if any(token in normalized for token in ("tw", "hk", "mo", "hant"))
                else "hans"
            )
            matched = next(
                (code for code in variants if preferred_token in code.casefold()),
                None,
            )
            if matched is not None:
                return matched
    return None


def _interaction_locales(interaction: discord.Interaction) -> tuple[str, ...]:
    values: list[str] = []
    for locale in (interaction.locale, interaction.guild_locale):
        if locale is None:
            continue
        value = getattr(locale, "value", str(locale))
        if value not in values:
            values.append(value)
    return tuple(values)


def _translation_target_autocomplete_choices(
    languages: tuple[TranslationLanguageItem, ...],
    current: str,
) -> list[app_commands.Choice[str]]:
    """Filter every available language while respecting Discord's 25-result cap."""

    terms = current.strip().casefold()
    matches = (
        item
        for item in languages
        if not terms
        or terms in item.code.casefold()
        or terms in item.english_name.casefold()
        or terms in item.native_name.casefold()
    )
    return [
        app_commands.Choice(
            name=(
                f"{_TRANSLATION_LANGUAGE_FLAGS.get(item.code, '🌐')} "
                f"{item.english_name} · {item.native_name} ({item.code})"
            )[:100],
            value=item.code,
        )
        for item in matches
    ][:25]


_TRANSLATION_RELIABLE_CONFIDENCE = 0.80
_TRANSLATION_SHORT_TEXT_CONFIDENCE = 0.95
_TRANSLATION_RELIABLE_MARGIN = 0.20
_TRANSLATION_RECOMMENDED_DETECTION_CHARACTERS = 20

_TRANSLATION_LANGUAGE_FLAGS = {
    "ar-AE": "🇦🇪",
    "zh-TW": "🇹🇼",
    "zh-HK": "🇭🇰",
    "zh": "🇨🇳",
    "da": "🇩🇰",
    "nl": "🇳🇱",
    "en-IN": "🇮🇳",
    "en-CA": "🇨🇦",
    "en-SG": "🇸🇬",
    "en-GB": "🇬🇧",
    "en-ZA": "🇿🇦",
    "en-AU": "🇦🇺",
    "en": "🇺🇸",
    "en-IE": "🇮🇪",
    "en-NZ": "🇳🇿",
    "fr-CA": "🇨🇦",
    "fr": "🇫🇷",
    "de-CH": "🇨🇭",
    "de": "🇩🇪",
    "hi": "🇮🇳",
    "id": "🇮🇩",
    "it-CH": "🇨🇭",
    "it": "🇮🇹",
    "ja": "🇯🇵",
    "ko": "🇰🇷",
    "nb": "🇳🇴",
    "pl": "🇵🇱",
    "pt": "🇧🇷",
    "pt-PT": "🇵🇹",
    "ru": "🇷🇺",
    "es-MX": "🇲🇽",
    "es": "🇪🇸",
    "es-US": "🇺🇸",
    "sv": "🇸🇪",
    "th": "🇹🇭",
    "tr": "🇹🇷",
    "uk": "🇺🇦",
    "vi": "🇻🇳",
}

_TRANSLATION_LANGUAGE_REGIONS: dict[str, tuple[str, ...]] = {
    "Asia & Pacific": (
        "zh-TW",
        "zh-HK",
        "zh",
        "en-IN",
        "en-SG",
        "en-AU",
        "en-NZ",
        "hi",
        "id",
        "ja",
        "ko",
        "th",
        "vi",
    ),
    "Europe": (
        "da",
        "nl",
        "en-GB",
        "en-IE",
        "fr",
        "de-CH",
        "de",
        "it-CH",
        "it",
        "nb",
        "pl",
        "pt-PT",
        "ru",
        "es",
        "sv",
        "tr",
        "uk",
    ),
    "Americas": ("en", "en-CA", "fr-CA", "pt", "es-MX", "es-US"),
    "Middle East & Africa": ("ar-AE", "en-ZA"),
}


def _translation_language_name(
    code: str,
    languages: tuple[TranslationLanguageItem, ...],
) -> str:
    matched = next(
        (item for item in languages if item.code.casefold() == code.casefold()),
        None,
    )
    return f"{matched.english_name} ({matched.code})" if matched is not None else code


def _translation_detection_margin(
    detection: TranslationDetectResponse,
) -> float:
    alternative = max(
        (
            confidence
            for code, confidence in detection.hypotheses
            if code.casefold() != detection.language.casefold()
        ),
        default=0.0,
    )
    return max(0.0, detection.confidence - alternative)


def _translation_detection_is_uncertain(
    text: str,
    detection: TranslationDetectResponse,
) -> bool:
    meaningful_length = len(re.sub(r"\s+", "", text))
    if detection.confidence < _TRANSLATION_RELIABLE_CONFIDENCE:
        return True
    if _translation_detection_margin(detection) < _TRANSLATION_RELIABLE_MARGIN:
        return True
    return (
        meaningful_length < _TRANSLATION_RECOMMENDED_DETECTION_CHARACTERS
        and detection.confidence < _TRANSLATION_SHORT_TEXT_CONFIDENCE
    )


class TranslationRegionSelect(discord.ui.Select["TranslationLanguagePickerView"]):
    """Choose a compact geographic group before selecting a language."""

    def __init__(self, languages: tuple[TranslationLanguageItem, ...]) -> None:
        available = {item.code for item in languages}
        super().__init__(
            placeholder="Choose a region",
            min_values=1,
            max_values=1,
            options=[
                discord.SelectOption(
                    label=region,
                    value=region,
                    description=f"{sum(code in available for code in codes)} languages",
                    emoji={
                        "Asia & Pacific": "🌏",
                        "Europe": "🌍",
                        "Americas": "🌎",
                        "Middle East & Africa": "🌍",
                    }[region],
                )
                for region, codes in _TRANSLATION_LANGUAGE_REGIONS.items()
                if any(code in available for code in codes)
            ],
            row=0,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        await self.view.choose_region(interaction, self.values[0])


class TranslationLanguageSelect(discord.ui.Select["TranslationLanguagePickerView"]):
    """Choose one language from a region-sized set of at most 25."""

    def __init__(
        self,
        region: str,
        languages: tuple[TranslationLanguageItem, ...],
    ) -> None:
        by_code = {item.code: item for item in languages}
        regional = tuple(
            by_code[code] for code in _TRANSLATION_LANGUAGE_REGIONS[region] if code in by_code
        )
        super().__init__(
            placeholder=f"Choose a language · {region}",
            min_values=1,
            max_values=1,
            options=[
                discord.SelectOption(
                    label=f"{item.english_name} ({item.code})"[:100],
                    value=item.code,
                    description=item.native_name[:100],
                    emoji=_TRANSLATION_LANGUAGE_FLAGS.get(item.code, "🌐"),
                )
                for item in regional
            ],
            row=1,
        )

    async def callback(self, interaction: discord.Interaction) -> None:
        assert self.view is not None
        await self.view.choose_language(interaction, self.values[0])


class TranslationLanguagePickerView(SafeView):
    """Region-first language picker without pagination or Next buttons."""

    def __init__(
        self,
        cog: TranslationCog,
        *,
        requester_id: int,
        message: discord.Message,
        languages: tuple[TranslationLanguageItem, ...],
        source_language: str,
        target_language: str,
        show_original: bool,
        mode: Literal["source", "target"],
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.message = message
        self.languages = languages
        self.source_language = source_language
        self.target_language = target_language
        self.show_original = show_original
        self.mode = mode
        self.add_item(TranslationRegionSelect(languages))
        self.add_item(
            discord.ui.Button(
                label="Jump",
                style=discord.ButtonStyle.link,
                url=message.jump_url,
                row=2,
            )
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Use Apps → Translate to choose your own language.",
            ephemeral=True,
        )
        return False

    async def choose_region(
        self,
        interaction: discord.Interaction,
        region: str,
    ) -> None:
        for item in tuple(self.children):
            if isinstance(item, TranslationLanguageSelect):
                self.remove_item(item)
        self.add_item(TranslationLanguageSelect(region, self.languages))
        await interaction.response.edit_message(view=self)

    async def choose_language(
        self,
        interaction: discord.Interaction,
        language: str,
    ) -> None:
        source: str = language if self.mode == "source" else self.source_language
        target: str = language if self.mode == "target" else self.target_language
        if source.casefold() == target.casefold():
            next_mode: Literal["source", "target"] = (
                "source" if self.mode == "target" else "target"
            )
            title = (
                "Choose the source language"
                if next_mode == "source"
                else "Choose a different target"
            )
            description = (
                "The detected source and selected target are both "
                f"**{_translation_language_name(source, self.languages)}**. "
                + (
                    "Choose the language actually used by the message so it can "
                    "still be translated into your selected target."
                    if next_mode == "source"
                    else "The source is now confirmed; choose the language to translate into."
                )
            )
            await interaction.response.edit_message(
                embed=command_embed(title, description=description),
                view=TranslationLanguagePickerView(
                    self.cog,
                    requester_id=self.requester_id,
                    message=self.message,
                    languages=self.languages,
                    source_language=source,
                    target_language=target,
                    show_original=self.show_original,
                    mode=next_mode,
                ),
            )
            return
        try:
            await self.cog._translate_message_from_language_override(
                interaction,
                message=self.message,
                source_language=source,
                target_language=target,
                show_original=self.show_original,
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @discord.ui.button(label="Type language", style=discord.ButtonStyle.secondary, row=2)
    async def type_language(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationLanguagePickerView],
    ) -> None:
        await interaction.response.send_modal(
            TranslationLanguageOverrideModal(
                self.cog,
                requester_id=self.requester_id,
                message=self.message,
                languages=self.languages,
                source_language=self.source_language,
                target_language=self.target_language,
                show_original=self.show_original,
                require_source_confirmation=self.mode == "source",
            )
        )


class TranslationPostView(SafeView):
    def __init__(
        self,
        cog: TranslationCog,
        *,
        requester_id: int,
        message: discord.Message,
        response: DiscordTranslateMessageResponse,
        show_original: bool = False,
        preference_saved: bool = False,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.message = message
        self.response = response
        self.show_original = show_original
        self.toggle_original.label = "Hide original" if show_original else "Show original"
        if preference_saved:
            self.use_by_default.disabled = True
            self.use_by_default.label = "Default saved"
        self.add_item(
            discord.ui.Button(
                label="Jump",
                style=discord.ButtonStyle.link,
                url=response.jump_url,
            )
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Use Apps → Translate to open your own translation.",
            ephemeral=True,
        )
        return False

    @discord.ui.button(label="Show original", style=discord.ButtonStyle.secondary)
    async def toggle_original(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[TranslationPostView],
    ) -> None:
        self.show_original = not self.show_original
        button.label = "Hide original" if self.show_original else "Show original"
        await interaction.response.edit_message(
            embeds=_translation_result_embeds(
                self.message,
                self.response,
                show_original=self.show_original,
            ),
            view=self,
        )

    @discord.ui.button(label="Change language", style=discord.ButtonStyle.secondary)
    async def change_language(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationPostView],
    ) -> None:
        await interaction.response.edit_message(
            embed=command_embed(
                "Choose a target language",
                description="Select a region, then the language you want.",
            ),
            view=TranslationLanguagePickerView(
                self.cog,
                requester_id=self.requester_id,
                message=self.message,
                languages=await self.cog._languages(),
                source_language=self.response.source_language,
                target_language=self.response.target_language,
                show_original=self.show_original,
                mode="target",
            ),
        )

    @discord.ui.button(label="Use by default", style=discord.ButtonStyle.secondary)
    async def use_by_default(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationPostView],
    ) -> None:
        await interaction.response.edit_message(
            embeds=[
                command_embed(
                    "Save as default",
                    description=(
                        f"Use **{self.response.target_language}** for translations "
                        "in this server or everywhere. This preference is private."
                    ),
                )
            ],
            view=TranslationPreferenceScopeView(
                self.cog,
                requester_id=self.requester_id,
                message=self.message,
                response=self.response,
                show_original=self.show_original,
                guild_id=interaction.guild_id,
            ),
        )

    @discord.ui.button(label="Post", style=discord.ButtonStyle.primary)
    async def post(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[TranslationPostView],
    ) -> None:
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await interaction.followup.send(
            embeds=_translation_result_embeds(
                self.message,
                self.response,
                show_original=self.show_original,
            ),
            view=_translation_jump_view(self.response.jump_url) or discord.utils.MISSING,
            allowed_mentions=discord.AllowedMentions.none(),
            silent=True,
        )


class TranslationPreferenceScopeView(SafeView):
    def __init__(
        self,
        cog: TranslationCog,
        *,
        requester_id: int,
        message: discord.Message,
        response: DiscordTranslateMessageResponse,
        show_original: bool,
        guild_id: int | None,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.message = message
        self.response = response
        self.show_original = show_original
        self.guild_id = guild_id
        self.this_server.disabled = guild_id is None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Use Apps → Translate to manage your own translation preference.",
            ephemeral=True,
        )
        return False

    async def _save(
        self,
        interaction: discord.Interaction,
        *,
        workspace_id: str | None,
    ) -> None:
        await self.cog.runtime.translation.set_preference(
            actor_id=str(interaction.user.id),
            workspace_id=workspace_id,
            target_language=self.response.target_language,
            show_original=self.show_original,
        )
        await interaction.response.edit_message(
            embeds=_translation_result_embeds(
                self.message,
                self.response,
                show_original=self.show_original,
            ),
            view=TranslationPostView(
                self.cog,
                requester_id=self.requester_id,
                message=self.message,
                response=self.response,
                show_original=self.show_original,
                preference_saved=True,
            ),
        )

    @discord.ui.button(label="This server", style=discord.ButtonStyle.primary)
    async def this_server(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationPreferenceScopeView],
    ) -> None:
        assert self.guild_id is not None
        await self._save(interaction, workspace_id=str(self.guild_id))

    @discord.ui.button(label="Everywhere", style=discord.ButtonStyle.secondary)
    async def everywhere(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationPreferenceScopeView],
    ) -> None:
        await self._save(interaction, workspace_id=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationPreferenceScopeView],
    ) -> None:
        await interaction.response.edit_message(
            embeds=_translation_result_embeds(
                self.message,
                self.response,
                show_original=self.show_original,
            ),
            view=TranslationPostView(
                self.cog,
                requester_id=self.requester_id,
                message=self.message,
                response=self.response,
                show_original=self.show_original,
            ),
        )


class TranslationLanguageOverrideModal(SafeModal):
    """Resolve source and target by name or BCP-47 without paginated menus."""

    def __init__(
        self,
        cog: TranslationCog,
        *,
        requester_id: int,
        message: discord.Message,
        languages: tuple[TranslationLanguageItem, ...],
        source_language: str,
        target_language: str,
        show_original: bool,
        require_source_confirmation: bool,
    ) -> None:
        super().__init__(title="Translation languages", timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.message = message
        self.languages = languages
        self.show_original = show_original
        source_default = (
            None
            if require_source_confirmation
            else _translation_language_name(source_language, languages)
        )
        self.source: discord.ui.TextInput[TranslationLanguageOverrideModal] = discord.ui.TextInput(
            label="Source language",
            placeholder="English, 日本語, or en",
            default=source_default,
            min_length=2,
            max_length=100,
        )
        self.target: discord.ui.TextInput[TranslationLanguageOverrideModal] = discord.ui.TextInput(
            label="Target language",
            placeholder="Japanese, 日本語, or ja",
            default=_translation_language_name(target_language, languages),
            min_length=2,
            max_length=100,
        )
        self.add_item(self.source)
        self.add_item(self.target)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "Use Apps → Translate to open your own translation.",
                ephemeral=True,
            )
            return
        try:
            source_language = _resolve_translation_target(
                str(self.source),
                self.languages,
            )
            target_language = _resolve_translation_target(
                str(self.target),
                self.languages,
            )
            await self.cog._translate_message_from_language_override(
                interaction,
                message=self.message,
                source_language=source_language,
                target_language=target_language,
                show_original=self.show_original,
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class TranslationDetectionReviewView(SafeView):
    """Let the requester correct uncertain automatic language detection."""

    def __init__(
        self,
        cog: TranslationCog,
        *,
        requester_id: int,
        message: discord.Message,
        languages: tuple[TranslationLanguageItem, ...],
        detected_language: str,
        target_language: str,
        show_original: bool,
    ) -> None:
        super().__init__(timeout=5 * 60)
        self.cog = cog
        self.requester_id = requester_id
        self.message = message
        self.languages = languages
        self.detected_language = detected_language
        self.target_language = target_language
        self.show_original = show_original
        self.add_item(
            discord.ui.Button(
                label="Jump",
                style=discord.ButtonStyle.link,
                url=message.jump_url,
            )
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Use Apps → Translate to review your own language detection.",
            ephemeral=True,
        )
        return False

    def _picker(
        self,
        *,
        mode: Literal["source", "target"],
    ) -> TranslationLanguagePickerView:
        return TranslationLanguagePickerView(
            self.cog,
            requester_id=self.requester_id,
            message=self.message,
            languages=self.languages,
            source_language=self.detected_language,
            target_language=self.target_language,
            show_original=self.show_original,
            mode=mode,
        )

    @discord.ui.button(label="Choose source", style=discord.ButtonStyle.primary)
    async def choose_source(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationDetectionReviewView],
    ) -> None:
        await interaction.response.edit_message(
            embed=command_embed(
                "Choose the source language",
                description="Select a region, then the language used by the message.",
            ),
            view=self._picker(mode="source"),
        )

    @discord.ui.button(label="Change target", style=discord.ButtonStyle.secondary)
    async def change_target(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[TranslationDetectionReviewView],
    ) -> None:
        await interaction.response.edit_message(
            embed=command_embed(
                "Choose a target language",
                description="Select a region, then the language you want.",
            ),
            view=self._picker(mode="target"),
        )


class TranslationCog(commands.Cog):
    """Thin Discord entrances for the provider-neutral translation service."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._language_cache: tuple[TranslationLanguageItem, ...] | None = None

    async def _languages(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguageItem, ...]:
        if source_language is None and self._language_cache is not None:
            return self._language_cache
        response = cast(
            TranslationLanguagesResponse,
            await self.runtime.registry.invoke(
                "translation.languages",
                TranslationLanguagesRequest(source_language=source_language),
                InvocationContext(
                    actor_id="discord-translation-ui",
                    workspace_id=None,
                    transport="discord",
                    request_id=secrets.token_hex(8),
                ),
            ),
        )
        if source_language is None:
            self._language_cache = response.languages
        return response.languages

    def _ranked_languages(
        self,
        languages: tuple[TranslationLanguageItem, ...],
        *,
        recent: tuple[str, ...],
        source_language: str | None,
    ) -> tuple[TranslationLanguageItem, ...]:
        recent_positions = {code: index for index, code in enumerate(recent)}
        common = {
            "en": 0,
            "ja": 1,
            "ko": 2,
            "zh": 3,
            "es": 4,
            "fr": 5,
            "de": 6,
        }
        available = tuple(
            item
            for item in languages
            if item.availability not in {"unsupported", "same_language"}
            and item.code != source_language
        )
        return tuple(
            sorted(
                available,
                key=lambda item: (
                    0 if item.code in recent_positions else 1,
                    recent_positions.get(item.code, 99),
                    common.get(item.code, 99),
                    item.english_name.casefold(),
                    item.code,
                ),
            )
        )

    async def _default_target(
        self,
        interaction: discord.Interaction,
        languages: tuple[TranslationLanguageItem, ...],
    ) -> str:
        target, _ = await self._default_translation_settings(
            interaction,
            languages,
        )
        return target

    async def _default_translation_settings(
        self,
        interaction: discord.Interaction,
        languages: tuple[TranslationLanguageItem, ...],
    ) -> tuple[str, bool]:
        preference = await self.runtime.translation.preference(
            actor_id=str(interaction.user.id),
            workspace_id=(str(interaction.guild_id) if interaction.guild_id is not None else None),
        )
        candidates = (
            ((preference.target_language,) if preference is not None else ())
            + _interaction_locales(interaction)
            + ("en",)
        )
        resolved = _locale_target(candidates, languages)
        if resolved is None:
            raise UserError("translation.target_required")
        return resolved, preference.show_original if preference is not None else False

    async def _translate_message_from_language_override(
        self,
        interaction: discord.Interaction,
        *,
        message: discord.Message,
        source_language: str,
        target_language: str,
        show_original: bool,
    ) -> None:
        if source_language.casefold() == target_language.casefold():
            raise UserError("translation.same_language")
        source_targets = await self._languages(source_language)
        target_status = next(
            (
                item.availability
                for item in source_targets
                if item.code.casefold() == target_language.casefold()
            ),
            "unsupported",
        )
        if target_status == "unsupported":
            raise UserError("translation.language_pair_unsupported")
        if target_status == "supported":
            raise UserError("translation.language_pair_not_installed")
        await interaction.response.edit_message(
            embed=async_progress_embed(
                interaction.client,
                "Translating…",
            ),
            view=None,
        )
        response = cast(
            DiscordTranslateMessageResponse,
            await self.runtime.registry.invoke(
                "discord.translate_message",
                DiscordTranslateMessageRequest(
                    channel_id=str(message.channel.id),
                    message_id=str(message.id),
                    source_language=source_language,
                    target_language=target_language,
                ),
                invocation_context(interaction),
            ),
        )
        await self.runtime.translation.record_recent_target(
            actor_id=str(interaction.user.id),
            code=target_language,
        )
        await interaction.edit_original_response(
            embeds=_translation_result_embeds(
                message,
                response,
                show_original=show_original,
            ),
            view=TranslationPostView(
                self,
                requester_id=interaction.user.id,
                message=message,
                response=response,
                show_original=show_original,
            ),
        )

    @app_commands.command(
        name="translate",
        description="Translate text into another language.",
    )
    @app_commands.describe(
        source="Source language; omit to detect it automatically",
        target="Target language; omit to follow your Discord language",
        text="Text to translate; omit to use the latest visible message",
    )
    async def translate_command(
        self,
        interaction: discord.Interaction,
        source: str | None = None,
        target: str | None = None,
        text: str | None = None,
    ) -> None:
        try:
            await interaction.response.send_message(
                embed=async_progress_embed(
                    interaction.client,
                    "Translating…",
                ),
                ephemeral=True,
                silent=True,
            )
            source_message: discord.Message | None = None
            source_text = text
            if source_text is None:
                channel = interaction.channel
                if not isinstance(
                    channel,
                    (
                        discord.TextChannel,
                        discord.Thread,
                        discord.VoiceChannel,
                        discord.StageChannel,
                    ),
                ):
                    raise UserError("translation.text_required")
                async for candidate in channel.history(limit=10):
                    if discord_translation_segments(candidate):
                        source_message = candidate
                        break
            if source_text is None and source_message is None:
                raise UserError("translation.text_required")
            languages = await self._languages()
            source_code = (
                _resolve_translation_target(source, languages) if source is not None else None
            )
            if target is not None:
                target_code = _resolve_translation_target(target, languages)
                show_original = False
            else:
                target_code, show_original = await self._default_translation_settings(
                    interaction, languages
                )
            if source_message is not None:
                document_response = cast(
                    DiscordTranslateMessageResponse,
                    await self.runtime.registry.invoke(
                        "discord.translate_message",
                        DiscordTranslateMessageRequest(
                            channel_id=str(source_message.channel.id),
                            message_id=str(source_message.id),
                            target_language=target_code,
                            source_language=source_code,
                        ),
                        invocation_context(interaction),
                    ),
                )
                await self.runtime.translation.record_recent_target(
                    actor_id=str(interaction.user.id),
                    code=target_code,
                )
                await interaction.edit_original_response(
                    embeds=_translation_result_embeds(
                        source_message,
                        document_response,
                        show_original=show_original,
                    ),
                    view=TranslationPostView(
                        self,
                        requester_id=interaction.user.id,
                        message=source_message,
                        response=document_response,
                        show_original=show_original,
                    ),
                )
                return
            assert source_text is not None
            response = cast(
                TranslationTranslateResponse,
                await self.runtime.registry.invoke(
                    "translation.translate",
                    TranslationTranslateRequest(
                        text=source_text,
                        target_language=target_code,
                        source_language=source_code,
                    ),
                    invocation_context(interaction),
                ),
            )
            await self.runtime.translation.record_recent_target(
                actor_id=str(interaction.user.id),
                code=target_code,
            )
            await interaction.edit_original_response(
                embed=translation_embed(
                    original=response.original,
                    translation=response.translation,
                    source_language=response.source_language,
                    target_language=response.target_language,
                    show_original=show_original,
                ),
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)

    @translate_command.autocomplete("source")
    async def source_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        try:
            languages = self._ranked_languages(
                await self._languages(),
                recent=(),
                source_language=None,
            )
        except Exception:
            return []
        return _translation_target_autocomplete_choices(languages, current)

    @translate_command.autocomplete("target")
    async def target_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        try:
            languages = self._ranked_languages(
                await self._languages(),
                recent=await self.runtime.translation.recent_targets(
                    actor_id=str(interaction.user.id)
                ),
                source_language=None,
            )
        except Exception:
            return []
        return _translation_target_autocomplete_choices(languages, current)

    async def translate_message(
        self,
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        try:
            await interaction.response.send_message(
                embed=async_progress_embed(
                    interaction.client,
                    "Translating…",
                ),
                ephemeral=True,
                silent=True,
            )
            segments = discord_translation_segments(message)
            if not segments:
                raise UserError("translation.message_text_required")
            detection = cast(
                TranslationDetectResponse,
                await self.runtime.registry.invoke(
                    "translation.detect",
                    TranslationDetectRequest(
                        text="\n".join(item.text for item in segments)[:4_000]
                    ),
                    invocation_context(interaction),
                ),
            )
            all_languages = await self._languages()
            source_languages = await self._languages(detection.language)
            languages = self._ranked_languages(
                source_languages,
                recent=await self.runtime.translation.recent_targets(
                    actor_id=str(interaction.user.id)
                ),
                source_language=detection.language,
            )
            if not languages:
                raise UserError("translation.language_pair_unsupported")
            target_language, show_original = await self._default_translation_settings(
                interaction,
                all_languages,
            )
            target_status = next(
                (
                    item.availability
                    for item in source_languages
                    if item.code.casefold() == target_language.casefold()
                ),
                "unsupported",
            )
            detection_text = "\n".join(item.text for item in segments)[:4_000]
            uncertain = _translation_detection_is_uncertain(
                detection_text,
                detection,
            )
            if uncertain or target_status != "installed":
                confidence = round(detection.confidence * 100)
                detected_name = _translation_language_name(
                    detection.language,
                    all_languages,
                )
                if target_status == "same_language":
                    status_text = (
                        "The best match is also your target language. "
                        "This is an estimate, not proof that translation is unnecessary."
                    )
                elif target_status == "unsupported":
                    status_text = (
                        "That source and target pair is unavailable. "
                        "Choose the source or a different target."
                    )
                elif target_status == "supported":
                    status_text = (
                        "The selected language pair is supported but is not installed yet."
                    )
                else:
                    status_text = (
                        "Short and mixed-language messages can be misidentified. "
                        "Confirm the source or choose another target."
                    )
                with suppress(Exception):
                    await self.runtime.journal.append(
                        kind="translation.detection_review",
                        payload={
                            "detected_language": detection.language,
                            "confidence": detection.confidence,
                            "margin": _translation_detection_margin(detection),
                            "meaningful_characters": len(re.sub(r"\s+", "", detection_text)),
                            "target_language": target_language,
                            "target_status": target_status,
                            "uncertain": uncertain,
                        },
                        actor_id=str(interaction.user.id),
                        workspace_id=(
                            str(interaction.guild_id) if interaction.guild_id is not None else None
                        ),
                        transport="discord",
                        request_id=str(interaction.id),
                    )
                await interaction.edit_original_response(
                    embed=command_embed(
                        "Check the language",
                        description=(
                            f"Best match: **{detected_name}** · **{confidence}%**\n{status_text}"
                        ),
                    ),
                    view=TranslationDetectionReviewView(
                        self,
                        requester_id=interaction.user.id,
                        message=message,
                        languages=all_languages,
                        detected_language=detection.language,
                        target_language=target_language,
                        show_original=show_original,
                    ),
                )
                return
            response = cast(
                DiscordTranslateMessageResponse,
                await self.runtime.registry.invoke(
                    "discord.translate_message",
                    DiscordTranslateMessageRequest(
                        channel_id=str(message.channel.id),
                        message_id=str(message.id),
                        target_language=target_language,
                        # Detection only explains the initial UI. Keep the
                        # automatic request provider-driven unless the user
                        # explicitly chooses a source language.
                        source_language=None,
                    ),
                    invocation_context(interaction),
                ),
            )
            await self.runtime.translation.record_recent_target(
                actor_id=str(interaction.user.id),
                code=target_language,
            )
            await interaction.edit_original_response(
                embeds=_translation_result_embeds(
                    message,
                    response,
                    show_original=show_original,
                ),
                view=TranslationPostView(
                    self,
                    requester_id=interaction.user.id,
                    message=message,
                    response=response,
                    show_original=show_original,
                ),
            )
        except Exception as exc:
            await edit_deferred_error(interaction, exc)


class MediaCog(commands.Cog):
    """Discord upload and download adapters for shared media capabilities."""

    media = app_commands.Group(
        name="media",
        description="Download public media or inspect synthetic media.",
    )

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime
        self._last_request: dict[int, float] = {}

    @media.command(
        name="detect-ai",
        description="Estimate AI-generation and deepfake likelihood.",
    )
    @app_commands.describe(media="Image or video attachment to analyze")
    async def detectai(
        self,
        interaction: discord.Interaction,
        media: discord.Attachment,
    ) -> None:
        await interaction.response.send_message(
            embed=async_progress_embed(
                interaction.client,
                "Analyzing the attachment…",
            ),
            silent=True,
        )
        try:
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            content = await read_attachment_bytes(media)
            if len(content) > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
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

    @media.command(
        name="download",
        description="Save video or audio from a supported public URL.",
    )
    @app_commands.describe(
        url="Public media URL to download",
        media_type="Keep the video or extract audio only",
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
                raise UserError("media.download_cooldown")
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
                        f"[Open source]({response.source_url})"
                    ),
                    fields=(
                        EmbedField(
                            "Size",
                            f"{response.size_bytes / 1_000_000:.1f} MB",
                        ),
                        EmbedField(
                            "Media",
                            "Video" if media_type == "video" else "Audio",
                        ),
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
    utility = app_commands.Group(
        name="utility",
        description="Create polls and make small random choices.",
    )

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @utility.command(
        name="roll",
        description="Roll one or more dice and show each result and the total.",
    )
    @app_commands.describe(
        dice="Number of dice, from 1 to 20",
        sides="Sides per die, from 2 to 1,000",
    )
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

    @utility.command(
        name="choose",
        description="Choose one item from a comma-separated list.",
    )
    @app_commands.describe(options="Comma-separated choices")
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
                    "Selected",
                    description=response.choice,
                    tone=EmbedTone.SUCCESS,
                )
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @utility.command(
        name="poll",
        description="Create a native Discord poll from comma-separated answers.",
    )
    @app_commands.describe(
        question="Question shown at the top of the poll",
        options="Comma-separated answers",
        hours="How long voting stays open, from 1 to 168 hours",
        multiple="Allow voters to select more than one answer",
    )
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
                raise UserError("discord.message_channel_unavailable")
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


def server_info_embed(response: DiscordServerResponse) -> discord.Embed:
    """Render a complete, compact public server profile."""

    owner = f"<@{response.owner_id}>" if response.owner_id is not None else "Unavailable"
    if response.owner_name:
        owner += f"\n{discord.utils.escape_markdown(response.owner_name)}"
    member_count = (
        str(response.member_count) if response.member_count is not None else "Unavailable"
    )
    population = f"**{member_count}** total"
    if response.human_count is not None and response.bot_count is not None:
        population += f"\n{response.human_count} people · {response.bot_count} bots"
    features = ", ".join(feature.replace("_", " ").title() for feature in response.features[:10])
    if len(response.features) > 10:
        features += f" · +{len(response.features) - 10} more"
    description = (
        discord.utils.escape_markdown(response.description)
        if response.description
        else "Public server information"
    )
    embed = command_embed(
        response.name,
        description=f"{description}\n`{response.server_id}`",
        fields=(
            EmbedField("Owner", owner),
            EmbedField("Created", _discord_time_pair(response.created_at_iso)),
            EmbedField("Population", population),
            EmbedField(
                "Channels",
                (
                    f"{response.text_channel_count} text · "
                    f"{response.voice_channel_count} voice\n"
                    f"{response.stage_channel_count} stage · "
                    f"{response.forum_channel_count} forum · "
                    f"{response.category_count} categories"
                ),
                inline=False,
            ),
            EmbedField(
                "Community",
                (
                    f"{response.role_count} roles · {response.emoji_count} emoji · "
                    f"{response.sticker_count} stickers"
                ),
                inline=False,
            ),
            EmbedField(
                "Safety",
                (
                    "Verification: "
                    f"**{_humanize_discord_value(response.verification_level)}**\n"
                    "Media filter: "
                    f"**{_humanize_discord_value(response.explicit_content_filter)}**"
                ),
            ),
            EmbedField(
                "Boosts",
                f"Level **{response.boost_level}** · **{response.boost_count}** boosts",
            ),
            EmbedField("Locale", response.preferred_locale or "Unavailable"),
            *((EmbedField("Features", features, inline=False),) if features else ()),
        ),
    )
    if response.icon_url:
        embed.set_thumbnail(url=response.icon_url)
    return embed


def user_info_embed(
    response: DiscordUserResponse,
    *,
    mention: str,
) -> discord.Embed:
    """Render account identity and guild-specific public membership details."""

    account_label = "BOT account" if response.bot else "User account"
    username = response.username or response.display_name
    embed = command_embed(
        response.display_name,
        description=(
            f"{mention} · `@{discord.utils.escape_markdown(username)}`\n`{response.user_id}`"
        ),
        fields=(),
    )
    if response.colour_value:
        embed.colour = discord.Colour(response.colour_value)
    embed.set_thumbnail(url=response.avatar_url)
    embed.add_field(
        name="Account",
        value=f"{account_label}\nCreated {_discord_time_pair(response.created_at_iso)}",
        inline=False,
    )
    if response.joined_at_iso:
        embed.add_field(
            name="Server membership",
            value=(
                f"Joined {_discord_time_pair(response.joined_at_iso)}\n"
                "Top role: "
                f"**{discord.utils.escape_markdown(response.top_role or 'None')}**"
            ),
            inline=False,
        )
    status_lines = [f"Presence: **{(response.status or 'unknown').title()}**"]
    if response.nickname:
        status_lines.append(f"Nickname: **{discord.utils.escape_markdown(response.nickname)}**")
    if response.pending:
        status_lines.append("Membership screening: **Pending**")
    if response.timed_out_until_iso:
        status_lines.append(f"Timeout ends {_discord_time_pair(response.timed_out_until_iso)}")
    embed.add_field(name="Status", value="\n".join(status_lines), inline=False)
    if response.role_names:
        visible_roles = tuple(
            discord.utils.escape_markdown(name) for name in reversed(response.role_names[-10:])
        )
        role_text = " · ".join(visible_roles)
        if response.role_count > len(visible_roles):
            role_text += f"\n+{response.role_count - len(visible_roles)} more"
        embed.add_field(
            name=f"Roles · {response.role_count}",
            value=role_text,
            inline=False,
        )
    if response.key_permissions:
        embed.add_field(
            name="Key server permissions",
            value=" · ".join(response.key_permissions),
            inline=False,
        )
    return embed


class InfoCog(commands.Cog):
    info = app_commands.Group(
        name="info",
        description="Inspect public Discord server and member information.",
    )

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @info.command(
        name="server",
        description="Show detailed public information about this server.",
    )
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
            await interaction.response.send_message(embed=server_info_embed(response))
        except Exception as exc:
            await send_error(interaction, exc)

    @info.command(
        name="user",
        description="Show public account and server-membership information.",
    )
    @app_commands.describe(user="Member to inspect; omit to inspect yourself")
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
            await interaction.response.send_message(
                embed=user_info_embed(response, mention=target.mention)
            )
        except Exception as exc:
            await send_error(interaction, exc)

    @info.command(
        name="avatar",
        description="Show a member's current display avatar at full size.",
    )
    @app_commands.describe(user="Member whose current display avatar should be shown")
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


def _discord_time_pair(value: str) -> str:
    parsed = discord.utils.parse_time(value)
    if parsed is None:
        return "Unavailable"
    epoch = int(parsed.timestamp())
    return f"<t:{epoch}:F>\n<t:{epoch}:R>"


def _humanize_discord_value(value: str | None) -> str:
    if not value:
        return "Unavailable"
    return value.replace("_", " ").replace(".", " ").title()


class MessageExpandCog(commands.Cog):
    """Expand one bare Discord message link with least-privilege checks."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot or message.webhook_id is not None:
            return
        link = parse_discord_message_link(message.content)
        if link is None:
            return
        try:
            cast(
                DiscordPostExpandedMessageResponse,
                await self.runtime.registry.invoke(
                    "discord.post_expanded_message",
                    DiscordPostExpandedMessageRequest(
                        source_guild_id=link.guild_id,
                        source_channel_id=link.channel_id,
                        source_message_id=link.message_id,
                        destination_channel_id=str(message.channel.id),
                    ),
                    message_context(message),
                ),
            )
            try:
                await message.delete()
            except discord.DiscordException:
                log.info(
                    "Expanded link retained because the source post could not be deleted "
                    "guild=%s channel=%s message=%s",
                    message.guild.id,
                    message.channel.id,
                    message.id,
                )
        except UserError as exc:
            await message.reply(
                embed=command_embed(
                    "Could not expand the message",
                    description=error_message(
                        exc,
                        request_id=str(message.id),
                    ),
                    tone=EmbedTone.WARNING,
                ),
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                silent=True,
            )
        except discord.DiscordException:
            log.exception(
                "Could not publish expanded Discord message guild=%s channel=%s "
                "message=%s source_message=%s",
                message.guild.id,
                message.channel.id,
                message.id,
                link.message_id,
            )


class QuoteCog(commands.Cog):
    """Create one local quote image from Discord's message context menu."""

    def __init__(self, runtime: SimajilordRuntime) -> None:
        self.runtime = runtime

    async def create_quote(
        self,
        interaction: discord.Interaction,
        message: discord.Message,
    ) -> None:
        if interaction.guild_id is None or interaction.channel_id is None:
            await send_error(
                interaction,
                UserError("workspace.required"),
            )
            return
        view = QuoteComposerView(
            self.runtime,
            requester_id=interaction.user.id,
            source_channel_id=message.channel.id,
            source_message_id=message.id,
            destination_channel_id=interaction.channel_id,
            has_animation=quote_message_has_animation(message),
        )
        await interaction.response.send_message(
            embed=view.embed(),
            view=view,
            ephemeral=True,
        )


class QuoteComposerView(SafeView):
    """Private, short-lived quote options without cluttering the result message."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        *,
        requester_id: int,
        source_channel_id: int,
        source_message_id: int,
        destination_channel_id: int,
        has_animation: bool = False,
    ) -> None:
        super().__init__(timeout=300)
        self.runtime = runtime
        self.requester_id = requester_id
        self.source_channel_id = source_channel_id
        self.source_message_id = source_message_id
        self.destination_channel_id = destination_channel_id
        self.color = False
        self.vertical = False
        self.bold = False
        self.flip = False
        self.has_animation = has_animation
        self.animate = False
        self.include_jump = True
        self._page: Literal["main", "layout", "style", "more"] = "main"
        self._show_page("main")

    def embed(self) -> discord.Embed:
        layout = "Vertical · 4:5" if self.vertical else "Landscape · 40:21"
        appearance = "Color" if self.color else "Black / White"
        enabled = [
            label
            for label, active in (
                ("Bold", self.bold),
                ("Flip", self.flip),
                ("Animation", self.animate),
                ("Jump", self.include_jump),
            )
            if active
        ]
        return discord.Embed(
            title="Quote",
            description=(
                {
                    "main": "Review the settings, then select Generate.",
                    "layout": "Choose the image orientation and ratio.",
                    "style": "Choose colour and text emphasis.",
                    "more": "Configure flip, Jump, and animation.",
                }[self._page]
            ),
            color=discord.Colour.green(),
        ).add_field(
            name="Preview settings",
            value=f"{layout}\n{appearance}\n{' · '.join(enabled) or 'Standard'}",
            inline=False,
        )

    def request(self) -> DiscordCreateQuoteImageRequest:
        return DiscordCreateQuoteImageRequest(
            source_channel_id=str(self.source_channel_id),
            source_message_id=str(self.source_message_id),
            destination_channel_id=str(self.destination_channel_id),
            color=self.color,
            vertical=self.vertical,
            bold=self.bold,
            flip=self.flip,
            animate=self.animate,
            include_jump=self.include_jump,
        )

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id == self.requester_id:
            return True
        await interaction.response.send_message(
            "Only the person who opened this Quote menu can use it.",
            ephemeral=True,
        )
        return False

    def _sync_labels(self) -> None:
        self.layout_menu_button.label = (
            "Layout · Vertical" if self.vertical else "Layout · Landscape"
        )
        style_parts = ["Color" if self.color else "B/W"]
        if self.bold:
            style_parts.append("Bold")
        self.style_menu_button.label = f"Style · {' + '.join(style_parts)}"
        more_count = sum((self.flip, self.include_jump, self.animate))
        self.more_menu_button.label = f"More · {more_count} On"
        self.color_button.label = f"Color {'On' if self.color else 'Off'}"
        self.vertical_button.label = "Vertical" if self.vertical else "Landscape"
        self.bold_button.label = f"Bold {'On' if self.bold else 'Off'}"
        self.flip_button.label = f"Flip {'On' if self.flip else 'Off'}"
        self.jump_button.label = f"Jump {'On' if self.include_jump else 'Off'}"
        self.animation_button.label = f"Animation {'On' if self.animate else 'Off'}"
        self.color_button.style = (
            discord.ButtonStyle.primary if self.color else discord.ButtonStyle.secondary
        )
        self.vertical_button.style = (
            discord.ButtonStyle.primary if self.vertical else discord.ButtonStyle.secondary
        )
        self.bold_button.style = (
            discord.ButtonStyle.primary if self.bold else discord.ButtonStyle.secondary
        )
        self.flip_button.style = (
            discord.ButtonStyle.primary if self.flip else discord.ButtonStyle.secondary
        )
        self.jump_button.style = (
            discord.ButtonStyle.primary if self.include_jump else discord.ButtonStyle.secondary
        )
        self.animation_button.style = (
            discord.ButtonStyle.primary if self.animate else discord.ButtonStyle.secondary
        )

    def _show_page(
        self,
        page: Literal["main", "layout", "style", "more"],
    ) -> None:
        self._page = page
        self._sync_labels()
        self.clear_items()
        if page == "main":
            for item in (
                self.layout_menu_button,
                self.style_menu_button,
                self.more_menu_button,
            ):
                item.row = 0
                self.add_item(item)
            for item in (self.generate_button, self.cancel_button):
                item.row = 1
                self.add_item(item)
            return
        choices: tuple[discord.ui.Button[QuoteComposerView], ...]
        if page == "layout":
            choices = (self.vertical_button,)
        elif page == "style":
            choices = (self.color_button, self.bold_button)
        else:
            choices = (self.flip_button, self.jump_button)
            if self.has_animation:
                choices = (*choices, self.animation_button)
        for item in choices:
            item.row = 0
            self.add_item(item)
        self.back_button.row = 1
        self.add_item(self.back_button)

    async def _refresh_composer(self, interaction: discord.Interaction) -> None:
        self._sync_labels()
        await interaction.response.edit_message(embed=self.embed(), view=self)

    async def _toggle_setting(
        self,
        interaction: discord.Interaction,
        setting: Literal["color", "vertical", "bold", "flip", "animate", "include_jump"],
    ) -> None:
        previous = getattr(self, setting)
        setattr(self, setting, not previous)
        try:
            await self._refresh_composer(interaction)
        except Exception:
            # Discord can reject a component acknowledgement after its short
            # interaction window. Keep server-side view state aligned with the
            # still-visible controls when no edit was accepted.
            setattr(self, setting, previous)
            self._sync_labels()
            raise

    async def _open_page(
        self,
        interaction: discord.Interaction,
        page: Literal["main", "layout", "style", "more"],
    ) -> None:
        previous = self._page
        self._show_page(page)
        try:
            await interaction.response.edit_message(embed=self.embed(), view=self)
        except Exception:
            self._show_page(previous)
            raise

    @discord.ui.button(label="Layout", style=discord.ButtonStyle.secondary, row=0)
    async def layout_menu_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._open_page(interaction, "layout")

    @discord.ui.button(label="Style", style=discord.ButtonStyle.secondary, row=0)
    async def style_menu_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._open_page(interaction, "style")

    @discord.ui.button(label="More", style=discord.ButtonStyle.secondary, row=0)
    async def more_menu_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._open_page(interaction, "more")

    @discord.ui.button(label="Color Off", emoji="🎨", row=2)
    async def color_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "color")

    @discord.ui.button(label="Landscape", emoji="↔️", row=2)
    async def vertical_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "vertical")

    @discord.ui.button(label="Bold Off", emoji="🅱️", row=2)
    async def bold_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "bold")

    @discord.ui.button(label="Flip Off", emoji="🔄", row=2)
    async def flip_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "flip")

    @discord.ui.button(label="Jump On", emoji="↗️", row=2)
    async def jump_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "include_jump")

    @discord.ui.button(label="Animation Off", emoji="🎞️", row=3)
    async def animation_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._toggle_setting(interaction, "animate")

    @discord.ui.button(label="Back", style=discord.ButtonStyle.secondary, row=4)
    async def back_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await self._open_page(interaction, "main")

    @discord.ui.button(
        label="Generate",
        emoji="✨",
        style=discord.ButtonStyle.success,
        row=1,
    )
    async def generate_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await interaction.response.defer()
        try:
            await self.runtime.registry.invoke(
                "discord.create_quote_image",
                self.request(),
                invocation_context(interaction),
            )
            await interaction.delete_original_response()
            self.stop()
        except Exception as exc:
            await interaction.edit_original_response(
                embed=command_embed(
                    "Could not generate the quote",
                    description=error_message(
                        exc,
                        request_id=str(interaction.id),
                    ),
                    tone=EmbedTone.ERROR,
                ),
                view=self,
            )

    @discord.ui.button(
        label="Cancel",
        style=discord.ButtonStyle.danger,
        row=1,
    )
    async def cancel_button(
        self,
        interaction: discord.Interaction,
        _: discord.ui.Button[QuoteComposerView],
    ) -> None:
        await interaction.response.defer()
        with suppress(discord.DiscordException):
            await interaction.delete_original_response()
        self.stop()


def discord_conversation_id(
    *,
    guild_id: int | None,
    channel_id: int,
    actor_id: int | str,
    grants: frozenset[str] = frozenset(),
    compatibility_epoch: int = 4,
) -> str:
    """Map one actor, channel, and capability profile to one private conversation."""

    if not 1 <= compatibility_epoch <= 10_000:
        raise ValueError("compatibility epoch must be between 1 and 10000")
    scope = f"guild:{guild_id}" if guild_id is not None else "direct"
    base = (
        f"discord:v{compatibility_epoch}:"
        f"{scope}:channel:{channel_id}:actor:{actor_id}"
    )
    if not grants:
        return base
    profile = "+".join(sorted(grants))
    return f"{base}:profile:{profile}"


async def _publish_autonomy_event(
    runtime: SimajilordRuntime,
    *,
    kind: AutonomyEventKind,
    deduplication_key: str,
    workspace_id: str,
    channel_id: str,
    occurred_at: datetime,
    actor_id: str | None,
    message_id: str | None,
    payload: dict[str, object],
) -> AutonomyEnqueueResult | None:
    """External-source hook shared by timer, audio, GitHub, and RSS adapters."""

    settings = runtime.settings
    if (
        runtime.agent is None
        or not settings.agent_autonomy_enabled
        or settings.agent_autonomy_mode is AgentAutonomyMode.OBSERVE
        or workspace_id not in settings.agent_autonomy_guild_ids
    ):
        return None
    try:
        await runtime.journal.append(
            kind=kind.value,
            actor_id=actor_id,
            workspace_id=workspace_id,
            transport="agent",
            request_id=deduplication_key,
            payload={"channel_id": channel_id, "message_id": message_id, **payload},
        )
        result = await runtime.autonomy_events.enqueue(
            kind=kind,
            deduplication_key=deduplication_key,
            workspace_id=workspace_id,
            channel_id=channel_id,
            occurred_at=occurred_at,
            actor_id=actor_id,
            message_id=message_id,
            payload=payload,
        )
        if result in {
            AutonomyEnqueueResult.QUEUE_FULL,
            AutonomyEnqueueResult.CHANNEL_QUEUE_FULL,
            AutonomyEnqueueResult.ACTOR_QUEUE_FULL,
        }:
            await runtime.journal.append(
                kind="agent.autonomy.event_rejected",
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport="agent",
                request_id=deduplication_key,
                payload={
                    "event_kind": kind.value,
                    "channel_id": channel_id,
                    "reason": result.value,
                },
            )
        return result
    except Exception:
        log.exception(
            "Could not publish autonomy event kind=%s workspace=%s channel=%s",
            kind.value,
            workspace_id,
            channel_id,
        )
        return None


def _agent_grants(
    runtime: SimajilordRuntime,
    *,
    actor_id: str,
    autonomous: bool = False,
    memory_curator: bool = False,
) -> frozenset[str]:
    settings = runtime.settings
    autonomy_mode = settings.agent_autonomy_mode if autonomous else None
    autonomy_policy_mode = getattr(
        settings,
        "agent_autonomy_policy_mode",
        AgentAutonomyPolicyMode.STRICT,
    )
    legacy_autonomy_act = (
        autonomous
        and autonomy_mode is AgentAutonomyMode.ACT
        and autonomy_policy_mode is AgentAutonomyPolicyMode.LEGACY
    )
    grants: set[str] = {AGENT_AUDIO_GRANT, AGENT_MEMORY_GRANT}
    if (
        not autonomous
        and (
            memory_curator
            or actor_id in settings.agent_admin_user_ids
        )
    ):
        grants.add(AGENT_MEMORY_CURATOR_GRANT)
    if not autonomous:
        grants.add(AGENT_FEEDBACK_GRANT)
    if not autonomous or legacy_autonomy_act:
        grants.update((AGENT_QUOTE_GRANT, AGENT_REPOST_GRANT))
    if not autonomous or autonomy_mode in {
        AgentAutonomyMode.ASSIST,
        AgentAutonomyMode.ACT,
    }:
        grants.update((AGENT_MESSAGE_GRANT, AGENT_REACTION_GRANT))
    if (
        (not autonomous or legacy_autonomy_act)
        and settings.agent_file_sandbox_enabled
        and runtime.files is not None
    ):
        grants.add(AGENT_FILE_GRANT)
    web_access = settings.agent_web_search_access
    if web_access is AgentFeatureAccess.EVERYONE or (
        web_access is AgentFeatureAccess.ADMINS and actor_id in settings.agent_admin_user_ids
    ):
        grants.add(AGENT_WEB_GRANT)
    if AGENT_FILE_GRANT in grants and AGENT_WEB_GRANT in grants:
        grants.add(AGENT_MEDIA_GRANT)
    compute_access = settings.agent_safe_compute_access
    if (
        (not autonomous or legacy_autonomy_act)
        and runtime.compute is not None
        and (
            compute_access is AgentFeatureAccess.EVERYONE
            or (
                compute_access is AgentFeatureAccess.ADMINS
                and actor_id in settings.agent_admin_user_ids
            )
        )
    ):
        grants.add(AGENT_COMPUTE_GRANT)
    if not autonomous:
        # Autonomous batches never inherit interactive host escape hatches,
        # regardless of their configured mode or the BOT principal's admin ID.
        shell_access = settings.agent_isolated_shell_access
        if shell_access is AgentFeatureAccess.EVERYONE or (
            shell_access is AgentFeatureAccess.ADMINS
            and actor_id in settings.agent_admin_user_ids
        ):
            grants.add(AGENT_SHELL_GRANT)
        connector_access = settings.agent_connector_access
        if runtime.connectors is not None and (
            connector_access is AgentFeatureAccess.EVERYONE
            or (
                connector_access is AgentFeatureAccess.ADMINS
                and actor_id in settings.agent_admin_user_ids
            )
        ):
            grants.add(AGENT_CONNECTOR_GRANT)
    if not autonomous or legacy_autonomy_act:
        grants.add(AGENT_MODERATION_GRANT)
    if runtime.moderation.provider is not None:
        grants.add(AGENT_HIVE_GRANT)
    image_access = settings.image_generation_access
    if (
        (not autonomous or legacy_autonomy_act)
        and runtime.image.provider is not None
        and (
            image_access is AgentFeatureAccess.EVERYONE
            or (
                image_access is AgentFeatureAccess.ADMINS
                and actor_id in settings.agent_admin_user_ids
            )
        )
    ):
        grants.add(AGENT_IMAGE_GRANT)
    if not autonomous and actor_id in settings.agent_admin_user_ids:
        grants.add(ACTION_UNDO_ANY_GRANT)
    return frozenset(grants)


async def _resolve_requester_principal(
    guild: discord.Guild,
    author: discord.User | discord.Member,
) -> RequesterPrincipal | None:
    """Resolve a human requester with one bounded REST fallback, or fail closed."""

    member = author if isinstance(author, discord.Member) else guild.get_member(author.id)
    if member is None:
        try:
            member = await guild.fetch_member(author.id)
        except (discord.NotFound, discord.Forbidden):
            return None
        except discord.DiscordException:
            log.warning(
                "Discord requester lookup failed guild=%s actor=%s",
                guild.id,
                author.id,
                exc_info=True,
            )
            return None
    try:
        return RequesterPrincipal(member)
    except ValueError:
        return None


_AUTONOMY_ASSIST_WRITES_BY_EVENT: dict[AutonomyEventKind, frozenset[str]] = {
    AutonomyEventKind.MESSAGE_CREATE: frozenset(
        {
            "discord.add_reaction",
            "discord.remove_own_reaction",
        }
    ),
    AutonomyEventKind.MESSAGE_EDIT: frozenset(
        {
            "discord.add_reaction",
            "discord.remove_own_reaction",
        }
    ),
    AutonomyEventKind.REACTION_ADD: frozenset(
        {"discord.add_reaction", "discord.remove_own_reaction"}
    ),
}
_AUTONOMY_ACT_WRITES_BY_EVENT: dict[AutonomyEventKind, frozenset[str]] = {
    AutonomyEventKind.MESSAGE_CREATE: frozenset(
        {
            "discord.send_message",
            "discord.send_embed",
            "discord.reply_message",
            "discord.edit_own_message",
            "discord.delete_own_message",
            *_AUTONOMY_ASSIST_WRITES_BY_EVENT[AutonomyEventKind.MESSAGE_CREATE],
        }
    ),
    AutonomyEventKind.MESSAGE_EDIT: frozenset(
        {
            "discord.send_message",
            "discord.send_embed",
            "discord.reply_message",
            "discord.edit_own_message",
            "discord.delete_own_message",
            *_AUTONOMY_ASSIST_WRITES_BY_EVENT[AutonomyEventKind.MESSAGE_EDIT],
        }
    ),
    AutonomyEventKind.REACTION_ADD: frozenset(
        {
            "discord.send_message",
            "discord.send_embed",
            *_AUTONOMY_ASSIST_WRITES_BY_EVENT[AutonomyEventKind.REACTION_ADD],
        }
    ),
    AutonomyEventKind.THREAD_CREATE: frozenset(
        {"discord.send_message", "discord.send_embed", "discord.reply_message"}
    ),
    # Service-principal voice events may report status, but they must not borrow
    # the bot's current voice membership to perform user-scoped audio writes.
    AutonomyEventKind.VOICE_STATE_UPDATE: frozenset(),
    AutonomyEventKind.AUDIO_ERROR: frozenset(),
}

_AUTONOMY_STRICT_BASE_READ_CAPABILITIES = frozenset(
    {
        "turn.evidence_plan",
        "moderation.status",
        "translation.detect",
        "translation.languages",
        "translation.translate",
        "translation.translate_batch",
        "utility.choose",
        "utility.roll",
        "web.status",
    }
)
_AUTONOMY_EVENT_CHANNEL_READ_CAPABILITIES = frozenset(
    {
        "discord.get_message",
        "discord.read_messages",
        "discord.search_messages",
        "discord.expand_message",
        "discord.translate_message",
        "discord.inspect_channel",
        "discord.list_pins",
        "discord.list_reaction_users",
        "discord.list_poll_voters",
        "discord.list_thread_members",
        "discord.view_custom_emoji",
        "discord.view_image_attachment",
        "discord.view_sticker",
    }
)
_AUTONOMY_CHANNEL_EVENT_KINDS = frozenset(
    {
        AutonomyEventKind.MESSAGE_CREATE,
        AutonomyEventKind.MESSAGE_EDIT,
        AutonomyEventKind.REACTION_ADD,
        AutonomyEventKind.THREAD_CREATE,
    }
)


def _autonomy_event_write_capabilities(
    mode: AgentAutonomyMode,
    event_kinds: frozenset[AutonomyEventKind],
) -> frozenset[str]:
    if mode is AgentAutonomyMode.OBSERVE:
        return frozenset()
    policy = (
        _AUTONOMY_ASSIST_WRITES_BY_EVENT
        if mode is AgentAutonomyMode.ASSIST
        else _AUTONOMY_ACT_WRITES_BY_EVENT
    )
    return frozenset(
        capability
        for event_kind in event_kinds
        for capability in policy.get(event_kind, frozenset())
    )


def _autonomy_approvals(
    runtime: SimajilordRuntime,
    mode: AgentAutonomyMode,
    event_kinds: frozenset[AutonomyEventKind] | None = None,
    *,
    policy_mode: AgentAutonomyPolicyMode | None = None,
) -> frozenset[str]:
    selected_policy = policy_mode or getattr(
        runtime.settings,
        "agent_autonomy_policy_mode",
        AgentAutonomyPolicyMode.STRICT,
    )
    if selected_policy is AgentAutonomyPolicyMode.LEGACY:
        if mode is AgentAutonomyMode.ASSIST:
            return frozenset(AGENT_TIMER_WRITE_CAPABILITIES)
        if mode is not AgentAutonomyMode.ACT:
            return frozenset()
        return frozenset(
            item.descriptor.name
            for item in runtime.registry.all()
            if item.descriptor.approval is ApprovalMode.WHEN_REQUESTED
        )
    kinds = event_kinds or frozenset(AutonomyEventKind)
    allowed = _autonomy_event_write_capabilities(mode, kinds)
    return frozenset(
        item.descriptor.name
        for item in runtime.registry.all()
        if item.descriptor.name in allowed
        and item.descriptor.approval is ApprovalMode.WHEN_REQUESTED
    )


def _autonomy_allowed_capabilities(
    runtime: SimajilordRuntime,
    mode: AgentAutonomyMode,
    event_kinds: frozenset[AutonomyEventKind],
    *,
    policy_mode: AgentAutonomyPolicyMode,
) -> frozenset[str] | None:
    """Return an event-specific catalog ceiling; legacy keeps the old catalog."""

    if policy_mode is AgentAutonomyPolicyMode.LEGACY:
        return None
    writes = _autonomy_event_write_capabilities(mode, event_kinds)
    reads = set(_AUTONOMY_STRICT_BASE_READ_CAPABILITIES)
    if event_kinds.intersection(_AUTONOMY_CHANNEL_EVENT_KINDS):
        reads.update(_AUTONOMY_EVENT_CHANNEL_READ_CAPABILITIES)
    return frozenset(
        endpoint.descriptor.name
        for endpoint in runtime.registry.all()
        if (
            endpoint.descriptor.egress is None
            and (
                endpoint.descriptor.name in reads
                or endpoint.descriptor.name in writes
            )
        )
    )


_discord_message_chunks = discord_message_chunks
_agent_message_groups = agent_message_groups
_agent_delivery_nonce = agent_delivery_nonce
_agent_error_text = agent_error_text
_retry_after_text = retry_after_text
_agent_progress_text = agent_progress_text
_AgentProgressMessage = AgentProgressMessage
_AGENT_INTERRUPTED_MENTION_MAX_AGE = timedelta(hours=24)


def _agent_invocation_context(request: AgentRequest) -> InvocationContext:
    """Mirror the provider context for host-delivered Discord action receipts."""

    return InvocationContext(
        actor_id=request.actor_id,
        workspace_id=request.workspace_id,
        transport="agent",
        request_id=request.event_id,
        resource_ids=request.resource_ids,
        grants=request.grants,
        origin_resource_id=request.channel_id,
        approvals=request.approvals,
        public_reference_id=request.public_reference_id,
        principal_kind=request.principal_kind,
        read_scope_mode=request.read_scope_mode,
        information_flow_mode=request.information_flow_mode.value,
        file_workspace_mode=request.file_workspace_mode.value,
        high_risk_authorization_mode=request.high_risk_authorization_mode.value,
        executor_principal_id=request.executor_principal_id,
        delegator_principal_id=request.delegator_principal_id,
        trigger_actor_ids=request.trigger_actor_ids,
        requester_principal_id=request.requester_principal_id,
        policy_id=request.policy_id,
        allowed_capabilities=request.allowed_capabilities,
    )


def _pending_host_invocation_context(
    pending: AgentPendingHostDelivery,
) -> InvocationContext:
    """Restore only persisted host-delivery authority after any restart."""

    principal_kind: AgentPrincipalKind = (
        pending.principal_kind
        if pending.principal_kind is not None
        else "legacy_unknown"
    )
    return InvocationContext(
        actor_id=pending.actor_id,
        workspace_id=pending.workspace_id,
        transport="agent",
        request_id=pending.event_id,
        resource_ids=(pending.channel_id,),
        origin_resource_id=pending.channel_id,
        public_reference_id=pending.public_reference_id,
        agent_trigger="mention",
        principal_kind=principal_kind,
        executor_principal_id=pending.executor_principal_id,
        delegator_principal_id=pending.delegator_principal_id,
        trigger_actor_ids=pending.trigger_actor_ids,
        requester_principal_id=pending.requester_principal_id,
        policy_id=pending.policy_id,
    )


async def _record_agent_host_posts(
    runtime: SimajilordRuntime,
    *,
    channel_id: str,
    message_ids: tuple[str, ...],
    context: InvocationContext,
) -> bool:
    """Receipt one already-sent response using IDs only."""

    receipts = getattr(runtime, "action_receipts", None)
    if receipts is None:
        return True
    try:
        receipt = await receipts.record_posted_messages(
            channel_id=channel_id,
            message_ids=message_ids,
            context=context,
        )
        return receipt is not None and receipt.tracked
    except Exception:
        log.exception(
            "Could not receipt agent host response channel=%s messages=%s request=%s",
            channel_id,
            ",".join(message_ids),
            context.request_id,
        )
        return False


def _agent_response_uses_host_delivery(content: str) -> bool:
    """Return false for responses already delivered by a tool or intentionally silent."""

    return content.strip() not in {
        AGENT_FINAL_DELIVERED_CONTENT,
        AGENT_NO_ACTION_CONTENT,
    }


class AgentCog(commands.Cog):
    """Wake one shared agent conversation only for explicit mentions."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        self._started_at = datetime.now(UTC)
        self._active_progress: dict[str, AgentProgressMessage] = {}
        self._host_delivery_locks = KeyedAsyncLockPool()
        self._host_delivery_wakeup = asyncio.Event()
        self._host_delivery_task: asyncio.Task[None] | None = None
        self._interrupted_recovery_task: asyncio.Task[None] | None = None

    async def cog_load(self) -> None:
        if self._host_delivery_task is None or self._host_delivery_task.done():
            self._host_delivery_task = asyncio.create_task(
                self._host_delivery_loop(),
                name="simajilord-agent-host-delivery",
            )
        if (
            self._interrupted_recovery_task is None
            or self._interrupted_recovery_task.done()
        ):
            self._interrupted_recovery_task = asyncio.create_task(
                self._recover_interrupted_mentions(),
                name="simajilord-agent-interrupted-recovery",
            )

    async def cog_unload(self) -> None:
        tasks = tuple(
            task
            for task in (
                self._host_delivery_task,
                self._interrupted_recovery_task,
            )
            if task is not None
        )
        self._host_delivery_task = None
        self._interrupted_recovery_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        await self._handle_mention(
            message,
            event_id=f"discord:message:{message.id}",
            occurred_at=message.created_at,
            message_edited_at=message.edited_at,
        )

    @commands.Cog.listener()
    async def on_raw_message_edit(
        self,
        payload: discord.RawMessageUpdateEvent,
    ) -> None:
        raw_edited_at = payload.data.get("edited_timestamp")
        if "content" not in payload.data or not isinstance(raw_edited_at, str):
            return
        edited_at = discord.utils.parse_time(raw_edited_at)
        if edited_at is None:
            return
        await self._handle_mention(
            payload.message,
            event_id=(
                f"discord:message-edit:{payload.message_id}:"
                f"{edited_at.isoformat()}"
            ),
            occurred_at=edited_at,
            message_edited_at=edited_at,
        )

    async def _confirm_high_risk_action(
        self,
        source: discord.Message,
        proposal: AgentHighRiskConfirmation,
    ) -> bool:
        """Render and verify one concrete requester-only Discord confirmation."""

        try:
            requester_id = int(proposal.requester_principal_id)
            authorization_message_id = int(proposal.authorization_message_id)
        except ValueError:
            return False
        channel = source.channel
        if not isinstance(
            channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            return False
        arguments = proposal.arguments_json.replace("```", "`​``")
        external_egress = proposal.confirmation_kind == "external_egress"
        description = (
            (
                "A labelled source is about to be sent to an external provider. "
                "The content is intentionally hidden here; confirm only if the "
                "provider and field categories match your intent.\n\n"
                if external_egress
                else (
                    "A high-risk capability is ready to run. Confirm only if this "
                    "exact target and change match your intent.\n\n"
                )
            )
            + f"Capability: `{proposal.capability}`\n"
            + f"Binding: `{proposal.binding_sha256[:16]}`\n"
            + f"```json\n{arguments}\n```"
        )
        timeout = float(
            self.runtime.settings.agent_high_risk_confirmation_timeout_seconds
        )
        view = AgentHighRiskConfirmationView(
            requester_id=requester_id,
            binding_sha256=proposal.binding_sha256,
            timeout=timeout,
        )
        try:
            confirmation_message = await channel.send(
                embed=command_embed(
                    (
                        "Confirm external data transfer"
                        if external_egress
                        else "Confirm high-risk action"
                    ),
                    description=description,
                    tone=EmbedTone.WARNING,
                ),
                view=view,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.DiscordException:
            log.exception(
                "Could not publish high-risk confirmation request=%s capability=%s",
                proposal.authorization_message_id,
                proposal.capability,
            )
            return False
        view.message = confirmation_message
        confirmed = await view.wait_for_decision()
        revision_valid = False
        if confirmed:
            try:
                authorization_message = await channel.fetch_message(
                    authorization_message_id
                )
            except discord.DiscordException:
                authorization_message = None
            if authorization_message is not None:
                actual_revision = (
                    authorization_message.edited_at.isoformat()
                    if authorization_message.edited_at is not None
                    else None
                )
                revision_valid = (
                    authorization_message.author.id == requester_id
                    and actual_revision
                    == proposal.authorization_message_edited_at
                )
        accepted = confirmed and revision_valid
        try:
            await self.runtime.journal.append(
                kind=(
                    f"agent.{'egress' if external_egress else 'high_risk'}.confirmed"
                    if accepted
                    else f"agent.{'egress' if external_egress else 'high_risk'}.rejected"
                ),
                actor_id=proposal.requester_principal_id,
                workspace_id=(
                    str(source.guild.id) if source.guild is not None else None
                ),
                transport="agent",
                request_id=proposal.authorization_message_id,
                payload={
                    "capability": proposal.capability,
                    "binding_sha256": proposal.binding_sha256,
                    "authorization_message_id": proposal.authorization_message_id,
                    "authorization_message_edited_at": (
                        proposal.authorization_message_edited_at
                    ),
                    "confirmation_kind": proposal.confirmation_kind,
                    "confirmed": accepted,
                    "revision_valid": revision_valid,
                },
            )
        except Exception:
            log.exception(
                "Could not persist high-risk confirmation request=%s capability=%s",
                proposal.authorization_message_id,
                proposal.capability,
            )
            accepted = False
        with suppress(discord.DiscordException):
            await confirmation_message.edit(
                embed=command_embed(
                    (
                        "External transfer confirmed"
                        if accepted and external_egress
                        else "External transfer stopped"
                        if external_egress
                        else "High-risk action confirmed"
                        if accepted
                        else "High-risk action stopped"
                    ),
                    description=(
                        "The exact binding was confirmed and will now be rechecked "
                        "against live permissions."
                        if accepted
                        else (
                            "No external action was dispatched. The authorization "
                            "message may have changed, expired, or been rejected."
                        )
                    ),
                    tone=EmbedTone.SUCCESS if accepted else EmbedTone.WARNING,
                ),
                view=view,
            )
        return accepted

    async def _handle_mention(
        self,
        message: discord.Message,
        *,
        event_id: str,
        occurred_at: datetime,
        message_edited_at: datetime | None = None,
        allow_routing: bool = True,
    ) -> None:
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
        requester_principal = await _resolve_requester_principal(
            message.guild,
            message.author,
        )
        if requester_principal is None:
            log.info(
                "Mention agent turn rejected because requester could not be resolved "
                "guild=%s channel=%s actor=%s",
                message.guild.id,
                message.channel.id,
                message.author.id,
            )
            return
        resource_ids = readable_for_requester(message.guild, requester_principal)
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
        requester_permissions = requester_principal.member.guild_permissions
        grants = _agent_grants(
            self.runtime,
            actor_id=actor_id,
            memory_curator=(
                permission_enabled(requester_permissions, "administrator")
                or permission_enabled(requester_permissions, "manage_guild")
            ),
        )
        approvals = frozenset(AGENT_REQUESTED_WRITE_CAPABILITIES)
        public_reference_id = (
            await self.runtime.agent_store.public_reference_id_for_event(event_id)
            or new_agent_public_reference_id()
        )
        existing_request = await self.runtime.agent_store.request_by_public_reference_id(
            public_reference_id
        )
        if existing_request is not None and existing_request.event_id != event_id:
            raise RuntimeError("agent event reference resolved to another request")
        task_id = (
            existing_request.task_id
            if existing_request is not None
            else await self.runtime.agent_store.task_id_for_event(event_id)
            or new_agent_task_id()
        )
        conversation_id = discord_conversation_id(
            guild_id=message.guild.id if message.guild else None,
            channel_id=message.channel.id,
            actor_id=actor_id,
            grants=grants,
            compatibility_epoch=(
                self.runtime.settings.agent_conversation_compatibility_epoch
            ),
        )
        request = AgentRequest(
            conversation_id=(
                existing_request.conversation_id
                if existing_request is not None
                else task_scoped_conversation_id(
                    conversation_id,
                    task_id,
                )
            ),
            event_id=event_id,
            trigger=AgentTrigger.MENTION,
            actor_id=actor_id,
            actor_name=message.author.display_name,
            workspace_id=str(message.guild.id) if message.guild else None,
            channel_id=str(message.channel.id),
            message_id=str(message.id),
            occurred_at=occurred_at,
            resource_ids=resource_ids,
            public_reference_id=public_reference_id,
            task_id=task_id,
            principal_kind="requester",
            read_scope_mode="requester_live",
            information_flow_mode=(
                self.runtime.settings.agent_information_flow_mode
            ),
            file_workspace_mode=self.runtime.settings.agent_file_workspace_mode,
            high_risk_authorization_mode=(
                self.runtime.settings.agent_high_risk_authorization_mode
            ),
            executor_principal_id=str(bot_user.id),
            delegator_principal_id=actor_id,
            trigger_actor_ids=(actor_id,),
            requester_principal_id=actor_id,
            policy_id="discord-mention-v2",
            message_edited_at=message_edited_at,
            grants=grants,
            approvals=approvals,
        )
        if allow_routing:
            try:
                route = await agent.route_candidate(request)
            except AgentBusyError as exc:
                await message.reply(
                    _agent_error_text(exc),
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                return
            except Exception as exc:
                log.exception(
                    "AI task candidate routing failed message=%s reference=%s",
                    message.id,
                    request.public_reference_id,
                )
                await message.reply(
                    _agent_error_text(exc, reference_id=request.public_reference_id),
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                return
            if route is not None and route.decision is not AgentTaskRouteDecision.SEPARATE:
                return
        progress = _AgentProgressMessage(
            message,
            delivery_key=request.event_id,
        )
        self._active_progress[request.event_id] = progress
        try:
            response = await agent.respond(
                request,
                on_progress=progress.update,
                on_high_risk_confirmation=lambda proposal: (
                    self._confirm_high_risk_action(message, proposal)
                ),
            )
        except asyncio.CancelledError:
            snapshot = (
                await self.runtime.agent_store.task_snapshot_by_public_reference_id(
                    request.public_reference_id
                )
            )
            if snapshot is not None and snapshot.state == "cancelled":
                await progress.cancelled()
                return
            raise
        except Exception as exc:
            log.exception(
                "Mention agent turn failed message=%s reference=%s",
                message.id,
                request.public_reference_id,
            )
            persisted_reference_id = (
                await self.runtime.agent_store.public_reference_id_for_event(
                    request.event_id
                )
            )
            await progress.fail(
                _agent_error_text(
                    exc,
                    reference_id=persisted_reference_id,
                )
            )
        else:
            chunks = await progress.prepare(response.content)
            pending = await self.runtime.agent_store.pending_host_delivery(
                request.event_id
            )
            if pending is None:
                if _agent_response_uses_host_delivery(response.content):
                    log.error(
                        "Completed agent turn has no pending host record request=%s",
                        request.event_id,
                    )
                else:
                    log.debug(
                        "Agent turn requires no host delivery request=%s",
                        request.event_id,
                    )
            else:
                try:
                    await self._deliver_host_response(
                        pending,
                        source=message,
                        expected_chunks=chunks,
                    )
                except Exception:
                    # The completed model response remains in SQLite and the
                    # recovery loop will retry without re-running any tools.
                    log.exception(
                        "Agent response delivery deferred request=%s reference=%s "
                        "channel=%s",
                        request.event_id,
                        request.public_reference_id,
                        request.channel_id,
                    )
                    self._host_delivery_wakeup.set()
        finally:
            if self._active_progress.get(request.event_id) is progress:
                self._active_progress.pop(request.event_id, None)

    async def _recover_interrupted_mentions(self) -> None:
        """Re-run recent explicit mentions interrupted by the prior process."""

        await self.bot.wait_until_ready()
        interrupted = await self.runtime.agent_store.interrupted_mentions(
            started_after=self._started_at - _AGENT_INTERRUPTED_MENTION_MAX_AGE,
            started_before=self._started_at,
        )
        for request in interrupted:
            try:
                replay_barrier = await _agent_request_replay_barrier_reason(
                    self.runtime,
                    request.event_id,
                    task_id=request.task_id,
                )
                if replay_barrier is not None:
                    await self.runtime.agent_store.fail_interrupted_mention(
                        request.event_id,
                        error_type="RecoveryBlockedByExternalEffect",
                    )
                    await self.runtime.journal.append(
                        kind="agent.turn.recovery_blocked",
                        actor_id=None,
                        workspace_id=None,
                        transport="agent",
                        request_id=request.event_id,
                        payload={
                            "public_reference_id": request.public_reference_id,
                            "reason": replay_barrier,
                        },
                    )
                    log.error(
                        "Interrupted mention was not replayed after an external "
                        "write request=%s reason=%s",
                        request.event_id,
                        replay_barrier,
                    )
                    continue
                channel = await self._agent_host_channel(request.channel_id)
                source = await self._agent_source_message(
                    channel,
                    request.source_message_id,
                )
                if source is None:
                    await self.runtime.agent_store.fail_interrupted_mention(
                        request.event_id,
                        error_type="RecoverySourceUnavailable",
                    )
                    log.warning(
                        "Interrupted mention source is unavailable request=%s "
                        "channel=%s message=%s",
                        request.event_id,
                        request.channel_id,
                        request.source_message_id,
                    )
                    continue
                log.info(
                    "Recovering interrupted mention request=%s channel=%s message=%s",
                    request.event_id,
                    request.channel_id,
                    request.source_message_id,
                )
                await self._handle_mention(
                    source,
                    event_id=request.event_id,
                    occurred_at=request.occurred_at,
                    message_edited_at=source.edited_at,
                    allow_routing=False,
                )
                skipped = await self.runtime.agent_store.fail_interrupted_mention(
                    request.event_id,
                    error_type="RecoverySkipped",
                )
                if skipped:
                    log.warning(
                        "Interrupted mention no longer qualified for processing "
                        "request=%s",
                        request.event_id,
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Leave the row in progress so a later restart can retry a
                # transient Discord or store failure.
                log.exception(
                    "Interrupted mention recovery failed request=%s channel=%s",
                    request.event_id,
                    request.channel_id,
                )

        await self._recover_unrouted_task_candidates()

    async def _recover_unrouted_task_candidates(self) -> None:
        """Default crash-interrupted semantic decisions to isolated tasks."""

        candidates = await self.runtime.agent_store.unrouted_task_candidates(
            created_before=self._started_at,
        )
        for candidate in candidates:
            try:
                defaulted = await self.runtime.agent_store.default_task_candidate_to_separate(
                    candidate.event_id,
                    reason="startup_default_separate",
                )
                if not defaulted:
                    continue
                replay_barrier = await _agent_request_replay_barrier_reason(
                    self.runtime,
                    candidate.event_id,
                )
                if replay_barrier is not None:
                    await self.runtime.agent_store.fail_unrouted_task_candidate(
                        candidate.event_id,
                        error_type="RecoveryBlockedByExternalEffect",
                    )
                    await self.runtime.journal.append(
                        kind="agent.task.recovery_blocked",
                        actor_id=None,
                        workspace_id=None,
                        transport="agent",
                        request_id=candidate.event_id,
                        payload={
                            "public_reference_id": candidate.public_reference_id,
                            "task_id": candidate.task_id,
                            "reason": replay_barrier,
                        },
                    )
                    continue
                channel = await self._agent_host_channel(candidate.channel_id)
                source = await self._agent_source_message(
                    channel,
                    candidate.source_message_id,
                )
                if source is None:
                    await self.runtime.agent_store.fail_unrouted_task_candidate(
                        candidate.event_id,
                        error_type="RecoverySourceUnavailable",
                    )
                    log.warning(
                        "Task candidate source is unavailable event=%s channel=%s "
                        "message=%s",
                        candidate.event_id,
                        candidate.channel_id,
                        candidate.source_message_id,
                    )
                    continue
                log.info(
                    "Recovering task candidate as separate event=%s task=%s",
                    candidate.event_id,
                    candidate.task_id,
                )
                await self._handle_mention(
                    source,
                    event_id=candidate.event_id,
                    occurred_at=candidate.occurred_at,
                    message_edited_at=source.edited_at,
                    allow_routing=False,
                )
                recovered = await self.runtime.agent_store.request_by_public_reference_id(
                    candidate.public_reference_id
                )
                if recovered is None:
                    await self.runtime.agent_store.fail_unrouted_task_candidate(
                        candidate.event_id,
                        error_type="RecoverySkipped",
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "Task candidate recovery failed event=%s task=%s",
                    candidate.event_id,
                    candidate.task_id,
                )

    async def _host_delivery_loop(self) -> None:
        """Reconcile completed mention responses without rerunning the model."""

        await self.bot.wait_until_ready()
        retry_delay = 1.0
        while True:
            self._host_delivery_wakeup.clear()
            had_failure = False
            try:
                pending = await self.runtime.agent_store.pending_host_deliveries(
                    limit=100
                )
                recovery_candidates: dict[
                    str,
                    tuple[discord.Message, ...],
                ] = {}
                for delivery in pending:
                    try:
                        await self._deliver_host_response(
                            delivery,
                            recovery_candidates=recovery_candidates,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        had_failure = True
                        log.exception(
                            "Agent host delivery recovery failed request=%s "
                            "reference=%s channel=%s",
                            delivery.event_id,
                            delivery.public_reference_id,
                            delivery.channel_id,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                had_failure = True
                log.exception("Agent host delivery recovery query failed")

            retry_delay = (
                min(30.0, retry_delay * 2) if had_failure else 1.0
            )
            with suppress(TimeoutError):
                await asyncio.wait_for(
                    self._host_delivery_wakeup.wait(),
                    timeout=retry_delay if had_failure else 30.0,
                )

    async def _deliver_host_response(
        self,
        pending: AgentPendingHostDelivery,
        *,
        source: discord.Message | None = None,
        expected_chunks: tuple[str, ...] | None = None,
        recovery_candidates: dict[str, tuple[discord.Message, ...]] | None = None,
    ) -> None:
        async with self._host_delivery_locks.hold(pending.event_id):
            chunks = agent_message_groups(pending.response_content)
            if pending.response_content.strip() in {
                AGENT_FINAL_DELIVERED_CONTENT,
                AGENT_NO_ACTION_CONTENT,
            }:
                chunks = ()
            if expected_chunks is not None and chunks != expected_chunks:
                raise RuntimeError("prepared Discord chunks changed before delivery")
            if not chunks:
                await self.runtime.agent_store.complete_host_delivery(
                    pending.event_id,
                    allow_empty=True,
                )
                return

            records = await self.runtime.agent_store.plan_host_delivery(
                event_id=pending.event_id,
                purpose="response",
                channel_id=pending.channel_id,
                contents=chunks,
            )
            channel = await self._agent_host_channel(pending.channel_id)
            source = source or await self._agent_source_message(
                channel,
                pending.source_message_id,
            )
            records = await self._reconcile_host_messages(
                channel,
                pending,
                records,
                recovery_candidates=recovery_candidates,
            )
            context = _pending_host_invocation_context(pending)
            for record, content in zip(records, chunks, strict=True):
                message_id = record.message_id
                if message_id is None:
                    posted = await self._send_host_chunk(
                        channel,
                        source,
                        pending,
                        record,
                        content,
                    )
                    message_id = str(posted.id)
                    record = await self.runtime.agent_store.record_host_delivery_message(
                        event_id=pending.event_id,
                        purpose=record.purpose,
                        chunk_index=record.chunk_index,
                        message_id=message_id,
                    )
            records = await self.runtime.agent_store.host_delivery_records(
                event_id=pending.event_id,
                purpose="response",
            )
            unreceipted = tuple(
                record for record in records if record.receipted_at is None
            )
            if unreceipted:
                message_ids = tuple(
                    record.message_id
                    for record in records
                    if record.message_id is not None
                )
                if len(message_ids) != len(records):
                    raise RuntimeError("agent host delivery message evidence is incomplete")
                receipted = await _record_agent_host_posts(
                    self.runtime,
                    channel_id=pending.channel_id,
                    message_ids=message_ids,
                    context=context,
                )
                if not receipted:
                    raise RuntimeError("agent host receipt persistence failed")
                for record in unreceipted:
                    await self.runtime.agent_store.mark_host_delivery_receipted(
                        event_id=pending.event_id,
                        purpose=record.purpose,
                        chunk_index=record.chunk_index,
                    )

            if not await self.runtime.agent_store.complete_host_delivery(
                pending.event_id
            ):
                raise RuntimeError("agent host delivery did not reach terminal state")

    async def _agent_host_channel(
        self,
        channel_id: str,
    ) -> discord.abc.Messageable:
        try:
            numeric_channel_id = int(channel_id)
        except ValueError as exc:
            raise RuntimeError("agent host channel ID is invalid") from exc
        channel = self.bot.get_channel(numeric_channel_id)
        if channel is None:
            channel = await self.bot.fetch_channel(numeric_channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            raise RuntimeError("agent host channel cannot contain messages")
        return channel

    async def _agent_source_message(
        self,
        channel: discord.abc.Messageable,
        source_message_id: str | None,
    ) -> discord.Message | None:
        if source_message_id is None:
            return None
        try:
            return await channel.fetch_message(int(source_message_id))
        except (ValueError, discord.NotFound, discord.Forbidden):
            return None

    async def _reconcile_host_messages(
        self,
        channel: discord.abc.Messageable,
        pending: AgentPendingHostDelivery,
        records: tuple[AgentHostDeliveryRecord, ...],
        *,
        recovery_candidates: dict[str, tuple[discord.Message, ...]] | None = None,
    ) -> tuple[AgentHostDeliveryRecord, ...]:
        missing = [record for record in records if record.message_id is None]
        bot_user = self.bot.user
        if not missing or bot_user is None:
            return records
        candidates = (
            list(recovery_candidates[pending.channel_id])
            if recovery_candidates is not None
            and pending.channel_id in recovery_candidates
            else None
        )
        if candidates is None:
            candidates = []
            after = pending.completed_at - timedelta(minutes=1)
            try:
                async for candidate in channel.history(
                    limit=1_000,
                    after=after,
                    oldest_first=True,
                ):
                    if candidate.author.id == bot_user.id:
                        candidates.append(candidate)
            except (discord.Forbidden, discord.NotFound):
                candidates = []
            if recovery_candidates is not None:
                recovery_candidates[pending.channel_id] = tuple(candidates)

        # Identical chunks are valid. Never let hash fallback reuse a message
        # that is already durable evidence for another chunk.
        used_ids = {
            int(record.message_id)
            for record in records
            if record.message_id is not None
        }
        reconciled: list[AgentHostDeliveryRecord] = []
        for record in records:
            if record.message_id is not None:
                reconciled.append(record)
                continue
            nonce = agent_delivery_nonce(
                pending.event_id,
                record.chunk_index,
                purpose=record.purpose,
            )
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.id not in used_ids
                    and str(candidate.nonce or "") == nonce
                ),
                None,
            )
            if match is None:
                reconciled.append(record)
                continue
            used_ids.add(match.id)
            reconciled.append(
                await self.runtime.agent_store.record_host_delivery_message(
                    event_id=pending.event_id,
                    purpose=record.purpose,
                    chunk_index=record.chunk_index,
                    message_id=str(match.id),
                )
            )
        return tuple(reconciled)

    async def _send_host_chunk(
        self,
        channel: discord.abc.Messageable,
        source: discord.Message | None,
        pending: AgentPendingHostDelivery,
        record: AgentHostDeliveryRecord,
        content: str,
    ) -> discord.Message:
        nonce = agent_delivery_nonce(
            pending.event_id,
            record.chunk_index,
            purpose=record.purpose,
        )
        if record.chunk_index == 0 and source is not None:
            return await source.reply(
                content,
                nonce=nonce,
                mention_author=False,
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
        return await channel.send(
            content,
            nonce=nonce,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )


async def _agent_request_replay_barrier_reason(
    runtime: SimajilordRuntime,
    request_id: str,
    *,
    task_id: str | None = None,
) -> str | None:
    """Conservatively refuse whole-turn replay after any durable write attempt."""

    action_receipts = getattr(runtime, "action_receipts", None)
    if (
        action_receipts is not None
        and await action_receipts.request_has_replay_barrier(request_id)
    ):
        return "external_effect_ledger"
    journal = getattr(runtime, "journal", None)
    if journal is None:
        return None
    trace = await journal.agent_trace(
        request_id=None if task_id is not None else request_id,
        task_id=task_id,
        limit=1_000,
    )
    for record in trace:
        if record.kind not in {"agent.tool.started", "agent.app_tool.started"}:
            continue
        if record.payload.get("write") is True:
            return "legacy_write_trace"
    return None


class ObservationCog(commands.Cog):
    """Publish Discord gateway events to the durable autonomous-agent queue."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        bot_user = self.bot.user
        mentions_bot = bot_user in message.mentions if bot_user else False
        payload: dict[str, object] = {
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "author_name": message.author.display_name,
            "author_is_bot": message.author.bot,
            "content_length": len(message.content),
            "mentions_bot": mentions_bot,
            "attachments": [
                {
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "size": attachment.size,
                }
                for attachment in message.attachments
            ],
        }
        await self.runtime.journal.append(
            kind="discord.message.created",
            actor_id=str(message.author.id),
            workspace_id=str(message.guild.id),
            transport="discord",
            request_id=str(message.id),
            payload=payload,
        )
        if (
            message.author.bot
            or message.webhook_id is not None
            or mentions_bot
            or not self._enabled_for(message.guild.id)
        ):
            return
        get_context = getattr(self.bot, "get_context", None)
        if callable(get_context):
            command_context = await get_context(message)
            if command_context.valid:
                return
        await self._enqueue(
            kind=AutonomyEventKind.MESSAGE_CREATE,
            deduplication_key=f"message-create:{message.id}",
            workspace_id=str(message.guild.id),
            channel_id=str(message.channel.id),
            actor_id=str(message.author.id),
            message_id=str(message.id),
            occurred_at=message.created_at,
            payload=payload,
        )

    @commands.Cog.listener()
    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if payload.guild_id is None or not self._enabled_for(payload.guild_id):
            return
        raw_edited_at = payload.data.get("edited_timestamp")
        content_changed = "content" in payload.data
        attachments_changed = "attachments" in payload.data
        if (
            not isinstance(raw_edited_at, str)
            or not (content_changed or attachments_changed)
        ):
            return
        occurred_at = discord.utils.parse_time(raw_edited_at)
        if occurred_at is None:
            return
        cached = payload.cached_message
        current = payload.message
        author_id: str | None = None
        author_is_bot = False
        author = getattr(current, "author", None)
        if author is None and cached is not None:
            author = cached.author
        if author is not None:
            author_id = str(author.id)
            author_is_bot = author.bot
        webhook_message = (
            getattr(current, "webhook_id", None) is not None
            or (cached is not None and cached.webhook_id is not None)
            or bool(payload.data.get("webhook_id"))
        )
        bot_user = self.bot.user
        current_mentions = getattr(current, "mentions", ())
        mentions_bot_after = (
            bot_user in current_mentions if bot_user is not None else False
        )
        mentions_bot_before = (
            cached is not None
            and bot_user is not None
            and bot_user in cached.mentions
        )
        mentions_bot = mentions_bot_before or mentions_bot_after
        event_payload: dict[str, object] = {
            "message_id": str(payload.message_id),
            "channel_id": str(payload.channel_id),
            "source_actor_id": author_id,
            "author_is_bot": author_is_bot,
            "mentions_bot": mentions_bot,
            "mentions_bot_before": mentions_bot_before,
            "mentions_bot_after": mentions_bot_after,
            "edited_at_iso": occurred_at.isoformat(),
            "content_changed": content_changed,
            "attachments_changed": attachments_changed,
        }
        await self.runtime.journal.append(
            kind=AutonomyEventKind.MESSAGE_EDIT.value,
            actor_id=author_id,
            workspace_id=str(payload.guild_id),
            transport="discord",
            request_id=f"edit:{payload.message_id}:{occurred_at.isoformat()}",
            payload=event_payload,
        )
        if author_id is None or author_is_bot or webhook_message or mentions_bot:
            return
        await self._enqueue(
            kind=AutonomyEventKind.MESSAGE_EDIT,
            deduplication_key=(
                f"message-edit:{payload.message_id}:{occurred_at.isoformat()}"
            ),
            workspace_id=str(payload.guild_id),
            channel_id=str(payload.channel_id),
            actor_id=author_id,
            message_id=str(payload.message_id),
            occurred_at=occurred_at,
            payload=event_payload,
        )

    @commands.Cog.listener()
    async def on_raw_reaction_add(
        self,
        payload: discord.RawReactionActionEvent,
    ) -> None:
        if payload.guild_id is None or not self._enabled_for(payload.guild_id):
            return
        bot_user = self.bot.user
        if (
            (bot_user is not None and payload.user_id == bot_user.id)
            or (payload.member is not None and payload.member.bot)
        ):
            return
        occurred_at = datetime.now(UTC)
        emoji = str(payload.emoji)
        event_payload: dict[str, object] = {
            "message_id": str(payload.message_id),
            "channel_id": str(payload.channel_id),
            "source_actor_id": str(payload.user_id),
            "emoji": emoji,
        }
        await self.runtime.journal.append(
            kind=AutonomyEventKind.REACTION_ADD.value,
            actor_id=str(payload.user_id),
            workspace_id=str(payload.guild_id),
            transport="discord",
            request_id=(
                f"reaction:{payload.message_id}:{payload.user_id}:"
                f"{occurred_at.isoformat()}"
            ),
            payload=event_payload,
        )
        await self._enqueue(
            kind=AutonomyEventKind.REACTION_ADD,
            deduplication_key=(
                f"reaction-add:{payload.message_id}:{payload.user_id}:"
                f"{emoji}:{int(occurred_at.timestamp())}"
            ),
            workspace_id=str(payload.guild_id),
            channel_id=str(payload.channel_id),
            actor_id=str(payload.user_id),
            message_id=str(payload.message_id),
            occurred_at=occurred_at,
            payload=event_payload,
        )

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread) -> None:
        if not self._enabled_for(thread.guild.id):
            return
        bot_user = self.bot.user
        if bot_user is not None and thread.owner_id == bot_user.id:
            return
        occurred_at = thread.created_at or datetime.now(UTC)
        event_payload: dict[str, object] = {
            "channel_id": str(thread.id),
            "parent_channel_id": str(thread.parent_id) if thread.parent_id else None,
            "source_actor_id": str(thread.owner_id) if thread.owner_id else None,
            "thread_name": thread.name,
        }
        await self.runtime.journal.append(
            kind=AutonomyEventKind.THREAD_CREATE.value,
            actor_id=str(thread.owner_id) if thread.owner_id else None,
            workspace_id=str(thread.guild.id),
            transport="discord",
            request_id=f"thread:{thread.id}",
            payload=event_payload,
        )
        await self._enqueue(
            kind=AutonomyEventKind.THREAD_CREATE,
            deduplication_key=f"thread-create:{thread.id}",
            workspace_id=str(thread.guild.id),
            channel_id=str(thread.id),
            actor_id=str(thread.owner_id) if thread.owner_id else None,
            message_id=None,
            occurred_at=occurred_at,
            payload=event_payload,
        )

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        if (
            member.bot
            or before.channel == after.channel
            or not self._enabled_for(member.guild.id)
        ):
            return
        channel = after.channel or before.channel
        if channel is None:
            return
        occurred_at = datetime.now(UTC)
        event_payload: dict[str, object] = {
            "channel_id": str(channel.id),
            "source_actor_id": str(member.id),
            "before_channel_id": (
                str(before.channel.id) if before.channel is not None else None
            ),
            "after_channel_id": (
                str(after.channel.id) if after.channel is not None else None
            ),
        }
        await self.runtime.journal.append(
            kind=AutonomyEventKind.VOICE_STATE_UPDATE.value,
            actor_id=str(member.id),
            workspace_id=str(member.guild.id),
            transport="discord",
            request_id=f"voice:{member.id}:{occurred_at.isoformat()}",
            payload=event_payload,
        )
        await self._enqueue(
            kind=AutonomyEventKind.VOICE_STATE_UPDATE,
            deduplication_key=f"voice-state:{member.id}:{occurred_at.isoformat()}",
            workspace_id=str(member.guild.id),
            channel_id=str(channel.id),
            actor_id=str(member.id),
            message_id=None,
            occurred_at=occurred_at,
            payload=event_payload,
        )

    def _enabled_for(self, guild_id: int) -> bool:
        settings = self.runtime.settings
        return (
            self.runtime.agent is not None
            and settings.agent_autonomy_enabled
            and settings.agent_autonomy_mode is not AgentAutonomyMode.OBSERVE
            and str(guild_id) in settings.agent_autonomy_guild_ids
        )

    async def _enqueue(
        self,
        *,
        kind: AutonomyEventKind,
        deduplication_key: str,
        workspace_id: str,
        channel_id: str,
        occurred_at: datetime,
        actor_id: str | None,
        message_id: str | None,
        payload: dict[str, object],
    ) -> None:
        result = await self.runtime.autonomy_events.enqueue(
            kind=kind,
            deduplication_key=deduplication_key,
            workspace_id=workspace_id,
            channel_id=channel_id,
            occurred_at=occurred_at,
            actor_id=actor_id,
            message_id=message_id,
            payload=payload,
        )
        if result in {
            AutonomyEnqueueResult.QUEUE_FULL,
            AutonomyEnqueueResult.CHANNEL_QUEUE_FULL,
            AutonomyEnqueueResult.ACTOR_QUEUE_FULL,
        }:
            await self.runtime.journal.append(
                kind="agent.autonomy.event_rejected",
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport="agent",
                request_id=deduplication_key,
                payload={
                    "event_kind": kind.value,
                    "channel_id": channel_id,
                    "reason": result.value,
                },
            )


class _AutonomyTerminalDrop(RuntimeError):
    """A queued pointer cannot become actionable without a configuration change."""


class _AutonomyReceiptDeferred(RuntimeError):
    """A delivered message still needs its durable Action Receipt."""


class AgentAutonomyCog(commands.Cog):
    """Consume durable same-channel event batches without polling the journal."""

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
            and settings.agent_autonomy_mode is not AgentAutonomyMode.OBSERVE
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
        queue = self.runtime.autonomy_events
        await self.bot.wait_until_ready()
        await self._journal(
            kind="agent.autonomy.started",
            transport="agent",
            payload={
                "mode": settings.agent_autonomy_mode.value,
                "batch_seconds": settings.agent_autonomy_batch_seconds,
                "max_runs": settings.agent_autonomy_max_runs,
                "candidate_limit": settings.agent_autonomy_candidate_limit,
            },
        )
        completed_runs = 0
        try:
            while (
                settings.agent_autonomy_max_runs == 0
                or completed_runs < settings.agent_autonomy_max_runs
            ):
                try:
                    batch = await self._next_batch()
                    if batch is None:
                        continue
                    model_called = await self._consume_batch(batch)
                    if model_called is None:
                        continue
                    if model_called:
                        completed_runs += 1
                    await self._journal(
                        kind="agent.autonomy.checked",
                        workspace_id=batch.workspace_id,
                        transport="agent",
                        request_id=batch.batch_id,
                        payload={
                            "run": completed_runs,
                            "candidate_count": len(batch.events),
                            "model_called": model_called,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One corrupt event or transient SQLite failure must not stop
                    # the only long-lived consumer task.
                    log.exception("Autonomous agent queue iteration failed")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        finally:
            try:
                pending_events = await queue.pending_count()
            except Exception:
                pending_events = -1
                log.exception("Could not count pending autonomy events at shutdown")
            await self._journal(
                kind="agent.autonomy.stopped",
                transport="agent",
                payload={
                    "completed_runs": completed_runs,
                    "pending_events": pending_events,
                },
            )

    async def _next_batch(self) -> AutonomyEventBatch | None:
        settings = self.runtime.settings
        queue = self.runtime.autonomy_events
        batch = await queue.next_batch(
            debounce_seconds=settings.agent_autonomy_batch_seconds,
            candidate_limit=settings.agent_autonomy_candidate_limit,
            lease_seconds=_AUTONOMY_LEASE_SECONDS,
        )
        if batch is None:
            queue.clear_wake()
            # Close the clear/enqueue race before sleeping.
            batch = await queue.next_batch(
                debounce_seconds=settings.agent_autonomy_batch_seconds,
                candidate_limit=settings.agent_autonomy_candidate_limit,
                lease_seconds=_AUTONOMY_LEASE_SECONDS,
            )
        if batch is not None:
            return batch
        delay = await queue.seconds_until_ready(
            debounce_seconds=settings.agent_autonomy_batch_seconds,
            candidate_limit=settings.agent_autonomy_candidate_limit,
        )
        await queue.wait(delay)
        return None

    async def _consume_batch(
        self,
        batch: AutonomyEventBatch,
    ) -> bool | None:
        heartbeat = asyncio.create_task(
            self._heartbeat(batch),
            name=f"simajilord-autonomy-lease-{batch.batch_id}",
        )
        try:
            try:
                model_called = await self._while_owned(
                    self._inspect(batch),
                    heartbeat,
                )
                await self._while_owned(
                    self._ack_with_retry(batch),
                    heartbeat,
                    operation_wins_tie=True,
                )
                return model_called
            except AgentRateLimitError as exc:
                await self._stop_heartbeat(heartbeat)
                retry_after = exc.retry_after_seconds or 60
                await self._safe_defer(
                    batch,
                    reason="rate_limited",
                    retry_after_seconds=retry_after,
                )
            except AgentBusyError:
                await self._stop_heartbeat(heartbeat)
                retry_after = min(
                    300,
                    max(
                        5,
                        self.runtime.settings.agent_autonomy_batch_seconds,
                    )
                    * (2 ** min(batch.deferral_count, 4)),
                )
                await self._safe_defer(
                    batch,
                    reason="agent_busy",
                    retry_after_seconds=retry_after,
                )
            except _AutonomyReceiptDeferred:
                await self._stop_heartbeat(heartbeat)
                await self._safe_defer(
                    batch,
                    reason="receipt_unavailable",
                    retry_after_seconds=60,
                )
            except (
                _AutonomyTerminalDrop,
                AutonomyDeliveryConflictError,
            ) as exc:
                await self._stop_heartbeat(heartbeat)
                await self._safe_dead_letter(batch, reason=str(exc))
            except AutonomyLeaseLostError:
                log.info(
                    "Autonomous agent batch ownership moved batch=%s",
                    batch.batch_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._stop_heartbeat(heartbeat)
                if batch.attempt_count >= 7:
                    await self._safe_dead_letter(
                        batch,
                        reason=f"retry_exhausted:{type(exc).__name__}",
                    )
                    log.exception(
                        "Autonomous agent batch dead-lettered batch=%s",
                        batch.batch_id,
                    )
                else:
                    retry_after = min(
                        3_600,
                        60 * (2 ** min(batch.attempt_count, 6)),
                    )
                    await self._safe_retry_failure(
                        batch,
                        reason=type(exc).__name__,
                        retry_after_seconds=retry_after,
                    )
                    log.exception(
                        "Autonomous agent batch failed batch=%s",
                        batch.batch_id,
                    )
            return None
        finally:
            await self._stop_heartbeat(heartbeat)

    async def _while_owned(
        self,
        operation: Awaitable[_AutonomyResultT],
        heartbeat: asyncio.Task[None],
        *,
        operation_wins_tie: bool = False,
    ) -> _AutonomyResultT:
        operation_task = asyncio.ensure_future(operation)
        done, _ = await asyncio.wait(
            (operation_task, heartbeat),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done and (
            heartbeat not in done or operation_wins_tie
        ):
            return await operation_task
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        if heartbeat.cancelled():
            raise AutonomyLeaseLostError(
                "Autonomy lease heartbeat ended before the operation"
            )
        await heartbeat
        raise AutonomyLeaseLostError(
            "Autonomy lease heartbeat stopped unexpectedly"
        )

    async def _heartbeat(self, batch: AutonomyEventBatch) -> None:
        lease_until = batch.lease_until
        while True:
            await asyncio.sleep(_AUTONOMY_LEASE_HEARTBEAT_SECONDS)
            while True:
                try:
                    lease_until = await self.runtime.autonomy_events.renew_lease(
                        batch,
                        lease_seconds=_AUTONOMY_LEASE_SECONDS,
                    )
                    break
                except AutonomyLeaseLostError:
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    remaining = (lease_until - datetime.now(UTC)).total_seconds()
                    if remaining <= 0:
                        raise AutonomyLeaseLostError(
                            f"Autonomy lease renewal failed until expiry: "
                            f"{batch.batch_id}"
                        ) from exc
                    log.exception(
                        "Autonomy lease heartbeat failed batch=%s remaining=%s",
                        batch.batch_id,
                        round(remaining),
                    )
                    await asyncio.sleep(min(5.0, remaining))

    async def _ack_with_retry(self, batch: AutonomyEventBatch) -> None:
        attempt = 0
        while True:
            try:
                await self.runtime.autonomy_events.mark_processed(batch)
                return
            except AutonomyLeaseLostError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                delay = min(30, 2 ** min(attempt, 5))
                attempt += 1
                log.exception(
                    "Autonomous agent batch ACK failed; retrying batch=%s "
                    "delay=%s",
                    batch.batch_id,
                    delay,
                )
                await asyncio.sleep(delay)

    @staticmethod
    async def _stop_heartbeat(heartbeat: asyncio.Task[None]) -> None:
        if not heartbeat.done():
            heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)

    async def _journal(
        self,
        *,
        kind: str,
        payload: dict[str, object],
        actor_id: str | None = None,
        workspace_id: str | None = None,
        transport: str | None = None,
        request_id: str | None = None,
    ) -> None:
        try:
            await self.runtime.journal.append(
                kind=kind,
                payload=payload,
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport=transport,
                request_id=request_id,
            )
        except Exception:
            log.exception(
                "Could not append autonomous-agent journal event kind=%s",
                kind,
            )

    async def _safe_dead_letter(
        self,
        batch: AutonomyEventBatch,
        *,
        reason: str,
    ) -> bool:
        try:
            await self.runtime.autonomy_events.dead_letter(batch, reason=reason)
        except AutonomyLeaseLostError:
            log.info(
                "Autonomy dead-letter skipped after ownership moved batch=%s",
                batch.batch_id,
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Autonomy dead-letter persistence failed batch=%s",
                batch.batch_id,
            )
            return False
        await self._journal(
            kind="agent.autonomy.dead_lettered",
            workspace_id=batch.workspace_id,
            transport="agent",
            request_id=batch.batch_id,
            payload={
                "channel_id": batch.channel_id,
                "event_count": len(batch.events),
                "attempt_count": batch.attempt_count + 1,
                "reason": reason,
            },
        )
        return True

    async def _safe_defer(
        self,
        batch: AutonomyEventBatch,
        *,
        reason: str,
        retry_after_seconds: int,
    ) -> bool:
        try:
            await self.runtime.autonomy_events.defer(
                batch,
                retry_after_seconds=retry_after_seconds,
            )
        except AutonomyLeaseLostError:
            log.info(
                "Autonomy deferral skipped after ownership moved batch=%s",
                batch.batch_id,
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Autonomy deferral persistence failed batch=%s",
                batch.batch_id,
            )
            return False
        await self._record_retry(
            batch,
            reason,
            retry_after_seconds,
            failure=False,
        )
        return True

    async def _safe_retry_failure(
        self,
        batch: AutonomyEventBatch,
        *,
        reason: str,
        retry_after_seconds: int,
    ) -> bool:
        try:
            await self.runtime.autonomy_events.reschedule(
                batch,
                retry_after_seconds=retry_after_seconds,
            )
        except AutonomyLeaseLostError:
            log.info(
                "Autonomy retry skipped after ownership moved batch=%s",
                batch.batch_id,
            )
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Autonomy retry persistence failed batch=%s",
                batch.batch_id,
            )
            return False
        await self._record_retry(
            batch,
            reason,
            retry_after_seconds,
            failure=True,
        )
        return True

    async def _record_retry(
        self,
        batch: AutonomyEventBatch,
        reason: str,
        retry_after_seconds: int,
        *,
        failure: bool,
    ) -> None:
        await self._journal(
            kind="agent.autonomy.deferred",
            workspace_id=batch.workspace_id,
            transport="agent",
            request_id=batch.batch_id,
            payload={
                "channel_id": batch.channel_id,
                "event_count": len(batch.events),
                "reason": reason,
                "retry_after_seconds": retry_after_seconds,
                "failure": failure,
                "failure_count": batch.attempt_count + int(failure),
                "deferral_count": batch.deferral_count + int(not failure),
            },
        )

    async def _deliver_response(
        self,
        batch: AutonomyEventBatch,
        *,
        channel: (
            discord.TextChannel
            | discord.Thread
            | discord.VoiceChannel
            | discord.StageChannel
        ),
        target: discord.Message | None,
        messages: tuple[str, ...],
        context: InvocationContext,
        bot_user_id: int,
    ) -> None:
        purpose = "response"
        specs = tuple(
            AutonomyDeliverySpec(
                purpose=purpose,
                chunk_index=index,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                nonce=agent_delivery_nonce(
                    batch.batch_id,
                    index,
                    purpose=purpose,
                ),
            )
            for index, content in enumerate(messages)
        )
        records = await self.runtime.autonomy_events.prepare_deliveries(
            batch,
            specs,
        )
        resolved: dict[int, discord.Message | None] = {}
        missing: list[AutonomyDeliveryRecord] = []
        for record in records:
            if record.message_id is None:
                missing.append(record)
                continue
            try:
                candidate = await channel.fetch_message(int(record.message_id))
            except discord.NotFound:
                # A saved ID proves the chunk was sent. Manual deletion must not
                # turn a queue ACK retry into an unsolicited repost.
                resolved[record.chunk_index] = None
                continue
            except discord.Forbidden as exc:
                raise _AutonomyTerminalDrop("channel_not_postable") from exc
            except discord.DiscordException as exc:
                raise RuntimeError(
                    "Saved autonomy delivery is temporarily unavailable"
                ) from exc
            self._validate_reconciled_message(
                batch,
                record,
                candidate,
                bot_user_id=bot_user_id,
            )
            resolved[record.chunk_index] = candidate

        if missing:
            unresolved = {
                (record.chunk_index, record.purpose): record
                for record in missing
            }
            # A previously persisted chunk can have the same body as a later
            # missing chunk. Only the event-owned nonce may recover an unsaved
            # message; a content hash alone could claim an unrelated bot post.
            used_message_ids = {
                int(record.message_id)
                for record in records
                if record.message_id is not None
            }
            search_after = min(
                record.prepared_at for record in missing
            ) - timedelta(minutes=1)
            try:
                async for candidate in channel.history(
                    limit=_AUTONOMY_DELIVERY_RECOVERY_LIMIT,
                    after=search_after,
                    oldest_first=True,
                ):
                    if candidate.author.id != bot_user_id:
                        continue
                    nonce = str(getattr(candidate, "nonce", None) or "")
                    matched_record = next(
                        (
                            item
                            for item in unresolved.values()
                            if candidate.id not in used_message_ids
                            and nonce == item.nonce
                        ),
                        None,
                    )
                    if matched_record is None:
                        continue
                    self._validate_reconciled_message(
                        batch,
                        matched_record,
                        candidate,
                        bot_user_id=bot_user_id,
                    )
                    resolved[matched_record.chunk_index] = candidate
                    used_message_ids.add(candidate.id)
                    unresolved.pop(
                        (matched_record.chunk_index, matched_record.purpose)
                    )
                    if not unresolved:
                        break
            except discord.Forbidden as exc:
                raise _AutonomyTerminalDrop("channel_not_postable") from exc
            except discord.DiscordException as exc:
                raise RuntimeError(
                    "Recent autonomy deliveries could not be reconciled"
                ) from exc

        records_by_index = {record.chunk_index: record for record in records}
        delivered_records: list[AutonomyDeliveryRecord] = []
        for index, content in enumerate(messages):
            record = records_by_index[index]
            posted = resolved.get(index)
            if record.message_id is None:
                if posted is None:
                    if index == 0 and target is not None:
                        posted = await target.reply(
                            content,
                            nonce=record.nonce,
                            mention_author=False,
                            allowed_mentions=discord.AllowedMentions.none(),
                            suppress_embeds=True,
                        )
                    else:
                        posted = await channel.send(
                            content,
                            nonce=record.nonce,
                            allowed_mentions=discord.AllowedMentions.none(),
                            suppress_embeds=True,
                        )
                record = await self.runtime.autonomy_events.mark_delivery_sent(
                    batch,
                    purpose=purpose,
                    chunk_index=index,
                    message_id=str(posted.id),
                )
            delivered_records.append(record)

        pending_receipts = tuple(
            record
            for record in delivered_records
            if record.receipt_state is AutonomyDeliveryReceiptState.PENDING
        )
        if pending_receipts:
            message_ids = tuple(
                record.message_id
                for record in delivered_records
                if record.message_id is not None
            )
            if len(message_ids) != len(delivered_records):
                raise AutonomyDeliveryConflictError(
                    f"Autonomy delivery evidence is incomplete for {batch.batch_id}"
                )
            receipted = await _record_agent_host_posts(
                self.runtime,
                channel_id=batch.channel_id,
                message_ids=message_ids,
                context=context,
            )
            if not receipted:
                raise _AutonomyReceiptDeferred
            for record in pending_receipts:
                await self.runtime.autonomy_events.mark_delivery_receipted(
                    batch,
                    purpose=purpose,
                    chunk_index=record.chunk_index,
                )

    @staticmethod
    def _validate_reconciled_message(
        batch: AutonomyEventBatch,
        record: AutonomyDeliveryRecord,
        message: discord.Message,
        *,
        bot_user_id: int,
    ) -> None:
        nonce = getattr(message, "nonce", None)
        if message.author.id != bot_user_id or (
            nonce is not None and str(nonce) != record.nonce
        ) or (
            hashlib.sha256(message.content.encode()).hexdigest()
            != record.content_sha256
        ):
            raise AutonomyDeliveryConflictError(
                f"Autonomy delivery reconciliation conflict for "
                f"{batch.batch_id}:{record.purpose}:{record.chunk_index}"
            )

    async def _inspect(self, batch: AutonomyEventBatch) -> bool:
        agent = self.runtime.agent
        if agent is None:
            raise _AutonomyTerminalDrop("agent_disabled")
        channel_id = batch.channel_id
        message_id = batch.message_id
        workspace_id = batch.workspace_id
        if workspace_id not in self.runtime.settings.agent_autonomy_guild_ids:
            raise _AutonomyTerminalDrop("guild_not_allowlisted")
        guild = self.bot.get_guild(int(workspace_id))
        if guild is None:
            raise _AutonomyTerminalDrop("guild_unavailable")
        bot_member = guild.me
        if bot_member is None:
            raise RuntimeError("Discord bot member is temporarily unavailable")
        try:
            service_principal = ServicePrincipal(bot_member)
        except ValueError as exc:
            raise _AutonomyTerminalDrop("service_principal_invalid") from exc
        service_resource_ids = readable_for_service(guild, service_principal)
        autonomy_policy_mode = self.runtime.settings.agent_autonomy_policy_mode
        resource_ids = (
            (channel_id,)
            if (
                autonomy_policy_mode is AgentAutonomyPolicyMode.STRICT
                and channel_id in service_resource_ids
            )
            else service_resource_ids
        )
        if channel_id not in resource_ids:
            raise _AutonomyTerminalDrop("channel_not_readable")
        channel = self.bot.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(int(channel_id))
            except (discord.NotFound, discord.Forbidden) as exc:
                raise _AutonomyTerminalDrop("channel_not_messageable") from exc
            except discord.DiscordException as exc:
                raise RuntimeError(
                    "Autonomy response channel is temporarily unavailable"
                ) from exc
        if not isinstance(
            channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            raise _AutonomyTerminalDrop("channel_not_messageable")
        permissions = channel.permissions_for(bot_member)
        can_send = (
            permission_enabled(permissions, "send_messages_in_threads")
            if isinstance(channel, discord.Thread)
            else permission_enabled(permissions, "send_messages")
        )
        if (
            not permission_enabled(permissions, "administrator")
            and (
                not permission_enabled(permissions, "view_channel")
                or not permission_enabled(permissions, "read_message_history")
                or not can_send
                or (
                    isinstance(channel, (discord.VoiceChannel, discord.StageChannel))
                    and not permission_enabled(permissions, "connect")
                )
            )
        ) or (
            isinstance(channel, discord.Thread)
            and (channel.archived or channel.locked)
        ):
            raise _AutonomyTerminalDrop("channel_not_postable")
        bot_user = self.bot.user
        if bot_user is None:
            raise RuntimeError("Discord bot identity is temporarily unavailable")
        autonomy_actor_id = str(bot_user.id)
        grants = _agent_grants(
            self.runtime,
            actor_id=autonomy_actor_id,
            autonomous=True,
        )
        mode = self.runtime.settings.agent_autonomy_mode
        event_kinds = frozenset(event.kind for event in batch.events)
        approvals = _autonomy_approvals(
            self.runtime,
            mode,
            event_kinds,
            policy_mode=autonomy_policy_mode,
        )
        allowed_capabilities = _autonomy_allowed_capabilities(
            self.runtime,
            mode,
            event_kinds,
            policy_mode=autonomy_policy_mode,
        )
        event_pointers = tuple(
            AgentEvent(
                event_id=f"autonomy:queue:{event.sequence}",
                kind=event.kind.value,
                occurred_at=event.occurred_at,
                workspace_id=event.workspace_id,
                payload={
                    **event.payload,
                    "source_actor_id": event.actor_id,
                    "channel_id": event.channel_id,
                    "message_id": event.message_id,
                },
            )
            for event in batch.events
        )
        public_reference_id = (
            await self.runtime.agent_store.public_reference_id_for_event(batch.batch_id)
            or new_agent_public_reference_id()
        )
        task_id = (
            await self.runtime.agent_store.task_id_for_event(batch.batch_id)
            or new_agent_task_id()
        )
        conversation_id = discord_conversation_id(
            guild_id=int(workspace_id) if workspace_id else None,
            channel_id=int(channel_id),
            actor_id=autonomy_actor_id,
            grants=grants,
            compatibility_epoch=(
                self.runtime.settings.agent_conversation_compatibility_epoch
            ),
        )
        request = AgentRequest(
            conversation_id=task_scoped_conversation_id(
                conversation_id,
                task_id,
            ),
            event_id=batch.batch_id,
            trigger=AgentTrigger.AUTONOMOUS,
            actor_id=autonomy_actor_id,
            actor_name="Simajilord autonomy",
            workspace_id=workspace_id,
            channel_id=channel_id,
            message_id=message_id,
            occurred_at=max(event.occurred_at for event in batch.events),
            resource_ids=resource_ids,
            public_reference_id=public_reference_id,
            task_id=task_id,
            principal_kind="service",
            read_scope_mode=(
                "resource_ids"
                if autonomy_policy_mode is AgentAutonomyPolicyMode.STRICT
                else "service_live"
            ),
            information_flow_mode=(
                self.runtime.settings.agent_information_flow_mode
            ),
            file_workspace_mode=self.runtime.settings.agent_file_workspace_mode,
            high_risk_authorization_mode=(
                self.runtime.settings.agent_high_risk_authorization_mode
            ),
            executor_principal_id=autonomy_actor_id,
            trigger_actor_ids=tuple(
                sorted(
                    {
                        event.actor_id
                        for event in batch.events
                        if event.actor_id is not None
                    }
                )
            ),
            policy_id=f"discord-autonomy-{autonomy_policy_mode.value}-v1",
            allowed_capabilities=allowed_capabilities,
            grants=grants,
            approvals=approvals,
            events=event_pointers,
        )
        response = await agent.respond(request)
        if response.content.strip() in {
            AGENT_FINAL_DELIVERED_CONTENT,
            AGENT_NO_ACTION_CONTENT,
        }:
            return True
        messages = _agent_message_groups(response.content)
        if not messages:
            return True
        target = None
        if message_id is not None:
            try:
                target = await channel.fetch_message(int(message_id))
            except discord.NotFound:
                target = None
            except discord.Forbidden as exc:
                raise _AutonomyTerminalDrop("channel_not_postable") from exc
            except discord.DiscordException as exc:
                raise RuntimeError(
                    "Autonomy reply target is temporarily unavailable"
                ) from exc
        host_post_context = _agent_invocation_context(request)
        try:
            await self._deliver_response(
                batch,
                channel=channel,
                target=target,
                messages=messages,
                context=host_post_context,
                bot_user_id=bot_user.id,
            )
        except discord.Forbidden as exc:
            raise _AutonomyTerminalDrop("channel_not_postable") from exc
        await self._journal(
            kind="agent.autonomy.acted",
            actor_id=autonomy_actor_id,
            workspace_id=workspace_id,
            transport="agent",
            request_id=batch.batch_id,
            payload={
                "channel_id": channel_id,
                "event_count": len(batch.events),
                "message_count": len(messages),
                "mode": mode.value,
            },
        )
        return True


class PrefixCog(commands.Cog):
    """Prefix presentation for the same APIs used by slash commands."""

    def __init__(self, bot: commands.Bot, runtime: SimajilordRuntime) -> None:
        self.bot = bot
        self.runtime = runtime
        dashboard = getattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, None)
        self.dashboard = dashboard if isinstance(dashboard, MusicDashboardManager) else None

    def _bind_music_dashboard(self, context: BotContext) -> None:
        if self.dashboard is not None:
            guild_id = context.guild.id if context.guild is not None else None
            self.dashboard.bind(guild_id, context.channel.id)

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
                "Health check",
                fields=(
                    EmbedField(
                        "Status",
                        "Operational" if response.status == "ok" else response.status,
                    ),
                    EmbedField(
                        "Discord latency",
                        f"{response.transport_latency_ms:.1f} ms",
                    ),
                ),
                tone=EmbedTone.SUCCESS,
            )
        )

    @commands.command(name="help")
    async def help(self, context: BotContext, *, topic: str = "") -> None:
        normalized = " ".join(topic.strip().removeprefix("/").split()).casefold()
        entry = HELP_ENTRIES_BY_TOPIC.get(normalized) if normalized else None
        if normalized and entry is None:
            await context.send(
                embed=command_embed(
                    "Unknown help topic",
                    description=(
                        f"No public command matches "
                        f"`{discord.utils.escape_markdown(topic)}`.\n"
                        "Use `help` without a topic to browse the command categories."
                    ),
                    tone=EmbedTone.WARNING,
                )
            )
            return
        await context.send(
            embed=_help_entry_embed(entry) if entry is not None else _help_overview_embed()
        )

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
                f"• `{item.name}` — {item.summary} — Risk: **{_risk_label(item.risk)}**"
                for item in response.capabilities
            )
            or "No capability matches that task."
        )
        await context.send(embed=command_embed("Capabilities", description=description))

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
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

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
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

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
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    async def detectai(self, context: BotContext) -> None:
        try:
            if not context.message.attachments:
                raise UserError("moderation.media_empty")
            media = context.message.attachments[0]
            if media.size > self.runtime.settings.hive_max_media_bytes:
                raise UserError("moderation.media_too_large")
            async with context.typing():
                content = await read_attachment_bytes(media)
                if len(content) > self.runtime.settings.hive_max_media_bytes:
                    raise UserError("moderation.media_too_large")
                response = cast(
                    SyntheticMediaAnalyzeResponse,
                    await self.runtime.registry.invoke(
                        "moderation.detect_synthetic_media",
                        SyntheticMediaAnalyzeRequest(
                            filename=media.filename,
                            content_type=media.content_type,
                            content=content,
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
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="play")
    async def play(self, context: BotContext, *, reference: str) -> None:
        try:
            self._bind_music_dashboard(context)
            guild_id = context.guild.id if context.guild else None
            session = _discord_audio_session(self.bot, self.runtime, guild_id)
            reservation = await session.reserve_manual_music_start()
            try:
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
                            silent=True,
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
            finally:
                await reservation.release()
            await context.send(
                embed=music_added_embed(response),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    @commands.command(name="audio")
    async def queue(self, context: BotContext, page: int = 1) -> None:
        try:
            self._bind_music_dashboard(context)
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
                embed=music_queue_embed(response, page=page),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
                silent=True,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    async def history(self, context: BotContext, limit: int = 10) -> None:
        try:
            self._bind_music_dashboard(context)
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
            await context.send(embed=music_history_embed(response), silent=True)
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    async def nowplaying(self, context: BotContext) -> None:
        try:
            self._bind_music_dashboard(context)
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
                embed=music_now_playing_embed(response),
                view=MusicControlsView(
                    self.runtime,
                    self.dashboard,
                    response=response,
                ),
                silent=True,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    async def _control(
        self,
        context: BotContext,
        action: AudioAction,
        *,
        position: int | None = None,
        to_position: int | None = None,
        music_percent: int | None = None,
        speech_percent: int | None = None,
    ) -> None:
        try:
            self._bind_music_dashboard(context)
            if context.guild is None:
                raise UserError("workspace.required")
            session = self.runtime.audio.require(str(context.guild.id))
            _require_same_voice(session, context.author)
            capability_name, request = audio_control_capability_call(
                action,
                position=position,
                to_position=to_position,
                music_percent=music_percent,
                speech_percent=speech_percent,
            )
            response = cast(
                AudioControlResponse,
                await self.runtime.registry.invoke(
                    capability_name,
                    request,
                    prefix_context(context),
                ),
            )
            description = _AUDIO_ACTION_MESSAGES.get(
                response.action,
                "Audio state updated.",
            )
            if response.action == AudioAction.MOVE.value:
                description = (
                    f"Moved **{response.affected_title}** to pending position **{to_position}**."
                )
            elif response.action == AudioAction.CLEAR_MINE.value:
                description = f"Removed **{response.removed_count or 0}** of your pending tracks."
            elif response.action == AudioAction.VOLUME.value:
                description = (
                    f"Music **{response.music_volume_percent}%** · "
                    f"Read aloud **{response.speech_volume_percent}%**"
                )
            await context.send(
                embed=command_embed(
                    "Audio updated",
                    description=description,
                    tone=EmbedTone.SUCCESS,
                ),
                silent=True,
                delete_after=8,
            )
        except Exception as exc:
            await context.send(
                embed=command_embed(
                    "Could not complete the request",
                    description=error_message(
                        exc,
                        request_id=str(context.message.id),
                    ),
                    tone=EmbedTone.ERROR,
                )
            )

    async def pause(self, context: BotContext) -> None:
        await self._control(context, AudioAction.PAUSE)

    async def resume(self, context: BotContext) -> None:
        await self._control(context, AudioAction.RESUME)

    async def skip(self, context: BotContext) -> None:
        await self._control(context, AudioAction.SKIP)

    async def stop(self, context: BotContext) -> None:
        await self._control(context, AudioAction.STOP)

    async def leave(self, context: BotContext) -> None:
        await self._control(context, AudioAction.LEAVE)

    async def volume(
        self,
        context: BotContext,
        music: int | None = None,
        read_aloud: int | None = None,
    ) -> None:
        await self._control(
            context,
            AudioAction.VOLUME,
            music_percent=music,
            speech_percent=read_aloud,
        )

    async def move(
        self,
        context: BotContext,
        from_position: int,
        to_position: int,
    ) -> None:
        await self._control(
            context,
            AudioAction.MOVE,
            position=from_position,
            to_position=to_position,
        )

    async def clear_mine(self, context: BotContext) -> None:
        await self._control(context, AudioAction.CLEAR_MINE)


async def setup_cogs(bot: commands.Bot, runtime: SimajilordRuntime) -> None:
    from .feedback import FeedbackCog

    dashboard = MusicDashboardManager(bot, runtime)
    setattr(bot, _MUSIC_DASHBOARD_ATTRIBUTE, dashboard)
    bot.add_view(MusicControlsView(runtime, dashboard))
    await bot.add_cog(HelpCog())
    await bot.add_cog(FeedbackCog(runtime))
    await bot.add_cog(SystemCog(bot, runtime))
    await bot.add_cog(FocusTimerCog(bot, runtime))
    music_cog = MusicCog(bot, runtime, dashboard)
    await bot.add_cog(music_cog)
    bot.tree.add_command(
        app_commands.ContextMenu(
            name=_PLAY_AUDIO_CONTEXT_MENU_NAME,
            callback=music_cog.play_attachment,
        )
    )
    await bot.add_cog(ReadAloudCog(bot, runtime))
    await bot.add_cog(VoiceLifecycleCog(bot, runtime))
    await bot.add_cog(WebCog(runtime))
    translation_cog = TranslationCog(runtime)
    await bot.add_cog(translation_cog)
    bot.tree.add_command(
        app_commands.ContextMenu(
            name=_TRANSLATE_CONTEXT_MENU_NAME,
            callback=translation_cog.translate_message,
        )
    )
    await bot.add_cog(MediaCog(runtime))
    await bot.add_cog(UtilityCog(runtime))
    await bot.add_cog(InfoCog(runtime))
    await bot.add_cog(MessageExpandCog(runtime))
    quote_cog = QuoteCog(runtime)
    await bot.add_cog(quote_cog)
    bot.tree.add_command(
        app_commands.ContextMenu(
            name=_QUOTE_CONTEXT_MENU_NAME,
            callback=quote_cog.create_quote,
        )
    )
    await bot.add_cog(AgentCog(bot, runtime))
    await bot.add_cog(ObservationCog(bot, runtime))
    await bot.add_cog(AgentAutonomyCog(bot, runtime))
    await bot.add_cog(PrefixCog(bot, runtime))
