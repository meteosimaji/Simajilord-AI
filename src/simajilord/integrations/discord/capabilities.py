"""Discord-specific permission boundaries shared by commands and the agent."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from pathlib import PurePath
from typing import Any, Literal, TypeAlias, cast
from urllib.parse import urlsplit

import discord
from discord.http import Route
from PIL import Image, UnidentifiedImageError

from simajilord.capabilities.audio import (
    AudioAutoLeaveRequest,
    AudioControlRequest,
    AudioControlResponse,
    AudioLoopRequest,
    AudioMixRequest,
    AudioMixResponse,
    AudioMoveRequest,
    AudioNoArgsRequest,
    AudioPlayRequest,
    AudioPlayResponse,
    AudioQueuePositionRequest,
    AudioSeekRequest,
    AudioTuneRequest,
    AudioVolumeRequest,
)
from simajilord.capabilities.file_scope import (
    file_provenance,
    file_workspace_id,
    provenance_observations,
)
from simajilord.capabilities.moderation import (
    SyntheticMediaAnalyzeRequest,
    SyntheticMediaAnalyzeResponse,
)
from simajilord.capabilities.read_aloud import (
    ReadAloudAction,
    ReadAloudAddSourcesRequest,
    ReadAloudAnnouncementsSetRequest,
    ReadAloudContentModeSetRequest,
    ReadAloudDictionaryListRequest,
    ReadAloudDictionaryRemoveRequest,
    ReadAloudDictionarySetRequest,
    ReadAloudDisableRequest,
    ReadAloudExclusionSetRequest,
    ReadAloudExclusionTarget,
    ReadAloudPolicyResponse,
    ReadAloudRemoveSourceRequest,
    ReadAloudRequest,
    ReadAloudResponse,
    ReadAloudSemanticsSetRequest,
    ReadAloudStatusRequest,
)
from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
)
from simajilord.capabilities.translation import (
    TranslationBatchRequest,
    TranslationBatchResponse,
    TranslationSegmentItem,
)
from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    DisclosureObservation,
    EgressDescriptor,
    EgressFieldKind,
    EgressSinkAudience,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime
from simajilord.services.files import (
    WorkspaceFileProvenance,
    WorkspaceFileRecord,
)
from simajilord.services.quote import (
    QuoteCustomEmojiAsset,
    QuoteRenderRequest,
    QuoteStickerAsset,
)

from .attachment_io import read_attachment_bytes
from .audio import DiscordAudioOutput
from .local_media import attachment_can_play, import_discord_attachment
from .permissions import (
    RequesterPrincipal,
    ServicePrincipal,
    inspect_read_aloud_audience,
    read_aloud_audience_relation,
    readable_for_requester,
    readable_for_service,
)
from .permissions import (
    can_post_expanded_message as _can_post_expanded_message,  # noqa: F401
)
from .permissions import (
    can_read_messages as _can_read_messages,
)
from .permissions import (
    can_read_private_thread as _can_read_private_thread,
)
from .permissions import channel_visibility as _channel_visibility
from .permissions import (
    disclosure_audience_relation as _disclosure_audience_relation,
)
from .permissions import permission_enabled as _permission_enabled
from .presenter import (
    EmbedField,
    EmbedTone,
    agent_embed,
    expanded_message_embeds,
    expanded_message_view,
    quote_message_view,
)

log = logging.getLogger(__name__)

DiscordMessageChannel: TypeAlias = (
    discord.TextChannel | discord.Thread | discord.VoiceChannel | discord.StageChannel
)
DiscordReadableChannel: TypeAlias = DiscordMessageChannel | discord.ForumChannel
DiscordDeliveryPurpose: TypeAlias = Literal[
    "progress",
    "requested_action",
    "final",
]
_OPTIONAL_TARGET_GUILD_DESCRIPTION = (
    "Target guild ID returned by discord.list_servers. Omit for the origin "
    "guild, or when channel_id already identifies a cached shared-guild channel; "
    "the host always rechecks requester and bot membership and permissions."
)
_CUSTOM_EMOJI_PATTERN = re.compile(
    r"<(?P<animated>a?):(?P<name>[A-Za-z0-9_]{2,32}):(?P<id>[0-9]{15,22})>"
)
_CUSTOM_EMOJI_PREVIEW_LIMIT = 25
_CUSTOM_EMOJI_MEDIA_MAX_BYTES = 5_000_000
_QUOTE_AVATAR_MAX_BYTES = 8_000_000
_UNCHANGED_CHANNEL_TOPIC = "__simajilord_unchanged_topic__"
_NO_EXPECTED_STRING_STATE = "__simajilord_no_expected_state__"


@dataclass(frozen=True, slots=True)
class DiscordServerRequest:
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordListServersRequest:
    offset: int = dataclass_field(
        default=0,
        metadata={
            "description": (
                "Zero-based bot-server page offset. For the next page, copy "
                "next_offset from the prior response."
            )
        },
    )
    limit: int = dataclass_field(
        default=15,
        metadata={"description": "Bot servers checked per page, from 1 through 15."},
    )


@dataclass(frozen=True, slots=True)
class DiscordServerRecord:
    server_id: str
    name: str
    readable_channel_count: int


@dataclass(frozen=True, slots=True)
class DiscordListServersResponse:
    servers: tuple[DiscordServerRecord, ...]
    checked_server_count: int = 0
    uncertain_membership_count: int = 0
    membership_checks_complete: bool = True
    next_offset: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordServerResponse:
    server_id: str
    name: str
    owner_id: str | None
    member_count: int | None
    text_channel_count: int
    voice_channel_count: int
    role_count: int
    created_at_iso: str
    icon_url: str | None
    description: str | None = None
    owner_name: str | None = None
    human_count: int | None = None
    bot_count: int | None = None
    category_count: int = 0
    stage_channel_count: int = 0
    forum_channel_count: int = 0
    emoji_count: int = 0
    sticker_count: int = 0
    boost_level: int = 0
    boost_count: int = 0
    preferred_locale: str | None = None
    verification_level: str | None = None
    explicit_content_filter: str | None = None
    features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordUserRequest:
    user_id: str
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordActivityRecord:
    name: str
    type: str
    details: str | None = None
    state: str | None = None
    url: str | None = None
    application_id: str | None = None
    created_at_iso: str | None = None
    start_iso: str | None = None
    end_iso: str | None = None
    emoji: str | None = None
    platform: str | None = None
    session_id: str | None = None
    sync_id: str | None = None
    details_url: str | None = None
    state_url: str | None = None
    large_image_url: str | None = None
    large_image_text: str | None = None
    small_image_url: str | None = None
    small_image_text: str | None = None
    buttons: tuple[str, ...] = ()
    party: str | None = None
    flags: str | None = None
    status_display_type: str | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    track_url: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordVoiceStateRecord:
    user_id: str
    display_name: str
    bot: bool
    channel_id: str
    channel_name: str
    channel_kind: str
    category_id: str | None
    server_muted: bool
    server_deafened: bool
    self_muted: bool
    self_deafened: bool
    streaming: bool
    video: bool
    suppressed: bool
    afk: bool
    requested_to_speak_at_iso: str | None


@dataclass(frozen=True, slots=True)
class DiscordListVoiceStatesRequest:
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )
    offset: int = dataclass_field(
        default=0,
        metadata={"description": "Zero-based voice-state offset."},
    )
    limit: int = dataclass_field(
        default=25,
        metadata={"description": "Visible connected members returned, from 1 through 25."},
    )


@dataclass(frozen=True, slots=True)
class DiscordListVoiceStatesResponse:
    voice_states: tuple[DiscordVoiceStateRecord, ...]
    source_guild_id: str
    total_visible_connected: int
    next_offset: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordUserResponse:
    user_id: str
    display_name: str
    bot: bool
    created_at_iso: str
    joined_at_iso: str | None
    top_role: str | None
    avatar_url: str
    username: str | None = None
    global_name: str | None = None
    nickname: str | None = None
    role_names: tuple[str, ...] = ()
    role_count: int = 0
    status: str | None = None
    pending: bool = False
    timed_out_until_iso: str | None = None
    key_permissions: tuple[str, ...] = ()
    colour_value: int = 0
    source_guild_id: str | None = None
    member: bool = False
    presence_available: bool = False
    desktop_status: str | None = None
    mobile_status: str | None = None
    web_status: str | None = None
    activities: tuple[DiscordActivityRecord, ...] = ()
    voice_state: DiscordVoiceStateRecord | None = None
    premium_since_iso: str | None = None
    guild_avatar_url: str | None = None
    guild_banner_url: str | None = None
    public_flags: tuple[str, ...] = ()
    member_flags: tuple[str, ...] = ()
    role_ids: tuple[str, ...] = ()
    enabled_guild_permissions: tuple[str, ...] = ()
    raw_status: str | None = None
    is_on_mobile: bool = False
    discriminator: str = "0"
    system: bool = False
    global_avatar_url: str | None = None
    banner_url: str | None = None
    display_banner_url: str | None = None
    accent_colour_value: int | None = None
    avatar_decoration_url: str | None = None
    avatar_decoration_sku_id: str | None = None
    mutual_guild_ids: tuple[str, ...] = ()
    primary_guild_id: str | None = None
    primary_guild_tag: str | None = None
    primary_guild_badge_url: str | None = None
    primary_guild_identity_enabled: bool = False


@dataclass(frozen=True, slots=True)
class DiscordListRolesRequest:
    query: str = dataclass_field(
        default="",
        metadata={
            "description": (
                "Optional case-insensitive role-name fragment. Empty returns all roles."
            )
        },
    )
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )
    offset: int = dataclass_field(
        default=0,
        metadata={"description": "Zero-based offset; copy next_offset for the next page."},
    )
    limit: int = dataclass_field(
        default=15,
        metadata={"description": "Role records returned per page, from 1 through 25."},
    )


@dataclass(frozen=True, slots=True)
class DiscordRoleRecord:
    role_id: str
    name: str
    position: int
    colour_value: int
    managed: bool
    mentionable: bool
    hoist: bool
    cached_member_count: int
    member_cache_complete: bool
    assignable_by_requester: bool
    assignable_by_bot: bool


@dataclass(frozen=True, slots=True)
class DiscordListRolesResponse:
    roles: tuple[DiscordRoleRecord, ...]
    source_guild_id: str = ""
    next_offset: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordListChannelsRequest:
    include_threads: bool = dataclass_field(
        default=True,
        metadata={"description": "Include currently discoverable threads."},
    )
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )
    offset: int = dataclass_field(
        default=0,
        metadata={
            "description": (
                "Zero-based readable-channel offset. For the next page, copy "
                "next_offset from the prior response."
            )
        },
    )
    limit: int = dataclass_field(
        default=15,
        metadata={"description": "Readable channels returned per page, from 1 through 15."},
    )


@dataclass(frozen=True, slots=True)
class DiscordChannelRecord:
    channel_id: str
    name: str
    kind: str
    category_id: str | None


@dataclass(frozen=True, slots=True)
class DiscordListChannelsResponse:
    channels: tuple[DiscordChannelRecord, ...]
    source_guild_id: str = ""
    next_offset: int | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordListArchivedThreadsRequest:
    parent_channel_id: str = dataclass_field(
        metadata={
            "description": (
                "Readable text or forum channel ID returned by discord.list_channels."
            )
        }
    )
    before_iso: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "For an older page, copy next_before_iso from the prior response."
            )
        },
    )
    limit: int = dataclass_field(
        default=15,
        metadata={"description": "Archived public threads per page, from 1 through 25."},
    )
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordArchivedThreadRecord:
    channel_id: str
    name: str
    parent_channel_id: str
    kind: str
    locked: bool
    archived_at_iso: str | None


@dataclass(frozen=True, slots=True)
class DiscordListArchivedThreadsResponse:
    threads: tuple[DiscordArchivedThreadRecord, ...]
    source_guild_id: str = ""
    source_parent_channel_id: str = ""
    has_more: bool = False
    next_before_iso: str | None = None
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordReadMessagesRequest:
    channel_id: str = dataclass_field(
        metadata={
            "description": (
                "Readable channel or thread ID returned by discord.list_channels."
            )
        }
    )
    limit: int = dataclass_field(
        default=10,
        metadata={"description": "Messages per chronological page, from 1 through 25."},
    )
    before_message_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "For an older page, copy next_before_message_id from the prior "
                "response; otherwise omit."
            )
        },
    )
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordAttachmentRecord:
    attachment_id: str
    filename: str
    content_type: str | None
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DiscordCustomEmojiRecord:
    index: int
    emoji_id: str
    name: str
    animated: bool
    occurrences: int


@dataclass(frozen=True, slots=True)
class DiscordStickerRecord:
    index: int
    sticker_id: str
    name: str
    format: str
    animated: bool


@dataclass(frozen=True, slots=True)
class DiscordReactionSummaryRecord:
    emoji: str
    count: int


@dataclass(frozen=True, slots=True)
class DiscordExpandedPollAnswerRecord:
    answer_id: str
    text: str
    emoji: str | None
    vote_count: int
    bot_voted: bool
    victor: bool


@dataclass(frozen=True, slots=True)
class DiscordExpandedPollRecord:
    question: str
    answers: tuple[DiscordExpandedPollAnswerRecord, ...]
    total_vote_count: int
    multiple: bool
    expires_at_iso: str | None
    duration_seconds: int | None
    finalized: bool
    counts_are_exact: bool
    victor_answer_id: str | None
    layout_type: str | None


@dataclass(frozen=True, slots=True)
class DiscordMessageRecord:
    message_id: str
    channel_id: str
    guild_id: str
    visibility: Literal["guild_public", "restricted", "uncertain"]
    disclosure_to_origin: Literal["same_or_narrower", "broader", "uncertain"]
    disclosure_warning: str | None
    author_id: str
    author_name: str
    author_is_bot: bool
    content_preview: str
    content_length: int
    preview_truncated: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    reference_message_id: str | None
    edited_at_iso: str | None = None
    reaction_count: int = 0
    reaction_summary: tuple[DiscordReactionSummaryRecord, ...] = ()
    thread_id: str | None = None
    poll: DiscordExpandedPollRecord | None = None


@dataclass(frozen=True, slots=True)
class DiscordReadMessagesResponse:
    messages: tuple[DiscordMessageRecord, ...]
    oldest_message_id: str | None
    newest_message_id: str | None = None
    order: Literal["oldest_to_newest"] = "oldest_to_newest"
    anchor_message_id: str | None = None
    anchor_is_active_message: bool = False
    immediate_predecessor_message_id: str | None = None
    source_guild_id: str = ""
    source_channel_id: str = ""
    visibility: Literal["guild_public", "restricted", "uncertain"] = "uncertain"
    disclosure_to_origin: Literal[
        "same_or_narrower", "broader", "uncertain"
    ] = "uncertain"
    disclosure_warning: str | None = None
    has_more: bool = False
    next_before_message_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordSearchMessagesRequest:
    content: str = dataclass_field(
        default="",
        metadata={
            "description": (
                "Words or phrase likely present in the original Discord text; this "
                "is indexed lexical search, not semantic paraphrase search. May be "
                "empty only when author or time filters are supplied."
            )
        },
    )
    channel_ids: tuple[str, ...] = dataclass_field(
        default=(),
        metadata={
            "description": (
                "Optional channel/thread IDs from discord.list_channels. Empty "
                "searches every channel currently readable by both requester and bot."
            )
        },
    )
    author_ids: tuple[str, ...] = dataclass_field(
        default=(),
        metadata={
            "description": (
                "Optional Discord user IDs. Combine with text or time filters when useful."
            )
        },
    )
    offset: int = dataclass_field(
        default=0,
        metadata={
            "description": (
                "For the next relevance page, copy next_offset only when it is returned. "
                "If next_cursor is returned instead, keep offset=0 and copy that cursor. "
                "Keep 0 when sort_by is timestamp."
            )
        },
    )
    cursor: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Opaque cursor returned as next_cursor when a relevance search spans "
                "more than 500 channels. Copy it unchanged and keep offset=0."
            )
        },
    )
    limit: int = dataclass_field(
        default=10,
        metadata={"description": "Results per page, from 1 through 25."},
    )
    sort_by: Literal["timestamp", "relevance"] = dataclass_field(
        default="relevance",
        metadata={
            "description": (
                "Use relevance for finding a phrase; use timestamp for chronological "
                "research or cursor paging."
            )
        },
    )
    sort_order: Literal["asc", "desc"] = dataclass_field(
        default="desc",
        metadata={
            "description": (
                "Newest first with desc or oldest first with asc for timestamp "
                "search. Discord ignores this field for relevance sorting."
            )
        },
    )
    before_message_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "For the next desc page, copy next_before_message_id from the prior "
                "response. Do not also set before_iso."
            )
        },
    )
    after_message_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "For the next asc page, copy next_after_message_id from the prior "
                "response. Do not also set after_iso."
            )
        },
    )
    before_iso: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Optional exclusive RFC 3339 upper time bound. Do not combine with "
                "before_message_id."
            )
        },
    )
    after_iso: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Optional exclusive RFC 3339 lower time bound. Do not combine with "
                "after_message_id."
            )
        },
    )
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={
            "description": (
                "Guild ID returned by discord.list_servers. Omit for the origin guild."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class DiscordSearchMessagesResponse:
    messages: tuple[DiscordMessageRecord, ...]
    total_results: int
    indexing: bool = False
    retry_after_seconds: float | None = None
    source_guild_id: str = ""
    oldest_message_id: str | None = None
    newest_message_id: str | None = None
    has_more: bool = False
    next_offset: int | None = None
    next_cursor: str | None = None
    cursor_pagination: bool = False
    next_before_message_id: str | None = None
    next_after_message_id: str | None = None
    search_window_exhausted: bool = False
    complete: bool = True


@dataclass(frozen=True, slots=True)
class DiscordGetMessageRequest:
    channel_id: str
    message_id: str
    offset: int = 0
    max_characters: int = 600
    include_reply_context: bool = True
    max_reply_depth: int = 2
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordReplyContextRecord:
    message_id: str
    author_id: str
    author_name: str
    author_is_bot: bool
    content_chunk: str
    content_length: int
    complete: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    reference_message_id: str | None
    poll: DiscordExpandedPollRecord | None = None


@dataclass(frozen=True, slots=True)
class DiscordGetMessageResponse:
    message_id: str
    channel_id: str
    guild_id: str
    visibility: Literal["guild_public", "restricted", "uncertain"]
    disclosure_to_origin: Literal["same_or_narrower", "broader", "uncertain"]
    disclosure_warning: str | None
    author_id: str
    author_name: str
    author_is_bot: bool
    content_chunk: str
    content_length: int
    offset: int
    next_offset: int | None
    complete: bool
    created_at_iso: str
    attachments: tuple[DiscordAttachmentRecord, ...]
    custom_emojis: tuple[DiscordCustomEmojiRecord, ...]
    stickers: tuple[DiscordStickerRecord, ...]
    reference_message_id: str | None
    reply_context: tuple[DiscordReplyContextRecord, ...]
    edited_at_iso: str | None = None
    reaction_count: int = 0
    reaction_summary: tuple[DiscordReactionSummaryRecord, ...] = ()
    thread_id: str | None = None
    content_format: Literal["discord_display_segments"] = "discord_display_segments"
    poll: DiscordExpandedPollRecord | None = None


@dataclass(frozen=True, slots=True)
class DiscordMessageLink:
    guild_id: str
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class DiscordExpandMessageRequest:
    guild_id: str
    channel_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class DiscordExpandedAttachmentRecord:
    filename: str
    content_type: str | None
    size_bytes: int
    url: str
    proxy_url: str
    spoiler: bool


@dataclass(frozen=True, slots=True)
class DiscordExpandedEmbedRecord:
    title: str | None
    description: str | None
    url: str | None
    image_url: str | None
    thumbnail_url: str | None


@dataclass(frozen=True, slots=True)
class DiscordExpandMessageResponse:
    guild_id: str
    channel_id: str
    channel_name: str
    message_id: str
    jump_url: str
    author_id: str
    author_name: str
    author_avatar_url: str
    author_is_bot: bool
    content: str
    created_at_iso: str
    edited_at_iso: str | None
    attachments: tuple[DiscordExpandedAttachmentRecord, ...]
    embeds: tuple[DiscordExpandedEmbedRecord, ...]
    sticker_names: tuple[str, ...]
    poll: DiscordExpandedPollRecord | None
    reply_author_name: str | None
    reply_content_preview: str | None


@dataclass(frozen=True, slots=True)
class DiscordPostExpandedMessageRequest:
    source_guild_id: str
    source_channel_id: str
    source_message_id: str
    destination_channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordPostExpandedMessageResponse:
    message_id: str
    channel_id: str
    source_jump_url: str


@dataclass(frozen=True, slots=True)
class DiscordCreateQuoteImageRequest:
    source_channel_id: str
    source_message_id: str
    destination_channel_id: str
    color: bool = False
    light: bool = False
    flip: bool = False
    bold: bool = False
    vertical: bool = False
    animate: bool = False
    include_jump: bool = True


@dataclass(frozen=True, slots=True)
class DiscordCreateQuoteImageResponse:
    message_id: str
    channel_id: str
    source_message_id: str
    source_jump_url: str
    filename: str
    width: int
    height: int
    rendered_custom_emojis: int
    rendered_stickers: int
    text_truncated: bool
    animated: bool


@dataclass(frozen=True, slots=True)
class DiscordViewCustomEmojiRequest:
    channel_id: str
    message_id: str
    emoji_index: int = 0
    mode: Literal["preview", "animation", "frame"] = "preview"
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewCustomEmojiResponse:
    emoji_index: int
    emoji_id: str
    name: str
    animated: bool
    occurrences: int
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None
    content_type: str
    image_data_url: str


@dataclass(frozen=True, slots=True)
class DiscordViewStickerRequest:
    channel_id: str
    message_id: str
    sticker_index: int = 0
    mode: Literal["preview", "animation", "frame"] = "preview"
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewStickerResponse:
    sticker_index: int
    sticker_id: str
    name: str
    format: str
    animated: bool
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None
    content_type: str
    image_data_url: str


@dataclass(frozen=True, slots=True)
class _DiscordAnimatedMedia:
    content: bytes
    content_type: str
    preview_kind: str
    frame_index: int | None
    frame_count: int
    duration_ms: int | None


@dataclass(frozen=True, slots=True)
class DiscordSendMessageRequest:
    channel_id: str
    content: str
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={"description": _OPTIONAL_TARGET_GUILD_DESCRIPTION},
    )
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordEmbedFieldRequest:
    name: str
    value: str
    inline: bool = False


@dataclass(frozen=True, slots=True)
class DiscordSendEmbedRequest:
    channel_id: str
    title: str
    description: str = ""
    fields: tuple[DiscordEmbedFieldRequest, ...] = ()
    tone: Literal["info", "success", "warning", "error"] = "info"
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={"description": _OPTIONAL_TARGET_GUILD_DESCRIPTION},
    )
    reply_to_message_id: str | None = None
    silent: bool = False
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordReactionRequest:
    channel_id: str
    message_id: str
    emoji: str


@dataclass(frozen=True, slots=True)
class DiscordReactionResponse:
    channel_id: str
    message_id: str
    emoji: str
    reacted: bool
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordAnalyzeAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordImportAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0
    destination_path: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordPlayAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordTranslateMessageRequest:
    channel_id: str
    message_id: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordTranslatedSegmentRecord:
    identifier: str
    original: str
    translation: str


@dataclass(frozen=True, slots=True)
class DiscordTranslateMessageResponse:
    message_id: str
    channel_id: str
    jump_url: str
    author_name: str
    original: str
    translation: str
    source_language: str
    target_language: str
    segments: tuple[DiscordTranslatedSegmentRecord, ...] = ()
    cached: bool = False


@dataclass(frozen=True, slots=True)
class DiscordViewImageAttachmentRequest:
    channel_id: str
    message_id: str
    attachment_index: int = 0


@dataclass(frozen=True, slots=True)
class DiscordViewImageAttachmentResponse:
    filename: str
    content_type: str
    size_bytes: int
    image_data_url: str


@dataclass(frozen=True, slots=True)
class DiscordSendFileRequest:
    channel_id: str
    path: str
    caption: str = ""
    description: str = ""
    spoiler: bool = False
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={"description": _OPTIONAL_TARGET_GUILD_DESCRIPTION},
    )
    reply_to_message_id: str | None = None
    silent: bool = False
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordSendFileResponse:
    message_id: str
    channel_id: str
    filename: str
    size_bytes: int
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordFileAttachmentRequest:
    path: str
    description: str = ""
    spoiler: bool = False


@dataclass(frozen=True, slots=True)
class DiscordSendFilesRequest:
    channel_id: str
    attachments: tuple[DiscordFileAttachmentRequest, ...]
    caption: str = ""
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={"description": _OPTIONAL_TARGET_GUILD_DESCRIPTION},
    )
    reply_to_message_id: str | None = None
    silent: bool = False
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordSendFilesResponse:
    message_id: str
    channel_id: str
    filenames: tuple[str, ...]
    size_bytes: tuple[int, ...]
    total_size_bytes: int
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordSendMessageResponse:
    message_id: str
    channel_id: str
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordReplyMessageRequest:
    channel_id: str
    message_id: str
    content: str
    guild_id: str | None = dataclass_field(
        default=None,
        metadata={"description": _OPTIONAL_TARGET_GUILD_DESCRIPTION},
    )
    silent: bool = False
    purpose: DiscordDeliveryPurpose = "requested_action"


@dataclass(frozen=True, slots=True)
class DiscordMessageWriteRequest:
    channel_id: str
    message_id: str
    content: str = ""
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_pinned: bool | None = None


@dataclass(frozen=True, slots=True)
class DiscordMessageWriteResponse:
    channel_id: str
    message_id: str
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordThreadCreateRequest:
    channel_id: str
    name: str
    message_id: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordThreadResponse:
    channel_id: str
    thread_id: str
    name: str
    archived: bool
    undo_fingerprint: str
    old_name: str | None = None
    old_archived: bool | None = None
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordThreadUpdateRequest:
    thread_id: str
    name: str | None = None
    archived: bool | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_name: str | None = None
    expected_archived: bool | None = None
    expected_undo_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordThreadMemberRequest:
    thread_id: str
    user_id: str
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_present: bool | None = None


@dataclass(frozen=True, slots=True)
class DiscordThreadMemberResponse:
    thread_id: str
    user_id: str
    present: bool
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordForumPostRequest:
    forum_id: str
    title: str
    content: str
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordForumPostResponse:
    channel_id: str
    message_id: str
    thread_id: str
    name: str
    archived: bool
    undo_fingerprint: str


@dataclass(frozen=True, slots=True)
class DiscordRoleCreateRequest:
    name: str
    colour: int = 0
    hoist: bool = False
    mentionable: bool = False
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordRoleResponse:
    role_id: str
    name: str
    undo_fingerprint: str
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordCreatedRoleDeleteRequest:
    role_id: str
    undo_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordCreatedChannelDeleteRequest:
    channel_id: str
    undo_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordCreatedEntityDeleteResponse:
    entity_id: str
    deleted: bool


@dataclass(frozen=True, slots=True)
class DiscordRoleMemberRequest:
    user_id: str
    role_id: str
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_assigned: bool | None = None


@dataclass(frozen=True, slots=True)
class DiscordRoleMemberResponse:
    user_id: str
    role_id: str
    assigned: bool
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordChannelSettingRequest:
    channel_id: str
    topic: str | None = _UNCHANGED_CHANNEL_TOPIC
    slowmode_seconds: int | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_topic: str | None = _NO_EXPECTED_STRING_STATE
    expected_slowmode_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class DiscordChannelSettingResponse:
    channel_id: str
    topic: str | None
    slowmode_seconds: int
    old_topic: str | None
    old_slowmode_seconds: int
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordChannelCreateRequest:
    name: str
    topic: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordChannelCreateResponse:
    channel_id: str
    name: str
    undo_fingerprint: str


@dataclass(frozen=True, slots=True)
class DiscordTimeoutRequest:
    user_id: str
    until_iso: str | None = None
    reason: str = ""
    evidence_message_ids: tuple[str, ...] = ()
    expected_until_iso: str | None = _NO_EXPECTED_STRING_STATE


@dataclass(frozen=True, slots=True)
class DiscordTimeoutResponse:
    user_id: str
    until_iso: str | None
    previous_until_iso: str | None
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordMemberModerationRequest:
    user_id: str
    reason: str
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordMemberModerationResponse:
    user_id: str
    action: str
    changed: bool = True


@dataclass(frozen=True, slots=True)
class DiscordBulkDeleteRequest:
    channel_id: str
    message_ids: tuple[str, ...]
    reason: str
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscordBulkDeleteResponse:
    channel_id: str
    deleted_message_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DiscordDeleteOwnMessageRequest:
    channel_id: str
    message_id: str
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordDeleteOwnMessageResponse:
    message_id: str
    channel_id: str
    deleted: bool
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordDeleteOwnMessagesRequest:
    """Internal bounded Undo request for one multi-post agent response."""

    channel_id: str
    message_ids: str
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordDeleteOwnMessagesResponse:
    channel_id: str
    deleted_message_ids: tuple[str, ...]
    guild_id: str | None = None


@dataclass(frozen=True, slots=True)
class DiscordPollRequest:
    channel_id: str
    question: str
    options: tuple[str, ...]
    duration_hours: int = 24
    multiple: bool = False


@dataclass(frozen=True, slots=True)
class DiscordPollResponse:
    message_id: str
    channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordConnectVoiceRequest:
    channel_id: str


@dataclass(frozen=True, slots=True)
class DiscordConnectVoiceResponse:
    channel_id: str
    connected: bool


def build_discord_endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> tuple[CapabilityEndpoint, ...]:
    async def inspect_server(
        request: DiscordServerRequest,
        context: InvocationContext,
    ) -> DiscordServerResponse:
        guild = _requested_guild(client, context, request.guild_id)
        if context.transport == "agent":
            await _require_common_guild(guild, context)
        cached_members_complete = guild.chunked or (
            guild.member_count is not None and len(guild.members) >= guild.member_count
        )
        bot_count = (
            sum(member.bot for member in guild.members)
            if cached_members_complete
            else None
        )
        owner = guild.get_member(guild.owner_id) if guild.owner_id is not None else None
        return DiscordServerResponse(
            server_id=str(guild.id),
            name=guild.name,
            owner_id=str(guild.owner_id) if guild.owner_id else None,
            member_count=guild.member_count,
            text_channel_count=len(guild.text_channels),
            voice_channel_count=len(guild.voice_channels),
            role_count=len(guild.roles),
            created_at_iso=guild.created_at.isoformat(),
            icon_url=str(guild.icon.url) if guild.icon else None,
            description=guild.description,
            owner_name=owner.display_name if owner is not None else None,
            human_count=(
                len(guild.members) - bot_count
                if cached_members_complete and bot_count is not None
                else None
            ),
            bot_count=bot_count,
            category_count=len(guild.categories),
            stage_channel_count=len(guild.stage_channels),
            forum_channel_count=len(guild.forums),
            emoji_count=len(guild.emojis),
            sticker_count=len(guild.stickers),
            boost_level=guild.premium_tier,
            boost_count=guild.premium_subscription_count or 0,
            preferred_locale=str(guild.preferred_locale),
            verification_level=str(guild.verification_level),
            explicit_content_filter=str(guild.explicit_content_filter),
            features=tuple(sorted(guild.features)),
        )

    async def list_servers(
        request: DiscordListServersRequest,
        context: InvocationContext,
    ) -> DiscordListServersResponse:
        if context.transport != "agent":
            guild = _guild(client, context)
            return DiscordListServersResponse(
                servers=(
                    DiscordServerRecord(
                        server_id=str(guild.id),
                        name=guild.name,
                        readable_channel_count=len(guild.text_channels),
                    ),
                ),
                checked_server_count=1,
            )
        if request.offset < 0:
            raise UserError("discord.server_offset_invalid")
        if not 1 <= request.limit <= 15:
            raise UserError("discord.server_limit_invalid")
        try:
            actor_id = int(context.actor_id)
        except ValueError as exc:
            raise UserError("discord.member_required") from exc
        guilds = sorted(
            client.guilds,
            key=lambda item: (item.name.casefold(), str(item.id)),
        )
        page = guilds[request.offset : request.offset + request.limit]
        records: list[DiscordServerRecord] = []
        uncertain_membership_count = 0
        for guild in page:
            actor = guild.get_member(actor_id)
            if actor is None:
                try:
                    actor = await guild.fetch_member(actor_id)
                except discord.NotFound:
                    continue
                except (discord.Forbidden, discord.HTTPException):
                    uncertain_membership_count += 1
                    continue
                if actor.id != actor_id:
                    uncertain_membership_count += 1
                    continue
            if guild.me is None:
                uncertain_membership_count += 1
                continue
            readable = _agent_readable_channel_ids(guild, actor, context)
            records.append(
                DiscordServerRecord(
                    server_id=str(guild.id),
                    name=guild.name,
                    readable_channel_count=len(readable),
                )
            )
        page_end = request.offset + len(page)
        next_offset = page_end if page_end < len(guilds) else None
        membership_checks_complete = uncertain_membership_count == 0
        return DiscordListServersResponse(
            servers=tuple(records),
            checked_server_count=len(page),
            uncertain_membership_count=uncertain_membership_count,
            membership_checks_complete=membership_checks_complete,
            next_offset=next_offset,
            complete=next_offset is None and membership_checks_complete,
        )

    async def inspect_user(
        request: DiscordUserRequest,
        context: InvocationContext,
    ) -> DiscordUserResponse:
        guild = _requested_guild(client, context, request.guild_id)
        if context.transport == "agent":
            await _require_common_guild(guild, context)
        try:
            user_id = int(request.user_id)
        except ValueError as exc:
            raise UserError("discord.user_id_invalid") from exc
        cached_member = guild.get_member(user_id)
        member = cached_member
        if member is None:
            try:
                member = await guild.fetch_member(user_id)
            except discord.NotFound:
                member = None
            except discord.Forbidden as exc:
                raise UserError("discord.member_lookup_failed") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.member_lookup_failed") from exc
        user = member or client.get_user(user_id)
        if user is None:
            try:
                user = await client.fetch_user(user_id)
            except discord.DiscordException as exc:
                raise UserError("discord.user_not_found") from exc
        key_permission_labels = {
            "administrator": "Administrator",
            "manage_guild": "Manage Server",
            "manage_channels": "Manage Channels",
            "manage_roles": "Manage Roles",
            "manage_messages": "Manage Messages",
            "moderate_members": "Timeout Members",
            "kick_members": "Kick Members",
            "ban_members": "Ban Members",
        }
        key_permissions = (
            tuple(
                label
                for permission, label in key_permission_labels.items()
                if _permission_enabled(member.guild_permissions, permission)
            )
            if member is not None
            else ()
        )
        member_roles = member.roles[1:] if member is not None else ()
        roles = tuple(role.name for role in member_roles)
        intents = getattr(client, "intents", None)
        presence_available = (
            cached_member is not None
            and isinstance(intents, discord.Intents)
            and intents.presences
        )
        presence_member = cached_member if presence_available else None
        voice_state = (
            _voice_state_record(cached_member)
            if cached_member is not None and cached_member.voice is not None
            else None
        )
        primary_guild = user.primary_guild
        global_avatar = user.avatar
        banner = user.banner
        display_banner = member.display_banner if member is not None else banner
        accent_colour = user.accent_colour
        avatar_decoration = user.avatar_decoration
        return DiscordUserResponse(
            user_id=str(user.id),
            display_name=member.display_name if member else user.display_name,
            bot=user.bot,
            created_at_iso=user.created_at.isoformat(),
            joined_at_iso=member.joined_at.isoformat() if member and member.joined_at else None,
            top_role=member.top_role.name if member else None,
            avatar_url=str(user.display_avatar.url),
            username=user.name,
            global_name=user.global_name,
            nickname=member.nick if member is not None else None,
            role_names=roles,
            role_count=len(roles),
            status=(
                str(presence_member.status)
                if presence_member is not None
                else None
            ),
            pending=member.pending is True if member is not None else False,
            timed_out_until_iso=(
                member.timed_out_until.isoformat()
                if member is not None and member.timed_out_until is not None
                else None
            ),
            key_permissions=key_permissions,
            colour_value=member.colour.value if member is not None else 0,
            source_guild_id=str(guild.id),
            member=member is not None,
            presence_available=presence_available,
            desktop_status=(
                str(presence_member.desktop_status)
                if presence_member is not None
                else None
            ),
            mobile_status=(
                str(presence_member.mobile_status)
                if presence_member is not None
                else None
            ),
            web_status=(
                str(presence_member.web_status)
                if presence_member is not None
                else None
            ),
            activities=(
                tuple(_activity_record(activity) for activity in presence_member.activities)
                if presence_member is not None
                else ()
            ),
            voice_state=voice_state,
            premium_since_iso=(
                member.premium_since.isoformat()
                if member is not None and member.premium_since is not None
                else None
            ),
            guild_avatar_url=(
                str(member.guild_avatar.url)
                if member is not None and member.guild_avatar is not None
                else None
            ),
            guild_banner_url=(
                str(member.guild_banner.url)
                if member is not None and member.guild_banner is not None
                else None
            ),
            public_flags=_enabled_flag_names(user.public_flags),
            member_flags=(
                _enabled_flag_names(member.flags)
                if member is not None
                else ()
            ),
            role_ids=tuple(str(role.id) for role in member_roles),
            enabled_guild_permissions=(
                _enabled_flag_names(member.guild_permissions)
                if member is not None
                else ()
            ),
            raw_status=(
                presence_member.raw_status
                if presence_member is not None
                else None
            ),
            is_on_mobile=(
                presence_member.is_on_mobile()
                if presence_member is not None
                else False
            ),
            discriminator=user.discriminator,
            system=user.system,
            global_avatar_url=str(global_avatar.url) if global_avatar else None,
            banner_url=str(banner.url) if banner else None,
            display_banner_url=(
                str(display_banner.url) if display_banner else None
            ),
            accent_colour_value=(
                accent_colour.value if accent_colour is not None else None
            ),
            avatar_decoration_url=(
                str(avatar_decoration.url)
                if avatar_decoration is not None
                else None
            ),
            avatar_decoration_sku_id=(
                str(user.avatar_decoration_sku_id)
                if user.avatar_decoration_sku_id is not None
                else None
            ),
            mutual_guild_ids=tuple(
                sorted(str(mutual_guild.id) for mutual_guild in user.mutual_guilds)
            ),
            primary_guild_id=(
                str(primary_guild.id) if primary_guild.id is not None else None
            ),
            primary_guild_tag=primary_guild.tag,
            primary_guild_badge_url=(
                str(primary_guild.badge.url)
                if primary_guild.badge is not None
                else None
            ),
            primary_guild_identity_enabled=primary_guild.identity_enabled is True,
        )

    async def list_voice_states(
        request: DiscordListVoiceStatesRequest,
        context: InvocationContext,
    ) -> DiscordListVoiceStatesResponse:
        if request.offset < 0:
            raise UserError("discord.voice_state_offset_invalid")
        if not 1 <= request.limit <= 25:
            raise UserError("discord.voice_state_limit_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        if bot is None:
            raise UserError("discord.guild_unavailable")
        records: list[DiscordVoiceStateRecord] = []
        channels = (*guild.voice_channels, *guild.stage_channels)
        for channel in channels:
            if not _can_view_channel(channel, actor) or not _can_view_channel(
                channel,
                bot,
            ):
                continue
            records.extend(
                _voice_state_record(member)
                for member in channel.members
                if member.voice is not None
            )
        records.sort(
            key=lambda item: (
                item.channel_name.casefold(),
                item.display_name.casefold(),
                item.user_id,
            )
        )
        page = tuple(records[request.offset : request.offset + request.limit])
        page_end = request.offset + len(page)
        next_offset = page_end if page_end < len(records) else None
        return DiscordListVoiceStatesResponse(
            voice_states=page,
            source_guild_id=str(guild.id),
            total_visible_connected=len(records),
            next_offset=next_offset,
            complete=next_offset is None,
        )

    async def list_roles(
        request: DiscordListRolesRequest,
        context: InvocationContext,
    ) -> DiscordListRolesResponse:
        if request.offset < 0:
            raise UserError("discord.role_offset_invalid")
        if not 1 <= request.limit <= 25:
            raise UserError("discord.role_limit_invalid")
        query = " ".join(request.query.split()).casefold()
        if len(query) > 100:
            raise UserError("discord.role_query_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        if bot is None:
            raise UserError("discord.guild_unavailable")
        roles = [
            role
            for role in guild.roles
            if not role.is_default() and (not query or query in role.name.casefold())
        ]
        roles.sort(key=lambda role: (-role.position, role.name.casefold(), role.id))
        records = tuple(
            DiscordRoleRecord(
                role_id=str(role.id),
                name=role.name,
                position=role.position,
                colour_value=role.colour.value,
                managed=role.managed,
                mentionable=role.mentionable,
                hoist=role.hoist,
                cached_member_count=len(role.members),
                member_cache_complete=guild.chunked is True
                or (
                    isinstance(guild.member_count, int)
                    and not isinstance(guild.member_count, bool)
                    and len(guild.members) >= guild.member_count
                ),
                assignable_by_requester=_role_assignable_by(actor, role),
                assignable_by_bot=_role_assignable_by(bot, role),
            )
            for role in roles[request.offset : request.offset + request.limit]
        )
        page_end = request.offset + len(records)
        next_offset = page_end if page_end < len(roles) else None
        return DiscordListRolesResponse(
            roles=records,
            source_guild_id=str(guild.id),
            next_offset=next_offset,
            complete=next_offset is None,
        )

    async def list_channels(
        request: DiscordListChannelsRequest,
        context: InvocationContext,
    ) -> DiscordListChannelsResponse:
        if request.offset < 0:
            raise UserError("discord.channel_offset_invalid")
        if not 1 <= request.limit <= 15:
            raise UserError("discord.channel_limit_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        allowed_ids: set[str] | None = None
        if context.transport == "agent":
            actor = await _require_common_guild(guild, context)
            allowed_ids = set(_agent_readable_channel_ids(guild, actor, context))
        records = [
            DiscordChannelRecord(
                channel_id=str(channel.id),
                name=channel.name,
                kind=str(channel.type),
                category_id=str(channel.category_id) if channel.category_id else None,
            )
            for channel in guild.channels
            if allowed_ids is None or str(channel.id) in allowed_ids
        ]
        if request.include_threads:
            records.extend(
                DiscordChannelRecord(
                    channel_id=str(thread.id),
                    name=thread.name,
                    kind="thread",
                    category_id=str(thread.parent_id) if thread.parent_id else None,
                )
                for thread in guild.threads
                if allowed_ids is None or str(thread.id) in allowed_ids
            )
        records.sort(key=lambda item: (item.kind, item.name, item.channel_id))
        page = records[request.offset : request.offset + request.limit]
        page_end = request.offset + len(page)
        next_offset = page_end if page_end < len(records) else None
        return DiscordListChannelsResponse(
            channels=tuple(page),
            source_guild_id=str(guild.id),
            next_offset=next_offset,
            complete=next_offset is None,
        )

    async def list_archived_threads(
        request: DiscordListArchivedThreadsRequest,
        context: InvocationContext,
    ) -> DiscordListArchivedThreadsResponse:
        if not 1 <= request.limit <= 25:
            raise UserError("discord.archived_thread_limit_invalid")
        guild = _requested_guild(client, context, request.guild_id)
        actor = await _require_common_guild(guild, context)
        bot = guild.me
        if bot is None:
            raise UserError("discord.guild_unavailable")
        parent_id = _snowflake(request.parent_channel_id, "channel")
        _assert_agent_channel_scope(context, request.parent_channel_id)
        parent = guild.get_channel(parent_id)
        if not isinstance(parent, (discord.TextChannel, discord.ForumChannel)):
            raise UserError("discord.archived_thread_parent_invalid")
        for member in (actor, bot):
            if not _can_read_messages(parent, member):
                raise UserError("discord.agent_read_channel_forbidden")
        before = _archived_threads_before(request.before_iso)
        discovered: list[discord.Thread] = []
        async for thread in parent.archived_threads(
            limit=request.limit + 1,
            before=before,
        ):
            if (
                context.read_scope_mode == "resource_ids"
                and str(thread.id) not in context.resource_ids
            ) or (
                not _can_read_messages(thread, actor)
                or not _can_read_private_thread(thread, actor)
                or not _can_read_messages(thread, bot)
                or not _can_read_private_thread(thread, bot)
            ):
                continue
            discovered.append(thread)
            if len(discovered) > request.limit:
                break
        has_more = len(discovered) > request.limit
        page = discovered[: request.limit]
        records = tuple(
            DiscordArchivedThreadRecord(
                channel_id=str(thread.id),
                name=thread.name,
                parent_channel_id=str(parent.id),
                kind=str(thread.type),
                locked=thread.locked,
                archived_at_iso=(
                    thread.archive_timestamp.isoformat()
                    if thread.archive_timestamp is not None
                    else None
                ),
            )
            for thread in page
        )
        next_before_iso = (
            records[-1].archived_at_iso if records and has_more else None
        )
        return DiscordListArchivedThreadsResponse(
            threads=records,
            source_guild_id=str(guild.id),
            source_parent_channel_id=str(parent.id),
            has_more=has_more,
            next_before_iso=next_before_iso,
            complete=not has_more,
        )

    async def read_messages(
        request: DiscordReadMessagesRequest,
        context: InvocationContext,
    ) -> DiscordReadMessagesResponse:
        guild, channel = await _readable_message_channel(
            client,
            context,
            guild_id=request.guild_id,
            channel_id=request.channel_id,
        )
        disclosure = _disclosure_to_origin(client, context, guild, channel)
        visibility = _channel_visibility(guild, channel)
        if not 1 <= request.limit <= 25:
            raise UserError("discord.message_limit_invalid")
        before = (
            discord.Object(_snowflake(request.before_message_id, "message"))
            if request.before_message_id
            else None
        )
        event_message_id = _discord_event_message_id(context)
        newest_first_with_lookahead = [
            _message_record(
                message,
                guild_id=str(guild.id),
                visibility=visibility,
                disclosure_to_origin=disclosure,
                event_message_id=event_message_id,
            )
            async for message in channel.history(
                limit=request.limit + 1,
                before=before,
            )
        ]
        has_more = len(newest_first_with_lookahead) > request.limit
        newest_first = newest_first_with_lookahead[: request.limit]
        messages = tuple(reversed(newest_first))
        anchor_is_active_message = (
            request.before_message_id is not None
            and request.before_message_id == context.active_message_id
        )
        return DiscordReadMessagesResponse(
            messages=messages,
            oldest_message_id=messages[0].message_id if messages else None,
            newest_message_id=messages[-1].message_id if messages else None,
            anchor_message_id=request.before_message_id,
            anchor_is_active_message=anchor_is_active_message,
            immediate_predecessor_message_id=(
                messages[-1].message_id
                if messages and anchor_is_active_message
                else None
            ),
            source_guild_id=str(guild.id),
            source_channel_id=str(channel.id),
            visibility=visibility,
            disclosure_to_origin=disclosure,
            disclosure_warning=_disclosure_warning(disclosure),
            has_more=has_more,
            next_before_message_id=(
                messages[0].message_id if messages and has_more else None
            ),
        )

    async def search_messages(
        request: DiscordSearchMessagesRequest,
        context: InvocationContext,
    ) -> DiscordSearchMessagesResponse:
        guild = _requested_guild(client, context, request.guild_id)
        content = request.content.strip()
        if len(content) > 1_024:
            raise UserError("discord.message_search_query_invalid")
        if (
            not isinstance(request.offset, int)
            or isinstance(request.offset, bool)
            or not 0 <= request.offset <= 9_975
            or (request.sort_by == "timestamp" and request.offset != 0)
        ):
            raise UserError("discord.message_search_offset_invalid")
        if (
            request.cursor is not None
            and (
                request.sort_by != "relevance"
                or request.offset != 0
                or not 1 <= len(request.cursor) <= 8_192
            )
        ):
            raise UserError("discord.message_search_cursor_invalid")
        if not 1 <= request.limit <= 25:
            raise UserError("discord.message_search_limit_invalid")
        before_message_id = _search_boundary_message_id(
            request.before_message_id,
            request.before_iso,
            high=False,
        )
        after_message_id = _search_boundary_message_id(
            request.after_message_id,
            request.after_iso,
            high=True,
        )
        if (
            not content
            and not request.author_ids
            and before_message_id is None
            and after_message_id is None
        ):
            raise UserError("discord.message_search_filter_required")
        if (
            before_message_id is not None
            and after_message_id is not None
            and int(after_message_id) >= int(before_message_id)
        ):
            raise UserError("discord.message_search_range_invalid")

        requested_ids = tuple(dict.fromkeys(request.channel_ids))
        actor: discord.Member | None = None
        readable_ids: set[str] | None = None
        if context.transport == "agent":
            actor = await _actor_member(guild, context)
            readable_ids = set(_agent_readable_channel_ids(guild, actor, context))
            if any(channel_id not in readable_ids for channel_id in requested_ids):
                raise UserError("discord.agent_read_channel_forbidden")
            scoped_ids = (
                requested_ids
                if requested_ids
                else tuple(sorted(readable_ids, key=int))
            )
            if not scoped_ids:
                return DiscordSearchMessagesResponse(
                    messages=(),
                    total_results=0,
                    source_guild_id=str(guild.id),
                )
        else:
            scoped_ids = requested_ids
        for channel_id in scoped_ids:
            _snowflake(channel_id, "channel")
        for author_id in request.author_ids:
            _snowflake(author_id, "user")
        if before_message_id:
            _snowflake(before_message_id, "message")
        if after_message_id:
            _snowflake(after_message_id, "message")

        channel_batches: tuple[tuple[str, ...], ...]
        if scoped_ids:
            channel_batches = tuple(
                scoped_ids[index : index + 500]
                for index in range(0, len(scoped_ids), 500)
            )
        else:
            channel_batches = ((),)
        cursor_fingerprint = _message_search_cursor_fingerprint(
            guild_id=str(guild.id),
            channel_batches=channel_batches,
            content=content,
            author_ids=request.author_ids,
            before_message_id=before_message_id,
            after_message_id=after_message_id,
            sort_by=request.sort_by,
            sort_order=request.sort_order,
        )
        if request.sort_by == "relevance":
            if request.cursor is not None:
                batch_offsets, next_batch_index = _decode_message_search_cursor(
                    request.cursor,
                    expected_fingerprint=cursor_fingerprint,
                    batch_count=len(channel_batches),
                )
            elif len(channel_batches) > 1:
                if request.offset != 0:
                    raise UserError("discord.message_search_cursor_required")
                batch_offsets = (0,) * len(channel_batches)
                next_batch_index = 0
            else:
                batch_offsets = (request.offset,)
                next_batch_index = 0
        else:
            batch_offsets = (0,) * len(channel_batches)
            next_batch_index = 0

        batch_hits: list[list[tuple[int, DiscordMessageRecord]]] = [
            [] for _ in channel_batches
        ]
        batch_fetched_counts = [0 for _ in channel_batches]
        batch_totals = [0 for _ in channel_batches]
        indexing = False
        retry_after_seconds: float | None = None
        for batch_index, channel_batch in enumerate(channel_batches):
            params: list[tuple[str, str]] = [
                ("limit", str(request.limit)),
                ("sort_by", request.sort_by),
                ("sort_order", request.sort_order),
                ("include_nsfw", "false"),
            ]
            if content:
                params.append(("content", content))
            if request.sort_by == "relevance":
                params.append(("offset", str(batch_offsets[batch_index])))
            params.extend(("channel_id", channel_id) for channel_id in channel_batch)
            params.extend(("author_id", author_id) for author_id in request.author_ids)
            if before_message_id:
                params.append(("max_id", before_message_id))
            if after_message_id:
                params.append(("min_id", after_message_id))
            try:
                payload = await client.http.request(
                    Route(
                        "GET",
                        "/guilds/{guild_id}/messages/search",
                        guild_id=guild.id,
                    ),
                    params=params,
                )
            except discord.Forbidden as exc:
                raise UserError("discord.message_search_forbidden") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.message_search_failed") from exc
            if not isinstance(payload, dict):
                raise UserError("discord.message_search_failed")
            if payload.get("code") == 110000:
                indexing = True
                retry = payload.get("retry_after")
                if isinstance(retry, (int, float)) and not isinstance(retry, bool):
                    retry_after_seconds = max(retry_after_seconds or 0.0, float(retry))
                continue
            if payload.get("doing_deep_historical_index") is True:
                # Discord can return usable recent hits while its older index is
                # still being built. Expose that partial state instead of
                # presenting the current page as complete.
                indexing = True
            raw_total = payload.get("total_results")
            if isinstance(raw_total, int) and not isinstance(raw_total, bool):
                batch_totals[batch_index] = max(0, raw_total)
            raw_groups = payload.get("messages")
            if not isinstance(raw_groups, list):
                continue
            batch_fetched_counts[batch_index] = len(raw_groups)
            raw_thread_parents = _search_thread_parents(payload)
            for group_position, raw_group in enumerate(raw_groups, start=1):
                if not isinstance(raw_group, list):
                    continue
                group_record: DiscordMessageRecord | None = None
                for raw_message in raw_group:
                    record = _search_message_record(raw_message)
                    if record is None:
                        continue
                    source = _search_result_source(
                        guild,
                        channel_id=record.channel_id,
                        readable_ids=readable_ids,
                        actor=actor,
                        raw_thread_parents=raw_thread_parents,
                    )
                    if source is None:
                        continue
                    disclosure = _disclosure_to_origin(
                        client,
                        context,
                        guild,
                        source,
                    )
                    record = DiscordMessageRecord(
                        message_id=record.message_id,
                        channel_id=record.channel_id,
                        guild_id=str(guild.id),
                        visibility=_channel_visibility(guild, source),
                        disclosure_to_origin=disclosure,
                        disclosure_warning=_disclosure_warning(disclosure),
                        author_id=record.author_id,
                        author_name=record.author_name,
                        author_is_bot=record.author_is_bot,
                        content_preview=record.content_preview,
                        content_length=record.content_length,
                        preview_truncated=record.preview_truncated,
                        created_at_iso=record.created_at_iso,
                        attachments=record.attachments,
                        reference_message_id=record.reference_message_id,
                        edited_at_iso=record.edited_at_iso,
                        reaction_count=record.reaction_count,
                        reaction_summary=record.reaction_summary,
                        thread_id=record.thread_id,
                    )
                    group_record = record
                    break
                if group_record is not None:
                    batch_hits[batch_index].append(
                        (group_position, group_record)
                    )

        total_results = sum(batch_totals)
        next_cursor: str | None = None
        if request.sort_by == "timestamp":
            records = [
                record
                for hits in batch_hits
                for _, record in hits
            ]
            records.sort(
                key=lambda item: (item.created_at_iso, int(item.message_id)),
                reverse=request.sort_order == "desc",
            )
            page = tuple(records[: request.limit])
            has_more = bool(page) and total_results > len(page)
            next_offset = None
            search_window_exhausted = False
        else:
            merged: list[DiscordMessageRecord] = []
            consumed_positions = [0 for _ in channel_batches]
            next_offsets = list(batch_offsets)
            seen_message_ids: set[str] = set()
            last_batch_index = next_batch_index
            while len(merged) < request.limit:
                progressed = False
                for step in range(len(channel_batches)):
                    batch_index = (
                        next_batch_index + step
                    ) % len(channel_batches)
                    position = consumed_positions[batch_index]
                    hits = batch_hits[batch_index]
                    if position >= len(hits):
                        continue
                    raw_position, record = hits[position]
                    consumed_positions[batch_index] += 1
                    next_offsets[batch_index] = (
                        batch_offsets[batch_index] + raw_position
                    )
                    last_batch_index = batch_index
                    progressed = True
                    if record.message_id not in seen_message_ids:
                        seen_message_ids.add(record.message_id)
                        merged.append(record)
                    if len(merged) >= request.limit:
                        break
                if not progressed:
                    break
                next_batch_index = (last_batch_index + 1) % len(channel_batches)
            for batch_index, hits in enumerate(batch_hits):
                if consumed_positions[batch_index] >= len(hits):
                    next_offsets[batch_index] = (
                        batch_offsets[batch_index]
                        + batch_fetched_counts[batch_index]
                    )
            page = tuple(merged)
            batch_has_more = tuple(
                total > consumed
                for total, consumed in zip(
                    batch_totals,
                    next_offsets,
                    strict=True,
                )
            )
            has_more = any(batch_has_more)
            search_window_exhausted = any(
                more and offset > 9_975
                for more, offset in zip(
                    batch_has_more,
                    next_offsets,
                    strict=True,
                )
            )
            if has_more and not search_window_exhausted:
                if len(channel_batches) == 1 and request.cursor is None:
                    next_offset = next_offsets[0]
                else:
                    next_offset = None
                    next_cursor = _encode_message_search_cursor(
                        fingerprint=cursor_fingerprint,
                        offsets=tuple(next_offsets),
                        next_batch_index=next_batch_index,
                    )
            else:
                next_offset = None
        oldest_message_id = (
            min((item.message_id for item in page), key=int) if page else None
        )
        newest_message_id = (
            max((item.message_id for item in page), key=int) if page else None
        )
        return DiscordSearchMessagesResponse(
            messages=page,
            total_results=total_results,
            indexing=indexing,
            retry_after_seconds=retry_after_seconds,
            source_guild_id=str(guild.id),
            oldest_message_id=oldest_message_id,
            newest_message_id=newest_message_id,
            has_more=has_more,
            next_offset=next_offset,
            next_cursor=next_cursor,
            cursor_pagination=(
                request.sort_by == "relevance" and len(channel_batches) > 1
            ),
            next_before_message_id=(
                oldest_message_id
                if (
                    has_more
                    and request.sort_by == "timestamp"
                    and request.sort_order == "desc"
                )
                else None
            ),
            next_after_message_id=(
                newest_message_id
                if (
                    has_more
                    and request.sort_by == "timestamp"
                    and request.sort_order == "asc"
                )
                else None
            ),
            search_window_exhausted=search_window_exhausted,
            complete=not indexing and not has_more,
        )

    async def get_message(
        request: DiscordGetMessageRequest,
        context: InvocationContext,
    ) -> DiscordGetMessageResponse:
        guild, channel = await _readable_message_channel(
            client,
            context,
            guild_id=request.guild_id,
            channel_id=request.channel_id,
        )
        disclosure = _disclosure_to_origin(client, context, guild, channel)
        message_id = _snowflake(request.message_id, "message")
        if request.offset < 0:
            raise UserError("discord.message_offset_invalid")
        if request.max_characters < 1:
            raise UserError("discord.message_chunk_limit_invalid")
        if not 0 <= request.max_reply_depth <= 3:
            raise UserError("discord.reply_depth_invalid")
        # The model may request a larger window than the per-tool response budget.
        # Clamp instead of failing so it can always read the exact trigger message,
        # then follow next_offset only when the remaining text is actually needed.
        chunk_characters = min(request.max_characters, 1_000)
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound as exc:
            raise UserError("discord.message_not_found") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_fetch_failed") from exc
        context_text = _message_context_text(message)
        content_length = len(context_text)
        if request.offset > content_length:
            raise UserError("discord.message_offset_invalid")
        end = min(content_length, request.offset + chunk_characters)
        next_offset = end if end < content_length else None
        reply_context = (
            await _reply_context(
                channel,
                message,
                max_depth=request.max_reply_depth,
                chunk_characters=min(chunk_characters, 600),
            )
            if request.include_reply_context and request.max_reply_depth
            else ()
        )
        return DiscordGetMessageResponse(
            message_id=str(message.id),
            channel_id=str(message.channel.id),
            guild_id=str(guild.id),
            visibility=_channel_visibility(guild, channel),
            disclosure_to_origin=disclosure,
            disclosure_warning=_disclosure_warning(disclosure),
            author_id=str(message.author.id),
            author_name=message.author.display_name,
            author_is_bot=message.author.bot,
            content_chunk=context_text[request.offset : end],
            content_length=content_length,
            offset=request.offset,
            next_offset=next_offset,
            complete=next_offset is None,
            created_at_iso=message.created_at.isoformat(),
            attachments=tuple(_attachment_record(attachment) for attachment in message.attachments),
            custom_emojis=_custom_emoji_records(message.content),
            stickers=_sticker_records(message.stickers),
            reference_message_id=(
                str(message.reference.message_id)
                if message.reference and message.reference.message_id
                else None
            ),
            reply_context=reply_context,
            edited_at_iso=_message_edited_at_iso(message),
            reaction_count=sum(
                reaction.count for reaction in getattr(message, "reactions", ())
            ),
            reaction_summary=tuple(
                DiscordReactionSummaryRecord(str(reaction.emoji), reaction.count)
                for reaction in getattr(message, "reactions", ())[:10]
            ),
            thread_id=_message_thread_id(message),
            poll=_expanded_poll(message.poll),
        )

    async def expand_message(
        request: DiscordExpandMessageRequest,
        context: InvocationContext,
    ) -> DiscordExpandMessageResponse:
        guild = _guild(client, context)
        if request.guild_id != str(guild.id):
            raise UserError("discord.expand_cross_guild_forbidden")
        channel, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        reply_author_name, reply_content_preview = _expanded_reply(message)
        return DiscordExpandMessageResponse(
            guild_id=str(guild.id),
            channel_id=str(channel.id),
            channel_name=channel.name,
            message_id=str(message.id),
            jump_url=message.jump_url,
            author_id=str(message.author.id),
            author_name=message.author.display_name,
            author_avatar_url=str(message.author.display_avatar.url),
            author_is_bot=message.author.bot,
            content=message.content,
            created_at_iso=message.created_at.isoformat(),
            edited_at_iso=(
                message.edited_at.isoformat() if message.edited_at is not None else None
            ),
            attachments=tuple(
                _expanded_attachment(attachment) for attachment in message.attachments[:10]
            ),
            embeds=tuple(_expanded_embed(item) for item in message.embeds[:10]),
            sticker_names=tuple(sticker.name for sticker in message.stickers[:10]),
            poll=_expanded_poll(message.poll),
            reply_author_name=reply_author_name,
            reply_content_preview=reply_content_preview,
        )

    async def translate_message(
        request: DiscordTranslateMessageRequest,
        context: InvocationContext,
    ) -> DiscordTranslateMessageResponse:
        guild = _guild(client, context)
        channel, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        segments = discord_translation_segments(message)
        if not segments:
            raise UserError("translation.message_text_required")
        translated = cast(
            TranslationBatchResponse,
            await runtime.registry.invoke(
                "translation.translate_batch",
                TranslationBatchRequest(
                    segments=segments,
                    target_language=request.target_language,
                    source_language=request.source_language,
                ),
                context,
            ),
        )
        response = DiscordTranslateMessageResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            jump_url=message.jump_url,
            author_name=message.author.display_name,
            original="\n".join(item.original for item in translated.segments),
            translation="\n".join(item.translation for item in translated.segments),
            source_language=translated.source_language,
            target_language=translated.target_language,
            segments=tuple(
                DiscordTranslatedSegmentRecord(
                    identifier=item.identifier,
                    original=item.original,
                    translation=item.translation,
                )
                for item in translated.segments
            ),
            cached=translated.cached,
        )
        await runtime.journal.append(
            kind="translation.document",
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            transport=context.transport,
            request_id=context.request_id,
            payload={
                "message_id": str(message.id),
                "channel_id": str(channel.id),
                "segment_count": len(response.segments),
                "source_language": response.source_language,
                "target_language": response.target_language,
                "cached": response.cached,
            },
        )
        log.info(
            "Translated Discord message message=%s channel=%s segments=%s "
            "source=%s target=%s cached=%s",
            response.message_id,
            response.channel_id,
            len(response.segments),
            response.source_language,
            response.target_language,
            response.cached,
        )
        return response

    async def post_expanded_message(
        request: DiscordPostExpandedMessageRequest,
        context: InvocationContext,
    ) -> DiscordPostExpandedMessageResponse:
        response = await expand_message(
            DiscordExpandMessageRequest(
                guild_id=request.source_guild_id,
                channel_id=request.source_channel_id,
                message_id=request.source_message_id,
            ),
            context,
        )
        _guild_value, destination, _actor, _bot = await _write_message_channel(
            client,
            context,
            request.destination_channel_id,
            required_permissions=("send_messages", "embed_links"),
        )
        source_guild = _requested_guild(client, context, response.guild_id)
        source_channel = _message_channel(source_guild, response.channel_id)
        _enforce_source_to_destination(
            context,
            source_guild,
            source_channel,
            _guild_value,
            destination,
        )
        try:
            posted = await destination.send(
                embeds=expanded_message_embeds(response),
                view=expanded_message_view(response.jump_url),
                allowed_mentions=discord.AllowedMentions.none(),
                silent=True,
            )
        except discord.Forbidden as exc:
            raise UserError("discord.expand_destination_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.expand_failed") from exc
        return DiscordPostExpandedMessageResponse(
            message_id=str(posted.id),
            channel_id=str(destination.id),
            source_jump_url=response.jump_url,
        )

    async def create_quote_image(
        request: DiscordCreateQuoteImageRequest,
        context: InvocationContext,
    ) -> DiscordCreateQuoteImageResponse:
        guild = _guild(client, context)
        source_channel, message = await _fetch_readable_message(
            guild,
            channel_id=request.source_channel_id,
            message_id=request.source_message_id,
            context=context,
        )
        _destination_guild, destination, _actor, _bot = await _write_message_channel(
            client,
            context,
            request.destination_channel_id,
            required_permissions=("send_messages", "attach_files"),
        )
        _enforce_source_to_destination(
            context,
            guild,
            source_channel,
            _destination_guild,
            destination,
        )
        avatar, custom_emojis, stickers = await asyncio.gather(
            _quote_avatar(client, message),
            _quote_custom_emojis(client, message.content),
            _quote_stickers(client, message),
        )
        try:
            rendered = await asyncio.to_thread(
                runtime.quote.render,
                QuoteRenderRequest(
                    text=_quote_text(message),
                    display_name=message.author.display_name,
                    username=message.author.name,
                    avatar=avatar,
                    custom_emojis=custom_emojis,
                    stickers=stickers,
                    color=request.color,
                    light=request.light,
                    flip=request.flip,
                    bold=request.bold,
                    vertical=request.vertical,
                    animate=request.animate,
                ),
            )
        except (OSError, ValueError) as exc:
            raise UserError("discord.quote_render_failed") from exc
        extension = "gif" if rendered.animated else "png"
        filename = f"quote-{message.id}.{extension}"
        try:
            image_file = discord.File(
                io.BytesIO(rendered.content),
                filename=filename,
            )
            if request.include_jump:
                jump_view = quote_message_view(message.jump_url)
                assert jump_view is not None
                posted = await destination.send(
                    file=image_file,
                    view=jump_view,
                    allowed_mentions=discord.AllowedMentions.none(),
                    silent=True,
                )
            else:
                posted = await destination.send(
                    file=image_file,
                    allowed_mentions=discord.AllowedMentions.none(),
                    silent=True,
                )
        except discord.Forbidden as exc:
            raise UserError("discord.quote_destination_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.quote_failed") from exc
        return DiscordCreateQuoteImageResponse(
            message_id=str(posted.id),
            channel_id=str(destination.id),
            source_message_id=str(message.id),
            source_jump_url=message.jump_url,
            filename=filename,
            width=rendered.width,
            height=rendered.height,
            rendered_custom_emojis=rendered.rendered_custom_emojis,
            rendered_stickers=rendered.rendered_stickers,
            text_truncated=rendered.text_truncated,
            animated=rendered.animated,
        )

    async def view_custom_emoji(
        request: DiscordViewCustomEmojiRequest,
        context: InvocationContext,
    ) -> DiscordViewCustomEmojiResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        custom_emojis = _custom_emoji_records(message.content)
        if not 0 <= request.emoji_index < len(custom_emojis):
            raise UserError("discord.custom_emoji_index_invalid")
        selected = custom_emojis[request.emoji_index]
        if request.frame_index < 0:
            raise UserError("discord.custom_emoji_frame_invalid")
        if request.mode == "animation" and not selected.animated:
            raise UserError("discord.custom_emoji_not_animated")
        if request.mode != "frame" and request.frame_index != 0:
            raise UserError("discord.custom_emoji_frame_mode_required")
        extension = "gif" if selected.animated and request.mode in {"animation", "frame"} else "png"
        emoji_url = (
            f"https://cdn.discordapp.com/emojis/{selected.emoji_id}.{extension}"
            "?size=128&quality=lossless"
        )
        try:
            content = await client.http.get_from_cdn(emoji_url)
        except (discord.NotFound, discord.Forbidden) as exc:
            raise UserError("discord.custom_emoji_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.custom_emoji_failed") from exc
        if not content or len(content) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            raise UserError("discord.custom_emoji_invalid")
        try:
            media = await asyncio.to_thread(
                _prepare_discord_animated_media,
                content,
                mode=request.mode,
                frame_index=request.frame_index,
            )
        except ValueError as exc:
            raise UserError("discord.custom_emoji_decode_failed") from exc
        return DiscordViewCustomEmojiResponse(
            emoji_index=selected.index,
            emoji_id=selected.emoji_id,
            name=selected.name,
            animated=selected.animated,
            occurrences=selected.occurrences,
            preview_kind=media.preview_kind,
            frame_index=media.frame_index,
            frame_count=media.frame_count,
            duration_ms=media.duration_ms,
            content_type=media.content_type,
            image_data_url=(
                f"data:{media.content_type};base64,"
                + base64.b64encode(media.content).decode("ascii")
            ),
        )

    async def view_sticker(
        request: DiscordViewStickerRequest,
        context: InvocationContext,
    ) -> DiscordViewStickerResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        if not 0 <= request.sticker_index < len(message.stickers):
            raise UserError("discord.sticker_index_invalid")
        selected = message.stickers[request.sticker_index]
        if request.frame_index < 0:
            raise UserError("discord.sticker_frame_invalid")
        if request.mode != "frame" and request.frame_index != 0:
            raise UserError("discord.sticker_frame_mode_required")
        if request.mode == "animation" and selected.format is discord.StickerFormatType.png:
            raise UserError("discord.sticker_not_animated")
        if selected.format is discord.StickerFormatType.lottie:
            if request.mode != "preview":
                raise UserError("discord.sticker_lottie_animation_unavailable")
            sticker_url = (
                f"https://media.discordapp.net/stickers/{selected.id}.png?size=160&quality=lossless"
            )
        else:
            sticker_url = selected.url
        try:
            content = await client.http.get_from_cdn(str(sticker_url))
        except (discord.NotFound, discord.Forbidden) as exc:
            raise UserError("discord.sticker_unavailable") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.sticker_failed") from exc
        if not content or len(content) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            raise UserError("discord.sticker_invalid")
        try:
            media = await asyncio.to_thread(
                _prepare_discord_animated_media,
                content,
                mode=request.mode,
                frame_index=request.frame_index,
            )
        except ValueError as exc:
            code = str(exc).replace("custom_emoji", "sticker")
            raise UserError(code) from exc
        return DiscordViewStickerResponse(
            sticker_index=request.sticker_index,
            sticker_id=str(selected.id),
            name=selected.name,
            format=selected.format.name,
            animated=selected.format is not discord.StickerFormatType.png,
            preview_kind=media.preview_kind,
            frame_index=media.frame_index,
            frame_count=media.frame_count,
            duration_ms=media.duration_ms,
            content_type=media.content_type,
            image_data_url=(
                f"data:{media.content_type};base64,"
                + base64.b64encode(media.content).decode("ascii")
            ),
        )

    async def analyze_attachment(
        request: DiscordAnalyzeAttachmentRequest,
        context: InvocationContext,
    ) -> SyntheticMediaAnalyzeResponse:
        _, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        if attachment.size > runtime.settings.hive_max_media_bytes:
            raise UserError("moderation.media_too_large")
        try:
            content = await read_attachment_bytes(attachment)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        if len(content) > runtime.settings.hive_max_media_bytes:
            raise UserError("moderation.media_too_large")
        return cast(
            SyntheticMediaAnalyzeResponse,
            await runtime.registry.invoke(
                "moderation.detect_synthetic_media",
                SyntheticMediaAnalyzeRequest(
                    filename=attachment.filename,
                    content_type=attachment.content_type,
                    content=content,
                ),
                context,
            ),
        )

    async def import_attachment(
        request: DiscordImportAttachmentRequest,
        context: InvocationContext,
    ) -> WorkspaceFileRecord:
        if runtime.files is None:
            raise UserError("files.disabled")
        if context.workspace_id is None:
            raise UserError("files.workspace_required")
        message, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        if attachment.size > runtime.files.max_file_bytes:
            raise UserError("files.file_too_large")
        try:
            content = await read_attachment_bytes(attachment)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        if len(content) > runtime.files.max_file_bytes:
            raise UserError("files.file_too_large")
        destination = request.destination_path or (
            f"attachments/{request.message_id}/"
            f"{_workspace_attachment_name(attachment)}"
        )
        source_guild = message.guild
        source_channel = message.channel
        if source_guild is None or not isinstance(
            source_channel,
            (
                discord.TextChannel,
                discord.Thread,
                discord.ForumChannel,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            raise UserError("discord.attachment_unavailable")
        return await asyncio.to_thread(
            runtime.files.import_bytes,
            file_workspace_id(context),
            destination,
            content,
            provenance=file_provenance(
                context,
                origin_guild_id=str(source_guild.id),
                origin_channel_id=str(source_channel.id),
                origin_message_id=str(message.id),
                origin_visibility=_channel_visibility(
                    source_guild,
                    source_channel,
                ),
            ),
        )

    async def view_image_attachment(
        request: DiscordViewImageAttachmentRequest,
        context: InvocationContext,
    ) -> DiscordViewImageAttachmentResponse:
        _, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        if attachment.size > 8 * 1024 * 1024:
            raise UserError("discord.image_attachment_too_large")
        try:
            content = await read_attachment_bytes(attachment)
        except discord.DiscordException as exc:
            raise UserError("discord.attachment_unavailable") from exc
        if len(content) > 8 * 1024 * 1024:
            raise UserError("discord.image_attachment_too_large")
        media_type = _image_media_type(content)
        if media_type is None:
            raise UserError("discord.attachment_not_supported_image")
        encoded = base64.b64encode(content).decode("ascii")
        return DiscordViewImageAttachmentResponse(
            filename=attachment.filename,
            content_type=media_type,
            size_bytes=len(content),
            image_data_url=f"data:{media_type};base64,{encoded}",
        )

    async def send_message(
        request: DiscordSendMessageRequest,
        context: InvocationContext,
    ) -> DiscordSendMessageResponse:
        guild, channel, actor, bot = await _write_message_channel(
            client,
            context,
            request.channel_id,
            guild_id=request.guild_id,
        )
        for member in (actor, bot):
            _require_channel_permissions(channel, member, "send_messages")
        if not 1 <= len(request.content) <= 2_000:
            raise UserError("discord.message_length_invalid")
        message = await channel.send(
            request.content,
            nonce=_discord_write_nonce(context, "message"),
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )
        return DiscordSendMessageResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            guild_id=str(guild.id),
        )

    async def send_embed(
        request: DiscordSendEmbedRequest,
        context: InvocationContext,
    ) -> DiscordSendMessageResponse:
        guild, channel, actor, bot = await _write_message_channel(
            client,
            context,
            request.channel_id,
            guild_id=request.guild_id,
        )
        for member in (actor, bot):
            _require_channel_permissions(channel, member, "send_messages")
            _require_channel_permissions(channel, member, "embed_links")
        title = request.title.strip()
        description = request.description.strip()
        if not 1 <= len(title) <= 256:
            raise UserError("discord.embed_title_invalid")
        if len(description) > 4_096:
            raise UserError("discord.embed_description_invalid")
        if len(request.fields) > 10:
            raise UserError("discord.embed_fields_invalid")
        fields: list[EmbedField] = []
        total_characters = len(title) + len(description)
        for field in request.fields:
            name = field.name.strip()
            value = field.value.strip()
            if not 1 <= len(name) <= 256 or not 1 <= len(value) <= 1_024:
                raise UserError("discord.embed_field_invalid")
            total_characters += len(name) + len(value)
            fields.append(EmbedField(name=name, value=value, inline=field.inline))
        if total_characters > 6_000:
            raise UserError("discord.embed_length_invalid")
        try:
            tone = EmbedTone(request.tone)
        except ValueError as exc:
            raise UserError("discord.embed_tone_invalid") from exc
        reply = (
            await _fetch_message_for_write(channel, request.reply_to_message_id)
            if request.reply_to_message_id is not None
            else None
        )
        send_arguments: dict[str, Any] = {
            "embed": agent_embed(
                title,
                description=description or None,
                fields=tuple(fields),
                tone=tone,
            ),
            "nonce": _discord_write_nonce(context, "embed"),
            "allowed_mentions": discord.AllowedMentions.none(),
        }
        if reply is not None:
            send_arguments["reference"] = reply
            send_arguments["mention_author"] = False
        if request.silent:
            send_arguments["silent"] = True
        message = await channel.send(**send_arguments)
        return DiscordSendMessageResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            guild_id=str(guild.id),
        )

    async def reply_message(
        request: DiscordReplyMessageRequest,
        context: InvocationContext,
    ) -> DiscordSendMessageResponse:
        guild, channel, actor, bot = await _write_message_channel(
            client,
            context,
            request.channel_id,
            guild_id=request.guild_id,
        )
        _require_channel_permissions(channel, actor, "send_messages")
        _require_channel_permissions(channel, bot, "send_messages")
        if not 1 <= len(request.content) <= 2_000:
            raise UserError("discord.message_length_invalid")
        message = await _fetch_message_for_write(channel, request.message_id)
        reply = await message.reply(
            request.content,
            nonce=_discord_write_nonce(context, "reply"),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
            silent=request.silent,
        )
        return DiscordSendMessageResponse(
            str(reply.id),
            str(channel.id),
            str(guild.id),
        )

    async def edit_own_message(
        request: DiscordMessageWriteRequest,
        context: InvocationContext,
    ) -> DiscordMessageWriteResponse:
        guild, channel, _actor, bot = await _write_message_channel(
            client, context, request.channel_id
        )
        del guild
        if not 1 <= len(request.content) <= 2_000:
            raise UserError("discord.message_length_invalid")
        message = await _fetch_message_for_write(channel, request.message_id)
        if message.author.id != bot.id:
            raise UserError("discord.message_not_owned")
        if message.content == request.content:
            return DiscordMessageWriteResponse(
                request.channel_id, request.message_id, changed=False
            )
        await message.edit(
            content=request.content,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress=True,
        )
        return DiscordMessageWriteResponse(request.channel_id, request.message_id)

    async def set_message_pin(
        request: DiscordMessageWriteRequest,
        context: InvocationContext,
        *,
        pinned: bool,
    ) -> DiscordMessageWriteResponse:
        _guild_value, channel, actor, bot = await _write_message_channel(
            client, context, request.channel_id
        )
        _require_channel_permissions(channel, actor, "pin_messages")
        _require_channel_permissions(channel, bot, "pin_messages")
        message = await _fetch_message_for_write(channel, request.message_id)
        if bool(message.pinned) == pinned:
            return DiscordMessageWriteResponse(
                request.channel_id, request.message_id, changed=False
            )
        if (
            request.expected_pinned is not None
            and bool(message.pinned) != request.expected_pinned
        ):
            raise UserError("action.undo_conflict")
        operation = message.pin if pinned else message.unpin
        await operation(reason=_audit_reason(request.reason, context))
        return DiscordMessageWriteResponse(request.channel_id, request.message_id)

    async def pin_message(
        request: DiscordMessageWriteRequest,
        context: InvocationContext,
    ) -> DiscordMessageWriteResponse:
        return await set_message_pin(request, context, pinned=True)

    async def unpin_message(
        request: DiscordMessageWriteRequest,
        context: InvocationContext,
    ) -> DiscordMessageWriteResponse:
        return await set_message_pin(request, context, pinned=False)

    async def create_thread(
        request: DiscordThreadCreateRequest,
        context: InvocationContext,
    ) -> DiscordThreadResponse:
        _guild_value, channel, actor, bot = await _write_message_channel(
            client, context, request.channel_id
        )
        if not isinstance(channel, discord.TextChannel):
            raise UserError("discord.thread_parent_invalid")
        for member in (actor, bot):
            _require_channel_permissions(channel, member, "create_public_threads")
            _require_channel_permissions(channel, member, "send_messages")
        message = (
            await _fetch_message_for_write(channel, request.message_id)
            if request.message_id is not None
            else None
        )
        thread = await channel.create_thread(
            name=_bounded_name(request.name, "discord.thread_name_invalid"),
            message=message,
            reason=_audit_reason(request.reason, context),
        )
        return DiscordThreadResponse(
            channel_id=str(channel.id),
            thread_id=str(thread.id),
            name=thread.name,
            archived=bool(thread.archived),
            undo_fingerprint=_thread_undo_fingerprint(thread),
        )

    async def update_thread(
        request: DiscordThreadUpdateRequest,
        context: InvocationContext,
    ) -> DiscordThreadResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        thread = guild.get_thread(_snowflake(request.thread_id, "channel"))
        if thread is None:
            raise UserError("discord.thread_not_found")
        _enforce_information_flow_to_destination(client, context, guild, thread)
        actor, bot = await _write_members(guild, context)
        for member in (actor, bot):
            _require_channel_permissions(thread, member, "manage_threads")
        old_name = thread.name
        old_archived = bool(thread.archived)
        name = (
            _bounded_name(request.name, "discord.thread_name_invalid")
            if request.name is not None
            else old_name
        )
        archived = old_archived if request.archived is None else request.archived
        if name == old_name and archived == old_archived:
            return DiscordThreadResponse(
                channel_id=str(thread.parent_id),
                thread_id=str(thread.id),
                name=old_name,
                archived=old_archived,
                undo_fingerprint=_thread_undo_fingerprint(thread),
                old_name=old_name,
                old_archived=old_archived,
                changed=False,
            )
        if (
            request.expected_name is not None
            and old_name != request.expected_name
        ) or (
            request.expected_archived is not None
            and old_archived != request.expected_archived
        ) or (
            request.expected_undo_fingerprint is not None
            and _thread_undo_fingerprint(thread)
            != request.expected_undo_fingerprint
        ):
            raise UserError("action.undo_conflict")
        updated = await thread.edit(
            name=name,
            archived=archived,
            reason=_audit_reason(request.reason, context),
        )
        return DiscordThreadResponse(
            channel_id=str(updated.parent_id),
            thread_id=str(updated.id),
            name=updated.name,
            archived=bool(updated.archived),
            undo_fingerprint=_thread_undo_fingerprint(updated),
            old_name=old_name,
            old_archived=old_archived,
        )

    async def set_thread_member(
        request: DiscordThreadMemberRequest,
        context: InvocationContext,
        *,
        present: bool,
    ) -> DiscordThreadMemberResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        thread = guild.get_thread(_snowflake(request.thread_id, "channel"))
        if thread is None:
            raise UserError("discord.thread_not_found")
        _enforce_information_flow_to_destination(client, context, guild, thread)
        actor, bot = await _write_members(guild, context)
        for member in (actor, bot):
            _require_channel_permissions(thread, member, "manage_threads")
        target = await _guild_member(guild, request.user_id)
        existing = any(item.id == target.id for item in thread.members)
        if existing == present:
            return DiscordThreadMemberResponse(
                request.thread_id, request.user_id, present, changed=False
            )
        if (
            present
            and isinstance(thread, discord.Thread)
            and thread.type is discord.ChannelType.private_thread
            and context.information_flow_mode != "disabled"
        ):
            _handle_information_flow_violations(
                context,
                guild,
                thread,
                [(str(guild.id), str(thread.id), "broader")],
            )
        if (
            request.expected_present is not None
            and existing != request.expected_present
        ):
            raise UserError("action.undo_conflict")
        if present:
            await thread.add_user(target)
        else:
            await thread.remove_user(target)
        return DiscordThreadMemberResponse(request.thread_id, request.user_id, present)

    async def add_thread_member(
        request: DiscordThreadMemberRequest,
        context: InvocationContext,
    ) -> DiscordThreadMemberResponse:
        return await set_thread_member(request, context, present=True)

    async def remove_thread_member(
        request: DiscordThreadMemberRequest,
        context: InvocationContext,
    ) -> DiscordThreadMemberResponse:
        return await set_thread_member(request, context, present=False)

    async def create_forum_post(
        request: DiscordForumPostRequest,
        context: InvocationContext,
    ) -> DiscordForumPostResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        forum = guild.get_channel(_snowflake(request.forum_id, "channel"))
        if not isinstance(forum, discord.ForumChannel):
            raise UserError("discord.forum_channel_required")
        _enforce_information_flow_to_destination(client, context, guild, forum)
        actor, bot = await _write_members(guild, context)
        for member in (actor, bot):
            _require_channel_permissions(forum, member, "send_messages")
            _require_channel_permissions(forum, member, "create_public_threads")
        created = await forum.create_thread(
            name=_bounded_name(request.title, "discord.thread_name_invalid"),
            content=request.content,
            reason=_audit_reason(request.reason, context),
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return DiscordForumPostResponse(
            channel_id=str(created.thread.id),
            message_id=str(created.message.id),
            thread_id=str(created.thread.id),
            name=created.thread.name,
            archived=bool(created.thread.archived),
            undo_fingerprint=_thread_undo_fingerprint(created.thread),
        )

    async def create_role(
        request: DiscordRoleCreateRequest,
        context: InvocationContext,
    ) -> DiscordRoleResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        _require_guild_permission(actor, "manage_roles")
        _require_guild_permission(bot, "manage_roles")
        if not 0 <= request.colour <= 0xFFFFFF:
            raise UserError("discord.role_colour_invalid")
        role = await guild.create_role(
            name=_bounded_name(request.name, "discord.role_name_invalid"),
            colour=discord.Colour(request.colour),
            hoist=request.hoist,
            mentionable=request.mentionable,
            reason=_audit_reason(request.reason, context),
        )
        return DiscordRoleResponse(
            str(role.id),
            role.name,
            _role_undo_fingerprint(role),
        )

    async def delete_created_role(
        request: DiscordCreatedRoleDeleteRequest,
        context: InvocationContext,
    ) -> DiscordCreatedEntityDeleteResponse:
        guild = _guild(client, context)
        actor, bot = await _write_members(guild, context)
        role = guild.get_role(_snowflake(request.role_id, "role"))
        if role is None:
            return DiscordCreatedEntityDeleteResponse(request.role_id, True)
        if (
            request.undo_fingerprint is not None
            and _role_undo_fingerprint(role) != request.undo_fingerprint
        ):
            raise UserError("action.undo_conflict")
        if not (
            guild.chunked
            or (
                guild.member_count is not None
                and len(guild.members) >= guild.member_count
            )
        ):
            raise UserError("action.undo_target_state_uncertain")
        if role.members:
            raise UserError("action.undo_target_in_use")
        for member in (actor, bot):
            _require_guild_permission(member, "manage_roles")
            _require_role_above(member, role)
        if await _role_has_channel_overwrite_reference(guild, role):
            raise UserError("action.undo_target_in_use")
        await role.delete(reason=_audit_reason("Undo created role", context))
        return DiscordCreatedEntityDeleteResponse(request.role_id, True)

    async def set_member_role(
        request: DiscordRoleMemberRequest,
        context: InvocationContext,
        *,
        assigned: bool,
    ) -> DiscordRoleMemberResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        target = await _guild_member(guild, request.user_id)
        role = guild.get_role(_snowflake(request.role_id, "role"))
        if role is None or role.is_default() or role.managed:
            raise UserError("discord.role_unassignable")
        _require_guild_permission(actor, "manage_roles")
        _require_guild_permission(bot, "manage_roles")
        _require_role_above(actor, role)
        _require_role_above(bot, role)
        _require_member_below(actor, target, guild)
        _require_member_below(bot, target, guild)
        existing = role in target.roles
        if existing == assigned:
            return DiscordRoleMemberResponse(
                request.user_id, request.role_id, assigned, changed=False
            )
        if (
            request.expected_assigned is not None
            and existing != request.expected_assigned
        ):
            raise UserError("action.undo_conflict")
        operation = target.add_roles if assigned else target.remove_roles
        await operation(role, reason=_audit_reason(request.reason, context))
        return DiscordRoleMemberResponse(request.user_id, request.role_id, assigned)

    async def assign_role(
        request: DiscordRoleMemberRequest,
        context: InvocationContext,
    ) -> DiscordRoleMemberResponse:
        return await set_member_role(request, context, assigned=True)

    async def remove_role(
        request: DiscordRoleMemberRequest,
        context: InvocationContext,
    ) -> DiscordRoleMemberResponse:
        return await set_member_role(request, context, assigned=False)

    async def update_channel_settings(
        request: DiscordChannelSettingRequest,
        context: InvocationContext,
    ) -> DiscordChannelSettingResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        channel = guild.get_channel(_snowflake(request.channel_id, "channel"))
        if not isinstance(channel, discord.TextChannel):
            raise UserError("discord.text_destination_invalid")
        _enforce_information_flow_to_destination(client, context, guild, channel)
        actor, bot = await _write_members(guild, context)
        _require_channel_permissions(channel, actor, "manage_channels")
        _require_channel_permissions(channel, bot, "manage_channels")
        old_topic = channel.topic
        old_slowmode = channel.slowmode_delay
        topic = (
            old_topic
            if request.topic == _UNCHANGED_CHANNEL_TOPIC
            else request.topic
        )
        slowmode = (
            old_slowmode
            if request.slowmode_seconds is None
            else request.slowmode_seconds
        )
        if len(topic or "") > 1_024 or not 0 <= slowmode <= 21_600:
            raise UserError("discord.channel_setting_invalid")
        changed = topic != old_topic or slowmode != old_slowmode
        if not changed:
            return DiscordChannelSettingResponse(
                request.channel_id,
                topic,
                slowmode,
                old_topic,
                old_slowmode,
                False,
            )
        if (
            request.expected_topic != _NO_EXPECTED_STRING_STATE
            and old_topic != request.expected_topic
        ) or (
            request.expected_slowmode_seconds is not None
            and old_slowmode != request.expected_slowmode_seconds
        ):
            raise UserError("action.undo_conflict")
        edit_arguments: dict[str, object] = {
            "slowmode_delay": slowmode,
            "reason": _audit_reason(request.reason, context),
        }
        edit_arguments["topic"] = topic
        await channel.edit(**edit_arguments)
        return DiscordChannelSettingResponse(
            request.channel_id,
            topic,
            slowmode,
            old_topic,
            old_slowmode,
            changed,
        )

    async def create_channel(
        request: DiscordChannelCreateRequest,
        context: InvocationContext,
    ) -> DiscordChannelCreateResponse:
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        _enforce_information_flow_to_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        _require_guild_permission(actor, "manage_channels")
        _require_guild_permission(bot, "manage_channels")
        channel_name = _bounded_name(request.name, "discord.channel_name_invalid")
        audit_reason = _audit_reason(request.reason, context)
        channel = (
            await guild.create_text_channel(
                channel_name,
                reason=audit_reason,
            )
            if request.topic is None
            else await guild.create_text_channel(
                channel_name,
                topic=request.topic,
                reason=audit_reason,
            )
        )
        return DiscordChannelCreateResponse(
            str(channel.id),
            channel.name,
            _channel_undo_fingerprint(channel),
        )

    async def delete_created_channel(
        request: DiscordCreatedChannelDeleteRequest,
        context: InvocationContext,
    ) -> DiscordCreatedEntityDeleteResponse:
        guild = _guild(client, context)
        actor, bot = await _write_members(guild, context)
        channel = guild.get_channel(_snowflake(request.channel_id, "channel"))
        if channel is None:
            return DiscordCreatedEntityDeleteResponse(request.channel_id, True)
        if (
            request.undo_fingerprint is not None
            and (
                not isinstance(channel, discord.TextChannel)
                or _channel_undo_fingerprint(channel) != request.undo_fingerprint
            )
        ):
            raise UserError("action.undo_conflict")
        if isinstance(channel, discord.TextChannel):
            async for _message in channel.history(limit=1):
                raise UserError("action.undo_target_in_use")
        for member in (actor, bot):
            _require_guild_permission(member, "manage_channels")
        await channel.delete(reason=_audit_reason("Undo created channel", context))
        return DiscordCreatedEntityDeleteResponse(request.channel_id, True)

    async def set_timeout(
        request: DiscordTimeoutRequest,
        context: InvocationContext,
    ) -> DiscordTimeoutResponse:
        _require_moderation_reason(request.reason)
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        target = await _guild_member(guild, request.user_id)
        for member in (actor, bot):
            _require_guild_permission(member, "moderate_members")
            _require_member_below(member, target, guild)
        until = _timeout_datetime(request.until_iso)
        previous = target.timed_out_until
        if previous == until:
            return DiscordTimeoutResponse(
                request.user_id,
                request.until_iso,
                previous.isoformat() if previous is not None else None,
                changed=False,
            )
        if request.expected_until_iso != _NO_EXPECTED_STRING_STATE:
            expected_until = _timeout_state_datetime(request.expected_until_iso)
            normalized_previous = (
                previous.astimezone(UTC) if previous is not None else None
            )
            if normalized_previous != expected_until:
                raise UserError("action.undo_conflict")
        await target.timeout(until, reason=_audit_reason(request.reason, context))
        return DiscordTimeoutResponse(
            request.user_id,
            until.isoformat() if until is not None else None,
            previous.isoformat() if previous is not None else None,
        )

    async def delete_message(
        request: DiscordMessageWriteRequest,
        context: InvocationContext,
    ) -> DiscordMessageWriteResponse:
        _require_moderation_reason(request.reason)
        _guild_value, channel, actor, bot = await _write_message_channel(
            client, context, request.channel_id
        )
        _require_channel_permissions(channel, actor, "manage_messages")
        _require_channel_permissions(channel, bot, "manage_messages")
        message = await _fetch_message_for_write(channel, request.message_id)
        await message.delete()
        return DiscordMessageWriteResponse(request.channel_id, request.message_id)

    async def bulk_delete_messages(
        request: DiscordBulkDeleteRequest,
        context: InvocationContext,
    ) -> DiscordBulkDeleteResponse:
        _require_moderation_reason(request.reason)
        _guild_value, channel, actor, bot = await _write_message_channel(
            client, context, request.channel_id
        )
        _require_channel_permissions(channel, actor, "manage_messages")
        _require_channel_permissions(channel, bot, "manage_messages")
        ids = tuple(dict.fromkeys(request.message_ids))
        if not 2 <= len(ids) <= 100:
            raise UserError("discord.bulk_delete_limit_invalid")
        messages = [
            await _fetch_message_for_write(channel, message_id)
            for message_id in ids
        ]
        oldest_bulk_delete_time = datetime.now(UTC) - timedelta(days=14)
        if any(message.created_at <= oldest_bulk_delete_time for message in messages):
            raise UserError("discord.bulk_delete_message_too_old")
        await channel.delete_messages(
            messages,
            reason=_audit_reason(request.reason, context),
        )
        return DiscordBulkDeleteResponse(request.channel_id, ids)

    async def moderate_member(
        request: DiscordMemberModerationRequest,
        context: InvocationContext,
        *,
        action: Literal["kick", "ban", "unban"],
    ) -> DiscordMemberModerationResponse:
        _require_moderation_reason(request.reason)
        guild = _guild(client, context)
        _assert_origin_guild(context, guild)
        actor, bot = await _write_members(guild, context)
        user_id = _snowflake(request.user_id, "user")
        if action == "unban":
            for member in (actor, bot):
                _require_guild_permission(member, "ban_members")
            try:
                await guild.unban(
                    discord.Object(user_id),
                    reason=_audit_reason(request.reason, context),
                )
            except discord.NotFound:
                return DiscordMemberModerationResponse(
                    request.user_id,
                    action,
                    changed=False,
                )
        else:
            target = await _guild_member(guild, request.user_id)
            permission = "kick_members" if action == "kick" else "ban_members"
            for member in (actor, bot):
                _require_guild_permission(member, permission)
                _require_member_below(member, target, guild)
            if action == "kick":
                await target.kick(reason=_audit_reason(request.reason, context))
            else:
                await guild.ban(
                    target,
                    reason=_audit_reason(request.reason, context),
                    delete_message_seconds=0,
                )
        return DiscordMemberModerationResponse(request.user_id, action)

    async def kick_member(
        request: DiscordMemberModerationRequest,
        context: InvocationContext,
    ) -> DiscordMemberModerationResponse:
        return await moderate_member(request, context, action="kick")

    async def ban_member(
        request: DiscordMemberModerationRequest,
        context: InvocationContext,
    ) -> DiscordMemberModerationResponse:
        return await moderate_member(request, context, action="ban")

    async def unban_member(
        request: DiscordMemberModerationRequest,
        context: InvocationContext,
    ) -> DiscordMemberModerationResponse:
        return await moderate_member(request, context, action="unban")

    async def delete_own_message(
        request: DiscordDeleteOwnMessageRequest,
        context: InvocationContext,
    ) -> DiscordDeleteOwnMessageResponse:
        guild = _requested_guild(client, context, request.guild_id)
        channel = _message_channel(guild, request.channel_id)
        bot_member = guild.me
        if bot_member is None:
            raise UserError("discord.message_delete_forbidden")
        actor = await _actor_member(guild, context)
        if not _can_read_messages(channel, actor) or not _can_read_private_thread(
            channel,
            actor,
        ):
            raise UserError("discord.message_delete_forbidden")
        if not _can_read_messages(channel, bot_member) or not _can_read_private_thread(
            channel,
            bot_member,
        ):
            raise UserError("discord.message_delete_forbidden")
        try:
            message = await channel.fetch_message(
                _snowflake(request.message_id, "message")
            )
        except discord.NotFound:
            return DiscordDeleteOwnMessageResponse(
                message_id=request.message_id,
                channel_id=request.channel_id,
                deleted=True,
                guild_id=str(guild.id),
            )
        except discord.Forbidden as exc:
            raise UserError("discord.message_delete_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_delete_failed") from exc
        if message.author.id != bot_member.id:
            raise UserError("discord.message_not_owned")
        try:
            await message.delete()
        except discord.NotFound:
            pass
        except discord.Forbidden as exc:
            raise UserError("discord.message_delete_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_delete_failed") from exc
        return DiscordDeleteOwnMessageResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            deleted=True,
            guild_id=str(guild.id),
        )

    async def delete_own_messages(
        request: DiscordDeleteOwnMessagesRequest,
        context: InvocationContext,
    ) -> DiscordDeleteOwnMessagesResponse:
        guild = _requested_guild(client, context, request.guild_id)
        channel = _message_channel(guild, request.channel_id)
        bot_member = guild.me
        if bot_member is None:
            raise UserError("discord.message_delete_forbidden")
        actor = await _actor_member(guild, context)
        if not _can_read_messages(channel, actor) or not _can_read_private_thread(
            channel,
            actor,
        ):
            raise UserError("discord.message_delete_forbidden")
        if not _can_read_messages(
            channel,
            bot_member,
        ) or not _can_read_private_thread(channel, bot_member):
            raise UserError("discord.message_delete_forbidden")

        message_ids = tuple(request.message_ids.split(","))
        if (
            not 1 <= len(message_ids) <= 100
            or len(set(message_ids)) != len(message_ids)
        ):
            raise UserError("action.undo_target_state_uncertain")
        numeric_ids = tuple(
            _snowflake(message_id, "message") for message_id in message_ids
        )
        messages: list[discord.Message] = []
        for message_id in numeric_ids:
            try:
                message = await channel.fetch_message(message_id)
            except discord.NotFound:
                continue
            except discord.Forbidden as exc:
                raise UserError("discord.message_delete_forbidden") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.message_delete_failed") from exc
            if message.author.id != bot_member.id:
                raise UserError("discord.message_not_owned")
            messages.append(message)

        for message in messages:
            try:
                await message.delete()
            except discord.NotFound:
                continue
            except discord.Forbidden as exc:
                raise UserError("discord.message_delete_forbidden") from exc
            except discord.DiscordException as exc:
                raise UserError("discord.message_delete_failed") from exc
        return DiscordDeleteOwnMessagesResponse(
            channel_id=str(channel.id),
            deleted_message_ids=message_ids,
            guild_id=str(guild.id),
        )

    async def add_reaction(
        request: DiscordReactionRequest,
        context: InvocationContext,
    ) -> DiscordReactionResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        emoji = _reaction_emoji(request.emoji)
        already_reacted = _bot_has_reaction(message, emoji)
        try:
            await message.add_reaction(emoji)
        except TypeError as exc:
            raise UserError("discord.reaction_emoji_invalid") from exc
        except discord.Forbidden as exc:
            raise UserError("discord.reaction_forbidden") from exc
        except discord.NotFound as exc:
            raise UserError("discord.reaction_target_not_found") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.reaction_failed") from exc
        return DiscordReactionResponse(
            channel_id=str(message.channel.id),
            message_id=str(message.id),
            emoji=emoji,
            reacted=True,
            changed=not already_reacted,
        )

    async def remove_own_reaction(
        request: DiscordReactionRequest,
        context: InvocationContext,
    ) -> DiscordReactionResponse:
        guild = _guild(client, context)
        _, message = await _fetch_readable_message(
            guild,
            channel_id=request.channel_id,
            message_id=request.message_id,
            context=context,
        )
        bot_member = guild.me
        if bot_member is None:
            raise UserError("discord.reaction_unavailable")
        emoji = _reaction_emoji(request.emoji)
        already_reacted = _bot_has_reaction(message, emoji)
        try:
            await message.remove_reaction(emoji, bot_member)
        except TypeError as exc:
            raise UserError("discord.reaction_emoji_invalid") from exc
        except discord.Forbidden as exc:
            raise UserError("discord.reaction_forbidden") from exc
        except discord.NotFound as exc:
            raise UserError("discord.reaction_target_not_found") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.reaction_failed") from exc
        return DiscordReactionResponse(
            channel_id=str(message.channel.id),
            message_id=str(message.id),
            emoji=emoji,
            reacted=False,
            changed=already_reacted,
        )

    async def send_files(
        request: DiscordSendFilesRequest,
        context: InvocationContext,
        *,
        _single_file: bool = False,
    ) -> DiscordSendFilesResponse:
        if runtime.files is None:
            raise UserError("files.disabled")
        if context.workspace_id is None:
            raise UserError("files.workspace_required")
        if not 1 <= len(request.attachments) <= 10:
            raise UserError("discord.file_count_invalid")
        if len(request.caption) > 2_000:
            raise UserError("discord.file_caption_too_long")
        if any(len(item.description) > 1_024 for item in request.attachments):
            raise UserError("discord.file_description_too_long")
        guild, channel, _actor, _bot = await _write_message_channel(
            client,
            context,
            request.channel_id,
            guild_id=request.guild_id,
            required_permissions=("send_messages", "attach_files"),
        )
        snapshots = await asyncio.gather(
            *(
                asyncio.to_thread(
                    runtime.files.snapshot_for_actor_delivery_with_provenance,
                    file_workspace_id(context),
                    context.actor_id,
                    item.path,
                )
                for item in request.attachments
            )
        )
        _enforce_file_provenance_to_destination(
            client,
            context,
            guild,
            channel,
            tuple(provenance for _filename, _content, provenance in snapshots),
        )
        sizes = tuple(len(content) for _filename, content, _provenance in snapshots)
        if any(size > guild.filesize_limit for size in sizes):
            raise UserError("discord.file_too_large")
        files = [
            discord.File(
                io.BytesIO(content),
                filename=filename,
                spoiler=item.spoiler,
                description=item.description or None,
            )
            for item, (filename, content, _provenance) in zip(
                request.attachments,
                snapshots,
                strict=True,
            )
        ]
        filenames = tuple(file.filename for file in files)
        reply = (
            await _fetch_message_for_write(channel, request.reply_to_message_id)
            if request.reply_to_message_id is not None
            else None
        )
        try:
            send_arguments: dict[str, Any] = {
                "allowed_mentions": discord.AllowedMentions.none(),
                "nonce": _discord_write_nonce(context, "attachments"),
                "suppress_embeds": True,
            }
            if _single_file:
                send_arguments["file"] = files[0]
            else:
                send_arguments["files"] = files
            if reply is not None:
                send_arguments["reference"] = reply
                send_arguments["mention_author"] = False
            if request.silent:
                send_arguments["silent"] = True
            message = await channel.send(
                request.caption or None,
                **send_arguments,
            )
        except discord.Forbidden as exc:
            raise UserError("discord.file_send_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.file_send_failed") from exc
        finally:
            for file in files:
                file.close()
        return DiscordSendFilesResponse(
            message_id=str(message.id),
            channel_id=str(channel.id),
            filenames=filenames,
            size_bytes=sizes,
            total_size_bytes=sum(sizes),
            guild_id=str(guild.id),
        )

    async def send_file(
        request: DiscordSendFileRequest,
        context: InvocationContext,
    ) -> DiscordSendFileResponse:
        response = await send_files(
            DiscordSendFilesRequest(
                channel_id=request.channel_id,
                attachments=(
                    DiscordFileAttachmentRequest(
                        path=request.path,
                        description=request.description,
                        spoiler=request.spoiler,
                    ),
                ),
                caption=request.caption,
                guild_id=request.guild_id,
                reply_to_message_id=request.reply_to_message_id,
                silent=request.silent,
                purpose=request.purpose,
            ),
            context,
            _single_file=True,
        )
        return DiscordSendFileResponse(
            message_id=response.message_id,
            channel_id=response.channel_id,
            filename=response.filenames[0],
            size_bytes=response.size_bytes[0],
            guild_id=response.guild_id,
        )

    async def create_poll(
        request: DiscordPollRequest,
        context: InvocationContext,
    ) -> DiscordPollResponse:
        guild = _guild(client, context)
        _assert_agent_update_scope(context, request.channel_id)
        channel = _text_channel(guild, request.channel_id)
        _enforce_information_flow_to_destination(client, context, guild, channel)
        actor = await _actor_member(guild, context)
        bot_member = guild.me
        actor_permissions = channel.permissions_for(actor)
        bot_permissions = (
            channel.permissions_for(bot_member)
            if bot_member is not None
            else None
        )
        if (
            bot_permissions is None
            or (
                not _permission_enabled(actor_permissions, "administrator")
                and (
                    not _permission_enabled(actor_permissions, "view_channel")
                    or not _permission_enabled(actor_permissions, "send_messages")
                    or not _permission_enabled(actor_permissions, "create_polls")
                )
            )
            or (
                not _permission_enabled(bot_permissions, "administrator")
                and (
                    not _permission_enabled(bot_permissions, "view_channel")
                    or not _permission_enabled(bot_permissions, "send_messages")
                    or not _permission_enabled(bot_permissions, "create_polls")
                )
            )
        ):
            raise UserError("discord.poll_forbidden")
        question = request.question.strip()
        options = tuple(option.strip() for option in request.options if option.strip())
        if not 1 <= len(question) <= 300:
            raise UserError("discord.poll_question_invalid")
        if not 2 <= len(options) <= 10:
            raise UserError("discord.poll_option_count_invalid")
        if any(len(option) > 55 for option in options):
            raise UserError("discord.poll_option_too_long")
        if not 1 <= request.duration_hours <= 168:
            raise UserError("discord.poll_duration_invalid")
        poll = discord.Poll(
            question,
            duration=timedelta(hours=request.duration_hours),
            multiple=request.multiple,
        )
        for option in options:
            poll.add_answer(text=option)
        try:
            message = await channel.send(
                poll=poll,
                allowed_mentions=discord.AllowedMentions.none(),
            )
        except discord.Forbidden as exc:
            raise UserError("discord.poll_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.poll_failed") from exc
        return DiscordPollResponse(message_id=str(message.id), channel_id=str(channel.id))

    async def connect_voice(
        request: DiscordConnectVoiceRequest,
        context: InvocationContext,
    ) -> DiscordConnectVoiceResponse:
        guild = _guild(client, context)
        channel = guild.get_channel(_snowflake(request.channel_id, "voice channel"))
        if not isinstance(channel, (discord.VoiceChannel, discord.StageChannel)):
            raise UserError("discord.voice_channel_unavailable")
        if context.transport == "agent":
            actor = await _actor_member(guild, context)
            actor_channel = _member_voice_channel(actor)
            if actor_channel is None or actor_channel.id != channel.id:
                raise UserError("audio.same_voice_required")
        workspace_id = str(guild.id)
        session = runtime.audio.get_or_create(
            workspace_id,
            lambda: DiscordAudioOutput(client, guild.id),
        )
        if session.current is not None:
            output = session.output
            if isinstance(output, DiscordAudioOutput) and output.destination_id != channel.id:
                raise UserError("audio.other_voice_active")
        await runtime.audio.connect(workspace_id, str(channel.id))
        return DiscordConnectVoiceResponse(channel_id=str(channel.id), connected=True)

    async def play_audio(
        request: AudioPlayRequest,
        context: InvocationContext,
    ) -> AudioPlayResponse:
        """Enforce Discord voice scope before entering the shared audio API."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        await _prepare_actor_audio(client, runtime, guild, member)
        response = await runtime.registry.invoke(
            "audio.play",
            AudioPlayRequest(
                reference=request.reference,
                requested_by_name=member.display_name,
            ),
            context,
        )
        return cast(AudioPlayResponse, response)

    async def play_attachment(
        request: DiscordPlayAttachmentRequest,
        context: InvocationContext,
    ) -> AudioPlayResponse:
        """Import one authorized Discord attachment, then use normal voice policy."""

        message, attachment = await _attachment(
            client,
            context,
            request.channel_id,
            request.message_id,
            request.attachment_index,
        )
        if not attachment_can_play(attachment):
            raise UserError("local_media.content_type_unsupported")
        record = await import_discord_attachment(
            runtime,
            attachment,
            source_message=message,
            uploader=message.author,
        )
        return await play_audio(
            AudioPlayRequest(reference=record.reference),
            context,
        )

    async def control_audio(
        request: AudioControlRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        """Enforce requester ownership before mutating a Discord audio session."""

        return await _invoke_audio_control("audio.control", request, context)

    async def _invoke_audio_control(
        capability_name: str,
        request: object,
        context: InvocationContext,
    ) -> AudioControlResponse:
        await _assert_audio_control_access(context)
        response = await runtime.registry.invoke(capability_name, request, context)
        return cast(AudioControlResponse, response)

    async def _assert_audio_control_access(context: InvocationContext) -> None:
        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        session = runtime.audio.require(str(guild.id))
        if session.output.connected:
            _assert_same_voice(
                session.destination_id,
                _member_voice_channel(member),
            )
        elif session.waiting_for_voice and not session.can_control_while_waiting(context.actor_id):
            raise UserError("audio.waiting_queue_restricted")

    async def pause_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.pause", request, context)

    async def resume_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.resume", request, context)

    async def skip_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.skip", request, context)

    async def stop_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.stop", request, context)

    async def leave_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.leave", request, context)

    async def set_audio_loop(
        request: AudioLoopRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_loop", request, context)

    async def remove_audio(
        request: AudioQueuePositionRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.remove", request, context)

    async def set_audio_auto_leave(
        request: AudioAutoLeaveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_auto_leave", request, context)

    async def shuffle_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.shuffle", request, context)

    async def seek_audio(
        request: AudioSeekRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.seek", request, context)

    async def tune_audio(
        request: AudioTuneRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.tune", request, context)

    async def set_audio_volume(
        request: AudioVolumeRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.set_volume", request, context)

    async def set_audio_radio(
        request: AudioMixRequest,
        context: InvocationContext,
    ) -> AudioMixResponse:
        await _assert_audio_control_access(context)
        response = await runtime.registry.invoke("audio.mix", request, context)
        return cast(AudioMixResponse, response)

    async def move_audio(
        request: AudioMoveRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.move", request, context)

    async def clear_my_audio(
        request: AudioNoArgsRequest,
        context: InvocationContext,
    ) -> AudioControlResponse:
        return await _invoke_audio_control("audio.clear_mine", request, context)

    def discord_audio_endpoint(
        name: str,
        summary: str,
        keywords: tuple[str, ...],
        request_type: type[Any],
        handler: Callable[..., Awaitable[AudioControlResponse]],
    ) -> CapabilityEndpoint:
        return endpoint(
            CapabilityDescriptor(
                name=name,
                summary=summary,
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", *keywords),
                side_effects=("Changes the server's persistent audio session.",),
                requires_workspace=True,
                requires_voice=True,
                requires_same_voice=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "audio.same_voice_required",
                ),
                timeout_seconds=15,
                user_visible_effect="Updates the shared Audio panel and playback state.",
            ),
            request_type,
            AudioControlResponse,
            handler,
        )

    async def speak(
        request: SpeechSpeakRequest,
        context: InvocationContext,
    ) -> SpeechSpeakResponse:
        """Prepare the requester's voice route before shared speech synthesis."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        destination = _member_voice_channel(member)
        if destination is None:
            raise UserError("discord.voice_join_required")
        _enforce_information_flow_to_destination(
            client,
            context,
            guild,
            destination,
        )
        _enforce_voice_listener_audience(runtime, context, guild, destination)
        await _prepare_actor_audio(client, runtime, guild, member)
        response = await runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=request.text,
                title=f"Requested by {member.display_name}",
                segments=request.segments,
            ),
            context,
        )
        return cast(SpeechSpeakResponse, response)

    async def manage_read_aloud(
        request: ReadAloudRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        """Protect persistent read-aloud mutations with Discord permissions."""

        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        mutating = request.action is not ReadAloudAction.STATUS
        can_manage_guild = _permission_enabled(
            member.guild_permissions,
            "administrator",
        ) or _permission_enabled(member.guild_permissions, "manage_guild")
        if mutating and not can_manage_guild:
            member_voice = _member_voice_channel(member)
            if request.action is ReadAloudAction.ADD_SOURCES:
                current_route = runtime.read_aloud.get(str(guild.id))
                self_service_allowed = (
                    bool(request.text_channel_ids)
                    and request.audio_destination_id is not None
                    and member_voice is not None
                    and str(member_voice.id) == request.audio_destination_id
                    and (
                        current_route is None
                        or current_route.audio_destination_id == request.audio_destination_id
                    )
                )
            elif request.action in {
                ReadAloudAction.CONFIGURE,
                ReadAloudAction.ADD_SOURCE,
            }:
                self_service_allowed = (
                    request.text_channel_id is not None
                    and request.audio_destination_id is not None
                    and context.origin_resource_id == request.text_channel_id
                    and member_voice is not None
                    and str(member_voice.id) == request.audio_destination_id
                )
            else:
                route = runtime.read_aloud.get(str(guild.id))
                self_service_allowed = (
                    route is not None
                    and member_voice is not None
                    and str(member_voice.id) == route.audio_destination_id
                    and (
                        request.action is ReadAloudAction.DISABLE
                        or (
                            request.action is ReadAloudAction.REMOVE_SOURCE
                            and request.text_channel_id is not None
                            and context.origin_resource_id == request.text_channel_id
                        )
                    )
                )
            if not self_service_allowed:
                raise UserError("discord.manage_guild_required")
        audience_sources: list[DiscordReadableChannel] = []
        audience_destination: discord.VoiceChannel | discord.StageChannel | None = None
        if request.action is ReadAloudAction.ADD_SOURCES:
            source_ids = tuple(dict.fromkeys(request.text_channel_ids))
            if not 1 <= len(source_ids) <= 25:
                raise UserError("read_aloud.source_channel_limit")
            if request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            for source_id in source_ids:
                source = _message_channel(guild, source_id)
                if not _can_read_messages(source, member):
                    raise UserError("discord.message_channel_unavailable")
                if guild.me is None or not _can_read_messages(source, guild.me):
                    raise UserError("discord.message_channel_unavailable")
                audience_sources.append(source)
            voice = guild.get_channel(_snowflake(request.audio_destination_id, "voice channel"))
            if not isinstance(voice, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("discord.voice_channel_required")
            audience_destination = voice
        elif request.action in {
            ReadAloudAction.CONFIGURE,
            ReadAloudAction.ADD_SOURCE,
        }:
            if request.text_channel_id is None or request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            source = _message_channel(guild, request.text_channel_id)
            if not _can_read_messages(source, member):
                raise UserError("discord.message_channel_unavailable")
            if guild.me is None or not _can_read_messages(source, guild.me):
                raise UserError("discord.message_channel_unavailable")
            voice = guild.get_channel(_snowflake(request.audio_destination_id, "voice channel"))
            if not isinstance(voice, (discord.VoiceChannel, discord.StageChannel)):
                raise UserError("discord.voice_channel_required")
            audience_sources.append(source)
            audience_destination = voice
        elif request.action is ReadAloudAction.REMOVE_SOURCE:
            if request.text_channel_id is None:
                raise UserError("read_aloud.source_channel_required")
            _message_channel(guild, request.text_channel_id)
        if audience_destination is not None:
            if request.action in {
                ReadAloudAction.ADD_SOURCES,
                ReadAloudAction.ADD_SOURCE,
            }:
                current_route = runtime.read_aloud.get(str(guild.id))
                if (
                    current_route is not None
                    and current_route.audio_destination_id == str(audience_destination.id)
                ):
                    known_source_ids = {source.id for source in audience_sources}
                    for existing_source_id in current_route.text_channel_ids:
                        existing_source = _message_channel(guild, existing_source_id)
                        if existing_source.id in known_source_ids:
                            continue
                        if not _can_read_messages(existing_source, member):
                            raise UserError("discord.message_channel_unavailable")
                        if guild.me is None or not _can_read_messages(
                            existing_source,
                            guild.me,
                        ):
                            raise UserError("discord.message_channel_unavailable")
                        audience_sources.append(existing_source)
                        known_source_ids.add(existing_source.id)
            _enforce_read_aloud_route_audience(
                runtime,
                context,
                guild,
                tuple(audience_sources),
                audience_destination,
            )
        response = await runtime.registry.invoke(
            "speech.manage_read_aloud",
            request,
            context,
        )
        return cast(ReadAloudResponse, response)

    async def read_aloud_status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(action=ReadAloudAction.STATUS),
            context,
        )

    async def read_aloud_add_sources(
        request: ReadAloudAddSourcesRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(
                action=ReadAloudAction.ADD_SOURCES,
                text_channel_ids=request.text_channel_ids,
                audio_destination_id=request.audio_destination_id,
                mode=request.mode,
            ),
            context,
        )

    async def read_aloud_remove_source(
        request: ReadAloudRemoveSourceRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(
                action=ReadAloudAction.REMOVE_SOURCE,
                text_channel_id=request.text_channel_id,
            ),
            context,
        )

    async def read_aloud_disable(
        _request: ReadAloudDisableRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return await manage_read_aloud(
            ReadAloudRequest(action=ReadAloudAction.DISABLE),
            context,
        )

    async def read_aloud_policy_status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        _guild(client, context)
        response = await runtime.registry.invoke(
            "speech.read_aloud_policy_status",
            ReadAloudStatusRequest(),
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_list(
        _request: ReadAloudDictionaryListRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        _guild(client, context)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_list",
            ReadAloudDictionaryListRequest(),
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_set(
        request: ReadAloudDictionarySetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_dictionary_remove(
        request: ReadAloudDictionaryRemoveRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_dictionary_remove",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_exclusion_set(
        request: ReadAloudExclusionSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        guild = _guild(client, context)
        member = await _actor_member(guild, context)
        target_id = _snowflake(request.target_id, request.target.value)
        if request.target is ReadAloudExclusionTarget.USER:
            if target_id != member.id:
                _require_manage_guild(member)
            target_member = guild.get_member(target_id)
            if target_member is None:
                try:
                    target_member = await guild.fetch_member(target_id)
                except discord.DiscordException as exc:
                    raise UserError("discord.member_lookup_failed") from exc
            if target_member.bot:
                raise UserError("read_aloud.ignore_bot_unnecessary")
        else:
            _require_manage_guild(member)
            if guild.get_role(target_id) is None:
                raise UserError("read_aloud.role_not_found")
        response = await runtime.registry.invoke(
            "speech.read_aloud_exclusion_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_announcements_set(
        request: ReadAloudAnnouncementsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_announcements_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_semantics_set(
        request: ReadAloudSemanticsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_semantics_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    async def read_aloud_content_mode_set(
        request: ReadAloudContentModeSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        member = await _actor_member(_guild(client, context), context)
        _require_manage_guild(member)
        response = await runtime.registry.invoke(
            "speech.read_aloud_content_mode_set",
            request,
            context,
        )
        return cast(ReadAloudPolicyResponse, response)

    endpoints = (
        endpoint(
            CapabilityDescriptor(
                name="discord.list_servers",
                summary=(
                    "Page through Discord servers shared by the active requester and "
                    "bot. Uncached memberships are checked live; follow next_offset, "
                    "and treat membership_checks_complete=false on any page as "
                    "incomplete discovery."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "servers",
                    "guilds",
                    "shared",
                    "search",
                    "サーバー",
                    "共有",
                    "横断検索",
                    "参加しているサーバーを一覧",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.server_offset_invalid",
                    "discord.server_limit_invalid",
                ),
                timeout_seconds=10,
            ),
            DiscordListServersRequest,
            DiscordListServersResponse,
            list_servers,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_server",
                summary="Inspect an accessible Discord server's structure and identifiers.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "server",
                    "guild",
                    "channels",
                    "roles",
                    "members",
                    "サーバー",
                    "チャンネル",
                    "メンバー",
                ),
                requires_workspace=True,
                expected_errors=("discord.guild_unavailable",),
            ),
            DiscordServerRequest,
            DiscordServerResponse,
            inspect_server,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.inspect_user",
                summary="Inspect a Discord user's public account and server membership details.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "user",
                    "member",
                    "avatar",
                    "role",
                    "ユーザー",
                    "メンバー",
                    "役職",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.user_id_invalid",
                    "discord.user_not_found",
                ),
            ),
            DiscordUserRequest,
            DiscordUserResponse,
            inspect_user,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_voice_states",
                summary=(
                    "List every currently connected member in voice or stage channels "
                    "visible to both the requester and bot, including mute, stream, "
                    "video, and speaker state."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "voice",
                    "stage",
                    "connected",
                    "presence",
                    "VC",
                    "ボイス",
                    "参加",
                    "通話",
                    "今VCに誰がいる",
                    "ボイス参加者一覧",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.guild_unavailable",
                    "discord.member_required",
                    "discord.voice_state_offset_invalid",
                    "discord.voice_state_limit_invalid",
                ),
                timeout_seconds=10,
            ),
            DiscordListVoiceStatesRequest,
            DiscordListVoiceStatesResponse,
            list_voice_states,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_roles",
                summary=(
                    "Search and page through existing server roles with stable IDs and "
                    "requester/bot assignability signals."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_PUBLIC_METADATA,
                keywords=(
                    "discord",
                    "roles",
                    "existing",
                    "assign",
                    "役職",
                    "ロール",
                    "既存",
                    "一覧",
                    "検索",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.role_query_invalid",
                    "discord.role_offset_invalid",
                    "discord.role_limit_invalid",
                ),
                timeout_seconds=10,
            ),
            DiscordListRolesRequest,
            DiscordListRolesResponse,
            list_roles,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_channels",
                summary=(
                    "Page through text, forum, voice, stage, and active-thread identifiers "
                    "currently readable by both requester and bot in a shared server."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "channels",
                    "threads",
                    "where",
                    "チャンネル",
                    "スレッド",
                    "一覧",
                    "サーバーのチャンネル一覧",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.channel_offset_invalid",
                    "discord.channel_limit_invalid",
                ),
                timeout_seconds=10,
            ),
            DiscordListChannelsRequest,
            DiscordListChannelsResponse,
            list_channels,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.list_archived_threads",
                summary=(
                    "Page through archived public threads or forum posts under one "
                    "readable text/forum parent, returning IDs usable for retrieval."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "archived",
                    "threads",
                    "forum",
                    "posts",
                    "アーカイブ",
                    "過去スレッド",
                    "フォーラム",
                    "投稿",
                    "一覧",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.archived_thread_parent_invalid",
                    "discord.archived_thread_limit_invalid",
                    "discord.archived_thread_time_invalid",
                    "discord.agent_read_channel_forbidden",
                ),
                timeout_seconds=20,
            ),
            DiscordListArchivedThreadsRequest,
            DiscordListArchivedThreadsResponse,
            list_archived_threads,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_messages",
                summary=(
                    "Read one bounded chronological page from an authorized channel, with "
                    "reply/thread and cached reaction signals for trend analysis. When "
                    "before_message_id is the typed active event, the response explicitly "
                    "identifies the active anchor and its immediate Discord predecessor."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "messages",
                    "history",
                    "conversation",
                    "moderation",
                    "trend",
                    "popularity",
                    "engagement",
                    "メッセージ",
                    "履歴",
                    "会話",
                    "読む",
                    "前後",
                    "人気",
                    "分析",
                    "最近の会話を読む",
                    "チャンネルの会話を順番に読む",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.message_limit_invalid",
                ),
                timeout_seconds=15,
            ),
            DiscordReadMessagesRequest,
            DiscordReadMessagesResponse,
            read_messages,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.search_messages",
                summary=(
                    "Search indexed text guild-wide or in selected authorized channels by "
                    "phrase, author, and message-ID/ISO time range. Results are capped at 25 "
                    "and return safe offset/time/opaque page cursors plus IDs for "
                    "discord.get_message. Forum posts are included through readable forum "
                    "parents. Discord's "
                    "total_results is approximate; complete means no known next page, not "
                    "proof that every historical message was indexed."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "messages",
                    "search",
                    "history",
                    "find",
                    "context",
                    "trend",
                    "popularity",
                    "engagement",
                    "メッセージ検索",
                    "過去メッセージ",
                    "履歴",
                    "探す",
                    "検索",
                    "人気",
                    "話題",
                    "分析",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.message_search_query_invalid",
                    "discord.message_search_offset_invalid",
                    "discord.message_search_cursor_invalid",
                    "discord.message_search_cursor_required",
                    "discord.message_search_limit_invalid",
                    "discord.message_search_filter_required",
                    "discord.message_search_time_invalid",
                    "discord.message_search_forbidden",
                ),
                timeout_seconds=20,
            ),
            DiscordSearchMessagesRequest,
            DiscordSearchMessagesResponse,
            search_messages,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.get_message",
                summary=(
                    "Read a Discord message by ID in bounded chunks, optionally including "
                    "the message it replies to in the same channel. Native polls include "
                    "answer IDs, vote counts, expiry, finalization, and winner state so "
                    "discord.list_poll_voters can be called without guessing."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "message",
                    "chunk",
                    "mention",
                    "content",
                    "メッセージ",
                    "取得",
                    "読む",
                    "本文",
                    "原文",
                    "メンション",
                    "poll results",
                    "vote count",
                    "投票結果",
                    "票数",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.message_offset_invalid",
                ),
                timeout_seconds=15,
            ),
            DiscordGetMessageRequest,
            DiscordGetMessageResponse,
            get_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.expand_message",
                summary=(
                    "Resolve a Discord message link for display after enforcing view permissions."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "message link",
                    "expand link",
                    "message preview",
                    "メッセージリンク",
                    "リンク展開",
                    "プレビュー",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.expand_cross_guild_forbidden",
                ),
                timeout_seconds=20,
            ),
            DiscordExpandMessageRequest,
            DiscordExpandMessageResponse,
            expand_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.translate_message",
                summary=(
                    "Read one authorized Discord message and translate its text locally "
                    "to a requested BCP-47 language."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "message",
                    "translate",
                    "translation",
                    "language",
                    "メッセージ",
                    "翻訳",
                    "言語",
                    "ドイツ語",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "translation.message_text_required",
                    "translation.language_unsupported",
                ),
                timeout_seconds=60,
            ),
            DiscordTranslateMessageRequest,
            DiscordTranslateMessageResponse,
            translate_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.post_expanded_message",
                summary=(
                    "Repost a visible Discord message as a text quotation, not an image, "
                    "in another visible channel while preserving attribution and a Jump link."
                ),
                risk=RiskLevel.WRITE,
                keywords=(
                    "discord",
                    "quoted repost",
                    "quote as text",
                    "repost message",
                    "jump link",
                    "引用投稿",
                    "引用して再投稿",
                    "メッセージを引用",
                    "文章のまま引用して別チャンネルに再投稿",
                    "ジャンプリンク",
                ),
                side_effects=("Posts one quotation of the source message to the chosen channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.expand_destination_unavailable",
                ),
                timeout_seconds=30,
                user_visible_effect="Posts a quoted message with attribution and a Jump button.",
            ),
            DiscordPostExpandedMessageRequest,
            DiscordPostExpandedMessageResponse,
            post_expanded_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_quote_image",
                summary=(
                    "Render a visible Discord message locally as a quote image, preserving "
                    "the author, avatar, and emoji, then send it with a source link."
                ),
                risk=RiskLevel.WRITE,
                keywords=(
                    "discord",
                    "quote image",
                    "render message",
                    "screenshot style",
                    "引用画像",
                    "メッセージを引用画像にする",
                    "メッセージを引用画像にして",
                    "画像化",
                    "スクショ風",
                ),
                side_effects=("Posts one locally rendered quote image to the chosen channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.quote_render_failed",
                    "discord.quote_destination_unavailable",
                ),
                timeout_seconds=60,
                user_visible_effect="Posts a locally rendered quote image with source attribution.",
            ),
            DiscordCreateQuoteImageRequest,
            DiscordCreateQuoteImageResponse,
            create_quote_image,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_custom_emoji",
                summary=(
                    "Inspect one selected custom emoji from a Discord message as an image, "
                    "including its full animation or a requested frame when available."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_PUBLIC_METADATA,
                keywords=(
                    "discord",
                    "custom emoji",
                    "animated emoji",
                    "emoji animation",
                    "emoji frame",
                    "カスタム絵文字",
                    "動く絵文字",
                    "アニメーション絵文字",
                    "絵文字のフレーム",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.custom_emoji_index_invalid",
                    "discord.custom_emoji_unavailable",
                ),
                timeout_seconds=30,
            ),
            DiscordViewCustomEmojiRequest,
            DiscordViewCustomEmojiResponse,
            view_custom_emoji,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_sticker",
                summary=(
                    "Inspect one selected sticker from a Discord message as an image, including "
                    "its full animation or a requested frame when the format supports it."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_PUBLIC_METADATA,
                keywords=(
                    "discord",
                    "sticker",
                    "animated sticker",
                    "sticker animation",
                    "sticker frame",
                    "スタンプ",
                    "ステッカー",
                    "動くスタンプ",
                    "スタンプのフレーム",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.sticker_index_invalid",
                    "discord.sticker_unavailable",
                ),
                timeout_seconds=30,
            ),
            DiscordViewStickerRequest,
            DiscordViewStickerResponse,
            view_sticker,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.analyze_attachment",
                summary=(
                    "Analyze an authorized Discord attachment for synthetic-media "
                    "signals without exposing its bytes or signed URL to the model."
                ),
                risk=RiskLevel.EXTERNAL,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                keywords=(
                    "discord",
                    "attachment",
                    "synthetic media",
                    "ai-generated",
                    "deepfake",
                    "manipulated image",
                    "hive",
                    "AI生成",
                    "ディープフェイク",
                    "合成画像",
                    "合成動画",
                    "加工画像",
                    "生成画像判定",
                ),
                side_effects=(
                    "Retrieves one authorized Discord attachment.",
                    "Uses one external analysis request when no cached result is available.",
                ),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="hive",
                    field_kinds=(EgressFieldKind.MEDIA,),
                    sink_audience=EgressSinkAudience.EXTERNAL_PRIVATE,
                    source_resource_fields=("channel_id",),
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.agent_read_channel_forbidden",
                    "discord.message_not_found",
                    "discord.message_fetch_failed",
                    "discord.attachment_missing",
                    "discord.attachment_unavailable",
                    "moderation.media_too_large",
                ),
                timeout_seconds=120,
            ),
            DiscordAnalyzeAttachmentRequest,
            SyntheticMediaAnalyzeResponse,
            analyze_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.import_attachment",
                summary=(
                    "Import a Discord attachment into this server's isolated workspace "
                    "and return its SHA-256 digest."
                ),
                risk=RiskLevel.WRITE,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "import attachment",
                    "workspace file",
                    "PDF",
                    "ZIP",
                    "添付ファイルを取り込む",
                    "作業領域に保存",
                    "添付を保存",
                ),
                side_effects=("Creates or replaces a file inside the isolated workspace.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "discord.member_required",
                    "discord.agent_read_channel_forbidden",
                    "discord.message_not_found",
                    "discord.message_fetch_failed",
                    "discord.attachment_missing",
                    "discord.attachment_unavailable",
                    "files.file_too_large",
                ),
                timeout_seconds=60,
                user_visible_effect="Creates or replaces a file in the isolated workspace.",
            ),
            DiscordImportAttachmentRequest,
            WorkspaceFileRecord,
            import_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.view_image_attachment",
                summary=(
                    "Expose a PNG, JPEG, GIF, or WebP attachment to model vision without "
                    "revealing the signed Discord URL."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.CHANNEL_SCOPED_CONTENT,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "view image attachment",
                    "inspect attached image",
                    "model vision",
                    "添付画像を見る",
                    "画像を確認",
                    "画像を読んで",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.member_required",
                    "discord.agent_read_channel_forbidden",
                    "discord.message_not_found",
                    "discord.message_fetch_failed",
                    "discord.attachment_missing",
                    "discord.attachment_unavailable",
                    "discord.image_attachment_too_large",
                    "discord.attachment_not_supported_image",
                ),
                timeout_seconds=30,
            ),
            DiscordViewImageAttachmentRequest,
            DiscordViewImageAttachmentResponse,
            view_image_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.add_reaction",
                summary=(
                    "Add one intentional bot reaction to an authorized Discord message."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "add reaction",
                    "add an emoji reaction to this message",
                    "react with emoji",
                    "リアクションを付ける",
                    "絵文字で反応",
                    "リアクション追加",
                    "絵文字でリアクションを付けて",
                ),
                side_effects=("Adds the bot's reaction to one visible message.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.reaction_emoji_invalid",
                    "discord.reaction_forbidden",
                    "discord.reaction_target_not_found",
                ),
                timeout_seconds=15,
                user_visible_effect="Adds one bot-authored reaction to a message.",
            ),
            DiscordReactionRequest,
            DiscordReactionResponse,
            add_reaction,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.remove_own_reaction",
                summary="Remove only the bot's own selected reaction from a Discord message.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "remove own reaction",
                    "remove the bot's own reaction",
                    "undo bot reaction",
                    "自分のリアクションを外す",
                    "BOTのリアクションを削除",
                    "リアクションを取り消す",
                    "BOT自身のリアクションを外して",
                ),
                side_effects=("Removes the bot's selected reaction from one message.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.reaction_emoji_invalid",
                    "discord.reaction_forbidden",
                    "discord.reaction_target_not_found",
                ),
                timeout_seconds=15,
                user_visible_effect="Removes one reaction previously added by the bot.",
            ),
            DiscordReactionRequest,
            DiscordReactionResponse,
            remove_own_reaction,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_message",
                summary=(
                    "Post one distinct plain Discord channel message, not a reply, embed, "
                    "file attachment, or DM. Routine progress is already shown by the host "
                    "UI; use purpose=progress only for a useful bespoke interim update, or "
                    "purpose=requested_action when the person explicitly requested a "
                    "separate post. Use purpose=final when this post is the complete answer "
                    "and host reply delivery should be suppressed."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "send message",
                    "post message",
                    "plain message",
                    "new channel message",
                    "メッセージを送る",
                    "メッセージを送って",
                    "通常メッセージ",
                    "新規投稿",
                    "チャンネルに投稿",
                    "途中経過を送る",
                    "別メッセージ",
                ),
                side_effects=("Creates a user-visible message in a Discord channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=("discord.agent_update_channel_forbidden",),
                timeout_seconds=15,
                user_visible_effect="Posts a new Discord message visible to channel members.",
            ),
            DiscordSendMessageRequest,
            DiscordSendMessageResponse,
            send_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_embed",
                summary=(
                    "Post one clean AI-authored Discord embed when a compact structured "
                    "card materially improves a requested result or useful interim update. "
                    "Supports a title, description, up to 10 fields, and a restrained tone; "
                    "it adds no timestamp, footer, provider label, image URL, or mentions. "
                    "It may reply to a selected message, and purpose=final makes the card "
                    "the complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "embed",
                    "card",
                    "structured",
                    "status",
                    "summary",
                    "埋め込み",
                    "カード",
                    "ステータス",
                ),
                side_effects=("Creates a user-visible embed in a Discord channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.agent_write_channel_forbidden",
                    "discord.embed_links_required",
                    "discord.embed_title_invalid",
                    "discord.embed_description_invalid",
                    "discord.embed_fields_invalid",
                    "discord.embed_field_invalid",
                    "discord.embed_length_invalid",
                    "discord.embed_tone_invalid",
                ),
                timeout_seconds=15,
                user_visible_effect="Posts a clean structured card visible to channel members.",
            ),
            DiscordSendEmbedRequest,
            DiscordSendMessageResponse,
            send_embed,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.reply_message",
                summary=(
                    "Reply as the bot to any readable Discord message in an authorized "
                    "shared server. Use purpose=final when this reply is the complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "reply",
                    "respond",
                    "message",
                    "返信",
                    "返事",
                    "メッセージ",
                ),
                requires_workspace=True,
            ),
            DiscordReplyMessageRequest,
            DiscordSendMessageResponse,
            reply_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.edit_own_message",
                summary="Edit one message authored by this bot.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "edit",
                    "own",
                    "message",
                    "correct",
                    "編集",
                    "修正",
                    "訂正",
                    "メッセージ",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
            ),
            DiscordMessageWriteRequest,
            DiscordMessageWriteResponse,
            edit_own_message,
        ),
        *(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=summary,
                    risk=RiskLevel.WRITE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=keywords,
                    requires_workspace=True,
                    idempotency="idempotent_write",
                ),
                DiscordMessageWriteRequest,
                DiscordMessageWriteResponse,
                handler,
            )
            for name, summary, keywords, handler in (
                (
                    "discord.pin_message",
                    "Pin one readable message after checking requester and bot permissions.",
                    (
                        "discord",
                        "pin",
                        "message",
                        "bookmark",
                        "ピン留め",
                        "固定",
                        "メッセージをピン留めして",
                    ),
                    pin_message,
                ),
                (
                    "discord.unpin_message",
                    "Unpin one message after checking requester and bot permissions.",
                    (
                        "discord",
                        "unpin",
                        "message",
                        "undo",
                        "ピン留め解除",
                        "固定解除",
                        "ピン留めを解除して",
                    ),
                    unpin_message,
                ),
            )
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_thread",
                summary="Create a public thread in the current Discord server.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "thread",
                    "create",
                    "discussion",
                    "split",
                    "スレッド",
                    "議論",
                    "会話",
                    "分ける",
                    "作成",
                ),
                requires_workspace=True,
            ),
            DiscordThreadCreateRequest,
            DiscordThreadResponse,
            create_thread,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.update_thread",
                summary="Rename, archive, or unarchive a Discord thread.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "thread",
                    "rename",
                    "archive",
                    "unarchive",
                    "スレッド",
                    "名前変更",
                    "アーカイブ",
                    "再開",
                    "スレッドの名前を変更して",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
            ),
            DiscordThreadUpdateRequest,
            DiscordThreadResponse,
            update_thread,
        ),
        *(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=summary,
                    risk=RiskLevel.WRITE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=keywords,
                    requires_workspace=True,
                    idempotency="idempotent_write",
                ),
                DiscordThreadMemberRequest,
                DiscordThreadMemberResponse,
                handler,
            )
            for name, summary, keywords, handler in (
                (
                    "discord.add_thread_member",
                    "Add one server member to a Discord thread.",
                    (
                        "discord",
                        "thread",
                        "member",
                        "add",
                        "invite",
                        "スレッド",
                        "メンバー",
                        "追加",
                        "招待",
                    ),
                    add_thread_member,
                ),
                (
                    "discord.remove_thread_member",
                    "Remove one server member from a Discord thread.",
                    (
                        "discord",
                        "thread",
                        "member",
                        "remove",
                        "スレッド",
                        "メンバー",
                        "削除",
                        "外す",
                        "スレッドから外して",
                    ),
                    remove_thread_member,
                ),
            )
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_forum_post",
                summary="Create a forum post with an initial bot-authored message.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "forum",
                    "post",
                    "bug",
                    "discussion",
                    "フォーラム",
                    "投稿",
                    "バグ報告",
                    "議論",
                    "整理",
                ),
                requires_workspace=True,
            ),
            DiscordForumPostRequest,
            DiscordForumPostResponse,
            create_forum_post,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_role",
                summary="Create a role below the bot and requester's role hierarchy.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "role",
                    "create",
                    "permission",
                    "役職",
                    "ロール",
                    "作成",
                    "権限",
                    "サーバーロールを作って",
                ),
                requires_workspace=True,
            ),
            DiscordRoleCreateRequest,
            DiscordRoleResponse,
            create_role,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_created_role",
                summary="Internal Undo for a role created by a recorded action receipt.",
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.ALWAYS,
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "action.undo_conflict",
                    "action.undo_target_in_use",
                    "action.undo_target_state_uncertain",
                    "discord.manage_roles_required",
                    "discord.role_hierarchy_forbidden",
                ),
            ),
            DiscordCreatedRoleDeleteRequest,
            DiscordCreatedEntityDeleteResponse,
            delete_created_role,
        ),
        *(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=summary,
                    risk=RiskLevel.WRITE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=keywords,
                    requires_workspace=True,
                    idempotency="idempotent_write",
                ),
                DiscordRoleMemberRequest,
                DiscordRoleMemberResponse,
                handler,
            )
            for name, summary, keywords, handler in (
                (
                    "discord.assign_role",
                    "Assign an eligible role to one lower-ranked server member.",
                    (
                        "discord",
                        "role",
                        "assign",
                        "member",
                        "grant",
                        "役職",
                        "ロール",
                        "付与",
                        "割り当て",
                        "メンバー",
                    ),
                    assign_role,
                ),
                (
                    "discord.remove_role",
                    "Remove an eligible role from one lower-ranked server member.",
                    (
                        "discord",
                        "role",
                        "remove",
                        "member",
                        "revoke",
                        "役職",
                        "ロール",
                        "解除",
                        "外す",
                        "メンバー",
                        "ロールを外して",
                    ),
                    remove_role,
                ),
            )
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.update_channel_settings",
                summary="Update a text channel topic or slowmode with scalar Undo support.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "channel",
                    "topic",
                    "slowmode",
                    "edit",
                    "チャンネル",
                    "トピック",
                    "低速モード",
                    "編集",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
            ),
            DiscordChannelSettingRequest,
            DiscordChannelSettingResponse,
            update_channel_settings,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_channel",
                summary="Create a text channel in the current Discord server.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "channel",
                    "create",
                    "text",
                    "チャンネル",
                    "作成",
                    "テキスト",
                ),
                requires_workspace=True,
            ),
            DiscordChannelCreateRequest,
            DiscordChannelCreateResponse,
            create_channel,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_created_channel",
                summary="Internal Undo for a channel created by a recorded action receipt.",
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.ALWAYS,
                requires_workspace=True,
                idempotency="idempotent_write",
            ),
            DiscordCreatedChannelDeleteRequest,
            DiscordCreatedEntityDeleteResponse,
            delete_created_channel,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.set_timeout",
                summary="Set or clear a member timeout after permission and hierarchy checks.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "moderation",
                    "timeout",
                    "clear",
                    "member",
                    "モデレーション",
                    "タイムアウト",
                    "発言禁止",
                    "ミュート",
                    "解除",
                    "メンバー",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
            ),
            DiscordTimeoutRequest,
            DiscordTimeoutResponse,
            set_timeout,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_message",
                summary="Delete one message as a moderated, audited non-undoable action.",
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "moderation",
                    "delete",
                    "message",
                    "モデレーション",
                    "削除",
                    "メッセージ",
                ),
                requires_workspace=True,
            ),
            DiscordMessageWriteRequest,
            DiscordMessageWriteResponse,
            delete_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.bulk_delete_messages",
                summary="Delete 2-100 exact message IDs without an unbounded history scan.",
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "moderation",
                    "bulk",
                    "delete",
                    "messages",
                    "モデレーション",
                    "一括削除",
                    "メッセージ",
                ),
                requires_workspace=True,
                expected_errors=(
                    "discord.bulk_delete_limit_invalid",
                    "discord.bulk_delete_message_too_old",
                ),
            ),
            DiscordBulkDeleteRequest,
            DiscordBulkDeleteResponse,
            bulk_delete_messages,
        ),
        *(
            endpoint(
                CapabilityDescriptor(
                    name=name,
                    summary=summary,
                    risk=(
                        RiskLevel.DESTRUCTIVE
                        if name in {"discord.kick_member", "discord.ban_member"}
                        else RiskLevel.WRITE
                    ),
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=keywords,
                    requires_workspace=True,
                    idempotency=(
                        "idempotent_write"
                        if name == "discord.unban_member"
                        else "non_idempotent_write"
                    ),
                ),
                DiscordMemberModerationRequest,
                DiscordMemberModerationResponse,
                handler,
            )
            for name, summary, keywords, handler in (
                (
                    "discord.kick_member",
                    "Kick a lower-ranked member with an explicit reason and evidence IDs.",
                    (
                        "discord",
                        "moderation",
                        "kick",
                        "member",
                        "モデレーション",
                        "キック",
                        "退出",
                        "メンバー",
                        "サーバーからキックして",
                    ),
                    kick_member,
                ),
                (
                    "discord.ban_member",
                    "Ban a lower-ranked member without deleting message history.",
                    (
                        "discord",
                        "moderation",
                        "ban",
                        "member",
                        "モデレーション",
                        "BAN",
                        "追放",
                        "メンバー",
                    ),
                    ban_member,
                ),
                (
                    "discord.unban_member",
                    "Remove a server ban for one user ID.",
                    (
                        "discord",
                        "moderation",
                        "unban",
                        "member",
                        "モデレーション",
                        "BAN解除",
                        "追放解除",
                        "メンバー",
                        "BANを解除して",
                    ),
                    unban_member,
                ),
            )
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_own_message",
                summary="Delete only a Discord message authored by this bot.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "delete own message",
                    "delete message authored by the bot",
                    "delete this message authored by the bot",
                    "undo bot message",
                    "BOTのメッセージを削除",
                    "自分の投稿を消す",
                    "送信を取り消す",
                    "BOT自身が送った投稿を消して",
                ),
                side_effects=("Deletes one bot-authored Discord message.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.message_not_owned",
                    "discord.message_delete_forbidden",
                ),
                timeout_seconds=15,
                user_visible_effect="Deletes one message previously posted by the bot.",
            ),
            DiscordDeleteOwnMessageRequest,
            DiscordDeleteOwnMessageResponse,
            delete_own_message,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.delete_own_messages",
                summary="Internal Undo for one multi-post agent response.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.ALWAYS,
                side_effects=("Deletes only bot-authored messages from one response.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.message_not_owned",
                    "discord.message_delete_forbidden",
                    "discord.message_delete_failed",
                    "action.undo_target_state_uncertain",
                ),
                timeout_seconds=30,
                user_visible_effect="Deletes one complete multi-message agent response.",
            ),
            DiscordDeleteOwnMessagesRequest,
            DiscordDeleteOwnMessagesResponse,
            delete_own_messages,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_files",
                summary=(
                    "Send 1-10 workspace files as real Discord attachments, with "
                    "optional descriptions, spoiler treatment, reply target, and "
                    "silent delivery. Use purpose=final when the attachment post is the "
                    "complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "files",
                    "attachments",
                    "send",
                    "deliver",
                    "export",
                    "複数ファイル",
                    "添付",
                    "送る",
                    "届ける",
                    "multiple files together",
                    "複数ファイルをまとめて送る",
                    "3つのファイルを添付してまとめて送って",
                ),
                side_effects=(
                    "Creates one Discord message containing 1-10 attachments.",
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "discord.file_count_invalid",
                    "discord.file_too_large",
                    "discord.file_send_forbidden",
                ),
                timeout_seconds=30,
                user_visible_effect=(
                    "Posts one Discord message containing the selected attachments."
                ),
            ),
            DiscordSendFilesRequest,
            DiscordSendFilesResponse,
            send_files,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.send_file",
                summary=(
                    "Send one workspace file as a real Discord attachment, with "
                    "optional description, spoiler treatment, reply target, and "
                    "silent delivery. Use purpose=final when the attachment post is the "
                    "complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "discord",
                    "file",
                    "attachment",
                    "send",
                    "deliver",
                    "export",
                    "ファイル",
                    "添付",
                    "送る",
                    "届ける",
                    "ファイルを一つ添付して送って",
                ),
                side_effects=("Creates one Discord message with an attachment.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "discord.file_too_large",
                    "discord.file_send_forbidden",
                ),
                timeout_seconds=30,
                user_visible_effect="Posts a Discord message containing the selected file.",
            ),
            DiscordSendFileRequest,
            DiscordSendFileResponse,
            send_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.create_poll",
                summary="Create a native poll in a Discord text channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "poll",
                    "vote",
                    "question",
                    "投票",
                    "アンケート",
                ),
                side_effects=("Creates a user-visible poll in a Discord channel.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.agent_update_channel_forbidden",
                    "discord.poll_forbidden",
                    "discord.poll_question_invalid",
                    "discord.poll_option_count_invalid",
                    "discord.poll_option_too_long",
                    "discord.poll_duration_invalid",
                    "discord.poll_failed",
                ),
                timeout_seconds=15,
                user_visible_effect="Posts a native Discord poll in the active channel.",
            ),
            DiscordPollRequest,
            DiscordPollResponse,
            create_poll,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.connect_voice",
                summary="Connect Simajilord's audio output to a Discord voice channel.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "voice",
                    "join",
                    "connect",
                    "vc",
                    "ボイス",
                    "通話",
                    "参加",
                ),
                side_effects=("Makes the bot join or move to a voice channel.",),
                requires_workspace=True,
                requires_voice=True,
                requires_same_voice=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.voice_channel_unavailable",
                    "audio.same_voice_required",
                    "audio.other_voice_active",
                ),
                timeout_seconds=30,
                user_visible_effect="Joins the requester's current Discord voice channel.",
            ),
            DiscordConnectVoiceRequest,
            DiscordConnectVoiceResponse,
            connect_voice,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.play_audio",
                summary=(
                    "Play public media in the requester's voice channel. If the requester "
                    "is not in voice, keep it queued until they join."
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "music",
                    "audio",
                    "play",
                    "queue",
                    "voice",
                    "曲",
                    "流して",
                    "再生",
                ),
                side_effects=(
                    "May join the requester's voice channel.",
                    "Adds one track to the server's persistent queue.",
                ),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="public_media",
                    field_kinds=(EgressFieldKind.URL, EgressFieldKind.QUERY),
                    request_fields=("reference",),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "audio.waiting_queue_restricted",
                    "audio.same_voice_required",
                ),
                timeout_seconds=60,
                user_visible_effect="Queues a track and refreshes the shared Audio panel.",
            ),
            AudioPlayRequest,
            AudioPlayResponse,
            play_audio,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.play_attachment",
                summary=(
                    "Validate and persist an audio or video attachment from an "
                    "authorized Discord message, then queue its audio."
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "attachment",
                    "audio",
                    "video",
                    "play",
                    "queue",
                    "voice",
                    "添付",
                    "音声",
                    "動画",
                    "流して",
                ),
                side_effects=(
                    "Copies the selected attachment into the private local media store.",
                    "May join the requester's voice channel.",
                    "Adds one item to the server's persistent queue.",
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.attachment_missing",
                    "discord.attachment_unavailable",
                    "local_media.content_type_unsupported",
                    "local_media.too_large",
                    "local_media.cache_full",
                ),
                timeout_seconds=120,
                user_visible_effect=(
                    "Stores the attachment locally, queues its audio, and refreshes "
                    "the shared Audio panel."
                ),
            ),
            DiscordPlayAttachmentRequest,
            AudioPlayResponse,
            play_attachment,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.control_audio",
                summary=(
                    "Control server audio when the requester shares the bot's voice channel "
                    "or owns the relevant waiting queue item."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("discord", "music", "pause", "resume", "skip", "stop", "loop"),
                side_effects=("Changes the server's persistent audio session.",),
                requires_workspace=True,
                requires_voice=True,
                requires_same_voice=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "audio.same_voice_required",
                ),
                timeout_seconds=15,
                user_visible_effect="Updates the shared Audio panel and playback state.",
            ),
            AudioControlRequest,
            AudioControlResponse,
            control_audio,
        ),
        discord_audio_endpoint(
            "discord.pause_audio",
            "Pause music playing in the requester's voice channel.",
            ("pause", "一時停止", "止める"),
            AudioNoArgsRequest,
            pause_audio,
        ),
        discord_audio_endpoint(
            "discord.resume_audio",
            "Resume paused music in the requester's voice channel.",
            ("resume", "再開", "続き"),
            AudioNoArgsRequest,
            resume_audio,
        ),
        discord_audio_endpoint(
            "discord.skip_audio",
            "Skip the current track in the requester's voice channel.",
            ("skip", "スキップ", "次の曲", "飛ばす"),
            AudioNoArgsRequest,
            skip_audio,
        ),
        discord_audio_endpoint(
            "discord.stop_audio",
            "Stop playback and clear the music queue in the requester's voice channel.",
            ("stop", "clear", "停止", "終了", "キューを空にする"),
            AudioNoArgsRequest,
            stop_audio,
        ),
        discord_audio_endpoint(
            "discord.leave_audio",
            "Clear the music queue and disconnect from the requester's voice channel.",
            (
                "leave",
                "disconnect bot from voice",
                "clear queue and disconnect",
                "disconnect the bot from voice and clear the queue",
                "切断",
                "退出",
                "BOTをVCから切断して",
            ),
            AudioNoArgsRequest,
            leave_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_loop",
            "Set the music loop mode.",
            (
                "loop",
                "repeat",
                "ループ",
                "リピート",
                "繰り返し",
                "リピート再生にして",
            ),
            AudioLoopRequest,
            set_audio_loop,
        ),
        discord_audio_endpoint(
            "discord.remove_audio",
            "Remove a waiting track at a specified queue position.",
            ("remove", "queue", "削除", "キュー"),
            AudioQueuePositionRequest,
            remove_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_auto_leave",
            "Configure automatic disconnect when no listeners remain.",
            (
                "automatic voice disconnect",
                "when no listeners remain",
                "自動退出",
                "自動切断",
            ),
            AudioAutoLeaveRequest,
            set_audio_auto_leave,
        ),
        discord_audio_endpoint(
            "discord.shuffle_audio",
            "Shuffle the waiting music queue.",
            ("shuffle", "queue", "シャッフル", "キュー"),
            AudioNoArgsRequest,
            shuffle_audio,
        ),
        discord_audio_endpoint(
            "discord.seek_audio",
            "Seek the current track to a specified time in seconds.",
            (
                "seek",
                "playback position",
                "シーク",
                "再生位置",
                "再生位置を移して",
            ),
            AudioSeekRequest,
            seek_audio,
        ),
        discord_audio_endpoint(
            "discord.tune_audio",
            "Set music playback speed and pitch.",
            ("speed", "pitch", "tune", "速度", "ピッチ"),
            AudioTuneRequest,
            tune_audio,
        ),
        discord_audio_endpoint(
            "discord.set_audio_volume",
            "Set music and read-aloud volume percentages.",
            ("volume", "speech", "音楽", "音量", "下げて", "上げて"),
            AudioVolumeRequest,
            set_audio_volume,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.set_audio_radio",
                summary=(
                    "Start or stop related-track selection from one or more YouTube seeds. "
                    "Human requests always take priority over automatically selected tracks."
                ),
                risk=RiskLevel.EXTERNAL,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "music",
                    "mix",
                    "autoplay",
                    "radio",
                    "related",
                    "自動再生",
                    "ラジオ",
                    "関連曲",
                ),
                side_effects=("Changes the server's persistent automatic selection settings.",),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="public_media",
                    field_kinds=(EgressFieldKind.URL,),
                    request_fields=("seed_references",),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
                requires_workspace=True,
                requires_voice=True,
                requires_same_voice=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "audio.same_voice_required",
                    "audio.mix_seed_limit",
                ),
                timeout_seconds=60,
                user_visible_effect="Changes Radio state and refreshes the shared Audio panel.",
            ),
            AudioMixRequest,
            AudioMixResponse,
            set_audio_radio,
        ),
        discord_audio_endpoint(
            "discord.move_audio",
            "Move a waiting track to another queue position.",
            ("move", "queue", "reorder", "移動", "並べ替え", "キュー"),
            AudioMoveRequest,
            move_audio,
        ),
        discord_audio_endpoint(
            "discord.clear_my_audio",
            "Remove only the waiting tracks added by the requester.",
            ("clear", "mine", "requester", "自分の曲", "取り消し"),
            AudioNoArgsRequest,
            clear_my_audio,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.speak",
                summary=(
                    "Speak a short passage with VOICEVOX in the requester's voice channel. "
                    "Music is coordinated automatically while speech is playing. Use "
                    "purpose=final when the spoken passage is the complete answer."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "voice synthesis",
                    "speak in voice",
                    "VOICEVOX",
                    "音声合成",
                    "VOICEVOXでしゃべる",
                    "VOICEVOXでVCにしゃべって",
                    "VCで読み上げる",
                    "声に出す",
                ),
                side_effects=(
                    "May join the requester's voice channel.",
                    "Plays synthesized speech.",
                ),
                requires_workspace=True,
                requires_voice=True,
                idempotency="non_idempotent_write",
                expected_errors=("discord.member_required",),
                timeout_seconds=90,
                user_visible_effect="Plays synthesized speech in the requester's voice channel.",
            ),
            SpeechSpeakRequest,
            SpeechSpeakResponse,
            speak,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_status",
                summary="Inspect the current Discord read-aloud routes without changing them.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "read aloud status",
                    "read aloud routes",
                    "読み上げ状態",
                    "読み上げ経路",
                    "読み上げ設定を確認",
                ),
                requires_workspace=True,
                expected_errors=("discord.member_required",),
                timeout_seconds=10,
            ),
            ReadAloudStatusRequest,
            ReadAloudResponse,
            read_aloud_status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_add_sources",
                summary=(
                    "Add selected conversation channels as sources for the current voice channel."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "add read aloud source",
                    "add conversation channel as read aloud source",
                    "add this conversation channel as read aloud source",
                    "enable channel read aloud",
                    "読み上げ対象を追加",
                    "読み上げチャンネルを追加",
                    "このチャンネルを読み上げ対象に追加",
                ),
                side_effects=("Adds read-aloud source channels to persistent settings.",),
                requires_workspace=True,
                requires_voice=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.voice_channel_required",
                    "read_aloud.audience_forbidden",
                    "read_aloud.source_channel_limit",
                ),
                timeout_seconds=20,
                user_visible_effect=(
                    "Enables automatic read aloud for the selected conversation channels."
                ),
            ),
            ReadAloudAddSourcesRequest,
            ReadAloudResponse,
            read_aloud_add_sources,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_remove_source",
                summary="Remove a selected conversation channel from read-aloud sources.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "remove read aloud source",
                    "disable channel read aloud",
                    "読み上げ対象から外す",
                    "チャンネルを読み上げ対象から外して",
                    "読み上げチャンネルを削除",
                ),
                side_effects=("Removes a read-aloud source channel from persistent settings.",),
            ),
            ReadAloudRemoveSourceRequest,
            ReadAloudResponse,
            read_aloud_remove_source,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_disable",
                summary="Disable read-aloud routes for this Discord server.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "disable read aloud",
                    "stop all read aloud",
                    "disable all read aloud routes for this server",
                    "読み上げを無効",
                    "読み上げを停止",
                    "読み上げを全部止める",
                ),
                side_effects=("Deletes this server's read-aloud routes.",),
            ),
            ReadAloudDisableRequest,
            ReadAloudResponse,
            read_aloud_disable,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_policy_status",
                summary=(
                    "Inspect current read-aloud dictionary, exclusion, and voice-event settings."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "read aloud policy",
                    "pronunciation and exclusion settings",
                    "読み上げポリシー",
                    "読み上げ辞書と除外設定",
                    "読み上げ詳細設定",
                ),
            ),
            ReadAloudStatusRequest,
            ReadAloudPolicyResponse,
            read_aloud_policy_status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_list",
                summary="List this Discord server's read-aloud dictionary.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "discord",
                    "list pronunciation dictionary",
                    "read aloud dictionary",
                    "読み上げ辞書",
                    "読み方一覧",
                    "発音辞書",
                ),
            ),
            ReadAloudDictionaryListRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_list,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_set",
                summary="Add a written form and pronunciation to the server read-aloud dictionary.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "register pronunciation",
                    "add dictionary entry",
                    "読み上げ辞書に登録",
                    "読み方を登録",
                    "発音を登録",
                    "読むようにして",
                    "この単語はこう読む",
                    "読み方を変える",
                ),
                side_effects=("Updates the server-specific read-aloud dictionary.",),
            ),
            ReadAloudDictionarySetRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_dictionary_remove",
                summary="Remove a written form from the server read-aloud dictionary.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "remove pronunciation",
                    "delete dictionary entry",
                    "読み上げ辞書から削除",
                    "読み方を削除",
                    "発音登録を消す",
                ),
                side_effects=("Updates the server-specific read-aloud dictionary.",),
            ),
            ReadAloudDictionaryRemoveRequest,
            ReadAloudPolicyResponse,
            read_aloud_dictionary_remove,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_exclusion_set",
                summary="Enable or disable read-aloud exclusions for a user or role.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "read aloud exclusion",
                    "exclude this user from read aloud",
                    "ignore user or role",
                    "読み上げ除外",
                    "読み上げ対象外",
                    "このユーザーを読まない",
                    "このロールを読まない",
                ),
                side_effects=("Updates read-aloud exclusion settings.",),
            ),
            ReadAloudExclusionSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_exclusion_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_announcements_set",
                summary=(
                    "Configure announcements for joining, leaving, and moving "
                    "between voice channels."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "voice event announcements",
                    "announce voice join leave",
                    "読み上げ",
                    "入退室",
                    "入室",
                    "退出",
                    "VC移動",
                    "入退室アナウンス",
                ),
                side_effects=("Updates read-aloud settings for voice-channel events.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.manage_guild_required",
                ),
                timeout_seconds=15,
                user_visible_effect="Changes which voice join, leave, and move events are spoken.",
            ),
            ReadAloudAnnouncementsSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_announcements_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_content_mode_set",
                summary="Choose the read-aloud content mode: all, messages, events, or off.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "read aloud content mode",
                    "messages or voice events",
                    "読み上げ内容",
                    "メッセージだけ読み上げ",
                    "音声イベントだけ読み上げ",
                    "読み上げモード",
                ),
                side_effects=("Persists the selected read-aloud content mode.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.manage_guild_required",
                ),
                timeout_seconds=15,
                user_visible_effect=(
                    "Changes whether messages, voice events, both, or neither are spoken."
                ),
            ),
            ReadAloudContentModeSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_content_mode_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.read_aloud_semantics_set",
                summary="Configure semantic reading of authors, replies, and attachments.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "read aloud semantics",
                    "narrate author reply attachment",
                    "名前も読み上げ",
                    "返信を読み上げ",
                    "添付を読み上げ",
                    "読み上げ方",
                    "返信元と添付も読み上げる",
                ),
                side_effects=("Updates semantic read-aloud settings.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "discord.member_required",
                    "discord.manage_guild_required",
                ),
                timeout_seconds=15,
                user_visible_effect=(
                    "Changes how authors, replies, attachments, and VC membership "
                    "are narrated."
                ),
            ),
            ReadAloudSemanticsSetRequest,
            ReadAloudPolicyResponse,
            read_aloud_semantics_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="discord.manage_read_aloud",
                summary=(
                    "Inspect read-aloud routes, or configure and disable them when the "
                    "requester has Manage Server permission."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "discord",
                    "legacy read aloud manager",
                    "読み上げ一括管理",
                ),
                side_effects=("May change persistent automatic read-aloud settings.",),
            ),
            ReadAloudRequest,
            ReadAloudResponse,
            manage_read_aloud,
        ),
    )
    # Import lazily so the low-frequency platform module can reuse the
    # authorization helpers above without creating an import-time cycle.
    from .platform_actions import build_discord_platform_action_endpoints
    from .platform_assets import build_discord_platform_asset_endpoints
    from .platform_automod import build_discord_automod_endpoints
    from .platform_capabilities import build_discord_platform_endpoints
    from .platform_operations import build_discord_platform_operation_endpoints

    return (
        *endpoints,
        *build_discord_platform_endpoints(client, runtime),
        *build_discord_platform_action_endpoints(client),
        *build_discord_platform_asset_endpoints(client, runtime),
        *build_discord_automod_endpoints(client),
        *build_discord_platform_operation_endpoints(client),
    )


def _guild(client: discord.Client, context: InvocationContext) -> discord.Guild:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    try:
        guild_id = int(context.workspace_id)
    except ValueError as exc:
        raise UserError("discord.guild_id_invalid") from exc
    guild = client.get_guild(guild_id)
    if guild is None:
        raise UserError("discord.guild_unavailable")
    return guild


def _requested_guild(
    client: discord.Client,
    context: InvocationContext,
    requested_guild_id: str | None,
) -> discord.Guild:
    """Resolve a guild without allowing a command to forge another workspace."""

    if requested_guild_id is None:
        return _guild(client, context)
    guild_id = _snowflake(requested_guild_id, "guild")
    if context.transport != "agent" and requested_guild_id != context.workspace_id:
        raise UserError("discord.cross_guild_read_forbidden")
    guild = client.get_guild(guild_id)
    if guild is None:
        raise UserError("discord.guild_unavailable")
    return guild


async def _readable_message_channel(
    client: discord.Client,
    context: InvocationContext,
    *,
    guild_id: str | None,
    channel_id: str,
) -> tuple[discord.Guild, DiscordMessageChannel]:
    """Enforce effective requester and bot permissions at retrieval time."""

    guild = _requested_guild(client, context, guild_id)
    _assert_agent_channel_scope(context, channel_id)
    resolved_channel_id = _snowflake(channel_id, "channel")
    cached = guild.get_channel_or_thread(resolved_channel_id)
    if isinstance(
        cached,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        channel = cached
    else:
        try:
            fetched = await client.fetch_channel(resolved_channel_id)
        except discord.NotFound as exc:
            raise UserError("discord.message_destination_invalid") from exc
        except discord.Forbidden as exc:
            raise UserError("discord.agent_read_channel_forbidden") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.message_channel_fetch_failed") from exc
        if (
            not isinstance(
                fetched,
                (
                    discord.TextChannel,
                    discord.Thread,
                    discord.VoiceChannel,
                    discord.StageChannel,
                ),
            )
            or fetched.guild.id != guild.id
        ):
            raise UserError("discord.message_destination_invalid")
        channel = fetched
    actor = await _require_common_guild(guild, context)
    bot_member = guild.me
    if (
        bot_member is None
        or not _can_read_messages(channel, actor)
        or not _can_read_private_thread(channel, actor)
        or not _can_read_messages(channel, bot_member)
        or not _can_read_private_thread(channel, bot_member)
    ):
        raise UserError("discord.agent_read_channel_forbidden")
    if context.transport != "agent":
        current_guild = _guild(client, context)
        if guild.id != current_guild.id:
            raise UserError("discord.cross_guild_read_forbidden")
    return guild, channel


async def _require_common_guild(
    guild: discord.Guild,
    context: InvocationContext,
) -> discord.Member:
    actor = await _actor_member(guild, context)
    if guild.me is None:
        raise UserError("discord.guild_unavailable")
    return actor


def _disclosure_to_origin(
    client: discord.Client,
    context: InvocationContext,
    source_guild: discord.Guild,
    source: DiscordReadableChannel,
) -> Literal["same_or_narrower", "broader", "uncertain"]:
    if context.workspace_id is None or context.origin_resource_id is None:
        return "uncertain"
    try:
        destination_guild_id = int(context.workspace_id)
        destination_channel_id = int(context.origin_resource_id)
    except ValueError:
        return "uncertain"
    destination_guild = client.get_guild(destination_guild_id)
    if destination_guild is None:
        return "uncertain"
    destination = destination_guild.get_channel_or_thread(destination_channel_id)
    if not isinstance(
        destination,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        return "uncertain"
    return _disclosure_audience_relation(
        source_guild,
        source,
        destination_guild,
        destination,
    )


def _disclosure_warning(
    relation: Literal["same_or_narrower", "broader", "uncertain"],
) -> str | None:
    if relation == "same_or_narrower":
        return None
    if relation == "broader":
        return (
            "The destination has at least one known reader who cannot read the source. "
            "The enforce policy blocks disclosure unless it is explicitly declassified."
        )
    return (
        "The complete source and destination audiences could not be proven. "
        "The enforce policy fails closed."
    )


def _enforce_information_flow_to_destination(
    client: discord.Client,
    context: InvocationContext,
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
) -> None:
    """Apply every observed source label to the actual write destination."""

    mode = context.information_flow_mode
    if mode == "disabled" or not context.disclosure_observations:
        return
    violations = _information_flow_violations(
        client,
        context.disclosure_observations,
        destination_guild,
        destination,
    )
    _handle_information_flow_violations(
        context,
        destination_guild,
        destination,
        violations,
    )


def _enforce_source_to_destination(
    context: InvocationContext,
    source_guild: discord.Guild,
    source: DiscordReadableChannel,
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
) -> None:
    """Check a source consumed and published inside one composite capability."""

    if context.information_flow_mode == "disabled":
        return
    relation = _disclosure_audience_relation(
        source_guild,
        source,
        destination_guild,
        destination,
    )
    violations: list[tuple[str, str, str]] = (
        []
        if relation == "same_or_narrower"
        else [(str(source_guild.id), str(source.id), relation)]
    )
    _handle_information_flow_violations(
        context,
        destination_guild,
        destination,
        violations,
    )


def _enforce_voice_listener_audience(
    runtime: SimajilordRuntime,
    context: InvocationContext,
    guild: discord.Guild,
    destination: discord.VoiceChannel | discord.StageChannel,
) -> None:
    """Apply the reversible read-aloud policy to one-off synthesized speech."""

    raw_mode = getattr(runtime.settings, "read_aloud_audience_mode", "enforce")
    mode = getattr(raw_mode, "value", raw_mode)
    if mode not in {"enforce", "audit", "disabled"}:
        mode = "enforce"
    if mode == "disabled":
        return
    relation: str = "uncertain"
    if context.workspace_id == str(guild.id) and context.origin_resource_id is not None:
        try:
            source_id = int(context.origin_resource_id)
        except ValueError:
            source_id = 0
        source = guild.get_channel_or_thread(source_id)
        if isinstance(
            source,
            (
                discord.TextChannel,
                discord.Thread,
                discord.VoiceChannel,
                discord.StageChannel,
            ),
        ):
            relation = read_aloud_audience_relation(guild, source, destination)
    if relation == "same_or_narrower":
        return
    log.warning(
        "One-off speech audience check failed mode=%s request=%s "
        "guild=%s source=%s destination=%s relation=%s",
        mode,
        context.request_id,
        guild.id,
        context.origin_resource_id,
        destination.id,
        relation,
    )
    if mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


def _enforce_read_aloud_route_audience(
    runtime: SimajilordRuntime,
    context: InvocationContext,
    guild: discord.Guild,
    sources: tuple[DiscordReadableChannel, ...],
    destination: discord.VoiceChannel | discord.StageChannel,
) -> None:
    """Reject a persistent route whose current listener audience is unproven."""

    raw_mode = getattr(runtime.settings, "read_aloud_audience_mode", "enforce")
    mode = getattr(raw_mode, "value", raw_mode)
    if mode not in {"enforce", "audit", "disabled"}:
        mode = "enforce"
    if mode == "disabled":
        return
    violations = tuple(
        (source.id, inspection.relation)
        for source in sources
        if (
            inspection := inspect_read_aloud_audience(guild, source, destination)
        ).relation
        != "same_or_narrower"
    )
    if not violations:
        return
    log.warning(
        "Read-aloud route audience preflight failed mode=%s request=%s "
        "guild=%s destination=%s sources=%s",
        mode,
        context.request_id,
        guild.id,
        destination.id,
        violations,
    )
    if mode == "enforce":
        raise UserError("read_aloud.audience_forbidden")


def _enforce_information_flow_to_guild(
    context: InvocationContext,
    destination_guild: discord.Guild,
) -> None:
    """Treat guild-wide resources as visible to the guild's broad audience."""

    if context.information_flow_mode == "disabled":
        return
    violations = [
        (
            observation.source_workspace_id,
            observation.source_resource_id,
            "broader" if observation.visibility == "restricted" else "uncertain",
        )
        for observation in context.disclosure_observations
        if observation.source_workspace_id != str(destination_guild.id)
        or observation.visibility != "guild_public"
    ]
    if not violations:
        return
    log.warning(
        "Discord guild-wide information-flow check failed mode=%s request=%s "
        "destination_guild=%s sources=%s",
        context.information_flow_mode,
        context.request_id,
        destination_guild.id,
        violations,
    )
    if context.information_flow_mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


def _enforce_unknown_audience(
    context: InvocationContext,
    *,
    sink: str,
) -> None:
    """Block publication to a global or otherwise unbounded audience."""

    if context.information_flow_mode == "disabled" or not context.disclosure_observations:
        return
    log.warning(
        "Discord unknown-audience information-flow check failed mode=%s "
        "request=%s sink=%s sources=%s",
        context.information_flow_mode,
        context.request_id,
        sink,
        [
            (item.source_workspace_id, item.source_resource_id, item.visibility)
            for item in context.disclosure_observations
        ],
    )
    if context.information_flow_mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


def _enforce_file_provenance_to_guild(
    context: InvocationContext,
    destination_guild: discord.Guild,
    provenance: WorkspaceFileProvenance | None,
) -> None:
    """Require a public same-guild source label before guild-wide file publication."""

    if context.information_flow_mode == "disabled":
        return
    allowed = (
        provenance is not None
        and provenance.declassified_at is not None
    ) or (
        provenance is not None
        and provenance.sensitivity == "guild_public"
        and all(
            workspace_id == str(destination_guild.id)
            and visibility == "guild_public"
            for workspace_id, _resource_id, visibility in provenance.source_resources
        )
        and bool(provenance.source_resources)
    )
    if allowed:
        return
    log.warning(
        "Discord guild-wide file provenance check failed mode=%s request=%s "
        "destination_guild=%s labelled=%s",
        context.information_flow_mode,
        context.request_id,
        destination_guild.id,
        provenance is not None,
    )
    if context.information_flow_mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


def _enforce_file_provenance_to_unknown_audience(
    context: InvocationContext,
    provenance: WorkspaceFileProvenance | None,
    *,
    sink: str,
) -> None:
    """Treat a durable file as restricted until explicitly declassified."""

    if context.information_flow_mode == "disabled" or (
        provenance is not None and provenance.declassified_at is not None
    ):
        return
    log.warning(
        "Discord unknown-audience file provenance check failed mode=%s "
        "request=%s sink=%s labelled=%s",
        context.information_flow_mode,
        context.request_id,
        sink,
        provenance is not None,
    )
    if context.information_flow_mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


def _enforce_file_provenance_to_destination(
    client: discord.Client,
    context: InvocationContext,
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
    provenances: tuple[WorkspaceFileProvenance | None, ...],
) -> None:
    """Fail closed when durable file sources cannot flow to the actual target."""

    if context.information_flow_mode == "disabled":
        return
    violations: list[tuple[str, str, str]] = []
    observations: list[DisclosureObservation] = []
    for provenance in provenances:
        if provenance is None:
            violations.append(("file", "unlabelled", "uncertain"))
            continue
        if provenance.declassified_at is not None:
            continue
        if provenance.unlabelled_input or provenance.sources_truncated:
            violations.append(("file", "provenance", "uncertain"))
        if provenance.sensitivity == "actor_private":
            same_origin = (
                provenance.owner_actor_ids == (context.actor_id,)
                and provenance.origin_guild_id == str(destination_guild.id)
                and provenance.origin_channel_id == str(destination.id)
            )
            if not same_origin:
                violations.append(
                    (
                        provenance.origin_guild_id or "file",
                        provenance.origin_channel_id or "actor_private",
                        "broader",
                    )
                )
            if not provenance.source_resources:
                continue
        file_observations = provenance_observations(provenance)
        if not file_observations:
            violations.append(
                (
                    provenance.origin_guild_id or "file",
                    provenance.origin_channel_id or "unknown",
                    "uncertain",
                )
            )
        observations.extend(file_observations)
    violations.extend(
        _information_flow_violations(
            client,
            tuple(dict.fromkeys(observations)),
            destination_guild,
            destination,
        )
    )
    _handle_information_flow_violations(
        context,
        destination_guild,
        destination,
        violations,
    )


def _information_flow_violations(
    client: discord.Client,
    observations: tuple[DisclosureObservation, ...],
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
) -> list[tuple[str, str, str]]:
    violations: list[tuple[str, str, str]] = []
    for observation in observations:
        try:
            source_guild_id = int(observation.source_workspace_id)
            source_channel_id = int(observation.source_resource_id)
        except ValueError:
            relation = "uncertain"
        else:
            source_guild = client.get_guild(source_guild_id)
            source = (
                source_guild.get_channel_or_thread(source_channel_id)
                if source_guild is not None
                else None
            )
            if source_guild is None or not isinstance(
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
                relation = _disclosure_audience_relation(
                    source_guild,
                    source,
                    destination_guild,
                    destination,
                )
        if relation != "same_or_narrower":
            violations.append(
                (
                    observation.source_workspace_id,
                    observation.source_resource_id,
                    relation,
                )
            )
    return violations


def _handle_information_flow_violations(
    context: InvocationContext,
    destination_guild: discord.Guild,
    destination: DiscordReadableChannel,
    violations: list[tuple[str, str, str]],
) -> None:
    if not violations:
        return
    mode = context.information_flow_mode
    log.warning(
        "Discord information-flow destination check failed mode=%s request=%s "
        "destination=%s/%s sources=%s",
        mode,
        context.request_id,
        destination_guild.id,
        destination.id,
        violations,
    )
    if mode == "enforce":
        raise UserError("discord.information_flow_forbidden")


async def _actor_member(
    guild: discord.Guild,
    context: InvocationContext,
) -> discord.Member:
    try:
        actor_id = int(context.actor_id)
    except ValueError as exc:
        raise UserError("discord.member_required") from exc
    member = guild.get_member(actor_id)
    if member is not None:
        return member
    try:
        member = await guild.fetch_member(actor_id)
    except (discord.NotFound, discord.Forbidden) as exc:
        raise UserError("discord.member_required") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.member_lookup_failed") from exc
    if member.id != actor_id:
        raise UserError("discord.member_required")
    return member


def _member_voice_channel(
    member: discord.Member,
) -> discord.VoiceChannel | discord.StageChannel | None:
    state = member.voice
    if state is None or not isinstance(
        state.channel,
        (discord.VoiceChannel, discord.StageChannel),
    ):
        return None
    return state.channel


def _voice_state_record(member: discord.Member) -> DiscordVoiceStateRecord:
    state = member.voice
    if state is None or not isinstance(
        state.channel,
        (discord.VoiceChannel, discord.StageChannel),
    ):
        raise ValueError("member is not connected to a visible voice channel")
    channel = state.channel
    return DiscordVoiceStateRecord(
        user_id=str(member.id),
        display_name=member.display_name,
        bot=member.bot,
        channel_id=str(channel.id),
        channel_name=channel.name,
        channel_kind=str(channel.type),
        category_id=(
            str(channel.category_id)
            if channel.category_id is not None
            else None
        ),
        server_muted=state.mute,
        server_deafened=state.deaf,
        self_muted=state.self_mute,
        self_deafened=state.self_deaf,
        streaming=state.self_stream,
        video=state.self_video,
        suppressed=state.suppress,
        afk=state.afk,
        requested_to_speak_at_iso=(
            state.requested_to_speak_at.isoformat()
            if state.requested_to_speak_at is not None
            else None
        ),
    )


def _activity_record(activity: object) -> DiscordActivityRecord:
    def optional_text(value: object) -> str | None:
        return str(value) if value is not None and str(value) else None

    def optional_timestamp(value: object) -> str | None:
        return value.isoformat() if isinstance(value, datetime) else None

    activity_type = getattr(activity, "type", None)
    activity_type_name = getattr(activity_type, "name", None)
    return DiscordActivityRecord(
        name=optional_text(getattr(activity, "name", None)) or "Unknown activity",
        type=(
            activity_type_name
            if isinstance(activity_type_name, str) and activity_type_name
            else optional_text(activity_type) or type(activity).__name__
        ),
        details=optional_text(getattr(activity, "details", None)),
        state=optional_text(getattr(activity, "state", None)),
        url=optional_text(getattr(activity, "url", None)),
        application_id=optional_text(getattr(activity, "application_id", None)),
        created_at_iso=optional_timestamp(getattr(activity, "created_at", None)),
        start_iso=optional_timestamp(getattr(activity, "start", None)),
        end_iso=optional_timestamp(getattr(activity, "end", None)),
        emoji=optional_text(getattr(activity, "emoji", None)),
        platform=optional_text(getattr(activity, "platform", None)),
        session_id=optional_text(getattr(activity, "session_id", None)),
        sync_id=optional_text(getattr(activity, "sync_id", None)),
        details_url=optional_text(getattr(activity, "details_url", None)),
        state_url=optional_text(getattr(activity, "state_url", None)),
        large_image_url=optional_text(getattr(activity, "large_image_url", None)),
        large_image_text=optional_text(getattr(activity, "large_image_text", None)),
        small_image_url=optional_text(getattr(activity, "small_image_url", None)),
        small_image_text=optional_text(getattr(activity, "small_image_text", None)),
        buttons=tuple(
            str(button)
            for button in (getattr(activity, "buttons", None) or ())
        ),
        party=optional_text(getattr(activity, "party", None)),
        flags=optional_text(getattr(activity, "flags", None)),
        status_display_type=optional_text(
            getattr(activity, "status_display_type", None)
        ),
        title=optional_text(getattr(activity, "title", None)),
        artist=optional_text(getattr(activity, "artist", None)),
        album=optional_text(getattr(activity, "album", None)),
        track_url=optional_text(getattr(activity, "track_url", None)),
    )


def _enabled_flag_names(flags: object) -> tuple[str, ...]:
    try:
        values: tuple[object, ...] = tuple(cast(Iterable[object], flags))
    except TypeError:
        return ()
    return tuple(
        str(name)
        for value in values
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance((name := value[0]), str)
            and value[1] is True
        )
    )


def _can_view_channel(
    channel: discord.abc.GuildChannel | discord.Thread,
    member: discord.Member,
) -> bool:
    permissions = channel.permissions_for(member)
    return _permission_enabled(permissions, "administrator") or _permission_enabled(
        permissions,
        "view_channel",
    )


def _assert_same_voice(
    destination_id: str | None,
    channel: discord.VoiceChannel | discord.StageChannel | None,
) -> None:
    if channel is None or destination_id is None or str(channel.id) != destination_id:
        raise UserError("audio.same_voice_required")


def _require_manage_guild(member: discord.Member) -> None:
    permissions = member.guild_permissions
    if not _permission_enabled(
        permissions,
        "administrator",
    ) and not _permission_enabled(permissions, "manage_guild"):
        raise UserError("discord.manage_guild_required")


async def _prepare_actor_audio(
    client: discord.Client,
    runtime: SimajilordRuntime,
    guild: discord.Guild,
    member: discord.Member,
) -> None:
    workspace_id = str(guild.id)
    session = runtime.audio.get_or_create(
        workspace_id,
        lambda: DiscordAudioOutput(client, guild.id),
    )
    channel = _member_voice_channel(member)
    if session.output.connected:
        _assert_same_voice(session.destination_id, channel)
        return
    if channel is not None:
        await runtime.audio.connect(workspace_id, str(channel.id))


def _snowflake(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        code = {
            "message": "discord.message_id_invalid",
            "channel": "discord.channel_id_invalid",
            "voice channel": "discord.voice_channel_id_invalid",
            "guild": "discord.guild_id_invalid",
            "user": "discord.user_id_invalid",
            "role": "discord.role_id_invalid",
        }.get(label, "discord.snowflake_invalid")
        raise UserError(code) from exc


def _search_boundary_message_id(
    message_id: str | None,
    timestamp_iso: str | None,
    *,
    high: bool,
) -> str | None:
    if message_id is not None and timestamp_iso is not None:
        raise UserError("discord.message_search_range_invalid")
    if message_id is not None:
        _snowflake(message_id, "message")
        return message_id
    if timestamp_iso is None:
        return None
    try:
        timestamp = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError("discord.message_search_time_invalid") from exc
    if timestamp.tzinfo is None:
        raise UserError("discord.message_search_time_invalid")
    return str(discord.utils.time_snowflake(timestamp.astimezone(UTC), high=high))


def _archived_threads_before(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError("discord.archived_thread_time_invalid") from exc
    if timestamp.tzinfo is None:
        raise UserError("discord.archived_thread_time_invalid")
    return timestamp.astimezone(UTC)


def _message_search_cursor_fingerprint(
    *,
    guild_id: str,
    channel_batches: tuple[tuple[str, ...], ...],
    content: str,
    author_ids: tuple[str, ...],
    before_message_id: str | None,
    after_message_id: str | None,
    sort_by: str,
    sort_order: str,
) -> str:
    canonical = json.dumps(
        {
            "guild_id": guild_id,
            "channel_batches": channel_batches,
            "content": content,
            "author_ids": author_ids,
            "before_message_id": before_message_id,
            "after_message_id": after_message_id,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _encode_message_search_cursor(
    *,
    fingerprint: str,
    offsets: tuple[int, ...],
    next_batch_index: int,
) -> str:
    payload = json.dumps(
        {
            "v": 1,
            "fingerprint": fingerprint,
            "offsets": offsets,
            "next_batch_index": next_batch_index,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_message_search_cursor(
    value: str,
    *,
    expected_fingerprint: str,
    batch_count: int,
) -> tuple[tuple[int, ...], int]:
    try:
        padding = "=" * (-len(value) % 4)
        payload = json.loads(
            base64.b64decode(
                value + padding,
                altchars=b"-_",
                validate=True,
            )
        )
        offsets = payload["offsets"]
        next_batch_index = payload["next_batch_index"]
        if (
            payload.get("v") != 1
            or payload.get("fingerprint") != expected_fingerprint
            or not isinstance(offsets, list)
            or len(offsets) != batch_count
            or any(
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not 0 <= offset <= 9_975
                for offset in offsets
            )
            or not isinstance(next_batch_index, int)
            or isinstance(next_batch_index, bool)
            or not 0 <= next_batch_index < batch_count
        ):
            raise ValueError("cursor payload mismatch")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise UserError("discord.message_search_cursor_invalid") from exc
    return tuple(offsets), next_batch_index


def _search_thread_parents(
    payload: dict[str, Any],
) -> dict[str, tuple[str, int | None]]:
    raw_threads = payload.get("threads")
    if not isinstance(raw_threads, list):
        return {}
    parents: dict[str, tuple[str, int | None]] = {}
    for raw_thread in raw_threads:
        if not isinstance(raw_thread, dict):
            continue
        thread_id = raw_thread.get("id")
        parent_id = raw_thread.get("parent_id")
        raw_type = raw_thread.get("type")
        if not isinstance(thread_id, str) or not isinstance(parent_id, str):
            continue
        parents[thread_id] = (
            parent_id,
            raw_type
            if isinstance(raw_type, int) and not isinstance(raw_type, bool)
            else None,
        )
    return parents


def _search_result_source(
    guild: discord.Guild,
    *,
    channel_id: str,
    readable_ids: set[str] | None,
    actor: discord.Member | None,
    raw_thread_parents: dict[str, tuple[str, int | None]],
) -> DiscordReadableChannel | None:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if isinstance(
        channel,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ) and (readable_ids is None or channel_id in readable_ids):
        return channel
    parent_record = raw_thread_parents.get(channel_id)
    if parent_record is None:
        return None
    parent_id, raw_type = parent_record
    if readable_ids is not None and parent_id not in readable_ids:
        return None
    parent = guild.get_channel(_snowflake(parent_id, "channel"))
    if not isinstance(parent, (discord.TextChannel, discord.ForumChannel)):
        return None
    if raw_type == discord.ChannelType.private_thread.value:
        if actor is None:
            return None
        permissions = parent.permissions_for(actor)
        if not (
            _permission_enabled(permissions, "administrator")
            or _permission_enabled(permissions, "manage_threads")
        ):
            return None
    return parent


def _assert_origin_guild(
    context: InvocationContext,
    guild: discord.Guild,
) -> None:
    if isinstance(guild.id, int) and context.workspace_id != str(guild.id):
        raise UserError("discord.agent_write_cross_guild_forbidden")


async def _write_members(
    guild: discord.Guild,
    context: InvocationContext,
) -> tuple[discord.Member, discord.Member]:
    actor = await _actor_member(guild, context)
    bot = guild.me
    if bot is None:
        raise UserError("discord.bot_member_required")
    return actor, bot


async def _write_message_channel(
    client: discord.Client,
    context: InvocationContext,
    channel_id: str,
    *,
    guild_id: str | None = None,
    required_permissions: tuple[str, ...] = (),
) -> tuple[discord.Guild, DiscordMessageChannel, discord.Member, discord.Member]:
    guild = _write_guild_for_channel(
        client,
        context,
        channel_id=channel_id,
        requested_guild_id=guild_id,
    )
    _assert_agent_update_scope(context, channel_id)
    channel = _message_channel(guild, channel_id)
    _enforce_information_flow_to_destination(client, context, guild, channel)
    actor, bot = await _write_members(guild, context)
    for member in (actor, bot):
        if not _can_read_messages(channel, member) or not _can_read_private_thread(
            channel, member
        ):
            raise UserError("discord.agent_write_channel_forbidden")
        for permission in required_permissions:
            _require_channel_permissions(channel, member, permission)
    return guild, channel, actor, bot


def _write_guild_for_channel(
    client: discord.Client,
    context: InvocationContext,
    *,
    channel_id: str,
    requested_guild_id: str | None,
) -> discord.Guild:
    """Infer an omitted cross-guild target from Discord's globally unique channel ID."""

    origin_or_requested = _requested_guild(client, context, requested_guild_id)
    if requested_guild_id is not None or context.transport != "agent":
        return origin_or_requested
    resolved_channel_id = _snowflake(channel_id, "channel")
    if origin_or_requested.get_channel_or_thread(resolved_channel_id) is not None:
        return origin_or_requested
    cached = client.get_channel(resolved_channel_id)
    if isinstance(cached, (discord.abc.GuildChannel, discord.Thread)):
        return cached.guild
    return origin_or_requested


async def _guild_member(guild: discord.Guild, user_id: str) -> discord.Member:
    member_id = _snowflake(user_id, "user")
    member = guild.get_member(member_id)
    if member is None:
        try:
            member = await guild.fetch_member(member_id)
        except (discord.NotFound, discord.Forbidden) as exc:
            raise UserError("discord.member_required") from exc
        except discord.DiscordException as exc:
            raise UserError("discord.member_lookup_failed") from exc
    return member


def _require_channel_permissions(
    channel: (
        DiscordMessageChannel
        | discord.ForumChannel
        | discord.CategoryChannel
    ),
    member: discord.Member,
    permission: str,
) -> None:
    permissions = channel.permissions_for(member)
    if _permission_enabled(permissions, "administrator"):
        return
    effective = permission
    if permission == "send_messages" and isinstance(channel, discord.Thread):
        effective = "send_messages_in_threads"
    if not _permission_enabled(
        permissions,
        "view_channel",
    ) or not _permission_enabled(permissions, effective):
        raise UserError(f"discord.{permission}_required")


def _require_guild_permission(member: discord.Member, permission: str) -> None:
    permissions = member.guild_permissions
    if not _permission_enabled(
        permissions,
        "administrator",
    ) and not _permission_enabled(permissions, permission):
        raise UserError(f"discord.{permission}_required")


def _discord_write_nonce(context: InvocationContext, purpose: str) -> str:
    """Deduplicate one exact model tool call without merging intentional calls."""

    invocation_id = context.tool_call_id or context.request_id
    digest = hashlib.sha256(
        f"{purpose}\0{invocation_id}".encode()
    ).hexdigest()
    return f"sla{digest[:22]}"


def _require_role_above(member: discord.Member, role: discord.Role) -> None:
    # Administrator bypasses channel overwrites, but Discord role hierarchy
    # still prevents every non-owner from managing an equal or higher role.
    if member.guild.owner_id == member.id:
        return
    if member.top_role <= role:
        raise UserError("discord.role_hierarchy_forbidden")


def _role_assignable_by(member: discord.Member, role: discord.Role) -> bool:
    if role.is_default() or role.managed:
        return False
    permissions = member.guild_permissions
    if member.guild.owner_id == member.id:
        return True
    can_manage = _permission_enabled(
        permissions,
        "administrator",
    ) or _permission_enabled(permissions, "manage_roles")
    return can_manage and member.top_role > role


def _require_member_below(
    member: discord.Member,
    target: discord.Member,
    guild: discord.Guild,
) -> None:
    if target.id == guild.owner_id or target.id == member.id:
        raise UserError("discord.member_hierarchy_forbidden")
    if member.id == guild.owner_id:
        return
    if member.top_role <= target.top_role:
        raise UserError("discord.member_hierarchy_forbidden")


async def _fetch_message_for_write(
    channel: DiscordMessageChannel,
    message_id: str | None,
) -> discord.Message:
    if message_id is None:
        raise UserError("discord.message_id_invalid")
    try:
        return await channel.fetch_message(_snowflake(message_id, "message"))
    except discord.NotFound as exc:
        raise UserError("discord.message_not_found") from exc
    except discord.Forbidden as exc:
        raise UserError("discord.message_write_forbidden") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.message_fetch_failed") from exc


def _bounded_name(value: str, code: str) -> str:
    name = value.strip()
    if not 1 <= len(name) <= 100:
        raise UserError(code)
    return name


def _undo_state_fingerprint(payload: object) -> str:
    """Hash a small live-state projection without retaining message bodies."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _thread_undo_fingerprint(thread: discord.Thread) -> str:
    applied_tag_ids = sorted(
        str(tag.id)
        for tag in getattr(thread, "applied_tags", ())
    )
    return _undo_state_fingerprint(
        {
            "name": thread.name,
            "archived": bool(thread.archived),
            "locked": bool(getattr(thread, "locked", False)),
            "invitable": bool(getattr(thread, "invitable", True)),
            "auto_archive_duration": getattr(
                thread,
                "auto_archive_duration",
                None,
            ),
            "slowmode_delay": getattr(thread, "slowmode_delay", 0),
            "last_message_id": (
                str(thread.last_message_id)
                if thread.last_message_id is not None
                else None
            ),
            "applied_tag_ids": applied_tag_ids,
        }
    )


def _role_undo_fingerprint(role: discord.Role) -> str:
    display_icon = getattr(role, "display_icon", None)
    return _undo_state_fingerprint(
        {
            "name": role.name,
            "colour": role.colour.value,
            "permissions": role.permissions.value,
            "hoist": bool(role.hoist),
            "mentionable": bool(role.mentionable),
            "display_icon": str(display_icon) if display_icon is not None else None,
        }
    )


async def _role_has_channel_overwrite_reference(
    guild: discord.Guild,
    role: discord.Role,
) -> bool:
    """Check the complete live set of overwrite-bearing guild channels.

    Discord threads do not have independent permission overwrites; they inherit
    the parent text/forum/media channel. ``fetch_channels`` therefore covers the
    relevant state for categories, thread parents, text, voice, stage, and forum
    channels without trusting the gateway cache.
    """

    try:
        channels = await guild.fetch_channels()
    except discord.DiscordException as exc:
        raise UserError("action.undo_target_state_uncertain") from exc
    try:
        for channel in channels:
            overwrites = channel.overwrites
            if not isinstance(overwrites, Mapping):
                raise TypeError("channel overwrites are not a mapping")
            for target in overwrites:
                if getattr(target, "id", None) != role.id:
                    continue
                if isinstance(target, discord.Role):
                    return True
                if isinstance(target, discord.Object):
                    target_type = target.type
                    if target_type is discord.Role:
                        return True
                    if target_type in {discord.Member, discord.User}:
                        continue
                elif isinstance(target, (discord.Member, discord.User)):
                    continue
                raise TypeError("overwrite target type is unresolved")
    except (AttributeError, TypeError, ValueError) as exc:
        raise UserError("action.undo_target_state_uncertain") from exc
    return False


def _channel_undo_fingerprint(channel: discord.TextChannel) -> str:
    overwrites: list[tuple[str, str, int, int]] = []
    for target, overwrite in channel.overwrites.items():
        allow, deny = overwrite.pair()
        overwrites.append(
            (
                "role" if isinstance(target, discord.Role) else "member",
                str(target.id),
                allow.value,
                deny.value,
            )
        )
    return _undo_state_fingerprint(
        {
            "name": channel.name,
            "topic": channel.topic,
            "slowmode_delay": channel.slowmode_delay,
            "nsfw": bool(channel.nsfw),
            "category_id": (
                str(channel.category_id)
                if channel.category_id is not None
                else None
            ),
            "default_auto_archive_duration": getattr(
                channel,
                "default_auto_archive_duration",
                None,
            ),
            "default_thread_slowmode_delay": getattr(
                channel,
                "default_thread_slowmode_delay",
                0,
            ),
            "overwrites": sorted(overwrites),
        }
    )


def _audit_reason(reason: str, context: InvocationContext) -> str:
    text = " ".join(reason.split())
    if len(text) > 400:
        raise UserError("discord.audit_reason_too_long")
    return f"Simajilord actor={context.actor_id} request={context.request_id}: {text}"[:512]


def _require_moderation_reason(reason: str) -> None:
    if not reason.strip():
        raise UserError("discord.moderation_reason_required")


def _timeout_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError("discord.timeout_invalid") from exc
    if parsed.tzinfo is None:
        raise UserError("discord.timeout_invalid")
    parsed = parsed.astimezone(UTC)
    now = datetime.now(UTC)
    if not now < parsed <= now + timedelta(days=28):
        raise UserError("discord.timeout_invalid")
    return parsed


def _timeout_state_datetime(value: str | None) -> datetime | None:
    """Parse a persisted current-state guard without applying future-time limits."""

    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UserError("discord.timeout_invalid") from exc
    if parsed.tzinfo is None:
        raise UserError("discord.timeout_invalid")
    return parsed.astimezone(UTC)


def _reaction_emoji(value: str) -> str:
    emoji = value.strip()
    if not emoji or len(emoji) > 100 or any(character.isspace() for character in emoji):
        raise UserError("discord.reaction_emoji_invalid")
    return emoji


def _bot_has_reaction(message: discord.Message, emoji: str) -> bool:
    reactions = getattr(message, "reactions", ())
    if not isinstance(reactions, (list, tuple)):
        return False
    return any(
        reaction.me and str(reaction.emoji) == emoji
        for reaction in reactions
    )


def _assert_agent_update_scope(
    context: InvocationContext,
    channel_id: str,
) -> None:
    _assert_agent_channel_scope(context, channel_id)


def _text_channel(
    guild: discord.Guild,
    channel_id: str,
) -> discord.TextChannel | discord.Thread:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if not isinstance(channel, (discord.TextChannel, discord.Thread)):
        raise UserError("discord.text_destination_invalid")
    return channel


def _message_channel(
    guild: discord.Guild,
    channel_id: str,
) -> DiscordMessageChannel:
    channel = guild.get_channel_or_thread(_snowflake(channel_id, "channel"))
    if not isinstance(
        channel,
        (
            discord.TextChannel,
            discord.Thread,
            discord.VoiceChannel,
            discord.StageChannel,
        ),
    ):
        raise UserError("discord.message_destination_invalid")
    return channel


def _message_record(
    message: discord.Message,
    *,
    guild_id: str,
    visibility: Literal["guild_public", "restricted", "uncertain"],
    disclosure_to_origin: Literal["same_or_narrower", "broader", "uncertain"],
    event_message_id: str | None = None,
) -> DiscordMessageRecord:
    context_text = _message_context_text(message)
    if str(message.id) == event_message_id:
        preview, truncated = _bounded_event_message(context_text)
    else:
        preview, truncated = _message_preview(context_text)
    return DiscordMessageRecord(
        message_id=str(message.id),
        channel_id=str(message.channel.id),
        guild_id=guild_id,
        visibility=visibility,
        disclosure_to_origin=disclosure_to_origin,
        disclosure_warning=_disclosure_warning(disclosure_to_origin),
        author_id=str(message.author.id),
        author_name=message.author.display_name,
        author_is_bot=message.author.bot,
        content_preview=preview,
        content_length=len(context_text),
        preview_truncated=truncated,
        created_at_iso=message.created_at.isoformat(),
        attachments=tuple(_attachment_record(attachment) for attachment in message.attachments),
        reference_message_id=(
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        ),
        edited_at_iso=_message_edited_at_iso(message),
        reaction_count=sum(
            reaction.count for reaction in getattr(message, "reactions", ())
        ),
        reaction_summary=tuple(
            DiscordReactionSummaryRecord(str(reaction.emoji), reaction.count)
            for reaction in getattr(message, "reactions", ())[:10]
        ),
        thread_id=_message_thread_id(message),
        poll=_expanded_poll(message.poll),
    )


def _attachment_record(attachment: discord.Attachment) -> DiscordAttachmentRecord:
    return DiscordAttachmentRecord(
        attachment_id=str(attachment.id),
        filename=attachment.filename,
        content_type=attachment.content_type,
        size_bytes=attachment.size,
    )


def _message_edited_at_iso(message: discord.Message) -> str | None:
    edited_at = getattr(message, "edited_at", None)
    return edited_at.isoformat() if isinstance(edited_at, datetime) else None


def _message_thread_id(message: discord.Message) -> str | None:
    thread = getattr(message, "thread", None)
    return str(thread.id) if isinstance(thread, discord.Thread) else None


def _workspace_attachment_name(attachment: discord.Attachment) -> str:
    """Keep separate same-named uploads collision-free and within sandbox limits."""

    filename = PurePath(attachment.filename).name or "attachment"
    prefix = f"{attachment.id}-"
    maximum_filename_characters = 180 - len(prefix)
    if len(filename) > maximum_filename_characters:
        suffix = PurePath(filename).suffix[:20]
        stem_limit = max(1, maximum_filename_characters - len(suffix))
        filename = f"{filename[:stem_limit]}{suffix}"
    return f"{prefix}{filename}"


def _search_message_record(value: object) -> DiscordMessageRecord | None:
    if not isinstance(value, dict):
        return None
    message_id = value.get("id")
    channel_id = value.get("channel_id")
    author = value.get("author")
    timestamp = value.get("timestamp")
    if not (
        isinstance(message_id, str)
        and message_id.isdecimal()
        and isinstance(channel_id, str)
        and channel_id.isdecimal()
        and isinstance(author, dict)
        and isinstance(timestamp, str)
    ):
        return None
    author_id = author.get("id")
    if not isinstance(author_id, str):
        return None
    content = value.get("content")
    text = content if isinstance(content, str) else ""
    preview, truncated = _message_preview(text)
    raw_attachments = value.get("attachments")
    attachments: list[DiscordAttachmentRecord] = []
    if isinstance(raw_attachments, list):
        for raw_attachment in raw_attachments[:10]:
            if not isinstance(raw_attachment, dict):
                continue
            attachment_id = raw_attachment.get("id")
            filename = raw_attachment.get("filename")
            size = raw_attachment.get("size")
            content_type = raw_attachment.get("content_type")
            if not (
                isinstance(attachment_id, str)
                and isinstance(filename, str)
                and isinstance(size, int)
                and not isinstance(size, bool)
            ):
                continue
            attachments.append(
                DiscordAttachmentRecord(
                    attachment_id=attachment_id,
                    filename=filename,
                    content_type=content_type if isinstance(content_type, str) else None,
                    size_bytes=max(0, size),
                )
            )
    reference = value.get("message_reference")
    reference_message_id = (
        reference.get("message_id") if isinstance(reference, dict) else None
    )
    display_name = author.get("global_name") or author.get("username") or author_id
    raw_reactions = value.get("reactions")
    reaction_summary: list[DiscordReactionSummaryRecord] = []
    if isinstance(raw_reactions, list):
        for raw_reaction in raw_reactions[:10]:
            if not isinstance(raw_reaction, dict):
                continue
            count = raw_reaction.get("count")
            emoji = raw_reaction.get("emoji")
            if not isinstance(count, int) or isinstance(count, bool) or not isinstance(
                emoji, dict
            ):
                continue
            emoji_name = emoji.get("name")
            emoji_id = emoji.get("id")
            if not isinstance(emoji_name, str):
                continue
            rendered = (
                f"<:{emoji_name}:{emoji_id}>"
                if isinstance(emoji_id, str)
                else emoji_name
            )
            reaction_summary.append(
                DiscordReactionSummaryRecord(rendered, max(0, count))
            )
    thread = value.get("thread")
    raw_thread_id = thread.get("id") if isinstance(thread, dict) else None
    return DiscordMessageRecord(
        message_id=message_id,
        channel_id=channel_id,
        guild_id="",
        visibility="uncertain",
        disclosure_to_origin="uncertain",
        disclosure_warning=_disclosure_warning("uncertain"),
        author_id=author_id,
        author_name=str(display_name),
        author_is_bot=bool(author.get("bot")),
        content_preview=preview,
        content_length=len(text),
        preview_truncated=truncated,
        created_at_iso=timestamp,
        attachments=tuple(attachments),
        reference_message_id=(
            reference_message_id if isinstance(reference_message_id, str) else None
        ),
        edited_at_iso=(
            str(value["edited_timestamp"])
            if isinstance(value.get("edited_timestamp"), str)
            else None
        ),
        reaction_count=sum(item.count for item in reaction_summary),
        reaction_summary=tuple(reaction_summary),
        thread_id=raw_thread_id if isinstance(raw_thread_id, str) else None,
    )


def _custom_emoji_records(content: str) -> tuple[DiscordCustomEmojiRecord, ...]:
    """Return bounded, de-duplicated metadata in first-appearance order."""

    ordered: list[tuple[str, str, bool]] = []
    occurrence_counts: dict[str, int] = {}
    for match in _CUSTOM_EMOJI_PATTERN.finditer(content):
        emoji_id = match.group("id")
        occurrence_counts[emoji_id] = occurrence_counts.get(emoji_id, 0) + 1
        if occurrence_counts[emoji_id] == 1 and len(ordered) < _CUSTOM_EMOJI_PREVIEW_LIMIT:
            ordered.append(
                (
                    emoji_id,
                    match.group("name"),
                    match.group("animated") == "a",
                )
            )
    return tuple(
        DiscordCustomEmojiRecord(
            index=index,
            emoji_id=emoji_id,
            name=name,
            animated=animated,
            occurrences=occurrence_counts[emoji_id],
        )
        for index, (emoji_id, name, animated) in enumerate(ordered)
    )


def _sticker_records(
    stickers: list[discord.StickerItem],
) -> tuple[DiscordStickerRecord, ...]:
    return tuple(
        DiscordStickerRecord(
            index=index,
            sticker_id=str(sticker.id),
            name=sticker.name,
            format=sticker.format.name,
            animated=sticker.format is not discord.StickerFormatType.png,
        )
        for index, sticker in enumerate(stickers[:10])
    )


def _prepare_discord_animated_media(
    content: bytes,
    *,
    mode: Literal["preview", "animation", "frame"],
    frame_index: int,
) -> _DiscordAnimatedMedia:
    """Validate model media and optionally extract an exact animation frame."""

    try:
        with Image.open(io.BytesIO(content)) as image:
            image_format = image.format
            frame_count = getattr(image, "n_frames", 1)
            durations: list[int] = []
            for index in range(frame_count):
                image.seek(index)
                raw_duration = image.info.get("duration")
                durations.append(int(raw_duration) if isinstance(raw_duration, (int, float)) else 0)
            duration_ms = sum(durations) or None
            if mode == "animation":
                if image_format == "GIF":
                    content_type = "image/gif"
                elif image_format == "PNG" and frame_count > 1:
                    content_type = "image/apng"
                else:
                    raise ValueError("discord.custom_emoji_invalid")
                return _DiscordAnimatedMedia(
                    content=content,
                    content_type=content_type,
                    preview_kind="full_animation",
                    frame_index=None,
                    frame_count=frame_count,
                    duration_ms=duration_ms,
                )
            selected_frame = frame_index if mode == "frame" else 0
            if selected_frame >= frame_count:
                raise ValueError("discord.custom_emoji_frame_invalid")
            image.seek(selected_frame)
            output = io.BytesIO()
            image.convert("RGBA").save(output, format="PNG", optimize=True)
            return _DiscordAnimatedMedia(
                content=output.getvalue(),
                content_type="image/png",
                preview_kind=(
                    "selected_animation_frame" if mode == "frame" else "representative_static_frame"
                ),
                frame_index=selected_frame,
                frame_count=frame_count,
                duration_ms=duration_ms,
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("discord.custom_emoji_invalid") from exc


def parse_discord_message_link(content: str) -> DiscordMessageLink | None:
    """Parse one bare Discord message link; angle brackets are an explicit opt-out."""

    value = content.strip()
    if not value or value.startswith("<") or value.endswith(">"):
        return None
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or (parsed.hostname or "").lower()
        not in {"discord.com", "ptb.discord.com", "canary.discord.com"}
    ):
        return None
    path = parsed.path.rstrip("/").split("/")
    if len(path) != 5 or path[1] != "channels":
        return None
    guild_id, channel_id, message_id = path[2:]
    if not all(value.isdigit() and int(value) > 0 for value in path[2:]):
        return None
    return DiscordMessageLink(
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
    )


def _expanded_attachment(
    attachment: discord.Attachment,
) -> DiscordExpandedAttachmentRecord:
    return DiscordExpandedAttachmentRecord(
        filename=attachment.filename,
        content_type=attachment.content_type,
        size_bytes=attachment.size,
        url=attachment.url,
        proxy_url=attachment.proxy_url,
        spoiler=attachment.is_spoiler(),
    )


def _expanded_embed(item: discord.Embed) -> DiscordExpandedEmbedRecord:
    return DiscordExpandedEmbedRecord(
        title=item.title,
        description=item.description,
        url=item.url,
        image_url=item.image.url if item.image.url else None,
        thumbnail_url=item.thumbnail.url if item.thumbnail.url else None,
    )


def _expanded_poll(poll: discord.Poll | None) -> DiscordExpandedPollRecord | None:
    if poll is None:
        return None
    answers = tuple(
        DiscordExpandedPollAnswerRecord(
            answer_id=str(answer.id),
            text=answer.text,
            emoji=str(answer.emoji) if answer.emoji is not None else None,
            vote_count=answer.vote_count,
            bot_voted=answer.self_voted,
            victor=answer.victor,
        )
        for answer in poll.answers
    )
    finalized_value = poll.is_finalized()
    expires_at = poll.expires_at
    layout_name = getattr(poll.layout_type, "name", None)
    return DiscordExpandedPollRecord(
        question=poll.question,
        answers=answers,
        total_vote_count=poll.total_votes,
        multiple=poll.multiple,
        expires_at_iso=expires_at.isoformat() if expires_at is not None else None,
        duration_seconds=int(poll.duration.total_seconds()),
        finalized=finalized_value,
        counts_are_exact=finalized_value,
        victor_answer_id=(
            str(poll.victor_answer_id)
            if poll.victor_answer_id is not None
            else None
        ),
        layout_type=(
            str(layout_name)
            if layout_name is not None
            else str(poll.layout_type)
        ),
    )


def discord_translation_segments(
    message: discord.Message,
) -> tuple[TranslationSegmentItem, ...]:
    """Extract translatable text in Discord display order with stable paths."""

    output: list[TranslationSegmentItem] = []

    def append(identifier: str, text: str | None) -> None:
        value = text.strip() if text is not None else ""
        if value:
            output.append(TranslationSegmentItem(identifier=identifier, text=value))

    primary_text = message.content
    if not primary_text.strip() and message.is_system():
        # System messages such as member join notices render through
        # ``system_content`` while their regular content remains empty.
        # Retain the stable ``content`` path for the translated rendering.
        primary_text = message.system_content
    append("content", primary_text)
    for embed_index, item in enumerate(message.embeds[:10]):
        append(f"embed.{embed_index}.author", item.author.name)
        append(f"embed.{embed_index}.title", item.title)
        append(f"embed.{embed_index}.description", item.description)
        for field_index, field in enumerate(item.fields[:25]):
            append(f"embed.{embed_index}.field.{field_index}.name", field.name)
            append(f"embed.{embed_index}.field.{field_index}.value", field.value)
        append(f"embed.{embed_index}.footer", item.footer.text)
    if message.poll is not None:
        append("poll.question", message.poll.question)
        for answer_index, answer in enumerate(message.poll.answers):
            append(f"poll.answer.{answer_index}", answer.text)
    for component_index, component in enumerate(message.components):
        _append_component_text(
            component.to_dict(),
            path=f"component.{component_index}",
            append=append,
        )
    for attachment_index, attachment in enumerate(message.attachments[:10]):
        append(
            f"attachment.{attachment_index}.description",
            attachment.description,
        )
    return tuple(output)


def _message_context_text(message: discord.Message) -> str:
    """Render every bounded Discord text surface with stable source labels."""

    lines: list[str] = []
    for segment in discord_translation_segments(message):
        if segment.identifier == "content":
            lines.append(segment.text)
        else:
            lines.append(f"[{segment.identifier}] {segment.text}")
    return "\n".join(lines)


def _append_component_text(
    value: object,
    *,
    path: str,
    append: Callable[[str, str | None], None],
) -> None:
    if isinstance(value, dict):
        component_type = value.get("type")
        content = value.get("content")
        if component_type == 10 and isinstance(content, str):
            append(f"{path}.content", content)
        for key in ("components",):
            children = value.get(key)
            if isinstance(children, list):
                for index, child in enumerate(children):
                    _append_component_text(
                        child,
                        path=f"{path}.{index}",
                        append=append,
                    )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _append_component_text(
                child,
                path=f"{path}.{index}",
                append=append,
            )


def _expanded_reply(message: discord.Message) -> tuple[str | None, str | None]:
    referenced = message.reference.resolved if message.reference is not None else None
    if not isinstance(referenced, discord.Message):
        return None, None
    content = referenced.content.strip()
    if not content and referenced.attachments:
        content = "Attachment"
    return referenced.author.display_name, content[:300] or None


async def _attachment(
    client: discord.Client,
    context: InvocationContext,
    channel_id: str,
    message_id: str,
    attachment_index: int,
) -> tuple[discord.Message, discord.Attachment]:
    guild = _guild(client, context)
    channel = _message_channel(guild, channel_id)
    if not 0 <= attachment_index <= 9:
        raise UserError("discord.attachment_index_invalid")
    if context.transport == "agent":
        _assert_agent_channel_scope(context, channel_id)
    actor = await _actor_member(guild, context)
    bot_member = guild.me
    if (
        bot_member is None
        or not _can_read_messages(channel, actor)
        or not _can_read_private_thread(channel, actor)
        or not _can_read_messages(channel, bot_member)
        or not _can_read_private_thread(channel, bot_member)
    ):
        raise UserError("discord.agent_read_channel_forbidden")
    try:
        message = await channel.fetch_message(_snowflake(message_id, "message"))
    except discord.NotFound as exc:
        raise UserError("discord.message_not_found") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.message_fetch_failed") from exc
    try:
        attachment = message.attachments[attachment_index]
    except IndexError as exc:
        raise UserError("discord.attachment_missing") from exc
    return message, attachment


def _image_media_type(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _reply_context(
    channel: DiscordMessageChannel,
    message: discord.Message,
    *,
    max_depth: int,
    chunk_characters: int,
) -> tuple[DiscordReplyContextRecord, ...]:
    records: list[DiscordReplyContextRecord] = []
    seen = {message.id}
    current = message
    for _ in range(max_depth):
        reference = current.reference
        if reference is None or reference.message_id is None:
            break
        if reference.channel_id is not None and reference.channel_id != channel.id:
            break
        if reference.message_id in seen:
            break
        seen.add(reference.message_id)
        resolved = reference.resolved
        if isinstance(resolved, discord.Message):
            parent = resolved
        else:
            try:
                parent = await channel.fetch_message(reference.message_id)
            except (discord.NotFound, discord.DiscordException):
                break
        context_text = _message_context_text(parent)
        chunk = context_text[:chunk_characters]
        records.append(
            DiscordReplyContextRecord(
                message_id=str(parent.id),
                author_id=str(parent.author.id),
                author_name=parent.author.display_name,
                author_is_bot=parent.author.bot,
                content_chunk=chunk,
                content_length=len(context_text),
                complete=len(chunk) == len(context_text),
                created_at_iso=parent.created_at.isoformat(),
                attachments=tuple(
                    _attachment_record(attachment) for attachment in parent.attachments
                ),
                reference_message_id=(
                    str(parent.reference.message_id)
                    if parent.reference and parent.reference.message_id
                    else None
                ),
                poll=_expanded_poll(parent.poll),
            )
        )
        current = parent
    return tuple(records)


def _message_preview(content: str) -> tuple[str, bool]:
    """Keep enough surrounding text to identify a historical Discord message."""

    maximum = 240
    if len(content) <= maximum:
        return content, False
    return f"{content[:200]}…{content[-39:]}", True


def _bounded_event_message(content: str) -> tuple[str, bool]:
    """Return one bounded chunk of the event that caused this agent turn."""

    maximum = 1_000
    return content[:maximum], len(content) > maximum


def _discord_event_message_id(context: InvocationContext) -> str | None:
    """Use the transport-typed event pointer, never text parsed from an identifier."""

    return context.active_message_id


def _assert_agent_channel_scope(
    context: InvocationContext,
    channel_id: str,
) -> None:
    if (
        context.transport == "agent"
        and context.read_scope_mode == "resource_ids"
        and channel_id not in context.resource_ids
    ):
        raise UserError("discord.agent_read_channel_forbidden")


def _agent_readable_channel_ids(
    guild: discord.Guild,
    actor: discord.Member,
    context: InvocationContext,
) -> tuple[str, ...]:
    """Resolve a typed principal, then apply any event-bounded read scope."""

    try:
        if context.principal_kind == "service":
            readable = readable_for_service(guild, ServicePrincipal(actor))
        elif context.principal_kind == "requester":
            readable = readable_for_requester(guild, RequesterPrincipal(actor))
        else:
            raise UserError("discord.agent_principal_invalid")
    except ValueError as exc:
        raise UserError("discord.agent_principal_invalid") from exc
    if context.read_scope_mode != "resource_ids":
        return readable
    bounded = set(context.resource_ids)
    return tuple(channel_id for channel_id in readable if channel_id in bounded)


async def _fetch_readable_message(
    guild: discord.Guild,
    *,
    channel_id: str,
    message_id: str,
    context: InvocationContext,
) -> tuple[DiscordMessageChannel, discord.Message]:
    """Fetch one exact message after actor, bot, thread, and voice checks."""

    channel = _message_channel(guild, channel_id)
    bot_member = guild.me
    if bot_member is None:
        raise UserError("discord.expand_unavailable")
    if context.transport == "agent":
        _assert_agent_channel_scope(context, channel_id)
    actor = await _actor_member(guild, context)
    if not _can_read_messages(channel, actor) or not _can_read_private_thread(
        channel,
        actor,
    ):
        raise UserError("discord.expand_unavailable")
    if not _can_read_messages(channel, bot_member) or not _can_read_private_thread(
        channel, bot_member
    ):
        raise UserError("discord.expand_unavailable")
    try:
        message = await channel.fetch_message(_snowflake(message_id, "message"))
    except (discord.NotFound, discord.Forbidden) as exc:
        raise UserError("discord.expand_unavailable") from exc
    except discord.DiscordException as exc:
        raise UserError("discord.expand_failed") from exc
    return channel, message


def _quote_text(message: discord.Message) -> str:
    """Resolve Discord mentions while retaining custom emoji tokens for rendering."""

    text = message.content
    for member in message.mentions:
        replacement = f"@{member.display_name}"
        text = text.replace(f"<@{member.id}>", replacement)
        text = text.replace(f"<@!{member.id}>", replacement)
    for role in message.role_mentions:
        text = text.replace(f"<@&{role.id}>", f"@{role.name}")
    for channel in message.channel_mentions:
        text = text.replace(f"<#{channel.id}>", f"#{channel.name}")
    if text.strip():
        return text
    if message.stickers:
        return ""
    if message.attachments:
        image_count = sum(
            1
            for attachment in message.attachments
            if (attachment.content_type or "").startswith("image/")
        )
        if image_count == len(message.attachments):
            return f"Sent {image_count} image{'s' if image_count != 1 else ''}"
        attachment_count = len(message.attachments)
        return f"Sent {attachment_count} attachment{'s' if attachment_count != 1 else ''}"
    if message.embeds:
        return "Embed message"
    return "…"


async def _quote_avatar(
    client: discord.Client,
    message: discord.Message,
) -> bytes | None:
    asset = message.author.display_avatar.with_static_format("png").with_size(512)
    try:
        content = await client.http.get_from_cdn(str(asset))
    except discord.DiscordException:
        return None
    if not content or len(content) > _QUOTE_AVATAR_MAX_BYTES:
        return None
    return content


async def _quote_custom_emojis(
    client: discord.Client,
    content: str,
) -> tuple[QuoteCustomEmojiAsset, ...]:
    async def fetch(record: DiscordCustomEmojiRecord) -> QuoteCustomEmojiAsset | None:
        extension = "gif" if record.animated else "png"
        url = (
            f"https://cdn.discordapp.com/emojis/{record.emoji_id}.{extension}"
            "?size=128&quality=lossless"
        )
        try:
            payload = await client.http.get_from_cdn(url)
        except discord.DiscordException:
            return None
        if not payload or len(payload) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            return None
        return QuoteCustomEmojiAsset(
            emoji_id=record.emoji_id,
            name=record.name,
            content=payload,
        )

    records = _custom_emoji_records(content)
    if not records:
        return ()
    fetched = await asyncio.gather(*(fetch(record) for record in records))
    return tuple(item for item in fetched if item is not None)


async def _quote_stickers(
    client: discord.Client,
    message: discord.Message,
) -> tuple[QuoteStickerAsset, ...]:
    async def fetch(sticker: discord.StickerItem) -> QuoteStickerAsset | None:
        url = (
            f"https://media.discordapp.net/stickers/{sticker.id}.png"
            if sticker.format is discord.StickerFormatType.lottie
            else str(sticker.url)
        )
        try:
            payload = await client.http.get_from_cdn(url)
        except discord.DiscordException:
            return None
        if not payload or len(payload) > _CUSTOM_EMOJI_MEDIA_MAX_BYTES:
            return None
        return QuoteStickerAsset(
            sticker_id=str(sticker.id),
            name=sticker.name,
            content=payload,
        )

    fetched = await asyncio.gather(*(fetch(sticker) for sticker in message.stickers[:3]))
    return tuple(item for item in fetched if item is not None)


def quote_message_has_animation(message: discord.Message) -> bool:
    """Return whether Quote can preserve at least one animated source asset."""

    if any(record.animated for record in _custom_emoji_records(message.content)):
        return True
    return any(
        sticker.format
        in {
            discord.StickerFormatType.apng,
            discord.StickerFormatType.gif,
        }
        for sticker in message.stickers
    )
