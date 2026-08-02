"""Transport-neutral records for event-driven agent execution."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

AGENT_NO_ACTION_CONTENT = "<simajilord:no-action>"
AGENT_FINAL_DELIVERED_CONTENT = "<simajilord:final-delivered>"
AGENT_MESSAGE_BREAK = "<simajilord:message-break>"
AGENT_DISCORD_SAFE_MESSAGE_CHARACTERS = 1_900
AGENT_AUTONOMY_ACTOR_ID = "simajilord:autonomy"
AGENT_AUDIO_GRANT = "audio"
AGENT_COMPUTE_GRANT = "safe_compute"
AGENT_FEEDBACK_GRANT = "feedback"
AGENT_FILE_GRANT = "files"
AGENT_HIVE_GRANT = "hive_analysis"
AGENT_IMAGE_GRANT = "image"
AGENT_MEDIA_GRANT = "media_download"
AGENT_MEMORY_GRANT = "memory"
AGENT_MESSAGE_GRANT = "discord_message"
AGENT_MODERATION_GRANT = "moderation"
AGENT_QUOTE_GRANT = "discord_quote"
AGENT_REACTION_GRANT = "discord_reaction"
AGENT_REPOST_GRANT = "discord_repost"
AGENT_WEB_GRANT = "web"
AGENT_TIMER_WRITE_CAPABILITIES = (
    "timer.create",
    "timer.cancel",
)
AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES = (
    "discord.delete_message",
    "discord.bulk_delete_messages",
    "discord.kick_member",
    "discord.ban_member",
    "discord.delete_guild_resource",
    "discord.delete_platform_asset",
    "discord.delete_automod_rule",
)
AGENT_DISCORD_MODERATION_CAPABILITIES = (
    "discord.set_timeout",
    *AGENT_DISCORD_DESTRUCTIVE_CAPABILITIES,
    "discord.unban_member",
)
AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES = (
    "discord.connect_voice",
    "discord.create_poll",
    "discord.reply_message",
    "discord.edit_own_message",
    "discord.pin_message",
    "discord.unpin_message",
    "discord.create_thread",
    "discord.update_thread",
    "discord.add_thread_member",
    "discord.remove_thread_member",
    "discord.create_forum_post",
    "discord.create_role",
    "discord.assign_role",
    "discord.remove_role",
    "discord.update_channel_settings",
    "discord.create_channel",
    "discord.create_guild_resource",
    "discord.update_guild_resource",
    "discord.message_action",
    "discord.set_channel_overwrite",
    "discord.create_platform_asset",
    "discord.update_platform_asset",
    "discord.create_automod_rule",
    "discord.update_automod_rule",
    "discord.channel_operation",
    "discord.forward_message",
    "discord.send_direct_message",
    "discord.set_bot_presence",
    *AGENT_DISCORD_MODERATION_CAPABILITIES,
)
AGENT_MEMORY_WRITE_CAPABILITIES = (
    "memory.remember",
    "memory.update",
    "memory.forget",
)
AGENT_AUDIO_CONTROL_CAPABILITIES = (
    "discord.pause_audio",
    "discord.resume_audio",
    "discord.skip_audio",
    "discord.stop_audio",
    "discord.leave_audio",
    "discord.set_audio_loop",
    "discord.remove_audio",
    "discord.set_audio_auto_leave",
    "discord.shuffle_audio",
    "discord.seek_audio",
    "discord.tune_audio",
    "discord.set_audio_volume",
    "discord.set_audio_radio",
    "discord.move_audio",
    "discord.clear_my_audio",
)
AGENT_AUDIO_WRITE_CAPABILITIES = (
    "discord.play_audio",
    "discord.play_attachment",
    *AGENT_AUDIO_CONTROL_CAPABILITIES,
    "discord.read_aloud_add_sources",
    "discord.read_aloud_remove_source",
    "discord.read_aloud_disable",
    "discord.read_aloud_dictionary_set",
    "discord.read_aloud_dictionary_remove",
    "discord.read_aloud_exclusion_set",
    "discord.read_aloud_announcements_set",
    "discord.read_aloud_semantics_set",
    "discord.read_aloud_content_mode_set",
    "discord.speak",
)
AGENT_REQUESTED_WRITE_CAPABILITIES = (
    *AGENT_AUDIO_WRITE_CAPABILITIES,
    *AGENT_DISCORD_REQUESTED_WRITE_CAPABILITIES,
    *AGENT_TIMER_WRITE_CAPABILITIES,
    *AGENT_MEMORY_WRITE_CAPABILITIES,
    "feedback.create",
)
_AGENT_PUBLIC_REFERENCE_HEX_CHARACTERS = 20
_AGENT_TASK_ID_HEX_CHARACTERS = 20


def new_agent_public_reference_id() -> str:
    """Return an opaque, Discord-safe identifier for one agent request."""

    return f"agt_{secrets.token_hex(_AGENT_PUBLIC_REFERENCE_HEX_CHARACTERS // 2)}"


def is_agent_public_reference_id(value: str) -> bool:
    """Validate the public format without decoding any internal identity."""

    prefix = "agt_"
    if not value.startswith(prefix):
        return False
    suffix = value[len(prefix) :]
    return len(suffix) == _AGENT_PUBLIC_REFERENCE_HEX_CHARACTERS and all(
        character in "0123456789abcdef" for character in suffix
    )


def new_agent_task_id() -> str:
    """Return an opaque identifier for one durable unit of agent work."""

    return f"tsk_{secrets.token_hex(_AGENT_TASK_ID_HEX_CHARACTERS // 2)}"


def is_agent_task_id(value: str) -> bool:
    """Validate the public task format without decoding transport identity."""

    prefix = "tsk_"
    if not value.startswith(prefix):
        return False
    suffix = value[len(prefix) :]
    return len(suffix) == _AGENT_TASK_ID_HEX_CHARACTERS and all(
        character in "0123456789abcdef" for character in suffix
    )


def task_scoped_conversation_id(conversation_id: str, task_id: str) -> str:
    """Derive a provider-continuity key that cannot bleed into another task."""

    normalized_conversation_id = conversation_id.strip()
    if not normalized_conversation_id or len(normalized_conversation_id) > 500:
        raise ValueError("conversation ID must contain 1 to 500 characters")
    if not is_agent_task_id(task_id):
        raise ValueError("invalid agent task ID")
    suffix = f":task:{task_id}"
    profile_marker = ":profile:"
    if profile_marker in normalized_conversation_id:
        base, profile = normalized_conversation_id.rsplit(profile_marker, 1)
        scoped_conversation_id = (
            normalized_conversation_id
            if base.endswith(suffix)
            else f"{base}{suffix}{profile_marker}{profile}"
        )
    else:
        scoped_conversation_id = (
            normalized_conversation_id
            if normalized_conversation_id.endswith(suffix)
            else f"{normalized_conversation_id}{suffix}"
        )
    if len(scoped_conversation_id) > 500:
        raise ValueError("task-scoped conversation ID exceeds 500 characters")
    return scoped_conversation_id


class GoalState(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentGoal:
    goal_id: str
    instruction: str
    created_at: datetime
    next_review_at: datetime | None
    state: GoalState = GoalState.ACTIVE


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_id: str
    kind: str
    occurred_at: datetime
    workspace_id: str | None
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class ActionProposal:
    capability_name: str
    reason: str
    arguments: dict[str, object]
    deduplication_key: str


class AgentTrigger(StrEnum):
    """Why the agent was woken without coupling it to a transport command."""

    MENTION = "mention"
    AUTONOMOUS = "autonomous"


class AgentAutonomyMode(StrEnum):
    """Host-enforced autonomy levels, independent of model instructions."""

    OBSERVE = "observe"
    ASSIST = "assist"
    ACT = "act"


class AgentResponseStatus(StrEnum):
    """Stable outcome exposed to transport adapters."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTaskRouteDecision(StrEnum):
    """Model-selected relationship between a new event and an active task."""

    ATTACH = "attach"
    SEPARATE = "separate"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class AgentTaskRouteResult:
    """Typed host result for a candidate routed against one active task."""

    decision: AgentTaskRouteDecision
    active_event_id: str
    active_task_id: str
    active_public_reference_id: str


class AgentProgressStage(StrEnum):
    """Low-cardinality public progress without model reasoning."""

    QUEUED = "queued"
    STARTING = "starting"
    READING_DISCORD = "reading_discord"
    SEARCHING_WEB = "searching_web"
    COMPUTING = "computing"
    COMPACTING_CONTEXT = "compacting_context"
    ANALYZING_MEDIA = "analyzing_media"
    GENERATING_IMAGE = "generating_image"
    USING_AUDIO = "using_audio"
    PREPARING_RESPONSE = "preparing_response"


@dataclass(frozen=True, slots=True)
class AgentProgressUpdate:
    """Public execution state with optional same-server FIFO position."""

    stage: AgentProgressStage
    queue_position: int | None = None


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """A compact event pointer.

    Message content is deliberately absent. The agent must retrieve bounded
    content through a capability when it decides that content is required.
    """

    conversation_id: str
    event_id: str
    trigger: AgentTrigger
    actor_id: str
    actor_name: str
    workspace_id: str | None
    channel_id: str
    message_id: str | None
    occurred_at: datetime
    resource_ids: tuple[str, ...]
    public_reference_id: str
    task_id: str = field(default_factory=new_agent_task_id)
    message_edited_at: datetime | None = None
    grants: frozenset[str] = frozenset()
    approvals: frozenset[str] = frozenset()
    events: tuple[AgentEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentTokenUsage:
    """Provider-reported usage for one completed turn."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0
    model_context_window: int | None = None


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """One final user-facing answer plus durable provider identity."""

    status: AgentResponseStatus
    conversation_id: str
    provider_thread_id: str | None
    model: str
    content: str
    usage: AgentTokenUsage = AgentTokenUsage()
