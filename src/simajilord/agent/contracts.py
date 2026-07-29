"""Transport-neutral records for event-driven agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

AGENT_NO_ACTION_CONTENT = "<simajilord:no-action>"
AGENT_MESSAGE_BREAK = "<simajilord:message-break>"
AGENT_AUTONOMY_ACTOR_ID = "simajilord:autonomy"
AGENT_AUDIO_GRANT = "audio"
AGENT_FILE_GRANT = "files"
AGENT_IMAGE_GRANT = "image"
AGENT_MESSAGE_GRANT = "discord_message"
AGENT_MODERATION_GRANT = "moderation"
AGENT_QUOTE_GRANT = "discord_quote"
AGENT_REPOST_GRANT = "discord_repost"
AGENT_WEB_GRANT = "web"
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


class AgentResponseStatus(StrEnum):
    """Stable outcome exposed to transport adapters."""

    COMPLETED = "completed"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class AgentProgressStage(StrEnum):
    """Low-cardinality public progress without model reasoning."""

    QUEUED = "queued"
    STARTING = "starting"
    READING_DISCORD = "reading_discord"
    SEARCHING_WEB = "searching_web"
    COMPUTING = "computing"
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
    grants: frozenset[str] = frozenset()
    approvals: frozenset[str] = frozenset()


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
