"""Read-aloud route management as a reusable capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.read_aloud import (
    ReadAloudContentMode,
    ReadAloudMode,
    ReadAloudPolicy,
    ReadAloudRoute,
    ReadAloudService,
    ReadAloudVoicePreset,
)


class ReadAloudAction(StrEnum):
    CONFIGURE = "configure"
    ADD_SOURCES = "add_sources"
    ADD_SOURCE = "add_source"
    REMOVE_SOURCE = "remove_source"
    DISABLE = "disable"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class ReadAloudRequest:
    action: ReadAloudAction
    text_channel_id: str | None = None
    text_channel_ids: tuple[str, ...] = ()
    audio_destination_id: str | None = None
    mode: ReadAloudMode = ReadAloudMode.QUEUE


@dataclass(frozen=True, slots=True)
class ReadAloudResponse:
    action: str
    enabled: bool
    text_channel_id: str | None
    text_channel_ids: tuple[str, ...]
    audio_destination_id: str | None
    mode: str | None


@dataclass(frozen=True, slots=True)
class ReadAloudStatusRequest:
    pass


@dataclass(frozen=True, slots=True)
class ReadAloudAddSourcesRequest:
    text_channel_ids: tuple[str, ...] = field(
        metadata={"description": "Discord channel IDs to add as read-aloud sources."}
    )
    audio_destination_id: str = field(
        metadata={"description": "Discord voice channel ID used for audio output."}
    )
    mode: ReadAloudMode = ReadAloudMode.QUEUE


@dataclass(frozen=True, slots=True)
class ReadAloudRemoveSourceRequest:
    text_channel_id: str = field(
        metadata={"description": "Discord channel ID to remove from read aloud."}
    )


@dataclass(frozen=True, slots=True)
class ReadAloudDisableRequest:
    pass


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryListRequest:
    pass


@dataclass(frozen=True, slots=True)
class ReadAloudDictionarySetRequest:
    surface: str = field(metadata={"description": "Written form to replace in messages."})
    reading: str = field(metadata={"description": "Natural reading passed to VOICEVOX."})


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryRemoveRequest:
    surface: str = field(metadata={"description": "Written form to remove."})


class ReadAloudExclusionTarget(StrEnum):
    USER = "user"
    ROLE = "role"


@dataclass(frozen=True, slots=True)
class ReadAloudExclusionSetRequest:
    target: ReadAloudExclusionTarget
    target_id: str = field(
        metadata={"description": "Target Discord user or role ID."}
    )
    ignored: bool = field(
        metadata={"description": "True to exclude from read aloud; false to restore."}
    )


@dataclass(frozen=True, slots=True)
class ReadAloudAnnouncementsSetRequest:
    join: bool | None = None
    leave: bool | None = None
    move: bool | None = None
    expected_join: bool | None = None
    expected_leave: bool | None = None
    expected_move: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadAloudSemanticsSetRequest:
    author_names: bool | None = None
    replies: bool | None = None
    attachments: bool | None = None
    vc_members_only: bool | None = field(
        default=None,
        metadata={
            "description": (
                "When true, read only messages from users currently in the output VC."
            )
        },
    )
    expected_author_names: bool | None = None
    expected_replies: bool | None = None
    expected_attachments: bool | None = None
    expected_vc_members_only: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadAloudContentModeSetRequest:
    mode: ReadAloudContentMode


@dataclass(frozen=True, slots=True)
class ReadAloudContentStateRestoreRequest:
    """Internal exact inverse for the four booleans collapsed by a mode preset."""

    read_messages: bool
    announce_join: bool
    announce_leave: bool
    announce_move: bool
    expected_read_messages: bool | None = None
    expected_announce_join: bool | None = None
    expected_announce_leave: bool | None = None
    expected_announce_move: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadAloudServerVoiceSetRequest:
    preset: ReadAloudVoicePreset


@dataclass(frozen=True, slots=True)
class ReadAloudUserVoiceSetRequest:
    preset: ReadAloudVoicePreset | None = field(
        default=None,
        metadata={"description": "Omit to clear the current actor's override."},
    )


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryItem:
    surface: str
    reading: str


@dataclass(frozen=True, slots=True)
class ReadAloudPolicyResponse:
    dictionary: tuple[ReadAloudDictionaryItem, ...]
    ignored_user_ids: tuple[str, ...]
    ignored_role_ids: tuple[str, ...]
    announce_join: bool
    announce_leave: bool
    announce_move: bool
    read_author_names: bool
    read_replies: bool
    read_attachments: bool
    vc_members_only: bool = False
    read_messages: bool = True
    content_mode: str = ReadAloudContentMode.MESSAGES.value
    default_voice_preset: str = ReadAloudVoicePreset.CLEAR.value
    user_voice_presets: tuple[tuple[str, str], ...] = ()
    previous_announce_join: bool | None = None
    previous_announce_leave: bool | None = None
    previous_announce_move: bool | None = None
    previous_read_author_names: bool | None = None
    previous_read_replies: bool | None = None
    previous_read_attachments: bool | None = None
    previous_vc_members_only: bool | None = None
    previous_content_mode: str | None = None
    previous_read_messages: bool | None = None


def build_read_aloud_endpoint(service: ReadAloudService) -> CapabilityEndpoint:
    async def manage(
        request: ReadAloudRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        if context.workspace_id is None:
            raise UserError("workspace.required")
        workspace_id = context.workspace_id
        route: ReadAloudRoute | None
        if request.action is ReadAloudAction.ADD_SOURCES:
            if not request.text_channel_ids or request.audio_destination_id is None:
                raise UserError("read_aloud.source_channels_required")
            try:
                route = await service.add_sources(
                    workspace_id=workspace_id,
                    text_channel_ids=request.text_channel_ids,
                    audio_destination_id=request.audio_destination_id,
                    mode=request.mode,
                )
            except ValueError as exc:
                code = (
                    "read_aloud.destination_conflict"
                    if str(exc) == "read_aloud.destination_conflict"
                    else "read_aloud.source_channels_required"
                )
                raise UserError(code) from exc
        elif request.action in {
            ReadAloudAction.CONFIGURE,
            ReadAloudAction.ADD_SOURCE,
        }:
            if request.text_channel_id is None or request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            if request.action is ReadAloudAction.CONFIGURE:
                route = ReadAloudRoute(
                    workspace_id=workspace_id,
                    text_channel_id=request.text_channel_id,
                    audio_destination_id=request.audio_destination_id,
                    mode=request.mode,
                )
                await service.configure(route)
            else:
                try:
                    route = await service.add_source(
                        workspace_id=workspace_id,
                        text_channel_id=request.text_channel_id,
                        audio_destination_id=request.audio_destination_id,
                        mode=request.mode,
                    )
                except ValueError as exc:
                    raise UserError("read_aloud.destination_conflict") from exc
        elif request.action is ReadAloudAction.REMOVE_SOURCE:
            if request.text_channel_id is None:
                raise UserError("read_aloud.source_channel_required")
            route = await service.remove_source(
                workspace_id=workspace_id,
                text_channel_id=request.text_channel_id,
            )
        elif request.action is ReadAloudAction.DISABLE:
            await service.disable(workspace_id)
            route = None
        else:
            route = service.get(workspace_id)

        if route is None:
            route = service.get(workspace_id)
        return ReadAloudResponse(
            action=request.action.value,
            enabled=route is not None,
            text_channel_id=route.text_channel_id if route else None,
            text_channel_ids=route.text_channel_ids if route else (),
            audio_destination_id=route.audio_destination_id if route else None,
            mode=route.mode.value if route else None,
        )

    return endpoint(
        CapabilityDescriptor(
            name="speech.manage_read_aloud",
            summary="Configure, inspect, or disable an automatic read-aloud route.",
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.WHEN_REQUESTED,
            keywords=("tts", "speech", "messages", "channel", "voice"),
            side_effects=(
                "Persists the workspace read-aloud route.",
                "Future messages may produce synthesized speech.",
            ),
        ),
        ReadAloudRequest,
        ReadAloudResponse,
        manage,
    )


def build_read_aloud_route_endpoints(
    service: ReadAloudService,
) -> tuple[CapabilityEndpoint, ...]:
    """Expose one exact schema per route action for reliable tool use."""

    async def status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        return _route_response(
            ReadAloudAction.STATUS,
            service.get(_workspace_id(context)),
        )

    async def add_sources(
        request: ReadAloudAddSourcesRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        try:
            route = await service.add_sources(
                workspace_id=_workspace_id(context),
                text_channel_ids=request.text_channel_ids,
                audio_destination_id=request.audio_destination_id,
                mode=request.mode,
            )
        except ValueError as exc:
            code = (
                "read_aloud.destination_conflict"
                if str(exc) == "read_aloud.destination_conflict"
                else "read_aloud.source_channels_required"
            )
            raise UserError(code) from exc
        return _route_response(ReadAloudAction.ADD_SOURCES, route)

    async def remove_source(
        request: ReadAloudRemoveSourceRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        route = await service.remove_source(
            workspace_id=_workspace_id(context),
            text_channel_id=request.text_channel_id,
        )
        return _route_response(ReadAloudAction.REMOVE_SOURCE, route)

    async def disable(
        _request: ReadAloudDisableRequest,
        context: InvocationContext,
    ) -> ReadAloudResponse:
        workspace_id = _workspace_id(context)
        await service.disable(workspace_id)
        return _route_response(ReadAloudAction.DISABLE, None)

    return (
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_status",
                summary="Inspect the current read-aloud route.",
                risk=RiskLevel.READ,
                keywords=("read aloud", "status", "route", "tts"),
            ),
            ReadAloudStatusRequest,
            ReadAloudResponse,
            status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_add_sources",
                summary="Add conversation channels to the current voice route.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "add", "channel", "voice", "tts"),
                side_effects=("Persists read-aloud source channels.",),
            ),
            ReadAloudAddSourcesRequest,
            ReadAloudResponse,
            add_sources,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_remove_source",
                summary="Remove a conversation channel from read aloud.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "remove", "channel", "tts"),
                side_effects=("Removes a source from persistent read-aloud settings.",),
            ),
            ReadAloudRemoveSourceRequest,
            ReadAloudResponse,
            remove_source,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_disable",
                summary="Disable the read-aloud route for this server.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "disable", "stop", "tts"),
                side_effects=("Deletes this server's read-aloud route.",),
            ),
            ReadAloudDisableRequest,
            ReadAloudResponse,
            disable,
        ),
    )


def build_read_aloud_policy_endpoints(
    service: ReadAloudService,
) -> tuple[CapabilityEndpoint, ...]:
    """Expose durable read-aloud policy operations with narrow schemas."""

    async def policy_status(
        _request: ReadAloudStatusRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        return _policy_response(service.policy(_workspace_id(context)))

    async def dictionary_list(
        _request: ReadAloudDictionaryListRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        return _policy_response(service.policy(_workspace_id(context)))

    async def dictionary_set(
        request: ReadAloudDictionarySetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        try:
            policy = await service.upsert_dictionary_entry(
                workspace_id=_workspace_id(context),
                surface=request.surface,
                reading=request.reading,
            )
        except ValueError as exc:
            raise UserError("read_aloud.dictionary_invalid") from exc
        return _policy_response(policy)

    async def dictionary_remove(
        request: ReadAloudDictionaryRemoveRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        try:
            policy, _ = await service.remove_dictionary_entry(
                workspace_id=_workspace_id(context),
                surface=request.surface,
            )
        except ValueError as exc:
            raise UserError("read_aloud.dictionary_invalid") from exc
        return _policy_response(policy)

    async def exclusion_set(
        request: ReadAloudExclusionSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        try:
            if request.target is ReadAloudExclusionTarget.USER:
                policy = await service.set_user_ignored(
                    workspace_id=_workspace_id(context),
                    user_id=request.target_id,
                    ignored=request.ignored,
                )
            else:
                policy = await service.set_role_ignored(
                    workspace_id=_workspace_id(context),
                    role_id=request.target_id,
                    ignored=request.ignored,
                )
        except ValueError as exc:
            raise UserError("read_aloud.exclusion_invalid") from exc
        return _policy_response(policy)

    async def announcements_set(
        request: ReadAloudAnnouncementsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        workspace_id = _workspace_id(context)
        try:
            policy, previous = await service.set_announcements_with_previous(
                workspace_id=workspace_id,
                join=request.join,
                leave=request.leave,
                move=request.move,
                expected_join=request.expected_join,
                expected_leave=request.expected_leave,
                expected_move=request.expected_move,
            )
        except ValueError as exc:
            raise UserError("read_aloud.announcement_value_invalid") from exc
        return _policy_response(
            policy,
            previous_announcements=previous,
        )

    async def semantics_set(
        request: ReadAloudSemanticsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        workspace_id = _workspace_id(context)
        try:
            policy, previous = await service.set_semantic_options_with_previous(
                workspace_id=workspace_id,
                author_names=request.author_names,
                replies=request.replies,
                attachments=request.attachments,
                vc_members_only=request.vc_members_only,
                expected_author_names=request.expected_author_names,
                expected_replies=request.expected_replies,
                expected_attachments=request.expected_attachments,
                expected_vc_members_only=request.expected_vc_members_only,
            )
        except ValueError as exc:
            raise UserError("read_aloud.semantic_value_invalid") from exc
        return _policy_response(
            policy,
            previous_semantics=previous,
        )

    async def content_mode_set(
        request: ReadAloudContentModeSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        policy, previous = await service.set_content_mode_with_previous(
            workspace_id=_workspace_id(context),
            mode=request.mode,
        )
        return _policy_response(
            policy,
            previous_announcements=previous,
            previous_content_mode=previous,
        )

    async def content_state_restore(
        request: ReadAloudContentStateRestoreRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        policy = await service.restore_content_state(
            workspace_id=_workspace_id(context),
            read_messages=request.read_messages,
            announce_join=request.announce_join,
            announce_leave=request.announce_leave,
            announce_move=request.announce_move,
            expected_read_messages=request.expected_read_messages,
            expected_announce_join=request.expected_announce_join,
            expected_announce_leave=request.expected_announce_leave,
            expected_announce_move=request.expected_announce_move,
        )
        return _policy_response(policy)

    async def server_voice_set(
        request: ReadAloudServerVoiceSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        policy = await service.set_default_voice_preset(
            workspace_id=_workspace_id(context),
            preset=request.preset,
        )
        return _policy_response(policy)

    async def user_voice_set(
        request: ReadAloudUserVoiceSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        if context.actor_id is None:
            raise UserError("actor.required")
        policy = await service.set_user_voice_preset(
            workspace_id=_workspace_id(context),
            user_id=context.actor_id,
            preset=request.preset,
        )
        return _policy_response(policy)

    return (
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_user_voice_set",
                summary="Set or clear the current actor's read-aloud voice preset.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "voice", "preset", "self"),
                side_effects=("Persists the current actor's voice preset.",),
            ),
            ReadAloudUserVoiceSetRequest,
            ReadAloudPolicyResponse,
            user_voice_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_server_voice_set",
                summary="Set the server's default read-aloud voice preset.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "voice", "preset", "server"),
                side_effects=("Persists the server's default voice preset.",),
            ),
            ReadAloudServerVoiceSetRequest,
            ReadAloudPolicyResponse,
            server_voice_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_content_mode_set",
                summary="Set read aloud to all, messages, events, or off.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "messages", "events", "off", "mode"),
                side_effects=("Persists which content types are read aloud.",),
                idempotency="idempotent_write",
            ),
            ReadAloudContentModeSetRequest,
            ReadAloudPolicyResponse,
            content_mode_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_content_state_restore",
                summary="Internal exact Undo for a recorded read-aloud content-mode change.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.ALWAYS,
                side_effects=("Restores four recorded read-aloud policy switches.",),
                idempotency="idempotent_write",
            ),
            ReadAloudContentStateRestoreRequest,
            ReadAloudPolicyResponse,
            content_state_restore,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_policy_status",
                summary="Inspect read-aloud dictionary, exclusions, and announcements.",
                risk=RiskLevel.READ,
                keywords=("read aloud", "policy", "settings", "tts"),
            ),
            ReadAloudStatusRequest,
            ReadAloudPolicyResponse,
            policy_status,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_dictionary_list",
                summary="List this server's read-aloud dictionary.",
                risk=RiskLevel.READ,
                keywords=("read aloud", "dictionary", "pronunciation", "tts"),
            ),
            ReadAloudDictionaryListRequest,
            ReadAloudPolicyResponse,
            dictionary_list,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_dictionary_set",
                summary="Add or update a server-specific pronunciation entry.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "dictionary", "pronunciation", "add"),
                side_effects=("Updates the server-specific pronunciation dictionary.",),
            ),
            ReadAloudDictionarySetRequest,
            ReadAloudPolicyResponse,
            dictionary_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_dictionary_remove",
                summary="Remove a pronunciation entry from this server.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "dictionary", "pronunciation", "remove"),
                side_effects=("Updates the server-specific pronunciation dictionary.",),
            ),
            ReadAloudDictionaryRemoveRequest,
            ReadAloudPolicyResponse,
            dictionary_remove,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_exclusion_set",
                summary="Set or clear a user or role read-aloud exclusion.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "ignore", "mute", "user", "role"),
                side_effects=("Updates persistent read-aloud exclusions.",),
            ),
            ReadAloudExclusionSetRequest,
            ReadAloudPolicyResponse,
            exclusion_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_announcements_set",
                summary="Configure voice join, leave, and move announcements.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "join", "leave", "move", "announce"),
                side_effects=("Updates persistent voice-event announcement settings.",),
                idempotency="idempotent_write",
            ),
            ReadAloudAnnouncementsSetRequest,
            ReadAloudPolicyResponse,
            announcements_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_semantics_set",
                summary=(
                    "Configure author, reply, attachment, and voice-member-only narration."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "read aloud",
                    "author",
                    "reply",
                    "attachment",
                    "semantic",
                    "voice members",
                ),
                side_effects=("Updates persistent semantic read-aloud settings.",),
                idempotency="idempotent_write",
            ),
            ReadAloudSemanticsSetRequest,
            ReadAloudPolicyResponse,
            semantics_set,
        ),
    )


def _workspace_id(context: InvocationContext) -> str:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    return context.workspace_id


def _route_response(
    action: ReadAloudAction,
    route: ReadAloudRoute | None,
) -> ReadAloudResponse:
    return ReadAloudResponse(
        action=action.value,
        enabled=route is not None,
        text_channel_id=route.text_channel_id if route else None,
        text_channel_ids=route.text_channel_ids if route else (),
        audio_destination_id=route.audio_destination_id if route else None,
        mode=route.mode.value if route else None,
    )


def _policy_response(
    policy: ReadAloudPolicy,
    *,
    previous_announcements: ReadAloudPolicy | None = None,
    previous_semantics: ReadAloudPolicy | None = None,
    previous_content_mode: ReadAloudPolicy | None = None,
) -> ReadAloudPolicyResponse:
    content_mode = _content_mode(policy)
    return ReadAloudPolicyResponse(
        dictionary=tuple(
            ReadAloudDictionaryItem(entry.surface, entry.reading)
            for entry in policy.dictionary
        ),
        ignored_user_ids=policy.ignored_user_ids,
        ignored_role_ids=policy.ignored_role_ids,
        announce_join=policy.announce_join,
        announce_leave=policy.announce_leave,
        announce_move=policy.announce_move,
        read_messages=policy.read_messages,
        content_mode=content_mode.value,
        read_author_names=policy.read_author_names,
        read_replies=policy.read_replies,
        read_attachments=policy.read_attachments,
        vc_members_only=policy.vc_members_only,
        default_voice_preset=policy.default_voice_preset.value,
        user_voice_presets=tuple(
            (user_id, preset.value)
            for user_id, preset in policy.user_voice_presets
        ),
        previous_announce_join=(
            previous_announcements.announce_join
            if previous_announcements is not None
            else None
        ),
        previous_announce_leave=(
            previous_announcements.announce_leave
            if previous_announcements is not None
            else None
        ),
        previous_announce_move=(
            previous_announcements.announce_move
            if previous_announcements is not None
            else None
        ),
        previous_read_author_names=(
            previous_semantics.read_author_names
            if previous_semantics is not None
            else None
        ),
        previous_read_replies=(
            previous_semantics.read_replies
            if previous_semantics is not None
            else None
        ),
        previous_read_attachments=(
            previous_semantics.read_attachments
            if previous_semantics is not None
            else None
        ),
        previous_vc_members_only=(
            previous_semantics.vc_members_only
            if previous_semantics is not None
            else None
        ),
        previous_content_mode=(
            _content_mode(previous_content_mode).value
            if previous_content_mode is not None
            else None
        ),
        previous_read_messages=(
            previous_content_mode.read_messages
            if previous_content_mode is not None
            else None
        ),
    )


def _content_mode(policy: ReadAloudPolicy) -> ReadAloudContentMode:
    has_events = (
        policy.announce_join
        or policy.announce_leave
        or policy.announce_move
    )
    if policy.read_messages:
        return (
            ReadAloudContentMode.ALL
            if has_events
            else ReadAloudContentMode.MESSAGES
        )
    return (
        ReadAloudContentMode.EVENTS
        if has_events
        else ReadAloudContentMode.OFF
    )
