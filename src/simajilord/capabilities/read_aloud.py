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
        metadata={"description": "読み上げ対象に追加するDiscordチャンネルID。"}
    )
    audio_destination_id: str = field(
        metadata={"description": "音声を流すDiscordボイスチャンネルID。"}
    )
    mode: ReadAloudMode = ReadAloudMode.QUEUE


@dataclass(frozen=True, slots=True)
class ReadAloudRemoveSourceRequest:
    text_channel_id: str = field(
        metadata={"description": "読み上げ対象から外すDiscordチャンネルID。"}
    )


@dataclass(frozen=True, slots=True)
class ReadAloudDisableRequest:
    pass


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryListRequest:
    pass


@dataclass(frozen=True, slots=True)
class ReadAloudDictionarySetRequest:
    surface: str = field(metadata={"description": "メッセージ中で置換する表記。"})
    reading: str = field(metadata={"description": "VOICEVOXへ渡す自然な読み。"})


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryRemoveRequest:
    surface: str = field(metadata={"description": "辞書から削除する表記。"})


class ReadAloudExclusionTarget(StrEnum):
    USER = "user"
    ROLE = "role"


@dataclass(frozen=True, slots=True)
class ReadAloudExclusionSetRequest:
    target: ReadAloudExclusionTarget
    target_id: str = field(
        metadata={"description": "対象のDiscordユーザーIDまたはロールID。"}
    )
    ignored: bool = field(
        metadata={"description": "trueなら読み上げから除外、falseなら解除。"}
    )


@dataclass(frozen=True, slots=True)
class ReadAloudAnnouncementsSetRequest:
    join: bool | None = None
    leave: bool | None = None
    move: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadAloudSemanticsSetRequest:
    author_names: bool | None = None
    replies: bool | None = None
    attachments: bool | None = None
    vc_members_only: bool | None = field(
        default=None,
        metadata={
            "description": (
                "trueなら読み上げ先VCに現在参加中のユーザーの投稿だけを読み上げます。"
            )
        },
    )


@dataclass(frozen=True, slots=True)
class ReadAloudContentModeSetRequest:
    mode: ReadAloudContentMode


@dataclass(frozen=True, slots=True)
class ReadAloudServerVoiceSetRequest:
    preset: ReadAloudVoicePreset


@dataclass(frozen=True, slots=True)
class ReadAloudUserVoiceSetRequest:
    preset: ReadAloudVoicePreset | None = field(
        default=None,
        metadata={"description": "省略すると本人の上書きを解除します。"},
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
            summary="チャンネルの自動読み上げ経路を設定・確認・無効化します。",
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.WHEN_REQUESTED,
            keywords=("tts", "speech", "messages", "channel", "voice"),
            side_effects=(
                "ワークスペースの読み上げ経路を保存します。",
                "今後届くメッセージの音声が再生される場合があります。",
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
                summary="現在の読み上げ経路だけを確認します。",
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
                summary="会話チャンネルを現在のVC読み上げ経路へ追加します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "add", "channel", "voice", "tts"),
                side_effects=("読み上げ対象チャンネルを永続設定します。",),
            ),
            ReadAloudAddSourcesRequest,
            ReadAloudResponse,
            add_sources,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_remove_source",
                summary="会話チャンネルを読み上げ対象から外します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "remove", "channel", "tts"),
                side_effects=("読み上げ対象チャンネルを永続設定から外します。",),
            ),
            ReadAloudRemoveSourceRequest,
            ReadAloudResponse,
            remove_source,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_disable",
                summary="このサーバーの読み上げ経路を無効にします。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "disable", "stop", "tts"),
                side_effects=("このサーバーの読み上げ経路を削除します。",),
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
            raise UserError(str(exc)) from exc
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
            raise UserError(str(exc)) from exc
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
            raise UserError(str(exc)) from exc
        return _policy_response(policy)

    async def announcements_set(
        request: ReadAloudAnnouncementsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        try:
            policy = await service.set_announcements(
                workspace_id=_workspace_id(context),
                join=request.join,
                leave=request.leave,
                move=request.move,
            )
        except ValueError as exc:
            raise UserError(str(exc)) from exc
        return _policy_response(policy)

    async def semantics_set(
        request: ReadAloudSemanticsSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        try:
            policy = await service.set_semantic_options(
                workspace_id=_workspace_id(context),
                author_names=request.author_names,
                replies=request.replies,
                attachments=request.attachments,
                vc_members_only=request.vc_members_only,
            )
        except ValueError as exc:
            raise UserError(str(exc)) from exc
        return _policy_response(policy)

    async def content_mode_set(
        request: ReadAloudContentModeSetRequest,
        context: InvocationContext,
    ) -> ReadAloudPolicyResponse:
        policy = await service.set_content_mode(
            workspace_id=_workspace_id(context),
            mode=request.mode,
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
                summary="自分の読み上げ音声プリセットを設定または解除します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "voice", "preset", "self"),
                side_effects=("本人の音声プリセットを永続設定します。",),
            ),
            ReadAloudUserVoiceSetRequest,
            ReadAloudPolicyResponse,
            user_voice_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_server_voice_set",
                summary="サーバー既定の読み上げ音声プリセットを設定します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "voice", "preset", "server"),
                side_effects=("サーバー既定の音声プリセットを永続設定します。",),
            ),
            ReadAloudServerVoiceSetRequest,
            ReadAloudPolicyResponse,
            server_voice_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_content_mode_set",
                summary="読み上げをall/messages/events/offのプリセットで設定します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "messages", "events", "off", "mode"),
                side_effects=("読み上げ対象の種類を永続設定します。",),
            ),
            ReadAloudContentModeSetRequest,
            ReadAloudPolicyResponse,
            content_mode_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_policy_status",
                summary="読み上げ辞書・除外・入退室通知の現在値を確認します。",
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
                summary="このサーバーの読み上げ辞書を一覧表示します。",
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
                summary="表記と読みをサーバー別読み上げ辞書へ登録します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "dictionary", "pronunciation", "add"),
                side_effects=("サーバー別読み上げ辞書を更新します。",),
            ),
            ReadAloudDictionarySetRequest,
            ReadAloudPolicyResponse,
            dictionary_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_dictionary_remove",
                summary="指定した表記をサーバー別読み上げ辞書から削除します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "dictionary", "pronunciation", "remove"),
                side_effects=("サーバー別読み上げ辞書を更新します。",),
            ),
            ReadAloudDictionaryRemoveRequest,
            ReadAloudPolicyResponse,
            dictionary_remove,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_exclusion_set",
                summary="ユーザーまたはロールの読み上げ除外を設定・解除します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "ignore", "mute", "user", "role"),
                side_effects=("読み上げ除外設定を更新します。",),
            ),
            ReadAloudExclusionSetRequest,
            ReadAloudPolicyResponse,
            exclusion_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_announcements_set",
                summary="VCへの参加・退出・移動の読み上げを設定します。",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("read aloud", "join", "leave", "move", "announce"),
                side_effects=("VC入退室通知の読み上げ設定を更新します。",),
            ),
            ReadAloudAnnouncementsSetRequest,
            ReadAloudPolicyResponse,
            announcements_set,
        ),
        endpoint(
            CapabilityDescriptor(
                name="speech.read_aloud_semantics_set",
                summary=(
                    "投稿者名・返信先・添付と、VC参加者限定の読み上げを設定します。"
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
                side_effects=("意味的な読み上げ設定を更新します。",),
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


def _policy_response(policy: ReadAloudPolicy) -> ReadAloudPolicyResponse:
    has_events = (
        policy.announce_join
        or policy.announce_leave
        or policy.announce_move
    )
    if policy.read_messages:
        content_mode = (
            ReadAloudContentMode.ALL
            if has_events
            else ReadAloudContentMode.MESSAGES
        )
    else:
        content_mode = (
            ReadAloudContentMode.EVENTS
            if has_events
            else ReadAloudContentMode.OFF
        )
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
    )
