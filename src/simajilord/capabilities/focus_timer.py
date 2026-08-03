"""Typed focus-timer capability endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.focus_timer import FocusTimer, FocusTimerService
from simajilord.services.read_aloud import (
    ReadAloudContentMode,
    ReadAloudPolicy,
    ReadAloudService,
    read_aloud_content_mode,
)


@dataclass(frozen=True, slots=True)
class FocusTimerCreateRequest:
    duration_seconds: int
    message: str = "Focus session complete."
    delivery_target_id: str | None = field(
        default=None,
        metadata={"description": "Transport-specific destination channel ID."},
    )
    voice_notify: bool = True
    focus_session: bool = False


@dataclass(frozen=True, slots=True)
class FocusTimerListRequest:
    own_only: bool = True


@dataclass(frozen=True, slots=True)
class FocusTimerCancelRequest:
    timer_id: str


@dataclass(frozen=True, slots=True)
class FocusTimerItem:
    timer_id: str
    due_at_epoch: int
    message: str
    voice_notify: bool
    focus_session: bool
    status: str


@dataclass(frozen=True, slots=True)
class FocusTimerResponse:
    timer: FocusTimerItem
    changed: bool = True


@dataclass(frozen=True, slots=True)
class FocusTimerListResponse:
    timers: tuple[FocusTimerItem, ...]


def build_focus_timer_endpoints(
    service: FocusTimerService,
    read_aloud: ReadAloudService | None = None,
) -> tuple[CapabilityEndpoint, ...]:
    async def create(
        request: FocusTimerCreateRequest,
        context: InvocationContext,
    ) -> FocusTimerResponse:
        workspace_id = _workspace(context)
        delivery_target_id = request.delivery_target_id or context.origin_resource_id
        if delivery_target_id is None:
            raise UserError("timer.delivery_target_required")
        restore_content_mode: str | None = None
        if request.focus_session and read_aloud is not None:
            async with service.focus_session_lock(workspace_id):
                active_focus = tuple(
                    timer
                    for timer in await service.active(workspace_id=workspace_id)
                    if timer.focus_session
                )
                current_mode = read_aloud_content_mode(
                    read_aloud.policy(workspace_id)
                )
                restore_content_mode = (
                    (
                        active_focus[0].restore_content_mode
                        or current_mode.value
                    )
                    if active_focus
                    else current_mode.value
                )
                if not active_focus:
                    await read_aloud.set_content_mode(
                        workspace_id=workspace_id,
                        mode=ReadAloudContentMode.EVENTS,
                    )
                try:
                    timer = await service.create(
                        workspace_id=workspace_id,
                        actor_id=context.actor_id,
                        delivery_target_id=delivery_target_id,
                        duration_seconds=request.duration_seconds,
                        message=request.message,
                        voice_notify=request.voice_notify,
                        focus_session=True,
                        restore_content_mode=restore_content_mode,
                    )
                except Exception:
                    if not active_focus:
                        await read_aloud.compare_and_set_content_mode(
                            workspace_id=workspace_id,
                            expected=ReadAloudContentMode.EVENTS,
                            mode=current_mode,
                        )
                    raise
        else:
            timer = await service.create(
                workspace_id=workspace_id,
                actor_id=context.actor_id,
                delivery_target_id=delivery_target_id,
                duration_seconds=request.duration_seconds,
                message=request.message,
                voice_notify=request.voice_notify,
                focus_session=request.focus_session,
                restore_content_mode=restore_content_mode,
            )
        return FocusTimerResponse(timer=_item(timer))

    async def list_active(
        request: FocusTimerListRequest,
        context: InvocationContext,
    ) -> FocusTimerListResponse:
        timers = await service.active(
            workspace_id=_workspace(context),
            actor_id=context.actor_id if request.own_only else None,
        )
        return FocusTimerListResponse(timers=tuple(_item(timer) for timer in timers))

    async def cancel(
        request: FocusTimerCancelRequest,
        context: InvocationContext,
    ) -> FocusTimerResponse:
        workspace_id = _workspace(context)
        async with service.focus_session_lock(workspace_id):
            timer, changed = await service.cancel_with_change(
                timer_id=request.timer_id,
                workspace_id=workspace_id,
                actor_id=context.actor_id,
            )
            if (
                changed
                and timer.focus_session
                and timer.restore_content_mode is not None
                and read_aloud is not None
                and not any(
                    active.focus_session
                    for active in await service.active(workspace_id=workspace_id)
                )
            ):
                await read_aloud.compare_and_set_content_mode(
                    workspace_id=workspace_id,
                    expected=ReadAloudContentMode.EVENTS,
                    mode=ReadAloudContentMode(timer.restore_content_mode),
                )
        return FocusTimerResponse(timer=_item(timer), changed=changed)

    async def restore(
        request: FocusTimerCancelRequest,
        context: InvocationContext,
    ) -> FocusTimerResponse:
        workspace_id = _workspace(context)
        async with service.focus_session_lock(workspace_id):
            active_focus = tuple(
                timer
                for timer in await service.active(workspace_id=workspace_id)
                if timer.focus_session
            )
            current_mode = (
                read_aloud_content_mode(read_aloud.policy(workspace_id))
                if read_aloud is not None
                else None
            )
            timer, changed = await service.restore_with_change(
                timer_id=request.timer_id,
                workspace_id=workspace_id,
                actor_id=context.actor_id,
            )
            try:
                if (
                    changed
                    and timer.focus_session
                    and read_aloud is not None
                    and current_mode is not None
                ):
                    restore_mode = (
                        (
                            active_focus[0].restore_content_mode
                            or current_mode.value
                        )
                        if active_focus
                        else current_mode.value
                    )
                    if restore_mode is not None:
                        timer = await service.set_restore_content_mode(
                            timer.timer_id,
                            restore_mode,
                        )
                    if not active_focus:
                        await read_aloud.set_content_mode(
                            workspace_id=workspace_id,
                            mode=ReadAloudContentMode.EVENTS,
                        )
            except Exception:
                if changed:
                    await service.cancel(
                        timer_id=timer.timer_id,
                        workspace_id=workspace_id,
                        actor_id=context.actor_id,
                    )
                raise
        return FocusTimerResponse(timer=_item(timer), changed=changed)

    return (
        endpoint(
            CapabilityDescriptor(
                name="timer.create",
                summary="Create a persistent Focus Timer and notify its destination.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "timer",
                    "focus",
                    "pomodoro",
                    "reminder",
                    "タイマー",
                    "集中",
                    "ポモドーロ",
                    "リマインダー",
                    "通知",
                ),
                side_effects=("Persists a timer.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=("workspace.required",),
                timeout_seconds=15,
                user_visible_effect="Creates a timer and later posts its notification.",
            ),
            FocusTimerCreateRequest,
            FocusTimerResponse,
            create,
        ),
        endpoint(
            CapabilityDescriptor(
                name="timer.list",
                summary="List active Focus Timers.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=(
                    "timer",
                    "focus",
                    "status",
                    "タイマー",
                    "集中",
                    "一覧",
                    "確認",
                ),
                requires_workspace=True,
                expected_errors=("workspace.required",),
                timeout_seconds=10,
            ),
            FocusTimerListRequest,
            FocusTimerListResponse,
            list_active,
        ),
        endpoint(
            CapabilityDescriptor(
                name="timer.cancel",
                summary="Cancel a Focus Timer created by the current actor.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "timer",
                    "focus",
                    "cancel",
                    "タイマー",
                    "取り消し",
                    "キャンセル",
                    "中止",
                ),
                side_effects=("Cancels a persisted timer.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=("workspace.required",),
                timeout_seconds=10,
                user_visible_effect="Cancels a timer owned by the requester.",
            ),
            FocusTimerCancelRequest,
            FocusTimerResponse,
            cancel,
        ),
        endpoint(
            CapabilityDescriptor(
                name="timer.restore",
                summary="Restore a just-cancelled Focus Timer for the same actor.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.ALWAYS,
                keywords=(
                    "timer",
                    "focus",
                    "restore",
                    "undo",
                    "タイマー",
                    "復元",
                    "元に戻す",
                ),
                side_effects=("Restores a cancelled persisted timer.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "workspace.required",
                    "timer.not_found",
                    "timer.not_owner",
                    "timer.not_cancelled",
                ),
                timeout_seconds=10,
                user_visible_effect="Restores a cancelled timer owned by the requester.",
            ),
            FocusTimerCancelRequest,
            FocusTimerResponse,
            restore,
        ),
    )


def _workspace(context: InvocationContext) -> str:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    return context.workspace_id


def _item(timer: FocusTimer) -> FocusTimerItem:
    return FocusTimerItem(
        timer_id=timer.timer_id,
        due_at_epoch=int(timer.due_at.timestamp()),
        message=timer.message,
        voice_notify=timer.voice_notify,
        focus_session=timer.focus_session,
        status=timer.status.value,
    )


def _content_mode(policy: ReadAloudPolicy) -> ReadAloudContentMode:
    return read_aloud_content_mode(policy)
