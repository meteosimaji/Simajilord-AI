"""Read-aloud route management as a reusable capability."""

from __future__ import annotations

from dataclasses import dataclass
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
from simajilord.services.read_aloud import ReadAloudMode, ReadAloudRoute, ReadAloudService


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
