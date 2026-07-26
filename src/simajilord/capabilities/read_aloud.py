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
    DISABLE = "disable"
    STATUS = "status"


@dataclass(frozen=True, slots=True)
class ReadAloudRequest:
    action: ReadAloudAction
    text_channel_id: str | None = None
    audio_destination_id: str | None = None
    mode: ReadAloudMode = ReadAloudMode.QUEUE


@dataclass(frozen=True, slots=True)
class ReadAloudResponse:
    action: str
    enabled: bool
    text_channel_id: str | None
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
        if request.action is ReadAloudAction.CONFIGURE:
            if request.text_channel_id is None or request.audio_destination_id is None:
                raise UserError("read_aloud.route_fields_required")
            route = ReadAloudRoute(
                workspace_id=workspace_id,
                text_channel_id=request.text_channel_id,
                audio_destination_id=request.audio_destination_id,
                mode=request.mode,
            )
            await service.configure(route)
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
            audio_destination_id=route.audio_destination_id if route else None,
            mode=route.mode.value if route else None,
        )

    return endpoint(
        CapabilityDescriptor(
            name="speech.manage_read_aloud",
            summary="Configure, inspect, or disable automatic channel read-aloud routing.",
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.WHEN_REQUESTED,
            keywords=("tts", "speech", "messages", "channel", "voice"),
            side_effects=(
                "Persists workspace routing.",
                "May cause future messages to play audio.",
            ),
        ),
        ReadAloudRequest,
        ReadAloudResponse,
        manage,
    )
