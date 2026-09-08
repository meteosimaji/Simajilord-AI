"""Explicit single-VC resume, relocation, and session-local follow consent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import discord

from simajilord.core import CapabilityDescriptor, InvocationContext, RiskLevel, endpoint
from simajilord.core.capabilities import CapabilityEndpoint
from simajilord.core.errors import UserError
from simajilord.services.read_aloud import ReadAloudMode, ReadAloudRoute

from .audio import DiscordAudioOutput

if TYPE_CHECKING:
    from simajilord.runtime import SimajilordRuntime


@dataclass(frozen=True, slots=True)
class AudioNavigateRequest:
    destination_id: str
    expected_destination_id: str | None = None
    mode: Literal["preserve", "speech", "both", "music", "follow", "unfollow"] = "preserve"
    source_id: str | None = None
    confirm_occupied: bool = False


@dataclass(frozen=True, slots=True)
class AudioNavigateResponse:
    destination_id: str
    speech_only: bool
    reading_sources: tuple[str, ...]
    following: bool


def build_audio_navigation_endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> tuple[CapabilityEndpoint, ...]:
    # Imported after the main capability module has defined its shared guards.
    from .capabilities import (
        _actor_member,
        _enforce_read_aloud_route_audience,
        _guild,
        _member_voice_channel,
        _message_channel,
    )
    from .permissions import can_read_messages as _can_read_messages
    from .permissions import permission_enabled as _permission_enabled

    locks: dict[str, asyncio.Lock] = {}

    async def navigate(
        request: AudioNavigateRequest, context: InvocationContext
    ) -> AudioNavigateResponse:
        guild = _guild(client, context)
        workspace = str(guild.id)
        async with locks.setdefault(workspace, asyncio.Lock()):
            member = await _actor_member(guild, context)
            destination = _member_voice_channel(member)
            if destination is None or str(destination.id) != request.destination_id:
                raise UserError("audio.same_voice_required")
            session = runtime.audio.get_or_create(
                workspace,
                lambda: DiscordAudioOutput(client, guild.id),
            )
            if session.destination_id != request.expected_destination_id:
                raise UserError("audio.navigation_changed")
            if (
                not session.output.connected
                and request.mode in {"both", "music", "preserve"}
                and not session.can_start_for(str(member.id))
            ):
                raise UserError("audio.waiting_queue_restricted")
            if request.mode in {"follow", "unfollow"}:
                if session.destination_id != request.destination_id or not session.output.connected:
                    raise UserError("audio.follow_start_required")
                owner = runtime.audio.follow_actors.get(workspace)
                if owner is not None and owner != str(member.id):
                    raise UserError("audio.follow_owned")
                await context.dispatch_external_effect()
                if request.mode == "follow":
                    runtime.audio.follow_actors[workspace] = str(member.id)
                else:
                    runtime.audio.follow_actors.pop(workspace, None)
                route = runtime.read_aloud.get(workspace)
                return AudioNavigateResponse(
                    request.destination_id,
                    session.speech_only,
                    route.text_channel_ids if route else (),
                    request.mode == "follow",
                )
            moving = session.destination_id not in {None, request.destination_id}
            origin = (
                guild.get_channel(int(session.destination_id)) if session.destination_id else None
            )
            occupied = (
                moving
                and session.output.connected
                and isinstance(origin, (discord.VoiceChannel, discord.StageChannel))
                and any(
                    not listener.bot and listener.id != member.id for listener in origin.members
                )
            )
            if occupied:
                if not request.confirm_occupied:
                    raise UserError("audio.move_confirmation_required")
                if not (
                    _permission_enabled(member.guild_permissions, "manage_guild")
                    or _permission_enabled(member.guild_permissions, "move_members")
                    or _permission_enabled(member.guild_permissions, "administrator")
                ):
                    raise UserError("audio.move_occupied_forbidden")
            current_route = runtime.read_aloud.get(workspace)
            route = runtime.read_aloud.resume_route(workspace, request.destination_id)
            if request.mode in {"preserve", "music"} and current_route is None:
                route = None
            if route is None and request.mode in {"speech", "both"}:
                if not request.source_id:
                    raise UserError("read_aloud.source_channels_required")
                route = ReadAloudRoute(
                    workspace, request.source_id, request.destination_id, ReadAloudMode.QUEUE
                )
            if route is not None:
                sources = tuple(_message_channel(guild, value) for value in route.text_channel_ids)
                for source in sources:
                    if (
                        not _can_read_messages(source, member)
                        or guild.me is None
                        or not _can_read_messages(source, guild.me)
                    ):
                        raise UserError("discord.message_channel_unavailable")
                _enforce_read_aloud_route_audience(runtime, context, guild, sources, destination)
                route = replace(route, audio_destination_id=request.destination_id)
            speech_only = (
                session.speech_only if request.mode == "preserve" else request.mode == "speech"
            )
            # Moving live speech would carry content admitted for another audience.
            # Let already-admitted speech drain before moving; preserve music in place.
            snapshot = await session.snapshot()
            if moving and (
                snapshot.speech_active
                or any(item.kind.value == "speech" for item in snapshot.pending)
            ):
                raise UserError("audio.move_speech_busy")
            await context.dispatch_external_effect()
            if current_route is not None and moving:
                # Pause message admission during the move. Restore the route if it fails.
                await runtime.read_aloud.configure(replace(current_route, enabled=False))
            try:
                if moving and session.output.connected:
                    await session.suspend()
                if moving:
                    await session.discard_relocation_speech()
                if route is not None:
                    await runtime.read_aloud.configure(route)
                await runtime.audio.connect(
                    workspace, request.destination_id, speech_only=speech_only
                )
            except BaseException:
                if current_route is not None:
                    await runtime.read_aloud.configure(current_route)
                elif route is not None:
                    await runtime.read_aloud.configure(replace(route, enabled=False))
                raise
            runtime.audio.follow_actors.pop(workspace, None)
            return AudioNavigateResponse(
                request.destination_id,
                session.speech_only,
                route.text_channel_ids if route else (),
                False,
            )

    return (
        endpoint(
            CapabilityDescriptor(
                name="discord.navigate_audio",
                summary="Resume audio or move saved music and read-aloud to your VC.",
                risk=RiskLevel.WRITE,
                requires_workspace=True,
                timeout_seconds=60,
                user_visible_effect=(
                    "Moves or resumes the single audio connection with its saved read-aloud route."
                ),
            ),
            AudioNavigateRequest,
            AudioNavigateResponse,
            navigate,
        ),
    )
