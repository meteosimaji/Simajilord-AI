from __future__ import annotations

import asyncio
import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import discord
import pytest
from discord import app_commands

from simajilord.capabilities.feedback import (
    FeedbackCreateRequest,
    FeedbackCreateResponse,
    build_feedback_endpoint,
)
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.diagnostics.feedback import main as feedback_main
from simajilord.integrations.discord.feedback import FeedbackCog, FeedbackModal
from simajilord.runtime import SimajilordRuntime
from simajilord.services.feedback import (
    FeedbackKind,
    FeedbackService,
    FeedbackStatus,
)


@pytest.mark.asyncio
async def test_feedback_is_restart_safe_private_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "private" / "feedback.sqlite3"
    service = FeedbackService(database)

    first = await service.create(
        title=None,
        details="  Playback stopped.\r\nIt never resumed.  ",
        expected="Continue playing",
        reporter_actor_id="discord-user-7",
        workspace_id="guild-2",
        source_transport="discord",
        source_event_id="interaction-3",
        source_channel_id="channel-4",
    )
    retry = await service.create(
        title=None,
        details="Playback stopped.\nIt never resumed.",
        expected="Continue playing",
        reporter_actor_id="discord-user-7",
        workspace_id="guild-2",
        source_transport="discord",
        source_event_id="interaction-3",
        source_channel_id="channel-4",
    )

    assert first.created is True
    assert retry.created is False
    assert retry.report.report_id == first.report.report_id
    assert first.report.status is FeedbackStatus.NEW
    assert first.report.kind is FeedbackKind.UNTRIAGED
    assert first.report.title == "Playback stopped."
    assert first.report.reporter_actor_id == "discord-user-7"
    assert first.report.source_event_id == "interaction-3"
    assert os.stat(database).st_mode & 0o777 == 0o600
    assert os.stat(database.parent).st_mode & 0o077 == 0
    assert first.report.resolved_at is None

    restarted = FeedbackService(database)
    restored = await restarted.get(first.report.report_id)
    assert restored == first.report


@pytest.mark.asyncio
async def test_feedback_endpoint_owns_identity_and_preserves_follow_up_actor(
    tmp_path: Path,
) -> None:
    service = FeedbackService(tmp_path / "feedback.sqlite3")
    endpoint = build_feedback_endpoint(service)
    request_fields = {item.name for item in fields(FeedbackCreateRequest)}

    assert "reporter_actor_id" not in request_fields
    assert "workspace_id" not in request_fields
    assert "source_event_id" not in request_fields
    assert "public_reference_id" not in request_fields

    response = cast(
        FeedbackCreateResponse,
        await endpoint.invoke(
            FeedbackCreateRequest(
                details="Please save the issue from this follow-up.",
            ),
            InvocationContext(
                actor_id="follow-up-author",
                workspace_id="guild",
                transport="agent",
                request_id="discord:message:22",
                origin_resource_id="channel",
                public_reference_id="agt_0123456789abcdefabcd",
            ),
        ),
    )
    report = await service.get(response.report_id)

    assert report.reporter_actor_id == "follow-up-author"
    assert report.workspace_id == "guild"
    assert report.source_transport == "agent"
    assert report.source_event_id == "discord:message:22"
    assert report.source_channel_id == "channel"
    assert report.public_reference_id == "agt_0123456789abcdefabcd"
@pytest.mark.asyncio
async def test_feedback_triage_is_admin_side_only(tmp_path: Path) -> None:
    service = FeedbackService(tmp_path / "feedback.sqlite3")
    first = await service.create(
        details="First",
        reporter_actor_id="1",
        source_transport="discord",
        source_event_id="1",
    )
    second = await service.create(
        details="Second",
        reporter_actor_id="2",
        source_transport="discord",
        source_event_id="2",
    )

    classified = await service.set_kind(first.report.report_id, FeedbackKind.BUG)
    duplicate = await service.set_status(
        second.report.report_id,
        FeedbackStatus.DUPLICATE,
        duplicate_of=first.report.report_id,
    )
    resolved = await service.set_status(
        first.report.report_id,
        FeedbackStatus.RESOLVED,
    )

    assert classified.kind is FeedbackKind.BUG
    assert duplicate.status is FeedbackStatus.DUPLICATE
    assert duplicate.duplicate_of == first.report.report_id
    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_feedback_modal_defers_and_acknowledges_ephemerally(
    tmp_path: Path,
) -> None:
    service = FeedbackService(tmp_path / "feedback.sqlite3")
    registry = CapabilityRegistry()
    registry.register(build_feedback_endpoint(service))
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(registry=registry),
    )
    modal = FeedbackModal(runtime, requester_id=71)
    modal.title_input._value = "Private form"
    modal.details_input._value = "The form submission should stay private."
    modal.expected_input._value = ""
    interaction = SimpleNamespace(
        id=9001,
        user=SimpleNamespace(id=71),
        guild_id=81,
        channel_id=91,
        response=SimpleNamespace(defer=AsyncMock()),
        edit_original_response=AsyncMock(),
    )

    await modal.on_submit(cast(discord.Interaction, interaction))

    interaction.response.defer.assert_awaited_once_with(
        ephemeral=True,
        thinking=True,
    )
    interaction.edit_original_response.assert_awaited_once()
    reports = await service.list()
    assert len(reports) == 1
    assert reports[0].reporter_actor_id == "71"
    assert reports[0].workspace_id == "81"
    assert reports[0].source_channel_id == "91"
    assert reports[0].kind is FeedbackKind.UNTRIAGED


def test_feedback_slash_command_has_no_user_triage_arguments() -> None:
    commands = tuple(
        command
        for command in FeedbackCog.__cog_app_commands__
        if isinstance(command, app_commands.Command)
    )

    assert len(commands) == 1
    assert commands[0].name == "feedback"
    assert commands[0].parameters == []


def test_feedback_cli_lists_and_exports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = tmp_path / "feedback.sqlite3"
    service = FeedbackService(database)
    created = asyncio.run(
        service.create(
            details="CLI report",
            reporter_actor_id="cli-user",
            source_transport="discord",
            source_event_id="cli-event",
        )
    )

    assert feedback_main(["--database", str(database), "list", "--json"]) == 0
    assert created.report.report_id in capsys.readouterr().out

    export = tmp_path / "export.json"
    assert (
        feedback_main(
            [
                "--database",
                str(database),
                "export",
                "--output",
                str(export),
            ]
        )
        == 0
    )
    assert created.report.report_id in export.read_text(encoding="utf-8")
    assert os.stat(export).st_mode & 0o777 == 0o600
