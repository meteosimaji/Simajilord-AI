from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest
from discord.ext import commands

from simajilord.core.errors import UserError
from simajilord.integrations.discord.temp_voice import (
    TempVoiceCog,
    TempVoiceCreateView,
    _unique_channel_name,
)
from simajilord.runtime import SimajilordRuntime
from simajilord.services.read_aloud import ReadAloudMode, ReadAloudRoute
from simajilord.services.temp_voice import (
    DEFAULT_TEMP_VOICE_GRACE_SECONDS,
    DEFAULT_TEMP_VOICE_ROOM_TEMPLATE,
    TempVoiceRoomEndReason,
    TempVoiceService,
    normalize_temp_voice_room_name,
    render_temp_voice_room_name,
    validate_temp_voice_room_template,
)


@pytest.mark.asyncio
async def test_temp_voice_state_survives_restart_and_preserves_owner_profile(
    tmp_path,
) -> None:
    path = tmp_path / "temp_voice.sqlite3"
    service = TempVoiceService(path)

    config = await service.ensure_config("guild")
    assert config.enabled is True
    assert config.empty_grace_seconds == DEFAULT_TEMP_VOICE_GRACE_SECONDS
    assert config.room_name_template == DEFAULT_TEMP_VOICE_ROOM_TEMPLATE
    creator = await service.add_creator(
        workspace_id="guild",
        channel_id="creator",
        category_id="category",
        permission_source_channel_id="permission-source",
    )
    assert creator.permission_source_channel_id == "permission-source"

    created_at = datetime.now(UTC)
    room = await service.register_room(
        workspace_id="guild",
        channel_id="room-one",
        creator_channel_id="creator",
        owner_id="owner",
        name="Owner room",
        user_limit=4,
        locked=False,
        base_everyone_connect=False,
    )
    assert room.empty_since is None
    assert room.base_everyone_connect is False
    marked = await service.mark_room_empty(
        room.channel_id,
        empty_since=created_at,
    )
    assert marked.empty_since == created_at

    restarted = TempVoiceService(path)
    restored = await restarted.room("room-one")
    assert restored is not None
    assert restored.empty_since == created_at
    restored = await restarted.rename_room(restored.channel_id, "Study room")
    restored = await restarted.set_room_user_limit(restored.channel_id, 8)
    restored = await restarted.set_room_locked(restored.channel_id, True)
    assert (restored.name, restored.user_limit, restored.locked) == (
        "Study room",
        8,
        True,
    )

    profile = await restarted.profile(workspace_id="guild", owner_id="owner")
    assert profile is not None
    assert profile.room_name == "Study room"
    assert profile.user_limit == 8
    assert profile.locked is True
    assert profile.last_room_channel_id == "room-one"

    finished = await restarted.finish_room(
        "room-one",
        reason=TempVoiceRoomEndReason.EMPTY,
    )
    assert finished is not None
    assert await restarted.room("room-one") is None
    remembered = await restarted.profile(workspace_id="guild", owner_id="owner")
    assert remembered is not None
    assert remembered.last_room_channel_id == "room-one"
    assert os.stat(path).st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_creator_permission_source_can_change_without_deleting_setup(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    creator = await service.add_creator(
        workspace_id="guild",
        channel_id="creator",
        category_id="category",
        permission_source_channel_id="source-a",
    )
    assert creator.permission_source_channel_id == "source-a"

    creator = await service.set_creator_permission_source("creator", "source-b")
    assert creator.permission_source_channel_id == "source-b"
    creator = await service.set_creator_permission_source("creator", None)
    assert creator.permission_source_channel_id is None
    assert (await service.config("guild")) is not None

    removed = await service.remove_creator("creator")
    assert removed is not None
    assert await service.creator("creator") is None
    assert (await service.config("guild")) is not None


def test_temp_voice_store_migrates_legacy_creator_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE temp_voice_creators (
                channel_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE temp_voice_rooms (
                channel_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                creator_channel_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                name TEXT NOT NULL,
                user_limit INTEGER NOT NULL,
                locked INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                empty_since TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE temp_voice_room_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                creator_channel_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                name TEXT NOT NULL,
                user_limit INTEGER NOT NULL,
                locked INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                empty_since TEXT,
                ended_at TEXT NOT NULL,
                end_reason TEXT NOT NULL
            )
            """
        )

    TempVoiceService(path)

    with sqlite3.connect(path) as connection:
        creator_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(temp_voice_creators)").fetchall()
        }
        room_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(temp_voice_rooms)").fetchall()
        }
        history_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(temp_voice_room_history)").fetchall()
        }
    assert "permission_source_channel_id" in creator_columns
    assert "base_everyone_connect" in room_columns
    assert "base_everyone_connect" in history_columns


def test_temp_voice_names_reject_invisible_control_characters() -> None:
    assert normalize_temp_voice_room_name("  visible   room ") == "visible room"
    assert (
        render_temp_voice_room_name(
            "{display_name}'s room",
            display_name="Meteo",
            username="meteo",
        )
        == "Meteo's room"
    )
    assert validate_temp_voice_room_template("{username} study") == "{username} study"
    with pytest.raises(UserError, match=r"temp_voice\.room_name_invalid"):
        normalize_temp_voice_room_name("unsafe\x00name")
    with pytest.raises(UserError, match=r"temp_voice\.room_name_invalid"):
        normalize_temp_voice_room_name("replacement \ufffd")
    with pytest.raises(UserError, match=r"temp_voice\.template_invalid"):
        validate_temp_voice_room_template("{unknown} room")
    with pytest.raises(UserError, match=r"temp_voice\.template_invalid"):
        validate_temp_voice_room_template("unsafe\x00template")


def test_unique_temp_voice_names_are_bounded_and_do_not_shadow_existing_names() -> None:
    assert _unique_channel_name("Room", ("Other",)) == "Room"
    duplicate = _unique_channel_name("R" * 100, ("R" * 100,))
    assert duplicate.endswith("· 2")
    assert len(duplicate) <= 100


def test_creation_retry_view_can_disable_retry_and_hide_setup() -> None:
    view = TempVoiceCreateView(
        cast(TempVoiceCog, Mock()),
        requester_id=1,
        guild_id=2,
        creator_channel_id=3,
        creation_enabled=False,
        can_manage=False,
    )
    children = {item.custom_id: item for item in view.children}
    create = children["simajilord:tempvc:create-room"]
    assert isinstance(create, discord.ui.Button)
    assert create.label == "Retry room creation"
    assert create.disabled is True
    assert "simajilord:tempvc:creator-setup" not in children


@pytest.mark.asyncio
async def test_full_category_blocks_creator_before_discord_mutation(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)
    guild = Mock(spec=discord.Guild)
    guild.create_voice_channel = AsyncMock()
    member = Mock(spec=discord.Member)
    member.guild = guild
    category = Mock(spec=discord.CategoryChannel)
    category.channels = [Mock() for _ in range(50)]

    with pytest.raises(UserError, match=r"temp_voice\.category_full"):
        await cog._create_creator_channel(member, category)

    guild.create_voice_channel.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_creator_channel_uses_join_to_create_label(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.me = Mock(spec=discord.Member)
    created_channel = Mock(spec=discord.VoiceChannel)
    created_channel.id = 20
    guild.create_voice_channel = AsyncMock(return_value=created_channel)
    member = Mock(spec=discord.Member)
    member.guild = guild
    category = Mock(spec=discord.CategoryChannel)
    category.id = 30
    category.guild = guild
    category.channels = []
    category.voice_channels = []
    category.permissions_for.return_value = discord.Permissions.all()

    creator = await cog._create_creator_channel(member, category)

    create_call = guild.create_voice_channel.await_args
    assert create_call.args[0] == "Join to create"
    assert create_call.kwargs["category"] is category
    assert creator.channel_id == "20"
    assert creator.permission_source_channel_id == "20"


@pytest.mark.asyncio
async def test_joining_creator_lobby_starts_automatic_room_creation(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
        permission_source_channel_id="20",
    )
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    bot = Mock(spec=commands.Bot)
    cog = TempVoiceCog(bot, runtime)
    created_channel = Mock(spec=discord.VoiceChannel)
    cog._create_room_for_member = AsyncMock(  # type: ignore[method-assign]
        return_value=created_channel
    )

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.me = Mock(spec=discord.Member)
    category = Mock(spec=discord.CategoryChannel)
    category.id = 30
    category.guild = guild
    category.channels = []
    category.permissions_for.return_value = discord.Permissions.all()
    member = Mock(spec=discord.Member)
    member.id = 40
    member.bot = False
    member.guild = guild
    creator_channel = Mock(spec=discord.VoiceChannel)
    creator_channel.id = 20
    creator_channel.guild = guild
    creator_channel.permissions_for.return_value = discord.Permissions.all()
    guild.get_channel.side_effect = lambda channel_id: {
        20: creator_channel,
        30: category,
    }.get(channel_id)
    member.voice = SimpleNamespace(channel=creator_channel)
    before = Mock(spec=discord.VoiceState)
    before.channel = None
    after = Mock(spec=discord.VoiceState)
    after.channel = creator_channel

    await cog.on_voice_state_update(member, before, after)

    creation_call = cog._create_room_for_member.await_args
    assert creation_call.args[0] is member
    assert creation_call.args[1] is creator_channel
    assert creation_call.args[2].channel_id == "20"
    assert creation_call.args[3] is category


@pytest.mark.asyncio
async def test_automatic_creation_rejection_is_journaled_for_private_retry(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
        permission_source_channel_id="20",
    )
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)
    cog._create_or_recover_room_from_creator = AsyncMock(  # type: ignore[method-assign]
        side_effect=UserError("temp_voice.creation_paused")
    )
    cog.record_event = AsyncMock()  # type: ignore[method-assign]

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    member = Mock(spec=discord.Member)
    member.id = 40
    member.bot = False
    member.guild = guild
    creator_channel = Mock(spec=discord.VoiceChannel)
    creator_channel.id = 20
    before = Mock(spec=discord.VoiceState)
    before.channel = None
    after = Mock(spec=discord.VoiceState)
    after.channel = creator_channel

    await cog.on_voice_state_update(member, before, after)

    cog.record_event.assert_awaited_once_with(
        "temp_voice.auto_create_rejected",
        workspace_id="10",
        actor_id="40",
        request_id="voice:40:20",
        payload={
            "creator_channel_id": "20",
            "error": "temp_voice.creation_paused",
        },
    )


@pytest.mark.asyncio
async def test_creation_retry_checks_lobby_and_invokes_room_creation(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
        permission_source_channel_id="20",
    )
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    bot = Mock(spec=commands.Bot)
    cog = TempVoiceCog(bot, runtime)

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    category = Mock(spec=discord.CategoryChannel)
    category.id = 30
    category.channels = []
    creator_channel = Mock(spec=discord.VoiceChannel)
    creator_channel.id = 20
    all_permissions = discord.Permissions.all()
    creator_channel.permissions_for.return_value = all_permissions
    category.permissions_for.return_value = all_permissions
    guild.get_channel.side_effect = lambda channel_id: {
        20: creator_channel,
        30: category,
    }.get(channel_id)
    guild.me = Mock(spec=discord.Member)

    member = Mock(spec=discord.Member)
    member.id = 40
    member.guild = guild
    member.voice = SimpleNamespace(channel=creator_channel)
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 50
    interaction.user = member
    created_channel = Mock(spec=discord.VoiceChannel)
    cog._create_room_for_member = AsyncMock(  # type: ignore[method-assign]
        return_value=created_channel
    )

    channel, recovered = await cog.create_room_from_creator(interaction, 20)

    assert channel is created_channel
    assert recovered is False
    cog._create_room_for_member.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_recovery_move_restores_empty_cleanup_timer(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
        permission_source_channel_id="20",
    )
    await service.register_room(
        workspace_id="10",
        channel_id="90",
        creator_channel_id="20",
        owner_id="40",
        name="Recovered room",
        user_limit=0,
        locked=False,
    )
    await service.mark_room_empty("90")
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    bot = Mock(spec=commands.Bot)
    cog = TempVoiceCog(bot, runtime)

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.me = Mock(spec=discord.Member)
    category = Mock(spec=discord.CategoryChannel)
    category.id = 30
    category.channels = []
    category.permissions_for.return_value = discord.Permissions.all()
    creator_channel = Mock(spec=discord.VoiceChannel)
    creator_channel.id = 20
    creator_channel.permissions_for.return_value = discord.Permissions.all()
    recovered = Mock(spec=discord.VoiceChannel)
    recovered.id = 90
    recovered.members = []
    guild.get_channel.side_effect = lambda channel_id: {
        20: creator_channel,
        30: category,
        90: recovered,
    }.get(channel_id)

    member = Mock(spec=discord.Member)
    member.id = 40
    member.guild = guild
    member.voice = SimpleNamespace(channel=creator_channel)
    member.move_to = AsyncMock(side_effect=RuntimeError("move failed"))
    interaction = Mock(spec=discord.Interaction)
    interaction.id = 50
    interaction.user = member
    cog._recover_owned_empty_room = AsyncMock(  # type: ignore[method-assign]
        return_value=recovered
    )
    cog.schedule_deletion = Mock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="move failed"):
        await cog.create_room_from_creator(interaction, 20)

    retained = await service.room("90")
    assert retained is not None
    assert retained.empty_since is not None
    cog.schedule_deletion.assert_called_once_with(retained)


@pytest.mark.asyncio
async def test_room_creation_copies_selected_voice_permissions(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    creator = await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
        permission_source_channel_id="25",
    )
    journal = SimpleNamespace(append=AsyncMock())
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(
            temp_voice=service,
            audio=SimpleNamespace(find=Mock(return_value=None)),
            read_aloud=SimpleNamespace(get=Mock(return_value=None)),
            journal=journal,
        ),
    )
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)
    cog._send_room_welcome = AsyncMock()  # type: ignore[method-assign]

    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.default_role = Mock(spec=discord.Role)
    bot_member = Mock(spec=discord.Member)
    guild.me = bot_member
    copied_role = Mock(spec=discord.Role)
    category_only_role = Mock(spec=discord.Role)
    copied = discord.PermissionOverwrite(view_channel=True, connect=True)
    source = Mock(spec=discord.VoiceChannel)
    source.id = 25
    source.overwrites = {
        copied_role: copied,
        guild.default_role: discord.PermissionOverwrite(connect=False),
    }
    source.permissions_for.return_value = discord.Permissions.all()
    category = Mock(spec=discord.CategoryChannel)
    category.id = 30
    category.overwrites = {category_only_role: discord.PermissionOverwrite(view_channel=False)}
    creator_channel = Mock(spec=discord.VoiceChannel)
    creator_channel.id = 20
    creator_channel.name = "Join to create"
    creator_channel.user_limit = 0
    category.voice_channels = [creator_channel]
    created = Mock(spec=discord.VoiceChannel)
    created.id = 90
    guild.create_voice_channel = AsyncMock(return_value=created)
    guild.get_channel.side_effect = lambda channel_id: {25: source}.get(channel_id)

    owner = Mock(spec=discord.Member)
    owner.id = 40
    owner.guild = guild
    owner.name = "meteo"
    owner.display_name = "Meteo"
    owner.move_to = AsyncMock()
    config = await service.ensure_config("10")

    result = await cog._create_room_for_member(
        owner,
        creator_channel,
        creator,
        category,
        config,
    )

    assert result is created
    create_call = guild.create_voice_channel.await_args
    overwrites = create_call.kwargs["overwrites"]
    assert copied_role in overwrites
    assert category_only_role not in overwrites
    assert overwrites[owner].view_channel is True
    assert overwrites[owner].connect is True
    assert overwrites[bot_member].manage_channels is True
    assert overwrites[bot_member].move_members is True
    owner.move_to.assert_awaited_once_with(
        created,
        reason="Simajilord TempVC join-to-create",
    )
    tracked = await service.room("90")
    assert tracked is not None
    assert tracked.base_everyone_connect is False


@pytest.mark.asyncio
async def test_unlock_restores_permission_source_everyone_connect_value(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
    )
    room = await service.register_room(
        workspace_id="10",
        channel_id="90",
        creator_channel_id="20",
        owner_id="40",
        name="Private room",
        user_limit=0,
        locked=True,
        base_everyone_connect=False,
    )
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)
    guild = Mock(spec=discord.Guild)
    guild.default_role = Mock(spec=discord.Role)
    channel = Mock(spec=discord.VoiceChannel)
    channel.guild = guild
    channel.overwrites_for.return_value = discord.PermissionOverwrite(connect=False)
    channel.set_permissions = AsyncMock()
    member = Mock(spec=discord.Member)
    interaction = Mock(spec=discord.Interaction)
    cog.require_room_controller = AsyncMock(  # type: ignore[method-assign]
        return_value=(room, channel, member)
    )

    updated = await cog.set_room_lock(interaction, 90, False)

    applied = channel.set_permissions.await_args.kwargs["overwrite"]
    assert applied.connect is False
    assert updated.locked is False
    assert updated.base_everyone_connect is False


@pytest.mark.asyncio
async def test_temp_voice_deletion_suspends_audio_even_when_auto_leave_is_disabled() -> None:
    session = SimpleNamespace(
        destination_id="90",
        output=SimpleNamespace(connected=True),
        auto_leave=False,
        suspend=AsyncMock(),
    )
    dashboard = SimpleNamespace(forget_channel=AsyncMock())
    bot = Mock(spec=commands.Bot)
    bot._simajilord_music_dashboard = dashboard
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(audio=SimpleNamespace(find=Mock(return_value=session))),
    )
    cog = TempVoiceCog(bot, runtime)
    channel = Mock(spec=discord.VoiceChannel)
    channel.id = 90
    room = SimpleNamespace(workspace_id="10", channel_id="90")

    await cog._prepare_audio_for_room_deletion(room, channel)

    session.suspend.assert_awaited_once_with()
    dashboard.forget_channel.assert_awaited_once_with("10", 90)


@pytest.mark.asyncio
async def test_recreated_room_remaps_audio_and_read_aloud_without_connecting() -> None:
    session = SimpleNamespace(remap_suspended_destination=AsyncMock(return_value=True))
    route = ReadAloudRoute(
        workspace_id="10",
        text_channel_id="70",
        audio_destination_id="90",
        mode=ReadAloudMode.QUEUE,
    )
    read_aloud = SimpleNamespace(
        get=Mock(return_value=route),
        configure=AsyncMock(),
    )
    journal = SimpleNamespace(append=AsyncMock())
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(
            audio=SimpleNamespace(find=Mock(return_value=session)),
            read_aloud=read_aloud,
            journal=journal,
        ),
    )
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)

    await cog._remap_saved_voice_destinations(
        10,
        previous_channel_id="90",
        replacement_channel_id="91",
    )

    session.remap_suspended_destination.assert_awaited_once_with(
        expected_destination_id="90",
        replacement_destination_id="91",
    )
    configured = read_aloud.configure.await_args.args[0]
    assert configured.audio_destination_id == "91"
    assert configured.text_channel_id == "70"
    assert journal.append.await_args.kwargs["payload"]["auto_connected"] is False


@pytest.mark.asyncio
async def test_intentional_delete_event_does_not_race_room_retirement(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
    )
    await service.register_room(
        workspace_id="10",
        channel_id="90",
        creator_channel_id="20",
        owner_id="40",
        name="Room",
        user_limit=0,
        locked=False,
    )
    runtime = cast(
        SimajilordRuntime,
        SimpleNamespace(
            temp_voice=service,
            audio=SimpleNamespace(find=Mock(return_value=None)),
        ),
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.VoiceChannel)
    channel.id = 90
    channel.guild = guild
    cog = TempVoiceCog(Mock(spec=commands.Bot), runtime)
    cog._room_deletions_in_flight.add(90)

    await cog.on_guild_channel_delete(channel)

    assert await service.room("90") is not None


@pytest.mark.asyncio
async def test_reconciliation_marks_missed_empty_room_for_delayed_cleanup(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="10",
        channel_id="20",
        category_id="30",
    )
    await service.register_room(
        workspace_id="10",
        channel_id="90",
        creator_channel_id="20",
        owner_id="40",
        name="Room",
        user_limit=0,
        locked=False,
    )
    runtime = cast(SimajilordRuntime, SimpleNamespace(temp_voice=service))
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    creator = Mock(spec=discord.VoiceChannel)
    creator.id = 20
    room_channel = Mock(spec=discord.VoiceChannel)
    room_channel.id = 90
    room_channel.members = []
    guild.get_channel.side_effect = lambda channel_id: {
        20: creator,
        90: room_channel,
    }.get(channel_id)
    bot = Mock(spec=commands.Bot)
    bot.guilds = [guild]
    bot.get_guild.return_value = guild
    cog = TempVoiceCog(bot, runtime)
    cog.schedule_deletion = Mock()  # type: ignore[method-assign]

    await cog.reconcile_tracked_rooms()

    marked = await service.room("90")
    assert marked is not None
    assert marked.empty_since is not None
    cog.schedule_deletion.assert_called_once()


@pytest.mark.asyncio
async def test_empty_grace_timestamp_is_not_reset_by_repeated_leave_events(tmp_path) -> None:
    service = TempVoiceService(tmp_path / "temp_voice.sqlite3")
    await service.add_creator(
        workspace_id="guild",
        channel_id="creator",
        category_id="category",
    )
    await service.register_room(
        workspace_id="guild",
        channel_id="room",
        creator_channel_id="creator",
        owner_id="owner",
        name="Room",
        user_limit=0,
        locked=False,
    )
    first = datetime.now(UTC) - timedelta(minutes=2)
    initially_empty = await service.mark_room_empty("room", empty_since=first)
    repeated = await service.mark_room_empty("room", empty_since=datetime.now(UTC))
    assert initially_empty.empty_since == first
    assert repeated.empty_since == first
