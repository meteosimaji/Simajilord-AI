from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import discord
import pytest

from simajilord.agent.actions import ActionClassification, action_policy
from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import CapabilityRegistry, InvocationContext, RiskLevel
from simajilord.core.errors import UserError
from simajilord.integrations.discord.capabilities import (
    DiscordBulkDeleteRequest,
    DiscordChannelSettingRequest,
    DiscordCreatedChannelDeleteRequest,
    DiscordCreatedRoleDeleteRequest,
    DiscordDeleteOwnMessagesRequest,
    DiscordMessageWriteRequest,
    DiscordRoleMemberRequest,
    DiscordThreadUpdateRequest,
    build_discord_endpoints,
)
from simajilord.runtime import SimajilordRuntime

ACT_WRITES = {
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
    "discord.set_timeout",
    "discord.delete_message",
    "discord.bulk_delete_messages",
    "discord.kick_member",
    "discord.ban_member",
    "discord.unban_member",
}


def _endpoints(client: discord.Client) -> dict[str, object]:
    return {
        item.descriptor.name: item
        for item in build_discord_endpoints(
            client,
            Mock(spec=SimajilordRuntime),
        )
    }


def _created_role_undo_harness(
    channels: list[object],
) -> tuple[dict[str, object], Mock, Mock]:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.owner_id = 1
    guild.chunked = True
    guild.member_count = 2
    actor = Mock(spec=discord.Member)
    actor.id = 7
    actor.guild = guild
    actor.guild_permissions = discord.Permissions(manage_roles=True)
    actor.top_role = MagicMock(spec=discord.Role)
    actor.top_role.__le__.return_value = False
    bot = Mock(spec=discord.Member)
    bot.id = 99
    bot.guild = guild
    bot.guild_permissions = discord.Permissions(manage_roles=True)
    bot.top_role = MagicMock(spec=discord.Role)
    bot.top_role.__le__.return_value = False
    guild.get_member.return_value = actor
    guild.me = bot
    guild.members = [actor, bot]
    role = Mock(spec=discord.Role)
    role.id = 40
    role.members = []
    role.delete = AsyncMock()
    guild.get_role.return_value = role
    guild.fetch_channels = AsyncMock(return_value=channels)
    client.get_guild.return_value = guild
    return _endpoints(cast(discord.Client, client)), guild, role


def test_act_capabilities_are_unique_discoverable_and_explicitly_classified() -> None:
    client = Mock(spec=discord.Client)
    endpoints = build_discord_endpoints(
        cast(discord.Client, client),
        Mock(spec=SimajilordRuntime),
    )
    registry = CapabilityRegistry()
    for item in endpoints:
        registry.register(item)
    names = {item.descriptor.name for item in endpoints}

    assert names >= ACT_WRITES
    assert {
        item.descriptor.name
        for item in registry.search("moderation ban timeout messages", limit=20)
    } & {"discord.ban_member", "discord.set_timeout"}
    assert {
        item.descriptor.name for item in registry.search("スレッド rename archive member", limit=20)
    } & {"discord.update_thread", "discord.add_thread_member"}
    for capability in ACT_WRITES:
        policy = action_policy(capability)
        assert (
            policy.classification is ActionClassification.NON_UNDOABLE
            or policy.undo_capability is not None
        ), capability
    assert (
        registry.endpoint("discord.ban_member").descriptor.risk
        is RiskLevel.DESTRUCTIVE
    )
    assert (
        registry.endpoint("discord.delete_message").descriptor.risk
        is RiskLevel.DESTRUCTIVE
    )


@pytest.mark.asyncio
async def test_bulk_delete_rejects_mixed_age_batch_before_any_delete() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        administrator=False,
        view_channel=True,
        read_message_history=True,
        manage_messages=True,
    )
    now = datetime.now(UTC)
    messages = {
        31: SimpleNamespace(id=31, created_at=now - timedelta(days=1)),
        32: SimpleNamespace(id=32, created_at=now - timedelta(days=15)),
    }
    channel.fetch_message = AsyncMock(side_effect=lambda message_id: messages[message_id])
    channel.delete_messages = AsyncMock()
    guild.get_channel_or_thread.return_value = channel
    client.get_guild.return_value = guild

    with pytest.raises(
        UserError,
        match=r"discord\.bulk_delete_message_too_old",
    ):
        await _endpoints(cast(discord.Client, client))[
            "discord.bulk_delete_messages"
        ].invoke(
            DiscordBulkDeleteRequest(
                channel_id="20",
                message_ids=("31", "32"),
                reason="Remove test spam",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="discord:message:30",
                resource_ids=("20",),
                origin_resource_id="20",
            ),
        )

    channel.delete_messages.assert_not_awaited()


def test_reversible_act_receipts_store_only_ids_and_small_scalar_state() -> None:
    thread_policy = action_policy("discord.update_thread")
    assert thread_policy.undo_arguments is not None
    assert thread_policy.undo_arguments(
        SimpleNamespace(),
        SimpleNamespace(
            changed=True,
            thread_id="41",
            name="after",
            archived=True,
            old_name="before",
            old_archived=False,
            undo_fingerprint="thread-state",
        ),
    ) == {
        "thread_id": "41",
        "name": "before",
        "archived": False,
        "expected_name": "after",
        "expected_archived": True,
        "expected_undo_fingerprint": "thread-state",
    }
    forum_policy = action_policy("discord.create_forum_post")
    assert forum_policy.undo_arguments is not None
    assert forum_policy.undo_arguments(
        SimpleNamespace(),
        SimpleNamespace(
            thread_id="42",
            name="Report",
            archived=False,
            undo_fingerprint="forum-state",
        ),
    ) == {
        "thread_id": "42",
        "archived": True,
        "expected_name": "Report",
        "expected_archived": False,
        "expected_undo_fingerprint": "forum-state",
    }

    channel_policy = action_policy("discord.update_channel_settings")
    assert channel_policy.undo_arguments is not None
    assert channel_policy.undo_arguments(
        SimpleNamespace(),
        SimpleNamespace(
            changed=True,
            channel_id="20",
            topic="after",
            slowmode_seconds=10,
            old_topic=None,
            old_slowmode_seconds=5,
        ),
    ) == {
        "channel_id": "20",
        "topic": None,
        "slowmode_seconds": 5,
        "expected_topic": "after",
        "expected_slowmode_seconds": 10,
    }
    assert (
        action_policy("discord.edit_own_message").classification
        is ActionClassification.NON_UNDOABLE
    )
    assert (
        action_policy("discord.delete_message").classification
        is ActionClassification.NON_UNDOABLE
    )


@pytest.mark.asyncio
async def test_destructive_moderation_is_searchable_only_with_explicit_policy() -> None:
    registry = CapabilityRegistry()
    for item in build_discord_endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        Mock(spec=SimajilordRuntime),
    ):
        registry.register(item)
    catalog = AgentToolCatalog(
        registry,
        ("discord.ban_member",),
        required_grants={"discord.ban_member": "moderation"},
        eager_capabilities=(),
        write_capabilities=("discord.ban_member",),
        destructive_capabilities=("discord.ban_member",),
    )
    context = InvocationContext(
        actor_id="7",
        workspace_id="10",
        transport="agent",
        request_id="discord:message:30",
        grants=frozenset({"moderation"}),
        approvals=frozenset({"discord.ban_member"}),
    )

    output = await catalog.invoke(
        namespace="simajilord",
        tool_name="capability_search",
        arguments={"query": "ban member moderation", "limit": 3},
        context=context,
        max_output_characters=4_000,
    )

    assert "discord.ban_member" in output.text
    assert '"risk":"destructive"' in output.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor_pin", "bot_pin"),
    ((False, True), (True, False)),
)
async def test_pin_requires_actor_and_bot_effective_pin_messages(
    actor_pin: bool,
    bot_pin: bool,
) -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    guild.get_channel_or_thread.return_value = channel

    def permissions(member: object) -> SimpleNamespace:
        return SimpleNamespace(
            view_channel=True,
            read_message_history=True,
            pin_messages=bot_pin if member is bot else actor_pin,
        )

    channel.permissions_for.side_effect = permissions
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))
    context = InvocationContext(
        actor_id="7",
        workspace_id="10",
        transport="agent",
        request_id="discord:message:30",
        resource_ids=("20",),
        origin_resource_id="30",
    )

    with pytest.raises(UserError, match=r"discord\.pin_messages_required"):
        await endpoints["discord.pin_message"].invoke(
            DiscordMessageWriteRequest(
                channel_id="20",
                message_id="31",
                reason="Keep the decision visible",
                evidence_message_ids=("30",),
            ),
            context,
        )

    channel.fetch_message.assert_not_called()


@pytest.mark.asyncio
async def test_pin_can_target_an_authorized_non_origin_channel() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
        pin_messages=True,
    )
    message = SimpleNamespace(
        id=31,
        pinned=False,
        pin=AsyncMock(),
    )
    channel.fetch_message = AsyncMock(return_value=message)
    guild.get_channel_or_thread.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    response = await endpoints["discord.pin_message"].invoke(
        DiscordMessageWriteRequest(
            channel_id="20",
            message_id="31",
            reason="Keep the decision visible",
            evidence_message_ids=("30",),
        ),
        InvocationContext(
            actor_id="7",
            workspace_id="10",
            transport="agent",
            request_id="discord:message:30",
            resource_ids=("20", "30"),
            origin_resource_id="30",
        ),
    )

    assert response.changed is True
    message.pin.assert_awaited_once()
    assert "actor=7" in message.pin.await_args.kwargs["reason"]


@pytest.mark.asyncio
async def test_pin_allows_administrator_without_explicit_pin_messages() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        administrator=True,
        view_channel=True,
        pin_messages=False,
    )
    message = SimpleNamespace(id=31, pinned=False, pin=AsyncMock())
    channel.fetch_message = AsyncMock(return_value=message)
    guild.get_channel_or_thread.return_value = channel
    client.get_guild.return_value = guild

    await _endpoints(cast(discord.Client, client))["discord.pin_message"].invoke(
        DiscordMessageWriteRequest(
            channel_id="20",
            message_id="31",
            reason="Keep the decision visible",
            evidence_message_ids=("30",),
        ),
        InvocationContext(
            actor_id="7",
            workspace_id="10",
            transport="agent",
            request_id="discord:message:30",
            resource_ids=("20", "30"),
            origin_resource_id="30",
        ),
    )

    message.pin.assert_awaited_once()


@pytest.mark.asyncio
async def test_grouped_host_undo_validates_ownership_before_deleting() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
    )
    first = SimpleNamespace(
        id=301,
        author=bot,
        delete=AsyncMock(),
    )
    foreign = SimpleNamespace(
        id=302,
        author=SimpleNamespace(id=8),
        delete=AsyncMock(),
    )
    channel.fetch_message = AsyncMock(side_effect=(first, foreign))
    guild.get_channel_or_thread.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.message_not_owned"):
        await endpoints["discord.delete_own_messages"].invoke(
            DiscordDeleteOwnMessagesRequest(
                channel_id="20",
                message_ids="301,302",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:response",
            ),
        )

    first.delete.assert_not_awaited()
    foreign.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_settings_undo_rejects_newer_human_state() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.topic = "human update"
    channel.slowmode_delay = 5
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        manage_channels=True,
    )
    channel.edit = AsyncMock()
    guild.get_channel.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"action\.undo_conflict"):
        await endpoints["discord.update_channel_settings"].invoke(
            DiscordChannelSettingRequest(
                channel_id="20",
                topic="before",
                slowmode_seconds=0,
                expected_topic="agent update",
                expected_slowmode_seconds=5,
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:channel",
                origin_resource_id="20",
            ),
        )

    channel.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_settings_undo_retry_accepts_already_restored_target() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.topic = "before"
    channel.slowmode_delay = 0
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        manage_channels=True,
    )
    channel.edit = AsyncMock()
    guild.get_channel.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    response = await endpoints["discord.update_channel_settings"].invoke(
        DiscordChannelSettingRequest(
            channel_id="20",
            topic="before",
            slowmode_seconds=0,
            expected_topic="agent update",
            expected_slowmode_seconds=5,
        ),
        InvocationContext(
            actor_id="7",
            workspace_id="10",
            transport="agent",
            request_id="undo:channel:retry",
            origin_resource_id="20",
        ),
    )

    assert response.changed is False
    channel.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_role_undo_rejects_a_modified_role() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    guild.chunked = True
    role = Mock(spec=discord.Role)
    role.id = 40
    role.name = "renamed by a human"
    role.colour = SimpleNamespace(value=0)
    role.permissions = SimpleNamespace(value=0)
    role.hoist = False
    role.mentionable = False
    role.display_icon = None
    role.members = []
    role.delete = AsyncMock()
    guild.get_role.return_value = role
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"action\.undo_conflict"):
        await endpoints["discord.delete_created_role"].invoke(
            DiscordCreatedRoleDeleteRequest(
                role_id="40",
                undo_fingerprint="state-before-human-edit",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:role",
            ),
        )

    role.delete.assert_not_awaited()


@pytest.mark.parametrize(
    "channel_type",
    (
        discord.CategoryChannel,
        discord.TextChannel,
        discord.VoiceChannel,
        discord.StageChannel,
        discord.ForumChannel,
    ),
    ids=("category", "text", "voice", "stage", "forum"),
)
@pytest.mark.asyncio
async def test_created_role_undo_rejects_live_channel_overwrite_reference(
    channel_type: type[object],
) -> None:
    channel = Mock(spec=channel_type)
    endpoints, _guild, role = _created_role_undo_harness([channel])
    channel.overwrites = {
        role: discord.PermissionOverwrite(view_channel=True),
    }

    with pytest.raises(UserError, match=r"action\.undo_target_in_use"):
        await endpoints["discord.delete_created_role"].invoke(
            DiscordCreatedRoleDeleteRequest(role_id="40"),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:role:overwrite",
            ),
        )

    role.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_role_undo_checks_thread_parent_overwrite() -> None:
    parent = Mock(spec=discord.ForumChannel)
    thread = Mock(spec=discord.Thread)
    parent.id = 20
    thread.parent_id = parent.id
    endpoints, guild, role = _created_role_undo_harness([parent])
    guild.threads = [thread]
    parent.overwrites = {
        role: discord.PermissionOverwrite(view_channel=True),
    }

    with pytest.raises(UserError, match=r"action\.undo_target_in_use"):
        await endpoints["discord.delete_created_role"].invoke(
            DiscordCreatedRoleDeleteRequest(role_id="40"),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:role:thread-parent",
            ),
        )

    role.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_role_undo_checks_synced_category_overwrite() -> None:
    category = Mock(spec=discord.CategoryChannel)
    child = Mock(spec=discord.TextChannel)
    category.id = 20
    child.category_id = category.id
    child.permissions_synced = True
    endpoints, _guild, role = _created_role_undo_harness([category, child])
    overwrite = discord.PermissionOverwrite(view_channel=True)
    category.overwrites = {role: overwrite}
    child.overwrites = {role: overwrite}

    with pytest.raises(UserError, match=r"action\.undo_target_in_use"):
        await endpoints["discord.delete_created_role"].invoke(
            DiscordCreatedRoleDeleteRequest(role_id="40"),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:role:synced-category",
            ),
        )

    role.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_role_undo_succeeds_after_overwrite_is_removed() -> None:
    channel = Mock(spec=discord.TextChannel)
    channel.overwrites = {}
    endpoints, guild, role = _created_role_undo_harness([channel])

    response = await endpoints["discord.delete_created_role"].invoke(
        DiscordCreatedRoleDeleteRequest(role_id="40"),
        InvocationContext(
            actor_id="7",
            workspace_id="10",
            transport="agent",
            request_id="undo:role:overwrite-removed",
        ),
    )

    assert response.deleted is True
    guild.fetch_channels.assert_awaited_once_with()
    role.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_created_role_undo_fails_closed_when_channel_state_is_unavailable() -> None:
    endpoints, guild, role = _created_role_undo_harness([])
    guild.fetch_channels.side_effect = discord.DiscordException(
        "channel lookup failed"
    )

    with pytest.raises(
        UserError,
        match=r"action\.undo_target_state_uncertain",
    ):
        await endpoints["discord.delete_created_role"].invoke(
            DiscordCreatedRoleDeleteRequest(role_id="40"),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:role:channel-fetch-failed",
            ),
        )

    role.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_channel_undo_rejects_modified_empty_channel() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.name = "renamed-by-human"
    channel.topic = None
    channel.slowmode_delay = 0
    channel.nsfw = False
    channel.category_id = None
    channel.default_auto_archive_duration = 1440
    channel.default_thread_slowmode_delay = 0
    channel.overwrites = {}
    channel.delete = AsyncMock()
    guild.get_channel.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"action\.undo_conflict"):
        await endpoints["discord.delete_created_channel"].invoke(
            DiscordCreatedChannelDeleteRequest(
                channel_id="20",
                undo_fingerprint="state-before-human-edit",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:channel",
            ),
        )

    channel.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_created_thread_undo_rejects_newer_discussion() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot
    thread = Mock(spec=discord.Thread)
    thread.id = 50
    thread.parent_id = 20
    thread.name = "bug report"
    thread.archived = False
    thread.locked = False
    thread.invitable = True
    thread.auto_archive_duration = 1440
    thread.slowmode_delay = 0
    thread.last_message_id = 999
    thread.applied_tags = []
    thread.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        manage_threads=True,
    )
    thread.edit = AsyncMock()
    guild.get_thread.return_value = thread
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"action\.undo_conflict"):
        await endpoints["discord.update_thread"].invoke(
            DiscordThreadUpdateRequest(
                thread_id="50",
                archived=True,
                expected_name="bug report",
                expected_archived=False,
                expected_undo_fingerprint="before-new-message",
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="undo:thread",
                origin_resource_id="20",
            ),
        )

    thread.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_role_assignment_enforces_both_hierarchies() -> None:
    class Rank:
        def __init__(self, value: int) -> None:
            self.value = value

        def __le__(self, other: object) -> bool:
            return isinstance(other, Rank) and self.value <= other.value

    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.owner_id = 1
    permissions = SimpleNamespace(manage_roles=True, administrator=False)
    actor = SimpleNamespace(
        id=7,
        bot=False,
        guild=guild,
        guild_permissions=permissions,
        top_role=Rank(5),
    )
    bot = SimpleNamespace(
        id=99,
        bot=True,
        guild=guild,
        guild_permissions=permissions,
        top_role=Rank(10),
    )
    target = SimpleNamespace(
        id=8,
        bot=False,
        guild=guild,
        guild_permissions=SimpleNamespace(administrator=False),
        top_role=Rank(6),
        roles=[],
        add_roles=AsyncMock(),
    )
    role = SimpleNamespace(
        id=40,
        managed=False,
        is_default=lambda: False,
    )
    guild.me = bot
    guild.get_member.side_effect = lambda member_id: {7: actor, 8: target}.get(member_id)
    guild.get_role.return_value = role
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.member_hierarchy_forbidden"):
        await endpoints["discord.assign_role"].invoke(
            DiscordRoleMemberRequest(
                user_id="8",
                role_id="40",
                reason="Grant project access",
                evidence_message_ids=("30",),
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="discord:message:30",
            ),
        )

    target.add_roles.assert_not_awaited()


@pytest.mark.asyncio
async def test_administrator_still_cannot_bypass_member_role_hierarchy() -> None:
    class Rank:
        def __init__(self, value: int) -> None:
            self.value = value

        def __le__(self, other: object) -> bool:
            return isinstance(other, Rank) and self.value <= other.value

    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.owner_id = 1
    administrator = SimpleNamespace(
        manage_roles=False,
        administrator=True,
    )
    actor = SimpleNamespace(
        id=7,
        bot=False,
        guild=guild,
        guild_permissions=administrator,
        top_role=Rank(5),
    )
    bot = SimpleNamespace(
        id=99,
        bot=True,
        guild=guild,
        guild_permissions=administrator,
        top_role=Rank(10),
    )
    target = SimpleNamespace(
        id=8,
        bot=False,
        guild=guild,
        guild_permissions=SimpleNamespace(administrator=False),
        top_role=Rank(6),
        roles=[],
        add_roles=AsyncMock(),
    )
    role = SimpleNamespace(
        id=40,
        managed=False,
        is_default=lambda: False,
    )
    guild.me = bot
    guild.get_member.side_effect = lambda member_id: {7: actor, 8: target}.get(member_id)
    guild.get_role.return_value = role
    client.get_guild.return_value = guild
    endpoints = _endpoints(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.member_hierarchy_forbidden"):
        await endpoints["discord.assign_role"].invoke(
            DiscordRoleMemberRequest(
                user_id="8",
                role_id="40",
                reason="Grant project access",
                evidence_message_ids=("30",),
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="10",
                transport="agent",
                request_id="discord:message:30",
            ),
        )

    target.add_roles.assert_not_awaited()
