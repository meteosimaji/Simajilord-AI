from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.capabilities.file_scope import file_workspace_id
from simajilord.core import (
    CapabilityEndpoint,
    DisclosureObservation,
    InvocationContext,
)
from simajilord.core.errors import UserError
from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.integrations.discord.platform_actions import (
    DiscordCreateGuildResourceRequest,
    DiscordSetChannelOverwriteRequest,
)
from simajilord.integrations.discord.platform_assets import (
    DiscordCreatePlatformAssetRequest,
)
from simajilord.integrations.discord.platform_automod import (
    DiscordAutoModActionInput,
    DiscordAutoModRuleInput,
    DiscordCreateAutoModRuleRequest,
)
from simajilord.integrations.discord.platform_capabilities import (
    DiscordInspectChannelRequest,
    DiscordListMembersRequest,
    DiscordListPlatformResourcesRequest,
    DiscordListPollVotersRequest,
)
from simajilord.integrations.discord.platform_operations import (
    DiscordChannelOperationRequest,
    DiscordSendDirectMessageRequest,
)
from simajilord.runtime import SimajilordRuntime
from simajilord.services import AgentFileSandbox
from simajilord.services.files import WorkspaceFileProvenance


def _context() -> InvocationContext:
    return InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        origin_resource_id="10",
        grants=frozenset({"files"}),
    )


def _endpoints(
    client: discord.Client,
    runtime: SimajilordRuntime,
) -> dict[str, CapabilityEndpoint]:
    return {
        item.descriptor.name: item
        for item in build_discord_endpoints(client, runtime)
    }


def _member_with_permissions(**permissions: bool) -> Mock:
    member = Mock(spec=discord.Member)
    member.guild_permissions = discord.Permissions(**permissions)
    return member


def test_platform_capability_names_are_unique_and_cover_resource_families() -> None:
    endpoints = _endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    assert len(endpoints) == 111
    assert {
        "discord.list_members",
        "discord.inspect_channel",
        "discord.list_pins",
        "discord.list_reaction_users",
        "discord.list_poll_voters",
        "discord.list_thread_members",
        "discord.list_platform_resources",
        "discord.inspect_application",
        "discord.create_guild_resource",
        "discord.update_guild_resource",
        "discord.delete_guild_resource",
        "discord.message_action",
        "discord.set_channel_overwrite",
        "discord.create_platform_asset",
        "discord.update_platform_asset",
        "discord.delete_platform_asset",
        "discord.create_automod_rule",
        "discord.update_automod_rule",
        "discord.delete_automod_rule",
        "discord.channel_operation",
        "discord.forward_message",
        "discord.send_direct_message",
        "discord.set_bot_presence",
    } <= endpoints.keys()


@pytest.mark.asyncio
async def test_list_members_exposes_presence_activity_voice_and_effective_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(spec=discord.Client)
    client.intents = discord.Intents.all()
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.chunked = True
    guild.member_count = 1
    member = Mock(spec=discord.Member)
    member.id = 7
    member.name = "meteosimaji"
    member.display_name = "めてお"
    member.global_name = "Meteo"
    member.nick = "めてお"
    member.bot = False
    member.system = False
    member.joined_at = datetime(2026, 1, 1, tzinfo=UTC)
    member.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    member.status = discord.Status.online
    activity = Mock(spec=discord.Activity)
    activity.type = discord.ActivityType.playing
    activity.name = "Discord API"
    activity.details = None
    activity.state = None
    activity.url = None
    activity.application_id = None
    activity.created_at = None
    activity.timestamps = {}
    activity.emoji = None
    activity.platform = None
    activity.session_id = None
    activity.sync_id = None
    activity.assets = {}
    activity.buttons = ()
    activity.party = {}
    activity.flags = None
    activity.status_display_type = None
    member.activities = (activity,)
    voice = Mock(spec=discord.VoiceState)
    voice.channel = Mock(spec=discord.VoiceChannel)
    voice.channel.id = 50
    voice.channel.name = "VC"
    member.voice = voice
    everyone = Mock(spec=discord.Role)
    everyone.id = 1
    everyone.name = "@everyone"
    role = Mock(spec=discord.Role)
    role.id = 2
    role.name = "Admin"
    member.roles = [everyone, role]
    member.guild_permissions = discord.Permissions(administrator=True)
    member.pending = False
    member.timed_out_until = None
    member.premium_since = None
    member.display_avatar.url = "https://cdn.discordapp.com/avatar.png"
    guild.members = [member]
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=member),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.list_members"].invoke(
        DiscordListMembersRequest(),
        _context(),
    )

    record = response.members[0]
    assert record.status == "online"
    assert record.presence_available is True
    assert record.activities == ("playing: Discord API",)
    assert record.voice_channel_id == "50"
    assert record.role_names == ("Admin",)
    assert "administrator" in record.enabled_guild_permissions


@pytest.mark.asyncio
async def test_list_poll_voters_pages_one_readable_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = cast(discord.Client, Mock(spec=discord.Client))
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    channel = Mock(spec=discord.TextChannel)
    channel.id = 10
    users = [
        SimpleNamespace(id=7, name="meteo", display_name="めてお", bot=False),
        SimpleNamespace(id=8, name="friend", display_name="Friend", bot=False),
    ]

    async def voters(*, limit: int) -> AsyncIterator[SimpleNamespace]:
        for user in users[:limit]:
            yield user

    answer = SimpleNamespace(id=40, text="Yes", voters=voters)
    poll = SimpleNamespace(
        get_answer=lambda answer_id: answer if answer_id == 40 else None,
    )
    message = Mock(spec=discord.Message)
    message.id = 30
    message.poll = poll
    channel.fetch_message = AsyncMock(return_value=message)
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._readable_message_channel",
        AsyncMock(return_value=(guild, channel)),
    )
    endpoint_by_name = _endpoints(
        client,
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.list_poll_voters"].invoke(
        DiscordListPollVotersRequest(
            channel_id="10",
            message_id="30",
            answer_id="40",
            limit=1,
        ),
        _context(),
    )

    assert response.answer_text == "Yes"
    assert tuple(item.user_id for item in response.voters) == ("7",)
    assert response.next_offset == 1
    assert response.complete is False


@pytest.mark.asyncio
async def test_application_skus_require_owner_admin_and_omit_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    actor.id = 7
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    application = SimpleNamespace(
        owner=SimpleNamespace(id=7),
        team=None,
    )
    client.application_info = AsyncMock(return_value=application)
    client.fetch_skus = AsyncMock(
        return_value=[
            SimpleNamespace(
                id=80,
                name="Supporter",
                type=SimpleNamespace(name="durable"),
                application_id=81,
                slug="supporter",
                flags=(),
            )
        ]
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="sku"),
        _context(),
    )

    assert response.complete is True
    assert len(response.resources) == 1
    resource = response.resources[0]
    assert resource.resource_id == "80"
    assert resource.name == "Supporter"
    fields = {field.key: field.value for field in resource.fields}
    assert fields == {
        "type": "durable",
        "application_id": "81",
        "slug": "supporter",
        "flags": "",
    }
    assert all("token" not in field.key and "secret" not in field.key for field in resource.fields)


@pytest.mark.asyncio
async def test_application_skus_reject_non_owner_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    actor.id = 7
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    client.application_info = AsyncMock(
        return_value=SimpleNamespace(
            owner=SimpleNamespace(id=99),
            team=None,
        )
    )
    client.fetch_skus = AsyncMock()
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    with pytest.raises(UserError, match=r"discord\.application_owner_required"):
        await endpoint_by_name["discord.list_platform_resources"].invoke(
            DiscordListPlatformResourcesRequest(kind="sku"),
            _context(),
        )

    client.fetch_skus.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_threads_and_guild_preview_are_fetched_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    thread = Mock(spec=discord.Thread)
    thread.id = 30
    thread.name = "live-thread"
    thread.type = discord.ChannelType.public_thread
    thread.parent_id = 10
    thread.owner_id = 7
    thread.archived = False
    thread.locked = False
    thread.invitable = True
    thread.message_count = 4
    thread.member_count = 2
    thread.total_message_sent = 5
    thread.slowmode_delay = 0
    thread.auto_archive_duration = 1_440
    thread.archive_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    thread.applied_tags = []
    guild.active_threads = AsyncMock(return_value=[thread])
    preview = SimpleNamespace(
        id=1,
        name="Server",
        description="A server",
        approximate_member_count=12,
        approximate_presence_count=3,
        features=["COMMUNITY"],
        emojis=[],
        stickers=[],
        icon=None,
        splash=None,
        discovery_splash=None,
    )
    client.fetch_guild_preview = AsyncMock(return_value=preview)
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_view_channel",
        lambda channel, member: True,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_read_messages",
        lambda channel, member: True,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_read_private_thread",
        lambda channel, member: True,
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    active = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="active_thread"),
        _context(),
    )
    guild_preview = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="guild_preview"),
        _context(),
    )

    assert tuple(item.resource_id for item in active.resources) == ("30",)
    assert guild_preview.resources[0].name == "Server"
    preview_fields = {
        field.key: field.value for field in guild_preview.resources[0].fields
    }
    assert preview_fields["approximate_presence_count"] == "3"
    guild.active_threads.assert_awaited_once_with()
    client.fetch_guild_preview.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_disabled_widget_returns_settings_instead_of_public_widget_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.http.request = AsyncMock(
        return_value={"enabled": False, "channel_id": None},
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    guild.widget = AsyncMock()
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="widget"),
        _context(),
    )

    fields = {field.key: field.value for field in response.resources[0].fields}
    assert fields["enabled"] == "false"
    assert fields["public_widget_available"] == "false"
    assert fields["image_url"] == "https://discord.com/api/guilds/1/widget.png"
    guild.widget.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfigured_welcome_screen_and_unavailable_vanity_are_state_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = cast(discord.Client, Mock(spec=discord.Client))
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.features = ["COMMUNITY"]
    guild.vanity_url_code = None
    guild.welcome_screen = AsyncMock(side_effect=discord.NotFound(Mock(), "missing"))
    guild.vanity_invite = AsyncMock()
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        client,
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    welcome = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="welcome_screen"),
        _context(),
    )
    vanity = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(kind="vanity_invite"),
        _context(),
    )

    welcome_fields = {
        field.key: field.value for field in welcome.resources[0].fields
    }
    vanity_fields = {
        field.key: field.value for field in vanity.resources[0].fields
    }
    assert welcome_fields["configured"] == "false"
    assert welcome_fields["community_enabled"] == "true"
    assert vanity_fields["feature_enabled"] == "false"
    assert vanity_fields["configured"] == "false"
    guild.vanity_invite.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_regions_and_prune_estimate_use_safe_official_get_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.http.request = AsyncMock(
        side_effect=[
            [
                {
                    "id": "japan",
                    "name": "Japan",
                    "optimal": True,
                    "deprecated": False,
                    "custom": False,
                }
            ],
            [
                {
                    "id": "vip-japan",
                    "name": "VIP Japan",
                    "optimal": False,
                    "deprecated": False,
                    "custom": True,
                }
            ],
            {"pruned": 3},
        ]
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(
        administrator=True,
        manage_guild=True,
        kick_members=True,
    )
    bot = _member_with_permissions(
        administrator=True,
        manage_guild=True,
        kick_members=True,
    )
    guild.me = bot
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    global_regions = await endpoint_by_name[
        "discord.list_platform_resources"
    ].invoke(
        DiscordListPlatformResourcesRequest(kind="voice_region"),
        _context(),
    )
    guild_regions = await endpoint_by_name[
        "discord.list_platform_resources"
    ].invoke(
        DiscordListPlatformResourcesRequest(kind="guild_voice_region"),
        _context(),
    )
    prune = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(
            kind="prune_count",
            prune_days=14,
            prune_role_ids=("22", "33"),
        ),
        _context(),
    )

    assert global_regions.resources[0].resource_id == "japan"
    assert guild_regions.resources[0].resource_id == "vip-japan"
    prune_fields = {
        field.key: field.value for field in prune.resources[0].fields
    }
    assert prune_fields == {
        "days": "14",
            "include_role_ids": "22, 33",
        "estimated_member_count": "3",
        "mutates_members": "false",
    }
    routes = [
        call.args[0] for call in client.http.request.await_args_list
    ]
    assert [(route.method, route.path) for route in routes] == [
        ("GET", "/voice/regions"),
        ("GET", "/guilds/{guild_id}/regions"),
        ("GET", "/guilds/{guild_id}/prune"),
    ]
    assert client.http.request.await_args_list[2].kwargs["params"] == {
        "days": 14,
        "include_roles": "22,33",
    }


@pytest.mark.asyncio
async def test_application_subscriptions_support_list_and_exact_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.application_info = AsyncMock(
        return_value=SimpleNamespace(
            owner=SimpleNamespace(id=7),
            team=None,
        )
    )
    subscription = {
        "id": "101",
        "user_id": "7",
        "sku_ids": ["80"],
        "entitlement_ids": ["90"],
        "renewal_sku_ids": ["80"],
        "current_period_start": "2026-07-01T00:00:00+00:00",
        "current_period_end": "2026-08-01T00:00:00+00:00",
        "status": 0,
        "canceled_at": None,
    }
    client.http.request = AsyncMock(
        side_effect=[[subscription], subscription],
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    actor.id = 7
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    with pytest.raises(
        UserError,
        match=r"discord\.subscription_user_id_required",
    ):
        await endpoint_by_name["discord.list_platform_resources"].invoke(
            DiscordListPlatformResourcesRequest(
                kind="subscription",
                resource_id="80",
            ),
            _context(),
        )
    listed = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(
            kind="subscription",
            resource_id="80",
            user_id="7",
        ),
        _context(),
    )
    exact = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(
            kind="subscription",
            resource_id="80",
            subresource_id="101",
        ),
        _context(),
    )

    assert listed.resources[0].resource_id == "101"
    assert exact.resources[0].resource_id == "101"
    exact_fields = {
        field.key: field.value for field in exact.resources[0].fields
    }
    assert exact_fields["user_id"] == "7"
    assert exact_fields["entitlement_ids"] == "90"
    routes = [call.args[0] for call in client.http.request.await_args_list]
    assert [(route.method, route.path) for route in routes] == [
        ("GET", "/skus/{sku_id}/subscriptions"),
        ("GET", "/skus/{sku_id}/subscriptions/{subscription_id}"),
    ]
    assert client.http.request.await_args_list[0].kwargs["params"] == {
        "limit": 16,
        "user_id": "7",
    }
    assert client.http.request.await_args_list[1].kwargs["params"] is None


@pytest.mark.asyncio
async def test_application_role_connection_metadata_is_owner_only_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.application_id = 55
    client.application_info = AsyncMock(
        return_value=SimpleNamespace(
            owner=SimpleNamespace(id=7),
            team=None,
        )
    )
    client.http.request = AsyncMock(
        return_value=[
            {
                "type": 7,
                "key": "verified",
                "name": "Verified",
                "name_localizations": {"ja": "確認済み"},
                "description": "Whether this member is verified.",
                "description_localizations": {"ja": "確認済みか"},
            }
        ]
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    actor.id = 7
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.list_platform_resources"].invoke(
        DiscordListPlatformResourcesRequest(
            kind="role_connection_metadata",
        ),
        _context(),
    )

    assert response.resources[0].resource_id == "verified"
    assert {field.key: field.value for field in response.resources[0].fields} == {
        "type": "7",
        "key": "verified",
        "description": "Whether this member is verified.",
        "name_locales": "ja",
        "description_locales": "ja",
    }
    route = client.http.request.await_args.args[0]
    assert (route.method, route.path) == (
        "GET",
        "/applications/{application_id}/role-connections/metadata",
    )


@pytest.mark.asyncio
async def test_private_thread_inspection_requires_actual_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = cast(discord.Client, Mock(spec=discord.Client))
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    guild.me = bot
    thread = Mock(spec=discord.Thread)
    thread.type = discord.ChannelType.private_thread
    guild.get_channel_or_thread.return_value = thread
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._require_common_guild",
        AsyncMock(return_value=actor),
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_view_channel",
        lambda channel, member: True,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_read_messages",
        lambda channel, member: True,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_capabilities._can_read_private_thread",
        lambda channel, member: member is bot,
    )
    endpoint_by_name = _endpoints(
        client,
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    with pytest.raises(
        UserError,
        match=r"discord\.agent_read_channel_forbidden",
    ):
        await endpoint_by_name["discord.inspect_channel"].invoke(
            DiscordInspectChannelRequest(channel_id="10"),
            _context(),
        )


@pytest.mark.asyncio
async def test_administrator_can_set_a_channel_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    channel = Mock(spec=discord.TextChannel)
    channel.id = 10
    channel.permissions_for.side_effect = (
        lambda member: member.guild_permissions
    )
    channel.set_permissions = AsyncMock()
    role = Mock(spec=discord.Role)
    role.id = 20
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._requested_guild",
        lambda client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._guild_channel",
        lambda selected_guild, channel_id: channel,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._guild_role",
        lambda selected_guild, role_id: role,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._require_role_above",
        lambda member, target: None,
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.set_channel_overwrite"].invoke(
        DiscordSetChannelOverwriteRequest(
            channel_id="10",
            target_kind="role",
            target_id="20",
            allowed_permissions=("view_channel", "send_messages"),
            denied_permissions=("manage_messages",),
        ),
        _context(),
    )

    assert response.changed is True
    overwrite = channel.set_permissions.await_args.kwargs["overwrite"]
    assert overwrite.view_channel is True
    assert overwrite.send_messages is True
    assert overwrite.manage_messages is False


@pytest.mark.asyncio
async def test_voice_status_uses_official_route_and_requires_live_permissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.http.request = AsyncMock(return_value=None)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(
        administrator=True,
        manage_channels=True,
        set_voice_channel_status=True,
    )
    bot = _member_with_permissions(
        administrator=True,
        manage_channels=True,
        set_voice_channel_status=True,
    )
    bot.voice = None
    channel = Mock(spec=discord.VoiceChannel)
    channel.id = 10
    channel.permissions_for.side_effect = (
        lambda member: member.guild_permissions
    )
    guild.get_channel_or_thread.return_value = channel
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_operations._requested_guild",
        lambda selected_client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_operations._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, client),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.channel_operation"].invoke(
        DiscordChannelOperationRequest(
            operation="set_voice_status",
            channel_id="10",
            name="  Audit   in progress  ",
            reason="Live API audit",
        ),
        _context(),
    )

    assert response.name == "Audit in progress"
    route = client.http.request.await_args.args[0]
    assert (route.method, route.path) == (
        "PUT",
        "/channels/{channel_id}/voice-status",
    )
    assert client.http.request.await_args.kwargs["json"] == {
        "status": "Audit in progress"
    }


@pytest.mark.asyncio
async def test_platform_asset_creation_reads_only_the_workspace_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "files")
    context = _context()
    runtime.files.import_bytes(
        file_workspace_id(context),
        "emoji.png",
        b"png",
        provenance=WorkspaceFileProvenance(
            owner_actor_ids=("7",),
            origin_guild_id="1",
            origin_channel_id="10",
            origin_visibility="guild_public",
            sensitivity="guild_public",
            source_resources=(("1", "10", "guild_public"),),
        ),
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    created = Mock(spec=discord.Emoji)
    created.id = 99
    created.name = "simaji"
    created.url = "https://cdn.discordapp.com/emoji.png"
    guild.create_custom_emoji = AsyncMock(return_value=created)
    actor = _member_with_permissions(create_expressions=True)
    bot = _member_with_permissions(create_expressions=True)
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_assets._requested_guild",
        lambda client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_assets._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        cast(SimajilordRuntime, runtime),
    )

    response = await endpoint_by_name["discord.create_platform_asset"].invoke(
        DiscordCreatePlatformAssetRequest(
            kind="guild_emoji",
            name="simaji",
            path="emoji.png",
        ),
        context,
    )

    assert response.resource_id == "99"
    assert guild.create_custom_emoji.await_args.kwargs["image"] == b"png"


@pytest.mark.asyncio
async def test_application_emoji_rejects_undeclassified_workspace_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = Mock(spec=SimajilordRuntime)
    runtime.files = AgentFileSandbox(tmp_path / "files")
    context = _context()
    runtime.files.import_bytes(
        file_workspace_id(context),
        "emoji.png",
        b"png",
        provenance=WorkspaceFileProvenance(
            owner_actor_ids=("7",),
            origin_guild_id="1",
            origin_channel_id="10",
            origin_visibility="guild_public",
            sensitivity="guild_public",
            source_resources=(("1", "10", "guild_public"),),
        ),
    )
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_assets._requested_guild",
        lambda client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_assets._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    client = Mock(spec=discord.Client)
    client.create_application_emoji = AsyncMock()

    with pytest.raises(UserError, match=r"discord\.information_flow_forbidden"):
        await _endpoints(
            cast(discord.Client, client),
            cast(SimajilordRuntime, runtime),
        )["discord.create_platform_asset"].invoke(
            DiscordCreatePlatformAssetRequest(
                kind="application_emoji",
                name="simaji",
                path="emoji.png",
            ),
            context,
        )

    client.create_application_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_public_guild_template_rejects_unknown_target_audience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._requested_guild",
        lambda client, context, guild_id: guild,
    )
    write_members = AsyncMock()
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_actions._write_members",
        write_members,
    )
    context = InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="agent",
        request_id="event",
        origin_resource_id="10",
        disclosure_observations=(
            DisclosureObservation(
                source_workspace_id="1",
                source_resource_id="10",
                visibility="guild_public",
                relation_to_origin="same_or_narrower",
            ),
        ),
    )

    with pytest.raises(UserError, match=r"discord\.information_flow_forbidden"):
        await _endpoints(
            cast(discord.Client, Mock(spec=discord.Client)),
            cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
        )["discord.create_guild_resource"].invoke(
            DiscordCreateGuildResourceRequest(
                kind="template",
                name="Public template",
            ),
            context,
        )

    write_members.assert_not_awaited()


@pytest.mark.asyncio
async def test_automod_create_uses_typed_trigger_actions_and_admin_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    created = Mock(spec=discord.AutoModRule)
    created.id = 88
    created.name = "Anti spam"
    created.enabled = True
    created.trigger.type = discord.AutoModRuleTriggerType.spam
    created.actions = (
        discord.AutoModRuleAction(
            type=discord.AutoModRuleActionType.block_message,
        ),
    )
    guild.create_automod_rule = AsyncMock(return_value=created)
    actor = _member_with_permissions(administrator=True)
    bot = _member_with_permissions(administrator=True)
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_automod._requested_guild",
        lambda client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_automod._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.create_automod_rule"].invoke(
        DiscordCreateAutoModRuleRequest(
            rule=DiscordAutoModRuleInput(
                name="Anti spam",
                trigger_kind="spam",
                actions=(DiscordAutoModActionInput(kind="block_message"),),
                enabled=True,
            )
        ),
        _context(),
    )

    assert response.rule_id == "88"
    call = guild.create_automod_rule.await_args.kwargs
    assert call["event_type"] is discord.AutoModRuleEventType.message_send
    assert call["trigger"].type is discord.AutoModRuleTriggerType.spam
    assert call["actions"][0].type is discord.AutoModRuleActionType.block_message

    guild.create_automod_rule.reset_mock()
    dispatch = AsyncMock()
    effect = Mock()
    effect.dispatch = dispatch
    effect.complete_without_dispatch = AsyncMock()
    with pytest.raises(UserError, match=r"discord\.audit_reason_too_long"):
        await endpoint_by_name["discord.create_automod_rule"].invoke(
            DiscordCreateAutoModRuleRequest(
                rule=DiscordAutoModRuleInput(
                    name="Anti spam",
                    trigger_kind="spam",
                    actions=(DiscordAutoModActionInput(kind="block_message"),),
                    enabled=True,
                ),
                reason="x" * 401,
            ),
            InvocationContext(
                actor_id="7",
                workspace_id="1",
                transport="agent",
                request_id="invalid-automod",
                origin_resource_id="10",
                grants=frozenset({"files"}),
                external_effect_dispatch=effect,
            ),
        )

    dispatch.assert_not_awaited()
    guild.create_automod_rule.assert_not_awaited()


@pytest.mark.asyncio
async def test_administrator_can_send_suppressed_dm_to_shared_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    actor = _member_with_permissions(administrator=True)
    actor.id = 7
    bot = _member_with_permissions(administrator=True)
    target = Mock(spec=discord.Member)
    target.id = 8
    posted = Mock(spec=discord.Message)
    posted.id = 90
    posted.channel.id = 91
    target.send = AsyncMock(return_value=posted)
    guild.get_member.return_value = target
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_operations._requested_guild",
        lambda client, context, guild_id: guild,
    )
    monkeypatch.setattr(
        "simajilord.integrations.discord.platform_operations._write_members",
        AsyncMock(return_value=(actor, bot)),
    )
    endpoint_by_name = _endpoints(
        cast(discord.Client, Mock(spec=discord.Client)),
        cast(SimajilordRuntime, Mock(spec=SimajilordRuntime)),
    )

    response = await endpoint_by_name["discord.send_direct_message"].invoke(
        DiscordSendDirectMessageRequest(user_id="8", content="hello"),
        _context(),
    )

    assert response.message_id == "90"
    assert target.send.await_args.kwargs["suppress_embeds"] is True
    assert target.send.await_args.kwargs["allowed_mentions"].to_dict() == {
        "parse": [],
    }
