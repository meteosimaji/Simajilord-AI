from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.agent.tools import AgentToolCatalog
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.core.errors import UserError
from simajilord.integrations.discord.capabilities import (
    DiscordConnectVoiceRequest,
    DiscordDeleteOwnMessageRequest,
    DiscordGetMessageRequest,
    DiscordListArchivedThreadsRequest,
    DiscordListChannelsRequest,
    DiscordListRolesRequest,
    DiscordListServersRequest,
    DiscordMemberModerationRequest,
    DiscordPollRequest,
    DiscordReactionRequest,
    DiscordReadMessagesRequest,
    DiscordSearchMessagesRequest,
    build_discord_endpoints,
)
from simajilord.runtime import SimajilordRuntime


def _endpoint_map(client: discord.Client) -> dict[str, object]:
    return {
        endpoint.descriptor.name: endpoint
        for endpoint in build_discord_endpoints(
            client,
            Mock(spec=SimajilordRuntime),
        )
    }


def test_discord_staged_search_tools_explain_filters_and_continuation() -> None:
    client = Mock(spec=discord.Client)
    registry = CapabilityRegistry()
    for item in build_discord_endpoints(
        cast(discord.Client, client),
        Mock(spec=SimajilordRuntime),
    ):
        if item.descriptor.name in {
            "discord.list_archived_threads",
            "discord.list_servers",
            "discord.list_channels",
            "discord.list_roles",
            "discord.read_messages",
            "discord.search_messages",
        }:
            registry.register(item)
    catalog = AgentToolCatalog(
        registry,
        (
            "discord.list_archived_threads",
            "discord.list_servers",
            "discord.list_channels",
            "discord.list_roles",
            "discord.read_messages",
            "discord.search_messages",
        ),
        eager_capabilities=(
            "discord.list_archived_threads",
            "discord.list_servers",
            "discord.list_channels",
            "discord.list_roles",
            "discord.read_messages",
            "discord.search_messages",
        ),
    )

    tools = {
        str(tool["name"]): tool
        for namespace in catalog.dynamic_specs(_agent_context())
        for tool in cast(list[dict[str, object]], namespace["tools"])
    }
    search_properties = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], tools["discord_search_messages"]["inputSchema"])[
            "properties"
        ],
    )
    read_properties = cast(
        dict[str, dict[str, object]],
        cast(dict[str, object], tools["discord_read_messages"]["inputSchema"])[
            "properties"
        ],
    )

    assert "not semantic paraphrase search" in str(
        search_properties["content"]["description"]
    )
    assert "next_before_message_id" in str(
        search_properties["before_message_id"]["description"]
    )
    assert "next_offset" in str(search_properties["offset"]["description"])
    assert "next_after_message_id" in str(
        search_properties["after_message_id"]["description"]
    )
    assert "both requester and bot" in str(
        search_properties["channel_ids"]["description"]
    )
    assert "next_before_message_id" in str(
        read_properties["before_message_id"]["description"]
    )
    assert "total_results is approximate" in str(
        tools["discord_search_messages"]["description"]
    )
    assert {
        "discord_list_servers",
        "discord_list_channels",
        "discord_list_archived_threads",
        "discord_list_roles",
    } <= tools.keys()


def test_discord_research_capabilities_are_found_from_natural_japanese() -> None:
    client = Mock(spec=discord.Client)
    registry = CapabilityRegistry()
    for item in build_discord_endpoints(
        cast(discord.Client, client),
        Mock(spec=SimajilordRuntime),
    ):
        if item.descriptor.name in {
            "discord.get_message",
            "discord.list_archived_threads",
            "discord.list_roles",
            "discord.read_messages",
            "discord.search_messages",
        }:
            registry.register(item)

    past = {
        item.descriptor.name
        for item in registry.search("過去のメッセージを探す", limit=3)
    }
    popularity = {
        item.descriptor.name
        for item in registry.search("このサーバーで人気の理由を分析", limit=3)
    }
    archived = {
        item.descriptor.name
        for item in registry.search("アーカイブ済みフォーラム投稿を探す", limit=3)
    }
    roles = {
        item.descriptor.name
        for item in registry.search("既存ロールを名前で検索", limit=3)
    }

    assert "discord.search_messages" in past
    assert {
        "discord.search_messages",
        "discord.read_messages",
    } <= popularity
    assert "discord.list_archived_threads" in archived
    assert "discord.list_roles" in roles


@pytest.mark.asyncio
async def test_list_channels_is_bounded_and_paginated() -> None:
    client = Mock(spec=discord.Client)
    guild, channel, _, _ = _visibility_guild(10, 20)
    channel.id = 20
    channel.name = "a"
    channel.type = discord.ChannelType.text
    channel.category_id = None
    second = Mock(spec=discord.TextChannel)
    second.id = 21
    second.name = "b"
    second.type = discord.ChannelType.text
    second.category_id = None
    second.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in {7, 99}
    )
    guild.channels = [second, channel]
    guild.text_channels = [second, channel]
    client.guilds = [guild]
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    first = await endpoints["discord.list_channels"].invoke(
        DiscordListChannelsRequest(limit=1),
        _agent_context(),
    )
    second_page = await endpoints["discord.list_channels"].invoke(
        DiscordListChannelsRequest(offset=cast(int, first.next_offset), limit=1),
        _agent_context(),
    )

    assert len(first.channels) == 1
    assert first.next_offset == 1
    assert first.complete is False
    assert len(second_page.channels) == 1
    assert second_page.next_offset is None
    assert second_page.complete is True


@pytest.mark.asyncio
async def test_forum_and_archived_posts_are_discoverable_and_searchable() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    other = SimpleNamespace(id=8, bot=False)
    forum = Mock(spec=discord.ForumChannel)
    forum.id = 20
    forum.name = "bugs"
    forum.type = discord.ChannelType.forum
    forum.category_id = None
    forum.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in {7, 8, 99}
    )
    archived = Mock(spec=discord.Thread)
    archived.id = 30
    archived.name = "old-bug"
    archived.type = discord.ChannelType.public_thread
    archived.parent_id = 20
    archived.guild = guild
    archived.locked = False
    archived.archive_timestamp = datetime(2026, 7, 1, tzinfo=UTC)
    archived.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in {7, 8, 99}
    )
    archived.members = []
    archived.fetch_message = AsyncMock(return_value=_fetched_message(archived))

    async def archived_threads(
        *,
        limit: int,
        before: datetime | None,
    ) -> AsyncIterator[Mock]:
        assert limit == 2
        assert before is None
        yield archived

    forum.archived_threads = archived_threads
    guild.get_member.side_effect = lambda member_id: {
        7: actor,
        99: bot,
    }.get(member_id)
    guild.fetch_member = AsyncMock(return_value=actor)
    guild.me = bot
    guild.members = [actor, other, bot]
    guild.member_count = 3
    guild.chunked = True
    guild.channels = [forum]
    guild.text_channels = []
    guild.forums = [forum]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    guild.get_channel.side_effect = lambda channel_id: (
        forum if channel_id == 20 else None
    )
    guild.get_channel_or_thread.side_effect = lambda channel_id: (
        forum if channel_id == 20 else None
    )
    client.get_guild.return_value = guild
    client.fetch_channel = AsyncMock(return_value=archived)
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={
            "total_results": 1,
            "threads": [
                {
                    "id": "30",
                    "parent_id": "20",
                    "type": discord.ChannelType.public_thread.value,
                }
            ],
            "messages": [
                [
                    {
                        "id": "31",
                        "channel_id": "30",
                        "author": {"id": "7", "username": "member"},
                        "content": "archived bug report",
                        "timestamp": "2026-07-01T00:00:00+00:00",
                        "attachments": [],
                    }
                ]
            ],
        }
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    listed = await endpoints["discord.list_channels"].invoke(
        DiscordListChannelsRequest(),
        _agent_context(),
    )
    archived_page = await endpoints["discord.list_archived_threads"].invoke(
        DiscordListArchivedThreadsRequest(parent_channel_id="20", limit=1),
        _agent_context(),
    )
    search = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="bug report",
            channel_ids=("20",),
        ),
        _agent_context(),
    )
    exact = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            channel_id="30",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert [(item.channel_id, item.kind) for item in listed.channels] == [
        ("20", "forum")
    ]
    assert archived_page.threads[0].channel_id == "30"
    assert archived_page.complete is True
    assert search.messages[0].channel_id == "30"
    assert search.messages[0].visibility == "guild_public"
    assert exact.channel_id == "30"


@pytest.mark.asyncio
async def test_list_roles_resolves_existing_role_ids_and_assignability() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    guild.owner_id = 700
    actor = SimpleNamespace(
        id=7,
        bot=False,
        guild=guild,
        guild_permissions=SimpleNamespace(
            administrator=True,
            manage_roles=True,
        ),
    )
    bot = SimpleNamespace(
        id=99,
        bot=True,
        guild=guild,
        guild_permissions=SimpleNamespace(
            administrator=False,
            manage_roles=False,
        ),
    )
    role = Mock(spec=discord.Role)
    role.id = 40
    role.name = "Verified Member"
    role.position = 5
    role.colour = SimpleNamespace(value=0x123456)
    role.managed = False
    role.mentionable = True
    role.hoist = False
    role.members = [actor]
    role.is_default.return_value = False
    guild.roles = [role]
    guild.me = bot
    guild.get_member.return_value = actor
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.list_roles"].invoke(
        DiscordListRolesRequest(query="verified"),
        _agent_context(),
    )

    assert response.roles[0].role_id == "40"
    assert response.roles[0].assignable_by_requester is True
    assert response.roles[0].assignable_by_bot is False


def _agent_context(*, resource_ids: tuple[str, ...] = ("20", "21")) -> InvocationContext:
    return InvocationContext(
        actor_id="7",
        workspace_id="10",
        transport="agent",
        request_id="discord:message:30",
        origin_resource_id="20",
        resource_ids=resource_ids,
    )


def _readable_guild(
    guild: Mock,
    channel: Mock,
    *,
    actor_id: int = 7,
) -> None:
    actor = SimpleNamespace(id=actor_id, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    permissions = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
        connect=True,
        administrator=False,
        manage_threads=False,
    )
    channel.permissions_for.return_value = permissions
    guild.get_member.return_value = actor
    guild.me = bot
    guild.members = [actor, bot]
    guild.member_count = 2
    guild.chunked = True
    guild.text_channels = [channel]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    guild.get_channel_or_thread.return_value = channel


def _permission(*, readable: bool) -> SimpleNamespace:
    return SimpleNamespace(
        view_channel=readable,
        read_message_history=readable,
        connect=True,
        administrator=False,
        manage_threads=False,
    )


def _visibility_guild(
    guild_id: int,
    channel_id: int,
    *,
    actor_id: int = 7,
    other_id: int | None = 8,
    source_readers: set[int] | None = None,
) -> tuple[Mock, Mock, SimpleNamespace, SimpleNamespace]:
    guild = Mock(spec=discord.Guild)
    guild.id = guild_id
    channel = Mock(spec=discord.TextChannel)
    channel.id = channel_id
    actor = SimpleNamespace(id=actor_id, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    other = SimpleNamespace(id=other_id, bot=False) if other_id is not None else None
    members = [actor, bot, *([other] if other is not None else [])]
    readers = source_readers or {member.id for member in members}
    channel.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in readers
    )
    guild.get_member.side_effect = lambda member_id: next(
        (member for member in members if member.id == member_id),
        None,
    )
    guild.fetch_member = AsyncMock(return_value=actor)
    guild.me = bot
    guild.members = members
    guild.member_count = len(members)
    guild.chunked = True
    guild.text_channels = [channel]
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    guild.get_channel_or_thread.return_value = channel
    return guild, channel, actor, bot


def _fetched_message(
    channel: Mock,
    *,
    message_id: int = 31,
    edited_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        channel=channel,
        author=SimpleNamespace(id=7, display_name="Member", bot=False),
        content="source content",
        created_at=datetime(2026, 7, 29, 9, 31, tzinfo=UTC),
        edited_at=edited_at,
        attachments=[],
        stickers=[],
        embeds=[],
        poll=None,
        components=[],
        reference=None,
        is_system=lambda: False,
    )


@pytest.mark.asyncio
async def test_discord_read_messages_returns_explicit_chronological_order() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    author = SimpleNamespace(id=7, display_name="Member", bot=False)

    def message(message_id: int, content: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=message_id,
            channel=channel,
            author=author,
            content=content,
            created_at=datetime(2026, 7, 29, 9, message_id, tzinfo=UTC),
            attachments=[],
            reference=None,
            embeds=[],
            poll=None,
            components=[],
        )

    newest = message(32, "newest")
    oldest = message(31, "oldest")
    lookahead = message(30, "lookahead")

    async def history(*, limit: int, before: object) -> AsyncIterator[SimpleNamespace]:
        assert limit == 3
        del before
        for item in (newest, oldest, lookahead):
            yield item

    channel.history = history
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.read_messages"].invoke(
        DiscordReadMessagesRequest(channel_id="20", limit=2),
        _agent_context(),
    )

    assert tuple(item.message_id for item in response.messages) == ("31", "32")
    assert response.oldest_message_id == "31"
    assert response.newest_message_id == "32"
    assert response.order == "oldest_to_newest"
    assert response.has_more is True
    assert response.next_before_message_id == "31"


@pytest.mark.asyncio
async def test_discord_search_messages_stays_inside_authorized_channels() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={
            "total_results": 2,
            "messages": [
                [
                    {
                        "id": "31",
                        "channel_id": "20",
                        "author": {
                            "id": "7",
                            "username": "member",
                            "global_name": "Member",
                        },
                        "content": "the exact historical phrase",
                        "timestamp": "2026-07-29T09:39:50.831000+00:00",
                        "edited_timestamp": "2026-07-29T09:40:00+00:00",
                        "attachments": [],
                        "reactions": [
                            {
                                "count": 4,
                                "emoji": {"id": None, "name": "🔥"},
                            }
                        ],
                        "thread": {"id": "41"},
                    }
                ]
            ],
        }
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="historical phrase",
            channel_ids=("20",),
            author_ids=("7",),
        ),
        _agent_context(),
    )

    assert response.total_results == 2
    assert tuple(message.message_id for message in response.messages) == ("31",)
    assert response.messages[0].reaction_count == 4
    assert response.messages[0].reaction_summary[0].emoji == "🔥"
    assert response.messages[0].thread_id == "41"
    assert (
        response.messages[0].edited_at_iso
        == "2026-07-29T09:40:00+00:00"
    )
    assert response.oldest_message_id == "31"
    assert response.newest_message_id == "31"
    assert response.has_more is True
    assert response.next_offset == 1
    assert response.next_before_message_id is None
    assert response.next_after_message_id is None
    assert response.complete is False
    params = dict(client.http.request.await_args.kwargs["params"])
    assert params["content"] == "historical phrase"
    assert params["channel_id"] == "20"
    assert params["author_id"] == "7"
    assert params["offset"] == "0"


@pytest.mark.asyncio
async def test_relevance_search_over_500_channels_uses_lossless_global_cursor() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot = SimpleNamespace(id=99, bot=True)
    channels: list[Mock] = []
    channel_by_id: dict[int, Mock] = {}
    for channel_id in range(1_000, 1_501):
        channel = Mock(spec=discord.TextChannel)
        channel.id = channel_id
        channel.permissions_for.side_effect = lambda member: _permission(
            readable=member.id in {7, 99}
        )
        channels.append(channel)
        channel_by_id[channel_id] = channel
    guild.get_member.return_value = actor
    guild.me = bot
    guild.members = [actor, bot]
    guild.member_count = 2
    guild.chunked = True
    guild.text_channels = channels
    guild.forums = []
    guild.threads = []
    guild.voice_channels = []
    guild.stage_channels = []
    guild.get_channel_or_thread.side_effect = channel_by_id.get
    client.get_guild.return_value = guild
    client.http = Mock()

    def payload(message_id: str, channel_id: str, total: int) -> dict[str, object]:
        return {
            "total_results": total,
            "messages": [
                [
                    {
                        "id": message_id,
                        "channel_id": channel_id,
                        "author": {"id": "7", "username": "member"},
                        "content": f"result {message_id}",
                        "timestamp": "2026-07-29T09:39:50+00:00",
                        "attachments": [],
                    }
                ]
            ],
        }

    async def search_request(_route: object, *, params: list[tuple[str, str]]):
        channel_ids = [value for key, value in params if key == "channel_id"]
        offset = dict(params)["offset"]
        if channel_ids == ["1500"]:
            return payload("9002", "1500", 1)
        if offset == "0":
            return payload("9001", "1000", 2)
        return payload("9003", "1001", 2)

    client.http.request = AsyncMock(side_effect=search_request)
    endpoints = _endpoint_map(cast(discord.Client, client))

    first = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(content="result", limit=1),
        _agent_context(),
    )
    second = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="result",
            limit=1,
            cursor=cast(str, first.next_cursor),
        ),
        _agent_context(),
    )

    assert tuple(item.message_id for item in first.messages) == ("9001",)
    assert first.next_offset is None
    assert first.next_cursor is not None
    assert tuple(item.message_id for item in second.messages) == ("9002",)
    second_call_params = client.http.request.await_args_list[2:4]
    assert [
        dict(call.kwargs["params"])["offset"]
        for call in second_call_params
    ] == ["1", "0"]


@pytest.mark.asyncio
async def test_discord_relevance_search_never_returns_an_invalid_offset() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={
            "total_results": 20_000,
            "messages": [
                [
                    {
                        "id": "31",
                        "channel_id": "20",
                        "author": {"id": "7", "username": "member"},
                        "content": "edge result",
                        "timestamp": "2026-07-29T09:39:50+00:00",
                        "attachments": [],
                    }
                ]
            ],
        }
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="edge",
            channel_ids=("20",),
            offset=9_975,
            limit=1,
        ),
        _agent_context(),
    )

    assert response.has_more is True
    assert response.next_offset is None
    assert response.search_window_exhausted is True
    assert response.complete is False


@pytest.mark.asyncio
async def test_discord_search_accepts_bounded_iso_period_without_phrase() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={"total_results": 0, "messages": []}
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            channel_ids=("20",),
            after_iso="2026-07-01T00:00:00+00:00",
            before_iso="2026-07-30T00:00:00+00:00",
            limit=5,
        ),
        _agent_context(),
    )

    params = dict(client.http.request.await_args.kwargs["params"])
    assert "content" not in params
    assert params["min_id"].isdecimal()
    assert params["max_id"].isdecimal()
    assert int(params["min_id"]) < int(params["max_id"])
    assert response.messages == ()
    assert response.next_before_message_id is None


@pytest.mark.asyncio
async def test_discord_search_messages_rejects_channel_outside_actor_scope() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock()
    endpoints = _endpoint_map(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.agent_read_channel_forbidden"):
        await endpoints["discord.search_messages"].invoke(
            DiscordSearchMessagesRequest(
                content="private phrase",
                channel_ids=("99",),
            ),
            _agent_context(),
        )

    client.http.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_discord_search_messages_reports_indexing_without_guessing() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={"code": 110000, "retry_after": 2.5}
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="newly indexed phrase",
            channel_ids=("20",),
        ),
        _agent_context(),
    )

    assert response.messages == ()
    assert response.indexing is True
    assert response.retry_after_seconds == 2.5


@pytest.mark.asyncio
async def test_discord_search_reports_partial_deep_historical_index() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    _readable_guild(guild, channel)
    client.get_guild.return_value = guild
    client.http = Mock()
    client.http.request = AsyncMock(
        return_value={
            "doing_deep_historical_index": True,
            "total_results": 0,
            "messages": [],
        }
    )
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.search_messages"].invoke(
        DiscordSearchMessagesRequest(
            content="older phrase",
            channel_ids=("20",),
        ),
        _agent_context(),
    )

    assert response.messages == ()
    assert response.indexing is True
    assert response.retry_after_seconds is None
    assert response.complete is False


@pytest.mark.asyncio
async def test_cross_guild_read_requires_common_actor_and_bot_membership() -> None:
    client = Mock(spec=discord.Client)
    origin_guild, origin_channel, _, _ = _visibility_guild(10, 20)
    source_guild, source_channel, _, _ = _visibility_guild(11, 21)
    edited_at = datetime(2026, 7, 29, 9, 35, tzinfo=UTC)
    source_channel.fetch_message = AsyncMock(
        return_value=_fetched_message(source_channel, edited_at=edited_at)
    )
    client.get_guild.side_effect = lambda guild_id: {
        10: origin_guild,
        11: source_guild,
    }.get(guild_id)
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            guild_id="11",
            channel_id="21",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert response.guild_id == "11"
    assert response.channel_id == "21"
    assert response.disclosure_to_origin == "same_or_narrower"
    assert response.disclosure_warning is None
    assert response.edited_at_iso == edited_at.isoformat()
    assert origin_channel.permissions_for.call_count > 0


@pytest.mark.asyncio
async def test_list_servers_live_checks_uncached_memberships_and_pages() -> None:
    client = Mock(spec=discord.Client)
    common, _, _, _ = _visibility_guild(10, 20)
    common.name = "B cached"
    uncached, _, actor, _ = _visibility_guild(11, 21)
    uncached.name = "A uncached"
    uncached.get_member.side_effect = lambda _member_id: None
    uncached.fetch_member = AsyncMock(return_value=actor)
    client.guilds = [common, uncached]
    client.get_guild.return_value = common
    endpoints = _endpoint_map(cast(discord.Client, client))

    first = await endpoints["discord.list_servers"].invoke(
        DiscordListServersRequest(limit=1),
        _agent_context(),
    )
    second = await endpoints["discord.list_servers"].invoke(
        DiscordListServersRequest(offset=cast(int, first.next_offset), limit=1),
        _agent_context(),
    )

    assert tuple(item.server_id for item in first.servers) == ("11",)
    assert first.checked_server_count == 1
    assert first.membership_checks_complete is True
    assert first.next_offset == 1
    assert first.complete is False
    uncached.fetch_member.assert_awaited_once_with(7)
    assert tuple(item.server_id for item in second.servers) == ("10",)
    assert second.servers[0].readable_channel_count == 1
    assert second.next_offset is None
    assert second.complete is True


@pytest.mark.asyncio
async def test_list_servers_reports_uncertain_live_membership_lookup() -> None:
    client = Mock(spec=discord.Client)
    guild, _, _, _ = _visibility_guild(10, 20)
    guild.name = "Uncertain"
    guild.get_member.side_effect = lambda _member_id: None
    http_response = Mock()
    http_response.status = 503
    http_response.reason = "Service Unavailable"
    http_response.headers = {}
    guild.fetch_member = AsyncMock(
        side_effect=discord.HTTPException(
            http_response,
            {"code": 0, "message": "temporary failure"},
        )
    )
    client.guilds = [guild]
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.list_servers"].invoke(
        DiscordListServersRequest(),
        _agent_context(),
    )

    assert response.servers == ()
    assert response.checked_server_count == 1
    assert response.uncertain_membership_count == 1
    assert response.membership_checks_complete is False
    assert response.next_offset is None
    assert response.complete is False


@pytest.mark.asyncio
async def test_cross_guild_read_rejects_requester_who_is_not_a_member() -> None:
    client = Mock(spec=discord.Client)
    origin_guild, _, _, _ = _visibility_guild(10, 20)
    source_guild, source_channel, _, _ = _visibility_guild(11, 21)
    source_guild.get_member.side_effect = lambda _member_id: None
    source_guild.fetch_member = AsyncMock(
        return_value=SimpleNamespace(id=123456, bot=False)
    )
    client.get_guild.side_effect = lambda guild_id: {
        10: origin_guild,
        11: source_guild,
    }.get(guild_id)
    endpoints = _endpoint_map(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.member_required"):
        await endpoints["discord.get_message"].invoke(
            DiscordGetMessageRequest(
                guild_id="11",
                channel_id="21",
                message_id="31",
            ),
            _agent_context(),
        )

    source_channel.fetch_message.assert_not_called()


@pytest.mark.asyncio
async def test_private_source_to_broader_destination_returns_advisory_warning() -> None:
    client = Mock(spec=discord.Client)
    guild, destination, _, _ = _visibility_guild(10, 20)
    source = Mock(spec=discord.TextChannel)
    source.id = 21
    source.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in {7, 99}
    )
    source.fetch_message = AsyncMock(return_value=_fetched_message(source))
    guild.text_channels = [destination, source]
    guild.get_channel_or_thread.side_effect = lambda channel_id: {
        20: destination,
        21: source,
    }.get(channel_id)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            channel_id="21",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert response.visibility == "restricted"
    assert response.disclosure_to_origin == "broader"
    assert response.disclosure_warning is not None
    assert "known reader" in response.disclosure_warning


@pytest.mark.asyncio
async def test_public_source_to_narrower_destination_has_no_warning() -> None:
    client = Mock(spec=discord.Client)
    guild, source, _, _ = _visibility_guild(10, 21)
    destination = Mock(spec=discord.TextChannel)
    destination.id = 20
    destination.permissions_for.side_effect = lambda member: _permission(
        readable=member.id in {7, 99}
    )
    source.fetch_message = AsyncMock(return_value=_fetched_message(source))
    guild.text_channels = [destination, source]
    guild.get_channel_or_thread.side_effect = lambda channel_id: {
        20: destination,
        21: source,
    }.get(channel_id)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            channel_id="21",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert response.visibility == "guild_public"
    assert response.disclosure_to_origin == "same_or_narrower"
    assert response.disclosure_warning is None


@pytest.mark.asyncio
async def test_role_gated_source_with_all_effective_members_has_no_false_warning() -> None:
    client = Mock(spec=discord.Client)
    guild, source, _, _ = _visibility_guild(10, 21)
    destination = Mock(spec=discord.TextChannel)
    destination.id = 20
    destination.permissions_for.return_value = _permission(readable=True)
    source.fetch_message = AsyncMock(return_value=_fetched_message(source))
    guild.text_channels = [destination, source]
    guild.get_channel_or_thread.side_effect = lambda channel_id: {
        20: destination,
        21: source,
    }.get(channel_id)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            channel_id="21",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert response.disclosure_to_origin == "same_or_narrower"
    assert response.disclosure_warning is None


@pytest.mark.asyncio
async def test_incomplete_member_cache_reports_uncertain_without_false_warning() -> None:
    client = Mock(spec=discord.Client)
    guild, source, _, _ = _visibility_guild(10, 21, other_id=None)
    destination = Mock(spec=discord.TextChannel)
    destination.id = 20
    destination.permissions_for.return_value = _permission(readable=True)
    source.fetch_message = AsyncMock(return_value=_fetched_message(source))
    guild.chunked = False
    guild.member_count = 100
    guild.text_channels = [destination, source]
    guild.get_channel_or_thread.side_effect = lambda channel_id: {
        20: destination,
        21: source,
    }.get(channel_id)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.get_message"].invoke(
        DiscordGetMessageRequest(
            channel_id="21",
            message_id="31",
            include_reply_context=False,
        ),
        _agent_context(),
    )

    assert response.visibility == "uncertain"
    assert response.disclosure_to_origin == "uncertain"
    assert response.disclosure_warning is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability_name", "reacted"),
    (
        ("discord.add_reaction", True),
        ("discord.remove_own_reaction", False),
    ),
)
async def test_reaction_capabilities_act_only_as_the_bot(
    capability_name: str,
    reacted: bool,
) -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    bot_member = Mock(spec=discord.Member)
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
    )
    guild.get_channel_or_thread.return_value = channel
    message = Mock(spec=discord.Message)
    message.id = 30
    message.channel = channel
    message.add_reaction = AsyncMock()
    message.remove_reaction = AsyncMock()
    channel.fetch_message = AsyncMock(return_value=message)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints[capability_name].invoke(
        DiscordReactionRequest(
            channel_id="20",
            message_id="30",
            emoji="✅",
        ),
        _agent_context(),
    )

    assert response.reacted is reacted
    assert response.changed is reacted
    if reacted:
        message.add_reaction.assert_awaited_once_with("✅")
        message.remove_reaction.assert_not_awaited()
    else:
        message.remove_reaction.assert_awaited_once_with("✅", bot_member)
        message.add_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_capability_rejects_whitespace_emoji() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    bot_member = Mock(spec=discord.Member)
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
    )
    guild.get_channel_or_thread.return_value = channel
    message = Mock(spec=discord.Message)
    message.id = 30
    message.channel = channel
    message.add_reaction = AsyncMock()
    channel.fetch_message = AsyncMock(return_value=message)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"discord\.reaction_emoji_invalid"):
        await endpoints["discord.add_reaction"].invoke(
            DiscordReactionRequest(
                channel_id="20",
                message_id="30",
                emoji="not one emoji",
            ),
            _agent_context(),
        )

    message.add_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_poll_is_limited_to_active_channel_and_effective_permissions() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False)
    bot_member = SimpleNamespace(id=99, bot=True)
    guild.get_member.return_value = actor
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        send_messages=True,
        create_polls=True,
    )
    channel.send = AsyncMock(return_value=SimpleNamespace(id=31))
    guild.get_channel_or_thread.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.create_poll"].invoke(
        DiscordPollRequest(
            channel_id="20",
            question="Which option?",
            options=("A", "B"),
        ),
        _agent_context(),
    )

    assert response.message_id == "31"
    channel.send.assert_awaited_once()

    with pytest.raises(UserError, match=r"discord\.agent_read_channel_forbidden"):
        await endpoints["discord.create_poll"].invoke(
            DiscordPollRequest(
                channel_id="99",
                question="Outside scope?",
                options=("A", "B"),
            ),
            _agent_context(),
        )


@pytest.mark.asyncio
async def test_agent_voice_connect_requires_requester_in_selected_channel() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    actor = SimpleNamespace(id=7, bot=False, voice=None)
    guild.get_member.return_value = actor
    channel = Mock(spec=discord.VoiceChannel)
    channel.id = 40
    guild.get_channel.return_value = channel
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    with pytest.raises(UserError, match=r"audio\.same_voice_required"):
        await endpoints["discord.connect_voice"].invoke(
            DiscordConnectVoiceRequest(channel_id="40"),
            _agent_context(),
        )


@pytest.mark.asyncio
async def test_delete_own_message_rejects_member_content_and_deletes_bot_content() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    bot_member = Mock(spec=discord.Member)
    bot_member.id = 99
    guild.me = bot_member
    channel = Mock(spec=discord.TextChannel)
    channel.id = 20
    channel.permissions_for.return_value = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
    )
    guild.get_channel_or_thread.return_value = channel
    message = Mock(spec=discord.Message)
    message.id = 30
    message.channel = channel
    message.author = SimpleNamespace(id=7)
    message.delete = AsyncMock()
    channel.fetch_message = AsyncMock(return_value=message)
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))
    request = DiscordDeleteOwnMessageRequest(channel_id="20", message_id="30")

    with pytest.raises(UserError, match=r"discord\.message_not_owned"):
        await endpoints["discord.delete_own_message"].invoke(
            request,
            _agent_context(),
        )
    message.delete.assert_not_awaited()

    message.author = SimpleNamespace(id=99)
    response = await endpoints["discord.delete_own_message"].invoke(
        request,
        _agent_context(),
    )
    assert response.deleted is True
    message.delete.assert_awaited_once_with()

    http_response = Mock()
    http_response.status = 404
    http_response.reason = "Not Found"
    http_response.headers = {}
    channel.fetch_message = AsyncMock(
        side_effect=discord.NotFound(
            http_response,
            {"code": 10008, "message": "Unknown Message"},
        )
    )
    repeated = await endpoints["discord.delete_own_message"].invoke(
        request,
        _agent_context(),
    )
    assert repeated.deleted is True


@pytest.mark.asyncio
async def test_unban_is_idempotent_when_member_is_already_unbanned() -> None:
    client = Mock(spec=discord.Client)
    guild = Mock(spec=discord.Guild)
    guild.id = 10
    permissions = SimpleNamespace(administrator=False, ban_members=True)
    actor = SimpleNamespace(id=7, guild_permissions=permissions)
    bot = SimpleNamespace(id=99, guild_permissions=permissions)
    guild.get_member.return_value = actor
    guild.me = bot
    http_response = Mock()
    http_response.status = 404
    http_response.reason = "Not Found"
    http_response.headers = {}
    guild.unban = AsyncMock(
        side_effect=discord.NotFound(
            http_response,
            {"code": 10026, "message": "Unknown Ban"},
        )
    )
    client.get_guild.return_value = guild
    endpoints = _endpoint_map(cast(discord.Client, client))

    response = await endpoints["discord.unban_member"].invoke(
        DiscordMemberModerationRequest(
            user_id="8",
            reason="Undo prior ban",
        ),
        _agent_context(),
    )

    assert response.action == "unban"
    assert response.changed is False
