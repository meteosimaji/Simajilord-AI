from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.integrations.discord.channel_lifecycle import reconcile_read_aloud_channels
from simajilord.integrations.discord.cogs import ReadAloudCog
from simajilord.services.read_aloud import ReadAloudMode, ReadAloudRoute, ReadAloudService


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(403, 50001), (500, 0), (404, 10008)])
async def test_unverified_missing_channel_keeps_configuration(tmp_path, status, code):
    service = ReadAloudService(tmp_path / "routes.json")
    route = ReadAloudRoute("1", "11", "20", ReadAloudMode.QUEUE)
    await service.configure(route)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel_or_thread.return_value = None
    error_type = discord.NotFound if status == 404 else discord.HTTPException
    guild.fetch_channel = AsyncMock(side_effect=error_type(
        SimpleNamespace(status=status, reason="failure"), {"code": code, "message": "failure"}
    ))
    await reconcile_read_aloud_channels(guild, service)
    assert ReadAloudService(service.state_file).get("1") == route


@pytest.mark.asyncio
async def test_uncached_existing_thread_is_returned_without_removing_route(tmp_path):
    service = ReadAloudService(tmp_path / "routes.json")
    route = ReadAloudRoute("1", "11", "20", ReadAloudMode.QUEUE)
    await service.configure(route)
    guild = Mock(spec=discord.Guild)
    guild.id = 1
    guild.get_channel_or_thread.side_effect = lambda value: object() if value == 20 else None
    thread = Mock(spec=discord.Thread)
    guild.fetch_channel = AsyncMock(return_value=thread)
    assert await reconcile_read_aloud_channels(guild, service) == {11: thread}
    assert service.get("1") == route


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_thread", [False, True])
async def test_gateway_deletion_repairs_saved_routes(tmp_path, raw_thread):
    service = ReadAloudService(tmp_path / "routes.json")
    await service.configure(ReadAloudRoute(
        "1", "11", "20", ReadAloudMode.QUEUE, additional_text_channel_ids=("12",)
    ))
    runtime = SimpleNamespace(read_aloud=service)
    cog = ReadAloudCog(SimpleNamespace(), runtime)
    if raw_thread:
        await cog.on_raw_thread_delete(SimpleNamespace(guild_id=1, thread_id=11))
    else:
        await cog.on_guild_channel_delete(SimpleNamespace(guild=SimpleNamespace(id=1), id=11))
    assert ReadAloudService(service.state_file).get("1").text_channel_ids == ("12",)
