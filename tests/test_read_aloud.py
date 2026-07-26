from __future__ import annotations

import pytest

from simajilord.services.read_aloud import (
    ReadAloudMode,
    ReadAloudRoute,
    ReadAloudService,
)


@pytest.mark.asyncio
async def test_read_aloud_route_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    route = ReadAloudRoute("guild", "text", "voice", ReadAloudMode.QUEUE)
    await service.configure(route)
    reloaded = ReadAloudService(path)
    assert reloaded.get("guild") == route
    assert reloaded.matches("guild", "text")
    assert path.stat().st_mode & 0o077 == 0


@pytest.mark.asyncio
async def test_disable_removes_route(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.configure(
        ReadAloudRoute("guild", "text", "voice", ReadAloudMode.SKIP_DURING_MUSIC)
    )
    assert await service.disable("guild")
    assert service.get("guild") is None


@pytest.mark.asyncio
async def test_read_aloud_routes_are_independent_per_guild(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    first = ReadAloudRoute("guild-one", "text-one", "voice-one", ReadAloudMode.QUEUE)
    second = ReadAloudRoute(
        "guild-two",
        "text-two",
        "voice-two",
        ReadAloudMode.SKIP_DURING_MUSIC,
    )
    await service.configure(first)
    await service.configure(second)
    assert service.get("guild-one") == first
    assert service.get("guild-two") == second
    await service.disable("guild-one")
    assert service.get("guild-one") is None
    assert service.get("guild-two") == second
    assert ReadAloudService(path).get("guild-two") == second
