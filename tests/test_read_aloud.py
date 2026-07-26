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


@pytest.mark.asyncio
async def test_multiple_conversation_channels_share_one_voice_destination(
    tmp_path,
) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)

    await service.add_source(
        workspace_id="guild",
        text_channel_id="conversation-one",
        audio_destination_id="voice",
        mode=ReadAloudMode.QUEUE,
    )
    route = await service.add_source(
        workspace_id="guild",
        text_channel_id="conversation-two",
        audio_destination_id="voice",
        mode=ReadAloudMode.QUEUE,
    )

    assert route.text_channel_ids == ("conversation-one", "conversation-two")
    assert service.matches("guild", "conversation-one")
    assert service.matches("guild", "conversation-two")
    assert ReadAloudService(path).get("guild") == route


@pytest.mark.asyncio
async def test_adding_multiple_sources_is_deduplicated_and_persistent(
    tmp_path,
) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    await service.add_source(
        workspace_id="guild",
        text_channel_id="existing",
        audio_destination_id="voice",
        mode=ReadAloudMode.QUEUE,
    )

    route = await service.add_sources(
        workspace_id="guild",
        text_channel_ids=("one", "two", "one", "three"),
        audio_destination_id="voice",
        mode=ReadAloudMode.QUEUE,
    )

    assert route.text_channel_ids == ("existing", "one", "two", "three")
    assert ReadAloudService(path).get("guild") == route


@pytest.mark.asyncio
async def test_removing_last_conversation_channel_disables_read_aloud(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.add_source(
        workspace_id="guild",
        text_channel_id="conversation",
        audio_destination_id="voice",
        mode=ReadAloudMode.QUEUE,
    )

    assert (
        await service.remove_source(
            workspace_id="guild",
            text_channel_id="conversation",
        )
        is None
    )
    assert service.get("guild") is None


@pytest.mark.asyncio
async def test_conversation_channels_cannot_silently_move_to_another_voice_channel(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.add_source(
        workspace_id="guild",
        text_channel_id="conversation-one",
        audio_destination_id="voice-one",
        mode=ReadAloudMode.QUEUE,
    )

    with pytest.raises(ValueError, match="destination_conflict"):
        await service.add_source(
            workspace_id="guild",
            text_channel_id="conversation-two",
            audio_destination_id="voice-two",
            mode=ReadAloudMode.QUEUE,
        )
