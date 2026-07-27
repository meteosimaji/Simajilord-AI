from __future__ import annotations

import json

import pytest

from simajilord.services.read_aloud import (
    ReadAloudDictionaryEntry,
    ReadAloudMode,
    ReadAloudPolicy,
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


@pytest.mark.asyncio
async def test_legacy_route_list_is_loaded_and_migrated_on_next_write(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    path.write_text(
        json.dumps(
            [
                {
                    "workspace_id": "guild",
                    "text_channel_id": "text",
                    "audio_destination_id": "voice",
                    "mode": "queue",
                }
            ]
        ),
        encoding="utf-8",
    )

    service = ReadAloudService(path)
    assert service.get("guild") == ReadAloudRoute(
        "guild",
        "text",
        "voice",
        ReadAloudMode.QUEUE,
    )

    await service.set_announcements(workspace_id="guild", join=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["routes"][0]["workspace_id"] == "guild"
    assert payload["policies"][0]["announce_join"] is True


@pytest.mark.asyncio
async def test_dictionary_is_literal_longest_first_and_persistent(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    await service.upsert_dictionary_entry(
        workspace_id="guild",
        surface="IUT",
        reading="あいゆーてぃー",
    )
    await service.upsert_dictionary_entry(
        workspace_id="guild",
        surface="IUT III",
        reading="あいゆーてぃー・すりー",
    )

    assert service.apply_dictionary("guild", "IUT III と IUT") == (
        "あいゆーてぃー・すりー と あいゆーてぃー"
    )
    assert ReadAloudService(path).policy("guild").dictionary == (
        ReadAloudDictionaryEntry("IUT", "あいゆーてぃー"),
        ReadAloudDictionaryEntry("IUT III", "あいゆーてぃー・すりー"),
    )


@pytest.mark.asyncio
async def test_dictionary_update_replaces_exact_surface_and_can_be_removed(
    tmp_path,
) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")
    await service.upsert_dictionary_entry(
        workspace_id="guild",
        surface="ABC",
        reading="えーびーしー",
    )
    updated = await service.upsert_dictionary_entry(
        workspace_id="guild",
        surface="ＡＢＣ",  # noqa: RUF001 - verifies NFKC normalization
        reading="あぶく",
    )

    assert updated.dictionary == (ReadAloudDictionaryEntry("ABC", "あぶく"),)
    policy, removed = await service.remove_dictionary_entry(
        workspace_id="guild",
        surface="ABC",
    )
    assert removed is True
    assert policy.dictionary == ()
    _, removed_again = await service.remove_dictionary_entry(
        workspace_id="guild",
        surface="ABC",
    )
    assert removed_again is False


@pytest.mark.asyncio
async def test_dictionary_rejects_empty_and_excessively_long_values(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")

    with pytest.raises(ValueError, match="dictionary_surface_required"):
        await service.upsert_dictionary_entry(
            workspace_id="guild",
            surface=" \n ",
            reading="よみ",
        )
    with pytest.raises(ValueError, match="dictionary_reading_too_long"):
        await service.upsert_dictionary_entry(
            workspace_id="guild",
            surface="word",
            reading="あ" * 201,
        )


@pytest.mark.asyncio
async def test_user_and_role_exclusions_are_applied_before_speech(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    await service.set_user_ignored(
        workspace_id="guild",
        user_id="user-two",
        ignored=True,
    )
    await service.set_role_ignored(
        workspace_id="guild",
        role_id="muted-role",
        ignored=True,
    )

    assert service.allows_message(
        workspace_id="guild",
        author_id="user-one",
        role_ids=("member",),
    )
    assert not service.allows_message(
        workspace_id="guild",
        author_id="user-two",
    )
    assert not service.allows_message(
        workspace_id="guild",
        author_id="user-three",
        role_ids=("muted-role",),
    )
    assert not service.allows_message(
        workspace_id="guild",
        author_id="bot",
        is_bot=True,
    )
    assert not service.allows_message(
        workspace_id="guild",
        author_id="webhook",
        is_webhook=True,
    )
    assert ReadAloudService(path).policy("guild").ignored_user_ids == (
        "user-two",
    )


@pytest.mark.asyncio
async def test_policy_survives_route_disable_and_can_unignore(tmp_path) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)
    await service.configure(
        ReadAloudRoute("guild", "text", "voice", ReadAloudMode.QUEUE)
    )
    await service.set_user_ignored(
        workspace_id="guild",
        user_id="user",
        ignored=True,
    )

    await service.disable("guild")

    assert service.get("guild") is None
    assert service.policy("guild").ignored_user_ids == ("user",)
    updated = await service.set_user_ignored(
        workspace_id="guild",
        user_id="user",
        ignored=False,
    )
    assert updated.ignored_user_ids == ()
    assert ReadAloudService(path).policy("guild").ignored_user_ids == ()


@pytest.mark.asyncio
async def test_announcement_and_semantic_options_are_selectively_updated(
    tmp_path,
) -> None:
    path = tmp_path / "read_aloud.json"
    service = ReadAloudService(path)

    announced = await service.set_announcements(
        workspace_id="guild",
        join=True,
        move=True,
    )
    semantic = await service.set_semantic_options(
        workspace_id="guild",
        author_names=False,
        attachments=False,
    )

    assert announced.announce_join is True
    assert announced.announce_leave is False
    assert announced.announce_move is True
    assert semantic.read_author_names is False
    assert semantic.read_replies is True
    assert semantic.read_attachments is False
    assert ReadAloudService(path).policy("guild") == semantic


def test_default_policy_is_not_stored_or_shared(tmp_path) -> None:
    service = ReadAloudService(tmp_path / "read_aloud.json")

    assert service.policy("one") == ReadAloudPolicy("one")
    assert service.policy("two") == ReadAloudPolicy("two")
    assert not service.state_file.exists()
