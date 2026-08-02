from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from simajilord.integrations.discord.bot import SimajilordDiscordBot
from simajilord.integrations.discord.command_sync import (
    CommandManifestStore,
    DiscordCommandSynchronizer,
    command_manifest_hash,
)


def test_command_manifest_hash_ignores_remote_identity_and_top_level_order() -> None:
    first = {
        "id": 100,
        "application_id": 200,
        "name": "alpha",
        "description": "Alpha",
        "type": 1,
        "contexts": [2, 0, 1],
        "name_localizations": {},
        "options": [],
    }
    second = {
        "name": "beta",
        "description": "Beta",
        "type": 1,
        "options": [{"name": "value", "description": "Value", "type": 3}],
    }
    remote_first = {
        **first,
        "id": 999,
        "version": 300,
        "contexts": [0, 1, 2],
    }

    assert command_manifest_hash((first, second)) == command_manifest_hash(
        (second, remote_first)
    )
    assert command_manifest_hash((first,)) != command_manifest_hash((second,))


@pytest.mark.asyncio
async def test_command_synchronizer_skips_remote_reads_after_applied_hash(
    tmp_path: Path,
) -> None:
    store = CommandManifestStore(tmp_path / "commands.sqlite3")
    synchronizer = DiscordCommandSynchronizer(store)
    desired = ({"name": "alpha", "description": "Alpha", "type": 1},)
    fetch_remote = AsyncMock(return_value=desired)

    first = await synchronizer.assess(
        scope="application:1:guild:2",
        desired=desired,
        fetch_remote=fetch_remote,
    )
    assert not first.needs_sync
    assert first.reason == "remote_already_current"
    fetch_remote.assert_awaited_once()

    fetch_remote.reset_mock(side_effect=True)
    fetch_remote.side_effect = AssertionError("durable manifest should skip remote read")
    second = await synchronizer.assess(
        scope="application:1:guild:2",
        desired=desired,
        fetch_remote=fetch_remote,
    )
    assert not second.needs_sync
    assert second.reason == "manifest_unchanged"
    fetch_remote.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_synchronizer_writes_only_after_remote_diff(
    tmp_path: Path,
) -> None:
    store = CommandManifestStore(tmp_path / "commands.sqlite3")
    synchronizer = DiscordCommandSynchronizer(store)
    old = ({"name": "alpha", "description": "Old", "type": 1},)
    desired = ({"name": "alpha", "description": "New", "type": 1},)

    decision = await synchronizer.assess(
        scope="application:1:global",
        desired=desired,
        fetch_remote=AsyncMock(return_value=old),
    )

    assert decision.needs_sync
    assert decision.reason == "remote_changed"
    assert await store.applied_hash("application:1:global") is None
    await synchronizer.mark_applied(
        "application:1:global",
        decision.manifest_hash,
    )
    assert await store.applied_hash("application:1:global") == decision.manifest_hash


@pytest.mark.asyncio
async def test_command_manifest_store_prunes_departed_guild_scopes(
    tmp_path: Path,
) -> None:
    store = CommandManifestStore(tmp_path / "commands.sqlite3")
    manifest_hash = command_manifest_hash(())
    for scope in (
        "application:1:global",
        "application:1:guild:2",
        "application:1:guild:3",
    ):
        await store.mark_applied(scope, manifest_hash)

    removed = await store.retain_scopes(
        frozenset({"application:1:global", "application:1:guild:2"})
    )

    assert removed == 1
    assert await store.applied_hash("application:1:guild:3") is None


@pytest.mark.asyncio
async def test_bot_command_scope_does_not_sync_an_unchanged_remote_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desired = ({"name": "alpha", "description": "Alpha", "type": 1},)

    class FakeTree:
        def __init__(self) -> None:
            self.fetch_count = 0
            self.sync_count = 0

        async def fetch_commands(self, *, guild: object = None) -> list[object]:
            del guild
            self.fetch_count += 1
            return list(desired)

        async def sync(self, *, guild: object = None) -> list[object]:
            del guild
            self.sync_count += 1
            return []

    class Harness:
        _command_manifest_scope = SimajilordDiscordBot._command_manifest_scope
        _sync_prepared_command_scope = (
            SimajilordDiscordBot._sync_prepared_command_scope
        )

        def __init__(self) -> None:
            self.runtime = SimpleNamespace(
                settings=SimpleNamespace(application_id=1)
            )
            self.tree = FakeTree()
            self._command_synchronizer = DiscordCommandSynchronizer(
                CommandManifestStore(tmp_path / "commands.sqlite3")
            )

    monkeypatch.setattr(
        "simajilord.integrations.discord.bot.remote_command_payloads",
        lambda commands: tuple(commands),
    )
    harness = Harness()

    assert await harness._sync_prepared_command_scope(  # type: ignore[misc]
        guild=None,
        desired=desired,
    ) == []
    assert harness.tree.fetch_count == 1
    assert harness.tree.sync_count == 0

    assert await harness._sync_prepared_command_scope(  # type: ignore[misc]
        guild=None,
        desired=desired,
    ) == []
    assert harness.tree.fetch_count == 1
    assert harness.tree.sync_count == 0
