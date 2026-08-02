"""Crash-safe Discord application-command manifest synchronization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from discord import app_commands

CommandPayload = Mapping[str, object]
CommandPayloadFetcher = Callable[[], Awaitable[Sequence[CommandPayload]]]

_VOLATILE_COMMAND_FIELDS = frozenset(
    {
        "application_id",
        "guild_id",
        "id",
        "version",
    }
)
_EMPTY_EQUIVALENT_COMMAND_FIELDS = frozenset(
    {
        "description_localizations",
        "name_localizations",
    }
)
_ORDER_INDEPENDENT_COMMAND_FIELDS = frozenset(
    {
        "contexts",
        "integration_types",
    }
)


@dataclass(frozen=True, slots=True)
class CommandSyncDecision:
    """One scope comparison result, before any Discord write."""

    needs_sync: bool
    manifest_hash: str
    reason: Literal["manifest_unchanged", "remote_already_current", "remote_changed"]


class CommandManifestStore:
    """Persist the last successfully applied hash independently per Discord scope."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def applied_hash(self, scope: str) -> str | None:
        async with self._lock:
            return await asyncio.to_thread(self._applied_hash, _normalized_scope(scope))

    async def mark_applied(self, scope: str, manifest_hash: str) -> None:
        normalized_hash = manifest_hash.strip()
        if len(normalized_hash) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_hash
        ):
            raise ValueError("command manifest hash must be lowercase SHA-256")
        async with self._lock:
            await asyncio.to_thread(
                self._mark_applied,
                _normalized_scope(scope),
                normalized_hash,
            )

    async def retain_scopes(self, scopes: frozenset[str]) -> int:
        normalized = frozenset(_normalized_scope(scope) for scope in scopes)
        async with self._lock:
            return await asyncio.to_thread(self._retain_scopes, normalized)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS discord_command_manifests (
                    scope TEXT PRIMARY KEY,
                    manifest_sha256 TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
        os.chmod(self.path, 0o600)

    def _applied_hash(self, scope: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT manifest_sha256
                FROM discord_command_manifests
                WHERE scope = ?
                """,
                (scope,),
            ).fetchone()
        return str(row[0]) if row is not None else None

    def _mark_applied(self, scope: str, manifest_hash: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO discord_command_manifests(
                    scope, manifest_sha256, applied_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    manifest_sha256 = excluded.manifest_sha256,
                    applied_at = excluded.applied_at
                """,
                (scope, manifest_hash, datetime.now(UTC).isoformat()),
            )

    def _retain_scopes(self, scopes: frozenset[str]) -> int:
        with self._connect() as connection:
            if not scopes:
                cursor = connection.execute("DELETE FROM discord_command_manifests")
                return cursor.rowcount
            placeholders = ", ".join("?" for _ in scopes)
            cursor = connection.execute(
                f"DELETE FROM discord_command_manifests WHERE scope NOT IN ({placeholders})",
                tuple(sorted(scopes)),
            )
            return cursor.rowcount

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


class DiscordCommandSynchronizer:
    """Compare local, durable, and remote command manifests before writing."""

    def __init__(self, store: CommandManifestStore) -> None:
        self.store = store

    async def assess(
        self,
        *,
        scope: str,
        desired: Sequence[CommandPayload],
        fetch_remote: CommandPayloadFetcher,
    ) -> CommandSyncDecision:
        manifest_hash = command_manifest_hash(desired)
        if await self.store.applied_hash(scope) == manifest_hash:
            return CommandSyncDecision(False, manifest_hash, "manifest_unchanged")
        remote = await fetch_remote()
        if command_manifest_hash(remote) == manifest_hash:
            await self.store.mark_applied(scope, manifest_hash)
            return CommandSyncDecision(False, manifest_hash, "remote_already_current")
        return CommandSyncDecision(True, manifest_hash, "remote_changed")

    async def mark_applied(self, scope: str, manifest_hash: str) -> None:
        await self.store.mark_applied(scope, manifest_hash)


def local_command_payloads(
    commands: Sequence[
        app_commands.Command[Any, ..., Any]
        | app_commands.Group
        | app_commands.ContextMenu
    ],
    tree: app_commands.CommandTree[Any],
) -> tuple[dict[str, object], ...]:
    """Return the writable local API payloads without mutating the tree."""

    return tuple(command.to_dict(tree) for command in commands)


def remote_command_payloads(
    commands: Sequence[app_commands.AppCommand],
) -> tuple[dict[str, object], ...]:
    """Project fetched commands onto fields represented by local definitions."""

    payloads: list[dict[str, object]] = []
    for command in commands:
        payload: dict[str, object] = dict(command.to_dict())
        permissions = command.default_member_permissions
        payload["default_member_permissions"] = (
            permissions.value if permissions is not None else None
        )
        payload["dm_permission"] = command.dm_permission
        payload["nsfw"] = command.nsfw
        payloads.append(payload)
    return tuple(payloads)


def command_manifest_hash(commands: Sequence[CommandPayload]) -> str:
    """Hash a stable, order-independent top-level command manifest."""

    canonical_commands = sorted(
        (_canonical_command_value(dict(command)) for command in commands),
        key=lambda value: json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    encoded = json.dumps(
        canonical_commands,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_command_value(value: object, *, field: str | None = None) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            if key in _VOLATILE_COMMAND_FIELDS or raw_value is None:
                continue
            if key in _EMPTY_EQUIVALENT_COMMAND_FIELDS and not raw_value:
                continue
            normalized[key] = _canonical_command_value(raw_value, field=key)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized_items = [
            _canonical_command_value(item)
            for item in value
        ]
        if field in _ORDER_INDEPENDENT_COMMAND_FIELDS:
            return sorted(
                normalized_items,
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        return normalized_items
    return value


def _normalized_scope(scope: str) -> str:
    normalized = scope.strip()
    if not normalized or len(normalized) > 200:
        raise ValueError("command manifest scope must be bounded and non-empty")
    return normalized
