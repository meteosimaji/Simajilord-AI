"""Durable temporary-voice configuration and room ownership state."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from simajilord.core.errors import UserError

DEFAULT_TEMP_VOICE_GRACE_SECONDS = 10 * 60
MIN_TEMP_VOICE_GRACE_SECONDS = 60
MAX_TEMP_VOICE_GRACE_SECONDS = 60 * 60
MAX_TEMP_VOICE_CREATORS_PER_WORKSPACE = 20
MAX_TEMP_VOICE_ROOM_NAME_LENGTH = 100
DEFAULT_TEMP_VOICE_ROOM_TEMPLATE = "{display_name}'s room"


class TempVoiceRoomEndReason(StrEnum):
    """Why one BOT-tracked room stopped being temporary."""

    EMPTY = "empty"
    MISSING = "missing"
    KEPT_PERMANENT = "kept_permanent"
    MOVE_FAILED = "move_failed"


@dataclass(frozen=True, slots=True)
class TempVoiceConfig:
    workspace_id: str
    enabled: bool
    room_name_template: str
    empty_grace_seconds: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TempVoiceCreator:
    workspace_id: str
    channel_id: str
    category_id: str
    permission_source_channel_id: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TempVoiceRoom:
    workspace_id: str
    channel_id: str
    creator_channel_id: str
    owner_id: str
    name: str
    user_limit: int
    locked: bool
    created_at: datetime
    empty_since: datetime | None = None


@dataclass(frozen=True, slots=True)
class TempVoiceProfile:
    workspace_id: str
    owner_id: str
    room_name: str
    user_limit: int
    locked: bool
    last_room_channel_id: str | None
    updated_at: datetime


class TempVoiceService:
    """SQLite authority for TempVC configuration, profiles, and active rooms."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def ensure_config(self, workspace_id: str) -> TempVoiceConfig:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._ensure_config,
                normalized_workspace_id,
            )

    async def config(self, workspace_id: str) -> TempVoiceConfig | None:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        async with self._lock:
            return await asyncio.to_thread(self._config, normalized_workspace_id)

    async def set_enabled(
        self,
        workspace_id: str,
        enabled: bool,
    ) -> TempVoiceConfig:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._set_enabled,
                normalized_workspace_id,
                enabled,
            )

    async def update_settings(
        self,
        workspace_id: str,
        *,
        room_name_template: str,
        empty_grace_seconds: int,
    ) -> TempVoiceConfig:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        normalized_template = validate_temp_voice_room_template(room_name_template)
        normalized_grace = validate_temp_voice_grace_seconds(empty_grace_seconds)
        async with self._lock:
            return await asyncio.to_thread(
                self._update_settings,
                normalized_workspace_id,
                normalized_template,
                normalized_grace,
            )

    async def add_creator(
        self,
        *,
        workspace_id: str,
        channel_id: str,
        category_id: str,
        permission_source_channel_id: str | None = None,
    ) -> TempVoiceCreator:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_category_id = _bounded_identifier(category_id, "category_id")
        normalized_permission_source_id = (
            _bounded_identifier(
                permission_source_channel_id,
                "permission_source_channel_id",
            )
            if permission_source_channel_id is not None
            else None
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._add_creator,
                normalized_workspace_id,
                normalized_channel_id,
                normalized_category_id,
                normalized_permission_source_id,
            )

    async def creator(self, channel_id: str) -> TempVoiceCreator | None:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        async with self._lock:
            return await asyncio.to_thread(self._creator, normalized_channel_id)

    async def creators(self, workspace_id: str) -> tuple[TempVoiceCreator, ...]:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        async with self._lock:
            return await asyncio.to_thread(self._creators, normalized_workspace_id)

    async def set_creator_permission_source(
        self,
        channel_id: str,
        permission_source_channel_id: str | None,
    ) -> TempVoiceCreator:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_permission_source_id = (
            _bounded_identifier(
                permission_source_channel_id,
                "permission_source_channel_id",
            )
            if permission_source_channel_id is not None
            else None
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._set_creator_permission_source,
                normalized_channel_id,
                normalized_permission_source_id,
            )

    async def remove_creator(self, channel_id: str) -> TempVoiceCreator | None:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._remove_creator,
                normalized_channel_id,
            )

    async def profile(
        self,
        *,
        workspace_id: str,
        owner_id: str,
    ) -> TempVoiceProfile | None:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        normalized_owner_id = _bounded_identifier(owner_id, "owner_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._profile,
                normalized_workspace_id,
                normalized_owner_id,
            )

    async def register_room(
        self,
        *,
        workspace_id: str,
        channel_id: str,
        creator_channel_id: str,
        owner_id: str,
        name: str,
        user_limit: int,
        locked: bool,
    ) -> TempVoiceRoom:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_creator_id = _bounded_identifier(
            creator_channel_id,
            "creator_channel_id",
        )
        normalized_owner_id = _bounded_identifier(owner_id, "owner_id")
        normalized_name = normalize_temp_voice_room_name(name)
        normalized_limit = validate_temp_voice_user_limit(user_limit)
        async with self._lock:
            return await asyncio.to_thread(
                self._register_room,
                normalized_workspace_id,
                normalized_channel_id,
                normalized_creator_id,
                normalized_owner_id,
                normalized_name,
                normalized_limit,
                locked,
            )

    async def room(self, channel_id: str) -> TempVoiceRoom | None:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        async with self._lock:
            return await asyncio.to_thread(self._room, normalized_channel_id)

    async def rooms(
        self,
        workspace_id: str | None = None,
    ) -> tuple[TempVoiceRoom, ...]:
        normalized_workspace_id = (
            _bounded_identifier(workspace_id, "workspace_id") if workspace_id is not None else None
        )
        async with self._lock:
            return await asyncio.to_thread(self._rooms, normalized_workspace_id)

    async def rooms_owned_by(
        self,
        *,
        workspace_id: str,
        owner_id: str,
    ) -> tuple[TempVoiceRoom, ...]:
        normalized_workspace_id = _bounded_identifier(workspace_id, "workspace_id")
        normalized_owner_id = _bounded_identifier(owner_id, "owner_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._rooms_owned_by,
                normalized_workspace_id,
                normalized_owner_id,
            )

    async def mark_room_empty(
        self,
        channel_id: str,
        *,
        empty_since: datetime | None = None,
    ) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        marked_at = _normalized_datetime(empty_since or datetime.now(UTC))
        async with self._lock:
            return await asyncio.to_thread(
                self._mark_room_empty,
                normalized_channel_id,
                marked_at,
            )

    async def mark_room_occupied(self, channel_id: str) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._mark_room_occupied,
                normalized_channel_id,
            )

    async def rename_room(self, channel_id: str, name: str) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_name = normalize_temp_voice_room_name(name)
        async with self._lock:
            return await asyncio.to_thread(
                self._update_room,
                normalized_channel_id,
                normalized_name,
                None,
                None,
            )

    async def set_room_user_limit(
        self,
        channel_id: str,
        user_limit: int,
    ) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_limit = validate_temp_voice_user_limit(user_limit)
        async with self._lock:
            return await asyncio.to_thread(
                self._update_room,
                normalized_channel_id,
                None,
                normalized_limit,
                None,
            )

    async def set_room_locked(self, channel_id: str, locked: bool) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._update_room,
                normalized_channel_id,
                None,
                None,
                locked,
            )

    async def transfer_room(self, channel_id: str, owner_id: str) -> TempVoiceRoom:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_owner_id = _bounded_identifier(owner_id, "owner_id")
        async with self._lock:
            return await asyncio.to_thread(
                self._transfer_room,
                normalized_channel_id,
                normalized_owner_id,
            )

    async def finish_room(
        self,
        channel_id: str,
        *,
        reason: TempVoiceRoomEndReason,
        remember_for_recreation: bool = True,
        replacement_last_channel_id: str | None = None,
    ) -> TempVoiceRoom | None:
        normalized_channel_id = _bounded_identifier(channel_id, "channel_id")
        normalized_replacement = (
            _bounded_identifier(replacement_last_channel_id, "replacement_last_channel_id")
            if replacement_last_channel_id is not None
            else None
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._finish_room,
                normalized_channel_id,
                reason,
                remember_for_recreation,
                normalized_replacement,
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS temp_voice_configs (
                    workspace_id TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    room_name_template TEXT NOT NULL,
                    empty_grace_seconds INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS temp_voice_creators (
                    channel_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    category_id TEXT NOT NULL,
                    permission_source_channel_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(workspace_id)
                        REFERENCES temp_voice_configs(workspace_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_temp_voice_creators_workspace
                    ON temp_voice_creators(workspace_id, created_at, channel_id);

                CREATE TABLE IF NOT EXISTS temp_voice_rooms (
                    channel_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    creator_channel_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    user_limit INTEGER NOT NULL,
                    locked INTEGER NOT NULL CHECK (locked IN (0, 1)),
                    created_at TEXT NOT NULL,
                    empty_since TEXT,
                    FOREIGN KEY(workspace_id)
                        REFERENCES temp_voice_configs(workspace_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_temp_voice_rooms_workspace
                    ON temp_voice_rooms(workspace_id, created_at, channel_id);
                CREATE INDEX IF NOT EXISTS idx_temp_voice_rooms_owner
                    ON temp_voice_rooms(workspace_id, owner_id, created_at, channel_id);

                CREATE TABLE IF NOT EXISTS temp_voice_profiles (
                    workspace_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    room_name TEXT NOT NULL,
                    user_limit INTEGER NOT NULL,
                    locked INTEGER NOT NULL CHECK (locked IN (0, 1)),
                    last_room_channel_id TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, owner_id),
                    FOREIGN KEY(workspace_id)
                        REFERENCES temp_voice_configs(workspace_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS temp_voice_room_history (
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
                );
                CREATE INDEX IF NOT EXISTS idx_temp_voice_history_workspace
                    ON temp_voice_room_history(workspace_id, ended_at, history_id);
                """
            )
            creator_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(temp_voice_creators)").fetchall()
            }
            if "permission_source_channel_id" not in creator_columns:
                connection.execute(
                    """
                    ALTER TABLE temp_voice_creators
                    ADD COLUMN permission_source_channel_id TEXT
                    """
                )
        os.chmod(self.path, 0o600)

    def _ensure_config(self, workspace_id: str) -> TempVoiceConfig:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO temp_voice_configs(
                    workspace_id,
                    enabled,
                    room_name_template,
                    empty_grace_seconds,
                    created_at,
                    updated_at
                ) VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(workspace_id) DO NOTHING
                """,
                (
                    workspace_id,
                    DEFAULT_TEMP_VOICE_ROOM_TEMPLATE,
                    DEFAULT_TEMP_VOICE_GRACE_SECONDS,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM temp_voice_configs WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        assert row is not None
        return _config_from_row(row)

    def _config(self, workspace_id: str) -> TempVoiceConfig | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temp_voice_configs WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        return _config_from_row(row) if row is not None else None

    def _set_enabled(self, workspace_id: str, enabled: bool) -> TempVoiceConfig:
        self._ensure_config(workspace_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE temp_voice_configs
                SET enabled = ?, updated_at = ?
                WHERE workspace_id = ?
                """,
                (int(enabled), datetime.now(UTC).isoformat(), workspace_id),
            )
        config = self._config(workspace_id)
        assert config is not None
        return config

    def _update_settings(
        self,
        workspace_id: str,
        room_name_template: str,
        empty_grace_seconds: int,
    ) -> TempVoiceConfig:
        self._ensure_config(workspace_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE temp_voice_configs
                SET room_name_template = ?,
                    empty_grace_seconds = ?,
                    updated_at = ?
                WHERE workspace_id = ?
                """,
                (
                    room_name_template,
                    empty_grace_seconds,
                    datetime.now(UTC).isoformat(),
                    workspace_id,
                ),
            )
        config = self._config(workspace_id)
        assert config is not None
        return config

    def _add_creator(
        self,
        workspace_id: str,
        channel_id: str,
        category_id: str,
        permission_source_channel_id: str | None,
    ) -> TempVoiceCreator:
        self._ensure_config(workspace_id)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if existing is not None:
                creator = _creator_from_row(existing)
                if creator.workspace_id != workspace_id:
                    raise UserError("temp_voice.creator_conflict")
                if (
                    creator.category_id != category_id
                    or creator.permission_source_channel_id != permission_source_channel_id
                ):
                    connection.execute(
                        """
                        UPDATE temp_voice_creators
                        SET category_id = ?, permission_source_channel_id = ?
                        WHERE channel_id = ?
                        """,
                        (
                            category_id,
                            permission_source_channel_id,
                            channel_id,
                        ),
                    )
                    updated = connection.execute(
                        "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                        (channel_id,),
                    ).fetchone()
                    assert updated is not None
                    return _creator_from_row(updated)
                return creator
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM temp_voice_creators
                    WHERE workspace_id = ?
                    """,
                    (workspace_id,),
                ).fetchone()[0]
            )
            if count >= MAX_TEMP_VOICE_CREATORS_PER_WORKSPACE:
                raise UserError("temp_voice.creator_limit")
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                    INSERT INTO temp_voice_creators(
                        channel_id,
                        workspace_id,
                        category_id,
                        permission_source_channel_id,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                (
                    channel_id,
                    workspace_id,
                    category_id,
                    permission_source_channel_id,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        assert row is not None
        return _creator_from_row(row)

    def _creator(self, channel_id: str) -> TempVoiceCreator | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return _creator_from_row(row) if row is not None else None

    def _creators(self, workspace_id: str) -> tuple[TempVoiceCreator, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM temp_voice_creators
                WHERE workspace_id = ?
                ORDER BY created_at, channel_id
                """,
                (workspace_id,),
            ).fetchall()
        return tuple(_creator_from_row(row) for row in rows)

    def _set_creator_permission_source(
        self,
        channel_id: str,
        permission_source_channel_id: str | None,
    ) -> TempVoiceCreator:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE temp_voice_creators
                SET permission_source_channel_id = ?
                WHERE channel_id = ?
                """,
                (permission_source_channel_id, channel_id),
            )
            row = connection.execute(
                "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        if row is None:
            raise UserError("temp_voice.creator_missing")
        return _creator_from_row(row)

    def _remove_creator(self, channel_id: str) -> TempVoiceCreator | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if row is None:
                return None
            creator = _creator_from_row(row)
            connection.execute(
                "DELETE FROM temp_voice_creators WHERE channel_id = ?",
                (channel_id,),
            )
        return creator

    def _profile(self, workspace_id: str, owner_id: str) -> TempVoiceProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM temp_voice_profiles
                WHERE workspace_id = ? AND owner_id = ?
                """,
                (workspace_id, owner_id),
            ).fetchone()
        return _profile_from_row(row) if row is not None else None

    def _register_room(
        self,
        workspace_id: str,
        channel_id: str,
        creator_channel_id: str,
        owner_id: str,
        name: str,
        user_limit: int,
        locked: bool,
    ) -> TempVoiceRoom:
        self._ensure_config(workspace_id)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            creator = connection.execute(
                """
                SELECT workspace_id
                FROM temp_voice_creators
                WHERE channel_id = ?
                """,
                (creator_channel_id,),
            ).fetchone()
            if creator is None or str(creator[0]) != workspace_id:
                raise UserError("temp_voice.creator_missing")
            try:
                connection.execute(
                    """
                    INSERT INTO temp_voice_rooms(
                        channel_id,
                        workspace_id,
                        creator_channel_id,
                        owner_id,
                        name,
                        user_limit,
                        locked,
                        created_at,
                        empty_since
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        channel_id,
                        workspace_id,
                        creator_channel_id,
                        owner_id,
                        name,
                        user_limit,
                        int(locked),
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise UserError("temp_voice.room_conflict") from exc
            _upsert_profile(
                connection,
                workspace_id=workspace_id,
                owner_id=owner_id,
                room_name=name,
                user_limit=user_limit,
                locked=locked,
                last_room_channel_id=channel_id,
                updated_at=now,
            )
            row = connection.execute(
                "SELECT * FROM temp_voice_rooms WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        assert row is not None
        return _room_from_row(row)

    def _room(self, channel_id: str) -> TempVoiceRoom | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temp_voice_rooms WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return _room_from_row(row) if row is not None else None

    def _rooms(self, workspace_id: str | None) -> tuple[TempVoiceRoom, ...]:
        with self._connect() as connection:
            if workspace_id is None:
                rows = connection.execute(
                    "SELECT * FROM temp_voice_rooms ORDER BY workspace_id, created_at, channel_id"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM temp_voice_rooms
                    WHERE workspace_id = ?
                    ORDER BY created_at, channel_id
                    """,
                    (workspace_id,),
                ).fetchall()
        return tuple(_room_from_row(row) for row in rows)

    def _rooms_owned_by(
        self,
        workspace_id: str,
        owner_id: str,
    ) -> tuple[TempVoiceRoom, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM temp_voice_rooms
                WHERE workspace_id = ? AND owner_id = ?
                ORDER BY created_at, channel_id
                """,
                (workspace_id, owner_id),
            ).fetchall()
        return tuple(_room_from_row(row) for row in rows)

    def _mark_room_empty(self, channel_id: str, empty_since: datetime) -> TempVoiceRoom:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE temp_voice_rooms
                SET empty_since = COALESCE(empty_since, ?)
                WHERE channel_id = ?
                """,
                (empty_since.isoformat(), channel_id),
            )
            row = _required_room_row(connection, channel_id)
        return _room_from_row(row)

    def _mark_room_occupied(self, channel_id: str) -> TempVoiceRoom:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE temp_voice_rooms
                SET empty_since = NULL
                WHERE channel_id = ?
                """,
                (channel_id,),
            )
            row = _required_room_row(connection, channel_id)
        return _room_from_row(row)

    def _update_room(
        self,
        channel_id: str,
        name: str | None,
        user_limit: int | None,
        locked: bool | None,
    ) -> TempVoiceRoom:
        with self._connect() as connection:
            current = _room_from_row(_required_room_row(connection, channel_id))
            updated_name = current.name if name is None else name
            updated_limit = current.user_limit if user_limit is None else user_limit
            updated_locked = current.locked if locked is None else locked
            connection.execute(
                """
                UPDATE temp_voice_rooms
                SET name = ?, user_limit = ?, locked = ?
                WHERE channel_id = ?
                """,
                (updated_name, updated_limit, int(updated_locked), channel_id),
            )
            now = datetime.now(UTC).isoformat()
            _upsert_profile(
                connection,
                workspace_id=current.workspace_id,
                owner_id=current.owner_id,
                room_name=updated_name,
                user_limit=updated_limit,
                locked=updated_locked,
                last_room_channel_id=current.channel_id,
                updated_at=now,
            )
            row = _required_room_row(connection, channel_id)
        return _room_from_row(row)

    def _transfer_room(self, channel_id: str, owner_id: str) -> TempVoiceRoom:
        with self._connect() as connection:
            current = _room_from_row(_required_room_row(connection, channel_id))
            if current.owner_id == owner_id:
                return current
            connection.execute(
                """
                UPDATE temp_voice_rooms
                SET owner_id = ?
                WHERE channel_id = ?
                """,
                (owner_id, channel_id),
            )
            connection.execute(
                """
                UPDATE temp_voice_profiles
                SET last_room_channel_id = NULL, updated_at = ?
                WHERE workspace_id = ?
                  AND owner_id = ?
                  AND last_room_channel_id = ?
                """,
                (
                    datetime.now(UTC).isoformat(),
                    current.workspace_id,
                    current.owner_id,
                    current.channel_id,
                ),
            )
            now = datetime.now(UTC).isoformat()
            _upsert_profile(
                connection,
                workspace_id=current.workspace_id,
                owner_id=owner_id,
                room_name=current.name,
                user_limit=current.user_limit,
                locked=current.locked,
                last_room_channel_id=current.channel_id,
                updated_at=now,
            )
            row = _required_room_row(connection, channel_id)
        return _room_from_row(row)

    def _finish_room(
        self,
        channel_id: str,
        reason: TempVoiceRoomEndReason,
        remember_for_recreation: bool,
        replacement_last_channel_id: str | None,
    ) -> TempVoiceRoom | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM temp_voice_rooms WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if row is None:
                return None
            room = _room_from_row(row)
            ended_at = datetime.now(UTC).isoformat()
            connection.execute(
                """
                INSERT INTO temp_voice_room_history(
                    workspace_id,
                    channel_id,
                    creator_channel_id,
                    owner_id,
                    name,
                    user_limit,
                    locked,
                    created_at,
                    empty_since,
                    ended_at,
                    end_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    room.workspace_id,
                    room.channel_id,
                    room.creator_channel_id,
                    room.owner_id,
                    room.name,
                    room.user_limit,
                    int(room.locked),
                    room.created_at.isoformat(),
                    room.empty_since.isoformat() if room.empty_since is not None else None,
                    ended_at,
                    reason.value,
                ),
            )
            connection.execute(
                "DELETE FROM temp_voice_rooms WHERE channel_id = ?",
                (channel_id,),
            )
            remembered_channel_id = (
                room.channel_id if remember_for_recreation else replacement_last_channel_id
            )
            _upsert_profile(
                connection,
                workspace_id=room.workspace_id,
                owner_id=room.owner_id,
                room_name=room.name,
                user_limit=room.user_limit,
                locked=room.locked,
                last_room_channel_id=remembered_channel_id,
                updated_at=ended_at,
            )
        return room

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


def validate_temp_voice_room_template(template: str) -> str:
    normalized = " ".join(template.split())
    if (
        not normalized
        or len(normalized) > MAX_TEMP_VOICE_ROOM_NAME_LENGTH
        or _contains_unsupported_channel_codepoint(normalized)
    ):
        raise UserError("temp_voice.template_invalid")
    remainder = normalized.replace("{display_name}", "").replace("{username}", "")
    if "{" in remainder or "}" in remainder:
        raise UserError("temp_voice.template_invalid")
    return normalized


def render_temp_voice_room_name(
    template: str,
    *,
    display_name: str,
    username: str,
) -> str:
    normalized_template = validate_temp_voice_room_template(template)
    rendered = normalized_template.replace(
        "{display_name}",
        " ".join(display_name.split()),
    ).replace(
        "{username}",
        " ".join(username.split()),
    )
    rendered = " ".join(rendered.split())
    if len(rendered) > MAX_TEMP_VOICE_ROOM_NAME_LENGTH:
        rendered = rendered[:MAX_TEMP_VOICE_ROOM_NAME_LENGTH].rstrip()
    return normalize_temp_voice_room_name(rendered)


def normalize_temp_voice_room_name(name: str) -> str:
    normalized = " ".join(name.split())
    if (
        not normalized
        or len(normalized) > MAX_TEMP_VOICE_ROOM_NAME_LENGTH
        or _contains_unsupported_channel_codepoint(normalized)
    ):
        raise UserError("temp_voice.room_name_invalid")
    return normalized


def validate_temp_voice_user_limit(user_limit: int) -> int:
    if isinstance(user_limit, bool) or not 0 <= user_limit <= 99:
        raise UserError("temp_voice.user_limit_invalid")
    return user_limit


def validate_temp_voice_grace_seconds(empty_grace_seconds: int) -> int:
    if isinstance(empty_grace_seconds, bool) or not (
        MIN_TEMP_VOICE_GRACE_SECONDS <= empty_grace_seconds <= MAX_TEMP_VOICE_GRACE_SECONDS
    ):
        raise UserError("temp_voice.grace_invalid")
    return empty_grace_seconds


def _bounded_identifier(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 100:
        raise ValueError(f"{field} must be bounded and non-empty")
    return normalized


def _contains_unsupported_channel_codepoint(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cs"} or character == "\ufffd"
        for character in value
    )


def _normalized_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("TempVC timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _row_datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _config_from_row(row: sqlite3.Row) -> TempVoiceConfig:
    return TempVoiceConfig(
        workspace_id=str(row["workspace_id"]),
        enabled=bool(row["enabled"]),
        room_name_template=str(row["room_name_template"]),
        empty_grace_seconds=int(row["empty_grace_seconds"]),
        created_at=_row_datetime(row["created_at"]),
        updated_at=_row_datetime(row["updated_at"]),
    )


def _creator_from_row(row: sqlite3.Row) -> TempVoiceCreator:
    permission_source_channel_id = row["permission_source_channel_id"]
    return TempVoiceCreator(
        workspace_id=str(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        category_id=str(row["category_id"]),
        permission_source_channel_id=(
            str(permission_source_channel_id) if permission_source_channel_id is not None else None
        ),
        created_at=_row_datetime(row["created_at"]),
    )


def _room_from_row(row: sqlite3.Row) -> TempVoiceRoom:
    raw_empty_since = row["empty_since"]
    return TempVoiceRoom(
        workspace_id=str(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        creator_channel_id=str(row["creator_channel_id"]),
        owner_id=str(row["owner_id"]),
        name=str(row["name"]),
        user_limit=int(row["user_limit"]),
        locked=bool(row["locked"]),
        created_at=_row_datetime(row["created_at"]),
        empty_since=(_row_datetime(raw_empty_since) if raw_empty_since is not None else None),
    )


def _profile_from_row(row: sqlite3.Row) -> TempVoiceProfile:
    raw_last_room_channel_id = row["last_room_channel_id"]
    return TempVoiceProfile(
        workspace_id=str(row["workspace_id"]),
        owner_id=str(row["owner_id"]),
        room_name=str(row["room_name"]),
        user_limit=int(row["user_limit"]),
        locked=bool(row["locked"]),
        last_room_channel_id=(
            str(raw_last_room_channel_id) if raw_last_room_channel_id is not None else None
        ),
        updated_at=_row_datetime(row["updated_at"]),
    )


def _required_room_row(connection: sqlite3.Connection, channel_id: str) -> sqlite3.Row:
    row = cast(
        sqlite3.Row | None,
        connection.execute(
            "SELECT * FROM temp_voice_rooms WHERE channel_id = ?",
            (channel_id,),
        ).fetchone(),
    )
    if row is None:
        raise UserError("temp_voice.room_missing")
    return row


def _upsert_profile(
    connection: sqlite3.Connection,
    *,
    workspace_id: str,
    owner_id: str,
    room_name: str,
    user_limit: int,
    locked: bool,
    last_room_channel_id: str | None,
    updated_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO temp_voice_profiles(
            workspace_id,
            owner_id,
            room_name,
            user_limit,
            locked,
            last_room_channel_id,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(workspace_id, owner_id) DO UPDATE SET
            room_name = excluded.room_name,
            user_limit = excluded.user_limit,
            locked = excluded.locked,
            last_room_channel_id = excluded.last_room_channel_id,
            updated_at = excluded.updated_at
        """,
        (
            workspace_id,
            owner_id,
            room_name,
            user_limit,
            int(locked),
            last_room_channel_id,
            updated_at,
        ),
    )
