"""Durable, scoped agent memory independent of provider conversation threads."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError

MAX_MEMORY_KEY_CHARACTERS = 80
MAX_MEMORY_SUMMARY_CHARACTERS = 320
MAX_MEMORY_QUERY_CHARACTERS = 200
MAX_MEMORY_SOURCE_MESSAGE_IDS = 8
MAX_MEMORY_SEARCH_RESULTS = 10
MAX_MEMORY_SEARCH_OFFSET = 500
MAX_MEMORY_TTL_DAYS = 365
MIN_MEMORY_CONFIDENCE = 0.8

_LATIN_SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CJK_SEARCH_RUN_PATTERN = re.compile(
    r"[\u3041-\u3096\u30a1-\u30fa\u30fc"
    r"\u3400-\u4dbf\u4e00-\u9fff\u3005\u3006\u30f5\u30f6]+"
)
_SEARCH_TERM_GROUPS = (
    frozenset(
        {
            "answer",
            "respond",
            "response",
            "reply",
            "回答",
            "応答",
            "返答",
        }
    ),
    frozenset(
        {
            "brief",
            "concise",
            "short",
            "succinct",
            "簡潔",
            "短い",
            "短く",
        }
    ),
    frozenset({"english", "英語"}),
    frozenset({"japanese", "日本語"}),
    frozenset(
        {
            "checklist",
            "procedure",
            "steps",
            "workflow",
            "やり方",
            "作業手順",
            "手順",
        }
    ),
    frozenset(
        {
            "preference",
            "prefer",
            "好み",
            "好む",
            "希望",
        }
    ),
    frozenset(
        {
            "memory",
            "remember",
            "メモリ",
            "記憶",
            "覚える",
            "思い出す",
        }
    ),
    frozenset(
        {
            "identity",
            "me",
            "my",
            "profile",
            "requester",
            "user",
            "プロフィール",
            "ユーザー",
            "本人",
            "私",
            "自己",
        }
    ),
    frozenset(
        {
            "admin",
            "administrator",
            "author",
            "built",
            "creator",
            "developer",
            "manager",
            "operator",
            "owner",
            "created",
            "作成者",
            "制作者",
            "管理者",
            "開発者",
            "運営",
            "責任者",
        }
    ),
)
_SEARCH_TERM_GROUP_BY_TOKEN = {
    token: f"concept:{index}"
    for index, group in enumerate(_SEARCH_TERM_GROUPS)
    for token in group
}
_CJK_SEARCH_STOP_TOKENS = frozenset(
    {
        "ある",
        "いる",
        "から",
        "こと",
        "した",
        "して",
        "する",
        "ため",
        "です",
        "ます",
        "よう",
    }
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    \b(?:api[_\s-]?key|access[_\s-]?token|auth(?:orization)?|
    client[_\s-]?secret|cookie|pass(?:word|wd)?|private[_\s-]?key|secret|token)
    \b\s*(?::|=|\bis\b|は)\s*\S+
    """
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,}\b"),
    re.compile(r"\bmfa\.[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(
        r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"
    ),
    re.compile(r"\b(?:[A-Fa-f0-9]{40,}|[A-Za-z0-9+/]{32,}={0,2})\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
)
_INFERENCE_PHRASES = (
    "probably",
    "likely",
    "maybe",
    "might",
    "appears to",
    "seems to",
    "presumably",
    "たぶん",
    "多分",
    "おそらく",
    "恐らく",
    "かもしれ",
    "ようだ",
    "らしい",
    "と思われ",
)


class AgentMemoryScope(StrEnum):
    """Host-enforced visibility boundary for a memory."""

    USER = "user"
    CHANNEL = "channel"
    WORKSPACE = "workspace"
    PROCEDURE = "procedure"


class AgentMemoryBasis(StrEnum):
    """Allowed non-inferential evidence for a durable memory."""

    USER_STATED = "user_stated"
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"


@dataclass(frozen=True, slots=True)
class AgentMemorySourceLocator:
    """Message provenance only; never an authorization or permission grant."""

    message_id: str
    channel_id: str | None
    guild_id: str | None


@dataclass(frozen=True, slots=True)
class AgentMemoryRecord:
    """Bounded projection; message bodies and attachments are never retained."""

    memory_id: str
    scope: AgentMemoryScope
    workspace_id: str
    owner_user_id: str | None
    channel_id: str | None
    key: str
    summary: str
    source_message_ids: tuple[str, ...]
    source_message_locators: tuple[AgentMemorySourceLocator, ...]
    basis: AgentMemoryBasis
    confidence: float
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentMemorySearchRequest:
    query: str = field(
        metadata={
            "description": (
                "Two to four likely key terms in Japanese or English. Search is "
                "case/width/punctuation tolerant and supports common wording variants; "
                "empty text returns the most recently used accessible memories."
            )
        }
    )
    scopes: tuple[AgentMemoryScope, ...] = field(
        default=(
            AgentMemoryScope.USER,
            AgentMemoryScope.CHANNEL,
            AgentMemoryScope.WORKSPACE,
            AgentMemoryScope.PROCEDURE,
        ),
        metadata={
            "description": (
                "Visibility filters: user is the current requester's private memory; "
                "channel is only the current origin channel; workspace is shared in "
                "the current guild; procedure is a guild-scoped verified workflow."
            )
        },
    )
    basis: AgentMemoryBasis | None = field(
        default=None,
        metadata={
            "description": (
                "Optional kind filter: user_stated preferences/rules or "
                "verified_success/verified_failure procedure outcomes."
            )
        },
    )
    min_confidence: float = field(
        default=MIN_MEMORY_CONFIDENCE,
        metadata={
            "description": (
                "Minimum stored confidence from 0.8 through 1.0."
            )
        },
    )
    updated_after: str | None = field(
        default=None,
        metadata={
            "description": (
                "Optional RFC 3339 timezone-aware timestamp; return only memories "
                "updated at or after it."
            )
        },
    )
    offset: int = field(
        default=0,
        metadata={
            "description": (
                "Zero-based result offset. Use next_offset from the prior response."
            )
        },
    )
    limit: int = field(
        default=5,
        metadata={
            "description": (
                "Results per page, from 1 through 10. If next_offset is not null, "
                "repeat the same filters with offset=next_offset."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class AgentMemorySearchResponse:
    query: str
    memories: tuple[AgentMemoryRecord, ...]
    next_offset: int | None


@dataclass(frozen=True, slots=True)
class AgentMemoryRememberRequest:
    scope: AgentMemoryScope
    key: str = field(
        metadata={
            "description": (
                "Stable semantic key used to upsert, such as response.language or "
                "pdf.review.checklist."
            )
        }
    )
    summary: str = field(
        metadata={
            "description": (
                "A short paraphrased preference, rule, or verified successful/failed "
                "procedure outcome; never a message body, attachment, credential, "
                "or inferred profile."
            )
        }
    )
    source_message_ids: tuple[str, ...] = field(
        metadata={
            "description": (
                "Exact Discord message IDs supporting this memory; content is not stored."
            )
        }
    )
    basis: AgentMemoryBasis
    confidence: float
    ttl_days: int | None = None
    source_message_locators: tuple[AgentMemorySourceLocator, ...] = field(
        default=(),
        metadata={
            "description": (
                "Optional guild/channel/message locators for evidence found outside "
                "the active channel. They are provenance only and never authorize "
                "reading or acting; message IDs must exactly match source_message_ids. "
                "When omitted, every ID is located in the active Discord channel."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class AgentMemoryUpdateRequest:
    memory_id: str
    summary: str
    source_message_ids: tuple[str, ...]
    confidence: float
    ttl_days: int | None = None
    source_message_locators: tuple[AgentMemorySourceLocator, ...] = field(
        default=(),
        metadata={
            "description": (
                "Optional guild/channel/message provenance locators. They never grant "
                "authority; message IDs must exactly match source_message_ids."
            )
        },
    )


@dataclass(frozen=True, slots=True)
class AgentMemoryWriteResponse:
    memory: AgentMemoryRecord
    created: bool


@dataclass(frozen=True, slots=True)
class AgentMemoryForgetRequest:
    memory_id: str


@dataclass(frozen=True, slots=True)
class AgentMemoryForgetResponse:
    memory_id: str
    forgotten: bool


def _memory_table_sql(table_name: str, *, if_not_exists: bool) -> str:
    if table_name not in {"agent_memories", "agent_memories_basis_v2"}:
        raise ValueError("unexpected memory table name")
    create_mode = "IF NOT EXISTS " if if_not_exists else ""
    return f"""
        CREATE TABLE {create_mode}{table_name} (
            memory_id TEXT PRIMARY KEY,
            locator TEXT NOT NULL UNIQUE,
            scope TEXT NOT NULL CHECK (
                scope IN ('user', 'channel', 'workspace', 'procedure')
            ),
            workspace_id TEXT NOT NULL,
            owner_user_id TEXT,
            channel_id TEXT,
            memory_key TEXT NOT NULL
                CHECK (length(memory_key) BETWEEN 1 AND {MAX_MEMORY_KEY_CHARACTERS}),
            summary TEXT NOT NULL
                CHECK (
                    length(summary) BETWEEN 1 AND
                    {MAX_MEMORY_SUMMARY_CHARACTERS}
                ),
            source_message_ids_json TEXT NOT NULL,
            source_message_locators_json TEXT NOT NULL DEFAULT '[]',
            basis TEXT NOT NULL CHECK (
                basis IN (
                    'user_stated',
                    'verified_success',
                    'verified_failure'
                )
            ),
            confidence REAL NOT NULL CHECK (
                confidence >= {MIN_MEMORY_CONFIDENCE}
                AND confidence <= 1.0
            ),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL,
            expires_at TEXT,
            CHECK (
                (scope = 'user' AND owner_user_id IS NOT NULL
                    AND channel_id IS NULL)
                OR
                (scope = 'channel' AND owner_user_id IS NULL
                    AND channel_id IS NOT NULL)
                OR
                (scope IN ('workspace', 'procedure')
                    AND owner_user_id IS NULL AND channel_id IS NULL)
            ),
            CHECK (
                (
                    scope = 'procedure'
                    AND basis IN ('verified_success', 'verified_failure')
                )
                OR
                (scope != 'procedure' AND basis = 'user_stated')
            )
        )
    """


def _create_memory_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_memories_workspace_scope
        ON agent_memories(workspace_id, scope)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_memories_user
        ON agent_memories(workspace_id, owner_user_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_memories_channel
        ON agent_memories(workspace_id, channel_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_memories_expiry
        ON agent_memories(expires_at)
        """
    )


def _migrate_memory_basis_constraint(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'agent_memories'
        """
    ).fetchone()
    if row is None or "verified_failure" in str(row["sql"]):
        return

    columns = (
        "memory_id, locator, scope, workspace_id, owner_user_id, channel_id, "
        "memory_key, summary, source_message_ids_json, "
        "source_message_locators_json, basis, confidence, created_at, "
        "updated_at, last_used_at, expires_at"
    )
    connection.commit()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            _memory_table_sql("agent_memories_basis_v2", if_not_exists=False)
        )
        connection.execute(
            f"""
            INSERT INTO agent_memories_basis_v2 ({columns})
            SELECT {columns}
            FROM agent_memories
            """
        )
        connection.execute("DROP TABLE agent_memories")
        connection.execute(
            "ALTER TABLE agent_memories_basis_v2 RENAME TO agent_memories"
        )
        _create_memory_indexes(connection)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


class AgentMemoryStore:
    """Restart-safe SQLite memory with bounded cardinality and expiry."""

    def __init__(
        self,
        path: Path,
        *,
        max_records: int = 2_000,
        max_records_per_workspace: int = 500,
        max_records_per_user: int = 100,
        max_records_per_channel: int = 100,
        max_workspace_records: int = 150,
        max_procedure_records: int = 150,
    ) -> None:
        per_scope_limits = (
            max_records_per_user,
            max_records_per_channel,
            max_workspace_records,
            max_procedure_records,
        )
        if max_records < 1:
            raise ValueError("memory max_records must be positive")
        if not 1 <= max_records_per_workspace <= max_records:
            raise ValueError(
                "memory max_records_per_workspace must be between 1 and max_records"
            )
        if any(
            limit < 1 or limit > max_records_per_workspace
            for limit in per_scope_limits
        ):
            raise ValueError(
                "memory per-scope limits must be between 1 and the workspace limit"
            )
        self.path = path
        self.max_records = max_records
        self.max_records_per_workspace = max_records_per_workspace
        self.max_records_per_user = max_records_per_user
        self.max_records_per_channel = max_records_per_channel
        self.max_workspace_records = max_workspace_records
        self.max_procedure_records = max_procedure_records
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def remember(
        self,
        *,
        scope: AgentMemoryScope,
        workspace_id: str,
        owner_user_id: str | None,
        channel_id: str | None,
        key: str,
        summary: str,
        source_message_ids: tuple[str, ...],
        source_message_locators: tuple[AgentMemorySourceLocator, ...] = (),
        basis: AgentMemoryBasis,
        confidence: float,
        expires_at: datetime | None,
        now: datetime,
    ) -> tuple[AgentMemoryRecord, bool]:
        if not source_message_locators:
            source_message_locators = tuple(
                AgentMemorySourceLocator(
                    message_id=message_id,
                    channel_id=channel_id,
                    guild_id=workspace_id,
                )
                for message_id in source_message_ids
            )
        async with self._lock:
            return await asyncio.to_thread(
                self._remember,
                scope,
                workspace_id,
                owner_user_id,
                channel_id,
                key,
                summary,
                source_message_ids,
                source_message_locators,
                basis,
                confidence,
                expires_at,
                _utc(now),
            )

    async def update(
        self,
        *,
        memory_id: str,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        summary: str,
        source_message_ids: tuple[str, ...],
        source_message_locators: tuple[AgentMemorySourceLocator, ...] = (),
        confidence: float,
        expires_at: datetime | None,
        now: datetime,
    ) -> AgentMemoryRecord:
        if not source_message_locators:
            source_message_locators = tuple(
                AgentMemorySourceLocator(
                    message_id=message_id,
                    channel_id=channel_id,
                    guild_id=workspace_id,
                )
                for message_id in source_message_ids
            )
        async with self._lock:
            return await asyncio.to_thread(
                self._update,
                memory_id,
                workspace_id,
                actor_id,
                channel_id,
                summary,
                source_message_ids,
                source_message_locators,
                confidence,
                expires_at,
                _utc(now),
            )

    async def forget(
        self,
        *,
        memory_id: str,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        now: datetime,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._forget,
                memory_id,
                workspace_id,
                actor_id,
                channel_id,
                _utc(now),
            )

    async def search(
        self,
        *,
        query: str,
        scopes: tuple[AgentMemoryScope, ...],
        limit: int,
        basis: AgentMemoryBasis | None = None,
        min_confidence: float = MIN_MEMORY_CONFIDENCE,
        updated_after: datetime | None = None,
        offset: int = 0,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        now: datetime,
        mark_used_limit: int | None = None,
    ) -> tuple[AgentMemoryRecord, ...]:
        async with self._lock:
            return await asyncio.to_thread(
                self._search,
                query,
                scopes,
                limit,
                basis,
                min_confidence,
                _utc(updated_after) if updated_after is not None else None,
                offset,
                workspace_id,
                actor_id,
                channel_id,
                _utc(now),
                mark_used_limit,
            )

    async def cleanup(self, *, now: datetime | None = None) -> int:
        current = _utc(now or datetime.now(UTC))
        async with self._lock:
            return await asyncio.to_thread(self._cleanup, current)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                _memory_table_sql("agent_memories", if_not_exists=True)
            )
            _create_memory_indexes(connection)
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(agent_memories)"
                ).fetchall()
            }
            if "source_message_locators_json" not in columns:
                connection.execute(
                    "ALTER TABLE agent_memories "
                    "ADD COLUMN source_message_locators_json TEXT NOT NULL DEFAULT '[]'"
                )
            legacy_rows = connection.execute(
                """
                SELECT memory_id, workspace_id, channel_id,
                       source_message_ids_json, source_message_locators_json
                FROM agent_memories
                WHERE source_message_locators_json = '[]'
                """
            ).fetchall()
            for row in legacy_rows:
                raw_ids = json.loads(str(row["source_message_ids_json"]))
                if not isinstance(raw_ids, list) or not all(
                    isinstance(value, str) for value in raw_ids
                ):
                    continue
                locators = tuple(
                    AgentMemorySourceLocator(
                        message_id=value,
                        channel_id=(
                            str(row["channel_id"])
                            if row["channel_id"] is not None
                            else None
                        ),
                        guild_id=str(row["workspace_id"]),
                    )
                    for value in raw_ids
                )
                connection.execute(
                    """
                    UPDATE agent_memories
                    SET source_message_locators_json = ?
                    WHERE memory_id = ?
                    """,
                    (_source_locators_json(locators), str(row["memory_id"])),
                )
            _migrate_memory_basis_constraint(connection)
        os.chmod(self.path, 0o600)

    def _remember(
        self,
        scope: AgentMemoryScope,
        workspace_id: str,
        owner_user_id: str | None,
        channel_id: str | None,
        key: str,
        summary: str,
        source_message_ids: tuple[str, ...],
        source_message_locators: tuple[AgentMemorySourceLocator, ...],
        basis: AgentMemoryBasis,
        confidence: float,
        expires_at: datetime | None,
        now: datetime,
    ) -> tuple[AgentMemoryRecord, bool]:
        locator = _memory_locator(
            scope=scope,
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            channel_id=channel_id,
            key=key,
        )
        now_text = now.isoformat()
        expiry_text = expires_at.isoformat() if expires_at is not None else None
        source_json = json.dumps(
            source_message_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        source_locators_json = _source_locators_json(source_message_locators)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now_text)
            existing = connection.execute(
                "SELECT memory_id FROM agent_memories WHERE locator = ?",
                (locator,),
            ).fetchone()
            created = existing is None
            memory_id = (
                f"mem_{uuid.uuid4().hex}"
                if existing is None
                else str(existing["memory_id"])
            )
            if created:
                connection.execute(
                    """
                    INSERT INTO agent_memories(
                        memory_id, locator, scope, workspace_id, owner_user_id,
                        channel_id, memory_key, summary, source_message_ids_json,
                        source_message_locators_json, basis, confidence, created_at,
                        updated_at, last_used_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        locator,
                        scope.value,
                        workspace_id,
                        owner_user_id,
                        channel_id,
                        key,
                        summary,
                        source_json,
                        source_locators_json,
                        basis.value,
                        confidence,
                        now_text,
                        now_text,
                        now_text,
                        expiry_text,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE agent_memories
                    SET summary = ?, source_message_ids_json = ?,
                        source_message_locators_json = ?, basis = ?,
                        confidence = ?, updated_at = ?, expires_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        summary,
                        source_json,
                        source_locators_json,
                        basis.value,
                        confidence,
                        now_text,
                        expiry_text,
                        memory_id,
                    ),
                )
            self._enforce_caps(
                connection,
                workspace_id=workspace_id,
                scope=scope,
                owner_user_id=owner_user_id,
                channel_id=channel_id,
                protected_memory_id=memory_id,
            )
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("memory cap enforcement removed the protected record")
            connection.commit()
            return _memory_from_row(row), created
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _update(
        self,
        memory_id: str,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        summary: str,
        source_message_ids: tuple[str, ...],
        source_message_locators: tuple[AgentMemorySourceLocator, ...],
        confidence: float,
        expires_at: datetime | None,
        now: datetime,
    ) -> AgentMemoryRecord:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now.isoformat())
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None or not _row_is_accessible(
                row,
                workspace_id=workspace_id,
                actor_id=actor_id,
                channel_id=channel_id,
            ):
                raise UserError("memory.not_found")
            connection.execute(
                """
                UPDATE agent_memories
                SET summary = ?, source_message_ids_json = ?,
                    source_message_locators_json = ?, confidence = ?,
                    updated_at = ?, expires_at = ?
                WHERE memory_id = ?
                """,
                (
                    summary,
                    json.dumps(
                        source_message_ids,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    _source_locators_json(source_message_locators),
                    confidence,
                    now.isoformat(),
                    expires_at.isoformat() if expires_at is not None else None,
                    memory_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if updated is None:
                raise RuntimeError("updated memory disappeared")
            connection.commit()
            return _memory_from_row(updated)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _forget(
        self,
        memory_id: str,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        now: datetime,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now.isoformat())
            row = connection.execute(
                "SELECT * FROM agent_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None or not _row_is_accessible(
                row,
                workspace_id=workspace_id,
                actor_id=actor_id,
                channel_id=channel_id,
            ):
                raise UserError("memory.not_found")
            connection.execute(
                "DELETE FROM agent_memories WHERE memory_id = ?",
                (memory_id,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _search(
        self,
        query: str,
        scopes: tuple[AgentMemoryScope, ...],
        limit: int,
        basis: AgentMemoryBasis | None,
        min_confidence: float,
        updated_after: datetime | None,
        offset: int,
        workspace_id: str,
        actor_id: str,
        channel_id: str | None,
        now: datetime,
        mark_used_limit: int | None,
    ) -> tuple[AgentMemoryRecord, ...]:
        scope_values = tuple(scope.value for scope in scopes)
        placeholders = ", ".join("?" for _ in scope_values)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now.isoformat())
            rows = connection.execute(
                f"""
                SELECT * FROM agent_memories
                WHERE workspace_id = ?
                  AND scope IN ({placeholders})
                  AND (
                    (scope = 'user' AND owner_user_id = ?)
                    OR (scope = 'channel' AND channel_id = ?)
                    OR scope IN ('workspace', 'procedure')
                  )
                ORDER BY last_used_at DESC, updated_at DESC
                LIMIT ?
                """,
                (
                    workspace_id,
                    *scope_values,
                    actor_id,
                    channel_id,
                    min(self.max_records_per_workspace, 500),
                ),
            ).fetchall()
            normalized_query = _normalize_memory_search_text(query)
            query_tokens = _search_tokens(normalized_query)
            scored: list[
                tuple[int, float, str, str, str, str, sqlite3.Row]
            ] = []
            for row in rows:
                if (
                    basis is not None
                    and str(row["basis"]) != basis.value
                ):
                    continue
                if float(row["confidence"]) < min_confidence:
                    continue
                if (
                    updated_after is not None
                    and str(row["updated_at"]) < updated_after.isoformat()
                ):
                    continue
                score = _memory_search_score(
                    normalized_query,
                    query_tokens,
                    key=str(row["memory_key"]),
                    summary=str(row["summary"]),
                )
                if normalized_query and score == 0:
                    continue
                scored.append(
                    (
                        score,
                        float(row["confidence"]),
                        str(row["last_used_at"]),
                        str(row["updated_at"]),
                        str(row["created_at"]),
                        str(row["memory_id"]),
                        row,
                    )
                )
            scored.sort(
                key=lambda item: item[:6],
                reverse=True,
            )
            selected_rows = tuple(
                item[6] for item in scored[offset : offset + limit]
            )
            used_count = (
                len(selected_rows)
                if mark_used_limit is None
                else min(mark_used_limit, len(selected_rows))
            )
            used_rows = selected_rows[:used_count]
            if used_rows:
                connection.executemany(
                    """
                    UPDATE agent_memories SET last_used_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        (now.isoformat(), str(row["memory_id"]))
                        for row in used_rows
                    ),
                )
            connection.commit()
            return tuple(
                _memory_from_row(
                    row,
                    last_used_at=now if index < used_count else None,
                )
                for index, row in enumerate(selected_rows)
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _cleanup(self, now: datetime) -> int:
        with self._connect() as connection:
            return self._delete_expired(connection, now.isoformat())

    def _delete_expired(
        self,
        connection: sqlite3.Connection,
        now_text: str,
    ) -> int:
        cursor = connection.execute(
            """
            DELETE FROM agent_memories
            WHERE expires_at IS NOT NULL AND expires_at <= ?
            """,
            (now_text,),
        )
        return cursor.rowcount

    def _enforce_caps(
        self,
        connection: sqlite3.Connection,
        *,
        workspace_id: str,
        scope: AgentMemoryScope,
        owner_user_id: str | None,
        channel_id: str | None,
        protected_memory_id: str,
    ) -> None:
        parameters: tuple[object, ...]
        if scope is AgentMemoryScope.USER:
            where_sql = (
                "workspace_id = ? AND scope = 'user' AND owner_user_id = ?"
            )
            parameters = (workspace_id, owner_user_id)
            limit = self.max_records_per_user
        elif scope is AgentMemoryScope.CHANNEL:
            where_sql = (
                "workspace_id = ? AND scope = 'channel' AND channel_id = ?"
            )
            parameters = (workspace_id, channel_id)
            limit = self.max_records_per_channel
        elif scope is AgentMemoryScope.WORKSPACE:
            where_sql = "workspace_id = ? AND scope = 'workspace'"
            parameters = (workspace_id,)
            limit = self.max_workspace_records
        else:
            where_sql = "workspace_id = ? AND scope = 'procedure'"
            parameters = (workspace_id,)
            limit = self.max_procedure_records
        self._evict_over_limit(
            connection,
            where_sql=where_sql,
            parameters=parameters,
            limit=limit,
            protected_memory_id=protected_memory_id,
        )
        # Enforce narrow ownership first. Otherwise a broad global eviction can
        # remove unrelated shared memory and the later narrow eviction removes a
        # second row even though deleting only one scoped row was sufficient.
        self._evict_over_limit(
            connection,
            where_sql="workspace_id = ?",
            parameters=(workspace_id,),
            limit=self.max_records_per_workspace,
            protected_memory_id=protected_memory_id,
        )
        self._evict_over_limit(
            connection,
            where_sql="1 = 1",
            parameters=(),
            limit=self.max_records,
            protected_memory_id=protected_memory_id,
        )

    @staticmethod
    def _evict_over_limit(
        connection: sqlite3.Connection,
        *,
        where_sql: str,
        parameters: tuple[object, ...],
        limit: int,
        protected_memory_id: str,
    ) -> None:
        count_row = connection.execute(
            f"SELECT COUNT(*) FROM agent_memories WHERE {where_sql}",
            parameters,
        ).fetchone()
        count = int(count_row[0]) if count_row is not None else 0
        overflow = count - limit
        if overflow <= 0:
            return
        connection.execute(
            f"""
            DELETE FROM agent_memories
            WHERE memory_id IN (
                SELECT memory_id FROM agent_memories
                WHERE {where_sql} AND memory_id != ?
                ORDER BY last_used_at ASC, updated_at ASC, created_at ASC
                LIMIT ?
            )
            """,
            (*parameters, protected_memory_id, overflow),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection


class AgentMemoryService:
    """Validate evidence and derive all scope ownership from trusted context."""

    def __init__(self, store: AgentMemoryStore) -> None:
        self.store = store

    async def context_for_turn(
        self,
        context: InvocationContext,
        *,
        limit: int = 4,
    ) -> tuple[AgentMemoryRecord, ...]:
        """Return a tiny requester-private context without claiming it was used."""

        if context.workspace_id is None:
            return ()
        if not 1 <= limit <= MAX_MEMORY_SEARCH_RESULTS:
            raise ValueError("turn memory context limit is invalid")
        return await self.store.search(
            query="",
            scopes=(AgentMemoryScope.USER,),
            limit=limit,
            min_confidence=MIN_MEMORY_CONFIDENCE,
            workspace_id=context.workspace_id,
            actor_id=context.actor_id,
            channel_id=context.origin_resource_id,
            now=datetime.now(UTC),
            mark_used_limit=0,
        )

    async def search(
        self,
        request: AgentMemorySearchRequest,
        context: InvocationContext,
    ) -> AgentMemorySearchResponse:
        workspace_id = _workspace(context)
        query = _bounded_query(request.query)
        scopes = _validated_scopes(request.scopes)
        if not 1 <= request.limit <= MAX_MEMORY_SEARCH_RESULTS:
            raise UserError(
                "memory.limit_invalid",
                minimum=1,
                maximum=MAX_MEMORY_SEARCH_RESULTS,
            )
        if (
            not isinstance(request.offset, int)
            or isinstance(request.offset, bool)
            or not 0 <= request.offset <= MAX_MEMORY_SEARCH_OFFSET
        ):
            raise UserError(
                "memory.offset_invalid",
                minimum=0,
                maximum=MAX_MEMORY_SEARCH_OFFSET,
            )
        min_confidence = _validated_confidence(request.min_confidence)
        try:
            updated_after = _optional_timestamp(request.updated_after)
        except (TypeError, ValueError) as exc:
            raise UserError("memory.updated_after_invalid") from exc
        memories = await self.store.search(
            query=query,
            scopes=scopes,
            limit=request.limit + 1,
            basis=request.basis,
            min_confidence=min_confidence,
            updated_after=updated_after,
            offset=request.offset,
            workspace_id=workspace_id,
            actor_id=context.actor_id,
            channel_id=context.origin_resource_id,
            now=datetime.now(UTC),
            mark_used_limit=request.limit,
        )
        has_more = len(memories) > request.limit
        return AgentMemorySearchResponse(
            query=query,
            memories=memories[: request.limit],
            next_offset=(
                request.offset + request.limit if has_more else None
            ),
        )

    async def remember(
        self,
        request: AgentMemoryRememberRequest,
        context: InvocationContext,
    ) -> AgentMemoryWriteResponse:
        workspace_id = _workspace(context)
        key = _bounded_text(
            request.key,
            limit=MAX_MEMORY_KEY_CHARACTERS,
            error_code="memory.key_invalid",
        )
        if _looks_secret(key):
            raise UserError("memory.secret_forbidden")
        summary = _validated_summary(request.summary)
        source_message_ids = _validated_source_message_ids(
            request.source_message_ids
        )
        source_message_locators = _validated_source_message_locators(
            request.source_message_locators,
            source_message_ids=source_message_ids,
            context=context,
        )
        confidence = _validated_confidence(request.confidence)
        _validate_basis(request.scope, request.basis, summary)
        owner_user_id, channel_id = _scope_owners(request.scope, context)
        now = datetime.now(UTC)
        record, created = await self.store.remember(
            scope=request.scope,
            workspace_id=workspace_id,
            owner_user_id=owner_user_id,
            channel_id=channel_id,
            key=key,
            summary=summary,
            source_message_ids=source_message_ids,
            source_message_locators=source_message_locators,
            basis=request.basis,
            confidence=confidence,
            expires_at=_expiry(request.ttl_days, now=now),
            now=now,
        )
        return AgentMemoryWriteResponse(memory=record, created=created)

    async def update(
        self,
        request: AgentMemoryUpdateRequest,
        context: InvocationContext,
    ) -> AgentMemoryWriteResponse:
        workspace_id = _workspace(context)
        memory_id = _validated_memory_id(request.memory_id)
        summary = _validated_summary(request.summary)
        source_message_ids = _validated_source_message_ids(
            request.source_message_ids
        )
        source_message_locators = _validated_source_message_locators(
            request.source_message_locators,
            source_message_ids=source_message_ids,
            context=context,
        )
        confidence = _validated_confidence(request.confidence)
        now = datetime.now(UTC)
        record = await self.store.update(
            memory_id=memory_id,
            workspace_id=workspace_id,
            actor_id=context.actor_id,
            channel_id=context.origin_resource_id,
            summary=summary,
            source_message_ids=source_message_ids,
            source_message_locators=source_message_locators,
            confidence=confidence,
            expires_at=_expiry(request.ttl_days, now=now),
            now=now,
        )
        _validate_basis(record.scope, record.basis, record.summary)
        return AgentMemoryWriteResponse(memory=record, created=False)

    async def forget(
        self,
        request: AgentMemoryForgetRequest,
        context: InvocationContext,
    ) -> AgentMemoryForgetResponse:
        workspace_id = _workspace(context)
        memory_id = _validated_memory_id(request.memory_id)
        await self.store.forget(
            memory_id=memory_id,
            workspace_id=workspace_id,
            actor_id=context.actor_id,
            channel_id=context.origin_resource_id,
            now=datetime.now(UTC),
        )
        return AgentMemoryForgetResponse(memory_id=memory_id, forgotten=True)


def build_memory_endpoints(
    service: AgentMemoryService,
) -> tuple[CapabilityEndpoint, ...]:
    """Build the searchable READ endpoint and authorized mutation endpoints."""

    expected_errors = (
        "workspace.required",
        "memory.channel_required",
        "memory.key_invalid",
        "memory.summary_invalid",
        "memory.source_message_ids_invalid",
        "memory.source_message_locators_invalid",
        "memory.source_message_not_read",
        "memory.source_message_locator_mismatch",
        "memory.basis_invalid",
        "memory.confidence_too_low",
        "memory.secret_forbidden",
        "memory.inference_forbidden",
        "memory.ttl_invalid",
    )
    return (
        endpoint(
            CapabilityDescriptor(
                name="memory.search",
                summary=(
                    "Search paginated durable preferences, rules, and verified "
                    "procedures by text, scope, basis, time, and confidence. Returns "
                    "memory IDs, source guild/channel/message locators, timestamps, and "
                    "next_offset for later source retrieval, update, or explicit forget."
                ),
                risk=RiskLevel.READ,
                keywords=(
                    "memory",
                    "preference",
                    "procedure",
                    "remembered",
                    "記憶",
                    "好み",
                    "手順",
                ),
                requires_workspace=True,
                expected_errors=(
                    "workspace.required",
                    "memory.limit_invalid",
                    "memory.query_invalid",
                    "memory.scopes_invalid",
                    "memory.offset_invalid",
                    "memory.updated_after_invalid",
                    "memory.confidence_too_low",
                ),
                timeout_seconds=10,
            ),
            AgentMemorySearchRequest,
            AgentMemorySearchResponse,
            service.search,
        ),
        endpoint(
            CapabilityDescriptor(
                name="memory.remember",
                summary=(
                    "Upsert one short user-stated preference/rule or verified "
                    "successful/failed procedure outcome, citing exact Discord "
                    "message locators."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "memory",
                    "remember",
                    "preference",
                    "procedure",
                    "覚える",
                    "記憶",
                ),
                side_effects=("Persists a bounded summary outside the model thread.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=expected_errors,
                timeout_seconds=10,
                user_visible_effect="Saves or refreshes one scoped durable memory.",
            ),
            AgentMemoryRememberRequest,
            AgentMemoryWriteResponse,
            service.remember,
        ),
        endpoint(
            CapabilityDescriptor(
                name="memory.update",
                summary=(
                    "Replace the summary, evidence locators, confidence, and expiry of "
                    "one accessible durable memory."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("memory", "update", "correct", "記憶", "修正"),
                side_effects=("Updates one bounded durable memory.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(*expected_errors, "memory.not_found"),
                timeout_seconds=10,
                user_visible_effect="Updates one accessible durable memory.",
            ),
            AgentMemoryUpdateRequest,
            AgentMemoryWriteResponse,
            service.update,
        ),
        endpoint(
            CapabilityDescriptor(
                name="memory.forget",
                summary=(
                    "Permanently delete one accessible durable memory by memory ID."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=("memory", "forget", "delete", "記憶", "忘れる", "削除"),
                side_effects=("Permanently deletes one short memory record.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=("workspace.required", "memory.not_found"),
                timeout_seconds=10,
                user_visible_effect="Forgets one memory; this action has no Undo.",
            ),
            AgentMemoryForgetRequest,
            AgentMemoryForgetResponse,
            service.forget,
        ),
    )


def _workspace(context: InvocationContext) -> str:
    if context.workspace_id is None:
        raise UserError("workspace.required")
    return context.workspace_id


def _scope_owners(
    scope: AgentMemoryScope,
    context: InvocationContext,
) -> tuple[str | None, str | None]:
    if scope is AgentMemoryScope.USER:
        return context.actor_id, None
    if scope is AgentMemoryScope.CHANNEL:
        if context.origin_resource_id is None:
            raise UserError("memory.channel_required")
        if (
            context.resource_ids
            and context.origin_resource_id not in context.resource_ids
        ):
            raise UserError("memory.channel_required")
        return None, context.origin_resource_id
    return None, None


def _validated_scopes(
    scopes: tuple[AgentMemoryScope, ...],
) -> tuple[AgentMemoryScope, ...]:
    unique = tuple(dict.fromkeys(scopes))
    if not unique or len(unique) != len(scopes):
        raise UserError("memory.scopes_invalid")
    return unique


def _bounded_query(value: str) -> str:
    query = " ".join(value.split())
    if len(query) > MAX_MEMORY_QUERY_CHARACTERS:
        raise UserError(
            "memory.query_invalid",
            maximum=MAX_MEMORY_QUERY_CHARACTERS,
        )
    return query


def _bounded_text(value: str, *, limit: int, error_code: str) -> str:
    text = " ".join(value.split())
    if not text or len(text) > limit or "\x00" in text:
        raise UserError(error_code, maximum=limit)
    return text


def _validated_summary(value: str) -> str:
    summary = _bounded_text(
        value,
        limit=MAX_MEMORY_SUMMARY_CHARACTERS,
        error_code="memory.summary_invalid",
    )
    if _looks_secret(summary):
        raise UserError("memory.secret_forbidden")
    normalized = unicodedata.normalize("NFKC", summary).casefold()
    if any(phrase in normalized for phrase in _INFERENCE_PHRASES):
        raise UserError("memory.inference_forbidden")
    return summary


def _validated_source_message_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(values))
    if (
        not unique
        or len(unique) != len(values)
        or len(unique) > MAX_MEMORY_SOURCE_MESSAGE_IDS
        or any(
            not value.isascii()
            or not value.isdigit()
            or len(value) > 32
            or int(value) <= 0
            for value in unique
        )
    ):
        raise UserError(
            "memory.source_message_ids_invalid",
            maximum=MAX_MEMORY_SOURCE_MESSAGE_IDS,
        )
    return unique


def _validated_source_message_locators(
    values: tuple[AgentMemorySourceLocator, ...],
    *,
    source_message_ids: tuple[str, ...],
    context: InvocationContext,
) -> tuple[AgentMemorySourceLocator, ...]:
    if not values:
        return tuple(
            AgentMemorySourceLocator(
                message_id=message_id,
                channel_id=context.origin_resource_id,
                guild_id=context.workspace_id,
            )
            for message_id in source_message_ids
        )
    if (
        len(values) != len(source_message_ids)
        or tuple(locator.message_id for locator in values) != source_message_ids
        or len({locator.message_id for locator in values}) != len(values)
    ):
        raise UserError(
            "memory.source_message_locators_invalid",
            maximum=MAX_MEMORY_SOURCE_MESSAGE_IDS,
        )
    validated: list[AgentMemorySourceLocator] = []
    for locator in values:
        try:
            message_id = _validated_snowflake_text(locator.message_id)
            channel_id = _validated_snowflake_text(locator.channel_id)
            guild_id = _validated_snowflake_text(locator.guild_id)
        except (TypeError, ValueError) as exc:
            raise UserError(
                "memory.source_message_locators_invalid",
                maximum=MAX_MEMORY_SOURCE_MESSAGE_IDS,
            ) from exc
        if channel_id is None or guild_id is None:
            raise UserError(
                "memory.source_message_locators_invalid",
                maximum=MAX_MEMORY_SOURCE_MESSAGE_IDS,
            )
        validated.append(
            AgentMemorySourceLocator(
                message_id=message_id or "",
                channel_id=channel_id,
                guild_id=guild_id,
            )
        )
    return tuple(validated)


def _validated_snowflake_text(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
        or len(value) > 32
        or int(value) <= 0
    ):
        raise ValueError("invalid Discord snowflake")
    return value


def _source_locators_json(
    locators: tuple[AgentMemorySourceLocator, ...],
) -> str:
    return json.dumps(
        [
            {
                "message_id": locator.message_id,
                "channel_id": locator.channel_id,
                "guild_id": locator.guild_id,
            }
            for locator in locators
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validated_confidence(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < MIN_MEMORY_CONFIDENCE
        or value > 1.0
    ):
        raise UserError(
            "memory.confidence_too_low",
            minimum=MIN_MEMORY_CONFIDENCE,
            maximum=1.0,
        )
    return value


def _validate_basis(
    scope: AgentMemoryScope,
    basis: AgentMemoryBasis,
    summary: str,
) -> None:
    if scope is AgentMemoryScope.PROCEDURE:
        if basis not in {
            AgentMemoryBasis.VERIFIED_SUCCESS,
            AgentMemoryBasis.VERIFIED_FAILURE,
        }:
            raise UserError("memory.basis_invalid")
    elif basis is not AgentMemoryBasis.USER_STATED:
        raise UserError("memory.basis_invalid")
    if _looks_secret(summary):
        raise UserError("memory.secret_forbidden")


def _expiry(ttl_days: int | None, *, now: datetime) -> datetime | None:
    if ttl_days is None:
        return None
    if (
        not isinstance(ttl_days, int)
        or isinstance(ttl_days, bool)
        or ttl_days < 1
        or ttl_days > MAX_MEMORY_TTL_DAYS
    ):
        raise UserError(
            "memory.ttl_invalid",
            minimum=1,
            maximum=MAX_MEMORY_TTL_DAYS,
        )
    return now + timedelta(days=ttl_days)


def _validated_memory_id(value: str) -> str:
    if (
        not value.startswith("mem_")
        or len(value) != 36
        or any(character not in "0123456789abcdef" for character in value[4:])
    ):
        raise UserError("memory.not_found")
    return value


def _looks_secret(value: str) -> bool:
    return bool(
        _SECRET_ASSIGNMENT_PATTERN.search(value)
        or any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)
    )


def _memory_locator(
    *,
    scope: AgentMemoryScope,
    workspace_id: str,
    owner_user_id: str | None,
    channel_id: str | None,
    key: str,
) -> str:
    normalized_key = _normalize(key)
    if scope is AgentMemoryScope.USER:
        owner = owner_user_id
    elif scope is AgentMemoryScope.CHANNEL:
        owner = channel_id
    else:
        owner = "-"
    if owner is None:
        raise ValueError("memory scope owner is missing")
    return "\x1f".join((scope.value, workspace_id, owner, normalized_key))


def _row_is_accessible(
    row: sqlite3.Row,
    *,
    workspace_id: str,
    actor_id: str,
    channel_id: str | None,
) -> bool:
    if str(row["workspace_id"]) != workspace_id:
        return False
    scope = AgentMemoryScope(str(row["scope"]))
    if scope is AgentMemoryScope.USER:
        return str(row["owner_user_id"]) == actor_id
    if scope is AgentMemoryScope.CHANNEL:
        return channel_id is not None and str(row["channel_id"]) == channel_id
    return True


def _memory_search_score(
    normalized_query: str,
    query_tokens: set[str],
    *,
    key: str,
    summary: str,
) -> int:
    if not normalized_query:
        return 1
    normalized_key = _normalize_memory_search_text(key)
    normalized_summary = _normalize_memory_search_text(summary)
    key_tokens = _search_tokens(normalized_key)
    summary_tokens = _search_tokens(normalized_summary)
    return (
        6 * int(normalized_query == normalized_key)
        + 4 * int(normalized_query in normalized_key)
        + 3 * int(normalized_query in normalized_summary)
        + 3 * len(query_tokens & key_tokens)
        + len(query_tokens & summary_tokens)
    )


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalize_memory_search_text(value: str) -> str:
    """Normalize display text without making punctuation part of a search term."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    searchable = "".join(
        character
        if (
            character.isspace()
            or unicodedata.category(character).startswith(("L", "N", "M"))
        )
        else " "
        for character in normalized
    )
    return " ".join(searchable.split())


def _search_tokens(value: str) -> set[str]:
    """Build a tiny bilingual lexical index suitable for <=500 scoped rows.

    SQLite FTS tokenizers do not segment Japanese by default. Scanning the already
    bounded candidate set lets us handle punctuation, inflection, and unspaced CJK
    text without another native dependency or a second index to keep in sync.
    """

    tokens: set[str] = set()
    for token in _LATIN_SEARCH_TOKEN_PATTERN.findall(value):
        tokens.add(token)
        stem = _english_search_stem(token)
        if stem:
            tokens.add(stem)
    for match in _CJK_SEARCH_RUN_PATTERN.finditer(value):
        run = match.group(0)
        tokens.add(run)
        for size in (2, 3):
            if len(run) < size:
                continue
            tokens.update(
                run[index : index + size]
                for index in range(len(run) - size + 1)
                if run[index : index + size] not in _CJK_SEARCH_STOP_TOKENS
            )
    tokens.update(
        group
        for token in tuple(tokens)
        if (group := _SEARCH_TERM_GROUP_BY_TOKEN.get(token)) is not None
    )
    return tokens


def _english_search_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ly"):
        stem = token[:-2]
        return f"{stem[:-1]}e" if stem.endswith("i") else stem
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith(("es", "ed")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _optional_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("memory timestamp must be non-empty text")
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    return _utc(datetime.fromisoformat(normalized))


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("memory timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _memory_from_row(
    row: sqlite3.Row,
    *,
    last_used_at: datetime | None = None,
) -> AgentMemoryRecord:
    source_value = json.loads(str(row["source_message_ids_json"]))
    if not isinstance(source_value, list) or not all(
        isinstance(value, str) for value in source_value
    ):
        raise RuntimeError("invalid memory source message IDs")
    raw_locators = json.loads(str(row["source_message_locators_json"]))
    if not isinstance(raw_locators, list):
        raise RuntimeError("invalid memory source message locators")
    source_locators: list[AgentMemorySourceLocator] = []
    for value in raw_locators:
        if not isinstance(value, dict):
            raise RuntimeError("invalid memory source message locators")
        message_id = value.get("message_id")
        channel_id = value.get("channel_id")
        guild_id = value.get("guild_id")
        if (
            not isinstance(message_id, str)
            or (channel_id is not None and not isinstance(channel_id, str))
            or (guild_id is not None and not isinstance(guild_id, str))
        ):
            raise RuntimeError("invalid memory source message locators")
        source_locators.append(
            AgentMemorySourceLocator(
                message_id=message_id,
                channel_id=channel_id,
                guild_id=guild_id,
            )
        )
    return AgentMemoryRecord(
        memory_id=str(row["memory_id"]),
        scope=AgentMemoryScope(str(row["scope"])),
        workspace_id=str(row["workspace_id"]),
        owner_user_id=(
            str(row["owner_user_id"])
            if row["owner_user_id"] is not None
            else None
        ),
        channel_id=(
            str(row["channel_id"]) if row["channel_id"] is not None else None
        ),
        key=str(row["memory_key"]),
        summary=str(row["summary"]),
        source_message_ids=tuple(source_value),
        source_message_locators=tuple(source_locators),
        basis=AgentMemoryBasis(str(row["basis"])),
        confidence=float(row["confidence"]),
        created_at=datetime.fromisoformat(str(row["created_at"])).astimezone(UTC),
        updated_at=datetime.fromisoformat(str(row["updated_at"])).astimezone(UTC),
        last_used_at=(
            last_used_at
            if last_used_at is not None
            else datetime.fromisoformat(str(row["last_used_at"])).astimezone(UTC)
        ),
        expires_at=(
            datetime.fromisoformat(str(row["expires_at"])).astimezone(UTC)
            if row["expires_at"] is not None
            else None
        ),
    )
