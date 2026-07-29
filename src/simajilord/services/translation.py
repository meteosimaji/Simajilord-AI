"""Provider-neutral language detection and on-device translation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from simajilord.core.errors import ProviderError, UserError


@dataclass(frozen=True, slots=True)
class TranslationHypothesis:
    """One language-detection candidate."""

    code: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TranslationDetection:
    """Detected source language and bounded alternatives."""

    language: str
    confidence: float
    hypotheses: tuple[TranslationHypothesis, ...]


@dataclass(frozen=True, slots=True)
class TranslationLanguage:
    """A provider language suitable for Discord autocomplete."""

    code: str
    english_name: str
    native_name: str
    availability: str


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """One completed translation."""

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    provider: str


@dataclass(frozen=True, slots=True)
class TranslationSegment:
    """One independently addressable fragment of a structured document."""

    identifier: str
    text: str


@dataclass(frozen=True, slots=True)
class TranslatedSegment:
    """One translated fragment while preserving its document identifier."""

    identifier: str
    source_text: str
    translated_text: str


@dataclass(frozen=True, slots=True)
class TranslationBatchResult:
    """A structure-preserving batch translation."""

    segments: tuple[TranslatedSegment, ...]
    source_language: str
    target_language: str
    provider: str
    cached: bool = False


@dataclass(frozen=True, slots=True)
class TranslationPreference:
    """A user's explicit translation target for one optional workspace."""

    target_language: str
    show_original: bool


class TranslationProvider(Protocol):
    """Replaceable local translation backend."""

    @property
    def name(self) -> str:
        """Stable provider name shown in diagnostics."""

    async def detect_language(self, text: str) -> TranslationDetection:
        """Detect the dominant source language."""

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        """Return target languages and pair availability."""

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationResult:
        """Translate text without a cloud model."""

    async def translate_batch(
        self,
        segments: tuple[TranslationSegment, ...],
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationBatchResult:
        """Translate a structured group without losing segment identifiers."""

    async def close(self) -> None:
        """Release provider resources."""


class TranslationProviderError(ProviderError):
    """A stable provider failure, optionally eligible for fallback."""

    def __init__(
        self,
        code: str,
        technical_detail: str = "",
        *,
        fallback_allowed: bool,
    ) -> None:
        super().__init__(technical_detail or code)
        self.code = code
        self.technical_detail = technical_detail
        self.fallback_allowed = fallback_allowed


class TranslationService:
    """Validate requests and select the first capable offline provider."""

    def __init__(
        self,
        providers: tuple[TranslationProvider, ...],
        *,
        max_characters: int = 8_000,
        store: TranslationStore | None = None,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._providers = providers
        self._max_characters = max_characters
        self._store = store

    @property
    def available(self) -> bool:
        return bool(self._providers)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(provider.name for provider in self._providers)

    def _validated_text(self, text: str) -> str:
        value = text.strip()
        if not value:
            raise UserError("translation.text_required")
        if len(value) > self._max_characters:
            raise UserError(
                "translation.text_too_long",
                maximum=self._max_characters,
            )
        return value

    async def detect_language(self, text: str) -> TranslationDetection:
        value = self._validated_text(text)
        last_error: TranslationProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.detect_language(value)
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        source = _normalized_language(source_language)
        last_error: TranslationProviderError | None = None
        merged: dict[str, TranslationLanguage] = {}
        for provider in self._providers:
            try:
                for language in await provider.supported_targets(source):
                    merged.setdefault(language.code, language)
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if merged:
            return tuple(
                sorted(
                    merged.values(),
                    key=lambda item: (item.english_name.casefold(), item.code),
                )
            )
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None = None,
        target_language: str,
    ) -> TranslationResult:
        value = self._validated_text(text)
        target = _normalized_language(target_language)
        if target is None:
            raise UserError("translation.target_required")
        source = _normalized_language(source_language)
        if source is None:
            source = (await self.detect_language(value)).language
        if source.casefold() == target.casefold():
            raise UserError("translation.same_language")
        last_error: TranslationProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.translate(
                    value,
                    source_language=source,
                    target_language=target,
                )
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def translate_batch(
        self,
        segments: tuple[TranslationSegment, ...],
        *,
        source_language: str | None = None,
        target_language: str,
    ) -> TranslationBatchResult:
        values = self._validated_segments(segments)
        target = _normalized_language(target_language)
        if target is None:
            raise UserError("translation.target_required")
        source = _normalized_language(source_language)
        if source is None:
            source = (await self.detect_language(_detection_text(values))).language
        if source.casefold() == target.casefold():
            raise UserError("translation.same_language")
        cache_key = _batch_cache_key(values, source, target)
        if self._store is not None:
            cached = await self._store.cached(cache_key)
            if cached is not None:
                return cached
        last_error: TranslationProviderError | None = None
        for provider in self._providers:
            try:
                result = await provider.translate_batch(
                    values,
                    source_language=source,
                    target_language=target,
                )
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
            else:
                if self._store is not None:
                    await self._store.save(cache_key, result)
                return result
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    def _validated_segments(
        self,
        segments: tuple[TranslationSegment, ...],
    ) -> tuple[TranslationSegment, ...]:
        if not segments:
            raise UserError("translation.text_required")
        seen: set[str] = set()
        validated: list[TranslationSegment] = []
        total = 0
        for segment in segments:
            identifier = segment.identifier.strip()
            if not identifier or identifier in seen:
                raise UserError("translation.segment_invalid")
            value = segment.text.strip()
            if not value:
                continue
            seen.add(identifier)
            total += len(value)
            validated.append(TranslationSegment(identifier=identifier, text=value))
        if not validated:
            raise UserError("translation.text_required")
        if total > self._max_characters:
            raise UserError(
                "translation.text_too_long",
                maximum=self._max_characters,
            )
        return tuple(validated)

    async def preference(
        self,
        *,
        actor_id: str,
        workspace_id: str | None,
    ) -> TranslationPreference | None:
        if self._store is None:
            return None
        return await self._store.preference(
            actor_id=actor_id,
            workspace_id=workspace_id,
        )

    async def set_preference(
        self,
        *,
        actor_id: str,
        workspace_id: str | None,
        target_language: str,
        show_original: bool = False,
    ) -> None:
        target = _normalized_language(target_language)
        if target is None:
            raise UserError("translation.target_required")
        if self._store is not None:
            await self._store.set_preference(
                actor_id=actor_id,
                workspace_id=workspace_id,
                target_language=target,
                show_original=show_original,
            )

    async def record_recent_target(self, *, actor_id: str, code: str) -> None:
        normalized = _normalized_language(code)
        if normalized is not None and self._store is not None:
            await self._store.record_recent_target(actor_id, normalized)

    async def recent_targets(
        self,
        *,
        actor_id: str,
        limit: int = 5,
    ) -> tuple[str, ...]:
        if self._store is None:
            return ()
        return await self._store.recent_targets(actor_id, limit=limit)

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()


class TranslationStore:
    """SQLite authority for preferences, recents, and reusable batch results."""

    def __init__(self, path: Path, *, max_cache_rows: int = 2_000) -> None:
        if max_cache_rows < 1:
            raise ValueError("max_cache_rows must be positive")
        self.path = path
        self.max_cache_rows = max_cache_rows
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def preference(
        self,
        *,
        actor_id: str,
        workspace_id: str | None,
    ) -> TranslationPreference | None:
        async with self._lock:
            return await asyncio.to_thread(
                self._preference,
                actor_id,
                _workspace_key(workspace_id),
            )

    async def set_preference(
        self,
        *,
        actor_id: str,
        workspace_id: str | None,
        target_language: str,
        show_original: bool,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._set_preference,
                actor_id,
                _workspace_key(workspace_id),
                target_language,
                show_original,
            )

    async def record_recent_target(self, actor_id: str, code: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._record_recent_target, actor_id, code)

    async def recent_targets(
        self,
        actor_id: str,
        *,
        limit: int = 5,
    ) -> tuple[str, ...]:
        bounded = min(max(limit, 1), 25)
        async with self._lock:
            return await asyncio.to_thread(
                self._recent_targets,
                actor_id,
                bounded,
            )

    async def cached(self, key: str) -> TranslationBatchResult | None:
        async with self._lock:
            return await asyncio.to_thread(self._cached, key)

    async def save(self, key: str, result: TranslationBatchResult) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save, key, result)

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS translation_preferences (
                    actor_id TEXT NOT NULL,
                    workspace_key TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    show_original INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (actor_id, workspace_key)
                );
                CREATE TABLE IF NOT EXISTS translation_recent_targets (
                    actor_id TEXT NOT NULL,
                    language_code TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    PRIMARY KEY (actor_id, language_code)
                );
                CREATE INDEX IF NOT EXISTS translation_recent_actor_time
                    ON translation_recent_targets(actor_id, last_used_at DESC);
                CREATE TABLE IF NOT EXISTS translation_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()
        with suppress(OSError):
            self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _preference(
        self,
        actor_id: str,
        workspace_key: str,
    ) -> TranslationPreference | None:
        connection = self._connect()
        try:
            keys = (workspace_key, "") if workspace_key else ("",)
            for key in keys:
                row = connection.execute(
                    """
                    SELECT target_language, show_original
                    FROM translation_preferences
                    WHERE actor_id = ? AND workspace_key = ?
                    """,
                    (actor_id, key),
                ).fetchone()
                if row is not None:
                    return TranslationPreference(
                        target_language=str(row[0]),
                        show_original=bool(row[1]),
                    )
        finally:
            connection.close()
        return None

    def _set_preference(
        self,
        actor_id: str,
        workspace_key: str,
        target_language: str,
        show_original: bool,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO translation_preferences (
                    actor_id, workspace_key, target_language,
                    show_original, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(actor_id, workspace_key) DO UPDATE SET
                    target_language = excluded.target_language,
                    show_original = excluded.show_original,
                    updated_at = excluded.updated_at
                """,
                (
                    actor_id,
                    workspace_key,
                    target_language,
                    int(show_original),
                    _now_iso(),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def _record_recent_target(self, actor_id: str, code: str) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO translation_recent_targets (
                    actor_id, language_code, last_used_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(actor_id, language_code) DO UPDATE SET
                    last_used_at = excluded.last_used_at
                """,
                (actor_id, code, _now_iso()),
            )
            connection.commit()
        finally:
            connection.close()

    def _recent_targets(self, actor_id: str, limit: int) -> tuple[str, ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT language_code
                FROM translation_recent_targets
                WHERE actor_id = ?
                ORDER BY last_used_at DESC
                LIMIT ?
                """,
                (actor_id, limit),
            ).fetchall()
        finally:
            connection.close()
        return tuple(str(row[0]) for row in rows)

    def _cached(self, key: str) -> TranslationBatchResult | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result_json FROM translation_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            payload = json.loads(str(row[0]))
            return TranslationBatchResult(
                segments=tuple(
                    TranslatedSegment(
                        identifier=str(item["identifier"]),
                        source_text=str(item["source_text"]),
                        translated_text=str(item["translated_text"]),
                    )
                    for item in payload["segments"]
                ),
                source_language=str(payload["source_language"]),
                target_language=str(payload["target_language"]),
                provider=str(payload["provider"]),
                cached=True,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save(self, key: str, result: TranslationBatchResult) -> None:
        payload = json.dumps(
            {
                "segments": [
                    {
                        "identifier": item.identifier,
                        "source_text": item.source_text,
                        "translated_text": item.translated_text,
                    }
                    for item in result.segments
                ],
                "source_language": result.source_language,
                "target_language": result.target_language,
                "provider": result.provider,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO translation_cache (
                    cache_key, result_json, created_at
                ) VALUES (?, ?, ?)
                """,
                (key, payload, _now_iso()),
            )
            connection.execute(
                """
                DELETE FROM translation_cache
                WHERE cache_key IN (
                    SELECT cache_key
                    FROM translation_cache
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_cache_rows,),
            )
            connection.commit()
        finally:
            connection.close()


def _detection_text(segments: tuple[TranslationSegment, ...]) -> str:
    return "\n".join(item.text for item in segments)[:4_000]


def _batch_cache_key(
    segments: tuple[TranslationSegment, ...],
    source: str,
    target: str,
) -> str:
    encoded = json.dumps(
        {
            "version": 1,
            "source": source,
            "target": target,
            "segments": [
                {"identifier": item.identifier, "text": item.text}
                for item in segments
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _workspace_key(value: str | None) -> str:
    return value.strip() if value is not None else ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalized_language(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("_", "-")
    if not normalized:
        return None
    if len(normalized) > 35 or not all(
        character.isalnum() or character == "-"
        for character in normalized
    ):
        raise UserError("translation.language_invalid")
    return normalized
