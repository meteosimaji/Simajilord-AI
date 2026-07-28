"""Reusable synthetic-media analysis with persistent cache and quota guards."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from simajilord.core.errors import UserError
from simajilord.domain.moderation import (
    SyntheticMediaAnalysis,
    SyntheticMediaModality,
    SyntheticMediaProviderResult,
    SyntheticMediaVerdict,
)
from simajilord.providers.moderation import SyntheticMediaProvider

_SUPPORTED_MEDIA_TYPES = {
    ".avi": "video/x-msvideo",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".wmv": "video/x-ms-wmv",
}


@dataclass(frozen=True, slots=True)
class ModerationStatus:
    configured: bool
    provider: str
    model: str
    quota_used: int
    quota_remaining: int
    quota_limit: int
    quota_reset_at_epoch: int


class ModerationStore:
    """SQLite authority for successful results and paid-request reservations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def cached(
        self,
        *,
        sha256: str,
        model: str,
        threshold: float,
    ) -> SyntheticMediaProviderResult | None:
        async with self._lock:
            return await asyncio.to_thread(
                self._cached,
                sha256,
                model,
                _threshold_key(threshold),
            )

    async def save(
        self,
        *,
        sha256: str,
        result: SyntheticMediaProviderResult,
    ) -> None:
        payload = json.dumps(
            {
                **asdict(result),
                "verdict": result.verdict.value,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        async with self._lock:
            await asyncio.to_thread(
                self._save,
                sha256,
                result.model,
                _threshold_key(result.threshold),
                payload,
            )

    async def reserve(self, *, provider: str, limit: int) -> int:
        """Atomically reserve one paid attempt and return today's used count."""

        day = datetime.now(UTC).date().isoformat()
        async with self._lock:
            return await asyncio.to_thread(self._reserve, provider, day, limit)

    async def used(self, *, provider: str) -> int:
        day = datetime.now(UTC).date().isoformat()
        async with self._lock:
            return await asyncio.to_thread(self._used, provider, day)

    async def prune(self, *, before: datetime) -> tuple[int, int]:
        """Remove expired cache entries and old daily quota rows."""

        if before.tzinfo is None:
            raise ValueError("Retention cutoffs must be timezone-aware.")
        cutoff = before.astimezone(UTC)
        async with self._lock:
            return await asyncio.to_thread(
                self._prune,
                cutoff.isoformat(),
                cutoff.date().isoformat(),
            )

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS synthetic_media_detections (
                    sha256 TEXT NOT NULL,
                    model TEXT NOT NULL,
                    threshold TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (sha256, model, threshold)
                );
                CREATE TABLE IF NOT EXISTS moderation_quota (
                    provider TEXT NOT NULL,
                    day_utc TEXT NOT NULL,
                    count INTEGER NOT NULL CHECK (count >= 0),
                    PRIMARY KEY (provider, day_utc)
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

    def _cached(
        self,
        sha256: str,
        model: str,
        threshold: str,
    ) -> SyntheticMediaProviderResult | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT result_json
                FROM synthetic_media_detections
                WHERE sha256 = ? AND model = ? AND threshold = ?
                """,
                (sha256, model, threshold),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return _decode_result(str(row[0]))

    def _save(
        self,
        sha256: str,
        model: str,
        threshold: str,
        payload: str,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO synthetic_media_detections (
                    sha256, model, threshold, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (sha256, model, threshold)
                DO UPDATE SET result_json = excluded.result_json,
                              created_at = excluded.created_at
                """,
                (sha256, model, threshold, payload, datetime.now(UTC).isoformat()),
            )
            connection.commit()
        finally:
            connection.close()

    def _reserve(self, provider: str, day: str, limit: int) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT count FROM moderation_quota WHERE provider = ? AND day_utc = ?",
                (provider, day),
            ).fetchone()
            used = int(row[0]) if row is not None else 0
            if used >= limit:
                connection.rollback()
                raise UserError("moderation.daily_limit_reached")
            next_used = used + 1
            connection.execute(
                """
                INSERT INTO moderation_quota (provider, day_utc, count)
                VALUES (?, ?, ?)
                ON CONFLICT (provider, day_utc)
                DO UPDATE SET count = excluded.count
                """,
                (provider, day, next_used),
            )
            connection.commit()
            return next_used
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _used(self, provider: str, day: str) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT count FROM moderation_quota WHERE provider = ? AND day_utc = ?",
                (provider, day),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def _prune(self, cutoff: str, cutoff_day: str) -> tuple[int, int]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            detections = connection.execute(
                "DELETE FROM synthetic_media_detections WHERE created_at < ?",
                (cutoff,),
            ).rowcount
            quotas = connection.execute(
                "DELETE FROM moderation_quota WHERE day_utc < ?",
                (cutoff_day,),
            ).rowcount
            connection.commit()
            return detections, quotas
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class ModerationService:
    """Provider-neutral entry point for image and video authenticity signals."""

    def __init__(
        self,
        *,
        provider: SyntheticMediaProvider | None,
        store: ModerationStore,
        daily_limit: int,
        max_media_bytes: int,
        threshold: float,
        provider_name: str = "hive",
        model_name: str = "hive/ai-generated-and-deepfake-content-detection",
    ) -> None:
        self.provider = provider
        self.store = store
        self.daily_limit = daily_limit
        self.max_media_bytes = max_media_bytes
        self.threshold = threshold
        self.provider_name = provider.name if provider is not None else provider_name
        self.model_name = provider.model if provider is not None else model_name

    async def analyze(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str | None,
    ) -> SyntheticMediaAnalysis:
        normalized_name, normalized_type = _validate_media(
            content=content,
            filename=filename,
            content_type=content_type,
            maximum=self.max_media_bytes,
        )
        import hashlib

        sha256 = hashlib.sha256(content).hexdigest()
        cached = await self.store.cached(
            sha256=sha256,
            model=self.model_name,
            threshold=self.threshold,
        )
        if cached is not None:
            used = await self.store.used(provider=self.provider_name)
            return _analysis(
                cached,
                sha256=sha256,
                filename=normalized_name,
                content_type=normalized_type,
                cached=True,
                quota_used=used,
                quota_limit=self.daily_limit,
            )
        provider = self.provider
        if provider is None:
            raise UserError("moderation.not_configured")
        used = await self.store.reserve(
            provider=self.provider_name,
            limit=self.daily_limit,
        )
        result = await provider.analyze(
            content=content,
            filename=normalized_name,
            content_type=normalized_type,
            threshold=self.threshold,
        )
        await self.store.save(sha256=sha256, result=result)
        return _analysis(
            result,
            sha256=sha256,
            filename=normalized_name,
            content_type=normalized_type,
            cached=False,
            quota_used=used,
            quota_limit=self.daily_limit,
        )

    async def status(self) -> ModerationStatus:
        used = await self.store.used(provider=self.provider_name)
        return ModerationStatus(
            configured=self.provider is not None,
            provider=self.provider_name,
            model=self.model_name,
            quota_used=used,
            quota_remaining=max(0, self.daily_limit - used),
            quota_limit=self.daily_limit,
            quota_reset_at_epoch=_next_utc_midnight_epoch(),
        )

    async def close(self) -> None:
        if self.provider is not None:
            await self.provider.close()


def _analysis(
    result: SyntheticMediaProviderResult,
    *,
    sha256: str,
    filename: str,
    content_type: str,
    cached: bool,
    quota_used: int,
    quota_limit: int,
) -> SyntheticMediaAnalysis:
    return SyntheticMediaAnalysis(
        sha256=sha256,
        filename=filename,
        content_type=content_type,
        modality=result.modality,
        ai_generated_score=result.ai_generated_score,
        not_ai_generated_score=result.not_ai_generated_score,
        deepfake_score=result.deepfake_score,
        deepfake_likely=result.deepfake_likely,
        sample_count=result.sample_count,
        model=result.model,
        threshold=result.threshold,
        top_source=result.top_source,
        top_source_score=result.top_source_score,
        verdict=result.verdict,
        version=result.version,
        cached=cached,
        quota_used=quota_used,
        quota_remaining=max(0, quota_limit - quota_used),
        quota_limit=quota_limit,
        quota_reset_at_epoch=_next_utc_midnight_epoch(),
    )


def _validate_media(
    *,
    content: bytes,
    filename: str,
    content_type: str | None,
    maximum: int,
) -> tuple[str, str]:
    if not content:
        raise UserError("moderation.media_empty")
    if len(content) > maximum:
        raise UserError("moderation.media_too_large")
    normalized_name = filename.strip()
    if (
        not normalized_name
        or len(normalized_name) > 200
        or "/" in normalized_name
        or "\\" in normalized_name
        or "\x00" in normalized_name
    ):
        raise UserError("moderation.filename_invalid")
    expected_type = _SUPPORTED_MEDIA_TYPES.get(Path(normalized_name).suffix.lower())
    if expected_type is None:
        raise UserError("moderation.media_type_unsupported")
    normalized_type = (content_type or "").partition(";")[0].strip().lower()
    if normalized_type in {"", "application/octet-stream"}:
        normalized_type = expected_type
    expected_family = expected_type.partition("/")[0]
    if normalized_type.partition("/")[0] != expected_family:
        raise UserError("moderation.media_type_unsupported")
    return normalized_name, normalized_type


def _decode_result(payload: str) -> SyntheticMediaProviderResult:
    raw = cast(dict[str, object], json.loads(payload))
    return SyntheticMediaProviderResult(
        modality=SyntheticMediaModality(str(raw["modality"])),
        ai_generated_score=_score(raw["ai_generated_score"]),
        not_ai_generated_score=_score(raw["not_ai_generated_score"]),
        deepfake_score=_score(raw["deepfake_score"]),
        deepfake_likely=bool(raw["deepfake_likely"]),
        sample_count=max(1, int(cast(int, raw["sample_count"]))),
        model=str(raw["model"]),
        threshold=_score(raw["threshold"]),
        top_source=(
            str(raw["top_source"]) if isinstance(raw.get("top_source"), str) else None
        ),
        top_source_score=_score(raw["top_source_score"]),
        verdict=SyntheticMediaVerdict(str(raw["verdict"])),
        version=str(raw["version"]) if isinstance(raw.get("version"), str) else None,
    )


def _score(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("Cached moderation score is invalid.")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("Cached moderation score is outside the expected range.")
    return score




def _threshold_key(value: float) -> str:
    return format(value, ".12g")


def _next_utc_midnight_epoch() -> int:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).date()
    return int(datetime.combine(tomorrow, datetime.min.time(), tzinfo=UTC).timestamp())
