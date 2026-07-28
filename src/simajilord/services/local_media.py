"""Persistent, content-addressed media imported from trusted transport adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import time
from urllib.parse import urlsplit

from simajilord.core.errors import MediaError, UserError
from simajilord.domain.audio import AudioItem

LOCAL_MEDIA_SCHEME = "local-media"
_ALLOWED_CONTENT_PREFIXES = ("audio/", "video/")
_SAFE_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


@dataclass(frozen=True, slots=True)
class LocalMediaRecord:
    """Durable metadata for one unique media payload."""

    reference: str
    sha256: str
    path: Path
    original_filename: str
    content_type: str
    duration_seconds: float
    size_bytes: int
    source_jump_url: str | None
    uploaded_by_id: str | None
    uploaded_by_name: str | None
    cover_path: Path | None
    reference_count: int
    last_used_epoch: int


class LocalMediaStore:
    """Validate, deduplicate, persist, resolve, and bound local media files."""

    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int,
        max_cache_bytes: int,
        max_duration_seconds: float,
        audio_state_path: Path | None = None,
    ) -> None:
        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")
        if max_cache_bytes < max_file_bytes:
            raise ValueError("max_cache_bytes must be at least max_file_bytes")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        self.root = root.resolve()
        self.files_dir = self.root / "files"
        self.covers_dir = self.root / "covers"
        self.database_path = self.root / "media.sqlite3"
        self.max_file_bytes = max_file_bytes
        self.max_cache_bytes = max_cache_bytes
        self.max_duration_seconds = max_duration_seconds
        self.audio_state_path = audio_state_path
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.files_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.covers_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._initialize()

    async def import_file(
        self,
        source: Path,
        *,
        original_filename: str,
        content_type: str | None,
        source_jump_url: str | None,
        uploaded_by_id: str | None,
        uploaded_by_name: str | None,
    ) -> LocalMediaRecord:
        """Validate and atomically import one attachment without trusting its suffix."""

        async with self._lock:
            return await asyncio.to_thread(
                self._import_file,
                source,
                original_filename=original_filename,
                content_type=content_type,
                source_jump_url=source_jump_url,
                uploaded_by_id=uploaded_by_id,
                uploaded_by_name=uploaded_by_name,
            )

    async def resolve_audio(self, reference: str) -> AudioItem:
        """Resolve a durable local-media reference to a playable filesystem path."""

        digest = _reference_digest(reference)
        async with self._lock:
            record = await asyncio.to_thread(self._record_for_digest, digest, True)
        if record is None or not record.path.is_file():
            raise MediaError("unavailable", "The imported media is no longer available.")
        return AudioItem(
            source=str(record.path),
            title=record.original_filename,
            page_url=record.source_jump_url or record.reference,
            duration_seconds=record.duration_seconds,
            resolver_reference=record.reference,
            uploader=record.uploaded_by_name,
            request_source="discord_attachment",
        )

    async def record(self, reference: str) -> LocalMediaRecord | None:
        digest = _reference_digest(reference)
        async with self._lock:
            return await asyncio.to_thread(self._record_for_digest, digest, False)

    async def cleanup(self) -> int:
        """Run protected LRU cleanup and return the number of removed objects."""

        async with self._lock:
            return await asyncio.to_thread(self._cleanup, None)

    async def cleanup_expired(self, *, before_epoch: int) -> int:
        """Remove old unreferenced objects while preserving durable audio queues."""

        async with self._lock:
            return await asyncio.to_thread(self._cleanup_expired, before_epoch)

    async def usage_bytes(self) -> int:
        async with self._lock:
            return await asyncio.to_thread(self._usage_bytes)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS local_media (
                    sha256 TEXT PRIMARY KEY,
                    stored_name TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    source_jump_url TEXT,
                    uploaded_by_id TEXT,
                    uploaded_by_name TEXT,
                    cover_name TEXT,
                    reference_count INTEGER NOT NULL DEFAULT 0,
                    created_at_epoch INTEGER NOT NULL,
                    last_used_epoch INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS local_media_lru "
                "ON local_media(reference_count, last_used_epoch)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _import_file(
        self,
        source: Path,
        *,
        original_filename: str,
        content_type: str | None,
        source_jump_url: str | None,
        uploaded_by_id: str | None,
        uploaded_by_name: str | None,
    ) -> LocalMediaRecord:
        source = source.resolve(strict=True)
        size = source.stat().st_size
        if size <= 0:
            raise UserError("local_media.empty")
        if size > self.max_file_bytes:
            raise UserError("local_media.too_large", maximum=self.max_file_bytes)
        normalized_content_type = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized_content_type and not normalized_content_type.startswith(
            _ALLOWED_CONTENT_PREFIXES
        ):
            raise UserError("local_media.content_type_unsupported")

        probe = _probe_media(source)
        duration = probe.duration_seconds
        if duration <= 0:
            raise UserError("local_media.duration_unknown")
        if duration > self.max_duration_seconds:
            raise UserError(
                "local_media.duration_too_long",
                maximum=self.max_duration_seconds,
            )
        if not probe.has_audio:
            raise UserError("local_media.audio_stream_missing")

        digest = _sha256_file(source)
        existing = self._record_for_digest(digest, True)
        if existing is not None and existing.path.is_file():
            return existing

        suffix = _safe_suffix(original_filename, probe.format_name)
        stored_name = f"{digest}{suffix}"
        destination = self.files_dir / stored_name
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        cover_name = _extract_cover(destination, self.covers_dir / f"{digest}.jpg")
        now = int(time())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO local_media (
                    sha256, stored_name, original_filename, content_type,
                    duration_seconds, size_bytes, source_jump_url,
                    uploaded_by_id, uploaded_by_name, cover_name,
                    reference_count, created_at_epoch, last_used_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    digest,
                    stored_name,
                    Path(original_filename).name or "Discord attachment",
                    normalized_content_type or probe.content_type,
                    duration,
                    size,
                    _safe_jump_url(source_jump_url),
                    uploaded_by_id,
                    uploaded_by_name,
                    cover_name,
                    now,
                    now,
                ),
            )
        try:
            self._cleanup(digest)
        except Exception:
            destination.unlink(missing_ok=True)
            if cover_name:
                (self.covers_dir / cover_name).unlink(missing_ok=True)
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM local_media WHERE sha256 = ?",
                    (digest,),
                )
            raise
        record = self._record_for_digest(digest, False)
        if record is None:
            raise MediaError("unavailable", "The imported media could not be indexed.")
        return record

    def _record_for_digest(
        self,
        digest: str,
        touch: bool,
    ) -> LocalMediaRecord | None:
        now = int(time())
        with self._connect() as connection:
            if touch:
                connection.execute(
                    "UPDATE local_media SET last_used_epoch = ? WHERE sha256 = ?",
                    (now, digest),
                )
            row = connection.execute(
                "SELECT * FROM local_media WHERE sha256 = ?",
                (digest,),
            ).fetchone()
        return None if row is None else self._row_to_record(row)

    def _cleanup(self, keep_digest: str | None) -> int:
        protected = self._protected_references()
        removed = 0
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sha256, stored_name, cover_name, size_bytes "
                "FROM local_media ORDER BY last_used_epoch ASC, sha256 ASC"
            ).fetchall()
            total = sum(int(row["size_bytes"]) for row in rows)
            if total <= self.max_cache_bytes:
                self._refresh_reference_counts(connection, protected)
                return 0
            self._refresh_reference_counts(connection, protected)
            for row in rows:
                digest = str(row["sha256"])
                if digest == keep_digest or digest in protected:
                    continue
                reference_count = connection.execute(
                    "SELECT reference_count FROM local_media WHERE sha256 = ?",
                    (digest,),
                ).fetchone()
                if reference_count is not None and int(reference_count[0]) > 0:
                    continue
                (self.files_dir / str(row["stored_name"])).unlink(missing_ok=True)
                cover_name = row["cover_name"]
                if isinstance(cover_name, str) and cover_name:
                    (self.covers_dir / cover_name).unlink(missing_ok=True)
                connection.execute(
                    "DELETE FROM local_media WHERE sha256 = ?",
                    (digest,),
                )
                total -= int(row["size_bytes"])
                removed += 1
                if total <= self.max_cache_bytes:
                    break
        if total > self.max_cache_bytes:
            raise UserError("local_media.cache_full")
        return removed

    def _cleanup_expired(self, before_epoch: int) -> int:
        protected = self._protected_references()
        removed = 0
        with self._connect() as connection:
            self._refresh_reference_counts(connection, protected)
            rows = connection.execute(
                """
                SELECT sha256, stored_name, cover_name
                FROM local_media
                WHERE last_used_epoch < ?
                ORDER BY last_used_epoch, sha256
                """,
                (before_epoch,),
            ).fetchall()
            for row in rows:
                digest = str(row["sha256"])
                if digest in protected:
                    continue
                reference_count = connection.execute(
                    "SELECT reference_count FROM local_media WHERE sha256 = ?",
                    (digest,),
                ).fetchone()
                if reference_count is not None and int(reference_count[0]) > 0:
                    continue
                self._unlink_record_files(row)
                connection.execute(
                    "DELETE FROM local_media WHERE sha256 = ?",
                    (digest,),
                )
                removed += 1
        removed += self._cleanup(None)
        return removed

    def _usage_bytes(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM local_media"
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def _unlink_record_files(self, row: sqlite3.Row) -> None:
        (self.files_dir / str(row["stored_name"])).unlink(missing_ok=True)
        cover_name = row["cover_name"]
        if isinstance(cover_name, str) and cover_name:
            (self.covers_dir / cover_name).unlink(missing_ok=True)

    def _protected_references(self) -> Counter[str]:
        path = self.audio_state_path
        if path is None or not path.is_file():
            return Counter()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Counter()
        protected: Counter[str] = Counter()
        sessions = payload.get("sessions", ()) if isinstance(payload, dict) else ()
        if not isinstance(sessions, list):
            return protected
        for session in sessions:
            if not isinstance(session, dict):
                continue
            items = session.get("items", ())
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                reference = item.get("reference")
                if isinstance(reference, str) and reference.startswith(
                    f"{LOCAL_MEDIA_SCHEME}://"
                ):
                    try:
                        protected[_reference_digest(reference)] += 1
                    except UserError:
                        continue
        return protected

    @staticmethod
    def _refresh_reference_counts(
        connection: sqlite3.Connection,
        protected: Counter[str],
    ) -> None:
        connection.execute("UPDATE local_media SET reference_count = 0")
        connection.executemany(
            "UPDATE local_media SET reference_count = ? WHERE sha256 = ?",
            ((count, digest) for digest, count in protected.items()),
        )

    def _row_to_record(self, row: sqlite3.Row) -> LocalMediaRecord:
        digest = str(row["sha256"])
        cover_name = row["cover_name"]
        return LocalMediaRecord(
            reference=f"{LOCAL_MEDIA_SCHEME}://{digest}",
            sha256=digest,
            path=(self.files_dir / str(row["stored_name"])).resolve(),
            original_filename=str(row["original_filename"]),
            content_type=str(row["content_type"]),
            duration_seconds=float(row["duration_seconds"]),
            size_bytes=int(row["size_bytes"]),
            source_jump_url=(
                str(row["source_jump_url"]) if row["source_jump_url"] else None
            ),
            uploaded_by_id=(
                str(row["uploaded_by_id"]) if row["uploaded_by_id"] else None
            ),
            uploaded_by_name=(
                str(row["uploaded_by_name"]) if row["uploaded_by_name"] else None
            ),
            cover_path=(
                (self.covers_dir / str(cover_name)).resolve()
                if isinstance(cover_name, str) and cover_name
                else None
            ),
            reference_count=int(row["reference_count"]),
            last_used_epoch=int(row["last_used_epoch"]),
        )


@dataclass(frozen=True, slots=True)
class _ProbeResult:
    duration_seconds: float
    format_name: str
    content_type: str
    has_audio: bool


def _probe_media(path: Path) -> _ProbeResult:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise MediaError("runtime", "ffprobe is required to import local media.")
    process = subprocess.run(
        (
            executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name:stream=codec_type",
            "-of",
            "json",
            str(path),
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise UserError("local_media.invalid_media")
    try:
        payload = json.loads(process.stdout)
        format_data = payload["format"]
        duration = float(format_data["duration"])
        format_name = str(format_data.get("format_name") or "")
        streams = payload.get("streams", ())
    except (KeyError, TypeError, ValueError) as exc:
        raise UserError("local_media.invalid_media") from exc
    has_audio = any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio"
        for stream in streams
    )
    content_type = (
        "video/" + format_name.split(",", 1)[0]
        if any(
            isinstance(stream, dict) and stream.get("codec_type") == "video"
            for stream in streams
        )
        else "audio/" + format_name.split(",", 1)[0]
    )
    return _ProbeResult(duration, format_name, content_type, has_audio)


def _extract_cover(source: Path, destination: Path) -> str | None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return None
    temporary = destination.with_suffix(".tmp.jpg")
    process = subprocess.run(
        (
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            "scale='min(640,iw)':-2",
            str(temporary),
        ),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if process.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        return None
    os.chmod(temporary, 0o600)
    temporary.replace(destination)
    return destination.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_digest(reference: str) -> str:
    parsed = urlsplit(reference)
    digest = parsed.netloc or parsed.path.lstrip("/")
    if (
        parsed.scheme != LOCAL_MEDIA_SCHEME
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise UserError("local_media.reference_invalid")
    return digest


def _safe_suffix(filename: str, format_name: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in _SAFE_SUFFIXES:
        return suffix
    preferred = format_name.split(",", 1)[0].lower()
    mapped = {
        "matroska": ".mkv",
        "mov": ".mp4",
        "mp4": ".mp4",
        "mpeg": ".mp3",
        "ogg": ".ogg",
        "wav": ".wav",
        "webm": ".webm",
    }.get(preferred)
    return mapped or ".media"


def _safe_jump_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme == "https" and parsed.hostname in {"discord.com", "www.discord.com"}:
        return value
    return None
