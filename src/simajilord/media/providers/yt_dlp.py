"""Platform provider boundary around the vendored yt-dlp snapshot."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from simajilord.core.errors import MediaError, UserError
from simajilord.domain.audio import AudioItem
from simajilord.domain.media import DownloadArtifact, DownloadFormat, MediaCandidate
from simajilord.media.security import (
    normalize_media_query,
    normalize_media_reference,
    validate_media_url,
    validate_public_media_url,
)

# The vendored source is trusted; unreviewed discovery from user plugin folders is not.
os.environ.setdefault("YTDLP_NO_PLUGINS", "1")
import yt_dlp  # type: ignore[import-untyped]

log = logging.getLogger(__name__)
_ALLOWED_EXTRACTORS = ["default", "-generic"]


class YtDlpProvider:
    """Resolve streams in-process and downloads in a cancellable subprocess."""

    def __init__(
        self,
        *,
        cookie_file: Path | None,
        download_timeout_seconds: float,
    ) -> None:
        self.cookie_file = cookie_file
        self.download_timeout_seconds = download_timeout_seconds

    async def resolve_audio(self, reference: str) -> AudioItem:
        normalized = normalize_media_reference(reference)
        if normalized.startswith("https://"):
            normalized = await validate_public_media_url(normalized)

        def extract() -> AudioItem:
            options: dict[str, Any] = {
                "allowed_extractors": _ALLOWED_EXTRACTORS,
                "format": "bestaudio/best",
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "plugin_dirs": [],
            }
            if self.cookie_file is not None:
                options["cookiefile"] = str(self.cookie_file)
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(normalized, download=False)
                info = _first_entry(info)
                source = str(info.get("url") or "")
                if not source:
                    raise MediaError("unavailable", "The media has no playable audio stream.")
                page_url = _safe_page_url(info.get("webpage_url"), fallback=reference)
                return AudioItem(
                    source=source,
                    title=str(info.get("title") or "Untitled media"),
                    duration_seconds=_safe_duration(info.get("duration")),
                    page_url=page_url,
                    http_headers=_safe_headers(info.get("http_headers")),
                    resolver_reference=page_url,
                    uploader=_optional_text(info.get("uploader") or info.get("channel")),
                    thumbnail_url=_safe_https_url(info.get("thumbnail")),
                )
            except MediaError:
                raise
            except yt_dlp.utils.DownloadError as exc:
                raise classify_yt_dlp_error(str(exc)) from exc

        try:
            async with asyncio.timeout(60):
                return await asyncio.to_thread(extract)
        except TimeoutError as exc:
            raise MediaError("timeout", "Media resolution timed out.") from exc

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        normalized = normalize_media_query(query)
        if not 1 <= limit <= 25:
            raise ValueError("limit must be between 1 and 25")

        def extract() -> tuple[MediaCandidate, ...]:
            options: dict[str, Any] = {
                "allowed_extractors": _ALLOWED_EXTRACTORS,
                "extract_flat": True,
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "plugin_dirs": [],
            }
            if self.cookie_file is not None:
                options["cookiefile"] = str(self.cookie_file)
            try:
                with yt_dlp.YoutubeDL(options) as downloader:
                    info = downloader.extract_info(
                        f"ytsearch{limit}:{normalized}",
                        download=False,
                    )
                candidates = _search_candidates(info, limit=limit)
                if not candidates:
                    raise MediaError("unavailable", "No media result was found.")
                return candidates
            except MediaError:
                raise
            except yt_dlp.utils.DownloadError as exc:
                raise classify_yt_dlp_error(str(exc)) from exc

        try:
            async with asyncio.timeout(60):
                return await asyncio.to_thread(extract)
        except TimeoutError as exc:
            raise MediaError("timeout", "Media search timed out.") from exc

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        """Combine bounded YouTube Mix candidate pools without resolving streams."""

        if not 1 <= len(seed_references) <= 8:
            raise ValueError("seed_references must contain between 1 and 8 items")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        seed_ids = tuple(
            dict.fromkeys(
                video_id
                for reference in seed_references
                if (video_id := _youtube_video_id(reference)) is not None
            )
        )
        if not seed_ids:
            raise MediaError(
                "unsupported",
                "A YouTube track is required to start an automatic mix.",
            )
        per_seed_limit = min(30, max(10, (limit * 2 + len(seed_ids) - 1) // len(seed_ids)))

        async def extract_seed(video_id: str) -> tuple[MediaCandidate, ...]:
            def extract() -> tuple[MediaCandidate, ...]:
                options: dict[str, Any] = {
                    "allowed_extractors": _ALLOWED_EXTRACTORS,
                    "extract_flat": "in_playlist",
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "playlistend": per_seed_limit,
                    "plugin_dirs": [],
                }
                if self.cookie_file is not None:
                    options["cookiefile"] = str(self.cookie_file)
                mix_url = (
                    "https://www.youtube.com/watch?"
                    + urlencode({"v": video_id, "list": f"RD{video_id}"})
                )
                try:
                    with yt_dlp.YoutubeDL(options) as downloader:
                        info = downloader.extract_info(mix_url, download=False)
                    return _search_candidates(info, limit=per_seed_limit)
                except yt_dlp.utils.DownloadError as exc:
                    raise classify_yt_dlp_error(str(exc)) from exc

            return await asyncio.to_thread(extract)

        try:
            async with asyncio.timeout(60):
                pools = await asyncio.gather(*(extract_seed(seed) for seed in seed_ids))
        except TimeoutError as exc:
            raise MediaError("timeout", "YouTube Mix lookup timed out.") from exc

        excluded = {
            f"https://www.youtube.com/watch?v={video_id}" for video_id in seed_ids
        }
        merged: list[MediaCandidate] = []
        seen = set(excluded)
        iterators: list[Iterator[MediaCandidate]] = [iter(pool) for pool in pools]
        while iterators and len(merged) < limit:
            remaining: list[Iterator[MediaCandidate]] = []
            for iterator in iterators:
                candidate = next(iterator, None)
                while candidate is not None and candidate.reference in seen:
                    candidate = next(iterator, None)
                if candidate is None:
                    continue
                seen.add(candidate.reference)
                merged.append(candidate)
                remaining.append(iterator)
                if len(merged) >= limit:
                    break
            iterators = remaining
        if not merged:
            raise MediaError("unavailable", "YouTube Mix returned no new tracks.")
        return tuple(merged)

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact:
        source_url = await validate_public_media_url(url)
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-plugin-dirs",
            "--use-extractors",
            "default,-generic",
            "--no-playlist",
            "--no-progress",
            "--no-warnings",
            "--restrict-filenames",
            "--max-filesize",
            str(max_bytes),
            "--paths",
            str(destination),
            "--output",
            "%(title).120B_[%(id)s].%(ext)s",
        ]
        if self.cookie_file is not None:
            command.extend(("--cookies", str(self.cookie_file)))
        if media_type is DownloadFormat.AUDIO:
            command.extend(("--extract-audio", "--audio-format", "mp3"))
        else:
            command.extend(("--format", "bv*+ba/b", "--merge-output-format", "mp4"))
        command.append(source_url)

        environment = os.environ.copy()
        environment["YTDLP_NO_PLUGINS"] = "1"
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.download_timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise MediaError("timeout", "The media download timed out.") from exc

        if process.returncode != 0:
            detail = (stderr or stdout).decode(errors="replace")[-2_000:]
            raise classify_yt_dlp_error(detail)

        files = tuple(
            path
            for path in destination.iterdir()
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        )
        if len(files) != 1:
            raise MediaError("unknown", "The media provider produced an unexpected result.")
        artifact_path = files[0].resolve()
        if destination.resolve() not in artifact_path.parents:
            raise MediaError("unsafe_path", "The media provider returned an unsafe path.")
        size = artifact_path.stat().st_size
        if size > max_bytes:
            artifact_path.unlink(missing_ok=True)
            raise MediaError("too_large", "The downloaded file exceeds this server's upload limit.")
        return DownloadArtifact(
            path=artifact_path,
            title=artifact_path.stem,
            media_type=media_type,
            source_url=source_url,
            size_bytes=size,
        )


def _first_entry(info: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not info:
        raise MediaError("unavailable", "No media result was found.")
    entries = info.get("entries")
    if entries is not None:
        first = next((entry for entry in entries if entry), None)
        if not isinstance(first, Mapping):
            raise MediaError("unavailable", "No media result was found.")
        return first
    return info


def _search_candidates(
    info: Mapping[str, Any] | None,
    *,
    limit: int,
) -> tuple[MediaCandidate, ...]:
    if not info:
        return ()
    raw_entries = info.get("entries")
    if raw_entries is None:
        raw_entries = (info,)
    candidates: list[MediaCandidate] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        reference = _candidate_reference(raw_entry)
        if reference is None or reference in seen:
            continue
        seen.add(reference)
        candidates.append(
            MediaCandidate(
                reference=reference,
                title=str(raw_entry.get("title") or "Untitled media"),
                duration_seconds=_safe_duration(raw_entry.get("duration")),
                uploader=_optional_text(
                    raw_entry.get("uploader") or raw_entry.get("channel")
                ),
                thumbnail_url=_safe_https_url(raw_entry.get("thumbnail")),
            )
        )
        if len(candidates) >= limit:
            break
    return tuple(candidates)


def _candidate_reference(info: Mapping[str, Any]) -> str | None:
    for key in ("webpage_url", "original_url", "url"):
        value = info.get(key)
        if isinstance(value, str) and value.startswith("https://"):
            try:
                return validate_media_url(value)
            except UserError:
                continue
    video_id = info.get("id")
    if not isinstance(video_id, str) or not video_id:
        return None
    reference = "https://www.youtube.com/watch?" + urlencode({"v": video_id})
    try:
        return validate_media_url(reference)
    except UserError:
        return None


def _youtube_video_id(reference: str) -> str | None:
    """Return a conservative video id from a canonical YouTube page URL."""

    try:
        parsed = urlsplit(validate_media_url(reference))
    except (UserError, ValueError):
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path != "/watch":
            return None
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    else:
        return None
    if not candidate or len(candidate) > 32:
        return None
    return (
        candidate
        if all(character.isalnum() or character in "_-" for character in candidate)
        else None
    )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip()
    return normalized or None


def _safe_duration(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return max(0.0, float(value))
    except (OverflowError, TypeError, ValueError):
        return 0.0


def _safe_https_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2_048:
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            return None
    except ValueError:
        return None
    return value


def _safe_page_url(value: object, *, fallback: str) -> str:
    if isinstance(value, str):
        try:
            return validate_media_url(value)
        except UserError:
            pass
    try:
        return validate_media_url(fallback)
    except UserError as exc:
        raise MediaError("unsafe_path", "The media provider returned an unsafe page URL.") from exc


def _safe_headers(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    safe: dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text.lower() in {"cookie", "authorization"}:
            continue
        safe[key_text] = str(item).replace("\r", " ").replace("\n", " ")
    return safe or None


def classify_yt_dlp_error(detail: str) -> MediaError:
    """Collapse unstable provider messages into a stable public taxonomy."""

    lowered = detail.lower()
    if any(term in lowered for term in ("cookie", "sign in", "login required")):
        return MediaError(
            "cookie_required",
            "This media requires authentication. Configure a private cookie file on the host.",
        )
    if "geo" in lowered and any(term in lowered for term in ("restrict", "block", "country")):
        return MediaError("geo_restricted", "This media is unavailable in the host region.")
    if any(term in lowered for term in ("unsupported url", "no suitable extractor")):
        return MediaError("unsupported", "This media URL is not supported.")
    if any(term in lowered for term in ("too many requests", "http error 429", "rate limit")):
        return MediaError("rate_limited", "The media service is rate-limiting requests.")
    if "larger than max-filesize" in lowered or "file is larger" in lowered:
        return MediaError("too_large", "The media exceeds this server's upload limit.")
    if any(
        term in lowered
        for term in ("private video", "video unavailable", "not available", "removed")
    ):
        return MediaError("unavailable", "This media is unavailable or private.")
    log.warning("Unclassified media provider failure: %s", detail[-500:])
    return MediaError("unknown", "The media provider could not complete the request.")
