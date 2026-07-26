"""Platform provider boundary around the vendored yt-dlp snapshot."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from simajilord.core.errors import MediaError
from simajilord.domain.audio import AudioItem
from simajilord.domain.media import DownloadArtifact, DownloadFormat
from simajilord.media.security import normalize_media_reference, validate_media_url

# The vendored source is trusted; unreviewed discovery from user plugin folders is not.
os.environ.setdefault("YTDLP_NO_PLUGINS", "1")
import yt_dlp  # type: ignore[import-untyped]

log = logging.getLogger(__name__)


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

        def extract() -> AudioItem:
            options: dict[str, Any] = {
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
                page_url = str(info.get("webpage_url") or reference)
                return AudioItem(
                    source=source,
                    title=str(info.get("title") or "Untitled media"),
                    duration_seconds=float(info.get("duration") or 0),
                    page_url=page_url,
                    http_headers=_safe_headers(info.get("http_headers")),
                    resolver_reference=page_url,
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

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact:
        source_url = validate_media_url(url)
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-plugin-dirs",
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
