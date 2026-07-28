"""Fast audio-path diagnostics without connecting to a Discord account."""

from __future__ import annotations

import argparse
import asyncio
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from yt_dlp.version import RELEASE_GIT_HEAD  # type: ignore[import-untyped]
from yt_dlp.version import __version__ as yt_dlp_version

from simajilord.integrations.discord.audio import (
    build_discord_audio_source,
    verify_ffmpeg_opus,
)
from simajilord.media.providers import YtDlpProvider


async def run_audio_doctor(
    reference: str | None,
    *,
    cookie_file: Path | None = None,
) -> tuple[str, ...]:
    """Validate the local toolchain and optionally exercise YouTube audio and Mix."""

    await verify_ffmpeg_opus()
    results = [
        f"yt-dlp: {yt_dlp_version} ({RELEASE_GIT_HEAD[:12]})",
        f"yt-dlp-ejs: {_package_version('yt-dlp-ejs')}",
        f"FFmpeg: {await _executable_version('ffmpeg', '-version')}",
        f"FFprobe: {await _executable_version('ffprobe', '-version')}",
        f"Deno: {await _executable_version('deno', '--version')}",
        f"Node.js: {await _executable_version('node', '--version')}",
        (
            f"Cookie file: configured ({cookie_file.name})"
            if cookie_file is not None
            else "Cookie file: not configured"
        ),
        "FFmpeg Opus encoder: OK",
    ]
    if reference is None:
        return tuple(results)

    provider = YtDlpProvider(
        cookie_file=cookie_file,
        download_timeout_seconds=180.0,
    )
    selected_reference = reference
    if "://" not in reference:
        candidates = await provider.search_audio(reference, limit=1)
        selected_reference = candidates[0].reference
    item = await provider.resolve_audio(selected_reference)
    source = build_discord_audio_source(item)
    try:
        packets = await asyncio.to_thread(lambda: tuple(source.read() for _ in range(5)))
        if not source.is_opus() or not all(packets):
            raise RuntimeError("FFmpeg did not produce valid Opus packets.")
    finally:
        source.cleanup()
    results.extend(
        (
            f"Media resolver: OK ({item.title})",
            f"Opus packets: OK ({len(packets)} packets, {sum(map(len, packets))} bytes)",
        )
    )
    if "youtube.com/watch?" in item.page_url or "youtu.be/" in item.page_url:
        mix = await provider.mix_audio((item.page_url,), limit=3)
        results.append(f"YouTube Mix: OK ({len(mix)} candidates)")
    else:
        results.append("YouTube Mix: skipped (resolved item is not a YouTube video)")
    return tuple(results)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


async def _executable_version(executable: str, *arguments: str) -> str:
    path = shutil.which(executable)
    if path is None:
        return "MISSING"
    process = await asyncio.create_subprocess_exec(
        path,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
    if process.returncode != 0:
        return f"FAILED ({process.returncode})"
    first_line = stdout.decode(errors="replace").splitlines()
    return first_line[0].strip() if first_line else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the Simajilord audio path without Discord voice or a bot token."
    )
    parser.add_argument(
        "reference",
        nargs="?",
        help="Optional provider-supported public media URL or search query.",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="Optional private Netscape cookie file for the diagnostic only.",
    )
    arguments = parser.parse_args()
    cookie_file = arguments.cookie_file
    if cookie_file is not None:
        cookie_file = cookie_file.expanduser().resolve()
        if not cookie_file.is_file():
            parser.error("--cookie-file must point to an existing file")
        if cookie_file.stat().st_mode & 0o077:
            parser.error("--cookie-file must not be accessible by group or other users")
    for result in asyncio.run(
        run_audio_doctor(arguments.reference, cookie_file=cookie_file)
    ):
        print(result)
