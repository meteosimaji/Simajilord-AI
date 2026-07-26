"""Fast audio-path diagnostics without connecting to a Discord account."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

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
    """Validate local Opus support and optionally the complete media provider path."""

    await verify_ffmpeg_opus()
    results = ("FFmpeg Opus encoder: OK",)
    if reference is None:
        return results

    provider = YtDlpProvider(
        cookie_file=cookie_file,
        download_timeout_seconds=180.0,
    )
    item = await provider.resolve_audio(reference)
    source = build_discord_audio_source(item)
    try:
        packets = await asyncio.to_thread(lambda: tuple(source.read() for _ in range(5)))
        if not source.is_opus() or not all(packets):
            raise RuntimeError("FFmpeg did not produce valid Opus packets.")
    finally:
        source.cleanup()
    return (
        *results,
        f"Media resolver: OK ({item.title})",
        f"Opus packets: OK ({len(packets)} packets, {sum(map(len, packets))} bytes)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the Simajilord audio path without Discord voice or a bot token."
    )
    parser.add_argument(
        "reference",
        nargs="?",
        help="Optional YouTube/TikTok URL or supported search query.",
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
