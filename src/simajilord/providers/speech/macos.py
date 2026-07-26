"""Offline speech synthesis through the macOS built-in `say` executable."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from simajilord.core.errors import ProviderError


class MacOSSayProvider:
    def __init__(self, voice: str, *, timeout_seconds: float = 30.0) -> None:
        # Provider construction must remain platform-neutral so the capability
        # graph can be validated on Linux. Availability is checked only when
        # speech is actually requested.
        self.executable = shutil.which("say")
        self.voice = voice
        self.timeout_seconds = timeout_seconds

    async def synthesize(self, text: str, destination: Path) -> None:
        executable = self.executable
        if executable is None:
            raise ProviderError("The macOS `say` executable is unavailable.")
        process = await asyncio.create_subprocess_exec(
            executable,
            "-v",
            self.voice,
            "-o",
            str(destination),
            "--",
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise ProviderError("Local speech synthesis timed out.") from exc
        if process.returncode != 0 or not destination.is_file():
            detail = stderr.decode(errors="replace").strip()[:300]
            raise ProviderError(f"Local speech synthesis failed: {detail or 'unknown error'}")

    async def close(self) -> None:
        """The macOS provider does not own a persistent process."""
