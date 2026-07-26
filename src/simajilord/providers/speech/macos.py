"""Offline speech synthesis through the macOS built-in `say` executable."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from simajilord.core.errors import ProviderError


class MacOSSayProvider:
    def __init__(self, voice: str, *, timeout_seconds: float = 30.0) -> None:
        executable = shutil.which("say")
        if executable is None:
            raise ProviderError("The macOS `say` executable is unavailable.")
        self.executable = executable
        self.voice = voice
        self.timeout_seconds = timeout_seconds

    async def synthesize(self, text: str, destination: Path) -> None:
        process = await asyncio.create_subprocess_exec(
            self.executable,
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
