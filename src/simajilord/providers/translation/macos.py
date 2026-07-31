"""macOS Natural Language and Translation framework provider."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simajilord.services.translation import (
    TranslatedSegment,
    TranslationBatchResult,
    TranslationDetection,
    TranslationHypothesis,
    TranslationLanguage,
    TranslationProviderError,
    TranslationResult,
    TranslationSegment,
)


@dataclass(frozen=True, slots=True)
class TranslationHelperResolution:
    """One explicit, source-checkout, or unavailable helper resolution."""

    ready: bool
    source: str
    command: tuple[str, ...]
    error_code: str | None
    detail: str


def source_translation_package(runtime_path: Path) -> Path | None:
    """Resolve the Swift package only beside an actual source checkout."""

    candidate = (
        runtime_path.resolve().parents[2]
        / "native"
        / "macos"
        / "TranslationHelper"
    )
    return candidate if (candidate / "Package.swift").is_file() else None


def resolve_translation_helper(
    package_path: Path | None,
    *,
    executable_path: Path | None = None,
) -> TranslationHelperResolution:
    """Resolve one runnable helper without guessing a wheel-adjacent source tree."""

    if executable_path is not None:
        configured = executable_path.expanduser().resolve()
        if not configured.is_file():
            return TranslationHelperResolution(
                ready=False,
                source="configured",
                command=(),
                error_code="translation.helper_missing",
                detail=(
                    "TRANSLATION_HELPER_PATH does not point to an existing file: "
                    f"{configured}"
                ),
            )
        mode = stat.S_IMODE(configured.stat().st_mode)
        if mode & 0o077 or not os.access(configured, os.X_OK):
            return TranslationHelperResolution(
                ready=False,
                source="configured",
                command=(),
                error_code="translation.helper_unavailable",
                detail=(
                    "TRANSLATION_HELPER_PATH must be an owner-executable private file. "
                    f"Run: chmod 700 {configured}"
                ),
            )
        return TranslationHelperResolution(
            ready=True,
            source="configured",
            command=(str(configured),),
            error_code=None,
            detail="Using TRANSLATION_HELPER_PATH.",
        )

    package = package_path.resolve() if package_path is not None else None
    if package is not None and (package / "Package.swift").is_file():
        candidates = (
            package / ".build" / "release" / "TranslationHelper",
            package
            / ".build"
            / "arm64-apple-macosx"
            / "release"
            / "TranslationHelper",
        )
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return TranslationHelperResolution(
                    ready=True,
                    source="source-build",
                    command=(str(candidate),),
                    error_code=None,
                    detail="Using a helper built in the source checkout.",
                )
        if shutil.which("swift") is None:
            return TranslationHelperResolution(
                ready=False,
                source="source-swift",
                command=(),
                error_code="translation.helper_unavailable",
                detail="The source Swift package exists, but the swift executable is unavailable.",
            )
        return TranslationHelperResolution(
            ready=True,
            source="source-swift",
            command=(
                "swift",
                "run",
                "--package-path",
                str(package),
                "--configuration",
                "release",
                "TranslationHelper",
            ),
            error_code=None,
            detail="The helper will be built from the source checkout with swift run.",
        )

    return TranslationHelperResolution(
        ready=False,
        source="installed-wheel",
        command=(),
        error_code="translation.helper_missing",
        detail=(
            "No source Swift package is available. Build TranslationHelper separately, "
            "run chmod 700 on it, and set TRANSLATION_HELPER_PATH."
        ),
    )


class MacOSTranslationProvider:
    """Invoke a small Swift JSON-Lines helper with no cloud translation API."""

    def __init__(
        self,
        package_path: Path | None,
        *,
        executable_path: Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._package_path = package_path.resolve() if package_path is not None else None
        self._executable_path = executable_path.resolve() if executable_path else None
        self._timeout_seconds = timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._process_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_tail = ""

    @property
    def name(self) -> str:
        return "apple-translation"

    def _command(self) -> tuple[str, ...]:
        resolution = resolve_translation_helper(
            self._package_path,
            executable_path=self._executable_path,
        )
        if not resolution.ready:
            raise TranslationProviderError(
                resolution.error_code or "translation.helper_unavailable",
                resolution.detail,
                fallback_allowed=True,
            )
        return resolution.command

    async def _start_process(self) -> asyncio.subprocess.Process:
        async with self._process_lock:
            if self._process is not None and self._process.returncode is None:
                return self._process
            await self._stop_process()
            self._stderr_tail = ""
            try:
                command = self._command()
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (OSError, TranslationProviderError) as exc:
                if isinstance(exc, TranslationProviderError):
                    raise
                raise TranslationProviderError(
                    "translation.helper_unavailable",
                    str(exc),
                    fallback_allowed=True,
                ) from exc
            self._process = process
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(process),
                name="translation-helper-stderr",
            )
            return process

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        try:
            while chunk := await process.stderr.read(1_024):
                self._stderr_tail = (
                    self._stderr_tail + chunk.decode(errors="replace")
                )[-4_000:]
        except (asyncio.CancelledError, OSError):
            return

    async def _stop_process(self) -> None:
        process = self._process
        self._process = None
        stderr_task = self._stderr_task
        self._stderr_task = None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except TimeoutError:
                process.kill()
                await process.wait()
        if stderr_task is not None:
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)

    async def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        async with self._request_lock:
            return await self._locked_request(payload)

    async def _locked_request(self, payload: dict[str, object]) -> dict[str, Any]:
        try:
            process = await self._start_process()
        except TranslationProviderError:
            raise
        if process.stdin is None or process.stdout is None:
            await self._stop_process()
            raise TranslationProviderError(
                "translation.helper_failed",
                "The translation helper has no JSON-Lines streams.",
                fallback_allowed=True,
            )
        input_data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        try:
            process.stdin.write(input_data)
            await process.stdin.drain()
            stdout = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self._timeout_seconds,
            )
        except (BrokenPipeError, ConnectionResetError, TimeoutError) as exc:
            await self._stop_process()
            code = (
                "translation.timeout"
                if isinstance(exc, TimeoutError)
                else "translation.helper_failed"
            )
            raise TranslationProviderError(
                code,
                (
                    "The macOS translation helper timed out."
                    if code == "translation.timeout"
                    else self._stderr_tail or str(exc)
                ),
                fallback_allowed=True,
            ) from exc
        if not stdout:
            detail = self._stderr_tail.strip()[-1_000:]
            await self._stop_process()
            raise TranslationProviderError(
                "translation.helper_failed",
                detail or "The translation helper closed its output stream.",
                fallback_allowed=True,
            )
        line = stdout.decode(errors="replace").strip()
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranslationProviderError(
                "translation.helper_failed",
                f"Invalid helper response: {line[:500]}",
                fallback_allowed=True,
            ) from exc
        if not isinstance(decoded, dict):
            raise TranslationProviderError(
                "translation.helper_failed",
                "Translation helper returned a non-object response.",
                fallback_allowed=True,
            )
        response = {str(key): value for key, value in decoded.items()}
        if response.get("ok") is not True:
            code = str(response.get("error", "translation.helper_failed"))
            fallback_allowed = code in {
                "translation.language_pair_unsupported",
                "translation.helper_unavailable",
                "translation.helper_failed",
            }
            raise TranslationProviderError(
                code,
                str(response.get("detail", code)),
                fallback_allowed=fallback_allowed,
            )
        return response

    async def detect_language(self, text: str) -> TranslationDetection:
        response = await self._request({"operation": "detect", "text": text})
        hypotheses = tuple(
            TranslationHypothesis(
                code=str(item["code"]),
                confidence=float(item["confidence"]),
            )
            for item in _object_list(response.get("hypotheses"))
        )
        return TranslationDetection(
            language=str(response["language"]),
            confidence=float(response.get("confidence", 0.0)),
            hypotheses=hypotheses,
        )

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        response = await self._request(
            {
                "operation": "languages",
                "source_language": source_language,
            }
        )
        return tuple(
            TranslationLanguage(
                code=str(item["code"]),
                english_name=str(item["english_name"]),
                native_name=str(item["native_name"]),
                availability=str(item["availability"]),
            )
            for item in _object_list(response.get("languages"))
        )

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationResult:
        response = await self._request(
            {
                "operation": "translate",
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            }
        )
        return TranslationResult(
            source_text=text,
            translated_text=str(response["translated_text"]),
            source_language=str(response["source_language"]),
            target_language=str(response["target_language"]),
            provider=self.name,
        )

    async def translate_batch(
        self,
        segments: tuple[TranslationSegment, ...],
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationBatchResult:
        response = await self._request(
            {
                "operation": "translate_batch",
                "segments": [
                    {"identifier": item.identifier, "text": item.text}
                    for item in segments
                ],
                "source_language": source_language,
                "target_language": target_language,
            }
        )
        translated = tuple(
            TranslatedSegment(
                identifier=str(item["identifier"]),
                source_text=str(item["source_text"]),
                translated_text=str(item["translated_text"]),
            )
            for item in _object_list(response.get("segments"))
        )
        if len(translated) != len(segments):
            raise TranslationProviderError(
                "translation.helper_failed",
                "The helper returned an incomplete translation batch.",
                fallback_allowed=True,
            )
        return TranslationBatchResult(
            segments=translated,
            source_language=str(response["source_language"]),
            target_language=str(response["target_language"]),
            provider=self.name,
        )

    async def close(self) -> None:
        async with self._request_lock:
            await self._stop_process()


def _object_list(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    output: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            output.append({str(key): entry for key, entry in item.items()})
    return tuple(output)
