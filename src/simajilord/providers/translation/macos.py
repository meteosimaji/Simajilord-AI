"""macOS Natural Language and Translation framework provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from simajilord.services.translation import (
    TranslationDetection,
    TranslationHypothesis,
    TranslationLanguage,
    TranslationProviderError,
    TranslationResult,
)


class MacOSTranslationProvider:
    """Invoke a small Swift JSON-Lines helper with no cloud translation API."""

    def __init__(
        self,
        package_path: Path,
        *,
        executable_path: Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._package_path = package_path.resolve()
        self._executable_path = executable_path.resolve() if executable_path else None
        self._timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "apple-translation"

    def _command(self) -> tuple[str, ...]:
        candidates = (
            self._executable_path,
            self._package_path / ".build" / "release" / "TranslationHelper",
            self._package_path
            / ".build"
            / "arm64-apple-macosx"
            / "release"
            / "TranslationHelper",
        )
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return (str(candidate),)
        if not (self._package_path / "Package.swift").is_file():
            raise TranslationProviderError(
                "translation.helper_missing",
                f"Swift package is missing: {self._package_path}",
                fallback_allowed=True,
            )
        return (
            "swift",
            "run",
            "--package-path",
            str(self._package_path),
            "--configuration",
            "release",
            "TranslationHelper",
        )

    async def _request(self, payload: dict[str, object]) -> dict[str, Any]:
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
        input_data = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise TranslationProviderError(
                "translation.timeout",
                "The macOS translation helper timed out.",
                fallback_allowed=True,
            ) from exc
        if process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()[-1_000:]
            raise TranslationProviderError(
                "translation.helper_failed",
                detail,
                fallback_allowed=True,
            )
        line = next(
            (item for item in stdout.decode(errors="replace").splitlines() if item.strip()),
            "",
        )
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

    async def close(self) -> None:
        return None


def _object_list(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    output: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            output.append({str(key): entry for key, entry in item.items()})
    return tuple(output)
