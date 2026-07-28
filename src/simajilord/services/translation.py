"""Provider-neutral language detection and on-device translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from simajilord.core.errors import ProviderError, UserError


@dataclass(frozen=True, slots=True)
class TranslationHypothesis:
    """One language-detection candidate."""

    code: str
    confidence: float


@dataclass(frozen=True, slots=True)
class TranslationDetection:
    """Detected source language and bounded alternatives."""

    language: str
    confidence: float
    hypotheses: tuple[TranslationHypothesis, ...]


@dataclass(frozen=True, slots=True)
class TranslationLanguage:
    """A provider language suitable for Discord autocomplete."""

    code: str
    english_name: str
    native_name: str
    availability: str


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """One completed translation."""

    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    provider: str


class TranslationProvider(Protocol):
    """Replaceable local translation backend."""

    @property
    def name(self) -> str:
        """Stable provider name shown in diagnostics."""

    async def detect_language(self, text: str) -> TranslationDetection:
        """Detect the dominant source language."""

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        """Return target languages and pair availability."""

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationResult:
        """Translate text without a cloud model."""

    async def close(self) -> None:
        """Release provider resources."""


class TranslationProviderError(ProviderError):
    """A stable provider failure, optionally eligible for fallback."""

    def __init__(
        self,
        code: str,
        technical_detail: str = "",
        *,
        fallback_allowed: bool,
    ) -> None:
        super().__init__(technical_detail or code)
        self.code = code
        self.technical_detail = technical_detail
        self.fallback_allowed = fallback_allowed


class TranslationService:
    """Validate requests and select the first capable offline provider."""

    def __init__(
        self,
        providers: tuple[TranslationProvider, ...],
        *,
        max_characters: int = 8_000,
    ) -> None:
        if max_characters < 1:
            raise ValueError("max_characters must be positive")
        self._providers = providers
        self._max_characters = max_characters

    @property
    def available(self) -> bool:
        return bool(self._providers)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(provider.name for provider in self._providers)

    def _validated_text(self, text: str) -> str:
        value = text.strip()
        if not value:
            raise UserError("translation.text_required")
        if len(value) > self._max_characters:
            raise UserError(
                "translation.text_too_long",
                maximum=self._max_characters,
            )
        return value

    async def detect_language(self, text: str) -> TranslationDetection:
        value = self._validated_text(text)
        last_error: TranslationProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.detect_language(value)
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        source = _normalized_language(source_language)
        last_error: TranslationProviderError | None = None
        merged: dict[str, TranslationLanguage] = {}
        for provider in self._providers:
            try:
                for language in await provider.supported_targets(source):
                    merged.setdefault(language.code, language)
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if merged:
            return tuple(
                sorted(
                    merged.values(),
                    key=lambda item: (item.english_name.casefold(), item.code),
                )
            )
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None = None,
        target_language: str,
    ) -> TranslationResult:
        value = self._validated_text(text)
        target = _normalized_language(target_language)
        if target is None:
            raise UserError("translation.target_required")
        source = _normalized_language(source_language)
        if source is None:
            source = (await self.detect_language(value)).language
        if source.casefold() == target.casefold():
            raise UserError("translation.same_language")
        last_error: TranslationProviderError | None = None
        for provider in self._providers:
            try:
                return await provider.translate(
                    value,
                    source_language=source,
                    target_language=target,
                )
            except TranslationProviderError as exc:
                last_error = exc
                if not exc.fallback_allowed:
                    break
        if last_error is not None:
            raise UserError(last_error.code) from last_error
        raise UserError("translation.unavailable")

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()


def _normalized_language(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().replace("_", "-")
    if not normalized:
        return None
    if len(normalized) > 35 or not all(
        character.isalnum() or character == "-"
        for character in normalized
    ):
        raise UserError("translation.language_invalid")
    return normalized
