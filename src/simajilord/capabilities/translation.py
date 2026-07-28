"""Typed translation capabilities shared by Discord and future transports."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.services.translation import TranslationService


@dataclass(frozen=True, slots=True)
class TranslationDetectRequest:
    text: str


@dataclass(frozen=True, slots=True)
class TranslationDetectResponse:
    language: str
    confidence: float
    hypotheses: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class TranslationLanguagesRequest:
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationLanguageItem:
    code: str
    english_name: str
    native_name: str
    availability: str


@dataclass(frozen=True, slots=True)
class TranslationLanguagesResponse:
    languages: tuple[TranslationLanguageItem, ...]


@dataclass(frozen=True, slots=True)
class TranslationTranslateRequest:
    text: str
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationTranslateResponse:
    original: str
    translation: str
    source_language: str
    target_language: str
    provider: str


def build_translation_endpoints(
    service: TranslationService,
) -> tuple[CapabilityEndpoint, ...]:
    async def detect(
        request: TranslationDetectRequest,
        _context: InvocationContext,
    ) -> TranslationDetectResponse:
        result = await service.detect_language(request.text)
        return TranslationDetectResponse(
            language=result.language,
            confidence=result.confidence,
            hypotheses=tuple(
                (item.code, item.confidence)
                for item in result.hypotheses
            ),
        )

    async def languages(
        request: TranslationLanguagesRequest,
        _context: InvocationContext,
    ) -> TranslationLanguagesResponse:
        result = await service.supported_targets(request.source_language)
        return TranslationLanguagesResponse(
            languages=tuple(
                TranslationLanguageItem(
                    code=item.code,
                    english_name=item.english_name,
                    native_name=item.native_name,
                    availability=item.availability,
                )
                for item in result
            )
        )

    async def translate(
        request: TranslationTranslateRequest,
        _context: InvocationContext,
    ) -> TranslationTranslateResponse:
        result = await service.translate(
            request.text,
            source_language=request.source_language,
            target_language=request.target_language,
        )
        return TranslationTranslateResponse(
            original=result.source_text,
            translation=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            provider=result.provider,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="translation.detect",
                summary="Detect the dominant BCP-47 language of supplied text locally.",
                risk=RiskLevel.READ,
                keywords=("language", "detect", "identify", "translation"),
                expected_errors=(
                    "translation.text_required",
                    "translation.unavailable",
                ),
                timeout_seconds=30,
            ),
            TranslationDetectRequest,
            TranslationDetectResponse,
            detect,
        ),
        endpoint(
            CapabilityDescriptor(
                name="translation.languages",
                summary=(
                    "List locally supported translation targets and installation status."
                ),
                risk=RiskLevel.READ,
                keywords=("language", "supported", "targets", "translation"),
                expected_errors=("translation.unavailable",),
                timeout_seconds=30,
            ),
            TranslationLanguagesRequest,
            TranslationLanguagesResponse,
            languages,
        ),
        endpoint(
            CapabilityDescriptor(
                name="translation.translate",
                summary="Translate supplied text with an offline on-device provider.",
                risk=RiskLevel.READ,
                keywords=("translate", "translation", "language", "local"),
                expected_errors=(
                    "translation.text_required",
                    "translation.target_required",
                    "translation.same_language",
                    "translation.language_invalid",
                    "translation.unavailable",
                ),
                timeout_seconds=60,
            ),
            TranslationTranslateRequest,
            TranslationTranslateResponse,
            translate,
        ),
    )
