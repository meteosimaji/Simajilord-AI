"""Typed translation capabilities shared by Discord and future transports."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.services.translation import TranslationSegment, TranslationService

log = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class TranslationSegmentItem:
    identifier: str
    text: str


@dataclass(frozen=True, slots=True)
class TranslatedSegmentItem:
    identifier: str
    original: str
    translation: str


@dataclass(frozen=True, slots=True)
class TranslationBatchRequest:
    segments: tuple[TranslationSegmentItem, ...]
    target_language: str
    source_language: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationBatchResponse:
    segments: tuple[TranslatedSegmentItem, ...]
    source_language: str
    target_language: str
    cached: bool


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
        log.info(
            "Translation completed provider=%s source=%s target=%s characters=%d",
            result.provider,
            result.source_language,
            result.target_language,
            len(result.source_text),
        )
        return TranslationTranslateResponse(
            original=result.source_text,
            translation=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
        )

    async def translate_batch(
        request: TranslationBatchRequest,
        _context: InvocationContext,
    ) -> TranslationBatchResponse:
        result = await service.translate_batch(
            tuple(
                TranslationSegment(
                    identifier=item.identifier,
                    text=item.text,
                )
                for item in request.segments
            ),
            source_language=request.source_language,
            target_language=request.target_language,
        )
        log.info(
            "Translation batch completed provider=%s source=%s target=%s "
            "segments=%d cached=%s",
            result.provider,
            result.source_language,
            result.target_language,
            len(result.segments),
            result.cached,
        )
        return TranslationBatchResponse(
            segments=tuple(
                TranslatedSegmentItem(
                    identifier=item.identifier,
                    original=item.source_text,
                    translation=item.translated_text,
                )
                for item in result.segments
            ),
            source_language=result.source_language,
            target_language=result.target_language,
            cached=result.cached,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="translation.detect",
                summary="Detect the dominant BCP-47 language of supplied text locally.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
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
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
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
                summary="Translate supplied text into the requested language.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
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
        endpoint(
            CapabilityDescriptor(
                name="translation.translate_batch",
                summary=(
                    "Translate identified document segments in one structured batch."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                keywords=(
                    "translate",
                    "translation",
                    "language",
                    "local",
                    "batch",
                    "structured",
                ),
                expected_errors=(
                    "translation.text_required",
                    "translation.target_required",
                    "translation.same_language",
                    "translation.language_invalid",
                    "translation.segment_invalid",
                    "translation.unavailable",
                ),
                timeout_seconds=60,
            ),
            TranslationBatchRequest,
            TranslationBatchResponse,
            translate_batch,
        ),
    )
