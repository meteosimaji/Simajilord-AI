from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simajilord.capabilities.translation import (
    TranslationDetectRequest,
    TranslationDetectResponse,
    TranslationLanguagesRequest,
    TranslationLanguagesResponse,
    TranslationTranslateRequest,
    TranslationTranslateResponse,
    build_translation_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.services.translation import (
    TranslationDetection,
    TranslationHypothesis,
    TranslationLanguage,
    TranslationProviderError,
    TranslationResult,
    TranslationService,
)


@dataclass(slots=True)
class FakeTranslationProvider:
    provider_name: str = "fake-local"
    detection: TranslationDetection = field(
        default_factory=lambda: TranslationDetection(
            language="ja",
            confidence=0.98,
            hypotheses=(
                TranslationHypothesis(code="ja", confidence=0.98),
                TranslationHypothesis(code="zh", confidence=0.01),
            ),
        )
    )
    languages: tuple[TranslationLanguage, ...] = (
        TranslationLanguage(
            code="en",
            english_name="English",
            native_name="English",
            availability="installed",
        ),
        TranslationLanguage(
            code="ja",
            english_name="Japanese",
            native_name="日本語",
            availability="same_language",
        ),
    )
    translated_text: str = "Good morning"
    failure: TranslationProviderError | None = None
    calls: list[tuple[str, object]] = field(default_factory=list)
    closed: bool = False

    @property
    def name(self) -> str:
        return self.provider_name

    async def detect_language(self, text: str) -> TranslationDetection:
        self.calls.append(("detect", text))
        if self.failure is not None:
            raise self.failure
        return self.detection

    async def supported_targets(
        self,
        source_language: str | None = None,
    ) -> tuple[TranslationLanguage, ...]:
        self.calls.append(("languages", source_language))
        if self.failure is not None:
            raise self.failure
        return self.languages

    async def translate(
        self,
        text: str,
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationResult:
        self.calls.append(
            (
                "translate",
                (text, source_language, target_language),
            )
        )
        if self.failure is not None:
            raise self.failure
        return TranslationResult(
            source_text=text,
            translated_text=self.translated_text,
            source_language=source_language or self.detection.language,
            target_language=target_language,
            provider=self.name,
        )

    async def close(self) -> None:
        self.closed = True


def _context() -> InvocationContext:
    return InvocationContext(
        actor_id="7",
        workspace_id="1",
        transport="test",
        request_id="translation-test",
    )


@pytest.mark.asyncio
async def test_translation_service_detects_and_translates_without_cloud() -> None:
    provider = FakeTranslationProvider()
    service = TranslationService((provider,), max_characters=100)

    result = await service.translate("  おはようございます  ", target_language="en")

    assert result.translated_text == "Good morning"
    assert result.source_language == "ja"
    assert result.target_language == "en"
    assert result.provider == "fake-local"
    assert provider.calls == [
        ("detect", "おはようございます"),
        ("translate", ("おはようございます", "ja", "en")),
    ]
    await service.close()
    assert provider.closed


@pytest.mark.asyncio
async def test_translation_service_falls_back_only_for_recoverable_provider_error() -> None:
    unavailable = FakeTranslationProvider(
        provider_name="unavailable",
        failure=TranslationProviderError(
            "translation.helper_unavailable",
            fallback_allowed=True,
        ),
    )
    fallback = FakeTranslationProvider(provider_name="fallback")
    service = TranslationService((unavailable, fallback))

    result = await service.translate(
        "おはよう",
        source_language="ja",
        target_language="en",
    )

    assert result.provider == "fallback"
    assert unavailable.calls == [
        ("translate", ("おはよう", "ja", "en")),
    ]
    assert fallback.calls == [
        ("translate", ("おはよう", "ja", "en")),
    ]


@pytest.mark.asyncio
async def test_translation_service_rejects_invalid_or_oversized_input() -> None:
    service = TranslationService((FakeTranslationProvider(),), max_characters=5)

    with pytest.raises(UserError, match=r"translation\.text_required"):
        await service.translate(" ", target_language="en")
    with pytest.raises(UserError, match=r"translation\.text_too_long"):
        await service.translate("123456", target_language="en")
    with pytest.raises(UserError, match=r"translation\.language_invalid"):
        await service.translate("hello", target_language="../en")
    with pytest.raises(UserError, match=r"translation\.same_language"):
        await service.translate(
            "hello",
            source_language="en",
            target_language="en",
        )


@pytest.mark.asyncio
async def test_translation_capabilities_are_typed_and_independently_invokable() -> None:
    service = TranslationService((FakeTranslationProvider(),))
    endpoints = {
        item.descriptor.name: item
        for item in build_translation_endpoints(service)
    }

    assert set(endpoints) == {
        "translation.detect",
        "translation.languages",
        "translation.translate",
    }
    detected = await endpoints["translation.detect"].invoke(
        TranslationDetectRequest(text="おはよう"),
        _context(),
    )
    languages = await endpoints["translation.languages"].invoke(
        TranslationLanguagesRequest(source_language="ja"),
        _context(),
    )
    translated = await endpoints["translation.translate"].invoke(
        TranslationTranslateRequest(
            text="おはよう",
            source_language="ja",
            target_language="en",
        ),
        _context(),
    )

    assert isinstance(detected, TranslationDetectResponse)
    assert detected.language == "ja"
    assert isinstance(languages, TranslationLanguagesResponse)
    assert languages.languages[0].code == "en"
    assert isinstance(translated, TranslationTranslateResponse)
    assert translated.translation == "Good morning"
    assert all(
        item.descriptor.risk.value == "read"
        for item in endpoints.values()
    )
