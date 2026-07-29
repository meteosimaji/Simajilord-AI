from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from simajilord.capabilities.translation import (
    TranslationBatchRequest,
    TranslationBatchResponse,
    TranslationDetectRequest,
    TranslationDetectResponse,
    TranslationLanguagesRequest,
    TranslationLanguagesResponse,
    TranslationSegmentItem,
    TranslationTranslateRequest,
    TranslationTranslateResponse,
    build_translation_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError
from simajilord.services.translation import (
    TranslatedSegment,
    TranslationBatchResult,
    TranslationDetection,
    TranslationHypothesis,
    TranslationLanguage,
    TranslationProviderError,
    TranslationResult,
    TranslationSegment,
    TranslationService,
    TranslationStore,
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

    async def translate_batch(
        self,
        segments: tuple[TranslationSegment, ...],
        *,
        source_language: str | None,
        target_language: str,
    ) -> TranslationBatchResult:
        self.calls.append(
            (
                "translate_batch",
                (segments, source_language, target_language),
            )
        )
        if self.failure is not None:
            raise self.failure
        return TranslationBatchResult(
            segments=tuple(
                TranslatedSegment(
                    identifier=item.identifier,
                    source_text=item.text,
                    translated_text=f"{self.translated_text}:{item.identifier}",
                )
                for item in segments
            ),
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
        "translation.translate_batch",
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
    batch = await endpoints["translation.translate_batch"].invoke(
        TranslationBatchRequest(
            segments=(
                TranslationSegmentItem(identifier="content", text="おはよう"),
                TranslationSegmentItem(identifier="embed.0.title", text="朝"),
            ),
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
    assert isinstance(batch, TranslationBatchResponse)
    assert tuple(item.identifier for item in batch.segments) == (
        "content",
        "embed.0.title",
    )
    assert all(
        item.descriptor.risk.value == "read"
        for item in endpoints.values()
    )


@pytest.mark.asyncio
async def test_translation_batch_cache_and_preferences_survive_restart(
    tmp_path,
) -> None:
    store_path = tmp_path / "translations.sqlite3"
    provider = FakeTranslationProvider()
    service = TranslationService(
        (provider,),
        store=TranslationStore(store_path),
    )
    segments = (
        TranslationSegment(identifier="content", text="おはよう"),
        TranslationSegment(identifier="embed.0.title", text="朝"),
    )

    first = await service.translate_batch(
        segments,
        source_language="ja",
        target_language="en",
    )
    second = await service.translate_batch(
        segments,
        source_language="ja",
        target_language="en",
    )
    await service.set_preference(
        actor_id="7",
        workspace_id=None,
        target_language="fr",
    )
    await service.set_preference(
        actor_id="7",
        workspace_id="1",
        target_language="en",
    )
    await service.record_recent_target(actor_id="7", code="en")
    await service.record_recent_target(actor_id="7", code="fr")

    restarted = TranslationService(
        (FakeTranslationProvider(),),
        store=TranslationStore(store_path),
    )
    workspace_preference = await restarted.preference(
        actor_id="7",
        workspace_id="1",
    )
    fallback_preference = await restarted.preference(
        actor_id="7",
        workspace_id="2",
    )

    assert first.cached is False
    assert second.cached is True
    assert sum(call[0] == "translate_batch" for call in provider.calls) == 1
    assert workspace_preference is not None
    assert workspace_preference.target_language == "en"
    assert fallback_preference is not None
    assert fallback_preference.target_language == "fr"
    assert await restarted.recent_targets(actor_id="7") == ("fr", "en")
