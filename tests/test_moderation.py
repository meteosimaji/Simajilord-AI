from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from simajilord.core.errors import ModerationError, UserError
from simajilord.domain.moderation import (
    SyntheticMediaModality,
    SyntheticMediaProviderResult,
    SyntheticMediaVerdict,
)
from simajilord.providers.moderation import (
    HIVE_V3_AI_DETECTION_MODEL,
    parse_hive_response,
)
from simajilord.services.moderation import ModerationService, ModerationStore


def _classes(
    *,
    ai: float,
    deepfake: float,
    source: str = "stablediffusionxl",
    source_score: float = 0.0,
) -> list[dict[str, object]]:
    return [
        {"class": "ai_generated", "value": ai},
        {"class": "not_ai_generated", "value": 1.0 - ai},
        {"class": "deepfake", "value": deepfake},
        {"class": "ai_generated_audio", "value": 0.0},
        {"class": "not_ai_generated_audio", "value": 1.0},
        {"class": source, "value": source_score},
        {"class": "none", "value": 1.0 - source_score},
    ]


def _payload(*outputs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": HIVE_V3_AI_DETECTION_MODEL,
        "version": 1,
        "output": [{"classes": classes} for classes in outputs],
    }


def test_hive_image_response_preserves_scores_and_generator() -> None:
    result = parse_hive_response(
        _payload(
            _classes(
                ai=0.999987,
                deepfake=0.00028,
                source_score=0.99166,
            )
        ),
        content_type="image/png",
    )

    assert result.modality is SyntheticMediaModality.IMAGE
    assert result.ai_generated_score == pytest.approx(0.999987)
    assert result.deepfake_score == pytest.approx(0.00028)
    assert result.deepfake_likely is False
    assert result.top_source == "stablediffusionxl"
    assert result.top_source_score == pytest.approx(0.99166)
    assert result.verdict is SyntheticMediaVerdict.AI_GENERATED


def test_hive_video_uses_any_frame_for_ai_and_documented_deepfake_rule() -> None:
    result = parse_hive_response(
        _payload(
            _classes(ai=0.1, deepfake=0.6),
            _classes(ai=0.95, deepfake=0.6, source="ltx", source_score=0.98),
            _classes(ai=0.2, deepfake=0.0),
        ),
        content_type="video/mp4",
    )

    assert result.modality is SyntheticMediaModality.VIDEO
    assert result.ai_generated_score == pytest.approx(0.95)
    assert result.deepfake_score == pytest.approx(0.6)
    assert result.deepfake_likely is True
    assert result.sample_count == 3
    assert result.top_source == "ltx"


def test_hive_parser_rejects_audio_for_the_enabled_visual_product() -> None:
    with pytest.raises(ValueError, match="Unsupported HIVE media content type"):
        parse_hive_response(
            _payload(_classes(ai=0.0, deepfake=0.0)),
            content_type="audio/mpeg",
        )


def test_hive_rejects_partial_or_non_normalized_score_heads() -> None:
    payload = _payload(_classes(ai=0.9, deepfake=0.0))
    output = payload["output"]
    assert isinstance(output, list)
    classes = output[0]["classes"]
    assert isinstance(classes, list)
    classes.pop(1)

    with pytest.raises(ModerationError, match="incomplete"):
        parse_hive_response(payload)


class _FakeProvider:
    name = "hive"
    model = HIVE_V3_AI_DETECTION_MODEL

    def __init__(self, result: SyntheticMediaProviderResult) -> None:
        self.result = result
        self.calls = 0
        self.closed = False

    async def analyze(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        threshold: float,
    ) -> SyntheticMediaProviderResult:
        self.calls += 1
        return replace(self.result, threshold=threshold)

    async def close(self) -> None:
        self.closed = True


def _provider_result() -> SyntheticMediaProviderResult:
    return SyntheticMediaProviderResult(
        modality=SyntheticMediaModality.IMAGE,
        ai_generated_score=0.95,
        not_ai_generated_score=0.05,
        deepfake_score=0.01,
        deepfake_likely=False,
        sample_count=1,
        model=HIVE_V3_AI_DETECTION_MODEL,
        threshold=0.9,
        top_source="gptimage1_5",
        top_source_score=0.93,
        verdict=SyntheticMediaVerdict.AI_GENERATED,
        version="1",
    )


@pytest.mark.asyncio
async def test_service_cache_avoids_second_paid_attempt(tmp_path) -> None:
    provider = _FakeProvider(_provider_result())
    service = ModerationService(
        provider=provider,
        store=ModerationStore(tmp_path / "moderation.sqlite3"),
        daily_limit=100,
        max_media_bytes=1_000,
        threshold=0.9,
    )

    first = await service.analyze(
        content=b"same-image",
        filename="sample.png",
        content_type="image/png",
    )
    second = await service.analyze(
        content=b"same-image",
        filename="renamed.png",
        content_type="image/png",
    )

    assert provider.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert first.quota_used == second.quota_used == 1
    assert second.top_source == "gptimage1_5"
    assert (tmp_path / "moderation.sqlite3").stat().st_mode & 0o077 == 0
    await service.close()
    assert provider.closed is True


@pytest.mark.asyncio
async def test_daily_limit_is_atomic_and_cached_results_remain_available(tmp_path) -> None:
    provider = _FakeProvider(_provider_result())
    service = ModerationService(
        provider=provider,
        store=ModerationStore(tmp_path / "moderation.sqlite3"),
        daily_limit=1,
        max_media_bytes=1_000,
        threshold=0.9,
    )
    results = await asyncio.gather(
        service.analyze(
            content=b"first",
            filename="first.png",
            content_type="image/png",
        ),
        service.analyze(
            content=b"second",
            filename="second.png",
            content_type="image/png",
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, Exception) for result in results) == 1
    errors = [result for result in results if isinstance(result, UserError)]
    assert len(errors) == 1
    assert errors[0].code == "moderation.daily_limit_reached"
    successful = next(result for result in results if not isinstance(result, Exception))
    cached = await service.analyze(
        content=b"first" if successful.filename == "first.png" else b"second",
        filename=successful.filename,
        content_type="image/png",
    )
    assert cached.cached is True
    assert cached.quota_used == 1


@pytest.mark.asyncio
async def test_service_validates_supported_media_before_reserving_quota(tmp_path) -> None:
    provider = _FakeProvider(_provider_result())
    service = ModerationService(
        provider=provider,
        store=ModerationStore(tmp_path / "moderation.sqlite3"),
        daily_limit=100,
        max_media_bytes=5,
        threshold=0.9,
    )

    with pytest.raises(UserError) as unsupported:
        await service.analyze(
            content=b"text",
            filename="sample.txt",
            content_type="text/plain",
        )
    assert unsupported.value.code == "moderation.media_type_unsupported"
    with pytest.raises(UserError) as too_large:
        await service.analyze(
            content=b"123456",
            filename="sample.wav",
            content_type="audio/wav",
        )
    assert too_large.value.code == "moderation.media_too_large"
    with pytest.raises(UserError) as audio:
        await service.analyze(
            content=b"1",
            filename="sample.wav",
            content_type="audio/wav",
        )
    assert audio.value.code == "moderation.media_type_unsupported"
    assert provider.calls == 0
