"""Transport-neutral synthetic-media detection endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.domain.moderation import SyntheticMediaModality, SyntheticMediaVerdict
from simajilord.services.moderation import ModerationService


@dataclass(frozen=True, slots=True)
class SyntheticMediaAnalyzeRequest:
    filename: str
    content_type: str | None
    content: bytes


@dataclass(frozen=True, slots=True)
class SyntheticMediaAnalyzeResponse:
    sha256: str
    filename: str
    content_type: str
    modality: SyntheticMediaModality
    ai_generated_score: float
    not_ai_generated_score: float
    deepfake_score: float
    deepfake_likely: bool
    sample_count: int
    model: str
    threshold: float
    top_source: str | None
    top_source_score: float
    verdict: SyntheticMediaVerdict
    version: str | None
    cached: bool
    quota_used: int
    quota_remaining: int
    quota_limit: int
    quota_reset_at_epoch: int


@dataclass(frozen=True, slots=True)
class ModerationStatusRequest:
    pass


@dataclass(frozen=True, slots=True)
class ModerationStatusResponse:
    configured: bool
    provider: str
    model: str
    quota_used: int
    quota_remaining: int
    quota_limit: int
    quota_reset_at_epoch: int


def build_moderation_endpoints(
    moderation: ModerationService,
) -> tuple[CapabilityEndpoint, ...]:
    async def analyze(
        request: SyntheticMediaAnalyzeRequest,
        _: InvocationContext,
    ) -> SyntheticMediaAnalyzeResponse:
        result = await moderation.analyze(
            content=request.content,
            filename=request.filename,
            content_type=request.content_type,
        )
        return SyntheticMediaAnalyzeResponse(
            sha256=result.sha256,
            filename=result.filename,
            content_type=result.content_type,
            modality=result.modality,
            ai_generated_score=result.ai_generated_score,
            not_ai_generated_score=result.not_ai_generated_score,
            deepfake_score=result.deepfake_score,
            deepfake_likely=result.deepfake_likely,
            sample_count=result.sample_count,
            model=result.model,
            threshold=result.threshold,
            top_source=result.top_source,
            top_source_score=result.top_source_score,
            verdict=result.verdict,
            version=result.version,
            cached=result.cached,
            quota_used=result.quota_used,
            quota_remaining=result.quota_remaining,
            quota_limit=result.quota_limit,
            quota_reset_at_epoch=result.quota_reset_at_epoch,
        )

    async def status(
        _: ModerationStatusRequest,
        __: InvocationContext,
    ) -> ModerationStatusResponse:
        result = await moderation.status()
        return ModerationStatusResponse(
            configured=result.configured,
            provider=result.provider,
            model=result.model,
            quota_used=result.quota_used,
            quota_remaining=result.quota_remaining,
            quota_limit=result.quota_limit,
            quota_reset_at_epoch=result.quota_reset_at_epoch,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="moderation.detect_synthetic_media",
                summary=(
                    "設定済みの解析サービスを使い、画像または動画のAI生成・"
                    "ディープフェイクの可能性を調べます。"
                ),
                risk=RiskLevel.EXTERNAL,
                keywords=(
                    "moderation",
                    "image",
                    "video",
                    "ai-generated",
                    "deepfake",
                    "hive",
                ),
                side_effects=("保存済みの結果がなければ、解析サービスを1回利用します。",),
            ),
            SyntheticMediaAnalyzeRequest,
            SyntheticMediaAnalyzeResponse,
            analyze,
        ),
        endpoint(
            CapabilityDescriptor(
                name="moderation.status",
                summary="解析サービスの準備状況と、本日のローカルAPI利用枠を表示します。",
                risk=RiskLevel.READ,
                keywords=("moderation", "status", "quota", "hive"),
            ),
            ModerationStatusRequest,
            ModerationStatusResponse,
            status,
        ),
    )
