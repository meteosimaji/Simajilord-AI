"""Transport-neutral synthetic-media detection endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    EgressDescriptor,
    EgressFieldKind,
    EgressSinkAudience,
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
                    "Estimate AI-generated and deepfake likelihood for an image "
                    "or video with the configured analysis provider."
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
                side_effects=(
                    "Uses one provider analysis when no cached result exists.",
                ),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="hive",
                    field_kinds=(EgressFieldKind.MEDIA,),
                    request_fields=("filename", "content_type", "content"),
                    sink_audience=EgressSinkAudience.EXTERNAL_PRIVATE,
                ),
            ),
            SyntheticMediaAnalyzeRequest,
            SyntheticMediaAnalyzeResponse,
            analyze,
        ),
        endpoint(
            CapabilityDescriptor(
                name="moderation.status",
                summary="Check synthetic-media analysis provider readiness.",
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.GUILD_MEMBER_METADATA,
                keywords=("moderation", "status", "quota", "hive"),
            ),
            ModerationStatusRequest,
            ModerationStatusResponse,
            status,
        ),
    )
