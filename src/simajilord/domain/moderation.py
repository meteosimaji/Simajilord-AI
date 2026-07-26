"""Transport-neutral models for synthetic-media analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SyntheticMediaVerdict(StrEnum):
    AI_GENERATED = "ai_generated"
    NOT_AI_GENERATED = "not_ai_generated"
    INCONCLUSIVE = "inconclusive"


class SyntheticMediaModality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class SyntheticMediaProviderResult:
    """Normalized scores returned by one concrete analysis provider."""

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


@dataclass(frozen=True, slots=True)
class SyntheticMediaAnalysis:
    """Reusable result plus local cache and daily-quota state."""

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
