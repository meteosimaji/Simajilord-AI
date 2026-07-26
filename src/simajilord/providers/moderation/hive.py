"""Hive V3 AI-generated image and video detection provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import aiohttp

from simajilord.core.errors import ModerationError
from simajilord.domain.moderation import (
    SyntheticMediaModality,
    SyntheticMediaProviderResult,
    SyntheticMediaVerdict,
)

HIVE_V3_AI_DETECTION_ENDPOINT = (
    "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"
)
HIVE_V3_AI_DETECTION_MODEL = "hive/ai-generated-and-deepfake-content-detection"
HIVE_RECOMMENDED_THRESHOLD = 0.9
_MAX_RESPONSE_BYTES = 1_000_000
_NON_SOURCE_CLASSES = frozenset(
    {
        "ai_generated",
        "not_ai_generated",
        "deepfake",
        "none",
        "inconclusive",
        "inconclusive_video",
        "ai_generated_audio",
        "not_ai_generated_audio",
        "ai_generated_music",
        "not_ai_generated_music",
        "ai_generated_music_vocal",
        "ai_generated_music_cover",
    }
)


@dataclass(frozen=True, slots=True)
class _FrameScores:
    ai_generated_score: float
    not_ai_generated_score: float
    deepfake_score: float
    top_source: str | None
    top_source_score: float


class HiveSyntheticMediaProvider:
    """Submit one bounded media object to Hive without blind paid retries."""

    def __init__(self, *, api_key: str, timeout_seconds: float) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("Hive API key must not be empty.")
        self._api_key = normalized_key
        self.timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "hive"

    @property
    def model(self) -> str:
        return HIVE_V3_AI_DETECTION_MODEL

    async def analyze(
        self,
        *,
        content: bytes,
        filename: str,
        content_type: str,
        threshold: float,
    ) -> SyntheticMediaProviderResult:
        form = aiohttp.FormData()
        form.add_field(
            "media",
            content,
            filename=filename,
            content_type=content_type,
        )
        try:
            async with self._client().post(
                HIVE_V3_AI_DETECTION_ENDPOINT,
                data=form,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                    "User-Agent": "SimajilordModeration/0.1 (+https://simajilord.com)",
                },
            ) as response:
                body = await _read_bounded(response, maximum=_MAX_RESPONSE_BYTES)
                if response.status != 200:
                    detail = _error_detail(body, response.reason)
                    category = _http_error_category(response.status)
                    raise ModerationError(
                        category,
                        detail,
                        http_status=response.status,
                    )
        except ModerationError:
            raise
        except TimeoutError as exc:
            raise ModerationError("timeout", "Hive request timed out.") from exc
        except aiohttp.ClientError as exc:
            raise ModerationError(
                "provider_unavailable",
                "Hive could not be reached.",
            ) from exc
        try:
            payload: object = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModerationError(
                "invalid_response",
                "Hive returned invalid JSON.",
            ) from exc
        return parse_hive_response(
            payload,
            threshold=threshold,
            content_type=content_type,
        )

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()

    def _client(self) -> aiohttp.ClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            connector=aiohttp.TCPConnector(limit=2),
            auto_decompress=True,
        )
        self._session = session
        return session


def parse_hive_response(
    payload: object,
    *,
    threshold: float = HIVE_RECOMMENDED_THRESHOLD,
    content_type: str = "image/png",
) -> SyntheticMediaProviderResult:
    """Normalize Hive's frame output and reject malformed score sets."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be greater than 0 and at most 1")
    modality = _modality(content_type)
    root = _mapping(payload, "Hive response")
    model = root.get("model")
    if model != HIVE_V3_AI_DETECTION_MODEL:
        raise ModerationError(
            "invalid_response",
            f"Hive returned an unexpected model: {model!r}.",
        )
    outputs = root.get("output")
    if not isinstance(outputs, list) or not outputs:
        raise ModerationError("invalid_response", "Hive output is missing.")
    frames: list[_FrameScores] = []
    for output_index, raw_output in enumerate(outputs):
        output = _mapping(raw_output, f"Hive output {output_index}")
        scores = _score_map(output, output_index=output_index)
        ai_score, not_ai_score = _score_pair(
            scores,
            positive="ai_generated",
            negative="not_ai_generated",
        )
        if ai_score is None or not_ai_score is None:
            raise ModerationError("invalid_response", "Hive visual score head is missing.")
        deepfake_score = scores.get("deepfake")
        if deepfake_score is None:
            raise ModerationError("invalid_response", "Hive deepfake score is missing.")
        source_candidates = tuple(
            (name, score)
            for name, score in scores.items()
            if name not in _NON_SOURCE_CLASSES
        )
        top_source, top_source_score = max(
            source_candidates,
            key=lambda pair: pair[1],
            default=(None, 0.0),
        )
        frames.append(
            _FrameScores(
                ai_generated_score=ai_score,
                not_ai_generated_score=not_ai_score,
                deepfake_score=deepfake_score,
                top_source=top_source,
                top_source_score=top_source_score,
            )
        )
    ai_score = max(frame.ai_generated_score for frame in frames)
    not_ai_score = min(frame.not_ai_generated_score for frame in frames)
    deepfake_score = max(frame.deepfake_score for frame in frames)
    if ai_score >= threshold:
        verdict = SyntheticMediaVerdict.AI_GENERATED
    elif not_ai_score >= threshold:
        verdict = SyntheticMediaVerdict.NOT_AI_GENERATED
    else:
        verdict = SyntheticMediaVerdict.INCONCLUSIVE
    peak = max(
        frames,
        key=lambda frame: frame.ai_generated_score,
    )
    deepfake_likely = _deepfake_likely(
        tuple(frame.deepfake_score for frame in frames),
        modality=modality,
        image_threshold=threshold,
    )
    raw_version = root.get("version")
    return SyntheticMediaProviderResult(
        modality=modality,
        ai_generated_score=ai_score,
        not_ai_generated_score=not_ai_score,
        deepfake_score=deepfake_score,
        deepfake_likely=deepfake_likely,
        sample_count=len(frames),
        model=HIVE_V3_AI_DETECTION_MODEL,
        threshold=threshold,
        top_source=peak.top_source,
        top_source_score=peak.top_source_score,
        verdict=verdict,
        version=str(raw_version)[:80] if raw_version is not None else None,
    )


def _score_map(output: Mapping[str, object], *, output_index: int) -> dict[str, float]:
    raw_classes = output.get("classes")
    if not isinstance(raw_classes, list):
        raise ModerationError(
            "invalid_response",
            f"Hive output {output_index} classes are missing.",
        )
    scores: dict[str, float] = {}
    for class_index, raw_class in enumerate(raw_classes):
        item = _mapping(
            raw_class,
            f"Hive output {output_index} class {class_index}",
        )
        class_name = item.get("class")
        raw_score = item.get("value", item.get("score"))
        if not isinstance(class_name, str) or not class_name:
            raise ModerationError("invalid_response", "Hive returned an invalid class.")
        if class_name in scores:
            raise ModerationError(
                "invalid_response",
                f"Hive returned duplicate class {class_name!r}.",
            )
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise ModerationError("invalid_response", "Hive returned an invalid score.")
        score = float(raw_score)
        if not 0.0 <= score <= 1.0:
            raise ModerationError(
                "invalid_response",
                "Hive returned a score outside the 0 to 1 range.",
            )
        scores[class_name] = score
    return scores


def _score_pair(
    scores: Mapping[str, float],
    *,
    positive: str,
    negative: str,
) -> tuple[float | None, float | None]:
    positive_score = scores.get(positive)
    negative_score = scores.get(negative)
    if (positive_score is None) != (negative_score is None):
        raise ModerationError(
            "invalid_response",
            f"Hive returned an incomplete {positive!r} score head.",
        )
    if (
        positive_score is not None
        and negative_score is not None
        and abs(positive_score + negative_score - 1.0) > 1e-5
    ):
        raise ModerationError(
            "invalid_response",
            f"Hive {positive!r} scores do not sum to 1.",
        )
    return positive_score, negative_score


def _modality(content_type: str) -> SyntheticMediaModality:
    family = content_type.partition(";")[0].strip().lower().partition("/")[0]
    try:
        return SyntheticMediaModality(family)
    except ValueError as exc:
        raise ValueError(f"Unsupported HIVE media content type: {content_type!r}") from exc


def _deepfake_likely(
    scores: tuple[float, ...],
    *,
    modality: SyntheticMediaModality,
    image_threshold: float,
) -> bool:
    present = scores
    if modality is SyntheticMediaModality.IMAGE:
        return max(present) >= image_threshold
    marked = tuple(score >= 0.5 for score in present)
    consecutive = any(left and right for left, right in pairwise(marked))
    proportion = sum(marked) / len(marked)
    return consecutive or proportion >= 0.05


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ModerationError("invalid_response", f"{label} is not an object.")
    return value


async def _read_bounded(response: aiohttp.ClientResponse, *, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1_024):
        total += len(chunk)
        if total > maximum:
            response.close()
            raise ModerationError(
                "response_too_large",
                "Hive response is too large.",
                http_status=response.status,
            )
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _error_detail(body: bytes, fallback: str | None) -> str:
    try:
        payload: object = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="replace")[:300] or fallback or "Hive error."
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value[:300]
    return fallback or "Hive error."


def _http_error_category(status: int) -> str:
    if status in {401, 403}:
        return "authentication_failed"
    if status == 429:
        return "rate_limited"
    if 400 <= status < 500:
        return "media_rejected"
    return "provider_unavailable"
