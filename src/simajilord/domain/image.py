"""Transport-neutral image-generation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ImageAspectRatio(StrEnum):
    SQUARE = "square"
    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class ImageRendering(StrEnum):
    PHOTO = "photo"
    ILLUSTRATION = "illustration"


class ImageJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ImageGenerationPrompt:
    """Structured fields compiled into the Ideogram 4 JSON-caption schema."""

    subject: str
    scene: str
    composition: str
    style: str
    lighting: str
    details: str = ""
    avoid: str = ""
    aspect_ratio: ImageAspectRatio = ImageAspectRatio.SQUARE
    rendering: ImageRendering = ImageRendering.PHOTO
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerationJob:
    job_id: str
    actor_id: str
    workspace_id: str
    delivery_target_id: str
    reply_to_message_id: str | None
    prompt: ImageGenerationPrompt
    caption_json: str
    status: ImageJobStatus
    output_path: Path | None
    width: int
    height: int
    seed: int
    created_at_iso: str
    completed_at_iso: str | None = None
    generation_seconds: float | None = None
    error_code: str | None = None
    progress_step: int = 0
    progress_total: int = 12
    delivery_message_id: str | None = None
    delivered: bool = False
