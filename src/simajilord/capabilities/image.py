"""Transport-neutral image-generation capabilities."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.domain.image import (
    ImageAspectRatio,
    ImageGenerationJob,
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.services.files import AgentFileSandbox
from simajilord.services.image import ImageGenerationService


@dataclass(frozen=True, slots=True)
class ImageGenerateRequest:
    """A structured prompt for one turn-owned generated image."""
    subject: str = field(
        metadata={
            "description": (
                "Production-ready visible subject description, not a rewritten short request: "
                "identity, exact count, appearance, pose, expression, clothing, action, and "
                "every user-requested attribute. Make coherent creative choices when omitted."
            )
        }
    )
    scene: str = field(
        metadata={
            "description": (
                "Concrete environment/background: location, time, weather, foreground and "
                "depth layers, materials, and relevant secondary objects. Do not use vague "
                "phrases such as 'nice background'."
            )
        }
    )
    composition: str = field(
        metadata={
            "description": (
                "Executable camera/framing plan: shot size, viewpoint, lens feel, subject "
                "placement, crop boundaries, foreground/background separation, and spatial "
                "relationships."
            )
        }
    )
    style: str = field(
        metadata={
            "description": (
                "Specific medium and visual language with references to texture, line/brush "
                "behavior, tonal treatment, or photographic genre. Generic words like "
                "'beautiful', 'high quality', or 'detailed' alone are invalid."
            )
        }
    )
    lighting: str = field(
        metadata={
            "description": (
                "Visible lighting design: sources and direction, softness, color temperature, "
                "contrast, shadow behavior, and atmosphere."
            )
        }
    )
    details: str = field(
        default="",
        metadata={
            "description": (
                "A positive checklist of important materials, colors, textures, anatomy, "
                "props, counts, relationships, and exact details that must survive generation."
            )
        },
    )
    avoid: str = field(
        default="",
        metadata={
            "description": (
                "Request-specific unwanted artifacts and deviations, including wrong counts, "
                "bad crops, malformed anatomy, unintended text/logos, and style drift."
            )
        },
    )
    aspect_ratio: ImageAspectRatio = ImageAspectRatio.SQUARE
    rendering: ImageRendering = ImageRendering.ILLUSTRATION
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ImageGenerateResponse:
    job_id: str
    status: ImageJobStatus
    path: str
    size_bytes: int
    sha256: str
    kind: str
    image_data_url: str
    width: int
    height: int
    provider_model: str
    generation_seconds: float
    next_action: str


@dataclass(frozen=True, slots=True)
class ImageStatusRequest:
    job_id: str


@dataclass(frozen=True, slots=True)
class ImageStatusResponse:
    job_id: str
    status: ImageJobStatus
    progress_step: int
    progress_total: int
    auto_delivery_enabled: bool
    runtime_delivery_completed: bool
    workspace_handoff_completed: bool
    workspace_path: str | None
    error_code: str | None
    terminal: bool
    retryable: bool
    next_action: str


def build_image_endpoints(
    service: ImageGenerationService,
    files: AgentFileSandbox | None,
) -> tuple[CapabilityEndpoint, ...]:
    async def generate(
        request: ImageGenerateRequest,
        context: InvocationContext,
    ) -> ImageGenerateResponse:
        if context.workspace_id is None:
            raise UserError("image.workspace_required")
        if files is None:
            raise UserError("files.disabled")
        job = await service.submit(
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            delivery_target_id=context.origin_resource_id or "",
            reply_to_message_id=_request_event_id(context.request_id),
            prompt=ImageGenerationPrompt(
                subject=request.subject,
                scene=request.scene,
                composition=request.composition,
                style=request.style,
                lighting=request.lighting,
                details=request.details,
                avoid=request.avoid,
                aspect_ratio=request.aspect_ratio,
                rendering=request.rendering,
                seed=request.seed,
            ),
            auto_deliver=False,
            idempotency_key=context.request_id,
        )
        terminal = await service.wait_for_terminal(job.job_id)
        if terminal.status is not ImageJobStatus.COMPLETED:
            raise UserError(
                "image.generation_failed",
                job_id=terminal.job_id,
                error_code=terminal.error_code,
            )
        return await _agent_image_result(terminal, files=files, service=service)

    async def status(
        request: ImageStatusRequest,
        context: InvocationContext,
    ) -> ImageStatusResponse:
        job = await service.owned_job(request.job_id, actor_id=context.actor_id)
        terminal = job.status in {ImageJobStatus.COMPLETED, ImageJobStatus.FAILED}
        retryable = job.status is ImageJobStatus.FAILED and job.error_code in {
            "caption_invalid",
            "generation_timeout",
            "provider_failed",
        }
        workspace_path: str | None = None
        if (
            job.status is ImageJobStatus.COMPLETED
            and files is not None
            and job.output_path is not None
            and job.output_path.is_file()
        ):
            content = await asyncio.to_thread(job.output_path.read_bytes)
            record = await asyncio.to_thread(
                files.import_bytes,
                job.workspace_id,
                _agent_image_path(job.job_id),
                content,
            )
            workspace_path = record.path
            await service.mark_handed_off(job.job_id)
        if job.status in {ImageJobStatus.QUEUED, ImageJobStatus.RUNNING}:
            next_action = "Wait for generation completion; do not submit a duplicate."
        elif job.status is ImageJobStatus.COMPLETED and workspace_path is not None:
            next_action = (
                "Use discord.send_file with workspace_path when the requested image "
                "should be posted."
            )
        elif job.status is ImageJobStatus.COMPLETED:
            next_action = "The result exists but is not available in the agent workspace."
        elif retryable:
            next_action = "Report the failure accurately and retry only when requested."
        else:
            next_action = "Report the failure accurately; do not claim a retry started."
        return ImageStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress_step=job.progress_step,
            progress_total=job.progress_total,
            auto_delivery_enabled=job.auto_deliver,
            runtime_delivery_completed=job.delivered,
            workspace_handoff_completed=(
                job.handoff_completed or workspace_path is not None
            ),
            workspace_path=workspace_path,
            error_code=job.error_code,
            terminal=terminal,
            retryable=retryable,
            next_action=next_action,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="image.generate",
                summary=(
                    "Generate one image through Codex OAuth from a production-ready "
                    "visual brief, wait for completion in this turn, and return both "
                    "the visible image and a workspace file."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "image",
                    "picture",
                    "illustration",
                    "photo",
                    "draw",
                    "画像",
                    "イラスト",
                    "写真",
                ),
                side_effects=(
                    "Starts hosted GPT Image generation through the saved Codex login.",
                    "Imports the generated file into Simajilord local storage.",
                    "Does not post to Discord; the agent chooses whether and how to send it.",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "image.provider_unavailable",
                    "image.generation_failed",
                ),
                timeout_seconds=900,
                user_visible_effect=(
                    "Creates a generated image file but does not post it automatically."
                ),
            ),
            ImageGenerateRequest,
            ImageGenerateResponse,
            generate,
        ),
        endpoint(
            CapabilityDescriptor(
                name="image.status",
                summary="Inspect an image-generation job requested by this actor.",
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("image", "progress", "job", "status", "進捗"),
                requires_workspace=True,
                expected_errors=("image.job_not_found",),
                timeout_seconds=10,
            ),
            ImageStatusRequest,
            ImageStatusResponse,
            status,
        ),
    )


async def _agent_image_result(
    job: ImageGenerationJob,
    *,
    files: AgentFileSandbox,
    service: ImageGenerationService,
) -> ImageGenerateResponse:
    if job.output_path is None or not job.output_path.is_file():
        raise UserError("image.output_unavailable", job_id=job.job_id)
    content = await asyncio.to_thread(job.output_path.read_bytes)
    record = await asyncio.to_thread(
        files.import_bytes,
        job.workspace_id,
        _agent_image_path(job.job_id),
        content,
    )
    await service.mark_handed_off(job.job_id)
    return ImageGenerateResponse(
        job_id=job.job_id,
        status=job.status,
        path=record.path,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        kind=record.kind,
        image_data_url="data:image/png;base64," + base64.b64encode(content).decode(),
        width=job.width,
        height=job.height,
        provider_model=job.provider_model or "GPT Image 2・Codex OAuth",
        generation_seconds=job.generation_seconds or 0.0,
        next_action=(
            "Inspect the image attached to this tool result. If it fulfills the request, "
            "call discord.send_file with path and the authorized current channel. If it "
            "does not, revise the brief deliberately. Never claim Discord delivery until "
            "discord.send_file succeeds."
        ),
    )


def _agent_image_path(job_id: str) -> str:
    return f"generated/simajilord-{job_id[:12]}.png"


def _request_event_id(request_id: str) -> str | None:
    prefix = "discord:message:"
    return request_id[len(prefix) :] if request_id.startswith(prefix) else None
