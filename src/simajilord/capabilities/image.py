"""Transport-neutral image-generation capabilities."""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass, field

from PIL import Image

from simajilord.core import (
    ApprovalMode,
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
from simajilord.core.errors import UserError
from simajilord.domain.image import (
    ImageAspectRatio,
    ImageGenerationJob,
    ImageGenerationModel,
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.services.files import AgentFileSandbox
from simajilord.services.image import ImageGenerationService

from .file_scope import file_provenance, file_workspace_id

_MAX_MODEL_IMAGE_PREVIEW_BYTES = 512_000
_MODEL_IMAGE_PREVIEW_DIMENSIONS = (1_024, 896, 768, 640, 512, 384)
_MODEL_IMAGE_PREVIEW_QUALITIES = (86, 78, 70, 62, 54, 46)


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
    preview_size_bytes: int
    preview_width: int
    preview_height: int
    width: int
    height: int
    generation_seconds: float
    requested_model: ImageGenerationModel
    provider_model: str
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
    requested_model: ImageGenerationModel
    provider_model: str | None
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
                model=ImageGenerationModel.GPT_IMAGE_2,
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
        return await _agent_image_result(
            terminal,
            files=files,
            service=service,
            context=context,
        )

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
                file_workspace_id(context),
                _agent_image_path(job.job_id),
                content,
                provenance=file_provenance(context),
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
            workspace_handoff_completed=(job.handoff_completed or workspace_path is not None),
            workspace_path=workspace_path,
            error_code=job.error_code,
            terminal=terminal,
            retryable=retryable,
            requested_model=job.prompt.model,
            provider_model=job.provider_model,
            next_action=next_action,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="image.generate",
                summary=(
                    "Generate one image from a production-ready visual brief, wait "
                    "for completion in this turn, and return both a bounded visual "
                    "preview and the original workspace file."
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
                    "Starts one image generation job.",
                    "Imports the generated file into Simajilord local storage.",
                    "Does not post by itself; publication is a separate model decision "
                    "based on the user's request.",
                ),
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="configured_image_generation",
                    field_kinds=(EgressFieldKind.PROMPT,),
                    request_fields=(
                        "subject",
                        "scene",
                        "composition",
                        "style",
                        "lighting",
                        "details",
                        "avoid",
                        "aspect_ratio",
                        "rendering",
                        "seed",
                    ),
                    sink_audience=EgressSinkAudience.EXTERNAL_PRIVATE,
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
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
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
    context: InvocationContext,
) -> ImageGenerateResponse:
    if job.output_path is None or not job.output_path.is_file():
        raise UserError("image.output_unavailable", job_id=job.job_id)
    content = await asyncio.to_thread(job.output_path.read_bytes)
    record = await asyncio.to_thread(
        files.import_bytes,
        file_workspace_id(context),
        _agent_image_path(job.job_id),
        content,
        provenance=file_provenance(context),
    )
    await service.mark_handed_off(job.job_id)
    preview, preview_media_type, preview_width, preview_height = await asyncio.to_thread(
        _model_image_preview,
        content,
    )
    return ImageGenerateResponse(
        job_id=job.job_id,
        status=job.status,
        path=record.path,
        size_bytes=record.size_bytes,
        sha256=record.sha256,
        kind=record.kind,
        image_data_url=(f"data:{preview_media_type};base64," + base64.b64encode(preview).decode()),
        preview_size_bytes=len(preview),
        preview_width=preview_width,
        preview_height=preview_height,
        width=job.width,
        height=job.height,
        generation_seconds=job.generation_seconds or 0.0,
        requested_model=job.prompt.model,
        provider_model=job.provider_model or job.prompt.model.value,
        next_action=(
            "Inspect the lightweight preview attached to this tool result. The path is "
            "the full-resolution original. Decide semantically from the exact active "
            "request and conversation context whether publishing it in Discord fulfills "
            "the user's intent. A request to create an image in the active public channel "
            "can imply that the result should be shown; no particular delivery verb is "
            "required. Keep it unpublished when the user requests private comparison or "
            "iteration, or when publication would introduce unresolved privacy or safety "
            "risk. Use discord.send_file or discord.send_files when you decide to publish. "
            "Never claim Discord delivery until the attachment send succeeds."
        ),
    )


def _model_image_preview(content: bytes) -> tuple[bytes, str, int, int]:
    """Return a bounded model preview while preserving the full workspace artifact."""

    with Image.open(io.BytesIO(content)) as source:
        source.load()
        source_format = source.format
        source_width, source_height = source.size
        image = source.convert("RGB")
    if (
        source_format == "PNG"
        and source_width <= 1_024
        and source_height <= 1_024
        and len(content) <= _MAX_MODEL_IMAGE_PREVIEW_BYTES
    ):
        return content, "image/png", source_width, source_height

    smallest: tuple[bytes, int, int] | None = None
    for maximum_dimension in _MODEL_IMAGE_PREVIEW_DIMENSIONS:
        candidate = image.copy()
        candidate.thumbnail(
            (maximum_dimension, maximum_dimension),
            Image.Resampling.LANCZOS,
        )
        for quality in _MODEL_IMAGE_PREVIEW_QUALITIES:
            output = io.BytesIO()
            candidate.save(
                output,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            jpeg = output.getvalue()
            smallest = (jpeg, candidate.width, candidate.height)
            if len(jpeg) <= _MAX_MODEL_IMAGE_PREVIEW_BYTES:
                return jpeg, "image/jpeg", candidate.width, candidate.height

    assert smallest is not None
    jpeg, width, height = smallest
    if len(jpeg) > _MAX_MODEL_IMAGE_PREVIEW_BYTES:
        raise UserError(
            "image.preview_too_large",
            preview_size_bytes=len(jpeg),
            maximum_size_bytes=_MAX_MODEL_IMAGE_PREVIEW_BYTES,
        )
    return jpeg, "image/jpeg", width, height


def _agent_image_path(job_id: str) -> str:
    return f"generated/simajilord-{job_id[:12]}.png"


def _request_event_id(request_id: str) -> str | None:
    prefix = "discord:message:"
    return request_id[len(prefix) :] if request_id.startswith(prefix) else None
