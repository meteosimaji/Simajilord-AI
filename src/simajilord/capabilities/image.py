"""Transport-neutral local image-generation capabilities."""

from __future__ import annotations

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
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.services.image import ImageGenerationService


@dataclass(frozen=True, slots=True)
class ImageGenerateRequest:
    """A structured prompt plus a transport-owned delivery route."""

    delivery_target_id: str = field(
        metadata={"description": "The authorized Discord channel ID from this event."}
    )
    reply_to_event_id: str = field(
        metadata={"description": "The exact triggering Discord event/message ID."}
    )
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
    accepted: bool
    status: ImageJobStatus
    width: int
    height: int
    seed: int
    progress_delivery: str
    retryable: bool
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
    delivered: bool
    error_code: str | None
    terminal: bool
    retryable: bool
    next_action: str


def build_image_endpoints(
    service: ImageGenerationService,
) -> tuple[CapabilityEndpoint, ...]:
    async def generate(
        request: ImageGenerateRequest,
        context: InvocationContext,
    ) -> ImageGenerateResponse:
        if context.workspace_id is None:
            raise UserError("image.workspace_required")
        if (
            context.transport == "agent"
            and request.delivery_target_id not in context.resource_ids
        ):
            raise UserError("image.delivery_target_forbidden")
        expected_event = _request_event_id(context.request_id)
        normalized_reply_event = _normalized_reply_event(
            request.reply_to_event_id,
            context=context,
        )
        if (
            context.transport == "agent"
            and expected_event is not None
            and normalized_reply_event != expected_event
        ):
            raise UserError("image.reply_target_forbidden")
        job = await service.submit(
            actor_id=context.actor_id,
            workspace_id=context.workspace_id,
            delivery_target_id=request.delivery_target_id,
            reply_to_message_id=normalized_reply_event,
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
        )
        return ImageGenerateResponse(
            job_id=job.job_id,
            accepted=True,
            status=job.status,
            width=job.width,
            height=job.height,
            seed=job.seed,
            progress_delivery=(
                "The runtime will post a start update and deliver the image "
                "in this Discord channel when complete."
            ),
            retryable=False,
            next_action=(
                "Do not submit a duplicate. Wait for runtime progress, or call "
                "image.status with this job_id when the user asks for an update."
            ),
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
        if job.status in {ImageJobStatus.QUEUED, ImageJobStatus.RUNNING}:
            next_action = "Wait for runtime progress; do not submit a duplicate."
        elif job.status is ImageJobStatus.COMPLETED and not job.delivered:
            next_action = "The runtime is preparing Discord delivery."
        elif job.status is ImageJobStatus.COMPLETED:
            next_action = "No further action is required."
        elif retryable:
            next_action = "Report the failure accurately and retry only when requested."
        else:
            next_action = "Report the failure accurately; do not claim a retry started."
        return ImageStatusResponse(
            job_id=job.job_id,
            status=job.status,
            progress_step=job.progress_step,
            progress_total=job.progress_total,
            delivered=job.delivered,
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
                    "構図や演出を具体化し、ローカルで画像を1枚生成して非同期で届けます。"
                    "短い依頼文を繰り返すのではなく、制作に使える視覚情報を指定します。"
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
                    "ローカルMLXで画像生成を開始します。",
                    "許可されたチャンネルへ進捗と生成画像を投稿します。",
                ),
            ),
            ImageGenerateRequest,
            ImageGenerateResponse,
            generate,
        ),
        endpoint(
            CapabilityDescriptor(
                name="image.status",
                summary="このユーザーが依頼した画像生成ジョブの状態を確認します。",
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("image", "progress", "job", "status", "進捗"),
            ),
            ImageStatusRequest,
            ImageStatusResponse,
            status,
        ),
    )


def _request_event_id(request_id: str) -> str | None:
    prefix = "discord:message:"
    return request_id[len(prefix) :] if request_id.startswith(prefix) else None


def _normalized_reply_event(
    reply_to_event_id: str,
    *,
    context: InvocationContext,
) -> str:
    if reply_to_event_id == context.request_id:
        return _request_event_id(context.request_id) or reply_to_event_id
    return reply_to_event_id
