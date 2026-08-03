"""Media download capability returning an artifact to the calling adapter."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from simajilord.core.capabilities import (
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.domain.media import DownloadFormat
from simajilord.services.files import AgentFileSandbox, WorkspaceFileRecord
from simajilord.services.media import MediaPriority, MediaService

from .file_scope import file_provenance, file_workspace_id


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    url: str
    media_type: DownloadFormat
    destination: Path
    max_bytes: int


@dataclass(frozen=True, slots=True)
class DownloadResponse:
    path: Path
    title: str
    size_bytes: int
    source_url: str


@dataclass(frozen=True, slots=True)
class MediaSaveRequest:
    url: str
    media_type: DownloadFormat = DownloadFormat.VIDEO
    max_items: int = 4


@dataclass(frozen=True, slots=True)
class MediaSaveResponse:
    source_url: str
    files: tuple[WorkspaceFileRecord, ...]
    skipped_items: int
    partial: bool


def build_download_endpoint(media: MediaService) -> CapabilityEndpoint:
    async def download(
        request: DownloadRequest,
        context: InvocationContext,
    ) -> DownloadResponse:
        workspace_id = context.workspace_id or f"actor:{context.actor_id}"
        artifact = await media.download(
            request.url,
            request.media_type,
            request.destination,
            max_bytes=request.max_bytes,
            workspace_id=workspace_id,
            priority=MediaPriority.NORMAL,
        )
        return DownloadResponse(
            path=artifact.path,
            title=artifact.title,
            size_bytes=artifact.size_bytes,
            source_url=artifact.source_url,
        )

    return endpoint(
        CapabilityDescriptor(
            name="media.download",
            summary="Download a supported public media URL to a bounded destination.",
            risk=RiskLevel.EXTERNAL,
            keywords=("media", "download", "video", "audio", "save", "attachment"),
            side_effects=(
                "Connects to a public media service.",
                "Creates a temporary local file.",
            ),
        ),
        DownloadRequest,
        DownloadResponse,
        download,
    )


def build_media_save_endpoint(
    media: MediaService,
    files: AgentFileSandbox,
) -> CapabilityEndpoint:
    """Save public media into the caller's isolated file workspace."""

    async def save(
        request: MediaSaveRequest,
        context: InvocationContext,
    ) -> MediaSaveResponse:
        if context.workspace_id is None:
            raise UserError("media.workspace_required")
        if not 1 <= request.max_items <= 4:
            raise UserError("media.item_limit_invalid")
        temporary = Path(tempfile.mkdtemp(prefix="simajilord-media-"))
        scoped_workspace_id = file_workspace_id(context)
        try:
            batch = await media.download_many(
                request.url,
                request.media_type,
                temporary,
                max_bytes=files.max_file_bytes,
                max_items=request.max_items,
                workspace_id=context.workspace_id,
                priority=MediaPriority.NORMAL,
            )
            pending_files: list[tuple[str, bytes]] = []
            for artifact in batch.artifacts:
                content = await asyncio.to_thread(artifact.path.read_bytes)
                digest = hashlib.sha256(content).hexdigest()
                filename = _bounded_media_filename(artifact.path.name)
                pending_files.append(
                    (f"media/{digest[:16]}-{filename}", content)
                )
            records = await asyncio.to_thread(
                files.import_batch,
                scoped_workspace_id,
                tuple(pending_files),
                provenance=file_provenance(context),
            )
            return MediaSaveResponse(
                source_url=batch.artifacts[0].source_url,
                files=records,
                skipped_items=batch.skipped_items,
                partial=batch.partial,
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, temporary, True)

    return endpoint(
        CapabilityDescriptor(
            name="media.save",
            summary=(
                "Save video or audio from a supported public URL into the isolated "
                "actor/task workspace for later file use or Discord delivery."
            ),
            risk=RiskLevel.WRITE,
            keywords=(
                "media",
                "download",
                "save",
                "DL",
                "ダウンロード",
                "保存",
                "public URL",
                "video",
                "動画",
                "audio",
                "音声",
                "social post",
                "YouTube",
                "TikTok",
                "X",
                "Twitter",
            ),
            side_effects=(
                "Connects to the public media URL.",
                "Creates up to four bounded files in the isolated workspace.",
            ),
            requires_workspace=True,
            idempotency="idempotent_write",
            expected_errors=(
                "media.url_unsupported",
                "media.url_private",
                "media.cookie_required",
                "media.extractor_challenge",
                "media.rate_limited",
                "media.too_large",
                "media.workspace_required",
                "media.item_limit_invalid",
                "files.workspace_quota",
            ),
            timeout_seconds=180,
            user_visible_effect=(
                "Saves reusable media files in this server's isolated workspace."
            ),
        ),
        MediaSaveRequest,
        MediaSaveResponse,
        save,
    )


def _bounded_media_filename(value: str) -> str:
    path = Path(value)
    suffix = path.suffix[:16]
    maximum_stem = max(1, 140 - len(suffix))
    stem = path.stem[:maximum_stem].rstrip(" .") or "media"
    return f"{stem}{suffix}"
