"""Typed capabilities for the isolated agent file workspace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.services.files import (
    AgentFileSandbox,
    WorkspaceFileRecord,
    WorkspaceReadResult,
)

from .file_scope import file_provenance, file_workspace_id


@dataclass(frozen=True, slots=True)
class FileListRequest:
    pass


@dataclass(frozen=True, slots=True)
class FileListResponse:
    files: tuple[WorkspaceFileRecord, ...]


@dataclass(frozen=True, slots=True)
class FileReadRequest:
    path: str
    offset: int = 0
    max_characters: int = 4_000
    page_start: int = 1
    page_count: int = 5


@dataclass(frozen=True, slots=True)
class FileWriteTextRequest:
    path: str
    content: str
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class FileReplaceTextRequest:
    path: str
    old: str
    new: str
    expected_sha256: str


def build_file_endpoints(
    service: AgentFileSandbox,
) -> tuple[CapabilityEndpoint, ...]:
    def workspace(context: InvocationContext) -> str:
        return file_workspace_id(context)

    async def list_files(
        _: FileListRequest,
        context: InvocationContext,
    ) -> FileListResponse:
        return FileListResponse(
            files=await asyncio.to_thread(
                service.list_for_actor,
                workspace(context),
                context.actor_id,
            )
        )

    async def read_file(
        request: FileReadRequest,
        context: InvocationContext,
    ) -> WorkspaceReadResult:
        return await asyncio.to_thread(
            service.read_for_actor,
            workspace(context),
            context.actor_id,
            request.path,
            offset=request.offset,
            max_characters=request.max_characters,
            page_start=request.page_start,
            page_count=request.page_count,
        )

    async def write_text(
        request: FileWriteTextRequest,
        context: InvocationContext,
    ) -> WorkspaceFileRecord:
        workspace_id = workspace(context)
        provenance = file_provenance(context)
        await asyncio.to_thread(
            service.validate_write_text_for_actor,
            workspace_id,
            context.actor_id,
            request.path,
            request.content,
            expected_sha256=request.expected_sha256,
            provenance=provenance,
        )
        await context.dispatch_external_effect()
        return await asyncio.to_thread(
            service.write_text_for_actor,
            workspace_id,
            context.actor_id,
            request.path,
            request.content,
            expected_sha256=request.expected_sha256,
            provenance=provenance,
        )

    async def replace_text(
        request: FileReplaceTextRequest,
        context: InvocationContext,
    ) -> WorkspaceFileRecord:
        workspace_id = workspace(context)
        provenance = file_provenance(context)
        await asyncio.to_thread(
            service.validate_replace_text_for_actor,
            workspace_id,
            context.actor_id,
            request.path,
            request.old,
            request.new,
            expected_sha256=request.expected_sha256,
            provenance=provenance,
        )
        await context.dispatch_external_effect()
        return await asyncio.to_thread(
            service.replace_text_for_actor,
            workspace_id,
            context.actor_id,
            request.path,
            request.old,
            request.new,
            expected_sha256=request.expected_sha256,
            provenance=provenance,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="files.list",
                summary=(
                    "List files owned by the current actor in the configured "
                    "workspace. Other actors and unlabelled legacy files are hidden."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "files",
                    "workspace",
                    "attachments",
                    "documents",
                    "ファイル",
                    "添付",
                    "文書",
                    "一覧",
                ),
                requires_workspace=True,
                expected_errors=("files.workspace_required",),
                timeout_seconds=10,
            ),
            FileListRequest,
            FileListResponse,
            list_files,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.read",
                summary=(
                    "Read text or inspect PDF and ZIP files owned by the current actor "
                    "in the configured workspace. "
                    "Use offset/next_offset within one chunk. For PDF files, use "
                    "page_start/page_count and continue from next_page until complete."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "file",
                    "read",
                    "pdf",
                    "zip",
                    "document",
                    "inspect",
                    "ファイル",
                    "読む",
                    "読み取り",
                    "添付",
                    "文書",
                    "内容",
                ),
                requires_workspace=True,
                expected_errors=(
                    "files.workspace_required",
                    "files.read_range_invalid",
                    "files.page_range_invalid",
                    "files.page_range_unsupported",
                    "files.not_found",
                ),
                timeout_seconds=30,
            ),
            FileReadRequest,
            WorkspaceReadResult,
            read_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.write_text",
                summary=(
                    "Create or update a UTF-8 text file owned by the current actor in "
                    "the configured workspace, "
                    "optionally checking the previous SHA-256."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "file",
                    "write",
                    "create",
                    "edit",
                    "document",
                    "code",
                    "ファイル",
                    "書く",
                    "作成",
                    "編集",
                    "修正",
                    "直して",
                    "文書",
                ),
                side_effects=(
                    "Creates or updates a text file in the isolated workspace.",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.path_conflict",
                    "files.provenance_invalid",
                ),
                timeout_seconds=15,
                user_visible_effect="Creates or updates a file in the isolated workspace.",
            ),
            FileWriteTextRequest,
            WorkspaceFileRecord,
            write_text,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.replace_text",
                summary="Replace one unique text match with optional SHA-256 locking.",
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "file",
                    "edit",
                    "replace",
                    "patch",
                    "ファイル",
                    "編集",
                    "置換",
                    "修正",
                    "書き換え",
                    "直して",
                ),
                side_effects=("Edits a text file in the isolated workspace.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.not_found",
                    "files.path_conflict",
                    "files.provenance_invalid",
                ),
                timeout_seconds=15,
                user_visible_effect="Edits a file in the isolated workspace.",
            ),
            FileReplaceTextRequest,
            WorkspaceFileRecord,
            replace_text,
        ),
    )
