"""Typed capabilities for the isolated agent file workspace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.files import (
    AgentFileSandbox,
    WorkspaceFileRecord,
    WorkspaceReadResult,
)


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
        if context.workspace_id is None:
            raise UserError("files.workspace_required")
        return context.workspace_id

    async def list_files(
        _: FileListRequest,
        context: InvocationContext,
    ) -> FileListResponse:
        return FileListResponse(
            files=await asyncio.to_thread(service.list, workspace(context))
        )

    async def read_file(
        request: FileReadRequest,
        context: InvocationContext,
    ) -> WorkspaceReadResult:
        return await asyncio.to_thread(
            service.read,
            workspace(context),
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
        return await asyncio.to_thread(
            service.write_text,
            workspace(context),
            request.path,
            request.content,
            expected_sha256=request.expected_sha256,
        )

    async def replace_text(
        request: FileReplaceTextRequest,
        context: InvocationContext,
    ) -> WorkspaceFileRecord:
        return await asyncio.to_thread(
            service.replace_text,
            workspace(context),
            request.path,
            request.old,
            request.new,
            expected_sha256=request.expected_sha256,
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="files.list",
                summary="List files in this Discord server's isolated workspace.",
                risk=RiskLevel.READ,
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
                    "Read text or inspect PDF and ZIP files in the isolated workspace. "
                    "Use offset/next_offset within one chunk. For PDF files, use "
                    "page_start/page_count and continue from next_page until complete."
                ),
                risk=RiskLevel.READ,
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
                    "Create or update a UTF-8 text file in the isolated workspace, "
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
                expected_errors=("files.workspace_required",),
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
                expected_errors=("files.workspace_required",),
                timeout_seconds=15,
                user_visible_effect="Edits a file in the isolated workspace.",
            ),
            FileReplaceTextRequest,
            WorkspaceFileRecord,
            replace_text,
        ),
    )
