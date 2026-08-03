"""Typed capabilities for the configured agent file workspace."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.services.files import (
    AgentFileSandbox,
    WorkspaceFileAction,
    WorkspaceFileRecord,
    WorkspaceManagedFile,
    WorkspaceManagedFileCatalog,
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
class FileCatalogRequest:
    section: Literal["all", "my", "task", "shared"] = "all"
    offset: int = 0
    limit: int = 50


@dataclass(frozen=True, slots=True)
class FileCatalogResponse:
    catalog: WorkspaceManagedFileCatalog
    offset: int
    next_offset: int | None
    complete: bool


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


@dataclass(frozen=True, slots=True)
class FileCopyToTaskRequest:
    file_ref: str
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class FileCopyToTaskResponse:
    file: WorkspaceManagedFile


@dataclass(frozen=True, slots=True)
class FileDeleteRequest:
    file_ref: str
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class FileDeleteResponse:
    file_ref: str
    filename: str
    deleted: bool


@dataclass(frozen=True, slots=True)
class FileHistoryRequest:
    file_ref: str
    limit: int = 20


@dataclass(frozen=True, slots=True)
class FileHistoryResponse:
    file_ref: str
    actions: tuple[WorkspaceFileAction, ...]


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

    async def catalog_files(
        request: FileCatalogRequest,
        context: InvocationContext,
    ) -> FileCatalogResponse:
        if request.section not in {"all", "my", "task", "shared"}:
            raise UserError("files.catalog_section_invalid")
        if request.offset < 0 or not 1 <= request.limit <= 100:
            raise UserError("files.catalog_page_invalid")
        guild_id = context.workspace_id
        if guild_id is None:
            raise UserError("files.workspace_required")
        catalog = await asyncio.to_thread(
            service.managed_catalog_for_actor,
            guild_id,
            context.actor_id,
            current_task_id=context.agent_task_id,
        )
        filtered = (
            catalog.files
            if request.section == "all"
            else tuple(
                item
                for item in catalog.files
                if item.section == request.section
            )
        )
        page = filtered[request.offset : request.offset + request.limit]
        next_offset = (
            request.offset + len(page)
            if request.offset + len(page) < len(filtered)
            else None
        )
        return FileCatalogResponse(
            catalog=WorkspaceManagedFileCatalog(
                files=page,
                my_count=catalog.my_count,
                task_count=catalog.task_count,
                shared_count=catalog.shared_count,
            ),
            offset=request.offset,
            next_offset=next_offset,
            complete=next_offset is None,
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

    async def copy_to_task(
        request: FileCopyToTaskRequest,
        context: InvocationContext,
    ) -> FileCopyToTaskResponse:
        guild_id = context.workspace_id
        task_id = context.agent_task_id
        if guild_id is None:
            raise UserError("files.workspace_required")
        if task_id is None:
            raise UserError("files.task_required")
        location = await asyncio.to_thread(
            service.resolve_managed_file_for_actor,
            request.file_ref,
            context.actor_id,
            guild_id,
        )
        if location.record.sha256 != request.expected_sha256:
            raise UserError("files.hash_conflict")
        await context.dispatch_external_effect()
        copied = await asyncio.to_thread(
            service.copy_managed_file_to_task_for_actor,
            request.file_ref,
            context.actor_id,
            guild_id,
            expected_sha256=request.expected_sha256,
            current_task_id=task_id,
            target_workspace_id=workspace(context),
        )
        if copied.file_ref is None:
            raise UserError("files.provenance_invalid")
        managed = await asyncio.to_thread(
            service.managed_file_for_actor,
            copied.file_ref,
            context.actor_id,
            guild_id,
            current_task_id=task_id,
        )
        return FileCopyToTaskResponse(file=managed)

    async def delete_file(
        request: FileDeleteRequest,
        context: InvocationContext,
    ) -> FileDeleteResponse:
        guild_id = context.workspace_id
        if guild_id is None:
            raise UserError("files.workspace_required")
        location = await asyncio.to_thread(
            service.resolve_managed_file_for_actor,
            request.file_ref,
            context.actor_id,
            guild_id,
        )
        if location.record.sha256 != request.expected_sha256:
            raise UserError("files.hash_conflict")
        await context.dispatch_external_effect()
        filename = await asyncio.to_thread(
            service.delete_managed_file_for_actor,
            request.file_ref,
            context.actor_id,
            guild_id,
            expected_sha256=request.expected_sha256,
        )
        return FileDeleteResponse(
            file_ref=request.file_ref,
            filename=filename,
            deleted=True,
        )

    async def file_history(
        request: FileHistoryRequest,
        context: InvocationContext,
    ) -> FileHistoryResponse:
        guild_id = context.workspace_id
        if guild_id is None:
            raise UserError("files.workspace_required")
        actions = await asyncio.to_thread(
            service.managed_history_for_actor,
            request.file_ref,
            context.actor_id,
            guild_id,
            limit=request.limit,
        )
        return FileHistoryResponse(file_ref=request.file_ref, actions=actions)

    return (
        endpoint(
            CapabilityDescriptor(
                name="files.catalog",
                summary=(
                    "List requester-private My, current Task, and explicitly Shared "
                    "files using opaque file references. The response never exposes raw "
                    "paths, actor IDs, or internal workspace identifiers."
                    " Use offset/next_offset to continue a bounded section page."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "files",
                    "file manager",
                    "my files",
                    "task files",
                    "shared files",
                    "ファイル管理",
                    "自分のファイル",
                    "タスクファイル",
                    "共有ファイル",
                ),
                requires_workspace=True,
                expected_errors=(
                    "files.workspace_required",
                    "files.catalog_section_invalid",
                    "files.catalog_page_invalid",
                ),
                timeout_seconds=15,
            ),
            FileCatalogRequest,
            FileCatalogResponse,
            catalog_files,
        ),
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
                    "Creates or updates a text file in the configured workspace.",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.path_conflict",
                    "files.provenance_invalid",
                ),
                timeout_seconds=15,
                user_visible_effect="Creates or updates a file in the configured workspace.",
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
                side_effects=("Edits a text file in the configured workspace.",),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.not_found",
                    "files.path_conflict",
                    "files.provenance_invalid",
                ),
                timeout_seconds=15,
                user_visible_effect="Edits a file in the configured workspace.",
            ),
            FileReplaceTextRequest,
            WorkspaceFileRecord,
            replace_text,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.copy_to_task",
                summary=(
                    "Copy one requester-owned file selected by opaque file_ref into the "
                    "current task in the configured workspace."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "files",
                    "copy to task",
                    "task copy",
                    "タスクへコピー",
                    "タスクファイル",
                ),
                side_effects=("Creates one private task copy in the configured workspace.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.task_required",
                    "files.file_ref_not_found",
                    "files.hash_conflict",
                ),
                timeout_seconds=15,
                user_visible_effect="Creates one private copy for the current task.",
            ),
            FileCopyToTaskRequest,
            FileCopyToTaskResponse,
            copy_to_task,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.delete",
                summary=(
                    "Permanently delete one requester-owned private file selected by "
                    "opaque file_ref and exact SHA-256. Shared publication copies must "
                    "be revoked instead."
                ),
                risk=RiskLevel.DESTRUCTIVE,
                approval=ApprovalMode.WHEN_REQUESTED,
                keywords=(
                    "files",
                    "delete file",
                    "remove file",
                    "ファイル削除",
                    "ファイルを消す",
                ),
                side_effects=("Permanently deletes one private workspace file.",),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "files.workspace_required",
                    "files.file_ref_not_found",
                    "files.hash_conflict",
                ),
                timeout_seconds=15,
                user_visible_effect="Permanently deletes one selected private file.",
            ),
            FileDeleteRequest,
            FileDeleteResponse,
            delete_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.history",
                summary=(
                    "Show body-free copy, publish, send, delete, and revoke history for "
                    "one requester-owned opaque file reference."
                ),
                risk=RiskLevel.READ,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "files",
                    "file history",
                    "share history",
                    "ファイル履歴",
                    "共有履歴",
                ),
                requires_workspace=True,
                expected_errors=(
                    "files.workspace_required",
                    "files.file_ref_not_found",
                    "files.history_limit_invalid",
                ),
                timeout_seconds=10,
            ),
            FileHistoryRequest,
            FileHistoryResponse,
            file_history,
        ),
    )
