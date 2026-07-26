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
                summary="List files in this Discord server's isolated agent workspace.",
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("files", "workspace", "attachments", "documents"),
            ),
            FileListRequest,
            FileListResponse,
            list_files,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.read",
                summary=(
                    "Read text or inspect a PDF/ZIP inside the isolated workspace "
                    "with bounded continuation offsets."
                ),
                risk=RiskLevel.READ,
                approval=ApprovalMode.NEVER,
                keywords=("file", "read", "pdf", "zip", "document", "inspect"),
            ),
            FileReadRequest,
            WorkspaceReadResult,
            read_file,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.write_text",
                summary=(
                    "Atomically create or update a UTF-8 text file in the isolated "
                    "workspace, optionally requiring the previous SHA-256."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("file", "write", "create", "edit", "document", "code"),
                side_effects=("Creates or replaces one sandbox text file.",),
            ),
            FileWriteTextRequest,
            WorkspaceFileRecord,
            write_text,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.replace_text",
                summary=(
                    "Replace one unique text occurrence with mandatory SHA-256 "
                    "conflict protection."
                ),
                risk=RiskLevel.WRITE,
                approval=ApprovalMode.NEVER,
                keywords=("file", "edit", "replace", "patch"),
                side_effects=("Atomically edits one sandbox text file.",),
            ),
            FileReplaceTextRequest,
            WorkspaceFileRecord,
            replace_text,
        ),
    )
