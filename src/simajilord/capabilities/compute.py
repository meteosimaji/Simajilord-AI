"""Typed capabilities for isolated workspace computation and downloads."""

from __future__ import annotations

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
from simajilord.services.compute import (
    ComputeRunResult,
    WorkspaceComputeService,
    WorkspaceDownloadResult,
)

from .file_scope import file_provenance, file_workspace_id


@dataclass(frozen=True, slots=True)
class ComputeRunRequest:
    runtime: Literal["python"]
    argv: tuple[str, ...]
    input_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileDownloadUrlRequest:
    url: str
    path: str


def build_compute_endpoints(
    service: WorkspaceComputeService,
) -> tuple[CapabilityEndpoint, ...]:
    def workspace(context: InvocationContext) -> str:
        try:
            return file_workspace_id(context)
        except UserError as exc:
            if exc.code == "files.workspace_required":
                raise UserError("compute.workspace_required") from exc
            raise

    async def run(
        request: ComputeRunRequest,
        context: InvocationContext,
    ) -> ComputeRunResult:
        return await service.run(
            workspace(context),
            runtime=request.runtime,
            argv=request.argv,
            input_paths=request.input_paths,
            actor_id=context.actor_id,
            provenance=file_provenance(context),
        )

    async def download_url(
        request: FileDownloadUrlRequest,
        context: InvocationContext,
    ) -> WorkspaceDownloadResult:
        return await service.download(
            workspace(context),
            url=request.url,
            path=request.path,
            provenance=file_provenance(context),
        )

    return (
        endpoint(
            CapabilityDescriptor(
                name="compute.run",
                summary=(
                    "Run one Python script owned by the current actor in the configured "
                    "workspace. Pass the script in argv[0] and every additional file the "
                    "script may read in input_paths; unspecified workspace files are not "
                    "staged. Shell commands are not accepted."
                ),
                risk=RiskLevel.WRITE,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "compute",
                    "python",
                    "execute",
                    "code",
                    "script",
                    "convert",
                    "transform",
                ),
                side_effects=(
                    "Runs one resource-bounded process without network access.",
                    "Commits validated file changes to the isolated workspace.",
                ),
                requires_workspace=True,
                idempotency="non_idempotent_write",
                expected_errors=(
                    "compute.sandbox_unavailable",
                    "compute.runtime_unsupported",
                    "compute.script_not_found",
                    "compute.input_not_found",
                    "compute.input_limit_invalid",
                    "compute.output_conflict",
                    "compute.timeout",
                    "compute.output_limit",
                    "compute.memory_limit",
                    "compute.workspace_monitor_unavailable",
                    "files.workspace_quota",
                ),
                timeout_seconds=150,
                user_visible_effect=(
                    "May create or update files in the isolated workspace."
                ),
            ),
            ComputeRunRequest,
            ComputeRunResult,
            run,
        ),
        endpoint(
            CapabilityDescriptor(
                name="files.download_url",
                summary=(
                    "Download one bounded public HTTP(S) file into this server's "
                    "isolated workspace with redirect and SSRF protection."
                ),
                risk=RiskLevel.WRITE,
                disclosure_class=DisclosureClass.ACTOR_PRIVATE,
                approval=ApprovalMode.NEVER,
                keywords=(
                    "file",
                    "download",
                    "url",
                    "pdf",
                    "document",
                    "attachment",
                    "workspace",
                ),
                side_effects=(
                    "Fetches one public HTTP or HTTPS resource.",
                    "Creates or replaces a file in the isolated workspace.",
                ),
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=(
                    "files.download_type_unsupported",
                    "files.file_too_large",
                    "files.workspace_quota",
                ),
                timeout_seconds=45,
                user_visible_effect=(
                    "Creates or replaces a downloaded file in the isolated workspace."
                ),
            ),
            FileDownloadUrlRequest,
            WorkspaceDownloadResult,
            download_url,
        ),
    )
