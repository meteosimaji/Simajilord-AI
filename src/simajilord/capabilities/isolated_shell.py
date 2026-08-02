"""A non-interactive shell confined to one Discord actor-and-task workspace."""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import json
import os
import signal
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError

_MAX_ARGUMENTS = 128
_MAX_ARGUMENT_CHARACTERS = 32_000
_MAX_OUTPUT_CHARACTERS = 12_000
_MAX_TIMEOUT_SECONDS = 120
_SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


@dataclass(frozen=True, slots=True)
class IsolatedShellRequest:
    argv: tuple[str, ...] = field(
        metadata={
            "description": (
                "Command and arguments as an array. Do not combine them into an "
                "unquoted command string."
            )
        }
    )
    working_directory: str = field(
        default=".",
        metadata={"description": "Relative directory inside this Discord workspace."},
    )
    timeout_seconds: int = field(default=30)


@dataclass(frozen=True, slots=True)
class IsolatedShellResponse:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool


def discord_workspace_for_context(
    workspace_root: Path,
    context: InvocationContext,
) -> Path:
    """Return one actor-and-task-owned path used by provider and shell."""

    task_scope = context.agent_task_id or context.request_id
    transport_scope = (
        f"guild:{context.workspace_id}"
        if context.workspace_id is not None
        else "direct"
    )
    scope = f"{transport_scope}:actor:{context.actor_id}:task:{task_scope}"
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:24]
    root = workspace_root.expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        root.chmod(0o700)
    workspace = root / digest
    workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        workspace.chmod(0o700)
    return workspace.resolve()


def build_isolated_shell_endpoint(workspace_root: Path) -> CapabilityEndpoint:
    async def run(
        request: IsolatedShellRequest,
        context: InvocationContext,
    ) -> IsolatedShellResponse:
        _validate_request(request)
        workspace = discord_workspace_for_context(workspace_root, context)
        working_directory = _resolve_working_directory(
            workspace,
            request.working_directory,
        )
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
            raise UserError("shell.isolation_unavailable")
        temporary_directory = workspace / ".tmp"
        temporary_directory.mkdir(mode=0o700, exist_ok=True)
        profile = _macos_sandbox_profile(workspace)
        try:
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/sandbox-exec",
                "-p",
                profile,
                "/usr/bin/env",
                "-i",
                f"PATH={_SYSTEM_PATH}",
                f"HOME={workspace}",
                f"TMPDIR={temporary_directory}",
                *request.argv,
                cwd=working_directory,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise UserError("shell.launch_failed") from exc
        if process.stdout is None or process.stderr is None:
            await _terminate_process_group(process)
            raise UserError("shell.launch_failed")
        stdout_task = asyncio.create_task(
            _read_bounded_stream(process.stdout),
            name="simajilord-shell-stdout",
        )
        stderr_task = asyncio.create_task(
            _read_bounded_stream(process.stderr),
            name="simajilord-shell-stderr",
        )
        timed_out = False
        try:
            async with asyncio.timeout(request.timeout_seconds):
                await process.wait()
        except TimeoutError:
            timed_out = True
            await _terminate_process_group(process)
        except BaseException:
            await _terminate_process_group(process)
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        stdout_result, stderr_result = await asyncio.gather(stdout_task, stderr_task)
        stdout, stdout_truncated = stdout_result
        stderr, stderr_truncated = stderr_result
        return IsolatedShellResponse(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    return endpoint(
        CapabilityDescriptor(
            name="system.shell",
            summary=(
                "Run one non-interactive command with no network and filesystem access "
                "limited to this Discord actor and task's dedicated workspace."
            ),
            risk=RiskLevel.WRITE,
            approval=ApprovalMode.WHEN_REQUESTED,
            keywords=("shell", "command", "terminal", "script", "build", "test"),
            side_effects=(
                "May create or modify files only inside the actor-and-task workspace.",
            ),
            requires_workspace=True,
            idempotency="non_idempotent_write",
            expected_errors=(
                "shell.arguments_invalid",
                "shell.timeout_invalid",
                "shell.login_shell_forbidden",
                "shell.working_directory_invalid",
                "shell.isolation_unavailable",
                "shell.launch_failed",
            ),
            timeout_seconds=_MAX_TIMEOUT_SECONDS + 5,
            audit_payload="metadata",
        ),
        IsolatedShellRequest,
        IsolatedShellResponse,
        run,
    )


def _validate_request(request: IsolatedShellRequest) -> None:
    if (
        not request.argv
        or len(request.argv) > _MAX_ARGUMENTS
        or any(not argument or "\x00" in argument for argument in request.argv)
        or sum(len(argument) for argument in request.argv) > _MAX_ARGUMENT_CHARACTERS
    ):
        raise UserError("shell.arguments_invalid")
    if not 1 <= request.timeout_seconds <= _MAX_TIMEOUT_SECONDS:
        raise UserError("shell.timeout_invalid")
    executable = Path(request.argv[0]).name
    if executable in {"bash", "sh", "zsh"} and any(
        argument in {"-l", "--login"} for argument in request.argv[1:]
    ):
        raise UserError("shell.login_shell_forbidden")


def _resolve_working_directory(workspace: Path, value: str) -> Path:
    requested = Path(value)
    if requested.is_absolute() or "\x00" in value:
        raise UserError("shell.working_directory_invalid")
    resolved = (workspace / requested).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise UserError("shell.working_directory_invalid") from exc
    if not resolved.is_dir():
        raise UserError("shell.working_directory_invalid")
    return resolved


def _macos_sandbox_profile(workspace: Path) -> str:
    quoted_workspace = json.dumps(str(workspace))
    return "\n".join(
        (
            "(version 1)",
            "(deny default)",
            '(import "system.sb")',
            "(allow process-exec)",
            "(allow process-fork)",
            "(allow signal (target self))",
            f"(allow file-read* (subpath {quoted_workspace}))",
            f"(allow file-write* (subpath {quoted_workspace}))",
            "(deny network*)",
        )
    )


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def _read_bounded_stream(stream: asyncio.StreamReader) -> tuple[str, bool]:
    """Drain a child pipe while retaining only the bounded response prefix."""

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    chunks: list[str] = []
    retained = 0
    truncated = False

    def accept(text: str) -> None:
        nonlocal retained, truncated
        remaining = max(0, _MAX_OUTPUT_CHARACTERS - retained)
        accepted = text[:remaining]
        if accepted:
            chunks.append(accepted)
            retained += len(accepted)
        if len(accepted) != len(text):
            truncated = True

    while chunk := await stream.read(16 * 1024):
        accept(decoder.decode(chunk))
    accept(decoder.decode(b"", final=True))
    return "".join(chunks), truncated
