"""OS-isolated code execution and public-file import for agent workspaces."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Protocol

from simajilord.core.errors import UserError
from simajilord.providers.web.base import PublicWebFetcher
from simajilord.providers.web.http import normalize_public_web_url

from .files import (
    AgentFileSandbox,
    WorkspaceFileProvenance,
    WorkspaceFileRecord,
    file_provenance_is_owned_by,
    merge_file_provenances,
    workspace_file_kind,
)

_MACOS_SANDBOX_EXECUTABLE = Path("/usr/bin/sandbox-exec")
_MACOS_OTOOL_EXECUTABLE = Path("/usr/bin/otool")
_MACOS_PRIVATE_READ_ROOTS = (
    Path("/Users"),
    Path("/Volumes"),
    Path("/Network"),
    Path("/private"),
    Path("/opt"),
    Path("/srv"),
)
_MAX_RUNTIME_MACHO_FILES = 2_048
_OTOOL_BATCH_SIZE = 128
_MAX_ARGUMENTS = 32
_MAX_ARGUMENT_CHARACTERS = 8_192
_MAX_SINGLE_ARGUMENT_CHARACTERS = 1_024
_MAX_INPUT_FILES = 32
_SUPPORTED_DOWNLOAD_CONTENT_TYPES = frozenset(
    {
        "application/epub+zip",
        "application/gzip",
        "application/json",
        "application/msword",
        "application/octet-stream",
        "application/pdf",
        "application/rtf",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.oasis.opendocument.presentation",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/x-gzip",
        "application/x-tar",
        "application/x-zip-compressed",
        "application/xml",
        "application/zip",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
_RUSAGE_INFO_V0 = 0


@dataclass(frozen=True, slots=True)
class ComputeLimits:
    """Host-enforced bounds for one isolated process."""

    timeout_seconds: float = 120.0
    cpu_seconds: int = 60
    memory_bytes: int = 512 * 1024 * 1024
    output_bytes: int = 4_096
    open_files: int = 64

    def __post_init__(self) -> None:
        if not 1.0 <= self.timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if not 1 <= self.cpu_seconds <= math.ceil(self.timeout_seconds):
            raise ValueError("cpu_seconds must be positive and no greater than timeout")
        if not 64 * 1024 * 1024 <= self.memory_bytes <= 2 * 1024 * 1024 * 1024:
            raise ValueError("memory_bytes must be between 64 MiB and 2 GiB")
        if not 4_096 <= self.output_bytes <= 1024 * 1024:
            raise ValueError("output_bytes must be between 4 KiB and 1 MiB")
        if not 16 <= self.open_files <= 256:
            raise ValueError("open_files must be between 16 and 256")


@dataclass(frozen=True, slots=True)
class ComputeProcessResult:
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ComputeRunResult:
    runtime: str
    argv: tuple[str, ...]
    exit_code: int
    duration_ms: float
    changed_files: tuple[WorkspaceFileRecord, ...]
    stdout: str
    stderr: str
    provenance: WorkspaceFileProvenance | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceDownloadResult:
    file: WorkspaceFileRecord
    source_url: str
    content_type: str


class SandboxedProcessLauncher(Protocol):
    """Port for a launcher that can enforce filesystem and network isolation."""

    @property
    def available(self) -> bool: ...

    async def run_python(
        self,
        *,
        workspace: Path,
        temporary_directory: Path,
        argv: tuple[str, ...],
        limits: ComputeLimits,
        max_file_bytes: int,
        max_workspace_bytes: int,
        max_files: int,
    ) -> ComputeProcessResult: ...


def _sandbox_read_metadata_ancestors(
    paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Return exact private ancestors needed to resolve allowed paths."""

    ancestors: list[Path] = []
    for path in paths:
        absolute_path = path.absolute()
        for private_root in _MACOS_PRIVATE_READ_ROOTS:
            try:
                absolute_path.relative_to(private_root)
            except ValueError:
                continue
            current = absolute_path.parent
            while current != current.parent:
                ancestors.append(current)
                if current == private_root:
                    break
                current = current.parent
            break
    return tuple(dict.fromkeys(ancestors))


def _parse_otool_dependencies(output: str) -> tuple[Path, ...]:
    dependencies: list[Path] = []
    for line in output.splitlines():
        if not line[:1].isspace():
            continue
        dependency = line.strip().split(" (compatibility version", 1)[0]
        path = Path(dependency)
        if path.is_absolute():
            dependencies.append(path)
    return tuple(dict.fromkeys(dependencies))


def _path_is_within(path: Path, roots: Sequence[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


@lru_cache(maxsize=8)
def _macos_runtime_dependency_files(
    python_executable: Path,
    runtime_read_roots: tuple[Path, ...],
    dependency_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Discover exact external Mach-O files needed by the Python runtime."""

    if (
        sys.platform != "darwin"
        or not _MACOS_OTOOL_EXECUTABLE.is_file()
        or not os.access(_MACOS_OTOOL_EXECUTABLE, os.X_OK)
    ):
        return ()

    candidates = {python_executable.resolve()}
    for root in (*runtime_read_roots, *dependency_paths):
        if not root.is_dir():
            continue
        for pattern in ("*.so", "*.dylib"):
            for candidate in root.rglob(pattern):
                if candidate.is_file():
                    candidates.add(candidate.resolve())
                    if len(candidates) > _MAX_RUNTIME_MACHO_FILES:
                        return ()

    allowed_roots = tuple(
        dict.fromkeys((*runtime_read_roots, *dependency_paths))
    )
    pending = sorted(candidates, key=str)
    examined: set[Path] = set()
    external_files: list[Path] = []
    external_file_set: set[Path] = set()
    while pending:
        batch: list[Path] = []
        while pending and len(batch) < _OTOOL_BATCH_SIZE:
            candidate = pending.pop()
            if candidate in examined:
                continue
            examined.add(candidate)
            batch.append(candidate)
        if not batch:
            continue
        try:
            result = subprocess.run(
                (
                    str(_MACOS_OTOOL_EXECUTABLE),
                    "-L",
                    *(str(path) for path in batch),
                ),
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return tuple(external_files)
        for dependency in _parse_otool_dependencies(result.stdout):
            if _path_is_within(dependency, allowed_roots):
                continue
            try:
                resolved_dependency = dependency.resolve(strict=True)
            except OSError:
                continue
            if not resolved_dependency.is_file():
                continue
            for path in (dependency, resolved_dependency):
                if path in external_file_set:
                    continue
                external_file_set.add(path)
                external_files.append(path)
                if len(external_files) > _MAX_RUNTIME_MACHO_FILES:
                    return ()
            if resolved_dependency not in examined:
                pending.append(resolved_dependency)
    return tuple(external_files)


class MacOSSandboxedPythonLauncher:
    """Run one Python process under the macOS Seatbelt sandbox."""

    def __init__(
        self,
        *,
        python_executable: Path | None = None,
        sandbox_executable: Path = _MACOS_SANDBOX_EXECUTABLE,
    ) -> None:
        executable = (python_executable or Path(sys.executable)).absolute()
        self.python_executable = executable
        self._resolved_python_executable = executable.resolve()
        self.sandbox_executable = sandbox_executable
        self._runtime_read_roots = tuple(
            dict.fromkeys(
                path.resolve()
                for path in (
                    Path(sys.prefix),
                    Path(sys.base_prefix),
                    self._resolved_python_executable.parent.parent,
                )
            )
        )
        interpreter_paths = sysconfig.get_paths()
        self._dependency_paths = tuple(
            dict.fromkeys(
                Path(value).resolve()
                for name in ("purelib", "platlib")
                if (value := interpreter_paths.get(name))
            )
        )
        self._runtime_dependency_read_roots = tuple(
            dict.fromkeys(
                path.parent
                for path in _macos_runtime_dependency_files(
                    self._resolved_python_executable,
                    self._runtime_read_roots,
                    self._dependency_paths,
                )
            )
        )

    @property
    def available(self) -> bool:
        return (
            sys.platform == "darwin"
            and self.sandbox_executable.is_file()
            and os.access(self.sandbox_executable, os.X_OK)
            and self.python_executable.is_file()
            and os.access(self.python_executable, os.X_OK)
        )

    async def run_python(
        self,
        *,
        workspace: Path,
        temporary_directory: Path,
        argv: tuple[str, ...],
        limits: ComputeLimits,
        max_file_bytes: int,
        max_workspace_bytes: int,
        max_files: int,
    ) -> ComputeProcessResult:
        if not self.available:
            raise UserError("compute.sandbox_unavailable")
        script = workspace.joinpath(*PurePosixPath(argv[0]).parts)
        profile = self._profile(
            workspace=workspace,
            temporary_directory=temporary_directory,
        )
        bootstrap_paths = (
            str(workspace),
            *(str(path) for path in self._dependency_paths),
        )
        bootstrap = (
            "import resource,runpy,sys;"
            "resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
            "resource.setrlimit("
            f"resource.RLIMIT_CPU,({limits.cpu_seconds},{limits.cpu_seconds})"
            ");"
            "resource.setrlimit("
            f"resource.RLIMIT_FSIZE,({max_file_bytes},{max_file_bytes})"
            ");"
            "resource.setrlimit("
            f"resource.RLIMIT_NOFILE,({limits.open_files},{limits.open_files})"
            ");"
            f"sys.path[:0]={bootstrap_paths!r};"
            "script=sys.argv.pop(1);"
            "sys.argv[0]=script;"
            "runpy.run_path(script,run_name='__main__')"
        )
        command = (
            str(self.sandbox_executable),
            "-p",
            profile,
            str(self._resolved_python_executable),
            "-I",
            "-B",
            "-c",
            bootstrap,
            str(script),
            *argv[1:],
        )
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(temporary_directory),
        }
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if process.stdout is None or process.stderr is None:
            await _terminate_process(process)
            raise UserError("compute.launch_failed")

        output_budget = _OutputBudget(limits.output_bytes)
        stdout_task = asyncio.create_task(
            _read_bounded_stream(process.stdout, output_budget),
            name="simajilord-compute-stdout",
        )
        stderr_task = asyncio.create_task(
            _read_bounded_stream(process.stderr, output_budget),
            name="simajilord-compute-stderr",
        )
        wait_task = asyncio.create_task(
            process.wait(),
            name="simajilord-compute-process",
        )
        memory_task = asyncio.create_task(
            _monitor_macos_memory(process.pid, limits.memory_bytes),
            name="simajilord-compute-memory-limit",
        )
        disk_task = asyncio.create_task(
            _monitor_workspace_usage(
                (workspace, temporary_directory),
                max_file_bytes=max_file_bytes,
                max_workspace_bytes=max_workspace_bytes,
                max_files=max_files,
            ),
            name="simajilord-compute-disk-limit",
        )
        exceeded_task = asyncio.create_task(
            output_budget.exceeded.wait(),
            name="simajilord-compute-output-limit",
        )
        try:
            done, _ = await asyncio.wait(
                (wait_task, memory_task, disk_task, exceeded_task),
                timeout=limits.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await _terminate_process(process)
                raise UserError("compute.timeout")
            if output_budget.exceeded.is_set():
                await _terminate_process(process)
                raise UserError("compute.output_limit")
            if memory_task in done:
                memory_outcome = memory_task.result()
                await _terminate_process(process)
                if memory_outcome == "exceeded":
                    raise UserError("compute.memory_limit")
                raise UserError("compute.memory_monitor_unavailable")
            if disk_task in done:
                disk_outcome = disk_task.result()
                await _terminate_process(process)
                raise UserError(disk_outcome)
            await wait_task
            stdout_bytes, stderr_bytes = await asyncio.gather(
                stdout_task,
                stderr_task,
            )
            if output_budget.exceeded.is_set():
                raise UserError("compute.output_limit")
        finally:
            for task in (
                stdout_task,
                stderr_task,
                wait_task,
                memory_task,
                disk_task,
                exceeded_task,
            ):
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                stdout_task,
                stderr_task,
                wait_task,
                memory_task,
                disk_task,
                exceeded_task,
                return_exceptions=True,
            )
            if process.returncode is None:
                await _terminate_process(process)
        return ComputeProcessResult(
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    def _profile(
        self,
        *,
        workspace: Path,
        temporary_directory: Path,
    ) -> str:
        read_denials = " ".join(
            f'(subpath "{_sandbox_string(path)}")'
            for path in _MACOS_PRIVATE_READ_ROOTS
        )
        read_roots = (
            workspace.resolve(),
            temporary_directory.resolve(),
            *self._runtime_read_roots,
            *self._runtime_dependency_read_roots,
        )
        read_subpath_exceptions = " ".join(
            f'(subpath "{_sandbox_string(path)}")'
            for path in read_roots
        )
        metadata_exceptions = " ".join(
            f'(literal "{_sandbox_string(path)}")'
            for path in _sandbox_read_metadata_ancestors(
                read_roots
            )
        )
        process_root = self._resolved_python_executable.parent.parent
        return (
            "(version 1)"
            "(allow default)"
            "(deny network*)"
            "(deny process-fork)"
            "(deny process-exec)"
            f'(allow process-exec (subpath "{_sandbox_string(process_root)}"))'
            "(deny file-write*)"
            f'(allow file-write* (subpath "{_sandbox_string(workspace)}") '
            f'(subpath "{_sandbox_string(temporary_directory)}"))'
            f"(deny file-read* {read_denials})"
            f"(allow file-read* {read_subpath_exceptions})"
            f"(allow file-read-metadata {metadata_exceptions})"
        )


class WorkspaceComputeService:
    """Stage, execute, validate, and commit code inside one server workspace."""

    def __init__(
        self,
        *,
        files: AgentFileSandbox,
        run_root: Path,
        web_fetcher: PublicWebFetcher,
        launcher: SandboxedProcessLauncher | None = None,
        limits: ComputeLimits | None = None,
        max_download_bytes: int | None = None,
        max_concurrent_processes: int = 2,
    ) -> None:
        if not 1 <= max_concurrent_processes <= 8:
            raise ValueError("max_concurrent_processes must be between 1 and 8")
        self.files = files
        self.run_root = run_root.resolve()
        self.run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.run_root.chmod(0o700)
        self.web_fetcher = web_fetcher
        self.launcher = launcher or MacOSSandboxedPythonLauncher()
        self.limits = limits or ComputeLimits()
        self.max_download_bytes = min(
            files.max_file_bytes,
            max_download_bytes or files.max_file_bytes,
        )
        self._process_slots = asyncio.Semaphore(max_concurrent_processes)
        self._workspace_locks: dict[str, asyncio.Lock] = {}

    @property
    def available(self) -> bool:
        return self.launcher.available

    async def run(
        self,
        workspace_id: str,
        *,
        runtime: str,
        argv: tuple[str, ...],
        input_paths: tuple[str, ...] = (),
        actor_id: str | None = None,
        provenance: WorkspaceFileProvenance | None = None,
        before_process: Callable[[], Awaitable[None]] | None = None,
    ) -> ComputeRunResult:
        if runtime != "python":
            raise UserError("compute.runtime_unsupported")
        normalized_argv = _validated_argv(argv)
        normalized_inputs = _validated_input_paths(
            input_paths,
            script_path=normalized_argv[0],
        )
        staged_paths = (normalized_argv[0], *normalized_inputs)
        lock = self._workspace_locks.setdefault(workspace_id, asyncio.Lock())
        async with lock:
            run_directory = Path(
                tempfile.mkdtemp(
                    prefix="simajilord-",
                    dir=self.run_root,
                )
            )
            workspace = run_directory / "workspace"
            temporary_directory = run_directory / "tmp"
            workspace.mkdir(mode=0o700)
            temporary_directory.mkdir(mode=0o700)
            try:
                before = await asyncio.to_thread(
                    self._stage_workspace,
                    workspace_id,
                    workspace,
                    staged_paths,
                    actor_id,
                )
                effective_provenance = merge_file_provenances(
                    (
                        provenance,
                        *(record.provenance for record in before.values()),
                    )
                )
                script = workspace.joinpath(*PurePosixPath(normalized_argv[0]).parts)
                if script.is_symlink() or not script.is_file():
                    raise UserError("compute.script_not_found")
                started = monotonic()
                async with self._process_slots:
                    if before_process is not None:
                        await before_process()
                    process_result = await self.launcher.run_python(
                        workspace=workspace,
                        temporary_directory=temporary_directory,
                        argv=normalized_argv,
                        limits=self.limits,
                        max_file_bytes=self.files.max_file_bytes,
                        max_workspace_bytes=self.files.max_workspace_bytes,
                        max_files=self.files.max_files,
                    )
                duration_ms = (monotonic() - started) * 1_000
                changed_files: tuple[WorkspaceFileRecord, ...] = ()
                if process_result.exit_code == 0:
                    changed_files = await asyncio.to_thread(
                        self._validate_and_commit,
                        workspace_id,
                        workspace,
                        before,
                        effective_provenance,
                    )
                return ComputeRunResult(
                    runtime=runtime,
                    argv=normalized_argv,
                    exit_code=process_result.exit_code,
                    duration_ms=duration_ms,
                    changed_files=changed_files,
                    stdout=process_result.stdout,
                    stderr=process_result.stderr,
                    provenance=effective_provenance,
                )
            finally:
                await asyncio.to_thread(
                    _remove_run_directory,
                    run_directory,
                )

    async def download(
        self,
        workspace_id: str,
        *,
        url: str,
        path: str,
        provenance: WorkspaceFileProvenance | None = None,
        before_fetch: Callable[[], Awaitable[None]] | None = None,
    ) -> WorkspaceDownloadResult:
        self.files.validate_path(workspace_id, path)
        normalized_url = normalize_public_web_url(url)
        if before_fetch is not None:
            await before_fetch()
        resource_result = await self.web_fetcher.fetch(
            normalized_url,
            max_bytes=self.max_download_bytes,
        )
        content_type = resource_result.content_type.lower().strip()
        if not _supported_download_content_type(content_type):
            raise UserError(
                "files.download_type_unsupported",
                content_type=content_type,
            )
        record = await asyncio.to_thread(
            self.files.import_bytes,
            workspace_id,
            path,
            resource_result.body,
            provenance=provenance,
        )
        return WorkspaceDownloadResult(
            file=record,
            source_url=resource_result.final_url,
            content_type=content_type,
        )

    def _stage_workspace(
        self,
        workspace_id: str,
        destination: Path,
        relative_paths: tuple[str, ...],
        actor_id: str | None,
    ) -> dict[str, WorkspaceFileRecord]:
        with self.files.locked_workspace(workspace_id):
            available = {
                record.path: record
                for record in self.files.list(workspace_id)
            }
            records: dict[str, WorkspaceFileRecord] = {}
            for index, relative_path in enumerate(relative_paths):
                self.files.validate_path(workspace_id, relative_path)
                record = available.get(relative_path)
                if record is None or (
                    actor_id is not None
                    and not file_provenance_is_owned_by(
                        record.provenance,
                        actor_id,
                    )
                ):
                    raise UserError(
                        "compute.script_not_found"
                        if index == 0
                        else "compute.input_not_found"
                    )
                records[relative_path] = record
            for relative_path, record in records.items():
                source = self.files.path_for_delivery(workspace_id, relative_path)
                target = destination.joinpath(*PurePosixPath(relative_path).parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o600)
                if (
                    target.stat().st_size != record.size_bytes
                    or hashlib.sha256(target.read_bytes()).hexdigest()
                    != record.sha256
                ):
                    raise UserError("compute.workspace_changed")
            return records

    def _validate_and_commit(
        self,
        workspace_id: str,
        workspace: Path,
        before: dict[str, WorkspaceFileRecord],
        provenance: WorkspaceFileProvenance | None,
    ) -> tuple[WorkspaceFileRecord, ...]:
        after = self._scan_staged_workspace(workspace_id, workspace)
        with self.files.locked_workspace(workspace_id):
            current = {
                record.path: record
                for record in self.files.list(workspace_id)
            }
            for path, staged_record in before.items():
                current_record = current.get(path)
                if (
                    current_record is None
                    or current_record.sha256 != staged_record.sha256
                ):
                    raise UserError("compute.workspace_changed")
            removed = set(before) - set(after)
            if removed:
                raise UserError("compute.file_deletion_unsupported")
            changed_paths = [
                path
                for path, record in after.items()
                if path not in before or before[path].sha256 != record.sha256
            ]
            if any(
                path not in before and path in current
                for path in changed_paths
            ):
                raise UserError("compute.output_conflict")
            changed_paths.sort(
                key=lambda path: (
                    after[path].size_bytes
                    - (before[path].size_bytes if path in before else 0),
                    path,
                )
            )
            return self.files.import_batch(
                workspace_id,
                tuple(
                    (
                        relative_path,
                        workspace.joinpath(
                            *PurePosixPath(relative_path).parts
                        ).read_bytes(),
                    )
                    for relative_path in changed_paths
                ),
                provenance=provenance,
            )

    def _scan_staged_workspace(
        self,
        workspace_id: str,
        workspace: Path,
    ) -> dict[str, WorkspaceFileRecord]:
        records: dict[str, WorkspaceFileRecord] = {}
        total_bytes = 0
        entry_count = 0
        for path in sorted(workspace.rglob("*")):
            if path.is_symlink():
                raise UserError("files.symlink_forbidden")
            entry_count += 1
            if entry_count > self.files.max_files:
                raise UserError("files.file_count_limit")
            if path.is_dir():
                continue
            if not path.is_file():
                raise UserError("compute.special_file_forbidden")
            relative_path = path.relative_to(workspace).as_posix()
            self.files.validate_path(workspace_id, relative_path)
            pure = PurePosixPath(relative_path)
            if pure.is_absolute() or ".." in pure.parts:
                raise UserError("files.path_invalid")
            size_bytes = path.stat().st_size
            if size_bytes > self.files.max_file_bytes:
                raise UserError("files.file_too_large")
            total_bytes += size_bytes
            if total_bytes > self.files.max_workspace_bytes:
                raise UserError("files.workspace_quota")
            data = path.read_bytes()
            records[relative_path] = WorkspaceFileRecord(
                path=relative_path,
                size_bytes=size_bytes,
                sha256=hashlib.sha256(data).hexdigest(),
                kind=workspace_file_kind(path, data),
            )
        return records


class _OutputBudget:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.used = 0
        self.exceeded = asyncio.Event()

    def accept(self, chunk: bytes) -> bytes:
        remaining = max(0, self.maximum - self.used)
        accepted = chunk[:remaining]
        self.used += len(accepted)
        if len(accepted) != len(chunk):
            self.exceeded.set()
        return accepted


async def _read_bounded_stream(
    stream: asyncio.StreamReader,
    budget: _OutputBudget,
) -> bytes:
    chunks: list[bytes] = []
    while not budget.exceeded.is_set():
        chunk = await stream.read(16 * 1024)
        if not chunk:
            break
        accepted = budget.accept(chunk)
        if accepted:
            chunks.append(accepted)
    return b"".join(chunks)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.kill()
    await process.wait()


class _DarwinRusageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


async def _monitor_macos_memory(
    pid: int,
    maximum_bytes: int,
) -> str:
    failures = 0
    while True:
        resident_size = _macos_process_memory_bytes(pid)
        if resident_size is None:
            failures += 1
            if failures >= 20:
                return "unavailable"
        else:
            failures = 0
            if resident_size > maximum_bytes:
                return "exceeded"
        await asyncio.sleep(0.05)


def _macos_process_memory_bytes(pid: int) -> int | None:
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pid_rusage = library.proc_pid_rusage
        proc_pid_rusage.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        proc_pid_rusage.restype = ctypes.c_int
        usage = _DarwinRusageInfoV0()
        result = proc_pid_rusage(
            pid,
            _RUSAGE_INFO_V0,
            ctypes.byref(usage),
        )
    except (AttributeError, OSError):
        return None
    if result != 0:
        return None
    return int(max(usage.ri_resident_size, usage.ri_phys_footprint))


async def _monitor_workspace_usage(
    roots: tuple[Path, ...],
    *,
    max_file_bytes: int,
    max_workspace_bytes: int,
    max_files: int,
) -> str:
    while True:
        try:
            violation = await asyncio.to_thread(
                _workspace_usage_violation,
                roots,
                max_file_bytes,
                max_workspace_bytes,
                max_files,
            )
        except OSError:
            return "compute.workspace_monitor_unavailable"
        if violation is not None:
            return violation
        await asyncio.sleep(0.05)


def _workspace_usage_violation(
    roots: tuple[Path, ...],
    max_file_bytes: int,
    max_workspace_bytes: int,
    max_files: int,
) -> str | None:
    file_count = 0
    total_bytes = 0
    for root in roots:
        for path in root.rglob("*"):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            file_count += 1
            if file_count > max_files:
                return "files.file_count_limit"
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                return "compute.special_file_forbidden"
            if metadata.st_size > max_file_bytes:
                return "files.file_too_large"
            total_bytes += metadata.st_size
            if total_bytes > max_workspace_bytes:
                return "files.workspace_quota"
    return None


def _remove_run_directory(run_directory: Path) -> None:
    if run_directory.is_symlink():
        run_directory.unlink(missing_ok=True)
        return
    with suppress(OSError):
        run_directory.chmod(0o700)
    for root, directory_names, filenames in os.walk(
        run_directory,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)
        with suppress(OSError):
            root_path.chmod(0o700)
        for name in directory_names:
            path = root_path / name
            if path.is_symlink():
                continue
            with suppress(OSError):
                path.chmod(0o700)
        for name in filenames:
            path = root_path / name
            if path.is_symlink():
                continue
            with suppress(OSError):
                path.chmod(0o600)
    shutil.rmtree(run_directory, ignore_errors=True)


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv or len(argv) > _MAX_ARGUMENTS:
        raise UserError("compute.argv_invalid")
    if any(
        not isinstance(argument, str)
        or not argument
        or "\x00" in argument
        or len(argument) > _MAX_SINGLE_ARGUMENT_CHARACTERS
        for argument in argv
    ):
        raise UserError("compute.argv_invalid")
    if sum(len(argument) for argument in argv) > _MAX_ARGUMENT_CHARACTERS:
        raise UserError("compute.argv_invalid")
    script = PurePosixPath(argv[0])
    if (
        argv[0].startswith("-")
        or not argv[0].endswith(".py")
        or argv[0].startswith("./")
        or script.is_absolute()
        or any(part in {"", ".", ".."} for part in script.parts)
    ):
        raise UserError("compute.script_invalid")
    return tuple(argv)


def _validated_input_paths(
    input_paths: Sequence[str],
    *,
    script_path: str,
) -> tuple[str, ...]:
    if len(input_paths) > _MAX_INPUT_FILES:
        raise UserError("compute.input_limit_invalid")
    normalized: list[str] = []
    for value in input_paths:
        if (
            not isinstance(value, str)
            or not value
            or "\x00" in value
            or len(value) > _MAX_SINGLE_ARGUMENT_CHARACTERS
        ):
            raise UserError("compute.input_limit_invalid")
        path = PurePosixPath(value)
        if (
            value.startswith("./")
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise UserError("compute.input_limit_invalid")
        if value != script_path:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def _sandbox_string(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _supported_download_content_type(content_type: str) -> bool:
    return (
        content_type.startswith("text/")
        or content_type in _SUPPORTED_DOWNLOAD_CONTENT_TYPES
    )
