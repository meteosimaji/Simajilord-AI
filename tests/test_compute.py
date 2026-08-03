from __future__ import annotations

import sys
from pathlib import Path

import pytest

from simajilord.capabilities.compute import (
    ComputeRunRequest,
    FileDownloadUrlRequest,
    build_compute_endpoints,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError, WebError
from simajilord.domain.web import FetchedWebResource
from simajilord.services.compute import (
    ComputeLimits,
    ComputeProcessResult,
    MacOSSandboxedPythonLauncher,
    WorkspaceComputeService,
    _parse_otool_dependencies,
    _sandbox_read_metadata_ancestors,
    _workspace_usage_violation,
)
from simajilord.services.files import AgentFileSandbox, WorkspaceFileProvenance


class _FakeLauncher:
    available = True

    def __init__(
        self,
        *,
        exit_code: int = 0,
        create_path: str | None = None,
        create_content: bytes = b"",
        remove_path: str | None = None,
        create_symlink: bool = False,
    ) -> None:
        self.exit_code = exit_code
        self.create_path = create_path
        self.create_content = create_content
        self.remove_path = remove_path
        self.create_symlink = create_symlink
        self.calls: list[tuple[str, ...]] = []
        self.staged_paths: list[tuple[str, ...]] = []

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
        del (
            temporary_directory,
            limits,
            max_file_bytes,
            max_workspace_bytes,
            max_files,
        )
        self.calls.append(argv)
        self.staged_paths.append(
            tuple(
                sorted(
                    path.relative_to(workspace).as_posix()
                    for path in workspace.rglob("*")
                    if path.is_file()
                )
            )
        )
        assert (workspace / argv[0]).is_file()
        if self.remove_path is not None:
            (workspace / self.remove_path).unlink()
        if self.create_path is not None:
            destination = workspace / self.create_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if self.create_symlink:
                destination.symlink_to("/etc/passwd")
            else:
                destination.write_bytes(self.create_content)
        return ComputeProcessResult(
            exit_code=self.exit_code,
            stdout="done\n",
            stderr="",
        )


class _FakeFetcher:
    def __init__(
        self,
        *,
        body: bytes = b"downloaded",
        content_type: str = "application/pdf",
    ) -> None:
        self.body = body
        self.content_type = content_type
        self.calls: list[tuple[str, int]] = []

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
    ) -> FetchedWebResource:
        self.calls.append((url, max_bytes))
        return FetchedWebResource(
            final_url="https://cdn.example.test/final",
            content_type=self.content_type,
            charset=None,
            body=self.body,
        )

    async def close(self) -> None:
        pass


def test_sandbox_metadata_exceptions_cover_only_exact_private_ancestors() -> None:
    ancestors = _sandbox_read_metadata_ancestors(
        (
            Path("/opt/homebrew/Cellar/python/3.14/lib/python3.14"),
            Path("/usr/lib/libSystem.B.dylib"),
        )
    )

    assert ancestors == (
        Path("/opt/homebrew/Cellar/python/3.14/lib"),
        Path("/opt/homebrew/Cellar/python/3.14"),
        Path("/opt/homebrew/Cellar/python"),
        Path("/opt/homebrew/Cellar"),
        Path("/opt/homebrew"),
        Path("/opt"),
    )


def test_otool_dependency_parser_accepts_only_absolute_dependency_lines() -> None:
    output = (
        "/runtime/lib-dynload/_ssl.so:\n"
        "\t/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib "
        "(compatibility version 3.0.0, current version 3.0.0)\n"
        "\t@rpath/libignored.dylib "
        "(compatibility version 1.0.0, current version 1.0.0)\n"
        "/not/an/indented/dependency\n"
    )

    assert _parse_otool_dependencies(output) == (
        Path("/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib"),
    )


def test_workspace_monitor_counts_directories_against_inode_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    for index in range(6):
        (root / f"directory-{index}").mkdir()

    assert (
        _workspace_usage_violation(
            (root,),
            max_file_bytes=1_000,
            max_workspace_bytes=1_000,
            max_files=5,
        )
        == "files.file_count_limit"
    )


def _service(
    tmp_path: Path,
    *,
    launcher: _FakeLauncher | MacOSSandboxedPythonLauncher | None = None,
    fetcher: _FakeFetcher | None = None,
    max_file_bytes: int = 1024,
) -> tuple[AgentFileSandbox, WorkspaceComputeService]:
    files = AgentFileSandbox(
        tmp_path / "files",
        max_file_bytes=max_file_bytes,
        max_workspace_bytes=4 * 1024,
        max_files=10,
    )
    return files, WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "runs",
        web_fetcher=fetcher or _FakeFetcher(),
        launcher=launcher or _FakeLauncher(),
        limits=ComputeLimits(
            timeout_seconds=5,
            cpu_seconds=2,
            memory_bytes=128 * 1024 * 1024,
            output_bytes=4_096,
            open_files=32,
        ),
        max_download_bytes=max_file_bytes,
    )


@pytest.mark.asyncio
async def test_compute_stages_script_and_commits_only_validated_changes(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher(
        create_path="outputs/result.txt",
        create_content=b"42\n",
    )
    files, service = _service(tmp_path, launcher=launcher)
    restricted = WorkspaceFileProvenance(
        owner_actor_ids=("actor",),
        origin_guild_id="guild",
        origin_channel_id="staff",
        origin_message_id="message",
        origin_visibility="restricted",
        created_task_id="task",
        sensitivity="restricted",
        source_resources=(("guild", "staff", "restricted"),),
    )
    files.write_text(
        "guild",
        "scripts/calculate.py",
        "print(6 * 7)",
        provenance=restricted,
    )

    result = await service.run(
        "guild",
        runtime="python",
        argv=("scripts/calculate.py", "--format", "text"),
        actor_id="actor",
        provenance=restricted,
    )

    assert result.exit_code == 0
    assert result.stdout == "done\n"
    assert launcher.calls == [
        ("scripts/calculate.py", "--format", "text")
    ]
    assert launcher.staged_paths == [("scripts/calculate.py",)]
    assert tuple(file.path for file in result.changed_files) == (
        "outputs/result.txt",
    )
    assert result.provenance is not None
    assert result.provenance.sensitivity == "restricted"
    assert result.changed_files[0].provenance is not None
    assert result.changed_files[0].provenance.sensitivity == "restricted"
    assert files.read(
        "guild",
        "outputs/result.txt",
        offset=0,
        max_characters=100,
    ).content == "42\n"


@pytest.mark.asyncio
async def test_compute_stages_only_explicit_actor_owned_inputs(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher()
    files, service = _service(tmp_path, launcher=launcher)
    actor = WorkspaceFileProvenance(owner_actor_ids=("actor",))
    other_actor = WorkspaceFileProvenance(owner_actor_ids=("other",))
    files.write_text(
        "guild",
        "run.py",
        "print('ok')",
        provenance=actor,
    )
    files.write_text(
        "guild",
        "selected.txt",
        "selected",
        provenance=actor,
    )
    files.write_text(
        "guild",
        "unrelated.txt",
        "unrelated",
        provenance=actor,
    )
    files.write_text(
        "guild",
        "other.txt",
        "private",
        provenance=other_actor,
    )

    await service.run(
        "guild",
        runtime="python",
        argv=("run.py",),
        input_paths=("selected.txt",),
        actor_id="actor",
        provenance=actor,
    )

    assert launcher.staged_paths == [("run.py", "selected.txt")]


@pytest.mark.asyncio
async def test_compute_rejects_cross_actor_and_unlabelled_inputs(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher()
    files, service = _service(tmp_path, launcher=launcher)
    actor = WorkspaceFileProvenance(owner_actor_ids=("actor",))
    other_actor = WorkspaceFileProvenance(owner_actor_ids=("other",))
    files.write_text("guild", "run.py", "pass", provenance=actor)
    files.write_text(
        "guild",
        "other-run.py",
        "pass",
        provenance=other_actor,
    )
    files.write_text(
        "guild",
        "other.txt",
        "private",
        provenance=other_actor,
    )
    files.write_text("guild", "legacy.txt", "unlabelled")

    with pytest.raises(UserError, match=r"compute\.script_not_found"):
        await service.run(
            "guild",
            runtime="python",
            argv=("other-run.py",),
            actor_id="actor",
            provenance=actor,
        )
    for path in ("other.txt", "legacy.txt"):
        with pytest.raises(UserError, match=r"compute\.input_not_found"):
            await service.run(
                "guild",
                runtime="python",
                argv=("run.py",),
                input_paths=(path,),
                actor_id="actor",
                provenance=actor,
            )

    assert launcher.calls == []


@pytest.mark.asyncio
async def test_compute_rejects_output_collision_with_unstaged_workspace_file(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher(
        create_path="existing.txt",
        create_content=b"replacement",
    )
    files, service = _service(tmp_path, launcher=launcher)
    actor = WorkspaceFileProvenance(owner_actor_ids=("actor",))
    files.write_text("guild", "run.py", "pass", provenance=actor)
    files.write_text(
        "guild",
        "existing.txt",
        "original",
        provenance=actor,
    )

    with pytest.raises(UserError, match=r"compute\.output_conflict"):
        await service.run(
            "guild",
            runtime="python",
            argv=("run.py",),
            actor_id="actor",
            provenance=actor,
        )

    assert files.read_for_actor(
        "guild",
        "actor",
        "existing.txt",
        offset=0,
        max_characters=100,
    ).content == "original"


@pytest.mark.asyncio
async def test_compute_taints_unlabelled_trusted_input_and_output(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher(
        create_path="result.txt",
        create_content=b"result",
    )
    files, service = _service(tmp_path, launcher=launcher)
    files.write_text("guild", "run.py", "pass")

    result = await service.run(
        "guild",
        runtime="python",
        argv=("run.py",),
    )

    assert result.provenance is not None
    assert result.provenance.owner_actor_ids == ()
    assert result.provenance.unlabelled_input
    assert result.provenance.sensitivity == "uncertain"
    output = files.read(
        "guild",
        "result.txt",
        offset=0,
        max_characters=100,
    )
    assert output.provenance == result.provenance
    with pytest.raises(UserError, match=r"files\.not_found"):
        files.snapshot_for_actor_delivery_with_provenance(
            "guild",
            "actor",
            "result.txt",
        )


@pytest.mark.asyncio
async def test_compute_does_not_commit_failed_process_output(tmp_path: Path) -> None:
    launcher = _FakeLauncher(
        exit_code=2,
        create_path="partial.txt",
        create_content=b"partial",
    )
    files, service = _service(tmp_path, launcher=launcher)
    files.write_text("guild", "run.py", "raise RuntimeError")

    result = await service.run(
        "guild",
        runtime="python",
        argv=("run.py",),
    )

    assert result.exit_code == 2
    assert result.changed_files == ()
    assert {file.path for file in files.list("guild")} == {"run.py"}


@pytest.mark.asyncio
async def test_compute_rejects_deletion_and_symlink_output(tmp_path: Path) -> None:
    deleting = _FakeLauncher(remove_path="input.txt")
    files, service = _service(tmp_path, launcher=deleting)
    files.write_text("guild", "run.py", "pass")
    files.write_text("guild", "input.txt", "keep")

    with pytest.raises(
        UserError,
        match=r"compute\.file_deletion_unsupported",
    ):
        await service.run(
            "guild",
            runtime="python",
            argv=("run.py",),
            input_paths=("input.txt",),
        )
    assert files.read(
        "guild",
        "input.txt",
        offset=0,
        max_characters=100,
    ).content == "keep"

    symlink = _FakeLauncher(
        create_path="leak",
        create_symlink=True,
    )
    _, service = _service(tmp_path, launcher=symlink)
    with pytest.raises(UserError, match=r"files\.symlink_forbidden"):
        await service.run("guild", runtime="python", argv=("run.py",))
    assert {file.path for file in files.list("guild")} == {
        "input.txt",
        "run.py",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "argv", "error"),
    (
        ("ruby", ("run.py",), r"compute\.runtime_unsupported"),
        ("python", ("-c", "print(1)"), r"compute\.script_invalid"),
        ("python", ("../run.py",), r"compute\.script_invalid"),
        ("python", (), r"compute\.argv_invalid"),
    ),
)
async def test_compute_accepts_only_allowlisted_runtime_and_script_argv(
    tmp_path: Path,
    runtime: str,
    argv: tuple[str, ...],
    error: str,
) -> None:
    _, service = _service(tmp_path)
    with pytest.raises(UserError, match=error):
        await service.run("guild", runtime=runtime, argv=argv)


@pytest.mark.asyncio
async def test_public_download_reuses_bounded_fetcher_and_file_quota(
    tmp_path: Path,
) -> None:
    fetcher = _FakeFetcher(
        body=b"%PDF-test",
        content_type="application/pdf",
    )
    files, service = _service(
        tmp_path,
        fetcher=fetcher,
        max_file_bytes=128,
    )

    result = await service.download(
        "guild",
        url="https://example.test/document.pdf",
        path="incoming/document.pdf",
    )

    assert fetcher.calls == [
        ("https://example.test/document.pdf", 128)
    ]
    assert result.source_url == "https://cdn.example.test/final"
    assert result.content_type == "application/pdf"
    assert result.file.kind == "pdf"
    assert files.path_for_delivery(
        "guild",
        "incoming/document.pdf",
    ).read_bytes() == b"%PDF-test"


@pytest.mark.asyncio
async def test_public_download_rejects_invalid_url_before_dispatch(
    tmp_path: Path,
) -> None:
    fetcher = _FakeFetcher()
    _, service = _service(tmp_path, fetcher=fetcher)
    dispatches = 0

    async def before_fetch() -> None:
        nonlocal dispatches
        dispatches += 1

    with pytest.raises(WebError) as captured:
        await service.download(
            "guild",
            url="file:///etc/passwd",
            path="incoming/file",
            before_fetch=before_fetch,
        )

    assert captured.value.category == "url_invalid"
    assert dispatches == 0
    assert fetcher.calls == []


@pytest.mark.asyncio
async def test_public_download_rejects_unhandled_mime_before_writing(
    tmp_path: Path,
) -> None:
    fetcher = _FakeFetcher(content_type="application/x-mach-binary")
    files, service = _service(tmp_path, fetcher=fetcher)

    with pytest.raises(
        UserError,
        match=r"files\.download_type_unsupported",
    ):
        await service.download(
            "guild",
            url="https://example.test/program",
            path="program",
        )
    assert files.list("guild") == ()


@pytest.mark.asyncio
async def test_compute_capabilities_require_workspace_and_keep_argv_typed(
    tmp_path: Path,
) -> None:
    launcher = _FakeLauncher()
    _, service = _service(tmp_path, launcher=launcher)
    endpoints = {
        item.descriptor.name: item
        for item in build_compute_endpoints(service)
    }
    assert set(endpoints) == {"compute.run", "files.download_url"}
    missing_workspace = InvocationContext(
        actor_id="7",
        workspace_id=None,
        transport="agent",
        request_id="event",
    )
    with pytest.raises(UserError, match=r"compute\.workspace_required"):
        await endpoints["compute.run"].invoke(
            ComputeRunRequest(runtime="python", argv=("run.py",)),
            missing_workspace,
        )
    with pytest.raises(UserError, match=r"compute\.workspace_required"):
        await endpoints["files.download_url"].invoke(
            FileDownloadUrlRequest(
                url="https://example.test/file.pdf",
                path="file.pdf",
            ),
            missing_workspace,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Seatbelt is a macOS execution boundary.",
)
async def test_macos_launcher_denies_network_fork_and_host_project_reads(
    tmp_path: Path,
) -> None:
    launcher = MacOSSandboxedPythonLauncher()
    if not launcher.available:
        pytest.skip("sandbox-exec is unavailable")
    files = AgentFileSandbox(
        tmp_path / "files",
        max_file_bytes=64 * 1024,
        max_workspace_bytes=256 * 1024,
        max_files=10,
    )
    service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=10,
            cpu_seconds=5,
            memory_bytes=512 * 1024 * 1024,
            output_bytes=16 * 1024,
            open_files=64,
        ),
    )
    project_file = Path(__file__).parents[1] / "pyproject.toml"
    outside_output = tmp_path / "outside.txt"
    script = f"""
import pathlib
import pypdf
import ssl
import sqlite3
import lzma
import _decimal
import socket
import subprocess
import os
import sys

blocked = 0
try:
    pathlib.Path({str(project_file)!r}).read_bytes()
except OSError:
    blocked += 1
try:
    socket.create_connection(("1.1.1.1", 80), timeout=1)
except OSError:
    blocked += 1
try:
    subprocess.run(["/bin/echo", "forbidden"], check=True)
except OSError:
    blocked += 1
try:
    pathlib.Path({str(outside_output)!r}).write_text("forbidden")
except OSError:
    blocked += 1
try:
    os.link({str(project_file)!r}, "linked-host-file")
except OSError:
    blocked += 1
if "DISCORD_TOKEN" not in os.environ and "PATH" not in os.environ:
    blocked += 1
if sys.dont_write_bytecode:
    blocked += 1
pathlib.Path("result.txt").write_text(
    f"{{blocked}}:{{pypdf.__version__}}",
    encoding="utf-8",
)
"""
    files.write_text("guild", "probe.py", script)

    result = await service.run(
        "guild",
        runtime="python",
        argv=("probe.py",),
    )

    assert result.exit_code == 0, result.stderr
    assert files.read(
        "guild",
        "result.txt",
        offset=0,
        max_characters=10,
    ).content.startswith("7:")
    assert not outside_output.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Seatbelt is a macOS execution boundary.",
)
async def test_macos_launcher_enforces_output_and_memory_limits(
    tmp_path: Path,
) -> None:
    launcher = MacOSSandboxedPythonLauncher()
    if not launcher.available:
        pytest.skip("sandbox-exec is unavailable")
    files = AgentFileSandbox(
        tmp_path / "files",
        max_file_bytes=64 * 1024,
        max_workspace_bytes=256 * 1024,
        max_files=10,
    )
    output_service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "output-runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=10,
            cpu_seconds=5,
            memory_bytes=128 * 1024 * 1024,
            output_bytes=4_096,
            open_files=64,
        ),
    )
    files.write_text("guild", "noisy.py", 'print("x" * 10000)')
    with pytest.raises(UserError, match=r"compute\.output_limit"):
        await output_service.run(
            "guild",
            runtime="python",
            argv=("noisy.py",),
        )

    memory_service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "memory-runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=10,
            cpu_seconds=5,
            memory_bytes=64 * 1024 * 1024,
            output_bytes=4_096,
            open_files=64,
        ),
    )
    files.write_text(
        "guild",
        "memory.py",
        (
            "import time\n"
            "chunks = [bytearray(8 * 1024 * 1024) for _ in range(20)]\n"
            "time.sleep(5)\n"
        ),
    )
    with pytest.raises(UserError, match=r"compute\.memory_limit"):
        await memory_service.run(
            "guild",
            runtime="python",
            argv=("memory.py",),
        )


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Seatbelt is a macOS execution boundary.",
)
async def test_macos_launcher_enforces_wall_and_cpu_time(
    tmp_path: Path,
) -> None:
    launcher = MacOSSandboxedPythonLauncher()
    if not launcher.available:
        pytest.skip("sandbox-exec is unavailable")
    files = AgentFileSandbox(
        tmp_path / "files",
        max_file_bytes=64 * 1024,
        max_workspace_bytes=256 * 1024,
        max_files=10,
    )
    timeout_service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "timeout-runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=1,
            cpu_seconds=1,
            memory_bytes=128 * 1024 * 1024,
            output_bytes=4_096,
            open_files=64,
        ),
    )
    files.write_text(
        "guild",
        "sleep.py",
        "import time\ntime.sleep(10)\n",
    )
    with pytest.raises(UserError, match=r"compute\.timeout"):
        await timeout_service.run(
            "guild",
            runtime="python",
            argv=("sleep.py",),
        )

    cpu_service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "cpu-runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=5,
            cpu_seconds=1,
            memory_bytes=128 * 1024 * 1024,
            output_bytes=4_096,
            open_files=64,
        ),
    )
    files.write_text("guild", "busy.py", "while True:\n    pass\n")
    result = await cpu_service.run(
        "guild",
        runtime="python",
        argv=("busy.py",),
    )
    assert result.exit_code < 0
    assert result.duration_ms < 4_000


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Seatbelt is a macOS execution boundary.",
)
async def test_macos_launcher_stops_runtime_workspace_quota_growth(
    tmp_path: Path,
) -> None:
    launcher = MacOSSandboxedPythonLauncher()
    if not launcher.available:
        pytest.skip("sandbox-exec is unavailable")
    files = AgentFileSandbox(
        tmp_path / "files",
        max_file_bytes=64 * 1024,
        max_workspace_bytes=256 * 1024,
        max_files=5,
    )
    service = WorkspaceComputeService(
        files=files,
        run_root=tmp_path / "runs",
        web_fetcher=_FakeFetcher(),
        launcher=launcher,
        limits=ComputeLimits(
            timeout_seconds=5,
            cpu_seconds=3,
            memory_bytes=128 * 1024 * 1024,
            output_bytes=4_096,
            open_files=64,
        ),
    )
    files.write_text(
        "guild",
        "many.py",
        (
            "import pathlib, time\n"
            "for index in range(20):\n"
            "    pathlib.Path(f'file-{index}').write_text('x')\n"
            "time.sleep(3)\n"
        ),
    )

    with pytest.raises(UserError, match=r"files\.file_count_limit"):
        await service.run(
            "guild",
            runtime="python",
            argv=("many.py",),
        )
    assert {record.path for record in files.list("guild")} == {"many.py"}
