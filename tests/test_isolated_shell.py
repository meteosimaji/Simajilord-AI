from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from simajilord.capabilities.isolated_shell import (
    IsolatedShellRequest,
    IsolatedShellResponse,
    build_isolated_shell_endpoint,
    discord_workspace_for_context,
)
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.core.errors import UserError
from simajilord.observability import EventJournal


def _context() -> InvocationContext:
    return InvocationContext(
        "actor",
        "guild",
        "discord",
        "request",
        public_reference_id="agt_0000000000000000000b",
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt integration")
@pytest.mark.asyncio
async def test_isolated_shell_writes_inside_and_denies_outside_reads(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(build_isolated_shell_endpoint(tmp_path / "workspaces"))
    context = _context()
    workspace = discord_workspace_for_context(tmp_path / "workspaces", context)

    write = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(
            argv=(
                "/bin/sh",
                "-c",
                "printf isolated-ok > proof.txt; /bin/ls proof.txt",
            ),
        ),
        context,
    )

    assert isinstance(write, IsolatedShellResponse)
    assert write.exit_code == 0
    assert write.timed_out is False
    assert write.stdout.strip() == "proof.txt"
    assert (workspace / "proof.txt").read_text(encoding="utf-8") == "isolated-ok"

    sibling = workspace.parent / "other-guild"
    sibling.mkdir(mode=0o700)
    parent = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(argv=("/bin/ls", "..")),
        context,
    )

    assert isinstance(parent, IsolatedShellResponse)
    assert parent.exit_code != 0
    assert parent.stdout == ""
    assert "Operation not permitted" in parent.stderr
    assert "other-guild" not in parent.stderr

    outside = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(
            argv=("/usr/bin/head", "-n", "1", str(Path(__file__).resolve())),
        ),
        context,
    )

    assert isinstance(outside, IsolatedShellResponse)
    assert outside.exit_code != 0
    assert outside.stdout == ""
    assert "Operation not permitted" in outside.stderr


@pytest.mark.asyncio
async def test_isolated_shell_rejects_directory_escape(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(build_isolated_shell_endpoint(tmp_path / "workspaces"))

    with pytest.raises(UserError, match=r"shell\.working_directory_invalid"):
        await registry.invoke(
            "system.shell",
            IsolatedShellRequest(argv=("/bin/pwd",), working_directory="../"),
            _context(),
        )


@pytest.mark.asyncio
async def test_isolated_shell_contract_lists_every_validation_error(tmp_path: Path) -> None:
    endpoint = build_isolated_shell_endpoint(tmp_path / "workspaces")

    assert set(endpoint.descriptor.expected_errors) == {
        "shell.arguments_invalid",
        "shell.timeout_invalid",
        "shell.login_shell_forbidden",
        "shell.working_directory_invalid",
        "shell.isolation_unavailable",
        "shell.launch_failed",
    }
    with pytest.raises(UserError, match=r"shell\.timeout_invalid"):
        await endpoint.invoke(
            IsolatedShellRequest(argv=("/bin/pwd",), timeout_seconds=0),
            _context(),
        )
    with pytest.raises(UserError, match=r"shell\.login_shell_forbidden"):
        await endpoint.invoke(
            IsolatedShellRequest(argv=("/bin/sh", "-l")),
            _context(),
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt integration")
@pytest.mark.asyncio
async def test_isolated_shell_bounds_output_without_blocking_child(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    registry.register(build_isolated_shell_endpoint(tmp_path / "workspaces"))

    response = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(argv=("/usr/bin/printf", "%s", "x" * 20_000)),
        _context(),
    )

    assert isinstance(response, IsolatedShellResponse)
    assert response.exit_code == 0
    assert response.stdout == "x" * 12_000
    assert response.stdout_truncated is True
    assert response.stderr_truncated is False

    unicode_response = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(argv=("/usr/bin/printf", "%s", "界" * 13_000)),
        _context(),
    )
    assert isinstance(unicode_response, IsolatedShellResponse)
    assert unicode_response.stdout == "界" * 12_000
    assert unicode_response.stdout_truncated is True


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt integration")
@pytest.mark.asyncio
async def test_isolated_shell_cancellation_terminates_process_group(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    workspace_root = tmp_path / "workspaces"
    registry.register(build_isolated_shell_endpoint(workspace_root))
    workspace = discord_workspace_for_context(workspace_root, _context())
    task = asyncio.create_task(
        registry.invoke(
            "system.shell",
            IsolatedShellRequest(
                argv=(
                    "/bin/sh",
                    "-c",
                    "echo $$ > process.pid; exec /bin/sleep 60",
                ),
            ),
            _context(),
        )
    )
    pid_file = workspace / "process.pid"
    for _ in range(100):
        if pid_file.is_file():
            break
        await asyncio.sleep(0.01)
    assert pid_file.is_file()
    pid = int(pid_file.read_text(encoding="utf-8").strip())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt integration")
@pytest.mark.asyncio
async def test_isolated_shell_journal_keeps_metadata_only(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "events.sqlite3")
    registry = CapabilityRegistry(journal)
    registry.register(build_isolated_shell_endpoint(tmp_path / "workspaces"))

    response = await registry.invoke(
        "system.shell",
        IsolatedShellRequest(argv=("/bin/echo", "never-persist-this-command-value")),
        _context(),
    )

    assert isinstance(response, IsolatedShellResponse)
    assert "never-persist-this-command-value" in response.stdout
    records = await journal.agent_trace(
        public_reference_id="agt_0000000000000000000b"
    )
    record = next(item for item in records if item.kind == "capability.invocation")
    serialized = json.dumps(record.payload, ensure_ascii=False)
    assert "never-persist-this-command-value" not in serialized
    assert "request" not in record.payload
    assert "response" not in record.payload
    request_fields = record.payload["request_fields"]
    assert isinstance(request_fields, list)
    assert set(request_fields) == {
        "argv",
        "timeout_seconds",
        "working_directory",
    }
    await journal.close()
