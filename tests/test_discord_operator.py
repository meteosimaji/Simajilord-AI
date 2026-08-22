from __future__ import annotations

import asyncio
import json
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
)
from simajilord.core import InvocationContext
from simajilord.integrations.discord.operator import (
    LocalOperatorError,
    LocalOperatorServer,
    request_vc_speech,
)
from simajilord.runtime import SimajilordRuntime

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="The local operator bridge uses a Unix-domain socket.",
)


@pytest.fixture
def socket_path() -> Iterator[Path]:
    # macOS limits AF_UNIX paths to 104 bytes. pytest's nested tmp_path can be
    # longer than that before the socket filename is appended.
    with tempfile.TemporaryDirectory(prefix="sl-operator-", dir="/tmp") as directory:
        yield Path(directory) / "operator.sock"


def _runtime(
    *,
    connected: bool = True,
) -> tuple[SimajilordRuntime, AsyncMock, AsyncMock]:
    registry = SimpleNamespace(invoke=AsyncMock())
    registry.invoke.return_value = SpeechSpeakResponse(
        title="Local operator announcement",
        queue_position=1,
        duration_seconds=1.25,
        destination_id="456",
        playback_state="playing",
    )
    session = SimpleNamespace(
        destination_id="456",
        output=SimpleNamespace(connected=connected),
    )
    connect = AsyncMock()

    async def mark_connected(workspace_id: str, destination_id: str) -> None:
        session.output.connected = True

    connect.side_effect = mark_connected
    runtime = SimpleNamespace(
        settings=SimpleNamespace(data_dir=Path(".data")),
        audio=SimpleNamespace(
            find=lambda workspace_id: session,
            connect=connect,
        ),
        registry=registry,
    )
    return cast(SimajilordRuntime, runtime), registry.invoke, connect


@pytest.mark.asyncio
async def test_local_operator_speaks_through_running_bot(socket_path: Path) -> None:
    runtime, invoke, connect = _runtime()
    server = LocalOperatorServer(runtime, socket_path)
    await server.start()
    socket_mode = stat.S_IMODE(socket_path.stat().st_mode)
    try:
        response = await request_vc_speech(
            socket_path=socket_path,
            workspace_id="123",
            text="  再起動します。  ",
            voice_preset="clear",
            request_id="operator:test",
        )
    finally:
        await server.close()

    assert response["ok"] is True
    assert response["destination_id"] == "456"
    assert response["queue_position"] == 1
    assert socket_mode == 0o600
    assert not socket_path.exists()
    invoke.assert_awaited_once()
    capability, request, context = invoke.await_args.args
    assert capability == "speech.speak"
    assert isinstance(request, SpeechSpeakRequest)
    assert request.text == "再起動します。"
    assert request.voice_preset == "clear"
    assert isinstance(context, InvocationContext)
    assert context.actor_id == "local-operator"
    assert context.workspace_id == "123"
    assert context.transport == "local_operator"
    assert context.resource_ids == ("456",)
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_operator_rejects_disconnected_audio(socket_path: Path) -> None:
    runtime, invoke, connect = _runtime(connected=False)
    server = LocalOperatorServer(runtime, socket_path)
    await server.start()
    try:
        with pytest.raises(LocalOperatorError, match=r"operator\.voice_not_connected"):
            await request_vc_speech(
                socket_path=socket_path,
                workspace_id="123",
                text="遅延させない",
                request_id="operator:disconnected",
            )
    finally:
        await server.close()

    invoke.assert_not_awaited()
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_operator_explicitly_connects_saved_audio(
    socket_path: Path,
) -> None:
    runtime, invoke, connect = _runtime(connected=False)
    server = LocalOperatorServer(runtime, socket_path)
    await server.start()
    try:
        response = await request_vc_speech(
            socket_path=socket_path,
            workspace_id="123",
            text="再起動後の直通テスト",
            request_id="operator:connect",
            connect_if_needed=True,
        )
    finally:
        await server.close()

    assert response["ok"] is True
    connect.assert_awaited_once_with("123", "456")
    invoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_local_operator_rejects_unknown_protocol_fields(
    socket_path: Path,
) -> None:
    runtime, invoke, connect = _runtime()
    server = LocalOperatorServer(runtime, socket_path)
    await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(
            (
                json.dumps(
                    {
                        "version": 1,
                        "operation": "speak",
                        "request_id": "operator:unknown",
                        "workspace_id": "123",
                        "text": "読まない",
                        "voice_preset": "clear",
                        "capability": "discord.send_message",
                    }
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        response = json.loads((await reader.readline()).decode())
        writer.close()
        await writer.wait_closed()
    finally:
        await server.close()

    assert response == {
        "ok": False,
        "version": 1,
        "error": "operator.invalid_request",
    }
    invoke.assert_not_awaited()
    connect.assert_not_awaited()
