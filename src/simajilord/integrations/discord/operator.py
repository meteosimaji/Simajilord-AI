"""Private local operator bridge for speaking through the running Discord BOT."""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from simajilord.capabilities.speech import (
    SpeechSpeakRequest,
    SpeechSpeakResponse,
)
from simajilord.core import InvocationContext
from simajilord.core.errors import UserError

if TYPE_CHECKING:
    from simajilord.runtime import SimajilordRuntime

log = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1
_MAX_REQUEST_BYTES = 4_096
_MAX_TEXT_CHARACTERS = 500
_SERVER_READ_TIMEOUT_SECONDS = 5.0
_CLIENT_RESPONSE_TIMEOUT_SECONDS = 90.0
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_VOICE_PRESETS = frozenset({"clear", "calm", "energetic", "cute", "narrator"})
_REQUEST_FIELDS = frozenset(
    {
        "version",
        "operation",
        "request_id",
        "workspace_id",
        "text",
        "voice_preset",
        "connect_if_needed",
    }
)


class LocalOperatorError(RuntimeError):
    """Stable client-visible failure from the local operator bridge."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class LocalOperatorServer:
    """Expose only VC speech over a user-private Unix-domain socket."""

    def __init__(
        self,
        runtime: SimajilordRuntime,
        socket_path: Path | None = None,
    ) -> None:
        self.runtime = runtime
        self.socket_path = socket_path or runtime.settings.data_dir / "operator.sock"
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if self._server is not None:
            return
        if sys.platform == "win32":
            log.warning("Local operator VC speech is unavailable on Windows")
            return
        await _remove_stale_socket(self.socket_path)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path),
                limit=_MAX_REQUEST_BYTES + 1,
            )
            self.socket_path.chmod(0o600)
        except BaseException:
            _unlink_socket(self.socket_path)
            raise
        log.info("Local operator VC speech ready at %s", self.socket_path)

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        _unlink_socket(self.socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            response = await self._read_and_dispatch(reader)
        except LocalOperatorError as exc:
            response = _error_response(exc.code)
        except UserError as exc:
            response = _error_response(exc.code)
        except (TimeoutError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            response = _error_response("operator.invalid_request")
        except Exception:
            log.exception("Local operator VC speech failed")
            response = _error_response("operator.internal_error")
        try:
            writer.write(_encode_message(response))
            await writer.drain()
        except (BrokenPipeError, ConnectionError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _read_and_dispatch(
        self,
        reader: asyncio.StreamReader,
    ) -> dict[str, object]:
        raw = await asyncio.wait_for(
            reader.readline(),
            timeout=_SERVER_READ_TIMEOUT_SECONDS,
        )
        if not raw or len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            raise LocalOperatorError("operator.invalid_request")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise LocalOperatorError("operator.invalid_request")
        return await self._dispatch(cast(dict[str, object], payload))

    async def _dispatch(self, payload: dict[str, object]) -> dict[str, object]:
        if set(payload) - _REQUEST_FIELDS:
            raise LocalOperatorError("operator.invalid_request")
        if payload.get("version") != _PROTOCOL_VERSION:
            raise LocalOperatorError("operator.protocol_version_invalid")
        if payload.get("operation") != "speak":
            raise LocalOperatorError("operator.operation_invalid")
        request_id = _required_request_id(payload.get("request_id"))
        workspace_id = _required_snowflake(payload.get("workspace_id"))
        text = _required_text(payload.get("text"))
        voice_preset = _voice_preset(payload.get("voice_preset"))
        connect_if_needed = _optional_bool(payload.get("connect_if_needed", False))

        session = self.runtime.audio.find(workspace_id)
        if session is None:
            raise UserError("operator.voice_not_connected")
        destination_id = session.destination_id
        if destination_id is None:
            raise UserError("operator.voice_not_connected")
        if not session.output.connected:
            if not connect_if_needed:
                raise UserError("operator.voice_not_connected")
            await self.runtime.audio.connect(workspace_id, destination_id)
            log.info(
                "Local operator connected VC request=%s workspace=%s destination=%s",
                request_id,
                workspace_id,
                destination_id,
            )

        result = await self.runtime.registry.invoke(
            "speech.speak",
            SpeechSpeakRequest(
                text=text,
                title="Local operator announcement",
                voice_preset=voice_preset,
            ),
            InvocationContext(
                actor_id="local-operator",
                workspace_id=workspace_id,
                transport="local_operator",
                request_id=request_id,
                resource_ids=(destination_id,),
                origin_resource_id=destination_id,
            ),
        )
        if not isinstance(result, SpeechSpeakResponse):
            raise RuntimeError("speech.speak returned an unexpected response")
        log.info(
            "Local operator queued VC speech request=%s workspace=%s destination=%s "
            "characters=%s position=%s",
            request_id,
            workspace_id,
            destination_id,
            len(text),
            result.queue_position,
        )
        return {
            "ok": True,
            "version": _PROTOCOL_VERSION,
            "request_id": request_id,
            "workspace_id": workspace_id,
            "destination_id": result.destination_id,
            "queue_position": result.queue_position,
            "duration_seconds": result.duration_seconds,
            "playback_state": result.playback_state,
        }


async def request_vc_speech(
    *,
    socket_path: Path,
    workspace_id: str,
    text: str,
    voice_preset: str = "clear",
    request_id: str | None = None,
    connect_if_needed: bool = False,
) -> dict[str, object]:
    """Ask the already-running BOT to synthesize one bounded passage."""

    payload = {
        "version": _PROTOCOL_VERSION,
        "operation": "speak",
        "request_id": request_id or f"operator:{uuid.uuid4().hex}",
        "workspace_id": _required_snowflake(workspace_id),
        "text": _required_text(text),
        "voice_preset": _voice_preset(voice_preset),
        "connect_if_needed": connect_if_needed,
    }
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path)),
            timeout=_SERVER_READ_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        raise LocalOperatorError("operator.bot_not_running") from exc
    try:
        writer.write(_encode_message(payload))
        await writer.drain()
        raw = await asyncio.wait_for(
            reader.readline(),
            timeout=_CLIENT_RESPONSE_TIMEOUT_SECONDS,
        )
        if not raw or len(raw) > _MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            raise LocalOperatorError("operator.invalid_response")
        response = json.loads(raw.decode("utf-8"))
        if not isinstance(response, dict):
            raise LocalOperatorError("operator.invalid_response")
        typed_response = cast(dict[str, object], response)
        if typed_response.get("ok") is not True:
            code = typed_response.get("error")
            raise LocalOperatorError(
                code if isinstance(code, str) else "operator.invalid_response"
            )
        return typed_response
    except (TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalOperatorError("operator.invalid_response") from exc
    finally:
        writer.close()
        await writer.wait_closed()


def _required_request_id(value: object) -> str:
    if not isinstance(value, str) or _REQUEST_ID_PATTERN.fullmatch(value) is None:
        raise LocalOperatorError("operator.request_id_invalid")
    return value


def _required_snowflake(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.isdigit()
        or not 1 <= len(value) <= 20
        or int(value) <= 0
    ):
        raise LocalOperatorError("operator.workspace_id_invalid")
    return value


def _required_text(value: object) -> str:
    if not isinstance(value, str):
        raise LocalOperatorError("operator.text_invalid")
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_TEXT_CHARACTERS:
        raise LocalOperatorError("operator.text_invalid")
    return normalized


def _voice_preset(value: object) -> str:
    if not isinstance(value, str) or value not in _VOICE_PRESETS:
        raise LocalOperatorError("operator.voice_preset_invalid")
    return value


def _optional_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise LocalOperatorError("operator.invalid_request")
    return value


def _error_response(code: str) -> dict[str, object]:
    return {
        "ok": False,
        "version": _PROTOCOL_VERSION,
        "error": code,
    }


def _encode_message(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


async def _remove_stale_socket(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_socket():
        raise RuntimeError(f"Local operator path is not a socket: {path}")
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path)),
            timeout=0.25,
        )
    except OSError as exc:
        if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
            raise
        _unlink_socket(path)
        return
    writer.close()
    await writer.wait_closed()
    raise RuntimeError(f"Another local operator server is already active: {path}")


def _unlink_socket(path: Path) -> None:
    if path.is_socket():
        path.unlink()


def _parser() -> argparse.ArgumentParser:
    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    parser = argparse.ArgumentParser(
        prog="simajilord-vc-speak",
        description="Ask the running local METEOBOT process to speak in an active VC.",
    )
    parser.add_argument("--guild", required=True, dest="workspace_id")
    parser.add_argument(
        "--voice",
        choices=tuple(sorted(_VOICE_PRESETS)),
        default="clear",
        dest="voice_preset",
    )
    parser.add_argument("--socket", type=Path, default=data_dir / "operator.sock")
    parser.add_argument(
        "--connect-if-needed",
        action="store_true",
        help="Connect to this guild's saved VC before speaking when the BOT is in standby.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("text", nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(
            request_vc_speech(
                socket_path=args.socket.expanduser().resolve(),
                workspace_id=args.workspace_id,
                text=" ".join(args.text),
                voice_preset=args.voice_preset,
                connect_if_needed=args.connect_if_needed,
            )
        )
    except (LocalOperatorError, OSError) as exc:
        code = exc.code if isinstance(exc, LocalOperatorError) else "operator.io_failed"
        print(f"Error: {code}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "Queued VC speech: "
            f"destination={result.get('destination_id')} "
            f"position={result.get('queue_position')} "
            f"duration={result.get('duration_seconds')}s "
            f"state={result.get('playback_state')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
