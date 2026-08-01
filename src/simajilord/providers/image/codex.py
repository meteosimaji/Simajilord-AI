"""GPT Image 2 generation through Codex app-server and saved OAuth."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image

from simajilord.core.errors import ProviderError
from simajilord.domain.image import ImageGenerationModel
from simajilord.providers.codex_features import (
    CODEX_THREAD_HISTORY_MODE,
    codex_feature_arguments,
)

from .base import ImageProgressCallback, ImageProviderResult

_MAX_IMAGE_BYTES = 50_000_000
_APP_SERVER_STDOUT_LIMIT_BYTES = 80_000_000
_MODEL_LABEL = "GPT Image 2・Codex OAuth"
log = logging.getLogger(__name__)


class CodexImageProvider:
    """Generate one image file through the Codex hosted image tool."""

    def __init__(
        self,
        *,
        executable: str,
        model: str,
        workspace_dir: Path,
        timeout_seconds: float,
    ) -> None:
        self.executable = executable
        self.model = model
        self.workspace_dir = workspace_dir.resolve()
        self.timeout_seconds = timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_sequence = 0
        self._pending: dict[int, asyncio.Future[object]] = {}
        self._notifications: asyncio.Queue[tuple[str, dict[str, object]]] = (
            asyncio.Queue()
        )
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._generation_lock = asyncio.Lock()

    async def generate(
        self,
        *,
        brief_json: str,
        destination: Path,
        width: int,
        height: int,
        seed: int,
        model: ImageGenerationModel = ImageGenerationModel.GPT_IMAGE_2,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageProviderResult:
        del seed  # The hosted Codex image tool does not expose deterministic seeds.
        if model is not ImageGenerationModel.GPT_IMAGE_2:
            raise ProviderError("Only gpt-image-2 is supported for image generation.")
        prompt = _image_prompt(brief_json, width=width, height=height)
        started = monotonic()
        async with self._generation_lock:
            async with asyncio.timeout(self.timeout_seconds):
                await self._ensure_started()
                _clear_queue(self._notifications)
                thread_id = await self._start_thread()
                if on_progress is not None:
                    await on_progress(1, 12)
                turn_id = await self._start_turn(thread_id, prompt)
                image_item = await self._await_image(
                    thread_id,
                    turn_id,
                    on_progress=on_progress,
                )
                actual_width, actual_height = await asyncio.to_thread(
                    self._import_image,
                    image_item,
                    destination,
                )
        return ImageProviderResult(
            generation_seconds=monotonic() - started,
            model=_MODEL_LABEL,
            width=actual_width,
            height=actual_height,
        )

    async def close(self) -> None:
        process = self._process
        self._process = None
        tasks = tuple(
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None
        )
        self._reader_task = None
        self._stderr_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.terminate()
            with suppress(TimeoutError):
                async with asyncio.timeout(2):
                    await process.wait()
            if process.returncode is None:
                process.kill()
                await process.wait()
        error = ProviderError("Codex image provider closed.")
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        _clear_queue(self._notifications)

    async def _ensure_started(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        async with self._start_lock:
            if self._process is not None and self._process.returncode is None:
                return
            executable = _resolve_executable(self.executable)
            self.workspace_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            environment = dict(os.environ)
            environment.setdefault("RUST_LOG", "warn")
            try:
                process = await asyncio.create_subprocess_exec(
                    executable,
                    "app-server",
                    "--listen",
                    "stdio://",
                    *codex_feature_arguments(allow_image_generation=True),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.workspace_dir,
                    env=environment,
                    limit=_APP_SERVER_STDOUT_LIMIT_BYTES,
                )
            except OSError as exc:
                raise ProviderError("Codex image app-server could not start.") from exc
            self._process = process
            self._reader_task = asyncio.create_task(
                self._reader_loop(process),
                name="simajilord-codex-image-reader",
            )
            self._stderr_task = asyncio.create_task(
                self._stderr_loop(process),
                name="simajilord-codex-image-stderr",
            )
            try:
                await self._request(
                    "initialize",
                    {
                        "clientInfo": {
                            "name": "simajilord-image",
                            "title": "Simajilord GPT Image Runtime",
                            "version": "0.1.0",
                        },
                        "capabilities": {
                            "experimentalApi": True,
                            "optOutNotificationMethods": [
                                "item/agentMessage/delta",
                            ],
                        },
                    },
                )
                await self._notify("initialized")
            except Exception:
                await self.close()
                raise

    async def _start_thread(self) -> str:
        response = _object(
            await self._request(
                "thread/start",
                {
                    "model": self.model,
                    "allowProviderModelFallback": False,
                    "cwd": str(self.workspace_dir),
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "baseInstructions": (
                        "You are a single-purpose image generator. Use the built-in "
                        "image generation tool exactly once for the supplied production "
                        "brief. Never use shell, web, plugins, or substitute artwork. "
                        "After the image tool finishes, return a short confirmation."
                    ),
                    "developerInstructions": (
                        "Preserve every requested visible fact. Do not post or upload the "
                        "result; the host imports the generated local file."
                    ),
                    "dynamicTools": [],
                    "environments": [],
                    "runtimeWorkspaceRoots": [],
                    "selectedCapabilityRoots": [],
                    "config": {
                        "allow_login_shell": False,
                        "web_search": "disabled",
                        "tool_output_token_limit": 1_000,
                    },
                    "ephemeral": True,
                    "historyMode": CODEX_THREAD_HISTORY_MODE,
                    "sessionStartSource": "startup",
                },
            ),
            "thread/start result",
        )
        thread = _object(response.get("thread"), "thread/start thread")
        return _text(thread.get("id"), "thread id")

    async def _start_turn(self, thread_id: str, prompt: str) -> str:
        response = _object(
            await self._request(
                "turn/start",
                {
                    "threadId": thread_id,
                    "input": [{"type": "text", "text": prompt}],
                    "model": self.model,
                    "effort": "low",
                    "approvalPolicy": "never",
                    "sandboxPolicy": {"type": "readOnly"},
                },
            ),
            "turn/start result",
        )
        turn = _object(response.get("turn"), "turn/start turn")
        return _text(turn.get("id"), "turn id")

    async def _await_image(
        self,
        thread_id: str,
        turn_id: str,
        *,
        on_progress: ImageProgressCallback | None,
    ) -> dict[str, object]:
        image_item: dict[str, object] | None = None
        image_started = False
        observed_item_types: list[str] = []
        while True:
            method, params = await self._notifications.get()
            if _optional_text(params.get("threadId")) not in {None, thread_id}:
                continue
            notification_turn_id = _optional_text(params.get("turnId"))
            if notification_turn_id not in {None, turn_id}:
                continue
            item = params.get("item")
            if isinstance(item, dict) and isinstance(item.get("type"), str):
                observed_item_types.append(str(item["type"]))
            if method == "item/started" and _is_image_item(item):
                image_started = True
                if on_progress is not None:
                    await on_progress(3, 12)
                continue
            if method == "item/completed" and _is_image_item(item):
                assert isinstance(item, dict)
                image_item = item
                if on_progress is not None:
                    await on_progress(12, 12)
                continue
            if method != "turn/completed":
                continue
            turn = _object(params.get("turn"), "turn/completed turn")
            if turn.get("status") != "completed":
                raise ProviderError(_turn_error(turn))
            if image_item is None:
                image_item = _image_item_from_turn(turn.get("items"))
            if image_item is None:
                observed = ", ".join(observed_item_types[-12:]) or "none"
                if not image_started:
                    raise ProviderError(
                        "The explicit $imagegen execution completed without starting "
                        f"image generation; observed item types: {observed}."
                    )
                raise ProviderError(
                    "Codex completed after starting image generation but without "
                    f"returning an image file; observed item types: {observed}."
                )
            status = _optional_text(image_item.get("status"))
            if status is not None and status.casefold() in {
                "failed",
                "error",
                "cancelled",
                "canceled",
            }:
                raise ProviderError(f"Codex image generation ended with status {status}.")
            return image_item

    def _import_image(
        self,
        item: dict[str, object],
        destination: Path,
    ) -> tuple[int, int]:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.partial")
        temporary.unlink(missing_ok=True)
        source: Path | None = None
        try:
            saved_path = _optional_text(item.get("savedPath"))
            if saved_path is not None:
                source = _validated_saved_path(saved_path)
                if source.stat().st_size > _MAX_IMAGE_BYTES:
                    raise ProviderError("Codex generated image exceeds the file limit.")
                shutil.copyfile(source, temporary)
            else:
                result = _text(item.get("result"), "image result")
                temporary.write_bytes(_decode_image_result(result))
            actual_width, actual_height = _verified_png_dimensions(temporary)
            os.replace(temporary, destination)
            destination.chmod(0o600)
            if source is not None:
                source.unlink(missing_ok=True)
                with suppress(OSError):
                    source.parent.rmdir()
            return actual_width, actual_height
        finally:
            temporary.unlink(missing_ok=True)

    async def _request(self, method: str, params: dict[str, object]) -> object:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise ProviderError("Codex image app-server is unavailable.")
        self._request_sequence += 1
        request_id = self._request_sequence
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def _notify(
        self,
        method: str,
        params: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {"method": method}
        if params is not None:
            payload["params"] = params
        await self._send(payload)

    async def _send(self, payload: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise ProviderError("Codex image app-server is not writable.")
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        async with self._write_lock:
            process.stdin.write(encoded)
            await process.stdin.drain()

    async def _reader_loop(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        cancelled = False
        try:
            while line := await process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                request_id = message.get("id")
                method = message.get("method")
                if request_id is not None and isinstance(method, str):
                    await self._send(
                        {
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "No client-side tools are available.",
                            },
                        }
                    )
                elif request_id is not None:
                    self._finish_request(request_id, message)
                elif isinstance(method, str):
                    params = message.get("params")
                    if isinstance(params, dict):
                        await self._notifications.put((method, params))
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            log.exception("Codex image app-server reader failed")
        finally:
            if not cancelled:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                if self._process is process:
                    self._process = None
                error = ProviderError("Codex image app-server stopped unexpectedly.")
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(error)

    async def _stderr_loop(self, process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        try:
            while await process.stderr.readline():
                pass
        except asyncio.CancelledError:
            raise

    def _finish_request(
        self,
        request_id: object,
        message: dict[str, object],
    ) -> None:
        if not isinstance(request_id, int):
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, dict):
            detail = _optional_text(error.get("message")) or "Protocol request failed."
            future.set_exception(ProviderError(f"Codex image request failed: {detail}"))
        else:
            future.set_result(message.get("result"))


def _image_prompt(brief_json: str, *, width: int, height: int) -> str:
    try:
        brief = json.loads(brief_json)
    except json.JSONDecodeError as exc:
        raise ProviderError("The image brief is not valid JSON.") from exc
    if not isinstance(brief, dict):
        raise ProviderError("The image brief must be a JSON object.")
    aspect = (
        "square"
        if width == height
        else ("landscape" if width > height else "portrait")
    )
    return (
        "$imagegen\n"
        "Generate exactly one image with the built-in image generation tool.\n"
        f"Requested composition: {aspect} ({width}:{height} target ratio).\n"
        "Treat this JSON as the complete production brief; preserve all positive "
        "requirements and avoid-list constraints. Do not render the JSON itself "
        "or add unrequested text.\n"
        f"{json.dumps(brief, ensure_ascii=False, separators=(',', ':'))}"
    )


def _validated_saved_path(value: str) -> Path:
    source = Path(value).expanduser().resolve(strict=True)
    root = (
        Path(os.getenv("CODEX_HOME", str(Path.home() / ".codex")))
        .expanduser()
        .resolve()
        / "generated_images"
    )
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ProviderError("Codex returned an image outside its generated directory.") from exc
    if not source.is_file():
        raise ProviderError("Codex generated image file is unavailable.")
    return source


def _decode_image_result(value: str) -> bytes:
    encoded = value.partition(",")[2] if value.startswith("data:image/") else value
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ProviderError("Codex returned invalid image data.") from exc
    if len(payload) > _MAX_IMAGE_BYTES:
        raise ProviderError("Codex generated image exceeds the file limit.")
    return payload


def _verified_png_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ProviderError("Codex generated image is not a PNG file.")
            return image.width, image.height
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError("Codex generated image failed PNG validation.") from exc


def _resolve_executable(value: str) -> str:
    if "/" in value:
        path = Path(value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise ProviderError("Codex is not installed or CODEX_EXECUTABLE is invalid.")
    return resolved


def _image_item_from_turn(value: object) -> dict[str, object] | None:
    if not isinstance(value, list):
        return None
    return next(
        (
            item
            for item in reversed(value)
            if isinstance(item, dict) and item.get("type") == "imageGeneration"
        ),
        None,
    )


def _is_image_item(value: object) -> bool:
    return isinstance(value, dict) and value.get("type") == "imageGeneration"


def _turn_error(turn: dict[str, object]) -> str:
    error = turn.get("error")
    if isinstance(error, dict):
        detail = _optional_text(error.get("message"))
        if detail is not None:
            return f"Codex image generation failed: {detail}"
    return f"Codex image turn ended with status {turn.get('status')}."


def _clear_queue(queue: asyncio.Queue[Any]) -> None:
    while not queue.empty():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ProviderError(f"Codex returned an invalid {label}.")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderError(f"Codex returned an invalid {label}.")
    return value


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
