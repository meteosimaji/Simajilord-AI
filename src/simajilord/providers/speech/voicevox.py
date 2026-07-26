"""Local VOICEVOX Engine speech provider with bounded lifecycle management."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from simajilord.core.errors import ProviderError

log = logging.getLogger(__name__)

_MAX_QUERY_BYTES = 2_000_000
_MAX_WAVE_BYTES = 50_000_000


class VoicevoxSpeechProvider:
    """Synthesize WAV audio and optionally own a local VOICEVOX Engine process."""

    def __init__(
        self,
        *,
        base_url: str,
        speaker_id: int,
        timeout_seconds: float,
        engine_path: Path | None,
        auto_start: bool,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        parsed = urlsplit(normalized_url)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "http"
            or host not in {"127.0.0.1", "::1", "localhost"}
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("VOICEVOX base URL must be a loopback HTTP URL with a port.")
        if not 0 <= speaker_id <= 65_535:
            raise ValueError("VOICEVOX speaker ID is out of range.")
        if timeout_seconds <= 0:
            raise ValueError("VOICEVOX timeout must be positive.")
        if engine_path is not None and (
            not engine_path.is_file() or not os.access(engine_path, os.X_OK)
        ):
            raise ValueError("VOICEVOX engine path must be an executable file.")

        self.base_url = normalized_url
        self.host = host
        self.port = parsed.port
        self.speaker_id = speaker_id
        self.timeout_seconds = timeout_seconds
        self.engine_path = engine_path
        self.auto_start = auto_start
        self._session: aiohttp.ClientSession | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._start_lock = asyncio.Lock()

    async def synthesize(self, text: str, destination: Path) -> None:
        await self._ensure_ready()
        query = await self._audio_query(text)
        wave = await self._synthesis(query)
        if len(wave) < 44 or not wave.startswith(b"RIFF") or wave[8:12] != b"WAVE":
            raise ProviderError("VOICEVOX returned an invalid WAV file.")

        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            await asyncio.to_thread(temporary.write_bytes, wave)
            with suppress(OSError):
                temporary.chmod(0o600)
            await asyncio.to_thread(temporary.replace, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()

        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def _ensure_ready(self) -> None:
        if await self._version_is_ready():
            return
        async with self._start_lock:
            if await self._version_is_ready():
                return
            if not self.auto_start:
                raise ProviderError(
                    f"VOICEVOX Engine is not responding at {self.base_url}."
                )
            if self.engine_path is None:
                raise ProviderError("VOICEVOX Engine executable is not configured.")
            process = self._process
            if process is None or process.returncode is not None:
                try:
                    process = await asyncio.create_subprocess_exec(
                        str(self.engine_path),
                        "--host",
                        self.host,
                        "--port",
                        str(self.port),
                        "--cors_policy_mode",
                        "localapps",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                except OSError as exc:
                    raise ProviderError("VOICEVOX Engine could not be started.") from exc
                self._process = process
                log.info(
                    "Started local VOICEVOX Engine pid=%s endpoint=%s",
                    process.pid,
                    self.base_url,
                )

            deadline = asyncio.get_running_loop().time() + self.timeout_seconds
            while asyncio.get_running_loop().time() < deadline:
                if await self._version_is_ready():
                    return
                if process.returncode is not None:
                    raise ProviderError(
                        f"VOICEVOX Engine exited during startup with code {process.returncode}."
                    )
                await asyncio.sleep(0.2)
            raise ProviderError("VOICEVOX Engine startup timed out.")

    async def _version_is_ready(self) -> bool:
        try:
            async with self._client().get(f"{self.base_url}/version") as response:
                body = await _read_bounded(response, maximum=4_096)
                if response.status != 200:
                    return False
                version = json.loads(body)
                return isinstance(version, str) and bool(version.strip())
        except (aiohttp.ClientError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
            return False

    async def _audio_query(self, text: str) -> dict[str, object]:
        try:
            async with self._client().post(
                f"{self.base_url}/audio_query",
                params={"text": text, "speaker": str(self.speaker_id)},
            ) as response:
                body = await _read_bounded(response, maximum=_MAX_QUERY_BYTES)
                if response.status != 200:
                    raise ProviderError(_voicevox_http_error("audio query", response, body))
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise ProviderError("VOICEVOX audio query timed out.") from exc
        except aiohttp.ClientError as exc:
            raise ProviderError("VOICEVOX audio query failed.") from exc
        try:
            payload: object = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("VOICEVOX returned an invalid audio query.") from exc
        if not isinstance(payload, Mapping):
            raise ProviderError("VOICEVOX returned an invalid audio query.")
        return {str(key): value for key, value in payload.items()}

    async def _synthesis(self, query: Mapping[str, object]) -> bytes:
        try:
            async with self._client().post(
                f"{self.base_url}/synthesis",
                params={"speaker": str(self.speaker_id)},
                json=query,
                headers={"Accept": "audio/wav"},
            ) as response:
                body = await _read_bounded(response, maximum=_MAX_WAVE_BYTES)
                if response.status != 200:
                    raise ProviderError(_voicevox_http_error("synthesis", response, body))
                return body
        except ProviderError:
            raise
        except TimeoutError as exc:
            raise ProviderError("VOICEVOX synthesis timed out.") from exc
        except aiohttp.ClientError as exc:
            raise ProviderError("VOICEVOX synthesis failed.") from exc

    def _client(self) -> aiohttp.ClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            connector=aiohttp.TCPConnector(limit=4),
            auto_decompress=True,
        )
        self._session = session
        return session


async def _read_bounded(
    response: aiohttp.ClientResponse,
    *,
    maximum: int,
) -> bytes:
    raw_length = response.headers.get("Content-Length", "").strip()
    if raw_length.isdigit() and int(raw_length) > maximum:
        response.close()
        raise ProviderError("VOICEVOX response is too large.")
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1_024):
        total += len(chunk)
        if total > maximum:
            response.close()
            raise ProviderError("VOICEVOX response is too large.")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _voicevox_http_error(
    operation: str,
    response: aiohttp.ClientResponse,
    body: bytes,
) -> str:
    detail = body.decode(errors="replace").strip().replace("\n", " ")[:300]
    return (
        f"VOICEVOX {operation} failed with HTTP {response.status}: "
        f"{detail or response.reason or 'unknown error'}"
    )
