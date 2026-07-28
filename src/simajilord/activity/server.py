"""Serve the official Discord Activity without exposing audio mutations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import aiohttp
import discord
from aiohttp import web
from aiohttp.typedefs import Handler, Middleware

from simajilord.domain.audio import AudioKind, QueueSnapshot
from simajilord.services.audio import AudioSession

if TYPE_CHECKING:
    from simajilord.runtime import SimajilordRuntime

log = logging.getLogger(__name__)

_DISCORD_API_BASE = "https://discord.com/api/v10"
_MAX_REQUEST_BYTES = 16_384
_FIRST_MESSAGE_TIMEOUT_SECONDS = 15.0
_AUTH_CACHE_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class _ActivitySubscriber:
    """Identity and exact voice scope attached to one Activity socket."""

    websocket: web.WebSocketResponse
    channel_id: str
    user_id: str


def build_activity_snapshot(
    snapshot: QueueSnapshot,
    *,
    read_aloud_enabled: bool,
) -> dict[str, object]:
    """Return the minimal, public playback projection used by the Activity."""

    current = snapshot.current
    if current is not None and current.kind is not AudioKind.MUSIC:
        current = None
    pending = tuple(
        item for item in snapshot.pending if item.kind is AudioKind.MUSIC
    )
    next_items = pending[:3]
    if not next_items and snapshot.autoplay_next is not None:
        next_items = (snapshot.autoplay_next,)

    return {
        "sampled_at_ms": round(time() * 1_000),
        "connected": snapshot.connected,
        "paused": snapshot.paused,
        "speech_active": snapshot.speech_active,
        "position_seconds": round(snapshot.position_seconds, 3),
        "loop": snapshot.loop.value,
        "radio": snapshot.autoplay_enabled,
        "read_aloud": read_aloud_enabled,
        "levels": {
            "music_percent": round(snapshot.music_volume * 100),
            "read_aloud_percent": round(snapshot.speech_volume * 100),
        },
        "current": _activity_track(current),
        "up_next": [_activity_track(item) for item in next_items],
    }


def _activity_track(item: object | None) -> dict[str, object] | None:
    if item is None:
        return None
    title = getattr(item, "title", None)
    if not isinstance(title, str):
        return None
    return {
        "title": title[:300],
        "page_url": _public_url(getattr(item, "page_url", None)),
        "thumbnail_url": _public_url(getattr(item, "thumbnail_url", None)),
        "uploader": _bounded_text(getattr(item, "uploader", None), maximum=200),
        "requested_by": _bounded_text(
            getattr(item, "requested_by_name", None),
            maximum=100,
        ),
        "duration_seconds": round(
            max(0.0, float(getattr(item, "duration_seconds", 0.0))),
            3,
        ),
    }


def _bounded_text(value: object, *, maximum: int) -> str | None:
    return value[:maximum] if isinstance(value, str) and value else None


def _public_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if any(ord(character) < 32 for character in value):
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return value


class ActivityServer:
    """Authenticated, read-only bridge from Discord Activity to audio state."""

    def __init__(
        self,
        bot: discord.Client,
        runtime: SimajilordRuntime,
    ) -> None:
        self.bot = bot
        self.runtime = runtime
        self._app = web.Application(
            client_max_size=_MAX_REQUEST_BYTES,
            middlewares=[cast(Middleware, _security_headers)],
        )
        self._app.router.add_get("/", self._index)
        self._app.router.add_get("/api/config", self._config)
        self._app.router.add_post("/api/token", self._token)
        self._app.router.add_get("/api/audio/ws", self._audio_websocket)
        self._static_dir = Path(__file__).with_name("static")
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._http: aiohttp.ClientSession | None = None
        self._subscribers: dict[
            str,
            dict[web.WebSocketResponse, _ActivitySubscriber],
        ] = {}
        self._identity_cache: dict[str, tuple[float, str]] = {}

    async def start(self) -> None:
        """Start only when the Activity is explicitly enabled."""

        if not self.runtime.settings.activity_enabled:
            return
        if not self._static_dir.joinpath("index.html").is_file():
            raise RuntimeError(
                "Discord Activity assets are missing. Run `npm run build` in activity/."
            )
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            raise_for_status=False,
        )
        self._app.router.add_static(
            "/assets",
            self._static_dir / "assets",
            append_version=True,
        )
        self._runner = web.AppRunner(
            self._app,
            access_log=None,
            handle_signals=False,
        )
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner,
            self.runtime.settings.activity_host,
            self.runtime.settings.activity_port,
        )
        await self._site.start()
        self.runtime.audio.add_state_listener(self.on_audio_state_changed)
        log.info(
            "Discord Activity ready on http://%s:%s",
            self.runtime.settings.activity_host,
            self.runtime.settings.activity_port,
        )

    async def close(self) -> None:
        self.runtime.audio.remove_state_listener(self.on_audio_state_changed)
        subscribers = tuple(
            subscriber.websocket
            for group in self._subscribers.values()
            for subscriber in group.values()
        )
        if subscribers:
            await asyncio.gather(
                *(subscriber.close(code=1001, message=b"shutdown") for subscriber in subscribers),
                return_exceptions=True,
            )
        self._subscribers.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        if self._http is not None:
            await self._http.close()
            self._http = None
        self._identity_cache.clear()

    async def on_audio_state_changed(self, session: AudioSession) -> None:
        subscribers = tuple(
            self._subscribers.get(session.workspace_id, {}).values()
        )
        if not subscribers:
            return
        payload = await self._snapshot_payload(session)
        for subscriber in subscribers:
            try:
                current = await self._authorized_session(
                    session.workspace_id,
                    subscriber.channel_id,
                    subscriber.user_id,
                )
                if current is not session:
                    raise web.HTTPForbidden(text="The audio session changed.")
                await subscriber.websocket.send_json(payload)
            except Exception:
                with suppress(Exception):
                    await subscriber.websocket.close(
                        code=1008,
                        message=b"voice access changed",
                    )
                self._discard_subscriber(
                    session.workspace_id,
                    subscriber.websocket,
                )

    async def _index(self, _: web.Request) -> web.FileResponse:
        return web.FileResponse(self._static_dir / "index.html")

    async def _config(self, _: web.Request) -> web.Response:
        return web.json_response(
            {
                "client_id": str(self.runtime.settings.application_id),
                "read_only": True,
            }
        )

    async def _token(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, web.HTTPBadRequest):
            raise web.HTTPBadRequest(text="Invalid JSON.") from None
        code = body.get("code") if isinstance(body, dict) else None
        if not isinstance(code, str) or not 1 <= len(code) <= 2_000:
            raise web.HTTPBadRequest(text="A valid authorization code is required.")
        secret = self.runtime.settings.activity_client_secret
        if secret is None:
            raise web.HTTPServiceUnavailable(text="Activity authentication is disabled.")
        http = self._require_http()
        response = await http.post(
            f"{_DISCORD_API_BASE}/oauth2/token",
            data={
                "client_id": str(self.runtime.settings.application_id),
                "client_secret": secret,
                "grant_type": "authorization_code",
                "code": code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        data = await _bounded_json(response)
        access_token = data.get("access_token")
        if response.status != 200 or not isinstance(access_token, str):
            raise web.HTTPUnauthorized(text="Discord authorization failed.")
        return web.json_response(
            {
                "access_token": access_token,
                "token_type": data.get("token_type", "Bearer"),
                "expires_in": data.get("expires_in"),
            }
        )

    async def _audio_websocket(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(
            heartbeat=30,
            max_msg_size=_MAX_REQUEST_BYTES,
            autoclose=True,
            autoping=True,
        )
        await websocket.prepare(request)
        workspace_id: str | None = None
        try:
            message = await websocket.receive_json(
                timeout=_FIRST_MESSAGE_TIMEOUT_SECONDS,
            )
            if not isinstance(message, dict):
                raise ValueError
            access_token = message.get("access_token")
            workspace_id = _snowflake(message.get("guild_id"))
            channel_id = _snowflake(message.get("channel_id"))
            if not isinstance(access_token, str) or not access_token:
                raise ValueError
            user_id = await self._discord_user_id(access_token)
            session = await self._authorized_session(
                workspace_id,
                channel_id,
                user_id,
            )
            self._subscribers.setdefault(workspace_id, {})[websocket] = (
                _ActivitySubscriber(
                    websocket=websocket,
                    channel_id=channel_id,
                    user_id=user_id,
                )
            )
            await websocket.send_json(await self._snapshot_payload(session))
            async for incoming in websocket:
                if incoming.type is aiohttp.WSMsgType.TEXT:
                    if incoming.data == "refresh":
                        session = await self._authorized_session(
                            workspace_id,
                            channel_id,
                            user_id,
                        )
                        await websocket.send_json(
                            await self._snapshot_payload(session)
                        )
                        continue
                    await websocket.close(code=1008, message=b"read-only")
                    break
                if incoming.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break
        except (TimeoutError, ValueError, json.JSONDecodeError):
            await websocket.close(code=1008, message=b"invalid authentication")
        except web.HTTPException as exc:
            await websocket.close(
                code=1008,
                message=(exc.text or "request denied").encode()[:120],
            )
        finally:
            if workspace_id is not None:
                self._discard_subscriber(workspace_id, websocket)
        return websocket

    async def _discord_user_id(self, access_token: str) -> str:
        cache_key = hashlib.sha256(access_token.encode()).hexdigest()
        now = time()
        cached = self._identity_cache.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]
        response = await self._require_http().get(
            f"{_DISCORD_API_BASE}/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        data = await _bounded_json(response)
        user_id = data.get("id")
        if response.status != 200 or not isinstance(user_id, str):
            raise web.HTTPUnauthorized(text="Discord identity verification failed.")
        self._identity_cache[cache_key] = (now + _AUTH_CACHE_SECONDS, user_id)
        return user_id

    async def _authorized_session(
        self,
        workspace_id: str,
        channel_id: str,
        user_id: str,
    ) -> AudioSession:
        guild = self.bot.get_guild(int(workspace_id))
        if guild is None:
            raise web.HTTPForbidden(text="This server is unavailable.")
        member = guild.get_member(int(user_id))
        if member is None:
            with suppress(discord.DiscordException):
                member = await guild.fetch_member(int(user_id))
        voice = member.voice if member is not None else None
        if voice is None or voice.channel is None or voice.channel.id != int(channel_id):
            raise web.HTTPForbidden(text="Join this voice channel to view its player.")
        session = self.runtime.audio.find(workspace_id)
        if (
            session is None
            or session.destination_id != channel_id
            or not session.output.connected
        ):
            raise web.HTTPNotFound(text="No active audio session is available here.")
        return session

    async def _snapshot_payload(self, session: AudioSession) -> dict[str, object]:
        snapshot = await session.snapshot()
        route = self.runtime.read_aloud.get(session.workspace_id)
        return {
            "type": "audio_state",
            "audio": build_activity_snapshot(
                snapshot,
                read_aloud_enabled=route is not None,
            ),
        }

    def _discard_subscriber(
        self,
        workspace_id: str,
        websocket: web.WebSocketResponse,
    ) -> None:
        subscribers = self._subscribers.get(workspace_id)
        if subscribers is None:
            return
        subscribers.pop(websocket, None)
        if not subscribers:
            self._subscribers.pop(workspace_id, None)

    def _require_http(self) -> aiohttp.ClientSession:
        if self._http is None:
            raise web.HTTPServiceUnavailable(text="Activity server is not ready.")
        return self._http


def _snowflake(value: object) -> str:
    if not isinstance(value, str) or not value.isdigit() or int(value) <= 0:
        raise ValueError
    return value


async def _bounded_json(response: aiohttp.ClientResponse) -> dict[str, Any]:
    body = await response.read()
    if len(body) > _MAX_REQUEST_BYTES:
        raise web.HTTPBadGateway(text="Discord returned an oversized response.")
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        raise web.HTTPBadGateway(text="Discord returned invalid JSON.") from None
    if not isinstance(value, dict):
        raise web.HTTPBadGateway(text="Discord returned an invalid response.")
    return value


@web.middleware
async def _security_headers(
    request: web.Request,
    handler: Handler,
) -> web.StreamResponse:
    response = await handler(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' https: data:; "
        "connect-src 'self' wss: https:; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors https://discord.com https://*.discord.com"
    )
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), display-capture=(), geolocation=()"
    )
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response
