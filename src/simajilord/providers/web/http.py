"""Bounded public-network HTTP client with redirect and DNS revalidation."""

from __future__ import annotations

import ipaddress
import socket
from contextlib import suppress
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver

from simajilord.core.errors import WebError
from simajilord.domain.web import FetchedWebResource

_MAX_WEB_URL_LENGTH = 2_048
_MAX_REDIRECTS = 4
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_USER_AGENT = "Simajilord/0.1 (+https://simajilord.com)"


class _PublicOnlyResolver(AbstractResolver):
    """Reject a hostname if any address used by aiohttp is not globally routable."""

    def __init__(self) -> None:
        self._delegate = DefaultResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        records = await self._delegate.resolve(host, port, family)
        if not records:
            raise OSError("The web host did not resolve.")
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for record in records:
            raw_address = str(record["host"]).split("%", maxsplit=1)[0]
            try:
                addresses.append(ipaddress.ip_address(raw_address))
            except ValueError as exc:
                raise OSError("The web host returned an invalid address.") from exc
        if any(not address.is_global for address in addresses):
            raise OSError("The web host resolves to a non-public address.")
        return records

    async def close(self) -> None:
        await self._delegate.close()


class AiohttpPublicWebFetcher:
    """Fetch a small public resource while preventing host-network access."""

    def __init__(self, *, timeout_seconds: float) -> None:
        self.timeout_seconds = timeout_seconds
        self._resolver: _PublicOnlyResolver | None = None
        self._session: aiohttp.ClientSession | None = None

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int,
    ) -> FetchedWebResource:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        current_url = normalize_public_web_url(url)
        session = self._client()
        try:
            for redirect_count in range(_MAX_REDIRECTS + 1):
                async with session.get(
                    current_url,
                    allow_redirects=False,
                    headers={
                        "Accept": (
                            "text/html,application/xhtml+xml,application/pdf,"
                            "text/plain,application/json,application/xml;q=0.9,*/*;q=0.1"
                        ),
                        "User-Agent": _USER_AGENT,
                    },
                ) as response:
                    if response.status in _REDIRECT_STATUSES:
                        location = response.headers.get("Location", "").strip()
                        if not location:
                            raise WebError("redirect_invalid", "Redirect has no Location header.")
                        if redirect_count >= _MAX_REDIRECTS:
                            raise WebError("too_many_redirects", "Too many URL redirects.")
                        current_url = normalize_public_web_url(urljoin(current_url, location))
                        continue
                    if response.status >= 400:
                        category = (
                            "url_rejected"
                            if 400 <= response.status < 500
                            else "upstream_unavailable"
                        )
                        raise WebError(category, f"URL returned HTTP {response.status}.")
                    content_length = _content_length(response.headers.get("Content-Length"))
                    if content_length is not None and content_length > max_bytes:
                        raise WebError("response_too_large", "URL response is too large.")
                    body = await _read_bounded(response, max_bytes=max_bytes)
                    content_type = (
                        response.headers.get("Content-Type", "application/octet-stream")
                        .split(";", maxsplit=1)[0]
                        .strip()
                        .lower()
                    )
                    return FetchedWebResource(
                        final_url=current_url,
                        content_type=content_type or "application/octet-stream",
                        charset=response.charset,
                        body=body,
                    )
        except WebError:
            raise
        except TimeoutError as exc:
            raise WebError("timeout", "URL fetch timed out.") from exc
        except aiohttp.InvalidURL as exc:
            raise WebError("url_invalid", "The URL is invalid.") from exc
        except aiohttp.ClientConnectorError as exc:
            raise WebError("url_unresolvable", "The web host could not be reached.") from exc
        except aiohttp.ClientError as exc:
            raise WebError("fetch_failed", "The URL fetch failed.") from exc
        raise WebError("fetch_failed", "The URL fetch did not complete.")

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()
        resolver = self._resolver
        self._resolver = None
        if resolver is not None:
            await resolver.close()

    def _client(self) -> aiohttp.ClientSession:
        session = self._session
        if session is not None and not session.closed:
            return session
        resolver = self._resolver
        if resolver is None:
            resolver = _PublicOnlyResolver()
            self._resolver = resolver
        connector = aiohttp.TCPConnector(
            resolver=resolver,
            use_dns_cache=False,
            limit=8,
        )
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
            auto_decompress=True,
        )
        self._session = session
        return session


def normalize_public_web_url(value: str) -> str:
    """Return a normalized public HTTP(S) URL or raise a stable web error."""

    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_WEB_URL_LENGTH
        or any(character.isspace() or ord(character) < 32 for character in normalized)
    ):
        raise WebError("url_invalid", "Provide a valid public URL.")
    try:
        parsed = urlsplit(normalized)
        port = parsed.port
    except ValueError as exc:
        raise WebError("url_invalid", "Provide a valid public URL.") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in {"http", "https"} or not host:
        raise WebError("url_invalid", "Only HTTP and HTTPS URLs are supported.")
    if parsed.username is not None or parsed.password is not None:
        raise WebError("url_invalid", "URLs with embedded credentials are not supported.")
    allowed_port = 80 if parsed.scheme == "http" else 443
    if port not in {None, allowed_port}:
        raise WebError("url_invalid", "Custom URL ports are not supported.")
    _assert_public_host(host)
    sanitized = SplitResult(
        scheme=parsed.scheme,
        netloc=parsed.netloc,
        path=parsed.path or "/",
        query=parsed.query,
        fragment="",
    )
    return urlunsplit(sanitized)


def _assert_public_host(host: str) -> None:
    if (
        host == "localhost"
        or host.endswith(".localhost")
        or host.endswith(".local")
        or host.endswith(".internal")
    ):
        raise WebError("url_private", "Private and local web addresses are not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            raise WebError("url_private", "Private and local web addresses are not allowed.")
        return
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise WebError("url_invalid", "The web hostname is invalid.") from exc
    if len(ascii_host) > 253 or "." not in ascii_host:
        raise WebError("url_invalid", "The web hostname is invalid.")
    for label in ascii_host.split("."):
        if (
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or any(not (character.isalnum() or character == "-") for character in label)
        ):
            raise WebError("url_invalid", "The web hostname is invalid.")


async def _read_bounded(
    response: aiohttp.ClientResponse,
    *,
    max_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1_024):
        total += len(chunk)
        if total > max_bytes:
            response.close()
            raise WebError("response_too_large", "URL response is too large.")
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    with suppress(ValueError):
        length = int(value)
        return length if length >= 0 else None
    return None
