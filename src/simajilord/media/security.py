"""Network-safe media input policy for user-controlled references."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from simajilord.core.errors import UserError

_MAX_MEDIA_URL_LENGTH = 2_048
_DNS_TIMEOUT_SECONDS = 5.0


def is_supported_media_url(value: str) -> bool:
    """Return whether a URL is structurally safe for a public media provider.

    This intentionally does not maintain a site allowlist. Provider support is
    determined by the provider itself; this boundary only rejects transports and
    authorities that should never be reached from user-controlled input.
    """

    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or not host
        or len(value) > _MAX_MEDIA_URL_LENGTH
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port not in (None, 443):
        return False
    return _host_is_structurally_public(host)


def validate_media_url(value: str) -> str:
    normalized = value.strip()
    if not is_supported_media_url(normalized):
        raise UserError("media.url_unsupported")
    return normalized


async def validate_public_media_url(value: str) -> str:
    """Validate a public HTTPS URL, including its currently resolved addresses."""

    normalized = validate_media_url(value)
    host = urlsplit(normalized).hostname
    if host is None:
        raise UserError("media.url_unsupported")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if not address.is_global:
            raise UserError("media.url_private")
        return normalized

    loop = asyncio.get_running_loop()
    try:
        async with asyncio.timeout(_DNS_TIMEOUT_SECONDS):
            records = await loop.getaddrinfo(
                host,
                443,
                type=socket.SOCK_STREAM,
            )
    except (TimeoutError, OSError) as exc:
        raise UserError("media.url_unresolvable") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for record in records:
        raw_address = record[4][0].split("%", maxsplit=1)[0]
        try:
            addresses.add(ipaddress.ip_address(raw_address))
        except ValueError:
            continue
    if not addresses:
        raise UserError("media.url_unresolvable")
    if any(not address.is_global for address in addresses):
        raise UserError("media.url_private")
    return normalized


def normalize_media_reference(value: str) -> str:
    normalized = _normalize_media_text(value)
    if "://" in normalized:
        return validate_media_url(normalized)
    return f"ytsearch1:{normalized}"


def normalize_media_query(value: str) -> str:
    """Validate free text before placing it inside a bounded provider search."""

    normalized = _normalize_media_text(value)
    if "://" in normalized:
        raise UserError("media.query_url_not_allowed")
    return normalized


def _normalize_media_text(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise UserError("media.reference_required")
    if len(normalized) > 300:
        raise UserError("media.reference_too_long")
    return normalized


def _host_is_structurally_public(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        pass
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if len(ascii_host) > 253 or "." not in ascii_host:
        return False
    labels = ascii_host.split(".")
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )
