"""Narrow media input policy for user-controlled references."""

from __future__ import annotations

from urllib.parse import urlsplit

from simajilord.core.errors import UserError

_ALLOWED_HOSTS = (
    "youtube.com",
    "youtu.be",
    "tiktok.com",
)


def is_supported_media_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != "https" or not host:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    if port not in (None, 443):
        return False
    return any(host == root or host.endswith(f".{root}") for root in _ALLOWED_HOSTS)


def validate_media_url(value: str) -> str:
    normalized = value.strip()
    if len(normalized) > 2_048 or not is_supported_media_url(normalized):
        raise UserError("media.url_unsupported")
    return normalized


def normalize_media_reference(value: str) -> str:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise UserError("media.reference_required")
    if len(normalized) > 300:
        raise UserError("media.reference_too_long")
    if "://" in normalized:
        return validate_media_url(normalized)
    return f"ytsearch1:{normalized}"
