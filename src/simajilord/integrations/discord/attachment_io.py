"""Safe Discord attachment downloads without exposing signed CDN URLs."""

from __future__ import annotations

import logging

import discord

log = logging.getLogger(__name__)


async def read_attachment_bytes(attachment: discord.Attachment) -> bytes:
    """Read the canonical attachment URL, falling back to Discord's media proxy.

    discord.py's ``use_cached=True`` selects only ``proxy_url``. Discord's proxy
    does not support every attachment type (notably some documents), so it must
    be a fallback rather than the primary download path.
    """

    try:
        return await attachment.read(use_cached=False)
    except discord.HTTPException as canonical_error:
        canonical_status = _http_status(canonical_error)
        log.info(
            "Canonical Discord attachment download failed; retrying media proxy "
            "attachment_id=%s canonical_status=%s",
            attachment.id,
            canonical_status,
        )
        try:
            return await attachment.read(use_cached=True)
        except discord.HTTPException as proxy_error:
            log.warning(
                "Discord attachment download failed on canonical and proxy endpoints "
                "attachment_id=%s canonical_status=%s proxy_status=%s",
                attachment.id,
                canonical_status,
                _http_status(proxy_error),
            )
            raise


def _http_status(error: discord.HTTPException) -> int | None:
    status = getattr(error, "status", None)
    return status if isinstance(status, int) else None
