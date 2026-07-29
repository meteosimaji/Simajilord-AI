from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock, call

import discord
import pytest

from simajilord.integrations.discord.attachment_io import read_attachment_bytes


def _http_error(status: int) -> discord.HTTPException:
    response = Mock()
    response.status = status
    response.reason = "attachment download failed"
    return discord.HTTPException(
        response,
        {"code": 0, "message": "attachment download failed"},
    )


@pytest.mark.asyncio
async def test_pdf_download_uses_canonical_url_before_unsupported_proxy() -> None:
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 123

    async def read(*, use_cached: bool = False) -> bytes:
        if use_cached:
            raise _http_error(415)
        return b"%PDF-1.7\nexample"

    attachment.read = AsyncMock(side_effect=read)

    assert await read_attachment_bytes(attachment) == b"%PDF-1.7\nexample"
    assert attachment.read.await_args_list == [call(use_cached=False)]


@pytest.mark.asyncio
async def test_attachment_download_falls_back_to_proxy_after_canonical_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 456
    attachment.read = AsyncMock(
        side_effect=(_http_error(404), b"cached attachment")
    )

    assert await read_attachment_bytes(attachment) == b"cached attachment"
    assert attachment.read.await_args_list == [
        call(use_cached=False),
        call(use_cached=True),
    ]
    assert "canonical_status=404" in caplog.text
    assert "https://" not in caplog.text


@pytest.mark.asyncio
async def test_attachment_download_logs_statuses_but_never_signed_urls(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 789
    attachment.read = AsyncMock(
        side_effect=(_http_error(403), _http_error(415))
    )

    with pytest.raises(discord.HTTPException):
        await read_attachment_bytes(attachment)

    assert "canonical_status=403" in caplog.text
    assert "proxy_status=415" in caplog.text
    assert "https://" not in caplog.text
