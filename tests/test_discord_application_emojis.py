from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import discord
import pytest

from simajilord.integrations.discord.application_emojis import (
    ApplicationEmojiCatalog,
    ApplicationEmojiName,
    application_emoji,
)


@dataclass(frozen=True, slots=True)
class _FakeEmoji:
    id: int
    name: str
    animated: bool = False

    def __str__(self) -> str:
        prefix = "a" if self.animated else ""
        return f"<{prefix}:{self.name}:{self.id}>"


@pytest.mark.asyncio
async def test_application_emojis_require_matching_id_and_exact_name() -> None:
    catalog = ApplicationEmojiCatalog(
        {
            ApplicationEmojiName.LOADING: 10,
            ApplicationEmojiName.SUCCESS: 11,
            ApplicationEmojiName.WARNING: 12,
            ApplicationEmojiName.AUDIO_WAVE: None,
            ApplicationEmojiName.RADIO: 13,
        }
    )
    client = Mock(spec=discord.Client)
    client.fetch_application_emojis = AsyncMock(
        return_value=[
            _FakeEmoji(10, "loading", animated=True),
            _FakeEmoji(11, "renamed_success"),
            _FakeEmoji(13, "radio"),
        ]
    )

    result = await catalog.refresh(client)

    assert result.loaded == (
        ApplicationEmojiName.LOADING,
        ApplicationEmojiName.RADIO,
    )
    assert result.problems == (
        "success:name_mismatch:renamed_success",
        "warning:missing",
    )
    assert catalog.render(ApplicationEmojiName.LOADING) == "<a:loading:10>"
    assert catalog.render(ApplicationEmojiName.RADIO) == "<:radio:13>"
    assert catalog.render(ApplicationEmojiName.SUCCESS) == "✅"
    assert catalog.render(ApplicationEmojiName.WARNING) == "⚠️"
    assert catalog.render(ApplicationEmojiName.AUDIO_WAVE) == "〰️"


@pytest.mark.asyncio
async def test_application_emoji_fetch_failure_is_non_fatal() -> None:
    catalog = ApplicationEmojiCatalog(
        {
            ApplicationEmojiName.LOADING: 10,
            ApplicationEmojiName.SUCCESS: None,
            ApplicationEmojiName.WARNING: None,
            ApplicationEmojiName.AUDIO_WAVE: None,
            ApplicationEmojiName.RADIO: None,
        }
    )
    client = Mock(spec=discord.Client)
    client.fetch_application_emojis = AsyncMock(
        side_effect=discord.MissingApplicationID()
    )

    result = await catalog.refresh(client)

    assert result.loaded == ()
    assert result.problems == ("fetch_failed:MissingApplicationID",)
    assert catalog.render(ApplicationEmojiName.LOADING) == "⏳"


@pytest.mark.asyncio
async def test_unconfigured_catalog_skips_discord_and_accessor_falls_back() -> None:
    catalog = ApplicationEmojiCatalog(
        {name: None for name in ApplicationEmojiName}
    )
    client = Mock(spec=discord.Client)
    client.fetch_application_emojis = AsyncMock()

    result = await catalog.refresh(client)

    client.fetch_application_emojis.assert_not_awaited()
    assert result.problems == ()
    assert result.fallback == tuple(ApplicationEmojiName)
    assert application_emoji(client, ApplicationEmojiName.RADIO) == "📻"

    client.application_emojis = catalog
    assert application_emoji(client, ApplicationEmojiName.LOADING) == "⏳"
