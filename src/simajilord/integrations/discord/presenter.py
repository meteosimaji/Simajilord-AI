"""Consistent Discord presentation for explicit human commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import discord


class EmbedTone(StrEnum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class EmbedField:
    name: str
    value: str
    inline: bool = True


_COLOURS = {
    EmbedTone.INFO: discord.Colour.blurple(),
    EmbedTone.SUCCESS: discord.Colour.green(),
    EmbedTone.WARNING: discord.Colour.orange(),
    EmbedTone.ERROR: discord.Colour.red(),
}


def command_embed(
    title: str,
    *,
    description: str | None = None,
    fields: tuple[EmbedField, ...] = (),
    tone: EmbedTone = EmbedTone.INFO,
) -> discord.Embed:
    """Build a compact branded embed for deterministic command feedback."""

    embed = discord.Embed(
        title=title,
        description=description,
        colour=_COLOURS[tone],
        timestamp=datetime.now(UTC),
    )
    for field in fields:
        embed.add_field(name=field.name, value=field.value, inline=field.inline)
    return embed
