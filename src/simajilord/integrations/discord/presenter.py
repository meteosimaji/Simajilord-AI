"""Consistent Discord presentation for explicit human commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    from .capabilities import DiscordExpandMessageResponse


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


def expanded_message_embeds(
    response: DiscordExpandMessageResponse,
) -> tuple[discord.Embed, ...]:
    """Render a bounded quote without allowing the source message to mention users."""

    content = response.content.strip()
    original_embed = response.embeds[0] if response.embeds else None
    if not content and original_embed is not None:
        content = "\n".join(
            value
            for value in (original_embed.title, original_embed.description)
            if value
        )
    if not content and not (
        response.attachments
        or response.embeds
        or response.sticker_names
        or response.poll
    ):
        content = "表示できる本文や添付はありません。"
    description = _bounded_quote_text(content, maximum=3_200)
    timestamp = discord.utils.parse_time(response.created_at_iso)
    embed = discord.Embed(
        title="元のメッセージへ",
        description=description or None,
        url=response.jump_url,
        colour=discord.Colour.blurple(),
        timestamp=timestamp,
    )
    embed.set_author(
        name=response.author_name,
        icon_url=response.author_avatar_url,
        url=response.jump_url,
    )
    if response.reply_author_name is not None:
        reply_text = response.reply_content_preview or "本文なし"
        embed.add_field(
            name=f"{response.reply_author_name}さんへの返信",
            value=_bounded_quote_text(reply_text, maximum=600),
            inline=False,
        )
    non_image_attachments = tuple(
        attachment
        for attachment in response.attachments
        if not _expanded_attachment_is_image(attachment.content_type)
        or attachment.spoiler
    )
    if non_image_attachments:
        lines = tuple(
            (
                f"[{discord.utils.escape_markdown(item.filename)}]({item.url})"
                f" · {_format_file_size(item.size_bytes)}"
            )
            for item in non_image_attachments[:8]
        )
        hidden_count = len(non_image_attachments) - len(lines)
        value = "\n".join(lines)
        if hidden_count:
            value += f"\nほか{hidden_count}件"
        embed.add_field(
            name="添付ファイル",
            value=_bounded_quote_text(value, maximum=900),
            inline=False,
        )
    if response.poll is not None:
        answers = "\n".join(
            f"{index}. {discord.utils.escape_markdown(answer)}"
            for index, answer in enumerate(response.poll.answers[:10], start=1)
        )
        embed.add_field(
            name=_bounded_quote_text(response.poll.question, maximum=200),
            value=_bounded_quote_text(answers, maximum=800) or "選択肢なし",
            inline=False,
        )
    if response.sticker_names:
        embed.add_field(
            name="スタンプ",
            value=", ".join(
                discord.utils.escape_markdown(name)
                for name in response.sticker_names
            )[:900],
            inline=False,
        )
    if original_embed is not None and response.content.strip():
        rich_text = "\n".join(
            value
            for value in (original_embed.title, original_embed.description)
            if value
        )
        if rich_text:
            embed.add_field(
                name="埋め込み",
                value=_bounded_quote_text(rich_text, maximum=700),
                inline=False,
            )
    edited = " · 編集済み" if response.edited_at_iso is not None else ""
    embed.set_footer(text=f"#{response.channel_name}{edited}")

    image_urls: list[str] = [
        attachment.proxy_url or attachment.url
        for attachment in response.attachments
        if _expanded_attachment_is_image(attachment.content_type)
        and not attachment.spoiler
    ]
    for item in response.embeds:
        for candidate in (item.image_url, item.thumbnail_url):
            if candidate and candidate not in image_urls:
                image_urls.append(candidate)
    if image_urls:
        embed.set_image(url=image_urls[0])
    extra_embeds: list[discord.Embed] = []
    for image_url in image_urls[1:4]:
        image_embed = discord.Embed(url=response.jump_url)
        image_embed.set_image(url=image_url)
        extra_embeds.append(image_embed)
    return (embed, *extra_embeds)


def expanded_message_view(jump_url: str) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label="Jump",
            emoji="↗️",
            style=discord.ButtonStyle.link,
            url=jump_url,
        )
    )
    return view


def _bounded_quote_text(value: str, *, maximum: int) -> str:
    normalized = value.strip()
    if len(normalized) <= maximum:
        return normalized
    suffix = "\n\n…続きは元のメッセージで確認できます。"
    return normalized[: maximum - len(suffix)].rstrip() + suffix


def _expanded_attachment_is_image(content_type: str | None) -> bool:
    return content_type is not None and content_type.lower().startswith("image/")


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1_000:
        return f"{size_bytes} B"
    if size_bytes < 1_000_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes / 1_000_000:.1f} MB"
