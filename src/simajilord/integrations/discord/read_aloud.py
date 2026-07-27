"""Discord-aware, natural Japanese text preparation for read-aloud."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import discord

from simajilord.services.read_aloud import ReadAloudService

_CUSTOM_EMOJI = re.compile(r"<a?:([^:>]+):\d+>")
_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_IMAGE_SUFFIXES = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
_VIDEO_SUFFIXES = {".m4v", ".mov", ".mp4", ".webm"}
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


@dataclass(frozen=True, slots=True)
class ReadAloudMessageText:
    text: str
    title: str


class ReadAloudMessageFormatter:
    """Resolve Discord markup only when a message is actually spoken."""

    def __init__(
        self,
        service: ReadAloudService,
        *,
        repeat_author_after_seconds: float = 90.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self.repeat_author_after_seconds = repeat_author_after_seconds
        self._clock = clock
        self._last_speakers: dict[tuple[int, int], tuple[int, float]] = {}

    async def format(self, message: discord.Message) -> ReadAloudMessageText | None:
        guild = message.guild
        if guild is None:
            return None
        workspace_id = str(guild.id)
        policy = self.service.policy(workspace_id)
        author_name = _display_name(message.author)
        parts: list[str] = []

        if policy.read_author_names and self._should_read_author(message):
            parts.append(f"{author_name}さん")

        if policy.read_replies:
            reply_author = await _reply_author_name(message)
            if reply_author is not None:
                parts.append(f"{reply_author}さんへの返信")

        content = _resolve_discord_markup(message, message.content.strip())
        if content:
            parts.append(content)

        if policy.read_attachments:
            parts.extend(_attachment_descriptions(message.attachments))
            parts.extend(_sticker_descriptions(message.stickers))

        if not parts:
            return None
        text = self.service.apply_dictionary(workspace_id, "。".join(parts))
        return ReadAloudMessageText(
            text=text,
            title=f"{author_name}さんのメッセージ",
        )

    def forget_workspace(self, workspace_id: str) -> None:
        """Drop only ephemeral speaker cadence when a route is disabled."""

        guild_id = int(workspace_id)
        self._last_speakers = {
            key: value
            for key, value in self._last_speakers.items()
            if key[0] != guild_id
        }

    def _should_read_author(self, message: discord.Message) -> bool:
        guild = message.guild
        if guild is None:
            return False
        key = (guild.id, message.channel.id)
        now = self._clock()
        previous = self._last_speakers.get(key)
        self._last_speakers[key] = (message.author.id, now)
        return (
            previous is None
            or previous[0] != message.author.id
            or now - previous[1] >= self.repeat_author_after_seconds
        )


def _resolve_discord_markup(message: discord.Message, content: str) -> str:
    if not content:
        return ""
    users = {
        str(user.id): _display_name(user)
        for user in message.mentions
    }
    roles = {
        str(role.id): role.name
        for role in message.role_mentions
    }
    channels = {
        str(channel.id): channel.name
        for channel in message.channel_mentions
    }

    value = _USER_MENTION.sub(
        lambda match: f"{users.get(match.group(1), 'ユーザー')}さん",
        content,
    )
    value = _ROLE_MENTION.sub(
        lambda match: f"{roles.get(match.group(1), 'ロール')}へのメンション",
        value,
    )
    value = _CHANNEL_MENTION.sub(
        lambda match: f"{channels.get(match.group(1), 'チャンネル')}チャンネル",
        value,
    )
    value = _CUSTOM_EMOJI.sub(
        lambda match: f"{match.group(1).replace('_', ' ')}の絵文字",
        value,
    )
    value = value.replace("@everyone", "全員へのメンション")
    value = value.replace("@here", "オンラインの皆さんへのメンション")
    return " ".join(value.split())


async def _reply_author_name(message: discord.Message) -> str | None:
    reference = message.reference
    if reference is None or reference.message_id is None:
        return None
    resolved = reference.resolved
    author = getattr(resolved, "author", None)
    if author is not None:
        return _display_name(author)
    fetch_message = getattr(message.channel, "fetch_message", None)
    if fetch_message is None:
        return None
    with suppress(discord.DiscordException):
        referenced = await fetch_message(reference.message_id)
        return _display_name(referenced.author)
    return None


def _attachment_descriptions(
    attachments: list[discord.Attachment],
) -> tuple[str, ...]:
    if not attachments:
        return ()
    groups: dict[str, list[discord.Attachment]] = {
        "画像": [],
        "動画": [],
        "音声": [],
        "ファイル": [],
    }
    for attachment in attachments:
        groups[_attachment_kind(attachment)].append(attachment)

    descriptions: list[str] = []
    for label in ("画像", "動画", "音声", "ファイル"):
        items = groups[label]
        if not items:
            continue
        if len(items) == 1 and label == "ファイル":
            filename = Path(items[0].filename).name[:100]
            descriptions.append(f"ファイル、{filename}を送信しました")
        else:
            descriptions.append(f"{label}を{len(items)}件送信しました")
    return tuple(descriptions)


def _attachment_kind(attachment: discord.Attachment) -> str:
    content_type = (attachment.content_type or "").lower()
    suffix = Path(attachment.filename).suffix.lower()
    if content_type.startswith("image/") or suffix in _IMAGE_SUFFIXES:
        return "画像"
    if content_type.startswith("video/") or suffix in _VIDEO_SUFFIXES:
        return "動画"
    if content_type.startswith("audio/") or suffix in _AUDIO_SUFFIXES:
        return "音声"
    return "ファイル"


def _sticker_descriptions(
    stickers: list[discord.StickerItem],
) -> tuple[str, ...]:
    if not stickers:
        return ()
    if len(stickers) == 1:
        return (f"スタンプ、{stickers[0].name}を送信しました",)
    return (f"スタンプを{len(stickers)}件送信しました",)


def _display_name(value: Any) -> str:
    name = getattr(value, "display_name", None) or getattr(value, "name", None)
    return str(name or "ユーザー").strip() or "ユーザー"
