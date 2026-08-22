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
from simajilord.services.speech import (
    SpeechSegment,
    SpeechSegmentKind,
    speech_prefix,
)

_CUSTOM_EMOJI = re.compile(r"<a?:([^:>]+):\d+>")
_CUSTOM_EMOJI_RUN = re.compile(
    r"(?P<emoji><a?:(?P<name>[^:>]+):\d+>)(?:\s*(?P=emoji)){1,}"
)
_USER_MENTION = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION = re.compile(r"<#(\d+)>")
_FENCED_CODE = re.compile(r"```(?:[^\n`]*)\n?.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)]\(https?://[^\s)]+\)")
_PLAIN_URL = re.compile(r"https?://[^\s<>]+")
_SPOILER = re.compile(r"\|\|(.+?)\|\|", re.DOTALL)
_IMAGE_SUFFIXES = {".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
_VIDEO_SUFFIXES = {".m4v", ".mov", ".mp4", ".webm"}
_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_OMISSION_TEXT = "以下略"


@dataclass(frozen=True, slots=True)
class ReadAloudMessageText:
    segments: tuple[SpeechSegment, ...]
    title: str

    @property
    def text(self) -> str:
        return "。".join(segment.text for segment in self.segments)


def merge_read_aloud_messages(
    messages: tuple[tuple[str, ReadAloudMessageText], ...],
) -> ReadAloudMessageText:
    """Merge one short channel burst while compacting exact consecutive spam."""

    if not messages:
        raise ValueError("At least one read-aloud message is required.")
    if len(messages) == 1:
        return messages[0][1]

    segments: list[SpeechSegment] = []
    index = 0
    while index < len(messages):
        author_id, prepared = messages[index]
        semantic = tuple(
            (segment.kind, segment.text)
            for segment in prepared.segments
            if segment.kind is not SpeechSegmentKind.AUTHOR
        )
        repeat_count = 1
        while index + repeat_count < len(messages):
            next_author, next_prepared = messages[index + repeat_count]
            next_semantic = tuple(
                (segment.kind, segment.text)
                for segment in next_prepared.segments
                if segment.kind is not SpeechSegmentKind.AUTHOR
            )
            if next_author != author_id or next_semantic != semantic:
                break
            repeat_count += 1
        segments.extend(prepared.segments)
        if repeat_count > 1:
            segments.append(
                SpeechSegment(
                    SpeechSegmentKind.EVENT,
                    f"同じ内容を{repeat_count}回送信しました",
                )
            )
        index += repeat_count

    return ReadAloudMessageText(
        segments=tuple(segments),
        title=f"{len(messages)}件のメッセージ",
    )


def abbreviate_read_aloud_segments(
    segments: tuple[SpeechSegment, ...],
    *,
    maximum: int,
) -> tuple[SpeechSegment, ...]:
    """Bound message semantics while retaining the speaker and an audible marker."""

    if maximum < 1:
        raise ValueError("Read-aloud abbreviation limit must be positive.")
    semantic_characters = sum(
        len(segment.text)
        for segment in segments
        if segment.kind is not SpeechSegmentKind.AUTHOR
    )
    if semantic_characters <= maximum:
        return segments

    remaining_characters = maximum
    abbreviated: list[SpeechSegment] = []
    for segment in segments:
        if segment.kind is SpeechSegmentKind.AUTHOR:
            abbreviated.append(segment)
            continue
        if remaining_characters == 0:
            break
        if len(segment.text) <= remaining_characters:
            abbreviated.append(segment)
            remaining_characters -= len(segment.text)
            continue
        prefix, _remainder = speech_prefix(segment.text, remaining_characters)
        abbreviated.append(SpeechSegment(segment.kind, prefix))
        remaining_characters = 0
        break
    abbreviated.append(
        SpeechSegment(
            SpeechSegmentKind.EVENT,
            _OMISSION_TEXT,
            cache_key="read-aloud:omission",
        )
    )
    return tuple(abbreviated)


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
        segments: list[SpeechSegment] = []

        if policy.read_author_names and self._should_read_author(message):
            segments.append(
                SpeechSegment(
                    SpeechSegmentKind.AUTHOR,
                    f"{author_name}さん",
                    cache_key=f"author:{message.author.id}:{author_name}",
                )
            )

        if policy.read_replies:
            reply_author = await _reply_author_name(message)
            if reply_author is not None:
                segments.append(
                    SpeechSegment(
                        SpeechSegmentKind.BODY,
                        f"{reply_author}さんへの返信",
                    )
                )

        content = _resolve_discord_markup(message, message.content.strip())
        if content:
            segments.append(SpeechSegment(SpeechSegmentKind.BODY, content))

        if policy.read_attachments:
            segments.extend(
                SpeechSegment(SpeechSegmentKind.ATTACHMENT, description)
                for description in (
                    *_attachment_descriptions(message.attachments),
                    *_sticker_descriptions(message.stickers),
                )
            )

        if not segments:
            return None
        spoken_segments = tuple(
            SpeechSegment(
                segment.kind,
                self.service.apply_dictionary(workspace_id, segment.text),
                segment.cache_key,
            )
            for segment in segments
        )
        if policy.abbreviate_long_messages:
            spoken_segments = abbreviate_read_aloud_segments(
                spoken_segments,
                maximum=policy.message_character_limit,
            )
        return ReadAloudMessageText(
            segments=spoken_segments,
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

    value = _FENCED_CODE.sub("コードブロックを送信しました", content)
    value = _INLINE_CODE.sub(
        lambda match: f"コード、{' '.join(match.group(1).split())}",
        value,
    )
    value = _MARKDOWN_LINK.sub(
        lambda match: f"{match.group(1)}のリンク",
        value,
    )
    value = _PLAIN_URL.sub("リンク", value)
    value = _SPOILER.sub(
        lambda match: f"スポイラー、{match.group(1)}",
        value,
    )
    value = _USER_MENTION.sub(
        lambda match: f"{users.get(match.group(1), 'ユーザー')}さん",
        value,
    )
    value = _ROLE_MENTION.sub(
        lambda match: f"{roles.get(match.group(1), 'ロール')}へのメンション",
        value,
    )
    value = _CHANNEL_MENTION.sub(
        lambda match: f"{channels.get(match.group(1), 'チャンネル')}チャンネル",
        value,
    )
    value = _CUSTOM_EMOJI_RUN.sub(
        lambda match: (
            f"{match.group('name').replace('_', ' ')}の絵文字を"
            f"{len(_CUSTOM_EMOJI.findall(match.group(0)))}個"
        ),
        value,
    )
    value = _CUSTOM_EMOJI.sub(
        lambda match: f"{match.group(1).replace('_', ' ')}の絵文字",
        value,
    )
    value = value.replace("@everyone", "全員へのメンション")
    value = value.replace("@here", "オンラインの皆さんへのメンション")
    lines: list[str] = []
    for raw_line in value.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            continue
        quoted = line.startswith(">")
        if quoted:
            line = line.lstrip(">").strip()
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-+*]\s+", "", line)
        line = line.replace("**", "").replace("__", "").replace("~~", "")
        if line:
            lines.append(f"引用、{line}" if quoted else line)
    return "\n".join(lines)


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
