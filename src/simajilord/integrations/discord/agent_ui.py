"""Discord presentation and delivery for long-running AI turns."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

import discord

from simajilord.agent import (
    AGENT_MESSAGE_BREAK,
    AGENT_NO_ACTION_CONTENT,
    AgentBusyError,
    AgentProgressStage,
    AgentProgressUpdate,
    AgentProviderLimitError,
    AgentRateLimitError,
    AgentUnavailableError,
)

from .presenter import EmbedTone, command_embed

log = logging.getLogger(__name__)


def discord_message_chunks(
    content: str,
    *,
    maximum: int = 1_900,
) -> tuple[str, ...]:
    """Bound Discord output without asking the model to repeat a long answer."""

    text = content.strip()
    if not text:
        return ()
    chunks: list[str] = []
    while text:
        if len(text) <= maximum:
            chunks.append(text)
            break
        boundary = text.rfind("\n", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = text.rfind(" ", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = maximum
        chunks.append(text[:boundary].rstrip())
        text = text[boundary:].lstrip()
    return tuple(chunk for chunk in chunks if chunk)


def agent_message_groups(content: str) -> tuple[str, ...]:
    """Convert every explicit boundary into an independent Discord post."""

    groups = content.split(AGENT_MESSAGE_BREAK)
    messages: list[str] = []
    for group in groups:
        messages.extend(discord_message_chunks(group))
    return tuple(messages)


def agent_error_text(error: Exception) -> str:
    if isinstance(error, AgentBusyError):
        return "AIへの依頼が混み合っています。少し待ってからもう一度お試しください。"
    if isinstance(error, AgentRateLimitError):
        if error.retry_after_seconds is not None:
            return (
                "AIの利用間隔を調整しています。"
                f"あと{retry_after_text(error.retry_after_seconds)}ほどお待ちください。"
            )
        return "AIの利用間隔を調整しています。時間を空けてもう一度お試しください。"
    if isinstance(error, AgentUnavailableError):
        return "現在、このホストではSimajilord AIを利用できません。"
    if isinstance(error, AgentProviderLimitError):
        return (
            "AIプロバイダーの利用上限に達しています。"
            "上限がリセットされてから、もう一度お試しください。"
        )
    return "AIの処理を完了できませんでした。"


def retry_after_text(total_seconds: int) -> str:
    seconds = max(1, total_seconds)
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}時間")
    if minutes:
        parts.append(f"{minutes}分")
    if seconds or not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


_PROGRESS_MESSAGES = {
    AgentProgressStage.QUEUED: "Waiting for an earlier AI request in this server…",
    AgentProgressStage.STARTING: "Checking your request…",
    AgentProgressStage.READING_DISCORD: "Reading the relevant Discord conversation…",
    AgentProgressStage.SEARCHING_WEB: "Searching the web…",
    AgentProgressStage.COMPUTING: "Running the calculation…",
    AgentProgressStage.ANALYZING_MEDIA: "Analyzing the attachment with HIVE…",
    AgentProgressStage.GENERATING_IMAGE: "Preparing local image generation…",
    AgentProgressStage.USING_AUDIO: "Preparing the server audio controls…",
    AgentProgressStage.PREPARING_RESPONSE: "Preparing the response…",
}


def agent_progress_text(update: AgentProgressUpdate) -> str:
    if (
        update.stage is AgentProgressStage.QUEUED
        and update.queue_position is not None
    ):
        return (
            "Waiting for earlier AI requests in this server…\n"
            f"Requests ahead of you: **{update.queue_position}**"
        )
    return _PROGRESS_MESSAGES[update.stage]


class AgentProgressMessage:
    """Coalesce real execution stages into one low-frequency Discord message."""

    def __init__(
        self,
        source: discord.Message,
        *,
        initial_delay_seconds: float = 1.0,
        minimum_update_seconds: float = 2.5,
    ) -> None:
        self.source = source
        self.initial_delay_seconds = initial_delay_seconds
        self.minimum_update_seconds = minimum_update_seconds
        self.message: discord.Message | None = None
        self._latest: AgentProgressUpdate | None = None
        self._published: AgentProgressUpdate | None = None
        self._last_update = 0.0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def update(self, update: AgentProgressUpdate) -> None:
        if self._closed:
            return
        self._latest = update
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._flush_later(),
                name=f"simajilord-agent-progress-{self.source.id}",
            )

    async def finish(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        if content.strip() == AGENT_NO_ACTION_CONTENT:
            async with self._lock:
                if self.message is not None:
                    with suppress(discord.DiscordException):
                        await self.message.delete()
                    self.message = None
            return
        messages = agent_message_groups(content)
        if not messages:
            return
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    messages[0],
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=messages[0],
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
        for message in messages[1:]:
            await self.source.channel.send(
                message,
                allowed_mentions=discord.AllowedMentions.none(),
            )

    async def fail(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        async with self._lock:
            if self.message is None:
                self.message = await self.source.reply(
                    content,
                    mention_author=False,
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            else:
                await self.message.edit(
                    content=content,
                    embed=None,
                    allowed_mentions=discord.AllowedMentions.none(),
                )

    async def _flush_later(self) -> None:
        try:
            if self.message is None:
                delay = self.initial_delay_seconds
            else:
                elapsed = time.monotonic() - self._last_update
                delay = max(0.0, self.minimum_update_seconds - elapsed)
            if delay:
                await asyncio.sleep(delay)
            if self._closed or self._latest is None or self._latest == self._published:
                return
            update = self._latest
            embed = command_embed(
                "Working",
                description=agent_progress_text(update),
                tone=EmbedTone.INFO,
            )
            async with self._lock:
                if self._closed:
                    return
                if self.message is None:
                    self.message = await self.source.reply(
                        embed=embed,
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await self.message.edit(content=None, embed=embed)
                self._published = update
                self._last_update = time.monotonic()
        except asyncio.CancelledError:
            raise
        except discord.DiscordException:
            # The Cog-level logger handles the enclosing turn if final delivery
            # also fails; a transient progress failure must not abort the AI.
            log.exception(
                "Failed to publish AI progress message source_message_id=%s",
                self.source.id,
            )
            return
        finally:
            self._task = None
            if (
                not self._closed
                and self._latest is not None
                and self._latest != self._published
            ):
                self._task = asyncio.create_task(
                    self._flush_later(),
                    name=f"simajilord-agent-progress-{self.source.id}",
                )

    async def _cancel_pending(self) -> None:
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
