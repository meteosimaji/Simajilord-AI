"""Discord presentation and delivery for long-running AI turns."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable

import discord

from simajilord.agent import (
    AGENT_DISCORD_SAFE_MESSAGE_CHARACTERS,
    AGENT_FINAL_DELIVERED_CONTENT,
    AGENT_MESSAGE_BREAK,
    AGENT_NO_ACTION_CONTENT,
    AgentBusyError,
    AgentProgressStage,
    AgentProgressUpdate,
    AgentProviderLimitError,
    AgentRateLimitError,
    AgentTimeoutError,
    AgentUnavailableError,
    is_agent_public_reference_id,
)

from .presenter import EmbedTone, command_embed

log = logging.getLogger(__name__)


def discord_message_chunks(
    content: str,
    *,
    maximum: int = AGENT_DISCORD_SAFE_MESSAGE_CHARACTERS,
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


def agent_delivery_nonce(
    delivery_key: str,
    index: int,
    *,
    purpose: str = "response",
) -> str:
    """Derive a stable Discord nonce for one event-owned host delivery.

    discord.py sends ``enforce_nonce=true`` whenever a nonce is provided. Discord
    can therefore return the already-created message when a retry crosses the
    narrow send/receipt/ACK crash window. Only a digest is exposed and Discord's
    25-character nonce limit is preserved.
    """

    if index < 0:
        raise ValueError("delivery nonce index must be non-negative")
    digest = hashlib.sha256(
        f"{purpose}\0{delivery_key}\0{index}".encode()
    ).hexdigest()
    return f"sla{digest[:22]}"


def agent_error_text(
    error: Exception,
    *,
    reference_id: str | None = None,
) -> str:
    if isinstance(error, AgentBusyError):
        return "The AI request queue is full. Please try again shortly."
    if isinstance(error, AgentRateLimitError):
        if error.retry_after_seconds is not None:
            return (
                "This AI request is rate-limited. Please wait about "
                f"{_english_duration(error.retry_after_seconds)}."
            )
        return "This AI request is rate-limited. Please try again later."
    if isinstance(error, AgentUnavailableError):
        message = "Simajilord AI is currently unavailable on this host."
    elif isinstance(error, AgentProviderLimitError):
        message = (
            "The AI provider usage limit has been reached. "
            "Please try again after the provider limit resets."
        )
    elif isinstance(error, AgentTimeoutError):
        limit = _english_duration(max(1, round(error.timeout_seconds)))
        recovery = (
            " The execution runtime was restarted automatically."
            if error.runtime_restarted
            else ""
        )
        if error.write_attempted:
            retry = (
                " The request was not replayed automatically because an external "
                "write may have started and replaying it could create a duplicate."
            )
        elif error.auto_retry_attempted:
            retry = " A safe automatic retry before any write also became inactive."
        else:
            retry = ""
        message = (
            f"The AI turn produced no observable activity for {limit} and was stopped."
            f"{recovery}{retry}"
            " Any partial response or operation result is unconfirmed."
        )
    else:
        message = "The AI request could not be completed."
    if reference_id is None:
        return message
    if not is_agent_public_reference_id(reference_id):
        log.error("Refusing to display malformed agent reference ID")
        return message
    return (
        f"{message}\n"
        f"Reference ID: `{reference_id}`. Share this ID with the administrator."
    )


def _english_duration(total_seconds: int) -> str:
    seconds = max(1, total_seconds)
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" + ("" if hours == 1 else "s"))
    if minutes:
        parts.append(f"{minutes} minute" + ("" if minutes == 1 else "s"))
    if seconds or not parts:
        parts.append(f"{seconds} second" + ("" if seconds == 1 else "s"))
    return " ".join(parts)


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
    AgentProgressStage.COMPACTING_CONTEXT: "Compacting retained context…",
    AgentProgressStage.ANALYZING_MEDIA: "Analyzing the attachment…",
    AgentProgressStage.GENERATING_IMAGE: "Generating image…",
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
    """Coalesce execution stages into one temporary Discord message."""

    def __init__(
        self,
        source: discord.Message,
        *,
        initial_delay_seconds: float = 1.0,
        minimum_update_seconds: float = 2.5,
        typing_interval_seconds: float = 7.5,
        typing_lease_seconds: float = 12.0,
        on_posted: Callable[[discord.Message], Awaitable[None]] | None = None,
        delivery_key: str | None = None,
    ) -> None:
        self.source = source
        self.initial_delay_seconds = initial_delay_seconds
        self.minimum_update_seconds = minimum_update_seconds
        self.typing_interval_seconds = typing_interval_seconds
        self.typing_lease_seconds = typing_lease_seconds
        self.on_posted = on_posted
        self.delivery_key = delivery_key or f"discord:message:{source.id}"
        self.message: discord.Message | None = None
        self._latest: AgentProgressUpdate | None = None
        self._published: AgentProgressUpdate | None = None
        self._last_update = 0.0
        self._task: asyncio.Task[None] | None = None
        self._typing_task: asyncio.Task[None] | None = None
        self._typing_deadline = 0.0
        self._stage_started_at_epoch: int | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def update(self, update: AgentProgressUpdate) -> None:
        if self._closed:
            return
        if update.stage is not AgentProgressStage.QUEUED:
            self._typing_deadline = time.monotonic() + self.typing_lease_seconds
            if self._typing_task is None or self._typing_task.done():
                self._typing_task = asyncio.create_task(
                    self._typing_while_active(),
                    name=f"simajilord-agent-typing-{self.source.id}",
                )
        self._latest = update
        if update == self._published:
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(
                self._flush_later(),
                name=f"simajilord-agent-progress-{self.source.id}",
            )

    async def finish(self, content: str) -> None:
        messages = await self.prepare(content)
        if not messages:
            return
        first = await self.source.reply(
            messages[0],
            nonce=agent_delivery_nonce(self.delivery_key, 0),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )
        await self._notify_posted(first)
        for index, message_content in enumerate(messages[1:], start=1):
            posted = await self.source.channel.send(
                message_content,
                nonce=agent_delivery_nonce(self.delivery_key, index),
                allowed_mentions=discord.AllowedMentions.none(),
                suppress_embeds=True,
            )
            await self._notify_posted(posted)

    async def prepare(self, content: str) -> tuple[str, ...]:
        """Close temporary UI and return final chunks without posting them."""

        self._closed = True
        await self._cancel_pending()
        await self._delete_progress_message()
        if content.strip() in {
            AGENT_FINAL_DELIVERED_CONTENT,
            AGENT_NO_ACTION_CONTENT,
        }:
            return ()
        return agent_message_groups(content)

    async def fail(self, content: str) -> None:
        self._closed = True
        await self._cancel_pending()
        await self._delete_progress_message()
        await self.source.reply(
            content,
            nonce=agent_delivery_nonce(
                self.delivery_key,
                0,
                purpose="error",
            ),
            mention_author=False,
            allowed_mentions=discord.AllowedMentions.none(),
            suppress_embeds=True,
        )

    async def cancelled(self) -> None:
        """Close active UI while leaving one explicit cancellation result."""

        self._closed = True
        await self._cancel_pending()
        async with self._lock:
            progress_message = self.message
            self.message = None
        embed = command_embed(
            "AI task cancelled",
            description=(
                "The active model turn was interrupted. External effects already "
                "confirmed before cancellation are not undone automatically."
            ),
            tone=EmbedTone.WARNING,
        )
        if progress_message is not None:
            try:
                await progress_message.edit(content=None, embed=embed)
                return
            except discord.DiscordException:
                log.exception(
                    "Failed to edit cancelled AI progress source_message_id=%s",
                    self.source.id,
                )
        await self.source.reply(
            embed=embed,
            nonce=agent_delivery_nonce(
                self.delivery_key,
                0,
                purpose="cancelled",
            ),
            mention_author=False,
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
            if self._published is None or self._published.stage is not update.stage:
                self._stage_started_at_epoch = int(time.time())
            description = agent_progress_text(update)
            if (
                self._stage_started_at_epoch is not None
                and update.stage is not AgentProgressStage.QUEUED
            ):
                description += (
                    f"\n-# Active since <t:{self._stage_started_at_epoch}:R>"
                )
            embed = command_embed(
                "Working",
                description=description,
                tone=EmbedTone.INFO,
            )
            async with self._lock:
                if self._closed:
                    return
                if self.message is None:
                    self.message = await self.source.reply(
                        embed=embed,
                        nonce=agent_delivery_nonce(
                            self.delivery_key,
                            0,
                            purpose="progress",
                        ),
                        mention_author=False,
                        allowed_mentions=discord.AllowedMentions.none(),
                    )
                else:
                    await self.message.edit(
                        content=None,
                        embed=embed,
                    )
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
        tasks = tuple(
            task
            for task in (self._task, self._typing_task)
            if task is not None and not task.done()
        )
        self._task = None
        self._typing_task = None
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _typing_while_active(self) -> None:
        """Pulse typing only while the harness keeps renewing activity."""

        try:
            while not self._closed:
                remaining = self._typing_deadline - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    await self.source.channel.typing()
                except discord.DiscordException:
                    log.exception(
                        "Failed to publish AI typing activity source_message_id=%s",
                        self.source.id,
                    )
                    return
                await asyncio.sleep(min(self.typing_interval_seconds, remaining))
        except asyncio.CancelledError:
            raise
        finally:
            if self._typing_task is asyncio.current_task():
                self._typing_task = None

    async def _delete_progress_message(self) -> None:
        async with self._lock:
            message = self.message
            self.message = None
        if message is not None:
            await self._delete_message(message, kind="temporary")

    async def _delete_message(
        self,
        message: discord.Message,
        *,
        kind: str,
    ) -> None:
        try:
            await message.delete()
        except discord.DiscordException:
            # Final delivery remains a new post even when Discord refuses
            # cleanup, so authored milestone updates keep their ordering.
            log.exception(
                "Failed to delete AI %s message source_message_id=%s "
                "temporary_message_id=%s",
                kind,
                self.source.id,
                message.id,
            )

    async def _notify_posted(self, message: discord.Message) -> None:
        callback = self.on_posted
        if callback is None:
            return
        try:
            await callback(message)
        except Exception:
            # Receipt persistence must never turn a successful Discord send into
            # an apparent delivery failure or cause the model response to repost.
            log.exception(
                "Failed to receipt AI host post source_message_id=%s "
                "posted_message_id=%s",
                self.source.id,
                message.id,
            )
