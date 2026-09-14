"""Reconcile durable channel references only after Discord confirms deletion."""

from __future__ import annotations

import logging

import discord

from simajilord.services.read_aloud import ReadAloudService

log = logging.getLogger(__name__)


async def reconcile_read_aloud_channels(
    guild: discord.Guild, service: ReadAloudService
) -> dict[int, discord.abc.GuildChannel | discord.Thread]:
    """Repair missed deletion events; a cache miss or Forbidden is not deletion.

    Return fetched channels for callers resolving uncached, archived threads.
    """

    resolved: dict[int, discord.abc.GuildChannel | discord.Thread] = {}
    workspace_id = str(guild.id)
    for channel_id in sorted(service.referenced_channel_ids(workspace_id)):
        numeric_id = int(channel_id)
        if guild.get_channel_or_thread(numeric_id) is not None:
            continue
        try:
            channel = await guild.fetch_channel(numeric_id)
        except discord.NotFound as exc:
            if exc.code != 10003:
                continue
            if await service.forget_channel(workspace_id, channel_id):
                log.info(
                    "Removed deleted channel from read-aloud routes guild=%s channel=%s",
                    workspace_id, channel_id,
                )
        except discord.HTTPException:
            log.warning(
                "Could not verify saved read-aloud channel; retaining configuration "
                "guild=%s channel=%s", workspace_id, channel_id,
            )
        else:
            resolved[numeric_id] = channel
    return resolved
