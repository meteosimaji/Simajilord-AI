"""Discord attachment adapter for the transport-neutral local media store."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path

import discord

from simajilord.core.errors import UserError
from simajilord.runtime import SimajilordRuntime
from simajilord.services.local_media import LocalMediaRecord

from .attachment_io import read_attachment_bytes

PLAYABLE_ATTACHMENT_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def attachment_can_play(attachment: discord.Attachment) -> bool:
    """Use MIME metadata as a hint; ffprobe remains the authority."""

    content_type = (attachment.content_type or "").lower()
    return content_type.startswith(("audio/", "video/")) or (
        Path(attachment.filename).suffix.lower() in PLAYABLE_ATTACHMENT_SUFFIXES
    )


async def import_discord_attachment(
    runtime: SimajilordRuntime,
    attachment: discord.Attachment,
    *,
    source_message: discord.Message | None,
    uploader: discord.abc.User,
) -> LocalMediaRecord:
    """Download once, then hand the untrusted payload to local validation."""

    if attachment.size <= 0:
        raise UserError("local_media.empty")
    if attachment.size > runtime.settings.local_media_max_file_bytes:
        raise UserError(
            "local_media.too_large",
            maximum=runtime.settings.local_media_max_file_bytes,
        )
    import_root = runtime.settings.data_dir / "local_media" / "incoming"
    import_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = import_root / f"{secrets.token_hex(16)}.part"
    try:
        content = await read_attachment_bytes(attachment)
        if not content:
            raise UserError("local_media.empty")
        if len(content) > runtime.settings.local_media_max_file_bytes:
            raise UserError(
                "local_media.too_large",
                maximum=runtime.settings.local_media_max_file_bytes,
            )
        await asyncio.to_thread(temporary.write_bytes, content)
        return await runtime.local_media.import_file(
            temporary,
            original_filename=attachment.filename,
            content_type=attachment.content_type,
            source_jump_url=(
                source_message.jump_url if source_message is not None else None
            ),
            uploaded_by_id=str(uploader.id),
            uploaded_by_name=uploader.display_name,
        )
    except discord.DiscordException as exc:
        raise UserError("discord.attachment_unavailable") from exc
    finally:
        temporary.unlink(missing_ok=True)
