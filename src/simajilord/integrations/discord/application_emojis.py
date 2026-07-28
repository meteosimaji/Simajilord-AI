"""Validated Discord Application Emoji with deterministic Unicode fallbacks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import discord

if TYPE_CHECKING:
    from simajilord.config import Settings

log = logging.getLogger(__name__)


class ApplicationEmojiName(StrEnum):
    """Stable names provisioned on the Discord application."""

    LOADING = "loading"
    SUCCESS = "success"
    WARNING = "warning"
    AUDIO_WAVE = "audio_wave"
    RADIO = "radio"


_FALLBACKS: dict[ApplicationEmojiName, str] = {
    ApplicationEmojiName.LOADING: "⏳",
    ApplicationEmojiName.SUCCESS: "✅",
    ApplicationEmojiName.WARNING: "⚠️",
    ApplicationEmojiName.AUDIO_WAVE: "〰️",
    ApplicationEmojiName.RADIO: "📻",
}


@dataclass(frozen=True, slots=True)
class ApplicationEmojiValidation:
    """One startup validation result, suitable for logs and status tests."""

    loaded: tuple[ApplicationEmojiName, ...]
    fallback: tuple[ApplicationEmojiName, ...]
    problems: tuple[str, ...]


class ApplicationEmojiCatalog:
    """Resolve configured IDs only after Discord confirms their exact names."""

    def __init__(
        self,
        configured_ids: dict[ApplicationEmojiName, int | None],
    ) -> None:
        self._configured_ids = dict(configured_ids)
        self._rendered = dict(_FALLBACKS)
        self._validation = ApplicationEmojiValidation(
            loaded=(),
            fallback=tuple(ApplicationEmojiName),
            problems=(),
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> ApplicationEmojiCatalog:
        return cls(
            {
                ApplicationEmojiName.LOADING: settings.discord_emoji_loading_id,
                ApplicationEmojiName.SUCCESS: settings.discord_emoji_success_id,
                ApplicationEmojiName.WARNING: settings.discord_emoji_warning_id,
                ApplicationEmojiName.AUDIO_WAVE: (
                    settings.discord_emoji_audio_wave_id
                ),
                ApplicationEmojiName.RADIO: settings.discord_emoji_radio_id,
            }
        )

    @property
    def validation(self) -> ApplicationEmojiValidation:
        return self._validation

    def render(self, name: ApplicationEmojiName) -> str:
        return self._rendered[name]

    async def refresh(
        self,
        client: discord.Client,
    ) -> ApplicationEmojiValidation:
        """Fetch once and reject stale IDs, renamed emoji, and duplicate bindings."""

        configured = {
            name: emoji_id
            for name, emoji_id in self._configured_ids.items()
            if emoji_id is not None
        }
        self._rendered = dict(_FALLBACKS)
        if not configured:
            self._validation = ApplicationEmojiValidation(
                loaded=(),
                fallback=tuple(ApplicationEmojiName),
                problems=(),
            )
            log.info("Application Emoji IDs are unset; using Unicode fallbacks")
            return self._validation

        problems: list[str] = []
        try:
            emojis = await client.fetch_application_emojis()
        except discord.DiscordException as exc:
            problems.append(f"fetch_failed:{type(exc).__name__}")
            self._validation = ApplicationEmojiValidation(
                loaded=(),
                fallback=tuple(ApplicationEmojiName),
                problems=tuple(problems),
            )
            log.warning(
                "Could not validate Discord Application Emoji; using Unicode fallbacks: %s",
                type(exc).__name__,
            )
            return self._validation

        by_id = {emoji.id: emoji for emoji in emojis}
        loaded: list[ApplicationEmojiName] = []
        for name, emoji_id in configured.items():
            emoji = by_id.get(emoji_id)
            if emoji is None:
                problems.append(f"{name.value}:missing")
                continue
            if emoji.name != name.value:
                problems.append(
                    f"{name.value}:name_mismatch:{emoji.name or 'unnamed'}"
                )
                continue
            self._rendered[name] = str(emoji)
            loaded.append(name)

        fallback = tuple(name for name in ApplicationEmojiName if name not in loaded)
        self._validation = ApplicationEmojiValidation(
            loaded=tuple(loaded),
            fallback=fallback,
            problems=tuple(problems),
        )
        if problems:
            log.warning(
                "Discord Application Emoji validation used fallbacks: %s",
                ", ".join(problems),
            )
        log.info(
            "Validated %s/%s Discord Application Emoji",
            len(loaded),
            len(ApplicationEmojiName),
        )
        return self._validation


def application_emoji(
    client: discord.Client,
    name: ApplicationEmojiName,
) -> str:
    """Read the catalog attached by SimajilordDiscordBot, or use Unicode."""

    catalog = getattr(client, "application_emojis", None)
    if isinstance(catalog, ApplicationEmojiCatalog):
        return catalog.render(name)
    return _FALLBACKS[name]


__all__ = [
    "ApplicationEmojiCatalog",
    "ApplicationEmojiName",
    "ApplicationEmojiValidation",
    "application_emoji",
]
