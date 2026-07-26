"""Validated local configuration."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

from .core.errors import ConfigurationError


class CommandScope(StrEnum):
    """Where application commands are synchronized."""

    GUILD = "guild"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings. Secret fields are excluded from representations."""

    token: str = field(repr=False)
    application_id: int
    command_scope: CommandScope
    command_prefix: str
    log_level: str
    data_dir: Path
    media_cookie_file: Path | None
    tts_provider: str
    tts_voice: str
    max_read_aloud_characters: int
    max_pending_speech: int
    max_concurrent_tts: int
    max_active_voice_guilds: int
    download_timeout_seconds: float


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} is required.")
    return value


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if not 1 <= value <= maximum:
        raise ConfigurationError(f"{name} must be between 1 and {maximum}.")
    return value


def _positive_float(name: str, default: float, *, maximum: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.") from exc
    if not 0 < value <= maximum:
        raise ConfigurationError(f"{name} must be greater than 0 and at most {maximum}.")
    return value


def _optional_private_file(name: str) -> Path | None:
    raw_path = os.getenv(name, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"{name} does not point to an existing file.")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ConfigurationError(
            f"{name} must not be readable or writable by group or other users. "
            f"Run: chmod 600 {path}"
        )
    return path


def load_settings(*, dotenv_path: str | Path = ".env") -> Settings:
    """Load settings without overriding variables already set by the process."""

    load_dotenv(dotenv_path=dotenv_path, override=False)

    raw_application_id = _required("DISCORD_APPLICATION_ID")
    try:
        application_id = int(raw_application_id)
    except ValueError as exc:
        raise ConfigurationError("DISCORD_APPLICATION_ID must be an integer.") from exc
    if application_id <= 0:
        raise ConfigurationError("DISCORD_APPLICATION_ID must be positive.")

    raw_scope = os.getenv("COMMAND_SCOPE", CommandScope.GUILD.value).strip().lower()
    try:
        command_scope = CommandScope(raw_scope)
    except ValueError as exc:
        raise ConfigurationError("COMMAND_SCOPE must be guild or global.") from exc

    command_prefix = os.getenv("COMMAND_PREFIX", "!").strip()
    if not 1 <= len(command_prefix) <= 5 or any(
        character.isspace() for character in command_prefix
    ):
        raise ConfigurationError("COMMAND_PREFIX must contain 1 to 5 non-space characters.")

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ConfigurationError("LOG_LEVEL is invalid.")

    data_dir = Path(os.getenv("DATA_DIR", ".data")).expanduser().resolve()
    data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        data_dir.chmod(0o700)

    tts_provider = os.getenv("TTS_PROVIDER", "macos").strip().lower()
    if tts_provider != "macos":
        raise ConfigurationError("TTS_PROVIDER must be macos in this milestone.")

    tts_voice = os.getenv("TTS_VOICE", "Samantha").strip()
    if not tts_voice:
        raise ConfigurationError("TTS_VOICE cannot be empty.")

    return Settings(
        token=_required("DISCORD_TOKEN"),
        application_id=application_id,
        command_scope=command_scope,
        command_prefix=command_prefix,
        log_level=log_level,
        data_dir=data_dir,
        media_cookie_file=_optional_private_file("MEDIA_COOKIE_FILE"),
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        max_read_aloud_characters=_positive_int(
            "MAX_READ_ALOUD_CHARACTERS", 400, maximum=2_000
        ),
        max_pending_speech=_positive_int("MAX_PENDING_SPEECH", 20, maximum=100),
        max_concurrent_tts=_positive_int("MAX_CONCURRENT_TTS", 2, maximum=10),
        max_active_voice_guilds=_positive_int("MAX_ACTIVE_VOICE_GUILDS", 8, maximum=100),
        download_timeout_seconds=_positive_float(
            "DOWNLOAD_TIMEOUT_SECONDS", 180.0, maximum=900.0
        ),
    )
