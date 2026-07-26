"""Run the Discord transport adapter."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from simajilord.config import Settings, load_settings
from simajilord.runtime import SimajilordRuntime

from .bot import SimajilordDiscordBot


def main() -> None:
    settings = load_settings()
    _configure_logging(settings)
    bot = SimajilordDiscordBot(SimajilordRuntime.build(settings))
    bot.run(settings.token, log_handler=None)


def _configure_logging(settings: Settings) -> None:
    """Keep local operational evidence without allowing unbounded log growth."""

    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = log_dir / "simajilord.log"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    local_file = RotatingFileHandler(
        Path(log_path),
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    local_file.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(settings.log_level)
    root.addHandler(console)
    root.addHandler(local_file)
    log_path.touch(mode=0o600, exist_ok=True)
    log_path.chmod(0o600)


if __name__ == "__main__":
    main()
