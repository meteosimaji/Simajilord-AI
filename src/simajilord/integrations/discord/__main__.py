"""Run the Discord transport adapter."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path

from simajilord.config import Settings, load_settings, security_policy_warnings
from simajilord.runtime import SimajilordRuntime

from .bot import SimajilordDiscordBot

log = logging.getLogger(__name__)


def main() -> None:
    settings = load_settings()
    _configure_logging(settings)
    for warning in security_policy_warnings(settings):
        log.critical("SECURITY WARNING: %s", warning)
    bot = SimajilordDiscordBot(SimajilordRuntime.build(settings))
    with suppress(KeyboardInterrupt):
        asyncio.run(_run_bot(bot, settings.token))


async def _run_bot(bot: SimajilordDiscordBot, token: str) -> None:
    """Let service-manager termination reach the same graceful close as Ctrl-C."""

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        async with bot:
            await bot.start(token)
    except asyncio.CancelledError:
        pass
    finally:
        if sys.platform != "win32":
            loop.remove_signal_handler(signal.SIGTERM)


def _configure_logging(settings: Settings) -> None:
    """Keep local operational evidence without allowing unbounded log growth."""

    log_dir = settings.data_dir / "logs"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = log_dir / "simajilord.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
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
