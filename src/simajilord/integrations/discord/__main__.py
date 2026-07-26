"""Run the Discord transport adapter."""

from __future__ import annotations

import logging

from simajilord.config import load_settings
from simajilord.runtime import SimajilordRuntime

from .bot import SimajilordDiscordBot


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = SimajilordDiscordBot(SimajilordRuntime.build(settings))
    bot.run(settings.token, log_handler=None)


if __name__ == "__main__":
    main()
