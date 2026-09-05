from __future__ import annotations

import asyncio
import signal
import sys
from unittest.mock import AsyncMock

import pytest

from simajilord.integrations.discord.__main__ import _run_bot


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX service-manager signal")
async def test_service_termination_closes_bot_and_removes_signal_handler(monkeypatch):
    loop = asyncio.get_running_loop()
    handlers = {}
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb: handlers.update({sig: cb}))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: handlers.pop(sig))
    entered = asyncio.Event()
    bot = AsyncMock()

    async def start(token):
        assert token == "test-token"
        entered.set()
        await asyncio.Event().wait()

    bot.start.side_effect = start
    task = asyncio.create_task(_run_bot(bot, "test-token"))
    await entered.wait()
    handlers[signal.SIGTERM]()
    await asyncio.wait_for(task, 1)
    bot.__aexit__.assert_awaited_once()
    assert handlers == {}
