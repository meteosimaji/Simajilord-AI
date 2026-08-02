"""Cancellation-safe, bounded keyed asyncio locks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass


async def finish_async_cleanup(awaitable: Awaitable[None]) -> None:
    """Finish mandatory cleanup before preserving an arriving cancellation."""

    cleanup_task = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    cleanup_task.result()
    if cancellation is not None:
        raise cancellation


@dataclass(slots=True)
class _KeyedLockState:
    lock: asyncio.Lock
    users: int = 0


class KeyedAsyncLockPool:
    """Serialize work per key without retaining inactive keys forever."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._states: dict[str, _KeyedLockState] = {}

    @property
    def size(self) -> int:
        """Return the number of active or waiting keys."""

        return len(self._states)

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("keyed lock key must not be empty")
        async with self._guard:
            state = self._states.get(normalized_key)
            if state is None:
                state = _KeyedLockState(asyncio.Lock())
                self._states[normalized_key] = state
            state.users += 1

        acquired = False
        try:
            await state.lock.acquire()
            acquired = True
            yield
        finally:

            async def cleanup() -> None:
                if acquired:
                    state.lock.release()
                async with self._guard:
                    state.users -= 1
                    if state.users < 0:
                        raise RuntimeError("keyed lock reference count became negative")
                    if (
                        state.users == 0
                        and self._states.get(normalized_key) is state
                    ):
                        self._states.pop(normalized_key, None)

            await finish_async_cleanup(cleanup())
