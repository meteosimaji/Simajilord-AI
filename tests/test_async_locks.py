from __future__ import annotations

import asyncio

import pytest

from simajilord.async_locks import KeyedAsyncLockPool


@pytest.mark.asyncio
async def test_keyed_lock_pool_serializes_one_key_and_evicts_it() -> None:
    pool = KeyedAsyncLockPool()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with pool.hold("conversation"):
            order.append("first")
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        await first_entered.wait()
        async with pool.hold("conversation"):
            order.append("second")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await first_entered.wait()
    await asyncio.sleep(0)
    assert pool.size == 1
    release_first.set()
    await asyncio.gather(first_task, second_task)

    assert order == ["first", "second"]
    assert pool.size == 0


@pytest.mark.asyncio
async def test_keyed_lock_pool_evicts_cancelled_waiter_without_unlocking_owner() -> None:
    pool = KeyedAsyncLockPool()
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()

    async def owner() -> None:
        async with pool.hold("delivery"):
            owner_entered.set()
            await release_owner.wait()

    owner_task = asyncio.create_task(owner())
    await owner_entered.wait()
    waiter = asyncio.create_task(_hold_once(pool, "delivery"))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert pool.size == 1

    release_owner.set()
    await owner_task
    assert pool.size == 0


async def _hold_once(pool: KeyedAsyncLockPool, key: str) -> None:
    async with pool.hold(key):
        return
