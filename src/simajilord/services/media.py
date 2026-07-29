"""Media use cases independent of chat transports."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from time import monotonic
from typing import Protocol, TypeVar, cast

from simajilord.domain.audio import AudioItem
from simajilord.domain.media import (
    DownloadArtifact,
    DownloadBatch,
    DownloadFormat,
    MediaCandidate,
)

from .metrics import ServiceMetricHook, ServiceOperationMetric

log = logging.getLogger(__name__)

_T = TypeVar("_T")


class MediaPriority(IntEnum):
    """Scheduling priority without coupling providers to a chat transport."""

    INTERACTIVE = 0
    NORMAL = 1
    BACKGROUND = 2


class MediaProvider(Protocol):
    async def resolve_audio(self, reference: str) -> AudioItem: ...

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]: ...

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]: ...

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
    ) -> DownloadArtifact: ...

    async def download_many(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
        max_items: int,
    ) -> DownloadBatch: ...


@dataclass(slots=True)
class _MediaWork:
    workspace_id: str
    priority: MediaPriority
    operation_name: str
    operation: Callable[[], Awaitable[object]]
    future: asyncio.Future[object]
    enqueued_at: float


class FairMediaScheduler:
    """Globally bounded, priority-aware round robin across workspaces."""

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_per_workspace: int,
        metric_hook: ServiceMetricHook | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not 1 <= max_per_workspace <= max_concurrent:
            raise ValueError(
                "max_per_workspace must be between 1 and max_concurrent"
            )
        self.max_concurrent = max_concurrent
        self.max_per_workspace = max_per_workspace
        self._metric_hook = metric_hook
        self._queues: dict[
            MediaPriority,
            dict[str, deque[_MediaWork]],
        ] = {
            priority: defaultdict(deque) for priority in MediaPriority
        }
        self._rings: dict[MediaPriority, deque[str]] = {
            priority: deque() for priority in MediaPriority
        }
        self._ring_members: set[tuple[MediaPriority, str]] = set()
        self._active_by_workspace: dict[str, int] = defaultdict(int)
        self._workers: list[asyncio.Task[None]] = []
        self._condition = asyncio.Condition()
        self._closed = False

    async def run(
        self,
        *,
        workspace_id: str,
        priority: MediaPriority,
        operation_name: str,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Queue one provider call and return its typed result."""

        if not workspace_id:
            raise ValueError("workspace_id cannot be empty")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[object] = loop.create_future()

        async def boxed_operation() -> object:
            return await operation()

        work = _MediaWork(
            workspace_id=workspace_id,
            priority=priority,
            operation_name=operation_name,
            operation=boxed_operation,
            future=future,
            enqueued_at=monotonic(),
        )
        async with self._condition:
            if self._closed:
                raise RuntimeError("Media scheduler is closed.")
            self._ensure_workers()
            self._queues[priority][workspace_id].append(work)
            self._offer_workspace(priority, workspace_id)
            self._condition.notify_all()
        return cast(_T, await asyncio.shield(future))

    async def close(self) -> None:
        """Cancel workers and reject work that has not started."""

        async with self._condition:
            if self._closed:
                return
            self._closed = True
            for priority_queues in self._queues.values():
                for queue in priority_queues.values():
                    while queue:
                        work = queue.popleft()
                        if not work.future.done():
                            work.future.cancel()
            workers = tuple(self._workers)
            self._condition.notify_all()
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()

    def _ensure_workers(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(
                self._worker(),
                name=f"simajilord-media-{index + 1}",
            )
            for index in range(self.max_concurrent)
        ]

    def _offer_workspace(
        self,
        priority: MediaPriority,
        workspace_id: str,
    ) -> None:
        key = (priority, workspace_id)
        if (
            self._active_by_workspace[workspace_id] >= self.max_per_workspace
            or not self._queues[priority][workspace_id]
            or key in self._ring_members
        ):
            return
        self._rings[priority].append(workspace_id)
        self._ring_members.add(key)

    def _has_ready_work(self) -> bool:
        return any(
            self._active_by_workspace[workspace_id] < self.max_per_workspace
            for priority in MediaPriority
            for workspace_id in self._rings[priority]
        )

    def _take_next_work(self) -> _MediaWork:
        for priority in MediaPriority:
            ring = self._rings[priority]
            for _ in range(len(ring)):
                workspace_id = ring.popleft()
                if self._active_by_workspace[workspace_id] >= self.max_per_workspace:
                    ring.append(workspace_id)
                    continue
                self._ring_members.discard((priority, workspace_id))
                queue = self._queues[priority][workspace_id]
                work = queue.popleft()
                self._active_by_workspace[workspace_id] += 1
                self._offer_workspace(priority, workspace_id)
                return work
        raise RuntimeError("Media scheduler was woken without ready work.")

    async def _worker(self) -> None:
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._closed or self._has_ready_work()
                )
                if self._closed:
                    return
                work = self._take_next_work()
            started_at = monotonic()
            outcome = "succeeded"
            try:
                result = await work.operation()
            except asyncio.CancelledError:
                outcome = "cancelled"
                if not work.future.done():
                    work.future.cancel()
                raise
            except Exception as exc:
                outcome = "failed"
                if not work.future.done():
                    work.future.set_exception(exc)
            else:
                if not work.future.done():
                    work.future.set_result(result)
            finally:
                finished_at = monotonic()
                await self._record_metric(
                    ServiceOperationMetric(
                        operation=f"media.{work.operation_name}",
                        workspace_id=work.workspace_id,
                        wait_ms=max(0.0, (started_at - work.enqueued_at) * 1_000),
                        duration_ms=max(0.0, (finished_at - started_at) * 1_000),
                        outcome=outcome,
                    )
                )
                async with self._condition:
                    active = self._active_by_workspace[work.workspace_id] - 1
                    if active:
                        self._active_by_workspace[work.workspace_id] = active
                    else:
                        self._active_by_workspace.pop(work.workspace_id, None)
                    for priority in MediaPriority:
                        self._offer_workspace(priority, work.workspace_id)
                    self._condition.notify_all()

    async def _record_metric(self, metric: ServiceOperationMetric) -> None:
        if self._metric_hook is None:
            return
        try:
            await self._metric_hook(metric)
        except Exception:
            log.exception(
                "Could not record media metric operation=%s workspace=%s",
                metric.operation,
                metric.workspace_id,
            )


class MediaService:
    def __init__(
        self,
        provider: MediaProvider,
        *,
        max_concurrent: int | None = None,
        max_per_workspace: int = 1,
        metric_hook: ServiceMetricHook | None = None,
    ) -> None:
        self.provider = provider
        self._scheduler = (
            None
            if max_concurrent is None
            else FairMediaScheduler(
                max_concurrent=max_concurrent,
                max_per_workspace=max_per_workspace,
                metric_hook=metric_hook,
            )
        )

    async def resolve_audio(
        self,
        reference: str,
        *,
        workspace_id: str = "system",
        priority: MediaPriority = MediaPriority.INTERACTIVE,
    ) -> AudioItem:
        return await self._run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name="resolve",
            operation=lambda: self.provider.resolve_audio(reference),
        )

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
        workspace_id: str = "system",
        priority: MediaPriority = MediaPriority.INTERACTIVE,
    ) -> tuple[MediaCandidate, ...]:
        return await self._run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name="search",
            operation=lambda: self.provider.search_audio(query, limit=limit),
        )

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
        workspace_id: str = "system",
        priority: MediaPriority = MediaPriority.BACKGROUND,
    ) -> tuple[MediaCandidate, ...]:
        return await self._run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name="mix",
            operation=lambda: self.provider.mix_audio(seed_references, limit=limit),
        )

    async def download(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
        workspace_id: str = "system",
        priority: MediaPriority = MediaPriority.NORMAL,
    ) -> DownloadArtifact:
        return await self._run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name="download",
            operation=lambda: self.provider.download(
                url,
                media_type,
                destination,
                max_bytes=max_bytes,
            ),
        )

    async def download_many(
        self,
        url: str,
        media_type: DownloadFormat,
        destination: Path,
        *,
        max_bytes: int,
        max_items: int,
        workspace_id: str = "system",
        priority: MediaPriority = MediaPriority.NORMAL,
    ) -> DownloadBatch:
        if not 1 <= max_items <= 10:
            raise ValueError("max_items must be between 1 and 10")
        return await self._run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name="download",
            operation=lambda: self.provider.download_many(
                url,
                media_type,
                destination,
                max_bytes=max_bytes,
                max_items=max_items,
            ),
        )

    async def close(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.close()

    async def _run(
        self,
        *,
        workspace_id: str,
        priority: MediaPriority,
        operation_name: str,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        if self._scheduler is None:
            return await operation()
        return await self._scheduler.run(
            workspace_id=workspace_id,
            priority=priority,
            operation_name=operation_name,
            operation=operation,
        )
