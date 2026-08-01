"""Persistent local image-generation queue and structured caption compiler."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import sqlite3
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from simajilord.core.errors import ProviderError, UserError
from simajilord.domain.image import (
    ImageAspectRatio,
    ImageGenerationJob,
    ImageGenerationModel,
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.observability import EventJournal
from simajilord.providers.image import ImageGenerationProvider, ImageProgressCallback

log = logging.getLogger(__name__)
ImageDeliveryHandler = Callable[[ImageGenerationJob], Awaitable[None]]

_DIMENSIONS = {
    ImageAspectRatio.SQUARE: (512, 512),
    ImageAspectRatio.LANDSCAPE: (768, 512),
    ImageAspectRatio.PORTRAIT: (512, 768),
}
_DISALLOWED_PROMPT = re.compile(
    r"(?:child|minor|underage|児童|未成年).{0,80}"
    r"(?:nude|naked|sexual|explicit|ヌード|性的|裸)|"
    r"(?:nude|naked|sexual|explicit|ヌード|性的|裸).{0,80}"
    r"(?:child|minor|underage|児童|未成年)|"
    r"(?:non[- ]?consensual intimate|盗撮|児童ポルノ|csam)",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class _DeliveryLockState:
    lock: asyncio.Lock
    users: int = 0


class ImageGenerationStore:
    """SQLite authority for queued work and restart-safe Discord delivery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def insert(self, job: ImageGenerationJob) -> None:
        with self._connect() as connection:
            self._insert(connection, job)

    def admit(
        self,
        job: ImageGenerationJob,
        *,
        enforce_rate_limits: bool,
        actor_since: datetime,
        actor_limit: int,
        workspace_since: datetime,
        workspace_limit: int,
        pending_limit: int,
    ) -> str | None:
        """Atomically validate every queue limit and reserve the accepted job."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM image_generation_jobs WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return "image.idempotent_replay"
            if enforce_rate_limits:
                actor_count = self._count_recent(
                    connection,
                    actor_id=job.actor_id,
                    workspace_id=None,
                    since=actor_since,
                )
                if actor_count >= actor_limit:
                    connection.rollback()
                    return "image.user_limit_reached"
                workspace_count = self._count_recent(
                    connection,
                    actor_id=None,
                    workspace_id=job.workspace_id,
                    since=workspace_since,
                )
                if workspace_count >= workspace_limit:
                    connection.rollback()
                    return "image.workspace_limit_reached"
            pending = connection.execute(
                """
                SELECT COUNT(*) AS count FROM image_generation_jobs
                WHERE status IN (?, ?)
                """,
                (ImageJobStatus.QUEUED.value, ImageJobStatus.RUNNING.value),
            ).fetchone()
            if int(pending["count"]) >= pending_limit:
                connection.rollback()
                return "image.queue_full"
            self._insert(connection, job)
            connection.commit()
        return None

    def requeue_running(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE image_generation_jobs
                SET status = ?, progress_step = 0, error_code = NULL
                WHERE status = ?
                """,
                (ImageJobStatus.QUEUED.value, ImageJobStatus.RUNNING.value),
            )
            return cursor.rowcount

    def claim_next(self) -> ImageGenerationJob | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT * FROM image_generation_jobs
                WHERE status = ?
                ORDER BY created_at, job_id
                LIMIT 1
                """,
                (ImageJobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET status = ?, progress_step = 0
                WHERE job_id = ?
                """,
                (ImageJobStatus.RUNNING.value, row["job_id"]),
            )
            connection.commit()
        return self.get(str(row["job_id"]))

    def update_progress(self, job_id: str, step: int, total: int) -> ImageGenerationJob:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET progress_step = ?, progress_total = ?
                WHERE job_id = ? AND status = ?
                """,
                (step, total, job_id, ImageJobStatus.RUNNING.value),
            )
        return self.require(job_id)

    def complete(
        self,
        job_id: str,
        *,
        output_path: Path,
        generation_seconds: float,
        provider_model: str,
        width: int,
        height: int,
    ) -> ImageGenerationJob:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET status = ?, output_path = ?, completed_at = ?,
                    generation_seconds = ?, provider_model = ?,
                    width = ?, height = ?,
                    progress_step = progress_total, error_code = NULL
                WHERE job_id = ?
                """,
                (
                    ImageJobStatus.COMPLETED.value,
                    str(output_path),
                    now,
                    generation_seconds,
                    provider_model,
                    width,
                    height,
                    job_id,
                ),
            )
        return self.require(job_id)

    def fail(self, job_id: str, *, error_code: str) -> ImageGenerationJob:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET status = ?, completed_at = ?, error_code = ?
                WHERE job_id = ?
                """,
                (ImageJobStatus.FAILED.value, now, error_code[:200], job_id),
            )
        return self.require(job_id)

    def mark_delivered(
        self,
        job_id: str,
        *,
        message_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET delivered = 1,
                    delivery_message_id = COALESCE(?, delivery_message_id)
                WHERE job_id = ?
                """,
                (message_id, job_id),
            )

    def set_delivery_message(self, job_id: str, message_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET delivery_message_id = ?
                WHERE job_id = ?
                """,
                (message_id, job_id),
            )

    def mark_handed_off(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE image_generation_jobs
                SET handoff_completed = 1
                WHERE job_id = ?
                """,
                (job_id,),
            )

    def get(self, job_id: str) -> ImageGenerationJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM image_generation_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return _row_job(row) if row is not None else None

    def require(self, job_id: str) -> ImageGenerationJob:
        job = self.get(job_id)
        if job is None:
            raise RuntimeError(f"Image generation job disappeared: {job_id}")
        return job

    def pending_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM image_generation_jobs
                WHERE status IN (?, ?)
                """,
                (ImageJobStatus.QUEUED.value, ImageJobStatus.RUNNING.value),
            ).fetchone()
        return int(row["count"])

    def undelivered_terminal(self) -> tuple[ImageGenerationJob, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM image_generation_jobs
                WHERE status IN (?, ?) AND delivered = 0
                ORDER BY created_at, job_id
                """,
                (ImageJobStatus.COMPLETED.value, ImageJobStatus.FAILED.value),
            ).fetchall()
        return tuple(_row_job(row) for row in rows)

    def recent_count(
        self,
        *,
        actor_id: str | None,
        workspace_id: str | None,
        since: datetime,
    ) -> int:
        with self._connect() as connection:
            return self._count_recent(
                connection,
                actor_id=actor_id,
                workspace_id=workspace_id,
                since=since,
            )

    def prune_delivered_terminal(
        self,
        *,
        before: datetime,
    ) -> tuple[int, tuple[Path, ...]]:
        """Forget old terminal jobs after delivery or agent file handoff."""

        if before.tzinfo is None:
            raise ValueError("Retention cutoffs must be timezone-aware.")
        cutoff = before.astimezone(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT output_path
                FROM image_generation_jobs
                WHERE status IN (?, ?)
                  AND (delivered = 1 OR handoff_completed = 1)
                  AND COALESCE(completed_at, created_at) < ?
                """,
                (
                    ImageJobStatus.COMPLETED.value,
                    ImageJobStatus.FAILED.value,
                    cutoff,
                ),
            ).fetchall()
            connection.execute(
                """
                DELETE FROM image_generation_jobs
                WHERE status IN (?, ?)
                  AND (delivered = 1 OR handoff_completed = 1)
                  AND COALESCE(completed_at, created_at) < ?
                """,
                (
                    ImageJobStatus.COMPLETED.value,
                    ImageJobStatus.FAILED.value,
                    cutoff,
                ),
            )
            connection.commit()
        return (
            len(rows),
            tuple(
                Path(str(row["output_path"]))
                for row in rows
                if row["output_path"]
            ),
        )

    def prune_oldest_delivered_terminal(self) -> tuple[bool, tuple[Path, ...]]:
        """Remove one oldest safely handed-off terminal job under storage pressure."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT job_id, output_path
                FROM image_generation_jobs
                WHERE status IN (?, ?)
                  AND (delivered = 1 OR handoff_completed = 1)
                ORDER BY COALESCE(completed_at, created_at), job_id
                LIMIT 1
                """,
                (
                    ImageJobStatus.COMPLETED.value,
                    ImageJobStatus.FAILED.value,
                ),
            ).fetchone()
            if row is None:
                connection.commit()
                return False, ()
            connection.execute(
                "DELETE FROM image_generation_jobs WHERE job_id = ?",
                (row["job_id"],),
            )
            connection.commit()
        return (
            True,
            (
                (Path(str(row["output_path"])),)
                if row["output_path"]
                else ()
            ),
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS image_generation_jobs (
                    job_id TEXT PRIMARY KEY,
                    actor_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    delivery_target_id TEXT NOT NULL,
                    reply_to_message_id TEXT,
                    prompt_json TEXT NOT NULL,
                    caption_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output_path TEXT,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    generation_seconds REAL,
                    provider_model TEXT,
                    error_code TEXT,
                    progress_step INTEGER NOT NULL DEFAULT 0,
                    progress_total INTEGER NOT NULL DEFAULT 12,
                    delivery_message_id TEXT,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    auto_deliver INTEGER NOT NULL DEFAULT 0,
                    handoff_completed INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS image_jobs_status_created
                ON image_generation_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS image_jobs_actor_created
                ON image_generation_jobs(actor_id, created_at);
                CREATE INDEX IF NOT EXISTS image_jobs_workspace_created
                ON image_generation_jobs(workspace_id, created_at);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(image_generation_jobs)"
                ).fetchall()
            }
            if "delivery_message_id" not in columns:
                connection.execute(
                    "ALTER TABLE image_generation_jobs "
                    "ADD COLUMN delivery_message_id TEXT"
                )
            if "provider_model" not in columns:
                connection.execute(
                    "ALTER TABLE image_generation_jobs "
                    "ADD COLUMN provider_model TEXT"
                )
                connection.execute(
                    """
                    UPDATE image_generation_jobs
                    SET provider_model = ?
                    WHERE status = ? AND output_path IS NOT NULL
                    """,
                    (
                        "Ideogram 4・ローカルMLX (旧経路)",
                        ImageJobStatus.COMPLETED.value,
                    ),
                )
            if "auto_deliver" not in columns:
                connection.execute(
                    "ALTER TABLE image_generation_jobs "
                    "ADD COLUMN auto_deliver INTEGER NOT NULL DEFAULT 0"
                )
            if "handoff_completed" not in columns:
                connection.execute(
                    "ALTER TABLE image_generation_jobs "
                    "ADD COLUMN handoff_completed INTEGER NOT NULL DEFAULT 0"
                )
        self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _insert(
        connection: sqlite3.Connection,
        job: ImageGenerationJob,
    ) -> None:
        connection.execute(
            """
            INSERT INTO image_generation_jobs (
                job_id, actor_id, workspace_id, delivery_target_id,
                reply_to_message_id, prompt_json, caption_json, status,
                output_path, width, height, seed, created_at, completed_at,
                generation_seconds, provider_model, error_code,
                progress_step, progress_total, delivery_message_id, delivered,
                auto_deliver, handoff_completed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _job_values(job),
        )

    @staticmethod
    def _count_recent(
        connection: sqlite3.Connection,
        *,
        actor_id: str | None,
        workspace_id: str | None,
        since: datetime,
    ) -> int:
        clauses = ["created_at >= ?"]
        values: list[object] = [since.isoformat()]
        if actor_id is not None:
            clauses.append("actor_id = ?")
            values.append(actor_id)
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            values.append(workspace_id)
        row = connection.execute(
            f"SELECT COUNT(*) AS count FROM image_generation_jobs "
            f"WHERE {' AND '.join(clauses)}",
            values,
        ).fetchone()
        return int(row["count"])


class ImageGenerationService:
    """One restart-safe local generation worker shared by all transports."""

    def __init__(
        self,
        *,
        provider: ImageGenerationProvider | None,
        store: ImageGenerationStore,
        journal: EventJournal,
        output_dir: Path,
        per_user_requests: int,
        per_user_window_seconds: int,
        per_workspace_requests: int,
        per_workspace_window_seconds: int,
        max_pending_jobs: int,
        rate_limit_exempt_actor_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.provider = provider
        self.store = store
        self.journal = journal
        self.output_dir = output_dir
        self.per_user_requests = per_user_requests
        self.per_user_window_seconds = per_user_window_seconds
        self.per_workspace_requests = per_workspace_requests
        self.per_workspace_window_seconds = per_workspace_window_seconds
        self.max_pending_jobs = max_pending_jobs
        self.rate_limit_exempt_actor_ids = rate_limit_exempt_actor_ids
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._delivery_handler: ImageDeliveryHandler | None = None
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._delivery_retry_tasks: dict[str, asyncio.Task[None]] = {}
        self._delivery_retry_requested: set[str] = set()
        self._delivery_locks: dict[str, _DeliveryLockState] = {}
        self._terminal_futures: dict[str, asyncio.Future[None]] = {}
        self._closed = False

    async def start(self, delivery_handler: ImageDeliveryHandler) -> None:
        self._delivery_handler = delivery_handler
        requeued = await asyncio.to_thread(self.store.requeue_running)
        if requeued:
            log.warning("Requeued %s interrupted image generation job(s)", requeued)
        self._ensure_worker()
        for job in await asyncio.to_thread(self.store.undelivered_terminal):
            if not job.auto_deliver:
                continue
            self._schedule_delivery_retry(job.job_id, immediate=True)
        if await asyncio.to_thread(self.store.pending_count):
            self._wake.set()

    async def submit(
        self,
        *,
        actor_id: str,
        workspace_id: str,
        delivery_target_id: str,
        reply_to_message_id: str | None,
        prompt: ImageGenerationPrompt,
        auto_deliver: bool = False,
        idempotency_key: str | None = None,
    ) -> ImageGenerationJob:
        job_id = (
            _idempotent_job_id(
                actor_id=actor_id,
                workspace_id=workspace_id,
                idempotency_key=idempotency_key,
            )
            if idempotency_key is not None
            else uuid.uuid4().hex
        )
        if idempotency_key is not None:
            existing = await asyncio.to_thread(self.store.get, job_id)
            if existing is not None:
                return existing
        if self.provider is None:
            raise UserError("image.not_configured")
        _validate_prompt(prompt)
        now = datetime.now(UTC)
        width, height = _DIMENSIONS[prompt.aspect_ratio]
        seed = prompt.seed if prompt.seed is not None else secrets.randbelow(1_000_000_000)
        if not 0 <= seed <= 2_147_483_647:
            raise UserError("image.seed_invalid")
        job = ImageGenerationJob(
            job_id=job_id,
            actor_id=actor_id,
            workspace_id=workspace_id,
            delivery_target_id=delivery_target_id,
            reply_to_message_id=reply_to_message_id,
            prompt=prompt,
            brief_json=build_image_brief(prompt),
            status=ImageJobStatus.QUEUED,
            output_path=None,
            width=width,
            height=height,
            seed=seed,
            created_at_iso=now.isoformat(),
            auto_deliver=auto_deliver,
        )
        rejection_code = await asyncio.to_thread(
            self.store.admit,
            job,
            enforce_rate_limits=actor_id not in self.rate_limit_exempt_actor_ids,
            actor_since=now - timedelta(seconds=self.per_user_window_seconds),
            actor_limit=self.per_user_requests,
            workspace_since=now - timedelta(seconds=self.per_workspace_window_seconds),
            workspace_limit=self.per_workspace_requests,
            pending_limit=self.max_pending_jobs,
        )
        if rejection_code == "image.idempotent_replay":
            return await asyncio.to_thread(self.store.require, job_id)
        if rejection_code is not None:
            raise UserError(rejection_code)
        self._ensure_worker()
        try:
            await self._append_journal(
                kind="image.job.queued",
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport="image",
                request_id=job.job_id,
                payload={
                    "job_id": job.job_id,
                    "aspect_ratio": prompt.aspect_ratio.value,
                    "rendering": prompt.rendering.value,
                    "width": width,
                    "height": height,
                },
            )
        finally:
            self._wake.set()
        return job

    async def wait_for_terminal(self, job_id: str) -> ImageGenerationJob:
        """Wait for durable generation completion independently of delivery."""

        while not self._closed:
            job = await asyncio.to_thread(self.store.require, job_id)
            if job.status in {ImageJobStatus.COMPLETED, ImageJobStatus.FAILED}:
                return job
            future = self._terminal_futures.get(job_id)
            if future is None or future.done():
                future = asyncio.get_running_loop().create_future()
                self._terminal_futures[job_id] = future
            # Recheck after publishing the shared one-shot future. Completion
            # can now either be observed durably here or resolve the future;
            # unlike a shared Event, one waiter cannot clear another waiter's
            # notification.
            job = await asyncio.to_thread(self.store.require, job_id)
            if job.status in {ImageJobStatus.COMPLETED, ImageJobStatus.FAILED}:
                self._resolve_terminal_waiters(job_id)
                return job
            await future
        raise RuntimeError("Image generation service closed before completion.")

    async def mark_delivered(
        self,
        job_id: str,
        *,
        message_id: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.store.mark_delivered,
            job_id,
            message_id=message_id,
        )

    async def set_delivery_message(self, job_id: str, message_id: str) -> None:
        await asyncio.to_thread(self.store.set_delivery_message, job_id, message_id)

    async def mark_handed_off(self, job_id: str) -> None:
        await asyncio.to_thread(self.store.mark_handed_off, job_id)

    async def owned_job(
        self,
        job_id: str,
        *,
        actor_id: str,
    ) -> ImageGenerationJob:
        job = await asyncio.to_thread(self.store.get, job_id)
        if job is None or job.actor_id != actor_id:
            raise UserError("image.job_not_found")
        return job

    async def close(self) -> None:
        self._closed = True
        self._wake.set()
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        retry_tasks = tuple(self._delivery_retry_tasks.values())
        for task in retry_tasks:
            task.cancel()
        if retry_tasks:
            await asyncio.gather(*retry_tasks, return_exceptions=True)
        self._delivery_retry_tasks.clear()
        self._delivery_retry_requested.clear()
        self._delivery_locks.clear()
        for future in self._terminal_futures.values():
            if not future.done():
                future.set_result(None)
        self._terminal_futures.clear()
        if self.provider is not None:
            await self.provider.close()

    def _ensure_worker(self) -> None:
        if self._closed or self.provider is None:
            return
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(
                self._run(),
                name="simajilord-image-generation",
            )

    async def _run(self) -> None:
        while not self._closed:
            job = await asyncio.to_thread(self.store.claim_next)
            if job is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._notify(job)
            output = self.output_dir / f"{job.job_id}.png"
            try:
                assert self.provider is not None
                result = await self.provider.generate(
                    brief_json=job.brief_json,
                    destination=output,
                    width=job.width,
                    height=job.height,
                    seed=job.seed,
                    model=job.prompt.model,
                    on_progress=self._progress_callback(job.job_id),
                )
                terminal = await asyncio.to_thread(
                    self.store.complete,
                    job.job_id,
                    output_path=output,
                    generation_seconds=result.generation_seconds,
                    provider_model=result.model,
                    width=result.width or job.width,
                    height=result.height or job.height,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Image generation failed job=%s", job.job_id)
                terminal = await asyncio.to_thread(
                    self.store.fail,
                    job.job_id,
                    error_code=_image_error_code(exc),
                )
                self._resolve_terminal_waiters(job.job_id)
                await self._append_journal(
                    kind="image.job.failed",
                    actor_id=job.actor_id,
                    workspace_id=job.workspace_id,
                    transport="image",
                    request_id=job.job_id,
                    payload={
                        "job_id": job.job_id,
                        "error_type": type(exc).__name__,
                        "error_code": terminal.error_code,
                    },
                )
            else:
                self._resolve_terminal_waiters(job.job_id)
                await self._append_journal(
                    kind="image.job.completed",
                    actor_id=job.actor_id,
                    workspace_id=job.workspace_id,
                    transport="image",
                    request_id=job.job_id,
                    payload={
                        "job_id": job.job_id,
                        "generation_seconds": result.generation_seconds,
                        "model": result.model,
                    },
                )
            await self._notify(terminal)

    def _resolve_terminal_waiters(self, job_id: str) -> None:
        future = self._terminal_futures.pop(job_id, None)
        if future is not None and not future.done():
            future.set_result(None)

    def _progress_callback(self, job_id: str) -> ImageProgressCallback:
        async def progress(step: int, total: int) -> None:
            current = await asyncio.to_thread(
                self.store.update_progress,
                job_id,
                step,
                total,
            )
            await self._notify(current)

        return progress

    async def _notify(self, job: ImageGenerationJob) -> None:
        if self._delivery_handler is None or not job.auto_deliver:
            return
        try:
            await self._deliver_latest(job.job_id)
        except Exception:
            log.exception("Image job delivery update failed job=%s", job.job_id)
            self._schedule_delivery_retry(job.job_id)

    def _schedule_delivery_retry(
        self,
        job_id: str,
        *,
        immediate: bool = False,
    ) -> None:
        if self._closed or self._delivery_handler is None:
            return
        current = self._delivery_retry_tasks.get(job_id)
        if current is not None and not current.done():
            self._delivery_retry_requested.add(job_id)
            return
        self._delivery_retry_requested.discard(job_id)
        self._delivery_retry_tasks[job_id] = asyncio.create_task(
            self._retry_delivery(job_id, immediate=immediate),
            name=f"simajilord-image-delivery-{job_id}",
        )

    async def _retry_delivery(self, job_id: str, *, immediate: bool) -> None:
        delay = 0.0 if immediate else 1.0
        try:
            while not self._closed:
                if delay:
                    await asyncio.sleep(delay)
                self._delivery_retry_requested.discard(job_id)
                job = await asyncio.to_thread(self.store.get, job_id)
                if job is None or job.delivered or not job.auto_deliver:
                    return
                handler = self._delivery_handler
                if handler is None:
                    return
                try:
                    await self._deliver_latest(job_id)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "Image job delivery retry failed job=%s delay=%s",
                        job_id,
                        delay,
                    )
                    delay = min(max(1.0, delay * 2), 60.0)
                    continue
                current = await asyncio.to_thread(self.store.get, job_id)
                if current is None or current.delivered:
                    return
                if job_id in self._delivery_retry_requested:
                    delay = 0.0
                    continue
                if current.status in {
                    ImageJobStatus.COMPLETED,
                    ImageJobStatus.FAILED,
                }:
                    log.warning(
                        "Image terminal delivery returned without being marked "
                        "delivered job=%s",
                        job_id,
                    )
                    delay = min(max(1.0, delay * 2), 60.0)
                    continue
                return
        finally:
            if self._delivery_retry_tasks.get(job_id) is asyncio.current_task():
                self._delivery_retry_tasks.pop(job_id, None)
            self._delivery_retry_requested.discard(job_id)

    async def _deliver_latest(self, job_id: str) -> None:
        """Serialize delivery and hand the transport the latest durable job state."""

        async with self._delivery_lock(job_id):
            job = await asyncio.to_thread(self.store.get, job_id)
            if job is None or job.delivered or not job.auto_deliver:
                return
            handler = self._delivery_handler
            if handler is None:
                return
            await handler(job)

    @asynccontextmanager
    async def _delivery_lock(self, job_id: str) -> AsyncIterator[None]:
        state = self._delivery_locks.get(job_id)
        if state is None:
            state = _DeliveryLockState(lock=asyncio.Lock())
            self._delivery_locks[job_id] = state
        state.users += 1
        acquired = False
        try:
            await state.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                state.lock.release()
            state.users -= 1
            if (
                state.users == 0
                and self._delivery_locks.get(job_id) is state
            ):
                self._delivery_locks.pop(job_id, None)

    async def _append_journal(
        self,
        *,
        kind: str,
        actor_id: str,
        workspace_id: str,
        transport: str,
        request_id: str,
        payload: dict[str, object],
    ) -> None:
        """Keep best-effort observability from changing durable job outcomes."""

        try:
            await self.journal.append(
                kind=kind,
                actor_id=actor_id,
                workspace_id=workspace_id,
                transport=transport,
                request_id=request_id,
                payload=payload,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "Image job journal append failed kind=%s job=%s",
                kind,
                request_id,
            )


def build_image_brief(prompt: ImageGenerationPrompt) -> str:
    """Compile typed fields into the provider-neutral GPT Image brief."""

    payload = {
        "subject": prompt.subject,
        "scene": prompt.scene,
        "composition": prompt.composition,
        "style": prompt.style,
        "lighting": prompt.lighting,
        "required_details": prompt.details.strip(),
        "avoid": prompt.avoid.strip(),
        "rendering": prompt.rendering.value,
        "aspect_ratio": prompt.aspect_ratio.value,
        "model": prompt.model.value,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_prompt(prompt: ImageGenerationPrompt) -> None:
    fields = {
        "subject": prompt.subject,
        "scene": prompt.scene,
        "composition": prompt.composition,
        "style": prompt.style,
        "lighting": prompt.lighting,
        "details": prompt.details,
        "avoid": prompt.avoid,
    }
    required = {"subject", "scene", "composition", "style", "lighting"}
    for name, value in fields.items():
        if name not in required and not value:
            continue
        if not value.strip():
            raise UserError("image.prompt_field_empty", field=name)
        if len(value) > 1_200:
            raise UserError("image.prompt_field_too_long", field=name)
    combined = "\n".join(fields.values())
    if len(combined) > 3_000:
        raise UserError("image.prompt_too_long")
    if _DISALLOWED_PROMPT.search(combined):
        raise UserError("image.prompt_rejected")


def _image_error_code(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "generation_timeout"
    if isinstance(exc, ProviderError):
        detail = str(exc).lower()
        if "brief" in detail or "strict" in detail:
            return "caption_invalid"
        if "timed out" in detail:
            return "generation_timeout"
        if "png" in detail or "image data" in detail:
            return "invalid_output"
        return "provider_failed"
    return type(exc).__name__


def _idempotent_job_id(
    *,
    actor_id: str,
    workspace_id: str,
    idempotency_key: str,
) -> str:
    material = f"image.generate\0{actor_id}\0{workspace_id}\0{idempotency_key}"
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def _job_values(job: ImageGenerationJob) -> tuple[object, ...]:
    prompt_json = json.dumps(
        {
            **asdict(job.prompt),
            "aspect_ratio": job.prompt.aspect_ratio.value,
            "rendering": job.prompt.rendering.value,
            "model": job.prompt.model.value,
        },
        ensure_ascii=False,
    )
    return (
        job.job_id,
        job.actor_id,
        job.workspace_id,
        job.delivery_target_id,
        job.reply_to_message_id,
        prompt_json,
        job.brief_json,
        job.status.value,
        str(job.output_path) if job.output_path else None,
        job.width,
        job.height,
        job.seed,
        job.created_at_iso,
        job.completed_at_iso,
        job.generation_seconds,
        job.provider_model,
        job.error_code,
        job.progress_step,
        job.progress_total,
        job.delivery_message_id,
        int(job.delivered),
        int(job.auto_deliver),
        int(job.handoff_completed),
    )


def _row_job(row: sqlite3.Row) -> ImageGenerationJob:
    raw: Any = json.loads(str(row["prompt_json"]))
    prompt = ImageGenerationPrompt(
        subject=str(raw["subject"]),
        scene=str(raw["scene"]),
        composition=str(raw["composition"]),
        style=str(raw["style"]),
        lighting=str(raw["lighting"]),
        details=str(raw.get("details", "")),
        avoid=str(raw.get("avoid", "")),
        aspect_ratio=ImageAspectRatio(str(raw["aspect_ratio"])),
        rendering=ImageRendering(str(raw["rendering"])),
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        model=ImageGenerationModel(
            str(raw.get("model", ImageGenerationModel.GPT_IMAGE_2.value))
        ),
    )
    output = row["output_path"]
    return ImageGenerationJob(
        job_id=str(row["job_id"]),
        actor_id=str(row["actor_id"]),
        workspace_id=str(row["workspace_id"]),
        delivery_target_id=str(row["delivery_target_id"]),
        reply_to_message_id=(
            str(row["reply_to_message_id"]) if row["reply_to_message_id"] else None
        ),
        prompt=prompt,
        brief_json=str(row["caption_json"]),
        status=ImageJobStatus(str(row["status"])),
        output_path=Path(str(output)) if output else None,
        width=int(row["width"]),
        height=int(row["height"]),
        seed=int(row["seed"]),
        created_at_iso=str(row["created_at"]),
        completed_at_iso=(
            str(row["completed_at"]) if row["completed_at"] else None
        ),
        generation_seconds=(
            float(row["generation_seconds"])
            if row["generation_seconds"] is not None
            else None
        ),
        provider_model=(
            str(row["provider_model"]) if row["provider_model"] else None
        ),
        error_code=str(row["error_code"]) if row["error_code"] else None,
        progress_step=int(row["progress_step"]),
        progress_total=int(row["progress_total"]),
        delivery_message_id=(
            str(row["delivery_message_id"]) if row["delivery_message_id"] else None
        ),
        delivered=bool(row["delivered"]),
        auto_deliver=bool(row["auto_deliver"]),
        handoff_completed=bool(row["handoff_completed"]),
    )
