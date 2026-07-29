from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from simajilord.capabilities.image import (
    ImageGenerateRequest,
    ImageGenerateResponse,
    build_image_endpoints,
)
from simajilord.core import CapabilityRegistry, InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.image import (
    ImageGenerationJob,
    ImageGenerationPrompt,
    ImageJobStatus,
    ImageRendering,
)
from simajilord.observability import EventJournal
from simajilord.providers.image import ImageProviderResult
from simajilord.services.image import (
    ImageGenerationService,
    ImageGenerationStore,
    build_ideogram_caption,
)


class FakeImageProvider:
    async def generate(
        self,
        *,
        caption_json: str,
        destination: Path,
        width: int,
        height: int,
        seed: int,
        on_progress: object = None,
    ) -> ImageProviderResult:
        del caption_json, width, height, seed
        if callable(on_progress):
            await on_progress(6, 12)
            await on_progress(12, 12)
        destination.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return ImageProviderResult(generation_seconds=0.01, model="fake")


def _service(
    tmp_path: Path,
    *,
    exempt: frozenset[str] = frozenset(),
) -> ImageGenerationService:
    return ImageGenerationService(
        provider=FakeImageProvider(),
        store=ImageGenerationStore(tmp_path / "image.sqlite3"),
        journal=EventJournal(tmp_path / "events.sqlite3"),
        output_dir=tmp_path / "output",
        per_user_requests=1,
        per_user_window_seconds=3_600,
        per_workspace_requests=1,
        per_workspace_window_seconds=3_600,
        max_pending_jobs=10,
        rate_limit_exempt_actor_ids=exempt,
    )


def _prompt(subject: str = "a cat") -> ImageGenerationPrompt:
    return ImageGenerationPrompt(
        subject=subject,
        scene="a quiet sunlit room",
        composition="eye-level portrait, centered subject",
        style="clean editorial illustration",
        lighting="soft window light",
        rendering=ImageRendering.ILLUSTRATION,
    )


def test_ideogram_caption_uses_rendering_specific_canonical_key_order() -> None:
    illustration = json.loads(build_ideogram_caption(_prompt()))
    assert tuple(illustration["style_description"]) == (
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
    )

    photo = json.loads(
        build_ideogram_caption(
            ImageGenerationPrompt(
                subject="a cat",
                scene="a studio",
                composition="centered",
                style="editorial photography",
                lighting="softbox",
                rendering=ImageRendering.PHOTO,
            )
        )
    )
    assert tuple(photo["style_description"]) == (
        "aesthetics",
        "lighting",
        "photo",
        "medium",
    )


def test_ideogram_caption_preserves_full_production_brief() -> None:
    prompt = ImageGenerationPrompt(
        subject="Exactly one orange cat sitting upright with amber eyes",
        scene="A rainy apartment window with a low walnut table and city bokeh",
        composition="Landscape eye-level portrait with the cat on the left third",
        style="Natural editorial pet photography with realistic fur texture",
        lighting="Cool window light from the right and a warm lamp rim from the left",
        details="Four coherent paws, two ears, one tail, sharp eyes, blue blanket",
        avoid="extra animals, extra limbs, cropped tail, text, logos, watermarks",
        rendering=ImageRendering.PHOTO,
    )

    caption = json.loads(build_ideogram_caption(prompt))

    assert prompt.subject in caption["high_level_description"]
    assert prompt.scene in caption["high_level_description"]
    assert prompt.composition in caption["high_level_description"]
    element = caption["compositional_deconstruction"]["elements"][0]["desc"]
    assert prompt.details in element
    assert prompt.avoid in element


def test_image_pressure_pruning_distinguishes_fileless_job_from_empty_store(
    tmp_path: Path,
) -> None:
    store = ImageGenerationStore(tmp_path / "image.sqlite3")
    delivered = ImageGenerationJob(
        job_id="delivered-failure",
        actor_id="actor",
        workspace_id="guild",
        delivery_target_id="channel",
        reply_to_message_id=None,
        prompt=_prompt(),
        caption_json=build_ideogram_caption(_prompt()),
        status=ImageJobStatus.FAILED,
        output_path=None,
        width=512,
        height=512,
        seed=1,
        created_at_iso=(datetime.now(UTC) - timedelta(days=2)).isoformat(),
        completed_at_iso=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        error_code="provider.failed",
        delivered=True,
    )
    undelivered = ImageGenerationJob(
        job_id="undelivered-failure",
        actor_id="actor",
        workspace_id="guild",
        delivery_target_id="channel",
        reply_to_message_id=None,
        prompt=_prompt(),
        caption_json=build_ideogram_caption(_prompt()),
        status=ImageJobStatus.FAILED,
        output_path=None,
        width=512,
        height=512,
        seed=2,
        created_at_iso=(datetime.now(UTC) - timedelta(days=2)).isoformat(),
        completed_at_iso=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        error_code="provider.failed",
        delivered=False,
    )
    store.insert(delivered)
    store.insert(undelivered)

    removed, paths = store.prune_oldest_delivered_terminal()

    assert removed is True
    assert paths == ()
    assert store.get(delivered.job_id) is None
    assert store.get(undelivered.job_id) is not None
    assert store.prune_oldest_delivered_terminal() == (False, ())


@pytest.mark.asyncio
async def test_image_worker_persists_progress_and_terminal_delivery(tmp_path: Path) -> None:
    service = _service(tmp_path)
    delivered: list[ImageJobStatus] = []

    async def delivery(job: object) -> None:
        delivered.append(job.status)  # type: ignore[attr-defined]

    await service.start(delivery)
    job = await service.submit(
        actor_id="actor",
        workspace_id="guild",
        delivery_target_id="channel",
        reply_to_message_id="message",
        prompt=_prompt(),
    )
    for _ in range(100):
        current = service.store.require(job.job_id)
        if current.status is ImageJobStatus.COMPLETED:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("image worker did not complete")

    current = service.store.require(job.job_id)
    assert current.output_path is not None and current.output_path.is_file()
    assert current.progress_step == 12
    assert ImageJobStatus.RUNNING in delivered
    assert ImageJobStatus.COMPLETED in delivered
    await service.set_delivery_message(job.job_id, "progress-message")
    assert service.store.require(job.job_id).delivery_message_id == "progress-message"
    await service.close()


@pytest.mark.asyncio
async def test_image_terminal_delivery_retries_without_service_restart(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    terminal_attempts = 0

    async def delivery(job: ImageGenerationJob) -> None:
        nonlocal terminal_attempts
        if job.status is not ImageJobStatus.COMPLETED:
            return
        terminal_attempts += 1
        if terminal_attempts == 1:
            raise ConnectionError("temporary Discord outage")
        await service.mark_delivered(job.job_id)

    await service.start(delivery)
    job = await service.submit(
        actor_id="actor",
        workspace_id="guild",
        delivery_target_id="channel",
        reply_to_message_id="message",
        prompt=_prompt(),
    )
    for _ in range(250):
        if service.store.require(job.job_id).delivered:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("terminal image delivery was not retried")

    assert terminal_attempts == 2
    await service.close()


@pytest.mark.asyncio
async def test_terminal_retry_is_not_lost_while_an_older_retry_is_in_flight(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    first_delivery_started = asyncio.Event()
    release_first_delivery = asyncio.Event()
    delivered = asyncio.Event()
    seen: list[ImageJobStatus] = []

    async def delivery(job: ImageGenerationJob) -> None:
        seen.append(job.status)
        if len(seen) == 1:
            first_delivery_started.set()
            await release_first_delivery.wait()
            return
        if job.status is ImageJobStatus.COMPLETED:
            await service.mark_delivered(job.job_id)
            delivered.set()

    await service.start(delivery)
    job = ImageGenerationJob(
        job_id="retry-race",
        actor_id="actor",
        workspace_id="guild",
        delivery_target_id="channel",
        reply_to_message_id="message",
        prompt=_prompt(),
        caption_json=build_ideogram_caption(_prompt()),
        status=ImageJobStatus.RUNNING,
        output_path=None,
        width=512,
        height=512,
        seed=1,
        created_at_iso=datetime.now(UTC).isoformat(),
    )
    service.store.insert(job)
    service._schedule_delivery_retry(job.job_id, immediate=True)
    await asyncio.wait_for(first_delivery_started.wait(), timeout=1)

    output = tmp_path / "output" / "retry-race.png"
    output.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    service.store.complete(
        job.job_id,
        output_path=output,
        generation_seconds=0.01,
    )
    service._schedule_delivery_retry(job.job_id)
    release_first_delivery.set()

    await asyncio.wait_for(delivered.wait(), timeout=1)
    assert seen == [ImageJobStatus.RUNNING, ImageJobStatus.COMPLETED]
    assert service.store.require(job.job_id).delivered is True
    await service.close()


@pytest.mark.asyncio
async def test_image_rate_limit_check_and_insert_are_atomic(tmp_path: Path) -> None:
    service = _service(tmp_path)

    async def submit(index: int) -> ImageGenerationJob | UserError:
        try:
            return await service.submit(
                actor_id="same-user",
                workspace_id="same-guild",
                delivery_target_id="channel",
                reply_to_message_id="message",
                prompt=_prompt(f"subject {index}"),
            )
        except UserError as exc:
            return exc

    results = await asyncio.gather(*(submit(index) for index in range(8)))
    accepted = [result for result in results if isinstance(result, ImageGenerationJob)]
    rejected = [result for result in results if isinstance(result, UserError)]

    assert len(accepted) == 1
    assert len(rejected) == 7
    assert {error.code for error in rejected} == {"image.user_limit_reached"}
    assert service.store.recent_count(
        actor_id="same-user",
        workspace_id=None,
        since=datetime.now(UTC) - timedelta(minutes=1),
    ) == 1
    await service.close()


@pytest.mark.asyncio
async def test_image_limits_exempt_only_configured_actor(tmp_path: Path) -> None:
    service = _service(tmp_path, exempt=frozenset({"admin"}))
    for subject in ("one", "two"):
        await service.submit(
            actor_id="admin",
            workspace_id="guild",
            delivery_target_id="channel",
            reply_to_message_id="message",
            prompt=_prompt(subject),
        )
    await service.submit(
        actor_id="user",
        workspace_id="other-guild",
        delivery_target_id="channel",
        reply_to_message_id="message",
        prompt=_prompt("first"),
    )
    with pytest.raises(UserError, match=r"image\.user_limit_reached"):
        await service.submit(
            actor_id="user",
            workspace_id="other-guild-2",
            delivery_target_id="channel",
            reply_to_message_id="message",
            prompt=_prompt("second"),
        )
    await service.close()


@pytest.mark.asyncio
async def test_image_capability_enforces_agent_delivery_scope(tmp_path: Path) -> None:
    service = _service(tmp_path)
    registry = CapabilityRegistry()
    for item in build_image_endpoints(service):
        registry.register(item)
    request = ImageGenerateRequest(
        delivery_target_id="forbidden-channel",
        reply_to_event_id="123",
        subject="a friendly shiba inu",
        scene="a park",
        composition="full-body portrait",
        style="polished animation still",
        lighting="golden hour",
    )
    context = InvocationContext(
        actor_id="actor",
        workspace_id="guild",
        transport="agent",
        request_id="discord:message:123",
        resource_ids=("allowed-channel",),
    )
    with pytest.raises(UserError, match=r"image\.delivery_target_forbidden"):
        await registry.invoke("image.generate", request, context)
    await service.close()


@pytest.mark.asyncio
async def test_image_capability_normalizes_exact_agent_event_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    registry = CapabilityRegistry()
    for item in build_image_endpoints(service):
        registry.register(item)
    response = cast(
        ImageGenerateResponse,
        await registry.invoke(
            "image.generate",
            ImageGenerateRequest(
                delivery_target_id="channel",
                reply_to_event_id="discord:message:123",
                subject="a friendly shiba inu",
                scene="a park",
                composition="full-body portrait",
                style="polished animation still",
                lighting="golden hour",
            ),
            InvocationContext(
                actor_id="actor",
                workspace_id="guild",
                transport="agent",
                request_id="discord:message:123",
                resource_ids=("channel",),
            ),
        ),
    )
    job = service.store.require(response.job_id)
    assert job.reply_to_message_id == "123"
    await service.close()


@pytest.mark.asyncio
async def test_image_retention_requires_terminal_delivery(tmp_path: Path) -> None:
    service = _service(tmp_path, exempt=frozenset({"actor"}))
    jobs = [
        await service.submit(
            actor_id="actor",
            workspace_id="guild",
            delivery_target_id="channel",
            reply_to_message_id="message",
            prompt=_prompt(subject),
        )
        for subject in ("delivered", "not delivered")
    ]
    old = datetime.now(UTC) - timedelta(days=31)
    output = tmp_path / "output" / "delivered.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"png")
    with sqlite3.connect(service.store.path) as connection:
        connection.execute(
            """
            UPDATE image_generation_jobs
            SET status = ?, completed_at = ?, output_path = ?, delivered = 1
            WHERE job_id = ?
            """,
            (
                ImageJobStatus.COMPLETED.value,
                old.isoformat(),
                str(output),
                jobs[0].job_id,
            ),
        )
        connection.execute(
            """
            UPDATE image_generation_jobs
            SET status = ?, completed_at = ?, delivered = 0
            WHERE job_id = ?
            """,
            (
                ImageJobStatus.FAILED.value,
                old.isoformat(),
                jobs[1].job_id,
            ),
        )

    removed_jobs, paths = service.store.prune_delivered_terminal(
        before=datetime.now(UTC) - timedelta(days=30)
    )
    assert removed_jobs == 1
    assert paths == (output,)
    assert service.store.get(jobs[0].job_id) is None
    assert service.store.get(jobs[1].job_id) is not None
    await service.close()
