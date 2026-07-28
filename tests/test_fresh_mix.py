from __future__ import annotations

from typing import cast

import pytest

from simajilord.capabilities.audio import (
    AudioQueueRequest,
    AudioQueueResponse,
    FreshMixEnqueueRequest,
    FreshMixEnqueueResponse,
    FreshMixPlanRequest,
    FreshMixPreviewResponse,
    FreshMixReviseRequest,
    build_audio_endpoints,
)
from simajilord.core import ApprovalMode, InvocationContext
from simajilord.core.errors import UserError
from simajilord.domain.audio import AudioItem
from simajilord.domain.media import MediaCandidate
from simajilord.services.audio import AudioSession, AudioSessionManager
from simajilord.services.fresh_mix import (
    FreshMixBrief,
    FreshMixService,
    FreshMixVocals,
)
from simajilord.services.media import MediaService


class DisconnectedOutput:
    connected = False
    paused = False

    async def connect(self, destination_id: str) -> None:
        del destination_id
        self.connected = True

    async def play(self, item: AudioItem) -> None:
        del item

    async def overlay_speech(
        self,
        music: AudioItem,
        speech: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        del music, speech, position_seconds

    async def update_music(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
    ) -> None:
        del music, position_seconds

    async def fade_out(
        self,
        music: AudioItem,
        *,
        position_seconds: float,
        duration_seconds: float,
    ) -> None:
        del music, position_seconds, duration_seconds

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None

    def stop(self) -> None:
        return None

    async def disconnect(self) -> None:
        self.connected = False


class SearchProvider:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    async def search_audio(
        self,
        query: str,
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        self.queries.append((query, limit))
        return (
            MediaCandidate(
                "https://youtube.example/excluded",
                "Excluded Song",
                180,
                uploader="Artist A",
            ),
            MediaCandidate(
                "https://youtube.example/one",
                "One (Official Video)",
                180,
                uploader="Artist A",
            ),
            MediaCandidate(
                "https://youtube.example/one-duplicate",
                "One official audio",
                181,
                uploader="Artist B",
            ),
            MediaCandidate(
                "https://youtube.example/cover",
                "Two Cover",
                190,
                uploader="Artist C",
            ),
            MediaCandidate(
                "https://youtube.example/instrumental",
                "Three Instrumental",
                200,
                uploader="Artist D",
            ),
            MediaCandidate(
                "https://youtube.example/two",
                "Four",
                210,
                uploader="Artist B",
            ),
            MediaCandidate(
                "https://youtube.example/three",
                "Five",
                220,
                uploader="Artist A",
            ),
            MediaCandidate(
                "https://youtube.example/four",
                "Six",
                230,
                uploader="Artist E",
            ),
        )

    async def resolve_audio(self, reference: str) -> AudioItem:
        raise AssertionError(f"planning must not resolve a stream: {reference}")

    async def mix_audio(
        self,
        seed_references: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[MediaCandidate, ...]:
        raise AssertionError((seed_references, limit))

    async def download(self, *args: object, **kwargs: object) -> object:
        raise AssertionError((args, kwargs))


@pytest.mark.asyncio
async def test_fresh_mix_uses_real_candidates_without_history_and_filters_versions() -> None:
    provider = SearchProvider()
    service = FreshMixService(MediaService(provider))

    draft = await service.plan(
        workspace_id="guild",
        actor_id="actor",
        brief=FreshMixBrief(
            prompt="quiet coding",
            target_minutes=15,
            max_tracks_per_artist=1,
        ),
        excluded_references=("https://youtube.example/excluded",),
    )

    assert provider.queries == [("quiet coding", 20)]
    references = {track.reference for track in draft.tracks}
    assert "https://youtube.example/excluded" not in references
    assert "https://youtube.example/cover" not in references
    assert "https://youtube.example/instrumental" not in references
    assert len({track.artist for track in draft.tracks}) == len(draft.tracks)
    assert "history_off" in draft.checks
    assert all(track.verified_by == "yt_dlp_search" for track in draft.tracks)


@pytest.mark.asyncio
async def test_fresh_mix_allows_instrumental_only_when_requested() -> None:
    provider = SearchProvider()
    service = FreshMixService(MediaService(provider))

    draft = await service.plan(
        workspace_id="guild",
        actor_id="actor",
        brief=FreshMixBrief(
            prompt="quiet coding",
            target_minutes=15,
            vocals=FreshMixVocals.LOW,
        ),
    )

    assert any("Instrumental" in track.title for track in draft.tracks)


@pytest.mark.asyncio
async def test_fresh_mix_revision_is_scoped_to_actor_and_workspace() -> None:
    service = FreshMixService(MediaService(SearchProvider()))
    draft = await service.plan(
        workspace_id="guild",
        actor_id="actor",
        brief=FreshMixBrief(prompt="coding", target_minutes=15),
    )

    with pytest.raises(UserError, match=r"audio\.fresh_mix_draft_not_found"):
        await service.revise_track(
            draft_id=draft.draft_id,
            workspace_id="other-guild",
            actor_id="actor",
            position=1,
            query="replacement",
        )
    with pytest.raises(UserError, match=r"audio\.fresh_mix_draft_not_found"):
        await service.require(draft.draft_id, "guild", "other-actor")


@pytest.mark.asyncio
async def test_fresh_mix_draft_can_only_be_claimed_by_one_operation() -> None:
    service = FreshMixService(MediaService(SearchProvider()))
    draft = await service.plan(
        workspace_id="guild",
        actor_id="actor",
        brief=FreshMixBrief(prompt="coding", target_minutes=15),
    )

    assert await service.claim(draft.draft_id, "guild", "actor") == draft
    with pytest.raises(UserError, match=r"audio\.fresh_mix_draft_busy"):
        await service.claim(draft.draft_id, "guild", "actor")

    await service.release(draft.draft_id)
    assert await service.claim(draft.draft_id, "guild", "actor") == draft
    await service.release(draft.draft_id)


@pytest.mark.asyncio
async def test_enqueue_many_is_atomic_when_actor_limit_would_be_exceeded() -> None:
    output = DisconnectedOutput()
    state_changes = 0

    async def changed(_: AudioSession) -> None:
        nonlocal state_changes
        state_changes += 1

    session = AudioSession(
        "guild",
        output,
        max_pending_speech=3,
        max_pending_music=10,
        max_pending_music_per_actor=2,
        state_hook=changed,
    )
    baseline = state_changes
    items = tuple(
        AudioItem(
            "",
            f"Track {index}",
            f"https://example.com/{index}",
            resolver_reference=f"https://example.com/{index}",
            requested_by_id="actor",
        )
        for index in range(3)
    )

    with pytest.raises(UserError, match=r"audio\.user_queue_full"):
        await session.enqueue_many(items, wait_for_actor_id="actor")

    snapshot = await session.snapshot()
    assert snapshot.pending == ()
    assert snapshot.waiting_actor_ids == ()
    assert state_changes == baseline
    await session.close()


@pytest.mark.asyncio
async def test_fresh_mix_endpoint_previews_then_atomically_enqueues_stable_refs() -> None:
    provider = SearchProvider()
    media = MediaService(provider)
    manager = AudioSessionManager(max_active=2, max_pending_speech=3)
    manager.get_or_create("guild", DisconnectedOutput)
    endpoints = {
        item.descriptor.name: item
        for item in build_audio_endpoints(media, manager, FreshMixService(media))
    }
    context = InvocationContext("actor", "guild", "test", "request")

    preview = await endpoints["audio.fresh_mix_plan"].invoke(
        FreshMixPlanRequest(prompt="coding", target_minutes=15),
        context,
    )
    assert isinstance(preview, FreshMixPreviewResponse)
    assert preview.history_used is False
    assert endpoints["audio.fresh_mix_enqueue"].descriptor.approval is ApprovalMode.WHEN_REQUESTED

    enqueued = await endpoints["audio.fresh_mix_enqueue"].invoke(
        FreshMixEnqueueRequest(
            draft_id=preview.draft_id,
            requested_by_name="Tester",
        ),
        context,
    )
    assert isinstance(enqueued, FreshMixEnqueueResponse)
    queue = await endpoints["audio.queue"].invoke(AudioQueueRequest(), context)
    assert isinstance(queue, AudioQueueResponse)
    assert len(queue.pending) == enqueued.track_count
    session = manager.require("guild")
    snapshot = await session.snapshot()
    assert all(item.source == "" for item in snapshot.pending)
    assert all(item.resolver_reference.startswith("https://") for item in snapshot.pending)
    assert snapshot.waiting_actor_ids == ("actor",)
    assert enqueued.playback_state == "waiting_for_voice"

    with pytest.raises(UserError, match=r"audio\.fresh_mix_draft_not_found"):
        await endpoints["audio.fresh_mix_enqueue"].invoke(
            FreshMixEnqueueRequest(draft_id=preview.draft_id),
            context,
        )
    await manager.close()


def test_fresh_mix_request_types_remain_transport_neutral() -> None:
    assert FreshMixReviseRequest("draft", 1, "new query").position == 1
    assert cast(object, FreshMixPlanRequest("focus")).history_policy == "ignore"
