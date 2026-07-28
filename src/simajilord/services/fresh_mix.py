"""History-free, validated music mix planning."""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum

from simajilord.core.errors import UserError
from simajilord.domain.media import MediaCandidate
from simajilord.services.media import MediaService

_VARIANT_TERMS = re.compile(
    r"\b(live|cover|remix|nightcore|slowed|sped\s*up|karaoke)\b",
    re.IGNORECASE,
)
_INSTRUMENTAL_TERM = re.compile(r"\binstrumental\b", re.IGNORECASE)
_LONG_FORM_TERMS = re.compile(r"\b(?:1|2|3|4|5|6|8|10|12|24)\s*hours?\b", re.IGNORECASE)
_SHORTS_TERMS = re.compile(r"(?:#shorts\b|/shorts/)", re.IGNORECASE)
_TITLE_NOISE = re.compile(
    r"\b(official|music|video|audio|lyrics?|mv|hd|4k|remaster(?:ed)?)\b",
    re.IGNORECASE,
)


class FreshMixVocals(StrEnum):
    LOW = "low"
    BALANCED = "balanced"
    ANY = "any"


class FreshMixEnergy(StrEnum):
    CALM = "calm"
    STEADY = "steady"
    RISING = "rising"


@dataclass(frozen=True, slots=True)
class FreshMixPhase:
    minutes: int
    queries: tuple[str, ...]
    label: str = ""


@dataclass(frozen=True, slots=True)
class FreshMixBrief:
    prompt: str
    target_minutes: int = 60
    phases: tuple[FreshMixPhase, ...] = ()
    vocals: FreshMixVocals = FreshMixVocals.BALANCED
    energy: FreshMixEnergy = FreshMixEnergy.STEADY
    max_tracks_per_artist: int = 2
    allow_variants: bool = False
    explicit: bool = False
    history_policy: str = "ignore"


@dataclass(frozen=True, slots=True)
class FreshMixTrack:
    candidate_id: str
    reference: str
    title: str
    artist: str
    duration_seconds: float
    thumbnail_url: str | None
    phase: str
    verified_by: str = "yt_dlp_search"


@dataclass(frozen=True, slots=True)
class FreshMixDraft:
    draft_id: str
    workspace_id: str
    actor_id: str
    brief: FreshMixBrief
    tracks: tuple[FreshMixTrack, ...]
    duration_seconds: float
    checks: tuple[str, ...]
    created_at: float


class FreshMixService:
    """Produce real-provider-backed drafts without reading personal history."""

    def __init__(
        self,
        media: MediaService,
        *,
        draft_ttl_seconds: float = 1800.0,
        max_drafts: int = 200,
    ) -> None:
        if draft_ttl_seconds <= 0:
            raise ValueError("draft_ttl_seconds must be positive")
        if max_drafts < 1:
            raise ValueError("max_drafts must be positive")
        self.media = media
        self.draft_ttl_seconds = draft_ttl_seconds
        self.max_drafts = max_drafts
        self._drafts: dict[str, FreshMixDraft] = {}
        self._claimed: set[str] = set()
        self._lock = asyncio.Lock()

    async def plan(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        brief: FreshMixBrief,
        excluded_references: tuple[str, ...] = (),
    ) -> FreshMixDraft:
        normalized = _validate_brief(brief)
        phases = normalized.phases or _default_phases(normalized)
        selected: list[FreshMixTrack] = []
        seen_references = set(excluded_references)
        seen_titles: set[str] = set()
        artist_counts: dict[str, int] = {}

        for phase_index, phase in enumerate(phases):
            pools = tuple(
                await asyncio.gather(
                    *(self.media.search_audio(query, limit=20) for query in phase.queries)
                )
            )
            candidates = _round_robin(pools)
            phase_target = phase.minutes * 60
            phase_duration = 0.0
            for candidate in candidates:
                track = _validated_track(
                    candidate,
                    phase=phase.label or f"Phase {phase_index + 1}",
                    allow_variants=normalized.allow_variants,
                    allow_instrumental=normalized.vocals is FreshMixVocals.LOW,
                    explicit=normalized.explicit,
                )
                if track is None or track.reference in seen_references:
                    continue
                normalized_title = _normalized_title(track.title)
                if normalized_title in seen_titles:
                    continue
                artist_key = _normalized_artist(track.artist)
                if artist_counts.get(artist_key, 0) >= normalized.max_tracks_per_artist:
                    continue
                if selected and _normalized_artist(selected[-1].artist) == artist_key:
                    continue
                if (
                    phase_index == 0
                    and normalized.energy is FreshMixEnergy.RISING
                    and _looks_high_energy(track.title)
                ):
                    continue
                selected.append(track)
                seen_references.add(track.reference)
                seen_titles.add(normalized_title)
                artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
                phase_duration += track.duration_seconds
                if phase_duration >= phase_target or len(selected) >= 20:
                    break

        if not selected:
            raise UserError("audio.fresh_mix_no_candidates")
        target_seconds = normalized.target_minutes * 60
        trimmed = _trim_to_target(selected, target_seconds)
        duration = sum(track.duration_seconds for track in trimmed)
        checks = (
            "history_off",
            "provider_candidates_verified",
            "duplicate_tracks_removed",
            "artist_limit_checked",
            "variant_policy_checked",
            (
                "duration_within_5_minutes"
                if abs(duration - target_seconds) <= 300
                else "duration_outside_5_minutes"
            ),
        )
        draft = FreshMixDraft(
            draft_id=uuid.uuid4().hex,
            workspace_id=workspace_id,
            actor_id=actor_id,
            brief=normalized,
            tracks=tuple(trimmed),
            duration_seconds=duration,
            checks=checks,
            created_at=time.time(),
        )
        async with self._lock:
            self._purge_expired_locked()
            self._store_locked(draft)
        return draft

    async def revise_track(
        self,
        *,
        draft_id: str,
        workspace_id: str,
        actor_id: str,
        position: int,
        query: str,
    ) -> FreshMixDraft:
        draft = await self.claim(draft_id, workspace_id, actor_id)
        try:
            if not 1 <= position <= len(draft.tracks):
                raise UserError("audio.fresh_mix_position_invalid")
            candidates = await self.media.search_audio(query, limit=10)
            other_tracks = tuple(
                track for index, track in enumerate(draft.tracks, start=1) if index != position
            )
            references = {track.reference for track in other_tracks}
            titles = {_normalized_title(track.title) for track in other_tracks}
            artist_counts: dict[str, int] = {}
            for track in other_tracks:
                key = _normalized_artist(track.artist)
                artist_counts[key] = artist_counts.get(key, 0) + 1
            replacement_track: FreshMixTrack | None = None
            for candidate in candidates:
                proposed = _validated_track(
                    candidate,
                    phase=draft.tracks[position - 1].phase,
                    allow_variants=draft.brief.allow_variants,
                    allow_instrumental=draft.brief.vocals is FreshMixVocals.LOW,
                    explicit=draft.brief.explicit,
                )
                if proposed is None or proposed.reference in references:
                    continue
                if _normalized_title(proposed.title) in titles:
                    continue
                artist_key = _normalized_artist(proposed.artist)
                if artist_counts.get(artist_key, 0) >= draft.brief.max_tracks_per_artist:
                    continue
                replacement_track = proposed
                break
            if replacement_track is None:
                raise UserError("audio.fresh_mix_no_replacement")
            tracks = list(draft.tracks)
            tracks[position - 1] = replacement_track
            revised = replace(
                draft,
                draft_id=uuid.uuid4().hex,
                tracks=tuple(tracks),
                duration_seconds=sum(track.duration_seconds for track in tracks),
                created_at=time.time(),
            )
            async with self._lock:
                self._purge_expired_locked()
                self._drafts.pop(draft.draft_id, None)
                self._claimed.discard(draft.draft_id)
                self._store_locked(revised)
            return revised
        except BaseException:
            await self.release(draft.draft_id)
            raise

    async def require(
        self,
        draft_id: str,
        workspace_id: str,
        actor_id: str,
    ) -> FreshMixDraft:
        async with self._lock:
            self._purge_expired_locked()
            draft = self._drafts.get(draft_id)
            if draft is None or draft.workspace_id != workspace_id or draft.actor_id != actor_id:
                raise UserError("audio.fresh_mix_draft_not_found")
            return draft

    async def claim(
        self,
        draft_id: str,
        workspace_id: str,
        actor_id: str,
    ) -> FreshMixDraft:
        """Reserve one draft for a single revision or enqueue operation."""

        async with self._lock:
            self._purge_expired_locked()
            draft = self._drafts.get(draft_id)
            if draft is None or draft.workspace_id != workspace_id or draft.actor_id != actor_id:
                raise UserError("audio.fresh_mix_draft_not_found")
            if draft_id in self._claimed:
                raise UserError("audio.fresh_mix_draft_busy")
            self._claimed.add(draft_id)
            return draft

    async def release(self, draft_id: str) -> None:
        async with self._lock:
            self._claimed.discard(draft_id)

    async def consume(self, draft_id: str) -> None:
        async with self._lock:
            self._drafts.pop(draft_id, None)
            self._claimed.discard(draft_id)

    def _purge_expired_locked(self) -> None:
        cutoff = time.time() - self.draft_ttl_seconds
        self._drafts = {
            draft_id: draft
            for draft_id, draft in self._drafts.items()
            if draft.created_at >= cutoff
        }
        self._claimed.intersection_update(self._drafts)

    def _store_locked(self, draft: FreshMixDraft) -> None:
        while len(self._drafts) >= self.max_drafts:
            evictable = (
                stored for stored in self._drafts.values() if stored.draft_id not in self._claimed
            )
            oldest = min(evictable, key=lambda item: item.created_at, default=None)
            if oldest is None:
                raise UserError("audio.fresh_mix_draft_capacity")
            self._drafts.pop(oldest.draft_id, None)
        self._drafts[draft.draft_id] = draft


def _validate_brief(brief: FreshMixBrief) -> FreshMixBrief:
    prompt = unicodedata.normalize("NFKC", " ".join(brief.prompt.split())).strip()
    if not prompt:
        raise UserError("audio.fresh_mix_prompt_required")
    if not 15 <= brief.target_minutes <= 240:
        raise UserError("audio.fresh_mix_duration_invalid")
    if not 1 <= brief.max_tracks_per_artist <= 4:
        raise UserError("audio.fresh_mix_artist_limit_invalid")
    if brief.history_policy != "ignore":
        raise UserError("audio.fresh_mix_history_must_be_off")
    if len(brief.phases) > 6:
        raise UserError("audio.fresh_mix_phase_limit")
    if sum(len(phase.queries) for phase in brief.phases) > 8:
        raise UserError("audio.fresh_mix_query_limit")
    for phase in brief.phases:
        if phase.minutes < 1 or not 1 <= len(phase.queries) <= 4:
            raise UserError("audio.fresh_mix_phase_invalid")
        if any(not query.strip() or len(query) > 150 for query in phase.queries):
            raise UserError("audio.fresh_mix_query_invalid")
    if brief.phases and sum(phase.minutes for phase in brief.phases) != brief.target_minutes:
        raise UserError("audio.fresh_mix_phase_duration_mismatch")
    return replace(brief, prompt=prompt)


def _default_phases(brief: FreshMixBrief) -> tuple[FreshMixPhase, ...]:
    vocals = " instrumental" if brief.vocals is FreshMixVocals.LOW else ""
    if brief.energy is FreshMixEnergy.RISING:
        first = max(5, brief.target_minutes // 4)
        last = max(5, brief.target_minutes // 4)
        middle = brief.target_minutes - first - last
        return (
            FreshMixPhase(first, (f"{brief.prompt} calm{vocals}",), "Calm start"),
            FreshMixPhase(middle, (f"{brief.prompt} focus steady{vocals}",), "Focus"),
            FreshMixPhase(last, (f"{brief.prompt} uplifting{vocals}",), "Lift"),
        )
    return (
        FreshMixPhase(
            brief.target_minutes,
            (f"{brief.prompt}{vocals}",),
            "Main",
        ),
    )


def _round_robin(
    pools: tuple[tuple[MediaCandidate, ...], ...],
) -> tuple[MediaCandidate, ...]:
    maximum = max((len(pool) for pool in pools), default=0)
    return tuple(pool[index] for index in range(maximum) for pool in pools if index < len(pool))


def _validated_track(
    candidate: MediaCandidate,
    *,
    phase: str,
    allow_variants: bool,
    allow_instrumental: bool,
    explicit: bool,
) -> FreshMixTrack | None:
    title = " ".join(candidate.title.split()).strip()
    artist = " ".join((candidate.uploader or "Unknown artist").split()).strip()
    combined = f"{title} {candidate.reference}"
    if (
        not title
        or not candidate.reference.startswith("https://")
        or not 60 <= candidate.duration_seconds <= 1200
        or _LONG_FORM_TERMS.search(title)
        or _SHORTS_TERMS.search(combined)
        or (not allow_variants and _VARIANT_TERMS.search(title))
        or (not allow_instrumental and _INSTRUMENTAL_TERM.search(title))
        or (not explicit and _looks_explicit(title))
    ):
        return None
    return FreshMixTrack(
        candidate_id=uuid.uuid5(
            uuid.NAMESPACE_URL,
            candidate.reference,
        ).hex,
        reference=candidate.reference,
        title=title,
        artist=artist,
        duration_seconds=candidate.duration_seconds,
        thumbnail_url=candidate.thumbnail_url,
        phase=phase,
    )


def _normalized_title(title: str) -> str:
    value = unicodedata.normalize("NFKC", title).casefold()
    value = _TITLE_NOISE.sub(" ", value)
    return re.sub(r"[\W_]+", "", value)


def _normalized_artist(artist: str) -> str:
    return re.sub(
        r"[\W_]+",
        "",
        unicodedata.normalize("NFKC", artist).casefold(),
    )


def _looks_high_energy(title: str) -> bool:
    return bool(re.search(r"\b(hardcore|intense|workout|festival|edm)\b", title, re.I))


def _looks_explicit(title: str) -> bool:
    return bool(re.search(r"\b(explicit|uncensored)\b", title, re.I))


def _trim_to_target(
    tracks: list[FreshMixTrack],
    target_seconds: int,
) -> list[FreshMixTrack]:
    while len(tracks) > 1:
        current = sum(track.duration_seconds for track in tracks)
        without_last = current - tracks[-1].duration_seconds
        if abs(without_last - target_seconds) >= abs(current - target_seconds):
            break
        tracks.pop()
    return tracks
