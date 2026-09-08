"""Persistent read-aloud routing independent of chat transport."""

from __future__ import annotations

import asyncio
import json
import math
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from simajilord.core.errors import ConfigurationError, UserError

DEFAULT_READ_ALOUD_MESSAGE_CHARACTER_LIMIT = 120
MIN_READ_ALOUD_MESSAGE_CHARACTER_LIMIT = 20
MAX_READ_ALOUD_MESSAGE_CHARACTER_LIMIT = 400
MIN_READ_ALOUD_SPEED_SCALE = 0.5
MAX_READ_ALOUD_SPEED_SCALE = 2.0
MIN_READ_ALOUD_PITCH_SCALE = -0.15
MAX_READ_ALOUD_PITCH_SCALE = 0.15


class ReadAloudMode(StrEnum):
    """How speech behaves while music is already playing."""

    QUEUE = "queue"
    SKIP_DURING_MUSIC = "skip_during_music"


class ReadAloudContentMode(StrEnum):
    """Which durable input classes may enter a guild speech queue."""

    ALL = "all"
    MESSAGES = "messages"
    EVENTS = "events"
    OFF = "off"


class ReadAloudVoicePreset(StrEnum):
    """Small, stable voice vocabulary exposed instead of raw style IDs."""

    CLEAR = "clear"
    CALM = "calm"
    ENERGETIC = "energetic"
    CUTE = "cute"
    NARRATOR = "narrator"


@dataclass(frozen=True, slots=True)
class ReadAloudRoute:
    workspace_id: str
    text_channel_id: str
    audio_destination_id: str
    mode: ReadAloudMode
    enabled: bool = True
    additional_text_channel_ids: tuple[str, ...] = ()

    @property
    def text_channel_ids(self) -> tuple[str, ...]:
        """Return the stable, de-duplicated set of message sources."""

        return tuple(dict.fromkeys((self.text_channel_id, *self.additional_text_channel_ids)))


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryEntry:
    """One guild-scoped, literal speech replacement."""

    surface: str
    reading: str


@dataclass(frozen=True, slots=True)
class ReadAloudVoiceTuning:
    """One member's provider-neutral VOICEVOX tuning values."""

    user_id: str
    speed_scale: float = 1.0
    pitch_scale: float = 0.0


@dataclass(frozen=True, slots=True)
class ReadAloudPolicy:
    """Durable read-aloud behavior that survives route removal."""

    workspace_id: str
    dictionary: tuple[ReadAloudDictionaryEntry, ...] = ()
    ignored_user_ids: tuple[str, ...] = ()
    ignored_role_ids: tuple[str, ...] = ()
    announce_join: bool = False
    announce_leave: bool = False
    announce_move: bool = False
    read_messages: bool = True
    read_author_names: bool = True
    read_replies: bool = True
    read_attachments: bool = True
    vc_members_only: bool = False
    abbreviate_long_messages: bool = False
    message_character_limit: int = DEFAULT_READ_ALOUD_MESSAGE_CHARACTER_LIMIT
    default_voice_preset: ReadAloudVoicePreset = ReadAloudVoicePreset.CLEAR
    user_voice_presets: tuple[tuple[str, ReadAloudVoicePreset], ...] = ()
    user_voice_tunings: tuple[ReadAloudVoiceTuning, ...] = ()
    ignore_bots: bool = True
    ignore_webhooks: bool = True


def read_aloud_content_mode(policy: ReadAloudPolicy) -> ReadAloudContentMode:
    """Project the four content switches into their public preset."""

    read_events = policy.announce_join or policy.announce_leave or policy.announce_move
    if policy.read_messages and read_events:
        return ReadAloudContentMode.ALL
    if policy.read_messages:
        return ReadAloudContentMode.MESSAGES
    if read_events:
        return ReadAloudContentMode.EVENTS
    return ReadAloudContentMode.OFF


class ReadAloudService:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._routes: dict[str, ReadAloudRoute] = {}
        self._policies: dict[str, ReadAloudPolicy] = {}
        self._route_profiles: dict[tuple[str, str], ReadAloudRoute] = {}
        self._lock = asyncio.Lock()
        self._load()

    def get(self, workspace_id: str) -> ReadAloudRoute | None:
        route = self._routes.get(workspace_id)
        return route if route and route.enabled else None

    def saved_route(self, workspace_id: str, destination_id: str) -> ReadAloudRoute | None:
        current = self.get(workspace_id)
        if current is not None and current.audio_destination_id == destination_id:
            return current
        return self._route_profiles.get((workspace_id, destination_id))

    def resume_route(self, workspace_id: str, destination_id: str) -> ReadAloudRoute | None:
        """Preview the same saved route that an explicit start or move will activate."""

        saved = self.saved_route(workspace_id, destination_id)
        if saved is not None:
            return saved
        current = self.get(workspace_id)
        if current is None:
            return None
        source_ids = tuple(
            dict.fromkeys(
                destination_id if value == current.audio_destination_id else value
                for value in current.text_channel_ids
            )
        )
        return replace(
            current,
            text_channel_id=source_ids[0],
            additional_text_channel_ids=source_ids[1:],
            audio_destination_id=destination_id,
        )

    def policy(self, workspace_id: str) -> ReadAloudPolicy:
        """Return a stored policy or a non-persisted default policy."""

        return self._policies.get(workspace_id, ReadAloudPolicy(workspace_id))

    async def configure(
        self,
        route: ReadAloudRoute,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        async with self._lock:
            if self.get(route.workspace_id) == route:
                if on_noop is not None:
                    await on_noop()
                return
            if before_mutation is not None:
                await before_mutation()
            self._routes[route.workspace_id] = route
            await asyncio.to_thread(self._save)

    async def add_sources(
        self,
        *,
        workspace_id: str,
        text_channel_ids: tuple[str, ...],
        audio_destination_id: str,
        mode: ReadAloudMode,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudRoute:
        """Atomically add a source set without discarding existing routes."""

        channel_ids = tuple(dict.fromkeys(text_channel_ids))
        if not channel_ids:
            raise ValueError("read_aloud.source_channels_required")
        async with self._lock:
            current = self.get(workspace_id)
            if current is None:
                combined_ids = channel_ids
                route_mode = mode
            else:
                if current.audio_destination_id != audio_destination_id:
                    raise ValueError("read_aloud.destination_conflict")
                combined_ids = tuple(dict.fromkeys((*current.text_channel_ids, *channel_ids)))
                route_mode = current.mode
            route = ReadAloudRoute(
                workspace_id=workspace_id,
                text_channel_id=combined_ids[0],
                audio_destination_id=audio_destination_id,
                mode=route_mode,
                additional_text_channel_ids=combined_ids[1:],
            )
            if current == route:
                if on_noop is not None:
                    await on_noop()
                return route
            if before_mutation is not None:
                await before_mutation()
            self._routes[workspace_id] = route
            await asyncio.to_thread(self._save)
            return route

    async def configure_sources(
        self,
        *,
        workspace_id: str,
        text_channel_ids: tuple[str, ...],
        audio_destination_id: str,
        mode: ReadAloudMode,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudRoute:
        """Atomically replace the complete source set and voice destination."""

        channel_ids = tuple(dict.fromkeys(text_channel_ids))
        if not channel_ids:
            raise ValueError("read_aloud.source_channels_required")
        route = ReadAloudRoute(
            workspace_id=workspace_id,
            text_channel_id=channel_ids[0],
            audio_destination_id=audio_destination_id,
            mode=mode,
            additional_text_channel_ids=channel_ids[1:],
        )
        await self.configure(
            route,
            before_mutation=before_mutation,
            on_noop=on_noop,
        )
        return route

    async def add_source(
        self,
        *,
        workspace_id: str,
        text_channel_id: str,
        audio_destination_id: str,
        mode: ReadAloudMode,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudRoute:
        """Add one source without discarding existing sources for the same VC."""

        return await self.add_sources(
            workspace_id=workspace_id,
            text_channel_ids=(text_channel_id,),
            audio_destination_id=audio_destination_id,
            mode=mode,
            before_mutation=before_mutation,
            on_noop=on_noop,
        )

    async def remove_source(
        self,
        *,
        workspace_id: str,
        text_channel_id: str,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudRoute | None:
        """Remove one source; deleting the last source disables the route."""

        async with self._lock:
            current = self.get(workspace_id)
            if current is None or text_channel_id not in current.text_channel_ids:
                if on_noop is not None:
                    await on_noop()
                return current
            channel_ids = tuple(
                item for item in current.text_channel_ids if item != text_channel_id
            )
            if not channel_ids:
                if before_mutation is not None:
                    await before_mutation()
                self._routes.pop(workspace_id, None)
                await asyncio.to_thread(self._save)
                return None
            route = ReadAloudRoute(
                workspace_id=workspace_id,
                text_channel_id=channel_ids[0],
                audio_destination_id=current.audio_destination_id,
                mode=current.mode,
                additional_text_channel_ids=channel_ids[1:],
            )
            if before_mutation is not None:
                await before_mutation()
            self._routes[workspace_id] = route
            await asyncio.to_thread(self._save)
            return route

    async def disable(
        self,
        workspace_id: str,
        *,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        async with self._lock:
            if workspace_id not in self._routes:
                if on_noop is not None:
                    await on_noop()
                return False
            if before_mutation is not None:
                await before_mutation()
            self._routes.pop(workspace_id, None)
            await asyncio.to_thread(self._save)
            return True

    def matches(self, workspace_id: str, text_channel_id: str) -> bool:
        route = self.get(workspace_id)
        return route is not None and text_channel_id in route.text_channel_ids

    async def upsert_dictionary_entry(
        self,
        *,
        workspace_id: str,
        surface: str,
        reading: str,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Add or replace a literal dictionary entry for one guild."""

        normalized_surface = self._dictionary_value(
            surface,
            field="surface",
            maximum=100,
        )
        normalized_reading = self._dictionary_value(
            reading,
            field="reading",
            maximum=200,
        )
        async with self._lock:
            current = self.policy(workspace_id)
            entries = tuple(
                entry for entry in current.dictionary if entry.surface != normalized_surface
            )
            updated = replace(
                current,
                dictionary=(
                    *entries,
                    ReadAloudDictionaryEntry(
                        surface=normalized_surface,
                        reading=normalized_reading,
                    ),
                ),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def remove_dictionary_entry(
        self,
        *,
        workspace_id: str,
        surface: str,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[ReadAloudPolicy, bool]:
        """Remove an exact dictionary surface and report whether it existed."""

        normalized_surface = self._dictionary_value(
            surface,
            field="surface",
            maximum=100,
        )
        async with self._lock:
            current = self.policy(workspace_id)
            entries = tuple(
                entry for entry in current.dictionary if entry.surface != normalized_surface
            )
            removed = len(entries) != len(current.dictionary)
            if not removed:
                if on_noop is not None:
                    await on_noop()
                return current, False
            updated = replace(current, dictionary=entries)
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, True

    async def set_user_ignored(
        self,
        *,
        workspace_id: str,
        user_id: str,
        ignored: bool,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Set whether messages from one user may enter the speech queue."""

        normalized_id = self._required_identifier(user_id, field="user_id")
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                ignored_user_ids=self._updated_identifier_set(
                    current.ignored_user_ids,
                    normalized_id,
                    enabled=ignored,
                ),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_role_ignored(
        self,
        *,
        workspace_id: str,
        role_id: str,
        ignored: bool,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Set whether members with one role may enter the speech queue."""

        normalized_id = self._required_identifier(role_id, field="role_id")
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                ignored_role_ids=self._updated_identifier_set(
                    current.ignored_role_ids,
                    normalized_id,
                    enabled=ignored,
                ),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_announcements(
        self,
        *,
        workspace_id: str,
        join: bool | None = None,
        leave: bool | None = None,
        move: bool | None = None,
    ) -> ReadAloudPolicy:
        """Update selected voice-state announcement switches."""

        updated, _ = await self.set_announcements_with_previous(
            workspace_id=workspace_id,
            join=join,
            leave=leave,
            move=move,
        )
        return updated

    async def set_announcements_with_previous(
        self,
        *,
        workspace_id: str,
        join: bool | None = None,
        leave: bool | None = None,
        move: bool | None = None,
        expected_join: bool | None = None,
        expected_leave: bool | None = None,
        expected_move: bool | None = None,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[ReadAloudPolicy, ReadAloudPolicy]:
        """Update announcement switches and atomically return the prior policy."""

        if join is None and leave is None and move is None:
            raise ValueError("read_aloud.announcement_value_required")
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                announce_join=current.announce_join if join is None else join,
                announce_leave=current.announce_leave if leave is None else leave,
                announce_move=current.announce_move if move is None else move,
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current, current
            if (
                (expected_join is not None and current.announce_join != expected_join)
                or (expected_leave is not None and current.announce_leave != expected_leave)
                or (expected_move is not None and current.announce_move != expected_move)
            ):
                raise UserError("action.undo_conflict")
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, current

    async def set_content_mode(
        self,
        *,
        workspace_id: str,
        mode: ReadAloudContentMode,
    ) -> ReadAloudPolicy:
        """Apply one explicit messages/events preset without deleting the route."""

        updated, _ = await self.set_content_mode_with_previous(
            workspace_id=workspace_id,
            mode=mode,
        )
        return updated

    async def set_content_mode_with_previous(
        self,
        *,
        workspace_id: str,
        mode: ReadAloudContentMode,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[ReadAloudPolicy, ReadAloudPolicy]:
        """Apply a content preset and atomically return the prior scalar policy."""

        normalized_mode = ReadAloudContentMode(mode)
        read_messages = normalized_mode in {
            ReadAloudContentMode.ALL,
            ReadAloudContentMode.MESSAGES,
        }
        read_events = normalized_mode in {
            ReadAloudContentMode.ALL,
            ReadAloudContentMode.EVENTS,
        }
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                read_messages=read_messages,
                announce_join=read_events,
                announce_leave=read_events,
                announce_move=read_events,
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current, current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, current

    async def compare_and_set_content_mode(
        self,
        *,
        workspace_id: str,
        expected: ReadAloudContentMode,
        mode: ReadAloudContentMode,
    ) -> tuple[ReadAloudPolicy, bool]:
        """Apply a preset only while the current preset still matches."""

        normalized_expected = ReadAloudContentMode(expected)
        normalized_mode = ReadAloudContentMode(mode)
        read_messages = normalized_mode in {
            ReadAloudContentMode.ALL,
            ReadAloudContentMode.MESSAGES,
        }
        read_events = normalized_mode in {
            ReadAloudContentMode.ALL,
            ReadAloudContentMode.EVENTS,
        }
        async with self._lock:
            current = self.policy(workspace_id)
            if read_aloud_content_mode(current) != normalized_expected:
                return current, False
            updated = replace(
                current,
                read_messages=read_messages,
                announce_join=read_events,
                announce_leave=read_events,
                announce_move=read_events,
            )
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, True

    async def restore_content_state(
        self,
        *,
        workspace_id: str,
        read_messages: bool,
        announce_join: bool,
        announce_leave: bool,
        announce_move: bool,
        expected_read_messages: bool | None = None,
        expected_announce_join: bool | None = None,
        expected_announce_leave: bool | None = None,
        expected_announce_move: bool | None = None,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Restore the four exact booleans collapsed by a content-mode preset."""

        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                read_messages=read_messages,
                announce_join=announce_join,
                announce_leave=announce_leave,
                announce_move=announce_move,
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if (
                (
                    expected_read_messages is not None
                    and current.read_messages != expected_read_messages
                )
                or (
                    expected_announce_join is not None
                    and current.announce_join != expected_announce_join
                )
                or (
                    expected_announce_leave is not None
                    and current.announce_leave != expected_announce_leave
                )
                or (
                    expected_announce_move is not None
                    and current.announce_move != expected_announce_move
                )
            ):
                raise UserError("action.undo_conflict")
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_default_voice_preset(
        self,
        *,
        workspace_id: str,
        preset: ReadAloudVoicePreset,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Set the server-wide voice used unless a member overrides it."""

        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(current, default_voice_preset=preset)
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_user_voice_preset(
        self,
        *,
        workspace_id: str,
        user_id: str,
        preset: ReadAloudVoicePreset | None,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Set or clear one member's self-managed voice override."""

        normalized_id = self._required_identifier(user_id, field="user_id")
        async with self._lock:
            current = self.policy(workspace_id)
            overrides = dict(current.user_voice_presets)
            if preset is None:
                overrides.pop(normalized_id, None)
            else:
                overrides[normalized_id] = preset
            updated = replace(
                current,
                user_voice_presets=tuple(sorted(overrides.items())),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_user_voice_tuning(
        self,
        *,
        workspace_id: str,
        user_id: str,
        speed_scale: float,
        pitch_scale: float,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> ReadAloudPolicy:
        """Set one member's speech speed and pitch without changing their voice."""

        normalized_id = self._required_identifier(user_id, field="user_id")
        normalized_speed = self._voice_tuning_value(
            speed_scale,
            field="speed_scale",
            minimum=MIN_READ_ALOUD_SPEED_SCALE,
            maximum=MAX_READ_ALOUD_SPEED_SCALE,
        )
        normalized_pitch = self._voice_tuning_value(
            pitch_scale,
            field="pitch_scale",
            minimum=MIN_READ_ALOUD_PITCH_SCALE,
            maximum=MAX_READ_ALOUD_PITCH_SCALE,
        )
        async with self._lock:
            current = self.policy(workspace_id)
            tunings = {tuning.user_id: tuning for tuning in current.user_voice_tunings}
            if normalized_speed == 1.0 and normalized_pitch == 0.0:
                tunings.pop(normalized_id, None)
            else:
                tunings[normalized_id] = ReadAloudVoiceTuning(
                    user_id=normalized_id,
                    speed_scale=normalized_speed,
                    pitch_scale=normalized_pitch,
                )
            updated = replace(
                current,
                user_voice_tunings=tuple(tunings[user_id] for user_id in sorted(tunings)),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    def voice_preset_for(
        self,
        *,
        workspace_id: str,
        user_id: str,
    ) -> ReadAloudVoicePreset:
        """Resolve a member override with the server default as fallback."""

        policy = self.policy(workspace_id)
        return dict(policy.user_voice_presets).get(
            user_id,
            policy.default_voice_preset,
        )

    def voice_tuning_for(
        self,
        *,
        workspace_id: str,
        user_id: str,
    ) -> ReadAloudVoiceTuning:
        """Resolve one member's tuning, falling back to the natural defaults."""

        for tuning in self.policy(workspace_id).user_voice_tunings:
            if tuning.user_id == user_id:
                return tuning
        return ReadAloudVoiceTuning(user_id=user_id)

    async def set_semantic_options(
        self,
        *,
        workspace_id: str,
        author_names: bool | None = None,
        replies: bool | None = None,
        attachments: bool | None = None,
        vc_members_only: bool | None = None,
        abbreviate_long_messages: bool | None = None,
        message_character_limit: int | None = None,
    ) -> ReadAloudPolicy:
        """Update selected semantic speech formatting switches."""

        updated, _ = await self.set_semantic_options_with_previous(
            workspace_id=workspace_id,
            author_names=author_names,
            replies=replies,
            attachments=attachments,
            vc_members_only=vc_members_only,
            abbreviate_long_messages=abbreviate_long_messages,
            message_character_limit=message_character_limit,
        )
        return updated

    async def set_semantic_options_with_previous(
        self,
        *,
        workspace_id: str,
        author_names: bool | None = None,
        replies: bool | None = None,
        attachments: bool | None = None,
        vc_members_only: bool | None = None,
        abbreviate_long_messages: bool | None = None,
        message_character_limit: int | None = None,
        expected_author_names: bool | None = None,
        expected_replies: bool | None = None,
        expected_attachments: bool | None = None,
        expected_vc_members_only: bool | None = None,
        expected_abbreviate_long_messages: bool | None = None,
        expected_message_character_limit: int | None = None,
        before_mutation: Callable[[], Awaitable[None]] | None = None,
        on_noop: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[ReadAloudPolicy, ReadAloudPolicy]:
        """Update semantic options and atomically return the prior policy."""

        if (
            author_names is None
            and replies is None
            and attachments is None
            and vc_members_only is None
            and abbreviate_long_messages is None
            and message_character_limit is None
        ):
            raise ValueError("read_aloud.semantic_value_required")
        normalized_message_character_limit = (
            None
            if message_character_limit is None
            else self._message_character_limit(message_character_limit)
        )
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                read_author_names=(
                    current.read_author_names if author_names is None else author_names
                ),
                read_replies=current.read_replies if replies is None else replies,
                read_attachments=(current.read_attachments if attachments is None else attachments),
                vc_members_only=(
                    current.vc_members_only if vc_members_only is None else vc_members_only
                ),
                abbreviate_long_messages=(
                    current.abbreviate_long_messages
                    if abbreviate_long_messages is None
                    else abbreviate_long_messages
                ),
                message_character_limit=(
                    current.message_character_limit
                    if normalized_message_character_limit is None
                    else normalized_message_character_limit
                ),
            )
            if updated == current:
                if on_noop is not None:
                    await on_noop()
                return current, current
            if (
                (
                    expected_author_names is not None
                    and current.read_author_names != expected_author_names
                )
                or (expected_replies is not None and current.read_replies != expected_replies)
                or (
                    expected_attachments is not None
                    and current.read_attachments != expected_attachments
                )
                or (
                    expected_vc_members_only is not None
                    and current.vc_members_only != expected_vc_members_only
                )
                or (
                    expected_abbreviate_long_messages is not None
                    and current.abbreviate_long_messages != expected_abbreviate_long_messages
                )
                or (
                    expected_message_character_limit is not None
                    and current.message_character_limit
                    != self._message_character_limit(expected_message_character_limit)
                )
            ):
                raise UserError("action.undo_conflict")
            if before_mutation is not None:
                await before_mutation()
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, current

    def allows_message(
        self,
        *,
        workspace_id: str,
        author_id: str,
        role_ids: tuple[str, ...] = (),
        is_bot: bool = False,
        is_webhook: bool = False,
    ) -> bool:
        """Reject excluded authors before text synthesis consumes resources."""

        policy = self.policy(workspace_id)
        if not policy.read_messages:
            return False
        if policy.ignore_bots and is_bot:
            return False
        if policy.ignore_webhooks and is_webhook:
            return False
        if author_id in policy.ignored_user_ids:
            return False
        return not any(role_id in policy.ignored_role_ids for role_id in role_ids)

    def apply_dictionary(self, workspace_id: str, text: str) -> str:
        """Apply safe literal replacements, preferring longer surfaces."""

        value = unicodedata.normalize("NFKC", text)
        entries = sorted(
            self.policy(workspace_id).dictionary,
            key=lambda entry: (-len(entry.surface), entry.surface),
        )
        for entry in entries:
            value = value.replace(entry.surface, entry.reading)
        return value

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw: Any = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                route_items = raw
                policy_items: list[Any] = []
            elif isinstance(raw, dict):
                if int(raw.get("version", 0)) != 2:
                    raise ValueError("unsupported state version")
                route_items = raw.get("routes", [])
                policy_items = raw.get("policies", [])
                if not isinstance(route_items, list) or not isinstance(policy_items, list):
                    raise ValueError("routes and policies must be lists")
            else:
                raise ValueError("expected a list or versioned object")
            profile_items = raw.get("route_profiles", []) if isinstance(raw, dict) else []
            if not isinstance(profile_items, list):
                raise ValueError("route_profiles must be a list")
            for is_current, item in [
                *((False, item) for item in profile_items),
                *((True, item) for item in route_items),
            ]:
                if not isinstance(item, dict):
                    raise ValueError("route must be an object")
                route = ReadAloudRoute(
                    workspace_id=str(item["workspace_id"]),
                    text_channel_id=str(item["text_channel_id"]),
                    audio_destination_id=str(item["audio_destination_id"]),
                    mode=ReadAloudMode(str(item["mode"])),
                    enabled=bool(item.get("enabled", True)),
                    additional_text_channel_ids=tuple(
                        str(value) for value in item.get("additional_text_channel_ids", ())
                    ),
                )
                if route.enabled:
                    self._route_profiles[(route.workspace_id, route.audio_destination_id)] = route
                if is_current:
                    self._routes[route.workspace_id] = route
            for item in policy_items:
                if not isinstance(item, dict):
                    raise ValueError("policy must be an object")
                policy = self._policy_from_json(item)
                self._policies[policy.workspace_id] = policy
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Invalid read-aloud state: {self.state_file}") from exc

    def _save(self) -> None:
        self.state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        for route in self._routes.values():
            if route.enabled:
                self._route_profiles[(route.workspace_id, route.audio_destination_id)] = route
        payload = {
            "version": 2,
            "route_profiles": [
                {**asdict(route), "mode": route.mode.value}
                for route in self._route_profiles.values()
            ],
            "routes": [
                {
                    **asdict(route),
                    "mode": route.mode.value,
                    "additional_text_channel_ids": list(route.additional_text_channel_ids),
                }
                for route in sorted(
                    self._routes.values(),
                    key=lambda value: value.workspace_id,
                )
            ],
            "policies": [
                self._policy_to_json(policy)
                for policy in sorted(
                    self._policies.values(),
                    key=lambda value: value.workspace_id,
                )
            ],
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self.state_file)
        self.state_file.chmod(0o600)

    @staticmethod
    def _dictionary_value(value: str, *, field: str, maximum: int) -> str:
        normalized = unicodedata.normalize("NFKC", " ".join(value.split()))
        if not normalized:
            raise ValueError(f"read_aloud.dictionary_{field}_required")
        if len(normalized) > maximum:
            raise ValueError(f"read_aloud.dictionary_{field}_too_long")
        return normalized

    @staticmethod
    def _required_identifier(value: str, *, field: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"read_aloud.{field}_required")
        return normalized

    @staticmethod
    def _message_character_limit(value: object) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not MIN_READ_ALOUD_MESSAGE_CHARACTER_LIMIT
            <= value
            <= MAX_READ_ALOUD_MESSAGE_CHARACTER_LIMIT
        ):
            raise ValueError("read_aloud.message_character_limit_invalid")
        return value

    @staticmethod
    def _voice_tuning_value(
        value: object,
        *,
        field: str,
        minimum: float,
        maximum: float,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise ValueError(f"read_aloud.{field}_invalid")
        return round(float(value), 3)

    @staticmethod
    def _updated_identifier_set(
        values: tuple[str, ...],
        value: str,
        *,
        enabled: bool,
    ) -> tuple[str, ...]:
        current = set(values)
        if enabled:
            current.add(value)
        else:
            current.discard(value)
        return tuple(sorted(current))

    @staticmethod
    def _policy_from_json(item: dict[str, Any]) -> ReadAloudPolicy:
        dictionary_raw = item.get("dictionary", [])
        if not isinstance(dictionary_raw, list):
            raise ValueError("dictionary must be a list")
        dictionary: list[ReadAloudDictionaryEntry] = []
        for entry in dictionary_raw:
            if not isinstance(entry, dict):
                raise ValueError("dictionary entry must be an object")
            dictionary.append(
                ReadAloudDictionaryEntry(
                    surface=ReadAloudService._dictionary_value(
                        str(entry["surface"]),
                        field="surface",
                        maximum=100,
                    ),
                    reading=ReadAloudService._dictionary_value(
                        str(entry["reading"]),
                        field="reading",
                        maximum=200,
                    ),
                )
            )
        voice_tunings_raw = item.get("user_voice_tunings", {})
        if not isinstance(voice_tunings_raw, dict):
            raise ValueError("user voice tunings must be an object")
        voice_tunings: list[ReadAloudVoiceTuning] = []
        for user_id, tuning in voice_tunings_raw.items():
            if not isinstance(tuning, dict):
                raise ValueError("user voice tuning must be an object")
            voice_tunings.append(
                ReadAloudVoiceTuning(
                    user_id=ReadAloudService._required_identifier(
                        str(user_id),
                        field="user_id",
                    ),
                    speed_scale=ReadAloudService._voice_tuning_value(
                        tuning.get("speed_scale", 1.0),
                        field="speed_scale",
                        minimum=MIN_READ_ALOUD_SPEED_SCALE,
                        maximum=MAX_READ_ALOUD_SPEED_SCALE,
                    ),
                    pitch_scale=ReadAloudService._voice_tuning_value(
                        tuning.get("pitch_scale", 0.0),
                        field="pitch_scale",
                        minimum=MIN_READ_ALOUD_PITCH_SCALE,
                        maximum=MAX_READ_ALOUD_PITCH_SCALE,
                    ),
                )
            )
        return ReadAloudPolicy(
            workspace_id=str(item["workspace_id"]),
            dictionary=tuple(dictionary),
            ignored_user_ids=tuple(
                sorted({str(value) for value in item.get("ignored_user_ids", ())})
            ),
            ignored_role_ids=tuple(
                sorted({str(value) for value in item.get("ignored_role_ids", ())})
            ),
            announce_join=bool(item.get("announce_join", False)),
            announce_leave=bool(item.get("announce_leave", False)),
            announce_move=bool(item.get("announce_move", False)),
            read_messages=bool(item.get("read_messages", True)),
            read_author_names=bool(item.get("read_author_names", True)),
            read_replies=bool(item.get("read_replies", True)),
            read_attachments=bool(item.get("read_attachments", True)),
            vc_members_only=bool(item.get("vc_members_only", False)),
            abbreviate_long_messages=bool(item.get("abbreviate_long_messages", False)),
            message_character_limit=ReadAloudService._message_character_limit(
                item.get(
                    "message_character_limit",
                    DEFAULT_READ_ALOUD_MESSAGE_CHARACTER_LIMIT,
                )
            ),
            default_voice_preset=ReadAloudVoicePreset(
                str(item.get("default_voice_preset", ReadAloudVoicePreset.CLEAR.value))
            ),
            user_voice_presets=tuple(
                sorted(
                    (
                        ReadAloudService._required_identifier(
                            str(user_id),
                            field="user_id",
                        ),
                        ReadAloudVoicePreset(str(preset)),
                    )
                    for user_id, preset in dict(item.get("user_voice_presets", {})).items()
                )
            ),
            user_voice_tunings=tuple(sorted(voice_tunings, key=lambda tuning: tuning.user_id)),
            ignore_bots=bool(item.get("ignore_bots", True)),
            ignore_webhooks=bool(item.get("ignore_webhooks", True)),
        )

    @staticmethod
    def _policy_to_json(policy: ReadAloudPolicy) -> dict[str, Any]:
        return {
            "workspace_id": policy.workspace_id,
            "dictionary": [asdict(entry) for entry in policy.dictionary],
            "ignored_user_ids": list(policy.ignored_user_ids),
            "ignored_role_ids": list(policy.ignored_role_ids),
            "announce_join": policy.announce_join,
            "announce_leave": policy.announce_leave,
            "announce_move": policy.announce_move,
            "read_messages": policy.read_messages,
            "read_author_names": policy.read_author_names,
            "read_replies": policy.read_replies,
            "read_attachments": policy.read_attachments,
            "vc_members_only": policy.vc_members_only,
            "abbreviate_long_messages": policy.abbreviate_long_messages,
            "message_character_limit": policy.message_character_limit,
            "default_voice_preset": policy.default_voice_preset.value,
            "user_voice_presets": {
                user_id: preset.value for user_id, preset in policy.user_voice_presets
            },
            "user_voice_tunings": {
                tuning.user_id: {
                    "speed_scale": tuning.speed_scale,
                    "pitch_scale": tuning.pitch_scale,
                }
                for tuning in policy.user_voice_tunings
            },
            "ignore_bots": policy.ignore_bots,
            "ignore_webhooks": policy.ignore_webhooks,
        }
