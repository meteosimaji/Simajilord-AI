"""Persistent read-aloud routing independent of chat transport."""

from __future__ import annotations

import asyncio
import json
import unicodedata
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from simajilord.core.errors import ConfigurationError


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

        return tuple(
            dict.fromkeys((self.text_channel_id, *self.additional_text_channel_ids))
        )


@dataclass(frozen=True, slots=True)
class ReadAloudDictionaryEntry:
    """One guild-scoped, literal speech replacement."""

    surface: str
    reading: str


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
    ignore_bots: bool = True
    ignore_webhooks: bool = True


class ReadAloudService:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._routes: dict[str, ReadAloudRoute] = {}
        self._policies: dict[str, ReadAloudPolicy] = {}
        self._lock = asyncio.Lock()
        self._load()

    def get(self, workspace_id: str) -> ReadAloudRoute | None:
        route = self._routes.get(workspace_id)
        return route if route and route.enabled else None

    def policy(self, workspace_id: str) -> ReadAloudPolicy:
        """Return a stored policy or a non-persisted default policy."""

        return self._policies.get(workspace_id, ReadAloudPolicy(workspace_id))

    async def configure(self, route: ReadAloudRoute) -> None:
        async with self._lock:
            self._routes[route.workspace_id] = route
            await asyncio.to_thread(self._save)

    async def add_sources(
        self,
        *,
        workspace_id: str,
        text_channel_ids: tuple[str, ...],
        audio_destination_id: str,
        mode: ReadAloudMode,
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
                combined_ids = tuple(
                    dict.fromkeys((*current.text_channel_ids, *channel_ids))
                )
                route_mode = current.mode
            route = ReadAloudRoute(
                workspace_id=workspace_id,
                text_channel_id=combined_ids[0],
                audio_destination_id=audio_destination_id,
                mode=route_mode,
                additional_text_channel_ids=combined_ids[1:],
            )
            self._routes[workspace_id] = route
            await asyncio.to_thread(self._save)
            return route

    async def add_source(
        self,
        *,
        workspace_id: str,
        text_channel_id: str,
        audio_destination_id: str,
        mode: ReadAloudMode,
    ) -> ReadAloudRoute:
        """Add one source without discarding existing sources for the same VC."""

        return await self.add_sources(
            workspace_id=workspace_id,
            text_channel_ids=(text_channel_id,),
            audio_destination_id=audio_destination_id,
            mode=mode,
        )

    async def remove_source(
        self,
        *,
        workspace_id: str,
        text_channel_id: str,
    ) -> ReadAloudRoute | None:
        """Remove one source; deleting the last source disables the route."""

        async with self._lock:
            current = self.get(workspace_id)
            if current is None or text_channel_id not in current.text_channel_ids:
                return current
            channel_ids = tuple(
                item for item in current.text_channel_ids if item != text_channel_id
            )
            if not channel_ids:
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
            self._routes[workspace_id] = route
            await asyncio.to_thread(self._save)
            return route

    async def disable(self, workspace_id: str) -> bool:
        async with self._lock:
            existed = self._routes.pop(workspace_id, None) is not None
            if existed:
                await asyncio.to_thread(self._save)
            return existed

    def matches(self, workspace_id: str, text_channel_id: str) -> bool:
        route = self.get(workspace_id)
        return route is not None and text_channel_id in route.text_channel_ids

    async def upsert_dictionary_entry(
        self,
        *,
        workspace_id: str,
        surface: str,
        reading: str,
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
                entry
                for entry in current.dictionary
                if entry.surface != normalized_surface
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
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def remove_dictionary_entry(
        self,
        *,
        workspace_id: str,
        surface: str,
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
                entry
                for entry in current.dictionary
                if entry.surface != normalized_surface
            )
            removed = len(entries) != len(current.dictionary)
            if not removed:
                return current, False
            updated = replace(current, dictionary=entries)
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated, True

    async def set_user_ignored(
        self,
        *,
        workspace_id: str,
        user_id: str,
        ignored: bool,
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
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_role_ignored(
        self,
        *,
        workspace_id: str,
        role_id: str,
        ignored: bool,
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
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_content_mode(
        self,
        *,
        workspace_id: str,
        mode: ReadAloudContentMode,
    ) -> ReadAloudPolicy:
        """Apply one explicit messages/events preset without deleting the route."""

        read_messages = mode in {
            ReadAloudContentMode.ALL,
            ReadAloudContentMode.MESSAGES,
        }
        read_events = mode in {
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
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

    async def set_semantic_options(
        self,
        *,
        workspace_id: str,
        author_names: bool | None = None,
        replies: bool | None = None,
        attachments: bool | None = None,
    ) -> ReadAloudPolicy:
        """Update selected semantic speech formatting switches."""

        if author_names is None and replies is None and attachments is None:
            raise ValueError("read_aloud.semantic_value_required")
        async with self._lock:
            current = self.policy(workspace_id)
            updated = replace(
                current,
                read_author_names=(
                    current.read_author_names
                    if author_names is None
                    else author_names
                ),
                read_replies=current.read_replies if replies is None else replies,
                read_attachments=(
                    current.read_attachments
                    if attachments is None
                    else attachments
                ),
            )
            self._policies[workspace_id] = updated
            await asyncio.to_thread(self._save)
            return updated

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
                if not isinstance(route_items, list) or not isinstance(
                    policy_items, list
                ):
                    raise ValueError("routes and policies must be lists")
            else:
                raise ValueError("expected a list or versioned object")
            for item in route_items:
                if not isinstance(item, dict):
                    raise ValueError("route must be an object")
                route = ReadAloudRoute(
                    workspace_id=str(item["workspace_id"]),
                    text_channel_id=str(item["text_channel_id"]),
                    audio_destination_id=str(item["audio_destination_id"]),
                    mode=ReadAloudMode(str(item["mode"])),
                    enabled=bool(item.get("enabled", True)),
                    additional_text_channel_ids=tuple(
                        str(value)
                        for value in item.get("additional_text_channel_ids", ())
                    ),
                )
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
        payload = {
            "version": 2,
            "routes": [
                {
                    **asdict(route),
                    "mode": route.mode.value,
                    "additional_text_channel_ids": list(
                        route.additional_text_channel_ids
                    ),
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
            "ignore_bots": policy.ignore_bots,
            "ignore_webhooks": policy.ignore_webhooks,
        }
