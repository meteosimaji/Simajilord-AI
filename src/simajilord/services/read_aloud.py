"""Persistent read-aloud routing independent of chat transport."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from simajilord.core.errors import ConfigurationError


class ReadAloudMode(StrEnum):
    """How speech behaves while music is already playing."""

    QUEUE = "queue"
    SKIP_DURING_MUSIC = "skip_during_music"


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


class ReadAloudService:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._routes: dict[str, ReadAloudRoute] = {}
        self._lock = asyncio.Lock()
        self._load()

    def get(self, workspace_id: str) -> ReadAloudRoute | None:
        route = self._routes.get(workspace_id)
        return route if route and route.enabled else None

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

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw: Any = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("expected a list")
            for item in raw:
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
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Invalid read-aloud state: {self.state_file}") from exc

    def _save(self) -> None:
        self.state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        payload = [
            {
                **asdict(route),
                "mode": route.mode.value,
                "additional_text_channel_ids": list(
                    route.additional_text_channel_ids
                ),
            }
            for route in sorted(self._routes.values(), key=lambda value: value.workspace_id)
        ]
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self.state_file)
        self.state_file.chmod(0o600)
