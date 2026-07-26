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

    async def disable(self, workspace_id: str) -> bool:
        async with self._lock:
            existed = self._routes.pop(workspace_id, None) is not None
            if existed:
                await asyncio.to_thread(self._save)
            return existed

    def matches(self, workspace_id: str, text_channel_id: str) -> bool:
        route = self.get(workspace_id)
        return route is not None and route.text_channel_id == text_channel_id

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
                )
                self._routes[route.workspace_id] = route
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Invalid read-aloud state: {self.state_file}") from exc

    def _save(self) -> None:
        self.state_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        payload = [
            {**asdict(route), "mode": route.mode.value}
            for route in sorted(self._routes.values(), key=lambda value: value.workspace_id)
        ]
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(self.state_file)
        self.state_file.chmod(0o600)
