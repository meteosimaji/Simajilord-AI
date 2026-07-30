from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import discord

from simajilord.diagnostics.live_discord_capability_audit import (
    EXPECTED_DISCORD_CAPABILITIES,
    minimal_request,
)
from simajilord.integrations.discord.capabilities import build_discord_endpoints
from simajilord.runtime import SimajilordRuntime


class Choice(StrEnum):
    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True, slots=True)
class Nested:
    label: str


@dataclass(frozen=True, slots=True)
class FixtureRequest:
    name: str
    count: int
    enabled: bool
    mode: Literal["safe", "unsafe"]
    choice: Choice
    nested: Nested
    values: tuple[str, ...]
    optional: str | None
    output: Path
    defaulted: str = "kept"


def test_minimal_request_builds_nested_typed_dataclass() -> None:
    request = minimal_request(FixtureRequest)

    assert request == FixtureRequest(
        name="0",
        count=0,
        enabled=False,
        mode="safe",
        choice=Choice.FIRST,
        nested=Nested(label="0"),
        values=(),
        optional=None,
        output=Path("audit-probe"),
    )


def test_minimal_request_constructs_every_discord_endpoint_request() -> None:
    endpoints = build_discord_endpoints(
        Mock(spec=discord.Client),
        Mock(spec=SimajilordRuntime),
    )

    assert len(endpoints) == EXPECTED_DISCORD_CAPABILITIES
    for endpoint in endpoints:
        request = minimal_request(endpoint.request_type)
        assert isinstance(request, endpoint.request_type)
