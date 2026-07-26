from __future__ import annotations

from dataclasses import dataclass

import pytest

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityRegistry,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import CapabilityError


@dataclass(frozen=True)
class Request:
    value: int


@dataclass(frozen=True)
class Response:
    doubled: int


def build_endpoint():
    async def handler(request: Request, _: InvocationContext) -> Response:
        return Response(request.value * 2)

    return endpoint(
        CapabilityDescriptor("test.double", "Double an integer.", RiskLevel.READ),
        Request,
        Response,
        handler,
    )


@pytest.mark.asyncio
async def test_registry_invokes_typed_endpoint() -> None:
    registry = CapabilityRegistry()
    registry.register(build_endpoint())
    result = await registry.invoke(
        "test.double",
        Request(4),
        InvocationContext("actor", "workspace", "test", "request"),
    )
    assert result == Response(8)
    assert registry.manifest()[0]["request_fields"] == ("value",)


def test_registry_rejects_duplicate_names() -> None:
    registry = CapabilityRegistry()
    registry.register(build_endpoint())
    with pytest.raises(CapabilityError, match="Duplicate"):
        registry.register(build_endpoint())


@pytest.mark.asyncio
async def test_registry_rejects_wrong_request_type() -> None:
    registry = CapabilityRegistry()
    registry.register(build_endpoint())
    with pytest.raises(CapabilityError, match="expects Request"):
        await registry.invoke(
            "test.double",
            object(),
            InvocationContext("actor", None, "test", "request"),
        )
