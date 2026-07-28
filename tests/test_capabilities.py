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


def test_manifest_exposes_agent_planning_metadata() -> None:
    registry = CapabilityRegistry()
    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.write",
                "Update one shared test value.",
                RiskLevel.WRITE,
                requires_workspace=True,
                idempotency="idempotent_write",
                expected_errors=("workspace.required",),
                timeout_seconds=3,
                user_visible_effect="Updates the visible test value.",
            ),
            Request,
            Response,
            lambda request, _: _response(request),
        )
    )

    manifest = registry.manifest()[0]
    assert manifest["requires_workspace"] is True
    assert manifest["requires_voice"] is False
    assert manifest["requires_same_voice"] is False
    assert manifest["idempotency"] == "idempotent_write"
    assert manifest["expected_errors"] == ("workspace.required",)
    assert manifest["timeout_seconds"] == 3
    assert manifest["user_visible_effect"] == "Updates the visible test value."


async def _response(request: Request) -> Response:
    return Response(request.value * 2)


def test_descriptor_rejects_contradictory_planning_metadata() -> None:
    with pytest.raises(ValueError, match="requires_same_voice"):
        CapabilityDescriptor(
            "test.voice",
            "Use voice.",
            RiskLevel.WRITE,
            requires_workspace=True,
            requires_same_voice=True,
        )
    with pytest.raises(ValueError, match="positive"):
        CapabilityDescriptor(
            "test.timeout",
            "Wait.",
            RiskLevel.READ,
            timeout_seconds=0,
        )
    with pytest.raises(ValueError, match="read capabilities"):
        CapabilityDescriptor(
            "test.read",
            "Read.",
            RiskLevel.READ,
            idempotency="idempotent_write",
        )


def test_write_descriptor_defaults_to_non_idempotent_write() -> None:
    descriptor = CapabilityDescriptor(
        "test.write",
        "Write.",
        RiskLevel.WRITE,
    )
    assert descriptor.idempotency == "non_idempotent_write"
