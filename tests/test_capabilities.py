from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from simajilord.core import (
    CapabilityDescriptor,
    CapabilityRegistry,
    DisclosureClass,
    EgressDescriptor,
    EgressFieldKind,
    EgressSinkAudience,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import CapabilityError, UserError


@dataclass(frozen=True)
class Request:
    value: int


@dataclass(frozen=True)
class Response:
    doubled: int


@dataclass(frozen=True)
class SearchRequest:
    duration_minutes: int
    thread_name: str


class RecordingJournal:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def record_invocation(self, **values: object) -> None:
        self.records.append(values)


class FailingJournal:
    async def record_invocation(self, **values: object) -> None:
        del values
        raise RuntimeError("journal unavailable")


def build_endpoint():
    async def handler(request: Request, _: InvocationContext) -> Response:
        return Response(request.value * 2)

    return endpoint(
        CapabilityDescriptor(
            "test.double",
            "Double an integer.",
            RiskLevel.READ,
            disclosure_class=DisclosureClass.NO_USER_CONTENT,
        ),
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
    assert registry.manifest()[0]["disclosure_class"] == "no_user_content"


def test_read_descriptor_requires_explicit_disclosure_class() -> None:
    with pytest.raises(
        ValueError,
        match="read capabilities require an explicit disclosure_class",
    ):
        CapabilityDescriptor("test.unclassified", "Unsafe read.", RiskLevel.READ)

    with pytest.raises(ValueError, match="must be a DisclosureClass"):
        CapabilityDescriptor(
            "test.invalid_class",
            "Invalid class.",
            RiskLevel.READ,
            disclosure_class="channel_scoped_content",  # type: ignore[arg-type]
        )


def test_egress_descriptor_is_typed_and_bound_to_request_schema() -> None:
    async def handler(request: Request, _: InvocationContext) -> Response:
        return Response(request.value)

    declared = endpoint(
        CapabilityDescriptor(
            "test.egress",
            "Send one query.",
            RiskLevel.EXTERNAL,
            audit_payload="metadata",
            egress=EgressDescriptor(
                provider="test_provider",
                field_kinds=(EgressFieldKind.QUERY,),
                request_fields=("value",),
                source_resource_fields=("value",),
                sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
            ),
        ),
        Request,
        Response,
        handler,
    )
    registry = CapabilityRegistry()
    registry.register(declared)
    assert registry.manifest()[0]["egress"] == {
        "provider": "test_provider",
        "field_kinds": ("query",),
        "request_fields": ("value",),
        "source_resource_fields": ("value",),
        "sink_audience": "external_public",
        "consent": "restricted_or_uncertain",
    }

    with pytest.raises(CapabilityError, match="egress fields are absent"):
        endpoint(
            CapabilityDescriptor(
                "test.invalid_egress",
                "Invalid transfer declaration.",
                RiskLevel.EXTERNAL,
                audit_payload="metadata",
                egress=EgressDescriptor(
                    provider="test_provider",
                    field_kinds=(EgressFieldKind.QUERY,),
                    request_fields=("missing",),
                    sink_audience=EgressSinkAudience.EXTERNAL_PUBLIC,
                ),
            ),
            Request,
            Response,
            handler,
        )


def test_registry_rejects_duplicate_names() -> None:
    registry = CapabilityRegistry()
    registry.register(build_endpoint())
    with pytest.raises(CapabilityError, match="Duplicate"):
        registry.register(build_endpoint())


def test_registry_search_handles_nfkc_punctuation_cjk_and_schema_fields() -> None:
    registry = CapabilityRegistry()

    async def searchable_handler(
        request: SearchRequest,
        _: InvocationContext,
    ) -> Response:
        return Response(request.duration_minutes)

    registry.register(
        endpoint(
            CapabilityDescriptor(
                "timer.create",
                "Create a delayed focus notification.",
                RiskLevel.WRITE,
                keywords=("timer", "タイマー", "集中", "通知"),
            ),
            SearchRequest,
            Response,
            searchable_handler,
        )
    )
    registry.register(
        endpoint(
            CapabilityDescriptor(
                "discord.create_thread",
                "Create a discussion container.",
                RiskLevel.WRITE,
                keywords=("thread", "スレッド", "議論", "分ける"),
            ),
            SearchRequest,
            Response,
            searchable_handler,
        )
    )

    assert registry.search("２５分後、集中タイマーをかけて")[0].descriptor.name == (
        "timer.create"
    )
    assert registry.search("この議論をスレッドに分けて")[0].descriptor.name == (
        "discord.create_thread"
    )
    assert registry.search("TIMER\uFF0FCREATE")[0].descriptor.name == "timer.create"
    assert {
        item.descriptor.name
        for item in registry.search("duration minutes", limit=5)
    } == {"timer.create", "discord.create_thread"}


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


@pytest.mark.asyncio
async def test_registry_enforces_descriptor_timeout_and_records_stable_error() -> None:
    cancelled = asyncio.Event()
    journal = RecordingJournal()

    async def wait_forever(_: Request, __: InvocationContext) -> Response:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    registry = CapabilityRegistry(journal=journal)
    registry.register(
        endpoint(
            CapabilityDescriptor(
                "test.timeout",
                "Wait beyond the declared deadline.",
                RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
                timeout_seconds=0.01,
            ),
            Request,
            Response,
            wait_forever,
        )
    )

    with pytest.raises(UserError) as captured:
        await registry.invoke(
            "test.timeout",
            Request(1),
            InvocationContext("actor", "workspace", "test", "request-timeout"),
        )

    assert captured.value.code == "capability.timeout"
    assert captured.value.details == {
        "capability": "test.timeout",
        "timeout_seconds": 0.01,
    }
    assert cancelled.is_set()
    assert len(journal.records) == 1
    assert journal.records[0]["error"] is captured.value
    assert journal.records[0]["response"] is None


@pytest.mark.asyncio
async def test_journal_failure_does_not_replace_capability_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    successful = CapabilityRegistry(journal=FailingJournal())
    successful.register(build_endpoint())
    result = await successful.invoke(
        "test.double",
        Request(4),
        InvocationContext("actor", "workspace", "test", "request-success"),
    )
    assert result == Response(8)

    async def fail(_: Request, __: InvocationContext) -> Response:
        raise CapabilityError("primary capability failure")

    failed = CapabilityRegistry(journal=FailingJournal())
    failed.register(
        endpoint(
            CapabilityDescriptor(
                "test.fail",
                "Fail predictably.",
                RiskLevel.READ,
                disclosure_class=DisclosureClass.NO_USER_CONTENT,
            ),
            Request,
            Response,
            fail,
        )
    )
    with pytest.raises(CapabilityError, match="primary capability failure"):
        await failed.invoke(
            "test.fail",
            Request(1),
            InvocationContext("actor", "workspace", "test", "request-failure"),
        )

    assert caplog.text.count("Capability journal record failed") == 2


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


@pytest.mark.parametrize(
    "risk",
    (RiskLevel.WRITE, RiskLevel.DESTRUCTIVE),
)
def test_mutating_descriptor_defaults_to_non_idempotent_write(
    risk: RiskLevel,
) -> None:
    descriptor = CapabilityDescriptor(
        "test.write",
        "Write.",
        risk,
    )
    assert descriptor.idempotency == "non_idempotent_write"
