from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from simajilord.capabilities import (
    ConnectorBroker,
    ConnectorDescribeRequest,
    ConnectorDescribeResponse,
    ConnectorInvokeRequest,
    ConnectorInvokeResponse,
    ConnectorSearchRequest,
    ConnectorSearchResponse,
)
from simajilord.core import (
    ApprovalMode,
    CapabilityRegistry,
    InvocationContext,
)
from simajilord.core.errors import UserError

_FIGMA_ID = "connector_68df038e0ba48191908c8434991bbac2"
_GITHUB_ID = "connector_76869538009648d5b282a4bb21c3d157"


class _AppServer:
    def __init__(self, inventory: tuple[Mapping[str, object], ...]) -> None:
        self.inventory = inventory
        self.inventory_threads: list[str] = []
        self.calls: list[tuple[str, str, str, Mapping[str, object]]] = []

    async def connector_tool_inventory(
        self,
        *,
        thread_id: str,
    ) -> tuple[Mapping[str, object], ...]:
        self.inventory_threads.append(thread_id)
        return self.inventory

    async def call_connector_tool(
        self,
        *,
        thread_id: str,
        server: str,
        tool: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        self.calls.append((thread_id, server, tool, arguments))
        return {"content": [{"type": "text", "text": "ok"}]}


def _tool(
    name: str,
    *,
    connector_id: str = _FIGMA_ID,
    read_only: bool | None = True,
    schema: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    annotations: dict[str, object] = {"destructiveHint": read_only is False}
    if read_only is not None:
        annotations["readOnlyHint"] = read_only
    return {
        "name": name,
        "title": name.replace("_", " ").title(),
        "description": f"Use {name} in the current design.",
        "inputSchema": schema
        or {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "annotations": annotations,
        "_meta": {"connector_id": connector_id},
    }


def _context() -> InvocationContext:
    return InvocationContext(
        actor_id="actor",
        workspace_id="guild",
        transport="discord",
        request_id="discord:message:123",
        origin_resource_id="channel:456",
        provider_thread_id="thread-1",
        provider_turn_id="turn-1",
    )


def _registry(server: _AppServer) -> CapabilityRegistry:
    broker = ConnectorBroker({_FIGMA_ID: "Figma"})
    broker.bind(server)
    registry = CapabilityRegistry()
    for connector_endpoint in broker.endpoints():
        registry.register(connector_endpoint)
    return registry


@pytest.mark.asyncio
async def test_connector_inventory_is_allowlisted_and_fail_closed() -> None:
    server = _AppServer(
        (
            _tool("figma_read", read_only=True),
            _tool("figma_write", read_only=False),
            _tool("figma_unknown", read_only=None),
            _tool("github_read", connector_id=_GITHUB_ID, read_only=True),
        )
    )
    registry = _registry(server)

    response = await registry.invoke(
        "connector.search",
        ConnectorSearchRequest(query="design"),
        _context(),
    )

    assert isinstance(response, ConnectorSearchResponse)
    assert {tool.tool for tool in response.tools} == {"figma_read", "figma_write"}
    assert response.classified_tool_count == 2
    assert response.denied_unclassified_tool_count == 1
    assert response.inventory_id.startswith("inv_")
    assert server.inventory_threads == ["thread-1"]

    with pytest.raises(UserError) as denied:
        await registry.invoke(
            "connector.describe",
            ConnectorDescribeRequest(_FIGMA_ID, "figma_unknown"),
            _context(),
        )
    assert denied.value.code == "connector.tool_unclassified"


@pytest.mark.asyncio
async def test_connector_contract_binds_identity_scope_thread_and_schema() -> None:
    server = _AppServer((_tool("figma_read"),))
    registry = _registry(server)
    context = _context()
    with pytest.raises(UserError) as missing_turn:
        await registry.invoke(
            "connector.describe",
            ConnectorDescribeRequest(_FIGMA_ID, "figma_read"),
            replace(context, provider_turn_id=None),
        )
    assert missing_turn.value.code == "connector.turn_required"
    described = await registry.invoke(
        "connector.describe",
        ConnectorDescribeRequest(_FIGMA_ID, "figma_read"),
        context,
    )
    assert isinstance(described, ConnectorDescribeResponse)
    request = ConnectorInvokeRequest(
        connector_id=_FIGMA_ID,
        tool="figma_read",
        contract_id=described.contract_id,
        arguments={"query": "current frame"},
    )

    invoked = await registry.invoke("connector.read", request, context)

    assert isinstance(invoked, ConnectorInvokeResponse)
    assert invoked.action_class == "read"
    assert '"text":"ok"' in invoked.content
    assert server.calls == [
        (
            "thread-1",
            "codex_apps",
            "figma_read",
            {"query": "current frame"},
        )
    ]

    for changed_context in (
        replace(context, actor_id="other"),
        replace(context, workspace_id="other"),
        replace(context, request_id="discord:message:other"),
        replace(context, origin_resource_id="channel:other"),
        replace(context, provider_thread_id="thread-2"),
        replace(context, provider_turn_id="turn-2"),
    ):
        with pytest.raises(UserError) as denied:
            await registry.invoke("connector.read", request, changed_context)
        assert denied.value.code == "connector.contract_invalid"

    server.inventory = (
        _tool(
            "figma_read",
            schema={"type": "object", "properties": {"node": {"type": "string"}}},
        ),
    )
    with pytest.raises(UserError) as stale:
        await registry.invoke("connector.read", request, context)
    assert stale.value.code == "connector.contract_invalid"


@pytest.mark.asyncio
async def test_connector_read_write_actions_cannot_cross_endpoints() -> None:
    server = _AppServer((_tool("figma_write", read_only=False),))
    registry = _registry(server)
    context = _context()
    described = await registry.invoke(
        "connector.describe",
        ConnectorDescribeRequest(_FIGMA_ID, "figma_write"),
        context,
    )
    assert isinstance(described, ConnectorDescribeResponse)
    request = ConnectorInvokeRequest(
        connector_id=_FIGMA_ID,
        tool="figma_write",
        contract_id=described.contract_id,
        arguments={"query": "new frame"},
    )

    with pytest.raises(UserError) as mismatch:
        await registry.invoke("connector.read", request, context)
    assert mismatch.value.code == "connector.action_class_mismatch"

    write_endpoint = registry.endpoint("connector.write")
    assert write_endpoint.descriptor.approval is ApprovalMode.WHEN_REQUESTED
    assert write_endpoint.descriptor.idempotency == "non_idempotent_write"
    response = await registry.invoke("connector.write", request, context)
    assert isinstance(response, ConnectorInvokeResponse)
    assert response.action_class == "write"

    server.inventory = (_tool("figma_write", read_only=True),)
    with pytest.raises(UserError) as reclassified:
        await registry.invoke("connector.write", request, context)
    assert reclassified.value.code == "connector.action_class_mismatch"


@pytest.mark.asyncio
async def test_connector_inventory_rejects_duplicate_tool_identity() -> None:
    duplicate = _tool("figma_read")
    registry = _registry(_AppServer((duplicate, duplicate)))

    with pytest.raises(UserError) as collision:
        await registry.invoke(
            "connector.search",
            ConnectorSearchRequest(query="figma"),
            _context(),
        )
    assert collision.value.code == "connector.inventory_collision"
