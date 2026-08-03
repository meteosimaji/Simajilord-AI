"""Host-brokered access to reviewed external design connectors."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from simajilord.core.capabilities import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityEndpoint,
    DisclosureClass,
    InvocationContext,
    RiskLevel,
    endpoint,
)
from simajilord.core.errors import UserError
from simajilord.core.search import (
    normalize_search_text,
    normalized_substring,
    search_overlap_score,
)

_CONNECTOR_SERVER = "codex_apps"
_MAX_ARGUMENT_CHARACTERS = 100_000
_MAX_DESCRIPTION_CHARACTERS = 1_000
_MAX_RESULT_CHARACTERS = 6_000
_MAX_SCHEMA_CHUNK_CHARACTERS = 6_000


class ConnectorActionClass(StrEnum):
    """Fail-closed action class derived from the live MCP annotation."""

    READ = "read"
    WRITE = "write"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConnectorToolDefinition:
    """One normalized tool from the current app-server inventory."""

    connector_id: str
    connector_name: str
    tool: str
    title: str
    description: str
    action_class: ConnectorActionClass
    destructive_hint: bool
    input_schema: Mapping[str, object]


class ConnectorAppServer(Protocol):
    """Narrow app-server port owned by the Simajilord effect broker."""

    async def connector_tool_inventory(
        self,
        *,
        thread_id: str,
    ) -> tuple[Mapping[str, object], ...]: ...

    async def call_connector_tool(
        self,
        *,
        thread_id: str,
        server: str,
        tool: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class ConnectorSearchRequest:
    query: str
    limit: int = 10


@dataclass(frozen=True, slots=True)
class ConnectorToolSummary:
    connector_id: str
    connector_name: str
    tool: str
    title: str
    summary: str
    action_class: str
    destructive_hint: bool


@dataclass(frozen=True, slots=True)
class ConnectorSearchResponse:
    tools: tuple[ConnectorToolSummary, ...]
    classified_tool_count: int
    denied_unclassified_tool_count: int
    inventory_id: str


@dataclass(frozen=True, slots=True)
class ConnectorDescribeRequest:
    connector_id: str
    tool: str
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ConnectorDescribeResponse:
    connector_id: str
    connector_name: str
    tool: str
    action_class: str
    destructive_hint: bool
    contract_id: str
    content: str
    offset: int
    next_offset: int | None
    total_characters: int
    complete: bool


@dataclass(frozen=True, slots=True)
class ConnectorInvokeRequest:
    connector_id: str
    tool: str
    contract_id: str
    arguments: Any = field(
        metadata={
            "description": (
                "An object matching the exact input schema returned by connector.describe."
            )
        }
    )


@dataclass(frozen=True, slots=True)
class ConnectorInvokeResponse:
    connector_id: str
    connector_name: str
    tool: str
    action_class: str
    destructive_hint: bool
    content: str
    content_truncated: bool


class ConnectorBroker:
    """Expose connector effects only after host identity and contract validation."""

    def __init__(self, allowed_connectors: Mapping[str, str]) -> None:
        if not allowed_connectors:
            raise ValueError("at least one connector must be allowlisted")
        if any(not key.strip() or not value.strip() for key, value in allowed_connectors.items()):
            raise ValueError("connector IDs and names must be non-empty")
        self._allowed_connectors = dict(allowed_connectors)
        self._server: ConnectorAppServer | None = None
        self._contract_secret = secrets.token_bytes(32)

    def bind(self, server: ConnectorAppServer) -> None:
        """Bind exactly one app-server after runtime composition."""

        if self._server is not None and self._server is not server:
            raise RuntimeError("connector broker is already bound")
        self._server = server

    def endpoints(self) -> tuple[CapabilityEndpoint, ...]:
        async def search(
            request: ConnectorSearchRequest,
            context: InvocationContext,
        ) -> ConnectorSearchResponse:
            query = " ".join(request.query.split())
            if not query or len(query) > 500:
                raise UserError("connector.query_invalid")
            if not 1 <= request.limit <= 20:
                raise UserError("connector.limit_invalid")
            tools, denied_count = await self._inventory(context)
            ranked = _rank_tools(tools, query)[: request.limit]
            return ConnectorSearchResponse(
                tools=tuple(_summary(tool) for tool in ranked),
                classified_tool_count=len(tools),
                denied_unclassified_tool_count=denied_count,
                inventory_id=_inventory_id(tools),
            )

        async def describe(
            request: ConnectorDescribeRequest,
            context: InvocationContext,
        ) -> ConnectorDescribeResponse:
            tool = await self._selected_tool(
                context,
                connector_id=request.connector_id,
                tool_name=request.tool,
            )
            if request.offset < 0:
                raise UserError("connector.offset_invalid")
            schema = json.dumps(
                tool.input_schema,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if request.offset > len(schema):
                raise UserError("connector.offset_invalid")
            end = min(len(schema), request.offset + _MAX_SCHEMA_CHUNK_CHARACTERS)
            next_offset = end if end < len(schema) else None
            return ConnectorDescribeResponse(
                connector_id=tool.connector_id,
                connector_name=tool.connector_name,
                tool=tool.tool,
                action_class=tool.action_class.value,
                destructive_hint=tool.destructive_hint,
                contract_id=self._contract_id(tool, context),
                content=schema[request.offset:end],
                offset=request.offset,
                next_offset=next_offset,
                total_characters=len(schema),
                complete=next_offset is None,
            )

        async def invoke_read(
            request: ConnectorInvokeRequest,
            context: InvocationContext,
        ) -> ConnectorInvokeResponse:
            return await self._invoke(
                request,
                context,
                expected_class=ConnectorActionClass.READ,
                expected_destructive=False,
            )

        async def invoke_write(
            request: ConnectorInvokeRequest,
            context: InvocationContext,
        ) -> ConnectorInvokeResponse:
            return await self._invoke(
                request,
                context,
                expected_class=ConnectorActionClass.WRITE,
                expected_destructive=False,
            )

        async def invoke_destructive(
            request: ConnectorInvokeRequest,
            context: InvocationContext,
        ) -> ConnectorInvokeResponse:
            return await self._invoke(
                request,
                context,
                expected_class=ConnectorActionClass.WRITE,
                expected_destructive=True,
            )

        common_errors = (
            "connector.runtime_unavailable",
            "connector.thread_required",
            "connector.turn_required",
            "connector.tool_unknown",
            "connector.tool_unclassified",
            "connector.inventory_collision",
            "connector.action_class_mismatch",
            "connector.destructive_class_mismatch",
            "connector.contract_invalid",
            "connector.arguments_invalid",
            "connector.tool_failed",
        )
        return (
            endpoint(
                CapabilityDescriptor(
                    name="connector.search",
                    summary=(
                        "Search the current host-reviewed design connector inventory. "
                        "Unknown or unclassified tools are omitted instead of treated as reads."
                    ),
                    risk=RiskLevel.EXTERNAL,
                    disclosure_class=DisclosureClass.NO_USER_CONTENT,
                    keywords=(
                        "Figma",
                        "Canva",
                        "Adobe",
                        "BioRender",
                        "design connector",
                    ),
                    requires_workspace=True,
                    idempotency="read",
                    expected_errors=(
                        "connector.query_invalid",
                        "connector.limit_invalid",
                        "connector.runtime_unavailable",
                        "connector.thread_required",
                        "connector.inventory_collision",
                    ),
                    timeout_seconds=30,
                    audit_payload="metadata",
                ),
                ConnectorSearchRequest,
                ConnectorSearchResponse,
                search,
            ),
            endpoint(
                CapabilityDescriptor(
                    name="connector.describe",
                    summary=(
                        "Load the exact current input schema and an opaque turn-bound contract "
                        "for one reviewed design connector tool."
                    ),
                    risk=RiskLevel.EXTERNAL,
                    disclosure_class=DisclosureClass.NO_USER_CONTENT,
                    keywords=("connector schema", "app tool contract", "design tool details"),
                    requires_workspace=True,
                    idempotency="read",
                    expected_errors=(
                        "connector.runtime_unavailable",
                        "connector.thread_required",
                        "connector.turn_required",
                        "connector.tool_unknown",
                        "connector.tool_unclassified",
                        "connector.inventory_collision",
                        "connector.offset_invalid",
                    ),
                    timeout_seconds=30,
                    audit_payload="metadata",
                ),
                ConnectorDescribeRequest,
                ConnectorDescribeResponse,
                describe,
            ),
            endpoint(
                CapabilityDescriptor(
                    name="connector.read",
                    summary=(
                        "Invoke one live connector tool whose current MCP metadata explicitly "
                        "classifies it as read-only."
                    ),
                    risk=RiskLevel.EXTERNAL,
                    disclosure_class=DisclosureClass.EXTERNAL_PRIVATE,
                    keywords=("read Figma", "inspect Canva", "search Adobe", "connector read"),
                    requires_workspace=True,
                    idempotency="read",
                    expected_errors=common_errors,
                    timeout_seconds=120,
                    audit_payload="metadata",
                ),
                ConnectorInvokeRequest,
                ConnectorInvokeResponse,
                invoke_read,
            ),
            endpoint(
                CapabilityDescriptor(
                    name="connector.write",
                    summary=(
                        "Invoke one live design connector mutation through the host effect "
                        "ledger after exact-event authorization and a current contract check."
                    ),
                    risk=RiskLevel.WRITE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=(
                        "create Figma design",
                        "edit Canva",
                        "generate diagram",
                        "Adobe design write",
                    ),
                    side_effects=(
                        "May create or modify a resource in the selected external design service.",
                    ),
                    requires_workspace=True,
                    idempotency="non_idempotent_write",
                    expected_errors=common_errors,
                    timeout_seconds=180,
                    user_visible_effect="Creates or changes an external design resource.",
                    audit_payload="metadata",
                ),
                ConnectorInvokeRequest,
                ConnectorInvokeResponse,
                invoke_write,
            ),
            endpoint(
                CapabilityDescriptor(
                    name="connector.destructive",
                    summary=(
                        "Invoke one live connector mutation explicitly marked destructive "
                        "through a separately classified, one-use authorization path."
                    ),
                    risk=RiskLevel.DESTRUCTIVE,
                    approval=ApprovalMode.WHEN_REQUESTED,
                    keywords=(
                        "delete external design",
                        "destructive connector action",
                        "remove Canva resource",
                    ),
                    side_effects=(
                        "May irreversibly delete or replace an external resource.",
                    ),
                    requires_workspace=True,
                    idempotency="non_idempotent_write",
                    expected_errors=common_errors,
                    timeout_seconds=180,
                    user_visible_effect=(
                        "Performs a connector operation marked destructive by the live tool."
                    ),
                    audit_payload="metadata",
                ),
                ConnectorInvokeRequest,
                ConnectorInvokeResponse,
                invoke_destructive,
            ),
        )

    async def _invoke(
        self,
        request: ConnectorInvokeRequest,
        context: InvocationContext,
        *,
        expected_class: ConnectorActionClass,
        expected_destructive: bool,
    ) -> ConnectorInvokeResponse:
        selected = await self._selected_tool(
            context,
            connector_id=request.connector_id,
            tool_name=request.tool,
        )
        if selected.action_class is not expected_class:
            raise UserError("connector.action_class_mismatch")
        if selected.destructive_hint is not expected_destructive:
            raise UserError("connector.destructive_class_mismatch")
        expected_contract = self._contract_id(selected, context)
        if not secrets.compare_digest(request.contract_id, expected_contract):
            raise UserError("connector.contract_invalid")
        if not isinstance(request.arguments, dict):
            raise UserError("connector.arguments_invalid")
        try:
            encoded_arguments = json.dumps(
                request.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise UserError("connector.arguments_invalid") from exc
        if len(encoded_arguments) > _MAX_ARGUMENT_CHARACTERS:
            raise UserError("connector.arguments_invalid")
        server = self._require_server()
        thread_id = _thread_id(context)
        try:
            result = await server.call_connector_tool(
                thread_id=thread_id,
                server=_CONNECTOR_SERVER,
                tool=selected.tool,
                arguments=request.arguments,
            )
        except Exception as exc:
            raise UserError("connector.tool_failed") from exc
        if result.get("isError") is True:
            raise UserError("connector.tool_failed")
        content, truncated = _bounded_result(result)
        return ConnectorInvokeResponse(
            connector_id=selected.connector_id,
            connector_name=selected.connector_name,
            tool=selected.tool,
            action_class=selected.action_class.value,
            destructive_hint=selected.destructive_hint,
            content=content,
            content_truncated=truncated,
        )

    async def _selected_tool(
        self,
        context: InvocationContext,
        *,
        connector_id: str,
        tool_name: str,
    ) -> ConnectorToolDefinition:
        tools, _ = await self._inventory(context)
        for tool in tools:
            if tool.connector_id == connector_id and tool.tool == tool_name:
                return tool
        if connector_id not in self._allowed_connectors:
            raise UserError("connector.tool_unknown")
        raise UserError("connector.tool_unclassified")

    async def _inventory(
        self,
        context: InvocationContext,
    ) -> tuple[tuple[ConnectorToolDefinition, ...], int]:
        server = self._require_server()
        thread_id = _thread_id(context)
        try:
            raw_tools = await server.connector_tool_inventory(
                thread_id=thread_id,
            )
        except Exception as exc:
            raise UserError("connector.runtime_unavailable") from exc
        classified: list[ConnectorToolDefinition] = []
        denied = 0
        identities: set[tuple[str, str]] = set()
        for raw in raw_tools:
            parsed = _parse_tool(raw, self._allowed_connectors)
            if parsed is None:
                continue
            identity = (parsed.connector_id, parsed.tool)
            if identity in identities:
                raise UserError("connector.inventory_collision")
            identities.add(identity)
            if parsed.action_class is ConnectorActionClass.UNKNOWN:
                denied += 1
                continue
            classified.append(parsed)
        classified.sort(key=lambda item: (item.connector_name, item.tool))
        return tuple(classified), denied

    def _require_server(self) -> ConnectorAppServer:
        if self._server is None:
            raise UserError("connector.runtime_unavailable")
        return self._server

    def _contract_id(
        self,
        tool: ConnectorToolDefinition,
        context: InvocationContext,
    ) -> str:
        schema_hash = hashlib.sha256(
            json.dumps(
                tool.input_schema,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        payload = "\0".join(
            (
                "connector-contract-v1",
                context.actor_id,
                context.workspace_id or "",
                context.request_id,
                context.origin_resource_id or "",
                context.provider_thread_id or "",
                _turn_id(context),
                tool.connector_id,
                tool.tool,
                tool.action_class.value,
                "destructive" if tool.destructive_hint else "non-destructive",
                schema_hash,
            )
        ).encode("utf-8")
        digest = hmac.new(
            self._contract_secret,
            payload,
            hashlib.sha256,
        ).hexdigest()[:32]
        return f"con_{digest}"


def _thread_id(context: InvocationContext) -> str:
    thread_id = context.provider_thread_id
    if not isinstance(thread_id, str) or not thread_id:
        raise UserError("connector.thread_required")
    return thread_id


def _turn_id(context: InvocationContext) -> str:
    turn_id = context.provider_turn_id
    if not isinstance(turn_id, str) or not turn_id:
        raise UserError("connector.turn_required")
    return turn_id


def _parse_tool(
    raw: Mapping[str, object],
    allowed_connectors: Mapping[str, str],
) -> ConnectorToolDefinition | None:
    metadata = raw.get("_meta")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    connector_id = metadata.get("connector_id")
    if not isinstance(connector_id, str):
        connector_id = metadata.get("connectorId")
    if not isinstance(connector_id, str) or connector_id not in allowed_connectors:
        return None
    tool = raw.get("name")
    if not isinstance(tool, str) or not tool or len(tool) > 300:
        return None
    annotations = raw.get("annotations")
    annotations = annotations if isinstance(annotations, Mapping) else {}
    read_only_hint = annotations.get("readOnlyHint")
    if read_only_hint is True:
        action_class = ConnectorActionClass.READ
    elif read_only_hint is False:
        action_class = ConnectorActionClass.WRITE
    else:
        action_class = ConnectorActionClass.UNKNOWN
    input_schema = raw.get("inputSchema")
    if not isinstance(input_schema, Mapping):
        action_class = ConnectorActionClass.UNKNOWN
        input_schema = {}
    title = raw.get("title")
    description = raw.get("description")
    return ConnectorToolDefinition(
        connector_id=connector_id,
        connector_name=allowed_connectors[connector_id],
        tool=tool,
        title=(title[:200] if isinstance(title, str) and title else tool),
        description=(
            description[:_MAX_DESCRIPTION_CHARACTERS]
            if isinstance(description, str)
            else ""
        ),
        action_class=action_class,
        destructive_hint=annotations.get("destructiveHint") is True,
        input_schema={str(key): value for key, value in input_schema.items()},
    )


def _rank_tools(
    tools: Sequence[ConnectorToolDefinition],
    query: str,
) -> tuple[ConnectorToolDefinition, ...]:
    normalized_query = normalize_search_text(query)
    scored: list[tuple[int, ConnectorToolDefinition]] = []
    for tool in tools:
        text = " ".join(
            (
                tool.connector_name,
                tool.tool,
                tool.title,
                tool.description,
                tool.action_class.value,
            )
        )
        score = (
            3 * search_overlap_score(query, tool.tool)
            + 2 * search_overlap_score(query, tool.connector_name)
            + search_overlap_score(query, text)
            + 2 * int(normalized_substring(query, text))
        )
        if score or normalized_query in {"design", "connector", "app"}:
            scored.append((score, tool))
    scored.sort(key=lambda entry: (-entry[0], entry[1].connector_name, entry[1].tool))
    return tuple(tool for _, tool in scored)


def _summary(tool: ConnectorToolDefinition) -> ConnectorToolSummary:
    return ConnectorToolSummary(
        connector_id=tool.connector_id,
        connector_name=tool.connector_name,
        tool=tool.tool,
        title=tool.title,
        summary=tool.description[:400],
        action_class=tool.action_class.value,
        destructive_hint=tool.destructive_hint,
    )


def _inventory_id(tools: Sequence[ConnectorToolDefinition]) -> str:
    payload = "\0".join(
        f"{tool.connector_id}:{tool.tool}:{tool.action_class.value}:"
        f"{int(tool.destructive_hint)}"
        for tool in tools
    ).encode("utf-8")
    return "inv_" + hashlib.sha256(payload).hexdigest()[:24]


def _bounded_result(result: Mapping[str, object]) -> tuple[str, bool]:
    try:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise UserError("connector.tool_failed") from exc
    if len(encoded) <= _MAX_RESULT_CHARACTERS:
        return encoded, False
    preview = encoded[:_MAX_RESULT_CHARACTERS]
    return json.dumps(
        {
            "preview": preview,
            "total_characters": len(encoded),
            "truncated": True,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ), True
