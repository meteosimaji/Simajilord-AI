"""Bounded dynamic-tool bridge from an agent provider to typed capabilities."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import hashlib
import hmac
import json
import logging
import math
import secrets
import types
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from simajilord.core import (
    ApprovalMode,
    CapabilityDescriptor,
    CapabilityRegistry,
    DisclosureClass,
    EgressDescriptor,
    InvocationContext,
    RiskLevel,
)
from simajilord.core.errors import CapabilityError

from .actions import ActionReceipt, ActionReceiptService
from .errors import AgentToolError

_TOOL_NAMESPACE = "simajilord"
_LIST_TOOL = "capability_list"
_SEARCH_TOOL = "capability_search"
_DESCRIBE_TOOL = "capability_describe"
_RESOLUTION_TOOL = "capability_resolution"
_INVOKE_TOOL = "capability_invoke"
_AUTHORIZATION_EVENT_ID = "authorization_event_id"
_MAX_CAPABILITY_LIST_OFFSET = 10_000
_MAX_CAPABILITY_SEARCH_LIMIT = 25
_DEFAULT_CONCRETE_SEARCH_LIMIT = 10
_CAPABILITY_LIST_CURSOR_PREFIX = "simajilord-tools-v1:"
_CAPABILITY_CATALOG_ID_PREFIX = "capcat_v1_"
_CAPABILITY_CONTRACT_ID_PREFIX = "capcon_v1_"
_CONTENT_RESULT_KEYS = ("content", "content_chunk", "text")
_RESULT_IDENTITY_KEYS = (
    "path",
    "url",
    "message_id",
    "channel_id",
    "guild_id",
    "source_guild_id",
    "source_channel_id",
    "kind",
    "query",
    "backend",
    "order",
)
_CONTINUATION_METADATA_KEYS = (
    "complete",
    "next_offset",
    "next_page",
    "next_before_message_id",
    "next_after_message_id",
    "next_cursor",
    "has_more",
    "indexing",
    "retry_after_seconds",
    "search_window_exhausted",
    "total_results",
    "offset",
    "page_start",
    "total_pages",
    "total_characters",
    "maybe_more",
)
log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class AgentToolOutput:
    """Model-facing text plus optional media carried outside the text budget."""

    text: str
    image_url: str | None = None

    def __len__(self) -> int:
        return len(self.text)

    def __contains__(self, value: str) -> bool:
        return value in self.text


@dataclasses.dataclass(frozen=True, slots=True)
class AgentToolCallMetadata:
    """Body-free routing and risk metadata for one provider tool request."""

    route: Literal[
        "eager",
        "capability_invoke",
        "capability_search",
        "capability_list",
        "capability_describe",
        "capability_resolution",
    ]
    capability_name: str | None
    risk: RiskLevel | None
    write: bool
    destructive: bool


class AgentToolCatalog:
    """Expose only an explicit capability allowlist to the model."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        allowed_capabilities: Sequence[str],
        *,
        required_grants: Mapping[str, str] | None = None,
        eager_capabilities: Sequence[str] | None = None,
        write_capabilities: Sequence[str] = (),
        destructive_capabilities: Sequence[str] = (),
        image_output_capabilities: Sequence[str] = (),
        action_receipts: ActionReceiptService | None = None,
    ) -> None:
        self._registry = registry
        self._allowed_capabilities = tuple(allowed_capabilities)
        duplicate_capabilities = _duplicates(self._allowed_capabilities)
        if duplicate_capabilities:
            raise AgentToolError(
                "Agent capability allowlist contains duplicates: "
                + ", ".join(duplicate_capabilities)
            )
        policy_sequences = {
            "eager capability policy": tuple(eager_capabilities or ()),
            "write capability policy": tuple(write_capabilities),
            "destructive capability policy": tuple(destructive_capabilities),
            "image output capability policy": tuple(image_output_capabilities),
        }
        for label, values in policy_sequences.items():
            duplicates = _duplicates(values)
            if duplicates:
                raise AgentToolError(
                    f"Agent {label} contains duplicates: "
                    + ", ".join(duplicates)
                )
        self._required_grants = dict(required_grants or {})
        self._eager_capabilities = (
            frozenset(self._allowed_capabilities)
            if eager_capabilities is None
            else frozenset(eager_capabilities)
        )
        self._write_capabilities = frozenset(write_capabilities)
        self._destructive_capabilities = frozenset(destructive_capabilities)
        self._image_output_capabilities = frozenset(image_output_capabilities)
        self._action_receipts = action_receipts
        # Opaque discovery proofs are scoped to this runtime instance and one
        # stable InvocationContext. The model can copy them but cannot mint a
        # contract without first calling capability_describe.
        self._discovery_secret = secrets.token_bytes(32)
        allowed = set(self._allowed_capabilities)
        unknown_eager = self._eager_capabilities - allowed
        unknown_writes = self._write_capabilities - allowed
        unknown_destructive = self._destructive_capabilities - allowed
        unknown_images = self._image_output_capabilities - allowed
        if unknown_eager or unknown_writes or unknown_destructive or unknown_images:
            unknown = sorted(
                unknown_eager
                | unknown_writes
                | unknown_destructive
                | unknown_images
            )
            raise AgentToolError(
                "Agent tool policies reference capabilities outside the allowlist: "
                + ", ".join(unknown)
            )
        unknown_policies = set(self._required_grants) - set(self._allowed_capabilities)
        if unknown_policies:
            raise AgentToolError(
                "Grant policies reference capabilities outside the allowlist: "
                + ", ".join(sorted(unknown_policies))
            )
        ungranted_writes = self._write_capabilities - set(self._required_grants)
        if ungranted_writes:
            raise AgentToolError(
                "Agent write capabilities require explicit grants: "
                + ", ".join(sorted(ungranted_writes))
            )
        if self._action_receipts is not None:
            unclassified_writes = {
                capability
                for capability in self._write_capabilities
                if not self._action_receipts.has_explicit_policy(capability)
            }
            if unclassified_writes:
                raise AgentToolError(
                    "Agent write capabilities require an explicit Action policy: "
                    + ", ".join(sorted(unclassified_writes))
                )
        unmanaged_destructive = (
            self._destructive_capabilities - self._write_capabilities
        )
        if unmanaged_destructive:
            raise AgentToolError(
                "Destructive capabilities must also be declared as writes: "
                + ", ".join(sorted(unmanaged_destructive))
            )
        aliases: dict[str, str] = {}
        for capability_name in self._allowed_capabilities:
            alias = _tool_alias(capability_name)
            if alias in {
                _LIST_TOOL,
                _SEARCH_TOOL,
                _DESCRIBE_TOOL,
                _RESOLUTION_TOOL,
                _INVOKE_TOOL,
            }:
                raise AgentToolError(
                    f"Dynamic tool alias is reserved by the capability broker: {alias}"
                )
            previous = aliases.setdefault(alias, capability_name)
            if previous != capability_name:
                raise AgentToolError(
                    f"Dynamic tool alias collision: {previous} and {capability_name}"
                )
        self._aliases = aliases

    @property
    def namespace(self) -> str:
        return _TOOL_NAMESPACE

    @property
    def allowed_capabilities(self) -> tuple[str, ...]:
        """Return the immutable model allowlist for diagnostics and coverage checks."""

        return self._allowed_capabilities

    def write_capability_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> str | None:
        """Resolve whether one dynamic call is an explicitly granted write."""

        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        return (
            capability_name
            if capability_name in self._write_capabilities
            else None
        )

    def authorization_event_id_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> str | None:
        """Return the exact event whose actor must authorize one resolved write."""

        if self.write_capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        ) is None:
            return None
        if not isinstance(arguments, dict):
            return None
        value = arguments.get(_AUTHORIZATION_EVENT_ID)
        if isinstance(value, str) and value:
            return value
        if tool_name == _INVOKE_TOOL:
            capability_arguments = arguments.get("arguments")
            if isinstance(capability_arguments, dict):
                nested_value = capability_arguments.get(_AUTHORIZATION_EVENT_ID)
                if isinstance(nested_value, str) and nested_value:
                    return nested_value
        return None

    def write_is_safe_to_retry(self, capability_name: str) -> bool:
        """Return whether an already failed write may be repeated automatically."""

        if capability_name not in self._write_capabilities:
            return False
        return (
            self._registry.endpoint(capability_name).descriptor.idempotency
            == "idempotent_write"
        )

    def capability_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> str | None:
        """Resolve the concrete capability behind eager and brokered calls."""

        if tool_name == _INVOKE_TOOL and isinstance(arguments, dict):
            name = arguments.get("name")
            return (
                name
                if isinstance(name, str) and name in self._allowed_capabilities
                else None
            )
        return self._aliases.get(tool_name)

    def trace_metadata_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> AgentToolCallMetadata:
        """Resolve trace metadata without retaining or returning call arguments."""

        if tool_name == _LIST_TOOL:
            route: Literal[
                "eager",
                "capability_invoke",
                "capability_search",
                "capability_list",
                "capability_describe",
                "capability_resolution",
            ] = "capability_list"
        elif tool_name == _SEARCH_TOOL:
            route = "capability_search"
        elif tool_name == _DESCRIBE_TOOL:
            route = "capability_describe"
        elif tool_name == _RESOLUTION_TOOL:
            route = "capability_resolution"
        elif tool_name == _INVOKE_TOOL:
            route = "capability_invoke"
        else:
            route = "eager"
        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        risk: RiskLevel | None = None
        if capability_name is not None:
            with suppress(CapabilityError):
                risk = self._registry.endpoint(capability_name).descriptor.risk
        return AgentToolCallMetadata(
            route=route,
            capability_name=capability_name,
            risk=risk,
            write=capability_name in self._write_capabilities,
            destructive=capability_name in self._destructive_capabilities,
        )

    def timeout_seconds_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> float | None:
        """Return the capability's own execution limit for host watchdog routing."""

        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        if capability_name is None:
            return None
        return self._registry.endpoint(capability_name).descriptor.timeout_seconds

    def disclosure_class_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> DisclosureClass | None:
        """Return the host-declared information class for one concrete call."""

        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        if capability_name is None:
            return None
        return self._registry.endpoint(capability_name).descriptor.disclosure_class

    def egress_descriptor_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> EgressDescriptor | None:
        """Return the host-declared external-transfer contract for one call."""

        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        if capability_name is None:
            return None
        return self._registry.endpoint(capability_name).descriptor.egress

    def canonical_tool_name_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> str:
        """Return the dedicated alias even when a call used the broker."""

        capability_name = self.capability_for_call(
            tool_name=tool_name,
            arguments=arguments,
        )
        return (
            _tool_alias(capability_name)
            if capability_name is not None
            else tool_name
        )

    def capability_arguments_for_call(
        self,
        *,
        tool_name: str,
        arguments: object,
    ) -> object:
        """Return request arguments independent of dedicated or brokered routing."""

        if tool_name == _INVOKE_TOOL and isinstance(arguments, dict):
            return arguments.get("arguments")
        return arguments

    def dynamic_specs(
        self,
        context: InvocationContext | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """Build app-server dynamic tool specs from currently registered endpoints."""

        tools: list[Mapping[str, object]] = []
        hidden_configured = False
        hidden_available = False
        any_available = False
        for alias, capability_name in sorted(self._aliases.items()):
            if capability_name not in self._eager_capabilities:
                hidden_configured = True
            if not self._is_available(capability_name, context):
                continue
            any_available = True
            endpoint = self._validated_endpoint(capability_name, context)
            if capability_name not in self._eager_capabilities:
                hidden_available = True
                continue
            tools.append(
                {
                    "type": "function",
                    "name": alias,
                    "description": _descriptor_description(endpoint.descriptor),
                    "inputSchema": _tool_input_schema(
                        endpoint.request_type,
                        write=capability_name in self._write_capabilities,
                    ),
                }
            )
        if hidden_configured:
            tools.append(_list_spec())
            tools.append(_search_spec())
            tools.append(_resolution_spec())
        if hidden_configured and any_available:
            tools.append(_describe_spec())
        if hidden_available:
            tools.append(_invoke_spec())
        if not tools:
            return ()
        return (
            {
                "type": "namespace",
                "name": self.namespace,
                "description": (
                    "Typed Simajilord capabilities. List compact summaries for general "
                    "ability questions; search only for a concrete requested action."
                ),
                "tools": tools,
            },
        )

    async def invoke(
        self,
        *,
        namespace: str | None,
        tool_name: str,
        arguments: object,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None = None,
    ) -> AgentToolOutput:
        if namespace != self.namespace:
            raise AgentToolError("The dynamic tool namespace is not allowed.")
        if tool_name == _LIST_TOOL:
            return AgentToolOutput(
                self._list(
                    arguments,
                    context=context,
                    max_output_characters=max_output_characters,
                    before_invoke=before_invoke,
                )
            )
        if tool_name == _SEARCH_TOOL:
            return AgentToolOutput(
                self._search(
                    arguments,
                    context=context,
                    max_output_characters=max_output_characters,
                    before_invoke=before_invoke,
                )
            )
        if tool_name == _DESCRIBE_TOOL:
            return AgentToolOutput(
                self._describe(
                    arguments,
                    context=context,
                    max_output_characters=max_output_characters,
                    before_invoke=before_invoke,
                )
            )
        if tool_name == _RESOLUTION_TOOL:
            return AgentToolOutput(
                self._resolve_discovery(
                    arguments,
                    context=context,
                    max_output_characters=max_output_characters,
                    before_invoke=before_invoke,
                )
            )
        if tool_name == _INVOKE_TOOL:
            return await self._invoke_discovered(
                arguments,
                context=context,
                max_output_characters=max_output_characters,
                before_invoke=before_invoke,
            )
        try:
            capability_name = self._aliases[tool_name]
        except KeyError as exc:
            raise AgentToolError("The dynamic tool is not allowed.") from exc
        if capability_name not in self._eager_capabilities:
            raise AgentToolError("Use capability_invoke for discovered capabilities.")
        return await self._invoke_capability(
            capability_name,
            _without_authorization_event_id(arguments),
            context=context,
            max_output_characters=max_output_characters,
            before_invoke=before_invoke,
        )

    def _list(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> str:
        """List compact available tools without injecting every input schema."""

        if not isinstance(arguments, dict):
            raise AgentToolError("Capability list arguments must be an object.")
        unknown = set(arguments) - {"cursor", "limit"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability list fields: {', '.join(sorted(unknown))}"
            )
        raw_cursor = arguments.get("cursor")
        limit = arguments.get("limit", _MAX_CAPABILITY_SEARCH_LIMIT)
        offset = _decode_capability_list_cursor(raw_cursor)
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= _MAX_CAPABILITY_SEARCH_LIMIT
        ):
            raise AgentToolError(
                "Capability list limit must be between 1 and "
                f"{_MAX_CAPABILITY_SEARCH_LIMIT}."
            )
        if before_invoke is not None:
            before_invoke()

        registered = {
            item.descriptor.name: item
            for item in self._registry.all()
            if item.descriptor.name in self._allowed_capabilities
        }
        available: list[Any] = []
        unavailable_reason_counts: dict[str, int] = {}
        for capability_name in sorted(self._allowed_capabilities):
            reason = self._unavailable_reason(capability_name, context)
            if reason is not None:
                unavailable_reason_counts[reason] = (
                    unavailable_reason_counts.get(reason, 0) + 1
                )
                continue
            item = registered.get(capability_name)
            if item is None:
                # _unavailable_reason reports this as endpoint_unregistered.
                continue
            self._validated_endpoint(capability_name, context)
            available.append(item)

        page = available[offset : offset + limit]
        next_offset = offset + len(page)
        has_more = next_offset < len(available)
        payload = {
            "tools": [
                {
                    "name": item.descriptor.name,
                    "summary": item.descriptor.summary,
                    "risk": item.descriptor.risk.value,
                    "use_for_concrete_need": _SEARCH_TOOL,
                    "authorization_event_id_required": (
                        item.descriptor.name in self._write_capabilities
                    ),
                }
                for item in page
            ],
            "next_cursor": (
                _encode_capability_list_cursor(next_offset) if has_more else None
            ),
            "has_more": has_more,
            "total_results": len(available),
            "unavailable_reason_counts": unavailable_reason_counts,
        }
        return _bounded_json(
            payload,
            max_output_characters=max_output_characters,
            request={"cursor": raw_cursor},
        )

    def _search(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> str:
        if not isinstance(arguments, dict):
            raise AgentToolError("Capability search arguments must be an object.")
        unknown = set(arguments) - {"query", "limit"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability search fields: {', '.join(sorted(unknown))}"
            )
        query = arguments.get("query", "")
        limit = arguments.get("limit")
        if not isinstance(query, str) or not query.strip():
            raise AgentToolError(
                "Capability search requires one concrete need; use capability_list "
                "for general ability browsing."
            )
        if limit is None:
            limit = _DEFAULT_CONCRETE_SEARCH_LIMIT
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= _MAX_CAPABILITY_SEARCH_LIMIT
        ):
            raise AgentToolError(
                "Capability search limit must be between 1 and "
                f"{_MAX_CAPABILITY_SEARCH_LIMIT}."
            )
        if before_invoke is not None:
            before_invoke()
        registered = self._registry.all()
        candidates = tuple(
            item
            for item in self._registry.search(
                query,
                limit=max(1, len(registered)),
            )
            if item.descriptor.name in self._allowed_capabilities
        )
        available_matches: list[Any] = []
        unavailable_reason_counts: dict[str, int] = {}
        for capability_name in sorted(self._allowed_capabilities):
            reason = self._unavailable_reason(capability_name, context)
            if reason is not None:
                unavailable_reason_counts[reason] = (
                    unavailable_reason_counts.get(reason, 0) + 1
                )
        for item in candidates:
            if self._unavailable_reason(item.descriptor.name, context) is None:
                self._validated_endpoint(item.descriptor.name, context)
                available_matches.append(item)
        page = available_matches[:limit]
        available_catalog = tuple(
            item
            for item in registered
            if (
                item.descriptor.name in self._allowed_capabilities
                and self._unavailable_reason(item.descriptor.name, context) is None
            )
        )
        payload = {
            "query": query,
            "detail": "summary",
            "matches": [
                {
                    "name": item.descriptor.name,
                    "summary": item.descriptor.summary,
                    "risk": item.descriptor.risk.value,
                    "describe_with": _DESCRIBE_TOOL,
                    "authorization_event_id_required": (
                        item.descriptor.name in self._write_capabilities
                    ),
                }
                for item in page
            ],
            "ranked_hints_returned": len(page),
            "ranked_hints_truncated": len(page) < len(available_matches),
            "total_ranked_results": len(available_matches),
            "unavailable_reason_counts": unavailable_reason_counts,
            "catalog_complete": True,
            "catalog_id": _capability_catalog_id(
                available_catalog,
                context=context,
                secret=self._discovery_secret,
            ),
            "catalog_index": _capability_catalog_index(available_catalog),
            "describe_with": _DESCRIBE_TOOL,
            "resolve_unavailable_with": _RESOLUTION_TOOL,
        }
        return _bounded_capability_search_json(
            payload,
            max_output_characters=max_output_characters,
            request={"query": query},
        )

    def _describe(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> str:
        """Return the exact contract for one AI-selected available capability."""

        if not isinstance(arguments, dict):
            raise AgentToolError("Capability description arguments must be an object.")
        unknown = set(arguments) - {"catalog_id", "name"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability description fields: {', '.join(sorted(unknown))}"
            )
        catalog_id = arguments.get("catalog_id")
        capability_name = arguments.get("name")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise AgentToolError("Capability description catalog_id must be text.")
        if not isinstance(capability_name, str) or not capability_name:
            raise AgentToolError("Capability description name must be text.")
        if (
            capability_name not in self._allowed_capabilities
            or self._unavailable_reason(capability_name, context) is not None
        ):
            raise AgentToolError("The capability is not available for this turn.")
        available_catalog = tuple(
            item
            for item in self._registry.all()
            if (
                item.descriptor.name in self._allowed_capabilities
                and self._unavailable_reason(item.descriptor.name, context) is None
            )
        )
        _validate_capability_catalog_id(
            catalog_id,
            available_catalog,
            context=context,
            secret=self._discovery_secret,
        )
        endpoint = self._validated_endpoint(capability_name, context)
        if before_invoke is not None:
            before_invoke()
        payload = {
            "catalog_id": catalog_id,
            "name": endpoint.descriptor.name,
            "summary": endpoint.descriptor.summary,
            "risk": endpoint.descriptor.risk.value,
            "metadata": _descriptor_metadata(endpoint.descriptor),
            "input_schema": _dataclass_schema(endpoint.request_type),
            "invoke_with": _INVOKE_TOOL,
            "contract_id": _capability_contract_id(
                catalog_id,
                endpoint.descriptor.name,
                context=context,
                secret=self._discovery_secret,
            ),
            "authorization_event_id_required": (
                endpoint.descriptor.name in self._write_capabilities
            ),
        }
        return _bounded_json(
            payload,
            max_output_characters=max_output_characters,
            request={"catalog_id": catalog_id, "name": capability_name},
        )

    def _resolve_discovery(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> str:
        """Record the model's semantic no-match conclusion against a full index."""

        if not isinstance(arguments, dict):
            raise AgentToolError("Capability resolution arguments must be an object.")
        unknown = set(arguments) - {"catalog_id", "conclusion", "reason"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability resolution fields: {', '.join(sorted(unknown))}"
            )
        catalog_id = arguments.get("catalog_id")
        conclusion = arguments.get("conclusion")
        reason = arguments.get("reason")
        if not isinstance(catalog_id, str) or not catalog_id:
            raise AgentToolError("Capability resolution catalog_id must be text.")
        if conclusion != "unavailable":
            raise AgentToolError(
                "Capability resolution conclusion must be unavailable."
            )
        if not isinstance(reason, str) or not reason.strip():
            raise AgentToolError("Capability resolution reason must be non-empty text.")
        available_catalog = tuple(
            item
            for item in self._registry.all()
            if (
                item.descriptor.name in self._allowed_capabilities
                and self._unavailable_reason(item.descriptor.name, context) is None
            )
        )
        _validate_capability_catalog_id(
            catalog_id,
            available_catalog,
            context=context,
            secret=self._discovery_secret,
        )
        if before_invoke is not None:
            before_invoke()
        return _bounded_json(
            {
                "catalog_id": catalog_id,
                "conclusion": "unavailable",
                "reason": " ".join(reason.split())[:1_000],
                "recorded": True,
            },
            max_output_characters=max_output_characters,
            request={"catalog_id": catalog_id},
        )

    async def _invoke_discovered(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> AgentToolOutput:
        if not isinstance(arguments, dict):
            raise AgentToolError("Capability invocation arguments must be an object.")
        unknown = set(arguments) - {
            "name",
            "contract_id",
            "arguments",
            _AUTHORIZATION_EVENT_ID,
        }
        if unknown:
            raise AgentToolError(
                f"Unknown capability invocation fields: {', '.join(sorted(unknown))}"
            )
        capability_name = arguments.get("name")
        contract_id = arguments.get("contract_id")
        capability_arguments = arguments.get("arguments")
        if not isinstance(capability_name, str):
            raise AgentToolError("Capability name must be text.")
        if not isinstance(contract_id, str) or not contract_id:
            raise AgentToolError(
                "Capability invocation requires contract_id from capability_describe."
            )
        if capability_name not in self._allowed_capabilities:
            raise AgentToolError("The capability is not allowed.")
        available_catalog = tuple(
            item
            for item in self._registry.all()
            if (
                item.descriptor.name in self._allowed_capabilities
                and self._unavailable_reason(item.descriptor.name, context) is None
            )
        )
        catalog_id = _capability_catalog_id(
            available_catalog,
            context=context,
            secret=self._discovery_secret,
        )
        expected_contract_id = _capability_contract_id(
            catalog_id,
            capability_name,
            context=context,
            secret=self._discovery_secret,
        )
        if not secrets.compare_digest(contract_id, expected_contract_id):
            raise AgentToolError(
                "Capability contract is missing, stale, or belongs to another capability; "
                "search and describe exactly one candidate again."
            )
        return await self._invoke_capability(
            capability_name,
            _without_authorization_event_id(capability_arguments),
            context=context,
            max_output_characters=max_output_characters,
            before_invoke=before_invoke,
        )

    async def _invoke_capability(
        self,
        capability_name: str,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
        before_invoke: Callable[[], None] | None,
    ) -> AgentToolOutput:
        if not self._is_available(capability_name, context):
            raise AgentToolError("The dynamic tool grant is not present for this turn.")
        endpoint = self._validated_endpoint(capability_name, context)
        request = _build_dataclass(endpoint.request_type, arguments)
        if before_invoke is not None:
            before_invoke()
        effect_id: str | None = None
        tracks_external_effect = (
            self._action_receipts is not None
            and capability_name in self._write_capabilities
        )
        if tracks_external_effect:
            assert self._action_receipts is not None
            try:
                effect_id = await self._action_receipts.begin_external_effect(
                    capability=capability_name,
                    context=context,
                )
            except Exception as exc:
                raise AgentToolError(
                    "The external effect ledger is unavailable; the write was not dispatched."
                ) from exc
        try:
            result = await self._registry.invoke(capability_name, request, context)
        except BaseException:
            if effect_id is not None:
                assert self._action_receipts is not None
                try:
                    await self._action_receipts.mark_external_effect_unknown(
                        effect_id,
                        context=context,
                    )
                except Exception:
                    log.critical(
                        "External effect state could not be closed effect=%s request_id=%s",
                        effect_id,
                        context.request_id,
                        exc_info=True,
                    )
            raise
        receipt: ActionReceipt | None = None
        if tracks_external_effect:
            assert self._action_receipts is not None
            try:
                receipt = await self._action_receipts.record(
                    capability=capability_name,
                    request=request,
                    response=result,
                    context=context,
                    effect_id=effect_id,
                )
            except Exception:
                log.exception(
                    "Action receipt recording failed capability=%s request_id=%s",
                    capability_name,
                    context.request_id,
                )
                if effect_id is not None:
                    try:
                        await self._action_receipts.mark_external_effect_unknown(
                            effect_id,
                            context=context,
                        )
                    except Exception:
                        log.critical(
                            "External effect state could not be marked unknown "
                            "effect=%s request_id=%s",
                            effect_id,
                            context.request_id,
                            exc_info=True,
                        )
            if (
                receipt is None
                and effect_id is not None
                and capability_name == "action.undo"
            ):
                # action.undo owns its inverse receipt internally; the positive
                # endpoint result still confirms this provider effect.
                await self._action_receipts.confirm_external_effect(
                    effect_id,
                    context=context,
                    action_id=None,
                )
        if capability_name in self._image_output_capabilities:
            if not dataclasses.is_dataclass(result):
                raise AgentToolError("Image capability returned an invalid record.")
            image_url = getattr(result, "image_data_url", None)
            if not isinstance(image_url, str) or not image_url.startswith("data:image/"):
                raise AgentToolError("Image capability returned invalid model media.")
            visible = {
                field.name: (
                    "[attached to this tool result]"
                    if field.name == "image_data_url"
                    else getattr(result, field.name)
                )
                for field in dataclasses.fields(cast(Any, result))
            }
            if receipt is not None:
                visible["action_receipt"] = receipt
            return AgentToolOutput(
                _bounded_json(
                    visible,
                    max_output_characters=max_output_characters,
                    request=request,
                ),
                image_url=image_url,
            )
        visible_result = _with_action_receipt(result, receipt)
        return AgentToolOutput(
            _bounded_json(
                visible_result,
                max_output_characters=max_output_characters,
                request=request,
            )
        )

    def _is_available(
        self,
        capability_name: str,
        context: InvocationContext | None,
    ) -> bool:
        return self._unavailable_reason(capability_name, context) is None

    def _unavailable_reason(
        self,
        capability_name: str,
        context: InvocationContext | None,
    ) -> str | None:
        """Return a coarse availability bucket without exposing endpoint metadata."""

        if (
            context is not None
            and context.allowed_capabilities is not None
            and capability_name not in context.allowed_capabilities
        ):
            return "policy_denied"

        required_grant = self._required_grants.get(capability_name)
        has_grant = required_grant is None or (
            context is not None and required_grant in context.grants
        )
        if not has_grant:
            return "missing_grant"
        try:
            descriptor = self._registry.endpoint(capability_name).descriptor
        except CapabilityError:
            # Discord transport endpoints are attached after the core runtime is built.
            return "endpoint_unregistered"
        if descriptor.requires_workspace and (
            context is None or context.workspace_id is None
        ):
            return "workspace_required"
        if descriptor.approval is ApprovalMode.NEVER:
            return None
        if descriptor.approval is ApprovalMode.WHEN_REQUESTED:
            return (
                None
                if context is not None and capability_name in context.approvals
                else "approval_required"
            )
        return "approval_required"

    def _validated_endpoint(
        self,
        capability_name: str,
        context: InvocationContext | None,
    ) -> Any:
        endpoint = self._registry.endpoint(capability_name)
        descriptor = endpoint.descriptor
        if (
            context is not None
            and context.allowed_capabilities is not None
            and capability_name not in context.allowed_capabilities
        ):
            raise AgentToolError(
                f"Agent policy does not allow {capability_name} in this turn."
            )
        required_grant = self._required_grants.get(capability_name)
        if descriptor.approval is ApprovalMode.ALWAYS:
            raise AgentToolError(
                f"Agent catalog cannot expose always-approved {capability_name}."
            )
        if (
            descriptor.approval is ApprovalMode.WHEN_REQUESTED
            and (context is None or capability_name not in context.approvals)
        ):
            raise AgentToolError(
                f"Agent catalog lacks turn approval for {capability_name}."
            )
        if (
            capability_name in self._destructive_capabilities
            and descriptor.risk is not RiskLevel.DESTRUCTIVE
        ):
            raise AgentToolError(
                "Agent destructive policy does not match capability risk metadata: "
                f"{capability_name}."
            )
        if descriptor.risk is RiskLevel.DESTRUCTIVE and (
            capability_name not in self._destructive_capabilities
            or capability_name not in self._write_capabilities
            or required_grant is None
        ):
            raise AgentToolError(
                f"Agent catalog cannot expose unmanaged destructive {capability_name}."
            )
        if descriptor.risk is RiskLevel.EXTERNAL and required_grant is None:
            raise AgentToolError(
                f"Agent external catalog requires a grant for {capability_name}."
            )
        if descriptor.risk in {RiskLevel.WRITE, RiskLevel.DESTRUCTIVE} and (
            capability_name not in self._write_capabilities or required_grant is None
        ):
            raise AgentToolError(
                f"Agent catalog cannot expose unapproved write {capability_name}."
            )
        return endpoint


def _descriptor_metadata(descriptor: CapabilityDescriptor) -> Mapping[str, object]:
    """Return compact, stable behavior metadata for model-side planning."""

    return {
        "approval": descriptor.approval.value,
        "requires_workspace": descriptor.requires_workspace,
        "requires_voice": descriptor.requires_voice,
        "requires_same_voice": descriptor.requires_same_voice,
        "idempotency": descriptor.idempotency,
        "expected_errors": descriptor.expected_errors,
        "timeout_seconds": descriptor.timeout_seconds,
        "user_visible_effect": descriptor.user_visible_effect,
        "disclosure_class": (
            descriptor.disclosure_class.value
            if descriptor.disclosure_class is not None
            else None
        ),
        "egress": (
            {
                "provider": descriptor.egress.provider,
                "field_kinds": tuple(
                    item.value for item in descriptor.egress.field_kinds
                ),
                "sink_audience": descriptor.egress.sink_audience.value,
                "consent": descriptor.egress.consent.value,
                "source_resource_fields": (
                    descriptor.egress.source_resource_fields
                ),
            }
            if descriptor.egress is not None
            else None
        ),
    }


def _descriptor_description(descriptor: CapabilityDescriptor) -> str:
    """Describe operational constraints without requiring another tool call."""

    constraints: list[str] = []
    if descriptor.requires_same_voice:
        constraints.append("requester must share the bot's voice channel")
    elif descriptor.requires_voice:
        constraints.append("requester must be in a voice channel")
    elif descriptor.requires_workspace:
        constraints.append("Discord server context required")
    if descriptor.idempotency == "read":
        constraints.append("read-only; safe to retry")
    elif descriptor.idempotency == "idempotent_write":
        constraints.append("idempotent write; safe to retry after timeout")
    else:
        constraints.append("non-idempotent write; inspect state before retrying")
    if descriptor.approval is not ApprovalMode.NEVER:
        constraints.append(f"approval: {descriptor.approval.value}")
    if descriptor.timeout_seconds is not None:
        constraints.append(f"timeout: {descriptor.timeout_seconds:g}s")
    if descriptor.user_visible_effect:
        constraints.append(f"visible effect: {descriptor.user_visible_effect}")
    if descriptor.egress is not None:
        constraints.append(
            "external transfer: "
            f"{descriptor.egress.provider} / "
            f"{descriptor.egress.sink_audience.value} / "
            + ", ".join(item.value for item in descriptor.egress.field_kinds)
        )
    if descriptor.expected_errors:
        constraints.append("expected errors: " + ", ".join(descriptor.expected_errors))
    return f"{descriptor.summary} Operational metadata: {'; '.join(constraints)}."


def _list_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _LIST_TOOL,
        "description": (
            "List available Simajilord capabilities as compact summaries without loading "
            "their input schemas. Use this for general ability questions. Copy next_cursor "
            "unchanged into cursor until has_more is false. For one concrete need, use "
            "capability_search, select a capability by meaning from its complete catalog "
            "index, then load only that contract with capability_describe; do not invoke "
            "directly from this list."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cursor": {
                    "type": "string",
                    "description": "Copy next_cursor exactly; omit it for the first page.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_CAPABILITY_SEARCH_LIMIT,
                    "default": _MAX_CAPABILITY_SEARCH_LIMIT,
                },
            },
            "additionalProperties": False,
        },
    }


def _search_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _SEARCH_TOOL,
        "description": (
            "Find candidates for one concrete need without loading any input schemas. A "
            "concrete result includes ranked summaries plus a complete compact index of every "
            "capability available in this turn. Select from that index by meaning rather than "
            "assuming rank is exhaustive, then call capability_describe for one candidate at a "
            "time, copying catalog_id. After invoking it, the same catalog can describe another "
            "necessary candidate. The returned contract_id is required to invoke it. "
            "If none is semantically suitable, call capability_resolution with the returned "
            "catalog_id before claiming the capability is unavailable. Use "
            "capability_list—not a specially worded search—for general ability browsing. "
            "Unavailable capability names and schemas stay hidden."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Describe the one concrete state, ability, or action that is needed. "
                        "The host uses this only for ranked hints; the AI must semantically "
                        "inspect the complete catalog_index itself."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_CAPABILITY_SEARCH_LIMIT,
                    "description": (
                        "Ranked hints return short summaries only and default to 10. The "
                        "complete name index is independent of this limit."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _describe_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _DESCRIBE_TOOL,
        "description": (
            "Load the exact input schema and operational metadata for one capability selected "
            "semantically from capability_search's complete catalog index. Load only the "
            "candidate you intend to invoke or cite as an available ability. Copy catalog_id "
            "from that search; this returns the contract_id required by capability_invoke."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "catalog_id": {
                    "type": "string",
                    "description": "Copy catalog_id from capability_search.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Copy one complete capability name from matches or catalog_index."
                    ),
                },
            },
            "required": ["catalog_id", "name"],
            "additionalProperties": False,
        },
    }


def _resolution_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _RESOLUTION_TOOL,
        "description": (
            "Record the AI's semantic conclusion that no currently available capability fits "
            "one concrete need. Use this only after reviewing the complete catalog_index from "
            "capability_search; it validates the opaque catalog_id. This is the required "
            "grounding step before saying a capability is unavailable."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "catalog_id": {
                    "type": "string",
                    "description": "Copy catalog_id from the concrete capability_search result.",
                },
                "conclusion": {
                    "type": "string",
                    "enum": ["unavailable"],
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "A concise semantic reason no indexed capability satisfies the need."
                    ),
                },
            },
            "required": ["catalog_id", "conclusion", "reason"],
            "additionalProperties": False,
        },
    }


def _encode_capability_list_cursor(offset: int) -> str:
    raw = f"{_CAPABILITY_LIST_CURSOR_PREFIX}{offset}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_capability_list_cursor(cursor: object) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or not cursor or len(cursor) > 200:
        raise AgentToolError("Capability list cursor is invalid.")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(cursor + padding).decode("ascii")
        if not decoded.startswith(_CAPABILITY_LIST_CURSOR_PREFIX):
            raise ValueError
        raw_offset = decoded.removeprefix(_CAPABILITY_LIST_CURSOR_PREFIX)
        if not raw_offset.isdigit():
            raise ValueError
        offset = int(raw_offset)
    except (UnicodeDecodeError, ValueError, binascii.Error) as exc:
        raise AgentToolError("Capability list cursor is invalid.") from exc
    if not 0 <= offset <= _MAX_CAPABILITY_LIST_OFFSET:
        raise AgentToolError("Capability list cursor is out of range.")
    return offset


def _capability_catalog_id(
    endpoints: Sequence[Any],
    *,
    context: InvocationContext,
    secret: bytes,
) -> str:
    """Return an opaque proof bound to this runtime, request, and full catalog."""

    names = sorted(item.descriptor.name for item in endpoints)
    payload = "\0".join(
        ("catalog", _capability_discovery_scope(context), *names)
    ).encode("utf-8")
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]
    return f"{_CAPABILITY_CATALOG_ID_PREFIX}{digest}"


def _validate_capability_catalog_id(
    catalog_id: str,
    endpoints: Sequence[Any],
    *,
    context: InvocationContext,
    secret: bytes,
) -> None:
    expected = _capability_catalog_id(
        endpoints,
        context=context,
        secret=secret,
    )
    if not secrets.compare_digest(catalog_id, expected):
        raise AgentToolError(
            "Capability catalog changed, belongs to another turn, or was not fully "
            "inspected; search again."
        )


def _capability_contract_id(
    catalog_id: str,
    capability_name: str,
    *,
    context: InvocationContext,
    secret: bytes,
) -> str:
    """Prove that one exact contract was loaded for this request and catalog."""

    payload = "\0".join(
        (
            "contract",
            _capability_discovery_scope(context),
            catalog_id,
            capability_name,
        )
    ).encode("utf-8")
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()[:32]
    return f"{_CAPABILITY_CONTRACT_ID_PREFIX}{digest}"


def _capability_discovery_scope(context: InvocationContext) -> str:
    """Use stable host identity fields; provider call IDs legitimately change per step."""

    return "\0".join(
        (
            context.actor_id,
            context.workspace_id or "",
            context.transport,
            context.request_id,
            context.origin_resource_id or "",
            context.active_message_id or "",
            *context.resource_ids,
        )
    )


def _capability_catalog_index(
    endpoints: Sequence[Any],
) -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in sorted(endpoints, key=lambda entry: entry.descriptor.name):
        name = item.descriptor.name
        namespace, _, _ = name.partition(".")
        grouped.setdefault(namespace, []).append(name)
    return {
        namespace: tuple(local_names)
        for namespace, local_names in sorted(grouped.items())
    }


def _invoke_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _INVOKE_TOOL,
        "description": (
            "Invoke one capability after loading its exact current contract with "
            "capability_describe. Copy its opaque contract_id; a guessed name or contract from "
            "another turn or capability is rejected."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "contract_id": {
                    "type": "string",
                    "description": "Opaque contract_id returned by capability_describe.",
                },
                "arguments": {"type": "object"},
                _AUTHORIZATION_EVENT_ID: {
                    "type": "string",
                    "description": (
                        "Required for a write: the exact Simajilord event ID whose "
                        "actor is authorizing this action."
                    ),
                },
            },
            "required": ["name", "contract_id", "arguments"],
            "additionalProperties": False,
        },
    }


def _tool_alias(capability_name: str) -> str:
    alias = capability_name.replace(".", "_").replace("-", "_")
    if not alias or len(alias) > 128 or not alias.replace("_", "").isalnum():
        raise AgentToolError(f"Capability name cannot become a dynamic tool: {capability_name}")
    return alias


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    """Return stable duplicate names before aliases silently shadow a policy entry."""

    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return tuple(sorted(duplicates))


def _dataclass_schema(model: type[Any]) -> Mapping[str, object]:
    if not dataclasses.is_dataclass(model):
        raise AgentToolError(f"{model.__name__} must be a dataclass.")
    hints = get_type_hints(model)
    properties: dict[str, object] = {}
    required: list[str] = []
    for field in dataclasses.fields(model):
        annotation = hints.get(field.name, Any)
        property_schema = dict(_annotation_schema(annotation))
        description = field.metadata.get("description")
        if isinstance(description, str) and description:
            property_schema["description"] = description
        properties[field.name] = property_schema
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            required.append(field.name)
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _tool_input_schema(
    model: type[Any],
    *,
    write: bool,
) -> Mapping[str, object]:
    schema = dict(_dataclass_schema(model))
    if not write:
        return schema
    properties = dict(cast(Mapping[str, object], schema["properties"]))
    if _AUTHORIZATION_EVENT_ID in properties:
        raise AgentToolError(
            f"{model.__name__} shadows the reserved "
            f"{_AUTHORIZATION_EVENT_ID} field."
        )
    properties[_AUTHORIZATION_EVENT_ID] = {
        "type": "string",
        "description": (
            "The exact Simajilord event ID whose actor is authorizing this write. "
            "Read its Discord message completely before acting when it has one."
        ),
    }
    schema["properties"] = properties
    required = list(cast(Sequence[str], schema.get("required", ())))
    required.append(_AUTHORIZATION_EVENT_ID)
    schema["required"] = required
    return schema


def _without_authorization_event_id(arguments: object) -> object:
    if not isinstance(arguments, dict) or _AUTHORIZATION_EVENT_ID not in arguments:
        return arguments
    return {
        key: value
        for key, value in arguments.items()
        if key != _AUTHORIZATION_EVENT_ID
    }


def _annotation_schema(annotation: object) -> Mapping[str, object]:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if not arguments or not all(isinstance(item, str) for item in arguments):
            raise AgentToolError(
                f"Unsupported dynamic tool literal annotation: {annotation!r}"
            )
        return {"type": "string", "enum": list(arguments)}
    if origin in (Union, types.UnionType):
        non_none = tuple(item for item in arguments if item is not type(None))
        if len(non_none) == 1 and len(non_none) != len(arguments):
            return {"anyOf": [_annotation_schema(non_none[0]), {"type": "null"}]}
        return {"anyOf": [_annotation_schema(item) for item in arguments]}
    if origin in (tuple, list, Sequence):
        item_type = arguments[0] if arguments else Any
        return {"type": "array", "items": _annotation_schema(item_type)}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {
            "type": "string",
            "enum": [item.value for item in annotation],
        }
    if annotation is str or annotation is Path:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is Any:
        return {}
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _dataclass_schema(annotation)
    raise AgentToolError(f"Unsupported dynamic tool annotation: {annotation!r}")


def _build_dataclass(model: type[Any], arguments: object) -> object:
    if not isinstance(arguments, dict):
        raise AgentToolError("Dynamic tool arguments must be an object.")
    fields_by_name = {field.name: field for field in dataclasses.fields(model)}
    unknown = set(arguments) - set(fields_by_name)
    if unknown:
        raise AgentToolError(f"Unknown dynamic tool fields: {', '.join(sorted(unknown))}")
    hints = get_type_hints(model)
    values: dict[str, object] = {}
    try:
        for field_name, value in arguments.items():
            values[field_name] = _convert_value(value, hints.get(field_name, Any))
        return model(**values)
    except (TypeError, ValueError) as exc:
        raise AgentToolError("Dynamic tool arguments are invalid.") from exc


def _convert_value(value: object, annotation: object) -> object:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if value not in arguments:
            raise ValueError("Value is not one of the allowed literal values.")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        failures: list[Exception] = []
        for choice in arguments:
            if choice is type(None):
                continue
            try:
                return _convert_value(value, choice)
            except (TypeError, ValueError) as exc:
                failures.append(exc)
        raise TypeError("Value does not match any allowed type.") from (
            failures[-1] if failures else None
        )
    if origin in (tuple, list, Sequence):
        if not isinstance(value, list):
            raise TypeError("Expected an array.")
        item_type = arguments[0] if arguments else Any
        converted = [_convert_value(item, item_type) for item in value]
        return tuple(converted) if origin is tuple else converted
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("Expected text.")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean.")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("Expected an integer.")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("Expected a number.")
        return float(value)
    if annotation is Path:
        if not isinstance(value, str):
            raise TypeError("Expected a path string.")
        return Path(value)
    if annotation is Any:
        return value
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return _build_dataclass(annotation, value)
    raise TypeError(f"Unsupported value type: {annotation!r}")


def _bounded_capability_search_json(
    value: Mapping[str, object],
    *,
    max_output_characters: int,
    request: object | None,
) -> str:
    """Keep a concrete search's complete name index ahead of extra ranked matches."""

    if max_output_characters < 200:
        raise AgentToolError("Dynamic tool output budget is too small.")
    original_matches = value.get("matches")
    if not isinstance(original_matches, list):
        return _bounded_json(
            value,
            max_output_characters=max_output_characters,
            request=request,
        )
    total_results = value.get("total_ranked_results")
    total = (
        total_results
        if isinstance(total_results, int) and not isinstance(total_results, bool)
        else len(original_matches)
    )
    visible_matches = list(original_matches)
    while True:
        candidate = dict(value)
        candidate["matches"] = visible_matches
        candidate["ranked_hints_returned"] = len(visible_matches)
        candidate["ranked_hints_truncated"] = len(visible_matches) < total
        if len(_encode_json(_json_value(candidate))) <= max_output_characters:
            return _encode_json(_json_value(candidate))
        if not visible_matches:
            break
        visible_matches.pop()

    # An incomplete catalog must never carry a valid proof that could ground an
    # unavailable conclusion. The caller can report the bounded limitation or
    # page capability_list, but cannot mistake a partial index for evidence.
    fallback = {
        "query": value.get("query"),
        "catalog_complete": False,
        "catalog_id": None,
        "catalog_fallback": _LIST_TOOL,
        "matches": [],
        "ranked_hints_returned": 0,
        "ranked_hints_truncated": True,
        "error": "complete_capability_index_exceeds_output_budget",
    }
    return _bounded_json(
        fallback,
        max_output_characters=max_output_characters,
        request=request,
    )


def _bounded_json(
    value: object,
    *,
    max_output_characters: int,
    request: object | None = None,
) -> str:
    if max_output_characters < 200:
        raise AgentToolError("Dynamic tool output budget is too small.")
    normalized = _json_value(value)
    encoded = _encode_json(normalized)
    if len(encoded) <= max_output_characters:
        return encoded
    if isinstance(normalized, Mapping):
        mapping = {str(key): item for key, item in normalized.items()}
        for key in _CONTENT_RESULT_KEYS:
            if isinstance(mapping.get(key), str):
                return _bounded_content_mapping(
                    mapping,
                    content_key=key,
                    request=request,
                    max_output_characters=max_output_characters,
                )
        list_key = _primary_result_list_key(mapping)
        if list_key is not None:
            return _bounded_list_mapping(
                mapping,
                list_key=list_key,
                request=request,
                max_output_characters=max_output_characters,
            )
    return _bounded_structured_value(
        normalized,
        max_output_characters=max_output_characters,
    )


def _bounded_content_mapping(
    value: Mapping[str, object],
    *,
    content_key: str,
    request: object | None,
    max_output_characters: int,
) -> str:
    original_content = value[content_key]
    assert isinstance(original_content, str)
    payload = _shallow_result_payload(value, excluded={content_key})
    payload[content_key] = ""
    offset_value = value.get("offset")
    if not isinstance(offset_value, int) or isinstance(offset_value, bool):
        offset_value = _request_integer(request, "offset")
    has_offset_continuation = (
        offset_value is not None
        and ("next_offset" in value or _request_has_field(request, "offset"))
    )
    protected = {
        "truncated",
        "reason",
        content_key,
        *(
            ("offset", "next_offset", "complete")
            if has_offset_continuation
            else ()
        ),
    }
    payload = _drop_optional_fields_to_fit(
        payload,
        protected=protected,
        max_output_characters=max_output_characters,
    )

    def candidate(visible_characters: int) -> dict[str, object]:
        result = dict(payload)
        result[content_key] = original_content[:visible_characters]
        if visible_characters < len(original_content):
            if has_offset_continuation:
                assert offset_value is not None
                result["offset"] = offset_value
                result["next_offset"] = offset_value + visible_characters
                result["complete"] = False
                if "next_page" in result:
                    # Finish the selected page chunk before advancing to a later page.
                    result["next_page"] = None
            result["content_truncated"] = True
        return result

    best = candidate(0)
    if len(_encode_json(best)) > max_output_characters:
        minimal: dict[str, object] = {
            "truncated": True,
            "reason": "agent_tool_output_budget",
            content_key: "",
        }
        if has_offset_continuation:
            assert offset_value is not None
            minimal.update(
                {
                    "offset": offset_value,
                    "next_offset": offset_value,
                    "complete": False,
                }
            )
        payload = minimal
        best = candidate(0)
    lower = 0
    upper = len(original_content)
    while lower <= upper:
        midpoint = (lower + upper) // 2
        proposed = candidate(midpoint)
        if len(_encode_json(proposed)) <= max_output_characters:
            best = proposed
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    if (
        original_content
        and not best[content_key]
        and has_offset_continuation
    ):
        # A supported budget always has room for progress once optional metadata is gone.
        one_character = candidate(1)
        if len(_encode_json(one_character)) <= max_output_characters:
            best = one_character
    return _encode_json(best)


def _bounded_list_mapping(
    value: Mapping[str, object],
    *,
    list_key: str,
    request: object | None,
    max_output_characters: int,
) -> str:
    original_items = value[list_key]
    assert isinstance(original_items, list)
    payload = _shallow_result_payload(value, excluded={list_key})
    payload[list_key] = []
    mode = _list_continuation_mode(value, request)
    base_offset = _request_integer(request, "offset")
    if base_offset is None:
        response_offset = value.get("offset")
        if isinstance(response_offset, int) and not isinstance(response_offset, bool):
            base_offset = response_offset
    protected = {"truncated", "reason", list_key}
    if mode == "offset":
        protected.update(("offset", "next_offset", "complete"))
    elif mode == "before":
        protected.update(("next_before_message_id", "has_more", "complete"))
    elif mode == "after":
        protected.update(("next_after_message_id", "has_more", "complete"))
    elif mode == "cursor":
        protected.update(("next_cursor", "has_more", "complete"))
    payload = _drop_optional_fields_to_fit(
        payload,
        protected=protected,
        max_output_characters=max_output_characters,
    )

    def selected(count: int, *, compact: bool = False) -> list[object]:
        if count <= 0:
            return []
        timestamp_desc = (
            mode == "before"
            and _request_text(request, "sort_by") == "timestamp"
            and _request_text(request, "sort_order") == "desc"
        )
        items = (
            original_items[-count:]
            if mode == "before" and not timestamp_desc
            else original_items[:count]
        )
        if not compact:
            return list(items)
        return [
            _compact_json_value(item, string_limit=96, list_limit=4, depth=0)[0]
            for item in items
        ]

    def candidate(items: list[object]) -> dict[str, object]:
        result = dict(payload)
        result[list_key] = items
        hidden_items = len(items) < len(original_items)
        if hidden_items:
            result["complete"] = False
            result["has_more"] = True
            if mode == "offset":
                start = base_offset or 0
                result["offset"] = start
                result["next_offset"] = start + len(items)
                if "next_before_message_id" in result:
                    result["next_before_message_id"] = None
                if "next_after_message_id" in result:
                    result["next_after_message_id"] = None
            elif mode == "before":
                if "next_offset" in result:
                    result["next_offset"] = None
                if "next_after_message_id" in result:
                    result["next_after_message_id"] = None
                timestamp_desc = (
                    _request_text(request, "sort_by") == "timestamp"
                    and _request_text(request, "sort_order") == "desc"
                )
                boundary = _message_boundary(items, first=not timestamp_desc)
                if boundary is not None:
                    result["next_before_message_id"] = boundary
            elif mode == "after":
                if "next_offset" in result:
                    result["next_offset"] = None
                if "next_before_message_id" in result:
                    result["next_before_message_id"] = None
                boundary = _message_boundary(items, first=False)
                if boundary is not None:
                    result["next_after_message_id"] = boundary
            elif mode == "cursor":
                result["next_offset"] = None
                result["next_cursor"] = None
                retry_arguments: dict[str, object] = {
                    "limit": max(1, len(items)),
                }
                request_cursor = _request_text(request, "cursor")
                if request_cursor is not None:
                    retry_arguments["cursor"] = request_cursor
                result["continuation_retry_required"] = True
                result["continuation_retry"] = {
                    "use_same_arguments": True,
                    "replace": retry_arguments,
                }
            result["items_truncated"] = True
        return result

    best = candidate([])
    for count in range(1, len(original_items) + 1):
        proposed = candidate(selected(count))
        if len(_encode_json(proposed)) > max_output_characters:
            break
        best = proposed
    if not best[list_key] and original_items:
        compact_item = selected(1, compact=True)
        proposed = candidate(compact_item)
        if len(_encode_json(proposed)) <= max_output_characters:
            best = proposed
        else:
            projected_item = (
                _capability_match_projection(
                    selected(1)[0]
                )
                if list_key == "matches"
                else _identity_projection(compact_item[0])
            )
            proposed = candidate([projected_item])
            if len(_encode_json(proposed)) <= max_output_characters:
                best = proposed
    if len(_encode_json(best)) <= max_output_characters:
        return _encode_json(best)
    return _bounded_structured_value(
        {
            "truncated": True,
            "reason": "agent_tool_output_budget",
            list_key: [],
        },
        max_output_characters=max_output_characters,
    )


def _bounded_structured_value(
    value: object,
    *,
    max_output_characters: int,
) -> str:
    for string_limit, list_limit in (
        (512, 20),
        (256, 10),
        (128, 5),
        (64, 2),
        (32, 1),
    ):
        compact, _changed = _compact_json_value(
            value,
            string_limit=string_limit,
            list_limit=list_limit,
            depth=0,
        )
        payload = (
            dict(compact)
            if isinstance(compact, Mapping)
            else {"value": compact}
        )
        payload["truncated"] = True
        payload["reason"] = "agent_tool_output_budget"
        encoded = _encode_json(payload)
        if len(encoded) <= max_output_characters:
            return encoded
    return _encode_json(
        {
            "truncated": True,
            "reason": "agent_tool_output_budget",
            "result_omitted": True,
        }
    )


def _shallow_result_payload(
    value: Mapping[str, object],
    *,
    excluded: set[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "truncated": True,
        "reason": "agent_tool_output_budget",
    }
    for key, item in value.items():
        if key in excluded:
            continue
        if item is None or isinstance(item, (bool, int, float)):
            payload[key] = item
            continue
        if isinstance(item, str):
            payload[key] = item if len(item) <= 512 else item[:511] + "…"
            continue
        if len(_encode_json(item)) <= 512:
            payload[key] = item
    return payload


def _drop_optional_fields_to_fit(
    payload: dict[str, object],
    *,
    protected: set[str],
    max_output_characters: int,
) -> dict[str, object]:
    bounded = dict(payload)
    while len(_encode_json(bounded)) > max_output_characters:
        optional = [key for key in bounded if key not in protected]
        if not optional:
            break
        selected = max(optional, key=lambda key: len(_encode_json(bounded[key])))
        bounded.pop(selected)
    return bounded


def _compact_json_value(
    value: object,
    *,
    string_limit: int,
    list_limit: int,
    depth: int,
) -> tuple[object, bool]:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value, False
        return value[: max(1, string_limit - 1)] + "…", True
    if isinstance(value, Mapping):
        compact: dict[str, object] = {}
        changed = False
        for key, item in value.items():
            compact_item, item_changed = _compact_json_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth + 1,
            )
            compact[str(key)] = compact_item
            changed = changed or item_changed
        if changed and depth > 0:
            compact["_output_truncated"] = True
        return compact, changed
    if isinstance(value, list):
        shown = value[:list_limit]
        compact_items: list[object] = []
        changed = len(shown) < len(value)
        for item in shown:
            compact_item, item_changed = _compact_json_value(
                item,
                string_limit=string_limit,
                list_limit=list_limit,
                depth=depth + 1,
            )
            compact_items.append(compact_item)
            changed = changed or item_changed
        return compact_items, changed
    return value, False


def _identity_projection(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    projected = {
        str(key): item
        for key, item in value.items()
        if (
            str(key) in _RESULT_IDENTITY_KEYS
            or str(key) in {"name", "title", "risk"}
            or str(key).endswith("_id")
        )
        and (item is None or isinstance(item, (str, bool, int, float)))
    }
    projected["_output_truncated"] = True
    return projected


def _capability_match_projection(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    return {
        key: _without_schema_descriptions(item)
        for key, item in value.items()
        if key in {"name", "risk", "input_schema"}
    }


def _without_schema_descriptions(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_schema_descriptions(item)
            for key, item in value.items()
            if str(key) != "description"
        }
    if isinstance(value, list):
        return [_without_schema_descriptions(item) for item in value]
    return value


def _primary_result_list_key(value: Mapping[str, object]) -> str | None:
    for key in (
        "matches",
        "messages",
        "files",
        "servers",
        "channels",
        "sources",
        "items",
        "timers",
        "events",
        "jobs",
        "results",
        "links",
    ):
        if isinstance(value.get(key), list):
            return key
    return next(
        (str(key) for key, item in value.items() if isinstance(item, list)),
        None,
    )


def _list_continuation_mode(
    value: Mapping[str, object],
    request: object | None,
) -> Literal["offset", "before", "after", "cursor", "none"]:
    sort_by = _request_text(request, "sort_by")
    sort_order = _request_text(request, "sort_order")
    if sort_by == "relevance":
        if (
            value.get("cursor_pagination") is True
            or _request_text(request, "cursor") is not None
            or isinstance(value.get("next_cursor"), str)
        ):
            return "cursor"
        if "next_offset" in value:
            return "offset"
    if sort_by == "timestamp":
        if "next_after_message_id" in value and sort_order == "asc":
            return "after"
        if "next_before_message_id" in value:
            return "before"
    if sort_by is None:
        if "next_offset" in value and _request_has_field(request, "offset"):
            return "offset"
        if "next_before_message_id" in value:
            return "before"
        if "next_after_message_id" in value:
            return "after"
    return "none"


def _message_boundary(items: list[object], *, first: bool) -> str | None:
    if not items:
        return None
    selected = items[0] if first else items[-1]
    if not isinstance(selected, Mapping):
        return None
    message_id = selected.get("message_id")
    return message_id if isinstance(message_id, str) and message_id else None


def _request_has_field(request: object | None, field_name: str) -> bool:
    if isinstance(request, Mapping):
        return field_name in request
    return request is not None and hasattr(request, field_name)


def _request_integer(request: object | None, field_name: str) -> int | None:
    value = (
        request.get(field_name)
        if isinstance(request, Mapping)
        else getattr(request, field_name, None)
    )
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _request_text(request: object | None, field_name: str) -> str | None:
    value = (
        request.get(field_name)
        if isinstance(request, Mapping)
        else getattr(request, field_name, None)
    )
    return value if isinstance(value, str) else None


def _encode_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _continuation_metadata(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    metadata: dict[str, object] = {}
    for key in _CONTINUATION_METADATA_KEYS:
        if key not in value:
            continue
        item = value[key]
        if item is None or isinstance(item, (str, bool, int)):
            metadata[key] = item
    return metadata


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return repr(value)


def _with_action_receipt(
    result: object,
    receipt: ActionReceipt | None,
) -> object:
    """Add receipt metadata while preserving every existing top-level result field."""

    if receipt is None:
        return result
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        visible = {
            field.name: getattr(result, field.name)
            for field in dataclasses.fields(result)
        }
        if "action_receipt" in visible:
            raise AgentToolError("Capability response shadows action_receipt.")
        visible["action_receipt"] = receipt
        return visible
    if isinstance(result, Mapping):
        visible = dict(result)
        if "action_receipt" in visible:
            raise AgentToolError("Capability response shadows action_receipt.")
        visible["action_receipt"] = receipt
        return visible
    return {"result": result, "action_receipt": receipt}
