"""Bounded dynamic-tool bridge from an agent provider to typed capabilities."""

from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, cast, get_args, get_origin, get_type_hints

from simajilord.core import ApprovalMode, CapabilityRegistry, InvocationContext, RiskLevel

from .errors import AgentToolError

_TOOL_NAMESPACE = "simajilord"
_SEARCH_TOOL = "capability_search"
_INVOKE_TOOL = "capability_invoke"


@dataclasses.dataclass(frozen=True, slots=True)
class AgentToolOutput:
    """Model-facing text plus optional media carried outside the text budget."""

    text: str
    image_url: str | None = None

    def __len__(self) -> int:
        return len(self.text)

    def __contains__(self, value: str) -> bool:
        return value in self.text


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
        image_output_capabilities: Sequence[str] = (),
    ) -> None:
        self._registry = registry
        self._allowed_capabilities = tuple(allowed_capabilities)
        self._required_grants = dict(required_grants or {})
        self._eager_capabilities = (
            frozenset(self._allowed_capabilities)
            if eager_capabilities is None
            else frozenset(eager_capabilities)
        )
        self._write_capabilities = frozenset(write_capabilities)
        self._image_output_capabilities = frozenset(image_output_capabilities)
        allowed = set(self._allowed_capabilities)
        unknown_eager = self._eager_capabilities - allowed
        unknown_writes = self._write_capabilities - allowed
        unknown_images = self._image_output_capabilities - allowed
        if unknown_eager or unknown_writes or unknown_images:
            unknown = sorted(unknown_eager | unknown_writes | unknown_images)
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
        aliases: dict[str, str] = {}
        for capability_name in self._allowed_capabilities:
            alias = _tool_alias(capability_name)
            if alias in {_SEARCH_TOOL, _INVOKE_TOOL}:
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

    def dynamic_specs(
        self,
        context: InvocationContext | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """Build app-server dynamic tool specs from currently registered endpoints."""

        tools: list[Mapping[str, object]] = []
        hidden_available = False
        for alias, capability_name in sorted(self._aliases.items()):
            if not self._is_available(capability_name, context):
                continue
            endpoint = self._validated_endpoint(capability_name, context)
            if capability_name not in self._eager_capabilities:
                hidden_available = True
                continue
            tools.append(
                {
                    "type": "function",
                    "name": alias,
                    "description": endpoint.descriptor.summary,
                    "inputSchema": _dataclass_schema(endpoint.request_type),
                }
            )
        if hidden_available:
            tools.extend((_search_spec(), _invoke_spec()))
        if not tools:
            return ()
        return (
            {
                "type": "namespace",
                "name": self.namespace,
                "description": (
                    "Typed Simajilord capabilities. Search only when a requested action "
                    "needs a capability that is not already shown."
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
    ) -> AgentToolOutput:
        if namespace != self.namespace:
            raise AgentToolError("The dynamic tool namespace is not allowed.")
        if tool_name == _SEARCH_TOOL:
            return AgentToolOutput(
                self._search(
                    arguments,
                    context=context,
                    max_output_characters=max_output_characters,
                )
            )
        if tool_name == _INVOKE_TOOL:
            return await self._invoke_discovered(
                arguments,
                context=context,
                max_output_characters=max_output_characters,
            )
        try:
            capability_name = self._aliases[tool_name]
        except KeyError as exc:
            raise AgentToolError("The dynamic tool is not allowed.") from exc
        if capability_name not in self._eager_capabilities:
            raise AgentToolError("Use capability_invoke for discovered capabilities.")
        return await self._invoke_capability(
            capability_name,
            arguments,
            context=context,
            max_output_characters=max_output_characters,
        )

    def _search(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
    ) -> str:
        if not isinstance(arguments, dict):
            raise AgentToolError("Capability search arguments must be an object.")
        unknown = set(arguments) - {"query", "limit"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability search fields: {', '.join(sorted(unknown))}"
            )
        query = arguments.get("query")
        limit = arguments.get("limit", 3)
        if not isinstance(query, str) or not query.strip():
            raise AgentToolError("Capability search query must be non-empty text.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5:
            raise AgentToolError("Capability search limit must be between 1 and 5.")
        available = {
            name
            for name in self._allowed_capabilities
            if self._is_available(name, context)
        }
        matches = tuple(
            item
            for item in self._registry.search(
                query,
                limit=max(1, len(self._registry.all())),
            )
            if item.descriptor.name in available
        )[:limit]
        payload = {
            "query": query,
            "matches": [
                {
                    "name": item.descriptor.name,
                    "summary": item.descriptor.summary,
                    "risk": item.descriptor.risk.value,
                    "input_schema": _dataclass_schema(item.request_type),
                }
                for item in matches
            ],
        }
        return _bounded_json(payload, max_output_characters=max_output_characters)

    async def _invoke_discovered(
        self,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
    ) -> AgentToolOutput:
        if not isinstance(arguments, dict):
            raise AgentToolError("Capability invocation arguments must be an object.")
        unknown = set(arguments) - {"name", "arguments"}
        if unknown:
            raise AgentToolError(
                f"Unknown capability invocation fields: {', '.join(sorted(unknown))}"
            )
        capability_name = arguments.get("name")
        capability_arguments = arguments.get("arguments")
        if not isinstance(capability_name, str):
            raise AgentToolError("Capability name must be text.")
        if capability_name not in self._allowed_capabilities:
            raise AgentToolError("The capability is not allowed.")
        if capability_name in self._eager_capabilities:
            raise AgentToolError("Use the dedicated dynamic tool for this capability.")
        return await self._invoke_capability(
            capability_name,
            capability_arguments,
            context=context,
            max_output_characters=max_output_characters,
        )

    async def _invoke_capability(
        self,
        capability_name: str,
        arguments: object,
        *,
        context: InvocationContext,
        max_output_characters: int,
    ) -> AgentToolOutput:
        if not self._is_available(capability_name, context):
            raise AgentToolError("The dynamic tool grant is not present for this turn.")
        endpoint = self._validated_endpoint(capability_name, context)
        request = _build_dataclass(endpoint.request_type, arguments)
        result = await self._registry.invoke(capability_name, request, context)
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
            return AgentToolOutput(
                _bounded_json(visible, max_output_characters=max_output_characters),
                image_url=image_url,
            )
        return AgentToolOutput(
            _bounded_json(result, max_output_characters=max_output_characters)
        )

    def _is_available(
        self,
        capability_name: str,
        context: InvocationContext | None,
    ) -> bool:
        required_grant = self._required_grants.get(capability_name)
        has_grant = required_grant is None or (
            context is not None and required_grant in context.grants
        )
        if not has_grant:
            return False
        descriptor = self._registry.endpoint(capability_name).descriptor
        if descriptor.approval is ApprovalMode.NEVER:
            return True
        if descriptor.approval is ApprovalMode.WHEN_REQUESTED:
            return context is not None and capability_name in context.approvals
        return False

    def _validated_endpoint(
        self,
        capability_name: str,
        context: InvocationContext | None,
    ) -> Any:
        endpoint = self._registry.endpoint(capability_name)
        descriptor = endpoint.descriptor
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
        if descriptor.risk is RiskLevel.DESTRUCTIVE:
            raise AgentToolError(
                f"Agent catalog cannot expose destructive {capability_name}."
            )
        if descriptor.risk is RiskLevel.EXTERNAL and required_grant is None:
            raise AgentToolError(
                f"Agent external catalog requires a grant for {capability_name}."
            )
        if descriptor.risk is RiskLevel.WRITE and (
            capability_name not in self._write_capabilities or required_grant is None
        ):
            raise AgentToolError(
                f"Agent catalog cannot expose unapproved write {capability_name}."
            )
        return endpoint


def _search_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _SEARCH_TOOL,
        "description": "Find a small relevant set of available Simajilord capabilities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _invoke_spec() -> Mapping[str, object]:
    return {
        "type": "function",
        "name": _INVOKE_TOOL,
        "description": "Invoke one capability returned by capability_search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        },
    }


def _tool_alias(capability_name: str) -> str:
    alias = capability_name.replace(".", "_").replace("-", "_")
    if not alias or len(alias) > 128 or not alias.replace("_", "").isalnum():
        raise AgentToolError(f"Capability name cannot become a dynamic tool: {capability_name}")
    return alias


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


def _bounded_json(value: object, *, max_output_characters: int) -> str:
    if max_output_characters < 200:
        raise AgentToolError("Dynamic tool output budget is too small.")
    encoded = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded) <= max_output_characters:
        return encoded
    wrapper = json.dumps(
        {
            "truncated": True,
            "reason": "agent_tool_output_budget",
            "preview": encoded[: max_output_characters - 120],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return wrapper[:max_output_characters]


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
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)
