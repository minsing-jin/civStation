"""Map civStation civ6-mcp planned actions into upstream MCP tool calls.

The VLM backend uses pixel actions (`click`, `press`, `drag`, ...). The
civ6-mcp backend instead executes JSON-RPC tool calls, so this module is the
small boundary that accepts backend-specific planned actions and produces the
executor's `ToolCall` shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.operations import (
    END_TURN_REFLECTION_FIELDS,
    END_TURN_TOOL,
    SUPPORTED_CIV6_MCP_TOOLS,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.results import ToolCall

_DIRECT_TOOL_CALL_TYPES = frozenset(
    {
        "tool_call",
        "mcp_tool_call",
        "civ6_mcp_tool_call",
        "civ6-mcp_tool_call",
    }
)
_VLM_ACTION_TYPES = frozenset({"click", "double_click", "press", "drag", "wait"})
_METADATA_FIELDS = frozenset(
    {
        "action",
        "action_type",
        "arguments",
        "args",
        "description",
        "name",
        "operation",
        "payload",
        "reason",
        "reasoning",
        "reflections",
        "tool",
        "tool_name",
        "type",
    }
)


class Civ6McpActionMappingError(ValueError):
    """Raised when a planned action cannot resolve to a supported civ6-mcp tool call."""


@dataclass(frozen=True)
class Civ6McpFreeFormActionType:
    """Describe a free-form civ6-mcp action type and its upstream tool target."""

    action_type: str
    tool: str
    aliases: tuple[str, ...] = ()

    @property
    def accepted_types(self) -> tuple[str, ...]:
        """Return the canonical action type plus all planner-facing aliases."""
        return (self.action_type, *self.aliases)


_CORE_FREE_FORM_ACTION_TYPES: tuple[Civ6McpFreeFormActionType, ...] = (
    Civ6McpFreeFormActionType(
        action_type="research",
        tool="set_research",
        aliases=("choose_research", "set_research"),
    ),
    Civ6McpFreeFormActionType(
        action_type="city_production",
        tool="set_city_production",
        aliases=("choose_production", "set_city_production"),
    ),
    Civ6McpFreeFormActionType(
        action_type="city_focus",
        tool="set_city_focus",
        aliases=("set_city_focus",),
    ),
    Civ6McpFreeFormActionType(
        action_type="purchase",
        tool="purchase_item",
        aliases=("purchase_item", "buy_item"),
    ),
    Civ6McpFreeFormActionType(
        action_type="purchase_tile",
        tool="purchase_tile",
        aliases=("buy_tile",),
    ),
    Civ6McpFreeFormActionType(
        action_type="government",
        tool="change_government",
        aliases=("change_government",),
    ),
    Civ6McpFreeFormActionType(
        action_type="set_policies",
        tool="set_policies",
        aliases=("policies", "policy_cards"),
    ),
    Civ6McpFreeFormActionType(
        action_type="end_turn",
        tool=END_TURN_TOOL,
        aliases=("finish_turn",),
    ),
)


def _build_free_form_action_type_registry() -> dict[str, Civ6McpFreeFormActionType]:
    registry = {entry.action_type: entry for entry in _CORE_FREE_FORM_ACTION_TYPES}
    reserved_action_types = {
        accepted_type for entry in _CORE_FREE_FORM_ACTION_TYPES for accepted_type in entry.accepted_types
    }
    for tool in sorted(SUPPORTED_CIV6_MCP_TOOLS):
        if tool not in reserved_action_types:
            registry[tool] = Civ6McpFreeFormActionType(action_type=tool, tool=tool)
    return registry


def _build_free_form_action_type_aliases(
    registry: Mapping[str, Civ6McpFreeFormActionType],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for action_type, entry in registry.items():
        for accepted_type in entry.accepted_types:
            existing = aliases.setdefault(accepted_type, action_type)
            if existing != action_type:
                raise RuntimeError(
                    f"Duplicate civ6-mcp free-form action type alias {accepted_type!r}: "
                    f"{existing!r} and {action_type!r}"
                )
    return aliases


CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY: dict[str, Civ6McpFreeFormActionType] = _build_free_form_action_type_registry()
"""Canonical civ6-mcp free-form action types keyed by action type."""

CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES: dict[str, str] = _build_free_form_action_type_aliases(
    CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY
)
"""Accepted civ6-mcp free-form action type aliases mapped to canonical types."""

CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL: dict[str, str] = {
    accepted_type: CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY[canonical_type].tool
    for accepted_type, canonical_type in CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES.items()
}
"""Accepted civ6-mcp free-form action type aliases mapped to upstream tools."""

PLANNED_ACTION_TYPE_TO_MCP_TOOL: dict[str, str] = dict(CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL)
"""Backward-compatible alias for the civ6-mcp planned action type map."""


@dataclass(frozen=True)
class MappedCiv6McpAction:
    """Store the canonical tool call metadata produced by action mapping."""

    tool: str
    arguments: dict[str, Any]
    reasoning: str = ""

    def to_tool_call(self) -> ToolCall:
        """Build the executor-facing tool call from canonical mapping metadata."""
        return ToolCall(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )


def map_civ6_mcp_action(planned_action: Any) -> ToolCall:
    """Map one supported planner action shape into a validated executor `ToolCall`."""
    mapped = map_civ6_mcp_action_details(planned_action)
    return mapped.to_tool_call()


def map_civ6_mcp_action_details(planned_action: Any) -> MappedCiv6McpAction:
    """Map one supported planner action while retaining canonical tool metadata."""
    try:
        if isinstance(planned_action, ToolCall):
            return _validate_tool_call(planned_action)

        to_tool_call = getattr(planned_action, "to_tool_call", None)
        if callable(to_tool_call):
            converted = to_tool_call()
            if not isinstance(converted, ToolCall):
                raise Civ6McpActionMappingError(
                    f"planned action to_tool_call() must return ToolCall, got {type(converted).__name__}"
                )
            return _validate_tool_call(converted)

        if isinstance(planned_action, Mapping):
            return _map_mapping_action(planned_action)

        object_mapping = _mapping_from_object(planned_action)
        if object_mapping is not None:
            return _map_mapping_action(object_mapping)
    except Civ6McpActionMappingError:
        raise
    except ValueError as exc:
        raise Civ6McpActionMappingError(str(exc)) from exc

    raise Civ6McpActionMappingError(f"Unsupported planned action shape: {type(planned_action).__name__}")


def map_civ6_mcp_actions(planned_actions: Iterable[Any]) -> list[ToolCall]:
    """Map an iterable of planner actions into validated executor tool calls."""
    if isinstance(planned_actions, Mapping) or isinstance(planned_actions, str | bytes):
        raise Civ6McpActionMappingError("planned actions must be an iterable of action objects, not a single action")
    return [map_civ6_mcp_action(action) for action in planned_actions]


def _validate_tool_call(call: ToolCall) -> MappedCiv6McpAction:
    _raise_if_vlm_action_type(call.tool)
    request = Civ6McpRequestBuilder.build(
        call.tool,
        call.arguments,
        reasoning=call.reasoning,
    )
    return MappedCiv6McpAction(
        tool=request.tool,
        arguments=dict(request.arguments),
        reasoning=request.reasoning,
    )


def _map_mapping_action(raw: Mapping[str, Any]) -> MappedCiv6McpAction:
    action_type = _coerce_action_type(raw)
    _raise_if_vlm_action_type(action_type)

    tool = _tool_for_mapping(raw, action_type)
    _raise_if_vlm_action_type(tool)
    arguments = _arguments_for_tool(tool, raw)
    reasoning = _reasoning_for_mapping(raw)
    request = Civ6McpRequestBuilder.build(tool, arguments, reasoning=reasoning)
    return MappedCiv6McpAction(
        tool=request.tool,
        arguments=dict(request.arguments),
        reasoning=request.reasoning,
    )


def _coerce_action_type(raw: Mapping[str, Any]) -> str:
    value = raw.get("type") or raw.get("action_type") or raw.get("action")
    if value is None and (raw.get("tool") or raw.get("name") or raw.get("tool_name")):
        return "tool_call"
    if not isinstance(value, str) or not value.strip():
        raise Civ6McpActionMappingError(f"civ6-mcp planned action missing non-empty type/tool: {dict(raw)!r}")
    return value.strip()


def _mapping_from_object(raw: Any) -> dict[str, Any] | None:
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)

    if not any(hasattr(raw, field) for field in ("type", "action_type", "action", "tool", "tool_name", "name")):
        return None

    try:
        values = dict(vars(raw))
    except TypeError:
        values = {}
    for field in ("type", "action_type", "action", "tool", "tool_name", "name", "arguments", "reasoning", "reason"):
        if hasattr(raw, field):
            values[field] = getattr(raw, field)
    return values


def _tool_for_mapping(raw: Mapping[str, Any], action_type: str) -> str:
    explicit_tool = raw.get("tool") or raw.get("tool_name") or raw.get("name")
    if action_type in _DIRECT_TOOL_CALL_TYPES:
        if not isinstance(explicit_tool, str) or not explicit_tool.strip():
            raise Civ6McpActionMappingError("civ6-mcp tool_call action requires a non-empty tool/name field.")
        return explicit_tool.strip()

    mapped_tool = PLANNED_ACTION_TYPE_TO_MCP_TOOL.get(action_type)
    if mapped_tool is None:
        raise Civ6McpActionMappingError(f"Unsupported civ6-mcp planned action type: {action_type!r}")

    if isinstance(explicit_tool, str) and explicit_tool.strip() and explicit_tool.strip() != mapped_tool:
        raise Civ6McpActionMappingError(
            f"civ6-mcp planned action type {action_type!r} maps to {mapped_tool!r}, not {explicit_tool!r}."
        )
    return mapped_tool


def _raise_if_vlm_action_type(action_type: str) -> None:
    if action_type in _VLM_ACTION_TYPES:
        raise Civ6McpActionMappingError(f"VLM/computer-use action {action_type!r} cannot run on the civ6-mcp backend.")


def _arguments_for_tool(tool: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    arguments = _inline_arguments(raw)
    arguments.update(_extract_argument_mapping(raw))

    if tool == "set_research":
        _normalize_set_research_arguments(arguments)
    elif tool == "set_city_production":
        _normalize_set_city_production_arguments(arguments)
    elif tool == END_TURN_TOOL:
        _normalize_end_turn_arguments(arguments, raw)

    return arguments


def _extract_argument_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("arguments", "args", "payload"):
        if key not in raw or raw[key] is None:
            continue
        value = raw[key]
        if not isinstance(value, Mapping):
            raise Civ6McpActionMappingError(f"civ6-mcp planned action {key} must be an object.")
        return dict(value)
    return {}


def _inline_arguments(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in raw.items() if str(key) not in _METADATA_FIELDS}


def _normalize_set_research_arguments(arguments: dict[str, Any]) -> None:
    if "tech_or_civic" in arguments:
        return
    for alias in ("target", "research", "tech", "civic"):
        if alias in arguments:
            arguments["tech_or_civic"] = arguments.pop(alias)
            return


def _normalize_set_city_production_arguments(arguments: dict[str, Any]) -> None:
    if "production" in arguments:
        return
    for alias in ("item", "production_item", "build"):
        if alias in arguments:
            arguments["production"] = arguments.pop(alias)
            return


def _normalize_end_turn_arguments(arguments: dict[str, Any], raw: Mapping[str, Any]) -> None:
    reflections = raw.get("reflections")
    if reflections is not None:
        if not isinstance(reflections, Mapping):
            raise Civ6McpActionMappingError("civ6-mcp end_turn reflections must be an object.")
        for field in END_TURN_REFLECTION_FIELDS:
            if field not in arguments and field in reflections:
                arguments[field] = reflections[field]


def _reasoning_for_mapping(raw: Mapping[str, Any]) -> str:
    reasoning = raw.get("reasoning", raw.get("reason", ""))
    return str(reasoning or "")


__all__ = [
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL",
    "Civ6McpActionMappingError",
    "Civ6McpFreeFormActionType",
    "MappedCiv6McpAction",
    "PLANNED_ACTION_TYPE_TO_MCP_TOOL",
    "map_civ6_mcp_action",
    "map_civ6_mcp_action_details",
    "map_civ6_mcp_actions",
]
