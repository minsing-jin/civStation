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

from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.operations import (
    END_TURN_REFLECTION_FIELDS,
    END_TURN_TOOL,
    SUPPORTED_CIV6_MCP_TOOLS,
    Civ6McpRequestBuilder,
)

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

PLANNED_ACTION_TYPE_TO_MCP_TOOL: dict[str, str] = {
    "research": "set_research",
    "choose_research": "set_research",
    "set_research": "set_research",
    "city_production": "set_city_production",
    "choose_production": "set_city_production",
    "set_city_production": "set_city_production",
    "city_focus": "set_city_focus",
    "set_city_focus": "set_city_focus",
    "purchase": "purchase_item",
    "purchase_item": "purchase_item",
    "buy_item": "purchase_item",
    "purchase_tile": "purchase_tile",
    "buy_tile": "purchase_tile",
    "government": "change_government",
    "change_government": "change_government",
    "policies": "set_policies",
    "set_policies": "set_policies",
    "unit_action": "unit_action",
    "city_action": "city_action",
    "skip_remaining_units": "skip_remaining_units",
    "end_turn": END_TURN_TOOL,
}
PLANNED_ACTION_TYPE_TO_MCP_TOOL.update({tool: tool for tool in SUPPORTED_CIV6_MCP_TOOLS})


class Civ6McpActionMappingError(ValueError):
    """Raised when a planned action cannot be mapped to a civ6-mcp tool call."""


@dataclass(frozen=True)
class MappedCiv6McpAction:
    """A validated civ6-mcp action mapping result."""

    tool: str
    arguments: dict[str, Any]
    reasoning: str = ""

    def to_tool_call(self) -> ToolCall:
        """Return the executor-facing tool call."""
        return ToolCall(
            tool=self.tool,
            arguments=dict(self.arguments),
            reasoning=self.reasoning,
        )


def map_civ6_mcp_action(planned_action: Any) -> ToolCall:
    """Translate one supported planned action into a validated `ToolCall`.

    Supported inputs are:
    - `ToolCall` or any object exposing `to_tool_call()`.
    - mappings with direct `tool`/`name` fields.
    - mappings whose `type` names a supported civ6-mcp tool or semantic alias.
    """
    mapped = map_civ6_mcp_action_details(planned_action)
    return mapped.to_tool_call()


def map_civ6_mcp_action_details(planned_action: Any) -> MappedCiv6McpAction:
    """Translate one supported planned action and retain mapping metadata."""
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
    """Translate a sequence of planned actions into executor tool calls."""
    if isinstance(planned_actions, Mapping) or isinstance(planned_actions, str | bytes):
        raise Civ6McpActionMappingError("planned actions must be an iterable of action objects, not a single action")
    return [map_civ6_mcp_action(action) for action in planned_actions]


def _validate_tool_call(call: ToolCall) -> MappedCiv6McpAction:
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
    if action_type in _VLM_ACTION_TYPES:
        raise Civ6McpActionMappingError(f"VLM/computer-use action {action_type!r} cannot run on the civ6-mcp backend.")

    tool = _tool_for_mapping(raw, action_type)
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

    if not any(hasattr(raw, field) for field in ("type", "action_type", "action")):
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
    "Civ6McpActionMappingError",
    "MappedCiv6McpAction",
    "PLANNED_ACTION_TYPE_TO_MCP_TOOL",
    "map_civ6_mcp_action",
    "map_civ6_mcp_action_details",
    "map_civ6_mcp_actions",
]
