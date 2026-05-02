"""civ6-mcp executor — invokes tool calls produced by the planner.

This is the civ6-mcp equivalent of `civStation/utils/screen.py:execute_action`,
except the "action" is a typed tool call against the upstream MCP server
rather than a normalized pixel coordinate.

The executor also pattern-matches civ6-mcp's text error semantics so the
turn loop can decide whether to retry, escalate, or stop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.action_mapping import (
    Civ6McpActionMappingError,
    map_civ6_mcp_action,
    map_civ6_mcp_actions,
)
from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol
from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
    classify_civ6_mcp_text,
)
from civStation.agent.modules.backend.civ6_mcp.results import (
    ToolCall,
    ToolCallResult,
    tool_call_result_from_dispatch,
)

_TERMINAL_CLASSIFICATIONS = frozenset({"game_over", "aborted", "hang"})
StopRequested = Callable[[], bool]


class Civ6McpExecutor:
    """Synchronous executor that runs ToolCalls against a civ6-mcp client."""

    def __init__(self, client: Civ6McpClientProtocol) -> None:
        self._dispatcher = Civ6McpOperationDispatcher(client)

    def execute(self, planned_action: Any) -> ToolCallResult:
        """Run one planned civ6-mcp action and classify its outcome.

        The public executor surface is intentionally structural: turn-loop code
        can pass a ``ToolCall``, a planner action with ``to_tool_call()``, or a
        planner JSON mapping. This keeps executor.py independent from
        planner_types.py while still accepting the backend action interface.
        """
        try:
            call = coerce_tool_call(planned_action)
        except ValueError as exc:
            return ToolCallResult(
                call=_invalid_tool_call(planned_action),
                success=False,
                error=str(exc),
                classification="error",
                status="fatal",
                terminal=False,
            )

        try:
            request = Civ6McpRequestBuilder.build(
                call.tool,
                call.arguments,
                reasoning=call.reasoning,
            )
        except ValueError as exc:
            return ToolCallResult(
                call=call,
                success=False,
                error=str(exc),
                classification="error",
                status="fatal",
                terminal=False,
            )

        outcome = self._dispatcher.dispatch(request)
        return tool_call_result_from_dispatch(call, outcome)

    def execute_many(
        self,
        calls: Iterable[Any],
        *,
        stop_requested: StopRequested | None = None,
    ) -> list[ToolCallResult]:
        """Run a sequence of tool calls; stop early on terminal classifications."""
        results: list[ToolCallResult] = []
        for call in calls:
            if stop_requested is not None and stop_requested():
                break
            outcome = self.execute(call)
            results.append(outcome)
            if _is_terminal_outcome(outcome):
                break
        return results

    @staticmethod
    def _classify(text: str) -> str:
        return classify_civ6_mcp_text(text)


def _is_terminal_outcome(outcome: ToolCallResult) -> bool:
    return outcome.terminal or outcome.classification in _TERMINAL_CLASSIFICATIONS


def coerce_tool_call(planned_action: Any) -> ToolCall:
    """Normalize one backend planned action into an executor ``ToolCall``."""
    try:
        return map_civ6_mcp_action(planned_action)
    except Civ6McpActionMappingError as exc:
        raise ValueError(str(exc)) from exc


def coerce_tool_calls(payload: Any) -> list[ToolCall]:
    """Normalize a planner JSON payload into a list of ToolCalls.

    Accepts either::

        {"tool_calls": [{"tool": "...", "arguments": {...}}, ...]}

    or just the list itself for resilience against minor format drift. Each
    entry may be a canonical tool call or a civ6-mcp free-form action type.
    """
    items = _tool_call_items_from_payload(payload)
    try:
        return map_civ6_mcp_actions(items)
    except Civ6McpActionMappingError as exc:
        raise ValueError(str(exc)) from exc


def _tool_call_items_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
    elif isinstance(payload, dict) and "actions" in payload:
        raise ValueError(
            "VLM/computer-use action-plan payload cannot run on the civ6-mcp backend; expected 'tool_calls'."
        )
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unexpected planner payload shape: {type(payload).__name__}")

    if not isinstance(items, list):
        raise ValueError("Planner payload 'tool_calls' must be a list.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Tool call entry must be an object, got {type(item).__name__}")
        if not any(item.get(key) for key in ("tool", "name", "type", "action_type", "action")):
            raise ValueError(f"Tool call missing 'tool' name: {item!r}")
    return list(items)


def _invalid_tool_call(raw: Any) -> ToolCall:
    if isinstance(raw, Mapping):
        tool = raw.get("tool") or raw.get("name") or "<invalid>"
        return ToolCall(tool=str(tool))
    tool = getattr(raw, "tool", None)
    return ToolCall(tool=str(tool or "<invalid>"))
