"""civ6-mcp executor — invokes tool calls produced by the planner.

This is the civ6-mcp equivalent of `civStation/utils/screen.py:execute_action`,
except the "action" is a typed tool call against the upstream MCP server
rather than a normalized pixel coordinate.

The executor also pattern-matches civ6-mcp's text error semantics so the
turn loop can decide whether to retry, escalate, or stop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol
from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
    classify_civ6_mcp_text,
    coerce_civ6_mcp_requests,
)
from civStation.agent.modules.backend.civ6_mcp.results import (
    ToolCall,
    ToolCallResult,
    tool_call_result_from_dispatch,
)

_TERMINAL_CLASSIFICATIONS = frozenset({"game_over", "aborted", "hang"})


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
            )

        outcome = self._dispatcher.dispatch(request)
        return tool_call_result_from_dispatch(call, outcome)

    def execute_many(self, calls: Iterable[Any]) -> list[ToolCallResult]:
        """Run a sequence of tool calls; stop early on terminal classifications."""
        results: list[ToolCallResult] = []
        for call in calls:
            outcome = self.execute(call)
            results.append(outcome)
            if outcome.classification in _TERMINAL_CLASSIFICATIONS:
                break
        return results

    @staticmethod
    def _classify(text: str) -> str:
        return classify_civ6_mcp_text(text)


def coerce_tool_call(planned_action: Any) -> ToolCall:
    """Normalize one backend planned action into an executor ``ToolCall``."""
    if isinstance(planned_action, ToolCall):
        return planned_action

    to_tool_call = getattr(planned_action, "to_tool_call", None)
    if callable(to_tool_call):
        converted = to_tool_call()
        if not isinstance(converted, ToolCall):
            raise ValueError(f"planned action to_tool_call() must return ToolCall, got {type(converted).__name__}")
        return converted

    if isinstance(planned_action, Mapping):
        return _tool_call_from_mapping(planned_action)

    tool = getattr(planned_action, "tool", None)
    if tool is not None:
        return _tool_call_from_attrs(planned_action)

    raise ValueError(f"Unsupported planned action shape: {type(planned_action).__name__}")


def coerce_tool_calls(payload: Any) -> list[ToolCall]:
    """Normalize a planner JSON payload into a list of ToolCalls.

    Accepts either::

        {"tool_calls": [{"tool": "...", "arguments": {...}}, ...]}

    or just the list itself for resilience against minor format drift.
    """
    return [
        ToolCall(
            tool=request.tool,
            arguments=request.arguments,
            reasoning=request.reasoning,
        )
        for request in coerce_civ6_mcp_requests(payload)
    ]


def _tool_call_from_mapping(raw: Mapping[str, Any]) -> ToolCall:
    tool = raw.get("tool") or raw.get("name")
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"Tool call missing 'tool' name: {dict(raw)!r}")
    arguments = raw.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be an object.")
    return ToolCall(
        tool=tool,
        arguments=dict(arguments),
        reasoning=str(raw.get("reasoning") or ""),
    )


def _tool_call_from_attrs(raw: Any) -> ToolCall:
    tool = getattr(raw, "tool", None)
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"Tool call missing 'tool' name on {type(raw).__name__}.")
    arguments = getattr(raw, "arguments", {}) or {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool call arguments must be an object.")
    return ToolCall(
        tool=tool,
        arguments=dict(arguments),
        reasoning=str(getattr(raw, "reasoning", "") or ""),
    )


def _invalid_tool_call(raw: Any) -> ToolCall:
    if isinstance(raw, Mapping):
        tool = raw.get("tool") or raw.get("name") or "<invalid>"
        return ToolCall(tool=str(tool))
    tool = getattr(raw, "tool", None)
    return ToolCall(tool=str(tool or "<invalid>"))
