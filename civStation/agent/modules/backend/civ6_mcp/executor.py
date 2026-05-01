"""civ6-mcp executor — invokes tool calls produced by the planner.

This is the civ6-mcp equivalent of `civStation/utils/screen.py:execute_action`,
except the "action" is a typed tool call against the upstream MCP server
rather than a normalized pixel coordinate.

The executor also pattern-matches civ6-mcp's text error semantics so the
turn loop can decide whether to retry, escalate, or stop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClient, Civ6McpError

logger = logging.getLogger(__name__)


_END_TURN_BLOCKED = re.compile(r"^Cannot end turn:", re.IGNORECASE)
_END_TURN_SOFT = re.compile(r"^End turn requested .* still ", re.IGNORECASE)
_GAME_OVER = re.compile(r"^\*\*\*\s*GAME OVER", re.I)
_RUN_ABORTED = re.compile(r"^RUN ABORTED", re.I)
_HANG_FAILED = re.compile(r"^HANG RECOVERY FAILED", re.I)
_GENERIC_ERROR = re.compile(r"^Error:", re.IGNORECASE)


@dataclass
class ToolCall:
    """A single planner-issued civ6-mcp tool invocation."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ToolCallResult:
    """Outcome of executing one ToolCall."""

    call: ToolCall
    success: bool = False
    text: str = ""
    error: str = ""
    classification: str = ""  # "ok", "blocked", "soft_block", "game_over", "aborted", "hang", "error"


class Civ6McpExecutor:
    """Synchronous executor that runs ToolCalls against a civ6-mcp client."""

    def __init__(self, client: Civ6McpClient) -> None:
        self._client = client

    def execute(self, call: ToolCall) -> ToolCallResult:
        """Run a single tool call and classify its outcome."""
        if not self._client.has_tool(call.tool):
            return ToolCallResult(
                call=call,
                success=False,
                error=f"Tool '{call.tool}' not exposed by civ6-mcp server.",
                classification="error",
            )
        try:
            text = self._client.call_tool(call.tool, call.arguments)
        except Civ6McpError as exc:
            return ToolCallResult(
                call=call,
                success=False,
                error=str(exc),
                classification="error",
            )

        classification = self._classify(text)
        success = classification in {"ok", "soft_block"}
        return ToolCallResult(
            call=call,
            success=success,
            text=text,
            error="" if success else text,
            classification=classification,
        )

    def execute_many(self, calls: list[ToolCall]) -> list[ToolCallResult]:
        """Run a sequence of tool calls; stop early on terminal classifications."""
        results: list[ToolCallResult] = []
        for call in calls:
            outcome = self.execute(call)
            results.append(outcome)
            if outcome.classification in {"game_over", "aborted", "hang"}:
                logger.warning(
                    "civ6-mcp executor stopping early: classification=%s tool=%s",
                    outcome.classification,
                    call.tool,
                )
                break
        return results

    @staticmethod
    def _classify(text: str) -> str:
        if not text:
            return "ok"
        if _GAME_OVER.search(text):
            return "game_over"
        if _RUN_ABORTED.search(text):
            return "aborted"
        if _HANG_FAILED.search(text):
            return "hang"
        if _END_TURN_BLOCKED.search(text):
            return "blocked"
        if _END_TURN_SOFT.search(text):
            return "soft_block"
        if _GENERIC_ERROR.search(text):
            return "error"
        return "ok"


def coerce_tool_calls(payload: Any) -> list[ToolCall]:
    """Normalize a planner JSON payload into a list of ToolCalls.

    Accepts either::

        {"tool_calls": [{"tool": "...", "arguments": {...}}, ...]}

    or just the list itself for resilience against minor format drift.
    """
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unexpected planner payload shape: {type(payload).__name__}")

    if not isinstance(items, list):
        raise ValueError("Planner payload 'tool_calls' must be a list.")

    calls: list[ToolCall] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"Tool call entry must be an object, got {type(raw).__name__}")
        name = raw.get("tool") or raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Tool call missing 'tool' name: {raw!r}")
        args = raw.get("arguments") or {}
        if not isinstance(args, dict):
            raise ValueError(f"Tool call arguments must be an object: {raw!r}")
        reasoning = str(raw.get("reasoning") or "")
        calls.append(ToolCall(tool=name, arguments=args, reasoning=reasoning))
    return calls
