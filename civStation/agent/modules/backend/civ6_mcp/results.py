"""Executor-result adapters for civ6-mcp MCP responses.

This module is deliberately backend-local. It converts MCP response shapes
into the result objects consumed by the civ6-mcp executor without importing or
calling the VLM/computer-use screen pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.response import (
    Civ6McpNormalizedResult,
    normalize_mcp_response_error,
    normalize_mcp_response_timeout,
    normalize_mcp_tool_result,
)


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
    classification: str = ""  # "ok", "blocked", "soft_block", "game_over", "aborted", "hang", "error", "timeout"
    status: str = ""  # "success", "blocked", "retryable", "fatal", "aborted", "hang", "game_over"
    retryable: bool = False
    terminal: bool = False
    timed_out: bool = False
    normalized_response: Civ6McpNormalizedResult | None = None
    content_blocks: tuple[str, ...] = ()
    structured_content: Any | None = None
    raw_response: Any | None = None

    @property
    def response(self) -> Civ6McpNormalizedResult | None:
        """Captured normalized MCP response for callers that prefer a short name."""
        return self.normalized_response


def executor_result_from_normalized_response(
    call: ToolCall,
    response: Civ6McpNormalizedResult,
) -> ToolCallResult:
    """Convert a normalized civ6-mcp response into an executor result."""
    classification = _classification_value(response.classification)
    status = _classification_value(response.status)
    return ToolCallResult(
        call=call,
        success=response.success,
        text=response.text,
        error=response.error,
        classification=classification,
        status=status,
        retryable=_is_retryable_result(status, classification),
        terminal=_is_terminal_result(status, classification),
        timed_out=bool(response.timed_out or classification == "timeout"),
        normalized_response=response,
        content_blocks=response.content_blocks,
        structured_content=response.structured_content,
        raw_response=response.raw,
    )


def executor_result_from_mcp_tool_result(
    call: ToolCall,
    result: Any,
) -> ToolCallResult:
    """Normalize a raw MCP SDK tool result into an executor result."""
    response = normalize_mcp_tool_result(call.tool, call.arguments, result)
    return executor_result_from_normalized_response(call, response)


def executor_result_from_mcp_error(
    call: ToolCall,
    error: str,
    *,
    raw: Any | None = None,
) -> ToolCallResult:
    """Normalize a transport, JSON-RPC, or validation error into an executor result."""
    response = normalize_mcp_response_error(call.tool, call.arguments, error, raw=raw)
    return executor_result_from_normalized_response(call, response)


def executor_result_from_mcp_timeout(
    call: ToolCall,
    *,
    timeout_seconds: float,
) -> ToolCallResult:
    """Normalize an MCP tool-call timeout into an executor result."""
    response = normalize_mcp_response_timeout(call.tool, call.arguments, timeout_seconds=timeout_seconds)
    return executor_result_from_normalized_response(call, response)


def tool_call_result_from_dispatch(call: ToolCall, dispatch_result: Any) -> ToolCallResult:
    """Convert a dispatcher result into the public executor result shape."""
    response = getattr(dispatch_result, "response", None)
    if isinstance(response, Civ6McpNormalizedResult):
        return executor_result_from_normalized_response(call, response)

    classification = str(getattr(dispatch_result, "classification", "") or "")
    status = str(getattr(dispatch_result, "status", "") or "")
    raw_response = getattr(dispatch_result, "raw_response", response)
    return ToolCallResult(
        call=call,
        success=bool(getattr(dispatch_result, "success", False)),
        text=str(getattr(dispatch_result, "text", "") or ""),
        error=str(getattr(dispatch_result, "error", "") or ""),
        classification=classification,
        status=status,
        retryable=_is_retryable_result(status, classification),
        terminal=_is_terminal_result(status, classification),
        timed_out=classification == "timeout",
        content_blocks=_content_blocks_from_dispatch(dispatch_result),
        structured_content=getattr(dispatch_result, "structured_content", None),
        raw_response=raw_response,
    )


def _classification_value(classification: Any) -> str:
    value = getattr(classification, "value", classification)
    return str(value or "")


def _is_retryable_result(status: str, classification: str) -> bool:
    return status == "retryable" or classification in {"soft_block", "timeout"}


def _is_terminal_result(status: str, classification: str) -> bool:
    return status in {"aborted", "hang", "game_over"} or classification in {"aborted", "hang", "game_over"}


def _content_blocks_from_dispatch(dispatch_result: Any) -> tuple[str, ...]:
    blocks = getattr(dispatch_result, "content_blocks", ())
    if isinstance(blocks, tuple):
        return tuple(str(block) for block in blocks)
    if isinstance(blocks, list):
        return tuple(str(block) for block in blocks)
    return ()


__all__ = [
    "ToolCall",
    "ToolCallResult",
    "executor_result_from_mcp_error",
    "executor_result_from_mcp_timeout",
    "executor_result_from_mcp_tool_result",
    "executor_result_from_normalized_response",
    "tool_call_result_from_dispatch",
]
