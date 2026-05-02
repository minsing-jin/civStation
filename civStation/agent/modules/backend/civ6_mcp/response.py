"""MCP response parsing and typed normalization for the civ6-mcp backend."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Civ6McpResponseClassification(str, Enum):
    """Normalized civ6-mcp response classes understood by the backend."""

    OK = "ok"
    SOFT_BLOCK = "soft_block"
    BLOCKED = "blocked"
    GAME_OVER = "game_over"
    ABORTED = "aborted"
    HANG = "hang"
    ERROR = "error"
    TIMEOUT = "timeout"


_END_TURN_BLOCKED = re.compile(r"^Cannot end turn:", re.IGNORECASE)
_END_TURN_SOFT = re.compile(r"^End turn requested .* still ", re.IGNORECASE)
_GAME_OVER = re.compile(r"^\*\*\*\s*GAME OVER", re.I)
_RUN_ABORTED = re.compile(r"^RUN ABORTED", re.I)
_HANG_FAILED = re.compile(r"^HANG RECOVERY FAILED", re.I)
_GENERIC_ERROR = re.compile(r"^Error:", re.IGNORECASE)
_TIMEOUT = re.compile(r"\btimed out after [0-9.]+s\b", re.IGNORECASE)


@dataclass(frozen=True)
class Civ6McpNormalizedResult:
    """Typed, backend-local representation of one MCP tool response."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    text: str = ""
    error: str = ""
    classification: Civ6McpResponseClassification = Civ6McpResponseClassification.OK
    is_error: bool = False
    timed_out: bool = False
    content_blocks: tuple[str, ...] = ()
    structured_content: Any | None = None
    raw: Any | None = None


def classify_civ6_mcp_text(text: str) -> Civ6McpResponseClassification:
    """Classify civ6-mcp's text-first success/error semantics."""
    if not text:
        return Civ6McpResponseClassification.OK
    if _TIMEOUT.search(text):
        return Civ6McpResponseClassification.TIMEOUT
    if _GAME_OVER.search(text):
        return Civ6McpResponseClassification.GAME_OVER
    if _RUN_ABORTED.search(text):
        return Civ6McpResponseClassification.ABORTED
    if _HANG_FAILED.search(text):
        return Civ6McpResponseClassification.HANG
    if _END_TURN_BLOCKED.search(text):
        return Civ6McpResponseClassification.BLOCKED
    if _END_TURN_SOFT.search(text):
        return Civ6McpResponseClassification.SOFT_BLOCK
    if _GENERIC_ERROR.search(text):
        return Civ6McpResponseClassification.ERROR
    return Civ6McpResponseClassification.OK


def normalize_mcp_tool_result(
    tool: str,
    arguments: dict[str, Any] | None,
    result: Any,
) -> Civ6McpNormalizedResult:
    """Parse an MCP SDK CallToolResult into a typed backend result."""
    args = dict(arguments or {})
    content_blocks = tuple(_extract_text_blocks(getattr(result, "content", []) or []))
    text = "\n".join(content_blocks).strip()
    is_error = bool(getattr(result, "isError", False))
    structured_content = _get_structured_content(result)
    classification = Civ6McpResponseClassification.ERROR if is_error else classify_civ6_mcp_text(text)
    success = _is_success(classification) and not is_error
    error = text if (is_error or not success) else ""

    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=args,
        success=success,
        text=text,
        error=error,
        classification=classification,
        is_error=is_error,
        content_blocks=content_blocks,
        structured_content=structured_content,
        raw=result,
    )


def normalize_mcp_response_text(
    tool: str,
    arguments: dict[str, Any] | None,
    text: str,
) -> Civ6McpNormalizedResult:
    """Normalize a legacy text-only MCP response."""
    args = dict(arguments or {})
    body = str(text or "").strip()
    classification = classify_civ6_mcp_text(body)
    success = _is_success(classification)
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=args,
        success=success,
        text=body,
        error="" if success else body,
        classification=classification,
        content_blocks=(body,) if body else (),
        raw=text,
    )


def normalize_mcp_response_error(
    tool: str,
    arguments: dict[str, Any] | None,
    error: str,
    *,
    raw: Any | None = None,
) -> Civ6McpNormalizedResult:
    """Normalize a transport, JSON-RPC, or validation error."""
    args = dict(arguments or {})
    message = str(error or "").strip()
    classification = classify_civ6_mcp_text(message)
    if classification == Civ6McpResponseClassification.OK:
        classification = Civ6McpResponseClassification.ERROR
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=args,
        success=False,
        text="",
        error=message,
        classification=classification,
        is_error=True,
        timed_out=classification == Civ6McpResponseClassification.TIMEOUT,
        raw=raw,
    )


def normalize_mcp_response_timeout(
    tool: str,
    arguments: dict[str, Any] | None,
    *,
    timeout_seconds: float,
) -> Civ6McpNormalizedResult:
    """Normalize a tool-call timeout without losing the timeout signal."""
    message = f"civ6-mcp tool '{tool}' timed out after {timeout_seconds:g}s"
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=dict(arguments or {}),
        success=False,
        error=message,
        classification=Civ6McpResponseClassification.TIMEOUT,
        timed_out=True,
        is_error=True,
    )


def _extract_text_blocks(content: list[Any]) -> list[str]:
    text_parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
            continue
        if isinstance(block, dict):
            dict_text = block.get("text")
            if isinstance(dict_text, str):
                text_parts.append(dict_text)
    return text_parts


def _get_structured_content(result: Any) -> Any | None:
    for name in ("structuredContent", "structured_content"):
        if hasattr(result, name):
            value = getattr(result, name)
            if value is not None:
                return _dump_model(value)
    return None


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    try:
        return json.loads(json.dumps(value))
    except TypeError:
        return str(value)


def _is_success(classification: Civ6McpResponseClassification) -> bool:
    return classification in {
        Civ6McpResponseClassification.OK,
        Civ6McpResponseClassification.SOFT_BLOCK,
    }


__all__ = [
    "Civ6McpNormalizedResult",
    "Civ6McpResponseClassification",
    "classify_civ6_mcp_text",
    "normalize_mcp_response_error",
    "normalize_mcp_response_text",
    "normalize_mcp_response_timeout",
    "normalize_mcp_tool_result",
]
