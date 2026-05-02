"""MCP response parsing and typed normalization for the civ6-mcp backend."""

from __future__ import annotations

import re
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from re import Pattern
from typing import Any

from civStation.agent.modules.backend.civ6_mcp._payload import (
    dump_model,
    extract_text_blocks,
    payload_value,
)


class Civ6McpClassificationStatus(str, Enum):
    """Canonical status taxonomy for normalized civ6-mcp outcomes."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    RETRYABLE = "retryable"
    FATAL = "fatal"
    ABORTED = "aborted"
    HANG = "hang"
    GAME_OVER = "game_over"


class Civ6McpResponseClassification(str, Enum):
    """Backward-compatible legacy response classifications exposed to existing callers."""

    OK = "ok"
    SOFT_BLOCK = "soft_block"
    BLOCKED = "blocked"
    GAME_OVER = "game_over"
    ABORTED = "aborted"
    HANG = "hang"
    ERROR = "error"
    TIMEOUT = "timeout"


_LINE_FLAGS = re.IGNORECASE | re.MULTILINE

_END_TURN_BLOCKED = re.compile(r"^\s*Cannot end turn\b", _LINE_FLAGS)
_END_TURN_SOFT = re.compile(r"^\s*End turn requested\b.*\bstill\b", _LINE_FLAGS)
_GAME_OVER = re.compile(r"^\s*(?:\*{3}\s*)?GAME OVER\b", _LINE_FLAGS)
_RUN_ABORTED = re.compile(r"^\s*RUN ABORTED\b", _LINE_FLAGS)
_HANG_FAILED = re.compile(r"^\s*(?:HANG:|HANG RECOVERY FAILED\b)", _LINE_FLAGS)
_GENERIC_ERROR = re.compile(
    r"^\s*(?:Error:|Tool failed:|Traceback\b|ERR:|NO_ENEMY\b|(?!(?:[A-Za-z_][\w.]*\.)?TimeoutError:)"
    r"[A-Za-z_][\w.]*Error:|[A-Za-z_][\w.]*Exception:)",
    _LINE_FLAGS,
)
_TIMEOUT = re.compile(
    r"\b(?:timed out after [0-9.]+s|timed out\b|request timed out\b|(?:[A-Za-z_][\w.]*\.)?TimeoutError:)",
    _LINE_FLAGS,
)
_RETRYABLE = re.compile(rf"(?:{_TIMEOUT.pattern})|(?:{_END_TURN_SOFT.pattern})", _LINE_FLAGS)
_SUCCESS_DEFAULT = re.compile(r".*", re.DOTALL)

_UPSTREAM_CIV6_MCP_ACTION_RESPONSE_REFERENCE = """
The upstream civ6-mcp architecture documentation describes action responses
as text strings that return either OK: for confirmations or ERR: for failures.
Documented error examples include ERR:UNIT_NOT_FOUND, ERR:STACKING_CONFLICT,
and ERR:CANNOT_MOVE.
"""


@dataclass(frozen=True)
class _UpstreamTextPrefixChecklistItem:
    """One documented upstream text response prefix and example inventory."""

    prefix: str
    legacy_classification: str
    examples: tuple[str, ...] = ()


def _extract_upstream_text_prefix_checklist(reference_text: str) -> tuple[_UpstreamTextPrefixChecklistItem, ...]:
    """Extract the canonical checklist of documented upstream text prefixes."""
    classifications: list[tuple[str, str]] = []
    examples_by_prefix: dict[str, list[str]] = {}
    for match in re.finditer(r"\b(?:OK|ERR):", reference_text):
        prefix = match.group(0)
        classification = "ok" if prefix == "OK:" else "error"
        item = (prefix, classification)
        if item not in classifications:
            classifications.append(item)
            examples_by_prefix[prefix] = []
    for match in re.finditer(r"\b(?:OK|ERR):[A-Z0-9_]+", reference_text):
        example = match.group(0)
        prefix = "OK:" if example.startswith("OK:") else "ERR:"
        if example not in examples_by_prefix.setdefault(prefix, []):
            examples_by_prefix[prefix].append(example)
    return tuple(
        _UpstreamTextPrefixChecklistItem(prefix=prefix, legacy_classification=classification, examples=tuple(examples))
        for prefix, classification in classifications
        for examples in (examples_by_prefix.get(prefix, []),)
    )


def _extract_upstream_text_prefix_classifications(reference_text: str) -> tuple[tuple[str, str], ...]:
    """Extract documented upstream text prefixes and their legacy classes."""
    return tuple(
        (item.prefix, item.legacy_classification) for item in _extract_upstream_text_prefix_checklist(reference_text)
    )


def _extract_upstream_text_error_prefixes(reference_text: str) -> tuple[str, ...]:
    """Extract documented text-error prefixes from upstream reference prose."""
    return tuple(
        prefix
        for prefix, classification in _extract_upstream_text_prefix_classifications(reference_text)
        if classification == "error"
    )


_UPSTREAM_TEXT_PREFIX_CHECKLIST = _extract_upstream_text_prefix_checklist(_UPSTREAM_CIV6_MCP_ACTION_RESPONSE_REFERENCE)
_UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS = tuple(
    (item.prefix, item.legacy_classification) for item in _UPSTREAM_TEXT_PREFIX_CHECKLIST
)
_UPSTREAM_TEXT_ERROR_PREFIXES = _extract_upstream_text_error_prefixes(_UPSTREAM_CIV6_MCP_ACTION_RESPONSE_REFERENCE)
_UPSTREAM_TEXT_ERROR_PREFIX_CHECKLIST = tuple(
    item for item in _UPSTREAM_TEXT_PREFIX_CHECKLIST if item.legacy_classification == "error"
)


@dataclass(frozen=True)
class Civ6McpClassificationRule:
    """Ordered text-pattern rule for the canonical civ6-mcp classifier."""

    status: Civ6McpClassificationStatus
    pattern: Pattern[str]
    legacy_classification: Civ6McpResponseClassification
    description: str


@dataclass(frozen=True)
class Civ6McpExceptionClassificationRule:
    """Ordered exception-type or predicate rule for the canonical civ6-mcp classifier."""

    status: Civ6McpClassificationStatus
    exception_types: tuple[type[BaseException], ...]
    legacy_classification: Civ6McpResponseClassification
    description: str
    predicate: Callable[[BaseException], bool] | None = None

    def matches(self, exc: BaseException) -> bool:
        """Return whether an exception matches this classifier rule."""
        if isinstance(exc, self.exception_types):
            return True
        return bool(self.predicate and self.predicate(exc))


CIV6_MCP_CLASSIFICATION_PRECEDENCE: tuple[Civ6McpClassificationRule, ...] = (
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.GAME_OVER,
        pattern=_GAME_OVER,
        legacy_classification=Civ6McpResponseClassification.GAME_OVER,
        description="Terminal Civilization VI game-over banner.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.ABORTED,
        pattern=_RUN_ABORTED,
        legacy_classification=Civ6McpResponseClassification.ABORTED,
        description="Operator or launcher aborted the civ6-mcp run.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.HANG,
        pattern=_HANG_FAILED,
        legacy_classification=Civ6McpResponseClassification.HANG,
        description="No-progress recovery exhausted and the backend is stuck.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.FATAL,
        pattern=_GENERIC_ERROR,
        legacy_classification=Civ6McpResponseClassification.ERROR,
        description="Non-retryable tool or validation failure.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.BLOCKED,
        pattern=_END_TURN_BLOCKED,
        legacy_classification=Civ6McpResponseClassification.BLOCKED,
        description="Game-state blocker requiring a different action before continuing.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.RETRYABLE,
        pattern=_RETRYABLE,
        legacy_classification=Civ6McpResponseClassification.TIMEOUT,
        description="Transient timeout or soft end-turn blocker that can be retried after remediation.",
    ),
    Civ6McpClassificationRule(
        status=Civ6McpClassificationStatus.SUCCESS,
        pattern=_SUCCESS_DEFAULT,
        legacy_classification=Civ6McpResponseClassification.OK,
        description="Default successful response when no failure, blocker, or terminal rule matches.",
    ),
)


def _exception_name_matches(*names: str) -> Callable[[BaseException], bool]:
    normalized = frozenset(names)

    def predicate(exc: BaseException) -> bool:
        """Return whether an exception's concrete class name is allowlisted."""
        return type(exc).__name__ in normalized

    return predicate


CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE: tuple[Civ6McpExceptionClassificationRule, ...] = (
    Civ6McpExceptionClassificationRule(
        status=Civ6McpClassificationStatus.RETRYABLE,
        exception_types=(TimeoutError, FutureTimeoutError),
        legacy_classification=Civ6McpResponseClassification.TIMEOUT,
        description="Timeout while waiting for the upstream civ6-mcp server or game bridge.",
        predicate=_exception_name_matches("TimeoutException", "McpTimeoutError", "MCPTimeoutError"),
    ),
    Civ6McpExceptionClassificationRule(
        status=Civ6McpClassificationStatus.ABORTED,
        exception_types=(
            BrokenPipeError,
            ConnectionError,
            ConnectionAbortedError,
            ConnectionResetError,
            EOFError,
            AsyncCancelledError,
            FutureCancelledError,
        ),
        legacy_classification=Civ6McpResponseClassification.ABORTED,
        description="Disconnected stdio, cancelled call, or closed FireTuner/MCP transport.",
        predicate=_exception_name_matches("ClosedResourceError", "EndOfStream", "BrokenResourceError"),
    ),
    Civ6McpExceptionClassificationRule(
        status=Civ6McpClassificationStatus.FATAL,
        exception_types=(ValueError, TypeError, LookupError),
        legacy_classification=Civ6McpResponseClassification.ERROR,
        description="Local validation, unsupported tool, or schema mismatch.",
    ),
    Civ6McpExceptionClassificationRule(
        status=Civ6McpClassificationStatus.FATAL,
        exception_types=(RuntimeError, OSError),
        legacy_classification=Civ6McpResponseClassification.ERROR,
        description="Non-retryable runtime or environment failure unless the message matches a terminal rule.",
    ),
)


@dataclass(frozen=True)
class Civ6McpNormalizedResult:
    """Typed backend-local metadata for one normalized civ6-mcp response."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    text: str = ""
    error: str = ""
    classification: Civ6McpResponseClassification = Civ6McpResponseClassification.OK
    status: Civ6McpClassificationStatus = Civ6McpClassificationStatus.SUCCESS
    is_error: bool = False
    timed_out: bool = False
    content_blocks: tuple[str, ...] = ()
    structured_content: Any | None = None
    raw: Any | None = None


def classify_civ6_mcp_status(text: str) -> Civ6McpClassificationStatus:
    """Classify response text into the canonical civ6-mcp status taxonomy."""
    body = str(text or "").strip()
    if not body:
        return Civ6McpClassificationStatus.SUCCESS
    for rule in CIV6_MCP_CLASSIFICATION_PRECEDENCE:
        if rule.pattern.search(body):
            return rule.status
    return Civ6McpClassificationStatus.SUCCESS


def classify_civ6_mcp_text(text: str) -> Civ6McpResponseClassification:
    """Classify response text into the legacy civ6-mcp response class."""
    body = str(text or "").strip()
    status = classify_civ6_mcp_status(body)
    if status == Civ6McpClassificationStatus.RETRYABLE and _END_TURN_SOFT.search(body):
        return Civ6McpResponseClassification.SOFT_BLOCK
    return _legacy_classification_for_status(status)


def classify_civ6_mcp_exception_status(exc: BaseException) -> Civ6McpClassificationStatus:
    """Classify an exception into the canonical civ6-mcp status taxonomy."""
    message = str(exc or "").strip()
    message_status = classify_civ6_mcp_status(message)
    if message_status not in {Civ6McpClassificationStatus.SUCCESS, Civ6McpClassificationStatus.FATAL}:
        return message_status
    for rule in CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE:
        if rule.matches(exc):
            return rule.status
    if message_status == Civ6McpClassificationStatus.FATAL:
        return message_status
    return Civ6McpClassificationStatus.FATAL


def classify_civ6_mcp_exception(exc: BaseException) -> Civ6McpResponseClassification:
    """Classify an exception into the legacy civ6-mcp response class."""
    message = str(exc or "").strip()
    status = classify_civ6_mcp_exception_status(exc)
    if status == Civ6McpClassificationStatus.RETRYABLE and _END_TURN_SOFT.search(message):
        return Civ6McpResponseClassification.SOFT_BLOCK
    return _legacy_classification_for_status(status)


def normalize_mcp_tool_result(
    tool: str,
    arguments: dict[str, Any] | None,
    result: Any,
) -> Civ6McpNormalizedResult:
    """Normalize an MCP SDK tool result into backend-local response metadata."""
    args = dict(arguments or {})
    content_blocks = tuple(extract_text_blocks(_result_value(result, "content") or []))
    text = "\n".join(content_blocks).strip()
    is_error = bool(_result_value(result, "isError") or _result_value(result, "is_error"))
    if is_error and not text:
        text = _extract_error_message(result)
    structured_content = _get_structured_content(result)
    status = classify_civ6_mcp_status(text)
    if is_error and status == Civ6McpClassificationStatus.SUCCESS:
        status = Civ6McpClassificationStatus.FATAL
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
        status=status,
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
    """Normalize text-only tool output into backend-local response metadata."""
    args = dict(arguments or {})
    body = str(text or "").strip()
    status = classify_civ6_mcp_status(body)
    classification = classify_civ6_mcp_text(body)
    success = _is_success(classification)
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=args,
        success=success,
        text=body,
        error="" if success else body,
        classification=classification,
        status=status,
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
    """Normalize explicit MCP or transport error text into backend-local response metadata."""
    args = dict(arguments or {})
    message = str(error or "").strip()
    status = classify_civ6_mcp_status(message)
    if status == Civ6McpClassificationStatus.SUCCESS:
        status = Civ6McpClassificationStatus.FATAL
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
        status=status,
        is_error=True,
        timed_out=classification == Civ6McpResponseClassification.TIMEOUT,
        raw=raw,
    )


def normalize_mcp_response_exception(
    tool: str,
    arguments: dict[str, Any] | None,
    exc: BaseException,
) -> Civ6McpNormalizedResult:
    """Normalize a tool invocation exception into backend-local response metadata."""
    args = dict(arguments or {})
    message = str(exc or "").strip() or type(exc).__name__
    status = classify_civ6_mcp_exception_status(exc)
    classification = classify_civ6_mcp_exception(exc)
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=args,
        success=False,
        text="",
        error=message,
        classification=classification,
        status=status,
        is_error=True,
        timed_out=_is_timeout_exception(exc) or classification == Civ6McpResponseClassification.TIMEOUT,
        raw=exc,
    )


def normalize_mcp_response_timeout(
    tool: str,
    arguments: dict[str, Any] | None,
    *,
    timeout_seconds: float,
) -> Civ6McpNormalizedResult:
    """Normalize a tool-call timeout into backend-local response metadata."""
    message = f"civ6-mcp tool '{tool}' timed out after {timeout_seconds:g}s"
    return Civ6McpNormalizedResult(
        tool=tool,
        arguments=dict(arguments or {}),
        success=False,
        error=message,
        classification=Civ6McpResponseClassification.TIMEOUT,
        status=Civ6McpClassificationStatus.RETRYABLE,
        timed_out=True,
        is_error=True,
    )


def _result_value(result: Any, name: str) -> Any:
    return payload_value(result, name)


def _extract_error_message(result: Any) -> str:
    error = _result_value(result, "error")
    if isinstance(error, Mapping):
        for key in ("message", "detail", "code"):
            value = error.get(key)
            if value:
                return str(value).strip()
    if error:
        return str(error).strip()

    message = _result_value(result, "message")
    return str(message or "").strip()


def _get_structured_content(result: Any) -> Any | None:
    for name in ("structuredContent", "structured_content"):
        value = _result_value(result, name)
        if value is not None:
            return dump_model(value, json_safe=True)
    return None


def _is_success(classification: Civ6McpResponseClassification) -> bool:
    return classification in {
        Civ6McpResponseClassification.OK,
        Civ6McpResponseClassification.SOFT_BLOCK,
    }


def _legacy_classification_for_status(status: Civ6McpClassificationStatus) -> Civ6McpResponseClassification:
    legacy_by_status = {
        Civ6McpClassificationStatus.SUCCESS: Civ6McpResponseClassification.OK,
        Civ6McpClassificationStatus.BLOCKED: Civ6McpResponseClassification.BLOCKED,
        Civ6McpClassificationStatus.RETRYABLE: Civ6McpResponseClassification.TIMEOUT,
        Civ6McpClassificationStatus.FATAL: Civ6McpResponseClassification.ERROR,
        Civ6McpClassificationStatus.ABORTED: Civ6McpResponseClassification.ABORTED,
        Civ6McpClassificationStatus.HANG: Civ6McpResponseClassification.HANG,
        Civ6McpClassificationStatus.GAME_OVER: Civ6McpResponseClassification.GAME_OVER,
    }
    return legacy_by_status[status]


def _is_timeout_exception(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError | FutureTimeoutError) or type(exc).__name__ in {
        "TimeoutException",
        "McpTimeoutError",
        "MCPTimeoutError",
    }


__all__ = [
    "CIV6_MCP_CLASSIFICATION_PRECEDENCE",
    "CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE",
    "Civ6McpExceptionClassificationRule",
    "Civ6McpClassificationRule",
    "Civ6McpClassificationStatus",
    "Civ6McpNormalizedResult",
    "Civ6McpResponseClassification",
    "classify_civ6_mcp_exception",
    "classify_civ6_mcp_exception_status",
    "classify_civ6_mcp_status",
    "classify_civ6_mcp_text",
    "normalize_mcp_response_error",
    "normalize_mcp_response_exception",
    "normalize_mcp_response_text",
    "normalize_mcp_response_timeout",
    "normalize_mcp_tool_result",
]
