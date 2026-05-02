"""Request construction and dispatch for registered civ6-mcp tool operations."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from civStation.agent.modules.backend.civ6_mcp._payload import planner_tool_call_dicts
from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol, Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.response import (
    _UPSTREAM_TEXT_ERROR_PREFIX_CHECKLIST,
    _UPSTREAM_TEXT_PREFIX_CHECKLIST,
    _UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS,
    Civ6McpNormalizedResult,
    normalize_mcp_response_exception,
    normalize_mcp_response_text,
    normalize_mcp_tool_result,
)
from civStation.agent.modules.backend.civ6_mcp.response import (
    classify_civ6_mcp_text as _classify_response_text,
)

logger = logging.getLogger(__name__)


class SupportedCiv6McpOperation(str, Enum):
    """Operation categories assigned to civ6-mcp tool requests."""

    OBSERVE = "observe"
    ACT = "act"
    END_TURN = "end_turn"


_OBSERVATION_TOOL_ORDER: tuple[str, ...] = (
    "get_game_overview",
    "get_units",
    "get_cities",
    "get_city_production",
    "get_empire_resources",
    "get_diplomacy",
    "get_tech_civics",
    "get_policies",
    "get_governors",
    "get_pantheon_beliefs",
    "get_religion_beliefs",
    "get_world_congress",
    "get_great_people",
    "get_notifications",
    "get_pending_diplomacy",
    "get_pending_trades",
    "get_victory_progress",
    "get_strategic_map",
    "get_dedications",
    "get_unit_promotions",
    "get_purchasable_tiles",
    "get_district_advisor",
    "get_wonder_advisor",
    "get_settle_advisor",
    "get_global_settle_advisor",
    "get_pathing_estimate",
    "get_trade_routes",
    "get_trade_destinations",
    "get_trade_options",
    "get_builder_tasks",
    "get_religion_spread",
)


_ACTION_TOOL_ORDER: tuple[str, ...] = (
    "unit_action",
    "skip_remaining_units",
    "city_action",
    "set_city_production",
    "set_city_focus",
    "purchase_item",
    "purchase_tile",
    "set_research",
    "set_policies",
    "change_government",
    "appoint_governor",
    "assign_governor",
    "promote_governor",
    "promote_unit",
    "upgrade_unit",
    "send_envoy",
    "send_diplomatic_action",
    "form_alliance",
    "propose_trade",
    "respond_to_trade",
    "propose_peace",
    "respond_to_diplomacy",
    "choose_pantheon",
    "found_religion",
    "choose_dedication",
    "queue_wc_votes",
    "patronize_great_person",
    "recruit_great_person",
    "reject_great_person",
    "spy_action",
    "dismiss_popup",
)


END_TURN_TOOL = "end_turn"
END_TURN_REFLECTION_FIELDS: tuple[str, ...] = (
    "tactical",
    "strategic",
    "tooling",
    "planning",
    "hypothesis",
)
OBSERVATION_TOOLS: frozenset[str] = frozenset(_OBSERVATION_TOOL_ORDER)
ACTION_TOOLS: frozenset[str] = frozenset(_ACTION_TOOL_ORDER)
SUPPORTED_CIV6_MCP_TOOLS: frozenset[str] = OBSERVATION_TOOLS | ACTION_TOOLS | frozenset({END_TURN_TOOL})

_TERMINAL_CLASSIFICATIONS = frozenset({"game_over", "aborted", "hang"})
StopRequested = Callable[[], bool]


def _documented_upstream_prefix_examples() -> tuple[tuple[str, str], ...]:
    """Build operations samples for every response.py documented prefix."""
    examples: list[tuple[str, str]] = []
    error_examples_by_prefix = {item.prefix: item.examples for item in _UPSTREAM_TEXT_ERROR_PREFIX_CHECKLIST}
    for prefix, classification in _UPSTREAM_TEXT_PREFIX_CLASSIFICATIONS:
        if classification == "ok":
            examples.append((f"{prefix} action accepted by civ6-mcp.", classification))
            continue
        if classification == "error":
            examples.append((f"{prefix} tool failed in civ6-mcp.", classification))
            examples.extend((example, classification) for example in error_examples_by_prefix.get(prefix, ()))
    return tuple(examples)


def _documented_prefix_classification_gaps(
    tables: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[tuple[str, str, str], ...]:
    """Return documented response.py prefixes absent from named sample tables."""
    gaps: list[tuple[str, str, str]] = []
    for table_name, table in tables.items():
        for item in _UPSTREAM_TEXT_PREFIX_CHECKLIST:
            if not any(
                expected == item.legacy_classification and text.startswith(item.prefix) for text, expected in table
            ):
                gaps.append((table_name, item.prefix, item.legacy_classification))
    return tuple(gaps)


# Documentation-only inventory for the public operations classifier wrapper.
# Runtime classification remains delegated to response.py so prefix behavior
# stays centralized and cannot diverge from normalized MCP result handling.
_DOCUMENTED_PREFIX_CLASSIFICATIONS: tuple[tuple[str, str], ...] = (
    ("GAME OVER - VICTORY! You won a Science victory.", "game_over"),
    ("*** GAME OVER - DEFEAT ***", "game_over"),
    ("RUN ABORTED: operator stopped automation.", "aborted"),
    ("HANG:57:AutoSave_0057|AI turn did not finish.", "hang"),
    ("HANG RECOVERY FAILED after repeated no-op turns.", "hang"),
    *_documented_upstream_prefix_examples(),
    ("Error: must specify a known tech", "error"),
    ("Tool failed:\nError: must specify a known tech", "error"),
    ("Traceback (most recent call last):", "error"),
    ("NO_ENEMY: target not attackable until next turn.", "error"),
    ("ERR:NO_ENEMY|target not attackable until next turn.", "error"),
    ("Cannot end turn: choose production first.", "blocked"),
    ("End turn requested, but units still need orders.", "soft_block"),
    ("civ6-mcp tool 'end_turn' timed out after 120s", "timeout"),
    ("request timed out while waiting for FireTuner", "timeout"),
    ("asyncio.exceptions.TimeoutError: civ6-mcp request exceeded 30s", "timeout"),
)
_UNCOVERED_DOCUMENTED_PREFIX_CLASSIFICATIONS: tuple[tuple[str, str], ...] = tuple(
    (prefix, classification)
    for _table_name, prefix, classification in _documented_prefix_classification_gaps(
        {"operations": _DOCUMENTED_PREFIX_CLASSIFICATIONS}
    )
)


@dataclass(frozen=True)
class Civ6McpRequest:
    """Validated request metadata for one upstream civ6-mcp tool call."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    operation: SupportedCiv6McpOperation = SupportedCiv6McpOperation.ACT
    reasoning: str = ""


@dataclass(frozen=True)
class Civ6McpDispatchResult:
    """Result metadata from dispatching one civ6-mcp request."""

    request: Civ6McpRequest
    success: bool = False
    text: str = ""
    error: str = ""
    classification: str = ""
    status: str = ""
    response: Civ6McpNormalizedResult | None = None


class Civ6McpRequestBuilder:
    """Build validated requests for registered tools and dynamic observations."""

    @classmethod
    def observation(cls, tool: str, arguments: dict[str, Any] | None = None, *, reasoning: str = "") -> Civ6McpRequest:
        """Build a validated registered or dynamic ``get_*`` observation request."""
        request = cls.build(tool, arguments, operation=SupportedCiv6McpOperation.OBSERVE, reasoning=reasoning)
        if not is_civ6_mcp_observation_tool(request.tool):
            raise ValueError(f"Tool {request.tool!r} is not a civ6-mcp observation operation.")
        return request

    @classmethod
    def action(cls, tool: str, arguments: dict[str, Any] | None = None, *, reasoning: str = "") -> Civ6McpRequest:
        """Build a validated action request such as ``set_research`` or ``unit_action``."""
        request = cls.build(tool, arguments, operation=SupportedCiv6McpOperation.ACT, reasoning=reasoning)
        if request.tool not in ACTION_TOOLS:
            raise ValueError(f"Tool {request.tool!r} is not a civ6-mcp action operation.")
        return request

    @classmethod
    def end_turn(
        cls,
        *,
        tactical: str,
        strategic: str,
        tooling: str,
        planning: str,
        hypothesis: str,
        reasoning: str = "",
    ) -> Civ6McpRequest:
        """Build the required ``end_turn`` request with all reflection fields."""
        arguments = {
            "tactical": tactical,
            "strategic": strategic,
            "tooling": tooling,
            "planning": planning,
            "hypothesis": hypothesis,
        }
        _validate_end_turn_arguments(arguments)
        return cls.build(
            END_TURN_TOOL,
            arguments,
            operation=SupportedCiv6McpOperation.END_TURN,
            reasoning=reasoning,
        )

    @classmethod
    def build(
        cls,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        operation: SupportedCiv6McpOperation | str | None = None,
        reasoning: str = "",
        require_supported: bool = True,
    ) -> Civ6McpRequest:
        """Build a normalized request and validate support, arguments, and operation."""
        if not isinstance(tool, str) or not tool:
            raise ValueError("civ6-mcp request tool must be a non-empty string.")
        if require_supported and tool not in SUPPORTED_CIV6_MCP_TOOLS and not _is_dynamic_observation_tool(tool):
            raise ValueError(f"Unsupported civ6-mcp tool: {tool!r}")

        normalized_arguments = _normalize_arguments(arguments)
        op = _normalize_operation(operation) if operation is not None else operation_for_tool(tool)
        if tool == END_TURN_TOOL:
            _validate_end_turn_arguments(normalized_arguments)
            op = SupportedCiv6McpOperation.END_TURN
        elif op == SupportedCiv6McpOperation.END_TURN:
            raise ValueError(f"Only {END_TURN_TOOL!r} may use the end_turn operation.")

        return Civ6McpRequest(
            tool=tool,
            arguments=normalized_arguments,
            operation=op,
            reasoning=str(reasoning or ""),
        )


class Civ6McpOperationDispatcher:
    """Dispatch validated civ6-mcp tool requests through a synchronous client."""

    def __init__(self, client: Civ6McpClientProtocol) -> None:
        self._client = client

    def dispatch(self, request: Civ6McpRequest) -> Civ6McpDispatchResult:
        """Validate and invoke one request, returning dispatch metadata."""
        try:
            validate_civ6_mcp_request(request)
        except ValueError as exc:
            return Civ6McpDispatchResult(
                request=request,
                success=False,
                error=str(exc),
                classification="error",
                status="fatal",
            )

        if not self._client.has_tool(request.tool):
            return Civ6McpDispatchResult(
                request=request,
                success=False,
                error=f"Tool '{request.tool}' not exposed by civ6-mcp server.",
                classification="error",
                status="fatal",
            )

        try:
            response = self._call_tool_result(request)
        except Civ6McpError as exc:
            response = normalize_mcp_response_exception(request.tool, request.arguments, exc)
            return Civ6McpDispatchResult(
                request=request,
                success=False,
                error=response.error,
                classification=response.classification.value,
                status=response.status.value,
                response=response,
            )
        except Exception as exc:  # noqa: BLE001
            response = normalize_mcp_response_exception(request.tool, request.arguments, exc)
            return Civ6McpDispatchResult(
                request=request,
                success=False,
                error=response.error,
                classification=response.classification.value,
                status=response.status.value,
                response=response,
            )

        return Civ6McpDispatchResult(
            request=request,
            success=response.success,
            text=response.text,
            error=response.error,
            classification=response.classification.value,
            status=response.status.value,
            response=response,
        )

    def _call_tool_result(self, request: Civ6McpRequest) -> Civ6McpNormalizedResult:
        typed_call = getattr(self._client, "call_tool_result", None)
        if callable(typed_call):
            result = typed_call(request.tool, request.arguments)
            if isinstance(result, Civ6McpNormalizedResult):
                return result
            if isinstance(result, str):
                return normalize_mcp_response_text(request.tool, request.arguments, result)
            return normalize_mcp_tool_result(request.tool, request.arguments, result)

        text = self._client.call_tool(request.tool, request.arguments)
        return normalize_mcp_response_text(request.tool, request.arguments, text)

    def dispatch_many(
        self,
        requests: list[Civ6McpRequest],
        *,
        stop_on_terminal: bool = True,
        stop_requested: StopRequested | None = None,
    ) -> list[Civ6McpDispatchResult]:
        """Dispatch requests in order, honoring terminal outcomes and stop requests."""
        results: list[Civ6McpDispatchResult] = []
        for request in requests:
            if stop_requested is not None and stop_requested():
                logger.info("civ6-mcp dispatcher stopping early before tool=%s: stop requested", request.tool)
                break
            outcome = self.dispatch(request)
            results.append(outcome)
            if stop_on_terminal and outcome.classification in _TERMINAL_CLASSIFICATIONS:
                logger.warning(
                    "civ6-mcp dispatcher stopping early: classification=%s tool=%s",
                    outcome.classification,
                    request.tool,
                )
                break
        return results


def operation_for_tool(tool: str) -> SupportedCiv6McpOperation:
    """Infer a tool operation category without requiring registered-tool support."""
    if tool == END_TURN_TOOL:
        return SupportedCiv6McpOperation.END_TURN
    if is_civ6_mcp_observation_tool(tool):
        return SupportedCiv6McpOperation.OBSERVE
    return SupportedCiv6McpOperation.ACT


def validate_civ6_mcp_request(request: Civ6McpRequest) -> None:
    """Validate a request's tool, arguments, operation, and end-turn fields."""
    if request.tool not in SUPPORTED_CIV6_MCP_TOOLS and not _is_dynamic_observation_tool(request.tool):
        raise ValueError(f"Unsupported civ6-mcp tool: {request.tool!r}")
    if not isinstance(request.arguments, dict):
        raise ValueError("civ6-mcp request arguments must be a dict.")
    expected_operation = operation_for_tool(request.tool)
    if request.operation != expected_operation:
        raise ValueError(
            f"Tool {request.tool!r} is a {expected_operation.value!r} operation, not {request.operation.value!r}."
        )
    if request.tool == END_TURN_TOOL:
        _validate_end_turn_arguments(request.arguments)


def coerce_civ6_mcp_requests(payload: Any) -> list[Civ6McpRequest]:
    """Convert planner tool-call envelopes or lists into validated requests."""
    requests: list[Civ6McpRequest] = []
    for raw in planner_tool_call_dicts(payload):
        tool = raw.get("tool") or raw.get("name")
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"Tool call missing 'tool' name: {raw!r}")
        arguments = raw.get("arguments") or {}
        operation = raw.get("operation")
        requests.append(
            Civ6McpRequestBuilder.build(
                tool,
                arguments,
                operation=operation,
                reasoning=str(raw.get("reasoning") or ""),
            )
        )
    return requests


def classify_civ6_mcp_text(text: str) -> str:
    """Classify civ6-mcp response text as a legacy response-class string."""
    return _classify_response_text(text).value


def is_civ6_mcp_observation_tool(tool: str) -> bool:
    """Return whether a tool name is an allowlisted or dynamic ``get_*`` observation."""
    return tool in OBSERVATION_TOOLS or _is_dynamic_observation_tool(tool)


def _normalize_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise ValueError("civ6-mcp request arguments must be an object.")
    return dict(arguments)


def _normalize_operation(operation: SupportedCiv6McpOperation | str) -> SupportedCiv6McpOperation:
    if isinstance(operation, SupportedCiv6McpOperation):
        return operation
    try:
        return SupportedCiv6McpOperation(str(operation))
    except ValueError as exc:
        valid = ", ".join(item.value for item in SupportedCiv6McpOperation)
        raise ValueError(f"Unsupported civ6-mcp operation {operation!r}; expected one of: {valid}") from exc


def _validate_end_turn_arguments(arguments: dict[str, Any]) -> None:
    missing = [field for field in END_TURN_REFLECTION_FIELDS if not str(arguments.get(field) or "").strip()]
    if missing:
        raise ValueError(f"end_turn requires non-empty reflection fields: {', '.join(missing)}")


def _is_dynamic_observation_tool(tool: str) -> bool:
    return isinstance(tool, str) and tool.startswith("get_")


__all__ = [
    "ACTION_TOOLS",
    "END_TURN_REFLECTION_FIELDS",
    "END_TURN_TOOL",
    "OBSERVATION_TOOLS",
    "SUPPORTED_CIV6_MCP_TOOLS",
    "Civ6McpDispatchResult",
    "Civ6McpOperationDispatcher",
    "Civ6McpRequest",
    "Civ6McpRequestBuilder",
    "SupportedCiv6McpOperation",
    "classify_civ6_mcp_text",
    "coerce_civ6_mcp_requests",
    "is_civ6_mcp_observation_tool",
    "operation_for_tool",
    "validate_civ6_mcp_request",
]
