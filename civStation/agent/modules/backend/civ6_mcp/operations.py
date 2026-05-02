"""Request construction and dispatch for supported civ6-mcp operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol, Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.response import (
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
    """Backend operation categories backed by upstream civ6-mcp tools."""

    OBSERVE = "observe"
    ACT = "act"
    END_TURN = "end_turn"


OBSERVATION_TOOLS: frozenset[str] = frozenset(
    {
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
    }
)


ACTION_TOOLS: frozenset[str] = frozenset(
    {
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
    }
)


END_TURN_TOOL = "end_turn"
END_TURN_REFLECTION_FIELDS: tuple[str, ...] = (
    "tactical",
    "strategic",
    "tooling",
    "planning",
    "hypothesis",
)
SUPPORTED_CIV6_MCP_TOOLS: frozenset[str] = OBSERVATION_TOOLS | ACTION_TOOLS | frozenset({END_TURN_TOOL})

_TERMINAL_CLASSIFICATIONS = frozenset({"game_over", "aborted", "hang"})
StopRequested = Callable[[], bool]


@dataclass(frozen=True)
class Civ6McpRequest:
    """A validated tool request to dispatch to the upstream civ6-mcp server."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    operation: SupportedCiv6McpOperation = SupportedCiv6McpOperation.ACT
    reasoning: str = ""


@dataclass(frozen=True)
class Civ6McpDispatchResult:
    """Outcome of dispatching a single civ6-mcp request."""

    request: Civ6McpRequest
    success: bool = False
    text: str = ""
    error: str = ""
    classification: str = ""
    status: str = ""
    response: Civ6McpNormalizedResult | None = None


class Civ6McpRequestBuilder:
    """Construct typed requests for the civ6-mcp backend operation surface."""

    @classmethod
    def observation(cls, tool: str, arguments: dict[str, Any] | None = None, *, reasoning: str = "") -> Civ6McpRequest:
        """Build an observation request, e.g. ``get_game_overview``."""
        request = cls.build(tool, arguments, operation=SupportedCiv6McpOperation.OBSERVE, reasoning=reasoning)
        if not is_civ6_mcp_observation_tool(request.tool):
            raise ValueError(f"Tool {request.tool!r} is not a civ6-mcp observation operation.")
        return request

    @classmethod
    def action(cls, tool: str, arguments: dict[str, Any] | None = None, *, reasoning: str = "") -> Civ6McpRequest:
        """Build an action request, e.g. ``set_research`` or ``unit_action``."""
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
        """Build the required civ6-mcp ``end_turn`` request with five reflections."""
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
        """Build and validate a civ6-mcp tool request."""
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
    """Dispatch validated civ6-mcp requests through a client protocol."""

    def __init__(self, client: Civ6McpClientProtocol) -> None:
        self._client = client

    def dispatch(self, request: Civ6McpRequest) -> Civ6McpDispatchResult:
        """Invoke one request and classify the textual response."""
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
        """Dispatch requests in order and stop on terminal civ6-mcp responses."""
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
    """Infer the backend operation category for a supported civ6-mcp tool."""
    if tool == END_TURN_TOOL:
        return SupportedCiv6McpOperation.END_TURN
    if is_civ6_mcp_observation_tool(tool):
        return SupportedCiv6McpOperation.OBSERVE
    return SupportedCiv6McpOperation.ACT


def validate_civ6_mcp_request(request: Civ6McpRequest) -> None:
    """Validate a request before dispatch."""
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
    """Normalize planner JSON into validated civ6-mcp requests.

    Accepts either ``{"tool_calls": [...]}`` or the bare list form.
    """
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unexpected planner payload shape: {type(payload).__name__}")

    if not isinstance(items, list):
        raise ValueError("Planner payload 'tool_calls' must be a list.")

    requests: list[Civ6McpRequest] = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError(f"Tool call entry must be an object, got {type(raw).__name__}")
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
    """Classify civ6-mcp text and return the stable string value."""
    return _classify_response_text(text).value


def is_civ6_mcp_observation_tool(tool: str) -> bool:
    """Return whether a tool name represents a read-only civ6-mcp observation."""
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
