"""civ6-mcp observer — pulls native game state and writes it into ContextManager.

This is the civ6-mcp counterpart of `agent/modules/context/context_updater.py`.
Where ContextUpdater asks a VLM to extract state from a screenshot, the
observer here calls multiple `get_*` tools on the upstream MCP server and
parses the textual responses.

Synchronous by design (one snapshot per turn-tick). A future iteration can
move this to a background thread the way ContextUpdater does, but for now
keeping it inline removes a lot of races during the migration.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    Civ6McpToolObservation,
    normalize_observation_bundle,
    parse_observation_tool_response,
)
from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpDispatchResult,
    Civ6McpOperationDispatcher,
    Civ6McpRequest,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.response import Civ6McpNormalizedResult
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    GameOverviewSnapshot,
    StateBundle,
)

if TYPE_CHECKING:
    from civStation.agent.modules.context.context_manager import ContextManager

logger = logging.getLogger(__name__)


DEFAULT_CIV6_MCP_OBSERVE_TIMEOUT_SECONDS = 30.0


DEFAULT_CIV6_MCP_OBSERVE_TOOLS: tuple[str, ...] = (
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


class Civ6McpObserver:
    """Polls the civ6-mcp server and updates ContextManager.

    Designed to be called synchronously at the start of each turn iteration.
    Errors from individual tools are caught so a single failing endpoint does
    not stall the whole observation.
    """

    def __init__(
        self,
        client: Civ6McpClientProtocol,
        context_manager: ContextManager,
        observe_tools: tuple[str, ...] | None = None,
        observe_timeout_seconds: float | None = DEFAULT_CIV6_MCP_OBSERVE_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._ctx = context_manager
        self._observe_tools = observe_tools if observe_tools is not None else discover_civ6_mcp_observe_tools(client)
        self._observe_timeout_seconds = _normalize_timeout_seconds(observe_timeout_seconds)
        self._dispatcher = Civ6McpOperationDispatcher(client)
        self._last_bundle: StateBundle | None = None
        self._last_observation: Civ6McpNormalizedObservation | None = None
        self._last_tool_observations: tuple[Civ6McpToolObservation, ...] = ()

    @property
    def last_bundle(self) -> StateBundle | None:
        return self._last_bundle

    @property
    def last_observation(self) -> Civ6McpNormalizedObservation | None:
        return self._last_observation

    @property
    def last_tool_observations(self) -> tuple[Civ6McpToolObservation, ...]:
        return self._last_tool_observations

    def observe(self) -> StateBundle:
        """Pull a fresh state bundle and push the structured fields into ContextManager."""
        bundle = StateBundle()
        tool_observations: list[Civ6McpToolObservation] = []
        attempted_tools = 0
        skipped_tools = 0
        deadline_exhausted = False
        deadline_at = (
            time.monotonic() + self._observe_timeout_seconds if self._observe_timeout_seconds is not None else None
        )
        for index, tool in enumerate(self._observe_tools):
            if deadline_at is not None and time.monotonic() >= deadline_at:
                deadline_exhausted = True
                skipped_tools += _record_deadline_skipped_tools(bundle, self._observe_tools[index:])
                break
            if not self._client.has_tool(tool):
                logger.debug("civ6-mcp tool not available: %s", tool)
                bundle.missing_tools = (*bundle.missing_tools, tool)
                continue
            try:
                request = Civ6McpRequestBuilder.observation(tool)
            except ValueError as exc:
                logger.warning("civ6-mcp observation tool '%s' is unsupported: %s", tool, exc)
                skipped_tools += 1
                continue

            attempted_tools += 1
            try:
                outcome = _dispatch_with_optional_timeout(
                    self._dispatcher,
                    request,
                    timeout_seconds=_remaining_seconds(deadline_at),
                )
            except FutureTimeoutError:
                deadline_exhausted = True
                timeout_text = _timeout_reason(self._observe_timeout_seconds)
                logger.warning("civ6-mcp tool '%s' observation timed out: %s", tool, timeout_text)
                logger.warning("recoverable civ6-mcp observation failure: tool=%s reason=%s", tool, timeout_text)
                bundle.failed_tools[tool] = timeout_text
                skipped_tools += _record_deadline_skipped_tools(bundle, self._observe_tools[index + 1 :])
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("civ6-mcp tool '%s' could not be observed: %s", tool, exc)
                reason = _observer_error_reason(exc)
                logger.warning("recoverable civ6-mcp observation failure: tool=%s reason=%s", tool, reason)
                bundle.failed_tools[tool] = reason
                continue

            if outcome is None:
                logger.warning("civ6-mcp tool '%s' returned no observation result", tool)
                bundle.malformed_tools[tool] = "absent response"
                continue

            if not getattr(outcome, "success", False):
                error_text = _outcome_error_text(outcome)
                logger.warning("civ6-mcp tool '%s' failed: %s", tool, error_text)
                logger.warning("recoverable civ6-mcp observation failure: tool=%s reason=%s", tool, error_text)
                bundle.failed_tools[tool] = error_text
                continue

            try:
                tool_observation = parse_observation_tool_response(tool, _extract_observation_payload(tool, outcome))
            except ValueError as exc:
                logger.warning("civ6-mcp tool '%s' returned malformed state: %s", tool, exc)
                bundle.malformed_tools[tool] = _malformed_reason(exc)
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("civ6-mcp tool '%s' returned unexpected state: %s", tool, exc)
                bundle.malformed_tools[tool] = _observer_error_reason(exc)
                continue

            _merge_observation_bundle(bundle, tool_observation.bundle)
            tool_observations.append(tool_observation)
            if tool == "get_game_overview" and not _overview_has_parsed_state(bundle.overview):
                logger.warning("civ6-mcp get_game_overview response did not contain parseable state")
                bundle.malformed_tools[tool] = "unrecognized overview payload"

        observation = normalize_observation_bundle(bundle)
        self._sync_to_context_manager(observation)
        if tool_observations or self._last_observation is None:
            self._last_bundle = bundle
            self._last_observation = observation
            self._last_tool_observations = tuple(tool_observations)
        logger.info(
            "civ6-mcp observation completed: attempted=%d successful=%d missing=%d failed=%d malformed=%d "
            "skipped=%d deadline_exhausted=%s",
            attempted_tools,
            len(tool_observations),
            len(bundle.missing_tools),
            len(bundle.failed_tools),
            len(bundle.malformed_tools),
            skipped_tools,
            deadline_exhausted,
        )
        return bundle

    def _sync_to_context_manager(self, observation: Civ6McpNormalizedObservation) -> None:
        if observation.global_context_updates:
            try:
                self._ctx.update_global_context(**observation.global_context_updates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ContextManager update_global_context failed: %s", exc)

        # Write a compact situation summary into HighLevelContext so existing
        # strategy/router code paths that read ctx.high_level_context.notes
        # still see something useful.
        if observation.game_observation_updates:
            try:
                self._ctx.update_game_observation(**observation.game_observation_updates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ContextManager update_game_observation failed: %s", exc)


def build_civ6_mcp_observer(
    *,
    client: Civ6McpClientProtocol,
    context_manager: ContextManager,
    observe_tools: tuple[str, ...] | None = None,
    observe_timeout_seconds: float | None = DEFAULT_CIV6_MCP_OBSERVE_TIMEOUT_SECONDS,
) -> Civ6McpObserver:
    """Build the observer used by the civ6-mcp backend turn loop."""
    return Civ6McpObserver(
        client=client,
        context_manager=context_manager,
        observe_tools=observe_tools,
        observe_timeout_seconds=observe_timeout_seconds,
    )


def discover_civ6_mcp_observe_tools(client: Civ6McpClientProtocol) -> tuple[str, ...]:
    """Discover observation tools exposed by civ6-mcp.

    The upstream backend uses ``get_*`` names for read-only observation tools.
    Filtering the live MCP catalog keeps the observer separate from action and
    end-turn tools while still picking up newly bundled observation endpoints.
    """
    available = _available_tool_names(client)
    if not available:
        return DEFAULT_CIV6_MCP_OBSERVE_TOOLS

    get_tools = {name for name in available if name.startswith("get_")}
    ordered = [name for name in DEFAULT_CIV6_MCP_OBSERVE_TOOLS if name in get_tools]
    ordered.extend(sorted(get_tools.difference(DEFAULT_CIV6_MCP_OBSERVE_TOOLS)))
    return tuple(ordered)


def _extract_observation_payload(tool: str, outcome: object) -> object:
    """Return the richest payload available for schema-level observation parsing."""
    response = getattr(outcome, "response", None)
    if response is not None:
        _validate_observation_response_tool(tool, response)
        return response
    return getattr(outcome, "text", None)


def _validate_observation_response_tool(expected_tool: str, response: object) -> None:
    """Reject successful normalized responses that belong to a different tool."""
    response_tool = getattr(response, "tool", None)
    if response_tool is None:
        return
    if not isinstance(response_tool, str) or not response_tool:
        raise ValueError("response tool must be a non-empty string")
    if isinstance(response, Civ6McpNormalizedResult) and response_tool != expected_tool:
        raise ValueError(f"response tool mismatch: expected {expected_tool}, got {response_tool}")


def _outcome_error_text(outcome: object) -> str:
    for attr in ("error", "text", "classification", "status"):
        value = getattr(outcome, attr, "")
        if value:
            return str(value)
    return "absent response"


def _remaining_seconds(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    return max(0.0, deadline_at - time.monotonic())


def _dispatch_with_optional_timeout(
    dispatcher: Civ6McpOperationDispatcher,
    request: Civ6McpRequest,
    *,
    timeout_seconds: float | None,
) -> Civ6McpDispatchResult:
    if timeout_seconds is None:
        return dispatcher.dispatch(request)
    if timeout_seconds <= 0:
        raise FutureTimeoutError()

    future: Future[Civ6McpDispatchResult] = Future()

    def dispatch_in_background() -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(dispatcher.dispatch(request))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)

    thread = threading.Thread(
        target=dispatch_in_background,
        name=f"civ6-mcp-observe-{request.tool}",
        daemon=True,
    )
    thread.start()
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        raise


def _timeout_reason(timeout_seconds: float | None) -> str:
    if timeout_seconds is None:
        return "timed out"
    return f"timed out after {timeout_seconds:.2f}s"


def _normalize_timeout_seconds(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    return max(0.0, float(timeout_seconds))


def _record_deadline_skipped_tools(bundle: StateBundle, tools: tuple[str, ...]) -> int:
    skipped_count = 0
    for tool in tools:
        if tool in bundle.failed_tools:
            continue
        reason = "skipped because observation deadline was exhausted"
        logger.warning("recoverable civ6-mcp observation failure: tool=%s reason=%s", tool, reason)
        bundle.failed_tools[tool] = reason
        skipped_count += 1
    return skipped_count


def _merge_observation_bundle(target: StateBundle, source: StateBundle) -> None:
    if source.overview.raw_text or _overview_has_parsed_state(source.overview):
        target.overview = source.overview
    for attr in (
        "units_text",
        "cities_text",
        "diplomacy_text",
        "tech_civics_text",
        "notifications_text",
        "pending_diplomacy_text",
        "pending_trades_text",
        "victory_progress_text",
    ):
        value = getattr(source, attr)
        if value:
            setattr(target, attr, value)
    target.extra.update(source.extra)
    target.missing_tools = (*target.missing_tools, *source.missing_tools)
    target.failed_tools.update(source.failed_tools)
    target.malformed_tools.update(source.malformed_tools)


def _malformed_reason(exc: ValueError) -> str:
    message = str(exc)
    if "empty response body" in message:
        return "empty response body"
    return message


def _observer_error_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{exc.__class__.__name__}: {message}"
    return exc.__class__.__name__


def _overview_has_parsed_state(overview: GameOverviewSnapshot) -> bool:
    return (
        any(
            value is not None
            for value in (
                overview.current_turn,
                overview.game_era,
                overview.game_speed,
                overview.civilization_name,
                overview.leader_name,
                overview.gold,
                overview.science_per_turn,
                overview.culture_per_turn,
                overview.gold_per_turn,
                overview.faith,
                overview.faith_per_turn,
                overview.total_population,
                overview.military_strength,
                overview.unit_count,
                overview.current_research,
                overview.current_civic,
                overview.victory_text,
            )
        )
        or overview.is_game_over
    )


def _available_tool_names(client: Civ6McpClientProtocol) -> set[str]:
    raw_tool_names = getattr(client, "tool_names", None)
    if raw_tool_names:
        return {name for name in raw_tool_names if isinstance(name, str)}

    tool_schemas = getattr(client, "tool_schemas", None)
    if callable(tool_schemas):
        schemas = tool_schemas()
        if isinstance(schemas, Mapping):
            return {name for name in schemas if isinstance(name, str)}

    return set()


__all__ = [
    "DEFAULT_CIV6_MCP_OBSERVE_TIMEOUT_SECONDS",
    "DEFAULT_CIV6_MCP_OBSERVE_TOOLS",
    "Civ6McpObserver",
    "build_civ6_mcp_observer",
    "discover_civ6_mcp_observe_tools",
]
