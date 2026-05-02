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

import json
import logging
from typing import TYPE_CHECKING

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    normalize_observation_bundle,
)
from civStation.agent.modules.backend.civ6_mcp.operations import (
    Civ6McpOperationDispatcher,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    GameOverviewSnapshot,
    StateBundle,
    parse_game_overview,
)

if TYPE_CHECKING:
    from civStation.agent.modules.context.context_manager import ContextManager

logger = logging.getLogger(__name__)


DEFAULT_CIV6_MCP_OBSERVE_TOOLS: tuple[str, ...] = (
    "get_game_overview",
    "get_units",
    "get_cities",
    "get_diplomacy",
    "get_tech_civics",
    "get_notifications",
    "get_pending_diplomacy",
    "get_pending_trades",
    "get_victory_progress",
)
_DEFAULT_OBSERVE_TOOLS = DEFAULT_CIV6_MCP_OBSERVE_TOOLS


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
        observe_tools: tuple[str, ...] = _DEFAULT_OBSERVE_TOOLS,
    ) -> None:
        self._client = client
        self._ctx = context_manager
        self._observe_tools = observe_tools
        self._dispatcher = Civ6McpOperationDispatcher(client)
        self._last_bundle: StateBundle | None = None
        self._last_observation: Civ6McpNormalizedObservation | None = None

    @property
    def last_bundle(self) -> StateBundle | None:
        return self._last_bundle

    @property
    def last_observation(self) -> Civ6McpNormalizedObservation | None:
        return self._last_observation

    def observe(self) -> StateBundle:
        """Pull a fresh state bundle and push the structured fields into ContextManager."""
        bundle = StateBundle()
        for tool in self._observe_tools:
            if not self._client.has_tool(tool):
                logger.debug("civ6-mcp tool not available: %s", tool)
                bundle.missing_tools = (*bundle.missing_tools, tool)
                continue
            try:
                request = Civ6McpRequestBuilder.observation(tool)
            except ValueError as exc:
                logger.warning("civ6-mcp observation tool '%s' is unsupported: %s", tool, exc)
                continue

            outcome = self._dispatcher.dispatch(request)
            if not outcome.success:
                logger.warning("civ6-mcp tool '%s' failed: %s", tool, outcome.error or outcome.text)
                bundle.failed_tools[tool] = outcome.error or outcome.text or outcome.classification
                continue
            text = _extract_observation_text(outcome)
            if not text.strip():
                logger.warning("civ6-mcp tool '%s' returned an empty state response", tool)
                bundle.malformed_tools[tool] = "empty response body"
                continue

            if tool == "get_game_overview":
                bundle.overview = parse_game_overview(text)
                if not _overview_has_parsed_state(bundle.overview):
                    logger.warning("civ6-mcp get_game_overview response did not contain parseable state")
                    bundle.malformed_tools[tool] = "unrecognized overview payload"
            elif tool == "get_units":
                bundle.units_text = text
            elif tool == "get_cities":
                bundle.cities_text = text
            elif tool == "get_diplomacy":
                bundle.diplomacy_text = text
            elif tool == "get_tech_civics":
                bundle.tech_civics_text = text
            elif tool == "get_notifications":
                bundle.notifications_text = text
            elif tool == "get_pending_diplomacy":
                bundle.pending_diplomacy_text = text
            elif tool == "get_pending_trades":
                bundle.pending_trades_text = text
            elif tool == "get_victory_progress":
                bundle.victory_progress_text = text
            else:
                bundle.extra[tool] = text

        observation = normalize_observation_bundle(bundle)
        self._sync_to_context_manager(observation)
        self._last_bundle = bundle
        self._last_observation = observation
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
) -> Civ6McpObserver:
    """Build the observer used by the civ6-mcp backend turn loop."""
    return Civ6McpObserver(
        client=client,
        context_manager=context_manager,
        observe_tools=DEFAULT_CIV6_MCP_OBSERVE_TOOLS if observe_tools is None else observe_tools,
    )


def _extract_observation_text(outcome) -> str:  # noqa: ANN001
    """Return the best textual representation of a successful observation result."""
    if outcome.text.strip():
        return outcome.text
    response = getattr(outcome, "response", None)
    structured = getattr(response, "structured_content", None)
    if structured is None:
        return ""
    try:
        return json.dumps(structured, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(structured)


def _overview_has_parsed_state(overview: GameOverviewSnapshot) -> bool:
    return (
        any(
            value is not None
            for value in (
                overview.current_turn,
                overview.game_era,
                overview.science_per_turn,
                overview.culture_per_turn,
                overview.gold_per_turn,
                overview.faith_per_turn,
                overview.current_research,
                overview.current_civic,
                overview.victory_text,
            )
        )
        or overview.is_game_over
    )


__all__ = [
    "DEFAULT_CIV6_MCP_OBSERVE_TOOLS",
    "Civ6McpObserver",
    "build_civ6_mcp_observer",
]
