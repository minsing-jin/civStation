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
from typing import TYPE_CHECKING

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClient, Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    StateBundle,
    parse_game_overview,
)

if TYPE_CHECKING:
    from civStation.agent.modules.context.context_manager import ContextManager

logger = logging.getLogger(__name__)


_DEFAULT_OBSERVE_TOOLS: tuple[str, ...] = (
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


class Civ6McpObserver:
    """Polls the civ6-mcp server and updates ContextManager.

    Designed to be called synchronously at the start of each turn iteration.
    Errors from individual tools are caught so a single failing endpoint does
    not stall the whole observation.
    """

    def __init__(
        self,
        client: Civ6McpClient,
        context_manager: ContextManager,
        observe_tools: tuple[str, ...] = _DEFAULT_OBSERVE_TOOLS,
    ) -> None:
        self._client = client
        self._ctx = context_manager
        self._observe_tools = observe_tools
        self._last_bundle: StateBundle | None = None

    @property
    def last_bundle(self) -> StateBundle | None:
        return self._last_bundle

    def observe(self) -> StateBundle:
        """Pull a fresh state bundle and push the structured fields into ContextManager."""
        bundle = StateBundle()
        for tool in self._observe_tools:
            if not self._client.has_tool(tool):
                logger.debug("civ6-mcp tool not available: %s", tool)
                continue
            try:
                text = self._client.call_tool(tool)
            except Civ6McpError as exc:
                logger.warning("civ6-mcp tool '%s' failed: %s", tool, exc)
                continue

            if tool == "get_game_overview":
                bundle.overview = parse_game_overview(text)
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

        self._sync_to_context_manager(bundle)
        self._last_bundle = bundle
        return bundle

    def _sync_to_context_manager(self, bundle: StateBundle) -> None:
        ov = bundle.overview
        updates: dict[str, object] = {}
        if ov.current_turn is not None:
            updates["current_turn"] = ov.current_turn
        if ov.game_era:
            updates["game_era"] = ov.game_era
        if ov.science_per_turn is not None:
            updates["science_per_turn"] = float(ov.science_per_turn)
        if ov.culture_per_turn is not None:
            updates["culture_per_turn"] = float(ov.culture_per_turn)
        if ov.gold_per_turn is not None:
            updates["gold_per_turn"] = float(ov.gold_per_turn)
        if ov.faith_per_turn is not None:
            updates["faith_per_turn"] = float(ov.faith_per_turn)
        if ov.current_research:
            updates["current_research"] = ov.current_research
        if ov.current_civic:
            updates["current_civic"] = ov.current_civic
        if updates:
            try:
                self._ctx.update_global_context(**updates)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ContextManager update_global_context failed: %s", exc)

        # Write a compact situation summary into HighLevelContext so existing
        # strategy/router code paths that read ctx.high_level_context.notes
        # still see something useful.
        summary_bits: list[str] = []
        if ov.current_turn is not None:
            summary_bits.append(f"Turn {ov.current_turn}")
        if ov.game_era:
            summary_bits.append(f"Era {ov.game_era}")
        if ov.science_per_turn is not None:
            summary_bits.append(f"Sci +{ov.science_per_turn:.1f}/t")
        if ov.culture_per_turn is not None:
            summary_bits.append(f"Cul +{ov.culture_per_turn:.1f}/t")
        if ov.current_research:
            summary_bits.append(f"Research {ov.current_research}")
        if ov.current_civic:
            summary_bits.append(f"Civic {ov.current_civic}")
        if ov.is_game_over and ov.victory_text:
            summary_bits.append(f"GAME OVER: {ov.victory_text}")
        if summary_bits:
            try:
                self._ctx.update_game_observation(situation_summary=" | ".join(summary_bits))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ContextManager update_game_observation failed: %s", exc)
