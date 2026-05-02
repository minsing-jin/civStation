"""Deterministic turn-planning hints for the civ6-mcp backend.

The LLM planner still decides exact mutating actions. This module maps the
normalized civ6-mcp state envelope into prioritized backend intents so prompts
consistently handle blockers before routine turn work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    section_texts_for_bundle,
)
from civStation.agent.modules.backend.civ6_mcp.operations import OBSERVATION_TOOLS
from civStation.agent.modules.backend.civ6_mcp.planner_types import (
    Civ6McpPlannerAction,
    Civ6McpPlannerIntent,
)
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    StateBundle,
    parse_game_overview,
    state_bundle_from_raw_mcp_state,
)

_NO_WORK_PATTERNS = (
    "none",
    "no active",
    "no current",
    "no incoming",
    "no pending",
    "empty",
    "없음",
)
_DIAGNOSTIC_TOOL_PATTERN = re.compile(r"\bget_[a-z0-9_]+\b")


@dataclass(frozen=True)
class Civ6McpPrioritizedIntent:
    """Prioritized planner intent inferred from normalized civ6-mcp state."""

    priority: int
    intent: Civ6McpPlannerIntent
    source_section: str = ""
    trigger: str = ""

    @property
    def tool(self) -> str:
        """Return the upstream civ6-mcp tool name for this intent."""
        return self.intent.tool

    @property
    def arguments(self) -> dict[str, Any]:
        """Return a copy of the tool arguments for this intent."""
        return dict(self.intent.arguments)

    @property
    def reasoning(self) -> str:
        """Return the planner-facing rationale for this intent."""
        return self.intent.reasoning

    def to_action(self) -> Civ6McpPlannerAction:
        """Convert this intent into an executable planner action."""
        return self.intent.to_action()

    def to_tool_call(self) -> ToolCall:
        """Convert this intent into an executor tool call."""
        return self.intent.to_tool_call()

    def render_for_prompt(self) -> str:
        """Render a compact prompt line for this intent."""
        detail = f" | source={self.source_section}" if self.source_section else ""
        trigger = f" | trigger={self.trigger}" if self.trigger else ""
        return f"P{self.priority:03d} {self.tool} - {self.reasoning}{detail}{trigger}"


@dataclass(frozen=True)
class Civ6McpTurnPlan:
    """Deterministic civ6-mcp intent plan for one normalized state."""

    intents: tuple[Civ6McpPrioritizedIntent, ...] = ()
    notes: tuple[str, ...] = ()
    backend: str = "civ6-mcp"

    def to_actions(self) -> list[Civ6McpPlannerAction]:
        """Return executable planner actions ordered by priority."""
        return [item.to_action() for item in self.intents]

    @property
    def actions(self) -> list[Civ6McpPlannerAction]:
        """Return actions for consumers that expect plan.actions."""
        return self.to_actions()

    def to_tool_calls(self) -> list[ToolCall]:
        """Return executor tool calls ordered by priority."""
        return [item.to_tool_call() for item in self.intents]

    def render_for_prompt(self) -> str:
        """Render deterministic turn guidance for the LLM planner."""
        if not self.intents:
            return "\n".join(self.notes) if self.notes else "(no deterministic civ6-mcp intents)"
        lines = [item.render_for_prompt() for item in self.intents]
        if self.notes:
            lines.append("Notes: " + " | ".join(self.notes))
        return "\n".join(lines)


def build_prioritized_turn_plan(
    state: StateBundle | Civ6McpNormalizedObservation | dict[str, Any],
    *,
    strategy: str = "",
    include_end_turn: bool = True,
) -> Civ6McpTurnPlan:
    """Build deterministic planner guidance from normalized civ6-mcp state."""
    bundle, sections = _coerce_state_and_sections(state)
    notes: list[str] = []
    candidates: list[Civ6McpPrioritizedIntent] = []

    if bundle is not None and bundle.overview.is_game_over:
        victory = f": {bundle.overview.victory_text}" if bundle.overview.victory_text else ""
        return Civ6McpTurnPlan(notes=(f"Game over detected{victory}; no end_turn intent emitted.",))

    _maybe_add_missing_overview(candidates, sections)
    _add_diagnostic_retries(candidates, sections)
    _add_pending_decision_intents(candidates, sections)
    _add_notification_intents(candidates, sections)
    _add_state_gap_intents(candidates, bundle, sections)
    _add_strategy_intents(candidates, sections, strategy)

    if include_end_turn:
        candidates.append(
            _intent(
                1000,
                "end_turn",
                arguments=_end_turn_reflections(bundle, sections, strategy),
                reasoning="End the turn after higher-priority blockers and routine work are handled.",
                source_section="TURN_PLAN",
                trigger="default close-turn action",
            )
        )

    if not candidates:
        notes.append("No actionable civ6-mcp planning hints were found.")

    return Civ6McpTurnPlan(intents=_dedupe_and_sort(candidates), notes=tuple(notes))


def _coerce_state_and_sections(
    state: StateBundle | Civ6McpNormalizedObservation | dict[str, Any],
) -> tuple[StateBundle | None, dict[str, str]]:
    if isinstance(state, Civ6McpNormalizedObservation):
        sections = {str(key): str(value) for key, value in state.raw_sections.items() if str(value).strip()}
        bundle = StateBundle(overview=parse_game_overview(sections.get("OVERVIEW", "")))
        return bundle, sections
    if isinstance(state, StateBundle):
        return state, section_texts_for_bundle(state)
    bundle = state_bundle_from_raw_mcp_state(state)
    return bundle, section_texts_for_bundle(bundle)


def _maybe_add_missing_overview(candidates: list[Civ6McpPrioritizedIntent], sections: dict[str, str]) -> None:
    if "OVERVIEW" not in sections:
        candidates.append(
            _intent(
                5,
                "get_game_overview",
                reasoning="Refresh missing turn overview before deciding actions.",
                source_section="OVERVIEW",
                trigger="missing section",
            )
        )


def _add_diagnostic_retries(candidates: list[Civ6McpPrioritizedIntent], sections: dict[str, str]) -> None:
    diagnostics = sections.get("STATE_DIAGNOSTICS", "")
    if not diagnostics.strip():
        return
    for offset, tool in enumerate(_DIAGNOSTIC_TOOL_PATTERN.findall(diagnostics)):
        if tool in OBSERVATION_TOOLS:
            candidates.append(
                _intent(
                    10 + offset,
                    tool,
                    reasoning="Retry observation that was missing, failed, or malformed in normalized state.",
                    source_section="STATE_DIAGNOSTICS",
                    trigger=_first_matching_line(diagnostics, tool),
                )
            )


def _add_pending_decision_intents(candidates: list[Civ6McpPrioritizedIntent], sections: dict[str, str]) -> None:
    pending_diplomacy = sections.get("PENDING_DIPLOMACY", "")
    if _has_actionable_text(pending_diplomacy):
        candidates.append(
            _intent(
                20,
                "get_pending_diplomacy",
                reasoning="Resolve incoming diplomacy before ending the turn.",
                source_section="PENDING_DIPLOMACY",
                trigger=_compact_trigger(pending_diplomacy),
            )
        )

    pending_trades = sections.get("PENDING_TRADES", "")
    if _has_actionable_text(pending_trades):
        candidates.append(
            _intent(
                25,
                "get_pending_trades",
                reasoning="Resolve incoming or pending trades before ending the turn.",
                source_section="PENDING_TRADES",
                trigger=_compact_trigger(pending_trades),
            )
        )


def _add_notification_intents(candidates: list[Civ6McpPrioritizedIntent], sections: dict[str, str]) -> None:
    notification_text = sections.get("NOTIFICATIONS", "")
    if not _has_actionable_text(notification_text):
        return
    lowered = notification_text.lower()

    if _contains_any(lowered, ("choose production", "production needed", "city production", "needs production")):
        candidates.append(
            _intent(
                40,
                "get_city_production",
                reasoning="Inspect city production choices surfaced by notifications.",
                source_section="NOTIFICATIONS",
                trigger=_compact_trigger(notification_text),
            )
        )
    if _contains_any(lowered, ("choose research", "choose civic", "technology", "civic")):
        candidates.append(
            _intent(
                45,
                "get_tech_civics",
                reasoning="Inspect available research or civic choices surfaced by notifications.",
                source_section="NOTIFICATIONS",
                trigger=_compact_trigger(notification_text),
            )
        )
    if _contains_any(lowered, ("unit needs orders", "needs orders", "unit available", "unit can move")):
        candidates.append(
            _intent(
                50,
                "get_units",
                reasoning="Inspect units requiring orders before ending the turn.",
                source_section="NOTIFICATIONS",
                trigger=_compact_trigger(notification_text),
            )
        )
    if _contains_any(lowered, ("promotion", "promote unit")):
        candidates.append(
            _intent(
                55,
                "get_unit_promotions",
                reasoning="Inspect available unit promotions before acting.",
                source_section="NOTIFICATIONS",
                trigger=_compact_trigger(notification_text),
            )
        )
    if _contains_any(lowered, ("policy", "government")):
        candidates.append(
            _intent(
                60,
                "get_policies",
                reasoning="Inspect policy or government choices surfaced by notifications.",
                source_section="NOTIFICATIONS",
                trigger=_compact_trigger(notification_text),
            )
        )


def _add_state_gap_intents(
    candidates: list[Civ6McpPrioritizedIntent],
    bundle: StateBundle | None,
    sections: dict[str, str],
) -> None:
    if bundle is not None:
        overview = bundle.overview
        if overview.raw_text and (overview.current_research is None or overview.current_civic is None):
            candidates.append(
                _intent(
                    70,
                    "get_tech_civics",
                    reasoning="Research or civic is unset or unparsed in the normalized overview.",
                    source_section="OVERVIEW",
                    trigger="missing current_research/current_civic",
                )
            )

    overview_text = sections.get("OVERVIEW", "").lower()
    if _contains_any(overview_text, ("research:", "civic:")) and _contains_any(
        overview_text,
        ("research:\ncivic:", "research: \ncivic:", "research:\r\ncivic:"),
    ):
        candidates.append(
            _intent(
                70,
                "get_tech_civics",
                reasoning="Overview indicates blank research and civic choices.",
                source_section="OVERVIEW",
                trigger="blank research/civic labels",
            )
        )

    units_text = sections.get("UNITS", "").lower()
    if _contains_any(units_text, ("needs orders", "awaiting orders", "can move", "idle")):
        candidates.append(
            _intent(
                80,
                "get_units",
                reasoning="Unit section indicates units may still need orders.",
                source_section="UNITS",
                trigger=_compact_trigger(sections.get("UNITS", "")),
            )
        )

    cities_text = sections.get("CITIES", "").lower()
    if _contains_any(cities_text, ("choose production", "no production", "needs production", "idle")):
        candidates.append(
            _intent(
                85,
                "get_city_production",
                reasoning="City section indicates a city may need production assignment.",
                source_section="CITIES",
                trigger=_compact_trigger(sections.get("CITIES", "")),
            )
        )


def _add_strategy_intents(
    candidates: list[Civ6McpPrioritizedIntent],
    sections: dict[str, str],
    strategy: str,
) -> None:
    lowered = strategy.lower()
    if not lowered:
        return
    if "science" in lowered or "campus" in lowered:
        if "CITIES" in sections:
            candidates.append(
                _intent(
                    120,
                    "get_district_advisor",
                    reasoning="Science strategy benefits from district/campus planning data.",
                    source_section="STRATEGY",
                    trigger="science/campus strategy",
                )
            )
        candidates.append(
            _intent(
                130,
                "get_victory_progress",
                reasoning="Track science-victory trajectory against the declared strategy.",
                source_section="STRATEGY",
                trigger="science strategy",
            )
        )
    if "culture" in lowered or "tourism" in lowered:
        candidates.append(
            _intent(
                130,
                "get_victory_progress",
                reasoning="Track culture-victory trajectory against the declared strategy.",
                source_section="STRATEGY",
                trigger="culture/tourism strategy",
            )
        )
    if "religion" in lowered or "faith" in lowered:
        candidates.append(
            _intent(
                130,
                "get_religion_spread",
                reasoning="Inspect religion spread for the declared faith strategy.",
                source_section="STRATEGY",
                trigger="religion/faith strategy",
            )
        )


def _end_turn_reflections(
    bundle: StateBundle | None,
    sections: dict[str, str],
    strategy: str,
) -> dict[str, str]:
    turn = bundle.overview.current_turn if bundle is not None else None
    turn_text = f"Turn {turn}" if turn is not None else "Current turn"
    strategy_text = strategy.strip() or "Continue the current high-level strategy."
    section_names = ", ".join(sections) if sections else "no normalized sections"
    return {
        "tactical": f"{turn_text}: handle higher-priority blockers and routine unit/city work first.",
        "strategic": strategy_text[:240],
        "tooling": f"Deterministic planner reviewed normalized sections: {section_names}.",
        "planning": "Next turn, re-observe game overview, blockers, units, cities, and strategy-specific state.",
        "hypothesis": "If no blockers remain, ending the turn advances the current strategic plan safely.",
    }


def _dedupe_and_sort(candidates: list[Civ6McpPrioritizedIntent]) -> tuple[Civ6McpPrioritizedIntent, ...]:
    best_by_tool: dict[str, Civ6McpPrioritizedIntent] = {}
    for candidate in candidates:
        previous = best_by_tool.get(candidate.tool)
        if previous is None or (candidate.priority, candidate.reasoning) < (previous.priority, previous.reasoning):
            best_by_tool[candidate.tool] = candidate
    return tuple(sorted(best_by_tool.values(), key=lambda item: (item.priority, item.tool)))


def _intent(
    priority: int,
    tool: str,
    *,
    arguments: dict[str, Any] | None = None,
    reasoning: str,
    source_section: str,
    trigger: str,
) -> Civ6McpPrioritizedIntent:
    return Civ6McpPrioritizedIntent(
        priority=priority,
        intent=Civ6McpPlannerIntent.from_tool(tool, arguments, reasoning=reasoning),
        source_section=source_section,
        trigger=trigger,
    )


def _has_actionable_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return not any(pattern in lowered for pattern in _NO_WORK_PATTERNS)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _compact_trigger(text: str, *, max_chars: int = 96) -> str:
    compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[: max_chars - 3] + "..." if len(compact) > max_chars else compact


def _first_matching_line(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line.strip()
    return _compact_trigger(text)


__all__ = [
    "Civ6McpPrioritizedIntent",
    "Civ6McpTurnPlan",
    "build_prioritized_turn_plan",
]
