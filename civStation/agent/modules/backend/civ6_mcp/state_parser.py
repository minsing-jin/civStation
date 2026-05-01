"""Light parsing of civ6-mcp text responses into ContextManager fields.

The upstream server returns narrated text (not JSON). We extract just the
fields ContextManager already understands; the full text body is preserved
separately so the planner LLM still sees rich context.

Heuristic by design — civ6-mcp's text format is not a stable contract. We
try multiple regex shapes and silently fall back to "leave the field
unchanged" rather than crashing the agent loop.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GameOverviewSnapshot:
    """Best-effort structured view of `get_game_overview` output."""

    raw_text: str = ""
    current_turn: int | None = None
    game_era: str | None = None
    science_per_turn: float | None = None
    culture_per_turn: float | None = None
    gold_per_turn: float | None = None
    faith_per_turn: float | None = None
    current_research: str | None = None
    current_civic: str | None = None
    is_game_over: bool = False
    victory_text: str | None = None


_TURN_PATTERNS = (
    re.compile(r"(?im)^\s*Turn[:\s]+(\d+)"),
    re.compile(r"(?i)\bturn\s*(\d{1,4})\b"),
)
_ERA_PATTERN = re.compile(r"(?im)^\s*Era[:\s]+([A-Za-z][A-Za-z _\-]+?)\s*(?:Era|$)")
_RESEARCH_PATTERN = re.compile(r"(?im)^\s*Research(?:ing)?[:\s]+([^\n]+)")
_CIVIC_PATTERN = re.compile(r"(?im)^\s*Civic(?:\s*Research(?:ing)?)?[:\s]+([^\n]+)")
_YIELD_PATTERN = re.compile(r"(?im)\b(Science|Culture|Gold|Faith)\b[^\n]*?([+\-]?\d+(?:\.\d+)?)\s*/\s*turn")
_GAME_OVER_PATTERN = re.compile(r"\*\*\*\s*GAME OVER\s*[—-]?\s*([^\n*]*)", re.I)


def parse_game_overview(text: str) -> GameOverviewSnapshot:
    """Extract turn/era/yields/research/civic from `get_game_overview` text."""
    snapshot = GameOverviewSnapshot(raw_text=text)
    if not text:
        return snapshot

    for pattern in _TURN_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                snapshot.current_turn = int(match.group(1))
                break
            except (TypeError, ValueError):
                pass

    era_match = _ERA_PATTERN.search(text)
    if era_match:
        era_name = era_match.group(1).strip()
        if era_name and len(era_name) <= 32:
            snapshot.game_era = era_name

    research_match = _RESEARCH_PATTERN.search(text)
    if research_match:
        snapshot.current_research = research_match.group(1).strip()

    civic_match = _CIVIC_PATTERN.search(text)
    if civic_match:
        snapshot.current_civic = civic_match.group(1).strip()

    for kind, value_str in _YIELD_PATTERN.findall(text):
        try:
            value = float(value_str)
        except ValueError:
            continue
        kind_lower = kind.lower()
        if kind_lower == "science":
            snapshot.science_per_turn = value
        elif kind_lower == "culture":
            snapshot.culture_per_turn = value
        elif kind_lower == "gold":
            snapshot.gold_per_turn = value
        elif kind_lower == "faith":
            snapshot.faith_per_turn = value

    over_match = _GAME_OVER_PATTERN.search(text)
    if over_match:
        snapshot.is_game_over = True
        snapshot.victory_text = over_match.group(1).strip() or None

    return snapshot


@dataclass
class StateBundle:
    """Aggregated civ6-mcp state pulled at the start of a turn."""

    overview: GameOverviewSnapshot = field(default_factory=GameOverviewSnapshot)
    units_text: str = ""
    cities_text: str = ""
    diplomacy_text: str = ""
    tech_civics_text: str = ""
    notifications_text: str = ""
    pending_diplomacy_text: str = ""
    pending_trades_text: str = ""
    victory_progress_text: str = ""
    extra: dict[str, str] = field(default_factory=dict)
    """Arbitrary additional `get_*` calls keyed by tool name."""

    def to_planner_context(self, *, max_section_chars: int = 1200) -> str:
        """Render a compact context block for the planner LLM."""
        sections: list[tuple[str, str]] = [
            ("OVERVIEW", self.overview.raw_text),
            ("UNITS", self.units_text),
            ("CITIES", self.cities_text),
            ("TECH_CIVICS", self.tech_civics_text),
            ("DIPLOMACY", self.diplomacy_text),
            ("NOTIFICATIONS", self.notifications_text),
            ("PENDING_DIPLOMACY", self.pending_diplomacy_text),
            ("PENDING_TRADES", self.pending_trades_text),
            ("VICTORY_PROGRESS", self.victory_progress_text),
        ]
        for key, value in self.extra.items():
            sections.append((key.upper(), value))

        rendered: list[str] = []
        for label, body in sections:
            if not body:
                continue
            trimmed = body.strip()
            if len(trimmed) > max_section_chars:
                trimmed = trimmed[:max_section_chars] + "\n…(truncated)"
            rendered.append(f"## {label}\n{trimmed}")
        return "\n\n".join(rendered) if rendered else "(no civ6-mcp state available)"
