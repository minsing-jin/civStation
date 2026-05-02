"""Light parsing of civ6-mcp text responses into ContextManager fields.

The upstream server returns narrated text (not JSON). We extract just the
fields ContextManager already understands; the full text body is preserved
separately so the planner LLM still sees rich context.

Heuristic by design — civ6-mcp's text format is not a stable contract. We
try multiple regex shapes and silently fall back to "leave the field
unchanged" rather than crashing the agent loop.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

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
    re.compile(r"(?im)^\s*Turn(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)(\d{1,4})[^\S\r\n]*$"),
    re.compile(r"(?i)\bturn\s*(\d{1,4})\b"),
)
_ERA_PATTERN = re.compile(r"(?im)^\s*Era[:\s]+([A-Za-z][A-Za-z _\-]+?)\s*(?:Era|$)")
_RESEARCH_PATTERN = re.compile(r"(?im)^\s*Research(?:ing)?(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)([^\n]+?)\s*$")
_CIVIC_PATTERN = re.compile(
    r"(?im)^\s*Civic(?:[^\S\r\n]+Research(?:ing)?)?(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)([^\n]+?)\s*$"
)
_YIELD_PATTERN = re.compile(
    r"(?im)^\s*(Science|Culture|Gold|Faith)[^\S\r\n]*:[^\S\r\n]*([+\-]?\d+(?:\.\d+)?)\s*/\s*turn\b"
)
_GAME_OVER_PATTERN = re.compile(r"\*\*\*\s*GAME OVER\s*[—-]?\s*([^\n*]*)", re.I)
_RAW_TOOL_ALIASES = {
    "overview": "get_game_overview",
    "game_overview": "get_game_overview",
    "units": "get_units",
    "cities": "get_cities",
    "diplomacy": "get_diplomacy",
    "tech_civics": "get_tech_civics",
    "notifications": "get_notifications",
    "pending_diplomacy": "get_pending_diplomacy",
    "pending_trades": "get_pending_trades",
    "victory_progress": "get_victory_progress",
}
_BUNDLE_TEXT_ATTRS = {
    "get_units": "units_text",
    "get_cities": "cities_text",
    "get_diplomacy": "diplomacy_text",
    "get_tech_civics": "tech_civics_text",
    "get_notifications": "notifications_text",
    "get_pending_diplomacy": "pending_diplomacy_text",
    "get_pending_trades": "pending_trades_text",
    "get_victory_progress": "victory_progress_text",
}
_OVERVIEW_PAYLOAD_KEYS = frozenset(
    {
        "current_turn",
        "turn",
        "turn_number",
        "game_turn",
        "game_era",
        "era",
        "yields",
        "science_per_turn",
        "science",
        "sciencePerTurn",
        "culture_per_turn",
        "culture",
        "culturePerTurn",
        "gold_per_turn",
        "gold",
        "goldPerTurn",
        "faith_per_turn",
        "faith",
        "faithPerTurn",
        "current_research",
        "research",
        "researching",
        "current_civic",
        "civic",
        "civic_research",
        "civicResearching",
        "is_game_over",
        "game_over",
        "gameOver",
        "victory_text",
        "victory",
        "victoryText",
    }
)


def parse_game_overview(text: Any) -> GameOverviewSnapshot:
    """Extract turn/era/yields/research/civic from `get_game_overview` text."""
    snapshot = GameOverviewSnapshot(raw_text=_render_state_payload(text))
    structured = _load_structured_payload(text)
    if isinstance(structured, dict):
        _apply_structured_overview(snapshot, structured)

    text = snapshot.raw_text
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


def state_bundle_from_raw_mcp_state(raw_state: Any) -> StateBundle:
    """Convert raw civ6-mcp tool payloads into a ``StateBundle``.

    Accepts direct tool-name keys (``get_units``), short aliases
    (``units``), and MCP SDK-ish result dictionaries containing ``content``
    and/or ``structuredContent``.
    """
    if isinstance(raw_state, StateBundle):
        return raw_state

    bundle = StateBundle()
    if not isinstance(raw_state, dict):
        bundle.overview = parse_game_overview(_unwrap_raw_mcp_payload(raw_state))
        return bundle

    if _looks_like_overview_payload(raw_state):
        bundle.overview = parse_game_overview(raw_state)
        return bundle

    bundle.missing_tools = _coerce_string_tuple(raw_state.get("missing_tools"))
    bundle.failed_tools = _coerce_string_dict(raw_state.get("failed_tools"))
    bundle.malformed_tools = _coerce_string_dict(raw_state.get("malformed_tools"))

    for raw_key, raw_payload in raw_state.items():
        if raw_key in {"missing_tools", "failed_tools", "malformed_tools"}:
            continue
        key = str(raw_key)
        tool = _RAW_TOOL_ALIASES.get(key, key)
        payload = _unwrap_raw_mcp_payload(raw_payload)
        if tool == "get_game_overview":
            bundle.overview = parse_game_overview(payload)
            continue

        text = _render_state_payload(payload).strip()
        if not text:
            continue
        attr = _BUNDLE_TEXT_ATTRS.get(tool)
        if attr is not None:
            setattr(bundle, attr, text)
        else:
            bundle.extra[tool] = text

    return bundle


def _unwrap_raw_mcp_payload(payload: Any) -> Any:
    content = _payload_value(payload, "content")
    if isinstance(content, list):
        text_blocks = _extract_text_blocks(content)
        if text_blocks:
            return "\n".join(text_blocks)

    content_blocks = _payload_value(payload, "content_blocks")
    if isinstance(content_blocks, list | tuple):
        text_blocks = [str(block) for block in content_blocks if str(block).strip()]
        if text_blocks:
            return "\n".join(text_blocks)

    text = _payload_value(payload, "text")
    if isinstance(text, str) and text.strip():
        return text

    for key in ("structuredContent", "structured_content"):
        value = _payload_value(payload, key)
        if value is not None:
            return _dump_model(value)

    return payload


def _looks_like_overview_payload(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in _OVERVIEW_PAYLOAD_KEYS)


def _extract_text_blocks(content: list[Any]) -> list[str]:
    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)
        if isinstance(text, str):
            text_blocks.append(text)
    return text_blocks


def _payload_value(payload: Any, key: str) -> Any | None:
    if isinstance(payload, dict):
        return payload.get(key)
    if hasattr(payload, key):
        return getattr(payload, key)
    return None


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return (str(value),)


def _coerce_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(reason) for key, reason in value.items()}


def _render_state_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    if isinstance(payload, list | tuple) and all(isinstance(item, str | bytes) for item in payload):
        return "\n".join(_render_state_payload(item) for item in payload)
    payload = _dump_model(payload)
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(payload)


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _load_structured_payload(payload: Any) -> Any | None:
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str):
        return None
    stripped = payload.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("civ6-mcp overview looked like JSON but could not be decoded", exc_info=True)
        return None


def _apply_structured_overview(snapshot: GameOverviewSnapshot, payload: dict[str, Any]) -> None:
    turn = _first_present(payload, "current_turn", "turn", "turn_number", "game_turn")
    parsed_turn = _coerce_int(turn)
    if parsed_turn is not None:
        snapshot.current_turn = parsed_turn

    era = _first_present(payload, "game_era", "era")
    if era is not None:
        snapshot.game_era = _normalize_era_name(str(era))

    yields = payload.get("yields") if isinstance(payload.get("yields"), dict) else {}
    snapshot.science_per_turn = _first_number(payload, yields, "science_per_turn", "science", "sciencePerTurn")
    snapshot.culture_per_turn = _first_number(payload, yields, "culture_per_turn", "culture", "culturePerTurn")
    snapshot.gold_per_turn = _first_number(payload, yields, "gold_per_turn", "gold", "goldPerTurn")
    snapshot.faith_per_turn = _first_number(payload, yields, "faith_per_turn", "faith", "faithPerTurn")

    research = _first_present(payload, "current_research", "research", "researching")
    if research is not None and str(research).strip():
        snapshot.current_research = str(research).strip()

    civic = _first_present(payload, "current_civic", "civic", "civic_research", "civicResearching")
    if civic is not None and str(civic).strip():
        snapshot.current_civic = str(civic).strip()

    game_over = _first_present(payload, "is_game_over", "game_over", "gameOver")
    if isinstance(game_over, bool):
        snapshot.is_game_over = game_over
    victory_text = _first_present(payload, "victory_text", "victory", "victoryText")
    if victory_text is not None and str(victory_text).strip():
        snapshot.victory_text = str(victory_text).strip()
        if snapshot.is_game_over is False:
            snapshot.is_game_over = True


def _first_present(*sources_and_keys: Any) -> Any | None:
    source = sources_and_keys[0]
    keys = sources_and_keys[1:]
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _first_number(payload: dict[str, Any], yields: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _first_present(payload, key)
        number = _coerce_float(value)
        if number is not None:
            return number
        value = _first_present(yields, key)
        number = _coerce_float(value)
        if number is not None:
            return number
    return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_era_name(value: str) -> str | None:
    era = value.strip()
    if not era:
        return None
    era = re.sub(r"\s+Era$", "", era, flags=re.IGNORECASE).strip()
    return era if len(era) <= 32 else None


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
    missing_tools: tuple[str, ...] = ()
    failed_tools: dict[str, str] = field(default_factory=dict)
    malformed_tools: dict[str, str] = field(default_factory=dict)
    """Observation diagnostics that are safe to show the planner."""

    def to_planner_context(self, *, max_section_chars: int = 1200) -> str:
        """Render a compact context block for the planner LLM."""
        from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
            normalize_observation_bundle,
        )

        return normalize_observation_bundle(
            self,
            max_section_chars=max_section_chars,
        ).planner_context
