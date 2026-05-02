"""Light parsing of civ6-mcp text responses into ContextManager fields.

The upstream server returns narrated text (not JSON). We extract just the
fields ContextManager already understands; the full text body is preserved
separately so the planner LLM still sees rich context.

Heuristic by design — civ6-mcp's text format is not a stable contract. We
try multiple regex shapes and silently fall back to "leave the field
unchanged" rather than crashing the agent loop.

Parsing helper inventory:
- ``parse_game_overview``: raw text, SDK-style result objects, or structured
  overview mappings to ``GameOverviewSnapshot``. The parsed shape contains the
  original rendered ``raw_text`` plus typed turn, era, speed, civilization,
  leader, yield, population, military, research, civic, and game-over fields.
- ``state_bundle_from_raw_mcp_state``: raw observation state mappings,
  single overview payloads, or SDK-style result objects to ``StateBundle``.
  The parsed shape contains one overview snapshot, fixed ``*_text`` sections
  for known observation tools, ``extra`` text keyed by unknown tool name, and
  observation diagnostic containers.
- ``StateBundle.to_planner_context``: parsed bundle to the normalized planner
  context string produced by ``observation_schema.normalize_observation_bundle``.

Output-shape invariants that downstream normalization depends on:
- ``parse_game_overview`` always returns ``GameOverviewSnapshot``; it never
  exposes upstream aliases or raw dictionaries to callers.
- ``GameOverviewSnapshot.raw_text`` is always a string, missing parsed values
  stay ``None``, and ``is_game_over`` defaults to ``False``.
- Parsed numeric overview fields are finite ``int``/``float`` values; malformed
  or non-finite source values are omitted instead of represented as strings.
- ``state_bundle_from_raw_mcp_state`` always returns ``StateBundle`` unless the
  input is already a ``StateBundle`` instance.
- Known observation tools are routed to fixed ``*_text`` fields, unknown
  non-empty tools are preserved in ``extra`` by tool name, and observation
  diagnostics keep their tuple/dict shapes.

Internal normalization helper inventory:
- ``_looks_like_overview_payload`` identifies structured overview-only mappings
  by known upstream aliases so bundle parsing can distinguish them from
  multi-tool observation mappings. It returns a boolean only and performs no
  value validation.
- ``_coerce_string_tuple`` and ``_coerce_string_dict`` normalize diagnostic
  payloads into planner-safe containers, accepting scalars or mixed key/value
  types and stringifying rather than failing. Unusable diagnostic mappings are
  dropped as empty dictionaries.
- ``render_payload_body`` is imported from ``_payload`` and applies the shared
  MCP wrapper precedence plus canonical text rendering: strings pass through,
  bytes decode with replacement, string/bytes sequences join with newlines,
  structured objects are model-dumped then JSON-rendered with sorted keys, and
  unrenderable values fall back to ``str``.
- ``_load_structured_payload`` delegates JSON string decoding to
  ``_payload._load_json_payload`` while preserving lenient overview probing:
  dictionaries and JSON object/array strings are accepted, empty/non-JSON
  strings and non-string inputs return ``None``, and malformed JSON is logged
  then ignored so regex parsing can continue.
- ``_apply_structured_overview`` maps upstream overview aliases to
  ``GameOverviewSnapshot`` fields and delegates all value cleaning to the
  scalar helpers below.
- ``_first_present``, ``_first_number``, and ``_first_int`` implement alias
  priority. Payload-level values win before nested ``yields`` values, and
  invalid candidates are skipped until a finite value is found.
- ``_coerce_int`` and ``_coerce_float`` reject ``None``, booleans, malformed
  numbers, and non-finite floats; valid ints are truncated through ``int`` to
  match existing ContextManager fields.
- ``_normalize_era_name`` and ``_clean_short_text`` strip labels/whitespace and
  length-limit noisy upstream text so malformed sections stay omitted.
- ``_parse_civilization_and_leader`` and ``_structured_civilization_fields``
  normalize either ``"Civ (Leader)"`` strings or structured civilization
  mappings into separate short-text fields.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from civStation.agent.modules.backend.civ6_mcp._payload import (
    _load_json_payload,
    render_payload_body,
)

logger = logging.getLogger(__name__)


@dataclass
class GameOverviewSnapshot:
    """Stable civ6-mcp overview fields plus the rendered source text."""

    raw_text: str = ""
    current_turn: int | None = None
    game_era: str | None = None
    game_speed: str | None = None
    civilization_name: str | None = None
    leader_name: str | None = None
    gold: int | None = None
    science_per_turn: float | None = None
    culture_per_turn: float | None = None
    gold_per_turn: float | None = None
    faith: int | None = None
    faith_per_turn: float | None = None
    total_population: int | None = None
    military_strength: int | None = None
    unit_count: int | None = None
    current_research: str | None = None
    current_civic: str | None = None
    is_game_over: bool = False
    victory_text: str | None = None


_TURN_PATTERNS = (
    re.compile(r"(?im)^\s*Turn(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)(\d{1,4})[^\S\r\n]*$"),
    re.compile(r"(?i)\bturn\s*(\d{1,4})\b"),
)
_ERA_PATTERN = re.compile(r"(?im)^\s*Era[:\s]+([A-Za-z][A-Za-z _\-]+?)\s*(?:Era|$)")
_GAME_SPEED_PATTERN = re.compile(r"(?im)^\s*Game\s*Speed[:\s]+([^\n]+?)\s*$")
_CIVILIZATION_PATTERN = re.compile(
    r"(?im)^\s*(?:Civilization|Civilisation|Player\s+Civilization|Civ)[:\s]+([^\n]+?)\s*$"
)
_LEADER_PATTERN = re.compile(r"(?im)^\s*(?:Leader|Player\s+Leader)[:\s]+([^\n]+?)\s*$")
_RESEARCH_PATTERN = re.compile(r"(?im)^\s*Research(?:ing)?(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)([^\n]+?)\s*$")
_CIVIC_PATTERN = re.compile(
    r"(?im)^\s*Civic(?:[^\S\r\n]+Research(?:ing)?)?(?:[^\S\r\n]*:[^\S\r\n]*|[^\S\r\n]+)([^\n]+?)\s*$"
)
_YIELD_PATTERN = re.compile(
    r"(?im)^\s*(Science|Culture|Gold|Faith)[^\S\r\n]*:[^\S\r\n]*([+\-]?\d+(?:\.\d+)?)\s*/\s*turn\b"
)
_BALANCE_WITH_PER_TURN_PATTERN = re.compile(
    r"(?im)^\s*(Gold|Faith)[^\S\r\n]*:[^\S\r\n]*(\d+)\s*\(\s*([+\-]?\d+(?:\.\d+)?)\s*/\s*turn\s*\)\s*$"
)
_TOTAL_POPULATION_PATTERN = re.compile(r"(?im)^\s*(?:Total\s+Population|Population)[:\s]+(\d+)\s*$")
_MILITARY_STRENGTH_PATTERN = re.compile(r"(?im)^\s*Military\s+Strength[:\s]+(\d+)\s*$")
_UNIT_COUNT_PATTERN = re.compile(r"(?im)^\s*(?:Unit\s+Count|Units?)[:\s]+(\d+)\s*$")
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
        "game_speed",
        "gameSpeed",
        "speed",
        "civilization_name",
        "civilization",
        "civilisation",
        "civ",
        "player_civilization",
        "playerCivilization",
        "leader_name",
        "leader",
        "player_leader",
        "playerLeader",
        "gold_balance",
        "gold_amount",
        "current_gold",
        "treasury",
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
        "faith_balance",
        "faith_amount",
        "current_faith",
        "faith_per_turn",
        "faith",
        "faithPerTurn",
        "total_population",
        "population",
        "totalPopulation",
        "military_strength",
        "militaryStrength",
        "unit_count",
        "unitCount",
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
    """Normalize a civ6-mcp overview response into a stable snapshot.

    Inputs may be raw text/bytes, JSON strings, structured mappings, or MCP
    SDK-style result wrappers. The output is always ``GameOverviewSnapshot``:
    ``raw_text`` contains the rendered source, missing fields remain ``None``,
    game-over defaults to ``False``, and malformed values are ignored rather
    than raised. Normalization strips era suffixes, splits ``"Civ (Leader)"``
    labels, length-limits short text fields, and keeps only finite numeric
    values for ContextManager-compatible ``int``/``float`` fields.
    """
    rendered_payload = render_payload_body(text)
    snapshot = GameOverviewSnapshot(raw_text=rendered_payload.text)
    structured = _load_structured_payload(rendered_payload.value)
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

    game_speed_match = _GAME_SPEED_PATTERN.search(text)
    if game_speed_match:
        snapshot.game_speed = _clean_short_text(game_speed_match.group(1))

    civilization_match = _CIVILIZATION_PATTERN.search(text)
    if civilization_match:
        civilization_name, leader_name = _parse_civilization_and_leader(civilization_match.group(1))
        snapshot.civilization_name = civilization_name
        if leader_name and not snapshot.leader_name:
            snapshot.leader_name = leader_name

    leader_match = _LEADER_PATTERN.search(text)
    if leader_match:
        snapshot.leader_name = _clean_short_text(leader_match.group(1))

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

    for kind, value_str, per_turn_str in _BALANCE_WITH_PER_TURN_PATTERN.findall(text):
        amount = _coerce_int(value_str)
        per_turn = _coerce_float(per_turn_str)
        kind_lower = kind.lower()
        if kind_lower == "gold":
            snapshot.gold = amount
            snapshot.gold_per_turn = per_turn
        elif kind_lower == "faith":
            snapshot.faith = amount
            snapshot.faith_per_turn = per_turn

    total_population_match = _TOTAL_POPULATION_PATTERN.search(text)
    if total_population_match:
        snapshot.total_population = _coerce_int(total_population_match.group(1))

    military_strength_match = _MILITARY_STRENGTH_PATTERN.search(text)
    if military_strength_match:
        snapshot.military_strength = _coerce_int(military_strength_match.group(1))

    unit_count_match = _UNIT_COUNT_PATTERN.search(text)
    if unit_count_match:
        snapshot.unit_count = _coerce_int(unit_count_match.group(1))

    over_match = _GAME_OVER_PATTERN.search(text)
    if over_match:
        snapshot.is_game_over = True
        snapshot.victory_text = over_match.group(1).strip() or None

    return snapshot


def state_bundle_from_raw_mcp_state(raw_state: Any) -> StateBundle:
    """Normalize raw civ6-mcp observation state into a stable bundle.

    Inputs may be existing ``StateBundle`` instances, overview-only payloads,
    multi-tool mappings with canonical or short tool names, or SDK-style result
    wrappers. Existing bundles are returned unchanged; every other input
    produces a ``StateBundle``. Unknown non-empty tools are preserved in
    ``extra``, known tools fill fixed ``*_text`` fields, and malformed or empty
    tool bodies are skipped. Diagnostic fields are normalized into
    tuple/dict-of-string containers so planner rendering does not fail on mixed
    upstream error shapes.
    """
    if isinstance(raw_state, StateBundle):
        return raw_state

    bundle = StateBundle()
    if not isinstance(raw_state, dict):
        bundle.overview = parse_game_overview(raw_state)
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
        if tool == "get_game_overview":
            bundle.overview = parse_game_overview(raw_payload)
            continue

        text = render_payload_body(raw_payload).text.strip()
        if not text:
            continue
        attr = _BUNDLE_TEXT_ATTRS.get(tool)
        if attr is not None:
            setattr(bundle, attr, text)
        else:
            bundle.extra[tool] = text

    return bundle


def _looks_like_overview_payload(payload: dict[str, Any]) -> bool:
    """Return whether a mapping should be treated as a standalone overview.

    Inputs are raw dictionaries from civ6-mcp or SDK result unwrapping. The
    helper checks only for known overview aliases and does not validate values;
    malformed or partial overview payloads still take the overview parsing path
    so missing fields remain ``None`` instead of being misrouted as extras.
    """
    return any(key in payload for key in _OVERVIEW_PAYLOAD_KEYS)


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a diagnostic value into a tuple of strings.

    ``None`` becomes an empty tuple, a single string remains one entry, common
    iterables are stringified element-by-element, and all other scalars become a
    one-item tuple. This keeps diagnostics displayable without rejecting unusual
    upstream error shapes.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return (str(value),)


def _coerce_string_dict(value: Any) -> dict[str, str]:
    """Normalize diagnostic mappings into ``dict[str, str]``.

    Non-mapping inputs are ignored because there is no reliable tool-to-reason
    association to preserve. Mapping keys and values are stringified so planner
    diagnostics remain serializable even when upstream errors use non-string
    identifiers or exception-like values.
    """
    if not isinstance(value, dict):
        return {}
    return {str(key): str(reason) for key, reason in value.items()}


def _load_structured_payload(payload: Any) -> Any | None:
    """Return structured overview data when the payload is already structured.

    Dictionaries are accepted directly. Strings are parsed only when they look
    like JSON objects or arrays after trimming whitespace; empty strings,
    non-JSON text, malformed JSON, and non-string values return ``None`` so the
    caller can continue with heuristic text parsing.
    """
    try:
        return _load_json_payload(payload, require_json_container=True)
    except json.JSONDecodeError:
        logger.debug("civ6-mcp overview looked like JSON but could not be decoded", exc_info=True)
        return None


def _apply_structured_overview(snapshot: GameOverviewSnapshot, payload: dict[str, Any]) -> None:
    """Apply structured overview aliases to an existing snapshot in place.

    The input is a dictionary from a direct structured payload or decoded JSON.
    Alias groups are checked in priority order, numeric values are coerced to
    finite numbers, short labels are cleaned and length-limited, nested
    ``yields`` are consulted after top-level values, and malformed fields are
    left unset rather than raising.
    """
    turn = _first_present(payload, "current_turn", "turn", "turn_number", "game_turn")
    parsed_turn = _coerce_int(turn)
    if parsed_turn is not None:
        snapshot.current_turn = parsed_turn

    era = _first_present(payload, "game_era", "era")
    if era is not None:
        snapshot.game_era = _normalize_era_name(str(era))

    game_speed = _first_present(payload, "game_speed", "gameSpeed", "speed")
    if game_speed is not None:
        snapshot.game_speed = _clean_short_text(str(game_speed))

    civilization = _first_present(
        payload,
        "civilization_name",
        "civilization",
        "civilisation",
        "civ",
        "player_civilization",
        "playerCivilization",
    )
    if civilization is not None:
        civilization_name, leader_name = _structured_civilization_fields(civilization)
        snapshot.civilization_name = civilization_name
        if leader_name and not snapshot.leader_name:
            snapshot.leader_name = leader_name

    leader = _first_present(payload, "leader_name", "leader", "player_leader", "playerLeader")
    if leader is not None:
        snapshot.leader_name = _clean_short_text(str(leader))

    yields = payload.get("yields") if isinstance(payload.get("yields"), dict) else {}
    snapshot.gold = _first_int(payload, "gold_balance", "gold_amount", "current_gold", "treasury")
    snapshot.science_per_turn = _first_number(payload, yields, "science_per_turn", "science", "sciencePerTurn")
    snapshot.culture_per_turn = _first_number(payload, yields, "culture_per_turn", "culture", "culturePerTurn")
    snapshot.gold_per_turn = _first_number(payload, yields, "gold_per_turn", "gold", "goldPerTurn")
    snapshot.faith = _first_int(payload, "faith_balance", "faith_amount", "current_faith")
    snapshot.faith_per_turn = _first_number(payload, yields, "faith_per_turn", "faith", "faithPerTurn")
    snapshot.total_population = _first_int(payload, "total_population", "population", "totalPopulation")
    snapshot.military_strength = _first_int(payload, "military_strength", "militaryStrength")
    snapshot.unit_count = _first_int(payload, "unit_count", "unitCount")

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
    """Return the first non-``None`` value for the provided keys.

    The first argument must be a dictionary source and all remaining arguments
    are candidate keys in priority order. Missing keys, explicit ``None``
    values, and non-dictionary sources return ``None``.
    """
    source = sources_and_keys[0]
    keys = sources_and_keys[1:]
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key in source and source[key] is not None:
            return source[key]
    return None


def _first_number(payload: dict[str, Any], yields: dict[str, Any], *keys: str) -> float | None:
    """Return the first finite numeric value from overview aliases.

    For each alias, the top-level payload is checked before the nested
    ``yields`` mapping. Invalid, boolean, missing, or non-finite values are
    skipped so a later alias can still populate the parsed overview field.
    """
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


def _first_int(payload: dict[str, Any], *keys: str) -> int | None:
    """Return the first finite integer-like value from top-level aliases.

    Values are parsed with ``_coerce_int`` and therefore reject booleans,
    malformed strings, and non-finite numbers. No nested ``yields`` lookup is
    performed because the integer fields represent balances or counts.
    """
    for key in keys:
        parsed = _coerce_int(_first_present(payload, key))
        if parsed is not None:
            return parsed
    return None


def _coerce_int(value: Any) -> int | None:
    """Coerce a finite scalar to ``int`` or return ``None``.

    The helper shares float validation with ``_coerce_float`` and truncates
    valid values through Python's ``int`` constructor to preserve the existing
    ContextManager count/balance field shape.
    """
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _coerce_float(value: Any) -> float | None:
    """Coerce a scalar to a finite ``float`` or return ``None``.

    ``None`` and booleans are rejected before conversion, malformed inputs are
    ignored, and ``NaN``/infinite values are omitted. This prevents untrusted
    upstream payload text from leaking non-finite numbers into normalized state.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_era_name(value: str) -> str | None:
    """Normalize an era label by removing a trailing ``Era`` suffix.

    Empty values and labels longer than 32 characters return ``None``. The
    length guard prevents regex or structured parsing from preserving a noisy
    line as a concise overview field.
    """
    era = value.strip()
    if not era:
        return None
    era = re.sub(r"\s+Era$", "", era, flags=re.IGNORECASE).strip()
    return era if len(era) <= 32 else None


def _clean_short_text(value: str) -> str | None:
    """Strip a short text field and reject empty or oversized values."""
    cleaned = value.strip()
    return cleaned if cleaned and len(cleaned) <= 80 else None


def _parse_civilization_and_leader(value: str) -> tuple[str | None, str | None]:
    """Split a text civilization label into civilization and leader fields.

    Inputs such as ``"Korea (Seondeok)"`` produce both fields. Plain labels
    produce only the civilization name, and empty or oversized segments are
    dropped through ``_clean_short_text``.
    """
    cleaned = value.strip()
    if not cleaned:
        return None, None
    parenthetical = re.match(r"^(?P<civ>.+?)\s*\((?P<leader>[^()]+)\)\s*$", cleaned)
    if parenthetical:
        return _clean_short_text(parenthetical.group("civ")), _clean_short_text(parenthetical.group("leader"))
    return _clean_short_text(cleaned), None


def _structured_civilization_fields(value: Any) -> tuple[str | None, str | None]:
    """Normalize structured or scalar civilization payloads.

    Mapping inputs may provide separate civilization and leader aliases; scalar
    inputs are parsed with the same parenthetical convention as text overview
    lines. Missing or noisy values return ``None`` for the affected field.
    """
    if isinstance(value, dict):
        name = _first_present(value, "name", "civilization_name", "civilization", "civ")
        leader = _first_present(value, "leader", "leader_name")
        return (
            _clean_short_text(str(name)) if name is not None else None,
            _clean_short_text(str(leader)) if leader is not None else None,
        )
    return _parse_civilization_and_leader(str(value))


@dataclass
class StateBundle:
    """Stable civ6-mcp observation sections and planner-safe diagnostics."""

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
        """Render this bundle into normalized planner context text."""
        from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
            normalize_observation_bundle,
        )

        return normalize_observation_bundle(
            self,
            max_section_chars=max_section_chars,
        ).planner_context
