"""Normalized observation models for the civ6-mcp backend.

The upstream civ6-mcp server returns human-readable text for observation
tools. This module defines the stable civStation-side schema used to map
those tool responses into planner sections and ContextManager updates.

Output-shape invariants for the parsing helpers:
- ``normalize_*`` helpers always return ``Civ6McpNormalizedObservation`` with
  ``backend == "civ6-mcp"``.
- Context update dictionaries use existing ContextManager field names only;
  parsed overview aliases such as ``turn`` or ``era`` are not exposed.
- Empty or unparsed values are omitted instead of represented as ``None``.
- ``raw_sections`` is keyed by planner section labels, while ``tool_results``
  is keyed by upstream civ6-mcp tool names.
- Unknown non-empty ``get_*`` payloads are preserved as dynamic sections using
  uppercase section labels and unchanged tool-result keys.
- Missing, failed, and malformed tool diagnostics appear only in the optional
  ``STATE_DIAGNOSTICS`` section, not in ``tool_results``.
- Empty state renders ``"(no civ6-mcp state available)"`` for planner context.
- ``parse_observation_tool_response`` returns ``Civ6McpToolObservation`` for
  one validated successful ``get_*`` payload and rejects non-observation or
  empty payloads with ``ValueError``.

Parsing helper catalog:
- ``normalize_observation_bundle(bundle, max_section_chars=1200)`` accepts a
  parsed ``StateBundle`` and returns ``Civ6McpNormalizedObservation``. It does
  not raise on empty or partial bundles; missing values normalize to empty
  update dictionaries, empty section mappings, and the empty-state planner
  sentinel. It preserves the civ6-mcp backend label, omits ``None``/empty
  fields, and applies planner section truncation only when rendering
  ``planner_context``.
- ``normalize_raw_mcp_game_state(raw_state, max_section_chars=1200)`` accepts a
  raw civ6-mcp payload mapping or SDK-like object, delegates shape parsing to
  ``state_bundle_from_raw_mcp_state``, then returns the same normalized
  envelope as ``normalize_observation_bundle``. Parser diagnostics are carried
  into ``STATE_DIAGNOSTICS`` rather than raised by this layer.
- ``parse_observation_tool_response(tool, payload, max_section_chars=1200)``
  accepts one upstream observation tool name and payload, returns
  ``Civ6McpToolObservation(tool, bundle, normalized)``, and raises
  ``ValueError`` when ``tool`` is not a ``get_*`` observation tool or when the
  payload normalizes to an empty response body. The returned tool name is the
  validated input string, and the normalized result is the single-tool
  observation envelope.
- ``build_global_context_updates(bundle)`` accepts a ``StateBundle`` and
  returns typed ``update_global_context`` kwargs from
  ``CIV6_MCP_CONTEXT_FIELD_MAPPINGS``. It skips unmapped, missing, ``None``,
  and empty-string values, coercing mapped values to the declared
  ``int``/``float``/``str`` ContextManager schema type.
- ``build_game_observation_updates(bundle)`` accepts a ``StateBundle`` and
  returns ``update_game_observation`` kwargs. It includes
  ``situation_summary`` and/or ``observation_fields`` only when those derived
  values are non-empty, so empty bundles produce ``{}``.
- ``build_game_observation_fields(bundle)`` accepts a ``StateBundle`` and
  returns non-empty parsed overview fields for observation sync. It omits
  ``None`` and empty strings, includes ``is_game_over`` only when true, and
  keeps canonical overview field names.
- ``build_situation_summary(bundle)`` accepts a ``StateBundle`` and returns a
  pipe-delimited summary string, or ``""`` when no summary fields are parsed.
  Numeric yields render to one decimal place, and game-over text is appended
  only when both ``is_game_over`` and ``victory_text`` are present.
- ``section_texts_for_bundle(bundle)`` accepts a ``StateBundle`` and returns
  planner sections keyed by canonical labels. Known sections follow
  ``CIV6_MCP_OBSERVATION_SECTION_MAPPINGS`` order, dynamic extras normalize to
  uppercase tool names, empty text is omitted, and diagnostics are appended as
  ``STATE_DIAGNOSTICS`` when present.
- ``tool_results_for_bundle(bundle)`` accepts a ``StateBundle`` and returns
  successful observation text keyed by upstream civ6-mcp tool names. It shares
  the same non-empty inventory as ``section_texts_for_bundle`` but deliberately
  excludes missing/failed/malformed diagnostics.
- ``render_planner_context(sections, max_section_chars=1200)`` accepts a
  section mapping and returns ``"## LABEL\nbody"`` blocks separated by blank
  lines. Bodies are stripped, truncated with ``"...(truncated)"`` after the
  configured character limit, and empty mappings return the empty-state
  sentinel.
- Private helpers centralize shared rules: ``_nonempty_observation_texts``
  builds the known-section/dynamic-extra inventory, ``_value_at_path`` resolves
  dotted attributes to ``None`` on the first gap, ``_coerce_context_value``
  performs the declared type coercion, ``_validate_observation_tool`` raises
  ``ValueError`` for non-string or non-``get_*`` tool names, and
  ``_diagnostic_section_for_bundle`` renders missing, failed, and malformed
  tool diagnostics for planner context only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from civStation.agent.modules.backend.civ6_mcp._payload import payload_has_body
from civStation.agent.modules.backend.civ6_mcp.state_parser import StateBundle, state_bundle_from_raw_mcp_state


@dataclass(frozen=True)
class Civ6McpObservationSectionMapping:
    """Map one registered observation tool to a normalized planner section."""

    tool: str
    bundle_attr: str
    planner_section: str
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class Civ6McpContextFieldMapping:
    """Map one parsed observation field to a ContextManager update target."""

    source_path: str
    target_context: str
    target_field: str
    value_type: str


@dataclass(frozen=True)
class Civ6McpNormalizedObservation:
    """Bundle normalized context updates, planner text, and tool results for one snapshot."""

    backend: str = "civ6-mcp"
    global_context_updates: dict[str, object] = field(default_factory=dict)
    game_observation_updates: dict[str, object] = field(default_factory=dict)
    raw_sections: dict[str, str] = field(default_factory=dict)
    planner_context: str = "(no civ6-mcp state available)"
    tool_results: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Civ6McpToolObservation:
    """Bundle one validated observation-tool response with parsed and normalized state."""

    tool: str
    bundle: StateBundle
    normalized: Civ6McpNormalizedObservation


CIV6_MCP_OBSERVATION_SECTION_MAPPINGS: tuple[Civ6McpObservationSectionMapping, ...] = (
    Civ6McpObservationSectionMapping(
        tool="get_game_overview",
        bundle_attr="overview.raw_text",
        planner_section="OVERVIEW",
        required=True,
        description="Turn, era, yields, current research/civic, and terminal game status.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_units",
        bundle_attr="units_text",
        planner_section="UNITS",
        description="Player unit list and unit-level tactical state.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_cities",
        bundle_attr="cities_text",
        planner_section="CITIES",
        description="Owned city summaries and city-level production state.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_tech_civics",
        bundle_attr="tech_civics_text",
        planner_section="TECH_CIVICS",
        description="Technology and civic progress/options.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_diplomacy",
        bundle_attr="diplomacy_text",
        planner_section="DIPLOMACY",
        description="Known civilizations, relationships, and diplomatic state.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_notifications",
        bundle_attr="notifications_text",
        planner_section="NOTIFICATIONS",
        description="Current game notifications and prompts.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_pending_diplomacy",
        bundle_attr="pending_diplomacy_text",
        planner_section="PENDING_DIPLOMACY",
        description="Incoming diplomacy choices requiring a response.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_pending_trades",
        bundle_attr="pending_trades_text",
        planner_section="PENDING_TRADES",
        description="Incoming or pending trade choices requiring a response.",
    ),
    Civ6McpObservationSectionMapping(
        tool="get_victory_progress",
        bundle_attr="victory_progress_text",
        planner_section="VICTORY_PROGRESS",
        description="Victory progress summaries exposed by civ6-mcp.",
    ),
)

CIV6_MCP_CONTEXT_FIELD_MAPPINGS: tuple[Civ6McpContextFieldMapping, ...] = (
    Civ6McpContextFieldMapping("overview.current_turn", "global_context", "current_turn", "int"),
    Civ6McpContextFieldMapping("overview.game_era", "global_context", "game_era", "str"),
    Civ6McpContextFieldMapping("overview.game_speed", "global_context", "game_speed", "str"),
    Civ6McpContextFieldMapping("overview.civilization_name", "global_context", "civilization_name", "str"),
    Civ6McpContextFieldMapping("overview.leader_name", "global_context", "leader_name", "str"),
    Civ6McpContextFieldMapping("overview.gold", "global_context", "gold", "int"),
    Civ6McpContextFieldMapping("overview.science_per_turn", "global_context", "science_per_turn", "float"),
    Civ6McpContextFieldMapping("overview.culture_per_turn", "global_context", "culture_per_turn", "float"),
    Civ6McpContextFieldMapping("overview.gold_per_turn", "global_context", "gold_per_turn", "float"),
    Civ6McpContextFieldMapping("overview.faith", "global_context", "faith", "int"),
    Civ6McpContextFieldMapping("overview.faith_per_turn", "global_context", "faith_per_turn", "float"),
    Civ6McpContextFieldMapping("overview.total_population", "global_context", "total_population", "int"),
    Civ6McpContextFieldMapping("overview.military_strength", "global_context", "military_strength", "int"),
    Civ6McpContextFieldMapping("overview.unit_count", "global_context", "unit_count", "int"),
    Civ6McpContextFieldMapping("overview.current_research", "global_context", "current_research", "str"),
    Civ6McpContextFieldMapping("overview.current_civic", "global_context", "current_civic", "str"),
)


def normalize_observation_bundle(
    bundle: StateBundle,
    *,
    max_section_chars: int = 1200,
) -> Civ6McpNormalizedObservation:
    """Build the stable civStation observation envelope from a parsed state bundle."""
    raw_sections = section_texts_for_bundle(bundle)
    return Civ6McpNormalizedObservation(
        global_context_updates=build_global_context_updates(bundle),
        game_observation_updates=build_game_observation_updates(bundle),
        tool_results=tool_results_for_bundle(bundle),
        raw_sections=raw_sections,
        planner_context=render_planner_context(raw_sections, max_section_chars=max_section_chars),
    )


def normalize_raw_mcp_game_state(
    raw_state: object,
    *,
    max_section_chars: int = 1200,
) -> Civ6McpNormalizedObservation:
    """Parse raw civ6-mcp state into the normalized observation envelope."""
    return normalize_observation_bundle(
        state_bundle_from_raw_mcp_state(raw_state),
        max_section_chars=max_section_chars,
    )


def parse_observation_tool_response(
    tool: str,
    payload: object,
    *,
    max_section_chars: int = 1200,
) -> Civ6McpToolObservation:
    """Validate one non-empty ``get_*`` payload and return parsed observation state."""
    normalized_tool = _validate_observation_tool(tool)
    if not payload_has_body(payload):
        raise ValueError(f"civ6-mcp observation tool {normalized_tool!r} returned an empty response body.")
    bundle = state_bundle_from_raw_mcp_state({normalized_tool: payload})
    normalized = normalize_observation_bundle(bundle, max_section_chars=max_section_chars)
    if not any(tool == normalized_tool and text.strip() for tool, _, text in _nonempty_observation_texts(bundle)):
        raise ValueError(f"civ6-mcp observation tool {normalized_tool!r} returned an empty response body.")
    return Civ6McpToolObservation(
        tool=normalized_tool,
        bundle=bundle,
        normalized=normalized,
    )


def build_global_context_updates(bundle: StateBundle) -> dict[str, object]:
    """Return ``update_global_context`` kwargs from mapped parsed overview fields."""
    updates: dict[str, object] = {}
    for mapping in CIV6_MCP_CONTEXT_FIELD_MAPPINGS:
        if mapping.target_context != "global_context":
            continue
        value = _value_at_path(bundle, mapping.source_path)
        if value is None or value == "":
            continue
        updates[mapping.target_field] = _coerce_context_value(value, mapping.value_type)
    return updates


def build_game_observation_updates(bundle: StateBundle) -> dict[str, object]:
    """Return ``update_game_observation`` kwargs for summary and fields."""
    updates: dict[str, object] = {}
    summary = build_situation_summary(bundle)
    if summary:
        updates["situation_summary"] = summary
    observation_fields = build_game_observation_fields(bundle)
    if observation_fields:
        updates["observation_fields"] = observation_fields
    return updates


def build_game_observation_fields(bundle: StateBundle) -> dict[str, object]:
    """Return non-empty parsed overview fields for high-level observation sync."""
    overview = bundle.overview
    fields: dict[str, object] = {}
    for field_name in (
        "current_turn",
        "game_era",
        "game_speed",
        "civilization_name",
        "leader_name",
        "gold",
        "science_per_turn",
        "culture_per_turn",
        "gold_per_turn",
        "faith",
        "faith_per_turn",
        "total_population",
        "military_strength",
        "unit_count",
        "current_research",
        "current_civic",
        "victory_text",
    ):
        value = getattr(overview, field_name)
        if value is not None and value != "":
            fields[field_name] = value
    if overview.is_game_over:
        fields["is_game_over"] = True
    return fields


def build_situation_summary(bundle: StateBundle) -> str:
    """Render parsed overview fields as a compact situation summary."""
    overview = bundle.overview
    summary_bits: list[str] = []
    if overview.current_turn is not None:
        summary_bits.append(f"Turn {overview.current_turn}")
    if overview.game_era:
        summary_bits.append(f"Era {overview.game_era}")
    if overview.science_per_turn is not None:
        summary_bits.append(f"Sci +{overview.science_per_turn:.1f}/t")
    if overview.culture_per_turn is not None:
        summary_bits.append(f"Cul +{overview.culture_per_turn:.1f}/t")
    if overview.current_research:
        summary_bits.append(f"Research {overview.current_research}")
    if overview.current_civic:
        summary_bits.append(f"Civic {overview.current_civic}")
    if overview.is_game_over and overview.victory_text:
        summary_bits.append(f"GAME OVER: {overview.victory_text}")
    return " | ".join(summary_bits)


def section_texts_for_bundle(bundle: StateBundle) -> dict[str, str]:
    """Return planner sections, dynamic extras, and diagnostics from a bundle."""
    sections = {section: text for _, section, text in _nonempty_observation_texts(bundle)}
    diagnostics = _diagnostic_section_for_bundle(bundle)
    if diagnostics:
        sections["STATE_DIAGNOSTICS"] = diagnostics
    return sections


def tool_results_for_bundle(bundle: StateBundle) -> dict[str, str]:
    """Return parsed observation text keyed by upstream civ6-mcp tool name."""
    return {tool: text for tool, _, text in _nonempty_observation_texts(bundle)}


def _nonempty_observation_texts(bundle: StateBundle) -> tuple[tuple[str, str, str], ...]:
    """Return the shared successful-observation inventory for bundle renderers.

    Each tuple is ``(tool_name, planner_section, text)``. Keeping the inventory
    in one helper prevents known-tool and dynamic-extra traversal from drifting
    between ``section_texts_for_bundle`` and ``tool_results_for_bundle``.
    """
    items: list[tuple[str, str, str]] = []
    for mapping in CIV6_MCP_OBSERVATION_SECTION_MAPPINGS:
        value = _value_at_path(bundle, mapping.bundle_attr)
        if isinstance(value, str) and value.strip():
            items.append((mapping.tool, mapping.planner_section, value))

    for tool, value in bundle.extra.items():
        if value.strip():
            items.append((tool, tool.upper(), value))
    return tuple(items)


def render_planner_context(
    sections: dict[str, str],
    *,
    max_section_chars: int = 1200,
) -> str:
    """Render section mappings as the compact planner context block."""
    rendered: list[str] = []
    for label, body in sections.items():
        trimmed = body.strip()
        if len(trimmed) > max_section_chars:
            trimmed = trimmed[:max_section_chars] + "\n...(truncated)"
        rendered.append(f"## {label}\n{trimmed}")
    return "\n\n".join(rendered) if rendered else "(no civ6-mcp state available)"


def _value_at_path(source: object, path: str) -> Any:
    """Resolve a dotted attribute path and return ``None`` at the first gap."""
    value: Any = source
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _coerce_context_value(value: object, value_type: str) -> object:
    """Coerce a parsed overview value to a ContextManager schema type."""
    if value_type == "float":
        return float(value)
    if value_type == "int":
        return int(value)
    if value_type == "str":
        return str(value)
    return value


def _validate_observation_tool(tool: str) -> str:
    """Return a valid civ6-mcp observation tool name or raise ``ValueError``."""
    if not isinstance(tool, str) or not tool.startswith("get_"):
        raise ValueError(f"{tool!r} is not a civ6-mcp observation tool.")
    return tool


def _diagnostic_section_for_bundle(bundle: StateBundle) -> str:
    """Render missing, failed, and malformed tool diagnostics for planner context."""
    lines: list[str] = []
    if bundle.missing_tools:
        lines.append(f"missing: {', '.join(bundle.missing_tools)}")
    if bundle.failed_tools:
        failed = ", ".join(f"{tool} ({reason})" for tool, reason in sorted(bundle.failed_tools.items()))
        lines.append(f"failed: {failed}")
    if bundle.malformed_tools:
        malformed = ", ".join(f"{tool} ({reason})" for tool, reason in sorted(bundle.malformed_tools.items()))
        lines.append(f"malformed: {malformed}")
    return "\n".join(lines)


__all__ = [
    "CIV6_MCP_CONTEXT_FIELD_MAPPINGS",
    "CIV6_MCP_OBSERVATION_SECTION_MAPPINGS",
    "Civ6McpContextFieldMapping",
    "Civ6McpNormalizedObservation",
    "Civ6McpObservationSectionMapping",
    "Civ6McpToolObservation",
    "build_game_observation_updates",
    "build_game_observation_fields",
    "build_global_context_updates",
    "build_situation_summary",
    "normalize_observation_bundle",
    "normalize_raw_mcp_game_state",
    "parse_observation_tool_response",
    "render_planner_context",
    "section_texts_for_bundle",
    "tool_results_for_bundle",
]
