"""Normalized observation models for the civ6-mcp backend.

The upstream civ6-mcp server returns human-readable text for observation
tools. This module defines the stable civStation-side schema used to map
those tool responses into planner sections and ContextManager updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.state_parser import StateBundle, state_bundle_from_raw_mcp_state


@dataclass(frozen=True)
class Civ6McpObservationSectionMapping:
    """Maps one civ6-mcp observation tool to a normalized planner section."""

    tool: str
    bundle_attr: str
    planner_section: str
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class Civ6McpContextFieldMapping:
    """Maps one parsed observation field to an existing ContextManager field."""

    source_path: str
    target_context: str
    target_field: str
    value_type: str


@dataclass(frozen=True)
class Civ6McpNormalizedObservation:
    """Normalized civStation observation envelope for one civ6-mcp snapshot."""

    backend: str = "civ6-mcp"
    global_context_updates: dict[str, object] = field(default_factory=dict)
    game_observation_updates: dict[str, object] = field(default_factory=dict)
    raw_sections: dict[str, str] = field(default_factory=dict)
    planner_context: str = "(no civ6-mcp state available)"
    tool_results: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Civ6McpToolObservation:
    """Validated structured observation parsed from one successful ``get_*`` response."""

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
    """Build the stable civStation observation envelope from a raw state bundle."""
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
    """Normalize a raw civ6-mcp tool-result mapping into civStation observation data."""
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
    """Validate and normalize one successful civ6-mcp ``get_*`` tool response."""
    normalized_tool = _validate_observation_tool(tool)
    if not _payload_has_observation_body(payload):
        raise ValueError(f"civ6-mcp observation tool {normalized_tool!r} returned an empty response body.")
    bundle = state_bundle_from_raw_mcp_state({normalized_tool: payload})
    normalized = normalize_observation_bundle(bundle, max_section_chars=max_section_chars)
    if not str(normalized.tool_results.get(normalized_tool) or "").strip():
        raise ValueError(f"civ6-mcp observation tool {normalized_tool!r} returned an empty response body.")
    return Civ6McpToolObservation(
        tool=normalized_tool,
        bundle=bundle,
        normalized=normalized,
    )


def build_global_context_updates(bundle: StateBundle) -> dict[str, object]:
    """Return ContextManager.update_global_context kwargs derived from a bundle."""
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
    """Return ContextManager.update_game_observation kwargs derived from a bundle."""
    updates: dict[str, object] = {}
    summary = build_situation_summary(bundle)
    if summary:
        updates["situation_summary"] = summary
    observation_fields = build_game_observation_fields(bundle)
    if observation_fields:
        updates["observation_fields"] = observation_fields
    return updates


def build_game_observation_fields(bundle: StateBundle) -> dict[str, object]:
    """Return parsed civ6-mcp overview fields for high-level observation sync."""
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
    """Render a compact situation summary for high-level context notes."""
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
    """Return non-empty normalized planner sections from a bundle."""
    sections: dict[str, str] = {}
    for mapping in CIV6_MCP_OBSERVATION_SECTION_MAPPINGS:
        value = _value_at_path(bundle, mapping.bundle_attr)
        if isinstance(value, str) and value.strip():
            sections[mapping.planner_section] = value

    for key, value in bundle.extra.items():
        if value.strip():
            sections[key.upper()] = value
    diagnostics = _diagnostic_section_for_bundle(bundle)
    if diagnostics:
        sections["STATE_DIAGNOSTICS"] = diagnostics
    return sections


def tool_results_for_bundle(bundle: StateBundle) -> dict[str, str]:
    """Return non-empty observation payloads keyed by upstream civ6-mcp tool name."""
    results: dict[str, str] = {}
    for mapping in CIV6_MCP_OBSERVATION_SECTION_MAPPINGS:
        value = _value_at_path(bundle, mapping.bundle_attr)
        if isinstance(value, str) and value.strip():
            results[mapping.tool] = value

    for tool, value in bundle.extra.items():
        if value.strip():
            results[tool] = value
    return results


def render_planner_context(
    sections: dict[str, str],
    *,
    max_section_chars: int = 1200,
) -> str:
    """Render normalized sections as the compact planner context block."""
    rendered: list[str] = []
    for label, body in sections.items():
        trimmed = body.strip()
        if len(trimmed) > max_section_chars:
            trimmed = trimmed[:max_section_chars] + "\n...(truncated)"
        rendered.append(f"## {label}\n{trimmed}")
    return "\n\n".join(rendered) if rendered else "(no civ6-mcp state available)"


def _value_at_path(source: object, path: str) -> Any:
    value: Any = source
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _coerce_context_value(value: object, value_type: str) -> object:
    if value_type == "float":
        return float(value)
    if value_type == "int":
        return int(value)
    if value_type == "str":
        return str(value)
    return value


def _validate_observation_tool(tool: str) -> str:
    if not isinstance(tool, str) or not tool.startswith("get_"):
        raise ValueError(f"{tool!r} is not a civ6-mcp observation tool.")
    return tool


def _payload_has_observation_body(payload: object) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str | bytes):
        return bool(payload.strip())
    looks_like_result = False
    text = getattr(payload, "text", None)
    looks_like_result = looks_like_result or hasattr(payload, "text")
    if isinstance(text, str) and text.strip():
        return True
    content_blocks = getattr(payload, "content_blocks", None)
    looks_like_result = looks_like_result or hasattr(payload, "content_blocks")
    if isinstance(content_blocks, list | tuple) and any(str(block).strip() for block in content_blocks):
        return True
    content = getattr(payload, "content", None)
    looks_like_result = looks_like_result or hasattr(payload, "content")
    if isinstance(content, list | tuple) and any(str(getattr(block, "text", "")).strip() for block in content):
        return True
    for structured_name in ("structured_content", "structuredContent"):
        looks_like_result = looks_like_result or hasattr(payload, structured_name)
        if getattr(payload, structured_name, None) is not None:
            return True
    if isinstance(payload, dict):
        return any(_payload_has_observation_body(value) for value in payload.values())
    if looks_like_result:
        return False
    return True


def _diagnostic_section_for_bundle(bundle: StateBundle) -> str:
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
