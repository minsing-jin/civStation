"""Normalized observation schema contract for the civ6-mcp backend."""

from __future__ import annotations

from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    CIV6_MCP_CONTEXT_FIELD_MAPPINGS,
    CIV6_MCP_OBSERVATION_SECTION_MAPPINGS,
    build_global_context_updates,
    build_situation_summary,
    normalize_observation_bundle,
    normalize_raw_mcp_game_state,
    section_texts_for_bundle,
)
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    GameOverviewSnapshot,
    StateBundle,
    parse_game_overview,
)

VALID_OVERVIEW = """\
Turn: 87
Era: Medieval Era
Science: +93.25/turn
Culture: +41.5/turn
Gold: +104/turn
Faith: +7.5/turn
Researching: EDUCATION (3 turns)
Civic Researching: FEUDALISM (2 turns)
"""


class FakeMcpTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeMcpCallToolResult:
    def __init__(
        self,
        *,
        content: list[object] | None = None,
        structured_content: object | None = None,
    ) -> None:
        self.content = content or []
        self.structured_content = structured_content


def test_observation_schema_maps_default_tools_to_bundle_sections() -> None:
    mapped_tools = {mapping.tool for mapping in CIV6_MCP_OBSERVATION_SECTION_MAPPINGS}

    assert {
        "get_game_overview",
        "get_units",
        "get_cities",
        "get_diplomacy",
        "get_tech_civics",
        "get_notifications",
        "get_pending_diplomacy",
        "get_pending_trades",
        "get_victory_progress",
    }.issubset(mapped_tools)

    overview_mapping = next(
        mapping for mapping in CIV6_MCP_OBSERVATION_SECTION_MAPPINGS if mapping.tool == "get_game_overview"
    )
    assert overview_mapping.bundle_attr == "overview.raw_text"
    assert overview_mapping.planner_section == "OVERVIEW"
    assert overview_mapping.required is True


def test_context_field_mappings_target_existing_context_manager_fields() -> None:
    mapped_targets = {
        (mapping.source_path, mapping.target_context, mapping.target_field)
        for mapping in CIV6_MCP_CONTEXT_FIELD_MAPPINGS
    }

    assert ("overview.current_turn", "global_context", "current_turn") in mapped_targets
    assert ("overview.game_era", "global_context", "game_era") in mapped_targets
    assert ("overview.science_per_turn", "global_context", "science_per_turn") in mapped_targets
    assert ("overview.current_research", "global_context", "current_research") in mapped_targets
    assert ("overview.current_civic", "global_context", "current_civic") in mapped_targets


def test_normalize_observation_bundle_builds_context_updates_and_sections() -> None:
    bundle = StateBundle(
        overview=parse_game_overview(VALID_OVERVIEW),
        units_text="Units:\n- Builder at (3, 4)",
        cities_text="Cities:\n- Seoul: pop 7",
        victory_progress_text="Victory Progress:\n- Science: 35%",
        extra={"get_governors": "Governors:\n- Pingala established in Seoul"},
    )

    observation = normalize_observation_bundle(bundle)

    assert observation.backend == "civ6-mcp"
    assert observation.global_context_updates == {
        "current_turn": 87,
        "game_era": "Medieval",
        "science_per_turn": 93.25,
        "culture_per_turn": 41.5,
        "gold_per_turn": 104.0,
        "faith_per_turn": 7.5,
        "current_research": "EDUCATION (3 turns)",
        "current_civic": "FEUDALISM (2 turns)",
    }
    assert observation.game_observation_updates == {
        "situation_summary": (
            "Turn 87 | Era Medieval | Sci +93.2/t | Cul +41.5/t | "
            "Research EDUCATION (3 turns) | Civic FEUDALISM (2 turns)"
        )
    }
    assert observation.raw_sections["OVERVIEW"] == VALID_OVERVIEW
    assert observation.raw_sections["UNITS"] == "Units:\n- Builder at (3, 4)"
    assert observation.raw_sections["CITIES"] == "Cities:\n- Seoul: pop 7"
    assert observation.raw_sections["VICTORY_PROGRESS"] == "Victory Progress:\n- Science: 35%"
    assert observation.raw_sections["GET_GOVERNORS"] == "Governors:\n- Pingala established in Seoul"
    assert "## OVERVIEW" in observation.planner_context
    assert "## GET_GOVERNORS" in observation.planner_context


def test_normalize_raw_mcp_game_state_builds_civstation_observation() -> None:
    observation = normalize_raw_mcp_game_state(
        {
            "get_game_overview": {
                "turn": 94,
                "era": "Modern Era",
                "yields": {"science": 212, "culture": 105},
                "current_research": "RADIO",
                "current_civic": "SUFFRAGE",
            },
            "units": "Units:\n- Tank at (7, 8)",
            "cities": "Cities:\n- Seoul: pop 12",
            "get_trade_routes": "Trade Routes:\n- Seoul -> Busan",
        }
    )

    assert observation.global_context_updates == {
        "current_turn": 94,
        "game_era": "Modern",
        "science_per_turn": 212.0,
        "culture_per_turn": 105.0,
        "current_research": "RADIO",
        "current_civic": "SUFFRAGE",
    }
    assert observation.raw_sections["UNITS"] == "Units:\n- Tank at (7, 8)"
    assert observation.raw_sections["CITIES"] == "Cities:\n- Seoul: pop 12"
    assert observation.raw_sections["GET_TRADE_ROUTES"] == "Trade Routes:\n- Seoul -> Busan"
    assert "Turn 94" in observation.game_observation_updates["situation_summary"]


def test_normalize_raw_mcp_game_state_preserves_mixed_sdk_sections_and_diagnostics(
    mixed_civ6_mcp_sdk_state_payload: dict[str, object],
) -> None:
    observation = normalize_raw_mcp_game_state(
        mixed_civ6_mcp_sdk_state_payload,
        max_section_chars=80,
    )

    assert observation.backend == "civ6-mcp"
    assert observation.global_context_updates == {
        "current_turn": 112,
        "game_era": "Information",
        "science_per_turn": 455.5,
        "culture_per_turn": 231.25,
        "gold_per_turn": 1000.0,
        "faith_per_turn": 22.0,
        "current_research": "SMART_MATERIALS",
        "current_civic": "GLOBALIZATION",
    }
    assert observation.game_observation_updates == {
        "situation_summary": (
            "Turn 112 | Era Information | Sci +455.5/t | Cul +231.2/t | Research SMART_MATERIALS | Civic GLOBALIZATION"
        )
    }
    assert observation.raw_sections["UNITS"] == "Units:\n- Mech Infantry at (10, 11)"
    assert observation.raw_sections["CITIES"] == "Cities:\n- Seoul: pop 18"
    assert observation.raw_sections["NOTIFICATIONS"] == "Notifications:\n- Choose production"
    assert observation.raw_sections["GET_TRADE_ROUTES"] == "Trade Routes:\n- Seoul -> Busan"
    assert observation.raw_sections["STATE_DIAGNOSTICS"] == (
        "missing: get_victory_progress\n"
        "failed: get_diplomacy (timeout)\n"
        "malformed: get_pending_trades (empty response body)"
    )
    assert "## STATE_DIAGNOSTICS" in observation.planner_context
    assert "missing: get_victory_progress" in observation.planner_context


def test_normalize_raw_mcp_game_state_converts_mcp_result_objects_to_planner_ready_state() -> None:
    observation = normalize_raw_mcp_game_state(
        {
            "get_game_overview": FakeMcpCallToolResult(
                structured_content={
                    "turn": "115",
                    "era": "Future Era",
                    "science": "600",
                    "culture": "250.5",
                    "current_research": "SMART_MATERIALS",
                    "current_civic": "GLOBALIZATION",
                }
            ),
            "get_units": FakeMcpCallToolResult(
                content=[
                    FakeMcpTextBlock("Units:"),
                    FakeMcpTextBlock("- Giant Death Robot at (8, 9)"),
                ]
            ),
        }
    )

    assert observation.global_context_updates == {
        "current_turn": 115,
        "game_era": "Future",
        "science_per_turn": 600.0,
        "culture_per_turn": 250.5,
        "current_research": "SMART_MATERIALS",
        "current_civic": "GLOBALIZATION",
    }
    assert observation.raw_sections["UNITS"] == "Units:\n- Giant Death Robot at (8, 9)"
    assert "## UNITS" in observation.planner_context
    assert "Turn 115" in observation.game_observation_updates["situation_summary"]


def test_normalize_observation_bundle_uses_canonical_context_field_names() -> None:
    bundle = StateBundle(overview=parse_game_overview(VALID_OVERVIEW))

    observation = normalize_observation_bundle(bundle)

    assert set(observation.global_context_updates) == {
        "current_turn",
        "game_era",
        "science_per_turn",
        "culture_per_turn",
        "gold_per_turn",
        "faith_per_turn",
        "current_research",
        "current_civic",
    }
    assert "turn" not in observation.global_context_updates
    assert "era" not in observation.global_context_updates
    assert "science" not in observation.global_context_updates
    assert "research" not in observation.global_context_updates
    assert "civic" not in observation.global_context_updates


def test_normalize_observation_bundle_defaults_when_state_is_empty() -> None:
    observation = normalize_observation_bundle(StateBundle())

    assert observation.backend == "civ6-mcp"
    assert observation.global_context_updates == {}
    assert observation.game_observation_updates == {}
    assert observation.raw_sections == {}
    assert observation.planner_context == "(no civ6-mcp state available)"


def test_global_context_updates_coerce_values_to_schema_types() -> None:
    bundle = StateBundle(
        overview=GameOverviewSnapshot(
            current_turn="12",
            game_era=101,
            science_per_turn="13.75",
            culture_per_turn=9,
            gold_per_turn="-2.5",
            faith_per_turn="0",
            current_research=404,
            current_civic=True,
        )
    )

    updates = build_global_context_updates(bundle)

    assert updates == {
        "current_turn": 12,
        "game_era": "101",
        "science_per_turn": 13.75,
        "culture_per_turn": 9.0,
        "gold_per_turn": -2.5,
        "faith_per_turn": 0.0,
        "current_research": "404",
        "current_civic": "True",
    }
    assert isinstance(updates["current_turn"], int)
    assert isinstance(updates["science_per_turn"], float)
    assert isinstance(updates["culture_per_turn"], float)
    assert isinstance(updates["gold_per_turn"], float)
    assert isinstance(updates["faith_per_turn"], float)
    assert isinstance(updates["game_era"], str)
    assert isinstance(updates["current_research"], str)
    assert isinstance(updates["current_civic"], str)


def test_global_context_updates_skip_unparsed_optional_fields() -> None:
    bundle = StateBundle(overview=parse_game_overview("Turn: 5\n"))

    assert build_global_context_updates(bundle) == {"current_turn": 5}
    assert build_situation_summary(bundle) == "Turn 5"


def test_section_texts_for_bundle_omits_empty_sections() -> None:
    bundle = StateBundle(cities_text="Cities:\n- Seoul")

    sections = section_texts_for_bundle(bundle)

    assert sections == {"CITIES": "Cities:\n- Seoul"}
