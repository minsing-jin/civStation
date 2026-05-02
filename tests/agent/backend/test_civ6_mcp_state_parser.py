"""Tests for civ6-mcp text-response heuristic parsing."""

from __future__ import annotations

import json
from dataclasses import asdict, fields

import pytest

from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    GameOverviewSnapshot,
    StateBundle,
    parse_game_overview,
    state_bundle_from_raw_mcp_state,
)

VALID_CIV6_MCP_OVERVIEW_PAYLOAD = """\
Game Overview
Turn 87
Era: Medieval Era
Civilization: Korea (Seondeok)
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
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
        text: str = "",
    ) -> None:
        self.content = content or []
        self.structured_content = structured_content
        self.text = text


def test_parse_overview_extracts_turn_and_era() -> None:
    text = """\
Turn: 42
Era: Classical Era
Civilization: Korea (Seondeok)
Science: +35.5/turn
Culture: +12.0/turn
Gold: -2.0/turn
Faith: +0.0/turn
Researching: WRITING
Civic Researching: CRAFTSMANSHIP
"""
    snap = parse_game_overview(text)
    assert snap.current_turn == 42
    assert snap.game_era == "Classical"
    assert snap.science_per_turn == 35.5
    assert snap.culture_per_turn == 12.0
    assert snap.gold_per_turn == -2.0
    assert snap.faith_per_turn == 0.0
    assert snap.current_research == "WRITING"
    assert snap.current_civic == "CRAFTSMANSHIP"
    assert snap.is_game_over is False


def test_parse_overview_handles_game_over() -> None:
    text = "*** GAME OVER — VICTORY! Science victory achieved at turn 240 ***"
    snap = parse_game_overview(text)
    assert snap.is_game_over is True
    assert snap.victory_text and "VICTORY" in snap.victory_text


def test_parse_overview_resilient_to_empty_input() -> None:
    snap = parse_game_overview("")
    assert snap.current_turn is None
    assert snap.game_era is None
    assert snap.is_game_over is False


def test_state_parser_output_shapes_remain_stable() -> None:
    snap = parse_game_overview("")
    bundle = state_bundle_from_raw_mcp_state({})

    assert tuple(field.name for field in fields(snap)) == (
        "raw_text",
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
        "is_game_over",
        "victory_text",
    )
    assert snap.raw_text == ""
    assert snap.is_game_over is False
    assert all(getattr(snap, field.name) is None for field in fields(snap)[1:-2])
    assert snap.victory_text is None

    assert isinstance(bundle, StateBundle)
    assert tuple(field.name for field in fields(bundle)) == (
        "overview",
        "units_text",
        "cities_text",
        "diplomacy_text",
        "tech_civics_text",
        "notifications_text",
        "pending_diplomacy_text",
        "pending_trades_text",
        "victory_progress_text",
        "extra",
        "missing_tools",
        "failed_tools",
        "malformed_tools",
    )
    assert bundle.units_text == ""
    assert bundle.extra == {}
    assert bundle.missing_tools == ()
    assert bundle.failed_tools == {}
    assert bundle.malformed_tools == {}


def test_parse_valid_civ6_mcp_overview_payload_extracts_context_fields() -> None:
    snap = parse_game_overview(VALID_CIV6_MCP_OVERVIEW_PAYLOAD)

    assert snap.raw_text == VALID_CIV6_MCP_OVERVIEW_PAYLOAD
    assert snap.current_turn == 87
    assert snap.game_era == "Medieval"
    assert snap.civilization_name == "Korea"
    assert snap.leader_name == "Seondeok"
    assert snap.science_per_turn == 93.25
    assert snap.culture_per_turn == 41.5
    assert snap.gold_per_turn == 104.0
    assert snap.faith_per_turn == 7.5
    assert snap.current_research == "EDUCATION (3 turns)"
    assert snap.current_civic == "FEUDALISM (2 turns)"
    assert snap.is_game_over is False
    assert snap.victory_text is None


@pytest.mark.parametrize(
    ("payload", "missing_field"),
    [
        (
            """\
Game Overview
Era: Medieval Era
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "current_turn",
        ),
        (
            """\
Game Overview
Turn 87
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "game_era",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "science_per_turn",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Science: +93.25 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "culture_per_turn",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Science: +93.25 / turn
Culture: +41.5 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "gold_per_turn",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
""",
            "faith_per_turn",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Civic: FEUDALISM (2 turns)
""",
            "current_research",
        ),
        (
            """\
Game Overview
Turn 87
Era: Medieval Era
Science: +93.25 / turn
Culture: +41.5 / turn
Gold: +104 / turn
Faith: +7.5 / turn
Research: EDUCATION (3 turns)
""",
            "current_civic",
        ),
    ],
)
def test_parse_civ6_mcp_overview_payload_leaves_missing_fields_unset(
    payload: str,
    missing_field: str,
) -> None:
    snap = parse_game_overview(payload)

    assert snap.raw_text == payload
    assert getattr(snap, missing_field) is None


def test_parse_civ6_mcp_overview_payload_ignores_malformed_scalar_values() -> None:
    payload = """\
Game Overview
Turn: 87-ish
Era: Medieval Era
Science: +many / turn
Culture: NaN / turn
Gold: infinity / turn
Faith: --7 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
"""

    snap = parse_game_overview(payload)

    assert snap.raw_text == payload
    assert snap.current_turn is None
    assert snap.game_era == "Medieval"
    assert snap.science_per_turn is None
    assert snap.culture_per_turn is None
    assert snap.gold_per_turn is None
    assert snap.faith_per_turn is None
    assert snap.current_research == "EDUCATION (3 turns)"
    assert snap.current_civic == "FEUDALISM (2 turns)"


def test_parse_civ6_mcp_overview_payload_does_not_cross_line_labels_for_blank_values() -> None:
    payload = """\
Game Overview
Turn 88
Era: Renaissance Era
Science: +20 / turn
Culture: +10 / turn
Gold: +5 / turn
Faith: +1 / turn
Research:
Civic:
"""

    snap = parse_game_overview(payload)

    assert snap.current_turn == 88
    assert snap.game_era == "Renaissance"
    assert snap.science_per_turn == 20.0
    assert snap.culture_per_turn == 10.0
    assert snap.gold_per_turn == 5.0
    assert snap.faith_per_turn == 1.0
    assert snap.current_research is None
    assert snap.current_civic is None


def test_parse_structured_civ6_mcp_overview_payload() -> None:
    payload = {
        "turn": 91,
        "era": "Industrial Era",
        "game_speed": "Online",
        "civilization": {"name": "Korea", "leader": "Seondeok"},
        "gold_balance": "245",
        "faith_balance": 18,
        "total_population": "27",
        "unit_count": 9,
        "military_strength": "312",
        "yields": {
            "science": 141.5,
            "culture": "88.25",
            "gold": -12,
            "faith": None,
        },
        "current_research": "STEAM_POWER",
        "current_civic": "NATIONALISM",
    }

    snap = parse_game_overview(payload)

    assert snap.current_turn == 91
    assert snap.game_era == "Industrial"
    assert snap.game_speed == "Online"
    assert snap.civilization_name == "Korea"
    assert snap.leader_name == "Seondeok"
    assert snap.gold == 245
    assert snap.science_per_turn == 141.5
    assert snap.culture_per_turn == 88.25
    assert snap.gold_per_turn == -12.0
    assert snap.faith == 18
    assert snap.faith_per_turn is None
    assert snap.total_population == 27
    assert snap.unit_count == 9
    assert snap.military_strength == 312
    assert snap.current_research == "STEAM_POWER"
    assert snap.current_civic == "NATIONALISM"
    assert "STEAM_POWER" in snap.raw_text


def test_parse_overview_preserves_fields_across_supported_payload_shapes() -> None:
    text_payload = """\
Game Overview
Turn 87
Era: Medieval Era
Game Speed: Online
Civilization: Korea (Seondeok)
Gold: 104 (+104 / turn)
Faith: 7 (+7.5 / turn)
Science: +93.25 / turn
Culture: +41.5 / turn
Research: EDUCATION (3 turns)
Civic: FEUDALISM (2 turns)
"""
    structured_payload = {
        "turn": 87,
        "era": "Medieval Era",
        "game_speed": "Online",
        "civilization": {"name": "Korea", "leader": "Seondeok"},
        "gold_balance": 104,
        "faith_balance": 7,
        "yields": {
            "science": 93.25,
            "culture": 41.5,
            "gold": 104,
            "faith": 7.5,
        },
        "current_research": "EDUCATION (3 turns)",
        "current_civic": "FEUDALISM (2 turns)",
    }
    json_payload = json.dumps(structured_payload)
    sdk_payload = FakeMcpCallToolResult(structured_content=structured_payload)
    expected_fields = {
        "current_turn": 87,
        "game_era": "Medieval",
        "game_speed": "Online",
        "civilization_name": "Korea",
        "leader_name": "Seondeok",
        "gold": 104,
        "science_per_turn": 93.25,
        "culture_per_turn": 41.5,
        "gold_per_turn": 104.0,
        "faith": 7,
        "faith_per_turn": 7.5,
        "current_research": "EDUCATION (3 turns)",
        "current_civic": "FEUDALISM (2 turns)",
        "is_game_over": False,
        "victory_text": None,
    }

    for payload in (text_payload, structured_payload, json_payload, sdk_payload):
        snapshot = parse_game_overview(payload)

        assert snapshot.raw_text
        for field_name, expected in expected_fields.items():
            assert getattr(snapshot, field_name) == expected, field_name


def test_parse_overview_accepts_sdk_style_structured_payload() -> None:
    payload = FakeMcpCallToolResult(
        structured_content={
            "turn_number": "92",
            "game_era": "Modern Era",
            "sciencePerTurn": "199.5",
            "current_research": "RADIO",
            "current_civic": "MASS_MEDIA",
        }
    )

    snap = parse_game_overview(payload)

    assert snap.current_turn == 92
    assert snap.game_era == "Modern"
    assert snap.science_per_turn == 199.5
    assert snap.current_research == "RADIO"
    assert snap.current_civic == "MASS_MEDIA"
    assert isinstance(snap.raw_text, str)
    assert "RADIO" in snap.raw_text


def test_state_bundle_from_raw_mcp_state_maps_tool_payloads() -> None:
    bundle = state_bundle_from_raw_mcp_state(
        {
            "get_game_overview": {
                "turn": "93",
                "era": "Modern Era",
                "yields": {"science": "211.5", "culture": 104, "gold": -3},
                "research": "FLIGHT",
                "civic": "IDEOLOGY",
            },
            "get_units": ["Builder at (3, 4)", "Scout at (1, 2)"],
            "get_cities": {"cities": [{"name": "Seoul", "population": 11}]},
            "get_trade_routes": "Trade Routes:\n- Seoul -> Busan",
            "missing_tools": ["get_notifications"],
            "failed_tools": {"get_diplomacy": "connection closed"},
            "malformed_tools": {"get_pending_trades": "empty response body"},
        }
    )

    assert bundle.overview.current_turn == 93
    assert bundle.overview.game_era == "Modern"
    assert bundle.overview.science_per_turn == 211.5
    assert bundle.overview.culture_per_turn == 104.0
    assert bundle.overview.gold_per_turn == -3.0
    assert bundle.overview.current_research == "FLIGHT"
    assert bundle.overview.current_civic == "IDEOLOGY"
    assert "Builder at (3, 4)" in bundle.units_text
    assert '"Seoul"' in bundle.cities_text
    assert bundle.extra["get_trade_routes"] == "Trade Routes:\n- Seoul -> Busan"
    assert bundle.missing_tools == ("get_notifications",)
    assert bundle.failed_tools == {"get_diplomacy": "connection closed"}
    assert bundle.malformed_tools == {"get_pending_trades": "empty response body"}


def test_state_bundle_from_raw_mcp_state_accepts_mcp_result_shape() -> None:
    bundle = state_bundle_from_raw_mcp_state(
        {
            "get_game_overview": {
                "content": [],
                "structuredContent": {
                    "turn_number": 101,
                    "game_era": "Atomic Era",
                    "sciencePerTurn": 302.25,
                },
            },
            "get_units": {
                "content": [
                    {"type": "text", "text": "Units:"},
                    {"type": "text", "text": "- Infantry at (5, 6)"},
                ]
            },
        }
    )

    assert bundle.overview.current_turn == 101
    assert bundle.overview.game_era == "Atomic"
    assert bundle.overview.science_per_turn == 302.25
    assert bundle.units_text == "Units:\n- Infantry at (5, 6)"


def test_state_bundle_from_raw_mcp_state_accepts_mcp_result_objects() -> None:
    bundle = state_bundle_from_raw_mcp_state(
        {
            "get_game_overview": FakeMcpCallToolResult(
                structured_content={
                    "turn_number": "102",
                    "game_era": "Atomic Era",
                    "sciencePerTurn": "305.5",
                    "current_research": "ROBOTICS",
                }
            ),
            "get_units": FakeMcpCallToolResult(
                content=[
                    FakeMcpTextBlock("Units:"),
                    {"type": "text", "text": "- Infantry at (5, 6)"},
                ]
            ),
            "get_notifications": FakeMcpCallToolResult(text="Notifications:\n- Choose civic"),
        }
    )

    assert bundle.overview.current_turn == 102
    assert bundle.overview.game_era == "Atomic"
    assert bundle.overview.science_per_turn == 305.5
    assert bundle.overview.current_research == "ROBOTICS"
    assert bundle.units_text == "Units:\n- Infantry at (5, 6)"
    assert bundle.notifications_text == "Notifications:\n- Choose civic"


def test_state_bundle_output_remains_stable_for_mixed_helper_payload_shapes() -> None:
    payload = {
        "game_overview": {
            "content": [],
            "structured_content": {
                "turn_number": "118",
                "era": "Future Era",
                "game_speed": "Online",
                "civilization": {"name": "Korea", "leader": "Seondeok"},
                "gold_balance": "345",
                "faith_balance": "27",
                "totalPopulation": "33",
                "militaryStrength": "512",
                "unitCount": "14",
                "yields": {
                    "science": "612.5",
                    "culture": 250,
                    "gold": "+101.5",
                    "faith": "18",
                },
                "researching": "OFFWORLD_MISSION",
                "civicResearching": "EXODUS_IMPERATIVE",
            },
        },
        "units": {
            "content": [
                {"type": "text", "text": "Units:"},
                {"type": "text", "text": "- Giant Death Robot at (8, 9)"},
            ]
        },
        "get_cities": {"content_blocks": ["Cities:", "- Seoul: pop 22"]},
        "get_diplomacy": {"text": "Diplomacy:\n- Gilgamesh: allied"},
        "get_trade_routes": {"content": [{"type": "text", "text": "Trade Routes:\n- Seoul -> Busan"}]},
        "missing_tools": ("get_victory_progress",),
        "failed_tools": {"get_pending_trades": "timeout"},
        "malformed_tools": {"get_notifications": "empty response body"},
    }

    assert asdict(state_bundle_from_raw_mcp_state(payload)) == {
        "overview": {
            "raw_text": (
                '{"civicResearching": "EXODUS_IMPERATIVE", '
                '"civilization": {"leader": "Seondeok", "name": "Korea"}, '
                '"era": "Future Era", "faith_balance": "27", "game_speed": "Online", '
                '"gold_balance": "345", "militaryStrength": "512", '
                '"researching": "OFFWORLD_MISSION", "totalPopulation": "33", '
                '"turn_number": "118", "unitCount": "14", '
                '"yields": {"culture": 250, "faith": "18", "gold": "+101.5", "science": "612.5"}}'
            ),
            "current_turn": 118,
            "game_era": "Future",
            "game_speed": "Online",
            "civilization_name": "Korea",
            "leader_name": "Seondeok",
            "gold": 345,
            "science_per_turn": 612.5,
            "culture_per_turn": 250.0,
            "gold_per_turn": 101.5,
            "faith": 27,
            "faith_per_turn": 18.0,
            "total_population": 33,
            "military_strength": 512,
            "unit_count": 14,
            "current_research": "OFFWORLD_MISSION",
            "current_civic": "EXODUS_IMPERATIVE",
            "is_game_over": False,
            "victory_text": None,
        },
        "units_text": "Units:\n- Giant Death Robot at (8, 9)",
        "cities_text": "Cities:\n- Seoul: pop 22",
        "diplomacy_text": "Diplomacy:\n- Gilgamesh: allied",
        "tech_civics_text": "",
        "notifications_text": "",
        "pending_diplomacy_text": "",
        "pending_trades_text": "",
        "victory_progress_text": "",
        "extra": {"get_trade_routes": "Trade Routes:\n- Seoul -> Busan"},
        "missing_tools": ("get_victory_progress",),
        "failed_tools": {"get_pending_trades": "timeout"},
        "malformed_tools": {"get_notifications": "empty response body"},
    }


def test_state_bundle_parsed_output_matches_direct_and_consolidated_payload_paths() -> None:
    direct_payload = {
        "get_game_overview": VALID_CIV6_MCP_OVERVIEW_PAYLOAD,
        "get_units": "Units:\n- Builder at (3, 4)\n- Scout at (1, 2)",
        "get_cities": "Cities:\n- Seoul: pop 7",
        "get_notifications": "Notifications:\n- Choose civic",
        "get_trade_routes": "Trade Routes:\n- Seoul -> Busan",
        "missing_tools": ("get_victory_progress",),
        "failed_tools": {"get_diplomacy": "timeout"},
        "malformed_tools": {"get_pending_trades": "empty response body"},
    }
    consolidated_payload = {
        "game_overview": {"content": [FakeMcpTextBlock(VALID_CIV6_MCP_OVERVIEW_PAYLOAD)]},
        "units": {
            "content": [
                {"type": "text", "text": "Units:"},
                {"type": "text", "text": "- Builder at (3, 4)"},
                {"type": "text", "text": "- Scout at (1, 2)"},
            ]
        },
        "cities": {"content_blocks": ["Cities:", "- Seoul: pop 7"]},
        "notifications": FakeMcpCallToolResult(text="Notifications:\n- Choose civic"),
        "get_trade_routes": {"text": "Trade Routes:\n- Seoul -> Busan"},
        "missing_tools": ["get_victory_progress"],
        "failed_tools": {"get_diplomacy": "timeout"},
        "malformed_tools": {"get_pending_trades": "empty response body"},
    }

    direct_bundle = state_bundle_from_raw_mcp_state(direct_payload)
    consolidated_bundle = state_bundle_from_raw_mcp_state(consolidated_payload)

    assert asdict(consolidated_bundle) == asdict(direct_bundle)


def test_state_bundle_matches_fixture_regression_baselines(
    civ6_mcp_parser_regression_cases: dict[str, dict[str, object]],
) -> None:
    assert set(civ6_mcp_parser_regression_cases) == {
        "direct_tool_mapping",
        "consolidated_alias_sdk_mapping",
        "overview_only_structured_mapping",
    }

    for case_name, case in civ6_mcp_parser_regression_cases.items():
        bundle = state_bundle_from_raw_mcp_state(case["raw_state"])

        assert _jsonable(asdict(bundle)) == case["expected_bundle"], case_name


def test_parse_game_overview_malformed_json_path_remains_lenient(caplog: pytest.LogCaptureFixture) -> None:
    malformed_json = '{"turn": 120, "era": "Future Era",'

    with caplog.at_level("DEBUG", logger="civStation.agent.modules.backend.civ6_mcp.state_parser"):
        snapshot = parse_game_overview(malformed_json)

    assert asdict(snapshot) == asdict(GameOverviewSnapshot(raw_text=malformed_json))
    assert "civ6-mcp overview looked like JSON but could not be decoded" in caplog.text


def test_state_bundle_diagnostic_error_shapes_remain_parser_safe() -> None:
    bundle = state_bundle_from_raw_mcp_state(
        {
            "get_units": None,
            "missing_tools": ["get_units", 404],
            "failed_tools": "timeout",
            "malformed_tools": {7: RuntimeError("boom")},
        }
    )

    assert asdict(bundle) == asdict(
        StateBundle(
            missing_tools=("get_units", "404"),
            malformed_tools={"7": "boom"},
        )
    )


def test_representative_parser_regression_inputs_match_baseline_sections(
    civ6_mcp_parser_regression_cases: dict[str, dict[str, object]],
) -> None:
    direct_case = civ6_mcp_parser_regression_cases["direct_tool_mapping"]
    consolidated_case = civ6_mcp_parser_regression_cases["consolidated_alias_sdk_mapping"]
    structured_case = civ6_mcp_parser_regression_cases["overview_only_structured_mapping"]

    direct_bundle = state_bundle_from_raw_mcp_state(direct_case["raw_state"])
    consolidated_bundle = state_bundle_from_raw_mcp_state(consolidated_case["raw_state"])
    structured_bundle = state_bundle_from_raw_mcp_state(structured_case["raw_state"])

    assert _jsonable(asdict(direct_bundle)) == direct_case["expected_bundle"]
    assert _jsonable(asdict(consolidated_bundle)) == direct_case["expected_bundle"]
    assert consolidated_case["expected_bundle"] == direct_case["expected_bundle"]

    assert _jsonable(asdict(structured_bundle)) == structured_case["expected_bundle"]
    assert structured_bundle.to_planner_context(max_section_chars=1200) == (
        f"## OVERVIEW\n{structured_case['expected_bundle']['overview']['raw_text']}"
    )
    assert direct_bundle.to_planner_context(max_section_chars=1200) == consolidated_bundle.to_planner_context(
        max_section_chars=1200
    )


def test_state_bundle_from_raw_mcp_state_renders_string_lists_as_text_sections() -> None:
    bundle = state_bundle_from_raw_mcp_state(
        {
            "get_units": ["Builder at (3, 4)", "Scout at (1, 2)"],
        }
    )

    assert bundle.units_text == "Builder at (3, 4)\nScout at (1, 2)"


def test_state_bundle_from_raw_mcp_state_normalizes_mixed_sdk_state_fixture(
    mixed_civ6_mcp_sdk_state_payload: dict[str, object],
) -> None:
    bundle = state_bundle_from_raw_mcp_state(mixed_civ6_mcp_sdk_state_payload)

    assert bundle.overview.current_turn == 112
    assert bundle.overview.game_era == "Information"
    assert bundle.overview.science_per_turn == 455.5
    assert bundle.overview.culture_per_turn == 231.25
    assert bundle.overview.gold_per_turn == 1000.0
    assert bundle.overview.faith_per_turn == 22.0
    assert bundle.overview.current_research == "SMART_MATERIALS"
    assert bundle.overview.current_civic == "GLOBALIZATION"
    assert bundle.units_text == "Units:\n- Mech Infantry at (10, 11)"
    assert bundle.cities_text == "Cities:\n- Seoul: pop 18"
    assert bundle.notifications_text == "Notifications:\n- Choose production"
    assert bundle.extra == {"get_trade_routes": "Trade Routes:\n- Seoul -> Busan"}
    assert bundle.missing_tools == ("get_victory_progress",)
    assert bundle.failed_tools == {"get_diplomacy": "timeout"}
    assert bundle.malformed_tools == {"get_pending_trades": "empty response body"}


def test_state_bundle_renders_required_overview_when_optional_sections_are_missing() -> None:
    bundle = StateBundle(overview=parse_game_overview(VALID_CIV6_MCP_OVERVIEW_PAYLOAD))

    rendered = bundle.to_planner_context()

    assert "## OVERVIEW" in rendered
    assert "Turn 87" in rendered
    assert "## UNITS" not in rendered
    assert "## CITIES" not in rendered
    assert "## DIPLOMACY" not in rendered
    assert "## TECH_CIVICS" not in rendered
    assert "## NOTIFICATIONS" not in rendered
    assert "## PENDING_DIPLOMACY" not in rendered
    assert "## PENDING_TRADES" not in rendered
    assert "## VICTORY_PROGRESS" not in rendered


def test_state_bundle_renders_available_optional_sections_when_overview_is_missing() -> None:
    bundle = StateBundle(
        units_text="Units:\n- Scout at (1, 2)",
        cities_text="Cities:\n- Seoul: pop 4",
        missing_tools=("get_game_overview",),
    )

    rendered = bundle.to_planner_context()

    assert "## OVERVIEW" not in rendered
    assert "## UNITS" in rendered
    assert "Scout at (1, 2)" in rendered
    assert "## CITIES" in rendered
    assert "Seoul: pop 4" in rendered
    assert "missing: get_game_overview" in rendered


def test_state_bundle_renders_valid_civ6_mcp_state_payload_sections() -> None:
    bundle = StateBundle(
        overview=parse_game_overview(VALID_CIV6_MCP_OVERVIEW_PAYLOAD),
        units_text="Units:\n- Builder at (3, 4), 1 charge\n- Archer at (5, 5), fortified",
        cities_text="Cities:\n- Seoul: pop 7, production Campus\n- Busan: pop 3, production Granary",
        diplomacy_text="Diplomacy:\n- Cleopatra: neutral\n- Hojo Tokimune: friendly",
        tech_civics_text="Tech/Civics:\n- Research EDUCATION\n- Civic FEUDALISM",
        notifications_text="Notifications:\n- City-state quest completed",
        pending_diplomacy_text="Pending diplomacy:\n- No incoming offers",
        pending_trades_text="Pending trades:\n- No active trade decisions",
        victory_progress_text="Victory Progress:\n- Science: 35%\n- Culture: 18%",
        extra={"get_governors": "Governors:\n- Pingala established in Seoul"},
    )

    rendered = bundle.to_planner_context(max_section_chars=1200)

    assert "## OVERVIEW" in rendered
    assert "Turn 87" in rendered
    assert "Research: EDUCATION (3 turns)" in rendered
    assert "## UNITS" in rendered
    assert "Builder at (3, 4)" in rendered
    assert "## CITIES" in rendered
    assert "Seoul: pop 7" in rendered
    assert "## DIPLOMACY" in rendered
    assert "Cleopatra: neutral" in rendered
    assert "## TECH_CIVICS" in rendered
    assert "## NOTIFICATIONS" in rendered
    assert "## PENDING_DIPLOMACY" in rendered
    assert "## PENDING_TRADES" in rendered
    assert "## VICTORY_PROGRESS" in rendered
    assert "## GET_GOVERNORS" in rendered
    assert "Pingala established in Seoul" in rendered


def test_state_bundle_renders_planner_context_with_truncation() -> None:
    bundle = StateBundle()
    bundle.units_text = "U" * 4000
    bundle.cities_text = "Compact city listing."
    rendered = bundle.to_planner_context(max_section_chars=200)
    assert "## CITIES" in rendered
    assert "## UNITS" in rendered
    assert "(truncated)" in rendered
    assert len(rendered) < 2500


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))
