"""Tests for civ6-mcp planned-action to MCP tool-call mapping."""

from __future__ import annotations

import pytest

from civStation.agent.models.schema import ClickAction
from civStation.agent.modules.backend.civ6_mcp.action_mapping import (
    Civ6McpActionMappingError,
    map_civ6_mcp_action,
    map_civ6_mcp_actions,
)
from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.planner_types import Civ6McpPlannerAction


def test_maps_direct_tool_call_object() -> None:
    call = ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}, reasoning="unlock campuses")

    mapped = map_civ6_mcp_action(call)

    assert mapped == call
    assert mapped.arguments is not call.arguments


def test_maps_planner_action_object() -> None:
    action = Civ6McpPlannerAction(
        tool="set_city_production",
        arguments={"city": "Seoul", "production": "Campus"},
        reasoning="science plan",
    )

    mapped = map_civ6_mcp_action(action)

    assert mapped == ToolCall(
        tool="set_city_production",
        arguments={"city": "Seoul", "production": "Campus"},
        reasoning="science plan",
    )


def test_maps_tool_call_mapping_with_name_alias() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "tool_call",
            "name": "set_research",
            "arguments": {"tech_or_civic": "WRITING"},
            "reasoning": "required tech",
        }
    )

    assert mapped == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="required tech",
    )


def test_maps_semantic_research_action_alias() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "choose_research",
            "tech": "WRITING",
            "category": "tech",
            "reasoning": "libraries",
        }
    )

    assert mapped.tool == "set_research"
    assert mapped.arguments == {"category": "tech", "tech_or_civic": "WRITING"}
    assert mapped.reasoning == "libraries"


def test_maps_semantic_city_production_action_alias() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "city_production",
            "city": "Seoul",
            "item": "Campus",
        }
    )

    assert mapped.tool == "set_city_production"
    assert mapped.arguments == {"city": "Seoul", "production": "Campus"}


def test_maps_end_turn_reflections_from_nested_payload() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "end_turn",
            "reflections": {
                "tactical": "Queued research.",
                "strategic": "Science victory.",
                "tooling": "Tool calls succeeded.",
                "planning": "Check cities next.",
                "hypothesis": "Campus improves output.",
            },
        }
    )

    assert mapped.tool == "end_turn"
    assert mapped.arguments == {
        "tactical": "Queued research.",
        "strategic": "Science victory.",
        "tooling": "Tool calls succeeded.",
        "planning": "Check cities next.",
        "hypothesis": "Campus improves output.",
    }


def test_maps_sequence_of_actions() -> None:
    mapped = map_civ6_mcp_actions(
        [
            {"type": "get_units"},
            {"type": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    )

    assert [call.tool for call in mapped] == ["get_units", "set_research"]


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "click", "x": 10, "y": 20},
        {"type": "press", "keys": ["enter"]},
        {"type": "wait", "duration": 1},
    ],
)
def test_rejects_vlm_pixel_actions(raw: dict[str, object]) -> None:
    with pytest.raises(Civ6McpActionMappingError, match="VLM/computer-use action"):
        map_civ6_mcp_action(raw)


def test_rejects_vlm_pixel_action_objects() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="VLM/computer-use action"):
        map_civ6_mcp_action(ClickAction(x=10, y=20))


def test_rejects_unsupported_action_type() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="Unsupported civ6-mcp planned action type"):
        map_civ6_mcp_action({"type": "launch_orbital_laser", "target": "barbarian camp"})


def test_rejects_non_object_arguments() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="arguments must be an object"):
        map_civ6_mcp_action({"type": "set_research", "arguments": ["WRITING"]})


def test_rejects_blank_end_turn_reflection() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="end_turn requires non-empty reflection fields"):
        map_civ6_mcp_action(
            {
                "type": "end_turn",
                "arguments": {
                    "tactical": "Queued research.",
                    "strategic": "",
                    "tooling": "Tool calls succeeded.",
                    "planning": "Check cities next.",
                    "hypothesis": "Campus improves output.",
                },
            }
        )
