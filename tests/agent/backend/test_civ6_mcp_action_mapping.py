"""Tests for civ6-mcp planned-action to MCP tool-call mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from civStation.agent.models.schema import ClickAction, DragAction, KeyPressAction
from civStation.agent.modules.backend.civ6_mcp.action_mapping import (
    CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES,
    CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY,
    PLANNED_ACTION_TYPE_TO_MCP_TOOL,
    Civ6McpActionMappingError,
    Civ6McpFreeFormActionType,
    map_civ6_mcp_action,
    map_civ6_mcp_actions,
)
from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.operations import END_TURN_TOOL, SUPPORTED_CIV6_MCP_TOOLS
from civStation.agent.modules.backend.civ6_mcp.planner_types import Civ6McpPlannerAction


@dataclass
class ToolNamedAction:
    tool: str
    arguments: dict[str, Any]
    reasoning: str = ""


@dataclass
class NameNamedAction:
    name: str
    arguments: dict[str, Any]
    reasoning: str = ""


def _minimal_arguments_for_tool(tool: str) -> dict[str, Any]:
    if tool == END_TURN_TOOL:
        return {
            "tactical": "No tactical blockers.",
            "strategic": "Continue current plan.",
            "tooling": "Tool calls are available.",
            "planning": "Review state next turn.",
            "hypothesis": "Current plan remains viable.",
        }
    return {}


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


@pytest.mark.parametrize("tool_field", ["tool", "tool_name", "name"])
def test_maps_direct_tool_mapping_from_canonical_tool_fields(tool_field: str) -> None:
    mapped = map_civ6_mcp_action(
        {
            tool_field: "set_research",
            "arguments": {"tech_or_civic": "WRITING"},
            "reasoning": "canonical tool-call shape",
        }
    )

    assert mapped == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="canonical tool-call shape",
    )


def test_maps_tool_named_object_without_action_type() -> None:
    mapped = map_civ6_mcp_action(
        ToolNamedAction(
            tool="set_research",
            arguments={"tech_or_civic": "WRITING"},
            reasoning="canonical tool attr",
        )
    )

    assert mapped == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="canonical tool attr",
    )


def test_maps_name_named_object_without_action_type() -> None:
    mapped = map_civ6_mcp_action(
        NameNamedAction(
            name="set_research",
            arguments={"tech_or_civic": "WRITING"},
            reasoning="canonical name attr",
        )
    )

    assert mapped == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="canonical name attr",
    )


def test_defines_free_form_action_type_registry_for_core_tools() -> None:
    assert CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY["research"] == Civ6McpFreeFormActionType(
        action_type="research",
        tool="set_research",
        aliases=("choose_research", "set_research"),
    )
    assert CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY["city_production"] == Civ6McpFreeFormActionType(
        action_type="city_production",
        tool="set_city_production",
        aliases=("choose_production", "set_city_production"),
    )
    assert CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY["set_policies"] == Civ6McpFreeFormActionType(
        action_type="set_policies",
        tool="set_policies",
        aliases=("policies", "policy_cards"),
    )
    assert CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY["end_turn"] == Civ6McpFreeFormActionType(
        action_type="end_turn",
        tool="end_turn",
        aliases=("finish_turn",),
    )


def test_free_form_action_aliases_resolve_to_canonical_types_and_tools() -> None:
    expected_aliases = {
        "research": "research",
        "choose_research": "research",
        "set_research": "research",
        "city_production": "city_production",
        "choose_production": "city_production",
        "set_city_production": "city_production",
        "set_policies": "set_policies",
        "policies": "set_policies",
        "policy_cards": "set_policies",
        "end_turn": "end_turn",
        "finish_turn": "end_turn",
    }

    for alias, canonical in expected_aliases.items():
        assert CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES[alias] == canonical
        assert PLANNED_ACTION_TYPE_TO_MCP_TOOL[alias] == CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY[canonical].tool


@pytest.mark.parametrize("action_field", ["type", "action_type", "action"])
@pytest.mark.parametrize("alias", ["research", "choose_research", "set_research"])
def test_free_form_research_aliases_resolve_from_all_action_type_fields(action_field: str, alias: str) -> None:
    mapped = map_civ6_mcp_action(
        {
            action_field: alias,
            "target": "WRITING",
            "reason": "resolve semantic research alias",
        }
    )

    assert mapped == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="resolve semantic research alias",
    )


@pytest.mark.parametrize("tool_field", ["tool", "tool_name", "name"])
def test_free_form_alias_resolution_allows_matching_explicit_tool_fields(tool_field: str) -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "choose_production",
            tool_field: "set_city_production",
            "item": "Granary",
        }
    )

    assert mapped == ToolCall(
        tool="set_city_production",
        arguments={"production": "Granary"},
    )


@pytest.mark.parametrize("tool_field", ["tool", "tool_name", "name"])
def test_free_form_alias_resolution_rejects_conflicting_explicit_tool_fields(tool_field: str) -> None:
    with pytest.raises(Civ6McpActionMappingError, match="maps to 'set_research', not 'set_city_production'"):
        map_civ6_mcp_action(
            {
                "type": "choose_research",
                tool_field: "set_city_production",
                "target": "WRITING",
            }
        )


def test_all_supported_canonical_tool_names_are_accepted_as_free_form_types() -> None:
    assert SUPPORTED_CIV6_MCP_TOOLS <= set(PLANNED_ACTION_TYPE_TO_MCP_TOOL)

    for tool in sorted(SUPPORTED_CIV6_MCP_TOOLS):
        mapped = map_civ6_mcp_action({"type": tool, "arguments": _minimal_arguments_for_tool(tool)})

        assert mapped.tool == tool


def test_all_declared_free_form_aliases_map_to_their_canonical_tools() -> None:
    for canonical_type, entry in CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY.items():
        for alias in entry.accepted_types:
            mapped = map_civ6_mcp_action(
                {
                    "type": alias,
                    "arguments": _minimal_arguments_for_tool(entry.tool),
                }
            )

            assert CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES[alias] == canonical_type
            assert mapped.tool == entry.tool


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


def test_maps_free_form_policy_alias() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "policy_cards",
            "economic": ["Urban Planning", "God King"],
            "military": ["Discipline"],
        }
    )

    assert mapped.tool == "set_policies"
    assert mapped.arguments == {
        "economic": ["Urban Planning", "God King"],
        "military": ["Discipline"],
    }


def test_maps_free_form_finish_turn_alias() -> None:
    mapped = map_civ6_mcp_action(
        {
            "type": "finish_turn",
            "reflections": {
                "tactical": "Queued city work.",
                "strategic": "Build toward science.",
                "tooling": "Policy call succeeded.",
                "planning": "Inspect notifications next.",
                "hypothesis": "Production stabilizes tempo.",
            },
        }
    )

    assert mapped.tool == "end_turn"
    assert mapped.arguments["strategic"] == "Build toward science."


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
        {"type": "drag", "start_x": 10, "start_y": 20, "end_x": 30, "end_y": 40},
        {"type": "press", "keys": ["enter"]},
        {"type": "wait", "duration": 1},
    ],
)
def test_rejects_vlm_pixel_actions(raw: dict[str, object]) -> None:
    with pytest.raises(Civ6McpActionMappingError, match="VLM/computer-use action"):
        map_civ6_mcp_action(raw)


@pytest.mark.parametrize(
    ("action", "action_type"),
    [
        pytest.param(ClickAction(x=10, y=20), "click", id="click"),
        pytest.param(DragAction(start_x=10, start_y=20, end_x=30, end_y=40), "drag", id="drag"),
        pytest.param(KeyPressAction(keys=["enter"]), "press", id="press"),
    ],
)
def test_rejects_click_drag_press_vlm_action_model_inputs(action: object, action_type: str) -> None:
    with pytest.raises(Civ6McpActionMappingError) as exc_info:
        map_civ6_mcp_action(action)
    assert f"VLM/computer-use action {action_type!r}" in str(exc_info.value)


@pytest.mark.parametrize("tool", ["click", "drag", "press"])
def test_rejects_vlm_tool_call_names(tool: str) -> None:
    with pytest.raises(Civ6McpActionMappingError, match="VLM/computer-use action"):
        map_civ6_mcp_action(ToolCall(tool=tool, arguments={}))


@pytest.mark.parametrize("tool", ["click", "drag", "press"])
def test_rejects_vlm_direct_tool_call_mapping_names(tool: str) -> None:
    with pytest.raises(Civ6McpActionMappingError, match="VLM/computer-use action"):
        map_civ6_mcp_action({"type": "tool_call", "tool": tool, "arguments": {}})


def test_rejects_unsupported_action_type() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="Unsupported civ6-mcp planned action type"):
        map_civ6_mcp_action({"type": "launch_orbital_laser", "target": "barbarian camp"})


def test_rejects_unsupported_free_form_type_even_with_supported_explicit_tool() -> None:
    with pytest.raises(Civ6McpActionMappingError, match="Unsupported civ6-mcp planned action type"):
        map_civ6_mcp_action(
            {
                "type": "launch_orbital_laser",
                "tool": "set_research",
                "arguments": {"tech_or_civic": "WRITING"},
            }
        )


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
