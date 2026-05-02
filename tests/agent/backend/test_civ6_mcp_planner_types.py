"""Tests for civ6-mcp planner-facing interfaces and backend action types."""

from __future__ import annotations

import pytest

from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner
from civStation.agent.modules.backend.civ6_mcp.planner_types import (
    Civ6McpActionType,
    Civ6McpIntentType,
    Civ6McpPlanner,
    Civ6McpPlannerAction,
    Civ6McpPlannerIntent,
    Civ6McpPlannerProvider,
    PlannerResult,
    infer_civ6_mcp_intent_type,
)


class FakeProvider:
    def _build_text_content(self, text: str) -> object:
        return {"type": "text", "text": text}

    def _send_to_api(self, content_parts, **kwargs):  # noqa: ANN001, ARG002
        raise AssertionError("not used")


def test_planner_protocol_matches_tool_planner_surface() -> None:
    planner = Civ6McpToolPlanner(
        provider=FakeProvider(),
        tool_catalog={"get_game_overview": {"description": "overview", "input_schema": {}}},
        allowed_tools=("get_game_overview",),
    )

    assert isinstance(FakeProvider(), Civ6McpPlannerProvider)
    assert isinstance(planner, Civ6McpPlanner)


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("get_game_overview", Civ6McpIntentType.OBSERVE),
        ("set_research", Civ6McpIntentType.ACT),
        ("end_turn", Civ6McpIntentType.END_TURN),
    ],
)
def test_infer_civ6_mcp_intent_type(tool: str, expected: Civ6McpIntentType) -> None:
    assert infer_civ6_mcp_intent_type(tool) is expected


def test_civ6_mcp_planner_intent_and_action_convert_to_tool_call() -> None:
    intent = Civ6McpPlannerIntent.from_tool(
        "set_research",
        {"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )

    action = intent.to_action()
    call = action.to_tool_call()

    assert intent.intent_type is Civ6McpIntentType.ACT
    assert intent.type is Civ6McpIntentType.ACT
    assert action.action_type is Civ6McpActionType.TOOL_CALL
    assert action.type is Civ6McpActionType.TOOL_CALL
    assert call == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )


def test_planner_action_from_tool_call_copies_arguments() -> None:
    call = ToolCall(tool="get_units", arguments={"owner": "PLAYER"}, reasoning="inspect units")
    action = Civ6McpPlannerAction.from_tool_call(call)
    action.arguments["owner"] = "AI"

    assert call.arguments == {"owner": "PLAYER"}
    assert action.to_tool_call().arguments == {"owner": "AI"}


def test_planner_result_actions_aliases_tool_calls() -> None:
    calls = [ToolCall(tool="get_game_overview")]
    result = PlannerResult(tool_calls=calls, raw_response="{}")

    assert result.actions is calls
