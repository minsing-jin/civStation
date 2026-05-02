"""Fixture-driven planner coverage for representative civ6-mcp observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.observation_schema import normalize_raw_mcp_game_state
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner
from civStation.agent.modules.backend.civ6_mcp.planner_types import Civ6McpActionType, Civ6McpPlannerAction
from civStation.agent.modules.backend.civ6_mcp.turn_planning import build_prioritized_turn_plan


@dataclass
class FakeResponse:
    content: str


class FakeProvider:
    def __init__(self, content: str) -> None:
        self._content = content
        self.captured_prompts: list[str] = []

    def _build_text_content(self, text: str) -> dict[str, str]:
        return {"type": "text", "text": text}

    def _send_to_api(self, content_parts: list[object], **kwargs: object) -> FakeResponse:  # noqa: ARG002
        prompt = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
        self.captured_prompts.append(prompt)
        return FakeResponse(content=self._content)


_PLANNER_TOOL_CATALOG = {
    "set_city_production": {
        "description": "Assign city production",
        "input_schema": {
            "properties": {
                "city": {"type": "string"},
                "production": {"type": "string"},
            },
            "required": ["city", "production"],
        },
    },
    "set_research": {
        "description": "Queue research or civic",
        "input_schema": {
            "properties": {
                "tech_or_civic": {"type": "string"},
                "category": {"type": "string", "default": "tech"},
            },
            "required": ["tech_or_civic"],
        },
    },
    "end_turn": {
        "description": "Advance to the next turn",
        "input_schema": {
            "properties": {
                "tactical": {"type": "string"},
                "strategic": {"type": "string"},
                "tooling": {"type": "string"},
                "planning": {"type": "string"},
                "hypothesis": {"type": "string"},
            },
            "required": ["tactical", "strategic", "tooling", "planning", "hypothesis"],
        },
    },
}


def test_planner_observation_fixtures_cover_required_scenarios(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    assert set(civ6_mcp_planner_observation_cases) == {
        "representative_science_blockers",
        "edge_blank_research_policy_promotion",
        "edge_diagnostics_retry_missing_overview",
        "edge_game_over",
    }
    for case in civ6_mcp_planner_observation_cases.values():
        assert isinstance(case["raw_state"], dict)
        assert isinstance(case["expected_action_tools"], list)


def test_prioritized_turn_plan_matches_fixture_expected_backend_actions(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    for case_name, case in civ6_mcp_planner_observation_cases.items():
        observation = normalize_raw_mcp_game_state(case["raw_state"])
        plan = build_prioritized_turn_plan(observation, strategy=str(case["strategy"]))

        actions = plan.to_actions()

        assert plan.backend == "civ6-mcp", case_name
        assert [action.tool for action in actions] == case["expected_action_tools"], case_name
        assert all(isinstance(action, Civ6McpPlannerAction) for action in actions), case_name
        assert all(action.type is Civ6McpActionType.TOOL_CALL for action in actions), case_name
        assert all(not hasattr(action, "x") and not hasattr(action, "y") for action in actions), case_name


def test_prioritized_turn_plan_fixture_end_turn_actions_have_required_reflections(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    required_reflections = {"tactical", "strategic", "tooling", "planning", "hypothesis"}

    for case_name, case in civ6_mcp_planner_observation_cases.items():
        observation = normalize_raw_mcp_game_state(case["raw_state"])
        plan = build_prioritized_turn_plan(observation, strategy=str(case["strategy"]))
        end_turn_actions = [action for action in plan.to_actions() if action.tool == "end_turn"]

        if not case["expected_action_tools"]:
            assert end_turn_actions == [], case_name
            assert "game over" in " ".join(plan.notes).lower()
            continue

        assert len(end_turn_actions) == 1, case_name
        assert set(end_turn_actions[0].arguments) == required_reflections, case_name
        assert all(str(value).strip() for value in end_turn_actions[0].arguments.values()), case_name


def test_llm_tool_planner_uses_fixture_context_and_returns_civ6_mcp_tool_calls(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    case = civ6_mcp_planner_observation_cases["representative_science_blockers"]
    observation = normalize_raw_mcp_game_state(case["raw_state"])
    planner_payload: dict[str, Any] = {
        "tool_calls": [
            {
                "tool": "set_city_production",
                "arguments": {"city": "Seoul", "production": "Campus"},
                "reasoning": "Use production on a science-specific district.",
            },
            {
                "tool": "set_research",
                "arguments": {"tech_or_civic": "EDUCATION", "category": "tech"},
                "reasoning": "Continue the science path surfaced by the observation.",
            },
            {
                "tool": "end_turn",
                "arguments": {
                    "tactical": "Resolved production and research choices.",
                    "strategic": "Campus and Education support science victory.",
                    "tooling": "Planner emitted civ6-mcp tool calls only.",
                    "planning": "Re-observe blockers next turn.",
                    "hypothesis": "Science output improves after campus investment.",
                },
            },
        ]
    }
    provider = FakeProvider(json.dumps(planner_payload))
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_PLANNER_TOOL_CATALOG,
        allowed_tools=("set_city_production", "set_research", "end_turn"),
    )

    result = planner.plan(
        strategy=str(case["strategy"]),
        state_context=observation.planner_context,
        recent_calls="get_game_overview -> get_notifications",
    )

    assert [call.tool for call in result.tool_calls] == ["set_city_production", "set_research", "end_turn"]
    assert result.tool_calls[0].arguments == {"city": "Seoul", "production": "Campus"}
    assert "## OVERVIEW" in provider.captured_prompts[0]
    assert "Pending diplomacy" in provider.captured_prompts[0]
    assert "set_city_production(city: string, production: string)" in provider.captured_prompts[0]


def test_planner_invocation_consumes_normalized_observation_payload_and_returns_ordered_tool_plan(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    case = civ6_mcp_planner_observation_cases["representative_science_blockers"]
    observation = normalize_raw_mcp_game_state(case["raw_state"])
    planner_payload: dict[str, Any] = {
        "tool_calls": [
            {
                "tool": "get_pending_diplomacy",
                "arguments": {},
                "reasoning": "Incoming diplomacy must be inspected first.",
            },
            {
                "tool": "set_city_production",
                "arguments": {"city": "Seoul", "production": "Campus"},
                "reasoning": "Resolve production blocker.",
            },
            {
                "tool": "end_turn",
                "arguments": {
                    "tactical": "Handled diplomacy and city production.",
                    "strategic": "Campus supports the science plan.",
                    "tooling": "Observation payload became an MCP tool plan.",
                    "planning": "Re-check units next turn.",
                    "hypothesis": "Science scaling improves after Campus investment.",
                },
            },
        ]
    }
    provider = FakeProvider(json.dumps(planner_payload))
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog={
            **_PLANNER_TOOL_CATALOG,
            "get_pending_diplomacy": {"description": "Inspect diplomacy", "input_schema": {"properties": {}}},
        },
        allowed_tools=("get_pending_diplomacy", "set_city_production", "end_turn"),
    )

    result = planner.plan_from_observation(
        observation=observation,
        strategy=str(case["strategy"]),
        recent_calls="(none)",
    )

    assert [call.tool for call in result.tool_calls] == [
        "get_pending_diplomacy",
        "set_city_production",
        "end_turn",
    ]
    assert result.tool_calls[1].arguments == {"city": "Seoul", "production": "Campus"}
    assert "## OVERVIEW" in provider.captured_prompts[0]
    assert "## PRIORITIZED_MCP_INTENTS" in provider.captured_prompts[0]
    assert "get_pending_diplomacy" in provider.captured_prompts[0]


def test_planner_invocation_accepts_raw_observation_mapping(
    civ6_mcp_planner_observation_cases: dict[str, dict[str, object]],
) -> None:
    case = civ6_mcp_planner_observation_cases["edge_blank_research_policy_promotion"]
    planner_payload: dict[str, Any] = {
        "tool_calls": [
            {
                "tool": "set_research",
                "arguments": {"tech_or_civic": "WRITING", "category": "tech"},
                "reasoning": "Research is blank in the observation payload.",
            },
            {
                "tool": "end_turn",
                "arguments": {
                    "tactical": "Set missing research.",
                    "strategic": "Writing unlocks science infrastructure.",
                    "tooling": "Raw observation mapping was normalized for planning.",
                    "planning": "Inspect policies next turn.",
                    "hypothesis": "Research progress resumes after selecting Writing.",
                },
            },
        ]
    }
    provider = FakeProvider(json.dumps(planner_payload))
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_PLANNER_TOOL_CATALOG,
        allowed_tools=("set_research", "end_turn"),
    )

    result = planner.plan_from_observation(
        observation=case["raw_state"],
        strategy=str(case["strategy"]),
        recent_calls="get_game_overview",
    )

    assert [call.tool for call in result.tool_calls] == ["set_research", "end_turn"]
    assert "Turn 16" in provider.captured_prompts[0]
    assert "Research:" in provider.captured_prompts[0]
    assert "P070 get_tech_civics" in provider.captured_prompts[0]
