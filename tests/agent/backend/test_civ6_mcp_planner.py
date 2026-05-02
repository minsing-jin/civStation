"""Tests for the civ6-mcp tool-call planner.

We stub the LLM provider rather than calling a real model. The planner's
job is purely to format a prompt, parse the JSON, and validate the schema —
all testable without network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from civStation.agent.modules.backend.civ6_mcp.planner import (
    Civ6McpToolPlanner,
    PlannerResult,
)


@dataclass
class FakeResponse:
    content: str


class FakeProvider:
    """Implements the slice of BaseVLMProvider that planner.plan() actually uses."""

    def __init__(self, content_sequence: list[str]) -> None:
        self._content_sequence = list(content_sequence)
        self.captured_prompts: list[str] = []

    def _build_text_content(self, text: str) -> dict:
        return {"type": "text", "text": text}

    def _send_to_api(self, content_parts, **kwargs):  # noqa: ARG002
        prompt = "".join(part.get("text", "") for part in content_parts if isinstance(part, dict))
        self.captured_prompts.append(prompt)
        if not self._content_sequence:
            raise AssertionError("FakeProvider exhausted content sequence")
        next_content = self._content_sequence.pop(0)
        return FakeResponse(content=next_content)


_TOOL_CATALOG = {
    "get_game_overview": {"description": "current overview", "input_schema": {"properties": {}}},
    "set_research": {
        "description": "queue research",
        "input_schema": {
            "properties": {
                "tech_or_civic": {"type": "string"},
                "category": {"type": "string", "default": "tech"},
            },
            "required": ["tech_or_civic"],
        },
    },
    "end_turn": {
        "description": "advance to next turn",
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


def _valid_plan_payload() -> dict:
    return {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
            {
                "tool": "end_turn",
                "arguments": {
                    "tactical": "Queued WRITING.",
                    "strategic": "Pursuing science victory.",
                    "tooling": "Plan parsed cleanly.",
                    "planning": "Next turn revisit cities.",
                    "hypothesis": "Library next.",
                },
            },
        ]
    }


def test_planner_happy_path() -> None:
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_TOOL_CATALOG,
        allowed_tools=("get_game_overview", "set_research", "end_turn"),
    )
    result = planner.plan(
        strategy="Pursue science victory.",
        state_context="Turn 1.",
        recent_calls="(none)",
    )
    assert isinstance(result, PlannerResult)
    assert [c.tool for c in result.tool_calls] == ["get_game_overview", "set_research", "end_turn"]
    assert "Pursue science victory" in provider.captured_prompts[0]


@pytest.mark.parametrize(
    ("payload", "expected_tools"),
    [
        (_valid_plan_payload(), ["get_game_overview", "set_research", "end_turn"]),
        (_valid_plan_payload()["tool_calls"], ["get_game_overview", "set_research", "end_turn"]),
        (
            {
                "tool_calls": [
                    {"name": "get_game_overview", "arguments": None, "reasoning": "scan"},
                    {
                        "tool": "end_turn",
                        "arguments": {
                            "tactical": "Observed only.",
                            "strategic": "Hold position.",
                            "tooling": "Planner used name alias.",
                            "planning": "Reassess next turn.",
                            "hypothesis": "No action needed.",
                        },
                    },
                ]
            },
            ["get_game_overview", "end_turn"],
        ),
    ],
)
def test_planner_accepts_supported_json_output_shapes(payload: object, expected_tools: list[str]) -> None:
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_TOOL_CATALOG,
        allowed_tools=("get_game_overview", "set_research", "end_turn"),
    )

    result = planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert [call.tool for call in result.tool_calls] == expected_tools
    assert result.raw_response == json.dumps(payload)


def test_planner_renders_tool_catalog_with_required_marker() -> None:
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_TOOL_CATALOG,
        allowed_tools=("set_research",),
    )
    catalog = planner.render_tool_catalog()
    assert "set_research(" in catalog
    # required field has no '?'
    assert "tech_or_civic: string" in catalog
    # optional field with default has '?'
    assert "category?: string" in catalog


def test_planner_retries_on_invalid_json_then_succeeds() -> None:
    bad = "not even json"
    good = json.dumps(_valid_plan_payload())
    provider = FakeProvider([bad, good])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=2)
    result = planner.plan(strategy="x", state_context="y", recent_calls="z")
    assert len(result.tool_calls) == 3


@pytest.mark.parametrize(
    ("payload", "error_match"),
    [
        ({"tool_calls": {"tool": "get_game_overview"}}, "tool_calls.*list"),
        ({"tool_calls": ["get_game_overview"]}, "Tool call entry must be an object"),
        ({"tool_calls": [{"arguments": {}}]}, "missing 'tool' name"),
        ({"tool_calls": [{"tool": "not_real", "arguments": {}}]}, "Unsupported civ6-mcp tool"),
        ({"tool_calls": [{"tool": "get_game_overview", "arguments": ["bad"]}]}, "arguments must be an object"),
        (
            {
                "tool_calls": [
                    {
                        "tool": "end_turn",
                        "arguments": {
                            "tactical": "Queued research.",
                            "strategic": "",
                            "tooling": "Tools succeeded.",
                            "planning": "Check cities next.",
                            "hypothesis": "Writing unlocks libraries.",
                        },
                    }
                ]
            },
            "end_turn requires non-empty reflection fields",
        ),
    ],
)
def test_planner_retries_malformed_json_payloads_then_reports_schema_error(
    payload: object,
    error_match: str,
) -> None:
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match=error_match):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


def test_planner_raises_after_exhausting_retries() -> None:
    provider = FakeProvider(["bad1", "bad2", "bad3"])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=2)
    with pytest.raises(RuntimeError):
        planner.plan(strategy="x", state_context="y", recent_calls="z")


def test_planner_strips_markdown_fences() -> None:
    fenced = "```json\n" + json.dumps(_valid_plan_payload()) + "\n```"
    provider = FakeProvider([fenced])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG)
    result = planner.plan(strategy="x", state_context="y", recent_calls="z")
    assert len(result.tool_calls) == 3
