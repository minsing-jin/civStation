"""Tests for the civ6-mcp tool-call planner.

We stub the LLM provider rather than calling a real model. The planner's
job is purely to format a prompt, parse the JSON, and validate the schema —
all testable without network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pytest

from civStation.agent.modules.backend.civ6_mcp import planner as planner_module
from civStation.agent.modules.backend.civ6_mcp import prompts as planner_prompts
from civStation.agent.modules.backend.civ6_mcp.action_mapping import PLANNED_ACTION_TYPE_TO_MCP_TOOL
from civStation.agent.modules.backend.civ6_mcp.observation_schema import Civ6McpNormalizedObservation
from civStation.agent.modules.backend.civ6_mcp.operations import (
    ACTION_TOOLS,
    END_TURN_TOOL,
    OBSERVATION_TOOLS,
    SUPPORTED_CIV6_MCP_TOOLS,
)
from civStation.agent.modules.backend.civ6_mcp.planner import (
    DEFAULT_PLANNER_TOOL_ALLOWLIST,
    Civ6McpToolPlanner,
    MissingEndTurnPlannerOutputError,
    PlannerResult,
)
from civStation.agent.modules.backend.civ6_mcp.planner_types import (
    DEFAULT_PLANNER_TOOL_ALLOWLIST as CANONICAL_PLANNER_TOOL_ALLOWLIST,
)
from civStation.agent.modules.backend.civ6_mcp.prompts import build_planner_system_prompt


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

_REQUIRED_REFLECTION_FIELDS = ("tactical", "strategic", "tooling", "planning", "hypothesis")


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


def test_default_planner_tool_allowlist_is_reusable_export() -> None:
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG)

    assert DEFAULT_PLANNER_TOOL_ALLOWLIST is planner_module._DEFAULT_PLANNER_TOOL_ALLOWLIST
    assert DEFAULT_PLANNER_TOOL_ALLOWLIST is CANONICAL_PLANNER_TOOL_ALLOWLIST
    assert DEFAULT_PLANNER_TOOL_ALLOWLIST
    assert DEFAULT_PLANNER_TOOL_ALLOWLIST[-1] == END_TURN_TOOL
    assert planner.allowed_tools is DEFAULT_PLANNER_TOOL_ALLOWLIST


def test_default_planner_tool_allowlist_preserves_operation_groups_and_end_turn_order() -> None:
    allowlist = DEFAULT_PLANNER_TOOL_ALLOWLIST
    action_start = allowlist.index("unit_action")

    assert set(allowlist) == SUPPORTED_CIV6_MCP_TOOLS
    assert len(allowlist) == len(set(allowlist))
    assert set(allowlist[:action_start]) == OBSERVATION_TOOLS
    assert set(allowlist[action_start:-1]) == ACTION_TOOLS
    assert allowlist[-1] == END_TURN_TOOL


def test_rendered_planner_prompt_lists_exact_canonical_allowlist_entries() -> None:
    tool_catalog = {
        tool: {"description": f"{tool} description", "input_schema": {"properties": {}}}
        for tool in CANONICAL_PLANNER_TOOL_ALLOWLIST
    }
    tool_catalog["run_lua"] = {"description": "raw gameplay scripts", "input_schema": {"properties": {}}}
    tool_catalog["click"] = {"description": "VLM pixel action", "input_schema": {"properties": {}}}
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=tool_catalog)

    planner.plan(strategy="Pursue science victory.", state_context="Turn 1.", recent_calls="(none)")

    rendered_catalog = (
        provider.captured_prompts[0]
        .split(
            "Available tools (subset; full schema is provided separately):\n",
            1,
        )[1]
        .split("\n\nOutput schema:", 1)[0]
    )
    rendered_tool_names = [
        line.removeprefix("- ").split("(", 1)[0] for line in rendered_catalog.splitlines() if line.startswith("- ")
    ]

    assert rendered_tool_names == list(CANONICAL_PLANNER_TOOL_ALLOWLIST)
    assert "run_lua" not in rendered_tool_names
    assert "click" not in rendered_tool_names


def test_planner_catalog_and_action_metadata_share_canonical_allowlist() -> None:
    tool_catalog = {
        tool: {"description": f"{tool} description", "input_schema": {"properties": {}}}
        for tool in CANONICAL_PLANNER_TOOL_ALLOWLIST
    }
    tool_catalog["click"] = {"description": "VLM pixel action", "input_schema": {"properties": {}}}
    planner = Civ6McpToolPlanner(provider=FakeProvider([]), tool_catalog=tool_catalog)

    catalog_tool_names = [
        line.removeprefix("- ").split("(", 1)[0] for line in planner.render_tool_catalog().splitlines()
    ]

    assert catalog_tool_names == list(CANONICAL_PLANNER_TOOL_ALLOWLIST)
    assert {tool: PLANNED_ACTION_TYPE_TO_MCP_TOOL[tool] for tool in CANONICAL_PLANNER_TOOL_ALLOWLIST} == {
        tool: tool for tool in CANONICAL_PLANNER_TOOL_ALLOWLIST
    }
    assert "click" not in catalog_tool_names
    assert "click" not in PLANNED_ACTION_TYPE_TO_MCP_TOOL


def test_default_planner_prompt_content_stays_in_sync_with_allowlist() -> None:
    tool_catalog = {
        tool: {"description": f"{tool} description", "input_schema": {"properties": {}}}
        for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST
    }
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=tool_catalog)

    planner.plan(strategy="Pursue science victory.", state_context="Turn 1.", recent_calls="(none)")

    prompt = provider.captured_prompts[0]
    rendered_catalog = prompt.split("Available tools (subset; full schema is provided separately):\n", 1)[1].split(
        "\n\nOutput schema:",
        1,
    )[0]
    catalog_lines = [line for line in rendered_catalog.splitlines() if line.startswith("- ")]
    expected_catalog_lines = [f"- {tool}() — {tool} description" for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST]
    first_observation_tool = next(tool for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST if tool.startswith(("get_", "list_")))
    first_action_tool = next(
        tool
        for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST
        if tool != END_TURN_TOOL and not tool.startswith(("get_", "list_"))
    )
    output_schema = prompt.split("Output schema:", 1)[1]

    assert catalog_lines == expected_catalog_lines
    assert f'{{"tool": "{first_observation_tool}"' in output_schema
    assert f'{{"tool": "{first_action_tool}"' in output_schema
    assert f'"tool": "{END_TURN_TOOL}"' in output_schema


def test_planner_prompt_allowlist_sections_track_private_default_without_stale_tools() -> None:
    stale_tools = ("run_lua", "click", "stale_removed_tool")
    tool_catalog = {
        tool: {"description": f"{tool} description", "input_schema": {"properties": {}}}
        for tool in (*planner_module._DEFAULT_PLANNER_TOOL_ALLOWLIST, *stale_tools)
    }
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=tool_catalog)

    planner.plan(strategy="Pursue science victory.", state_context="Turn 1.", recent_calls="(none)")

    prompt = provider.captured_prompts[0]
    rendered_catalog = prompt.split("Available tools (subset; full schema is provided separately):\n", 1)[1].split(
        "\n\nOutput schema:",
        1,
    )[0]
    output_schema = prompt.split("Output schema:", 1)[1]
    catalog_tool_names = [
        line.removeprefix("- ").split("(", 1)[0] for line in rendered_catalog.splitlines() if line.startswith("- ")
    ]
    example_tool_names = re.findall(r'"tool": "([^"]+)"', output_schema)

    assert catalog_tool_names == list(planner_module._DEFAULT_PLANNER_TOOL_ALLOWLIST)
    assert set(example_tool_names).issubset(planner_module._DEFAULT_PLANNER_TOOL_ALLOWLIST)
    assert not set(stale_tools).intersection(catalog_tool_names)
    assert not set(stale_tools).intersection(example_tool_names)


def test_planner_prompt_examples_use_planner_default_even_if_prompt_default_drifts(monkeypatch) -> None:
    monkeypatch.setattr(
        planner_prompts,
        "DEFAULT_PLANNER_TOOL_ALLOWLIST",
        ("list_stale_prompt_default", "stale_prompt_action", END_TURN_TOOL),
    )
    tool_catalog = {
        tool: {"description": f"{tool} description", "input_schema": {"properties": {}}}
        for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST
    }
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=tool_catalog)

    planner.plan(strategy="Pursue science victory.", state_context="Turn 1.", recent_calls="(none)")

    output_schema = provider.captured_prompts[0].split("Output schema:", 1)[1]
    first_observation_tool = next(tool for tool in planner.allowed_tools if tool.startswith(("get_", "list_")))
    first_action_tool = next(
        tool for tool in planner.allowed_tools if tool != END_TURN_TOOL and not tool.startswith(("get_", "list_"))
    )

    assert f'{{"tool": "{first_observation_tool}"' in output_schema
    assert f'{{"tool": "{first_action_tool}"' in output_schema
    assert "stale_prompt" not in output_schema


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


def test_planner_system_prompt_examples_follow_allowed_tools() -> None:
    prompt = build_planner_system_prompt(
        tool_catalog="- get_game_overview() — current overview\n- purchase_tile() — buy tile\n- end_turn() — advance",
        allowed_tools=("get_game_overview", "purchase_tile", "end_turn"),
    )

    assert '{"tool": "get_game_overview"' in prompt
    assert '{"tool": "purchase_tile"' in prompt
    assert '"tool": "end_turn"' in prompt
    assert '{"tool": "get_units"' not in prompt
    assert '{"tool": "set_research"' not in prompt


def test_planner_system_prompt_examples_default_to_canonical_allowlist() -> None:
    prompt = build_planner_system_prompt(tool_catalog="- end_turn() — advance")
    first_observation_tool = next(tool for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST if tool.startswith(("get_", "list_")))
    first_action_tool = next(
        tool
        for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST
        if tool != END_TURN_TOOL and not tool.startswith(("get_", "list_"))
    )

    assert f'{{"tool": "{first_observation_tool}"' in prompt
    assert f'{{"tool": "{first_action_tool}"' in prompt
    assert f'"tool": "{END_TURN_TOOL}"' in prompt


def test_planner_system_prompt_examples_default_to_planner_module_allowlist(monkeypatch) -> None:
    canonical_allowlist = ("list_planner_canonical", "planner_canonical_action", END_TURN_TOOL)
    monkeypatch.setattr(planner_module, "DEFAULT_PLANNER_TOOL_ALLOWLIST", canonical_allowlist)

    prompt = build_planner_system_prompt(tool_catalog="- end_turn() — advance")

    assert '{"tool": "list_planner_canonical"' in prompt
    assert '{"tool": "planner_canonical_action"' in prompt
    assert '{"tool": "get_game_overview"' not in prompt
    assert '{"tool": "set_research"' not in prompt


def test_planner_from_observation_rejects_non_civ6_mcp_payload() -> None:
    provider = FakeProvider([json.dumps(_valid_plan_payload())])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG)
    observation = Civ6McpNormalizedObservation(backend="vlm", planner_context="VLM screenshot observation")

    with pytest.raises(ValueError, match="cannot consume backend 'vlm'"):
        planner.plan_from_observation(
            observation=observation,
            strategy="Pursue science victory.",
            recent_calls="(none)",
        )

    assert provider.captured_prompts == []


@pytest.mark.parametrize(
    ("payload", "expected_tools"),
    [
        (_valid_plan_payload(), ["get_game_overview", "set_research", "end_turn"]),
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


def test_planner_rejects_top_level_tool_call_list() -> None:
    provider = FakeProvider([json.dumps(_valid_plan_payload()["tool_calls"])])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="top-level JSON object.*tool_calls"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


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


def test_planner_retries_on_invalid_plan_structure_then_succeeds() -> None:
    bad = json.dumps({"tool_calls": {"tool": "get_game_overview", "arguments": {}}})
    good = json.dumps(_valid_plan_payload())
    provider = FakeProvider([bad, good])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=2)

    result = planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(result.tool_calls) == 3
    assert len(provider.captured_prompts) == 2
    assert "Planner validation error" in provider.captured_prompts[1]
    assert "'tool_calls' must be a list" in provider.captured_prompts[1]


def test_planner_retries_when_plan_does_not_end_turn_then_succeeds() -> None:
    missing_end_turn = {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    }
    good = _valid_plan_payload()
    provider = FakeProvider([json.dumps(missing_end_turn), json.dumps(good)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=1)

    result = planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert [call.tool for call in result.tool_calls] == ["get_game_overview", "set_research", "end_turn"]
    assert len(provider.captured_prompts) == 2
    assert "final tool call must be end_turn" in provider.captured_prompts[1]


def test_planner_rejects_plan_without_end_turn_after_exhausting_retries() -> None:
    payload = {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    }
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="final tool call must be end_turn"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


def test_planner_exposes_missing_end_turn_output_for_backend_fallback() -> None:
    payload = {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    }
    raw = json.dumps(payload)
    provider = FakeProvider([raw])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(MissingEndTurnPlannerOutputError, match="final tool call must be end_turn") as exc_info:
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    exc = exc_info.value
    assert [call.tool for call in exc.tool_calls] == ["get_game_overview", "set_research"]
    assert exc.raw_response == raw
    assert exc.parsed_payload == payload
    assert len(provider.captured_prompts) == 1


@pytest.mark.parametrize("field", _REQUIRED_REFLECTION_FIELDS)
def test_planner_rejects_successful_plan_missing_required_end_turn_reflection(field: str) -> None:
    payload = _valid_plan_payload()
    del payload["tool_calls"][-1]["arguments"][field]
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match=f"end_turn requires non-empty reflection fields: {field}"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


@pytest.mark.parametrize("field", _REQUIRED_REFLECTION_FIELDS)
def test_planner_rejects_successful_plan_blank_required_end_turn_reflection(field: str) -> None:
    payload = _valid_plan_payload()
    payload["tool_calls"][-1]["arguments"][field] = "   "
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match=f"end_turn requires non-empty reflection fields: {field}"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


def test_planner_rejects_successful_plan_non_string_end_turn_reflection() -> None:
    payload = _valid_plan_payload()
    payload["tool_calls"][-1]["arguments"]["tactical"] = ["not", "a", "reflection"]
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="end_turn reflection fields must be strings: tactical"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


def test_planner_repairs_trailing_calls_after_end_turn() -> None:
    payload = _valid_plan_payload()
    payload["tool_calls"].append({"tool": "get_game_overview", "arguments": {}, "reasoning": "too late"})
    provider = FakeProvider([json.dumps(payload)])
    planner = Civ6McpToolPlanner(
        provider=provider,
        tool_catalog=_TOOL_CATALOG,
        allowed_tools=("get_game_overview", "set_research", "end_turn"),
        max_retries=0,
    )

    result = planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert [call.tool for call in result.tool_calls] == ["get_game_overview", "set_research", "end_turn"]
    assert result.parsed_payload is not None
    assert [call["tool"] for call in result.parsed_payload["tool_calls"]] == [
        "get_game_overview",
        "set_research",
        "end_turn",
    ]


def test_planner_rejects_vlm_action_plan_output_for_civ6_mcp_backend() -> None:
    vlm_plan = {
        "primitive_name": "click_next_turn",
        "reasoning": "VLM-style pixel plan should not be accepted by civ6-mcp.",
        "actions": [{"type": "click", "x": 800, "y": 900, "button": "left"}],
    }
    provider = FakeProvider([json.dumps(vlm_plan)])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="VLM/computer-use action-plan payload cannot run"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


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
    with pytest.raises(RuntimeError, match="civ6-mcp planner exhausted retries"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 3


def test_planner_raises_runtime_error_after_exhausting_invalid_plan_structure_retries() -> None:
    bad = json.dumps({"tool_calls": {"tool": "get_game_overview", "arguments": {}}})
    provider = FakeProvider([bad, bad, bad])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=2)

    with pytest.raises(RuntimeError, match="'tool_calls' must be a list"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 3
    assert "Planner validation error" in provider.captured_prompts[1]
    assert "Planner validation error" in provider.captured_prompts[2]


def test_planner_rejects_markdown_fenced_json_without_fallback_parsing() -> None:
    fenced = "```json\n" + json.dumps(_valid_plan_payload()) + "\n```"
    provider = FakeProvider([fenced])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="Expecting value"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1


def test_planner_rejects_json_embedded_in_prose_without_fallback_parsing() -> None:
    prose_wrapped = "Here is the plan:\n" + json.dumps(_valid_plan_payload())
    provider = FakeProvider([prose_wrapped])
    planner = Civ6McpToolPlanner(provider=provider, tool_catalog=_TOOL_CATALOG, max_retries=0)

    with pytest.raises(RuntimeError, match="Expecting value"):
        planner.plan(strategy="x", state_context="y", recent_calls="z")

    assert len(provider.captured_prompts) == 1
