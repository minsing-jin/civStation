"""Tests for the civ6-mcp executor and tool-call coercion.

These tests use a fake client so they run without FireTuner / Civ6 / `uv`.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    ToolCall,
    coerce_tool_call,
    coerce_tool_calls,
)
from civStation.agent.modules.backend.civ6_mcp.planner_types import Civ6McpPlannerAction


class FakeCiv6McpClient:
    """Minimal stand-in matching Civ6McpClient's surface used by the executor."""

    def __init__(self, responses: dict[str, str], known: set[str] | None = None) -> None:
        self._responses = responses
        self._known = known if known is not None else set(responses.keys())
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._known

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        text = self._responses.get(name, "")
        if isinstance(text, Exception):
            raise text
        return text


class StrictMcpOnlyClient:
    """Fake client that proves executor dispatch goes through MCP tool calls only."""

    def __init__(self, response: str | BaseException, known: set[str]) -> None:
        self._response = response
        self._known = known
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._known

    def call_tool_result(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


@pytest.fixture
def fail_if_vlm_dispatch_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test if civ6-mcp executor reaches the VLM/computer-use path."""

    def forbidden_execute_action(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp executor must dispatch only through the MCP client")

    screen_module = sys.modules.get("civStation.utils.screen")
    turn_executor_module = sys.modules.get("civStation.agent.turn_executor")
    if screen_module is None:
        screen_module = SimpleNamespace(execute_action=forbidden_execute_action)
        monkeypatch.setitem(sys.modules, "civStation.utils.screen", screen_module)
    else:
        monkeypatch.setattr(screen_module, "execute_action", forbidden_execute_action)
    if turn_executor_module is None:
        turn_executor_module = SimpleNamespace(execute_action=forbidden_execute_action)
        monkeypatch.setitem(sys.modules, "civStation.agent.turn_executor", turn_executor_module)
    else:
        monkeypatch.setattr(turn_executor_module, "execute_action", forbidden_execute_action)


def test_coerce_accepts_dict_with_tool_calls_key() -> None:
    payload = {"tool_calls": [{"tool": "get_units", "arguments": {}}]}
    calls = coerce_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0].tool == "get_units"


def test_coerce_accepts_bare_list() -> None:
    payload = [{"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}, "reasoning": "lit boost"}]
    calls = coerce_tool_calls(payload)
    assert len(calls) == 1
    assert calls[0].tool == "set_research"
    assert calls[0].arguments == {"tech_or_civic": "WRITING"}
    assert calls[0].reasoning == "lit boost"


def test_coerce_rejects_missing_tool_field() -> None:
    with pytest.raises(ValueError):
        coerce_tool_calls({"tool_calls": [{"arguments": {}}]})


def test_coerce_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError):
        coerce_tool_calls(42)


def test_coerce_accepts_planner_action_object() -> None:
    action = Civ6McpPlannerAction(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )

    call = coerce_tool_call(action)

    assert call == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )


def test_executor_accepts_planned_action_object() -> None:
    action = Civ6McpPlannerAction(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )
    client = FakeCiv6McpClient(responses={"set_research": "Research set."}, known={"set_research"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(action)

    assert result.success is True
    assert result.classification == "ok"
    assert result.call == ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )
    assert client.calls == [("set_research", {"tech_or_civic": "WRITING"})]


def test_executor_accepts_mapping_planned_action() -> None:
    client = FakeCiv6McpClient(responses={"set_research": "Research set."}, known={"set_research"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(
        {
            "tool": "set_research",
            "arguments": {"tech_or_civic": "WRITING"},
            "reasoning": "unlock campuses",
        }
    )

    assert result.success is True
    assert result.call.tool == "set_research"
    assert result.call.arguments == {"tech_or_civic": "WRITING"}
    assert result.call.reasoning == "unlock campuses"


def test_executor_dispatch_success_uses_only_mcp_client(fail_if_vlm_dispatch_is_used: None) -> None:
    client = StrictMcpOnlyClient(response="Research set.", known={"set_research"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}))

    assert result.success is True
    assert result.classification == "ok"
    assert result.text == "Research set."
    assert result.error == ""
    assert client.calls == [("set_research", {"tech_or_civic": "WRITING"})]


def test_executor_unsupported_action_does_not_dispatch_to_mcp_or_vlm(
    fail_if_vlm_dispatch_is_used: None,
) -> None:
    client = StrictMcpOnlyClient(response="should not be called", known={"set_research"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute({"type": "click", "x": 500, "y": 600})

    assert result.success is False
    assert result.classification == "error"
    assert "missing 'tool' name" in result.error
    assert client.calls == []


def test_executor_mcp_client_failure_does_not_fallback_to_vlm(
    fail_if_vlm_dispatch_is_used: None,
) -> None:
    client = StrictMcpOnlyClient(
        response=Civ6McpError("RUN ABORTED: upstream civ6-mcp server exited."),
        known={"set_research"},
    )
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}))

    assert result.success is False
    assert result.classification == "aborted"
    assert result.text == ""
    assert result.error == "RUN ABORTED: upstream civ6-mcp server exited."
    assert client.calls == [("set_research", {"tech_or_civic": "WRITING"})]


def test_executor_marks_unknown_tool() -> None:
    client = FakeCiv6McpClient(responses={"get_units": "u1"}, known={"get_units"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    result = executor.execute(ToolCall(tool="not_real"))
    assert result.success is False
    assert result.classification == "error"
    assert "not_real" in result.error


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("set_research", {"tech_or_civic": "WRITING"}),
        ("unit_action", {"unit_id": 7, "action": "FORTIFY"}),
        ("dismiss_popup", {"popup_id": "goody_hut"}),
    ],
)
def test_executor_classifies_supported_action_tools_as_ok(tool: str, arguments: dict[str, Any]) -> None:
    client = FakeCiv6McpClient(responses={tool: f"{tool} completed."}, known={tool})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(ToolCall(tool=tool, arguments=arguments))

    assert result.success is True
    assert result.classification == "ok"
    assert result.text == f"{tool} completed."
    assert result.error == ""
    assert client.calls == [(tool, arguments)]


def test_executor_rejects_unknown_action_without_calling_client() -> None:
    client = FakeCiv6McpClient(responses={"set_research": "Research set."}, known={"set_research"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]

    result = executor.execute(ToolCall(tool="launch_orbital_laser", arguments={"target": "barbarian_camp"}))

    assert result.success is False
    assert result.classification == "error"
    assert "Unsupported civ6-mcp tool" in result.error
    assert "launch_orbital_laser" in result.error
    assert client.calls == []


def test_executor_classifies_game_over() -> None:
    client = FakeCiv6McpClient(
        responses={"end_turn": "*** GAME OVER — VICTORY! ***"},
        known={"end_turn"},
    )
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    result = executor.execute(
        ToolCall(
            tool="end_turn",
            arguments={
                "tactical": "a",
                "strategic": "b",
                "tooling": "c",
                "planning": "d",
                "hypothesis": "e",
            },
        )
    )
    assert result.classification == "game_over"
    assert result.success is False  # game_over halts execution


def test_executor_classifies_blocked_end_turn() -> None:
    client = FakeCiv6McpClient(
        responses={"end_turn": "Cannot end turn: incoming trade deal pending."},
        known={"end_turn"},
    )
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    result = executor.execute(
        ToolCall(
            tool="end_turn",
            arguments={
                "tactical": "a",
                "strategic": "b",
                "tooling": "c",
                "planning": "d",
                "hypothesis": "e",
            },
        )
    )
    assert result.classification == "blocked"


def test_executor_many_stops_on_terminal() -> None:
    client = FakeCiv6McpClient(
        responses={
            "get_units": "ok",
            "end_turn": "*** GAME OVER — DEFEAT ***",
            "set_research": "should_not_be_called",
        },
    )
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    results = executor.execute_many(
        [
            ToolCall(tool="get_units"),
            ToolCall(
                tool="end_turn",
                arguments={
                    "tactical": "a",
                    "strategic": "b",
                    "tooling": "c",
                    "planning": "d",
                    "hypothesis": "e",
                },
            ),
            ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}),
        ]
    )
    tools_called = [name for name, _ in client.calls]
    assert "set_research" not in tools_called
    assert results[-1].classification == "game_over"


def test_executor_classifies_generic_error_prefix() -> None:
    client = FakeCiv6McpClient(
        responses={"set_research": "Error: must specify a known tech"},
        known={"set_research"},
    )
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    result = executor.execute(ToolCall(tool="set_research", arguments={"tech_or_civic": "X"}))
    assert result.classification == "error"
    assert result.success is False
