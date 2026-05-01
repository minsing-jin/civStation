"""Tests for the civ6-mcp executor and tool-call coercion.

These tests use a fake client so they run without FireTuner / Civ6 / `uv`.
"""

from __future__ import annotations

from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    ToolCall,
    coerce_tool_calls,
)


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


def test_executor_marks_unknown_tool() -> None:
    client = FakeCiv6McpClient(responses={"get_units": "u1"}, known={"get_units"})
    executor = Civ6McpExecutor(client)  # type: ignore[arg-type]
    result = executor.execute(ToolCall(tool="not_real"))
    assert result.success is False
    assert result.classification == "error"
    assert "not_real" in result.error


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
