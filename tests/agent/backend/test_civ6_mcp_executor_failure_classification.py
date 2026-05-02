"""Failure classification tests for the civ6-mcp executor."""

from __future__ import annotations

from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.executor import Civ6McpExecutor, ToolCall
from civStation.agent.modules.backend.civ6_mcp.response import normalize_mcp_response_timeout

END_TURN_ARGUMENTS = {
    "tactical": "no tactical blockers",
    "strategic": "continue expansion",
    "tooling": "all tools responded",
    "planning": "advance one turn",
    "hypothesis": "next observation will expose new choices",
}


class FakeTextClient:
    """Fake legacy text-only civ6-mcp client used by executor tests."""

    def __init__(self, responses: dict[str, object], known: set[str] | None = None) -> None:
        self._responses = responses
        self._known = set(responses) if known is None else known
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._known

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, dict(arguments or {})))
        response = self._responses.get(name, "")
        if isinstance(response, BaseException):
            raise response
        return str(response)


class FakeTypedClient:
    """Fake client exposing normalized MCP results through call_tool_result."""

    def __init__(self, result: object, known: set[str]) -> None:
        self._result = result
        self._known = known
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._known

    def call_tool_result(self, name: str, arguments: dict[str, Any] | None = None) -> object:
        self.calls.append((name, dict(arguments or {})))
        return self._result


def test_executor_classifies_supported_tool_missing_from_backend_as_error() -> None:
    client = FakeTextClient(responses={}, known=set())
    executor = Civ6McpExecutor(client)

    result = executor.execute(ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}))

    assert result.success is False
    assert result.classification == "error"
    assert result.text == ""
    assert "not exposed by civ6-mcp server" in result.error
    assert client.calls == []


def test_executor_classifies_invalid_end_turn_payload_as_error_without_backend_call() -> None:
    client = FakeTextClient(responses={"end_turn": "should not be called"}, known={"end_turn"})
    executor = Civ6McpExecutor(client)

    result = executor.execute(ToolCall(tool="end_turn", arguments={"tactical": "only one reflection"}))

    assert result.success is False
    assert result.classification == "error"
    assert "end_turn requires non-empty reflection fields" in result.error
    assert client.calls == []


@pytest.mark.parametrize(
    ("text", "expected_category", "expected_success"),
    [
        ("Error: must specify a known tech", "error", False),
        ("Cannot end turn: incoming trade deal pending.", "blocked", False),
        ("End turn requested but units still need orders.", "soft_block", True),
        ("RUN ABORTED: FireTuner bridge disconnected.", "aborted", False),
        ("HANG RECOVERY FAILED after repeated end-turn checks.", "hang", False),
        ("*** GAME OVER - VICTORY ***", "game_over", False),
    ],
)
def test_executor_maps_tool_response_failures_to_expected_categories(
    text: str,
    expected_category: str,
    expected_success: bool,
) -> None:
    client = FakeTextClient(responses={"end_turn": text}, known={"end_turn"})
    executor = Civ6McpExecutor(client)

    result = executor.execute(ToolCall(tool="end_turn", arguments=END_TURN_ARGUMENTS))

    assert result.classification == expected_category
    assert result.success is expected_success
    assert result.text == text
    assert result.error == ("" if expected_success else text)
    assert client.calls == [("end_turn", END_TURN_ARGUMENTS)]


def test_executor_classifies_backend_exception_with_civ6_mcp_failure_prefix() -> None:
    client = FakeTextClient(
        responses={"end_turn": Civ6McpError("RUN ABORTED: upstream civ6-mcp server exited.")},
        known={"end_turn"},
    )
    executor = Civ6McpExecutor(client)

    result = executor.execute(ToolCall(tool="end_turn", arguments=END_TURN_ARGUMENTS))

    assert result.success is False
    assert result.classification == "aborted"
    assert result.text == ""
    assert result.error == "RUN ABORTED: upstream civ6-mcp server exited."
    assert client.calls == [("end_turn", END_TURN_ARGUMENTS)]


def test_executor_preserves_typed_timeout_failure_category() -> None:
    timeout = normalize_mcp_response_timeout("end_turn", END_TURN_ARGUMENTS, timeout_seconds=2.5)
    client = FakeTypedClient(result=timeout, known={"end_turn"})
    executor = Civ6McpExecutor(client)

    result = executor.execute(ToolCall(tool="end_turn", arguments=END_TURN_ARGUMENTS))

    assert result.success is False
    assert result.classification == "timeout"
    assert result.text == ""
    assert "timed out after 2.5s" in result.error
    assert client.calls == [("end_turn", END_TURN_ARGUMENTS)]


@pytest.mark.parametrize(
    ("terminal_text", "expected_category"),
    [
        ("RUN ABORTED: operator stopped automation.", "aborted"),
        ("HANG RECOVERY FAILED after repeated no-op turns.", "hang"),
    ],
)
def test_executor_many_stops_on_terminal_failure_categories(terminal_text: str, expected_category: str) -> None:
    client = FakeTextClient(
        responses={
            "get_units": "Unit 1: Warrior",
            "end_turn": terminal_text,
            "set_research": "should not be called",
        },
    )
    executor = Civ6McpExecutor(client)

    results = executor.execute_many(
        [
            ToolCall(tool="get_units"),
            ToolCall(tool="end_turn", arguments=END_TURN_ARGUMENTS),
            ToolCall(tool="set_research", arguments={"tech_or_civic": "WRITING"}),
        ]
    )

    assert [result.classification for result in results] == ["ok", expected_category]
    assert client.calls == [
        ("get_units", {}),
        ("end_turn", END_TURN_ARGUMENTS),
    ]
