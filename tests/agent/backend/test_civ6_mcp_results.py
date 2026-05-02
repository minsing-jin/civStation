"""Executor-result normalization for civ6-mcp MCP responses."""

from __future__ import annotations

import inspect
from typing import Any

from civStation.agent.modules.backend.civ6_mcp import results
from civStation.agent.modules.backend.civ6_mcp.results import (
    ToolCall,
    executor_result_from_mcp_error,
    executor_result_from_mcp_timeout,
    executor_result_from_mcp_tool_result,
)


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeToolResult:
    def __init__(self, content: list[Any], *, is_error: bool = False) -> None:
        self.content = content
        self.isError = is_error


def test_tool_response_normalizes_to_executor_result() -> None:
    call = ToolCall(
        tool="set_research",
        arguments={"tech_or_civic": "WRITING"},
        reasoning="unlock campuses",
    )
    raw = FakeToolResult([FakeTextBlock("Research set.")])

    result = executor_result_from_mcp_tool_result(call, raw)

    assert result.call is call
    assert result.success is True
    assert result.text == "Research set."
    assert result.error == ""
    assert result.classification == "ok"


def test_json_rpc_error_normalizes_to_executor_result_without_raising() -> None:
    call = ToolCall(tool="set_research", arguments={"tech_or_civic": "UNKNOWN"})
    raw = FakeToolResult([FakeTextBlock("must specify a known tech")], is_error=True)

    result = executor_result_from_mcp_tool_result(call, raw)

    assert result.success is False
    assert result.text == "must specify a known tech"
    assert result.error == "must specify a known tech"
    assert result.classification == "error"


def test_transport_error_normalizes_to_executor_result() -> None:
    call = ToolCall(tool="end_turn")

    result = executor_result_from_mcp_error(call, "RUN ABORTED: upstream server exited")

    assert result.success is False
    assert result.text == ""
    assert result.error == "RUN ABORTED: upstream server exited"
    assert result.classification == "aborted"


def test_timeout_normalizes_to_executor_result() -> None:
    call = ToolCall(tool="end_turn")

    result = executor_result_from_mcp_timeout(call, timeout_seconds=2.5)

    assert result.success is False
    assert result.text == ""
    assert "timed out after 2.5s" in result.error
    assert result.classification == "timeout"


def test_results_module_does_not_reference_vlm_or_computer_use_pipeline() -> None:
    source = inspect.getsource(results)

    assert "civStation.utils.screen" not in source
    assert "execute_action" not in source
    assert "pyautogui" not in source
