"""civ6-mcp command-outcome classification tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from civStation.agent.modules.backend.civ6_mcp import outcome
from civStation.agent.modules.backend.civ6_mcp.observation_schema import Civ6McpNormalizedObservation
from civStation.agent.modules.backend.civ6_mcp.outcome import (
    CIV6_MCP_COMMAND_OUTCOME_RULES,
    Civ6McpCommandOutcome,
    append_civ6_mcp_turn_outcome,
    build_civ6_mcp_turn_outcome_record,
    classify_civ6_mcp_command_outcome,
)
from civStation.agent.modules.backend.civ6_mcp.response import (
    CIV6_MCP_CLASSIFICATION_PRECEDENCE,
    Civ6McpClassificationStatus,
    Civ6McpResponseClassification,
)
from civStation.agent.modules.backend.civ6_mcp.results import ToolCall, ToolCallResult


def test_command_outcome_rules_are_the_civ6_mcp_backend_rules() -> None:
    assert CIV6_MCP_COMMAND_OUTCOME_RULES == CIV6_MCP_CLASSIFICATION_PRECEDENCE
    assert [rule.status for rule in CIV6_MCP_COMMAND_OUTCOME_RULES] == [
        Civ6McpClassificationStatus.GAME_OVER,
        Civ6McpClassificationStatus.ABORTED,
        Civ6McpClassificationStatus.HANG,
        Civ6McpClassificationStatus.FATAL,
        Civ6McpClassificationStatus.BLOCKED,
        Civ6McpClassificationStatus.RETRYABLE,
        Civ6McpClassificationStatus.SUCCESS,
    ]


def test_command_outcome_module_has_no_vlm_or_computer_use_dependencies() -> None:
    source = inspect.getsource(outcome)

    assert "civStation.utils.screen" not in source
    assert "civStation.agent.turn_executor" not in source
    assert "BaseVLMProvider" not in source
    assert "execute_action" not in source
    assert "pyautogui" not in source


@pytest.mark.parametrize(
    (
        "text",
        "expected_status",
        "expected_classification",
        "expected_success",
        "expected_retryable",
        "expected_terminal",
    ),
    [
        (
            "Research set.",
            Civ6McpClassificationStatus.SUCCESS,
            Civ6McpResponseClassification.OK,
            True,
            False,
            False,
        ),
        (
            "Cannot end turn: incoming trade deal pending.",
            Civ6McpClassificationStatus.BLOCKED,
            Civ6McpResponseClassification.BLOCKED,
            False,
            False,
            False,
        ),
        (
            "End turn requested, but units still need orders.",
            Civ6McpClassificationStatus.RETRYABLE,
            Civ6McpResponseClassification.SOFT_BLOCK,
            True,
            True,
            False,
        ),
        (
            "RUN ABORTED: upstream civ6-mcp server exited.",
            Civ6McpClassificationStatus.ABORTED,
            Civ6McpResponseClassification.ABORTED,
            False,
            False,
            True,
        ),
        (
            "HANG RECOVERY FAILED after repeated no-op turns.",
            Civ6McpClassificationStatus.HANG,
            Civ6McpResponseClassification.HANG,
            False,
            False,
            True,
        ),
        (
            "*** GAME OVER - VICTORY ***",
            Civ6McpClassificationStatus.GAME_OVER,
            Civ6McpResponseClassification.GAME_OVER,
            False,
            False,
            True,
        ),
    ],
)
def test_classify_command_outcome_uses_civ6_mcp_text_semantics(
    text: str,
    expected_status: Civ6McpClassificationStatus,
    expected_classification: Civ6McpResponseClassification,
    expected_success: bool,
    expected_retryable: bool,
    expected_terminal: bool,
) -> None:
    result = classify_civ6_mcp_command_outcome(text)

    assert isinstance(result, Civ6McpCommandOutcome)
    assert result.status == expected_status
    assert result.classification == expected_classification
    assert result.success is expected_success
    assert result.retryable is expected_retryable
    assert result.terminal is expected_terminal


@pytest.mark.parametrize(
    "vlm_pipeline_text",
    [
        "VLM returned no action",
        "No UI change for 2 consecutive steps",
        "Semantic verification failed",
        "computer-use action produced no screenshot progress",
    ],
)
def test_command_outcome_does_not_treat_vlm_pipeline_phrases_as_civ6_mcp_failures(
    vlm_pipeline_text: str,
) -> None:
    result = classify_civ6_mcp_command_outcome(vlm_pipeline_text)

    assert result.status == Civ6McpClassificationStatus.SUCCESS
    assert result.classification == Civ6McpResponseClassification.OK
    assert result.success is True
    assert result.error == ""


def test_command_outcome_can_mark_transport_error_without_vlm_fallback_semantics() -> None:
    result = classify_civ6_mcp_command_outcome("plain transport failure", is_error=True)

    assert result.status == Civ6McpClassificationStatus.FATAL
    assert result.classification == Civ6McpResponseClassification.ERROR
    assert result.success is False
    assert result.error == "plain transport failure"


def test_command_outcome_can_mark_timeout_without_screen_progress_semantics() -> None:
    result = classify_civ6_mcp_command_outcome("", timed_out=True)

    assert result.status == Civ6McpClassificationStatus.RETRYABLE
    assert result.classification == Civ6McpResponseClassification.TIMEOUT
    assert result.success is False
    assert result.retryable is True
    assert result.timed_out is True
    assert result.error == "civ6-mcp command timed out"


def test_turn_outcome_record_includes_structured_per_turn_fields() -> None:
    normalized_observation = Civ6McpNormalizedObservation(
        global_context_updates={"current_turn": 42, "game_era": "Classical"},
        game_observation_updates={"situation_summary": "Turn 42 | Era Classical"},
        raw_sections={"OVERVIEW": "Turn: 42", "NOTIFICATIONS": "Unit needs orders"},
    )
    state = SimpleNamespace(
        phase="failed",
        normalized_observation=normalized_observation,
        planner_output={
            "raw_response": '{"tool_calls":[]}',
            "tool_calls": [{"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}}],
        },
        error_message="civ6-mcp operation failed",
    )
    tool_result = ToolCallResult(
        call=ToolCall(
            tool="set_research",
            arguments={"tech_or_civic": "UNKNOWN"},
            reasoning="Choose writing if available.",
        ),
        success=False,
        text="Error: unknown tech",
        error="Error: unknown tech",
        classification="error",
        status="fatal",
    )
    turn_result = SimpleNamespace(
        turn_index=3,
        success=False,
        tool_results=[tool_result],
        error_message="civ6-mcp operation failed",
        end_turn_called=False,
        game_over=False,
        terminal_condition="",
        state=state,
    )

    record = build_civ6_mcp_turn_outcome_record(turn_result)
    payload = record.to_dict()

    assert payload["backend"] == "civ6-mcp"
    assert payload["turn_index"] == 3
    assert payload["observation_summary"] == "Turn 42 | Era Classical"
    assert payload["observation"]["raw_section_names"] == ["NOTIFICATIONS", "OVERVIEW"]
    assert payload["planner_output"]["tool_calls"][0]["tool"] == "set_research"
    assert payload["executed_tool_calls"][0]["tool"] == "set_research"
    assert payload["executed_tool_calls"][0]["arguments"] == {"tech_or_civic": "UNKNOWN"}
    assert payload["errors"] == ["civ6-mcp operation failed", "Error: unknown tech"]
    assert payload["final_turn_status"] == "failed"


def test_append_turn_outcome_persists_jsonl_record(tmp_path: Path) -> None:
    turn_result = SimpleNamespace(
        turn_index=1,
        success=True,
        tool_results=[
            ToolCallResult(
                call=ToolCall(tool="end_turn", arguments={"strategic": "Keep science pace."}),
                success=True,
                text="Turn advanced.",
                classification="ok",
                status="success",
            )
        ],
        error_message="",
        end_turn_called=True,
        game_over=False,
        terminal_condition="",
        synthesized_end_turn_reflection={
            "backend": "civ6-mcp",
            "action": "end_turn",
            "source": "civ6_mcp_turn_loop",
            "reason": "planner_missing_end_turn",
            "synthesized": True,
            "turn_index": 1,
            "reflection_fields": {"strategic": "Keep science pace."},
            "planner_tool_calls": ["set_research"],
        },
        state=SimpleNamespace(
            phase="completed",
            normalized_observation=Civ6McpNormalizedObservation(
                game_observation_updates={"situation_summary": "Turn 1"}
            ),
            planner_output={"tool_calls": [{"tool": "end_turn"}]},
            error_message="",
        ),
    )
    record = build_civ6_mcp_turn_outcome_record(turn_result)
    output_path = tmp_path / "turn_outcomes.jsonl"

    returned_path = append_civ6_mcp_turn_outcome(record, path=output_path)

    assert returned_path == output_path
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["final_turn_status"] == "completed"
    assert payload["executed_tool_calls"][0]["tool"] == "end_turn"
    assert payload["synthesized_end_turn_reflection"] == {
        "backend": "civ6-mcp",
        "action": "end_turn",
        "source": "civ6_mcp_turn_loop",
        "reason": "planner_missing_end_turn",
        "synthesized": True,
        "turn_index": 1,
        "reflection_fields": {"strategic": "Keep science pace."},
        "planner_tool_calls": ["set_research"],
    }
