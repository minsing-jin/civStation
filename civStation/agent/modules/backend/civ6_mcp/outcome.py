"""civ6-mcp command-outcome classification.

These rules are backend-local: they classify textual outcomes returned by the
upstream civ6-mcp tool server, not screenshot progress, VLM planning quality,
or computer-use action execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.response import (
    CIV6_MCP_CLASSIFICATION_PRECEDENCE,
    Civ6McpClassificationRule,
    Civ6McpClassificationStatus,
    Civ6McpResponseClassification,
    classify_civ6_mcp_status,
    classify_civ6_mcp_text,
)
from civStation.agent.modules.backend.civ6_mcp.results import ToolCallResult
from civStation.utils.project_runtime import get_project_runtime_root

Civ6McpCommandOutcomeRule = Civ6McpClassificationRule
CIV6_MCP_COMMAND_OUTCOME_RULES: tuple[Civ6McpCommandOutcomeRule, ...] = CIV6_MCP_CLASSIFICATION_PRECEDENCE
CIV6_MCP_TURN_OUTCOME_LOG_FILENAME = "civ6_mcp_turn_outcomes.jsonl"

_SUCCESS_CLASSIFICATIONS = frozenset(
    {
        Civ6McpResponseClassification.OK,
        Civ6McpResponseClassification.SOFT_BLOCK,
    }
)
_TERMINAL_STATUSES = frozenset(
    {
        Civ6McpClassificationStatus.ABORTED,
        Civ6McpClassificationStatus.HANG,
        Civ6McpClassificationStatus.GAME_OVER,
    }
)


@dataclass(frozen=True)
class Civ6McpCommandOutcome:
    """Backend-local command outcome consumed by civ6-mcp runtime decisions."""

    status: Civ6McpClassificationStatus
    classification: Civ6McpResponseClassification
    success: bool
    text: str = ""
    error: str = ""
    retryable: bool = False
    terminal: bool = False
    timed_out: bool = False


@dataclass(frozen=True)
class Civ6McpExecutedToolCallOutcome:
    """Structured record for one executed civ6-mcp tool call."""

    tool: str
    arguments: dict[str, Any]
    reasoning: str
    success: bool
    text: str
    error: str
    classification: str
    status: str
    retryable: bool
    terminal: bool
    timed_out: bool
    content_blocks: tuple[str, ...] = ()
    structured_content: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render this tool-call outcome as a JSON-safe mapping."""
        return {
            "tool": self.tool,
            "arguments": _json_safe(self.arguments),
            "reasoning": self.reasoning,
            "success": self.success,
            "text": self.text,
            "error": self.error,
            "classification": self.classification,
            "status": self.status,
            "retryable": self.retryable,
            "terminal": self.terminal,
            "timed_out": self.timed_out,
            "content_blocks": list(self.content_blocks),
            "structured_content": _json_safe(self.structured_content),
        }


@dataclass(frozen=True)
class Civ6McpTurnOutcomeRecord:
    """Durable structured outcome for one civ6-mcp turn."""

    turn_index: int
    observation_summary: str
    planner_output: dict[str, Any]
    executed_tool_calls: list[Civ6McpExecutedToolCallOutcome]
    errors: list[str]
    final_turn_status: str
    success: bool
    end_turn_called: bool
    game_over: bool
    terminal_condition: str
    created_at_utc: str
    backend: str = "civ6-mcp"
    schema_version: int = 1
    observation: dict[str, Any] | None = None
    synthesized_end_turn_reflection: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render this turn outcome as a JSON-safe mapping."""
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "created_at_utc": self.created_at_utc,
            "turn_index": self.turn_index,
            "observation_summary": self.observation_summary,
            "observation": _json_safe(self.observation or {}),
            "planner_output": _json_safe(self.planner_output),
            "executed_tool_calls": [outcome.to_dict() for outcome in self.executed_tool_calls],
            "errors": list(self.errors),
            "final_turn_status": self.final_turn_status,
            "success": self.success,
            "end_turn_called": self.end_turn_called,
            "game_over": self.game_over,
            "terminal_condition": self.terminal_condition,
            "synthesized_end_turn_reflection": _json_safe(self.synthesized_end_turn_reflection or {}),
        }


def classify_civ6_mcp_command_outcome(
    text: str,
    *,
    is_error: bool = False,
    timed_out: bool = False,
) -> Civ6McpCommandOutcome:
    """Classify one civ6-mcp command result without VLM/computer-use semantics."""
    body = str(text or "").strip()

    if timed_out:
        status = Civ6McpClassificationStatus.RETRYABLE
        classification = Civ6McpResponseClassification.TIMEOUT
    else:
        status = classify_civ6_mcp_status(body)
        classification = classify_civ6_mcp_text(body)

    if is_error and status == Civ6McpClassificationStatus.SUCCESS:
        status = Civ6McpClassificationStatus.FATAL
    if is_error and classification == Civ6McpResponseClassification.OK:
        classification = Civ6McpResponseClassification.ERROR

    success = not is_error and classification in _SUCCESS_CLASSIFICATIONS
    error = "" if success else body
    if timed_out and not error:
        error = "civ6-mcp command timed out"

    return Civ6McpCommandOutcome(
        status=status,
        classification=classification,
        success=success,
        text=body,
        error=error,
        retryable=status == Civ6McpClassificationStatus.RETRYABLE,
        terminal=status in _TERMINAL_STATUSES,
        timed_out=timed_out or classification == Civ6McpResponseClassification.TIMEOUT,
    )


def get_civ6_mcp_turn_outcome_log_path(base_dir: Path | str | None = None) -> Path:
    """Return the deterministic JSONL path for latest civ6-mcp turn outcomes."""
    return get_project_runtime_root(base_dir=base_dir) / CIV6_MCP_TURN_OUTCOME_LOG_FILENAME


def append_civ6_mcp_turn_outcome(
    record: Civ6McpTurnOutcomeRecord,
    *,
    path: Path | str | None = None,
) -> Path:
    """Append one structured turn outcome as a JSONL record and return the path."""
    output_path = Path(path) if path is not None else get_civ6_mcp_turn_outcome_log_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return output_path


def build_civ6_mcp_turn_outcome_record(turn_result: object) -> Civ6McpTurnOutcomeRecord:
    """Build a durable structured outcome from a ``Civ6McpTurnResult``-like object."""
    state = getattr(turn_result, "state", None)
    normalized_observation = getattr(state, "normalized_observation", None)
    tool_results = list(getattr(turn_result, "tool_results", []) or [])
    planner_output = getattr(state, "planner_output", {}) if state is not None else {}
    observation_summary = _observation_summary(normalized_observation)

    return Civ6McpTurnOutcomeRecord(
        turn_index=int(getattr(turn_result, "turn_index", 0) or 0),
        observation_summary=observation_summary,
        observation=_observation_payload(normalized_observation, observation_summary),
        planner_output=dict(planner_output)
        if isinstance(planner_output, dict)
        else {"value": _json_safe(planner_output)},
        executed_tool_calls=[_tool_result_record(outcome) for outcome in tool_results],
        errors=_collect_turn_errors(turn_result, state, tool_results),
        final_turn_status=_final_turn_status(turn_result, state),
        success=bool(getattr(turn_result, "success", False)),
        end_turn_called=bool(getattr(turn_result, "end_turn_called", False)),
        game_over=bool(getattr(turn_result, "game_over", False)),
        terminal_condition=str(getattr(turn_result, "terminal_condition", "") or ""),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        synthesized_end_turn_reflection=_synthesized_end_turn_reflection_payload(turn_result),
    )


def _tool_result_record(outcome: ToolCallResult) -> Civ6McpExecutedToolCallOutcome:
    call = outcome.call
    return Civ6McpExecutedToolCallOutcome(
        tool=str(call.tool),
        arguments=dict(call.arguments),
        reasoning=str(call.reasoning or ""),
        success=bool(outcome.success),
        text=str(outcome.text or ""),
        error=str(outcome.error or ""),
        classification=str(outcome.classification or ""),
        status=str(outcome.status or ""),
        retryable=bool(outcome.retryable),
        terminal=bool(outcome.terminal),
        timed_out=bool(outcome.timed_out),
        content_blocks=tuple(str(block) for block in outcome.content_blocks),
        structured_content=outcome.structured_content,
    )


def _synthesized_end_turn_reflection_payload(turn_result: object) -> dict[str, Any]:
    payload = getattr(turn_result, "synthesized_end_turn_reflection", {}) or {}
    if not isinstance(payload, dict):
        return {"value": _json_safe(payload)}
    return dict(payload)


def _observation_summary(normalized_observation: object | None) -> str:
    if normalized_observation is None:
        return ""
    game_updates = getattr(normalized_observation, "game_observation_updates", {}) or {}
    summary = game_updates.get("situation_summary") if isinstance(game_updates, dict) else ""
    if summary:
        return str(summary)

    global_updates = getattr(normalized_observation, "global_context_updates", {}) or {}
    if isinstance(global_updates, dict):
        parts = []
        if global_updates.get("current_turn") is not None:
            parts.append(f"Turn {global_updates['current_turn']}")
        if global_updates.get("game_era"):
            parts.append(f"Era {global_updates['game_era']}")
        if global_updates.get("current_research"):
            parts.append(f"Research {global_updates['current_research']}")
        if global_updates.get("current_civic"):
            parts.append(f"Civic {global_updates['current_civic']}")
        if parts:
            return " | ".join(parts)

    raw_sections = getattr(normalized_observation, "raw_sections", {}) or {}
    if isinstance(raw_sections, dict):
        overview = str(raw_sections.get("OVERVIEW") or "").strip()
        if overview:
            return overview.splitlines()[0][:240]
    return ""


def _observation_payload(normalized_observation: object | None, summary: str) -> dict[str, Any]:
    if normalized_observation is None:
        return {"summary": summary}
    raw_sections = getattr(normalized_observation, "raw_sections", {}) or {}
    return {
        "summary": summary,
        "backend": str(getattr(normalized_observation, "backend", "") or ""),
        "global_context_updates": _json_safe(getattr(normalized_observation, "global_context_updates", {}) or {}),
        "game_observation_updates": _json_safe(getattr(normalized_observation, "game_observation_updates", {}) or {}),
        "raw_section_names": sorted(str(key) for key in raw_sections) if isinstance(raw_sections, dict) else [],
    }


def _collect_turn_errors(
    turn_result: object,
    state: object | None,
    tool_results: list[ToolCallResult],
) -> list[str]:
    candidates = [
        str(getattr(turn_result, "error_message", "") or ""),
        str(getattr(state, "error_message", "") or "") if state is not None else "",
    ]
    for outcome in tool_results:
        if outcome.success:
            continue
        text = outcome.error or outcome.text
        if text:
            candidates.append(str(text))
    return _dedupe_nonempty(candidates)


def _final_turn_status(turn_result: object, state: object | None) -> str:
    phase = str(getattr(state, "phase", "") or "") if state is not None else ""
    if phase:
        return phase
    terminal_condition = str(getattr(turn_result, "terminal_condition", "") or "")
    if terminal_condition:
        return "game_over" if bool(getattr(turn_result, "game_over", False)) else terminal_condition
    if bool(getattr(turn_result, "success", False)):
        return "completed"
    if str(getattr(turn_result, "error_message", "") or ""):
        return "failed"
    return "unknown"


def _dedupe_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_json_safe(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return _json_safe(enum_value)
    return repr(value)


__all__ = [
    "CIV6_MCP_COMMAND_OUTCOME_RULES",
    "CIV6_MCP_TURN_OUTCOME_LOG_FILENAME",
    "Civ6McpCommandOutcome",
    "Civ6McpCommandOutcomeRule",
    "Civ6McpExecutedToolCallOutcome",
    "Civ6McpTurnOutcomeRecord",
    "append_civ6_mcp_turn_outcome",
    "build_civ6_mcp_turn_outcome_record",
    "classify_civ6_mcp_command_outcome",
    "get_civ6_mcp_turn_outcome_log_path",
]
