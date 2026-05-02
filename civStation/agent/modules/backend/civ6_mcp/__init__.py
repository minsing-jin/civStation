"""civ6-mcp backend public interface.

The upstream project (github.com/lmwilki/civ6-mcp) exposes Civ6's internal
state and commands through a Python MCP server that talks to the game over
FireTuner TCP. This package keeps exports lazy so importing the client startup
path does not initialize VLM/computer-use modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "ACTION_TOOLS": "operations",
    "CIV6_MCP_CLASSIFICATION_PRECEDENCE": "response",
    "CIV6_MCP_COMMAND_OUTCOME_RULES": "outcome",
    "CIV6_MCP_EXCEPTION_CLASSIFICATION_PRECEDENCE": "response",
    "CIV6_MCP_CONTEXT_FIELD_MAPPINGS": "observation_schema",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES": "action_mapping",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY": "action_mapping",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL": "action_mapping",
    "CIV6_MCP_OBSERVATION_SECTION_MAPPINGS": "observation_schema",
    "CIV6_MCP_TURN_OUTCOME_LOG_FILENAME": "outcome",
    "DEFAULT_CIV6_MCP_OBSERVE_TOOLS": "observer",
    "END_TURN_REFLECTION_FIELDS": "operations",
    "END_TURN_TOOL": "operations",
    "OBSERVATION_TOOLS": "operations",
    "PLANNED_ACTION_TYPE_TO_MCP_TOOL": "action_mapping",
    "SUPPORTED_CIV6_MCP_TOOLS": "operations",
    "Civ6McpAction": "planner_types",
    "Civ6McpActionMappingError": "action_mapping",
    "Civ6McpActionType": "planner_types",
    "Civ6McpBackendAction": "planner_types",
    "Civ6McpBackendIntent": "planner_types",
    "Civ6McpClient": "client",
    "Civ6McpClientFactory": "turn_loop",
    "Civ6McpClientProtocol": "client",
    "Civ6McpCommandOutcome": "outcome",
    "Civ6McpCommandOutcomeRule": "outcome",
    "Civ6McpClassificationRule": "response",
    "Civ6McpClassificationStatus": "response",
    "Civ6McpConfig": "client",
    "Civ6McpContextFieldMapping": "observation_schema",
    "Civ6McpDispatchResult": "operations",
    "Civ6McpError": "client",
    "Civ6McpExceptionClassificationRule": "response",
    "Civ6McpExecutor": "executor",
    "Civ6McpExecutedToolCallOutcome": "outcome",
    "Civ6McpFreeFormActionType": "action_mapping",
    "Civ6McpHealth": "client",
    "Civ6McpIntent": "planner_types",
    "Civ6McpIntentKind": "planner_types",
    "Civ6McpIntentType": "planner_types",
    "Civ6McpNormalizedObservation": "observation_schema",
    "Civ6McpNormalizedResult": "response",
    "Civ6McpObservationSectionMapping": "observation_schema",
    "Civ6McpObserver": "observer",
    "Civ6McpObserverFactory": "turn_loop",
    "Civ6McpOperationDispatcher": "operations",
    "Civ6McpPlanner": "planner_types",
    "Civ6McpPlannerAction": "planner_types",
    "Civ6McpPlannerIntent": "planner_types",
    "Civ6McpPlannerProvider": "planner_types",
    "Civ6McpPlannerResponse": "planner_types",
    "Civ6McpPrioritizedIntent": "turn_planning",
    "Civ6McpRequest": "operations",
    "Civ6McpRequestBuilder": "operations",
    "Civ6McpResponseClassification": "response",
    "Civ6McpToolPlanner": "planner",
    "Civ6McpTurnConfig": "turn_loop",
    "Civ6McpTurnLoopConfig": "turn_loop",
    "Civ6McpTurnOutcomeRecord": "outcome",
    "Civ6McpTurnPlan": "turn_planning",
    "Civ6McpTurnRequestContext": "turn_loop",
    "Civ6McpTurnResult": "turn_loop",
    "Civ6McpTurnState": "turn_loop",
    "Civ6McpUnavailableError": "client",
    "GameOverviewSnapshot": "state_parser",
    "MappedCiv6McpAction": "action_mapping",
    "MissingEndTurnPlannerOutputError": "planner",
    "PlannerResult": "planner",
    "StateBundle": "state_parser",
    "SupportedCiv6McpOperation": "operations",
    "ToolCall": "results",
    "ToolCallResult": "results",
    "append_civ6_mcp_turn_outcome": "outcome",
    "build_civ6_mcp_client": "turn_loop",
    "build_civ6_mcp_observer": "observer",
    "build_civ6_mcp_turn_outcome_record": "outcome",
    "build_game_observation_fields": "observation_schema",
    "build_game_observation_updates": "observation_schema",
    "build_global_context_updates": "observation_schema",
    "build_prioritized_turn_plan": "turn_planning",
    "build_situation_summary": "observation_schema",
    "classify_civ6_mcp_exception": "response",
    "classify_civ6_mcp_exception_status": "response",
    "classify_civ6_mcp_command_outcome": "outcome",
    "classify_civ6_mcp_status": "response",
    "classify_civ6_mcp_text": "operations",
    "coerce_civ6_mcp_requests": "operations",
    "coerce_tool_call": "executor",
    "coerce_tool_calls": "executor",
    "executor_result_from_mcp_error": "results",
    "executor_result_from_mcp_timeout": "results",
    "executor_result_from_mcp_tool_result": "results",
    "executor_result_from_normalized_response": "results",
    "get_civ6_mcp_turn_outcome_log_path": "outcome",
    "infer_civ6_mcp_intent_type": "planner_types",
    "map_civ6_mcp_action": "action_mapping",
    "map_civ6_mcp_action_details": "action_mapping",
    "map_civ6_mcp_actions": "action_mapping",
    "normalize_mcp_response_error": "response",
    "normalize_mcp_response_exception": "response",
    "normalize_mcp_response_text": "response",
    "normalize_mcp_response_timeout": "response",
    "normalize_mcp_tool_result": "response",
    "normalize_observation_bundle": "observation_schema",
    "normalize_raw_mcp_game_state": "observation_schema",
    "operation_for_tool": "operations",
    "parse_game_overview": "state_parser",
    "render_planner_context": "observation_schema",
    "run_civ6_mcp_turn_loop": "turn_loop",
    "run_multi_turn_civ6_mcp": "turn_loop",
    "run_one_turn_civ6_mcp": "turn_loop",
    "section_texts_for_bundle": "observation_schema",
    "state_bundle_from_raw_mcp_state": "state_parser",
    "tool_call_result_from_dispatch": "results",
    "tool_results_for_bundle": "observation_schema",
    "validate_civ6_mcp_request": "operations",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve public backend exports without eager cross-backend imports."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
