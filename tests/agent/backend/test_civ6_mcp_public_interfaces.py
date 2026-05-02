"""Public package interfaces for the civ6-mcp backend."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import civStation.agent.modules.backend.civ6_mcp.turn_loop as turn_loop_module
from civStation.agent.modules.backend import (
    Civ6McpObserver as BackendCiv6McpObserver,
)
from civStation.agent.modules.backend import (
    build_civ6_mcp_observer as build_backend_civ6_mcp_observer,
)
from civStation.agent.modules.backend import (
    civ6_mcp,
)
from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol, Civ6McpHealth
from civStation.agent.modules.backend.civ6_mcp.observer import (
    DEFAULT_CIV6_MCP_OBSERVE_TOOLS,
    Civ6McpObserver,
    build_civ6_mcp_observer,
)
from civStation.agent.modules.backend.civ6_mcp.planner import DEFAULT_PLANNER_TOOL_ALLOWLIST
from civStation.agent.modules.backend.civ6_mcp.response import Civ6McpNormalizedResult, normalize_mcp_response_text
from civStation.agent.modules.backend.civ6_mcp.turn_loop import run_civ6_mcp_turn_loop


class FakeClient:
    @property
    def tool_names(self) -> set[str]:
        return {"get_game_overview"}

    @property
    def tool_catalog(self) -> dict[str, dict[str, Any]]:
        return {"get_game_overview": {"description": "overview", "input_schema": {}}}

    def has_tool(self, name: str) -> bool:
        return name in self.tool_names

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return self.tool_catalog

    def health_check(self, required_tools: set[str] | frozenset[str] | None = None) -> Civ6McpHealth:  # noqa: ARG002
        return Civ6McpHealth(ok=True, started=True, initialized=True, tool_count=1)

    @property
    def startup_health(self) -> Civ6McpHealth:
        return Civ6McpHealth(ok=True, started=True, initialized=True, tool_count=1)

    @property
    def has_required_tools(self) -> bool:
        return True

    @property
    def missing_required_tools(self) -> frozenset[str]:
        return frozenset()

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:  # noqa: ARG002
        return "Turn: 1"

    def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Civ6McpNormalizedResult:
        return normalize_mcp_response_text(name, arguments, "Turn: 1")


def test_civ6_mcp_package_exports_public_interfaces() -> None:
    expected = {
        "DEFAULT_CIV6_MCP_OBSERVE_TOOLS",
        "DEFAULT_PLANNER_TOOL_ALLOWLIST",
        "CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES",
        "CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY",
        "CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL",
        "Civ6McpClient",
        "Civ6McpClientProtocol",
        "Civ6McpConfig",
        "Civ6McpError",
        "Civ6McpHealth",
        "Civ6McpUnavailableError",
        "Civ6McpAction",
        "Civ6McpActionMappingError",
        "Civ6McpActionType",
        "Civ6McpFreeFormActionType",
        "Civ6McpBackendAction",
        "Civ6McpBackendIntent",
        "Civ6McpExecutor",
        "Civ6McpIntent",
        "Civ6McpIntentKind",
        "Civ6McpIntentType",
        "Civ6McpObserver",
        "Civ6McpNormalizedResult",
        "Civ6McpNormalizedObservation",
        "Civ6McpObservationSectionMapping",
        "Civ6McpContextFieldMapping",
        "Civ6McpResponseClassification",
        "Civ6McpPlanner",
        "Civ6McpPlannerAction",
        "Civ6McpPlannerIntent",
        "Civ6McpPlannerProvider",
        "Civ6McpPlannerResponse",
        "Civ6McpPrioritizedIntent",
        "Civ6McpToolPlanner",
        "Civ6McpTurnPlan",
        "Civ6McpTurnConfig",
        "Civ6McpTurnLoopConfig",
        "Civ6McpTurnResult",
        "Civ6McpTurnState",
        "GameOverviewSnapshot",
        "MappedCiv6McpAction",
        "PlannerResult",
        "StateBundle",
        "ToolCall",
        "ToolCallResult",
        "CIV6_MCP_CONTEXT_FIELD_MAPPINGS",
        "CIV6_MCP_OBSERVATION_SECTION_MAPPINGS",
        "build_civ6_mcp_observer",
        "build_civ6_mcp_client",
        "build_game_observation_fields",
        "build_game_observation_updates",
        "build_global_context_updates",
        "build_situation_summary",
        "build_prioritized_turn_plan",
        "coerce_tool_call",
        "coerce_tool_calls",
        "infer_civ6_mcp_intent_type",
        "map_civ6_mcp_action",
        "map_civ6_mcp_action_details",
        "map_civ6_mcp_actions",
        "normalize_observation_bundle",
        "normalize_raw_mcp_game_state",
        "normalize_mcp_response_error",
        "normalize_mcp_response_text",
        "normalize_mcp_response_timeout",
        "normalize_mcp_tool_result",
        "render_planner_context",
        "parse_game_overview",
        "run_civ6_mcp_turn_loop",
        "run_multi_turn_civ6_mcp",
        "run_one_turn_civ6_mcp",
        "section_texts_for_bundle",
        "state_bundle_from_raw_mcp_state",
        "tool_results_for_bundle",
    }
    assert expected.issubset(set(civ6_mcp.__all__))
    for name in expected:
        assert hasattr(civ6_mcp, name)


def test_civ6_mcp_allowlist_public_api_resolves_to_planner_export() -> None:
    assert "DEFAULT_PLANNER_TOOL_ALLOWLIST" in dir(civ6_mcp)
    assert civ6_mcp.DEFAULT_PLANNER_TOOL_ALLOWLIST is DEFAULT_PLANNER_TOOL_ALLOWLIST
    assert civ6_mcp.DEFAULT_PLANNER_TOOL_ALLOWLIST[-1] == "end_turn"


def test_civ6_mcp_client_protocol_documents_minimal_surface() -> None:
    client = FakeClient()
    assert isinstance(client, Civ6McpClientProtocol)
    assert client.has_tool("get_game_overview") is True
    assert client.call_tool("get_game_overview") == "Turn: 1"


def test_civ6_mcp_observer_factory_is_available_from_backend_packages() -> None:
    ctx = object()
    client = FakeClient()

    observer = build_civ6_mcp_observer(
        client=client,  # type: ignore[arg-type]
        context_manager=ctx,  # type: ignore[arg-type]
        observe_tools=("get_game_overview",),
    )

    assert isinstance(observer, Civ6McpObserver)
    assert observer._observe_tools == ("get_game_overview",)
    assert DEFAULT_CIV6_MCP_OBSERVE_TOOLS[0] == "get_game_overview"
    assert BackendCiv6McpObserver is Civ6McpObserver
    assert build_backend_civ6_mcp_observer is build_civ6_mcp_observer


def test_run_civ6_mcp_turn_loop_interface_excludes_vlm_computer_use_parameters() -> None:
    signature = inspect.signature(run_civ6_mcp_turn_loop)

    forbidden_parameters = {
        "router_provider",
        "router_img_config",
        "planner_img_config",
        "context_img_config",
        "turn_detector",
        "turn_detector_img_config",
        "macro_turn_manager",
        "context_updater",
        "knowledge_manager",
        "normalizing_range",
        "delay_before_action",
        "prompt_language",
    }

    assert forbidden_parameters.isdisjoint(signature.parameters)
    assert signature.parameters["planner_provider"].annotation == "Civ6McpPlannerProvider"
    assert signature.parameters["install_path"].annotation == "str | None"
    assert signature.parameters["launcher"].annotation == "str | None"


def test_civ6_mcp_turn_loop_module_does_not_import_vlm_or_computer_use_runtime() -> None:
    source = inspect.getsource(turn_loop_module)
    module_ast = ast.parse(source)
    imported_modules: set[str] = set()

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden_prefixes = (
        "civStation.agent.turn_executor",
        "civStation.agent.modules.router",
        "civStation.utils.llm_provider",
        "civStation.utils.screen",
        "civStation.utils.image_pipeline",
        "PIL",
    )

    assert not {module for module in imported_modules if module == "pyautogui" or module.startswith(forbidden_prefixes)}
