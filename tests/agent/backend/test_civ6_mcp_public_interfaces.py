"""Public package interfaces for the civ6-mcp backend."""

from __future__ import annotations

from typing import Any

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
from civStation.agent.modules.backend.civ6_mcp.response import Civ6McpNormalizedResult, normalize_mcp_response_text


class FakeClient:
    @property
    def tool_names(self) -> set[str]:
        return {"get_game_overview"}

    def has_tool(self, name: str) -> bool:
        return name in self.tool_names

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {"get_game_overview": {"description": "overview", "input_schema": {}}}

    def health_check(self, required_tools: set[str] | frozenset[str] | None = None) -> Civ6McpHealth:  # noqa: ARG002
        return Civ6McpHealth(ok=True, started=True, initialized=True, tool_count=1)

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
        "Civ6McpClient",
        "Civ6McpClientProtocol",
        "Civ6McpConfig",
        "Civ6McpError",
        "Civ6McpHealth",
        "Civ6McpUnavailableError",
        "Civ6McpAction",
        "Civ6McpActionMappingError",
        "Civ6McpActionType",
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
        "run_multi_turn_civ6_mcp",
        "run_one_turn_civ6_mcp",
        "section_texts_for_bundle",
        "state_bundle_from_raw_mcp_state",
    }
    assert expected.issubset(set(civ6_mcp.__all__))
    for name in expected:
        assert hasattr(civ6_mcp, name)


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
