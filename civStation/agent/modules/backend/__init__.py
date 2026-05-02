"""Backend selection layer.

Currently supported backends:
- vlm: existing VLM/computer-use pipeline (default).
- civ6_mcp: tool-call backend that drives Civ6 via the civ6-mcp MCP server
  (https://github.com/lmwilki/civ6-mcp) over the FireTuner protocol.

Backends are mutually exclusive at runtime; users pick one with --backend.
"""

from civStation.agent.modules.backend.selector import (
    BackendKind,
    BackendNotConfiguredError,
    parse_backend_kind,
)

_CIV6_MCP_LAZY_EXPORTS = (
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_ALIASES",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_REGISTRY",
    "CIV6_MCP_FREE_FORM_ACTION_TYPE_TO_MCP_TOOL",
    "DEFAULT_CIV6_MCP_OBSERVE_TOOLS",
    "Civ6McpClient",
    "Civ6McpClientFactory",
    "Civ6McpClientProtocol",
    "Civ6McpConfig",
    "Civ6McpError",
    "Civ6McpHealth",
    "Civ6McpUnavailableError",
    "Civ6McpActionMappingError",
    "Civ6McpFreeFormActionType",
    "Civ6McpExecutor",
    "MappedCiv6McpAction",
    "Civ6McpObserver",
    "Civ6McpObserverFactory",
    "Civ6McpPrioritizedIntent",
    "Civ6McpTurnPlan",
    "Civ6McpTurnConfig",
    "Civ6McpTurnLoopConfig",
    "Civ6McpTurnRequestContext",
    "Civ6McpTurnResult",
    "Civ6McpTurnState",
    "build_civ6_mcp_client",
    "build_civ6_mcp_observer",
    "build_prioritized_turn_plan",
    "map_civ6_mcp_action",
    "map_civ6_mcp_action_details",
    "map_civ6_mcp_actions",
    "run_civ6_mcp_turn_loop",
    "run_multi_turn_civ6_mcp",
    "run_one_turn_civ6_mcp",
)


def __getattr__(name: str):
    """Lazily expose civ6-mcp hooks without importing MCP dependencies on selector import."""
    if name in _CIV6_MCP_LAZY_EXPORTS:
        from civStation.agent.modules.backend import civ6_mcp

        return getattr(civ6_mcp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendKind",
    "BackendNotConfiguredError",
    "parse_backend_kind",
    *_CIV6_MCP_LAZY_EXPORTS,
]
