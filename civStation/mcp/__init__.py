"""Lazy public exports for layered MCP server and civ6-mcp adapters."""

from __future__ import annotations

from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "CIV6_MCP_ADAPTER_NAME": "civ6_mcp_adapters",
    "CIV6_MCP_TOOL_CALL_ACTION": "civ6_mcp_adapters",
    "CIV6_MCP_TOOL_PLAN_ACTION": "civ6_mcp_adapters",
    "Civ6McpPlannerTransportPayload": "civ6_mcp_adapters",
    "Civ6McpRouteResult": "civ6_mcp_adapters",
    "LayerAdapterRegistry": "runtime",
    "LayeredComputerUseMCP": "server",
    "LayeredSession": "session",
    "SessionRegistry": "session",
    "SessionRuntimeConfig": "runtime",
    "adapter_overrides_for_backend": "runtime",
    "encode_civ6_mcp_planner_result": "civ6_mcp_adapters",
    "make_civ6_mcp_executor_adapter": "civ6_mcp_adapters",
    "make_civ6_mcp_observer_adapter": "civ6_mcp_adapters",
    "make_civ6_mcp_planner_adapter": "civ6_mcp_adapters",
    "make_civ6_mcp_router_adapter": "civ6_mcp_adapters",
    "register_civ6_mcp_adapters": "civ6_mcp_adapters",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve MCP package exports without importing optional backend modules eagerly."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    if module_name == "runtime":
        from civStation.mcp.runtime import LayerAdapterRegistry, SessionRuntimeConfig, adapter_overrides_for_backend

        value = {
            "LayerAdapterRegistry": LayerAdapterRegistry,
            "SessionRuntimeConfig": SessionRuntimeConfig,
            "adapter_overrides_for_backend": adapter_overrides_for_backend,
        }[name]
    elif module_name == "session":
        from civStation.mcp.session import LayeredSession, SessionRegistry

        value = {
            "LayeredSession": LayeredSession,
            "SessionRegistry": SessionRegistry,
        }[name]
    elif module_name == "server":
        from civStation.mcp.server import LayeredComputerUseMCP

        value = LayeredComputerUseMCP
    elif module_name == "civ6_mcp_adapters":
        from civStation.mcp import civ6_mcp_adapters

        value = getattr(civ6_mcp_adapters, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List lazy public MCP exports alongside loaded module globals."""
    return sorted((*globals(), *__all__))
