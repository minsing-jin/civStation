"""civ6-mcp backend — drives Civ6 via the upstream civ6-mcp MCP server.

The upstream project (github.com/lmwilki/civ6-mcp) exposes Civ6's internal
state and commands through a Python MCP server that talks to the game over
FireTuner TCP. We treat it as a black box and wrap it as a backend, leaving
the existing VLM/computer-use stack untouched.
"""

from civStation.agent.modules.backend.civ6_mcp.client import (
    Civ6McpClient,
    Civ6McpConfig,
    Civ6McpError,
    Civ6McpUnavailableError,
)
from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    ToolCall,
    ToolCallResult,
)
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner

__all__ = [
    "Civ6McpClient",
    "Civ6McpConfig",
    "Civ6McpError",
    "Civ6McpUnavailableError",
    "Civ6McpExecutor",
    "Civ6McpObserver",
    "Civ6McpToolPlanner",
    "ToolCall",
    "ToolCallResult",
]
