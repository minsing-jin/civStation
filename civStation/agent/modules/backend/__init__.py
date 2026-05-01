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

__all__ = [
    "BackendKind",
    "BackendNotConfiguredError",
    "parse_backend_kind",
]
