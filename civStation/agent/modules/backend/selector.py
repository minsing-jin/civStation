"""Backend kind enum and parsing.

Kept as a tiny standalone module so other layers can import it without
pulling in the heavy civ6_mcp client dependency tree.
"""

from __future__ import annotations

from enum import Enum


class BackendKind(str, Enum):
    """Which action/observation backend the agent runs against."""

    VLM = "vlm"
    CIV6_MCP = "civ6-mcp"


class BackendNotConfiguredError(RuntimeError):
    """Raised when a backend is selected but its environment is missing."""


def parse_backend_kind(raw: str | None) -> BackendKind:
    """Parse a CLI/yaml string into a BackendKind, defaulting to VLM."""
    if not raw:
        return BackendKind.VLM
    value = raw.strip().lower().replace("_", "-")
    if value in {"vlm", "computer-use", "cu"}:
        return BackendKind.VLM
    if value in {"civ6-mcp", "civ-mcp", "mcp", "civmcp"}:
        return BackendKind.CIV6_MCP
    raise ValueError(f"Unknown backend '{raw}'. Choose from: vlm, civ6-mcp.")
