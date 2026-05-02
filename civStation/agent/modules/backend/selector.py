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
    """Parse a CLI/yaml string into a BackendKind, defaulting to VLM.

    The VLM/computer-use backend is intentionally conservative: only the
    explicit civ6-mcp selection enters the MCP backend. Every other value
    resolves to VLM so older configs and omitted flags keep existing behavior.
    """
    if not raw:
        return BackendKind.VLM
    value = raw.strip().lower()
    if value == "civ6-mcp":
        return BackendKind.CIV6_MCP
    return BackendKind.VLM
