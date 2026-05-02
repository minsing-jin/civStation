"""Tests for the backend selector enum + parser."""

from __future__ import annotations

import pytest

from civStation.agent.modules.backend import BackendKind, parse_backend_kind


def test_default_backend_is_vlm() -> None:
    assert parse_backend_kind(None) is BackendKind.VLM
    assert parse_backend_kind("") is BackendKind.VLM


def test_vlm_aliases() -> None:
    assert parse_backend_kind("vlm") is BackendKind.VLM
    assert parse_backend_kind("VLM") is BackendKind.VLM
    assert parse_backend_kind("computer-use") is BackendKind.VLM
    assert parse_backend_kind("computer_use") is BackendKind.VLM
    assert parse_backend_kind("cu") is BackendKind.VLM


def test_exact_civ6_mcp_selects_civ6_mcp() -> None:
    assert parse_backend_kind("civ6-mcp") is BackendKind.CIV6_MCP


@pytest.mark.parametrize(
    "raw",
    [
        "CIV6-MCP",
        "Civ6-Mcp",
        " civ6-mcp ",
        "\tciv6-mcp\n",
    ],
)
def test_normalized_civ6_mcp_variants_select_civ6_mcp(raw: str) -> None:
    assert parse_backend_kind(raw) is BackendKind.CIV6_MCP


@pytest.mark.parametrize(
    "raw",
    [
        "anthropic",
        "civ6",
        "civ-mcp",
        "civ6_mcp",
        "mcp",
        "civmcp",
        "vlm,civ6-mcp",
        "computer-use,civ6-mcp",
        "vlm+civ6-mcp",
        "mcp-vlm",
    ],
)
def test_non_civ6_mcp_backend_values_select_vlm(raw: str) -> None:
    assert parse_backend_kind(raw) is BackendKind.VLM


def test_backend_kind_string_value() -> None:
    assert BackendKind.VLM.value == "vlm"
    assert BackendKind.CIV6_MCP.value == "civ6-mcp"
