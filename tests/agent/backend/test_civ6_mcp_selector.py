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


def test_civ6_mcp_aliases() -> None:
    assert parse_backend_kind("civ6-mcp") is BackendKind.CIV6_MCP
    assert parse_backend_kind("civ-mcp") is BackendKind.CIV6_MCP
    assert parse_backend_kind("civ6_mcp") is BackendKind.CIV6_MCP
    assert parse_backend_kind("MCP") is BackendKind.CIV6_MCP
    assert parse_backend_kind("civmcp") is BackendKind.CIV6_MCP


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError):
        parse_backend_kind("anthropic")


def test_backend_kind_string_value() -> None:
    assert BackendKind.VLM.value == "vlm"
    assert BackendKind.CIV6_MCP.value == "civ6-mcp"
