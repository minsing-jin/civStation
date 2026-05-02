"""Lifecycle tests for the civ6-mcp stdio MCP client.

These tests use fake MCP SDK objects so they verify client lifecycle behavior
without spawning civ6-mcp, Civ6, FireTuner, or uv.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import (
    Civ6McpClient,
    Civ6McpConfig,
    Civ6McpError,
)


class FakeTool:
    def __init__(self, name: str, description: str = "", input_schema: dict[str, Any] | None = None) -> None:
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {}


class FakeToolsResponse:
    def __init__(self, tools: list[FakeTool]) -> None:
        self.tools = tools


class FakeResultBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeCallResult:
    isError = False

    def __init__(self, text: str) -> None:
        self.content = [FakeResultBlock(text)]


class FakeStdioServerParameters:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.env = env


class FakeImplementation:
    def __init__(self, *, name: str, version: str) -> None:
        self.name = name
        self.version = version


class FakeStdioContext:
    events: list[str] = []
    entered_params = None

    async def __aenter__(self):
        self.events.append("stdio_enter")
        return object(), object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.events.append("stdio_exit")


class FakeClientSession:
    events: list[str] = []
    tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
    ]
    ping_count = 0

    def __init__(self, read_stream, write_stream, **kwargs) -> None:  # noqa: ANN001
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.kwargs = kwargs

    async def __aenter__(self):
        self.events.append("session_enter")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.events.append("session_exit")

    async def initialize(self) -> None:
        self.events.append("initialize")

    async def list_tools(self) -> FakeToolsResponse:
        self.events.append("list_tools")
        return FakeToolsResponse(self.tools)

    async def ping(self) -> None:
        self.events.append("ping")
        self.ping_count += 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeCallResult:
        self.events.append(f"call_tool:{name}")
        return FakeCallResult(f"{name} ok {arguments}")


@pytest.fixture
def fake_mcp_sdk(monkeypatch):
    mcp = types.ModuleType("mcp")
    client = types.ModuleType("mcp.client")
    stdio = types.ModuleType("mcp.client.stdio")
    mcp_types = types.ModuleType("mcp.types")

    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)
    monkeypatch.setattr(mcp, "client", client, raising=False)
    monkeypatch.setattr(client, "stdio", stdio, raising=False)
    monkeypatch.setattr(mcp, "StdioServerParameters", FakeStdioServerParameters, raising=False)
    monkeypatch.setattr(mcp_types, "Implementation", FakeImplementation, raising=False)

    FakeStdioContext.events = []
    FakeStdioContext.entered_params = None
    FakeClientSession.events = []
    FakeClientSession.ping_count = 0
    FakeClientSession.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
    ]

    def fake_stdio_client(params):
        FakeStdioContext.entered_params = params
        return FakeStdioContext()

    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession, raising=False)
    monkeypatch.setattr(stdio, "stdio_client", fake_stdio_client, raising=False)
    return FakeClientSession, FakeStdioContext


@pytest.fixture
def install_path(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    return tmp_path


def test_start_initializes_session_lists_tools_and_shutdown_closes_contexts(
    fake_mcp_sdk,
    install_path: Path,
) -> None:
    session, stdio = fake_mcp_sdk
    config = Civ6McpConfig(install_path=install_path, launcher="python")
    client = Civ6McpClient(config)

    client.start()

    assert client.tool_names == {"get_game_overview", "end_turn"}
    assert client.tool_schemas()["get_game_overview"]["description"] == "overview"
    assert session.events[:3] == ["session_enter", "initialize", "list_tools"]
    assert stdio.events == ["stdio_enter"]
    assert stdio.entered_params.command == "python"
    assert stdio.entered_params.args == ["-m", "civ_mcp"]

    client.stop()

    assert session.events[-1] == "session_exit"
    assert stdio.events[-1] == "stdio_exit"
    assert client.tool_names == set()


def test_health_check_pings_session_and_refreshes_tool_catalog(fake_mcp_sdk, install_path: Path) -> None:
    session, _stdio = fake_mcp_sdk
    config = Civ6McpConfig(install_path=install_path, launcher="python")
    client = Civ6McpClient(config)
    client.start()

    session.tools = [
        FakeTool("get_game_overview", "overview", {"type": "object"}),
        FakeTool("end_turn", "end the turn", {"type": "object"}),
        FakeTool("get_units", "units", {"type": "object"}),
    ]
    health = client.health_check(required_tools={"get_game_overview", "end_turn"})

    assert health.ok is True
    assert health.started is True
    assert health.initialized is True
    assert health.tool_count == 3
    assert health.missing_required_tools == set()
    assert client.has_tool("get_units") is True
    assert "ping" in session.events

    client.stop()


def test_health_check_reports_unstarted_client_without_spawning(install_path: Path) -> None:
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))

    default_health = client.health_check()
    health = client.health_check(required_tools={"get_game_overview"})

    assert default_health.ok is False
    assert default_health.missing_required_tools == {"get_game_overview", "end_turn"}
    assert health.ok is False
    assert health.started is False
    assert health.initialized is False
    assert health.missing_required_tools == {"get_game_overview"}
    assert "not started" in health.message


def test_health_check_wraps_ping_failure(fake_mcp_sdk, install_path: Path, monkeypatch) -> None:
    session, _stdio = fake_mcp_sdk

    async def fail_ping(self) -> None:  # noqa: ANN001
        raise RuntimeError("transport closed")

    monkeypatch.setattr(session, "ping", fail_ping)
    client = Civ6McpClient(Civ6McpConfig(install_path=install_path, launcher="python"))
    client.start()

    with pytest.raises(Civ6McpError, match="health check failed"):
        client.health_check()

    client.stop()
