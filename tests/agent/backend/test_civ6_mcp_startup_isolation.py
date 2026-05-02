"""Startup isolation tests for the civ6-mcp backend client."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES = (
    "civStation.agent.turn_executor",
    "civStation.agent.modules.context.context_updater",
    "civStation.agent.modules.context.macro_turn_manager",
    "civStation.agent.modules.context.turn_detector",
    "civStation.agent.modules.primitive.multi_step_process",
    "civStation.agent.modules.router.primitive_registry",
    "civStation.utils.image_pipeline",
    "civStation.utils.llm_provider",
    "civStation.utils.screen",
)


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.inputSchema = {"type": "object"}


class _FakeToolsResponse:
    def __init__(self) -> None:
        self.tools = [_FakeTool("get_game_overview"), _FakeTool("end_turn")]


class _FakeStdioServerParameters:
    def __init__(self, *, command: str, args: list[str], env: dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.env = env


class _FakeStdioContext:
    async def __aenter__(self) -> tuple[object, object]:
        return object(), object()

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class _FakeClientSession:
    def __init__(self, read_stream: object, write_stream: object, **kwargs: object) -> None:
        self.read_stream = read_stream
        self.write_stream = write_stream
        self.kwargs = kwargs

    async def __aenter__(self) -> _FakeClientSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> _FakeToolsResponse:
        return _FakeToolsResponse()


@pytest.fixture
def fake_mcp_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    mcp = types.ModuleType("mcp")
    client = types.ModuleType("mcp.client")
    stdio = types.ModuleType("mcp.client.stdio")

    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio)
    monkeypatch.setattr(mcp, "client", client, raising=False)
    monkeypatch.setattr(client, "stdio", stdio, raising=False)
    monkeypatch.setattr(mcp, "ClientSession", _FakeClientSession, raising=False)
    monkeypatch.setattr(mcp, "StdioServerParameters", _FakeStdioServerParameters, raising=False)
    monkeypatch.setattr(stdio, "stdio_client", lambda _params: _FakeStdioContext(), raising=False)


def _unload_backend_and_vlm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    prefixes = ("civStation.agent.modules.backend.civ6_mcp", *_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES)
    for module_name in list(sys.modules):
        if module_name.startswith(prefixes):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


def _loaded_forbidden_modules() -> set[str]:
    return {
        module_name for module_name in sys.modules if module_name.startswith(_FORBIDDEN_VLM_RUNTIME_MODULE_PREFIXES)
    }


def test_civ6_mcp_client_start_does_not_import_vlm_or_computer_use_runtime(
    fake_mcp_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _unload_backend_and_vlm_modules(monkeypatch)
    install_path = tmp_path / "civ6-mcp"
    install_path.mkdir()
    (install_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n", encoding="utf-8")

    client_module = importlib.import_module("civStation.agent.modules.backend.civ6_mcp.client")
    assert _loaded_forbidden_modules() == set()

    monkeypatch.setattr(client_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    config = client_module.Civ6McpConfig(install_path=install_path, launcher="python")
    client = client_module.Civ6McpClient(config)

    client.start()
    try:
        assert client.tool_names == {"get_game_overview", "end_turn"}
        assert _loaded_forbidden_modules() == set()
    finally:
        client.stop()
