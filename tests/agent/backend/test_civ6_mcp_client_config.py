"""Configuration-only tests for Civ6McpConfig (no subprocess spawning)."""

from __future__ import annotations

from pathlib import Path

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import (
    Civ6McpConfig,
    Civ6McpUnavailableError,
)


def test_from_environment_uses_explicit_path(tmp_path: Path) -> None:
    config = Civ6McpConfig.from_environment(install_path=tmp_path)
    assert config.install_path == tmp_path.resolve()
    assert config.launcher == "uv"


def test_from_environment_reads_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CIV6_MCP_PATH", str(tmp_path))
    monkeypatch.setenv("CIV6_MCP_LAUNCHER", "python")
    config = Civ6McpConfig.from_environment()
    assert config.install_path == tmp_path.resolve()
    assert config.launcher == "python"


def test_validate_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    config = Civ6McpConfig(install_path=missing)
    with pytest.raises(Civ6McpUnavailableError):
        config.validate()


def test_validate_rejects_directory_without_pyproject(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path)
    with pytest.raises(Civ6McpUnavailableError):
        config.validate()


def test_validate_passes_when_pyproject_exists(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    config = Civ6McpConfig(install_path=tmp_path, launcher="python")
    # python launcher does not require uv on PATH
    config.validate()


def test_server_command_uv_form(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path, launcher="uv")
    cmd = config.server_command()
    assert cmd[0] == "uv"
    assert "run" in cmd
    assert "--directory" in cmd
    assert str(tmp_path) in cmd
    assert "civ-mcp" in cmd


def test_server_command_python_form(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path, launcher="python")
    cmd = config.server_command()
    assert cmd == ["python", "-m", "civ_mcp"]


def test_server_command_unknown_launcher_raises(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path, launcher="docker")
    with pytest.raises(Civ6McpUnavailableError):
        config.server_command()
