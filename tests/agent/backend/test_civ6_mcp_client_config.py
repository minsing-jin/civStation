"""Configuration-only tests for Civ6McpConfig (no subprocess spawning)."""

from __future__ import annotations

from pathlib import Path

import pytest

from civStation.agent.modules.backend.civ6_mcp.client import (
    Civ6McpConfig,
    Civ6McpUnavailableError,
)


def _validation_error_message(config: Civ6McpConfig) -> str:
    with pytest.raises(Civ6McpUnavailableError) as exc_info:
        config.validate()
    return str(exc_info.value)


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


def test_config_constructor_requires_install_path() -> None:
    with pytest.raises(TypeError, match="install_path"):
        Civ6McpConfig()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("field_name", "missing_value"),
    [
        ("install_path", None),
        ("install_path", ""),
        ("launcher", None),
        ("launcher", ""),
    ],
)
def test_validate_rejects_missing_required_configuration_fields(
    tmp_path: Path,
    field_name: str,
    missing_value: object,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    kwargs = {"install_path": tmp_path, "launcher": "python"}
    kwargs[field_name] = missing_value

    config = Civ6McpConfig(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(Civ6McpUnavailableError, match=field_name):
        config.validate()


def test_missing_install_path_error_is_actionable() -> None:
    message = _validation_error_message(Civ6McpConfig(install_path=None, launcher="python"))  # type: ignore[arg-type]

    assert "install_path" in message
    assert "--civ6-mcp-path" in message
    assert "CIV6_MCP_PATH" in message
    assert "github.com/lmwilki/civ6-mcp" in message


def test_missing_launcher_error_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")

    message = _validation_error_message(Civ6McpConfig(install_path=tmp_path, launcher=""))  # type: ignore[arg-type]

    assert "launcher" in message
    assert "--civ6-mcp-launcher" in message
    assert "CIV6_MCP_LAUNCHER" in message
    assert "uv" in message
    assert "python" in message


@pytest.mark.parametrize("invalid_path", [123, object()])
def test_validate_rejects_non_path_like_install_path(invalid_path: object) -> None:
    config = Civ6McpConfig(install_path=invalid_path, launcher="python")  # type: ignore[arg-type]
    with pytest.raises(Civ6McpUnavailableError, match="install_path must be a path-like value"):
        config.validate()


@pytest.mark.parametrize("invalid_launcher", [123, object()])
def test_validate_rejects_non_string_launcher(tmp_path: Path, invalid_launcher: object) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    config = Civ6McpConfig(install_path=tmp_path, launcher=invalid_launcher)  # type: ignore[arg-type]
    with pytest.raises(Civ6McpUnavailableError, match="launcher must be a string"):
        config.validate()


def test_validate_rejects_unsupported_launcher_from_environment(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    monkeypatch.setenv("CIV6_MCP_PATH", str(tmp_path))
    monkeypatch.setenv("CIV6_MCP_LAUNCHER", "docker")

    config = Civ6McpConfig.from_environment()
    with pytest.raises(Civ6McpUnavailableError, match="Unsupported launcher"):
        config.validate()


def test_unsupported_launcher_error_lists_supported_values(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")

    message = _validation_error_message(Civ6McpConfig(install_path=tmp_path, launcher="docker"))

    assert "docker" in message
    assert "uv" in message
    assert "python" in message
    assert "--civ6-mcp-launcher" in message
    assert "CIV6_MCP_LAUNCHER" in message


def test_validate_rejects_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    config = Civ6McpConfig(install_path=missing)
    with pytest.raises(Civ6McpUnavailableError):
        config.validate()


def test_missing_directory_error_points_to_path_sources(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    message = _validation_error_message(Civ6McpConfig(install_path=missing, launcher="python"))

    assert str(missing) in message
    assert "--civ6-mcp-path" in message
    assert "CIV6_MCP_PATH" in message
    assert "civ6-mcp checkout" in message


def test_validate_rejects_directory_without_pyproject(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path)
    with pytest.raises(Civ6McpUnavailableError):
        config.validate()


def test_missing_pyproject_error_names_expected_checkout_setup(tmp_path: Path) -> None:
    message = _validation_error_message(Civ6McpConfig(install_path=tmp_path, launcher="python"))

    assert str(tmp_path) in message
    assert "pyproject.toml" in message
    assert "git clone https://github.com/lmwilki/civ6-mcp" in message
    assert "uv sync" in message


def test_missing_uv_binary_error_suggests_python_launcher(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'civ6-mcp'\n")
    monkeypatch.setattr("shutil.which", lambda _name: None)

    message = _validation_error_message(Civ6McpConfig(install_path=tmp_path, launcher="uv"))

    assert "uv" in message
    assert "PATH" in message
    assert "--civ6-mcp-launcher python" in message


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


def test_server_command_rejects_non_string_launcher(tmp_path: Path) -> None:
    config = Civ6McpConfig(install_path=tmp_path, launcher=123)  # type: ignore[arg-type]
    with pytest.raises(Civ6McpUnavailableError, match="launcher must be a string"):
        config.server_command()
