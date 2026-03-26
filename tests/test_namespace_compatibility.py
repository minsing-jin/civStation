from __future__ import annotations

import subprocess
import sys


def test_legacy_turn_runner_module_path_still_works():
    result = subprocess.run(
        [sys.executable, "-m", "computer_use_test.agent.turn_runner", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Run Civilization VI AI Agent" in result.stdout


def test_legacy_mcp_installer_module_path_still_works():
    result = subprocess.run(
        [sys.executable, "-m", "computer_use_test.mcp.install_client_assets", "--client", "codex"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "computer-use-layered" in result.stdout
