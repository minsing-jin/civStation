from __future__ import annotations

import subprocess
import sys

import pytest

from civStation import cli


def test_root_cli_without_args_shows_onboarding(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0

    captured = capsys.readouterr()

    assert "CivStation CLI" in captured.out
    assert "gh repo star minsing-jin/civStation" in captured.out
    assert "Keep Civilization VI visible on your main monitor" in captured.out
    assert "Use your phone or a secondary device" in captured.out
    assert "uv run civstation run" in captured.out


def test_star_command_shows_fast_actions(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["star"]) == 0

    captured = capsys.readouterr()

    assert "If CivStation helps you, a GitHub star really helps." in captured.out
    assert "gh repo star minsing-jin/civStation" in captured.out
    assert "https://github.com/minsing-jin/civStation" in captured.out


def test_run_command_prints_preflight_and_forwards_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forwarded: list[list[str] | None] = []

    monkeypatch.setattr(cli, "_run_turn_runner", lambda argv=None: forwarded.append(argv))

    assert cli.main(["run", "--provider", "gemini", "--turns", "5"]) == 0

    captured = capsys.readouterr()

    assert "Preflight Checklist" in captured.out
    assert "main monitor" in captured.out
    assert forwarded == [["--provider", "gemini", "--turns", "5"]]


def test_root_cli_direct_args_alias_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forwarded: list[list[str] | None] = []

    monkeypatch.setattr(cli, "_run_turn_runner", lambda argv=None: forwarded.append(argv))

    assert cli.main(["--provider", "gemini", "--turns", "3"]) == 0

    captured = capsys.readouterr()

    assert "Preflight Checklist" in captured.out
    assert forwarded == [["--provider", "gemini", "--turns", "3"]]


def test_run_guide_only_does_not_start_runner(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_runner(argv=None) -> None:
        raise AssertionError("runner should not be called in guide-only mode")

    monkeypatch.setattr(cli, "_run_turn_runner", fail_runner)

    assert cli.main(["run", "--guide-only"]) == 0

    captured = capsys.readouterr()

    assert "Preflight Checklist" in captured.out
    assert "mobile controller" in captured.out.lower()


def test_python_module_entrypoint_shows_guide() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "civStation"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "CivStation CLI" in result.stdout
    assert "uv run civstation run" in result.stdout
