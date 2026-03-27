from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from civStation import cli


def test_root_cli_without_args_shows_onboarding(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0

    captured = capsys.readouterr()

    assert "CivStation CLI" in captured.out
    assert "Run Setup" in captured.out
    assert "civstation star" in captured.out
    assert "Keep Civilization VI visible on your main monitor" in captured.out
    assert "Use your phone or a secondary device" in captured.out
    assert "uv run civstation run" in captured.out
    assert "Support CivStation" not in captured.out
    assert "gh repo star minsing-jin/civStation" not in captured.out


def test_star_command_shows_cli_only_actions(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["star"]) == 0

    captured = capsys.readouterr()

    assert "Support CivStation" in captured.out
    assert "If CivStation has been helpful, I'd really appreciate a GitHub star." in captured.out
    assert "civstation star --yes" in captured.out
    assert "gh auth login" in captured.out
    assert "https://github.com/minsing-jin/civStation" not in captured.out
    assert "Action " not in captured.out
    assert "gh api -X PUT user/starred/minsing-jin/civStation" not in captured.out


def test_interactive_cli_run_prompts_again_when_user_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CIVSTATION_CLI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    assert cli.main([]) == 0

    first = capsys.readouterr()
    assert "If CivStation has been helpful" in first.out
    assert "would you mind starring it" in first.out
    assert first.out.index("Support CivStation") < first.out.index("CivStation CLI")
    assert "No browser opens." in first.out
    assert "type `no` to skip." in first.out
    assert "Thanks either way." in first.out
    assert "No problem. If CivStation helps you later, I'll ask again next time." in first.out
    assert "Prompt " not in first.out
    assert "Input " not in first.out
    assert "gh api -X PUT user/starred/minsing-jin/civStation" not in first.out

    assert cli.main([]) == 0

    second = capsys.readouterr()
    assert "If CivStation has been helpful" in second.out


def test_legacy_seen_state_does_not_suppress_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CIVSTATION_CLI_STATE_DIR", str(tmp_path))
    marker = cli._star_prompt_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("seen\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    assert cli.main([]) == 0

    captured = capsys.readouterr()
    assert "If CivStation has been helpful" in captured.out


@pytest.mark.parametrize("reply", ["y", "yes", "Yes", ""])
def test_star_command_can_run_from_cli_with_yes_or_enter_and_marks_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    reply: str,
) -> None:
    monkeypatch.setenv("CIVSTATION_CLI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: reply)
    monkeypatch.setattr(cli.shutil, "which", lambda _: "/usr/bin/gh")
    calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli.main(["star"]) == 0

    captured = capsys.readouterr()
    assert "If CivStation has been helpful" in captured.out
    assert "No browser opens." in captured.out
    assert "type `no` to skip." in captured.out
    assert "Trying to star CivStation via GitHub CLI..." in captured.out
    assert "Thanks for starring CivStation. That genuinely helps." in captured.out
    assert "https://github.com/minsing-jin/civStation" not in captured.out
    assert "gh api -X PUT user/starred/minsing-jin/civStation" not in captured.out
    assert calls == [
        ["gh", "api", "-X", "PUT", "-H", "Accept: application/vnd.github+json", "user/starred/minsing-jin/civStation"]
    ]
    assert cli._star_prompt_marker_path().read_text(encoding="utf-8").strip() == "done"

    monkeypatch.setattr("builtins.input", lambda _: "no")
    assert cli.main([]) == 0
    after = capsys.readouterr()
    assert "Would you like to star CivStation from this terminal now?" not in after.out


def test_run_command_prints_preflight_and_forwards_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forwarded: list[list[str] | None] = []

    monkeypatch.setattr(cli, "_run_turn_runner", lambda argv=None: forwarded.append(argv))

    assert cli.main(["run", "--provider", "gemini", "--turns", "5"]) == 0

    captured = capsys.readouterr()

    assert "Run Setup" in captured.out
    assert "main monitor" in captured.out
    assert forwarded == [["--provider", "gemini", "--turns", "5"]]


def test_run_guide_only_shows_star_prompt_before_preflight_when_interactive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CIVSTATION_CLI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    assert cli.main(["run", "--guide-only"]) == 0

    captured = capsys.readouterr()
    assert "If CivStation has been helpful" in captured.out
    assert "Run Setup" in captured.out
    assert captured.out.index("Support CivStation") < captured.out.index("Run Setup")


def test_root_cli_direct_args_alias_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forwarded: list[list[str] | None] = []

    monkeypatch.setattr(cli, "_run_turn_runner", lambda argv=None: forwarded.append(argv))

    assert cli.main(["--provider", "gemini", "--turns", "3"]) == 0

    captured = capsys.readouterr()

    assert "Run Setup" in captured.out
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

    assert "Run Setup" in captured.out
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


def test_mcp_install_command_forwards_args(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded: list[list[str] | None] = []

    monkeypatch.setattr(cli, "_run_mcp_install", lambda argv=None: forwarded.append(argv))

    assert cli.main(["mcp-install", "--client", "codex", "--write"]) == 0
    assert forwarded == [["--client", "codex", "--write"]]


def test_help_mentions_mcp_install(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0

    captured = capsys.readouterr()
    assert "civstation mcp-install" in captured.out
