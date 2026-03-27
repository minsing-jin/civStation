from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - fallback when rich is unavailable
    Console = None
    Panel = None
    Table = None

REPO_SLUG = "minsing-jin/civStation"
REPO_URL = f"https://github.com/{REPO_SLUG}"
STAR_PROMPT_MARKER = "star_prompt_seen"


def _console() -> Console | None:
    if Console is None:  # pragma: no cover - exercised via plain fallback
        return None
    return Console(highlight=False, soft_wrap=True)


def _plain(text: str) -> None:
    print(dedent(text).strip())


def _cli_state_dir() -> Path:
    override = os.environ.get("CIVSTATION_CLI_STATE_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "civstation"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "civstation"
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / "civstation"
    return Path.home() / ".local" / "state" / "civstation"


def _star_prompt_marker_path() -> Path:
    return _cli_state_dir() / STAR_PROMPT_MARKER


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _read_star_prompt_state() -> str | None:
    marker = _star_prompt_marker_path()
    try:
        if not marker.exists():
            return None
        state = marker.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    return state or None


def _has_completed_star_prompt() -> bool:
    return _read_star_prompt_state() == "done"


def _write_star_prompt_state(state: str) -> None:
    marker = _star_prompt_marker_path()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{state}\n", encoding="utf-8")
    except OSError:
        pass


def _mark_star_completed() -> None:
    _write_star_prompt_state("done")


def _print_star_section() -> None:
    console = _console()
    if console is None:
        _plain(
            f"""
            Support CivStation

            If CivStation helps you, a GitHub star really helps.

            Star directly from CLI:
              civstation star
              civstation star --yes
              gh api -X PUT user/starred/{REPO_SLUG}

            If `gh` needs auth:
              gh auth login
            """
        )
        return

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Thanks", "If CivStation helps you, a GitHub star really helps.")
    table.add_row("Action", "civstation star")
    table.add_row("Fast path", "civstation star --yes")
    table.add_row("Raw", f"gh api -X PUT user/starred/{REPO_SLUG}")
    table.add_row("Auth", "gh auth login")

    console.print(Panel(table, title="Support CivStation", border_style="yellow"))


def _print_star_prompt_banner() -> None:
    console = _console()
    if console is None:
        _plain(
            f"""
            Support CivStation

            Would you like to star CivStation from this terminal now?
            Press Enter to star now. Type `no` to skip. No browser opens.
            If you star it, thank you. That genuinely helps the project.

            If you choose yes, CivStation runs:
              gh api -X PUT user/starred/{REPO_SLUG}
            """
        )
        return

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Prompt", "Would you like to star CivStation from this terminal now?")
    table.add_row("Input", "Press Enter to star now. Type `no` to skip. No browser opens.")
    table.add_row("Thanks", "If you star it, thank you. That genuinely helps the project.")
    table.add_row("Runs", f"gh api -X PUT user/starred/{REPO_SLUG}")

    console.print(Panel(table, title="Support CivStation", border_style="yellow"))


def _prompt_yes_no(prompt: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{prompt} {suffix}: ").strip()
    except EOFError:
        return default
    if not response:
        return default
    return response.casefold() not in {"n", "no"}


def _star_repo_via_gh() -> int:
    if shutil.which("gh") is None:
        _plain(
            """
            GitHub CLI `gh` is not installed.
            Install it, run `gh auth login`, then use `civstation star` again.
            """
        )
        return 1

    _plain("Trying to star CivStation via GitHub CLI...")
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "PUT",
                "-H",
                "Accept: application/vnd.github+json",
                f"user/starred/{REPO_SLUG}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        _plain(
            """
            GitHub CLI is taking too long.
            Run `gh auth login` if needed, then try `civstation star` again.
            """
        )
        return 1
    output = (result.stdout or result.stderr).strip()

    if result.returncode == 0:
        _mark_star_completed()
        if output:
            _plain(output)
        _plain("Thanks for starring CivStation. That genuinely helps.")
        return 0

    lowered = output.lower()
    if "already" in lowered and "star" in lowered:
        _mark_star_completed()
        _plain(output)
        _plain("Thanks. CivStation is already starred on this GitHub account.")
        return 0

    _plain(
        output
        or f"`gh api -X PUT user/starred/{REPO_SLUG}` failed. Run `gh auth login` and try `civstation star` again."
    )
    return 1


def _prompt_for_star() -> int:
    _print_star_prompt_banner()
    if not _prompt_yes_no("Star CivStation now?"):
        _plain("No problem. If CivStation helps you later, I'll ask again next time.")
        return 0
    return _star_repo_via_gh()


def _maybe_prompt_for_star() -> None:
    if not _is_interactive():
        return
    if _has_completed_star_prompt():
        return
    _prompt_for_star()


def _print_preflight_checklist() -> None:
    console = _console()
    if console is None:
        _plain(
            """
            Run Setup

            1. Keep Civilization VI visible on your main monitor.
            2. Click back into the actual Civ6 game so the captured screen is the gameplay UI, not another app.
            3. Do not cover the game window with the local dashboard or terminal.
            4. Use your phone or a secondary device for the controller when possible.
            5. On macOS, grant Screen Recording and Accessibility to your terminal or Python app.
            6. Prefer windowed or borderless mode and keep the game resolution stable during the run.
            7. If you use the mobile controller, start the relay host,
               scan the QR code, then press Start from the phone.
            8. For live gameplay, prefer the host machine's local uv/venv environment. Docker is not recommended.

            Live Run
              uv run civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start
            """
        )
        return

    checklist = Table.grid(padding=(0, 1))
    checklist.add_column(style="bold cyan", no_wrap=True)
    checklist.add_column()
    checklist.add_row("1.", "Keep Civilization VI visible on your main monitor.")
    checklist.add_row("2.", "Click back into the actual Civ6 game so the captured screen is the gameplay UI.")
    checklist.add_row("3.", "Do not cover the game window with the local dashboard or terminal.")
    checklist.add_row("4.", "Use your phone or a secondary device for the controller when possible.")
    checklist.add_row("5.", "Grant Screen Recording and Accessibility to your terminal or Python app on macOS.")
    checklist.add_row("6.", "Prefer windowed or borderless mode and keep the game resolution stable.")
    checklist.add_row("7.", "For mobile controller flow, start the relay host, scan the QR code, then press Start.")
    checklist.add_row("8.", "Prefer the host machine's local uv/venv environment. Docker is not recommended.")

    console.print(Panel(checklist, title="Run Setup", border_style="cyan"))
    console.print(
        Panel(
            "uv run civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start",
            title="Live Run",
            border_style="green",
        )
    )


def _print_onboarding() -> None:
    console = _console()
    if console is None:
        _plain(
            f"""
            CivStation CLI

            Better than `python -m ...`:
              uv run civstation
              uv run civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start

            Installed command:
              civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start

            Git clone flow:
              git clone {REPO_URL}.git
              cd civStation
              uv sync
              uv run civstation

            Main commands:
              civstation                Show this onboarding guide
              civstation run ...        Start the Civ6 agent with a preflight checklist
              civstation guide          Print the setup and operator checklist
              civstation star           Show the fastest GitHub star actions
              civstation mcp ...        Run the layered MCP server
              civstation mcp-install ...  Render or write MCP client templates
            """
        )
        print()
        _print_preflight_checklist()
        return

    overview = Table.grid(padding=(0, 1))
    overview.add_column(style="bold cyan", no_wrap=True)
    overview.add_column()
    overview.add_row("Recommended", "uv run civstation")
    overview.add_row(
        "Quick run",
        "uv run civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start",
    )
    overview.add_row(
        "Installed",
        "civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start",
    )
    overview.add_row("Clone", f"git clone {REPO_URL}.git")
    overview.add_row("Setup", "cd civStation && uv sync && uv run civstation")
    overview.add_row("Support", "civstation star")
    overview.add_row("Operator UX", "Keep Civ6 on the main monitor and use a phone or second device for control.")
    overview.add_row("MCP install", "uv run civstation mcp-install --client codex --write")

    console.print(Panel(overview, title="CivStation CLI", border_style="magenta"))
    _print_preflight_checklist()


def _print_root_help() -> None:
    _plain(
        """
        Usage:
          civstation
          civstation guide
          civstation star
          civstation run [--guide-only] [--skip-guide] [turn_runner args...]
          civstation mcp [mcp server args...]
          civstation mcp-install [install args...]

        Tip:
          If you pass runner flags directly, they are treated like `civstation run ...`.
        """
    )


def _run_turn_runner(argv: list[str] | None = None) -> None:
    from civStation.agent.turn_runner import main as turn_runner_main

    turn_runner_main(argv)


def _run_mcp_server(argv: list[str] | None = None) -> None:
    from civStation.mcp.server import main as mcp_main

    mcp_main(argv)


def _run_mcp_install(argv: list[str] | None = None) -> None:
    from civStation.mcp.install_client_assets import main as install_main

    install_main(argv)


def _handle_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="civstation run",
        description="Run the CivStation agent with a preflight operator guide.",
    )
    parser.add_argument("--guide-only", action="store_true", help="Print the preflight guide and exit.")
    parser.add_argument("--skip-guide", action="store_true", help="Skip the preflight guide before launching.")
    known, passthrough = parser.parse_known_args(argv)

    if not known.skip_guide:
        _maybe_prompt_for_star()
        _print_preflight_checklist()

    if known.guide_only:
        return 0

    _run_turn_runner(passthrough)
    return 0


def _handle_mcp(argv: list[str]) -> int:
    _run_mcp_server(argv)
    return 0


def _handle_mcp_install(argv: list[str]) -> int:
    _run_mcp_install(argv)
    return 0


def _handle_star(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="civstation star",
        description="Star the CivStation GitHub repo directly from the CLI.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Run the star command without prompting.")
    known = parser.parse_args(argv)

    if known.yes:
        return _star_repo_via_gh()

    if not _is_interactive():
        _print_star_section()
        return 0

    return _prompt_for_star()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
        _maybe_prompt_for_star()
        _print_onboarding()
        return 0

    command = argv[0]
    rest = argv[1:]

    if command in {"-h", "--help", "help"}:
        _print_root_help()
        print()
        _print_onboarding()
        return 0

    if command == "guide":
        _maybe_prompt_for_star()
        _print_onboarding()
        return 0

    if command == "star":
        return _handle_star(rest)

    if command == "run":
        return _handle_run(rest)

    if command == "mcp":
        return _handle_mcp(rest)

    if command == "mcp-install":
        return _handle_mcp_install(rest)

    return _handle_run(argv)


def app() -> int:
    return main()
