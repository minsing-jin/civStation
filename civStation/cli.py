from __future__ import annotations

import argparse
import sys
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


def _console() -> Console | None:
    if Console is None:  # pragma: no cover - exercised via plain fallback
        return None
    return Console(highlight=False, soft_wrap=True)


def _plain(text: str) -> None:
    print(dedent(text).strip())


def _print_star_section() -> None:
    console = _console()
    if console is None:
        _plain(
            f"""
            Support CivStation

            If CivStation helps you, a GitHub star really helps.

            Fastest CLI option:
              gh repo star {REPO_SLUG}

            Browser fallback:
              {REPO_URL}
            """
        )
        return

    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Thanks", "If CivStation helps you, a GitHub star really helps.")
    table.add_row("CLI", f"gh repo star {REPO_SLUG}")
    table.add_row("Browser", REPO_URL)

    console.print(Panel(table, title="Support CivStation", border_style="yellow"))


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
        print()
        _print_star_section()
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
    overview.add_row("Operator UX", "Keep Civ6 on the main monitor and use a phone or second device for control.")
    overview.add_row("MCP install", "uv run civstation mcp-install --client codex --write")

    console.print(Panel(overview, title="CivStation CLI", border_style="magenta"))
    _print_preflight_checklist()
    _print_star_section()


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
        _print_preflight_checklist()
        _print_star_section()

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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv:
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
        _print_onboarding()
        return 0

    if command == "star":
        _print_star_section()
        return 0

    if command == "run":
        return _handle_run(rest)

    if command == "mcp":
        return _handle_mcp(rest)

    if command == "mcp-install":
        return _handle_mcp_install(rest)

    return _handle_run(argv)


def app() -> int:
    return main()
