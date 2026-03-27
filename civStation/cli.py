from __future__ import annotations

import argparse
import sys
from textwrap import dedent

REPO_SLUG = "minsing-jin/civStation"
REPO_URL = f"https://github.com/{REPO_SLUG}"


def _print(text: str) -> None:
    print(dedent(text).strip())


def _print_star_section() -> None:
    _print(
        f"""
        If CivStation helps you, a GitHub star really helps.

        Fastest CLI option:
          gh repo star {REPO_SLUG}

        Browser fallback:
          {REPO_URL}
        """
    )


def _print_preflight_checklist() -> None:
    _print(
        """
        Preflight Checklist

        1. Keep Civilization VI visible on your main monitor.
        2. Click back into the actual Civ6 game so the captured screen is the gameplay UI, not another app.
        3. Do not cover the game window with the local dashboard or terminal.
        4. Use your phone or a secondary device for the controller when possible.
        5. On macOS, grant Screen Recording and Accessibility to your terminal or Python app.
        6. Prefer windowed or borderless mode and keep the game resolution stable during the run.
        7. If you use the mobile controller, start the relay host, scan the QR code, then press Start from the phone.

        Recommended quick start:
          uv run civstation run --provider gemini --model gemini-3-flash --turns 100 --status-ui --wait-for-start
        """
    )


def _print_onboarding() -> None:
    _print(
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

        Operator UX:
          - Keep Civ6 visible on the main monitor at all times.
          - Use your phone or a secondary device for live controls when possible.
          - Let the game screen be the thing the agent actually sees and clicks.
        """
    )
    print()
    _print_preflight_checklist()
    print()
    _print_star_section()


def _print_root_help() -> None:
    _print(
        """
        Usage:
          civstation
          civstation guide
          civstation star
          civstation run [--guide-only] [--skip-guide] [turn_runner args...]
          civstation mcp [mcp server args...]

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
        print()
        _print_star_section()
        print()

    if known.guide_only:
        return 0

    _run_turn_runner(passthrough)
    return 0


def _handle_mcp(argv: list[str]) -> int:
    _run_mcp_server(argv)
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

    return _handle_run(argv)


def app() -> int:
    return main()
