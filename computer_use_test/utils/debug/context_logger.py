"""
Context Logger — Rich-based structured display for context injected into primitive prompts.

Enabled via DebugOptions.log_context=True.
Mirrors the style of RichLogger for consistent terminal output.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_console = Console()


def log_context(
    primitive_name: str,
    strategy_string: str | None,
    context_string: str,
) -> None:
    """
    Display the full context injected into a primitive prompt using Rich formatting.

    Args:
        primitive_name: Name of the primitive being executed.
        strategy_string: High-level strategy text (may be None).
        context_string: Full context string from ContextManager.
    """
    # ── Header panel ──────────────────────────────────────────
    _console.print(
        Panel(
            f"[bold white]CONTEXT DUMP — {primitive_name}[/bold white]",
            style="magenta",
            expand=False,
        )
    )

    # ── Strategy section ──────────────────────────────────────
    strategy_text = strategy_string or "(전략 없음)"
    strategy_panel = Panel(
        Text(strategy_text, overflow="fold"),
        title="[bold yellow]Strategy[/bold yellow]",
        border_style="yellow",
        expand=False,
    )
    _console.print(strategy_panel)

    # ── Context string: parse sections and render as table ────
    _render_context_table(context_string)


def _render_context_table(context_string: str) -> None:
    """Parse context_string into sections and render as a Rich table."""
    lines = context_string.splitlines()

    # Group lines into sections by === ... === headers
    sections: list[tuple[str, list[str]]] = []
    current_title = "Context"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped.strip("= ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    if not sections:
        _console.print(Text("(컨텍스트 없음)", style="dim"))
        return

    for title, body_lines in sections:
        table = Table(
            title=title,
            title_style="bold cyan",
            show_header=False,
            expand=False,
            padding=(0, 1),
        )
        table.add_column("Line", style="dim", no_wrap=False)

        for raw_line in body_lines:
            if not raw_line.strip():
                table.add_row("")
                continue
            # Colour-code emoji-prefixed lines
            if raw_line.lstrip().startswith("⚔️") or raw_line.lstrip().startswith("⚠️"):
                table.add_row(Text(raw_line, style="red"))
            elif raw_line.lstrip().startswith("💡"):
                table.add_row(Text(raw_line, style="green"))
            elif raw_line.lstrip().startswith("🔬"):
                table.add_row(Text(raw_line, style="blue"))
            elif raw_line.lstrip().startswith("📊"):
                table.add_row(Text(raw_line, style="yellow"))
            elif raw_line.lstrip().startswith("🏛️"):
                table.add_row(Text(raw_line, style="magenta"))
            else:
                table.add_row(raw_line)

        _console.print(table)
