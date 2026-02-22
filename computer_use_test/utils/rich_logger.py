"""
RichLogger — Structured, colorful terminal output for agent execution.

Singleton that replaces verbose logger.info() calls in turn_executor.py
with Rich-based tables and panels for at-a-glance monitoring.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class RichLogger:
    """Singleton for structured Rich terminal output."""

    _instance: RichLogger | None = None

    @classmethod
    def get(cls) -> RichLogger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.console = Console()

    def turn_header(self, turn: int, total: int) -> None:
        """Display turn start header as a Rich Panel."""
        self.console.print(
            Panel(
                f"[bold white]TURN {turn} / {total}[/bold white]",
                style="cyan",
                expand=False,
            )
        )

    def route_result(
        self,
        primitive: str,
        reasoning: str,
        game_turn: int | None,
        macro_turn: int,
        micro_turn: int,
    ) -> None:
        """Display routing result as a keyed table."""
        table = Table(title="Route", title_style="bold yellow", show_header=False, expand=False)
        table.add_column("Key", style="bold", width=14)
        table.add_column("Value")

        table.add_row("Primitive", f"[green]{primitive}[/green]")
        table.add_row("Game Turn", str(game_turn) if game_turn is not None else "-")
        table.add_row("Turns", f"macro={macro_turn}  micro={micro_turn}")
        table.add_row("Reasoning", reasoning if reasoning else "-")

        self.console.print(table)

    def action_result(
        self,
        action_type: str,
        coords: tuple[int, int] | None,
        reasoning: str,
        extra: dict | None = None,
    ) -> None:
        """Display VLM action result as a keyed table."""
        table = Table(title="Action", title_style="bold blue", show_header=False, expand=False)
        table.add_column("Key", style="bold", width=14)
        table.add_column("Value")

        table.add_row("Type", f"[cyan]{action_type}[/cyan]")
        if coords:
            table.add_row("Coords", f"({coords[0]}, {coords[1]})")
        table.add_row("Reasoning", reasoning if reasoning else "-")

        if extra:
            for k, v in extra.items():
                table.add_row(k, str(v))

        self.console.print(table)

    def execution_status(self, success: bool, message: str = "") -> None:
        """Display execution success/failure indicator."""
        if success:
            self.console.print(Text.from_markup("[green bold]✓[/green bold] Action executed"))
        else:
            self.console.print(Text.from_markup(f"[red bold]✗[/red bold] Execution failed: {message}"))

    def strategy_update(self, victory_goal: str, summary: str) -> None:
        """Display strategy change notification."""
        self.console.print(
            Panel(
                f"[bold]{victory_goal}[/bold]\n{summary}",
                title="Strategy Update",
                style="magenta",
                expand=False,
            )
        )

    def hitl_event(self, event_type: str, detail: str = "") -> None:
        """Display HITL events (PAUSE, RESUME, STOP, etc.)."""
        color = {"STOP": "red", "PAUSE": "yellow", "RESUME": "green"}.get(event_type, "white")
        msg = f"[{color} bold][HITL] {event_type}[/{color} bold]"
        if detail:
            msg += f"  {detail}"
        self.console.print(msg)

    def turn_summary(self, turn: int, primitive: str, action: str, success: bool) -> None:
        """Display one-line turn completion summary."""
        status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
        self.console.print(f"  Turn {turn} | {primitive} | {action} | {status}")

    def multi_turn_progress(self, current: int, total: int, primitive: str) -> None:
        """Display multi-turn progress indicator."""
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        self.console.print(f"  [{bar}] {current}/{total}  {primitive}")
