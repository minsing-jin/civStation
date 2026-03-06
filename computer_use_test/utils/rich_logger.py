"""
RichLogger — Real-time split-view dashboard for agent execution.

Singleton that provides a Rich Live dashboard showing:
- Left panel: Main pipeline state (phase, primitive, action, per-phase timing)
- Right panel: Background workers (ContextUpdater, StrategyUpdater, TurnDetector)
- Bottom panel: Recent log lines

Each pipeline phase and background worker call is timed.
Falls back to plain console output when Live is not active.
"""

from __future__ import annotations

import time
from collections import deque

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_ms(seconds: float | None) -> str:
    """Format seconds as human-readable duration string."""
    if seconds is None:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


class RichLogger:
    """Singleton for structured Rich terminal output with live dashboard."""

    _instance: RichLogger | None = None

    @classmethod
    def get(cls) -> RichLogger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.console = Console()
        self._live: Live | None = None

        # Main pipeline state
        self._main_state: dict = {
            "phase": "idle",
            "primitive": "-",
            "reasoning": "-",
            "action": "-",
            "action_reasoning": "-",
            "status": "-",
            "turn_info": "-",
        }

        # Per-phase timing (main pipeline)
        self._phase_timings: dict[str, float | None] = {
            "routing": None,
            "planning": None,
            "executing": None,
        }
        self._phase_start: float | None = None
        self._turn_start: float | None = None
        self._turn_total: float | None = None

        # Background worker state
        self._bg_state: dict = {
            "context": {
                "turn": None,
                "era": None,
                "threats": [],
                "opportunities": [],
                "timestamp": None,
                "duration": None,
            },
            "strategy": {
                "goal": "-",
                "phase": "-",
                "trigger": "-",
                "text": "",
                "directives": {},
                "timestamp": None,
                "duration": None,
            },
            "turn_detector": {
                "turn": None,
                "timestamp": None,
                "duration": None,
            },
        }

        # Recent log lines (max 10)
        self._log_lines: deque[str] = deque(maxlen=10)

    # ------------------------------------------------------------------
    # Live dashboard management
    # ------------------------------------------------------------------

    def start_live(self) -> None:
        """Start the Live dashboard display."""
        if self._live is not None:
            return
        self._live = Live(
            self._build_dashboard(),
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

    def stop_live(self) -> None:
        """Stop the Live dashboard display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _refresh(self) -> None:
        """Refresh the Live display with current state."""
        if self._live is not None:
            self._live.update(self._build_dashboard())

    def _build_dashboard(self) -> Layout:
        """Build the split-view dashboard layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="top", ratio=4),
            Layout(name="log", ratio=1, minimum_size=5),
        )
        layout["top"].split_row(
            Layout(name="main", ratio=1),
            Layout(name="bg", ratio=1),
        )

        layout["main"].update(self._build_main_panel())
        layout["bg"].update(self._build_bg_panel())

        # --- Recent Log Panel ---
        log_text = "\n".join(self._log_lines) if self._log_lines else "[dim]No logs yet[/dim]"
        layout["log"].update(Panel(log_text, title="[bold]Recent Log[/bold]", border_style="bright_black"))

        return layout

    def _build_main_panel(self) -> Panel:
        """Build the Main Pipeline panel with phase timing."""
        t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        t.add_column("K", style="bold bright_cyan", width=10)
        t.add_column("V", ratio=1)

        # Phase with animated indicator
        phase = self._main_state["phase"]
        phase_style = {
            "routing": "[bold yellow]",
            "planning": "[bold blue]",
            "executing": "[bold green]",
            "idle": "[dim]",
        }
        phase_icon = {"routing": "⟳ ", "planning": "◆ ", "executing": "▶ ", "idle": "● "}.get(phase, "")
        style = phase_style.get(phase, "")
        t.add_row("Phase", f"{style}{phase_icon}{phase.upper()}[/]")

        t.add_row("Primitive", f"[bold green]{self._main_state['primitive']}[/]")
        t.add_row("Reasoning", f"[white]{self._main_state['reasoning'][:55]}[/]")
        t.add_row("", "")  # spacer

        t.add_row("Action", f"[bold cyan]{self._main_state['action']}[/]")
        t.add_row("Reason", f"[white]{self._main_state['action_reasoning'][:55]}[/]")
        t.add_row("Status", self._main_state["status"])
        t.add_row("Turn", f"[bold]{self._main_state['turn_info']}[/]")

        # Timing bar
        t.add_row("", "")
        t.add_row("[bold]Timing[/]", self._build_timing_bar())

        return Panel(t, title="[bold cyan]◼ Main Pipeline[/]", border_style="cyan", padding=(0, 1))

    def _build_timing_bar(self) -> str:
        """Build a compact timing summary for main pipeline phases."""
        parts = []
        labels = [("R", "routing"), ("P", "planning"), ("E", "executing")]
        for label, key in labels:
            val = self._phase_timings.get(key)
            if val is not None:
                parts.append(f"[bold]{label}[/]={_fmt_ms(val)}")
            else:
                parts.append(f"[dim]{label}=-[/dim]")

        total = self._turn_total
        if total is not None:
            parts.append(f"[bold]Total[/]={_fmt_ms(total)}")

        return "  ".join(parts)

    def _build_bg_panel(self) -> Panel:
        """Build the Background Workers panel with full strategy info."""
        t = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        t.add_column("K", style="bold bright_magenta", width=12)
        t.add_column("V", ratio=1)

        # ── ContextUpdater ──
        ctx = self._bg_state["context"]
        ctx_ts = ctx["timestamp"]
        ctx_dur = ctx["duration"]
        if ctx_ts:
            ago = int(time.time() - ctx_ts)
            dur_str = f" ({_fmt_ms(ctx_dur)})" if ctx_dur else ""
            ctx_status = f"[green]✓[/] {ago}s ago{dur_str}"
        else:
            ctx_status = "[dim]waiting…[/]"
        t.add_row("[bold]Context[/]", ctx_status)

        turn_str = str(ctx["turn"]) if ctx["turn"] is not None else "-"
        era_str = ctx["era"] or "-"
        t.add_row("  Turn/Era", f"[bold white]{turn_str}[/] | {era_str}")

        for threat in (ctx["threats"] or [])[:2]:
            t.add_row("  [red]Threat[/]", f"[red]{threat[:50]}[/]")
        for opp in (ctx["opportunities"] or [])[:2]:
            t.add_row("  [green]Opp[/]", f"[green]{opp[:50]}[/]")

        if not ctx["threats"] and not ctx["opportunities"]:
            t.add_row("  Info", "[dim]-[/]")

        t.add_row("", "")  # spacer

        # ── StrategyUpdater ──
        strat = self._bg_state["strategy"]
        strat_ts = strat["timestamp"]
        strat_dur = strat["duration"]
        if strat_ts:
            ago = int(time.time() - strat_ts)
            dur_str = f" ({_fmt_ms(strat_dur)})" if strat_dur else ""
            strat_status = f"[green]✓[/] {ago}s ago{dur_str}"
        else:
            strat_status = "[dim]waiting…[/]"
        t.add_row("[bold]Strategy[/]", strat_status)

        # Goal + Phase
        goal = strat["goal"]
        goal_icon = {
            "science": "🔬",
            "culture": "🎭",
            "domination": "⚔️",
            "religious": "🕌",
            "diplomatic": "🤝",
            "score": "📊",
        }.get(goal.lower() if isinstance(goal, str) else "", "")
        t.add_row("  Goal", f"[bold yellow]{goal_icon} {goal}[/]")
        t.add_row("  Phase", f"{strat['phase']}")
        t.add_row("  Trigger", f"[dim]{strat['trigger']}[/]")

        # Strategy text (truncated)
        text = strat["text"]
        if text:
            # Show first 2 lines of strategy text
            lines = text.strip().split("\n")
            for line in lines[:2]:
                t.add_row("  Text", f"[italic]{line[:55]}[/]")

        # Primitive directives (show all)
        directives = strat.get("directives", {})
        if directives:
            t.add_row("  [bold]Directives[/]", "")
            for prim, directive in list(directives.items())[:6]:
                t.add_row(f"    {prim[:8]}", f"[dim]{directive[:45]}[/]")

        t.add_row("", "")  # spacer

        # ── TurnDetector ──
        td = self._bg_state["turn_detector"]
        td_turn = str(td["turn"]) if td["turn"] is not None else "-"
        td_ts = td["timestamp"]
        td_dur = td["duration"]
        if td_ts:
            ago = int(time.time() - td_ts)
            dur_str = f" ({_fmt_ms(td_dur)})" if td_dur else ""
            td_status = f"[green]✓[/] {ago}s ago{dur_str}"
        else:
            td_status = "[dim]waiting…[/]"
        t.add_row("[bold]TurnDet[/]", td_status)
        t.add_row("  Turn", f"[bold white]{td_turn}[/]")

        return Panel(t, title="[bold magenta]◼ Background Workers[/]", border_style="magenta", padding=(0, 1))

    # ------------------------------------------------------------------
    # Main pipeline update methods
    # ------------------------------------------------------------------

    def update_phase(self, phase: str) -> None:
        """Update the current pipeline phase and track timing.

        Args:
            phase: One of 'routing', 'planning', 'executing', 'idle'
        """
        now = time.monotonic()

        # Close previous phase timing
        prev_phase = self._main_state["phase"]
        if self._phase_start is not None and prev_phase in self._phase_timings:
            self._phase_timings[prev_phase] = now - self._phase_start

        # Start new phase
        if phase == "idle":
            # Turn is done — compute total
            if self._turn_start is not None:
                self._turn_total = now - self._turn_start
            self._phase_start = None
        else:
            if prev_phase == "idle":
                # First non-idle phase → new turn start
                self._turn_start = now
                self._turn_total = None
                # Reset phase timings
                for k in self._phase_timings:
                    self._phase_timings[k] = None
            self._phase_start = now

        self._main_state["phase"] = phase
        self._refresh()

    def turn_header(self, turn: int, total: int) -> None:
        """Display turn start header."""
        self._main_state["turn_info"] = f"{turn}/{total}"
        self._main_state["phase"] = "idle"
        self._main_state["primitive"] = "-"
        self._main_state["reasoning"] = "-"
        self._main_state["action"] = "-"
        self._main_state["action_reasoning"] = "-"
        self._main_state["status"] = "-"
        self._refresh()

        if self._live is None:
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
        """Display routing result."""
        self._main_state["primitive"] = primitive
        self._main_state["reasoning"] = reasoning or "-"
        gt = str(game_turn) if game_turn is not None else "-"
        self._main_state["turn_info"] = f"G:{gt} M:{macro_turn} m:{micro_turn}"
        self._refresh()

        if self._live is None:
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
        """Display VLM action result."""
        coord_str = f"({coords[0]}, {coords[1]})" if coords else ""
        self._main_state["action"] = f"{action_type} {coord_str}"
        self._main_state["action_reasoning"] = reasoning or "-"
        self._refresh()

        if self._live is None:
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
            self._main_state["status"] = "[green bold]✓ Executed[/]"
        else:
            self._main_state["status"] = f"[red bold]✗ Failed: {message}[/]"
        self._refresh()

        if self._live is None:
            if success:
                self.console.print(Text.from_markup("[green bold]✓[/green bold] Action executed"))
            else:
                self.console.print(Text.from_markup(f"[red bold]✗[/red bold] Execution failed: {message}"))

    def strategy_update(self, victory_goal: str, summary: str) -> None:
        """Display strategy change notification (called from turn_executor HITL path)."""
        self._bg_state["strategy"]["goal"] = victory_goal
        self._bg_state["strategy"]["trigger"] = "HITL"
        self._bg_state["strategy"]["timestamp"] = time.time()
        self._refresh()

        if self._live is None:
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
        ts = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{ts}] [HITL] {event_type} {detail}")
        self._refresh()

        if self._live is None:
            self.console.print(msg)

    def turn_summary(self, turn: int, primitive: str, action: str, success: bool) -> None:
        """Display one-line turn completion summary."""
        status_str = "[green]OK[/]" if success else "[red]FAIL[/]"
        ts = time.strftime("%H:%M:%S")
        timing = self._build_timing_bar()
        self._log_lines.append(f"[{ts}] Turn {turn} │ {primitive} │ {action} │ {status_str} │ {timing}")
        self._refresh()

        if self._live is None:
            status = "[green]OK[/green]" if success else "[red]FAIL[/red]"
            self.console.print(f"  Turn {turn} | {primitive} | {action} | {status}")

    def multi_turn_progress(self, current: int, total: int, primitive: str) -> None:
        """Display multi-turn progress indicator."""
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        if self._live is None:
            self.console.print(f"  [{bar}] {current}/{total}  {primitive}")

    # ------------------------------------------------------------------
    # Background worker update methods
    # ------------------------------------------------------------------

    def update_context(
        self,
        turn: int | None = None,
        era: str | None = None,
        threats: list[str] | None = None,
        opportunities: list[str] | None = None,
        duration: float | None = None,
    ) -> None:
        """Update ContextUpdater results in the dashboard."""
        ctx = self._bg_state["context"]
        if turn is not None:
            ctx["turn"] = turn
        if era is not None:
            ctx["era"] = era
        if threats is not None:
            ctx["threats"] = threats
        if opportunities is not None:
            ctx["opportunities"] = opportunities
        if duration is not None:
            ctx["duration"] = duration
        ctx["timestamp"] = time.time()
        self._refresh()

    def update_strategy(
        self,
        goal: str,
        trigger: str,
        phase: str = "-",
        text: str = "",
        directives: dict[str, str] | None = None,
        duration: float | None = None,
    ) -> None:
        """Update StrategyUpdater results in the dashboard.

        Args:
            goal: Victory goal (e.g., "science", "culture")
            trigger: What triggered the update (e.g., "INITIAL", "NEW_GAME_TURN")
            phase: Game phase (e.g., "early_expansion", "mid_development")
            text: Full strategy text
            directives: Primitive-specific directives dict
            duration: VLM call duration in seconds
        """
        strat = self._bg_state["strategy"]
        strat["goal"] = goal
        strat["trigger"] = trigger
        strat["phase"] = phase
        strat["text"] = text
        if directives is not None:
            strat["directives"] = directives
        if duration is not None:
            strat["duration"] = duration
        strat["timestamp"] = time.time()
        self._refresh()

    def update_turn_detector(self, turn: int, duration: float | None = None) -> None:
        """Update TurnDetector results in the dashboard."""
        td = self._bg_state["turn_detector"]
        td["turn"] = turn
        if duration is not None:
            td["duration"] = duration
        td["timestamp"] = time.time()
        self._refresh()
