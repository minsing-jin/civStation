"""
Turn Checkpoint Module for Async HITL (Human-in-the-Loop) Interrupts.

Provides:
- TurnSummary: Dataclass summarizing one turn's result
- CheckpointDecision: Enum for user decisions at checkpoints (continue/change strategy/stop)
- TurnCheckpoint: Displays turn summary and prompts for c/s/q decision
- InterruptMonitor: Daemon thread monitoring stdin for Enter key to trigger pause
"""

from __future__ import annotations

import logging
import select
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

"""
Turn Checkpoint Module for Async HITL (Human-in-the-Loop) Interrupts.
Turn Checkpoint allows the agent to pause after each turn, display a summary of what happened,
and prompt the user for input on whether to continue, change strategy, or stop.

Turn Summary captures the details of the turn execution, including success/failure, reasoning, and any errors.
"""


@dataclass
class TurnSummary:
    """Summary of a single turn execution."""

    turn_number: int
    primitive: str
    action_type: str
    success: bool
    reasoning: str = ""
    error_message: str = ""
    coords: tuple[int, int] = field(default_factory=lambda: (0, 0))

    def display(self) -> str:
        """Format the turn summary for display."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"  Turn {self.turn_number}: [{status}]",
            f"  Primitive: {self.primitive}",
            f"  Action: {self.action_type} at ({self.coords[0]}, {self.coords[1]})",
        ]
        if self.reasoning:
            lines.append(f"  Reasoning: {self.reasoning}")
        if self.error_message:
            lines.append(f"  Error: {self.error_message}")
        return "\n".join(lines)


class CheckpointDecision(str, Enum):
    """User decisions available at a checkpoint."""

    CONTINUE = "continue"
    CHANGE_STRATEGY = "change_strategy"
    STOP = "stop"


class TurnCheckpoint:
    """
    Displays turn summary and prompts user for a decision at a checkpoint.

    Uses HITLInputManager for getting user input (supports text/voice/auto).
    """

    def __init__(self) -> None:
        self._text_prompt = "[c] 계속 / [s] 전략 변경 / [q] 중단: "

    def prompt(self, summary: TurnSummary) -> CheckpointDecision:
        """
        Display the turn summary and prompt the user for a decision.

        Args:
            summary: The turn summary to display

        Returns:
            CheckpointDecision indicating what to do next
        """
        print(f"\n{'=' * 50}")
        print("CHECKPOINT - 일시정지됨 (Enter 감지)")
        print("=" * 50)
        print(summary.display())
        print("-" * 50)

        while True:
            try:
                choice = input(self._text_prompt).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return CheckpointDecision.STOP

            if choice in ("c", ""):
                return CheckpointDecision.CONTINUE
            elif choice == "s":
                return CheckpointDecision.CHANGE_STRATEGY
            elif choice == "q":
                return CheckpointDecision.STOP
            else:
                print("  잘못된 입력입니다. c, s, q 중 하나를 입력하세요.")

    def prompt_new_strategy(self) -> str:
        """
        Prompt the user for a new strategy string.

        Returns:
            New strategy string entered by the user
        """
        print("-" * 50)
        try:
            new_strategy = input("새 전략을 입력하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""
        return new_strategy


class InterruptMonitor:
    """
    Monitors stdin for Enter key press in a daemon thread.

    When Enter is detected, sets an internal threading.Event so the main
    loop can check `is_interrupted()` between turns.

    Usage:
        monitor = InterruptMonitor()
        monitor.start()
        # ... in your loop ...
        if monitor.is_interrupted():
            # pause and interact
            monitor.reset()
        monitor.stop()
    """

    def __init__(self) -> None:
        self._interrupt_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the interrupt monitor daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        # Only monitor if stdin is a real terminal (not pipe/redirect)
        if not sys.stdin.isatty():
            logger.warning("InterruptMonitor: stdin is not a TTY, skipping monitor")
            return

        self._stop_event.clear()
        self._interrupt_event.clear()

        # Drain any pending input before starting monitor
        try:
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
        except (OSError, ValueError):
            pass

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("InterruptMonitor started (press Enter to pause)")
        print("Async HITL: 실행 중 Enter를 누르면 다음 턴 후 일시정지합니다.")

    def stop(self) -> None:
        """Stop the interrupt monitor."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("InterruptMonitor stopped")

    def is_interrupted(self) -> bool:
        """Check if an interrupt (Enter key) has been detected."""
        return self._interrupt_event.is_set()

    def reset(self) -> None:
        """Reset the interrupt flag after handling."""
        self._interrupt_event.clear()
        # Drain any remaining input from stdin
        try:
            while select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
        except (OSError, ValueError):
            pass

    def _monitor_loop(self) -> None:
        """Background loop that watches stdin for Enter key presses."""
        while not self._stop_event.is_set():
            try:
                # Wait up to 0.5s for stdin to become readable
                readable, _, _ = select.select([sys.stdin], [], [], 0.5)
                if readable:
                    line = sys.stdin.readline()
                    if line == "":
                        # EOF — stdin closed, stop monitoring
                        break
                    self._interrupt_event.set()
                    logger.info("Interrupt detected (Enter pressed)")
                    # Wait until reset before listening again
                    while not self._stop_event.is_set() and self._interrupt_event.is_set():
                        self._stop_event.wait(timeout=0.5)
            except (OSError, ValueError):
                # stdin closed or not selectable — stop monitoring
                break
