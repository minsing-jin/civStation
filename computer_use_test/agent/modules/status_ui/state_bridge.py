"""
State Bridge — Thread-safe snapshot provider for the Status UI.

Collects state from ContextManager and CommandQueue, provides
a frozen AgentStatus snapshot for the FastAPI endpoint.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context.context_manager import ContextManager
    from computer_use_test.agent.modules.hitl.command_queue import CommandQueue


@dataclass
class AgentStatus:
    """Frozen snapshot of agent state for the dashboard."""

    # Strategy
    current_strategy: str = ""
    victory_goal: str = ""
    game_phase: str = ""

    # Queue
    queued_directives: list[dict[str, Any]] = field(default_factory=list)

    # Current action
    current_primitive: str = ""
    current_action: str = ""
    current_reasoning: str = ""

    # Turn tracking
    game_turn: int = 1
    micro_turn: int = 0
    macro_turn: int = 1

    # History
    recent_actions: list[dict[str, Any]] = field(default_factory=list)

    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_strategy": self.current_strategy,
            "victory_goal": self.victory_goal,
            "game_phase": self.game_phase,
            "queued_directives": self.queued_directives,
            "current_primitive": self.current_primitive,
            "current_action": self.current_action,
            "current_reasoning": self.current_reasoning,
            "game_turn": self.game_turn,
            "micro_turn": self.micro_turn,
            "macro_turn": self.macro_turn,
            "recent_actions": self.recent_actions,
            "last_updated": self.last_updated,
        }


class AgentStateBridge:
    """
    Thread-safe bridge between the main agent loop and the Status UI.

    The main loop writes state updates; the FastAPI thread reads snapshots.
    """

    def __init__(self, context_manager: ContextManager, command_queue: CommandQueue) -> None:
        self._ctx = context_manager
        self._queue = command_queue
        self._lock = threading.Lock()

        # Mutable state updated by main loop
        self._current_primitive: str = ""
        self._current_action: str = ""
        self._current_reasoning: str = ""
        self._micro_turn: int = 0
        self._macro_turn: int = 1

    def update_current_action(self, primitive: str, action_desc: str, reasoning: str) -> None:
        """Called by main loop after planning an action."""
        with self._lock:
            self._current_primitive = primitive
            self._current_action = action_desc
            self._current_reasoning = reasoning

    def update_micro_turn(self, n: int) -> None:
        """Called by main loop at each micro-turn."""
        with self._lock:
            self._micro_turn = n

    def update_macro_turn(self, n: int) -> None:
        """Called when a macro-turn boundary is detected."""
        with self._lock:
            self._macro_turn = n

    def get_status(self) -> AgentStatus:
        """Build a frozen snapshot (called by FastAPI thread)."""
        with self._lock:
            # Strategy from context
            strategy_str = self._ctx.get_strategy_string()
            hlc = self._ctx.high_level_context
            victory_goal = ""
            game_phase = ""
            if hlc.current_strategy:
                victory_goal = getattr(hlc.current_strategy, "victory_goal", "")
                if hasattr(victory_goal, "value"):
                    victory_goal = victory_goal.value
                game_phase = getattr(hlc.current_strategy, "current_phase", "")

            # Queued directives
            pending = self._queue.peek()
            queued = [{"type": d.directive_type.value, "payload": d.payload, "source": d.source} for d in pending]

            # Recent actions from primitive context
            recent = []
            for a in self._ctx.primitive_context.get_last_actions(5):
                recent.append({"type": a.action_type, "primitive": a.primitive, "x": a.x, "y": a.y, "result": a.result})

            return AgentStatus(
                current_strategy=strategy_str,
                victory_goal=str(victory_goal),
                game_phase=str(game_phase),
                queued_directives=queued,
                current_primitive=self._current_primitive,
                current_action=self._current_action,
                current_reasoning=self._current_reasoning,
                game_turn=self._ctx.global_context.current_turn,
                micro_turn=self._micro_turn,
                macro_turn=self._macro_turn,
                recent_actions=recent,
                last_updated=datetime.now().isoformat(),
            )
