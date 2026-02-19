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
    from computer_use_test.agent.modules.hitl.agent_gate import AgentGate
    from computer_use_test.agent.modules.hitl.command_queue import CommandQueue
    from computer_use_test.agent.modules.relay.relay_client import RelayClient
    from computer_use_test.agent.modules.status_ui.websocket_manager import WebSocketManager


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

    # Agent lifecycle
    agent_state: str = "idle"

    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_state": self.agent_state,
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

    def __init__(
        self,
        context_manager: ContextManager,
        command_queue: CommandQueue,
        ws_manager: WebSocketManager | None = None,
        agent_gate: AgentGate | None = None,
        relay_client: RelayClient | None = None,
    ) -> None:
        self._ctx = context_manager
        self._queue = command_queue
        self._ws_manager = ws_manager
        self._agent_gate = agent_gate
        self._relay_client = relay_client
        self._lock = threading.Lock()

        # Mutable state updated by main loop
        self._current_primitive: str = ""
        self._current_action: str = ""
        self._current_reasoning: str = ""
        self._micro_turn: int = 0
        self._macro_turn: int = 1

    def _broadcast_if_connected(self) -> None:
        """Push current status to all WebSocket clients and the relay server."""
        if self._ws_manager and self._ws_manager.connection_count > 0:
            status = self.get_status()
            data = status.to_dict()
            self._ws_manager.broadcast({"type": "status", "data": data})
            if self._relay_client:
                self._relay_client.send_status(data)
        elif self._relay_client:
            status = self.get_status()
            self._relay_client.send_status(status.to_dict())

    def update_current_action(self, primitive: str, action_desc: str, reasoning: str) -> None:
        """Called by main loop after planning an action."""
        with self._lock:
            self._current_primitive = primitive
            self._current_action = action_desc
            self._current_reasoning = reasoning
        self._broadcast_if_connected()

    def update_micro_turn(self, n: int) -> None:
        """Called by main loop at each micro-turn."""
        with self._lock:
            self._micro_turn = n
        self._broadcast_if_connected()

    def update_macro_turn(self, n: int) -> None:
        """Called when a macro-turn boundary is detected."""
        with self._lock:
            self._macro_turn = n
        self._broadcast_if_connected()

    def broadcast_agent_phase(self, phase: str) -> None:
        """Broadcast agent execution phase (e.g. '추론 중...', '실행 중', '대기 중')."""
        if self._ws_manager:
            self._ws_manager.broadcast({"type": "phase", "phase": phase})

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

            agent_state = self._agent_gate.state.value if self._agent_gate else "unknown"

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
                agent_state=agent_state,
                last_updated=datetime.now().isoformat(),
            )
