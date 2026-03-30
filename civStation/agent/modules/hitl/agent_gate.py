"""
Agent Gate — Lifecycle state machine for receiving external control signals.

Receives start/pause/resume/stop signals from external controllers
(web UI, mobile app, etc.) and manages the agent's execution state.

State transitions:
    IDLE ──start()──→ RUNNING ──pause()──→ PAUSED
                        ↑                     │
                        └──── resume() ───────┘
    Any state ──stop()──→ STOPPED
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civStation.agent.modules.hitl.command_queue import CommandQueue

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent lifecycle states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class AgentGate:
    """
    Thread-safe gate that receives external control signals.

    External controllers (web UI, API) call start/pause/resume/stop.
    The agent loop reads state and blocks on wait_for_start().
    """

    def __init__(self, command_queue: CommandQueue) -> None:
        self._command_queue = command_queue
        self._state = AgentState.IDLE
        self._lock = threading.Lock()
        self._start_event = threading.Event()

    @property
    def state(self) -> AgentState:
        """Current agent state (thread-safe read)."""
        with self._lock:
            return self._state

    def set_state(self, state: AgentState) -> None:
        """
        Directly set state (called by turn_executor to sync).

        Use this when the agent loop processes directives from any source
        (WebSocket, HTTP, stdin) to keep the gate state in sync.
        """
        with self._lock:
            self._state = state

    def start(self) -> bool:
        """
        Signal the agent to begin execution.

        Returns False if already stopped.
        """
        from civStation.agent.modules.hitl.command_queue import Directive, DirectiveType

        with self._lock:
            if self._state == AgentState.STOPPED:
                return False
            self._state = AgentState.RUNNING
        self._start_event.set()

        # Push RESUME so any blocking wait() in turn_executor unblocks
        self._command_queue.push(Directive(directive_type=DirectiveType.RESUME, payload="", source="agent_gate"))
        logger.info("AgentGate: START signal received")
        return True

    def pause(self) -> bool:
        """
        Pause the agent. Returns False if not currently running.
        """
        from civStation.agent.modules.hitl.command_queue import Directive, DirectiveType

        with self._lock:
            if self._state != AgentState.RUNNING:
                return False
            self._state = AgentState.PAUSED

        self._command_queue.push(Directive(directive_type=DirectiveType.PAUSE, payload="", source="agent_gate"))
        logger.info("AgentGate: PAUSE signal received")
        return True

    def resume(self) -> bool:
        """
        Resume the agent from pause. Returns False if not paused.
        """
        from civStation.agent.modules.hitl.command_queue import Directive, DirectiveType

        with self._lock:
            if self._state != AgentState.PAUSED:
                return False
            self._state = AgentState.RUNNING

        self._command_queue.push(Directive(directive_type=DirectiveType.RESUME, payload="", source="agent_gate"))
        logger.info("AgentGate: RESUME signal received")
        return True

    def stop(self) -> bool:
        """
        Stop the agent. Unblocks wait_for_start() if waiting.
        Returns False if already stopped.
        """
        from civStation.agent.modules.hitl.command_queue import Directive, DirectiveType

        with self._lock:
            if self._state == AgentState.STOPPED:
                return False
            self._state = AgentState.STOPPED

        self._command_queue.push(Directive(directive_type=DirectiveType.STOP, payload="", source="agent_gate"))
        # Unblock wait_for_start() if the agent hasn't started yet
        self._start_event.set()
        logger.info("AgentGate: STOP signal received")
        return True

    def wait_for_start(self, timeout: float | None = None) -> bool:
        """
        Block until start() or stop() is called.

        Used by turn_runner when --wait-for-start is set.
        Returns True if start was received, False on timeout.
        """
        logger.info("AgentGate: Waiting for external start signal...")
        return self._start_event.wait(timeout=timeout)

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            return self._state == AgentState.STOPPED
