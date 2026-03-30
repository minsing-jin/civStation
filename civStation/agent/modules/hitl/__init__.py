"""
HITL (Human-in-the-Loop) Module.

Provides human-in-the-loop capabilities for the Civilization VI agent:
- Input via external chat apps (Discord, WhatsApp, etc.)
- Command queue and queue listener
- Agent gate (lifecycle state machine)
- Turn checkpoints
- Status UI (real-time web dashboard)
- Relay client (remote HITL via external relay server)

Example:
    from civStation.agent.modules.hitl import HITLInputManager, ChatAppInputProvider

    # Discord chatapp input
    manager = HITLInputManager(chatapp_provider=some_chatapp_provider)
    user_input = manager.get_input("Enter your strategy: ")
"""

from civStation.agent.modules.hitl.agent_gate import AgentGate, AgentState
from civStation.agent.modules.hitl.base_input import BaseInputProvider, InputMode
from civStation.agent.modules.hitl.chatapp_input import ChatAppInputProvider
from civStation.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from civStation.agent.modules.hitl.input_manager import HITLInputManager
from civStation.agent.modules.hitl.queue_listener import QueueListener
from civStation.agent.modules.hitl.relay.relay_client import RelayClient
from civStation.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge, AgentStatus
from civStation.agent.modules.hitl.status_ui.websocket_manager import WebSocketManager
from civStation.agent.modules.hitl.turn_checkpoint import (
    CheckpointDecision,
    InterruptMonitor,
    TurnCheckpoint,
    TurnSummary,
)

__all__ = [
    "AgentGate",
    "AgentState",
    "AgentStateBridge",
    "AgentStatus",
    "CommandQueue",
    "Directive",
    "DirectiveType",
    "InputMode",
    "BaseInputProvider",
    "ChatAppInputProvider",
    "CheckpointDecision",
    "HITLInputManager",
    "InterruptMonitor",
    "QueueListener",
    "RelayClient",
    "TurnCheckpoint",
    "TurnSummary",
    "WebSocketManager",
]
