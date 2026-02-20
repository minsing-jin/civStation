"""
HITL (Human-in-the-Loop) Module.

Provides human-in-the-loop capabilities for the Civilization VI agent:
- Input providers (voice, text, chatapp)
- Command queue and queue listener
- Agent gate (lifecycle state machine)
- Turn checkpoints
- Status UI (real-time web dashboard)
- Relay client (remote HITL via external relay server)

Example:
    from computer_use_test.agent.modules.hitl import HITLInputManager, InputMode

    # Text input (default)
    manager = HITLInputManager(input_mode="text")
    user_input = manager.get_input("Enter your strategy: ")

    # Voice input with Whisper STT
    manager = HITLInputManager(input_mode="voice", stt_provider="whisper")
    user_input = manager.get_input()

    # Auto mode (voice with text fallback)
    manager = HITLInputManager(input_mode="auto")
    user_input = manager.get_input()
"""

from computer_use_test.agent.modules.hitl.agent_gate import AgentGate, AgentState
from computer_use_test.agent.modules.hitl.base_input import BaseInputProvider, InputMode
from computer_use_test.agent.modules.hitl.chatapp_input import ChatAppInputProvider
from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from computer_use_test.agent.modules.hitl.input_manager import HITLInputManager
from computer_use_test.agent.modules.hitl.queue_listener import QueueListener
from computer_use_test.agent.modules.hitl.relay.relay_client import RelayClient
from computer_use_test.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge, AgentStatus
from computer_use_test.agent.modules.hitl.status_ui.websocket_manager import WebSocketManager
from computer_use_test.agent.modules.hitl.text_input import TextInputProvider
from computer_use_test.agent.modules.hitl.turn_checkpoint import (
    CheckpointDecision,
    InterruptMonitor,
    TurnCheckpoint,
    TurnSummary,
)
from computer_use_test.agent.modules.hitl.voice_input import VoiceInputProvider

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
    "TextInputProvider",
    "TurnCheckpoint",
    "TurnSummary",
    "VoiceInputProvider",
    "WebSocketManager",
]
