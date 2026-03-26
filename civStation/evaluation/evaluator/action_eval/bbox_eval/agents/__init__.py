"""
Agent runners for bbox-based evaluation.

Provides multiple ways to integrate agents:
    - SubprocessAgentRunner:  External agent via stdin/stdout JSON protocol.
    - BuiltinAgentRunner:     Wraps existing VLM providers (Claude, Gemini, GPT).
    - MockAgentRunner:        Fixed responses for testing.

To implement a custom agent runner, subclass BaseAgentRunner and implement run_case().
"""

from .base import AgentRunnerError, BaseAgentRunner
from .builtin_adapter import BuiltinAgentRunner
from .mock_runner import MockAgentRunner
from .subprocess_runner import SubprocessAgentRunner

__all__ = [
    "AgentRunnerError",
    "BaseAgentRunner",
    "BuiltinAgentRunner",
    "MockAgentRunner",
    "SubprocessAgentRunner",
]
