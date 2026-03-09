"""
Base agent runner interface for bbox evaluation.

Subclass BaseAgentRunner and implement run_case() to integrate any agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import AgentResponse, DatasetCase


class AgentRunnerError(Exception):
    """Raised when agent execution fails."""


class BaseAgentRunner(ABC):
    """
    Abstract base class for running an agent on evaluation cases.

    To integrate a custom agent:
        1. Subclass BaseAgentRunner.
        2. Implement run_case() -> AgentResponse.
        3. Pass your runner to run_evaluation() or evaluate_case().

    Example:
        >>> class MyAgent(BaseAgentRunner):
        ...     def run_case(self, case):
        ...         # your logic here
        ...         return AgentResponse(actions=[...])
    """

    @abstractmethod
    def run_case(self, case: DatasetCase) -> AgentResponse:
        """
        Run the agent on a single evaluation case.

        Args:
            case: The evaluation case with instruction and screenshot path.

        Returns:
            AgentResponse with predicted actions.

        Raises:
            AgentRunnerError: On timeout, parse failure, or other errors.
        """
