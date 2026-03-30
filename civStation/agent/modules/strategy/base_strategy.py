"""
Base Strategy Planner - Abstract interface for strategy generation.

Defines the abstract base class that all strategy planners must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civStation.agent.modules.context import ContextManager
    from civStation.agent.modules.strategy.strategy_schemas import StructuredStrategy


class BaseStrategyPlanner(ABC):
    """
    Abstract base class for strategy planners.

    Strategy planners take game context and optionally human input,
    and generate a StructuredStrategy to guide game decisions.
    """

    @abstractmethod
    def generate_strategy(
        self,
        context: "ContextManager",
        human_input: str | None = None,
    ) -> "StructuredStrategy":
        """
        Generate a strategy based on context and optional human input.

        In HITL mode, this should use human_input to refine the strategy.
        In autonomous mode, this should generate strategy from context alone.

        Args:
            context: The ContextManager instance with current game state
            human_input: Optional human-provided strategy guidance

        Returns:
            A StructuredStrategy object

        Raises:
            HITLInputRequiredError: If HITL mode is enabled but no input provided
        """
        pass

    @abstractmethod
    def refine_strategy(
        self,
        raw_input: str,
        context: "ContextManager",
    ) -> "StructuredStrategy":
        """
        Refine raw human input into a structured strategy.

        Takes free-form human text and uses LLM to convert it into
        a proper StructuredStrategy with all fields populated.

        Args:
            raw_input: Free-form human strategy input
            context: The ContextManager instance for context

        Returns:
            A StructuredStrategy object
        """
        pass

    @abstractmethod
    def update_strategy(
        self,
        current_strategy: "StructuredStrategy",
        context: "ContextManager",
        reason: str = "",
    ) -> "StructuredStrategy":
        """
        Update an existing strategy based on new context.

        Called periodically to adjust strategy based on game changes.

        Args:
            current_strategy: The current strategy to update
            context: The ContextManager instance with new game state
            reason: Optional reason for the update

        Returns:
            Updated StructuredStrategy object
        """
        pass
