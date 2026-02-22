"""
Base input provider for HITL (Human-in-the-Loop) input.

Defines the abstract base class and InputMode enum for all input providers.
"""

from abc import ABC, abstractmethod
from enum import Enum


class InputMode(str, Enum):
    """Input modes for HITL interaction."""

    CHATAPP = "chatapp"


class BaseInputProvider(ABC):
    """Abstract base class for HITL input providers."""

    @abstractmethod
    def get_input(self, prompt: str = "") -> str:
        """
        Get input from the user.

        Args:
            prompt: Optional prompt to display to the user

        Returns:
            User input as string
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this input method is available.

        Returns:
            True if the input method can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_mode(self) -> InputMode:
        """
        Return the input mode of this provider.

        Returns:
            InputMode enum value
        """
        pass
