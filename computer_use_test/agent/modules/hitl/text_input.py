"""
Text input provider for HITL (Human-in-the-Loop) input.

Provides terminal-based text input functionality.
"""

from computer_use_test.agent.modules.hitl.base_input import BaseInputProvider, InputMode


class TextInputProvider(BaseInputProvider):
    """Terminal-based text input provider."""

    def __init__(self, default_prompt: str = "전략을 입력하세요: "):
        """
        Initialize the text input provider.

        Args:
            default_prompt: Default prompt to display when no prompt is provided
        """
        self.default_prompt = default_prompt

    def get_input(self, prompt: str = "") -> str:
        """
        Get input from the user via terminal.

        Args:
            prompt: Optional prompt to display (uses default_prompt if empty)

        Returns:
            User input as string
        """
        display_prompt = prompt or self.default_prompt
        return input(display_prompt)

    def is_available(self) -> bool:
        """
        Check if text input is available.

        Returns:
            Always True since terminal input is always available
        """
        return True

    def get_mode(self) -> InputMode:
        """
        Return the input mode.

        Returns:
            InputMode.TEXT
        """
        return InputMode.TEXT
