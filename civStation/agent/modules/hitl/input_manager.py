"""
HITL Input Manager - Chat-app-based input handling for Human-in-the-Loop.

Input is always routed through an external chat application
(Discord, WhatsApp, etc.).  Local voice/text/auto modes have been removed
because those are now handled inside the third-party apps themselves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from civStation.agent.modules.hitl.base_input import InputMode

if TYPE_CHECKING:
    from civStation.agent.modules.hitl.chatapp_input import ChatAppInputProvider

logger = logging.getLogger(__name__)


class HITLInputManager:
    """
    HITL input manager that delegates to a chat-app provider.

    Raises RuntimeError if get_input() is called without a connected
    chatapp_provider (i.e. --chatapp original with no interactive input).
    """

    def __init__(
        self,
        chatapp_provider: ChatAppInputProvider | None = None,
    ):
        self.input_mode = InputMode.CHATAPP
        self._chatapp_provider = chatapp_provider

        if chatapp_provider is not None:
            logger.info("HITLInputManager initialized: chatapp provider connected")
        else:
            logger.info("HITLInputManager initialized: no chatapp provider (original mode)")

    def get_input(self, prompt: str = "") -> str:
        """
        Get input via the configured chat-app provider.

        Args:
            prompt: Prompt text to send to the user

        Returns:
            User input as string

        Raises:
            RuntimeError: If no chatapp provider is connected
        """
        if self._chatapp_provider and self._chatapp_provider.is_available():
            try:
                return self._chatapp_provider.get_input(prompt)
            except RuntimeError as e:
                logger.warning(f"Chat app input failed: {e}")
                raise

        raise RuntimeError(
            "No chat app provider connected. Set --chatapp discord or --chatapp whatsapp with proper credentials."
        )

    def is_available(self) -> bool:
        """Return True if a chatapp provider is connected."""
        return self._chatapp_provider is not None and self._chatapp_provider.is_available()

    def get_current_mode(self) -> InputMode:
        return self.input_mode

    def get_effective_mode(self) -> str:
        """
        Backward-compatible mode accessor used by QueueListener.

        Returns:
            String mode label (e.g. "chatapp")
        """
        return self.input_mode.value
