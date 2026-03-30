"""
Chat App Input Provider for HITL.

Bridges chat application messages to the synchronous BaseInputProvider interface,
allowing chat apps to be used as HITL input sources.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from civStation.agent.modules.hitl.base_input import BaseInputProvider, InputMode

if TYPE_CHECKING:
    from civStation.utils.chatapp.base import BaseChatApp

logger = logging.getLogger(__name__)


class ChatAppInputProvider(BaseInputProvider):
    """
    HITL input provider that bridges chat app messages to synchronous input.

    Sends the prompt to the user via chat app (DM or channel),
    then waits for their response.
    """

    def __init__(
        self,
        chat_app: BaseChatApp,
        user_id: str,
        channel_id: str | None = None,
        timeout: float = 300.0,
    ):
        """
        Initialize chat app input provider.

        Args:
            chat_app: Connected BaseChatApp instance
            user_id: User ID to send prompts to and receive input from
            channel_id: Optional channel ID for channel-based prompts (DM if None)
            timeout: Maximum seconds to wait for user response
        """
        self._chat_app = chat_app
        self._user_id = user_id
        self._channel_id = channel_id
        self._timeout = timeout

    def get_input(self, prompt: str = "") -> str:
        """
        Send prompt via chat app and wait for user response.

        Args:
            prompt: Prompt text to send to the user

        Returns:
            User's response text

        Raises:
            RuntimeError: If chat app is not connected or response times out
        """
        if not self._chat_app.is_connected():
            raise RuntimeError("Chat app is not connected")

        # Send prompt to user
        if prompt:
            try:
                if self._channel_id:
                    self._chat_app.send_message(self._channel_id, prompt)
                else:
                    self._chat_app.send_dm(self._user_id, prompt)
            except Exception as e:
                logger.error(f"Failed to send prompt via chat app: {e}")
                raise RuntimeError(f"Failed to send prompt: {e}") from e

        # Wait for response
        message = self._chat_app.wait_for_message(user_id=self._user_id, timeout=self._timeout)
        if message is None:
            raise RuntimeError(f"Chat app input timed out after {self._timeout}s")

        logger.info(f"Received chat app input from {message.author_name}: {message.content[:100]}...")
        return message.content

    def is_available(self) -> bool:
        """Check if the chat app is connected and available."""
        return self._chat_app.is_connected()

    def get_mode(self) -> InputMode:
        """Return CHATAPP input mode."""
        return InputMode.CHATAPP
