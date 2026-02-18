"""
Base interface for chat application integrations.

Defines the abstract base class and data models for all chat app platforms
(Discord, WhatsApp, etc.) used for strategy discussion and HITL input.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ChatAppPlatform(str, Enum):
    """Supported chat application platforms."""

    DISCORD = "discord"
    WHATSAPP = "whatsapp"


@dataclass
class ChatAppConfig:
    """Base configuration for chat app connections."""

    bot_token: str = ""
    allowed_user_ids: list[str] = field(default_factory=list)
    language: str = "ko"
    max_message_length: int = 2000


@dataclass
class ChatMessage:
    """A message from a chat application."""

    content: str
    author_id: str
    author_name: str = ""
    is_bot: bool = False
    is_dm: bool = False
    platform: ChatAppPlatform = ChatAppPlatform.DISCORD
    channel_id: str = ""


class BaseChatApp(ABC):
    """
    Abstract base class for chat application integrations.

    All chat app implementations (Discord, WhatsApp, etc.) must implement
    this interface to provide unified messaging capabilities.
    """

    def __init__(self, config: ChatAppConfig):
        self.config = config

    @abstractmethod
    def start(self) -> None:
        """Start the chat app connection (bot login, webhook setup, etc.)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the chat app connection and clean up resources."""

    @abstractmethod
    def send_message(self, channel_id: str, content: str) -> None:
        """
        Send a message to a specific channel.

        Args:
            channel_id: Target channel identifier
            content: Message text to send
        """

    @abstractmethod
    def send_dm(self, user_id: str, content: str) -> None:
        """
        Send a direct message to a user.

        Args:
            user_id: Target user identifier
            content: Message text to send
        """

    @abstractmethod
    def wait_for_message(self, user_id: str | None = None, timeout: float = 300.0) -> ChatMessage | None:
        """
        Wait for a message from a specific user (or any user).

        Args:
            user_id: If provided, only accept messages from this user
            timeout: Maximum seconds to wait

        Returns:
            ChatMessage if received, None on timeout
        """

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the chat app is currently connected."""

    @abstractmethod
    def get_platform(self) -> ChatAppPlatform:
        """Return the platform type of this chat app."""

    @abstractmethod
    def register_handler(self, handler: Callable[[ChatMessage], None]) -> None:
        """
        Register a message handler callback.

        The handler will be called for every incoming message that passes
        permission checks (allowed_user_ids, etc.).

        Args:
            handler: Callable that receives ChatMessage objects
        """
