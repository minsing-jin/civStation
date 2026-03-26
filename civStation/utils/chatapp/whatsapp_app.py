"""
WhatsApp chat app implementation (stub).

Placeholder for future WhatsApp Business API integration.
All methods raise NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from civStation.utils.chatapp.base import BaseChatApp, ChatAppConfig, ChatAppPlatform, ChatMessage


@dataclass
class WhatsAppConfig(ChatAppConfig):
    """WhatsApp-specific configuration."""

    phone_number_id: str = ""
    business_account_id: str = ""


class WhatsAppChatApp(BaseChatApp):
    """
    WhatsApp Business API integration (stub).

    All methods raise NotImplementedError. This class serves as a
    placeholder for future WhatsApp integration.
    """

    def __init__(self, config: WhatsAppConfig):
        super().__init__(config)

    def start(self) -> None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")

    def stop(self) -> None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")

    def send_message(self, channel_id: str, content: str) -> None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")

    def send_dm(self, user_id: str, content: str) -> None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")

    def wait_for_message(self, user_id: str | None = None, timeout: float = 300.0) -> ChatMessage | None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")

    def is_connected(self) -> bool:
        return False

    def get_platform(self) -> ChatAppPlatform:
        return ChatAppPlatform.WHATSAPP

    def register_handler(self, handler: Callable[[ChatMessage], None]) -> None:
        raise NotImplementedError("WhatsApp integration is not yet implemented")
