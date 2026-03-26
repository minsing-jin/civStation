"""
Chat Application Integration Module.

Provides a unified interface for chat app integrations (Discord, WhatsApp, etc.)
used for strategy discussion and HITL input in the Civilization VI agent.

Example:
    from civStation.utils.chatapp import create_chat_app, BaseChatApp

    app = create_chat_app("discord", bot_token="...", allowed_user_ids=["123"])
    app.start()
    app.send_message(channel_id, "Hello!")
"""

from civStation.utils.chatapp.base import BaseChatApp, ChatAppConfig, ChatAppPlatform, ChatMessage


def create_chat_app(platform: str | ChatAppPlatform, **kwargs) -> BaseChatApp:
    """
    Factory function to create a chat app instance.

    Args:
        platform: Platform name ("discord", "whatsapp") or ChatAppPlatform enum
        **kwargs: Platform-specific configuration passed to the constructor

    Returns:
        BaseChatApp instance for the specified platform

    Raises:
        ValueError: If the platform is not supported
    """
    if isinstance(platform, str):
        platform = ChatAppPlatform(platform.lower())

    if platform == ChatAppPlatform.DISCORD:
        from civStation.utils.chatapp.discord_app import DiscordChatApp, DiscordConfig

        config = DiscordConfig(
            bot_token=kwargs.get("bot_token", ""),
            allowed_user_ids=kwargs.get("allowed_user_ids", []),
            language=kwargs.get("language", "ko"),
            command_prefix=kwargs.get("command_prefix", "!"),
            allowed_channel_ids=kwargs.get("allowed_channel_ids", []),
        )
        return DiscordChatApp(config=config)

    elif platform == ChatAppPlatform.WHATSAPP:
        from civStation.utils.chatapp.whatsapp_app import WhatsAppChatApp, WhatsAppConfig

        config = WhatsAppConfig(
            bot_token=kwargs.get("bot_token", ""),
            allowed_user_ids=kwargs.get("allowed_user_ids", []),
            language=kwargs.get("language", "ko"),
            phone_number_id=kwargs.get("phone_number_id", ""),
            business_account_id=kwargs.get("business_account_id", ""),
        )
        return WhatsAppChatApp(config=config)

    raise ValueError(f"Unsupported chat app platform: {platform}")


__all__ = [
    "BaseChatApp",
    "ChatAppConfig",
    "ChatAppPlatform",
    "ChatMessage",
    "create_chat_app",
]
