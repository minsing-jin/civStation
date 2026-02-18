"""
Discord chat app implementation.

Provides Discord bot integration for strategy discussions and HITL input
using discord.py library.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from computer_use_test.utils.chatapp.base import BaseChatApp, ChatAppConfig, ChatAppPlatform, ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig(ChatAppConfig):
    """Discord-specific configuration."""

    command_prefix: str = "!"
    allowed_channel_ids: list[str] = field(default_factory=list)


class DiscordChatApp(BaseChatApp):
    """
    Discord bot integration using discord.py.

    Runs the Discord bot in a background thread with its own event loop,
    providing synchronous methods for the agent to interact with.
    """

    def __init__(self, config: DiscordConfig):
        super().__init__(config)
        self._config: DiscordConfig = config
        self._bot = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = False
        self._message_queue: asyncio.Queue | None = None
        self._handlers: list[Callable[[ChatMessage], None]] = []

    def start(self) -> None:
        """Start the Discord bot in a background thread."""
        try:
            import discord
        except ImportError as e:
            raise ImportError("discord.py is required for Discord integration. Install it with: pip install 'computer-use-test[chatapp]'") from e

        intents = discord.Intents.default()
        intents.message_content = True

        self._bot = discord.Client(intents=intents)
        self._loop = asyncio.new_event_loop()
        self._message_queue = asyncio.Queue()

        bot = self._bot
        config = self._config
        queue = self._message_queue
        handlers = self._handlers
        chat_app = self

        @bot.event
        async def on_ready():
            chat_app._connected = True
            logger.info(f"Discord bot connected as {bot.user}")

        @bot.event
        async def on_message(message):
            # Ignore own messages
            if message.author == bot.user:
                return

            author_id = str(message.author.id)

            # Check allowed users
            if config.allowed_user_ids and author_id not in config.allowed_user_ids:
                return

            # Check allowed channels (skip check for DMs)
            is_dm = isinstance(message.channel, discord.DMChannel)
            channel_id = str(message.channel.id)
            if not is_dm and config.allowed_channel_ids and channel_id not in config.allowed_channel_ids:
                return

            chat_msg = ChatMessage(
                content=message.content,
                author_id=author_id,
                author_name=str(message.author.display_name),
                is_bot=message.author.bot,
                is_dm=is_dm,
                platform=ChatAppPlatform.DISCORD,
                channel_id=channel_id,
            )

            # Put in queue for wait_for_message
            await queue.put(chat_msg)

            # Call registered handlers
            for handler in handlers:
                try:
                    handler(chat_msg)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

        def run_bot():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(bot.start(config.bot_token))

        self._thread = threading.Thread(target=run_bot, daemon=True, name="discord-bot")
        self._thread.start()

        # Wait for connection with timeout
        for _ in range(30):
            if self._connected:
                break
            import time

            time.sleep(1)

        if not self._connected:
            logger.warning("Discord bot connection timed out (30s)")

    def stop(self) -> None:
        """Stop the Discord bot."""
        if self._bot and self._loop:
            future = asyncio.run_coroutine_threadsafe(self._bot.close(), self._loop)
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning(f"Error closing Discord bot: {e}")
        self._connected = False

    def send_message(self, channel_id: str, content: str) -> None:
        """Send a message to a Discord channel, splitting if > 2000 chars."""
        if not self._bot or not self._loop:
            logger.error("Discord bot not started")
            return

        async def _send():
            channel = self._bot.get_channel(int(channel_id))
            if not channel:
                logger.error(f"Channel {channel_id} not found")
                return
            for chunk in self._split_message(content):
                await channel.send(chunk)

        future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
        future.result(timeout=30)

    def send_dm(self, user_id: str, content: str) -> None:
        """Send a DM to a Discord user, splitting if > 2000 chars."""
        if not self._bot or not self._loop:
            logger.error("Discord bot not started")
            return

        async def _send():
            user = await self._bot.fetch_user(int(user_id))
            if not user:
                logger.error(f"User {user_id} not found")
                return
            dm_channel = await user.create_dm()
            for chunk in self._split_message(content):
                await dm_channel.send(chunk)

        future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
        future.result(timeout=30)

    def wait_for_message(self, user_id: str | None = None, timeout: float = 300.0) -> ChatMessage | None:
        """
        Wait for a message, optionally from a specific user.

        Bridges async Discord events to synchronous agent code.
        """
        if not self._loop or not self._message_queue:
            logger.error("Discord bot not started")
            return None

        async def _wait():
            while True:
                try:
                    msg = await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
                    if user_id is None or msg.author_id == user_id:
                        return msg
                    # Not from target user, put back? No, just skip
                except TimeoutError:
                    return None

        future = asyncio.run_coroutine_threadsafe(_wait(), self._loop)
        try:
            return future.result(timeout=timeout + 5)
        except Exception:
            return None

    def is_connected(self) -> bool:
        """Check if the Discord bot is connected."""
        return self._connected

    def get_platform(self) -> ChatAppPlatform:
        return ChatAppPlatform.DISCORD

    def register_handler(self, handler: Callable[[ChatMessage], None]) -> None:
        """Register a handler for incoming messages."""
        self._handlers.append(handler)

    def _split_message(self, content: str) -> list[str]:
        """Split a message into chunks that fit Discord's 2000 char limit."""
        max_len = self._config.max_message_length
        if len(content) <= max_len:
            return [content]

        chunks = []
        while content:
            if len(content) <= max_len:
                chunks.append(content)
                break
            # Try to split at newline
            split_at = content.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(content[:split_at])
            content = content[split_at:].lstrip("\n")
        return chunks
