"""
Discord Discussion Handler - Routes Discord commands to the discussion engine.

Handles Discord-specific commands (!strategy, !discuss, !status, !finalize, !help)
and connects them to the platform-agnostic StrategyDiscussion engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from computer_use_test.utils.chatapp.discussion.discussion_schemas import DiscussionMode

if TYPE_CHECKING:
    from computer_use_test.utils.chatapp.base import BaseChatApp, ChatMessage
    from computer_use_test.utils.chatapp.discussion.discussion_engine import StrategyDiscussion

logger = logging.getLogger(__name__)

HELP_TEXT = """\
**Civ6 전략 토론 봇 명령어**

`!strategy [초기 전략]` - 새 전략 토론 시작
`!discuss <내용>` - 전략 토론 계속
`!status` - 현재 게임 상태/전략 표시
`!finalize` - 전략 확정
`!help` - 이 도움말 표시

DM으로 메시지를 보내면 자동으로 토론 세션에 연결됩니다.
"""


class DiscordDiscussionHandler:
    """
    Handles Discord commands and routes them to the discussion engine.

    Registered as a message handler on the Discord chat app to process
    strategy-related commands.
    """

    def __init__(
        self,
        chat_app: BaseChatApp,
        discussion_engine: StrategyDiscussion,
        command_prefix: str = "!",
    ):
        self._chat_app = chat_app
        self._engine = discussion_engine
        self._prefix = command_prefix
        self._active_users: set[str] = set()

        # Register self as message handler
        chat_app.register_handler(self._handle_message)

    def _handle_message(self, message: ChatMessage) -> None:
        """Route incoming messages to appropriate command handlers."""
        content = message.content.strip()

        # Command dispatch
        if content.startswith(f"{self._prefix}strategy"):
            args = content[len(f"{self._prefix}strategy") :].strip()
            self._cmd_strategy(message, args)
        elif content.startswith(f"{self._prefix}discuss"):
            args = content[len(f"{self._prefix}discuss") :].strip()
            self._cmd_discuss(message, args)
        elif content.startswith(f"{self._prefix}status"):
            self._cmd_status(message)
        elif content.startswith(f"{self._prefix}finalize"):
            self._cmd_finalize(message)
        elif content.startswith(f"{self._prefix}help"):
            self._cmd_help(message)
        elif message.is_dm:
            # DMs without commands go to active discussion session
            self._handle_dm(message)

    def _cmd_strategy(self, message: ChatMessage, initial_strategy: str) -> None:
        """Handle !strategy command - start a new discussion session."""
        user_id = message.author_id
        channel_id = message.channel_id

        # Create a new session
        session_id = self._engine.create_session(user_id, mode=DiscussionMode.PRE_GAME)
        self._active_users.add(user_id)

        if initial_strategy:
            # Process the initial strategy message
            response = self._engine.process_message(session_id, initial_strategy)
            self._send_response(channel_id, message.is_dm, user_id, response)
        else:
            self._send_response(
                channel_id,
                message.is_dm,
                user_id,
                "전략 토론 세션이 시작되었습니다! 원하는 승리 유형이나 전략 방향을 말씀해주세요.",
            )

    def _cmd_discuss(self, message: ChatMessage, content: str) -> None:
        """Handle !discuss command - continue a discussion."""
        user_id = message.author_id
        channel_id = message.channel_id

        if not content:
            self._send_response(channel_id, message.is_dm, user_id, "토론할 내용을 입력해주세요. 예: `!discuss 과학 승리로 갈까?`")
            return

        session = self._engine.get_active_session(user_id)
        if not session:
            # Auto-create a session
            session_id = self._engine.create_session(user_id, mode=DiscussionMode.IN_GAME)
            self._active_users.add(user_id)
        else:
            session_id = session.session_id

        response = self._engine.process_message(session_id, content)
        self._send_response(channel_id, message.is_dm, user_id, response)

    def _cmd_status(self, message: ChatMessage) -> None:
        """Handle !status command - show current game state and strategy."""
        user_id = message.author_id
        channel_id = message.channel_id

        lines = []

        # Show active session info
        session = self._engine.get_active_session(user_id)
        if session:
            lines.append(f"**활성 세션**: {session.session_id} (모드: {session.mode.value})")
            lines.append(f"**메시지 수**: {session.get_message_count()}")
        else:
            lines.append("활성 토론 세션이 없습니다.")

        # Show current strategy if context manager is available
        if self._engine.context_manager:
            strategy_str = self._engine.context_manager.get_strategy_string()
            if strategy_str and strategy_str != "전략 미설정":
                lines.append(f"\n**현재 전략**:\n{strategy_str}")
            else:
                lines.append("\n전략이 설정되지 않았습니다.")

        self._send_response(channel_id, message.is_dm, user_id, "\n".join(lines))

    def _cmd_finalize(self, message: ChatMessage) -> None:
        """Handle !finalize command - finalize the current strategy."""
        user_id = message.author_id
        channel_id = message.channel_id

        session = self._engine.get_active_session(user_id)
        if not session:
            self._send_response(channel_id, message.is_dm, user_id, "확정할 활성 세션이 없습니다. `!strategy`로 새 세션을 시작하세요.")
            return

        strategy = self._engine.finalize_session(session.session_id)
        if strategy:
            self._send_response(
                channel_id,
                message.is_dm,
                user_id,
                f"전략이 확정되었습니다!\n\n{strategy.to_prompt_string()}",
            )
        else:
            self._send_response(channel_id, message.is_dm, user_id, "전략 확정에 실패했습니다. 다시 시도해주세요.")

    def _cmd_help(self, message: ChatMessage) -> None:
        """Handle !help command."""
        self._send_response(message.channel_id, message.is_dm, message.author_id, HELP_TEXT)

    def _handle_dm(self, message: ChatMessage) -> None:
        """Handle DM messages - route to active session."""
        user_id = message.author_id
        session = self._engine.get_active_session(user_id)

        if session:
            response = self._engine.process_message(session.session_id, message.content)
            self._send_response(message.channel_id, True, user_id, response)
        # Ignore DMs without active sessions (no spam)

    def notify_turn_complete(self, summary: str) -> None:
        """
        Notify active users about turn completion.

        Args:
            summary: Turn summary text to send
        """
        notification = f"**턴 완료 알림**\n{summary}\n\n`!discuss`로 전략 조정을 논의할 수 있습니다."

        for user_id in self._active_users:
            try:
                self._chat_app.send_dm(user_id, notification)
            except Exception as e:
                logger.error(f"Failed to notify user {user_id}: {e}")

    def _send_response(self, channel_id: str, is_dm: bool, user_id: str, content: str) -> None:
        """Send a response via channel message or DM."""
        try:
            if is_dm:
                self._chat_app.send_dm(user_id, content)
            else:
                self._chat_app.send_message(channel_id, content)
        except Exception as e:
            logger.error(f"Failed to send response: {e}")
