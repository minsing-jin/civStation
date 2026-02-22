"""
Strategy Discussion Engine - Platform-agnostic multi-turn discussion.

Manages strategy discussion sessions where users can have multi-turn
conversations with the LLM to refine game strategy.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum
from typing import TYPE_CHECKING

from computer_use_test.utils.chatapp.discussion.discussion_schemas import (
    DiscussionMode,
    DiscussionSession,
)
from computer_use_test.utils.chatapp.discussion.prompts.discussion_prompts import (
    DISCUSSION_FINALIZE_PROMPT,
    DISCUSSION_LANGUAGE_INSTRUCTION,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_TURN_FEEDBACK_PROMPT,
)
from computer_use_test.utils.llm_provider.parser import strip_markdown

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context import ContextManager
    from computer_use_test.agent.modules.strategy.strategy_schemas import StructuredStrategy
    from computer_use_test.utils.llm_provider.base import BaseVLMProvider

logger = logging.getLogger(__name__)


class DiscussionState(str, Enum):
    """State of the discussion engine."""

    IDLE = "idle"
    DISCUSSING = "discussing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


class StrategyDiscussion:
    """
    Platform-agnostic strategy discussion engine.

    Manages multi-turn conversation sessions between the user and the LLM
    for strategy planning and in-game feedback.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider,
        context_manager: ContextManager | None = None,
    ):
        self.provider = vlm_provider
        self.context_manager = context_manager
        self._sessions: dict[str, DiscussionSession] = {}
        self._user_sessions: dict[str, str] = {}  # user_id -> active session_id
        self._state = DiscussionState.IDLE

    def create_session(self, user_id: str, mode: DiscussionMode = DiscussionMode.PRE_GAME) -> str:
        """
        Create a new discussion session.

        Args:
            user_id: User ID who initiated the session
            mode: Discussion mode (pre_game, in_game, post_turn)

        Returns:
            Session ID string
        """
        session_id = str(uuid.uuid4())[:8]
        session = DiscussionSession(
            session_id=session_id,
            mode=mode,
            user_id=user_id,
        )
        self._sessions[session_id] = session
        self._user_sessions[user_id] = session_id
        self._state = DiscussionState.DISCUSSING
        logger.info(f"Discussion session created: {session_id} (mode={mode.value}, user={user_id})")
        return session_id

    def get_active_session(self, user_id: str) -> DiscussionSession | None:
        """Get the active session for a user, if any."""
        session_id = self._user_sessions.get(user_id)
        if session_id:
            session = self._sessions.get(session_id)
            if session and not session.is_finalized:
                return session
        return None

    def process_message(self, session_id: str, user_message: str, language: str = "ko") -> str:
        """
        Process a user message in a discussion session.

        Sends the conversation history + new message to the LLM
        and returns the response.

        Args:
            session_id: Session ID to add the message to
            user_message: User's message text
            language: Response language code (ko, en, ja, zh). Defaults to "ko".

        Returns:
            LLM response string
        """
        session = self._sessions.get(session_id)
        if not session:
            return "세션을 찾을 수 없습니다. '!strategy'로 새 세션을 시작하세요."

        if session.is_finalized:
            return "이 세션은 이미 확정되었습니다. '!strategy'로 새 세션을 시작하세요."

        # Add user message
        session.add_user_message(user_message)

        # Build context
        context_str = ""
        if self.context_manager:
            context_str = f"현재 게임 상태:\n{self.context_manager.get_combined_context()}"

        # Build conversation prompt with language instruction
        lang_instruction = DISCUSSION_LANGUAGE_INSTRUCTION.get(language, DISCUSSION_LANGUAGE_INSTRUCTION["ko"])
        system_prompt = DISCUSSION_SYSTEM_PROMPT.format(context=context_str, language_instruction=lang_instruction)

        # Build conversation as a single prompt (since we use _send_to_api with text content)
        conversation_text = system_prompt + "\n\n"
        for msg in session.get_conversation_history():
            role_label = "플레이어" if msg["role"] == "user" else "상담사"
            conversation_text += f"{role_label}: {msg['content']}\n\n"
        conversation_text += "상담사: "

        # Call LLM
        try:
            response = self._call_vlm(conversation_text)
            session.add_assistant_message(response)
            logger.info(f"Discussion response generated for session {session_id}")
            return response
        except Exception as e:
            logger.error(f"Discussion LLM call failed: {e}")
            return f"응답 생성 중 오류가 발생했습니다: {e}"

    def finalize_session(self, session_id: str) -> StructuredStrategy | None:
        """
        Finalize a discussion session and extract a StructuredStrategy.

        Uses the LLM to convert the conversation into a structured strategy JSON.

        Args:
            session_id: Session ID to finalize

        Returns:
            StructuredStrategy if successful, None otherwise
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found for finalization")
            return None

        self._state = DiscussionState.FINALIZING

        # Build conversation history string
        history_lines = []
        for msg in session.get_conversation_history():
            role_label = "플레이어" if msg["role"] == "user" else "상담사"
            history_lines.append(f"{role_label}: {msg['content']}")
        conversation_history = "\n".join(history_lines)

        # Build finalize prompt
        prompt = DISCUSSION_FINALIZE_PROMPT.format(conversation_history=conversation_history)

        try:
            response = self._call_vlm(prompt)
            strategy = self._parse_strategy_response(response)

            # Mark session as finalized
            session.is_finalized = True
            self._state = DiscussionState.COMPLETED

            # Set strategy in context manager if available
            if self.context_manager and strategy:
                self.context_manager.set_strategy(strategy)
                logger.info(f"Strategy set from discussion: {strategy.victory_goal.value} victory")

            return strategy
        except Exception as e:
            logger.error(f"Failed to finalize discussion: {e}")
            self._state = DiscussionState.DISCUSSING
            return None

    def extract_strategy(self, session_id: str) -> StructuredStrategy | None:
        """
        Extract strategy from a finalized session.

        Alias for finalize_session that also works on already-finalized sessions.
        """
        session = self._sessions.get(session_id)
        if session and session.is_finalized:
            # Re-parse the last assistant message as strategy
            for msg in reversed(session.messages):
                if msg.role == "assistant":
                    try:
                        return self._parse_strategy_response(msg.content)
                    except Exception:
                        break
        return self.finalize_session(session_id)

    def process_turn_feedback(self, session_id: str, turn_summary: str) -> str:
        """
        Process turn completion feedback in an active session.

        Args:
            session_id: Active session ID
            turn_summary: Summary of the completed turn

        Returns:
            LLM feedback response
        """
        session = self._sessions.get(session_id)
        if not session:
            return "활성 세션이 없습니다."

        # Get current strategy string
        current_strategy = ""
        context_str = ""
        if self.context_manager:
            current_strategy = self.context_manager.get_strategy_string()
            context_str = f"게임 상태:\n{self.context_manager.get_combined_context()}"

        prompt = DISCUSSION_TURN_FEEDBACK_PROMPT.format(
            turn_summary=turn_summary,
            current_strategy=current_strategy,
            context=context_str,
        )

        try:
            response = self._call_vlm(prompt)
            session.add_assistant_message(response)
            return response
        except Exception as e:
            logger.error(f"Turn feedback failed: {e}")
            return f"턴 피드백 생성 중 오류: {e}"

    @property
    def state(self) -> DiscussionState:
        """Get the current discussion state."""
        return self._state

    def _call_vlm(self, prompt: str) -> str:
        """Call the VLM with a text-only prompt."""
        content_parts = [self.provider._build_text_content(prompt)]
        response = self.provider._send_to_api(
            content_parts,
            temperature=0.7,
            max_tokens=2048,
        )
        return response.content

    def _parse_strategy_response(self, response: str) -> StructuredStrategy:
        """Parse VLM response into StructuredStrategy."""
        from computer_use_test.agent.modules.strategy.strategy_schemas import parse_strategy_json

        return parse_strategy_json(strip_markdown(response))
