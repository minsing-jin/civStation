"""
Discussion Schemas - Data structures for strategy discussion sessions.

Defines the models for multi-turn strategy discussions between
human players and the LLM via chat applications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class DiscussionMode(str, Enum):
    """Mode of the strategy discussion session."""

    PRE_GAME = "pre_game"  # Before the game starts
    IN_GAME = "in_game"  # During gameplay
    POST_TURN = "post_turn"  # After a turn completes


@dataclass
class DiscussionMessage:
    """A single message in a discussion session."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiscussionSession:
    """
    A multi-turn strategy discussion session.

    Tracks the conversation history between the user and the LLM
    for strategy refinement and feedback.
    """

    session_id: str
    mode: DiscussionMode = DiscussionMode.PRE_GAME
    messages: list[DiscussionMessage] = field(default_factory=list)
    is_finalized: bool = False
    user_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(DiscussionMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant (LLM) response to the conversation."""
        self.messages.append(DiscussionMessage(role="assistant", content=content))

    def get_conversation_history(self) -> list[dict[str, str]]:
        """
        Get the conversation as a list of role/content dicts.

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def get_message_count(self) -> int:
        """Get the total number of messages in the session."""
        return len(self.messages)
