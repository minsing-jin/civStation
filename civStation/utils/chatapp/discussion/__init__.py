"""
Strategy Discussion Module.

Provides multi-turn strategy discussion capabilities for the Civilization VI agent,
allowing players to discuss and refine strategies with the LLM via chat applications.

Components:
- StrategyDiscussion: Platform-agnostic discussion engine
- DiscussionSession: Session state with conversation history
- DiscordDiscussionHandler: Discord command routing
"""

from civStation.utils.chatapp.discussion.discussion_engine import (
    DiscussionState,
    StrategyDiscussion,
)
from civStation.utils.chatapp.discussion.discussion_schemas import (
    DiscussionMessage,
    DiscussionMode,
    DiscussionSession,
)

__all__ = [
    "DiscussionMessage",
    "DiscussionMode",
    "DiscussionSession",
    "DiscussionState",
    "StrategyDiscussion",
]
