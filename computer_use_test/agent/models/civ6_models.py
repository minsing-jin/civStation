"""
Backward compatibility module for Civ6 models.

This module re-exports models from schema.py to maintain backward compatibility
with existing code that imports from civ6_models.py.

DEPRECATED: New code should import directly from
computer_use_test.agent.models.schema instead.
"""

# Re-export everything from schema.py for backward compatibility
from computer_use_test.agent.models.schema import (
    Action,
    ActionType,
    AgentPlan,
    BaseAction,
    ClickAction,
    DragAction,
    DoubleClickAction,
    KeyPressAction,
    WaitAction,
)

# For backward compatibility with code that uses action_type field
# The new schema uses 'type' as the discriminator, but we keep both available
__all__ = [
    "Action",
    "ActionType",
    "AgentPlan",
    "BaseAction",
    "ClickAction",
    "DragAction",
    "DoubleClickAction",
    "KeyPressAction",
    "WaitAction",
]
