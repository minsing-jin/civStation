"""Civilization VI game agent — primitive-based architecture with VLM reasoning."""

from civStation.agent.models.schema import (
    Action,
    AgentPlan,
    ClickAction,
    DragAction,
    KeyPressAction,
    WaitAction,
)

__all__ = [
    # Execution (lazy-loaded to avoid circular imports with llm_provider)
    "run_one_turn",
    "run_multi_turn",
    "route_primitive",
    "plan_action",
    # Models
    "Action",
    "AgentPlan",
    "ClickAction",
    "DragAction",
    "KeyPressAction",
    "WaitAction",
]

# Lazy imports for turn_executor functions to avoid circular dependency:
# agent.__init__ → turn_executor → llm_provider.base → agent.models.schema → agent.__init__
_EXECUTOR_ATTRS = {"run_one_turn", "run_multi_turn", "route_primitive", "plan_action"}


def __getattr__(name: str):
    if name in _EXECUTOR_ATTRS:
        from civStation.agent import turn_executor

        return getattr(turn_executor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
