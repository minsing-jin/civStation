"""
Strategy Module - Strategic planning for Civilization VI agent.

Provides:
- StructuredStrategy: Data structure for representing game strategies
- StrategyPlanner: HITL and autonomous strategy generation
- BaseStrategyPlanner: Abstract base class for custom planners
- InputMode: Enum for HITL input modes (voice, text, auto)
"""

from computer_use_test.agent.modules.hitl import InputMode
from computer_use_test.agent.modules.strategy.base_strategy import BaseStrategyPlanner
from computer_use_test.agent.modules.strategy.strategy_planner import StrategyPlanner
from computer_use_test.agent.modules.strategy.strategy_schemas import (
    GamePhase,
    HITLInputRequiredError,
    StructuredStrategy,
    VictoryType,
    parse_strategy_json,
)
from computer_use_test.agent.modules.strategy.strategy_updater import (
    StrategyRequest,
    StrategyTrigger,
    StrategyUpdater,
)

__all__ = [
    "BaseStrategyPlanner",
    "StrategyPlanner",
    "StrategyUpdater",
    "StrategyRequest",
    "StrategyTrigger",
    "StructuredStrategy",
    "VictoryType",
    "GamePhase",
    "HITLInputRequiredError",
    "InputMode",
    "parse_strategy_json",
]
