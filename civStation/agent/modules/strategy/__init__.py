"""
Strategy Module - Strategic planning for Civilization VI agent.

Provides:
- StructuredStrategy: Data structure for representing game strategies
- StrategyPlanner: HITL and autonomous strategy generation
- BaseStrategyPlanner: Abstract base class for custom planners
- InputMode: Enum for HITL input modes (voice, text, auto)
"""

from civStation.agent.modules.hitl import InputMode
from civStation.agent.modules.strategy.base_strategy import BaseStrategyPlanner
from civStation.agent.modules.strategy.strategy_planner import StrategyPlanner
from civStation.agent.modules.strategy.strategy_schemas import (
    GamePhase,
    HITLInputRequiredError,
    StructuredStrategy,
    VictoryType,
    parse_strategy_json,
)
from civStation.agent.modules.strategy.strategy_updater import (
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
