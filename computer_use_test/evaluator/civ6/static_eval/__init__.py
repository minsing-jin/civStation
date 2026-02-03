"""Static evaluation package - re-exports from interfaces for convenience."""

from computer_use_test.agent.modules.primitive.base_primitive import BasePrimitive
from computer_use_test.agent.modules.router.base_router import PrimitiveRouter
from computer_use_test.evaluator.civ6.static_eval.interfaces import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
    within_tolerance,
)

__all__ = [
    "BasePrimitive",
    "PrimitiveRouter",
    "BaseEvaluator",
    "GroundTruth",
    "EvalResult",
    "within_tolerance",
]
