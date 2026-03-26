"""Static evaluation package - re-exports from interfaces for convenience."""

from civStation.agent.modules.primitive.base_primitive import BasePrimitive
from civStation.agent.modules.router.base_router import PrimitiveRouter
from civStation.evaluation.evaluator.action_eval.interfaces import (
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
