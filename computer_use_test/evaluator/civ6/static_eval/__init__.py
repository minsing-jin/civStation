"""Static evaluation package - re-exports from interfaces for convenience."""

from computer_use_test.evaluator.civ6.static_eval.interfaces import (
    BaseEvaluator,
    BasePrimitive,
    EvalResult,
    GroundTruth,
    PrimitiveRouter,
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
