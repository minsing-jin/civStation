"""Evaluator engines for static and bbox-based evaluation."""

from civStation.evaluation.evaluator.action_eval import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
    within_tolerance,
)

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "GroundTruth",
    "within_tolerance",
]
