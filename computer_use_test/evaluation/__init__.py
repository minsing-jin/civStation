"""Evaluation framework — static and bbox-based agent evaluation."""

from computer_use_test.evaluation.evaluator.action_eval import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
    within_tolerance,
)
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import run_evaluation

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "GroundTruth",
    "within_tolerance",
    "run_evaluation",
]
