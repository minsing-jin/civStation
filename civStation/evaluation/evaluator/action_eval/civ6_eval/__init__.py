"""Civ6-specific static evaluation with primitive matching and action comparison."""

from civStation.evaluation.evaluator.action_eval.civ6_eval.civ6_static_evaluator import (
    Civ6StaticEvaluator,
)
from civStation.evaluation.evaluator.action_eval.civ6_eval.main import (
    load_ground_truth_from_json,
)

__all__ = [
    "Civ6StaticEvaluator",
    "load_ground_truth_from_json",
]
