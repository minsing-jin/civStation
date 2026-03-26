"""
Enhanced interfaces for static primitive evaluation.

This module re-exports base classes and provides helper utilities
for action comparison with coordinate tolerance.
"""

from civStation.evaluation.evaluator.action_eval.base_static_primitive_evaluator import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
)

# Re-export all base classes for convenience
__all__ = [
    "BaseEvaluator",
    "GroundTruth",
    "EvalResult",
    "within_tolerance",
]


def within_tolerance(coord1: int, coord2: int, tolerance: int = 5) -> bool:
    """
    Check if two coordinates are within a specified tolerance.

    This is useful for comparing predicted vs ground truth coordinates
    where exact pixel-perfect matches are not required.

    Args:
        coord1: First coordinate value
        coord2: Second coordinate value
        tolerance: Maximum allowed difference (default: 5 pixels)

    Returns:
        True if absolute difference is within tolerance, False otherwise

    Examples:
        >>> within_tolerance(100, 103, tolerance=5)
        True
        >>> within_tolerance(100, 106, tolerance=5)
        False
        >>> within_tolerance(500, 497, tolerance=5)
        True
    """
    return abs(coord1 - coord2) <= tolerance
