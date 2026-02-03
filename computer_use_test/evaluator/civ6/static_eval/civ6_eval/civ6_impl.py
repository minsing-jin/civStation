"""
Civilization VI specific implementation of the Static Primitive Evaluator.

This module contains:
- Seven distinct primitive implementations for different Civ6 game scenarios
- Mock router for primitive selection based on screenshot filename
- Enhanced evaluator with 5-pixel coordinate tolerance
"""

from computer_use_test.agent.models.schema import (
    Action,
    AgentPlan,
    ClickAction,
    DragAction,
    KeyPressAction,
)
from computer_use_test.evaluator.civ6.static_eval.interfaces import (
    BaseEvaluator,
    EvalResult,
    GroundTruth,
    within_tolerance,
)

# ==================== Evaluator Implementation ====================


class Civ6StaticEvaluator(BaseEvaluator):
    """
    Enhanced evaluator with 5-pixel coordinate tolerance.

    Compares predicted actions against ground truth with appropriate
    tolerances for different action types.
    """

    COORD_TOLERANCE = 5

    def _compare_actions(self, gt_action: Action, pred_action: Action) -> bool:
        """
        Compare two actions with appropriate tolerance.

        Args:
            gt_action: Ground truth action
            pred_action: Predicted action

        Returns:
            True if actions match within tolerance, False otherwise
        """
        # Type must match exactly
        if type(gt_action) is not type(pred_action):
            return False

        # ClickAction: allow coordinate tolerance
        if isinstance(gt_action, ClickAction):
            assert isinstance(pred_action, ClickAction)  # Type narrowing
            return (
                within_tolerance(gt_action.x, pred_action.x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.y, pred_action.y, self.COORD_TOLERANCE)
                and gt_action.button == pred_action.button
            )

        # DragAction: allow coordinate tolerance for all coordinates
        elif isinstance(gt_action, DragAction):
            assert isinstance(pred_action, DragAction)
            return (
                within_tolerance(gt_action.start_x, pred_action.start_x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.start_y, pred_action.start_y, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.end_x, pred_action.end_x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.end_y, pred_action.end_y, self.COORD_TOLERANCE)
            )

        # KeyPressAction: exact match required for keys
        elif isinstance(gt_action, KeyPressAction):
            assert isinstance(pred_action, KeyPressAction)
            return gt_action.keys == pred_action.keys

        # Unknown action type
        return False

    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        """
        Compare predicted plan against ground truth.

        Args:
            gt: Ground truth data
            selected_prim: Name of selected primitive
            plan: Generated action plan

        Returns:
            Evaluation result with match status
        """
        # 1. Check primitive selection
        prim_match = gt.expected_primitive == selected_prim

        # 2. Check action sequence
        actions_match = True
        if len(gt.expected_actions) != len(plan.actions):
            actions_match = False
        else:
            for gt_act, pred_act in zip(gt.expected_actions, plan.actions, strict=False):
                if not self._compare_actions(gt_act, pred_act):
                    actions_match = False
                    break

        return EvalResult(
            screenshot_path=gt.screenshot_path,
            selected_primitive=selected_prim,
            primitive_match=prim_match,
            action_sequence_match=actions_match,
            levenshtein_distance=0,  # TODO: Implement edit distance for partial credit
        )
