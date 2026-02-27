"""
Pytest tests for 5-pixel coordinate tolerance in action comparison.

Tests verify that the evaluator correctly handles coordinate comparisons
with the specified tolerance for different action types.
"""

import pytest

from computer_use_test.agent.models.schema import ClickAction, DragAction, KeyPressAction
from computer_use_test.evaluation.evaluator.action_eval.civ6_eval.civ6_impl import Civ6StaticEvaluator


@pytest.fixture
def evaluator():
    """Create a Civ6StaticEvaluator instance for testing."""
    return Civ6StaticEvaluator(router=None, primitives={})


class TestClickActionTolerance:
    """Tests for ClickAction coordinate tolerance."""

    def test_exact_match(self, evaluator):
        """Test that exact coordinate match passes."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = ClickAction(type="click", x=100, y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_within_tolerance_3_pixels(self, evaluator):
        """Test that 3-pixel difference passes (within 5-pixel tolerance)."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = ClickAction(type="click", x=103, y=197)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_at_tolerance_boundary_5_pixels(self, evaluator):
        """Test that exactly 5-pixel difference passes."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = ClickAction(type="click", x=105, y=195)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_beyond_tolerance_6_pixels(self, evaluator):
        """Test that 6-pixel difference fails (exceeds tolerance)."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = ClickAction(type="click", x=106, y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_beyond_tolerance_y_axis(self, evaluator):
        """Test that tolerance applies to Y axis as well."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = ClickAction(type="click", x=100, y=206)
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_different_button(self, evaluator):
        """Test that different mouse buttons fail comparison."""
        gt_action = ClickAction(type="click", x=100, y=200, button="left")
        pred_action = ClickAction(type="click", x=100, y=200, button="right")
        assert evaluator._compare_actions(gt_action, pred_action) is False

    @pytest.mark.parametrize(
        "gt_x,gt_y,pred_x,pred_y,expected",
        [
            (100, 200, 100, 200, True),  # Exact match
            (100, 200, 101, 201, True),  # 1 pixel diff
            (100, 200, 103, 198, True),  # 3 pixel diff
            (100, 200, 105, 205, True),  # 5 pixel diff (boundary)
            (100, 200, 106, 206, False),  # 6 pixel diff (over)
            (500, 300, 495, 305, True),  # 5 pixel diff both axes
            (500, 300, 494, 306, False),  # 6 pixel diff both axes
        ],
    )
    def test_tolerance_parametrized(self, evaluator, gt_x, gt_y, pred_x, pred_y, expected):
        """Parametrized test for various coordinate differences."""
        gt_action = ClickAction(type="click", x=gt_x, y=gt_y)
        pred_action = ClickAction(type="click", x=pred_x, y=pred_y)
        assert evaluator._compare_actions(gt_action, pred_action) is expected


class TestDragActionTolerance:
    """Tests for DragAction coordinate tolerance."""

    def test_exact_match(self, evaluator):
        """Test that exact drag coordinates match."""
        gt_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        pred_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_within_tolerance_start_coords(self, evaluator):
        """Test tolerance on start coordinates."""
        gt_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        pred_action = DragAction(type="drag", start_x=13, start_y=17, end_x=100, end_y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_within_tolerance_end_coords(self, evaluator):
        """Test tolerance on end coordinates."""
        gt_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        pred_action = DragAction(type="drag", start_x=10, start_y=20, end_x=104, end_y=196)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_within_tolerance_all_coords(self, evaluator):
        """Test tolerance on all coordinates simultaneously."""
        gt_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        pred_action = DragAction(type="drag", start_x=13, start_y=23, end_x=105, end_y=205)
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_beyond_tolerance_start_x(self, evaluator):
        """Test failure when start_x exceeds tolerance."""
        gt_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        pred_action = DragAction(type="drag", start_x=16, start_y=20, end_x=100, end_y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is False


class TestKeyPressAction:
    """Tests for KeyPressAction exact matching."""

    def test_exact_key_match(self, evaluator):
        """Test that exact key match passes."""
        gt_action = KeyPressAction(type="press", keys=["esc"])
        pred_action = KeyPressAction(type="press", keys=["esc"])
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_different_keys_fail(self, evaluator):
        """Test that different keys fail."""
        gt_action = KeyPressAction(type="press", keys=["esc"])
        pred_action = KeyPressAction(type="press", keys=["enter"])
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_multiple_keys_match(self, evaluator):
        """Test that multiple keys must match exactly."""
        gt_action = KeyPressAction(type="press", keys=["ctrl", "c"])
        pred_action = KeyPressAction(type="press", keys=["ctrl", "c"])
        assert evaluator._compare_actions(gt_action, pred_action) is True

    def test_multiple_keys_different_order_fail(self, evaluator):
        """Test that key order matters."""
        gt_action = KeyPressAction(type="press", keys=["ctrl", "c"])
        pred_action = KeyPressAction(type="press", keys=["c", "ctrl"])
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_different_key_count_fail(self, evaluator):
        """Test that different number of keys fail."""
        gt_action = KeyPressAction(type="press", keys=["ctrl", "c"])
        pred_action = KeyPressAction(type="press", keys=["ctrl"])
        assert evaluator._compare_actions(gt_action, pred_action) is False


class TestActionTypeMismatch:
    """Tests for action type mismatches."""

    def test_click_vs_keypress(self, evaluator):
        """Test that ClickAction doesn't match KeyPressAction."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = KeyPressAction(type="press", keys=["a"])
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_click_vs_drag(self, evaluator):
        """Test that ClickAction doesn't match DragAction."""
        gt_action = ClickAction(type="click", x=100, y=200)
        pred_action = DragAction(type="drag", start_x=100, start_y=200, end_x=150, end_y=250)
        assert evaluator._compare_actions(gt_action, pred_action) is False

    def test_keypress_vs_drag(self, evaluator):
        """Test that KeyPressAction doesn't match DragAction."""
        gt_action = KeyPressAction(type="press", keys=["m"])
        pred_action = DragAction(type="drag", start_x=10, start_y=20, end_x=100, end_y=200)
        assert evaluator._compare_actions(gt_action, pred_action) is False
