"""Tests for bbox_scorer.py scoring algorithms."""

import pytest

from computer_use_test.agent.models.schema import (
    ClickAction,
    DragAction,
    KeyPressAction,
    WaitAction,
)
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    BBox,
    CaseResult,
    GTClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    SequenceResult,
    StepResult,
    aggregate_results,
    compare_sequence,
    compare_step,
    levenshtein_distance,
    levenshtein_similarity,
    select_best_gt_set,
)

# ==================== Levenshtein ====================


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0

    def test_empty_vs_nonempty(self):
        assert levenshtein_distance([], ["a", "b"]) == 2
        assert levenshtein_distance(["a"], []) == 1

    def test_both_empty(self):
        assert levenshtein_distance([], []) == 0

    def test_single_substitution(self):
        assert levenshtein_distance(["a"], ["b"]) == 1

    def test_insertion_deletion(self):
        assert levenshtein_distance(["a", "b", "c"], ["a", "c"]) == 1

    def test_similarity_identical(self):
        assert levenshtein_similarity(["enter"], ["enter"]) == pytest.approx(1.0)

    def test_similarity_empty(self):
        assert levenshtein_similarity([], []) == pytest.approx(1.0)

    def test_similarity_partial(self):
        sim = levenshtein_similarity(["ctrl", "c"], ["ctrl", "v"])
        assert 0.0 < sim < 1.0


# ==================== Step Comparison ====================


class TestCompareStep:
    def test_click_inside_bbox(self):
        gt = GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220), button="left")
        pred = ClickAction(type="click", x=100, y=200, button="left")
        result = compare_step(gt, pred)
        assert result.correct
        assert result.gt_type == "click"
        assert result.distance_to_center is not None
        assert result.distance_to_center == pytest.approx(0.0, abs=0.01)

    def test_click_outside_bbox(self):
        gt = GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220), button="left")
        pred = ClickAction(type="click", x=50, y=200, button="left")
        result = compare_step(gt, pred)
        assert not result.correct
        assert result.distance_to_center > 0

    def test_click_wrong_button(self):
        gt = GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220), button="left")
        pred = ClickAction(type="click", x=100, y=200, button="right")
        result = compare_step(gt, pred)
        assert not result.correct

    def test_click_on_boundary(self):
        gt = GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220), button="left")
        pred = ClickAction(type="click", x=80, y=180, button="left")
        result = compare_step(gt, pred)
        assert result.correct

    def test_click_type_mismatch(self):
        gt = GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))
        pred = KeyPressAction(type="press", keys=["enter"])
        result = compare_step(gt, pred)
        assert not result.correct
        assert result.pred_type == "press"

    def test_drag_both_inside(self):
        gt = GTDragAction(
            start_bbox=BBox(x_min=10, y_min=10, x_max=30, y_max=30),
            end_bbox=BBox(x_min=200, y_min=200, x_max=220, y_max=220),
        )
        pred = DragAction(type="drag", start_x=20, start_y=20, end_x=210, end_y=210)
        result = compare_step(gt, pred)
        assert result.correct

    def test_drag_start_outside(self):
        gt = GTDragAction(
            start_bbox=BBox(x_min=10, y_min=10, x_max=30, y_max=30),
            end_bbox=BBox(x_min=200, y_min=200, x_max=220, y_max=220),
        )
        pred = DragAction(type="drag", start_x=5, start_y=20, end_x=210, end_y=210)
        result = compare_step(gt, pred)
        assert not result.correct

    def test_drag_end_outside(self):
        gt = GTDragAction(
            start_bbox=BBox(x_min=10, y_min=10, x_max=30, y_max=30),
            end_bbox=BBox(x_min=200, y_min=200, x_max=220, y_max=220),
        )
        pred = DragAction(type="drag", start_x=20, start_y=20, end_x=230, end_y=210)
        result = compare_step(gt, pred)
        assert not result.correct

    def test_keypress_exact_match(self):
        gt = GTKeyPressAction(keys=["ctrl", "c"])
        pred = KeyPressAction(type="press", keys=["ctrl", "c"])
        result = compare_step(gt, pred)
        assert result.correct
        assert result.levenshtein_similarity == pytest.approx(1.0)

    def test_keypress_different_keys(self):
        gt = GTKeyPressAction(keys=["ctrl", "c"])
        pred = KeyPressAction(type="press", keys=["ctrl", "v"])
        result = compare_step(gt, pred)
        assert not result.correct
        assert result.levenshtein_similarity is not None
        assert 0.0 < result.levenshtein_similarity < 1.0

    def test_wait_ignored(self):
        gt = GTWaitAction(duration=1.0)
        pred = ClickAction(type="click", x=100, y=200)
        result = compare_step(gt, pred, ignore_wait=True)
        assert result.correct

    def test_wait_not_ignored_type_mismatch(self):
        gt = GTWaitAction(duration=1.0)
        pred = ClickAction(type="click", x=100, y=200)
        result = compare_step(gt, pred, ignore_wait=False)
        assert not result.correct

    def test_wait_type_match(self):
        gt = GTWaitAction(duration=1.0)
        pred = WaitAction(type="wait", duration=2.0)
        result = compare_step(gt, pred, ignore_wait=False)
        assert result.correct


# ==================== Sequence Comparison ====================


class TestCompareSequence:
    def test_perfect_match(self):
        gt_actions = [
            GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
            GTKeyPressAction(keys=["enter"]),
        ]
        pred_actions = [
            ClickAction(type="click", x=100, y=200),
            KeyPressAction(type="press", keys=["enter"]),
        ]
        result = compare_sequence(gt_actions, pred_actions)
        assert result.strict_success
        assert result.prefix_len == 2
        assert result.step_accuracy == pytest.approx(1.0)

    def test_first_wrong(self):
        gt_actions = [
            GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
            GTKeyPressAction(keys=["enter"]),
        ]
        pred_actions = [
            ClickAction(type="click", x=0, y=0),  # outside bbox
            KeyPressAction(type="press", keys=["enter"]),
        ]
        result = compare_sequence(gt_actions, pred_actions)
        assert not result.strict_success
        assert result.prefix_len == 0
        assert result.step_accuracy == pytest.approx(0.5)

    def test_length_mismatch_pred_shorter(self):
        gt_actions = [
            GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
            GTKeyPressAction(keys=["enter"]),
        ]
        pred_actions = [
            ClickAction(type="click", x=100, y=200),
        ]
        result = compare_sequence(gt_actions, pred_actions)
        assert not result.strict_success
        assert result.prefix_len == 1
        assert result.step_accuracy == pytest.approx(0.5)

    def test_length_mismatch_pred_longer(self):
        gt_actions = [
            GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
        ]
        pred_actions = [
            ClickAction(type="click", x=100, y=200),
            KeyPressAction(type="press", keys=["enter"]),
        ]
        result = compare_sequence(gt_actions, pred_actions)
        assert not result.strict_success
        assert result.prefix_len == 1
        assert result.step_accuracy == pytest.approx(1.0)

    def test_empty_gt_empty_pred(self):
        result = compare_sequence([], [])
        assert result.strict_success
        assert result.step_accuracy == pytest.approx(1.0)

    def test_empty_gt_nonempty_pred(self):
        result = compare_sequence([], [ClickAction(type="click", x=100, y=200)])
        assert not result.strict_success

    def test_ignore_wait_filters_both(self):
        gt_actions = [
            GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
            GTWaitAction(duration=1.0),
            GTKeyPressAction(keys=["enter"]),
        ]
        pred_actions = [
            ClickAction(type="click", x=100, y=200),
            KeyPressAction(type="press", keys=["enter"]),
        ]
        result = compare_sequence(gt_actions, pred_actions, ignore_wait=True)
        assert result.strict_success
        assert result.prefix_len == 2


# ==================== Best GT Set Selection ====================


class TestSelectBestGTSet:
    def test_selects_matching_set(self):
        gt_sets = [
            [GTClickAction(target_bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10))],  # won't match
            [GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))],  # will match
        ]
        pred = [ClickAction(type="click", x=100, y=200)]
        result, index = select_best_gt_set(gt_sets, pred)
        assert result.strict_success
        assert index == 1

    def test_selects_best_prefix(self):
        gt_sets = [
            [
                GTClickAction(target_bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10)),
                GTKeyPressAction(keys=["enter"]),
            ],
            [
                GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
                GTKeyPressAction(keys=["esc"]),
            ],
        ]
        pred = [
            ClickAction(type="click", x=100, y=200),
            KeyPressAction(type="press", keys=["enter"]),
        ]
        result, index = select_best_gt_set(gt_sets, pred)
        # Set 1 has prefix_len=1 (click matches but esc!=enter)
        # Set 0 has prefix_len=0 (click outside bbox)
        assert index == 1
        assert result.prefix_len == 1


# ==================== Aggregation ====================


class TestAggregateResults:
    def test_all_success(self):
        cases = [
            CaseResult(
                case_id="c1",
                best_sequence=SequenceResult(
                    strict_success=True,
                    prefix_len=2,
                    step_accuracy=1.0,
                    step_results=[
                        StepResult(step_index=0, correct=True, gt_type="click"),
                        StepResult(step_index=1, correct=True, gt_type="press"),
                    ],
                ),
                agent_actions_count=2,
            ),
        ]
        metrics = aggregate_results(cases)
        assert metrics.total_cases == 1
        assert metrics.strict_success_rate == pytest.approx(1.0)
        assert metrics.avg_step_accuracy == pytest.approx(1.0)
        assert metrics.error_count == 0

    def test_with_errors(self):
        cases = [
            CaseResult(case_id="c1", error="parse failure"),
            CaseResult(
                case_id="c2",
                best_sequence=SequenceResult(
                    strict_success=True,
                    prefix_len=1,
                    step_accuracy=1.0,
                    step_results=[StepResult(step_index=0, correct=True, gt_type="click")],
                ),
                agent_actions_count=1,
            ),
        ]
        metrics = aggregate_results(cases)
        assert metrics.total_cases == 2
        assert metrics.error_count == 1
        assert metrics.strict_success_rate == pytest.approx(1.0)

    def test_empty(self):
        metrics = aggregate_results([])
        assert metrics.total_cases == 0

    def test_per_action_type_breakdown(self):
        cases = [
            CaseResult(
                case_id="c1",
                best_sequence=SequenceResult(
                    strict_success=False,
                    prefix_len=1,
                    step_accuracy=0.5,
                    step_results=[
                        StepResult(step_index=0, correct=True, gt_type="click"),
                        StepResult(step_index=1, correct=False, gt_type="press"),
                    ],
                ),
                agent_actions_count=2,
            ),
        ]
        metrics = aggregate_results(cases)
        type_map = {m.action_type: m for m in metrics.per_action_type}
        assert type_map["click"].accuracy == pytest.approx(1.0)
        assert type_map["press"].accuracy == pytest.approx(0.0)
