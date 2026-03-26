"""Tests for bbox_schema.py Pydantic models."""

import pytest
from pydantic import ValidationError

from civStation.evaluation.evaluator.action_eval.bbox_eval import (
    BBox,
    CaseResult,
    DatasetCase,
    EvalReport,
    GTActionSet,
    GTClickAction,
    GTDoubleClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    ImageSize,
    SequenceResult,
    StepResult,
)


class TestBBox:
    def test_valid_bbox(self):
        bbox = BBox(x_min=10, y_min=20, x_max=100, y_max=200)
        assert bbox.x_min == 10
        assert bbox.y_max == 200

    def test_invalid_x_range(self):
        with pytest.raises(ValidationError, match="x_min"):
            BBox(x_min=100, y_min=20, x_max=10, y_max=200)

    def test_invalid_y_range(self):
        with pytest.raises(ValidationError, match="y_min"):
            BBox(x_min=10, y_min=200, x_max=100, y_max=20)

    def test_zero_size_bbox(self):
        bbox = BBox(x_min=50, y_min=50, x_max=50, y_max=50)
        assert bbox.contains_point(50, 50)
        assert not bbox.contains_point(51, 50)

    def test_contains_point_inside(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        assert bbox.contains_point(100, 200)

    def test_contains_point_on_boundary(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        assert bbox.contains_point(80, 180)
        assert bbox.contains_point(120, 220)

    def test_contains_point_outside(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        assert not bbox.contains_point(79, 200)
        assert not bbox.contains_point(121, 200)
        assert not bbox.contains_point(100, 179)
        assert not bbox.contains_point(100, 221)

    def test_center(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        cx, cy = bbox.center()
        assert cx == 100.0
        assert cy == 200.0

    def test_distance_to_center_at_center(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        dist = bbox.distance_to_center(100, 200)
        assert dist == pytest.approx(0.0)

    def test_distance_to_center_away(self):
        bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
        dist = bbox.distance_to_center(80, 200)
        assert dist > 0.0

    def test_negative_coordinates_rejected(self):
        with pytest.raises(ValidationError):
            BBox(x_min=-1, y_min=0, x_max=10, y_max=10)


class TestGTActions:
    def test_gt_click_action(self):
        action = GTClickAction(
            target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220),
            button="left",
        )
        assert action.type == "click"
        assert action.target_bbox.contains_point(100, 200)

    def test_gt_double_click_action(self):
        action = GTDoubleClickAction(
            target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220),
        )
        assert action.type == "double_click"

    def test_gt_drag_action(self):
        action = GTDragAction(
            start_bbox=BBox(x_min=10, y_min=10, x_max=30, y_max=30),
            end_bbox=BBox(x_min=200, y_min=200, x_max=220, y_max=220),
        )
        assert action.type == "drag"
        assert action.start_bbox.contains_point(20, 20)
        assert action.end_bbox.contains_point(210, 210)

    def test_gt_key_press_action(self):
        action = GTKeyPressAction(keys=["ctrl", "c"])
        assert action.type == "press"
        assert action.keys == ["ctrl", "c"]

    def test_gt_wait_action(self):
        action = GTWaitAction(duration=2.0)
        assert action.type == "wait"
        assert action.duration == 2.0

    def test_gt_action_set_discriminator(self):
        action_set = GTActionSet(
            actions=[
                {"type": "click", "target_bbox": {"x_min": 80, "y_min": 180, "x_max": 120, "y_max": 220}},
                {"type": "press", "keys": ["enter"]},
            ]
        )
        assert len(action_set.actions) == 2
        assert isinstance(action_set.actions[0], GTClickAction)
        assert isinstance(action_set.actions[1], GTKeyPressAction)


class TestDatasetCase:
    def test_valid_case(self):
        case = DatasetCase(
            case_id="test_001",
            instruction="Click the button",
            screenshot_path="screenshots/test.png",
            image_size=ImageSize(width=1000, height=1000),
            action_sets=[
                GTActionSet(
                    actions=[
                        GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220)),
                    ]
                )
            ],
        )
        assert case.case_id == "test_001"

    def test_empty_action_sets_rejected(self):
        with pytest.raises(ValidationError, match="action_sets"):
            DatasetCase(
                case_id="test_001",
                instruction="Click",
                screenshot_path="test.png",
                image_size=ImageSize(width=1000, height=1000),
                action_sets=[],
            )

    def test_multiple_action_sets(self):
        case = DatasetCase(
            case_id="test_002",
            instruction="Click",
            screenshot_path="test.png",
            image_size=ImageSize(width=1000, height=1000),
            action_sets=[
                GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))]),
                GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=75, y_min=175, x_max=125, y_max=225))]),
            ],
            metadata={"turn": 10},
        )
        assert len(case.action_sets) == 2
        assert case.metadata["turn"] == 10


class TestResultModels:
    def test_step_result(self):
        sr = StepResult(step_index=0, correct=True, gt_type="click", pred_type="click", distance_to_center=0.1)
        assert sr.correct

    def test_sequence_result(self):
        seq = SequenceResult(
            strict_success=True,
            prefix_len=2,
            step_accuracy=1.0,
            step_results=[
                StepResult(step_index=0, correct=True, gt_type="click", pred_type="click"),
                StepResult(step_index=1, correct=True, gt_type="press", pred_type="press"),
            ],
        )
        assert seq.strict_success
        assert seq.prefix_len == 2

    def test_case_result_with_error(self):
        cr = CaseResult(case_id="err_001", error="timeout")
        assert cr.error == "timeout"
        assert cr.best_sequence is None

    def test_eval_report(self):
        from civStation.evaluation.evaluator.action_eval.bbox_eval import AggregateMetrics, EvalConfig

        report = EvalReport(
            aggregate=AggregateMetrics(total_cases=1, strict_success_rate=1.0),
            cases=[CaseResult(case_id="test", best_sequence=None)],
            config=EvalConfig(dataset_path="test.jsonl"),
        )
        assert report.aggregate.total_cases == 1
        assert report.timestamp  # auto-generated
