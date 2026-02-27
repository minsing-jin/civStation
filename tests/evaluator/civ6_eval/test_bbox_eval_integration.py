"""End-to-end integration test for bbox evaluation pipeline."""

import json
from pathlib import Path

import pytest

from computer_use_test.agent.models.schema import ClickAction, DragAction, KeyPressAction, WaitAction
from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import (
    AgentResponse,
    BaseAgentRunner,
    BBox,
    DatasetCase,
    GTActionSet,
    GTClickAction,
    GTDoubleClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    ImageSize,
    MockAgentRunner,
    evaluate_case,
    run_evaluation,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATASET = FIXTURES_DIR / "sample_bbox_dataset.jsonl"


class PerfectAgentRunner(BaseAgentRunner):
    """Agent that returns the center of GT bboxes from the first action set."""

    def __init__(self, dataset_cases: list[DatasetCase]):
        self._cases = {c.case_id: c for c in dataset_cases}

    def run_case(self, case: DatasetCase) -> AgentResponse:
        from computer_use_test.agent.models.schema import DoubleClickAction

        gt_actions = case.action_sets[0].actions
        pred_actions = []
        for gt in gt_actions:
            if isinstance(gt, GTClickAction):
                cx, cy = gt.target_bbox.center()
                pred_actions.append(ClickAction(type="click", x=int(cx), y=int(cy), button=gt.button))
            elif isinstance(gt, GTDoubleClickAction):
                cx, cy = gt.target_bbox.center()
                pred_actions.append(DoubleClickAction(type="double_click", x=int(cx), y=int(cy), button=gt.button))
            elif isinstance(gt, GTDragAction):
                sx, sy = gt.start_bbox.center()
                ex, ey = gt.end_bbox.center()
                pred_actions.append(DragAction(type="drag", start_x=int(sx), start_y=int(sy), end_x=int(ex), end_y=int(ey)))
            elif isinstance(gt, GTKeyPressAction):
                pred_actions.append(KeyPressAction(type="press", keys=gt.keys))
            elif isinstance(gt, GTWaitAction):
                pred_actions.append(WaitAction(type="wait", duration=gt.duration))
        return AgentResponse(actions=pred_actions, meta={"agent": "perfect"})


class TestEvaluateCase:
    def test_perfect_click_case(self):
        case = DatasetCase(
            case_id="test_click",
            instruction="Click",
            screenshot_path="test.png",
            image_size=ImageSize(width=1000, height=1000),
            action_sets=[GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))])],
        )

        class ClickCenterRunner(BaseAgentRunner):
            def run_case(self, case):
                return AgentResponse(actions=[ClickAction(type="click", x=100, y=200, button="left")])

        result = evaluate_case(case, ClickCenterRunner())
        assert result.best_sequence is not None
        assert result.best_sequence.strict_success
        assert result.error is None

    def test_error_case(self):
        case = DatasetCase(
            case_id="test_error",
            instruction="Click",
            screenshot_path="test.png",
            image_size=ImageSize(width=1000, height=1000),
            action_sets=[GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=80, y_min=180, x_max=120, y_max=220))])],
        )

        class ErrorRunner(BaseAgentRunner):
            def run_case(self, case):
                from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import AgentRunnerError

                raise AgentRunnerError("test error")

        result = evaluate_case(case, ErrorRunner())
        assert result.error is not None
        assert result.best_sequence is None

    def test_multiple_gt_sets_selects_best(self):
        case = DatasetCase(
            case_id="test_multi",
            instruction="Click",
            screenshot_path="test.png",
            image_size=ImageSize(width=1000, height=1000),
            action_sets=[
                # Set 0: tight bbox, won't match point (500, 300)
                GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=0, y_min=0, x_max=10, y_max=10))]),
                # Set 1: wide bbox, will match point (500, 300)
                GTActionSet(actions=[GTClickAction(target_bbox=BBox(x_min=400, y_min=200, x_max=600, y_max=400))]),
            ],
        )

        runner = MockAgentRunner()  # Returns click at (500, 300)
        result = evaluate_case(case, runner)
        assert result.gt_set_index == 1
        assert result.best_sequence is not None
        # MockAgentRunner returns 2 actions but GT has 1, so not strict success
        assert not result.best_sequence.strict_success


@pytest.mark.integration
class TestRunEvaluationEndToEnd:
    def test_with_mock_runner(self):
        """Run full evaluation with mock runner on fixture dataset."""
        runner = MockAgentRunner()
        report = run_evaluation(
            dataset_path=str(SAMPLE_DATASET),
            runner=runner,
            verbose=True,
        )

        assert report.aggregate.total_cases == 3
        assert len(report.cases) == 3
        assert report.config.dataset_path == str(SAMPLE_DATASET)

    def test_with_perfect_agent(self):
        """Run full evaluation with agent that returns perfect predictions."""
        from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import load_dataset

        cases = load_dataset(SAMPLE_DATASET)
        runner = PerfectAgentRunner(cases)
        report = run_evaluation(
            dataset_path=str(SAMPLE_DATASET),
            runner=runner,
            verbose=True,
        )

        assert report.aggregate.total_cases == 3
        # All cases should be strict successes with the perfect agent
        for case_result in report.cases:
            assert case_result.best_sequence is not None
            assert case_result.best_sequence.strict_success, f"Case {case_result.case_id} failed"

    def test_report_serialization(self):
        """Verify report can be serialized to JSON and parsed back."""
        runner = MockAgentRunner()
        report = run_evaluation(
            dataset_path=str(SAMPLE_DATASET),
            runner=runner,
        )

        json_str = report.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["aggregate"]["total_cases"] == 3
        assert len(parsed["cases"]) == 3
        assert "timestamp" in parsed

    def test_output_to_file(self, tmp_path):
        """Verify report can be written to file."""
        runner = MockAgentRunner()
        report = run_evaluation(
            dataset_path=str(SAMPLE_DATASET),
            runner=runner,
        )

        output_path = tmp_path / "results.json"
        with output_path.open("w") as f:
            f.write(report.model_dump_json(indent=2))

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["aggregate"]["total_cases"] == 3
