"""
Integration tests for Civ6 Static Primitive Evaluator with real screenshots.

These tests verify the complete evaluation pipeline using actual screenshot files
and ground truth data. Screenshots should be placed in the fixtures/screenshots directory.
"""

import json
from pathlib import Path

import pytest

from computer_use_test.agent.models.schema import (
    Action,
    AgentPlan,
    ClickAction,
    DragAction,
    KeyPressAction,
)
from computer_use_test.agent.modules.primitive.primitives import (
    CityProductionPrimitive,
    CultureDecisionPrimitive,
    PopupPrimitive,
    ResearchSelectPrimitive,
    ScienceDecisionPrimitive,
    UnitOpsPrimitive,
)
from computer_use_test.agent.modules.router.router import Civ6MockRouter
from computer_use_test.evaluator.civ6.static_eval import EvalResult, GroundTruth
from computer_use_test.evaluator.civ6.static_eval.civ6_eval.civ6_impl import (
    Civ6StaticEvaluator,
)

# Fixtures directory paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCREENSHOTS_DIR = FIXTURES_DIR / "screenshots"
GT_DATA_FILE = FIXTURES_DIR / "ground_truth.json"


@pytest.fixture
def primitives():
    """Create all primitive instances."""
    return {
        "unit_ops_primitive": UnitOpsPrimitive(),
        "popup_primitive": PopupPrimitive(),
        "research_select_primitive": ResearchSelectPrimitive(),
        "city_production_primitive": CityProductionPrimitive(),
        "science_decision_primitive": ScienceDecisionPrimitive(),
        "culture_decision_primitive": CultureDecisionPrimitive(),
    }


@pytest.fixture
def router():
    """Create router instance."""
    return Civ6MockRouter()


@pytest.fixture
def evaluator(router, primitives):
    """Create evaluator instance with router and primitives."""
    return Civ6StaticEvaluator(router, primitives)


@pytest.fixture
def ground_truth_data():
    """
    Load ground truth data from JSON file.

    The JSON file should be at: tests/evaluator/static_eval/civ6_eval/fixtures/ground_truth.json

    Returns:
        List of GroundTruth objects, or empty list if file doesn't exist
    """
    if not GT_DATA_FILE.exists():
        pytest.skip(f"Ground truth file not found: {GT_DATA_FILE}")

    with open(GT_DATA_FILE) as f:
        data = json.load(f)

    ground_truths = []
    for item in data:
        # Parse actions
        actions: list[Action] = []
        for action_dict in item["gt_actions"]:
            action_type = action_dict["type"]

            if action_type == "click":
                actions.append(
                    ClickAction(
                        type="click",
                        x=action_dict["x"],
                        y=action_dict["y"],
                        button=action_dict.get("button", "left"),
                    )
                )
            elif action_type == "press":
                actions.append(
                    KeyPressAction(
                        type="press",
                        keys=action_dict["keys"],
                        interval=action_dict.get("interval", 0.1),
                    )
                )
            elif action_type == "drag":
                actions.append(
                    DragAction(
                        type="drag",
                        start_x=action_dict["start_x"],
                        start_y=action_dict["start_y"],
                        end_x=action_dict["end_x"],
                        end_y=action_dict["end_y"],
                        duration=action_dict.get("duration", 0.5),
                        button=action_dict.get("button", "left"),
                    )
                )

        # Create full screenshot path
        screenshot_path = str(SCREENSHOTS_DIR / item["screenshot"])

        ground_truths.append(
            GroundTruth(
                screenshot_path=screenshot_path,
                expected_primitive=item["gt_primitive"],
                expected_actions=actions,
            )
        )

    return ground_truths


class TestPrimitiveRouting:
    """Tests for primitive routing based on screenshot filenames."""

    def test_route_science_screenshot(self, router):
        """Test that science screenshots are routed to science primitive."""
        screenshot_path = str(SCREENSHOTS_DIR / "turn_10_science.png")
        result = router.route(screenshot_path)
        assert result == "science_decision_primitive"

    def test_route_unit_screenshot(self, router):
        """Test that unit screenshots are routed to unit ops primitive."""
        screenshot_path = str(SCREENSHOTS_DIR / "unit_settler_move.png")
        result = router.route(screenshot_path)
        assert result == "unit_ops_primitive"

    def test_route_production_screenshot(self, router):
        """Test that production screenshots are routed to city production primitive."""
        screenshot_path = str(SCREENSHOTS_DIR / "city_production_queue.png")
        result = router.route(screenshot_path)
        assert result == "city_production_primitive"

    def test_route_popup_screenshot(self, router):
        """Test that popup screenshots are routed to popup primitive."""
        screenshot_path = str(SCREENSHOTS_DIR / "popup_next_turn.png")
        result = router.route(screenshot_path)
        assert result == "popup_primitive"

    def test_route_culture_screenshot(self, router):
        """Test that culture screenshots are routed to culture primitive."""
        screenshot_path = str(SCREENSHOTS_DIR / "culture_civic_choice.png")
        result = router.route(screenshot_path)
        assert result == "culture_decision_primitive"


class TestPrimitivePlanGeneration:
    """Tests for primitive plan generation."""

    def test_unit_ops_generates_valid_plan(self, primitives):
        """Test that UnitOpsPrimitive generates a valid AgentPlan."""
        primitive = primitives["unit_ops_primitive"]
        screenshot_path = str(SCREENSHOTS_DIR / "unit_test.png")

        plan = primitive.generate_plan_and_action(screenshot_path)

        assert isinstance(plan, AgentPlan)
        assert plan.primitive_name == "unit_ops_primitive"
        assert len(plan.actions) > 0
        assert plan.reasoning != ""

    def test_science_primitive_generates_valid_plan(self, primitives):
        """Test that ScienceDecisionPrimitive generates a valid AgentPlan."""
        primitive = primitives["science_decision_primitive"]
        screenshot_path = str(SCREENSHOTS_DIR / "science_test.png")

        plan = primitive.generate_plan_and_action(screenshot_path)

        assert isinstance(plan, AgentPlan)
        assert plan.primitive_name == "science_decision_primitive"
        assert len(plan.actions) > 0

    def test_plan_generation_is_deterministic(self, primitives):
        """Test that same screenshot produces same plan (deterministic mock)."""
        primitive = primitives["unit_ops_primitive"]
        screenshot_path = str(SCREENSHOTS_DIR / "unit_test.png")

        plan1 = primitive.generate_plan_and_action(screenshot_path)
        plan2 = primitive.generate_plan_and_action(screenshot_path)

        # Should produce identical plans
        assert len(plan1.actions) == len(plan2.actions)
        assert plan1.reasoning == plan2.reasoning

        # Check first action matches
        if len(plan1.actions) > 0:
            action1 = plan1.actions[0]
            action2 = plan2.actions[0]
            assert type(action1) is type(action2)


class TestEvaluationPipeline:
    """Tests for the complete evaluation pipeline."""

    def test_evaluate_single_ground_truth(self, evaluator, ground_truth_data):
        """Test evaluation of a single ground truth case."""
        if not ground_truth_data:
            pytest.skip("No ground truth data available")

        gt = ground_truth_data[0]
        result = evaluator.evaluate_single(gt)

        assert isinstance(result, EvalResult)
        assert result.screenshot_path == gt.screenshot_path
        assert result.selected_primitive in [
            "unit_ops_primitive",
            "popup_primitive",
            "research_select_primitive",
            "city_production_primitive",
            "science_decision_primitive",
            "culture_decision_primitive",
        ]
        assert isinstance(result.primitive_match, bool)
        assert isinstance(result.action_sequence_match, bool)

    def test_evaluate_all_ground_truth_cases(self, evaluator, ground_truth_data):
        """Test evaluation of all ground truth cases."""
        if not ground_truth_data:
            pytest.skip("No ground truth data available")

        results = []
        for gt in ground_truth_data:
            result = evaluator.evaluate_single(gt)
            results.append(result)

        # Basic sanity checks
        assert len(results) == len(ground_truth_data)
        assert all(isinstance(r, EvalResult) for r in results)

        # Calculate metrics
        primitive_accuracy = sum(r.primitive_match for r in results) / len(results)
        _ = sum(r.action_sequence_match for r in results) / len(
            results
        )  # TODO: action_accuracy unused. It will be implemented later.

        # Primitive routing should be 100% with mock router (keyword-based)
        assert primitive_accuracy == 1.0, "Keyword-based routing should be 100% accurate"

    def test_primitive_selection_accuracy(self, evaluator, ground_truth_data):
        """Test that primitive selection is accurate with keyword-based routing."""
        if not ground_truth_data:
            pytest.skip("No ground truth data available")

        for gt in ground_truth_data:
            result = evaluator.evaluate_single(gt)

            # With keyword-based routing, primitive should always match
            assert result.primitive_match, f"Expected {gt.expected_primitive}, got {result.selected_primitive}"

    @pytest.mark.skipif(
        not GT_DATA_FILE.exists(),
        reason="Ground truth data file not found",
    )
    def test_metrics_calculation(self, evaluator, ground_truth_data):
        """Test calculation of evaluation metrics."""
        results = [evaluator.evaluate_single(gt) for gt in ground_truth_data]

        # Calculate metrics
        total = len(results)
        primitive_correct = sum(r.primitive_match for r in results)
        action_correct = sum(r.action_sequence_match for r in results)
        both_correct = sum(r.primitive_match and r.action_sequence_match for r in results)

        primitive_accuracy = (primitive_correct / total) * 100 if total > 0 else 0
        action_accuracy = (action_correct / total) * 100 if total > 0 else 0
        overall_accuracy = (both_correct / total) * 100 if total > 0 else 0

        # Assertions
        assert 0 <= primitive_accuracy <= 100
        assert 0 <= action_accuracy <= 100
        assert 0 <= overall_accuracy <= 100


class TestScreenshotFileHandling:
    """Tests for screenshot file handling."""

    def test_screenshots_directory_exists(self):
        """Test that screenshots directory exists."""
        assert SCREENSHOTS_DIR.exists(), f"Screenshots directory not found: {SCREENSHOTS_DIR}"
        assert SCREENSHOTS_DIR.is_dir(), f"Screenshots path is not a directory: {SCREENSHOTS_DIR}"

    def test_ground_truth_references_valid_files(self, ground_truth_data):
        """Test that ground truth references existing screenshot files."""
        if not ground_truth_data:
            pytest.skip("No ground truth data available")

        for gt in ground_truth_data:
            screenshot_path = Path(gt.screenshot_path)
            # Note: Files may not exist yet, but paths should be valid
            assert screenshot_path.parent == SCREENSHOTS_DIR
            assert screenshot_path.suffix in [".png", ".jpg", ".jpeg"]

    @pytest.mark.skipif(
        not any(SCREENSHOTS_DIR.glob("*.png")) if SCREENSHOTS_DIR.exists() else True,
        reason="No screenshot files found",
    )
    def test_can_load_actual_screenshot_files(self):
        """Test that actual screenshot files can be accessed."""
        screenshots = list(SCREENSHOTS_DIR.glob("*.png"))
        assert len(screenshots) > 0, "No PNG screenshots found in fixtures directory"

        # Verify files are readable
        for screenshot in screenshots:
            assert screenshot.exists()
            assert screenshot.stat().st_size > 0, f"Screenshot {screenshot.name} is empty"


class TestEndToEndEvaluation:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_full_evaluation_workflow(self, evaluator, ground_truth_data):
        """
        Test the complete evaluation workflow from start to finish.

        This test simulates the full pipeline:
        1. Load ground truth
        2. Route to primitive
        3. Generate plan
        4. Compare with ground truth
        5. Calculate metrics
        """
        if not ground_truth_data:
            pytest.skip("No ground truth data available for integration test")

        print(f"\nRunning full evaluation on {len(ground_truth_data)} test cases...")

        results = []
        for idx, gt in enumerate(ground_truth_data, 1):
            print(f"  [{idx}/{len(ground_truth_data)}] {Path(gt.screenshot_path).name}")
            result = evaluator.evaluate_single(gt)
            results.append(result)

            print(f"    Primitive: {'✓' if result.primitive_match else '✗'}")
            print(f"    Actions:   {'✓' if result.action_sequence_match else '✗'}")

        # Calculate final metrics
        total = len(results)
        primitive_accuracy = (sum(r.primitive_match for r in results) / total) * 100
        action_accuracy = (sum(r.action_sequence_match for r in results) / total) * 100
        overall_accuracy = (sum(r.primitive_match and r.action_sequence_match for r in results) / total) * 100

        print("\nFinal Metrics:")
        print(f"  Primitive Accuracy: {primitive_accuracy:.2f}%")
        print(f"  Action Accuracy:    {action_accuracy:.2f}%")
        print(f"  Overall Accuracy:   {overall_accuracy:.2f}%")

        # Assert baseline expectations
        assert primitive_accuracy == 100.0, "Keyword routing should be 100% accurate"


# Utility function for manual testing
def evaluate_screenshot_directory(screenshots_dir: str, ground_truth_file: str, output_file: str = None):
    """
    Evaluate all screenshots in a directory against ground truth.

    This function can be called directly for manual evaluation:

    >>> from tests.evaluator.static_eval.civ6_eval.test_evaluation_integration import evaluate_screenshot_directory
    >>> evaluate_screenshot_directory(
    ...     "tests/evaluator/static_eval/civ6_eval/fixtures/screenshots",
    ...     "tests/evaluator/static_eval/civ6_eval/fixtures/ground_truth.json",
    ...     "results.json"
    ... )

    Args:
        screenshots_dir: Directory containing screenshot files
        ground_truth_file: Path to ground truth JSON file
        output_file: Optional path to save results JSON
    """
    from computer_use_test.evaluator.civ6.static_eval.civ6_eval.main import (
        load_ground_truth_from_json,
    )

    # Initialize pipeline
    primitives = {
        "unit_ops_primitive": UnitOpsPrimitive(),
        "science_decision_primitive": ScienceDecisionPrimitive(),
        "culture_decision_primitive": CultureDecisionPrimitive(),
    }
    router = Civ6MockRouter()
    evaluator = Civ6StaticEvaluator(router, primitives)

    # Load ground truth
    ground_truths = load_ground_truth_from_json(ground_truth_file)

    # Run evaluation
    results = []
    for gt in ground_truths:
        result = evaluator.evaluate_single(gt)
        results.append(result)

    # Calculate metrics
    total = len(results)
    primitive_accuracy = (sum(r.primitive_match for r in results) / total) * 100
    action_accuracy = (sum(r.action_sequence_match for r in results) / total) * 100
    overall_accuracy = (sum(r.primitive_match and r.action_sequence_match for r in results) / total) * 100

    metrics = {
        "primitive_accuracy": primitive_accuracy,
        "action_accuracy": action_accuracy,
        "overall_accuracy": overall_accuracy,
        "total_cases": total,
        "results": [
            {
                "screenshot": Path(r.screenshot_path).name,
                "primitive_match": r.primitive_match,
                "action_match": r.action_sequence_match,
                "selected_primitive": r.selected_primitive,
            }
            for r in results
        ],
    }

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
