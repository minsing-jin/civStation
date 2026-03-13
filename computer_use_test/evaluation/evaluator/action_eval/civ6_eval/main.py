"""
Main runner script for Civilization VI Static Primitive Evaluator.

This script:
1. Loads ground truth test cases from JSON
2. Initializes all primitives and the evaluation pipeline
3. Runs evaluation on each test case
4. Calculates and reports final metrics
"""

import argparse
import json
import sys
from pathlib import Path

from computer_use_test.agent.models.schema import (
    Action,
    ClickAction,
    DragAction,
    KeyPressAction,
)
from computer_use_test.agent.modules.primitive.primitives import (
    CityProductionPrimitive,
    CultureDecisionPrimitive,
    PopupPrimitive,
    ResearchSelectPrimitive,
    UnitOpsPrimitive,
)
from computer_use_test.agent.modules.router.router import Civ6MockRouter
from computer_use_test.evaluation.evaluator.action_eval.civ6_eval.civ6_impl import (
    Civ6StaticEvaluator,
)
from computer_use_test.evaluation.evaluator.action_eval.interfaces import GroundTruth
from computer_use_test.utils.llm_provider import create_provider, get_available_providers
from computer_use_test.utils.llm_provider.base import BaseVLMProvider


def load_ground_truth_from_json(json_path: str) -> list[GroundTruth]:
    """
    Load test cases from JSON file.

    The JSON file should contain a list of test cases with the following structure:
    [
        {
            "screenshot": "path/to/screenshot.png",
            "gt_primitive": "primitive_name",
            "gt_actions": [
                {"type": "click", "x": 100, "y": 200},
                {"type": "press", "keys": ["esc"]},
                {"type": "drag", "start_x": 10, "start_y": 20, "end_x": 100, "end_y": 200}
            ]
        }
    ]

    Args:
        json_path: Path to the JSON file containing test cases

    Returns:
        List of GroundTruth objects

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON is malformed
        KeyError: If required fields are missing
    """
    json_file = Path(json_path)
    if not json_file.exists():
        raise FileNotFoundError(f"Test set file not found: {json_path}")

    with open(json_file) as f:
        data = json.load(f)

    ground_truths = []
    for idx, item in enumerate(data):
        try:
            # Parse actions based on discriminator type field
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
                else:
                    print(f"Warning: Unknown action type '{action_type}' in test case {idx}")
                    continue

            ground_truths.append(
                GroundTruth(
                    screenshot_path=item["screenshot"],
                    expected_primitive=item["gt_primitive"],
                    expected_actions=actions,
                )
            )

        except KeyError as e:
            print(f"Error: Missing required field {e} in test case {idx}")
            raise

    return ground_truths


def main(
    json_path: str = "test_set.json",
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> None:
    """
    Main evaluation loop.

    Args:
        json_path: Path to JSON file containing test cases (default: test_set.json)
        provider_name: Optional VLM provider ("claude", "gemini", "gpt", "mock").
                      If None, uses deterministic mocking without API calls.
        api_key: API key for the provider (optional if using environment variables)
        model: Model identifier (optional, uses provider default)
    """
    print("=" * 60)
    print("Civilization VI Static Primitive Evaluator")
    print("=" * 60)
    print()

    # Initialize VLM provider if specified
    vlm_provider: BaseVLMProvider | None = None
    if provider_name:
        try:
            vlm_provider = create_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=model,
            )
            print(f"Using VLM Provider: {vlm_provider.get_provider_name()}")
            print(f"Model: {vlm_provider.model}")
            print()
        except Exception as e:
            print(f"Warning: Failed to initialize VLM provider: {e}")
            print("Falling back to deterministic mocking\n")
            vlm_provider = None
    else:
        print("Using deterministic mocking (no VLM provider)")
        print()

    # Initialize all primitives with optional VLM provider
    primitives = {
        "unit_ops_primitive": UnitOpsPrimitive(vlm_provider=vlm_provider),
        "popup_primitive": PopupPrimitive(vlm_provider=vlm_provider),
        "research_select_primitive": ResearchSelectPrimitive(vlm_provider=vlm_provider),
        "city_production_primitive": CityProductionPrimitive(vlm_provider=vlm_provider),
        "culture_decision_primitive": CultureDecisionPrimitive(vlm_provider=vlm_provider),
    }

    # Create evaluation pipeline
    router = Civ6MockRouter()
    evaluator = Civ6StaticEvaluator(router, primitives)

    # Load test set
    try:
        test_set = load_ground_truth_from_json(json_path)
        print(f"Loaded {len(test_set)} test cases from {json_path}\n")
    except Exception as e:
        print(f"Error loading test set: {e}")
        sys.exit(1)

    if not test_set:
        print("No test cases found. Exiting.")
        sys.exit(0)

    # Evaluation loop
    print("Running evaluation...\n")
    results = []

    for idx, gt in enumerate(test_set, 1):
        print(f"[{idx}/{len(test_set)}] {gt.screenshot_path}")

        # Evaluate single test case
        result = evaluator.evaluate_single(gt)
        results.append(result)

        # Print detailed results
        primitive_status = "✓" if result.primitive_match else "✗"
        action_status = "✓" if result.action_sequence_match else "✗"

        print(f"  Primitive: {primitive_status} (Expected: {gt.expected_primitive}, Got: {result.selected_primitive})")
        print(f"  Actions:   {action_status} ({len(gt.expected_actions)} actions)")
        print()

    # Calculate final metrics
    print("=" * 60)
    print("Final Metrics")
    print("=" * 60)

    primitive_correct = sum(r.primitive_match for r in results)
    action_correct = sum(r.action_sequence_match for r in results)
    both_correct = sum(r.primitive_match and r.action_sequence_match for r in results)

    total = len(results)

    primitive_accuracy = (primitive_correct / total) * 100 if total > 0 else 0
    action_accuracy = (action_correct / total) * 100 if total > 0 else 0
    overall_accuracy = (both_correct / total) * 100 if total > 0 else 0

    print(f"Primitive Selection Accuracy: {primitive_accuracy:6.2f}%  ({primitive_correct}/{total})")
    print(f"Action Sequence Accuracy:     {action_accuracy:6.2f}%  ({action_correct}/{total})")
    print(f"Overall Accuracy (Both):      {overall_accuracy:6.2f}%  ({both_correct}/{total})")
    print()

    # Summary
    if overall_accuracy == 100.0:
        print("✓ Perfect score! All test cases passed.")
    elif overall_accuracy >= 80.0:
        print("✓ Good performance. Most test cases passed.")
    elif overall_accuracy >= 50.0:
        print("⚠ Moderate performance. Consider improving the model.")
    else:
        print("✗ Low performance. Significant improvements needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Civilization VI AI Agent with Static Primitive Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run with mock (no API calls)
  python -m computer_use_test.evaluator.static_eval.civ6_eval.main

  # Run with Claude
  python -m computer_use_test.evaluator.static_eval.civ6_eval.main --provider claude

  # Run with GPT-4o mini
  python -m computer_use_test.evaluator.static_eval.civ6_eval.main --provider gpt --model gpt-4o-mini

  # Run with Gemini and custom test file
  python -m computer_use_test.evaluator.static_eval.civ6_eval.main --provider gemini --test-file my_tests.json

Available providers: {", ".join(get_available_providers().keys())}
        """,
    )

    parser.add_argument(
        "test_file",
        nargs="?",
        default="test_set.json",
        help="Path to JSON file with test cases (default: test_set.json)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=list(get_available_providers().keys()),
        help="VLM provider to use (default: mock/deterministic)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="API key for the provider (or set via environment variable)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model identifier (uses provider default if not specified)",
    )

    args = parser.parse_args()

    # Handle relative paths
    json_file = args.test_file
    if not Path(json_file).is_absolute():
        script_dir = Path(__file__).parent
        json_file = str(script_dir / json_file)

    main(
        json_path=json_file,
        provider_name=args.provider,
        api_key=args.api_key,
        model=args.model,
    )
