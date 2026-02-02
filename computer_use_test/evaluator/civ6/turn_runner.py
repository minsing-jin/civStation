"""
Civilization VI One-Turn Runner.

Executes one full game turn using the primitive-based agent architecture:
1. Observation: Capture screenshot
2. Routing: VLM analyzes screenshot → selects appropriate primitive
3. Planning: Primitive's VLM analyzes screenshot → generates action
4. Execution: Convert normalized coordinates → execute via PyAutoGUI

Supports separate providers/models for routing and planning:
    # Same provider for both
    python -m computer_use_test.evaluator.civ6.turn_runner --provider claude

    # Different models for router vs planner
    python -m computer_use_test.evaluator.civ6.turn_runner --provider gemini \
        --router-model gemini-2.0-flash --planner-model gemini-2.5-pro

    # Completely different providers
    python -m computer_use_test.evaluator.civ6.turn_runner \
        --router-provider gemini --planner-provider claude
"""

import argparse
import json
import logging
import time
from typing import Optional

from computer_use_test.utils.provider import create_provider, get_available_providers
from computer_use_test.utils.provider.base import AgentAction, BaseVLMProvider
from computer_use_test.utils.prompts.civ6_prompts import (
    ROUTER_PROMPT,
    get_primitive_prompt,
)
from computer_use_test.utils.screen import capture_screen_pil, execute_action

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Available primitive names
PRIMITIVE_NAMES = [
    "unit_ops_primitive",
    "popup_primitive",
    "research_select_primitive",
    "city_production_primitive",
    "science_decision_primitive",
    "culture_decision_primitive",
]

# finish_reason values that indicate truncation across providers
_TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}


# TODO: route primitive지 screenshot을 route하는게 맞냐? 진짜 모름
def route_screenshot(
    provider: BaseVLMProvider,
    pil_image,
) -> str:
    """
    Use VLM to classify a screenshot and select the appropriate primitive.

    Sends the screenshot + ROUTER_PROMPT to the VLM, which returns
    a JSON with the selected primitive name.

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen

    Returns:
        Primitive name string (e.g. "unit_ops_primitive")
    """
    content_parts = [
        provider._build_pil_image_content(pil_image),
        provider._build_text_content(ROUTER_PROMPT),
    ]

    response = None
    try:
        # TODO: For long-horizon tasks, reduce max_tokens and remove "reasoning"
        #       field from ROUTER_PROMPT JSON format to save tokens.
        response = provider._send_to_api(
            content_parts,
            temperature=0.2,
            max_tokens=8192,
        )

        # Check for truncation BEFORE attempting JSON parse
        if response.finish_reason in _TRUNCATION_REASONS:
            logger.warning(
                f"Router response TRUNCATED (finish_reason={response.finish_reason}). "
                f"JSON is likely incomplete. Raw response:\n{response.content}"
            )

        content = provider._strip_markdown(response.content)
        data = json.loads(content)

        selected = data.get("primitive", "")
        reasoning = data.get("reasoning", "")

        if selected not in PRIMITIVE_NAMES:
            logger.warning(
                f"Router returned unknown primitive '{selected}', "
                f"defaulting to unit_ops_primitive"
            )
            selected = "unit_ops_primitive"

        logger.info(f"Router selected: {selected}")
        logger.info(f"Router reasoning: {reasoning}")

        return selected

    except (json.JSONDecodeError, KeyError, RuntimeError) as e:
        logger.error(f"Router failed to parse response: {e}")
        if response is not None:
            logger.error(f"Raw response:\n{response.content}")
            if response.finish_reason in _TRUNCATION_REASONS:
                logger.error(
                    f"Response was truncated (finish_reason={response.finish_reason}) "
                    f"-- this is the likely cause of the parse failure."
                )
        logger.error("Defaulting to unit_ops_primitive")
        return "unit_ops_primitive"


def plan_action(
    provider: BaseVLMProvider,
    pil_image,
    primitive_name: str,
    normalizing_range: int = 1000,
) -> Optional[AgentAction]:
    """
    Use VLM to generate the next action for the selected primitive.

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen
        primitive_name: Selected primitive (determines the prompt)
        normalizing_range: Coordinate normalization range

    Returns:
        AgentAction with normalized coordinates, or None on failure
    """
    instruction = get_primitive_prompt(primitive_name)

    return provider.analyze(
        pil_image=pil_image,
        instruction=instruction,
        normalizing_range=normalizing_range,
    )


def run_one_turn(
    router_provider: BaseVLMProvider,
    planner_provider: BaseVLMProvider,
    normalizing_range: int = 1000,
    delay_before_action: float = 0.5,
) -> bool:
    """
    Execute one full game turn.

    Flow:
        1. Capture screenshot (observation)
        2. Route: router_provider classifies game state → selects primitive
        3. Plan: planner_provider generates action with normalized coordinates
        4. Execute: Convert coords and execute via PyAutoGUI

    Args:
        router_provider: VLM provider for routing (primitive selection)
        planner_provider: VLM provider for planning (action generation)
        normalizing_range: Coordinate normalization range (default 1000)
        delay_before_action: Seconds to wait before executing the action

    Returns:
        True if action was executed successfully, False otherwise
    """
    logger.info("=" * 50)
    logger.info("Starting one-turn execution")
    logger.info("=" * 50)

    # Step 1: Observation
    logger.info("[1/4] Capturing screenshot...")
    pil_image, screen_w, screen_h = capture_screen_pil()
    logger.info(f"  Screen: {screen_w}x{screen_h} (logical)")

    # Step 2: Routing
    logger.info("[2/4] Routing: analyzing game state...")
    primitive_name = route_screenshot(router_provider, pil_image)
    logger.info(f"  Selected primitive: {primitive_name}")

    # Step 3: Planning
    logger.info(f"[3/4] Planning: generating action for {primitive_name}...")
    action = plan_action(
        planner_provider, pil_image, primitive_name, normalizing_range
    )

    if action is None:
        logger.error("  VLM returned no action. Turn aborted.")
        return False

    logger.info(f"  Action: {action.action}")
    logger.info(f"  Coords: ({action.x}, {action.y})")
    if action.action == "drag":
        logger.info(f"  End coords: ({action.end_x}, {action.end_y})")
    if action.key:
        logger.info(f"  Key: {action.key}")
    if action.text:
        logger.info(f"  Text: {action.text}")
    logger.info(f"  Reasoning: {action.reasoning}")

    # Step 4: Execution
    if delay_before_action > 0:
        logger.info(f"  Waiting {delay_before_action}s before execution...")
        time.sleep(delay_before_action)

    logger.info("[4/4] Executing action...")
    execute_action(action, screen_w, screen_h, normalizing_range)
    logger.info("  Action executed.")

    logger.info("Turn complete.")
    return True


def run_multi_turn(
    router_provider: BaseVLMProvider,
    planner_provider: BaseVLMProvider,
    num_turns: int = 1,
    normalizing_range: int = 1000,
    delay_between_turns: float = 1.0,
    delay_before_action: float = 0.5,
) -> None:
    """
    Execute multiple consecutive turns.

    Args:
        router_provider: VLM provider for routing
        planner_provider: VLM provider for planning
        num_turns: Number of turns to execute
        normalizing_range: Coordinate normalization range
        delay_between_turns: Seconds to wait between turns
        delay_before_action: Seconds to wait before each action
    """
    logger.info(
        f"Running {num_turns} turn(s) with "
        f"router={router_provider.get_provider_name()}/{router_provider.model}, "
        f"planner={planner_provider.get_provider_name()}/{planner_provider.model}"
    )

    for turn in range(1, num_turns + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"TURN {turn}/{num_turns}")
        logger.info(f"{'='*60}")

        success = run_one_turn(
            router_provider=router_provider,
            planner_provider=planner_provider,
            normalizing_range=normalizing_range,
            delay_before_action=delay_before_action,
        )

        if not success:
            logger.warning(f"Turn {turn} failed. Stopping.")
            break

        if turn < num_turns:
            logger.info(f"Waiting {delay_between_turns}s before next turn...")
            time.sleep(delay_between_turns)

    logger.info("\nAll turns completed.")


def main():
    available = get_available_providers()
    provider_choices = list(available.keys())

    parser = argparse.ArgumentParser(
        description="Run Civilization VI AI Agent for one or more turns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Same provider for router and planner
  python -m computer_use_test.evaluator.civ6.turn_runner --provider claude

  # Different models (same provider)
  python -m computer_use_test.evaluator.civ6.turn_runner --provider gemini \\
      --router-model gemini-2.0-flash --planner-model gemini-2.5-pro

  # Different providers for router vs planner
  python -m computer_use_test.evaluator.civ6.turn_runner \\
      --router-provider gemini --router-model gemini-3.0-flash-preview \\
      --planner-provider claude --planner-model claude-sonnet-4-20250514

Available providers: {', '.join(provider_choices)}
        """,
    )

    # Shared default provider/model
    parser.add_argument(
        "--provider", "-p",
        default=None,
        choices=provider_choices,
        help="Default VLM provider for both router and planner",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Default model for both router and planner",
    )

    # Router-specific overrides
    parser.add_argument(
        "--router-provider",
        default=None,
        choices=provider_choices,
        help="VLM provider for routing (overrides --provider)",
    )
    parser.add_argument(
        "--router-model",
        default=None,
        help="Model for routing (overrides --model)",
    )

    # Planner-specific overrides
    parser.add_argument(
        "--planner-provider",
        default=None,
        choices=provider_choices,
        help="VLM provider for planning (overrides --provider)",
    )
    parser.add_argument(
        "--planner-model",
        default=None,
        help="Model for planning (overrides --model)",
    )

    # Execution parameters
    parser.add_argument(
        "--turns", "-t",
        type=int, default=1,
        help="Number of turns to execute (default: 1)",
    )
    parser.add_argument(
        "--range",
        type=int, default=1000,
        help="Coordinate normalization range (default: 1000)",
    )
    parser.add_argument(
        "--delay-action",
        type=float, default=0.5,
        help="Delay before executing each action in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--delay-turn",
        type=float, default=1.0,
        help="Delay between turns in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Resolve provider/model for router and planner
    router_provider_name = args.router_provider or args.provider
    router_model = args.router_model or args.model
    planner_provider_name = args.planner_provider or args.provider
    planner_model = args.planner_model or args.model

    if not router_provider_name:
        parser.error(
            "--provider is required, or specify both --router-provider and --planner-provider"
        )
    if not planner_provider_name:
        parser.error(
            "--provider is required, or specify both --router-provider and --planner-provider"
        )

    # Create providers
    router_provider = create_provider(
        provider_name=router_provider_name,
        model=router_model,
    )
    logger.info(f"Router:  {router_provider.get_provider_name()} ({router_provider.model})")

    # Reuse same instance if config is identical
    if planner_provider_name == router_provider_name and planner_model == router_model:
        planner_provider = router_provider
        logger.info(f"Planner: same as router (shared instance)")
    else:
        planner_provider = create_provider(
            provider_name=planner_provider_name,
            model=planner_model,
        )
        logger.info(f"Planner: {planner_provider.get_provider_name()} ({planner_provider.model})")

    # Run
    if args.turns == 1:
        run_one_turn(
            router_provider=router_provider,
            planner_provider=planner_provider,
            normalizing_range=args.range,
            delay_before_action=args.delay_action,
        )
    else:
        run_multi_turn(
            router_provider=router_provider,
            planner_provider=planner_provider,
            num_turns=args.turns,
            normalizing_range=args.range,
            delay_between_turns=args.delay_turn,
            delay_before_action=args.delay_action,
        )


if __name__ == "__main__":
    main()