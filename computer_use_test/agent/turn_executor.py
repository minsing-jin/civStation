"""
Turn Executor — Core execution logic for the Civilization VI agent.

Executes one full game turn using the primitive-based agent architecture:
1. Observation: Capture screenshot
2. Routing: VLM analyzes screenshot → selects appropriate primitive
3. Planning: Primitive's VLM analyzes screenshot → generates action
4. Execution: Convert normalized coordinates → execute via PyAutoGUI

Supports:
- Separate providers/models for routing and planning
- Context management (ContextManager singleton)
- Strategy planning (HITL and autonomous modes)
- Knowledge retrieval (RAG-based)

Contains the pure execution functions that run the agent loop:
- route_primitive: VLM classifies screenshot → selects primitive
- plan_action: VLM generates action for the selected primitive
- run_one_turn: Single turn (observe → route → plan → execute → record)
- run_multi_turn: Multi-turn loop with HITL checkpoints
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.context import ContextManager
from computer_use_test.agent.modules.hitl.turn_checkpoint import (
    CheckpointDecision,
    InterruptMonitor,
    TurnCheckpoint,
    TurnSummary,
)
from computer_use_test.agent.modules.router.primitive_registry import (
    PRIMITIVE_NAMES,
    ROUTER_PROMPT,
    get_primitive_prompt,
)
from computer_use_test.utils.llm_provider.base import BaseVLMProvider
from computer_use_test.utils.llm_provider.parser import AgentAction, strip_markdown
from computer_use_test.utils.screen import capture_screen_pil, execute_action

if TYPE_CHECKING:
    from computer_use_test.agent.modules.knowledge import KnowledgeManager
    from computer_use_test.agent.modules.strategy import StrategyPlanner

logger = logging.getLogger(__name__)

# finish_reason values that indicate truncation across providers
_TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}


def route_primitive(
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

        content = strip_markdown(response.content)
        data = json.loads(content)

        selected = data.get("primitive", "")
        reasoning = data.get("reasoning", "")

        if selected not in PRIMITIVE_NAMES:
            logger.warning(f"Router returned unknown primitive '{selected}', defaulting to unit_ops_primitive")
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
    high_level_strategy: str | None = None,
    context_string: str | None = None,
) -> AgentAction | None:
    """
    Use VLM to generate the next action for the selected primitive.

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen
        primitive_name: Selected primitive (determines the prompt)
        normalizing_range: Coordinate normalization range
        high_level_strategy: Optional high-level strategy/goal to guide action selection
        context_string: Optional context string from ContextManager

    Returns:
        AgentAction with normalized coordinates, or None on failure
    """
    instruction = get_primitive_prompt(
        primitive_name,
        normalizing_range,
        high_level_strategy=high_level_strategy,
        context=context_string or "현재 게임 상태 정보 없음",
    )

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
    high_level_strategy: str | None = None,
    context_manager: ContextManager | None = None,
    strategy_planner: StrategyPlanner | None = None,
    knowledge_manager: KnowledgeManager | None = None,
    turn_number: int = 0,
) -> TurnSummary | None:
    """
    Execute one full game turn.

    Flow:
        0. Strategy refinement (if HITL input provided)
        1. Capture screenshot (observation)
        2. Route: router_provider classifies game state → selects primitive
        3. Plan: planner_provider generates action with normalized coordinates
        4. Execute: Convert coords and execute via PyAutoGUI
        5. Record action in context

    Args:
        router_provider: VLM provider for routing (primitive selection)
        planner_provider: VLM provider for planning (action generation)
        normalizing_range: Coordinate normalization range (default 1000)
        delay_before_action: Seconds to wait before executing the action
        high_level_strategy: Optional high-level strategy to guide action selection
        context_manager: Optional ContextManager instance (uses singleton if None)
        strategy_planner: Optional StrategyPlanner for HITL strategy refinement
        knowledge_manager: Optional KnowledgeManager for RAG-based assistance
        turn_number: Current turn number (used in TurnSummary)

    Returns:
        TurnSummary with turn result details, or None on critical failure
    """
    logger.info("=" * 50)
    logger.info("Starting one-turn execution")
    logger.info("=" * 50)

    # Get or create context manager singleton
    ctx = context_manager or ContextManager.get_instance()

    # Step 0: Strategy generation/refinement
    strategy_string = None
    if strategy_planner:
        try:
            # generate_strategy() handles both HITL and autonomous modes:
            # - HITL mode: prompts user for input (voice/text), then refines
            # - Autonomous mode: generates strategy from context alone
            structured_strategy = strategy_planner.generate_strategy(
                context=ctx,
                human_input=high_level_strategy,  # None → will prompt user in HITL mode
            )
            ctx.set_strategy(structured_strategy)
            strategy_string = structured_strategy.to_prompt_string()
            logger.info(f"Strategy refined: {structured_strategy.victory_goal.value} victory")
        except Exception as e:
            logger.warning(f"Strategy generation failed: {e}, using fallback")
            strategy_string = high_level_strategy
    elif high_level_strategy:
        # Use raw strategy string if no planner
        strategy_string = high_level_strategy
    else:
        # Use existing strategy from context
        strategy_string = ctx.get_strategy_string()

    # Step 1: Observation
    logger.info("[1/5] Capturing screenshot...")
    pil_image, screen_w, screen_h = capture_screen_pil()
    logger.info(f"  Screen: {screen_w}x{screen_h} (logical)")

    # Step 2: Routing
    logger.info("[2/5] Routing: analyzing game state...")
    primitive_name = route_primitive(router_provider, pil_image)
    logger.info(f"  Selected primitive: {primitive_name}")

    # Update context with current primitive
    ctx.set_current_primitive(primitive_name)

    # Step 3: Planning with context
    logger.info(f"[3/5] Planning: generating action for {primitive_name}...")

    # Get context string for this primitive
    context_string = ctx.get_context_for_primitive(primitive_name)
    logger.debug(f"Context for primitive:\n{context_string}")

    # Optionally augment with knowledge
    if knowledge_manager and knowledge_manager.is_available():
        # Query relevant knowledge for this primitive
        query = f"{primitive_name} 전략 가이드"
        try:
            knowledge_result = knowledge_manager.query(query, top_k=2)
            if not knowledge_result.is_empty():
                knowledge_section = knowledge_result.to_prompt_string(max_chunks=2, max_tokens=300)
                context_string = f"{context_string}\n\n{knowledge_section}"
                logger.info(f"  Added {len(knowledge_result.chunks)} knowledge chunks")
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")

    action = plan_action(
        planner_provider,
        pil_image,
        primitive_name,
        normalizing_range,
        high_level_strategy=strategy_string,
        context_string=context_string,
    )

    if action is None:
        logger.error("  VLM returned no action. Turn aborted.")
        ctx.record_action(
            action_type="none",
            primitive=primitive_name,
            result="failed",
            error_message="VLM returned no action",
        )
        return TurnSummary(
            turn_number=turn_number,
            primitive=primitive_name,
            action_type="none",
            success=False,
            error_message="VLM returned no action",
        )

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

    logger.info("[4/5] Executing action...")
    try:
        execute_action(action, screen_w, screen_h, normalizing_range)
        logger.info("  Action executed.")
        execution_result = "success"
        error_message = ""
    except Exception as e:
        logger.error(f"  Action execution failed: {e}")
        execution_result = "failed"
        error_message = str(e)

    # Step 5: Record action in context
    logger.info("[5/5] Recording action in context...")
    ctx.record_action(
        action_type=action.action,
        primitive=primitive_name,
        x=action.x,
        y=action.y,
        end_x=action.end_x if action.action == "drag" else 0,
        end_y=action.end_y if action.action == "drag" else 0,
        key=action.key,
        text=action.text,
        result=execution_result,
        error_message=error_message,
    )

    logger.info("Turn complete.")
    return TurnSummary(
        turn_number=turn_number,
        primitive=primitive_name,
        action_type=action.action,
        success=execution_result == "success",
        reasoning=action.reasoning or "",
        error_message=error_message,
        coords=(action.x, action.y),
    )


def run_multi_turn(
    router_provider: BaseVLMProvider,
    planner_provider: BaseVLMProvider,
    num_turns: int = 1,
    normalizing_range: int = 1000,
    delay_between_turns: float = 1.0,
    delay_before_action: float = 0.5,
    high_level_strategy: str | None = None,
    context_manager: ContextManager | None = None,
    strategy_planner: StrategyPlanner | None = None,
    knowledge_manager: KnowledgeManager | None = None,
    hitl_mode: str | None = None,
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
        high_level_strategy: Optional high-level strategy to guide action selection
        context_manager: Optional ContextManager instance
        strategy_planner: Optional StrategyPlanner for HITL strategy refinement
        knowledge_manager: Optional KnowledgeManager for RAG-based assistance
        hitl_mode: HITL mode ("async" for Enter-key interrupt, None for default)
    """
    # Get or create context manager singleton
    ctx = context_manager or ContextManager.get_instance()

    # Set up async interrupt monitoring if requested
    interrupt_monitor: InterruptMonitor | None = None
    checkpoint: TurnCheckpoint | None = None
    if hitl_mode == "async":
        interrupt_monitor = InterruptMonitor()
        checkpoint = TurnCheckpoint()
        interrupt_monitor.start()

    logger.info(
        f"Running {num_turns} turn(s) with "
        f"router={router_provider.get_provider_name()}/{router_provider.model}, "
        f"planner={planner_provider.get_provider_name()}/{planner_provider.model}"
    )

    strategy_changed = False

    try:
        for turn in range(1, num_turns + 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"TURN {turn}/{num_turns}")
            logger.info(f"{'=' * 60}")

            # Pass strategy on first turn or when it was changed at a checkpoint
            turn_strategy = high_level_strategy if (turn == 1 or strategy_changed) else None
            strategy_changed = False

            summary = run_one_turn(
                router_provider=router_provider,
                planner_provider=planner_provider,
                normalizing_range=normalizing_range,
                delay_before_action=delay_before_action,
                high_level_strategy=turn_strategy,
                context_manager=ctx,
                strategy_planner=strategy_planner,
                knowledge_manager=knowledge_manager,
                turn_number=turn,
            )

            if summary is None or not summary.success:
                logger.warning(f"Turn {turn} failed. Stopping.")
                break

            # Check for async interrupt between turns
            if interrupt_monitor and checkpoint and interrupt_monitor.is_interrupted():
                decision = checkpoint.prompt(summary)

                if decision == CheckpointDecision.STOP:
                    logger.info("User requested stop at checkpoint.")
                    break
                elif decision == CheckpointDecision.CHANGE_STRATEGY:
                    new_strategy = checkpoint.prompt_new_strategy()
                    if new_strategy:
                        high_level_strategy = new_strategy
                        strategy_changed = True
                        logger.info(f"Strategy changed to: {new_strategy}")
                    else:
                        logger.info("No strategy provided, continuing with current strategy.")

                interrupt_monitor.reset()

            if turn < num_turns:
                logger.info(f"Waiting {delay_between_turns}s before next turn...")
                time.sleep(delay_between_turns)
    finally:
        if interrupt_monitor:
            interrupt_monitor.stop()

    logger.info("\nAll turns completed.")
    logger.info(f"Context summary: {ctx}")
