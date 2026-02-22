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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.context import ContextManager
from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, DirectiveType
from computer_use_test.agent.modules.hitl.turn_checkpoint import (
    CheckpointDecision,
    InterruptMonitor,
    TurnCheckpoint,
    TurnSummary,
)
from computer_use_test.agent.modules.router.primitive_registry import (
    PRIMITIVE_NAMES,
    ROUTER_PROMPT,
    RouterResult,
    get_primitive_prompt,
)
from computer_use_test.utils.debug import DebugOptions, TurnValidator, log_context
from computer_use_test.utils.llm_provider.base import BaseVLMProvider
from computer_use_test.utils.llm_provider.parser import AgentAction, strip_markdown
from computer_use_test.utils.rich_logger import RichLogger
from computer_use_test.utils.screen import capture_screen_pil, execute_action

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context.context_updater import ContextUpdater
    from computer_use_test.agent.modules.context.macro_turn_manager import MacroTurnManager
    from computer_use_test.agent.modules.hitl.agent_gate import AgentGate
    from computer_use_test.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
    from computer_use_test.agent.modules.knowledge import KnowledgeManager
    from computer_use_test.agent.modules.strategy import StrategyPlanner

logger = logging.getLogger(__name__)

# finish_reason values that indicate truncation across providers
_TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}


# The last turn number observed by the Router, used to detect new in-game turns.
_last_observed_turn: int | None = None

# Module-level TurnValidator singleton (created on first use)
_turn_validator: TurnValidator | None = None


def _get_turn_validator() -> TurnValidator:
    """Return (or create) the module-level TurnValidator singleton."""
    global _turn_validator  # noqa: PLW0603
    if _turn_validator is None:
        _turn_validator = TurnValidator()
    return _turn_validator


@dataclass
class QueueCheckResult:
    """Result of checking the CommandQueue for mid-turn interrupts."""

    should_stop: bool = False
    override_action: AgentAction | None = field(default=None)
    strategy_override: str | None = None


def _check_queue_for_interrupt(
    command_queue: CommandQueue | None,
    agent_gate: AgentGate | None = None,
) -> QueueCheckResult:
    """
    Check CommandQueue for user directives before VLM planning.

    Priority: STOP > PRIMITIVE_OVERRIDE > PAUSE > CHANGE_STRATEGY
    """
    if not command_queue or not command_queue.has_pending():
        return QueueCheckResult()

    result = QueueCheckResult()
    directives = command_queue.drain()

    for d in directives:
        if d.directive_type == DirectiveType.STOP:
            if agent_gate:
                from computer_use_test.agent.modules.hitl.agent_gate import AgentState

                agent_gate.set_state(AgentState.STOPPED)
            result.should_stop = True
            return result

        elif d.directive_type == DirectiveType.PRIMITIVE_OVERRIDE:
            try:
                payload = json.loads(d.payload) if isinstance(d.payload, str) else d.payload
                result.override_action = AgentAction(
                    action=payload.get("action", "click"),
                    x=int(payload.get("x", 0)),
                    y=int(payload.get("y", 0)),
                    end_x=int(payload.get("end_x", 0)),
                    end_y=int(payload.get("end_y", 0)),
                    button=payload.get("button", "left"),
                    key=payload.get("key", ""),
                    text=payload.get("text", ""),
                    reasoning=f"[HITL Override] {payload.get('reasoning', 'User command')}",
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Invalid primitive override payload: {e}")

        elif d.directive_type == DirectiveType.PAUSE:
            RichLogger.get().hitl_event("PAUSE", "Waiting for RESUME...")
            if agent_gate:
                from computer_use_test.agent.modules.hitl.agent_gate import AgentState

                agent_gate.set_state(AgentState.PAUSED)
            command_queue.wait(timeout=None)
            if agent_gate:
                agent_gate.set_state(AgentState.RUNNING)
            for rd in command_queue.drain():
                if rd.directive_type == DirectiveType.STOP:
                    if agent_gate:
                        agent_gate.set_state(AgentState.STOPPED)
                    result.should_stop = True
                    return result
                elif rd.directive_type == DirectiveType.CHANGE_STRATEGY:
                    result.strategy_override = rd.payload

        elif d.directive_type == DirectiveType.CHANGE_STRATEGY:
            result.strategy_override = d.payload

    return result


def route_primitive(
    provider: BaseVLMProvider,
    pil_image,
) -> RouterResult:
    """
    Use VLM to classify a screenshot and select the appropriate primitive.

    Also reads the in-game turn number from the top-right corner of the
    screenshot and detects when a new game turn has started.

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen

    Returns:
        RouterResult with primitive name, reasoning, and turn metadata
    """
    # TODO: 추후 턴 및 버튼 인식의 정확도와 속도를 높이기 위해,
    #       단순 VLM 프롬프팅 대신 OCR을 도입하거나 가벼운 Small VLM(sVLM)을
    #       파인튜닝하여 대체할 것.
    global _last_observed_turn  # noqa: PLW0603

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
            logger.warning(f"Router response TRUNCATED (finish_reason={response.finish_reason}). JSON is likely incomplete. Raw response:\n{response.content}")

        content = strip_markdown(response.content)
        data = json.loads(content)

        selected = data.get("primitive", "")
        reasoning = data.get("reasoning", "")

        if selected not in PRIMITIVE_NAMES:
            logger.warning(f"Router returned unknown primitive '{selected}', defaulting to unit_ops_primitive")
            selected = "unit_ops_primitive"

        # --- Turn recognition ---
        raw_turn = data.get("turn_number")
        observed_turn: int | None = None
        if isinstance(raw_turn, int):
            observed_turn = raw_turn
        elif isinstance(raw_turn, float) and raw_turn == int(raw_turn):
            observed_turn = int(raw_turn)

        is_new_turn = False
        if observed_turn is not None:
            if _last_observed_turn is not None and observed_turn > _last_observed_turn:
                is_new_turn = True
                logger.debug(f"New game turn detected: {_last_observed_turn} → {observed_turn}")
            _last_observed_turn = observed_turn

        logger.debug(f"Router selected: {selected} (turn={observed_turn}, new={is_new_turn})")

        return RouterResult(
            primitive=selected,
            reasoning=reasoning,
            observed_turn=observed_turn,
            is_new_turn=is_new_turn,
        )

    except (json.JSONDecodeError, KeyError, RuntimeError) as e:
        logger.error(f"Router failed to parse response: {e}")
        if response is not None:
            logger.error(f"Raw response:\n{response.content}")
            if response.finish_reason in _TRUNCATION_REASONS:
                logger.error(f"Response was truncated (finish_reason={response.finish_reason}) -- this is the likely cause of the parse failure.")
        logger.error("Defaulting to unit_ops_primitive")
        return RouterResult(primitive="unit_ops_primitive")


def plan_action(
    provider: BaseVLMProvider,
    pil_image,
    primitive_name: str,
    normalizing_range: int = 1000,
    high_level_strategy: str | None = None,
    context_string: str | None = None,
    hitl_directive: str | None = None,
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
        hitl_directive: Optional micro-level HITL directive (e.g., "병영을 최우선 선택")

    Returns:
        AgentAction with normalized coordinates, or None on failure
    """
    instruction = get_primitive_prompt(
        primitive_name,
        normalizing_range,
        high_level_strategy=high_level_strategy,
        context=context_string or "현재 게임 상태 정보 없음",
        hitl_directive=hitl_directive,
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
    macro_turn_manager: MacroTurnManager | None = None,
    state_bridge: AgentStateBridge | None = None,
    context_updater: ContextUpdater | None = None,
    debug_options: DebugOptions | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
) -> TurnSummary | None:
    """
    Execute one full game turn.

    Flow:
        0. Strategy refinement (if HITL input provided)
        1. Capture screenshot (observation)
        2. Route: router_provider classifies game state → selects primitive
           2a. Simultaneously submit screenshot to ContextUpdater (background)
        3. Plan: planner_provider generates action with normalized coordinates
           (reads the *latest* context — may already include background updates)
        4. Execute: Convert coords and execute via PyAutoGUI
        5. Record action in context
        6. Macro-turn tracking (turn-change detected by Router metadata)

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
        context_updater: Optional ContextUpdater for background screenshot analysis
        debug_options: Optional DebugOptions to enable context logging / turn validation

    Returns:
        TurnSummary with turn result details, or None on critical failure
    """
    rl = RichLogger.get()

    # Get or create context manager singleton
    ctx = context_manager or ContextManager.get_instance()
    dbg = debug_options or DebugOptions.none()

    # Step 0: Strategy generation/refinement
    strategy_string = None
    if strategy_planner:
        try:
            structured_strategy = strategy_planner.generate_strategy(
                context=ctx,
                human_input=high_level_strategy,
            )
            ctx.set_strategy(structured_strategy)
            strategy_string = structured_strategy.to_prompt_string()
            rl.strategy_update(structured_strategy.victory_goal.value, strategy_string or "")
        except Exception as e:
            logger.warning(f"Strategy generation failed: {e}, using fallback")
            strategy_string = high_level_strategy
    elif high_level_strategy:
        strategy_string = high_level_strategy
    else:
        strategy_string = ctx.get_strategy_string()

    # Step 1: Observation
    pil_image, screen_w, screen_h = capture_screen_pil()

    # Step 2: Routing (also reads turn number from screen)
    router_result = route_primitive(router_provider, pil_image)
    primitive_name = router_result.primitive
    macro_turn = macro_turn_manager.macro_turn_number if macro_turn_manager else 1
    rl.route_result(primitive_name, router_result.reasoning, router_result.observed_turn, macro_turn, turn_number)

    # Step 2a: Turn-number validation
    if dbg.validate_turns:
        _get_turn_validator().validate(
            observed_turn=router_result.observed_turn,
            is_new_turn=router_result.is_new_turn,
            micro_turn=turn_number,
            macro_turn=macro_turn,
        )

    # Step 2b: Submit screenshot to ContextUpdater (non-blocking background analysis).
    # The updater will parse game-state info (gold, science, era, etc.) and write
    # it into ContextManager while we proceed with planning.
    if context_updater:
        context_updater.submit(pil_image)

    # Handle game-turn transition detected by router
    if router_result.is_new_turn and macro_turn_manager:
        macro_summary = macro_turn_manager.handle_macro_turn_end()
        logger.debug(f"Macro-turn {macro_summary.macro_turn_number} ended (turn {router_result.observed_turn}): {macro_summary.llm_summary[:80]}...")
        if state_bridge:
            state_bridge.update_macro_turn(macro_summary.macro_turn_number + 1)

    # Update context with current primitive
    ctx.set_current_primitive(primitive_name)

    # Step 2c: Check queue for mid-turn user directives (before VLM call)
    if state_bridge:
        state_bridge.broadcast_agent_phase("명령 확인 중...")

    queue_result = _check_queue_for_interrupt(command_queue, agent_gate=agent_gate)

    if queue_result.should_stop:
        rl.hitl_event("STOP", "Directive received mid-turn. Aborting.")
        return None

    primitive_hint = ""
    if queue_result.strategy_override:
        if strategy_planner:
            try:
                refined = strategy_planner.refine_strategy(queue_result.strategy_override, ctx)
                ctx.set_strategy(refined)
                strategy_string = refined.to_prompt_string()
                primitive_hint = refined.primitive_hint
                rl.strategy_update(refined.victory_goal.value, queue_result.strategy_override or "")
            except Exception as e:
                logger.warning(f"HITL strategy refinement failed: {e}, using raw override")
                strategy_string = f"[사용자 최우선 지시] {queue_result.strategy_override}\n\n{strategy_string or ''}"
        else:
            strategy_string = f"[사용자 최우선 지시] {queue_result.strategy_override}\n\n{strategy_string or ''}"
        logger.debug(f"Strategy overridden by user: {queue_result.strategy_override[:80]}...")

    if queue_result.override_action:
        # Skip VLM planning — execute user's primitive override directly
        action = queue_result.override_action
        rl.hitl_event("OVERRIDE", f"{action.action} ({action.x}, {action.y})")
        if state_bridge:
            state_bridge.update_current_action(primitive_name, f"[HITL] {action.action} ({action.x}, {action.y})", action.reasoning)
            state_bridge.broadcast_agent_phase("사용자 명령 실행 중")
    else:
        # Step 3: Planning with context (normal VLM flow)
        # NOTE: get_context_for_primitive() reads the *latest* context, which may
        # already include updates from the background ContextUpdater.
        if state_bridge:
            state_bridge.broadcast_agent_phase("추론 중...")
        logger.debug(f"Planning: generating action for {primitive_name}...")

        context_string = ctx.get_context_for_primitive(primitive_name)

        if dbg.log_context:
            log_context(primitive_name, strategy_string, context_string)
        else:
            logger.debug(f"Context for primitive:\n{context_string}")

        # Optionally augment with knowledge
        if knowledge_manager and knowledge_manager.is_available():
            query = f"{primitive_name} 전략 가이드"
            try:
                knowledge_result = knowledge_manager.query(query, top_k=2)
                if not knowledge_result.is_empty():
                    knowledge_section = knowledge_result.to_prompt_string(max_chunks=2, max_tokens=300)
                    context_string = f"{context_string}\n\n{knowledge_section}"
                    logger.debug(f"Added {len(knowledge_result.chunks)} knowledge chunks")
            except Exception as e:
                logger.warning(f"Knowledge retrieval failed: {e}")

        action = plan_action(
            planner_provider,
            pil_image,
            primitive_name,
            normalizing_range,
            high_level_strategy=strategy_string,
            context_string=context_string,
            hitl_directive=primitive_hint if primitive_hint else None,
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

        extra = {}
        if action.action == "drag":
            extra["End Coords"] = f"({action.end_x}, {action.end_y})"
        if action.key:
            extra["Key"] = action.key
        if action.text:
            extra["Text"] = action.text
        rl.action_result(
            action_type=action.action,
            coords=(action.x, action.y),
            reasoning=action.reasoning or "",
            extra=extra or None,
        )

        # Update state bridge with current action info
        if state_bridge:
            action_desc = f"{action.action} ({action.x}, {action.y})"
            state_bridge.update_current_action(primitive_name, action_desc, action.reasoning or "")

    # Step 4: Execution
    if delay_before_action > 0:
        time.sleep(delay_before_action)

    try:
        execute_action(action, screen_w, screen_h, normalizing_range)
        rl.execution_status(True)
        execution_result = "success"
        error_message = ""
    except Exception as e:
        rl.execution_status(False, str(e))
        execution_result = "failed"
        error_message = str(e)

    # Step 5: Record action in context
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

    # Step 6: Macro-turn tracking (fallback: keyword-based detection from old flow)
    if macro_turn_manager:
        macro_turn_manager.record_micro_turn(primitive_name, action.reasoning or "")
        # Keyword-based fallback detection (supplements router-based detection above)
        if not router_result.is_new_turn and macro_turn_manager.is_next_turn_action(primitive_name, action):
            macro_summary = macro_turn_manager.handle_macro_turn_end()
            logger.debug(f"Macro-turn {macro_summary.macro_turn_number} ended (keyword fallback): {macro_summary.llm_summary[:80]}...")
            if state_bridge:
                state_bridge.update_macro_turn(macro_summary.macro_turn_number + 1)

    rl.turn_summary(turn_number, primitive_name, action.action, execution_result == "success")
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
    command_queue: CommandQueue | None = None,
    macro_turn_manager: MacroTurnManager | None = None,
    state_bridge: AgentStateBridge | None = None,
    context_updater: ContextUpdater | None = None,
    debug_options: DebugOptions | None = None,
    agent_gate: AgentGate | None = None,
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
        context_updater: Optional ContextUpdater for background screenshot analysis
        debug_options: Optional DebugOptions to enable context logging / turn validation
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

    rl = RichLogger.get()
    logger.info(
        f"Running {num_turns} turn(s) with router={router_provider.get_provider_name()}/{router_provider.model}, "
        f"planner={planner_provider.get_provider_name()}/{planner_provider.model}"
    )

    strategy_changed = False

    try:
        for turn in range(1, num_turns + 1):
            rl.turn_header(turn, num_turns)

            # Pass strategy on first turn or when it was changed at a checkpoint
            turn_strategy = high_level_strategy if (turn == 1 or strategy_changed) else None
            strategy_changed = False

            # Update state bridge with micro-turn counter
            if state_bridge:
                state_bridge.update_micro_turn(turn)

            # Sync gate state at turn start
            if agent_gate:
                from computer_use_test.agent.modules.hitl.agent_gate import AgentState

                agent_gate.set_state(AgentState.RUNNING)

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
                macro_turn_manager=macro_turn_manager,
                state_bridge=state_bridge,
                context_updater=context_updater,
                debug_options=debug_options,
                command_queue=command_queue,
                agent_gate=agent_gate,
            )

            if summary is None or not summary.success:
                logger.warning(f"Turn {turn} failed. Stopping.")
                break

            # Check command queue at checkpoint (preferred over InterruptMonitor)
            stop_requested = False
            if command_queue and command_queue.has_pending():
                directives = command_queue.drain()
                for d in directives:
                    if d.directive_type == DirectiveType.STOP:
                        rl.hitl_event("STOP", "Directive received from command queue.")
                        if agent_gate:
                            agent_gate.set_state(AgentState.STOPPED)
                        stop_requested = True
                        break
                    elif d.directive_type == DirectiveType.PAUSE:
                        rl.hitl_event("PAUSE", "Waiting for RESUME...")
                        if agent_gate:
                            agent_gate.set_state(AgentState.PAUSED)
                        command_queue.wait(timeout=None)
                        if agent_gate:
                            agent_gate.set_state(AgentState.RUNNING)
                        # After wake-up, drain again to consume RESUME
                        for rd in command_queue.drain():
                            if rd.directive_type == DirectiveType.STOP:
                                if agent_gate:
                                    agent_gate.set_state(AgentState.STOPPED)
                                stop_requested = True
                                break
                            elif rd.directive_type == DirectiveType.CHANGE_STRATEGY:
                                high_level_strategy = rd.payload
                                strategy_changed = True
                                rl.strategy_update("User Override", rd.payload or "")
                    elif d.directive_type == DirectiveType.CHANGE_STRATEGY:
                        high_level_strategy = d.payload
                        strategy_changed = True
                        rl.strategy_update("User Override", d.payload or "")
            if stop_requested:
                break

            # Fallback: Check for async interrupt between turns (legacy)
            if not command_queue and interrupt_monitor and checkpoint and interrupt_monitor.is_interrupted():
                decision = checkpoint.prompt(summary)

                if decision == CheckpointDecision.STOP:
                    rl.hitl_event("STOP", "User requested stop at checkpoint.")
                    break
                elif decision == CheckpointDecision.CHANGE_STRATEGY:
                    new_strategy = checkpoint.prompt_new_strategy()
                    if new_strategy:
                        high_level_strategy = new_strategy
                        strategy_changed = True
                        rl.strategy_update("User Override", new_strategy)
                    else:
                        logger.debug("No strategy provided, continuing with current strategy.")

                interrupt_monitor.reset()

            if turn < num_turns:
                time.sleep(delay_between_turns)
    finally:
        if interrupt_monitor:
            interrupt_monitor.stop()

    rl.console.print("[bold green]All turns completed.[/bold green]")
    logger.debug(f"Context summary: {ctx}")
