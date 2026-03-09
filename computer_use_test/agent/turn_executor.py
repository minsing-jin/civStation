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
    get_directive_for_primitive,
    get_primitive_prompt,
)
from computer_use_test.utils.debug import DebugOptions, TurnValidator, log_context
from computer_use_test.utils.image_pipeline import ImagePipelineConfig, process_image
from computer_use_test.utils.llm_provider.base import BaseVLMProvider
from computer_use_test.utils.llm_provider.parser import AgentAction, strip_markdown
from computer_use_test.utils.rich_logger import RichLogger
from computer_use_test.utils.screen import capture_screen_pil, execute_action

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context.context_updater import ContextUpdater
    from computer_use_test.agent.modules.context.macro_turn_manager import MacroTurnManager
    from computer_use_test.agent.modules.context.turn_detector import TurnDetector
    from computer_use_test.agent.modules.hitl.agent_gate import AgentGate
    from computer_use_test.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
    from computer_use_test.agent.modules.knowledge import KnowledgeManager
    from computer_use_test.agent.modules.strategy import StrategyPlanner
    from computer_use_test.agent.modules.strategy.strategy_updater import StrategyUpdater

logger = logging.getLogger(__name__)

# finish_reason values that indicate truncation across providers
_TRUNCATION_REASONS = {"max_tokens", "length", "MAX_TOKENS"}

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
    img_config: ImagePipelineConfig | None = None,
) -> RouterResult:
    """
    Use VLM to classify a screenshot and select the appropriate primitive.

    Turn detection is handled separately by TurnDetector (background thread).

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen
        img_config: Image pipeline config (defaults to ROUTER_DEFAULT)

    Returns:
        RouterResult with primitive name and reasoning
    """
    from computer_use_test.utils.image_pipeline import ROUTER_DEFAULT

    cfg = img_config or ROUTER_DEFAULT
    result = process_image(pil_image, cfg)
    prepared = result.image
    jpeg_quality = cfg.jpeg_quality if cfg.jpeg_quality > 0 else None
    content_parts = [
        provider._build_pil_image_content(prepared, jpeg_quality=jpeg_quality),
        provider._build_text_content(ROUTER_PROMPT),
    ]

    response = None
    try:
        response = provider._send_to_api(
            content_parts,
            temperature=0.2,
            max_tokens=1024,
            use_thinking=False,
        )

        # Check for truncation BEFORE attempting JSON parse
        if response.finish_reason in _TRUNCATION_REASONS:
            logger.warning(
                f"Router response TRUNCATED (finish_reason={response.finish_reason}). "
                f"JSON is likely incomplete. Raw:\n{response.content}"
            )

        content = strip_markdown(response.content)
        data = json.loads(content)

        selected = data.get("primitive", "")
        reasoning = data.get("reasoning", "")

        if selected not in PRIMITIVE_NAMES:
            logger.warning(f"Router returned unknown primitive '{selected}', defaulting to unit_ops_primitive")
            selected = "unit_ops_primitive"

        logger.debug(f"Router selected: {selected}")

        return RouterResult(
            primitive=selected,
            reasoning=reasoning,
        )

    except (json.JSONDecodeError, KeyError, RuntimeError) as e:
        logger.error(f"Router failed to parse response: {e}")
        if response is not None:
            logger.error(f"Raw response:\n{response.content}")
            if response.finish_reason in _TRUNCATION_REASONS:
                logger.error(
                    f"Response was truncated (finish_reason={response.finish_reason})"
                    " -- this is the likely cause of the parse failure."
                )
        logger.error("Defaulting to unit_ops_primitive")
        return RouterResult(primitive="unit_ops_primitive")


def _build_recent_actions_string(ctx: ContextManager) -> str:
    """Build a compressed string of the last 3 actions for repetition avoidance."""
    last_actions = ctx.primitive_context.get_last_actions(3)
    if not last_actions:
        return "없음"

    parts = []
    for a in last_actions:
        if a.action_type == "click":
            parts.append(f"click({a.x},{a.y})")
        elif a.action_type == "press":
            parts.append(f"press({a.key})")
        elif a.action_type == "drag":
            parts.append(f"drag({a.x},{a.y}→{a.end_x},{a.end_y})")
        elif a.action_type == "type":
            parts.append(f"type({a.text[:20]})")
        else:
            parts.append(a.action_type)
    return ", ".join(parts)


def _build_strategy_with_directive(ctx: ContextManager, primitive_name: str) -> str:
    """Combine strategy + primitive directive + live game observations.

    Assembles:
    1. strategy.text (from StrategyUpdater)
    2. primitive_directives match (행동 기준 for this primitive)
    3. ContextUpdater observations (위협/기회/상황요약 — the live game state)

    If the primitive has no Korean mapping (unmapped), falls back to
    full context from ``get_context_for_primitive()``.
    """
    from computer_use_test.agent.modules.router.primitive_registry import PRIMITIVE_TO_KOREAN

    strategy_str = ctx.get_strategy_string()

    # --- Primitive-specific directive ---
    hl_ctx = ctx.high_level_context
    if hl_ctx.current_strategy and hl_ctx.current_strategy.primitive_directives:
        directive = get_directive_for_primitive(primitive_name, hl_ctx.current_strategy.primitive_directives)
        if directive:
            strategy_str = f"{strategy_str}\n\n[이 프리미티브 행동 기준] {directive}"
        elif primitive_name not in PRIMITIVE_TO_KOREAN:
            full_ctx = ctx.get_context_for_primitive(primitive_name)
            strategy_str = f"{strategy_str}\n\n[게임 상태 컨텍스트]\n{full_ctx}"
            logger.debug(f"Unmapped primitive '{primitive_name}': injecting full context as fallback")

    # --- Live game observations from ContextUpdater (threats/opportunities/summary) ---
    observation_parts = []

    # Latest situation summary (from ContextUpdater background analysis)
    if hl_ctx.notes:
        observation_parts.append(hl_ctx.notes[-1])  # most recent summary

    # Active threats
    if hl_ctx.active_threats:
        observation_parts.append(f"위협: {', '.join(hl_ctx.active_threats[:3])}")

    # Opportunities
    if hl_ctx.opportunities:
        observation_parts.append(f"기회: {', '.join(hl_ctx.opportunities[:3])}")

    if observation_parts:
        strategy_str = f"{strategy_str}\n\n[현재 상황] {' | '.join(observation_parts)}"

    return strategy_str


def plan_action(
    provider: BaseVLMProvider,
    pil_image,
    primitive_name: str,
    normalizing_range: int = 1000,
    high_level_strategy: str | None = None,
    recent_actions_string: str | None = None,
    hitl_directive: str | None = None,
    img_config: ImagePipelineConfig | None = None,
) -> AgentAction | list[AgentAction] | None:
    """
    Use VLM to generate the next action(s) for the selected primitive.

    For multi-action primitives (e.g., policy_primitive), returns a list of
    AgentActions to be executed sequentially. Otherwise returns a single action.

    Args:
        provider: VLM provider instance
        pil_image: PIL Image of the current game screen
        primitive_name: Selected primitive (determines the prompt)
        normalizing_range: Coordinate normalization range
        high_level_strategy: Optional high-level strategy/goal to guide action selection
        recent_actions_string: Compressed string of recent actions (for repetition avoidance)
        hitl_directive: Optional micro-level HITL directive (e.g., "병영을 최우선 선택")
        img_config: Image pipeline config for planner (defaults to PLANNER_DEFAULT)

    Returns:
        AgentAction, list[AgentAction], or None on failure
    """
    instruction = get_primitive_prompt(
        primitive_name,
        normalizing_range,
        high_level_strategy=high_level_strategy,
        recent_actions=recent_actions_string or "없음",
        hitl_directive=hitl_directive,
    )

    if primitive_name == "policy_primitive":
        return provider.analyze_multi(
            pil_image=pil_image,
            instruction=instruction,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )

    return provider.analyze(
        pil_image=pil_image,
        instruction=instruction,
        normalizing_range=normalizing_range,
        img_config=img_config,
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
    strategy_updater: StrategyUpdater | None = None,
    turn_detector: TurnDetector | None = None,
    router_img_config: ImagePipelineConfig | None = None,
    planner_img_config: ImagePipelineConfig | None = None,
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

    # Step 0: Read strategy (non-blocking when StrategyUpdater is active)
    if high_level_strategy:
        # CLI --strategy flag takes top priority
        strategy_string = high_level_strategy
    elif strategy_updater:
        # Background StrategyUpdater keeps ctx updated — just read the latest
        strategy_string = ctx.get_strategy_string()
    elif strategy_planner:
        # Fallback: synchronous VLM call (legacy path when no updater)
        strategy_string = None
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
    else:
        strategy_string = ctx.get_strategy_string()

    # Step 1: Observation (cropped to game window if detected)
    pil_image, screen_w, screen_h, x_offset, y_offset = capture_screen_pil()

    # Step 1a: Submit to TurnDetector (background, non-blocking)
    if turn_detector:
        turn_detector.submit(pil_image)

    # Step 2: Routing (classification only — no turn detection)
    rl.update_phase("routing")
    router_result = route_primitive(router_provider, pil_image, img_config=router_img_config)
    primitive_name = router_result.primitive
    macro_turn = macro_turn_manager.macro_turn_number if macro_turn_manager else 1

    # Read turn from TurnDetector (instant)
    detected_turn = turn_detector.latest_turn if turn_detector else None
    rl.route_result(primitive_name, router_result.reasoning, detected_turn, macro_turn, turn_number)

    # Step 2a: Turn-number validation
    is_new_turn = turn_detector.check_new_turn() if turn_detector else False
    if dbg.validate_turns:
        _get_turn_validator().validate(
            observed_turn=detected_turn,
            is_new_turn=is_new_turn,
            micro_turn=turn_number,
            macro_turn=macro_turn,
        )

    # Step 2b: Submit screenshot to ContextUpdater (non-blocking background analysis).
    # The updater will parse game-state info (gold, science, era, etc.) and write
    # it into ContextManager while we proceed with planning.
    if context_updater:
        context_updater.submit(pil_image)

    # Submit strategy triggers based on TurnDetector result
    if strategy_updater and is_new_turn:
        from computer_use_test.agent.modules.strategy.strategy_updater import StrategyRequest, StrategyTrigger

        strategy_updater.submit(StrategyRequest(StrategyTrigger.NEW_GAME_TURN))
        strategy_updater.submit_if_periodic_due()

    # Handle game-turn transition detected by TurnDetector
    if is_new_turn and macro_turn_manager:
        macro_summary = macro_turn_manager.handle_macro_turn_end()
        mt_num = macro_summary.macro_turn_number
        logger.debug(f"Macro-turn {mt_num} ended (turn {detected_turn}): {macro_summary.llm_summary[:80]}...")
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
        if strategy_updater:
            # Non-blocking: submit to background worker + apply text override immediately
            from computer_use_test.agent.modules.strategy.strategy_updater import StrategyRequest, StrategyTrigger

            strategy_updater.submit(
                StrategyRequest(StrategyTrigger.HITL_CHANGE, human_input=queue_result.strategy_override)
            )
            # CRITICAL: micro command is applied immediately this turn — don't wait for BG result
            primitive_hint = queue_result.strategy_override
            strategy_string = f"[사용자 최우선 지시] {queue_result.strategy_override}\n\n{strategy_string or ''}"
            rl.strategy_update("HITL Override", queue_result.strategy_override or "")
        elif strategy_planner:
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
            state_bridge.update_current_action(
                primitive_name, f"[HITL] {action.action} ({action.x}, {action.y})", action.reasoning
            )
            state_bridge.broadcast_agent_phase("사용자 명령 실행 중")
    else:
        # Step 3: Planning — strategy + directive + recent_actions
        rl.update_phase("planning")
        if state_bridge:
            state_bridge.broadcast_agent_phase("추론 중...")
        logger.debug(f"Planning: generating action for {primitive_name}...")

        # Build strategy with primitive-specific directive
        if strategy_updater and not high_level_strategy:
            strategy_string = _build_strategy_with_directive(ctx, primitive_name)
        elif not strategy_string:
            strategy_string = _build_strategy_with_directive(ctx, primitive_name)

        # Build recent actions string (compressed, for repetition avoidance)
        recent_actions_str = _build_recent_actions_string(ctx)

        if dbg.log_context:
            log_context(primitive_name, strategy_string, recent_actions_str)
        else:
            logger.debug(f"Recent actions for primitive: {recent_actions_str}")

        # Optionally augment strategy with knowledge
        if knowledge_manager and knowledge_manager.is_available():
            query = f"{primitive_name} 전략 가이드"
            try:
                knowledge_result = knowledge_manager.query(query, top_k=2)
                if not knowledge_result.is_empty():
                    knowledge_section = knowledge_result.to_prompt_string(max_chunks=2, max_tokens=300)
                    strategy_string = f"{strategy_string}\n\n{knowledge_section}"
                    logger.debug(f"Added {len(knowledge_result.chunks)} knowledge chunks")
            except Exception as e:
                logger.warning(f"Knowledge retrieval failed: {e}")

        action = plan_action(
            planner_provider,
            pil_image,
            primitive_name,
            normalizing_range,
            high_level_strategy=strategy_string,
            recent_actions_string=recent_actions_str,
            hitl_directive=primitive_hint if primitive_hint else None,
            img_config=planner_img_config,
        )

        # Handle None and empty list
        if action is None or (isinstance(action, list) and len(action) == 0):
            logger.error("  VLM returned no action. Turn aborted.")
            rl.update_phase("idle")
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

        # Display action result(s)
        display_action = action[0] if isinstance(action, list) else action
        extra = {}
        if display_action.action == "drag":
            extra["End Coords"] = f"({display_action.end_x}, {display_action.end_y})"
        if display_action.key:
            extra["Key"] = display_action.key
        if display_action.text:
            extra["Text"] = display_action.text
        if isinstance(action, list) and len(action) > 1:
            extra["Multi-Action"] = f"{len(action)} actions"
        rl.action_result(
            action_type=display_action.action,
            coords=(display_action.x, display_action.y),
            reasoning=display_action.reasoning or "",
            extra=extra or None,
        )

        # Update state bridge with current action info
        if state_bridge:
            action_desc = f"{display_action.action} ({display_action.x}, {display_action.y})"
            state_bridge.update_current_action(primitive_name, action_desc, display_action.reasoning or "")

    # Step 4: Execution — handle both single action and multi-action lists
    rl.update_phase("executing")
    if delay_before_action > 0:
        time.sleep(delay_before_action)

    # Normalize to list for uniform handling
    actions_list: list[AgentAction] = action if isinstance(action, list) else [action]

    execution_result = "success"
    error_message = ""
    for i, act in enumerate(actions_list):
        try:
            execute_action(act, screen_w, screen_h, normalizing_range, x_offset, y_offset)
            if len(actions_list) > 1:
                logger.debug(f"Multi-action {i + 1}/{len(actions_list)} executed: {act.action}")
                if i < len(actions_list) - 1:
                    time.sleep(0.3)
        except Exception as e:
            rl.execution_status(False, str(e))
            execution_result = "failed"
            error_message = str(e)
            break

    if execution_result == "success":
        rl.execution_status(True)

    rl.update_phase("idle")

    # Step 5: Record action(s) in context
    # Use the first action for summary; record all in context
    first_action = actions_list[0]
    for act in actions_list:
        ctx.record_action(
            action_type=act.action,
            primitive=primitive_name,
            x=act.x,
            y=act.y,
            end_x=act.end_x if act.action == "drag" else 0,
            end_y=act.end_y if act.action == "drag" else 0,
            key=act.key,
            text=act.text,
            result=execution_result,
            error_message=error_message,
        )

    # Step 6: Macro-turn tracking (fallback: keyword-based detection from old flow)
    if macro_turn_manager:
        macro_turn_manager.record_micro_turn(primitive_name, first_action.reasoning or "")
        # Keyword-based fallback detection (supplements TurnDetector-based detection above)
        if not is_new_turn and macro_turn_manager.is_next_turn_action(primitive_name, first_action):
            macro_summary = macro_turn_manager.handle_macro_turn_end()
            mt_num = macro_summary.macro_turn_number
            logger.debug(f"Macro-turn {mt_num} ended (keyword fallback): {macro_summary.llm_summary[:80]}...")
            if state_bridge:
                state_bridge.update_macro_turn(macro_summary.macro_turn_number + 1)

    action_desc = first_action.action if len(actions_list) == 1 else f"{first_action.action}(x{len(actions_list)})"
    rl.turn_summary(turn_number, primitive_name, action_desc, execution_result == "success")
    return TurnSummary(
        turn_number=turn_number,
        primitive=primitive_name,
        action_type=first_action.action,
        success=execution_result == "success",
        reasoning=first_action.reasoning or "",
        error_message=error_message,
        coords=(first_action.x, first_action.y),
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
    strategy_updater: StrategyUpdater | None = None,
    turn_detector: TurnDetector | None = None,
    router_img_config: ImagePipelineConfig | None = None,
    planner_img_config: ImagePipelineConfig | None = None,
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
                strategy_updater=strategy_updater,
                turn_detector=turn_detector,
                router_img_config=router_img_config,
                planner_img_config=planner_img_config,
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
