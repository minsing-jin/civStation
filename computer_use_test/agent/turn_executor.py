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

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.context import ContextManager
from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, DirectiveType
from computer_use_test.agent.modules.hitl.turn_checkpoint import (
    CheckpointDecision,
    InterruptMonitor,
    TurnCheckpoint,
    TurnSummary,
)
from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
from computer_use_test.agent.modules.primitive.multi_step_process import (
    _POLICY_RIGHT_CARD_LIST_RATIOS,
    _POLICY_RIGHT_TAB_BAR_RATIOS,
    StageTransition,
    VerificationResult,
    get_multi_step_process,
)
from computer_use_test.agent.modules.router.primitive_registry import (
    PRIMITIVE_NAMES,
    PRIMITIVE_REGISTRY,
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
from computer_use_test.utils.run_log_cache import get_run_log_cache_path
from computer_use_test.utils.screen import capture_screen_pil, execute_action, move_cursor_to_center
from computer_use_test.utils.ui_change_detector import screenshots_similar

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
_SCROLL_LIST_POST_ACTION_WAIT_SECONDS = 0.55

# Module-level TurnValidator singleton (created on first use)
_turn_validator: TurnValidator | None = None
_policy_artifact_session_dir: Path | None = None


def _get_turn_validator() -> TurnValidator:
    """Return (or create) the module-level TurnValidator singleton."""
    global _turn_validator  # noqa: PLW0603
    if _turn_validator is None:
        _turn_validator = TurnValidator()
    return _turn_validator


def _save_policy_semantic_failure_artifacts(
    *,
    primitive_name: str,
    stage: str,
    semantic_reason: str,
    semantic_details: dict[str, object] | None,
    memory: ShortTermMemory,
    pil_image,
) -> None:
    """Best-effort hook for capturing policy semantic-failure context."""

    def _crop_ratio_region(image, ratios: tuple[float, float, float, float]):
        width, height = image.size
        left = max(0, min(width, round(width * ratios[0])))
        top = max(0, min(height, round(height * ratios[1])))
        right = max(left + 1, min(width, round(width * ratios[2])))
        bottom = max(top + 1, min(height, round(height * ratios[3])))
        return image.crop((left, top, right, bottom))

    def _get_policy_artifact_session_dir() -> Path:
        global _policy_artifact_session_dir  # noqa: PLW0603
        if _policy_artifact_session_dir is None:
            run_log_path = get_run_log_cache_path()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            _policy_artifact_session_dir = run_log_path.parent / "policy_artifacts" / stamp
        _policy_artifact_session_dir.mkdir(parents=True, exist_ok=True)
        return _policy_artifact_session_dir

    try:
        artifact_dir = _get_policy_artifact_session_dir()
        full_path = artifact_dir / "full.png"
        tab_bar_path = artifact_dir / "tab_bar.png"
        card_list_path = artifact_dir / "card_list.png"
        manifest_path = artifact_dir / "manifest.json"

        pil_image.save(full_path)
        _crop_ratio_region(pil_image, _POLICY_RIGHT_TAB_BAR_RATIOS).save(tab_bar_path)
        _crop_ratio_region(pil_image, _POLICY_RIGHT_CARD_LIST_RATIOS).save(card_list_path)

        details = dict(semantic_details or {})
        manifest = {
            "primitive": primitive_name,
            "stage": stage,
            "semantic_reason": semantic_reason,
            "expected_tab": details.get("expected_tab", ""),
            "observed_active_tab": details.get("observed_active_tab", details.get("tab_bar_observed", "")),
            "policy_tab_outcome": details.get("policy_tab_outcome", ""),
            "tab_content_state": details.get("tab_content_state", ""),
            "selected_tab_name": memory.get_policy_selected_tab(),
            "current_tab": memory.get_policy_current_tab_name(),
            "current_tab_index": memory.policy_state.current_tab_index,
            "overview_mode": memory.policy_state.overview_mode,
            "last_event": memory.policy_state.last_event,
            "last_tab_check_result": memory.policy_state.last_tab_check_result,
            "image_size": getattr(pil_image, "size", None),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.warning(
            "Policy semantic failure artifact | primitive=%s stage=%s "
            "reason=%s path=%s details=%s memory=%s image_size=%s",
            primitive_name,
            stage,
            semantic_reason,
            artifact_dir,
            details,
            memory.to_prompt_string()[:400],
            getattr(pil_image, "size", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Policy semantic failure artifact save failed | primitive=%s stage=%s reason=%s error=%s",
            primitive_name,
            stage,
            semantic_reason,
            exc,
        )


def _primitive_trace_tag(primitive_name: str) -> str:
    """Return a compact display tag for one primitive trace line."""
    overrides = {
        "city_production_primitive": "CITY_PROD",
        "governor_primitive": "GOVERNOR",
        "research_select_primitive": "RESEARCH",
        "culture_decision_primitive": "CULTURE",
        "policy_primitive": "POLICY",
        "religion_primitive": "RELIGION",
        "voting_primitive": "VOTING",
    }
    if primitive_name in overrides:
        return overrides[primitive_name]
    return primitive_name.replace("_primitive", "").upper()


def _emit_runtime_trace(
    *,
    rl: RichLogger,
    state_bridge: AgentStateBridge | None,
    primitive_name: str,
    stage: str,
    phase: str,
    summary: str,
    detail: str = "",
) -> None:
    """Emit one structured runtime trace event to Rich and HITL status."""
    rl.primitive_event(
        _primitive_trace_tag(primitive_name),
        f"{phase} | stage={stage or '-'} | {summary}{f' | {detail}' if detail else ''}",
    )
    if state_bridge:
        state_bridge.append_trace_event(
            primitive=primitive_name,
            stage=stage or "",
            phase=phase,
            summary=summary,
            detail=detail,
        )


def _post_action_wait_seconds(
    primitive_name: str,
    actions: list[AgentAction],
    *,
    delay_before_action: float,
    stage_name: str = "",
) -> float:
    """Return a capture-settle delay tuned to the executed action bundle."""
    if primitive_name == "policy_primitive":
        return 0.5
    default_wait = min(delay_before_action, 0.3)
    if primitive_name == "religion_primitive" and any(
        action.action in {"click", "double_click", "press"} for action in actions
    ):
        return max(default_wait, 0.5)
    if primitive_name in {"city_production_primitive", "governor_primitive"} and any(
        action.action == "scroll" for action in actions
    ):
        return max(default_wait, _SCROLL_LIST_POST_ACTION_WAIT_SECONDS)
    if (
        primitive_name == "city_production_primitive"
        and stage_name == "select_from_memory"
        and any(action.action in {"click", "double_click"} for action in actions)
    ):
        return max(default_wait, 0.5)
    return default_wait


@dataclass
class QueueCheckResult:
    """Result of checking the CommandQueue for mid-turn interrupts."""

    should_stop: bool = False
    override_action: AgentAction | None = field(default=None)
    override_primitive: str | None = None  # HITL forced primitive (e.g. "war_primitive")
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
                # Check if this is a primitive-name override (e.g. {"primitive": "war_primitive"})
                if "primitive" in payload and payload["primitive"] in PRIMITIVE_REGISTRY:
                    result.override_primitive = payload["primitive"]
                    logger.info(f"HITL primitive override: {result.override_primitive}")
                else:
                    # Legacy: coordinate-based action override
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
    prompt_language: str = "eng",
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
        prompt_language: Primitive prompt language (`eng` default, `kor` optional)
        img_config: Image pipeline config for planner (defaults to PLANNER_DEFAULT)

    Returns:
        AgentAction, list[AgentAction], or None on failure
    """
    instruction = get_primitive_prompt(
        primitive_name,
        normalizing_range,
        high_level_strategy=high_level_strategy,
        recent_actions=recent_actions_string,
        hitl_directive=hitl_directive,
        language=prompt_language,
    )

    return provider.analyze(
        pil_image=pil_image,
        instruction=instruction,
        normalizing_range=normalizing_range,
        img_config=img_config,
    )


@dataclass
class PrimitiveLoopResult:
    """Result from a multi-step primitive execution loop."""

    success: bool = False
    steps_taken: int = 0
    completed: bool = False  # task_status == "complete"
    re_route: bool = False  # UI didn't change → wrong primitive
    last_action: AgentAction | None = None
    error_message: str = ""


def _should_follow_up_route(loop_result: PrimitiveLoopResult) -> bool:
    """Whether run_one_turn should attempt a follow-up route after a loop result."""
    if loop_result.re_route or loop_result.completed:
        return True
    if not loop_result.error_message:
        return False
    return loop_result.error_message != "STOP directive received during loop"


def run_primitive_loop(
    planner_provider: BaseVLMProvider,
    primitive_name: str,
    screen_w: int,
    screen_h: int,
    normalizing_range: int,
    x_offset: int,
    y_offset: int,
    strategy_string: str,
    recent_actions_str: str,
    hitl_directive: str | None,
    memory: ShortTermMemory,
    ctx: ContextManager,
    max_steps: int,
    completion_condition: str,
    prompt_language: str = "eng",
    planner_img_config: ImagePipelineConfig | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge: AgentStateBridge | None = None,
    strategy_updater: StrategyUpdater | None = None,
    delay_before_action: float = 0.5,
) -> PrimitiveLoopResult:
    """Execute a class-based multi-step primitive until completion, rollback, or reroute."""
    rl = RichLogger.get()
    result = PrimitiveLoopResult()
    process = get_multi_step_process(primitive_name, completion_condition)
    process.initialize(memory)
    rollback_used = False
    last_policy_event_emitted = ""
    active_strategy_string = strategy_string or ""
    active_hitl_directive = (hitl_directive or "").strip()
    step_start = time.monotonic()
    plan_end = step_start
    exec_end = step_start
    step = 0
    loop_iterations = 0

    def _strip_user_priority_prefix(text: str) -> str:
        prefix = "[사용자 최우선 지시] "
        if not text.startswith(prefix):
            return text
        parts = text.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else ""

    base_strategy_string = _strip_user_priority_prefix(active_strategy_string)
    if active_hitl_directive:
        memory.set_task_hitl_directive(active_hitl_directive, reason="initial task hitl directive")

    def _call_process_method_with_optional_prompt_language(method, /, *args, **kwargs):
        signature = inspect.signature(method)
        parameters = signature.parameters
        if "prompt_language" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            kwargs["prompt_language"] = prompt_language
        return method(*args, **kwargs)

    def _emit_policy_event_if_changed() -> None:
        nonlocal last_policy_event_emitted
        if primitive_name != "policy_primitive" or not memory.policy_state.enabled:
            return
        current_event = memory.policy_state.last_event.strip()
        if current_event and current_event != last_policy_event_emitted:
            rl.policy_event(current_event)
            last_policy_event_emitted = current_event

    def _policy_cache_summary() -> str:
        if primitive_name != "policy_primitive" or not memory.policy_state.enabled:
            return "-"
        ordered_tabs = memory.policy_state.eligible_tabs_queue or list(memory.policy_state.tab_positions.keys())
        parts: list[str] = []
        for tab_name in ordered_tabs:
            tab = memory.policy_state.tab_positions.get(tab_name)
            if tab is None:
                parts.append(f"{tab_name}=missing")
                continue
            marker = "*" if tab.confirmed else ("?" if tab_name in memory.policy_state.provisional_tabs else "")
            parts.append(f"{tab_name}=({tab.screen_x},{tab.screen_y}){marker}")
        return ", ".join(parts) if parts else "-"

    def _sync_policy_cache_to_context() -> None:
        if primitive_name != "policy_primitive" or not memory.policy_state.enabled:
            return
        payload = memory.export_policy_tab_cache()
        ctx.replace_policy_tab_cache(
            positions=payload.get("positions"),
            confirmed_tabs=payload.get("confirmed_tabs"),
            provisional_tabs=payload.get("provisional_tabs"),
            capture_geometry=payload.get("capture_geometry"),
            calibration_complete=bool(payload.get("calibration_complete", False)),
        )
        ctx.clear_policy_tab_cache_failures()
        for tab_name in sorted(memory.policy_state.distinct_failed_tabs):
            ctx.add_policy_tab_cache_failure(tab_name)

    def _policy_action_summary(actions: list[AgentAction]) -> str:
        parts: list[str] = []
        for idx, planned_action in enumerate(actions, start=1):
            coord = f"{planned_action.coord_space}:({planned_action.x},{planned_action.y})"
            if planned_action.action == "drag":
                coord = f"{coord}->{planned_action.coord_space}:({planned_action.end_x},{planned_action.end_y})"
            parts.append(f"{idx}:{planned_action.action}{coord}")
        return " | ".join(parts) if parts else "-"

    def _publish_error_state(message: str) -> None:
        if not message:
            return
        rl.action_result(
            action_type="error",
            coords=None,
            reasoning=message,
        )
        if state_bridge:
            state_bridge.update_current_action(primitive_name, "error", message)

    def _log_policy_state(prefix: str, actions: list[AgentAction] | None = None) -> None:
        if primitive_name != "policy_primitive" or not memory.policy_state.enabled:
            return
        queue = " -> ".join(memory.get_policy_remaining_queue()) or "<empty>"
        completed = " -> ".join(memory.policy_state.completed_tabs) or "-"
        current_tab = memory.get_policy_current_tab_name() or "-"
        selected_tab = memory.get_policy_selected_tab() or "-"
        logger.info(
            "[POLICY] %s | mode=%s stage=%s queue_idx=%s remaining=%s "
            "completed=%s current=%s selected=%s "
            "bootstrap=%s cache_source=%s geometry=%s cache_geometry=%s "
            "visible=%s cache=%s provisional=%s bundle_count=%s "
            "similarity=%s tab_check=%s relocalize=%s fallback=%s event=%s actions=%s",
            prefix,
            memory.policy_state.mode,
            memory.current_stage or "-",
            memory.policy_state.current_tab_index,
            queue,
            completed,
            current_tab,
            selected_tab,
            memory.policy_state.bootstrap_summary or "-",
            memory.policy_state.cache_source or "-",
            memory._format_policy_capture_geometry(memory.policy_state.capture_geometry),
            memory._format_policy_capture_geometry(memory.policy_state.cache_geometry),
            ",".join(memory.policy_state.visible_tabs) or "-",
            _policy_cache_summary(),
            ",".join(sorted(memory.policy_state.provisional_tabs)) or "-",
            memory.policy_state.last_bundle_action_count,
            memory.policy_state.last_similarity_result or "-",
            memory.policy_state.last_tab_check_result or "-",
            memory.policy_state.last_relocalize_result or "-",
            ",".join(sorted(memory.policy_state.distinct_failed_tabs)) or "-",
            memory.policy_state.last_event or "-",
            _policy_action_summary(actions or []),
        )

    def _best_choice_summary() -> str:
        best_choice = memory.get_best_choice()
        if best_choice is None:
            return ""
        return f"{best_choice.label} ({best_choice.position_hint})"

    def _multi_step_stm_summary() -> str:
        stm_raw = memory.to_prompt_string()
        if primitive_name in {"policy_primitive", "city_production_primitive"}:
            return stm_raw
        return stm_raw[:150]

    def _format_debug_action_summary(action: AgentAction) -> str:
        parts = [action.action]
        if action.action in {"click", "double_click", "scroll"}:
            parts.append(f"@ ({action.x}, {action.y})")
        if action.action == "drag":
            parts.append(f"@ ({action.x}, {action.y}) -> ({action.end_x}, {action.end_y})")
        if action.action == "scroll":
            parts.append(f"amount={action.scroll_amount}")
        if action.action == "press" and action.key:
            parts.append(f"key={action.key}")
        if action.reasoning:
            parts.append(action.reasoning[:120])
        return " | ".join(parts)

    def _emit_city_production_event(label: str, detail: str) -> None:
        if primitive_name != "city_production_primitive":
            return
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage=memory.current_stage or "-",
            phase=label,
            summary=detail,
        )

    def _visible_progress(displayed_step: int) -> tuple[int, int]:
        step_value, max_value = process.get_visible_progress(
            memory,
            executed_steps=displayed_step,
            hard_max_steps=max_steps,
        )
        step_value = max(0, step_value)
        max_value = max(0, max_value)
        if max_value and step_value > max_value:
            step_value = max_value
        return step_value, max_value

    def _iteration_limit() -> int:
        try:
            requested_limit = int(process.get_iteration_limit(memory, action_limit=max_steps))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Iteration limit hook failed for %s: %s", primitive_name, exc)
            requested_limit = max_steps
        return max(max_steps, requested_limit)

    def _policy_semantic_recheck_once(
        initial_result,
        *,
        action: AgentAction,
    ):
        if primitive_name != "policy_primitive":
            return initial_result
        if not initial_result.handled or initial_result.passed:
            return initial_result
        time.sleep(0.1)
        recheck_image, *_ = capture_screen_pil()
        recheck_result = process.verify_action_success(
            planner_provider,
            recheck_image,
            memory,
            action,
            img_config=planner_img_config,
        )
        if recheck_result.handled:
            memory.set_last_semantic_verify(
                "pass" if recheck_result.passed else "fail",
                recheck_result.reason,
            )
        if recheck_result.passed:
            logger.info(
                "Policy semantic verification recovered on delayed recheck | stage=%s reason=%s",
                memory.current_stage or "-",
                recheck_result.reason,
            )
        return recheck_result

    def _governor_completion_recheck_once(initial_verification: VerificationResult) -> VerificationResult:
        if primitive_name != "governor_primitive":
            return initial_verification
        if initial_verification.complete:
            return initial_verification
        time.sleep(0.2)
        recheck_image, *_ = capture_screen_pil()
        recheck_result = process.verify_completion(
            planner_provider,
            recheck_image,
            memory,
            img_config=planner_img_config,
        )
        if recheck_result.complete:
            logger.info(
                "Governor completion recovered on delayed recheck | stage=%s reason=%s",
                memory.current_stage or "-",
                recheck_result.reason,
            )
        return recheck_result

    def _refresh_multi_step_debug() -> None:
        stm_str = _multi_step_stm_summary()
        visible_step, visible_max_steps = _visible_progress(result.steps_taken)
        rl.update_multi_step(
            active=True,
            step=visible_step,
            max_steps=visible_max_steps,
            plan_ms=(plan_end - step_start) * 1000,
            exec_ms=(exec_end - plan_end) * 1000,
            stage=memory.current_stage,
            stall_count=memory.failure_count,
            best_choice=_best_choice_summary(),
            stm_summary=stm_str,
        )
        if state_bridge:
            state_bridge.update_multi_step(
                active=True,
                step=visible_step,
                max_steps=visible_max_steps,
                plan_ms=(plan_end - step_start) * 1000,
                exec_ms=(exec_end - plan_end) * 1000,
                stage=memory.current_stage,
                stall_count=memory.failure_count,
                best_choice=_best_choice_summary(),
                stm_summary=stm_str,
            )

    def _clear_current_action_debug() -> None:
        rl.clear_current_action()
        if state_bridge:
            state_bridge.clear_current_action()

    def _complete_from_terminal_state() -> bool:
        if not process.is_terminal_state(memory):
            return False
        result.completed = True
        result.success = True
        _refresh_multi_step_debug()
        reason = process.terminal_state_reason(memory)
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage=memory.current_stage or "-",
            phase="complete",
            summary=reason,
        )
        logger.info("Primitive loop completed from terminal state: %s in %s steps", primitive_name, step + 1)
        return True

    while True:
        if result.steps_taken >= max_steps:
            result.error_message = f"Primitive loop reached safety action cap ({max_steps}) without completion"
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="error",
                summary=result.error_message,
            )
            logger.warning(result.error_message)
            break

        current_iteration_limit = _iteration_limit()
        if loop_iterations >= current_iteration_limit:
            result.error_message = (
                f"Primitive loop reached safety iteration cap ({current_iteration_limit}) without completion"
            )
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="error",
                summary=result.error_message,
            )
            logger.warning(result.error_message)
            break

        step = loop_iterations
        loop_iterations += 1
        pre_image, screen_w, screen_h, x_offset, y_offset = capture_screen_pil()
        if primitive_name == "policy_primitive":
            memory.set_policy_capture_geometry(screen_w, screen_h, x_offset, y_offset)
            if memory.policy_state.tab_positions and memory.policy_state.cache_geometry is not None:
                if not memory.policy_cache_matches_current_geometry():
                    preserve_entry_done = memory.is_policy_entry_done()
                    memory.clear_policy_bootstrap(
                        preserve_entry_done=preserve_entry_done,
                        preserve_progress=True,
                        preserve_tab_positions=False,
                    )
                    memory.set_policy_mode("structured")
                    memory.set_policy_event("policy cache invalidated: geometry changed")
                    memory.begin_stage("bootstrap_tabs" if preserve_entry_done else "policy_entry")
                    _sync_policy_cache_to_context()
        _log_policy_state(f"loop-start step={step + 1}")

        if command_queue:
            queue_check = _check_queue_for_interrupt(command_queue, agent_gate=agent_gate)
            if queue_check.should_stop:
                result.error_message = "STOP directive received during loop"
                break
            if queue_check.strategy_override:
                override_text = str(queue_check.strategy_override).strip()
                if override_text:
                    memory.set_task_hitl_directive(override_text, reason="mid-task hitl directive")
                    active_hitl_directive = override_text
                    active_strategy_string = f"[사용자 최우선 지시] {override_text}\n\n{base_strategy_string}".strip()
                    if strategy_updater:
                        from computer_use_test.agent.modules.strategy.strategy_updater import (
                            StrategyRequest,
                            StrategyTrigger,
                        )

                        strategy_updater.submit(StrategyRequest(StrategyTrigger.HITL_CHANGE, human_input=override_text))

        step_start = time.monotonic()
        action: AgentAction | list[AgentAction] | None = None

        if process.should_observe(memory):
            move_cursor_to_center(screen_w, screen_h, x_offset, y_offset)
            pre_image, screen_w, screen_h, x_offset, y_offset = capture_screen_pil()
            if primitive_name == "policy_primitive":
                memory.set_policy_capture_geometry(screen_w, screen_h, x_offset, y_offset)
            observation = _call_process_method_with_optional_prompt_language(
                process.observe,
                planner_provider,
                pre_image,
                memory,
                normalizing_range=normalizing_range,
                img_config=planner_img_config,
            )
            plan_end = time.monotonic()
            if observation is None:
                result.error_message = "Observation step failed"
                break

            action = process.consume_observation(memory, observation)
            _emit_city_production_event(
                "observe",
                f"stage={memory.current_stage or '-'} | {memory.last_observation_summary or '-'} | "
                f"anchor={memory.last_observation_anchor or '-'}",
            )
            if primitive_name != "city_production_primitive":
                _emit_runtime_trace(
                    rl=rl,
                    state_bridge=state_bridge,
                    primitive_name=primitive_name,
                    stage=memory.current_stage or "-",
                    phase="observe",
                    summary=memory.last_observation_summary or "observation completed",
                    detail=memory.last_observation_anchor or "",
                )
            stm_str = _multi_step_stm_summary()
            visible_step, visible_max_steps = _visible_progress(result.steps_taken)
            rl.update_multi_step(
                active=True,
                step=visible_step,
                max_steps=visible_max_steps,
                plan_ms=(plan_end - step_start) * 1000,
                exec_ms=0,
                stage=memory.current_stage,
                stall_count=memory.failure_count,
                best_choice=_best_choice_summary(),
                stm_summary=stm_str,
            )
            if state_bridge:
                state_bridge.update_multi_step(
                    active=True,
                    step=visible_step,
                    max_steps=visible_max_steps,
                    plan_ms=(plan_end - step_start) * 1000,
                    exec_ms=0,
                    stage=memory.current_stage,
                    stall_count=memory.failure_count,
                    best_choice=_best_choice_summary(),
                    stm_summary=stm_str,
                )
            _emit_policy_event_if_changed()
            _log_policy_state("post-observe")

            if action is None:
                continue
        else:
            if (
                process.supports_observation
                and memory.choice_catalog.end_reached
                and memory.get_best_choice() is None
                and process.should_auto_decide_from_memory(memory)
            ):
                decided = _call_process_method_with_optional_prompt_language(
                    process.decide_from_memory,
                    planner_provider,
                    memory,
                    high_level_strategy=active_strategy_string,
                )
                if not decided:
                    result.error_message = "Failed to decide best choice from short-term memory"
                    break

            task_recent_actions = memory.recent_actions_prompt()
            combined_recent_actions = recent_actions_str
            if task_recent_actions != "없음":
                if combined_recent_actions and combined_recent_actions != "없음":
                    combined_recent_actions = f"{combined_recent_actions}\n{task_recent_actions}"
                else:
                    combined_recent_actions = task_recent_actions

            action = _call_process_method_with_optional_prompt_language(
                process.plan_action,
                planner_provider,
                pre_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=active_strategy_string,
                recent_actions=combined_recent_actions or "없음",
                hitl_directive=active_hitl_directive or None,
                img_config=planner_img_config,
            )
            plan_end = time.monotonic()
            _sync_policy_cache_to_context()

        if action is None:
            logger.warning(f"Primitive loop step {step + 1}: VLM returned no action")
            result.error_message = "VLM returned no action"
            break

        if isinstance(action, StageTransition):
            if primitive_name == "policy_primitive" and action.reason:
                memory.set_policy_event(action.reason)
            logger.info(
                "Primitive loop internal stage transition | primitive=%s stage=%s reason=%s",
                primitive_name,
                action.stage,
                action.reason or "-",
            )
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="stage",
                summary=action.reason or "internal stage transition",
                detail=f"next={action.stage}",
            )
            visible_step, visible_max_steps = _visible_progress(result.steps_taken)
            rl.update_multi_step(
                active=True,
                step=visible_step,
                max_steps=visible_max_steps,
                plan_ms=(plan_end - step_start) * 1000,
                exec_ms=0,
                stage=memory.current_stage,
                stall_count=memory.failure_count,
                best_choice=_best_choice_summary(),
                stm_summary=_multi_step_stm_summary(),
            )
            if state_bridge:
                state_bridge.update_multi_step(
                    active=True,
                    step=visible_step,
                    max_steps=visible_max_steps,
                    plan_ms=(plan_end - step_start) * 1000,
                    exec_ms=0,
                    stage=memory.current_stage,
                    stall_count=memory.failure_count,
                    best_choice=_best_choice_summary(),
                    stm_summary=_multi_step_stm_summary(),
                )
            _sync_policy_cache_to_context()
            _emit_policy_event_if_changed()
            _log_policy_state("stage-transition")
            if _complete_from_terminal_state():
                break
            continue

        is_action_bundle = isinstance(action, list)
        actions_list = action if is_action_bundle else [action]
        actions_list = process.resolve_actions(actions_list, memory)
        display_action = actions_list[0]
        planned_summary = (
            f"{_format_debug_action_summary(display_action)} +{len(actions_list) - 1}"
            if is_action_bundle and len(actions_list) > 1
            else _format_debug_action_summary(display_action)
        )
        memory.set_last_planned_action_debug(planned_summary)
        result.last_action = actions_list[-1]
        result.steps_taken += 1

        visible_step, visible_max_steps = _visible_progress(result.steps_taken)
        action_extra = {
            "Step": f"{visible_step}/{visible_max_steps}",
            "Status": display_action.task_status or "in_progress",
        }
        if is_action_bundle:
            action_extra["Multi-Action"] = f"{len(actions_list)} actions"
        _emit_city_production_event(
            "plan",
            f"stage={memory.current_stage or '-'} | {memory.last_planned_action or planned_summary}",
        )
        if primitive_name != "city_production_primitive":
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="plan",
                summary=memory.last_planned_action or planned_summary,
            )
        _log_policy_state("planned-actions", actions_list)
        rl.action_result(
            action_type=display_action.action,
            coords=(display_action.x, display_action.y),
            reasoning=display_action.reasoning or "",
            extra=action_extra,
        )

        if state_bridge:
            action_desc = f"{display_action.action} ({display_action.x}, {display_action.y})"
            if is_action_bundle and len(actions_list) > 1:
                action_desc = f"{action_desc} +{len(actions_list) - 1}"
            state_bridge.update_current_action(primitive_name, action_desc, display_action.reasoning or "")

        if delay_before_action > 0:
            time.sleep(delay_before_action)

        try:
            for idx, planned_action in enumerate(actions_list):
                if primitive_name == "policy_primitive":
                    logger.info(
                        "[POLICY] execute %s/%s | stage=%s current=%s action=%s "
                        "coord_space=%s x=%s y=%s end_x=%s end_y=%s "
                        "button=%s key=%s status=%s reason=%s",
                        idx + 1,
                        len(actions_list),
                        memory.current_stage or "-",
                        memory.get_policy_current_tab_name() or "-",
                        planned_action.action,
                        planned_action.coord_space,
                        planned_action.x,
                        planned_action.y,
                        planned_action.end_x,
                        planned_action.end_y,
                        planned_action.button or "-",
                        planned_action.key or "-",
                        planned_action.task_status or "in_progress",
                        (planned_action.reasoning or "")[:120],
                    )
                execute_action(planned_action, screen_w, screen_h, normalizing_range, x_offset, y_offset)
                is_policy_tab_click = (
                    primitive_name == "policy_primitive"
                    and planned_action.action in ("click", "double_click")
                    and memory.current_stage in {"calibrate_tabs", "click_cached_tab", "click_next_tab"}
                )
                if (
                    primitive_name != "policy_primitive" and planned_action.action in ("click", "double_click")
                ) or is_policy_tab_click:
                    move_cursor_to_center(screen_w, screen_h, x_offset, y_offset)
                if idx < len(actions_list) - 1:
                    time.sleep(0.25)
        except Exception as e:
            logger.error(f"Primitive loop execution failed: {e}")
            result.error_message = str(e)
            break
        exec_end = time.monotonic()

        executed_summary = (
            " || ".join(_format_debug_action_summary(planned_action) for planned_action in actions_list[:2])
            if len(actions_list) > 1
            else _format_debug_action_summary(actions_list[0])
        )
        if len(actions_list) > 2:
            executed_summary = f"{executed_summary} || +{len(actions_list) - 2} more"
        memory.set_last_executed_action_debug(executed_summary)
        _emit_city_production_event(
            "exec",
            f"stage={memory.current_stage or '-'} | {memory.last_executed_action or executed_summary}",
        )
        if primitive_name != "city_production_primitive":
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="exec",
                summary=memory.last_executed_action or executed_summary,
            )

        stm_str = _multi_step_stm_summary()
        visible_step, visible_max_steps = _visible_progress(result.steps_taken)
        rl.update_multi_step(
            active=True,
            step=visible_step,
            max_steps=visible_max_steps,
            plan_ms=(plan_end - step_start) * 1000,
            exec_ms=(exec_end - plan_end) * 1000,
            stage=memory.current_stage,
            stall_count=memory.failure_count,
            best_choice=_best_choice_summary(),
            stm_summary=stm_str,
        )
        if state_bridge:
            state_bridge.update_multi_step(
                active=True,
                step=visible_step,
                max_steps=visible_max_steps,
                plan_ms=(plan_end - step_start) * 1000,
                exec_ms=(exec_end - plan_end) * 1000,
                stage=memory.current_stage,
                stall_count=memory.failure_count,
                best_choice=_best_choice_summary(),
                stm_summary=stm_str,
            )
        _emit_policy_event_if_changed()
        _log_policy_state("post-exec", actions_list)

        for planned_action in actions_list:
            ctx.record_action(
                action_type=planned_action.action,
                primitive=primitive_name,
                x=planned_action.x,
                y=planned_action.y,
                end_x=planned_action.end_x if planned_action.action == "drag" else 0,
                end_y=planned_action.end_y if planned_action.action == "drag" else 0,
                key=planned_action.key,
                text=planned_action.text,
                result="success",
            )

        semantic = (display_action.reasoning or display_action.action)[:60]
        action_summary = (
            f"[bundle x{len(actions_list)}] {semantic}"
            if len(actions_list) > 1
            else f"[{display_action.action}] {semantic}"
        )
        memory.add_observation(
            reasoning=display_action.reasoning or "",
            action_summary=action_summary,
            result=display_action.task_status or "in_progress",
            stage=memory.current_stage,
        )

        post_action_wait = _post_action_wait_seconds(
            primitive_name,
            actions_list,
            delay_before_action=delay_before_action,
            stage_name=memory.current_stage or "",
        )
        if post_action_wait > 0:
            time.sleep(post_action_wait)
        post_image, *_ = capture_screen_pil()
        if primitive_name == "policy_primitive":
            is_policy_tab_click = display_action.action in {
                "click",
                "double_click",
            } and memory.current_stage in {"calibrate_tabs", "click_cached_tab", "click_next_tab"}
            policy_requires_semantic_verify = is_policy_tab_click and process.should_verify_action_without_ui_change(
                memory, display_action
            )
            semantic_verify_result = None
            if policy_requires_semantic_verify:
                semantic_verify_result = process.verify_action_success(
                    planner_provider,
                    post_image,
                    memory,
                    display_action,
                    img_config=planner_img_config,
                )
                if semantic_verify_result.handled:
                    memory.set_last_semantic_verify(
                        "pass" if semantic_verify_result.passed else "fail",
                        semantic_verify_result.reason,
                    )
                semantic_verify_result = _policy_semantic_recheck_once(
                    semantic_verify_result,
                    action=display_action,
                )
                verify_status = (
                    "pass"
                    if semantic_verify_result.handled and semantic_verify_result.passed
                    else "fail"
                    if semantic_verify_result.handled
                    else "unhandled"
                )
                memory.set_policy_last_similarity_result(f"skipped(policy semantic-only) tab-check {verify_status}")
                logger.info(
                    "[POLICY] post-action-check | stage=%s current=%s similarity=skipped semantic_status=%s",
                    memory.current_stage or "-",
                    memory.get_policy_current_tab_name() or "-",
                    verify_status,
                )
                if not semantic_verify_result.handled or not semantic_verify_result.passed:
                    logger.info(
                        "Semantic verification failed for %s at stage %s: %s",
                        primitive_name,
                        memory.current_stage,
                        semantic_verify_result.reason if semantic_verify_result else "unhandled",
                    )
                    _save_policy_semantic_failure_artifacts(
                        primitive_name=primitive_name,
                        stage=memory.current_stage or "-",
                        semantic_reason=semantic_verify_result.reason if semantic_verify_result else "unhandled",
                        semantic_details=getattr(semantic_verify_result, "details", {}),
                        memory=memory,
                        pil_image=post_image,
                    )
                    no_progress_resolution = _call_process_method_with_optional_prompt_language(
                        process.handle_no_progress,
                        planner_provider,
                        post_image,
                        memory,
                        last_action=actions_list[-1],
                        normalizing_range=normalizing_range,
                        high_level_strategy=strategy_string,
                        recent_actions=recent_actions_str,
                        hitl_directive=hitl_directive,
                        img_config=planner_img_config,
                    )
                    if no_progress_resolution.handled:
                        memory.failure_count = 0
                        _sync_policy_cache_to_context()
                        _refresh_multi_step_debug()
                        _log_policy_state("semantic-no-progress handled", actions_list)
                        continue
                    if no_progress_resolution.reroute:
                        result.re_route = True
                        result.error_message = no_progress_resolution.error_message or "Semantic verification failed"
                        _sync_policy_cache_to_context()
                        _log_policy_state("semantic-no-progress reroute", actions_list)
                        break
            elif is_policy_tab_click:
                skip_reason = (
                    "skipped(policy confirmed absolute cache)"
                    if display_action.coord_space == "absolute"
                    else "skipped(policy semantic-only)"
                )
                memory.set_policy_last_similarity_result(skip_reason)
                logger.info(
                    "[POLICY] post-action-check | stage=%s current=%s similarity=%s semantic_status=not-required",
                    memory.current_stage or "-",
                    memory.get_policy_current_tab_name() or "-",
                    skip_reason,
                )
            else:
                logger.info(
                    "[POLICY] post-action-check | stage=%s current=%s "
                    "similarity=not-applicable semantic_status=not-required",
                    memory.current_stage or "-",
                    memory.get_policy_current_tab_name() or "-",
                )

            bundle_task_complete = any(planned_action.task_status == "complete" for planned_action in actions_list)
            if bundle_task_complete:
                verification = process.verify_completion(
                    planner_provider,
                    post_image,
                    memory,
                    img_config=planner_img_config,
                )
                verification = _governor_completion_recheck_once(verification)
                if verification.complete:
                    result.completed = True
                    result.success = True
                    _emit_runtime_trace(
                        rl=rl,
                        state_bridge=state_bridge,
                        primitive_name=primitive_name,
                        stage=memory.current_stage or "-",
                        phase="complete",
                        summary=verification.reason or "completion verified",
                    )
                    logger.info(f"Primitive loop completed: {primitive_name} in {step + 1} steps")
                    break
                logger.warning(
                    "Completion verification failed for %s at step %s: %s",
                    primitive_name,
                    step + 1,
                    verification.reason,
                )
                no_progress_resolution = _call_process_method_with_optional_prompt_language(
                    process.handle_no_progress,
                    planner_provider,
                    post_image,
                    memory,
                    last_action=actions_list[-1],
                    normalizing_range=normalizing_range,
                    high_level_strategy=strategy_string,
                    recent_actions=recent_actions_str,
                    hitl_directive=hitl_directive,
                    img_config=planner_img_config,
                )
                if no_progress_resolution.handled:
                    memory.failure_count = 0
                    _sync_policy_cache_to_context()
                    _refresh_multi_step_debug()
                    _log_policy_state("completion-no-progress handled", actions_list)
                    continue
                if no_progress_resolution.reroute:
                    result.re_route = True
                    result.error_message = no_progress_resolution.error_message or "Completion verification failed"
                    _sync_policy_cache_to_context()
                    _log_policy_state("completion-no-progress reroute", actions_list)
                    break

            stage_before_success = memory.current_stage
            if is_action_bundle:
                process.on_actions_success(memory, actions_list)
            else:
                process.on_action_success(memory, display_action)
            process.on_stage_success(memory, display_action, stage_name=stage_before_success)
            memory.failure_count = 0
            rollback_used = False
            _sync_policy_cache_to_context()
            _log_policy_state("stage-success", actions_list)
            if _complete_from_terminal_state():
                break
            continue

        raw_ui_changed = not screenshots_similar(
            pre_image,
            post_image,
            action=display_action,
            normalizing_range=normalizing_range,
        )
        effective_ui_changed = raw_ui_changed

        verify_without_ui_change = not effective_ui_changed and process.should_verify_action_without_ui_change(
            memory, display_action
        )
        semantic_verify_result = None
        if verify_without_ui_change:
            semantic_verify_result = process.verify_action_success(
                planner_provider,
                post_image,
                memory,
                display_action,
                img_config=planner_img_config,
            )
            if semantic_verify_result.handled:
                memory.set_last_semantic_verify(
                    "pass" if semantic_verify_result.passed else "fail",
                    semantic_verify_result.reason,
                )
                if semantic_verify_result.passed:
                    effective_ui_changed = True

        bundle_task_complete = any(planned_action.task_status == "complete" for planned_action in actions_list)
        if bundle_task_complete:
            verification = process.verify_completion(
                planner_provider,
                post_image,
                memory,
                img_config=planner_img_config,
            )
            verification = _governor_completion_recheck_once(verification)
            if verification.complete:
                result.completed = True
                result.success = True
                _emit_runtime_trace(
                    rl=rl,
                    state_bridge=state_bridge,
                    primitive_name=primitive_name,
                    stage=memory.current_stage or "-",
                    phase="complete",
                    summary=verification.reason or "completion verified",
                )
                logger.info(f"Primitive loop completed: {primitive_name} in {step + 1} steps")
                break
            logger.warning(
                "Completion verification failed for %s at step %s: %s",
                primitive_name,
                step + 1,
                verification.reason,
            )

        if effective_ui_changed:
            if semantic_verify_result is None and not process.should_verify_action_after_ui_change(
                memory, display_action
            ):
                stage_before_success = memory.current_stage
                if is_action_bundle:
                    process.on_actions_success(memory, actions_list)
                else:
                    process.on_action_success(memory, display_action)
                process.on_stage_success(memory, display_action, stage_name=stage_before_success)
                memory.failure_count = 0
                rollback_used = False
                _sync_policy_cache_to_context()
                _log_policy_state("stage-success", actions_list)
                if _complete_from_terminal_state():
                    break
                continue

            semantic_verify = semantic_verify_result
            if semantic_verify is None or not verify_without_ui_change:
                semantic_verify = process.verify_action_success(
                    planner_provider,
                    post_image,
                    memory,
                    display_action,
                    img_config=planner_img_config,
                )
            if semantic_verify.handled:
                memory.set_last_semantic_verify(
                    "pass" if semantic_verify.passed else "fail",
                    semantic_verify.reason,
                )
            if semantic_verify.handled and not semantic_verify.passed:
                logger.info(
                    "Semantic verification failed for %s at stage %s: %s",
                    primitive_name,
                    memory.current_stage,
                    semantic_verify.reason,
                )
                no_progress_resolution = _call_process_method_with_optional_prompt_language(
                    process.handle_no_progress,
                    planner_provider,
                    post_image,
                    memory,
                    last_action=actions_list[-1],
                    normalizing_range=normalizing_range,
                    high_level_strategy=strategy_string,
                    recent_actions=recent_actions_str,
                    hitl_directive=hitl_directive,
                    img_config=planner_img_config,
                )
                if no_progress_resolution.handled:
                    memory.failure_count = 0
                    _sync_policy_cache_to_context()
                    _refresh_multi_step_debug()
                    _log_policy_state("semantic-no-progress handled", actions_list)
                    continue
                if no_progress_resolution.reroute:
                    result.re_route = True
                    result.error_message = no_progress_resolution.error_message or "Semantic verification failed"
                    _sync_policy_cache_to_context()
                    _log_policy_state("semantic-no-progress reroute", actions_list)
                    break

            stage_before_success = memory.current_stage
            if is_action_bundle:
                process.on_actions_success(memory, actions_list)
            else:
                process.on_action_success(memory, display_action)
            process.on_stage_success(memory, display_action, stage_name=stage_before_success)
            memory.failure_count = 0
            rollback_used = False
            _sync_policy_cache_to_context()
            _log_policy_state("stage-success", actions_list)
            if _complete_from_terminal_state():
                break
            continue

        no_progress_resolution = _call_process_method_with_optional_prompt_language(
            process.handle_no_progress,
            planner_provider,
            post_image,
            memory,
            last_action=actions_list[-1],
            normalizing_range=normalizing_range,
            high_level_strategy=strategy_string,
            recent_actions=recent_actions_str,
            hitl_directive=hitl_directive,
            img_config=planner_img_config,
        )
        if no_progress_resolution.handled:
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="retry",
                summary="no progress handled",
                detail=memory.last_planned_action or "",
            )
            memory.failure_count = 0
            _sync_policy_cache_to_context()
            _refresh_multi_step_debug()
            _log_policy_state("no-progress handled", actions_list)
            continue
        if no_progress_resolution.reroute:
            result.re_route = True
            result.error_message = no_progress_resolution.error_message or "Process requested reroute"
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="error",
                summary=result.error_message,
            )
            _sync_policy_cache_to_context()
            _log_policy_state("no-progress reroute", actions_list)
            break

        memory.failure_count += 1
        logger.debug("No UI change detected for %s (count=%s)", primitive_name, memory.failure_count)

        if memory.failure_count < 2:
            continue

        if process.supports_observation and not rollback_used and memory.restore_last_checkpoint():
            rollback_used = True
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage=memory.current_stage or "-",
                phase="retry",
                summary="rollback to last checkpoint",
            )
            logger.info("No progress twice -> rollback to last observation checkpoint for %s", primitive_name)
            continue

        result.re_route = True
        result.error_message = "No UI change for 2 consecutive steps"
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage=memory.current_stage or "-",
            phase="error",
            summary=result.error_message,
        )
        logger.info("No UI change for 2 steps — signaling re-route from %s", primitive_name)
        break

    if not result.completed and not result.re_route and not result.error_message:
        current_iteration_limit = _iteration_limit()
        if result.steps_taken >= max_steps:
            result.error_message = f"Primitive loop reached safety action cap ({max_steps}) without completion"
        else:
            result.error_message = (
                f"Primitive loop reached safety iteration cap ({current_iteration_limit}) without completion"
            )
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage=memory.current_stage or "-",
            phase="error",
            summary=result.error_message,
        )
        logger.warning(result.error_message)

    if result.error_message:
        _publish_error_state(result.error_message)

    _sync_policy_cache_to_context()
    if result.success or result.completed:
        _clear_current_action_debug()
    rl.update_multi_step(active=False)
    if state_bridge:
        state_bridge.update_multi_step(active=False)

    return result


def run_one_turn(
    router_provider: BaseVLMProvider,
    planner_provider: BaseVLMProvider,
    normalizing_range: int = 1000,
    delay_before_action: float = 0.5,
    prompt_language: str = "eng",
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
    reroute_retry_budget: int = 3,
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
        prompt_language: Primitive prompt language (`eng` default, `kor` optional)
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

    def _retry_turn_via_reroute(error_message: str) -> TurnSummary | None:
        if reroute_retry_budget <= 0:
            return None
        if error_message == "STOP directive received during loop":
            return None
        logger.info(
            "Recoverable turn failure -> retry full reroute for turn %s (%s retry left after this): %s",
            turn_number,
            reroute_retry_budget - 1,
            error_message,
        )
        return run_one_turn(
            router_provider=router_provider,
            planner_provider=planner_provider,
            normalizing_range=normalizing_range,
            delay_before_action=delay_before_action,
            prompt_language=prompt_language,
            high_level_strategy=high_level_strategy,
            context_manager=ctx,
            strategy_planner=strategy_planner,
            knowledge_manager=knowledge_manager,
            turn_number=turn_number,
            macro_turn_manager=macro_turn_manager,
            state_bridge=state_bridge,
            context_updater=context_updater,
            debug_options=dbg,
            command_queue=command_queue,
            agent_gate=agent_gate,
            strategy_updater=strategy_updater,
            turn_detector=turn_detector,
            router_img_config=router_img_config,
            planner_img_config=planner_img_config,
            reroute_retry_budget=reroute_retry_budget - 1,
        )

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

    # HITL primitive override — force a specific primitive (e.g. war_primitive, deal_primitive)
    if queue_result.override_primitive:
        primitive_name = queue_result.override_primitive
        ctx.set_current_primitive(primitive_name)
        rl.hitl_event("PRIMITIVE_OVERRIDE", f"Forced primitive: {primitive_name}")

    if queue_result.override_action:
        # Skip VLM planning — execute user's primitive override directly
        action = queue_result.override_action
        rl.hitl_event("OVERRIDE", f"{action.action} ({action.x}, {action.y})")
        if state_bridge:
            state_bridge.update_current_action(
                primitive_name, f"[HITL] {action.action} ({action.x}, {action.y})", action.reasoning
            )
            state_bridge.broadcast_agent_phase("사용자 명령 실행 중")

        # Execute single override action (existing path)
        rl.update_phase("executing")
        if delay_before_action > 0:
            time.sleep(delay_before_action)
        try:
            execute_action(action, screen_w, screen_h, normalizing_range, x_offset, y_offset)
            rl.execution_status(True)
        except Exception as e:
            rl.execution_status(False, str(e))

        ctx.record_action(
            action_type=action.action,
            primitive=primitive_name,
            x=action.x,
            y=action.y,
            result="success",
        )
        rl.update_phase("idle")
        rl.turn_summary(turn_number, primitive_name, action.action, True)
        return TurnSummary(
            turn_number=turn_number,
            primitive=primitive_name,
            action_type=action.action,
            success=True,
            reasoning=action.reasoning or "",
            coords=(action.x, action.y),
        )

    # --- Step 3: Build strategy & recent actions (shared by both paths) ---
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

    # --- Check if this primitive is multi-step ---
    registry_entry = PRIMITIVE_REGISTRY.get(primitive_name, {})
    is_multi_step = registry_entry.get("multi_step", False)

    if is_multi_step:
        # === Multi-step primitive loop ===
        memory = ShortTermMemory()
        memory.start_task(
            primitive_name,
            registry_entry.get("max_steps", 10),
            normalizing_range=normalizing_range,
            enable_choice_catalog=registry_entry.get("process_kind") == "observation_assisted",
            enable_policy_state=primitive_name == "policy_primitive",
            enable_voting_state=primitive_name == "voting_primitive",
        )
        if primitive_name == "policy_primitive":
            memory.set_policy_capture_geometry(screen_w, screen_h, x_offset, y_offset)
            policy_cache = ctx.get_policy_tab_cache()
            cache_geometry = getattr(policy_cache, "capture_geometry", None)
            geometry_matches = (
                cache_geometry is not None
                and int(getattr(cache_geometry, "region_w", -1)) == screen_w
                and int(getattr(cache_geometry, "region_h", -1)) == screen_h
                and int(getattr(cache_geometry, "x_offset", -1)) == x_offset
                and int(getattr(cache_geometry, "y_offset", -1)) == y_offset
            )
            if policy_cache.positions and not geometry_matches:
                logger.info("Cleared stale session policy tab cache before loop start due to geometry mismatch")
                ctx.clear_policy_tab_cache()
                policy_cache = ctx.get_policy_tab_cache()
            memory.seed_policy_tab_cache(policy_cache)

        # Override image config if primitive has a specific preset
        primitive_img_override = registry_entry.get("img_config_preset")
        if primitive_img_override:
            from computer_use_test.utils.image_pipeline import PRESETS

            effective_img_config = PRESETS.get(primitive_img_override, planner_img_config)
        else:
            effective_img_config = planner_img_config

        loop_result = run_primitive_loop(
            planner_provider=planner_provider,
            primitive_name=primitive_name,
            screen_w=screen_w,
            screen_h=screen_h,
            normalizing_range=normalizing_range,
            x_offset=x_offset,
            y_offset=y_offset,
            strategy_string=strategy_string or "",
            recent_actions_str=recent_actions_str,
            hitl_directive=primitive_hint if primitive_hint else None,
            memory=memory,
            ctx=ctx,
            max_steps=registry_entry.get("max_steps", 10),
            completion_condition=registry_entry.get("completion_condition", ""),
            prompt_language=prompt_language,
            planner_img_config=effective_img_config,
            command_queue=command_queue,
            agent_gate=agent_gate,
            state_bridge=state_bridge,
            strategy_updater=strategy_updater,
            delay_before_action=delay_before_action,
        )

        # Handle follow-up routing after multi-step completion, reroute, or recoverable loop failure.
        if _should_follow_up_route(loop_result):
            if loop_result.error_message and not (loop_result.re_route or loop_result.completed):
                logger.info(
                    "Recoverable multi-step failure for %s -> follow-up reroute: %s",
                    primitive_name,
                    loop_result.error_message,
                )
            same_primitive_restart_used = False
            for _ in range(3):
                if not _should_follow_up_route(loop_result):
                    break
                re_image, *_ = capture_screen_pil()
                rl.update_phase("routing")
                if state_bridge:
                    state_bridge.broadcast_agent_phase("라우팅 중...")
                new_router = route_primitive(router_provider, re_image, img_config=router_img_config)
                reroute_macro_turn = macro_turn_manager.macro_turn_number if macro_turn_manager else macro_turn
                reroute_detected_turn = turn_detector.latest_turn if turn_detector else detected_turn
                rl.route_result(
                    new_router.primitive,
                    new_router.reasoning,
                    reroute_detected_turn,
                    reroute_macro_turn,
                    turn_number,
                )
                reroute_detail = f"follow-up route -> {new_router.primitive}"
                if new_router.reasoning:
                    reroute_detail = f"{reroute_detail} | {new_router.reasoning}"
                rl.primitive_event("ROUTER", reroute_detail)
                if (
                    new_router.primitive == primitive_name
                    and loop_result.completed
                    and primitive_name in {"voting_primitive", "city_production_primitive"}
                ):
                    logger.info(
                        "Completed %s rerouted back to same primitive "
                        "-> retrying routing without restarting completed loop",
                        primitive_name,
                    )
                    continue
                if new_router.primitive == primitive_name and loop_result.re_route:
                    if primitive_name == "policy_primitive" and not same_primitive_restart_used:
                        same_primitive_restart_used = True
                        logger.info("Policy requested re-route to same primitive -> restarting same primitive once")
                        preserve_entry_done = memory.is_policy_entry_done()
                        memory.clear_policy_bootstrap(
                            preserve_entry_done=preserve_entry_done,
                            preserve_progress=True,
                        )
                        memory.begin_stage("bootstrap_tabs" if preserve_entry_done else "policy_entry")
                        memory.set_policy_mode("structured")
                        memory.set_policy_event("same-primitive reroute -> preserved restart")
                        rl.update_phase("planning")
                        if state_bridge:
                            state_bridge.broadcast_agent_phase("추론 중...")
                        loop_result = run_primitive_loop(
                            planner_provider=planner_provider,
                            primitive_name=primitive_name,
                            screen_w=screen_w,
                            screen_h=screen_h,
                            normalizing_range=normalizing_range,
                            x_offset=x_offset,
                            y_offset=y_offset,
                            strategy_string=strategy_string or "",
                            recent_actions_str=recent_actions_str,
                            hitl_directive=primitive_hint if primitive_hint else None,
                            memory=memory,
                            ctx=ctx,
                            max_steps=registry_entry.get("max_steps", 10),
                            completion_condition=registry_entry.get("completion_condition", ""),
                            prompt_language=prompt_language,
                            planner_img_config=effective_img_config,
                            command_queue=command_queue,
                            agent_gate=agent_gate,
                            state_bridge=state_bridge,
                            strategy_updater=strategy_updater,
                            delay_before_action=delay_before_action,
                        )
                        if not loop_result.re_route:
                            break
                        continue
                    break  # Same primitive → give up re-routing
                primitive_name = new_router.primitive
                ctx.set_current_primitive(primitive_name)
                logger.info(f"Follow-up routed to: {primitive_name}")

                entry = PRIMITIVE_REGISTRY.get(primitive_name, {})
                rl.update_phase("planning")
                if state_bridge:
                    state_bridge.broadcast_agent_phase("추론 중...")
                if not entry.get("multi_step", False):
                    # Single-step primitive → run once and exit
                    action = plan_action(
                        planner_provider,
                        re_image,
                        primitive_name,
                        normalizing_range,
                        high_level_strategy=strategy_string,
                        recent_actions_string=recent_actions_str,
                        hitl_directive=primitive_hint if primitive_hint else None,
                        prompt_language=prompt_language,
                        img_config=planner_img_config,
                    )
                    if action is not None and not (isinstance(action, list) and len(action) == 0):
                        single_act = action[0] if isinstance(action, list) else action
                        _emit_runtime_trace(
                            rl=rl,
                            state_bridge=state_bridge,
                            primitive_name=primitive_name,
                            stage="single_step",
                            phase="plan",
                            summary=f"{single_act.action} @ ({single_act.x}, {single_act.y})",
                            detail=single_act.reasoning or "",
                        )
                        if delay_before_action > 0:
                            time.sleep(delay_before_action)
                        execute_action(single_act, screen_w, screen_h, normalizing_range, x_offset, y_offset)
                        _emit_runtime_trace(
                            rl=rl,
                            state_bridge=state_bridge,
                            primitive_name=primitive_name,
                            stage="single_step",
                            phase="exec",
                            summary=f"{single_act.action} executed",
                            detail=single_act.reasoning or "",
                        )
                        ctx.record_action(
                            action_type=single_act.action,
                            primitive=primitive_name,
                            x=single_act.x,
                            y=single_act.y,
                            result="success",
                        )
                        loop_result.last_action = single_act
                        loop_result.success = True
                        loop_result.completed = single_act.task_status == "complete"
                        loop_result.re_route = False
                        loop_result.steps_taken = 1
                    break

                # Re-routed to another multi-step primitive
                memory.reset()
                memory.start_task(
                    primitive_name,
                    entry.get("max_steps", 10),
                    normalizing_range=normalizing_range,
                    enable_choice_catalog=entry.get("process_kind") == "observation_assisted",
                    enable_policy_state=primitive_name == "policy_primitive",
                    enable_voting_state=primitive_name == "voting_primitive",
                )
                if primitive_name == "policy_primitive":
                    memory.set_policy_capture_geometry(screen_w, screen_h, x_offset, y_offset)
                    policy_cache = ctx.get_policy_tab_cache()
                    cache_geometry = getattr(policy_cache, "capture_geometry", None)
                    geometry_matches = (
                        cache_geometry is not None
                        and int(getattr(cache_geometry, "region_w", -1)) == screen_w
                        and int(getattr(cache_geometry, "region_h", -1)) == screen_h
                        and int(getattr(cache_geometry, "x_offset", -1)) == x_offset
                        and int(getattr(cache_geometry, "y_offset", -1)) == y_offset
                    )
                    if policy_cache.positions and not geometry_matches:
                        logger.info(
                            "Cleared stale session policy tab cache before rerouted loop due to geometry mismatch"
                        )
                        ctx.clear_policy_tab_cache()
                        policy_cache = ctx.get_policy_tab_cache()
                    memory.seed_policy_tab_cache(policy_cache)
                loop_result = run_primitive_loop(
                    planner_provider=planner_provider,
                    primitive_name=primitive_name,
                    screen_w=screen_w,
                    screen_h=screen_h,
                    normalizing_range=normalizing_range,
                    x_offset=x_offset,
                    y_offset=y_offset,
                    strategy_string=strategy_string or "",
                    recent_actions_str=recent_actions_str,
                    hitl_directive=primitive_hint if primitive_hint else None,
                    memory=memory,
                    ctx=ctx,
                    max_steps=entry.get("max_steps", 10),
                    completion_condition=entry.get("completion_condition", ""),
                    prompt_language=prompt_language,
                    planner_img_config=planner_img_config,
                    command_queue=command_queue,
                    agent_gate=agent_gate,
                    state_bridge=state_bridge,
                    strategy_updater=strategy_updater,
                    delay_before_action=delay_before_action,
                )
                if not (loop_result.re_route or loop_result.completed):
                    break

        memory.reset()
        rl.update_phase("idle")

        if not loop_result.success:
            retry_summary = _retry_turn_via_reroute(loop_result.error_message or "multi-step loop failed")
            if retry_summary is not None:
                return retry_summary

        # Build TurnSummary from loop result
        last = loop_result.last_action
        action_type = last.action if last else "none"
        action_desc = f"{action_type}(loop:{loop_result.steps_taken})"
        rl.turn_summary(turn_number, primitive_name, action_desc, loop_result.success)

        if macro_turn_manager and last:
            macro_turn_manager.record_micro_turn(primitive_name, last.reasoning or "")

        return TurnSummary(
            turn_number=turn_number,
            primitive=primitive_name,
            action_type=action_type,
            success=loop_result.success,
            reasoning=last.reasoning if last else "",
            error_message=loop_result.error_message,
            coords=(last.x, last.y) if last else (0, 0),
        )

    # === Single-step primitive (existing path) ===
    action = plan_action(
        planner_provider,
        pil_image,
        primitive_name,
        normalizing_range,
        high_level_strategy=strategy_string,
        recent_actions_string=recent_actions_str,
        hitl_directive=primitive_hint if primitive_hint else None,
        prompt_language=prompt_language,
        img_config=planner_img_config,
    )

    # Handle None and empty list
    if action is None or (isinstance(action, list) and len(action) == 0):
        logger.error("  VLM returned no action. Turn aborted.")
        rl.update_phase("idle")
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage="single_step",
            phase="error",
            summary="VLM returned no action",
        )
        ctx.record_action(
            action_type="none",
            primitive=primitive_name,
            result="failed",
            error_message="VLM returned no action",
        )
        retry_summary = _retry_turn_via_reroute("VLM returned no action")
        if retry_summary is not None:
            return retry_summary
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
    plan_summary = (
        f"{display_action.action} @ ({display_action.x}, {display_action.y})"
        if display_action.action != "drag"
        else (
            f"{display_action.action} @ ({display_action.x}, {display_action.y}) -> "
            f"({display_action.end_x}, {display_action.end_y})"
        )
    )
    _emit_runtime_trace(
        rl=rl,
        state_bridge=state_bridge,
        primitive_name=primitive_name,
        stage="single_step",
        phase="plan",
        summary=plan_summary,
        detail=display_action.reasoning or "",
    )
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
    first_action = actions_list[0]

    execution_result = "success"
    error_message = ""
    for i, act in enumerate(actions_list):
        try:
            execute_action(act, screen_w, screen_h, normalizing_range, x_offset, y_offset)
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage="single_step",
                phase="exec",
                summary=f"{act.action} executed",
                detail=act.reasoning or "",
            )
            if len(actions_list) > 1:
                logger.debug(f"Multi-action {i + 1}/{len(actions_list)} executed: {act.action}")
                if i < len(actions_list) - 1:
                    time.sleep(0.3)
        except Exception as e:
            rl.execution_status(False, str(e))
            execution_result = "failed"
            error_message = str(e)
            _emit_runtime_trace(
                rl=rl,
                state_bridge=state_bridge,
                primitive_name=primitive_name,
                stage="single_step",
                phase="error",
                summary=f"{act.action} failed",
                detail=str(e),
            )
            break

    if execution_result == "success":
        rl.execution_status(True)
        _emit_runtime_trace(
            rl=rl,
            state_bridge=state_bridge,
            primitive_name=primitive_name,
            stage="single_step",
            phase="complete",
            summary="single-step execution complete",
            detail=first_action.reasoning or "",
        )

    rl.update_phase("idle")

    # Step 5: Record action(s) in context
    # Use the first action for summary; record all in context
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

    if execution_result != "success":
        retry_summary = _retry_turn_via_reroute(error_message or "single-step execution failed")
        if retry_summary is not None:
            return retry_summary

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
    prompt_language: str = "eng",
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
        prompt_language: Primitive prompt language (`eng` default, `kor` optional)
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
    completed_all_turns = True

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
                prompt_language=prompt_language,
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
                completed_all_turns = False
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
                completed_all_turns = False
                break

            # Fallback: Check for async interrupt between turns (legacy)
            if not command_queue and interrupt_monitor and checkpoint and interrupt_monitor.is_interrupted():
                decision = checkpoint.prompt(summary)

                if decision == CheckpointDecision.STOP:
                    rl.hitl_event("STOP", "User requested stop at checkpoint.")
                    completed_all_turns = False
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

    if completed_all_turns:
        rl.console.print("[bold green]All turns completed.[/bold green]")
    else:
        rl.console.print("[bold yellow]Run stopped before all turns completed.[/bold yellow]")
    logger.debug(f"Context summary: {ctx}")
