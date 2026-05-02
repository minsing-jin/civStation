"""civ6-mcp turn loop — parallel to turn_executor.run_one_turn for the VLM backend.

This is intentionally a separate top-level entry rather than a branch inside
run_one_turn(). Mixing the two pipelines invites accidental coupling and
makes diffs hard to read. The shared surface is just:
    - ContextManager
    - StrategyPlanner / StrategyUpdater
    - HITL: AgentGate / CommandQueue / state bridge
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from civStation.agent.modules.backend.civ6_mcp.client import (
    Civ6McpClient,
    Civ6McpConfig,
    Civ6McpError,
    Civ6McpUnavailableError,
)
from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    ToolCall,
    ToolCallResult,
)
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    normalize_observation_bundle,
)
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver, build_civ6_mcp_observer
from civStation.agent.modules.backend.civ6_mcp.operations import (
    END_TURN_REFLECTION_FIELDS,
    Civ6McpRequestBuilder,
)
from civStation.agent.modules.backend.civ6_mcp.outcome import (
    append_civ6_mcp_turn_outcome,
    build_civ6_mcp_turn_outcome_record,
)
from civStation.agent.modules.backend.civ6_mcp.planner import (
    Civ6McpToolPlanner,
    MissingEndTurnPlannerOutputError,
)
from civStation.agent.modules.backend.civ6_mcp.planner_types import Civ6McpPlannerProvider
from civStation.agent.modules.backend.civ6_mcp.turn_planning import Civ6McpTurnPlan, build_prioritized_turn_plan

if TYPE_CHECKING:
    from civStation.agent.modules.backend.civ6_mcp.state_parser import StateBundle
    from civStation.agent.modules.context.context_manager import ContextManager
    from civStation.agent.modules.hitl.agent_gate import AgentGate
    from civStation.agent.modules.hitl.command_queue import CommandQueue

logger = logging.getLogger(__name__)

Civ6McpObserverFactory = Callable[..., Civ6McpObserver]
Civ6McpClientFactory = Callable[..., Civ6McpClient]
DEFAULT_CIV6_MCP_STRATEGY = "Pursue a science victory while avoiding unnecessary wars."


@dataclass(frozen=True)
class Civ6McpTurnLoopConfig:
    """Configuration inputs for the civ6-mcp turn loop."""

    max_planner_calls_per_turn: int = 25
    delay_between_turns: float = 1.0
    default_strategy: str = DEFAULT_CIV6_MCP_STRATEGY
    synthesize_end_turn_on_missing: bool = True
    persist_turn_outcomes: bool = True
    outcome_log_path: Path | str | None = None

    def validate(self) -> None:
        """Validate runtime bounds before a turn starts."""
        if self.max_planner_calls_per_turn < 1:
            raise ValueError("max_planner_calls_per_turn must be at least 1")
        if self.delay_between_turns < 0:
            raise ValueError("delay_between_turns must be non-negative")
        if not self.default_strategy.strip():
            raise ValueError("default_strategy must be non-empty")


Civ6McpTurnConfig = Civ6McpTurnLoopConfig


@dataclass
class Civ6McpTurnState:
    """Observable state for one civ6-mcp turn-loop iteration."""

    turn_index: int = 0
    phase: str = "initialized"
    strategy: str = DEFAULT_CIV6_MCP_STRATEGY
    observation: StateBundle | None = None
    normalized_observation: Civ6McpNormalizedObservation | None = None
    turn_plan: Civ6McpTurnPlan | None = None
    planner_context: str = ""
    planner_output: dict[str, Any] = field(default_factory=dict)
    recent_tool_results: list[ToolCallResult] = field(default_factory=list)
    error_message: str = ""


@dataclass(frozen=True)
class Civ6McpTurnRequestContext:
    """Stable request context passed into a single civ6-mcp turn iteration."""

    turn_index: int
    planner_provider: object
    context_manager: ContextManager
    strategy_planner: object | None = None
    strategy_updater: object | None = None
    high_level_strategy: str | None = None
    command_queue: CommandQueue | None = None
    agent_gate: AgentGate | None = None
    state_bridge: object | None = None

    def to_run_one_kwargs(self, *, civ6_mcp_client: Civ6McpClient) -> dict[str, object]:
        """Render this context as keyword arguments for ``run_one_turn_civ6_mcp``."""
        return {
            "civ6_mcp_client": civ6_mcp_client,
            "planner_provider": self.planner_provider,
            "context_manager": self.context_manager,
            "strategy_planner": self.strategy_planner,
            "strategy_updater": self.strategy_updater,
            "high_level_strategy": self.high_level_strategy,
            "command_queue": self.command_queue,
            "agent_gate": self.agent_gate,
            "state_bridge": self.state_bridge,
            "turn_index": self.turn_index,
        }


@dataclass
class Civ6McpTurnResult:
    """Outcome of one civ6-mcp turn iteration."""

    turn_index: int
    success: bool = False
    tool_results: list[ToolCallResult] = field(default_factory=list)
    end_turn_called: bool = False
    end_turn_text: str = ""
    error_message: str = ""
    game_over: bool = False
    terminal_condition: str = ""
    state: Civ6McpTurnState = field(default_factory=Civ6McpTurnState)
    structured_outcome: dict[str, Any] = field(default_factory=dict)
    outcome_log_path: str = ""
    synthesized_end_turn_reflection: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep default result state aligned with the result turn index."""
        if self.state.turn_index == 0 and self.turn_index != 0:
            self.state.turn_index = self.turn_index


def _format_recent_tool_history(history: list[ToolCallResult], limit: int = 10) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for outcome in history[-limit:]:
        prefix = "OK" if outcome.success else outcome.classification.upper()
        line = f"[{prefix}] {outcome.call.tool}({_brief_args(outcome.call.arguments)})"
        if outcome.text and not outcome.success:
            line += f" -> {outcome.text[:120]}"
        lines.append(line)
    return "\n".join(lines)


def _brief_args(arguments: dict) -> str:
    if not arguments:
        return ""
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 32:
            value = value[:29] + "..."
        parts.append(f"{key}={value!r}")
    return ", ".join(parts)


def _record_tool_outcomes(ctx, primitive_label: str, outcomes: list[ToolCallResult]) -> None:
    """Push tool outcomes into ContextManager.primitive_context.recent_actions."""
    for outcome in outcomes:
        # Reuse 'type' as a synthetic action_type so the existing action log
        # rendering in get_context_for_primitive() shows something readable.
        try:
            ctx.record_action(
                action_type="tool",
                primitive=primitive_label,
                text=outcome.call.tool,
                result="success" if outcome.success else "failed",
                error_message="" if outcome.success else outcome.text or outcome.error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ContextManager.record_action failed: %s", exc)


def _format_failed_operation(turn_index: int, outcome: ToolCallResult) -> str:
    classification = outcome.classification or "error"
    message = f"civ6-mcp operation failed: tool={outcome.call.tool!r} classification={classification!r}"
    logger.warning(
        "civ6-mcp operation failed: turn=%d tool=%s classification=%s error=%s text=%s",
        turn_index,
        outcome.call.tool,
        classification,
        outcome.error,
        outcome.text,
    )
    return message


def _finalize_failed_turn(result: Civ6McpTurnResult, *, phase: str = "failed") -> Civ6McpTurnResult:
    result.success = False
    result.state.phase = phase
    result.state.error_message = result.error_message
    logger.warning(
        "civ6-mcp turn %d failed: phase=%s error=%s",
        result.turn_index,
        phase,
        result.error_message,
    )
    return result


def _finalize_terminal_turn(result: Civ6McpTurnResult) -> Civ6McpTurnResult:
    result.success = False
    result.state.phase = "game_over" if result.game_over else "failed"
    result.state.error_message = result.error_message
    log = logger.info if result.game_over else logger.warning
    log(
        "civ6-mcp turn %d reached terminal condition: %s",
        result.turn_index,
        result.terminal_condition or "unknown",
    )
    return result


def _finalize_stop_requested_turn(result: Civ6McpTurnResult) -> Civ6McpTurnResult:
    """Finalize a turn stopped by HITL before the next MCP tool dispatch."""
    result.error_message = "stop requested mid-turn"
    result.state.error_message = result.error_message
    result.state.phase = "stopped"
    return _finalize_failed_turn(result, phase="stopped")


def _persist_structured_turn_outcome(
    result: Civ6McpTurnResult,
    *,
    config: Civ6McpTurnLoopConfig,
) -> Civ6McpTurnResult:
    """Persist one backend-local structured turn record without affecting turn flow."""
    if not config.persist_turn_outcomes:
        return result
    try:
        record = build_civ6_mcp_turn_outcome_record(result)
        output_path = append_civ6_mcp_turn_outcome(record, path=config.outcome_log_path)
        result.structured_outcome = record.to_dict()
        result.outcome_log_path = str(output_path)
        logger.debug("civ6-mcp turn %d structured outcome persisted: %s", result.turn_index, output_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("civ6-mcp turn %d structured outcome persistence failed: %s", result.turn_index, exc)
    return result


def _planner_output_snapshot(plan: object) -> dict[str, Any]:
    """Return the planner output fields required by per-turn outcome persistence."""
    tool_calls = list(getattr(plan, "tool_calls", []) or [])
    return {
        "raw_response": str(getattr(plan, "raw_response", "") or ""),
        "parsed_payload": getattr(plan, "parsed_payload", None),
        "tool_calls": [_tool_call_snapshot(call) for call in tool_calls],
    }


def _tool_call_snapshot(call: ToolCall) -> dict[str, Any]:
    return {
        "tool": call.tool,
        "arguments": dict(call.arguments),
        "reasoning": call.reasoning,
    }


def synthesize_missing_end_turn_call(*, strategy_text: str) -> ToolCall:
    """Build a backend-local civ6-mcp ``end_turn`` call for missing planner output."""
    request = Civ6McpRequestBuilder.end_turn(
        tactical="Planner did not include end_turn. Closing turn defensively.",
        strategic=strategy_text[:240] or "Hold course on current strategy.",
        tooling="Detected missing end_turn in planner output.",
        planning="Will request a fuller plan next turn.",
        hypothesis="If this recurs the planner prompt may need tightening.",
        reasoning="synthesized end_turn",
    )
    return ToolCall(
        tool=request.tool,
        arguments=dict(request.arguments),
        reasoning=request.reasoning,
    )


def build_synthesized_end_turn_reflection_metadata(
    *,
    turn_index: int,
    call: ToolCall,
    planner_output: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build backend-local metadata for a synthesized civ6-mcp ``end_turn`` call."""
    return {
        "backend": "civ6-mcp",
        "action": call.tool,
        "source": "civ6_mcp_turn_loop",
        "reason": "planner_missing_end_turn",
        "synthesized": True,
        "turn_index": turn_index,
        "reflection_fields": {
            field: str(call.arguments.get(field, ""))
            for field in END_TURN_REFLECTION_FIELDS
            if str(call.arguments.get(field, "")).strip()
        },
        "planner_tool_calls": _planner_tool_names(planner_output),
    }


def _planner_tool_names(planner_output: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(planner_output, Mapping):
        return []
    raw_tool_calls = planner_output.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        return []
    names: list[str] = []
    for item in raw_tool_calls:
        if not isinstance(item, Mapping):
            continue
        tool = item.get("tool")
        if isinstance(tool, str) and tool:
            names.append(tool)
    return names


def run_one_turn_civ6_mcp(
    *,
    civ6_mcp_client: Civ6McpClient,
    planner_provider,
    context_manager: ContextManager,
    strategy_planner=None,
    strategy_updater=None,
    high_level_strategy: str | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge=None,
    turn_index: int = 0,
    max_planner_calls_per_turn: int | None = None,
    turn_config: Civ6McpTurnLoopConfig | None = None,
    observer_factory: Civ6McpObserverFactory | None = None,
) -> Civ6McpTurnResult:
    """Drive a single Civ6 turn through the civ6-mcp tool-call backend."""
    config = turn_config or Civ6McpTurnLoopConfig()
    config.validate()
    effective_max_planner_calls = (
        max_planner_calls_per_turn if max_planner_calls_per_turn is not None else config.max_planner_calls_per_turn
    )
    if effective_max_planner_calls < 1:
        raise ValueError("max_planner_calls_per_turn must be at least 1")

    logger.info("civ6-mcp turn %d starting", turn_index)
    result = Civ6McpTurnResult(
        turn_index=turn_index,
        state=Civ6McpTurnState(
            turn_index=turn_index,
            strategy=high_level_strategy or config.default_strategy,
        ),
    )

    if _check_stop_requested(agent_gate, command_queue):
        return _persist_structured_turn_outcome(_finalize_stop_requested_turn(result), config=config)

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase("civ6-mcp: observing")
        except Exception:  # noqa: BLE001
            pass

    result.state.phase = "observing"
    observer_builder = observer_factory or build_civ6_mcp_observer
    observer = observer_builder(client=civ6_mcp_client, context_manager=context_manager)
    try:
        bundle = observer.observe()
    except Civ6McpError as exc:
        result.error_message = f"observe failed: {exc}"
        result.state.error_message = result.error_message
        result.state.phase = "failed"
        logger.error("civ6-mcp turn %d observe failed: %s", turn_index, exc)
        return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)
    normalized_observation = normalize_observation_bundle(bundle)
    result.state.observation = bundle
    result.state.normalized_observation = normalized_observation

    if bundle.overview.is_game_over:
        result.game_over = True
        result.terminal_condition = "game_over"
        result.end_turn_text = bundle.overview.victory_text or "GAME OVER"
        result.error_message = "terminal classification 'game_over' from observation"
        result.state.phase = "game_over"
        return _persist_structured_turn_outcome(_finalize_terminal_turn(result), config=config)

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase("civ6-mcp: planning")
        except Exception:  # noqa: BLE001
            pass

    # Resolve effective strategy text (StrategyPlanner output if available,
    # else the raw --strategy CLI string, else a sane default).
    result.state.phase = "planning"
    strategy_text = high_level_strategy or config.default_strategy
    try:
        ctx_strategy = context_manager.get_strategy_string()
        if ctx_strategy:
            strategy_text = ctx_strategy
    except Exception:  # noqa: BLE001
        pass
    result.state.strategy = strategy_text

    turn_plan = build_prioritized_turn_plan(normalized_observation, strategy=strategy_text)
    state_context = (
        f"{normalized_observation.planner_context}\n\n## PRIORITIZED_MCP_INTENTS\n{turn_plan.render_for_prompt()}"
    )
    result.state.turn_plan = turn_plan
    result.state.planner_context = state_context
    recent_history: list[ToolCallResult] = []
    recent_str = _format_recent_tool_history(recent_history)

    planner = Civ6McpToolPlanner(
        provider=planner_provider,
        tool_catalog=civ6_mcp_client.tool_schemas(),
    )

    try:
        if hasattr(planner, "plan_from_observation"):
            plan = planner.plan_from_observation(
                observation=normalized_observation,
                strategy=strategy_text,
                recent_calls=recent_str,
            )
        else:
            plan = planner.plan(
                strategy=strategy_text,
                state_context=state_context,
                recent_calls=recent_str,
            )
    except MissingEndTurnPlannerOutputError as exc:
        result.state.planner_output = _planner_output_snapshot(exc)
        plan_calls: list[ToolCall] = []
    except RuntimeError as exc:
        result.error_message = f"planner failed: {exc}"
        result.state.error_message = result.error_message
        result.state.phase = "failed"
        logger.error("civ6-mcp turn %d planner failed: %s", turn_index, exc)
        return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)
    else:
        result.state.planner_output = _planner_output_snapshot(plan)
        if not plan.tool_calls:
            result.error_message = "planner returned empty tool list"
            result.state.error_message = result.error_message
            result.state.phase = "failed"
            logger.error("civ6-mcp turn %d planner returned empty tool list", turn_index)
            return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)

        # Hard cap to defend against runaway planners.
        plan_calls = plan.tool_calls[:effective_max_planner_calls]

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase(f"civ6-mcp: executing ({len(plan_calls)} calls)")
        except Exception:  # noqa: BLE001
            pass

    result.state.phase = "executing"
    executor = Civ6McpExecutor(civ6_mcp_client)

    end_turn_outcome: ToolCallResult | None = None
    failed_operation: ToolCallResult | None = None
    for call in plan_calls:
        if _check_stop_requested(agent_gate, command_queue):
            return _persist_structured_turn_outcome(_finalize_stop_requested_turn(result), config=config)

        outcome = executor.execute(call)
        result.tool_results.append(outcome)
        recent_history.append(outcome)
        result.state.recent_tool_results.append(outcome)

        if call.tool == "end_turn":
            end_turn_outcome = outcome

        if outcome.classification in {"game_over", "aborted", "hang"}:
            result.game_over = outcome.classification == "game_over"
            result.terminal_condition = outcome.classification
            result.error_message = f"terminal classification {outcome.classification!r} at tool {call.tool!r}"
            result.state.error_message = result.error_message
            result.state.phase = "game_over" if result.game_over else "failed"
            break

        if not outcome.success:
            failed_operation = outcome
            result.error_message = _format_failed_operation(turn_index, outcome)
            result.state.error_message = result.error_message
            result.state.phase = "failed"
            break

    _record_tool_outcomes(context_manager, "civ6_mcp", result.tool_results)

    if result.terminal_condition:
        if end_turn_outcome is not None:
            result.end_turn_called = True
            result.end_turn_text = end_turn_outcome.text or end_turn_outcome.error
        return _persist_structured_turn_outcome(_finalize_terminal_turn(result), config=config)

    if failed_operation is not None:
        if end_turn_outcome is not None:
            result.end_turn_called = True
            result.end_turn_text = end_turn_outcome.text or end_turn_outcome.error
        return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)

    if end_turn_outcome is not None:
        result.end_turn_called = True
        result.end_turn_text = end_turn_outcome.text
        result.success = end_turn_outcome.success
    else:
        # Planner forgot end_turn — synthesize one with backend-local
        # reflections so the turn can advance without invoking the VLM path.
        if not config.synthesize_end_turn_on_missing:
            result.error_message = "planner did not emit end_turn"
            result.state.error_message = result.error_message
            result.state.phase = "failed"
            return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)
        logger.warning("civ6-mcp planner did not emit end_turn; synthesizing backend-local end_turn.")
        synth_call = synthesize_missing_end_turn_call(strategy_text=strategy_text)
        result.synthesized_end_turn_reflection = build_synthesized_end_turn_reflection_metadata(
            turn_index=turn_index,
            call=synth_call,
            planner_output=result.state.planner_output,
        )
        if _check_stop_requested(agent_gate, command_queue):
            return _persist_structured_turn_outcome(_finalize_stop_requested_turn(result), config=config)
        synth_outcome = executor.execute(synth_call)
        result.tool_results.append(synth_outcome)
        result.state.recent_tool_results.append(synth_outcome)
        result.end_turn_called = True
        result.end_turn_text = synth_outcome.text
        result.success = synth_outcome.success
        if synth_outcome.classification in {"game_over", "aborted", "hang"}:
            result.game_over = synth_outcome.classification == "game_over"
            result.terminal_condition = synth_outcome.classification
            result.error_message = (
                f"terminal classification {synth_outcome.classification!r} at tool {synth_call.tool!r}"
            )
            result.state.error_message = result.error_message
            return _persist_structured_turn_outcome(_finalize_terminal_turn(result), config=config)
        if not synth_outcome.success:
            result.error_message = _format_failed_operation(turn_index, synth_outcome)
            return _persist_structured_turn_outcome(_finalize_failed_turn(result), config=config)

    if strategy_updater is not None and result.success:
        try:
            from civStation.agent.modules.strategy.strategy_updater import (
                StrategyRequest,
                StrategyTrigger,
            )

            strategy_updater.submit(StrategyRequest(StrategyTrigger.PERIODIC))
        except Exception:  # noqa: BLE001
            pass

    try:
        if result.success:
            context_manager.advance_turn(primitive_used="civ6_mcp", success=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("advance_turn failed: %s", exc)

    result.state.phase = "completed" if result.success else "failed"
    result.state.error_message = result.error_message
    if result.success:
        logger.info(
            "civ6-mcp turn %d completed successfully: tools=%d end_turn=%s",
            turn_index,
            len(result.tool_results),
            result.end_turn_called,
        )
    else:
        logger.warning(
            "civ6-mcp turn %d completed unsuccessfully: tools=%d error=%s",
            turn_index,
            len(result.tool_results),
            result.error_message,
        )
    return _persist_structured_turn_outcome(result, config=config)


def run_multi_turn_civ6_mcp(
    *,
    num_turns: int,
    civ6_mcp_client: Civ6McpClient,
    planner_provider,
    context_manager: ContextManager,
    strategy_planner=None,
    strategy_updater=None,
    high_level_strategy: str | None = None,
    delay_between_turns: float | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge=None,
    turn_config: Civ6McpTurnLoopConfig | None = None,
    observer_factory: Civ6McpObserverFactory | None = None,
) -> list[Civ6McpTurnResult]:
    """Sequential multi-turn driver. Stops early on game-over / abort / stop."""
    config = turn_config or Civ6McpTurnLoopConfig()
    config.validate()
    effective_delay_between_turns = (
        delay_between_turns if delay_between_turns is not None else config.delay_between_turns
    )
    if effective_delay_between_turns < 0:
        raise ValueError("delay_between_turns must be non-negative")

    outcomes: list[Civ6McpTurnResult] = []
    for turn_index in range(num_turns):
        if _check_stop_requested(agent_gate, command_queue):
            logger.info("civ6-mcp: stop requested before turn %d", turn_index)
            break
        request_context = Civ6McpTurnRequestContext(
            turn_index=turn_index,
            planner_provider=planner_provider,
            context_manager=context_manager,
            strategy_planner=strategy_planner,
            strategy_updater=strategy_updater,
            high_level_strategy=high_level_strategy,
            command_queue=command_queue,
            agent_gate=agent_gate,
            state_bridge=state_bridge,
        )
        outcome = run_one_turn_civ6_mcp(
            **request_context.to_run_one_kwargs(civ6_mcp_client=civ6_mcp_client),
            turn_config=config,
            observer_factory=observer_factory,
        )
        outcomes.append(outcome)
        if outcome.terminal_condition:
            logger.info(
                "civ6-mcp: stopping after terminal condition on turn %d: %s",
                turn_index,
                outcome.terminal_condition,
            )
            break
        if not outcome.success and outcome.error_message:
            logger.warning("civ6-mcp turn %d ended unsuccessfully: %s", turn_index, outcome.error_message)
            break
        if effective_delay_between_turns > 0 and turn_index < num_turns - 1:
            time.sleep(effective_delay_between_turns)
    return outcomes


def run_civ6_mcp_turn_loop(
    *,
    num_turns: int,
    install_path: str | None,
    launcher: str | None,
    planner_provider: Civ6McpPlannerProvider,
    context_manager: ContextManager,
    strategy_planner: object | None = None,
    strategy_updater: object | None = None,
    high_level_strategy: str | None = None,
    delay_between_turns: float | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge: object | None = None,
    turn_config: Civ6McpTurnLoopConfig | None = None,
    observer_factory: Civ6McpObserverFactory | None = None,
    client_factory: Civ6McpClientFactory | None = None,
    env_overrides: dict[str, str] | None = None,
) -> list[Civ6McpTurnResult]:
    """Own civ6-mcp connection lifecycle around one or more turn iterations."""
    if num_turns < 1:
        return []
    if _check_stop_requested(agent_gate, command_queue):
        logger.info("civ6-mcp: stop requested before backend startup")
        return []

    client_builder = client_factory or build_civ6_mcp_client
    civ6_mcp_client = client_builder(
        install_path=install_path,
        launcher=launcher,
        env_overrides=env_overrides,
    )
    try:
        _start_civ6_mcp_client(civ6_mcp_client)
        _validate_startup_health(civ6_mcp_client)
        _wait_for_civ6_mcp_readiness(civ6_mcp_client)

        return run_multi_turn_civ6_mcp(
            num_turns=num_turns,
            civ6_mcp_client=civ6_mcp_client,
            planner_provider=planner_provider,
            context_manager=context_manager,
            strategy_planner=strategy_planner,
            strategy_updater=strategy_updater,
            high_level_strategy=high_level_strategy,
            delay_between_turns=delay_between_turns,
            command_queue=command_queue,
            agent_gate=agent_gate,
            state_bridge=state_bridge,
            turn_config=turn_config,
            observer_factory=observer_factory,
        )
    finally:
        try:
            civ6_mcp_client.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("civ6-mcp client stop failed: %s", exc)


def _start_civ6_mcp_client(civ6_mcp_client: object) -> None:
    """Start a civ6-mcp client if the injected client exposes a lifecycle hook."""
    if bool(getattr(civ6_mcp_client, "started", False)) or bool(getattr(civ6_mcp_client, "_started", False)):
        return
    start = getattr(civ6_mcp_client, "start", None)
    if callable(start):
        start()


def _validate_startup_health(civ6_mcp_client: object) -> None:
    startup_health = getattr(civ6_mcp_client, "startup_health", None)
    if startup_health is None or getattr(startup_health, "ok", True):
        return
    message = getattr(startup_health, "message", "") or "startup required-tool validation failed"
    raise Civ6McpError(f"civ6-mcp startup health check failed: {message}")


def _wait_for_civ6_mcp_readiness(civ6_mcp_client: object) -> None:
    """Probe the started civ6-mcp client before handing it to turn execution."""
    health_check = getattr(civ6_mcp_client, "health_check", None)
    if not callable(health_check):
        raise Civ6McpError("civ6-mcp client does not expose a health_check() readiness probe.")

    health = health_check()
    if not getattr(health, "ok", False):
        message = getattr(health, "message", "") or "readiness probe failed"
        raise Civ6McpError(f"civ6-mcp health check failed: {message}")


def _check_stop_requested(
    agent_gate: AgentGate | None,
    command_queue: CommandQueue | None,
) -> bool:
    if agent_gate is not None and getattr(agent_gate, "is_stopped", False):
        return True
    if command_queue is None:
        return False
    try:
        from civStation.agent.modules.hitl.command_queue import DirectiveType
    except Exception:  # noqa: BLE001
        return False
    pending = []
    try:
        pending = list(command_queue.peek())  # type: ignore[attr-defined]
    except AttributeError:
        # Older CommandQueue without peek() — drain non-destructively isn't ideal
        # but we can fall back to checking the gate alone.
        return False
    except Exception:  # noqa: BLE001
        return False
    for directive in pending:
        if getattr(directive, "directive_type", None) == DirectiveType.STOP:
            return True
    return False


def build_civ6_mcp_client(
    *,
    install_path: str | None,
    launcher: str | None,
    env_overrides: dict[str, str] | None = None,
) -> Civ6McpClient:
    """Helper used by turn_runner to construct + start a client safely.

    Raises Civ6McpUnavailableError before start() so the caller can report a
    crisp configuration error instead of a stack trace.
    """
    config = Civ6McpConfig.from_environment(
        install_path=install_path,
        launcher=launcher,
        env_overrides=env_overrides,
    )
    config.validate()
    client = Civ6McpClient(config)
    client.start()
    return client


__all__ = [
    "Civ6McpClientFactory",
    "Civ6McpObserverFactory",
    "Civ6McpTurnConfig",
    "Civ6McpTurnLoopConfig",
    "Civ6McpTurnRequestContext",
    "Civ6McpTurnResult",
    "Civ6McpTurnState",
    "build_civ6_mcp_client",
    "build_civ6_mcp_observer",
    "build_synthesized_end_turn_reflection_metadata",
    "run_civ6_mcp_turn_loop",
    "run_multi_turn_civ6_mcp",
    "run_one_turn_civ6_mcp",
    "Civ6McpUnavailableError",
]
