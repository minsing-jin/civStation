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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner

if TYPE_CHECKING:
    from civStation.agent.modules.hitl.agent_gate import AgentGate
    from civStation.agent.modules.hitl.command_queue import CommandQueue

logger = logging.getLogger(__name__)


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


def run_one_turn_civ6_mcp(
    *,
    civ6_mcp_client: Civ6McpClient,
    planner_provider,
    context_manager,
    strategy_planner=None,
    strategy_updater=None,
    high_level_strategy: str | None = None,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge=None,
    turn_index: int = 0,
    max_planner_calls_per_turn: int = 25,
) -> Civ6McpTurnResult:
    """Drive a single Civ6 turn through the civ6-mcp tool-call backend."""
    result = Civ6McpTurnResult(turn_index=turn_index)

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase("civ6-mcp: observing")
        except Exception:  # noqa: BLE001
            pass

    observer = Civ6McpObserver(civ6_mcp_client, context_manager)
    try:
        bundle = observer.observe()
    except Civ6McpError as exc:
        result.error_message = f"observe failed: {exc}"
        logger.error(result.error_message)
        return result

    if bundle.overview.is_game_over:
        result.game_over = True
        result.end_turn_text = bundle.overview.victory_text or "GAME OVER"
        return result

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase("civ6-mcp: planning")
        except Exception:  # noqa: BLE001
            pass

    # Resolve effective strategy text (StrategyPlanner output if available,
    # else the raw --strategy CLI string, else a sane default).
    strategy_text = high_level_strategy or ""
    try:
        ctx_strategy = context_manager.get_strategy_string()
        if ctx_strategy:
            strategy_text = ctx_strategy
    except Exception:  # noqa: BLE001
        pass
    if not strategy_text:
        strategy_text = "Pursue a science victory while avoiding unnecessary wars."

    state_context = bundle.to_planner_context()
    recent_history: list[ToolCallResult] = []
    recent_str = _format_recent_tool_history(recent_history)

    planner = Civ6McpToolPlanner(
        provider=planner_provider,
        tool_catalog=civ6_mcp_client.tool_schemas(),
    )

    try:
        plan = planner.plan(
            strategy=strategy_text,
            state_context=state_context,
            recent_calls=recent_str,
        )
    except RuntimeError as exc:
        result.error_message = f"planner failed: {exc}"
        logger.error(result.error_message)
        return result

    if not plan.tool_calls:
        result.error_message = "planner returned empty tool list"
        logger.error(result.error_message)
        return result

    # Hard cap to defend against runaway planners.
    plan_calls = plan.tool_calls[:max_planner_calls_per_turn]

    if state_bridge:
        try:
            state_bridge.broadcast_agent_phase(f"civ6-mcp: executing ({len(plan_calls)} calls)")
        except Exception:  # noqa: BLE001
            pass

    executor = Civ6McpExecutor(civ6_mcp_client)

    end_turn_outcome: ToolCallResult | None = None
    for call in plan_calls:
        if _check_stop_requested(agent_gate, command_queue):
            result.error_message = "stop requested mid-turn"
            return result

        outcome = executor.execute(call)
        result.tool_results.append(outcome)
        recent_history.append(outcome)

        if call.tool == "end_turn":
            end_turn_outcome = outcome

        if outcome.classification in {"game_over", "aborted", "hang"}:
            result.game_over = outcome.classification == "game_over"
            result.error_message = f"terminal classification {outcome.classification!r} at tool {call.tool!r}"
            logger.warning(result.error_message)
            break

    _record_tool_outcomes(context_manager, "civ6_mcp", result.tool_results)

    if end_turn_outcome is not None:
        result.end_turn_called = True
        result.end_turn_text = end_turn_outcome.text
        result.success = end_turn_outcome.success
    else:
        # Planner forgot end_turn — synthesize one with apologetic reflections so
        # the turn still advances. Better than getting stuck.
        logger.warning("civ6-mcp planner did not emit end_turn; synthesizing fallback.")
        synth_call = ToolCall(
            tool="end_turn",
            arguments={
                "tactical": "Planner did not include end_turn. Closing turn defensively.",
                "strategic": strategy_text[:240] or "Hold course on current strategy.",
                "tooling": "Detected missing end_turn in planner output.",
                "planning": "Will request a fuller plan next turn.",
                "hypothesis": "If this recurs the planner prompt may need tightening.",
            },
            reasoning="synthesized end_turn",
        )
        synth_outcome = executor.execute(synth_call)
        result.tool_results.append(synth_outcome)
        result.end_turn_called = True
        result.end_turn_text = synth_outcome.text
        result.success = synth_outcome.success

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
        context_manager.advance_turn(primitive_used="civ6_mcp", success=result.success)
    except Exception as exc:  # noqa: BLE001
        logger.debug("advance_turn failed: %s", exc)

    return result


def run_multi_turn_civ6_mcp(
    *,
    num_turns: int,
    civ6_mcp_client: Civ6McpClient,
    planner_provider,
    context_manager,
    strategy_planner=None,
    strategy_updater=None,
    high_level_strategy: str | None = None,
    delay_between_turns: float = 1.0,
    command_queue: CommandQueue | None = None,
    agent_gate: AgentGate | None = None,
    state_bridge=None,
) -> list[Civ6McpTurnResult]:
    """Sequential multi-turn driver. Stops early on game-over / abort / stop."""
    outcomes: list[Civ6McpTurnResult] = []
    for turn_index in range(num_turns):
        if _check_stop_requested(agent_gate, command_queue):
            logger.info("civ6-mcp: stop requested before turn %d", turn_index)
            break
        outcome = run_one_turn_civ6_mcp(
            civ6_mcp_client=civ6_mcp_client,
            planner_provider=planner_provider,
            context_manager=context_manager,
            strategy_planner=strategy_planner,
            strategy_updater=strategy_updater,
            high_level_strategy=high_level_strategy,
            command_queue=command_queue,
            agent_gate=agent_gate,
            state_bridge=state_bridge,
            turn_index=turn_index,
        )
        outcomes.append(outcome)
        if outcome.game_over:
            logger.info("civ6-mcp: game over after turn %d", turn_index)
            break
        if not outcome.success and outcome.error_message:
            logger.warning("civ6-mcp turn %d ended unsuccessfully: %s", turn_index, outcome.error_message)
        if delay_between_turns > 0:
            time.sleep(delay_between_turns)
    return outcomes


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
    "Civ6McpTurnResult",
    "build_civ6_mcp_client",
    "run_multi_turn_civ6_mcp",
    "run_one_turn_civ6_mcp",
    "Civ6McpUnavailableError",
]
