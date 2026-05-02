"""Turn-loop integration hooks for the civ6-mcp observer."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp import turn_loop as turn_loop_module
from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpError
from civStation.agent.modules.backend.civ6_mcp.executor import ToolCall, ToolCallResult
from civStation.agent.modules.backend.civ6_mcp.planner_types import PlannerResult
from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    GameOverviewSnapshot,
    StateBundle,
    parse_game_overview,
)
from civStation.agent.modules.backend.civ6_mcp.turn_loop import (
    Civ6McpTurnLoopConfig,
    Civ6McpTurnRequestContext,
    Civ6McpTurnResult,
    Civ6McpTurnState,
    run_civ6_mcp_turn_loop,
    run_multi_turn_civ6_mcp,
    run_one_turn_civ6_mcp,
    synthesize_missing_end_turn_call,
)
from civStation.agent.modules.hitl.agent_gate import AgentGate
from civStation.agent.modules.hitl.command_queue import CommandQueue


class FakeObserver:
    def __init__(self, bundle: StateBundle) -> None:
        self._bundle = bundle
        self.observe_calls = 0

    def observe(self) -> StateBundle:
        self.observe_calls += 1
        return self._bundle


class FakeLifecycleClient:
    def __init__(
        self,
        *,
        healthy: bool = True,
        message: str = "healthy",
        startup_health: object | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.healthy = healthy
        self.message = message
        self._startup_health = startup_health
        self.start_error = start_error
        self.started = False
        self.start_calls = 0
        self.health_checks = 0
        self.stopped = False
        self.events: list[str] = []

    @property
    def startup_health(self) -> object | None:
        self.events.append("startup_health")
        return self._startup_health

    def start(self) -> None:
        self.events.append("start")
        self.start_calls += 1
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def health_check(self) -> object:
        self.events.append("health_check")
        self.health_checks += 1
        return type("Health", (), {"ok": self.healthy, "message": self.message})()

    def stop(self) -> None:
        self.events.append("stop")
        self.stopped = True


class FakeContextManager:
    def __init__(self) -> None:
        self.recorded_actions: list[dict[str, Any]] = []
        self.advanced_turns: list[dict[str, Any]] = []

    def get_strategy_string(self) -> str:
        return "Prioritize campuses and safe scouting."

    def record_action(self, **kwargs) -> None:  # noqa: ANN003
        self.recorded_actions.append(dict(kwargs))

    def advance_turn(self, **kwargs) -> None:  # noqa: ANN003
        self.advanced_turns.append(dict(kwargs))


class FakePlannerProvider:
    """Text-only provider stub used by the real civ6-mcp planner."""

    def __init__(self, content_sequence: list[str]) -> None:
        self._content_sequence = list(content_sequence)

    def _build_text_content(self, text: str) -> dict[str, str]:
        return {"type": "text", "text": text}

    def _send_to_api(self, content_parts: list[object], **kwargs: object) -> SimpleNamespace:  # noqa: ARG002
        if not self._content_sequence:
            raise AssertionError("FakePlannerProvider exhausted content sequence")
        return SimpleNamespace(content=self._content_sequence.pop(0))


def _game_over_bundle() -> StateBundle:
    return StateBundle(
        overview=GameOverviewSnapshot(
            is_game_over=True,
            victory_text="GAME OVER - victory",
        )
    )


def _active_turn_bundle() -> StateBundle:
    return StateBundle(
        overview=parse_game_overview(
            {
                "turn": 42,
                "era": "Classical Era",
                "current_research": "WRITING",
                "current_civic": "CRAFTSMANSHIP",
            }
        ),
        notifications_text="Notifications:\n- Unit needs orders",
    )


def test_run_one_turn_uses_observer_factory_hook_before_planning() -> None:
    observer = FakeObserver(_game_over_bundle())
    factory_calls: list[dict[str, Any]] = []
    client = object()
    ctx = object()

    def observer_factory(**kwargs) -> FakeObserver:  # noqa: ANN003
        factory_calls.append(dict(kwargs))
        return observer

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=client,  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=ctx,
        observer_factory=observer_factory,
    )

    assert result.game_over is True
    assert result.end_turn_text == "GAME OVER - victory"
    assert observer.observe_calls == 1
    assert factory_calls == [{"client": client, "context_manager": ctx}]
    assert isinstance(result.state, Civ6McpTurnState)
    assert result.state.turn_index == 0
    assert result.state.observation is observer._bundle
    assert result.state.phase == "game_over"


def test_run_one_turn_logs_successful_completion(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class EndTurnPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, **_kwargs: object) -> PlannerResult:
            return PlannerResult(
                tool_calls=[
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "No blockers.",
                            "strategic": "Keep science pace.",
                            "tooling": "All civ6-mcp calls completed.",
                            "planning": "Observe next turn.",
                            "hypothesis": "Next turn exposes new choices.",
                        },
                    )
                ]
            )

    class SuccessfulExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            return ToolCallResult(call=call, success=True, text="Turn advanced.", classification="ok")

    fake_client = SimpleNamespace(tool_schemas=lambda: {"end_turn": {}})
    ctx = FakeContextManager()
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", EndTurnPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", SuccessfulExecutor)

    with caplog.at_level(logging.INFO, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=object(),
            context_manager=ctx,  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is True
    assert result.end_turn_called is True
    assert result.state.phase == "completed"
    assert result.error_message == ""
    assert "civ6-mcp turn 0 completed successfully" in caplog.text
    assert ctx.advanced_turns == [{"primitive_used": "civ6_mcp", "success": True}]


def test_run_one_turn_persists_structured_turn_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class EndTurnPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, **_kwargs: object) -> PlannerResult:
            return PlannerResult(
                tool_calls=[
                    ToolCall(
                        tool="set_research",
                        arguments={"tech_or_civic": "WRITING"},
                        reasoning="Prioritize campuses.",
                    ),
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "No blockers.",
                            "strategic": "Keep science pace.",
                            "tooling": "All civ6-mcp calls completed.",
                            "planning": "Observe next turn.",
                            "hypothesis": "Next turn exposes new choices.",
                        },
                    ),
                ],
                raw_response='{"tool_calls":[{"tool":"set_research"},{"tool":"end_turn"}]}',
                parsed_payload={"tool_calls": [{"tool": "set_research"}, {"tool": "end_turn"}]},
            )

    class SuccessfulExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            return ToolCallResult(
                call=call,
                success=True,
                text=f"{call.tool} ok",
                classification="ok",
                status="success",
            )

    output_path = tmp_path / "outcomes" / "turns.jsonl"
    fake_client = SimpleNamespace(tool_schemas=lambda: {"set_research": {}, "end_turn": {}})
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", EndTurnPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", SuccessfulExecutor)

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=fake_client,  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=FakeContextManager(),  # type: ignore[arg-type]
        observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        turn_config=Civ6McpTurnLoopConfig(outcome_log_path=output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert result.outcome_log_path == str(output_path)
    assert result.structured_outcome == payload
    assert payload["observation_summary"] == "Turn 42 | Era Classical | Research WRITING | Civic CRAFTSMANSHIP"
    assert payload["planner_output"]["tool_calls"] == [
        {
            "arguments": {"tech_or_civic": "WRITING"},
            "reasoning": "Prioritize campuses.",
            "tool": "set_research",
        },
        {
            "arguments": {
                "hypothesis": "Next turn exposes new choices.",
                "planning": "Observe next turn.",
                "strategic": "Keep science pace.",
                "tactical": "No blockers.",
                "tooling": "All civ6-mcp calls completed.",
            },
            "reasoning": "",
            "tool": "end_turn",
        },
    ]
    assert [call["tool"] for call in payload["executed_tool_calls"]] == ["set_research", "end_turn"]
    assert payload["errors"] == []
    assert payload["final_turn_status"] == "completed"


def test_run_one_turn_preserves_terminal_game_over(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class TerminalPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, **_kwargs: object) -> PlannerResult:
            return PlannerResult(
                tool_calls=[
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "No blockers.",
                            "strategic": "Keep science pace.",
                            "tooling": "All civ6-mcp calls completed.",
                            "planning": "Observe next turn.",
                            "hypothesis": "Next turn exposes new choices.",
                        },
                    ),
                    ToolCall(tool="get_units"),
                ]
            )

    class TerminalExecutor:
        def __init__(self, _client: object) -> None:
            self.calls: list[str] = []

        def execute(self, call: ToolCall) -> ToolCallResult:
            self.calls.append(call.tool)
            return ToolCallResult(
                call=call,
                success=False,
                text="*** GAME OVER - VICTORY ***",
                classification="game_over",
            )

    fake_client = SimpleNamespace(tool_schemas=lambda: {"end_turn": {}, "get_units": {}})
    ctx = FakeContextManager()
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", TerminalPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", TerminalExecutor)

    with caplog.at_level(logging.INFO, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=object(),
            context_manager=ctx,  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is False
    assert result.game_over is True
    assert result.terminal_condition == "game_over"
    assert result.end_turn_called is True
    assert result.state.phase == "game_over"
    assert result.error_message == "terminal classification 'game_over' at tool 'end_turn'"
    assert "civ6-mcp turn 0 reached terminal condition: game_over" in caplog.text
    assert ctx.advanced_turns == []


def test_run_one_turn_fails_and_logs_failed_operation_before_synthesizing_end_turn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, **_kwargs: object) -> PlannerResult:
            return PlannerResult(
                tool_calls=[
                    ToolCall(tool="set_research", arguments={"tech_or_civic": "UNKNOWN"}),
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "No blockers.",
                            "strategic": "Keep science pace.",
                            "tooling": "All civ6-mcp calls completed.",
                            "planning": "Observe next turn.",
                            "hypothesis": "Next turn exposes new choices.",
                        },
                    ),
                ]
            )

    class FailingExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            if call.tool == "set_research":
                return ToolCallResult(
                    call=call,
                    success=False,
                    text="Error: must specify a known tech",
                    error="Error: must specify a known tech",
                    classification="error",
                )
            return ToolCallResult(call=call, success=True, text="Turn advanced.", classification="ok")

    fake_client = SimpleNamespace(tool_schemas=lambda: {"set_research": {}, "end_turn": {}})
    ctx = FakeContextManager()
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", FailingPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", FailingExecutor)

    with caplog.at_level(logging.WARNING, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=object(),
            context_manager=ctx,  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is False
    assert result.end_turn_called is False
    assert [outcome.call.tool for outcome in result.tool_results] == ["set_research"]
    assert result.state.phase == "failed"
    assert result.error_message == "civ6-mcp operation failed: tool='set_research' classification='error'"
    assert "civ6-mcp operation failed: turn=0 tool=set_research classification=error" in caplog.text
    assert "synthesizing backend-local end_turn" not in caplog.text
    assert ctx.advanced_turns == []


def test_synthesize_missing_end_turn_call_builds_valid_backend_local_action() -> None:
    call = synthesize_missing_end_turn_call(strategy_text="Expand safely and keep a science lead." * 20)

    assert call.tool == "end_turn"
    assert call.reasoning == "synthesized end_turn"
    assert call.arguments == {
        "tactical": "Planner did not include end_turn. Closing turn defensively.",
        "strategic": ("Expand safely and keep a science lead." * 20)[:240],
        "tooling": "Detected missing end_turn in planner output.",
        "planning": "Will request a fuller plan next turn.",
        "hypothesis": "If this recurs the planner prompt may need tightening.",
    }


def test_run_one_turn_routes_missing_end_turn_planner_output_to_civ6_mcp_synthesis(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    }
    provider = FakePlannerProvider([json.dumps(payload), json.dumps(payload), json.dumps(payload)])
    executed_calls: list[ToolCall] = []

    class FallbackExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            executed_calls.append(call)
            return ToolCallResult(call=call, success=True, text="Turn advanced.", classification="ok")

    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("missing end_turn synthesis must stay inside the civ6-mcp backend")

    fake_client = SimpleNamespace(
        tool_schemas=lambda: {"get_game_overview": {}, "set_research": {}, "end_turn": {}},
    )
    ctx = FakeContextManager()
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", FallbackExecutor)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    with caplog.at_level(logging.WARNING, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=provider,
            context_manager=ctx,  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is True
    assert result.end_turn_called is True
    assert [call.tool for call in executed_calls] == ["end_turn"]
    assert executed_calls[0].arguments["tooling"] == "Detected missing end_turn in planner output."
    assert result.synthesized_end_turn_reflection == {
        "backend": "civ6-mcp",
        "action": "end_turn",
        "source": "civ6_mcp_turn_loop",
        "reason": "planner_missing_end_turn",
        "synthesized": True,
        "turn_index": 0,
        "reflection_fields": {
            "tactical": "Planner did not include end_turn. Closing turn defensively.",
            "strategic": "Prioritize campuses and safe scouting.",
            "tooling": "Detected missing end_turn in planner output.",
            "planning": "Will request a fuller plan next turn.",
            "hypothesis": "If this recurs the planner prompt may need tightening.",
        },
        "planner_tool_calls": ["get_game_overview", "set_research"],
    }
    assert result.structured_outcome["synthesized_end_turn_reflection"] == result.synthesized_end_turn_reflection
    assert result.state.planner_output["parsed_payload"] == payload
    assert [call["tool"] for call in result.state.planner_output["tool_calls"]] == [
        "get_game_overview",
        "set_research",
    ]
    assert "civ6-mcp planner did not emit end_turn; synthesizing backend-local end_turn." in caplog.text
    assert "fallback" not in caplog.text
    assert ctx.advanced_turns == [{"primitive_used": "civ6_mcp", "success": True}]


def test_run_one_turn_checks_agent_gate_before_synthesized_end_turn_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "tool_calls": [
            {"tool": "get_game_overview", "arguments": {}, "reasoning": "scan"},
            {"tool": "set_research", "arguments": {"tech_or_civic": "WRITING"}},
        ]
    }
    provider = FakePlannerProvider([json.dumps(payload), json.dumps(payload), json.dumps(payload)])

    class ForbiddenExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            raise AssertionError(f"stop requested before dispatch; executed {call.tool}")

    fake_client = SimpleNamespace(
        tool_schemas=lambda: {"get_game_overview": {}, "set_research": {}, "end_turn": {}},
    )
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", ForbiddenExecutor)

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=fake_client,  # type: ignore[arg-type]
        planner_provider=provider,
        context_manager=FakeContextManager(),  # type: ignore[arg-type]
        agent_gate=SimpleNamespace(is_stopped=True),  # type: ignore[arg-type]
        observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
    )

    assert result.success is False
    assert result.end_turn_called is False
    assert result.tool_results == []
    assert result.state.phase == "stopped"
    assert result.error_message == "stop requested mid-turn"


def test_run_one_turn_stop_before_observe_stays_inside_civ6_mcp_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp stop handling must not invoke the VLM/computer-use backend")

    def forbidden_observer_factory(**_kwargs: object) -> FakeObserver:
        raise AssertionError("stop requested before turn start; observer must not run")

    class ForbiddenPlanner:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("stop requested before turn start; planner must not run")

    class ForbiddenExecutor:
        def __init__(self, _client: object) -> None:
            raise AssertionError("stop requested before turn start; executor must not run")

    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", ForbiddenPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", ForbiddenExecutor)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=SimpleNamespace(tool_schemas=lambda: {"end_turn": {}}),  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=FakeContextManager(),  # type: ignore[arg-type]
        agent_gate=SimpleNamespace(is_stopped=True),  # type: ignore[arg-type]
        observer_factory=forbidden_observer_factory,
        turn_config=Civ6McpTurnLoopConfig(persist_turn_outcomes=False),
    )

    assert result.success is False
    assert result.tool_results == []
    assert result.end_turn_called is False
    assert result.state.phase == "stopped"
    assert result.error_message == "stop requested mid-turn"


def test_run_one_turn_agent_gate_allows_first_tool_call_then_stops_before_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed_calls: list[ToolCall] = []
    command_queue = CommandQueue()
    agent_gate = AgentGate(command_queue)
    agent_gate.start()

    class MultiCallPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan(self, **_kwargs: object) -> PlannerResult:
            return PlannerResult(
                tool_calls=[
                    ToolCall(tool="get_units", reasoning="First call should be allowed."),
                    ToolCall(
                        tool="set_research",
                        arguments={"tech_or_civic": "WRITING"},
                        reasoning="Second call must not run after stop.",
                    ),
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "No blockers.",
                            "strategic": "Keep science pace.",
                            "tooling": "All civ6-mcp calls completed.",
                            "planning": "Observe next turn.",
                            "hypothesis": "Next turn exposes new choices.",
                        },
                    ),
                ]
            )

    class StopAfterFirstExecutor:
        def __init__(self, _client: object) -> None:
            pass

        def execute(self, call: ToolCall) -> ToolCallResult:
            executed_calls.append(call)
            if len(executed_calls) == 1:
                assert agent_gate.stop() is True
            return ToolCallResult(call=call, success=True, text=f"{call.tool} ok", classification="ok")

    fake_client = SimpleNamespace(tool_schemas=lambda: {"get_units": {}, "set_research": {}, "end_turn": {}})
    ctx = FakeContextManager()
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", MultiCallPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", StopAfterFirstExecutor)

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=fake_client,  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=ctx,  # type: ignore[arg-type]
        agent_gate=agent_gate,
        command_queue=command_queue,
        observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        turn_config=Civ6McpTurnLoopConfig(persist_turn_outcomes=False),
    )

    assert [call.tool for call in executed_calls] == ["get_units"]
    assert [outcome.call.tool for outcome in result.tool_results] == ["get_units"]
    assert [outcome.call.tool for outcome in result.state.recent_tool_results] == ["get_units"]
    assert result.success is False
    assert result.end_turn_called is False
    assert result.state.phase == "stopped"
    assert result.error_message == "stop requested mid-turn"
    assert ctx.advanced_turns == []


def test_run_one_turn_does_not_synthesize_end_turn_for_vlm_planner_payload(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    vlm_payload = {
        "primitive_name": "click_next_turn",
        "reasoning": "A screenshot/computer-use plan must not run on civ6-mcp.",
        "actions": [{"type": "click", "x": 1120, "y": 690, "button": "left"}],
    }
    provider = FakePlannerProvider([json.dumps(vlm_payload), json.dumps(vlm_payload), json.dumps(vlm_payload)])

    class ForbiddenExecutor:
        def __init__(self, _client: object) -> None:
            raise AssertionError("non-civ6-mcp planner failures must not reach backend-local end_turn synthesis")

    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp planner failure handling must not invoke the VLM backend")

    fake_client = SimpleNamespace(
        tool_schemas=lambda: {"get_game_overview": {}, "end_turn": {}},
    )
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", ForbiddenExecutor)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    with caplog.at_level(logging.WARNING, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=provider,
            context_manager=FakeContextManager(),  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is False
    assert result.end_turn_called is False
    assert result.synthesized_end_turn_reflection == {}
    assert result.tool_results == []
    assert result.state.phase == "failed"
    assert "planner failed: civ6-mcp planner exhausted retries" in result.error_message
    assert "VLM/computer-use action-plan payload cannot run" in result.error_message
    assert "synthesizing backend-local end_turn" not in caplog.text
    assert "planner failed" in caplog.text


def test_run_one_turn_does_not_synthesize_end_turn_for_generic_planner_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingPlanner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def plan_from_observation(self, **_kwargs: object) -> PlannerResult:
            raise RuntimeError("planner transport unavailable")

    class ForbiddenExecutor:
        def __init__(self, _client: object) -> None:
            raise AssertionError("generic planner failures must not reach backend-local end_turn synthesis")

    fake_client = SimpleNamespace(tool_schemas=lambda: {"end_turn": {}})
    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", FailingPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", ForbiddenExecutor)

    with caplog.at_level(logging.WARNING, logger=turn_loop_module.__name__):
        result = run_one_turn_civ6_mcp(
            civ6_mcp_client=fake_client,  # type: ignore[arg-type]
            planner_provider=object(),
            context_manager=FakeContextManager(),  # type: ignore[arg-type]
            observer_factory=lambda **_kwargs: FakeObserver(_active_turn_bundle()),
        )

    assert result.success is False
    assert result.end_turn_called is False
    assert result.synthesized_end_turn_reflection == {}
    assert result.tool_results == []
    assert result.state.phase == "failed"
    assert result.error_message == "planner failed: planner transport unavailable"
    assert "synthesizing backend-local end_turn" not in caplog.text


def test_run_one_turn_executes_observe_plan_execute_without_vlm(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    captured_plan_inputs: list[dict[str, str]] = []
    executed_calls: list[ToolCall] = []
    bundle = StateBundle(
        overview=parse_game_overview(
            {
                "turn": 42,
                "era": "Classical Era",
                "current_research": "WRITING",
                "current_civic": "CRAFTSMANSHIP",
            }
        ),
        notifications_text="Notifications:\n- Unit needs orders\n- Choose production",
    )

    class SequencedObserver:
        def __init__(self) -> None:
            events.append("observer:init")

        def observe(self) -> StateBundle:
            events.append("observe")
            return bundle

    class SequencedPlanner:
        def __init__(self, *, provider: object, tool_catalog: dict[str, dict]) -> None:
            assert provider == "planner-provider"
            assert "end_turn" in tool_catalog
            events.append("planner:init")

        def plan(self, *, strategy: str, state_context: str, recent_calls: str) -> PlannerResult:
            events.append("plan")
            captured_plan_inputs.append(
                {
                    "strategy": strategy,
                    "state_context": state_context,
                    "recent_calls": recent_calls,
                }
            )
            return PlannerResult(
                tool_calls=[
                    ToolCall(tool="get_units", reasoning="Inspect units surfaced by observation."),
                    ToolCall(
                        tool="end_turn",
                        arguments={
                            "tactical": "Handled observed blockers.",
                            "strategic": "Keep science plan moving.",
                            "tooling": "Executed through civ6-mcp.",
                            "planning": "Re-observe next turn.",
                            "hypothesis": "Scouting remains safe.",
                        },
                    ),
                ]
            )

    class SequencedExecutor:
        def __init__(self, client: object) -> None:
            assert client is fake_client
            events.append("executor:init")

        def execute(self, call: ToolCall) -> ToolCallResult:
            events.append(f"execute:{call.tool}")
            executed_calls.append(call)
            return ToolCallResult(
                call=call,
                success=True,
                text=f"{call.tool} ok",
                classification="ok",
            )

    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp observe-plan-execute must not invoke the VLM backend")

    fake_client = SimpleNamespace(
        tool_schemas=lambda: {"get_units": {}, "end_turn": {}},
    )
    ctx = FakeContextManager()

    monkeypatch.setattr(turn_loop_module, "Civ6McpToolPlanner", SequencedPlanner)
    monkeypatch.setattr(turn_loop_module, "Civ6McpExecutor", SequencedExecutor)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=fake_client,  # type: ignore[arg-type]
        planner_provider="planner-provider",
        context_manager=ctx,  # type: ignore[arg-type]
        observer_factory=lambda **_kwargs: SequencedObserver(),
    )

    assert result.success is True
    assert result.end_turn_called is True
    assert events == [
        "observer:init",
        "observe",
        "planner:init",
        "plan",
        "executor:init",
        "execute:get_units",
        "execute:end_turn",
    ]
    assert [call.tool for call in executed_calls] == ["get_units", "end_turn"]
    assert captured_plan_inputs == [
        {
            "strategy": "Prioritize campuses and safe scouting.",
            "state_context": captured_plan_inputs[0]["state_context"],
            "recent_calls": "(none)",
        }
    ]
    planner_context = captured_plan_inputs[0]["state_context"]
    assert "## OVERVIEW" in planner_context
    assert '"turn": 42' in planner_context
    assert "## NOTIFICATIONS" in planner_context
    assert "## PRIORITIZED_MCP_INTENTS" in planner_context
    assert "get_city_production" in planner_context
    assert "get_units" in planner_context
    assert result.state.observation is bundle
    assert result.state.normalized_observation is not None
    assert result.state.normalized_observation.backend == "civ6-mcp"
    assert result.state.normalized_observation.global_context_updates == {
        "current_turn": 42,
        "game_era": "Classical",
        "current_research": "WRITING",
        "current_civic": "CRAFTSMANSHIP",
    }
    assert result.state.normalized_observation.raw_sections["NOTIFICATIONS"] == (
        "Notifications:\n- Unit needs orders\n- Choose production"
    )
    assert planner_context.startswith(result.state.normalized_observation.planner_context)
    assert result.state.planner_context == planner_context
    assert ctx.recorded_actions == [
        {
            "action_type": "tool",
            "primitive": "civ6_mcp",
            "text": "get_units",
            "result": "success",
            "error_message": "",
        },
        {
            "action_type": "tool",
            "primitive": "civ6_mcp",
            "text": "end_turn",
            "result": "success",
            "error_message": "",
        },
    ]
    assert ctx.advanced_turns == [{"primitive_used": "civ6_mcp", "success": True}]


def test_run_multi_turn_passes_observer_factory_hook_to_each_turn() -> None:
    observers = [FakeObserver(_game_over_bundle())]
    client = object()
    ctx = object()

    def observer_factory(**_kwargs) -> FakeObserver:  # noqa: ANN003
        return observers.pop(0)

    results = run_multi_turn_civ6_mcp(
        num_turns=3,
        civ6_mcp_client=client,  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=ctx,
        delay_between_turns=0,
        observer_factory=observer_factory,
    )

    assert len(results) == 1
    assert results[0].game_over is True


def test_run_multi_turn_stops_on_non_game_over_terminal_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    run_one_calls: list[int] = []

    def fake_run_one(**kwargs) -> Civ6McpTurnResult:  # noqa: ANN003
        run_one_calls.append(kwargs["turn_index"])
        return Civ6McpTurnResult(
            turn_index=kwargs["turn_index"],
            success=False,
            error_message="terminal classification 'aborted' at tool 'end_turn'",
            terminal_condition="aborted",
        )

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", fake_run_one)

    results = run_multi_turn_civ6_mcp(
        num_turns=3,
        civ6_mcp_client=object(),  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=object(),  # type: ignore[arg-type]
        delay_between_turns=0,
    )

    assert len(results) == 1
    assert results[0].terminal_condition == "aborted"
    assert run_one_calls == [0]


def test_turn_loop_config_carries_runtime_inputs() -> None:
    config = Civ6McpTurnLoopConfig(
        max_planner_calls_per_turn=7,
        delay_between_turns=0.25,
        default_strategy="Build campuses.",
    )

    assert config.max_planner_calls_per_turn == 7
    assert config.delay_between_turns == 0.25
    assert config.default_strategy == "Build campuses."


def test_run_one_turn_accepts_turn_loop_config_defaults() -> None:
    observer = FakeObserver(_game_over_bundle())

    result = run_one_turn_civ6_mcp(
        civ6_mcp_client=object(),  # type: ignore[arg-type]
        planner_provider=object(),
        context_manager=object(),  # type: ignore[arg-type]
        turn_index=3,
        turn_config=Civ6McpTurnLoopConfig(default_strategy="Culture victory."),
        observer_factory=lambda **_kwargs: observer,
    )

    assert result.turn_index == 3
    assert result.state.turn_index == 3
    assert result.state.strategy == "Culture victory."


def test_turn_request_context_renders_run_one_kwargs() -> None:
    client = object()
    provider = object()
    ctx = object()
    request_context = Civ6McpTurnRequestContext(
        turn_index=4,
        planner_provider=provider,
        context_manager=ctx,  # type: ignore[arg-type]
        high_level_strategy="science",
    )

    kwargs = request_context.to_run_one_kwargs(civ6_mcp_client=client)  # type: ignore[arg-type]

    assert kwargs["civ6_mcp_client"] is client
    assert kwargs["planner_provider"] is provider
    assert kwargs["context_manager"] is ctx
    assert kwargs["turn_index"] == 4
    assert kwargs["high_level_strategy"] == "science"


def test_run_civ6_mcp_turn_loop_builds_context_health_checks_and_stops(monkeypatch) -> None:
    client = FakeLifecycleClient()
    provider = object()
    ctx = object()
    factory_calls: list[dict[str, Any]] = []
    run_one_calls: list[dict[str, Any]] = []

    def client_factory(**kwargs):  # noqa: ANN003
        factory_calls.append(dict(kwargs))
        return client

    def fake_run_one(**kwargs):  # noqa: ANN003
        run_one_calls.append(dict(kwargs))
        return Civ6McpTurnResult(turn_index=kwargs["turn_index"], success=True)

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", fake_run_one)

    results = run_civ6_mcp_turn_loop(
        num_turns=1,
        install_path="/tmp/civ6-mcp",
        launcher="python",
        planner_provider=provider,
        context_manager=ctx,  # type: ignore[arg-type]
        high_level_strategy="science",
        client_factory=client_factory,
    )

    assert len(results) == 1
    assert factory_calls == [
        {
            "install_path": "/tmp/civ6-mcp",
            "launcher": "python",
            "env_overrides": None,
        }
    ]
    assert client.health_checks == 1
    assert client.start_calls == 1
    assert client.stopped is True
    assert client.events[:3] == ["start", "startup_health", "health_check"]
    assert run_one_calls[0]["civ6_mcp_client"] is client
    assert run_one_calls[0]["planner_provider"] is provider
    assert run_one_calls[0]["context_manager"] is ctx
    assert run_one_calls[0]["turn_index"] == 0
    assert run_one_calls[0]["high_level_strategy"] == "science"


def test_run_civ6_mcp_turn_loop_executes_multiple_turns_with_backend_only_components(monkeypatch) -> None:
    client = FakeLifecycleClient()
    provider = object()
    ctx = object()
    observer_factory = object()
    turn_config = Civ6McpTurnLoopConfig(delay_between_turns=0)
    run_one_calls: list[dict[str, Any]] = []

    def fake_run_one(**kwargs) -> Civ6McpTurnResult:  # noqa: ANN003
        run_one_calls.append(dict(kwargs))
        return Civ6McpTurnResult(turn_index=kwargs["turn_index"], success=True)

    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run_civ6_mcp_turn_loop must not invoke the VLM backend")

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", fake_run_one)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    results = run_civ6_mcp_turn_loop(
        num_turns=3,
        install_path="/tmp/civ6-mcp",
        launcher="python",
        planner_provider=provider,
        context_manager=ctx,  # type: ignore[arg-type]
        high_level_strategy="science",
        turn_config=turn_config,
        observer_factory=observer_factory,  # type: ignore[arg-type]
        client_factory=lambda **_kwargs: client,
    )

    assert [result.turn_index for result in results] == [0, 1, 2]
    assert client.events == ["start", "startup_health", "health_check", "stop"]
    assert all(call["civ6_mcp_client"] is client for call in run_one_calls)
    assert [call["turn_index"] for call in run_one_calls] == [0, 1, 2]
    assert all(call["planner_provider"] is provider for call in run_one_calls)
    assert all(call["context_manager"] is ctx for call in run_one_calls)
    assert all(call["high_level_strategy"] == "science" for call in run_one_calls)
    assert all(call["turn_config"] is turn_config for call in run_one_calls)
    assert all(call["observer_factory"] is observer_factory for call in run_one_calls)


def test_run_civ6_mcp_turn_loop_stop_before_startup_does_not_fallback_or_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command_queue = CommandQueue()
    agent_gate = AgentGate(command_queue)
    assert agent_gate.stop() is True
    client_factory_calls: list[dict[str, Any]] = []

    def forbidden_client_factory(**kwargs: object) -> FakeLifecycleClient:
        client_factory_calls.append(dict(kwargs))
        raise AssertionError("stop requested before backend startup; civ6-mcp client must not start")

    def forbidden_run_one(**_kwargs: object) -> Civ6McpTurnResult:
        raise AssertionError("stop requested before backend startup; turn execution must not run")

    def forbidden_vlm_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("civ6-mcp stop handling must not invoke the VLM/computer-use backend")

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", forbidden_run_one)
    monkeypatch.setitem(
        sys.modules,
        "civStation.agent.turn_executor",
        SimpleNamespace(run_one_turn=forbidden_vlm_call, run_multi_turn=forbidden_vlm_call),
    )
    monkeypatch.setitem(
        sys.modules,
        "civStation.utils.screen",
        SimpleNamespace(execute_action=forbidden_vlm_call),
    )

    results = run_civ6_mcp_turn_loop(
        num_turns=1,
        install_path="/tmp/civ6-mcp",
        launcher="python",
        planner_provider=object(),
        context_manager=object(),  # type: ignore[arg-type]
        command_queue=command_queue,
        agent_gate=agent_gate,
        client_factory=forbidden_client_factory,
    )

    assert results == []
    assert client_factory_calls == []


def test_run_civ6_mcp_turn_loop_does_not_restart_already_started_client(monkeypatch) -> None:
    client = FakeLifecycleClient()
    client.started = True
    client.events.append("factory_start")
    run_one_calls: list[int] = []

    def fake_run_one(**kwargs) -> Civ6McpTurnResult:  # noqa: ANN003
        run_one_calls.append(kwargs["turn_index"])
        return Civ6McpTurnResult(turn_index=kwargs["turn_index"], success=True)

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", fake_run_one)

    results = run_civ6_mcp_turn_loop(
        num_turns=1,
        install_path="/tmp/civ6-mcp",
        launcher="python",
        planner_provider=object(),
        context_manager=object(),  # type: ignore[arg-type]
        client_factory=lambda **_kwargs: client,
    )

    assert [result.turn_index for result in results] == [0]
    assert run_one_calls == [0]
    assert client.start_calls == 0
    assert client.events == ["factory_start", "startup_health", "health_check", "stop"]


def test_run_civ6_mcp_turn_loop_stops_client_when_health_fails() -> None:
    client = FakeLifecycleClient(healthy=False, message="missing end_turn")

    with pytest.raises(Civ6McpError, match="missing end_turn"):
        run_civ6_mcp_turn_loop(
            num_turns=1,
            install_path="/tmp/civ6-mcp",
            launcher="python",
            planner_provider=object(),
            context_manager=object(),  # type: ignore[arg-type]
            client_factory=lambda **_kwargs: client,
        )

    assert client.start_calls == 1
    assert client.health_checks == 1
    assert client.stopped is True


def test_run_civ6_mcp_turn_loop_validates_startup_health_before_probe() -> None:
    startup_health = type("Health", (), {"ok": False, "message": "Missing required civ6-mcp tools: ['end_turn']"})()
    client = FakeLifecycleClient(startup_health=startup_health)

    with pytest.raises(Civ6McpError, match="Missing required civ6-mcp tools"):
        run_civ6_mcp_turn_loop(
            num_turns=1,
            install_path="/tmp/civ6-mcp",
            launcher="python",
            planner_provider=object(),
            context_manager=object(),  # type: ignore[arg-type]
            client_factory=lambda **_kwargs: client,
        )

    assert client.start_calls == 1
    assert client.health_checks == 0
    assert client.stopped is True
    assert client.events == ["start", "startup_health", "stop"]


def test_run_civ6_mcp_turn_loop_stops_client_when_start_fails() -> None:
    client = FakeLifecycleClient(start_error=Civ6McpError("stdio handshake failed"))

    with pytest.raises(Civ6McpError, match="stdio handshake failed"):
        run_civ6_mcp_turn_loop(
            num_turns=1,
            install_path="/tmp/civ6-mcp",
            launcher="python",
            planner_provider=object(),
            context_manager=object(),  # type: ignore[arg-type]
            client_factory=lambda **_kwargs: client,
        )

    assert client.start_calls == 1
    assert client.health_checks == 0
    assert client.stopped is True
    assert client.events == ["start", "stop"]


def test_run_civ6_mcp_turn_loop_stops_client_when_turn_execution_fails(monkeypatch) -> None:
    client = FakeLifecycleClient()

    def fail_run_one(**_kwargs) -> Civ6McpTurnResult:  # noqa: ANN003
        raise RuntimeError("planner crashed")

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", fail_run_one)

    with pytest.raises(RuntimeError, match="planner crashed"):
        run_civ6_mcp_turn_loop(
            num_turns=1,
            install_path="/tmp/civ6-mcp",
            launcher="python",
            planner_provider=object(),
            context_manager=object(),  # type: ignore[arg-type]
            client_factory=lambda **_kwargs: client,
        )

    assert client.start_calls == 1
    assert client.health_checks == 1
    assert client.stopped is True
    assert client.events == ["start", "startup_health", "health_check", "stop"]


def test_run_civ6_mcp_turn_loop_stops_client_when_interrupted(monkeypatch) -> None:
    client = FakeLifecycleClient()

    def interrupt_run_one(**_kwargs) -> Civ6McpTurnResult:  # noqa: ANN003
        raise KeyboardInterrupt

    monkeypatch.setattr(turn_loop_module, "run_one_turn_civ6_mcp", interrupt_run_one)

    with pytest.raises(KeyboardInterrupt):
        run_civ6_mcp_turn_loop(
            num_turns=1,
            install_path="/tmp/civ6-mcp",
            launcher="python",
            planner_provider=object(),
            context_manager=object(),  # type: ignore[arg-type]
            client_factory=lambda **_kwargs: client,
        )

    assert client.start_calls == 1
    assert client.health_checks == 1
    assert client.stopped is True
    assert client.events == ["start", "startup_health", "health_check", "stop"]
