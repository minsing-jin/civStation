"""Turn-loop integration hooks for the civ6-mcp observer."""

from __future__ import annotations

import logging
import sys
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
)


class FakeObserver:
    def __init__(self, bundle: StateBundle) -> None:
        self._bundle = bundle
        self.observe_calls = 0

    def observe(self) -> StateBundle:
        self.observe_calls += 1
        return self._bundle


class FakeLifecycleClient:
    def __init__(self, *, healthy: bool = True, message: str = "healthy") -> None:
        self.healthy = healthy
        self.message = message
        self.health_checks = 0
        self.stopped = False

    def health_check(self):
        self.health_checks += 1
        return type("Health", (), {"ok": self.healthy, "message": self.message})()

    def stop(self) -> None:
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
    assert "synthesizing fallback" not in caplog.text
    assert ctx.advanced_turns == []


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
    assert client.stopped is True
    assert run_one_calls[0]["civ6_mcp_client"] is client
    assert run_one_calls[0]["planner_provider"] is provider
    assert run_one_calls[0]["context_manager"] is ctx
    assert run_one_calls[0]["turn_index"] == 0
    assert run_one_calls[0]["high_level_strategy"] == "science"


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

    assert client.health_checks == 1
    assert client.stopped is True
