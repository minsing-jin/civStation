"""Tests for Civ6McpObserver — verify state ingestion into ContextManager."""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

import pytest

import civStation.agent.modules.backend.civ6_mcp.observer as observer_module
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.operations import Civ6McpDispatchResult, Civ6McpRequest
from civStation.agent.modules.backend.civ6_mcp.planner import DEFAULT_PLANNER_TOOL_ALLOWLIST
from civStation.agent.modules.backend.civ6_mcp.response import (
    Civ6McpNormalizedResult,
    Civ6McpResponseClassification,
    normalize_mcp_response_text,
)
from civStation.agent.modules.context.context_manager import ContextManager


class FakeClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def call_tool(self, name: str, arguments=None) -> str:  # noqa: ARG002
        return self._responses[name]


class FakeTypedClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    @property
    def tool_names(self) -> set[str]:
        return set(self._responses)

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def call_tool_result(self, name: str, arguments=None) -> Civ6McpNormalizedResult | str | object:  # noqa: ARG002
        self.calls.append(name)
        response = self._responses[name]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, Civ6McpNormalizedResult):
            return response
        if not isinstance(response, str):
            return response
        return normalize_mcp_response_text(name, arguments, response)


class CatalogedMissingResponseClient:
    def __init__(self, tool_names: set[str], responses: dict[str, Civ6McpNormalizedResult | str | None]) -> None:
        self._tool_names = tool_names
        self._responses = responses
        self.calls: list[str] = []

    @property
    def tool_names(self) -> set[str]:
        return self._tool_names

    def has_tool(self, name: str) -> bool:
        return name in self._tool_names

    def call_tool_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Civ6McpNormalizedResult | str | None:
        self.calls.append(name)
        return self._responses[name]


@pytest.fixture(autouse=True)
def _reset_context_manager_singleton():
    ContextManager.reset_instance()
    yield
    ContextManager.reset_instance()


def test_observer_pushes_overview_into_global_context() -> None:
    overview = """\
Turn: 17
Era: Classical Era
Science: +12.5/turn
Culture: +5.0/turn
Gold: +1.0/turn
Faith: +0.0/turn
Researching: BRONZE_WORKING
Civic Researching: STATE_WORKFORCE
"""
    fake = FakeClient(
        responses={
            "get_game_overview": overview,
            "get_units": "Settler at (1,2)",
            "get_cities": "Seoul (Capital)",
            "get_diplomacy": "(none met yet)",
            "get_tech_civics": "BRONZE_WORKING researching",
            "get_notifications": "(none)",
            "get_pending_diplomacy": "",
            "get_pending_trades": "",
            "get_victory_progress": "Science 5%",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(fake, ctx)  # type: ignore[arg-type]
    bundle = observer.observe()

    assert bundle.overview.current_turn == 17
    assert bundle.overview.game_era == "Classical"
    assert ctx.global_context.current_turn == 17
    assert ctx.global_context.science_per_turn == pytest.approx(12.5)
    assert ctx.global_context.current_research == "BRONZE_WORKING"
    assert observer.last_observation is not None
    assert observer.last_observation.global_context_updates["current_turn"] == 17
    assert observer.last_observation.tool_results["get_game_overview"] == overview.strip()
    assert observer.last_observation.tool_results["get_units"] == "Settler at (1,2)"
    assert observer.last_observation.raw_sections["UNITS"] == "Settler at (1,2)"
    # situation_summary should land in HighLevelContext.notes
    assert ctx.high_level_context.notes
    assert "Turn 17" in ctx.high_level_context.notes[-1]
    assert ctx.high_level_context.latest_game_observation["current_turn"] == 17
    assert ctx.high_level_context.latest_game_observation["game_era"] == "Classical"
    assert ctx.high_level_context.latest_game_observation["current_research"] == "BRONZE_WORKING"


def test_default_observe_tools_are_derived_from_planner_allowlist() -> None:
    assert observer_module.DEFAULT_CIV6_MCP_OBSERVE_TOOLS == tuple(
        tool for tool in DEFAULT_PLANNER_TOOL_ALLOWLIST if tool.startswith("get_")
    )
    assert "DEFAULT_PLANNER_TOOL_ALLOWLIST" in inspect.getsource(observer_module)


def test_observer_skips_missing_tools_silently() -> None:
    fake = FakeClient(responses={"get_game_overview": "Turn: 1\nEra: Ancient Era\n"})
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(fake, ctx)  # type: ignore[arg-type]
    bundle = observer.observe()
    assert bundle.overview.current_turn == 1
    assert bundle.units_text == ""
    assert bundle.cities_text == ""


def test_observer_handles_call_errors() -> None:
    class ErroringClient:
        def has_tool(self, name: str) -> bool:
            return True

        def call_tool(self, name: str, arguments=None):  # noqa: ARG002
            from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpError

            raise Civ6McpError("simulated failure")

    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(ErroringClient(), ctx)  # type: ignore[arg-type]
    bundle = observer.observe()
    # Should not raise; just produce an empty bundle.
    assert bundle.overview.current_turn is None
    assert bundle.units_text == ""


def test_observer_reads_structured_game_state_when_text_is_missing() -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": Civ6McpNormalizedResult(
                tool="get_game_overview",
                success=True,
                structured_content={
                    "turn": 91,
                    "era": "Industrial Era",
                    "yields": {"science": 141.5},
                    "current_research": "STEAM_POWER",
                },
            )
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(fake, ctx)  # type: ignore[arg-type]

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview"]
    assert bundle.overview.current_turn == 91
    assert bundle.overview.game_era == "Industrial"
    assert bundle.overview.science_per_turn == pytest.approx(141.5)
    assert ctx.global_context.current_turn == 91
    assert bundle.malformed_tools == {}


def test_observer_parses_successful_get_responses_through_observation_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_tools: list[str] = []
    real_parser = observer_module.parse_observation_tool_response

    def recording_parser(tool: str, payload: object):
        parsed_tools.append(tool)
        return real_parser(tool, payload)

    monkeypatch.setattr(observer_module, "parse_observation_tool_response", recording_parser)
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 22\nEra: Renaissance Era\n",
            "get_units": "Units:\n- Archer at (2, 3)",
            "get_cities": Civ6McpNormalizedResult(
                tool="get_cities",
                success=False,
                error="transport failure",
                classification=Civ6McpResponseClassification.ERROR,
                is_error=True,
            ),
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_cities"),
    )

    bundle = observer.observe()

    assert parsed_tools == ["get_game_overview", "get_units"]
    assert bundle.overview.current_turn == 22
    assert bundle.units_text == "Units:\n- Archer at (2, 3)"
    assert bundle.failed_tools == {"get_cities": "transport failure"}
    assert observer.last_tool_observations is not None
    assert [item.tool for item in observer.last_tool_observations] == ["get_game_overview", "get_units"]
    assert observer.last_tool_observations[0].normalized.global_context_updates["current_turn"] == 22


def test_observer_records_missing_and_malformed_state_without_context_update() -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": normalize_mcp_response_text("get_game_overview", {}, "   "),
            "get_units": normalize_mcp_response_text("get_units", {}, "   "),
            "get_cities": "Cities:\n- Seoul",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_cities", "get_diplomacy"),
    )

    bundle = observer.observe()

    assert "get_diplomacy" in bundle.missing_tools
    assert bundle.malformed_tools["get_game_overview"] == "empty response body"
    assert bundle.malformed_tools["get_units"] == "empty response body"
    assert bundle.overview.current_turn is None
    assert ctx.global_context.current_turn == 1
    assert "## CITIES" in bundle.to_planner_context()
    assert "malformed: get_game_overview" in bundle.to_planner_context()


def test_observer_recovers_when_cataloged_get_response_is_missing() -> None:
    fake = CatalogedMissingResponseClient(
        tool_names={"get_game_overview", "get_units"},
        responses={"get_game_overview": "Turn: 31\nEra: Industrial Era\n"},
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units"),
    )

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview", "get_units"]
    assert bundle.overview.current_turn == 31
    assert bundle.units_text == ""
    assert "get_units" in bundle.failed_tools["get_units"]
    assert ctx.global_context.current_turn == 31
    assert observer.last_observation is not None
    assert observer.last_observation.global_context_updates["current_turn"] == 31
    assert "failed: get_units" in observer.last_observation.planner_context


def test_observer_records_error_payloads_and_tool_exceptions_without_aborting_turn() -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 44\nEra: Modern Era\n",
            "get_units": {
                "isError": True,
                "error": {"message": "Error: unit report unavailable"},
            },
            "get_cities": RuntimeError("transport disconnected"),
            "get_notifications": "Notifications:\n- Choose production in Seoul",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_cities", "get_notifications"),
    )

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview", "get_units", "get_cities", "get_notifications"]
    assert bundle.overview.current_turn == 44
    assert bundle.notifications_text == "Notifications:\n- Choose production in Seoul"
    assert bundle.failed_tools == {
        "get_units": "Error: unit report unavailable",
        "get_cities": "transport disconnected",
    }
    assert ctx.global_context.current_turn == 44
    assert observer.last_observation is not None
    assert "failed: get_cities" in observer.last_observation.planner_context
    assert "get_units (Error: unit report unavailable)" in observer.last_observation.planner_context
    assert observer.last_observation.tool_results["get_notifications"] == "Notifications:\n- Choose production in Seoul"


def test_observer_returns_empty_bundle_when_get_response_is_absent() -> None:
    fake = CatalogedMissingResponseClient(
        tool_names={"get_game_overview"},
        responses={"get_game_overview": None},
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview",),
    )

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview"]
    assert bundle.overview.current_turn is None
    assert bundle.malformed_tools == {"get_game_overview": "empty response body"}
    assert ctx.global_context.current_turn == 1
    assert observer.last_observation is not None
    assert observer.last_observation.global_context_updates == {}
    assert "malformed: get_game_overview" in observer.last_observation.planner_context


def test_observer_rejects_successful_get_response_for_different_tool_without_state_corruption() -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 58\nEra: Atomic Era\n",
            "get_units": Civ6McpNormalizedResult(
                tool="get_cities",
                success=True,
                text="Cities:\n- Seoul: pop 15",
            ),
            "get_notifications": "Notifications:\n- Choose research",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_notifications"),
    )

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview", "get_units", "get_notifications"]
    assert bundle.overview.current_turn == 58
    assert bundle.units_text == ""
    assert bundle.notifications_text == "Notifications:\n- Choose research"
    assert bundle.malformed_tools == {"get_units": "response tool mismatch: expected get_units, got get_cities"}
    assert ctx.global_context.current_turn == 58
    assert observer.last_observation is not None
    assert observer.last_observation.tool_results == {
        "get_game_overview": "Turn: 58\nEra: Atomic Era",
        "get_notifications": "Notifications:\n- Choose research",
    }
    assert "Cities:" not in observer.last_observation.planner_context
    assert "malformed: get_units" in observer.last_observation.planner_context


def test_observer_keeps_previous_valid_observation_when_next_get_payload_shape_is_unusable() -> None:
    fake = FakeTypedClient(responses={"get_game_overview": "Turn: 12\nEra: Ancient Era\n"})
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(fake, ctx, observe_tools=("get_game_overview",))  # type: ignore[arg-type]

    first_bundle = observer.observe()
    first_observation = observer.last_observation

    fake._responses["get_game_overview"] = {
        "content": [{"type": "image", "data": "not text"}],
        "structuredContent": None,
    }
    second_bundle = observer.observe()

    assert first_bundle.overview.current_turn == 12
    assert second_bundle.overview.current_turn is None
    assert second_bundle.malformed_tools == {"get_game_overview": "empty response body"}
    assert ctx.global_context.current_turn == 12
    assert observer.last_observation is first_observation
    assert observer.last_bundle is first_bundle


def test_observer_discovers_available_get_tools_only() -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 12\nEra: Medieval Era\n",
            "get_builder_tasks": "Builder Tasks:\n- Improve bananas near Seoul",
            "get_custom_report": "Custom Report:\n- Dynamic upstream observation",
            "set_research": "Should not be called by observer",
            "end_turn": "Should not be called by observer",
            "debug_dump_state": "Should not be called by observer",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(fake, ctx)  # type: ignore[arg-type]

    bundle = observer.observe()

    assert fake.calls == ["get_game_overview", "get_builder_tasks", "get_custom_report"]
    assert bundle.overview.current_turn == 12
    assert bundle.extra["get_builder_tasks"] == "Builder Tasks:\n- Improve bananas near Seoul"
    assert bundle.extra["get_custom_report"] == "Custom Report:\n- Dynamic upstream observation"
    assert observer.last_observation is not None
    assert observer.last_observation.tool_results == {
        "get_game_overview": "Turn: 12\nEra: Medieval Era",
        "get_builder_tasks": "Builder Tasks:\n- Improve bananas near Seoul",
        "get_custom_report": "Custom Report:\n- Dynamic upstream observation",
    }
    assert "set_research" not in fake.calls
    assert "end_turn" not in fake.calls


def test_observer_logs_recoverable_failures_and_completion_summary(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 66\nEra: Information Era\n",
            "get_units": Civ6McpNormalizedResult(
                tool="get_units",
                success=False,
                error="unit report unavailable",
                classification=Civ6McpResponseClassification.ERROR,
                is_error=True,
            ),
            "get_cities": "   ",
            "get_notifications": "Notifications:\n- Choose production",
        }
    )
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_cities", "get_notifications", "get_diplomacy"),
    )

    with caplog.at_level(logging.INFO, logger=observer_module.__name__):
        bundle = observer.observe()

    messages = [record.getMessage() for record in caplog.records]
    assert bundle.overview.current_turn == 66
    assert bundle.failed_tools == {"get_units": "unit report unavailable"}
    assert bundle.malformed_tools == {"get_cities": "empty response body"}
    assert bundle.missing_tools == ("get_diplomacy",)
    assert any(
        "recoverable civ6-mcp observation failure: tool=get_units reason=unit report unavailable" in message
        for message in messages
    )
    assert any(
        "civ6-mcp observation completed: attempted=4 successful=2 missing=1 failed=1 malformed=1 skipped=0" in message
        for message in messages
    )


def test_observer_deadline_records_timeout_and_returns_partial_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeTypedClient(
        responses={
            "get_game_overview": "Turn: 77\nEra: Atomic Era\n",
            "get_units": "Units:\n- Infantry at (4, 5)",
            "get_cities": "Cities:\n- Seoul",
        }
    )
    original_dispatch = observer_module.Civ6McpOperationDispatcher.dispatch

    def slow_units_dispatch(self: object, request: Civ6McpRequest) -> Civ6McpDispatchResult:
        if request.tool == "get_units":
            time.sleep(0.3)
        return original_dispatch(self, request)

    monkeypatch.setattr(observer_module.Civ6McpOperationDispatcher, "dispatch", slow_units_dispatch)
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(
        fake,  # type: ignore[arg-type]
        ctx,
        observe_tools=("get_game_overview", "get_units", "get_cities"),
        observe_timeout_seconds=0.05,
    )

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger=observer_module.__name__):
        bundle = observer.observe()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert bundle.overview.current_turn == 77
    assert "timed out after" in bundle.failed_tools["get_units"]
    assert bundle.failed_tools["get_cities"] == "skipped because observation deadline was exhausted"
    assert bundle.missing_tools == ()
    assert ctx.global_context.current_turn == 77
    assert observer.last_observation is not None
    assert "get_units (timed out after" in observer.last_observation.planner_context
    assert (
        "get_cities (skipped because observation deadline was exhausted)" in observer.last_observation.planner_context
    )
    assert any("deadline_exhausted=True" in record.getMessage() for record in caplog.records)
