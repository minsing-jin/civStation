"""Tests for Civ6McpObserver — verify state ingestion into ContextManager."""

from __future__ import annotations

import pytest

from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.context.context_manager import ContextManager


class FakeClient:
    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def call_tool(self, name: str, arguments=None) -> str:  # noqa: ARG002
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
    # situation_summary should land in HighLevelContext.notes
    assert ctx.high_level_context.notes
    assert "Turn 17" in ctx.high_level_context.notes[-1]


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
