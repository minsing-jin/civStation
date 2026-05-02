"""Focused ContextManager synchronization tests for the civ6-mcp observer."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.response import Civ6McpNormalizedResult, normalize_mcp_response_text
from civStation.agent.modules.context.context_manager import ContextManager


class FakeObservationClient:
    def __init__(self, responses: dict[str, Civ6McpNormalizedResult | str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has_tool(self, name: str) -> bool:
        return name in self._responses

    def call_tool_result(self, name: str, arguments: dict[str, Any] | None = None) -> Civ6McpNormalizedResult:
        request_arguments = dict(arguments or {})
        self.calls.append((name, request_arguments))
        response = self._responses[name]
        if isinstance(response, Civ6McpNormalizedResult):
            return response
        return normalize_mcp_response_text(name, request_arguments, response)


class RecordingContextManager:
    def __init__(self, *, fail_global_update: bool = False) -> None:
        self.fail_global_update = fail_global_update
        self.global_updates: list[dict[str, Any]] = []
        self.game_observation_updates: list[dict[str, Any]] = []

    def update_global_context(self, **kwargs: Any) -> None:
        self.global_updates.append(dict(kwargs))
        if self.fail_global_update:
            raise RuntimeError("global context write failed")

    def update_game_observation(self, **kwargs: Any) -> None:
        self.game_observation_updates.append(dict(kwargs))


@pytest.fixture(autouse=True)
def _reset_context_manager_singleton():
    ContextManager.reset_instance()
    yield
    ContextManager.reset_instance()


def test_observer_synchronizes_parsed_overview_fields_into_context_manager() -> None:
    overview = """\
Turn: 42
Era: Medieval Era
Science: +33.25/turn
Culture: +11.0/turn
Gold: -2.5/turn
Faith: +7.0/turn
Researching: EDUCATION
Civic Researching: FEUDALISM
"""
    client = FakeObservationClient({"get_game_overview": overview})
    ctx = ContextManager.get_instance()
    observer = Civ6McpObserver(client, ctx, observe_tools=("get_game_overview",))  # type: ignore[arg-type]

    bundle = observer.observe()

    assert client.calls == [("get_game_overview", {})]
    assert bundle.overview.current_turn == 42
    assert ctx.global_context.current_turn == 42
    assert ctx.global_context.game_era == "Medieval"
    assert ctx.global_context.science_per_turn == pytest.approx(33.25)
    assert ctx.global_context.culture_per_turn == pytest.approx(11.0)
    assert ctx.global_context.gold_per_turn == pytest.approx(-2.5)
    assert ctx.global_context.faith_per_turn == pytest.approx(7.0)
    assert ctx.global_context.current_research == "EDUCATION"
    assert ctx.global_context.current_civic == "FEUDALISM"
    assert ctx.high_level_context.notes == [
        "Turn 42 | Era Medieval | Sci +33.2/t | Cul +11.0/t | Research EDUCATION | Civic FEUDALISM"
    ]
    assert observer.last_observation is not None
    assert observer.last_observation.global_context_updates == {
        "current_turn": 42,
        "game_era": "Medieval",
        "science_per_turn": 33.25,
        "culture_per_turn": 11.0,
        "gold_per_turn": -2.5,
        "faith_per_turn": 7.0,
        "current_research": "EDUCATION",
        "current_civic": "FEUDALISM",
    }


def test_observer_preserves_existing_context_when_snapshot_has_no_context_updates() -> None:
    client = FakeObservationClient({"get_game_overview": "No active turn data available yet."})
    ctx = ContextManager.get_instance()
    ctx.update_global_context(current_turn=88, game_era="Modern", science_per_turn=120.0, current_research="RADIO")
    ctx.update_game_observation("Existing strategic note")
    observer = Civ6McpObserver(client, ctx, observe_tools=("get_game_overview",))  # type: ignore[arg-type]

    bundle = observer.observe()

    assert bundle.overview.current_turn is None
    assert bundle.malformed_tools == {"get_game_overview": "unrecognized overview payload"}
    assert ctx.global_context.current_turn == 88
    assert ctx.global_context.game_era == "Modern"
    assert ctx.global_context.science_per_turn == pytest.approx(120.0)
    assert ctx.global_context.current_research == "RADIO"
    assert ctx.high_level_context.notes == ["Existing strategic note"]
    assert observer.last_observation is not None
    assert observer.last_observation.global_context_updates == {}
    assert observer.last_observation.game_observation_updates == {}


def test_observer_still_syncs_game_observation_when_global_context_sync_fails(caplog: pytest.LogCaptureFixture) -> None:
    overview = """\
Turn: 7
Era: Ancient Era
Science: +4.0/turn
Culture: +2.0/turn
Researching: POTTERY
"""
    client = FakeObservationClient({"get_game_overview": overview})
    ctx = RecordingContextManager(fail_global_update=True)
    observer = Civ6McpObserver(client, ctx, observe_tools=("get_game_overview",))  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING, logger="civStation.agent.modules.backend.civ6_mcp.observer"):
        observer.observe()

    assert ctx.global_updates == [
        {
            "current_turn": 7,
            "game_era": "Ancient",
            "science_per_turn": 4.0,
            "culture_per_turn": 2.0,
            "current_research": "POTTERY",
        }
    ]
    assert ctx.game_observation_updates == [
        {"situation_summary": "Turn 7 | Era Ancient | Sci +4.0/t | Cul +2.0/t | Research POTTERY"}
    ]
    assert "ContextManager update_global_context failed: global context write failed" in caplog.text
