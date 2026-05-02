"""Shared civ6-mcp backend test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class FakeMcpTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


@pytest.fixture
def mixed_civ6_mcp_sdk_state_payload() -> dict[str, object]:
    """Representative MCP SDK-ish state payload with mixed result shapes."""
    return {
        "game_overview": {
            "content": [],
            "structured_content": {
                "game_turn": "112",
                "era": "Information Era",
                "sciencePerTurn": "455.5",
                "culturePerTurn": 231.25,
                "goldPerTurn": "+1000",
                "faithPerTurn": "22",
                "researching": "SMART_MATERIALS",
                "civicResearching": "GLOBALIZATION",
            },
        },
        "units": {
            "content": [
                FakeMcpTextBlock("Units:"),
                FakeMcpTextBlock("- Mech Infantry at (10, 11)"),
            ]
        },
        "get_cities": {"content": [{"type": "text", "text": "Cities:\n- Seoul: pop 18"}]},
        "get_notifications": {
            "content": [
                {"type": "image", "data": "<ignored>"},
                FakeMcpTextBlock("Notifications:\n- Choose production"),
            ]
        },
        "get_trade_routes": {"content": [FakeMcpTextBlock("Trade Routes:\n- Seoul -> Busan")]},
        "missing_tools": "get_victory_progress",
        "failed_tools": {"get_diplomacy": "timeout"},
        "malformed_tools": {"get_pending_trades": "empty response body"},
    }


@pytest.fixture(scope="session")
def civ6_mcp_planner_observation_cases() -> dict[str, dict[str, object]]:
    """Representative planner observation cases shared by civ6-mcp tests."""
    fixture_path = Path(__file__).with_name("fixtures") / "civ6_mcp_planner_observations.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))
