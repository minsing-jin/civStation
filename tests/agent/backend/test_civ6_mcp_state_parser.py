"""Tests for civ6-mcp text-response heuristic parsing."""

from __future__ import annotations

from civStation.agent.modules.backend.civ6_mcp.state_parser import (
    StateBundle,
    parse_game_overview,
)


def test_parse_overview_extracts_turn_and_era() -> None:
    text = """\
Turn: 42
Era: Classical Era
Civilization: Korea (Seondeok)
Science: +35.5/turn
Culture: +12.0/turn
Gold: -2.0/turn
Faith: +0.0/turn
Researching: WRITING
Civic Researching: CRAFTSMANSHIP
"""
    snap = parse_game_overview(text)
    assert snap.current_turn == 42
    assert snap.game_era == "Classical"
    assert snap.science_per_turn == 35.5
    assert snap.culture_per_turn == 12.0
    assert snap.gold_per_turn == -2.0
    assert snap.faith_per_turn == 0.0
    assert snap.current_research == "WRITING"
    assert snap.current_civic == "CRAFTSMANSHIP"
    assert snap.is_game_over is False


def test_parse_overview_handles_game_over() -> None:
    text = "*** GAME OVER — VICTORY! Science victory achieved at turn 240 ***"
    snap = parse_game_overview(text)
    assert snap.is_game_over is True
    assert snap.victory_text and "VICTORY" in snap.victory_text


def test_parse_overview_resilient_to_empty_input() -> None:
    snap = parse_game_overview("")
    assert snap.current_turn is None
    assert snap.game_era is None
    assert snap.is_game_over is False


def test_state_bundle_renders_planner_context_with_truncation() -> None:
    bundle = StateBundle()
    bundle.units_text = "U" * 4000
    bundle.cities_text = "Compact city listing."
    rendered = bundle.to_planner_context(max_section_chars=200)
    assert "## CITIES" in rendered
    assert "## UNITS" in rendered
    assert "(truncated)" in rendered
    assert len(rendered) < 2500
