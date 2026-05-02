"""Tests for deterministic civ6-mcp turn-planning hints."""

from __future__ import annotations

from civStation.agent.modules.backend.civ6_mcp.observation_schema import Civ6McpNormalizedObservation
from civStation.agent.modules.backend.civ6_mcp.state_parser import StateBundle, parse_game_overview
from civStation.agent.modules.backend.civ6_mcp.turn_planning import (
    Civ6McpPrioritizedIntent,
    Civ6McpTurnPlan,
    build_prioritized_turn_plan,
)


def test_turn_plan_prioritizes_blockers_before_normal_turn_work() -> None:
    bundle = StateBundle(
        overview=parse_game_overview(
            {
                "turn": 88,
                "era": "Medieval Era",
                "current_research": "EDUCATION",
                "current_civic": "FEUDALISM",
            }
        ),
        units_text="Units:\n- Archer needs orders\n- Builder can move",
        cities_text="Cities:\n- Seoul: Choose production",
        notifications_text="Notifications:\n- Choose production\n- Unit needs orders",
        pending_diplomacy_text="Pending diplomacy:\n- Caesar demands tribute",
        pending_trades_text="Pending trades:\n- Cleopatra offers open borders",
    )

    plan = build_prioritized_turn_plan(bundle, strategy="Pursue science victory.")

    assert isinstance(plan, Civ6McpTurnPlan)
    assert [item.tool for item in plan.intents] == [
        "get_pending_diplomacy",
        "get_pending_trades",
        "get_city_production",
        "get_units",
        "get_district_advisor",
        "get_victory_progress",
        "end_turn",
    ]
    assert plan.intents == tuple(sorted(plan.intents, key=lambda item: (item.priority, item.tool)))
    assert "PENDING_DIPLOMACY" in plan.render_for_prompt()
    assert plan.to_tool_calls()[-1].tool == "end_turn"
    assert plan.to_tool_calls()[-1].arguments["tooling"]


def test_turn_plan_does_not_end_turn_after_game_over() -> None:
    bundle = StateBundle(overview=parse_game_overview("*** GAME OVER - VICTORY ***"))

    plan = build_prioritized_turn_plan(bundle)

    assert plan.intents == ()
    assert plan.to_actions() == []
    assert "game over" in " ".join(plan.notes).lower()


def test_turn_plan_accepts_normalized_observation_and_retries_failed_observations() -> None:
    observation = Civ6McpNormalizedObservation(
        raw_sections={
            "OVERVIEW": "Turn 9\nResearch:\nCivic:",
            "STATE_DIAGNOSTICS": "missing: get_victory_progress\nfailed: get_diplomacy (timeout)",
            "NOTIFICATIONS": "Choose research and civic.",
        },
        global_context_updates={"current_turn": 9},
    )

    plan = build_prioritized_turn_plan(observation)

    assert [item.tool for item in plan.intents[:4]] == [
        "get_victory_progress",
        "get_diplomacy",
        "get_tech_civics",
        "end_turn",
    ]
    assert len({item.tool for item in plan.intents}) == len(plan.intents)
    assert all(isinstance(item, Civ6McpPrioritizedIntent) for item in plan.intents)


def test_turn_plan_can_disable_synthetic_end_turn() -> None:
    bundle = StateBundle(notifications_text="Notifications:\n- Choose production")

    plan = build_prioritized_turn_plan(bundle, include_end_turn=False)

    assert [item.tool for item in plan.intents] == ["get_game_overview", "get_city_production"]
    assert [action.tool for action in plan.to_actions()] == ["get_game_overview", "get_city_production"]
