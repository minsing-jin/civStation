"""
Core action model parsing and serialization tests.

These tests used to live under the legacy evaluation tree, but they validate
runtime schema behavior and should stay with the main test suite.
"""

import json

import pytest

from civStation.agent.models.schema import (
    AgentPlan,
    ClickAction,
    DragAction,
    KeyPressAction,
    WaitAction,
)


class TestActionParsing:
    def test_parse_click_action_from_dict(self):
        click_dict = {"type": "click", "x": 100, "y": 200, "button": "left"}
        action = ClickAction(**click_dict)

        assert action.x == 100
        assert action.y == 200
        assert action.button == "left"
        assert action.type == "click"

    def test_parse_click_action_minimal(self):
        click_dict = {"type": "click", "x": 100, "y": 200}
        action = ClickAction(**click_dict)

        assert action.x == 100
        assert action.y == 200
        assert action.button == "left"

    def test_parse_keypress_action_from_dict(self):
        press_dict = {"type": "press", "keys": ["esc"]}
        action = KeyPressAction(**press_dict)

        assert action.keys == ["esc"]
        assert action.interval == 0.1
        assert action.type == "press"

    def test_parse_keypress_action_multiple_keys(self):
        press_dict = {"type": "press", "keys": ["ctrl", "c"], "interval": 0.05}
        action = KeyPressAction(**press_dict)

        assert action.keys == ["ctrl", "c"]
        assert action.interval == 0.05

    def test_parse_drag_action_from_dict(self):
        drag_dict = {
            "type": "drag",
            "start_x": 10,
            "start_y": 20,
            "end_x": 100,
            "end_y": 200,
        }
        action = DragAction(**drag_dict)

        assert action.start_x == 10
        assert action.start_y == 20
        assert action.end_x == 100
        assert action.end_y == 200
        assert action.duration == 0.5
        assert action.button == "left"

    def test_parse_drag_action_with_duration(self):
        drag_dict = {
            "type": "drag",
            "start_x": 10,
            "start_y": 20,
            "end_x": 100,
            "end_y": 200,
            "duration": 1.0,
            "button": "right",
        }
        action = DragAction(**drag_dict)

        assert action.duration == 1.0
        assert action.button == "right"

    def test_parse_wait_action(self):
        wait_dict = {"type": "wait", "duration": 2.0}
        action = WaitAction(**wait_dict)

        assert action.duration == 2.0
        assert action.type == "wait"


class TestAgentPlanParsing:
    def test_parse_plan_with_mixed_actions(self):
        plan_dict = {
            "primitive_name": "test_primitive",
            "reasoning": "Testing discriminated union",
            "actions": [
                {"type": "click", "x": 100, "y": 200},
                {"type": "press", "keys": ["m"]},
                {"type": "drag", "start_x": 0, "start_y": 0, "end_x": 50, "end_y": 50},
            ],
        }

        plan = AgentPlan(**plan_dict)

        assert plan.primitive_name == "test_primitive"
        assert plan.reasoning == "Testing discriminated union"
        assert len(plan.actions) == 3
        assert isinstance(plan.actions[0], ClickAction)
        assert isinstance(plan.actions[1], KeyPressAction)
        assert isinstance(plan.actions[2], DragAction)

    def test_parse_plan_all_action_types(self):
        plan_dict = {
            "primitive_name": "comprehensive_test",
            "reasoning": "Testing all action types",
            "actions": [
                {"type": "click", "x": 100, "y": 200},
                {"type": "press", "keys": ["esc"]},
                {"type": "drag", "start_x": 10, "start_y": 10, "end_x": 100, "end_y": 100},
                {"type": "wait", "duration": 1.5},
                {"type": "double_click", "x": 500, "y": 600},
            ],
        }

        plan = AgentPlan(**plan_dict)

        assert len(plan.actions) == 5
        assert plan.actions[0].type == "click"
        assert plan.actions[1].type == "press"
        assert plan.actions[2].type == "drag"
        assert plan.actions[3].type == "wait"
        assert plan.actions[4].type == "double_click"

    def test_parse_plan_empty_actions(self):
        plan_dict = {
            "primitive_name": "empty_plan",
            "reasoning": "No actions needed",
            "actions": [],
        }

        plan = AgentPlan(**plan_dict)

        assert len(plan.actions) == 0
        assert plan.actions == []


class TestJSONSerialization:
    def test_json_roundtrip_click_action(self):
        original = ClickAction(type="click", x=960, y=540, description="Select unit")

        json_str = original.model_dump_json()
        json_dict = json.loads(json_str)
        restored = ClickAction(**json_dict)

        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.description == original.description

    def test_json_roundtrip_agent_plan(self):
        original_plan = AgentPlan(
            primitive_name="unit_ops",
            reasoning="Move unit to target location",
            actions=[
                ClickAction(type="click", x=960, y=540, description="Select unit"),
                KeyPressAction(type="press", keys=["m"], description="Move command"),
                ClickAction(type="click", x=1000, y=500, description="Target tile"),
            ],
        )

        json_str = original_plan.model_dump_json(indent=2)
        json_dict = json.loads(json_str)
        restored_plan = AgentPlan(**json_dict)

        assert restored_plan.primitive_name == original_plan.primitive_name
        assert restored_plan.reasoning == original_plan.reasoning
        assert len(restored_plan.actions) == 3
        assert isinstance(restored_plan.actions[0], ClickAction)
        assert isinstance(restored_plan.actions[1], KeyPressAction)
        assert isinstance(restored_plan.actions[2], ClickAction)
        assert restored_plan.actions[0].x == 960
        assert restored_plan.actions[1].keys == ["m"]
        assert restored_plan.actions[2].x == 1000

    def test_json_with_all_action_types(self):
        plan = AgentPlan(
            primitive_name="full_test",
            reasoning="Testing all actions",
            actions=[
                ClickAction(type="click", x=100, y=200),
                KeyPressAction(type="press", keys=["ctrl", "s"]),
                DragAction(type="drag", start_x=0, start_y=0, end_x=100, end_y=100),
                WaitAction(type="wait", duration=2.0),
            ],
        )

        json_str = plan.model_dump_json()
        restored = AgentPlan(**json.loads(json_str))

        assert len(restored.actions) == 4
        assert all(type(orig) is type(rest) for orig, rest in zip(plan.actions, restored.actions, strict=False))


class TestDiscriminatorValidation:
    def test_discriminator_field_present(self):
        actions = [
            ClickAction(type="click", x=100, y=200),
            KeyPressAction(type="press", keys=["a"]),
            DragAction(type="drag", start_x=0, start_y=0, end_x=10, end_y=10),
            WaitAction(type="wait", duration=1.0),
        ]

        for action in actions:
            assert hasattr(action, "type")
            assert action.type in ["click", "press", "drag", "wait", "double_click"]

    def test_action_serialization_includes_type(self):
        action = ClickAction(type="click", x=100, y=200)
        serialized = action.model_dump()

        assert "type" in serialized
        assert serialized["type"] == "click"

    @pytest.mark.parametrize(
        "action_type,expected_class",
        [
            ("click", ClickAction),
            ("press", KeyPressAction),
            ("drag", DragAction),
            ("wait", WaitAction),
        ],
    )
    def test_discriminator_routes_to_correct_type(self, action_type, expected_class):
        action_dicts = {
            "click": {"type": "click", "x": 0, "y": 0},
            "press": {"type": "press", "keys": ["a"]},
            "drag": {"type": "drag", "start_x": 0, "start_y": 0, "end_x": 10, "end_y": 10},
            "wait": {"type": "wait", "duration": 1.0},
        }

        plan_dict = {
            "primitive_name": "test",
            "reasoning": "test",
            "actions": [action_dicts[action_type]],
        }

        plan = AgentPlan(**plan_dict)
        assert isinstance(plan.actions[0], expected_class)
