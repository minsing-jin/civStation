"""Unit tests for action parsing and validation."""

from computer_use_test.utils.llm_provider.parser import (
    AgentAction,
    parse_action_json,
    parse_action_json_list,
    validate_action,
)


def test_parse_scroll_action():
    action = parse_action_json(
        """
        {
          "action": "scroll",
          "x": 512,
          "y": 444,
          "scroll_amount": -650,
          "reasoning": "리스트 중앙에 hover 후 아래로 스크롤"
        }
        """
    )

    assert action is not None
    assert action.action == "scroll"
    assert action.x == 512
    assert action.y == 444
    assert action.scroll_amount == -650
    assert validate_action(action) == []


def test_validate_scroll_requires_non_zero_amount():
    action = AgentAction(action="scroll", x=500, y=500, scroll_amount=0)
    errors = validate_action(action)
    assert any("scroll_amount" in error for error in errors)


def test_parse_policy_drag_metadata_from_multi_action_json():
    actions = parse_action_json_list(
        """
        [
          {
            "action": "drag",
            "x": 820,
            "y": 220,
            "end_x": 150,
            "end_y": 190,
            "reasoning": "정책 카드 드래그",
            "policy_card_name": "규율",
            "policy_source_tab": "군사",
            "policy_target_slot_id": "military_1",
            "policy_reasoning": "야만인 대응"
          }
        ]
        """
    )

    assert actions is not None
    assert len(actions) == 1
    assert actions[0].policy_card_name == "규율"
    assert actions[0].policy_source_tab == "군사"
    assert actions[0].policy_target_slot_id == "military_1"
    assert actions[0].policy_reasoning == "야만인 대응"


def test_parse_click_action_requires_explicit_coordinates():
    action = parse_action_json(
        """
        {
          "action": "click",
          "reasoning": "버튼을 클릭"
        }
        """
    )

    assert action is None


def test_parse_scroll_action_requires_explicit_coordinates():
    action = parse_action_json(
        """
        {
          "action": "scroll",
          "scroll_amount": -650,
          "reasoning": "리스트를 스크롤"
        }
        """
    )

    assert action is None


def test_parse_move_action():
    action = parse_action_json(
        """
        {
          "action": "move",
          "x": 640,
          "y": 420,
          "reasoning": "생산 목록 중앙 hover"
        }
        """
    )

    assert action is not None
    assert action.action == "move"
    assert action.x == 640
    assert action.y == 420
    assert validate_action(action) == []
