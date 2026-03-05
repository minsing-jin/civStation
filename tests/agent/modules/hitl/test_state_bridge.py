"""Unit tests for AgentStateBridge — snapshot consistency."""

from computer_use_test.agent.modules.context.context_manager import ContextManager
from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from computer_use_test.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge


class TestAgentStateBridge:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()
        self.queue = CommandQueue()
        self.bridge = AgentStateBridge(self.ctx, self.queue)

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_initial_status(self):
        status = self.bridge.get_status()
        assert status.game_turn == 1
        assert status.micro_turn == 0
        assert status.macro_turn == 1
        assert status.current_primitive == ""
        assert status.queued_directives == []
        assert status.recent_actions == []

    def test_update_current_action(self):
        self.bridge.update_current_action("unit_ops_primitive", "click (500, 300)", "전사 이동")
        status = self.bridge.get_status()
        assert status.current_primitive == "unit_ops_primitive"
        assert status.current_action == "click (500, 300)"
        assert status.current_reasoning == "전사 이동"

    def test_update_turns(self):
        self.bridge.update_micro_turn(5)
        self.bridge.update_macro_turn(3)
        status = self.bridge.get_status()
        assert status.micro_turn == 5
        assert status.macro_turn == 3

    def test_queued_directives_reflected(self):
        self.queue.push(Directive(directive_type=DirectiveType.CHANGE_STRATEGY, payload="문화 승리", source="test"))
        status = self.bridge.get_status()
        assert len(status.queued_directives) == 1
        assert status.queued_directives[0]["type"] == "change_strategy"
        assert status.queued_directives[0]["payload"] == "문화 승리"

    def test_recent_actions_from_context(self):
        self.ctx.record_action(action_type="click", primitive="unit_ops_primitive", x=100, y=200)
        status = self.bridge.get_status()
        assert len(status.recent_actions) == 1
        assert status.recent_actions[0]["type"] == "click"
        assert status.recent_actions[0]["x"] == 100

    def test_to_dict_completeness(self):
        self.bridge.update_current_action("test", "click (0,0)", "test reason")
        status = self.bridge.get_status()
        d = status.to_dict()
        expected_keys = {
            "agent_state",
            "current_strategy",
            "victory_goal",
            "game_phase",
            "queued_directives",
            "current_primitive",
            "current_action",
            "current_reasoning",
            "game_turn",
            "micro_turn",
            "macro_turn",
            "recent_actions",
            "last_updated",
        }
        assert set(d.keys()) == expected_keys
