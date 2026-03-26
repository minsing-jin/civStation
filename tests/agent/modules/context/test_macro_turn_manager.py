"""Unit tests for MacroTurnManager — micro-turn recording and next-turn detection."""

from __future__ import annotations

from dataclasses import dataclass

from civStation.agent.modules.context.context_manager import ContextManager
from civStation.agent.modules.context.macro_turn_manager import MacroTurnManager


@dataclass
class FakeAction:
    """Minimal stand-in for AgentAction."""

    action: str = "click"
    x: int = 0
    y: int = 0
    end_x: int = 0
    end_y: int = 0
    key: str = ""
    text: str = ""
    reasoning: str = ""


class TestIsNextTurnAction:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()
        self.mgr = MacroTurnManager(self.ctx, vlm_provider=None)

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_positive_popup_click_next_turn_korean(self):
        action = FakeAction(action="click", reasoning="다음 턴 버튼을 클릭합니다.")
        assert self.mgr.is_next_turn_action("popup_primitive", action) is True

    def test_positive_popup_click_end_turn_english(self):
        action = FakeAction(action="click", reasoning="Click the next turn button.")
        assert self.mgr.is_next_turn_action("popup_primitive", action) is True

    def test_positive_turn_end_keyword(self):
        action = FakeAction(action="click", reasoning="턴 종료를 위해 클릭합니다.")
        assert self.mgr.is_next_turn_action("popup_primitive", action) is True

    def test_negative_wrong_primitive(self):
        action = FakeAction(action="click", reasoning="다음 턴 버튼을 클릭합니다.")
        assert self.mgr.is_next_turn_action("unit_ops_primitive", action) is False

    def test_negative_wrong_action_type(self):
        action = FakeAction(action="press", key="enter", reasoning="다음 턴")
        assert self.mgr.is_next_turn_action("popup_primitive", action) is False

    def test_negative_no_keyword(self):
        action = FakeAction(action="click", reasoning="팝업 확인 버튼 클릭")
        assert self.mgr.is_next_turn_action("popup_primitive", action) is False


class TestMacroTurnTracking:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()
        self.mgr = MacroTurnManager(self.ctx, vlm_provider=None)

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_record_micro_turn(self):
        self.mgr.record_micro_turn("unit_ops_primitive", "전사 이동")
        self.mgr.record_micro_turn("popup_primitive", "팝업 확인")
        assert self.mgr.micro_turn_count == 2

    def test_handle_macro_turn_end(self):
        self.mgr.record_micro_turn("unit_ops_primitive", "전사 이동")
        self.mgr.record_micro_turn("research_select_primitive", "광업 연구")
        self.mgr.record_micro_turn("popup_primitive", "다음 턴 클릭")

        summary = self.mgr.handle_macro_turn_end()
        assert summary.macro_turn_number == 1
        assert summary.micro_turn_count == 3
        assert "unit_ops_primitive" in summary.primitives_used
        assert len(summary.llm_summary) > 0  # fallback summary

        # Verify reset
        assert self.mgr.macro_turn_number == 2
        assert self.mgr.micro_turn_count == 0

        # Verify context manager got the summary
        summaries = self.ctx.get_macro_turn_summaries()
        assert len(summaries) == 1

    def test_fallback_summary_format(self):
        self.mgr.record_micro_turn("combat_primitive", "공격")
        summary = self.mgr.handle_macro_turn_end()
        assert "턴 1" in summary.llm_summary
        assert "1개 행동" in summary.llm_summary
