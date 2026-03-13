from PIL import Image

from computer_use_test.agent.modules.context.context_manager import ContextManager
from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
from computer_use_test.agent.modules.primitive.multi_step_process import (
    BaseMultiStepProcess,
    SemanticVerifyResult,
    StageTransition,
    VerificationResult,
)
from computer_use_test.agent.turn_executor import run_primitive_loop
from computer_use_test.utils.llm_provider.base import BaseVLMProvider, VLMResponse
from computer_use_test.utils.llm_provider.parser import AgentAction


class DummyProvider(BaseVLMProvider):
    def __init__(self):
        super().__init__(api_key=None, model="dummy", resize_for_vlm=False)

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=8192, use_thinking=True) -> VLMResponse:
        raise AssertionError("DummyProvider should not be called in these tests")

    def _build_image_content(self, image_path):
        return {"image_path": str(image_path)}

    def _build_pil_image_content(self, pil_image, jpeg_quality=None):
        return {"pil_size": getattr(pil_image, "size", None), "jpeg_quality": jpeg_quality}

    def _build_text_content(self, text: str):
        return {"text": text}

    def get_provider_name(self) -> str:
        return "dummy"


class TransitionProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("research_select_primitive", "")
        self.calls = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.begin_stage("stage_a")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            memory.begin_stage("stage_b")
            return StageTransition(stage="stage_b", reason="internal transition")
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="complete on second iteration",
            task_status="complete",
        )

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class PolicySemanticOnlyProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("policy_primitive", "")
        self.calls = 0
        self.success_called = False
        self.verify_called = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        return None

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AgentAction(
                action="click",
                coord_space="absolute",
                x=200,
                y=100,
                reasoning="confirmed cached tab click",
                task_status="in_progress",
            )
        return AgentAction(
            action="click",
            x=850,
            y=920,
            reasoning="finish after bypassed tab click",
            task_status="complete",
        )

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        return memory.should_verify_policy_tab_click()

    def verify_action_success(self, provider, pil_image, memory, action, **kwargs) -> SemanticVerifyResult:
        self.verify_called += 1
        return SemanticVerifyResult(handled=True, passed=True, reason="semantic tab ok")

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        self.success_called = True
        memory.mark_policy_tab_confirmed("군사")
        memory.begin_stage("finalize_policy")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class PolicySemanticGateProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("policy_primitive", "")
        self.calls = 0
        self.verify_called = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        return None

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AgentAction(
                action="click",
                coord_space="absolute",
                x=200,
                y=100,
                reasoning="provisional cached tab click",
                task_status="in_progress",
            )
        return AgentAction(
            action="click",
            x=850,
            y=920,
            reasoning="finish after semantic tab verification",
            task_status="complete",
        )

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        return memory.should_verify_policy_tab_click()

    def verify_action_success(self, provider, pil_image, memory, action, **kwargs) -> SemanticVerifyResult:
        self.verify_called += 1
        return SemanticVerifyResult(handled=True, passed=True, reason="semantic tab ok")

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        memory.mark_policy_tab_confirmed("군사")
        memory.begin_stage("finalize_policy")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class PolicyDragProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("policy_primitive", "")
        self.calls = 0
        self.drag_success_called = False

    def initialize(self, memory: ShortTermMemory) -> None:
        return None

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return [
                AgentAction(
                    action="drag",
                    x=800,
                    y=240,
                    end_x=160,
                    end_y=220,
                    reasoning="apply policy card drag",
                    task_status="in_progress",
                )
            ]
        return AgentAction(
            action="click",
            x=850,
            y=920,
            reasoning="finish after drag",
            task_status="complete",
        )

    def on_actions_success(self, memory: ShortTermMemory, actions: list[AgentAction]) -> None:
        self.drag_success_called = True
        memory.begin_stage("finalize_policy")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class TestRunPrimitiveLoop:
    def setup_method(self):
        ContextManager.reset_instance()
        self.ctx = ContextManager.get_instance()

    def teardown_method(self):
        ContextManager.reset_instance()

    def test_stage_transition_does_not_execute_action_or_consume_step(self, monkeypatch):
        process = TransitionProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        executed = []

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.execute_action",
            lambda action, *args: executed.append(action.action),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="research_select_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=4,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert process.calls == 2
        assert result.steps_taken == 1
        assert executed == ["click"]

    def test_policy_confirmed_tab_uses_semantic_gate_without_similarity_check(self, monkeypatch):
        process = PolicySemanticOnlyProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 200, "y": 100, "confirmed": True}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.screenshots_similar",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("policy should not use screenshot similarity")
            ),
        )

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="policy_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=4,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert process.success_called is True
        assert process.verify_called == 0
        assert memory.policy_state.last_similarity_result == "skipped(policy confirmed absolute cache)"
        assert memory.last_semantic_verify == ""
        assert self.ctx.get_policy_tab_cache().positions["군사"].screen_x == 200

    def test_policy_provisional_tab_uses_semantic_gate_and_confirms_cache(self, monkeypatch):
        process = PolicySemanticGateProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 200, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["군사"],
        )
        memory.begin_stage("click_cached_tab")

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.screenshots_similar",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("policy should not use screenshot similarity")
            ),
        )

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="policy_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=4,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert process.verify_called == 1
        assert memory.policy_state.last_similarity_result == "skipped(policy semantic-only) tab-check pass"
        assert memory.last_semantic_verify == "pass: semantic tab ok"
        assert self.ctx.get_policy_tab_cache().positions["군사"].confirmed is True

    def test_policy_drag_progress_does_not_call_screenshot_similarity(self, monkeypatch):
        process = PolicyDragProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 200, "y": 100, "confirmed": True}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("plan_current_tab")

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.screenshots_similar",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("policy should not use screenshot similarity")
            ),
        )

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="policy_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=4,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert process.drag_success_called is True
        assert memory.policy_state.last_similarity_result == ""
