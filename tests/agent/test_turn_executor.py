from PIL import Image

from computer_use_test.agent.modules.context.context_manager import ContextManager
from computer_use_test.agent.modules.hitl.command_queue import CommandQueue, Directive, DirectiveType
from computer_use_test.agent.modules.hitl.status_ui.state_bridge import AgentStateBridge
from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
from computer_use_test.agent.modules.primitive.multi_step_process import (
    BaseMultiStepProcess,
    ObservationBundle,
    SemanticVerifyResult,
    StageTransition,
    VerificationResult,
)
from computer_use_test.agent.modules.router.primitive_registry import RouterResult
from computer_use_test.agent.turn_executor import PrimitiveLoopResult, run_one_turn, run_primitive_loop
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


class QueuedProvider(BaseVLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__(api_key=None, model="queued", resize_for_vlm=False)
        self.responses = list(responses)

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=8192, use_thinking=True) -> VLMResponse:
        if not self.responses:
            raise AssertionError("No more queued provider responses")
        return VLMResponse(content=self.responses.pop(0))

    def _build_image_content(self, image_path):
        return {"image_path": str(image_path)}

    def _build_pil_image_content(self, pil_image, jpeg_quality=None):
        return {"pil_size": getattr(pil_image, "size", None), "jpeg_quality": jpeg_quality}

    def _build_text_content(self, text: str):
        return {"text": text}

    def get_provider_name(self) -> str:
        return "queued"


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


class ObservationPrepProcess(BaseMultiStepProcess):
    supports_observation = True

    def __init__(self):
        super().__init__("world_congress_primitive", "")
        self.observed_image = None

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.begin_stage("observe_choices")

    def should_observe(self, memory: ShortTermMemory) -> bool:
        return self.observed_image is None

    def observe(self, provider, pil_image, memory, **kwargs) -> ObservationBundle | None:
        self.observed_image = pil_image
        return ObservationBundle(visible_options=[{"label": "안건"}], end_of_list=True)

    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        return AgentAction(
            action="click",
            x=800,
            y=400,
            reasoning="complete after observation",
            task_status="complete",
        )

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class MemoryDecisionFailureProcess(BaseMultiStepProcess):
    supports_observation = True

    def __init__(self):
        super().__init__("city_production_primitive", "")

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.begin_stage("choose_from_memory")

    def decide_from_memory(self, provider, memory, *, high_level_strategy: str) -> bool:
        return False

    def plan_action(self, provider, pil_image, memory, **kwargs):
        raise AssertionError("plan_action should not run when decide_from_memory fails first")


class CityProductionVisibleProgressProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("city_production_primitive", "")

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("hover_scroll_anchor")

    def get_visible_progress(
        self, memory: ShortTermMemory, *, executed_steps: int, hard_max_steps: int
    ) -> tuple[int, int]:
        return (2, 4)

    def plan_action(self, provider, pil_image, memory, **kwargs):
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="city production visible progress should come from stage mapping",
            task_status="complete",
        )

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class TaskLocalHitlLoopProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("city_production_primitive", "")
        self.seen_task_hitl_directive = ""
        self.seen_high_level_strategy = ""
        self.seen_hitl_directive = ""

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.begin_stage("choose_from_memory")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.seen_task_hitl_directive = memory.get_task_hitl_directive()
        self.seen_high_level_strategy = kwargs.get("high_level_strategy", "")
        self.seen_hitl_directive = kwargs.get("hitl_directive", "")
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="complete after capturing task-local hitl directive",
            task_status="complete",
        )

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class CityProductionScrollSettleProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("city_production_primitive", "")
        self.calls = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("scroll_down_for_hidden_choices")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AgentAction(
                action="scroll",
                x=880,
                y=520,
                scroll_amount=-420,
                reasoning="scroll the production list downward before selecting",
                task_status="in_progress",
            )
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="finish after stabilized city production scroll",
            task_status="complete",
        )

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        if action.action == "scroll":
            memory.begin_stage("choose_from_memory")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class GovernorScrollSettleProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("governor_primitive", "")
        self.calls = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.mark_substep("governor_entry_done")
        memory.begin_stage("scroll_down_for_hidden_choices")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return AgentAction(
                action="scroll",
                x=500,
                y=520,
                scroll_amount=-120,
                reasoning="scroll the governor list downward before selecting",
                task_status="in_progress",
            )
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="finish after stabilized governor scroll",
            task_status="complete",
        )

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        if action.action == "scroll":
            memory.begin_stage("choose_from_memory")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class TerminalStateProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("city_production_primitive", "")
        self.verify_completion_called = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("resolve_post_select_followup")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="advance city production into an explicit terminal state",
            task_status="in_progress",
        )

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        memory.begin_stage("production_complete")

    def is_terminal_state(self, memory: ShortTermMemory) -> bool:
        return memory.current_stage == "production_complete"

    def terminal_state_reason(self, memory: ShortTermMemory) -> str:
        return "city production reached explicit terminal state"

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        self.verify_completion_called += 1
        return VerificationResult(False, "city production should not use verify_completion here")


class CityProductionSelectiveScrollVerifyProcess(BaseMultiStepProcess):
    def __init__(self):
        super().__init__("city_production_primitive", "")
        self.plan_calls = 0
        self.scroll_verify_calls = 0

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("scroll_down_for_hidden_choices")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        self.plan_calls += 1
        if self.plan_calls == 1:
            return AgentAction(
                action="scroll",
                x=880,
                y=520,
                scroll_amount=-420,
                reasoning="scroll the production list downward",
                task_status="in_progress",
            )
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="finish after scroll handling",
            task_status="complete",
        )

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        return action.action == "scroll"

    def should_verify_action_after_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        return action.action != "scroll"

    def verify_action_success(self, provider, pil_image, memory, action, **kwargs) -> SemanticVerifyResult:
        if action.action == "scroll":
            self.scroll_verify_calls += 1
        return SemanticVerifyResult(handled=True, passed=True, reason="scroll verified")

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        if action.action == "scroll":
            memory.begin_stage("choose_from_memory")

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class GovernorTraceProcess(BaseMultiStepProcess):
    supports_observation = True

    def __init__(self):
        super().__init__("governor_primitive", "")

    def initialize(self, memory: ShortTermMemory) -> None:
        memory.begin_stage("observe_choices")

    def should_observe(self, memory: ShortTermMemory) -> bool:
        return memory.current_stage == "observe_choices"

    def observe(self, provider, pil_image, memory, **kwargs) -> ObservationBundle | None:
        return ObservationBundle(
            visible_options=[{"id": "pingala", "label": "핑갈라", "note": "진급_가능"}],
            end_of_list=False,
            scroll_anchor={"x": 500, "y": 520, "left": 250, "top": 120, "right": 760, "bottom": 920},
        )

    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        memory.begin_stage("scroll_down_for_hidden_choices")
        return AgentAction(
            action="scroll",
            x=500,
            y=520,
            scroll_amount=-120,
            reasoning="governor trace scroll",
            task_status="in_progress",
        )

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        if action.action == "scroll":
            memory.begin_stage("after_governor_scroll")
        elif action.action == "click":
            memory.begin_stage("governor_done")

    def plan_action(self, provider, pil_image, memory, **kwargs):
        if memory.current_stage == "after_governor_scroll":
            memory.begin_stage("governor_promote_click")
            return StageTransition(stage="governor_promote_click", reason="continue governor flow")
        return AgentAction(
            action="click",
            x=500,
            y=500,
            reasoning="finish traced governor flow",
            task_status="complete",
        )

    def verify_completion(self, provider, pil_image, memory, **kwargs) -> VerificationResult:
        return VerificationResult(True, "ok")


class RecordingRichLogger:
    def __init__(self):
        self.phase_updates: list[str] = []
        self.route_results: list[tuple[str, str]] = []
        self.primitive_events: list[tuple[str, str]] = []

    def update_phase(self, phase: str) -> None:
        self.phase_updates.append(phase)

    def route_result(
        self,
        primitive: str,
        reasoning: str,
        game_turn: int | None,
        macro_turn: int,
        micro_turn: int,
    ) -> None:
        self.route_results.append((primitive, reasoning))

    def primitive_event(self, primitive_tag: str, detail: str) -> None:
        self.primitive_events.append((primitive_tag, detail))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


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

    def test_culture_complete_action_ends_loop_from_terminal_state_when_generic_completion_verifier_fails(
        self, monkeypatch
    ):
        provider = QueuedProvider(
            [
                '{"action":"click","x":500,"y":500,"end_x":0,"end_y":0,"scroll_amount":0,'
                '"button":"left","key":"","text":"","reasoning":"pick civics","task_status":"complete"}',
                '{"complete": false, "reason": "post-click screenshot alone is ambiguous"}',
            ]
        )
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("culture_decision_primitive")
        memory.mark_substep("culture_entry_done")
        memory.begin_stage("direct_culture_select")

        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="culture_decision_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="문화 승리",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=1,
            completion_condition="사회 제도 클릭 완료 시 task_status='complete'.",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert result.completed is True
        assert memory.current_stage == "culture_complete"

    def test_research_complete_action_ends_loop_from_terminal_state_when_generic_completion_verifier_fails(
        self, monkeypatch
    ):
        provider = QueuedProvider(
            [
                '{"action":"click","x":500,"y":500,"end_x":0,"end_y":0,"scroll_amount":0,'
                '"button":"left","key":"","text":"","reasoning":"pick research","task_status":"complete"}',
                '{"complete": false, "reason": "post-click screenshot alone is ambiguous"}',
            ]
        )
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        memory.mark_substep("research_entry_done")
        memory.begin_stage("direct_research_select")

        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
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
            strategy_string="과학 승리",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=1,
            completion_condition="기술 클릭 완료 시 task_status='complete'.",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert result.completed is True
        assert memory.current_stage == "research_complete"

    def test_culture_entry_no_diff_uses_semantic_verify_and_continues_loop(self, monkeypatch):
        provider = QueuedProvider(
            [
                '{"culture_screen_ready": false, "notification_visible": true, "reasoning": "우하단 알림만 보임"}',
                '{"culture_screen_ready": true, "notification_visible": false, '
                '"reasoning": "좌측 상단 선택 창이 열림"}',
                '{"action":"click","x":500,"y":500,"end_x":0,"end_y":0,"scroll_amount":0,'
                '"button":"left","key":"","text":"","reasoning":"pick civics","task_status":"complete"}',
            ]
        )
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("culture_decision_primitive")

        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.screenshots_similar",
            lambda *args, action=None, **kwargs: action.action == "press",
        )

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="culture_decision_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="문화 승리",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=3,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert result.completed is True
        assert "culture_entry_done" in memory.completed_substeps

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

    def test_multistep_observation_moves_cursor_to_center_before_observe(self, monkeypatch):
        process = ObservationPrepProcess()
        provider = DummyProvider()
        initial_image = Image.new("RGB", (100, 100), "red")
        observation_image = Image.new("RGB", (100, 100), "blue")
        post_image = Image.new("RGB", (100, 100), "green")
        memory = ShortTermMemory()
        memory.start_task("world_congress_primitive", enable_choice_catalog=True)
        call_order: list[str] = []
        captures = iter(
            [
                (initial_image, 1440, 900, 0, 0),
                (observation_image, 1440, 900, 0, 0),
                (post_image, 1440, 900, 0, 0),
            ]
        )

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)

        def fake_capture():
            index = len([item for item in call_order if item.startswith("capture")]) + 1
            call_order.append(f"capture{index}")
            return next(captures)

        monkeypatch.setattr("computer_use_test.agent.turn_executor.capture_screen_pil", fake_capture)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.move_cursor_to_center",
            lambda *args: call_order.append("center"),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="world_congress_primitive",
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
            max_steps=2,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert call_order[:3] == ["capture1", "center", "capture2"]
        assert process.observed_image is observation_image

    def test_multistep_memory_decision_failure_updates_status_with_real_error(self, monkeypatch):
        process = MemoryDecisionFailureProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.remember_choices(
            [{"id": "cand1", "label": "기념비"}],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.choice_catalog.end_reached = True
        bridge = AgentStateBridge(self.ctx, CommandQueue())
        bridge.update_current_action("city_production_primitive", "scroll (8800, 5100)", "숨은 선택지 확인")

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
            state_bridge=bridge,
            delay_before_action=0,
        )

        status = bridge.get_status()

        assert result.success is False
        assert result.error_message == "Failed to decide best choice from short-term memory"
        assert status.current_action == "error"
        assert status.current_reasoning == "Failed to decide best choice from short-term memory"

    def test_multistep_trace_events_are_published_to_state_bridge(self, monkeypatch):
        process = TransitionProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        bridge = AgentStateBridge(self.ctx, CommandQueue())

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
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
            state_bridge=bridge,
            delay_before_action=0,
        )

        status = bridge.get_status()

        assert result.success is True
        assert len(status.recent_trace_events) >= 3
        assert any(event["phase"] == "stage" for event in status.recent_trace_events)
        assert any(event["phase"] == "plan" for event in status.recent_trace_events)
        assert any(event["phase"] == "exec" for event in status.recent_trace_events)

    def test_governor_multistep_trace_events_are_published_to_rich(self, monkeypatch):
        process = GovernorTraceProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        rich = RecordingRichLogger()

        monkeypatch.setattr("computer_use_test.agent.turn_executor.RichLogger.get", lambda: rich)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="governor_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="과학 승리",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=5,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert any(tag == "GOVERNOR" and "observe" in detail for tag, detail in rich.primitive_events)
        assert any(tag == "GOVERNOR" and "plan" in detail for tag, detail in rich.primitive_events)
        assert any(tag == "GOVERNOR" and "exec" in detail for tag, detail in rich.primitive_events)
        assert any(tag == "GOVERNOR" and "stage" in detail for tag, detail in rich.primitive_events)

    def test_city_production_uses_process_visible_progress_in_state_updates(self, monkeypatch):
        process = CityProductionVisibleProgressProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        bridge = AgentStateBridge(self.ctx, CommandQueue())
        progress_updates: list[tuple[bool, int, int, str]] = []

        original_update_multi_step = bridge.update_multi_step

        def record_update_multi_step(**kwargs):
            progress_updates.append(
                (
                    bool(kwargs.get("active", False)),
                    int(kwargs.get("step", 0)),
                    int(kwargs.get("max_steps", 0)),
                    str(kwargs.get("stage", "")),
                )
            )
            return original_update_multi_step(**kwargs)

        bridge.update_multi_step = record_update_multi_step

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
            max_steps=18,
            completion_condition="",
            planner_img_config=None,
            state_bridge=bridge,
            delay_before_action=0,
        )

        assert result.success is True
        assert any(active and step == 2 and max_steps == 4 for active, step, max_steps, _ in progress_updates)

    def test_city_production_scroll_waits_before_post_action_capture(self, monkeypatch):
        process = CityProductionScrollSettleProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        sleeps: list[float] = []

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.time.sleep", lambda seconds: sleeps.append(seconds))

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
            max_steps=18,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert sleeps == [0.55]

    def test_governor_scroll_waits_before_post_action_capture(self, monkeypatch):
        process = GovernorScrollSettleProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        sleeps: list[float] = []

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.time.sleep", lambda seconds: sleeps.append(seconds))

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="governor_primitive",
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
            max_steps=18,
            completion_condition="",
            planner_img_config=None,
            delay_before_action=0,
        )

        assert result.success is True
        assert sleeps == [0.55]

    def test_city_production_terminal_state_completes_without_verify_completion(self, monkeypatch):
        process = TerminalStateProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
        assert result.completed is True
        assert process.verify_completion_called == 0

    def test_city_production_scroll_skips_semantic_verify_when_raw_ui_changed(self, monkeypatch):
        process = CityProductionSelectiveScrollVerifyProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
        assert process.scroll_verify_calls == 0

    def test_city_production_scroll_semantic_verifies_when_raw_ui_unchanged(self, monkeypatch):
        process = CityProductionSelectiveScrollVerifyProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.screenshots_similar",
            lambda *args, action=None, **kwargs: action.action == "scroll",
        )

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
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
        assert process.scroll_verify_calls == 1

    def test_run_one_turn_reroutes_after_completed_city_production(self, monkeypatch):
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        routed_primitives = iter(
            [
                RouterResult("city_production_primitive", "initial city production"),
                RouterResult("popup_primitive", "follow-up popup after production"),
            ]
        )
        executed_actions: list[str] = []

        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.route_primitive",
            lambda *args, **kwargs: next(routed_primitives),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.run_primitive_loop",
            lambda **kwargs: PrimitiveLoopResult(
                success=True,
                completed=True,
                re_route=False,
                steps_taken=2,
                last_action=AgentAction(
                    action="click",
                    x=500,
                    y=500,
                    reasoning="city production completed",
                    task_status="in_progress",
                ),
            ),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.plan_action",
            lambda *args, **kwargs: AgentAction(
                action="press",
                key="enter",
                reasoning="dismiss popup after production",
                task_status="complete",
            ),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.execute_action",
            lambda action, *args: executed_actions.append(action.action),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.time.sleep", lambda *_args: None)

        summary = run_one_turn(
            router_provider=provider,
            planner_provider=provider,
            context_manager=self.ctx,
            turn_number=1,
            delay_before_action=0,
        )

        assert summary is not None
        assert summary.primitive == "popup_primitive"
        assert summary.action_type == "press"
        assert executed_actions == ["press"]

    def test_run_primitive_loop_persists_mid_task_hitl_override_in_memory(self, monkeypatch):
        process = TaskLocalHitlLoopProcess()
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive")
        command_queue = CommandQueue()
        command_queue.push(
            Directive(
                directive_type=DirectiveType.CHANGE_STRATEGY,
                payload="이번 task에서는 캠퍼스 먼저 지어",
                source="test",
            )
        )

        monkeypatch.setattr("computer_use_test.agent.turn_executor.get_multi_step_process", lambda *args: process)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.move_cursor_to_center", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.screenshots_similar", lambda *args, **kwargs: False)

        result = run_primitive_loop(
            planner_provider=provider,
            primitive_name="city_production_primitive",
            screen_w=1440,
            screen_h=900,
            normalizing_range=1000,
            x_offset=0,
            y_offset=0,
            strategy_string="과학 승리",
            recent_actions_str="없음",
            hitl_directive=None,
            memory=memory,
            ctx=self.ctx,
            max_steps=3,
            completion_condition="",
            planner_img_config=None,
            command_queue=command_queue,
            delay_before_action=0,
        )

        assert result.success is True
        assert process.seen_task_hitl_directive == "이번 task에서는 캠퍼스 먼저 지어"
        assert process.seen_hitl_directive == "이번 task에서는 캠퍼스 먼저 지어"
        assert "[사용자 최우선 지시] 이번 task에서는 캠퍼스 먼저 지어" in process.seen_high_level_strategy

    def test_run_one_turn_refreshes_rich_routing_after_completed_city_production(self, monkeypatch):
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        rich = RecordingRichLogger()
        routed_primitives = iter(
            [
                RouterResult("city_production_primitive", "initial city production"),
                RouterResult("popup_primitive", "follow-up popup after production"),
            ]
        )

        monkeypatch.setattr("computer_use_test.agent.turn_executor.RichLogger.get", lambda: rich)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.route_primitive",
            lambda *args, **kwargs: next(routed_primitives),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.run_primitive_loop",
            lambda **kwargs: PrimitiveLoopResult(
                success=True,
                completed=True,
                re_route=False,
                steps_taken=2,
                last_action=AgentAction(
                    action="click",
                    x=500,
                    y=500,
                    reasoning="city production completed",
                    task_status="in_progress",
                ),
            ),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.plan_action",
            lambda *args, **kwargs: AgentAction(
                action="press",
                key="enter",
                reasoning="dismiss popup after production",
                task_status="complete",
            ),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.execute_action", lambda *args: None)
        monkeypatch.setattr("computer_use_test.agent.turn_executor.time.sleep", lambda *_args: None)

        summary = run_one_turn(
            router_provider=provider,
            planner_provider=provider,
            context_manager=self.ctx,
            turn_number=1,
            delay_before_action=0,
        )

        assert summary is not None
        assert rich.phase_updates.count("routing") == 2
        assert rich.route_results == [
            ("city_production_primitive", "initial city production"),
            ("popup_primitive", "follow-up popup after production"),
        ]

    def test_run_one_turn_does_not_restart_completed_voting_on_same_primitive_reroute(self, monkeypatch):
        provider = DummyProvider()
        image = Image.new("RGB", (100, 100))
        routed_primitives = iter(
            [
                RouterResult("voting_primitive", "initial voting"),
                RouterResult("voting_primitive", "stale reroute still sees voting"),
                RouterResult("popup_primitive", "follow-up popup after voting"),
            ]
        )
        executed_actions: list[str] = []
        loop_calls = {"count": 0}

        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.capture_screen_pil",
            lambda: (image, 1440, 900, 0, 0),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.route_primitive",
            lambda *args, **kwargs: next(routed_primitives),
        )

        def fake_run_primitive_loop(**kwargs):
            loop_calls["count"] += 1
            if loop_calls["count"] > 1:
                raise AssertionError("completed voting primitive should not restart on same-primitive reroute")
            return PrimitiveLoopResult(
                success=True,
                completed=True,
                re_route=False,
                steps_taken=3,
                last_action=AgentAction(
                    action="press",
                    key="escape",
                    reasoning="voting completed and exited",
                    task_status="in_progress",
                ),
            )

        monkeypatch.setattr("computer_use_test.agent.turn_executor.run_primitive_loop", fake_run_primitive_loop)
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.plan_action",
            lambda *args, **kwargs: AgentAction(
                action="press",
                key="enter",
                reasoning="dismiss popup after voting",
                task_status="complete",
            ),
        )
        monkeypatch.setattr(
            "computer_use_test.agent.turn_executor.execute_action",
            lambda action, *args: executed_actions.append(action.action),
        )
        monkeypatch.setattr("computer_use_test.agent.turn_executor.time.sleep", lambda *_args: None)

        summary = run_one_turn(
            router_provider=provider,
            planner_provider=provider,
            context_manager=self.ctx,
            turn_number=1,
            delay_before_action=0,
        )

        assert summary is not None
        assert summary.primitive == "popup_primitive"
        assert summary.action_type == "press"
        assert executed_actions == ["press"]
        assert loop_calls["count"] == 1
