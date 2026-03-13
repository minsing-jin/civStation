"""Unit tests for class-based multi-step processes."""

import json

from PIL import Image

from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
from computer_use_test.agent.modules.primitive.multi_step_process import (
    _POLICY_RIGHT_CARD_LIST_RATIOS,
    _POLICY_RIGHT_TAB_BAR_RATIOS,
    ObservationBundle,
    StageTransition,
    get_multi_step_process,
)
from computer_use_test.agent.modules.router.primitive_registry import get_primitive_prompt
from computer_use_test.utils.llm_provider.base import BaseVLMProvider, VLMResponse


class FakeProvider(BaseVLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__(api_key=None, model="fake", resize_for_vlm=False)
        self.responses = list(responses)
        self.last_text = ""
        self.last_pil_size = None

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=8192, use_thinking=True) -> VLMResponse:
        for part in content_parts:
            if isinstance(part, dict) and "text" in part:
                self.last_text = str(part["text"])
            if isinstance(part, dict) and "pil_size" in part:
                self.last_pil_size = part["pil_size"]
        if not self.responses:
            raise AssertionError("No more fake responses queued")
        return VLMResponse(content=self.responses.pop(0))

    def _build_image_content(self, image_path):
        return {"image_path": str(image_path)}

    def _build_pil_image_content(self, pil_image, jpeg_quality=None):
        return {"pil_size": getattr(pil_image, "size", None), "jpeg_quality": jpeg_quality}

    def _build_text_content(self, text: str):
        return {"text": text}

    def get_provider_name(self) -> str:
        return "fake"


def _set_default_policy_geometry(
    memory: ShortTermMemory,
    *,
    region_w: int = 1000,
    region_h: int = 1000,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    memory.set_policy_capture_geometry(region_w, region_h, x_offset, y_offset)


def _policy_tabbar_global_norm(process, image, *, x: int, y: int, normalizing_range: int) -> tuple[int, int]:
    _, crop_box = process._crop_policy_region(image, _POLICY_RIGHT_TAB_BAR_RATIOS)  # noqa: SLF001
    return process._crop_local_norm_to_global_norm(  # noqa: SLF001
        x,
        y,
        crop_box,
        image,
        normalizing_range=normalizing_range,
    )


def _policy_crop_size(process, image, ratios) -> tuple[int, int]:
    cropped, _ = process._crop_policy_region(image, ratios)  # noqa: SLF001
    return cropped.size


class TestObservationAssistedProcess:
    def test_religion_process_scrolls_from_observer_anchor(self):
        process = get_multi_step_process(
            "religion_primitive",
            "초록색 '종교관 세우기' 버튼 클릭 완료 시 task_status='complete'.",
        )
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "신성한 불꽃"}],
                end_of_list=False,
                scroll_anchor={"x": 410, "y": 520, "left": 280, "top": 160, "right": 640, "bottom": 900},
            ),
        )

        assert action is not None
        assert action.action == "scroll"
        assert action.x == 410
        assert action.y == 520
        assert action.scroll_amount < 0

    def test_scroll_action_is_resolved_back_to_saved_anchor(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_anchor={"x": 600, "y": 500, "left": 450, "top": 120, "right": 850, "bottom": 920},
        )

        action = process.resolve_action(
            process.consume_observation(
                memory,
                ObservationBundle(visible_options=[{"label": "개척자"}], end_of_list=False),
            ),
            memory,
        )
        assert action is not None
        assert action.x == 600
        assert action.y == 500

    def test_observer_prompt_uses_passed_normalizing_range_for_anchor_contract(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", normalizing_range=777, enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"label": "신성한 불꽃"}],
                        "end_of_list": True,
                        "scroll_anchor": {"x": 300, "y": 400, "left": 100, "top": 100, "right": 700, "bottom": 700},
                    }
                )
            ]
        )

        observation = process.observe(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=777,
        )

        assert observation is not None
        assert "0-777 normalized coordinates" in provider.last_text


class TestPromptUpdates:
    def test_policy_prompt_contains_two_entry_branches(self):
        prompt = get_primitive_prompt("policy_primitive")
        assert "사회제도 완성" in prompt
        assert "새 정부 선택" in prompt
        assert "모든 정책 배정" in prompt
        assert "실패한 탭 하나만 다시 찾아 cached 좌표를 수정한다" in prompt
        assert "'전체' 탭은 초기 overview 상태" in prompt
        assert "혼합 overview 목록이면 '전체' 상태" in prompt

    def test_popup_prompt_handles_policy_change_popup(self):
        prompt = get_primitive_prompt("popup_primitive")
        assert "정책변경" in prompt


class TestPolicyProcess:
    def test_policy_entry_does_not_bootstrap_before_card_screen(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps({"policy_screen_ready": False}),
                json.dumps({"action": "click", "x": 500, "y": 720, "reasoning": "정책변경 버튼 클릭"}),
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "click"
        assert memory.current_stage == "policy_entry"
        assert memory.has_policy_bootstrap() is False

    def test_bootstrap_builds_tab_cache_and_first_lazy_verified_tab_click(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        _set_default_policy_geometry(memory)
        process.initialize(memory)
        image = Image.new("RGB", (100, 100))
        provider = FakeProvider(
            [
                json.dumps({"policy_screen_ready": True}),
                json.dumps(
                    {
                        "policy_screen_ready": True,
                        "overview_mode": True,
                        "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],
                        "wild_slot_active": True,
                        "slot_inventory": [
                            {"slot_id": "military_1", "slot_type": "군사", "is_empty": True},
                            {
                                "slot_id": "wild_1",
                                "slot_type": "와일드카드",
                                "is_empty": True,
                                "is_wild": True,
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "tab_positions": [
                            {"tab_name": "군사", "x": 340, "y": 500},
                            {"tab_name": "경제", "x": 460, "y": 500},
                            {"tab_name": "외교", "x": 580, "y": 500},
                            {"tab_name": "와일드카드", "x": 700, "y": 500},
                            {"tab_name": "암흑", "x": 820, "y": 500},
                        ],
                    }
                ),
            ]
        )
        expected_military = _policy_tabbar_global_norm(process, image, x=340, y=500, normalizing_range=1000)

        action = process.plan_action(
            provider,
            image,
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "click"
        assert action.coord_space == "absolute"
        assert action.x == expected_military[0]
        assert action.y == expected_military[1]
        assert memory.is_policy_entry_done() is True
        assert memory.has_policy_bootstrap() is True
        assert memory.policy_state.eligible_tabs_queue == ["군사", "경제", "외교", "와일드카드", "암흑"]
        assert memory.policy_state.overview_mode is True
        assert memory.current_stage == "click_cached_tab"
        assert set(memory.policy_state.tab_positions) == {"군사", "경제", "외교", "와일드카드", "암흑"}
        assert memory.policy_state.calibration_pending_tabs == []
        assert memory.policy_state.provisional_tabs == {"군사", "경제", "외교", "와일드카드", "암흑"}
        assert provider.last_pil_size == _policy_crop_size(process, image, _POLICY_RIGHT_TAB_BAR_RATIOS)

    def test_bootstrap_upscales_legacy_1000_positions_for_10000_range(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=10000, enable_policy_state=True)
        _set_default_policy_geometry(memory, region_w=10000, region_h=10000)
        memory.mark_policy_entry_done()
        image = Image.new("RGB", (100, 100))
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "policy_screen_ready": True,
                        "overview_mode": True,
                        "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],
                        "wild_slot_active": True,
                        "slot_inventory": [{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
                    }
                ),
                json.dumps(
                    {
                        "tab_positions": [
                            {"tab_name": "군사", "x": 340, "y": 500},
                            {"tab_name": "경제", "x": 460, "y": 500},
                            {"tab_name": "외교", "x": 580, "y": 500},
                            {"tab_name": "와일드카드", "x": 700, "y": 500},
                            {"tab_name": "암흑", "x": 820, "y": 500},
                        ],
                    }
                ),
            ]
        )
        expected_military = _policy_tabbar_global_norm(process, image, x=3400, y=5000, normalizing_range=10000)

        bootstrapped = process._bootstrap_policy_screen(  # noqa: SLF001
            provider,
            image,
            memory,
            high_level_strategy="과학 승리",
            normalizing_range=10000,
        )

        assert bootstrapped is True
        assert memory.policy_state.tab_positions["군사"].screen_x == expected_military[0]
        assert memory.policy_state.tab_positions["군사"].screen_y == expected_military[1]
        assert "coord_scale=legacy1000x10" in memory.policy_state.bootstrap_summary
        assert memory.current_stage == "click_cached_tab"
        assert memory.policy_state.calibration_pending_tabs == []
        assert provider.last_pil_size == _policy_crop_size(process, image, _POLICY_RIGHT_TAB_BAR_RATIOS)

        action = process.plan_action(
            FakeProvider([]),
            image,
            memory,
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "click"
        assert action.coord_space == "absolute"
        assert action.x == expected_military[0]
        assert action.y == expected_military[1]

    def test_click_cached_tab_verification_uses_semantic_verifier_for_provisional_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "경제", "x": 760, "y": 100}],
            eligible_tabs_queue=["경제"],
            slot_inventory=[{"slot_id": "economic_1", "slot_type": "경제", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["경제"],
        )
        memory.begin_stage("click_cached_tab")
        provider = FakeProvider([json.dumps({"match": True, "observed_tab": "경제", "reason": "노란 카드"})])

        verified = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is True
        assert "기대 탭: 경제" in provider.last_text
        assert "이 이미지는 정책 화면 오른쪽 카드 목록만 crop한 이미지다." in provider.last_text
        assert "이 crop에는 좌측 파란 슬롯 영역이 포함되지 않는다." in provider.last_text
        assert provider.last_pil_size == _policy_crop_size(
            process, Image.new("RGB", (100, 100)), _POLICY_RIGHT_CARD_LIST_RATIOS
        )

    def test_bootstrap_fails_when_any_visible_policy_tab_coord_is_missing(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        _set_default_policy_geometry(memory)
        memory.mark_policy_entry_done()
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "policy_screen_ready": True,
                        "overview_mode": True,
                        "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],
                        "wild_slot_active": True,
                        "slot_inventory": [{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
                    }
                ),
                json.dumps(
                    {
                        "tab_positions": [
                            {"tab_name": "군사", "x": 340, "y": 500},
                            {"tab_name": "경제", "x": 460, "y": 500},
                            {"tab_name": "외교", "x": 580, "y": 500},
                            {"tab_name": "와일드카드", "x": 700, "y": 500},
                        ],
                    }
                ),
            ]
        )

        assert (
            process._bootstrap_policy_screen(  # noqa: SLF001
                provider,
                Image.new("RGB", (100, 100)),
                memory,
                high_level_strategy="과학 승리",
                normalizing_range=1000,
            )
            is False
        )

    def test_plan_current_tab_builds_direct_drag_actions(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.mark_policy_tab_confirmed("군사")
        memory.begin_stage("plan_current_tab")
        provider = FakeProvider(
            [
                json.dumps(
                    [
                        {
                            "action": "drag",
                            "x": 820,
                            "y": 240,
                            "end_x": 140,
                            "end_y": 220,
                            "reasoning": "규율 카드를 군사 슬롯으로 드래그",
                            "policy_card_name": "규율",
                            "policy_target_slot_id": "military_1",
                            "policy_source_tab": "군사",
                            "policy_reasoning": "야만인 대응용 군사 카드 장착",
                            "task_status": "in_progress",
                        }
                    ]
                ),
            ]
        )

        actions = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(actions, list)
        assert len(actions) == 1
        assert actions[0].action == "drag"
        assert actions[0].x == 820
        assert actions[0].end_x == 140
        assert actions[0].policy_card_name == "규율"
        assert actions[0].policy_target_slot_id == "military_1"
        assert memory.is_policy_tab_confirmed("군사") is True
        assert memory.current_stage == "plan_current_tab"

    def test_plan_current_tab_rejects_invalid_cross_category_drag_bundle(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "경제", "x": 760, "y": 100},
                {"tab_name": "군사", "x": 700, "y": 100},
            ],
            eligible_tabs_queue=["경제", "군사"],
            slot_inventory=[
                {"slot_id": "military_1", "slot_type": "군사", "is_empty": True},
                {"slot_id": "economic_1", "slot_type": "경제", "is_empty": True},
            ],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="경제",
        )
        memory.mark_policy_tab_confirmed("경제")
        memory.begin_stage("plan_current_tab")
        provider = FakeProvider(
            [
                json.dumps(
                    [
                        {
                            "action": "drag",
                            "x": 810,
                            "y": 210,
                            "end_x": 130,
                            "end_y": 190,
                            "reasoning": "잘못된 슬롯으로 드래그",
                            "policy_card_name": "도시 계획",
                            "policy_target_slot_id": "military_1",
                            "policy_source_tab": "경제",
                            "policy_reasoning": "잘못된 슬롯",
                        }
                    ]
                ),
            ]
        )

        planned = process._plan_current_tab_actions(  # noqa: SLF001
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert planned is None

    def test_wild_slot_prompt_uses_previous_selection_reason_conservatively(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "경제", "x": 760, "y": 100},
                {"tab_name": "와일드카드", "x": 880, "y": 100},
            ],
            eligible_tabs_queue=["경제", "와일드카드"],
            slot_inventory=[
                {
                    "slot_id": "wild_1",
                    "slot_type": "와일드카드",
                    "current_card_name": "도시 계획",
                    "is_empty": False,
                    "is_wild": True,
                    "selected_from_tab": "경제",
                    "selection_reason": "과학 승리 기준으로 초기 성장 보너스가 중요",
                }
            ],
            wild_slot_active=True,
            overview_mode=False,
            selected_tab_name="경제",
        )
        memory.mark_policy_tab_confirmed("경제")
        memory.begin_stage("plan_current_tab")
        provider = FakeProvider([json.dumps([])])

        planned = process._plan_current_tab_actions(  # noqa: SLF001
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(planned, StageTransition)
        assert "현재 선택 이유:과학 승리 기준으로 초기 성장 보너스가 중요" in provider.last_text
        assert "단순히 현재 탭에도 좋은 카드가 보인다는 이유만으로 와일드 슬롯을 매 탭 바꾸지 마." in provider.last_text

    def test_policy_plan_with_no_replacements_transitions_to_next_tab_stage(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100},
                {"tab_name": "경제", "x": 760, "y": 100},
                {"tab_name": "외교", "x": 820, "y": 100},
                {"tab_name": "와일드카드", "x": 880, "y": 100},
                {"tab_name": "암흑", "x": 940, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.mark_policy_tab_confirmed("군사")
        memory.begin_stage("plan_current_tab")
        provider = FakeProvider([json.dumps([])])

        actions = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert actions is not None
        assert isinstance(actions, StageTransition)
        assert actions.stage == "click_next_tab"
        assert memory.current_stage == "click_next_tab"
        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.completed_tabs == ["군사"]

    def test_policy_next_tab_click_success_confirms_advanced_current_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100},
                {"tab_name": "경제", "x": 760, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.mark_policy_tab_confirmed("군사")
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("click_next_tab")

        process.on_action_success(
            memory,
            type(
                "A",
                (),
                {"action": "click", "reasoning": "다음 정책 카테고리 탭 '경제'을 cached position으로 클릭"},
            )(),
        )

        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.completed_tabs == ["군사"]
        assert memory.is_policy_tab_confirmed("경제") is True
        assert memory.current_stage == "plan_current_tab"

    def test_policy_click_next_tab_targets_already_advanced_current_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100},
                {"tab_name": "경제", "x": 760, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.mark_policy_tab_confirmed("군사")
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("click_next_tab")

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "click"
        assert action.x == 760
        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.completed_tabs == ["군사"]

    def test_finalize_policy_plan_forces_in_progress_until_confirm_popup(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("finalize_policy")

        action = process.plan_action(
            FakeProvider(
                [
                    json.dumps(
                        {
                            "action": "click",
                            "x": 860,
                            "y": 930,
                            "reasoning": "모든 정책 배정 버튼 클릭",
                            "task_status": "complete",
                        }
                    )
                ]
            ),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.task_status == "in_progress"

    def test_finalize_click_success_moves_to_confirm_policy_popup(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("finalize_policy")

        process.on_action_success(
            memory,
            type(
                "A",
                (),
                {"action": "click", "reasoning": "모든 정책 배정 버튼 클릭"},
            )(),
        )

        assert memory.current_stage == "confirm_policy_popup"

    def test_confirm_policy_popup_click_is_terminal_complete(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("confirm_policy_popup")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "action": "click",
                        "x": 610,
                        "y": 560,
                        "reasoning": "확인 팝업의 예 버튼 클릭",
                        "task_status": "in_progress",
                    }
                )
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.task_status == "complete"
        assert "'예' 또는 확인 버튼" in provider.last_text

    def test_first_tab_click_failure_relocalizes_current_tab_only(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        _set_default_policy_geometry(memory)
        memory.mark_policy_entry_done()
        image = Image.new("RGB", (100, 100))
        military_pos = _policy_tabbar_global_norm(process, image, x=340, y=500, normalizing_range=1000)
        economic_pos = _policy_tabbar_global_norm(process, image, x=460, y=500, normalizing_range=1000)
        diplomatic_pos = _policy_tabbar_global_norm(process, image, x=580, y=500, normalizing_range=1000)
        wildcard_pos = _policy_tabbar_global_norm(process, image, x=700, y=500, normalizing_range=1000)
        dark_pos = _policy_tabbar_global_norm(process, image, x=820, y=500, normalizing_range=1000)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": military_pos[0], "y": military_pos[1]},
                {"tab_name": "경제", "x": economic_pos[0], "y": economic_pos[1]},
                {"tab_name": "외교", "x": diplomatic_pos[0], "y": diplomatic_pos[1]},
                {"tab_name": "와일드카드", "x": wildcard_pos[0], "y": wildcard_pos[1]},
                {"tab_name": "암흑", "x": dark_pos[0], "y": dark_pos[1]},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")
        provider = FakeProvider([json.dumps({"found": True, "tab_name": "군사", "x": 360, "y": 520})])
        expected = _policy_tabbar_global_norm(process, image, x=360, y=520, normalizing_range=1000)

        resolution = process.handle_no_progress(
            provider,
            image,
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "click_cached_tab"
        assert memory.policy_state.tab_positions["군사"].screen_x == expected[0]
        assert memory.policy_state.tab_positions["군사"].screen_y == expected[1]
        assert memory.policy_state.provisional_tabs == {"군사"}
        assert memory.policy_state.eligible_tabs_queue == ["군사", "경제", "외교", "와일드카드", "암흑"]
        assert memory.policy_state.entry_done is True
        assert memory.policy_state.last_event == "tab click retry=군사 relocalized=yes"
        assert memory.policy_state.last_relocalize_result == (
            f"군사:raw=(360,520) -> abs=({expected[0]},{expected[1]})"
        )
        assert provider.last_pil_size == _policy_crop_size(process, image, _POLICY_RIGHT_TAB_BAR_RATIOS)

    def test_first_tab_click_failure_upscales_legacy_relocalize_for_10000_range(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=10000, enable_policy_state=True)
        _set_default_policy_geometry(memory, region_w=10000, region_h=10000)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 6910, "y": 2680},
                {"tab_name": "경제", "x": 7420, "y": 2680},
                {"tab_name": "외교", "x": 7930, "y": 2680},
                {"tab_name": "와일드카드", "x": 8440, "y": 2680},
                {"tab_name": "암흑", "x": 8950, "y": 2680},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")
        image = Image.new("RGB", (100, 100))
        provider = FakeProvider([json.dumps({"found": True, "tab_name": "군사", "x": 340, "y": 500})])
        expected = _policy_tabbar_global_norm(process, image, x=3400, y=5000, normalizing_range=10000)

        resolution = process.handle_no_progress(
            provider,
            image,
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "click_cached_tab"
        assert memory.policy_state.tab_positions["군사"].screen_x == expected[0]
        assert memory.policy_state.tab_positions["군사"].screen_y == expected[1]
        assert memory.policy_state.last_relocalize_result == (
            f"군사:raw=(340,500) -> (3400,5000) scaled(x10) -> abs=({expected[0]},{expected[1]})"
        )
        assert memory.policy_state.last_event == "tab click retry=군사 relocalized=yes"
        assert provider.last_pil_size == _policy_crop_size(process, image, _POLICY_RIGHT_TAB_BAR_RATIOS)

    def test_first_tab_click_failure_keeps_current_range_relocalize_for_10000_range(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=10000, enable_policy_state=True)
        _set_default_policy_geometry(memory, region_w=10000, region_h=10000)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 6910, "y": 2680},
                {"tab_name": "경제", "x": 7420, "y": 2680},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")
        image = Image.new("RGB", (100, 100))
        provider = FakeProvider([json.dumps({"found": True, "tab_name": "군사", "x": 3410, "y": 4980})])
        expected = _policy_tabbar_global_norm(process, image, x=3410, y=4980, normalizing_range=10000)

        resolution = process.handle_no_progress(
            provider,
            image,
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.policy_state.tab_positions["군사"].screen_x == expected[0]
        assert memory.policy_state.tab_positions["군사"].screen_y == expected[1]
        assert memory.policy_state.last_relocalize_result == (
            f"군사:raw=(3410,4980) -> abs=({expected[0]},{expected[1]})"
        )

    def test_first_tab_click_failure_rejects_implausible_relocalize_for_10000_range(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", normalizing_range=10000, enable_policy_state=True)
        _set_default_policy_geometry(memory, region_w=10000, region_h=10000)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 6910, "y": 2680},
                {"tab_name": "경제", "x": 7420, "y": 2680},
            ],
            eligible_tabs_queue=["군사", "경제"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")
        provider = FakeProvider([json.dumps({"found": True, "tab_name": "군사", "x": 120, "y": 950})])

        resolution = process.handle_no_progress(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "click_cached_tab"
        assert memory.policy_state.tab_positions["군사"].screen_x == 6910
        assert memory.policy_state.tab_positions["군사"].screen_y == 2680
        assert memory.policy_state.last_relocalize_result.startswith(
            "군사:reject raw=(120,950) existing=(6910,2680) budget=400"
        )
        assert memory.policy_state.last_event == "tab click retry=군사 relocalized=no"

    def test_next_tab_click_failure_relocalizes_next_tab_only(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        _set_default_policy_geometry(memory)
        memory.mark_policy_entry_done()
        image = Image.new("RGB", (100, 100))
        military_pos = _policy_tabbar_global_norm(process, image, x=340, y=500, normalizing_range=1000)
        economic_pos = _policy_tabbar_global_norm(process, image, x=460, y=500, normalizing_range=1000)
        diplomatic_pos = _policy_tabbar_global_norm(process, image, x=580, y=500, normalizing_range=1000)
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": military_pos[0], "y": military_pos[1]},
                {"tab_name": "경제", "x": economic_pos[0], "y": economic_pos[1]},
                {"tab_name": "외교", "x": diplomatic_pos[0], "y": diplomatic_pos[1]},
            ],
            eligible_tabs_queue=["군사", "경제", "외교"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.mark_policy_tab_confirmed("군사")
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("click_next_tab")
        provider = FakeProvider(
            [
                json.dumps({"found": True, "tab_name": "경제", "x": 470, "y": 520}),
            ]
        )
        expected = _policy_tabbar_global_norm(process, image, x=470, y=520, normalizing_range=1000)

        resolution = process.handle_no_progress(
            provider,
            image,
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.policy_state.tab_positions["군사"].screen_x == military_pos[0]
        assert memory.policy_state.tab_positions["군사"].screen_y == military_pos[1]
        assert memory.policy_state.tab_positions["경제"].screen_x == expected[0]
        assert memory.policy_state.tab_positions["경제"].screen_y == expected[1]
        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.last_relocalize_result == (
            f"경제:raw=(470,520) -> abs=({expected[0]},{expected[1]})"
        )

    def test_second_tab_click_failure_switches_to_generic_fallback(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100},
                {"tab_name": "경제", "x": 760, "y": 100},
                {"tab_name": "외교", "x": 820, "y": 100},
                {"tab_name": "와일드카드", "x": 880, "y": 100},
                {"tab_name": "암흑", "x": 940, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("click_cached_tab")
        memory.increment_stage_failure("click_cached_tab:군사")
        provider = FakeProvider([])

        resolution = process.handle_no_progress(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "generic_fallback"

    def test_full_session_cache_bootstrap_skips_tab_positions_scan(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        _set_default_policy_geometry(memory)
        memory.mark_policy_entry_done()
        memory.seed_policy_tab_cache(
            {
                "positions": {
                    "군사": {"screen_x": 700, "screen_y": 100, "confirmed": True},
                    "경제": {"screen_x": 760, "screen_y": 100, "confirmed": True},
                    "외교": {"screen_x": 820, "screen_y": 100, "confirmed": True},
                    "와일드카드": {"screen_x": 880, "screen_y": 100, "confirmed": True},
                    "암흑": {"screen_x": 940, "screen_y": 100, "confirmed": True},
                },
                "confirmed_tabs": ["군사", "경제", "외교", "와일드카드", "암흑"],
                "provisional_tabs": [],
                "capture_geometry": {
                    "region_w": 1000,
                    "region_h": 1000,
                    "x_offset": 0,
                    "y_offset": 0,
                },
                "calibration_complete": True,
            }
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "policy_screen_ready": True,
                        "overview_mode": True,
                        "visible_tabs": ["군사", "경제", "외교", "와일드카드", "암흑"],
                        "wild_slot_active": True,
                        "slot_inventory": [{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
                    }
                )
            ]
        )

        bootstrapped = process._bootstrap_policy_screen(  # noqa: SLF001
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            high_level_strategy="과학 승리",
            normalizing_range=1000,
        )

        assert bootstrapped is True
        assert "tab_positions는 반환하지 마" in provider.last_text
        assert memory.current_stage == "click_cached_tab"
        assert memory.policy_state.cache_source == "session_cache"
        assert memory.policy_state.calibration_pending_tabs == []

    def test_policy_calibration_verification_passes_for_matching_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "경제", "x": 760, "y": 100}],
            eligible_tabs_queue=["경제"],
            slot_inventory=[{"slot_id": "economic_1", "slot_type": "경제", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["경제"],
            calibration_pending_tabs=["경제"],
        )
        memory.begin_stage("calibrate_tabs")
        provider = FakeProvider([json.dumps({"match": True, "observed_tab": "경제", "reason": "노란 카드"})])
        image = Image.new("RGB", (100, 100))

        verified = process.verify_action_success(
            provider,
            image,
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is True
        assert "기대 탭: 경제" in provider.last_text
        assert "이 이미지는 정책 화면 오른쪽 카드 목록만 crop한 이미지다." in provider.last_text
        assert "이 crop에는 좌측 파란 슬롯 영역이 포함되지 않는다." in provider.last_text
        assert "'전체'는 여러 색이 섞인 혼합 overview 목록" in provider.last_text
        assert provider.last_pil_size == _policy_crop_size(process, image, _POLICY_RIGHT_CARD_LIST_RATIOS)

    def test_policy_calibration_verification_fails_for_wrong_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "경제", "x": 760, "y": 100}],
            eligible_tabs_queue=["경제"],
            slot_inventory=[{"slot_id": "economic_1", "slot_type": "경제", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["경제"],
            calibration_pending_tabs=["경제"],
        )
        memory.begin_stage("calibrate_tabs")
        provider = FakeProvider([json.dumps({"match": False, "observed_tab": "외교", "reason": "초록 카드"})])

        verified = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is False
        assert memory.policy_state.last_tab_check_result.startswith("경제->외교:fail")

    def test_distinct_tab_failures_restart_full_recalibration(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100, "confirmed": True},
                {"tab_name": "경제", "x": 760, "y": 100, "confirmed": True},
                {"tab_name": "외교", "x": 820, "y": 100, "confirmed": True},
                {"tab_name": "와일드카드", "x": 880, "y": 100, "confirmed": True},
                {"tab_name": "암흑", "x": 940, "y": 100, "confirmed": True},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.record_policy_failed_tab("군사")
        memory.begin_stage("click_next_tab")

        resolution = process.handle_no_progress(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "bootstrap_tabs"
        assert memory.policy_state.tab_positions == {}
        assert memory.policy_state.eligible_tabs_queue == ["군사", "경제", "외교", "와일드카드", "암흑"]
        assert memory.policy_state.completed_tabs == ["군사"]
        assert memory.get_policy_current_tab_name() == "경제"
        assert memory.policy_state.last_event == "distinct tab failures -> full recalibration"

    def test_generic_fallback_no_progress_restarts_same_policy_primitive(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[
                {"tab_name": "군사", "x": 700, "y": 100},
                {"tab_name": "경제", "x": 760, "y": 100},
                {"tab_name": "외교", "x": 820, "y": 100},
                {"tab_name": "와일드카드", "x": 880, "y": 100},
                {"tab_name": "암흑", "x": 940, "y": 100},
            ],
            eligible_tabs_queue=["군사", "경제", "외교", "와일드카드", "암흑"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
        )
        memory.begin_stage("generic_fallback")
        memory.set_fallback_return_stage("plan_current_tab", "plan_current_tab:군사")
        provider = FakeProvider([])

        resolution = process.handle_no_progress(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            last_action=type("A", (), {"action": "click"})(),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert resolution.handled is True
        assert memory.current_stage == "bootstrap_tabs"
        assert memory.has_policy_bootstrap() is False
        assert memory.is_policy_entry_done() is True
