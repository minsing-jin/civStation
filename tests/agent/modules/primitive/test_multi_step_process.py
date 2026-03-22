"""Unit tests for class-based multi-step processes."""

import json

from PIL import Image

from computer_use_test.agent.modules.memory.short_term_memory import ChoiceCandidate, ShortTermMemory
from computer_use_test.agent.modules.primitive.multi_step_process import (
    _POLICY_RIGHT_CARD_LIST_RATIOS,
    _POLICY_RIGHT_TAB_BAR_RATIOS,
    ObservationBundle,
    StageTransition,
    get_multi_step_process,
)
from computer_use_test.agent.modules.router.primitive_registry import PRIMITIVE_REGISTRY, get_primitive_prompt
from computer_use_test.utils.llm_provider.base import BaseVLMProvider, VLMResponse
from computer_use_test.utils.llm_provider.parser import AgentAction


class FakeProvider(BaseVLMProvider):
    def __init__(self, responses: list[str]):
        super().__init__(api_key=None, model="fake", resize_for_vlm=False)
        self.responses = list(responses)
        self.last_text = ""
        self.last_pil_size = None
        self.last_max_tokens = None
        self.last_use_thinking = None

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=8192, use_thinking=True) -> VLMResponse:
        self.last_max_tokens = max_tokens
        self.last_use_thinking = use_thinking
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
    def test_religion_process_moves_to_hover_stage_before_scrolling(self):
        process = get_multi_step_process(
            "religion_primitive",
            "초록색 '종교관 세우기' 버튼 클릭 완료 시 task_status='complete'.",
        )
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("observe_choices")

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "신성한 불꽃"}],
                end_of_list=False,
                scroll_anchor={"x": 160, "y": 520, "left": 80, "top": 160, "right": 320, "bottom": 900},
            ),
        )

        assert action is not None
        assert action.action == "move"
        assert action.y == 520
        assert action.x > 160
        assert memory.current_stage == "hover_scroll_anchor"

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
        assert action.x >= 820
        assert action.x <= 850
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

    def test_city_production_observer_prompt_does_not_echo_full_choice_catalog(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.begin_stage("observe_choices")
        memory.remember_choices(
            [
                {"id": "ghost_unit", "label": "유령전사"},
                {"id": "ghost_building", "label": "유령기념비"},
            ],
            end_of_list=False,
        )

        prompt = process.observer.build_prompt("city_production_primitive", memory, normalizing_range=1000)

        assert "현재 stage: observe_choices" in prompt
        assert "[choice_catalog]" not in prompt
        assert "유령전사" not in prompt
        assert "유령기념비" not in prompt

    def test_decide_from_memory_prompt_includes_earliest_candidates(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider([json.dumps({"best_option_id": "후보0", "reason": "처음 본 후보도 고려"})])

        for idx in range(10):
            memory.remember_choices(
                [{"id": f"후보{idx}", "label": f"후보{idx}"}],
                end_of_list=idx == 9,
                scroll_direction="down",
            )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="문화 승리",
        )

        assert decided is True
        assert "id=후보0" in provider.last_text
        assert "후보0" in provider.last_text
        assert "후보9" in provider.last_text

    def test_decide_from_memory_uses_best_option_id_not_label_text(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "cand_old", "label": "알렉산드리아 도서관"},
                {"id": "cand_new", "label": "개척자"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "cand_old",
                        "best_option_label": "알렉산드리아 도서관(불가사의)",
                        "reason": "핵심 선택",
                    }
                )
            ]
        )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="[사용자 최우선 지시] 이번 task에서는 캠퍼스 먼저 지어\n\n과학 승리",
        )

        assert decided is True
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().id == "cand_old"

    def test_decide_from_memory_scales_max_tokens_with_candidate_count(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "후보0",
                        "best_option_label": "후보0",
                        "reason": "긴 목록",
                    }
                )
            ]
        )

        for idx in range(26):
            memory.remember_choices(
                [{"id": f"후보{idx}", "label": f"후보{idx}"}],
                end_of_list=idx == 25,
                scroll_direction="down",
            )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="문화 승리",
        )

        assert decided is True
        assert provider.last_max_tokens is not None
        assert provider.last_max_tokens > 512

    def test_decide_from_memory_calls_provider_without_thinking(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [{"id": "cand1", "label": "기념비"}],
            end_of_list=True,
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "cand1",
                        "best_option_label": "기념비",
                        "reason": "빠른 선택",
                    }
                )
            ]
        )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="[사용자 최우선 지시] 이번 task에서는 캠퍼스 먼저 지어\n\n과학 승리",
        )

        assert decided is True
        assert provider.last_use_thinking is False

    def test_decide_from_memory_applies_task_hitl_when_one_candidate_matches(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "campus", "label": "캠퍼스"},
                {"id": "market", "label": "시장"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.set_task_hitl_directive("이번 task에서는 캠퍼스 먼저 지어")
        provider = FakeProvider([])

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="[사용자 최우선 지시] 이번 task에서는 캠퍼스 먼저 지어\n\n과학 승리",
        )

        assert decided is True
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().id == "campus"
        assert memory.choice_catalog.best_option_reason == "task HITL matched candidate '캠퍼스'"
        assert provider.last_text == ""

    def test_decide_from_memory_ignores_impossible_task_hitl_and_falls_back_to_strategy(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "market", "label": "시장"},
                {"id": "settler", "label": "개척자"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.set_task_hitl_directive("이번 task에서는 캠퍼스 먼저 지어")
        provider = FakeProvider([json.dumps({"best_option_id": "market", "reason": "전략상 금 수급 우선"})])

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="[사용자 최우선 지시] 이번 task에서는 캠퍼스 먼저 지어\n\n과학 승리",
        )

        assert decided is True
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().id == "market"
        assert "상위 전략" in provider.last_text
        assert "캠퍼스" not in provider.last_text

    def test_governor_decide_from_memory_applies_task_hitl_before_vlm(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "pingala", "label": "핑갈라", "note": "진급_가능"},
                {"id": "magnus", "label": "마그누스", "note": "임명_가능"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.set_task_hitl_directive("이번 task에서는 핑갈라 진급")
        provider = FakeProvider([])

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="과학 승리",
        )

        assert decided is True
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().id == "pingala"
        assert memory.branch == "governor_promote"
        assert memory.current_stage == "governor_promote_click"
        assert provider.last_text == ""


class TestPromptUpdates:
    def test_unit_ops_prompt_forbids_moving_onto_occupied_tiles(self):
        prompt = get_primitive_prompt("unit_ops_primitive")
        assert "빈 타일로만" in prompt
        assert "다른 유닛이 서 있는 타일" in prompt
        assert "공격일 때만" in prompt

    def test_unit_ops_prompt_handles_great_person_white_tile_activation(self):
        prompt = get_primitive_prompt("unit_ops_primitive")
        assert "위대한 위인" in prompt
        assert "명령이 필요한" in prompt
        assert "하얀색 타일" in prompt
        assert "오른쪽 아래" in prompt
        assert "사람 흉상" in prompt

    def test_governor_prompt_stage_note_driven(self):
        prompt = get_primitive_prompt("governor_primitive")
        assert "stage note" in prompt
        assert "확정" in prompt
        assert "배정" in prompt

    def test_governor_registry_criteria_includes_lower_right_governor_entry(self):
        criteria = PRIMITIVE_REGISTRY["governor_primitive"]["criteria"]
        assert "우하단" in criteria
        assert "총독 타이틀" in criteria
        assert "펜" in criteria

    def test_religion_registry_criteria_includes_lower_right_angel_icon_entry(self):
        criteria = PRIMITIVE_REGISTRY["religion_primitive"]["criteria"]
        assert "우하단" in criteria
        assert "천사" in criteria
        assert "원형" in criteria

    def test_governor_prompt_contains_essential_rules(self):
        prompt = get_primitive_prompt("governor_primitive")
        assert "확정" in prompt
        assert "비활성" in prompt
        assert "배정" in prompt

    def test_religion_prompt_mentions_angel_icon_entry_and_esc_exit(self):
        prompt = get_primitive_prompt("religion_primitive")
        assert "천사 문양" in prompt
        assert "종교관 준비" in prompt
        assert "Esc" in prompt
        assert 'task_status="complete"' in prompt

    def test_policy_prompt_contains_two_entry_branches(self):
        prompt = get_primitive_prompt("policy_primitive")
        assert "사회제도 완성" in prompt
        assert "새 정부 선택" in prompt
        assert "모든 정책 배정" in prompt
        assert "실패한 탭 하나만 다시 찾아 cached 좌표를 수정한다" in prompt
        assert "'전체' 탭은 초기 overview 상태" in prompt
        assert "혼합 overview 목록이면 '전체' 상태" in prompt

    def test_city_production_registry_allows_retry_heavy_flow(self):
        assert PRIMITIVE_REGISTRY["city_production_primitive"]["max_steps"] >= 18

    def test_religion_registry_allows_hover_scroll_flow_budget(self):
        assert PRIMITIVE_REGISTRY["religion_primitive"]["max_steps"] >= 18

    def test_popup_prompt_handles_policy_change_popup(self):
        prompt = get_primitive_prompt("popup_primitive")
        assert "정책변경" in prompt

    def test_popup_prompt_handles_hero_discovery_continue_button(self):
        prompt = get_primitive_prompt("popup_primitive")
        assert "발견된 영웅" in prompt
        assert "영웅을 보라" in prompt
        assert "계속" in prompt
        assert "클릭" in prompt

    def test_popup_prompt_handles_secret_society_discovery_with_continue_button(self):
        prompt = get_primitive_prompt("popup_primitive")
        assert "결사 발견" in prompt
        assert "계속" in prompt
        assert "총독화면으로 이동" not in prompt

    def test_popup_prompt_does_not_own_lower_right_screen_entry_buttons(self):
        prompt = get_primitive_prompt("popup_primitive")
        assert "우하단 '연구 선택'" not in prompt
        assert "우하단 '생산 품목'" not in prompt
        assert "우하단 '사회 제도 선택'" not in prompt

    def test_popup_registry_criteria_only_keeps_next_turn_lower_right_button(self):
        criteria = PRIMITIVE_REGISTRY["popup_primitive"]["criteria"]
        assert "다음 턴" in criteria
        assert "연구 선택" not in criteria
        assert "생산 품목" not in criteria
        assert "사회 제도 선택" not in criteria


class TestEntryGatedProcesses:
    def test_religion_process_factory_returns_religion_process(self):
        process = get_multi_step_process("religion_primitive", "")

        assert type(process).__name__ == "ReligionProcess"

    def test_religion_process_starts_in_entry_stage(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)

        process.initialize(memory)

        assert memory.current_stage == "religion_entry"

    def test_religion_process_uses_entry_press_when_labeled_button_is_ready(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "religion_screen_ready": False,
                        "entry_button_visible": True,
                        "prep_popup_visible": False,
                        "angel_button_visible": True,
                        "reasoning": "우하단 '종교관 선택' 버튼이 보여 enter로 진입 가능",
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
        assert action.action == "press"
        assert action.key == "enter"
        assert memory.current_stage == "religion_entry"

    def test_religion_entry_click_fallback_scales_with_runtime_normalizing_range(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "religion_screen_ready": False,
                        "entry_button_visible": False,
                        "prep_popup_visible": False,
                        "angel_button_visible": True,
                        "reasoning": "우하단 천사 문양 원형 버튼만 보여 click fallback 필요",
                    }
                )
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "click"
        assert action.x == 9300
        assert action.y == 8850

    def test_religion_entry_press_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="press", key="enter"),
        )

        assert should_verify is True

    def test_religion_entry_press_semantic_verify_promotes_to_observe_choices(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "religion_screen_ready": True,
                        "entry_button_visible": False,
                        "prep_popup_visible": False,
                        "angel_button_visible": False,
                        "reasoning": "왼쪽 종교관 목록 팝업이 열림",
                    }
                )
            ]
        )
        action = AgentAction(action="press", key="enter")

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            action,
        )

        assert verify.handled is True
        assert verify.passed is True
        process.on_action_success(memory, action)
        assert "religion_entry_done" in memory.completed_substeps
        assert memory.current_stage == "observe_choices"

    def test_religion_select_stage_promotes_to_confirm_when_confirm_button_is_ready(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("select_from_memory")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "followup_state": "confirm",
                        "belief_selected": True,
                        "confirm_button_visible": True,
                        "confirm_button_enabled": True,
                        "prep_popup_visible": False,
                        "angel_button_visible": False,
                        "reason": "선택된 종교관과 초록색 확정 버튼이 모두 보임",
                    }
                ),
                (
                    '{"action":"click","x":500,"y":790,"end_x":0,"end_y":0,"scroll_amount":0,'
                    '"button":"left","key":"","text":"","reasoning":"초록색 종교관 세우기 버튼 클릭",'
                    '"task_status":"in_progress"}'
                ),
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
        assert memory.current_stage == "religion_confirm"

    def test_religion_select_stage_returns_escape_when_prep_popup_is_visible(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("select_from_memory")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "followup_state": "exit",
                        "belief_selected": True,
                        "confirm_button_visible": False,
                        "confirm_button_enabled": False,
                        "prep_popup_visible": True,
                        "angel_button_visible": False,
                        "reason": "종교창시중 팝업이 떠 있어 Esc로 닫아야 함",
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
        assert action.action == "press"
        assert action.key == "escape"
        assert action.task_status == "complete"
        assert memory.current_stage == "religion_exit"

    def test_religion_select_stage_scrolls_to_reveal_hidden_confirm_button(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("select_from_memory")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "followup_state": "select",
                        "belief_selected": True,
                        "confirm_button_visible": False,
                        "confirm_button_enabled": False,
                        "prep_popup_visible": False,
                        "angel_button_visible": False,
                        "reason": "선택된 종교관은 보이지만 확정 버튼이 아직 아래에 가려짐",
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
        assert action.action == "scroll"
        assert action.scroll_amount == -120
        assert memory.current_stage == "select_from_memory"

    def test_religion_confirm_click_semantic_verify_promotes_to_exit_stage(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("religion_confirm")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "followup_state": "exit",
                        "belief_selected": True,
                        "confirm_button_visible": False,
                        "confirm_button_enabled": False,
                        "prep_popup_visible": True,
                        "angel_button_visible": False,
                        "reason": "확정 후 종교창시중 팝업이 뜸",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="click", x=500, y=790),
        )

        assert verify.handled is True
        assert verify.passed is True
        assert memory.current_stage == "religion_exit"

    def test_religion_completion_verifier_accepts_closed_prep_popup_and_non_angel_button(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("religion_exit")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "prep_popup_visible": False,
                        "angel_button_visible": False,
                        "complete": True,
                        "reason": "준비 팝업이 닫혔고 우하단 버튼이 더 이상 천사 문양이 아님",
                    }
                )
            ]
        )

        verification = process.verify_completion(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
        )

        assert verification.complete is True

    def test_religion_completion_verifier_rejects_when_angel_button_still_visible(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("religion_exit")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "prep_popup_visible": False,
                        "angel_button_visible": True,
                        "complete": False,
                        "reason": "우하단 버튼이 아직 천사 문양이라 완료 상태가 아님",
                    }
                )
            ]
        )

        verification = process.verify_completion(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
        )

        assert verification.complete is False

    def test_religion_completion_verifier_accepts_next_turn_screen_without_complete_flag(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("religion_exit")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "religion_screen_ready": False,
                        "entry_button_visible": False,
                        "prep_popup_visible": False,
                        "angel_button_visible": False,
                        "complete": False,
                        "reason": (
                            "현재 우하단 버튼은 '다음 턴' 상태이며, "
                            "종교관 선택과 관련된 팝업이나 버튼이 화면에 나타나지 않음"
                        ),
                    }
                )
            ]
        )

        verification = process.verify_completion(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
        )

        assert verification.complete is True

    def test_religion_default_anchor_targets_left_side_list_panel(self):
        process = get_multi_step_process("religion_primitive", "")

        anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert anchor["x"] < 450
        assert anchor["right"] < 450
        assert anchor["right"] > anchor["left"]

    def test_religion_hover_success_transitions_to_small_scroll_stage(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("hover_scroll_anchor")
        memory.remember_choices(
            [{"label": "신성한 불꽃"}],
            end_of_list=False,
            scroll_anchor={"x": 120, "y": 520, "left": 60, "top": 160, "right": 300, "bottom": 900},
        )

        process.on_action_success(
            memory,
            process.resolve_action(AgentAction(action="move", x=120, y=520), memory),
        )
        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert memory.current_stage == "scroll_down_for_hidden_choices"
        assert action is not None
        assert action.action == "scroll"
        assert action.scroll_amount == -120
        assert action.x > 120
        assert action.y == 520

    def test_religion_repairs_invalid_memory_anchor_before_hover_scroll(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("hover_scroll_anchor")
        memory.remember_choices(
            [{"label": "신성한 불꽃"}],
            end_of_list=False,
            scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
        )

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        expected_anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert action is not None
        assert action.action == "move"
        assert (action.x, action.y) == (expected_anchor["x"], expected_anchor["y"])
        assert memory.get_scroll_anchor() is not None
        assert memory.get_scroll_anchor().x == expected_anchor["x"]

    def test_religion_scan_does_not_complete_after_first_scrolled_observation_finds_no_new_choices(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "신성한 불꽃"}, {"label": "종교적 정착지"}],
                end_of_list=False,
                scroll_anchor={"x": 170, "y": 520, "left": 80, "top": 160, "right": 320, "bottom": 900},
            ),
        )
        assert first_action is not None
        process.on_action_success(memory, first_action)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert scroll_action is not None
        process.on_action_success(memory, scroll_action)

        second_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "신성한 불꽃"}, {"label": "종교적 정착지"}],
                end_of_list=False,
                scroll_anchor={"x": 170, "y": 520, "left": 80, "top": 160, "right": 320, "bottom": 900},
            ),
        )

        assert second_action is not None
        assert second_action.action == "move"
        assert memory.choice_catalog.end_reached is False
        assert memory.current_stage == "hover_scroll_anchor"

    def test_religion_scroll_verification_rejects_unchanged_visible_options(self):
        process = get_multi_step_process("religion_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("religion_primitive", enable_choice_catalog=True)
        memory.mark_substep("religion_entry_done")
        memory.begin_stage("scroll_down_for_hidden_choices")
        memory.remember_choices(
            [
                {"id": "fire", "label": "신성한 불꽃"},
                {"id": "settlements", "label": "종교적 정착지"},
                {"id": "city_patron", "label": "도시의 수호자", "disabled": True},
            ],
            end_of_list=False,
            scroll_anchor={"x": 200, "y": 520, "left": 80, "top": 160, "right": 320, "bottom": 900},
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [
                            {"id": "fire", "label": "신성한 불꽃"},
                            {"id": "settlements", "label": "종교적 정착지"},
                            {"id": "city_patron", "label": "도시의 수호자", "disabled": True},
                        ],
                        "end_of_list": False,
                        "scroll_anchor": {
                            "x": 200,
                            "y": 520,
                            "left": 80,
                            "top": 160,
                            "right": 320,
                            "bottom": 900,
                        },
                        "reasoning": "같은 목록이 그대로 보임",
                    }
                )
            ]
        )

        verification = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(action="scroll", x=220, y=520, scroll_amount=-120, task_status="in_progress"),
        )

        assert verification.handled is True
        assert verification.passed is False
        assert "같은 선택지" in verification.reason

    def test_governor_process_factory_returns_governor_process(self):
        process = get_multi_step_process("governor_primitive", "")

        assert type(process).__name__ == "GovernorProcess"

    def test_governor_process_uses_entry_action_before_screen_is_ready(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive")
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "governor_mode": "notification",
                        "governor_screen_ready": False,
                        "notification_visible": True,
                        "confirm_enabled": False,
                        "assign_enabled": False,
                        "left_city_popup_visible": False,
                        "reasoning": "우하단 총독 타이틀 버튼만 보임",
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
        assert action.action == "press"
        assert action.key == "enter"
        assert memory.current_stage == "governor_entry"

    def test_governor_entry_press_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="press", key="enter"),
        )

        assert should_verify is True

    def test_governor_entry_press_semantic_verify_marks_entry_done(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "governor_mode": "overview",
                        "governor_screen_ready": True,
                        "notification_visible": False,
                        "reasoning": "총독 카드와 진급 버튼이 보임",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="press", key="enter"),
        )

        assert verify.handled is True
        assert verify.passed is True
        assert "governor_entry_done" in memory.completed_substeps

    def test_governor_hover_move_semantic_verify_succeeds_without_ui_change(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.begin_stage("hover_scroll_anchor")

        verify = process.verify_action_success(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="move", x=500, y=520),
        )

        assert verify.handled is True
        assert verify.passed is True

    def test_governor_scroll_semantic_verify_detects_list_progress(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.begin_stage("scroll_down_for_hidden_choices")
        memory.remember_choices(
            [{"id": "pingala", "label": "핑갈라", "note": "진급_가능"}],
            end_of_list=False,
            scroll_anchor={"x": 500, "y": 520, "left": 250, "top": 120, "right": 760, "bottom": 920},
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"id": "victor", "label": "빅토르", "note": "진급_가능"}],
                        "end_of_list": False,
                        "scroll_anchor": {
                            "x": 500,
                            "y": 520,
                            "left": 250,
                            "top": 120,
                            "right": 760,
                            "bottom": 920,
                        },
                        "reasoning": "다른 총독 카드가 새로 보임",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="scroll", x=500, y=520, scroll_amount=-120),
        )

        assert verify.handled is True
        assert verify.passed is True

    def test_governor_promote_select_click_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_promote")
        memory.begin_stage("governor_promote_select")

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="click", x=500, y=500),
        )

        assert should_verify is True

    def test_governor_promote_select_semantic_verify_accepts_green_confirm_button(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_promote")
        memory.begin_stage("governor_promote_select")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "confirm_enabled": True,
                        "reasoning": "하단 버튼이 돌아가기에서 초록색 확정으로 바뀌어 누를 수 있음",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="click", x=500, y=500),
        )

        assert verify.handled is True
        assert verify.passed is True

    def test_governor_appoint_city_click_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city")

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="click", x=500, y=500),
        )

        assert should_verify is True

    def test_governor_appoint_city_semantic_verify_rejects_city_with_governor_portrait(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "valid_unassigned_city_selected": False,
                        "reasoning": "선택된 도시 왼쪽 동그라미에 총독 얼굴 아이콘이 보여 이미 배정된 도시임",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="click", x=500, y=500),
        )

        assert verify.handled is True
        assert verify.passed is False

    def test_governor_appoint_city_observer_prompt_describes_empty_circle_rule(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_observe")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [
                            {"id": "lugdunum", "label": "루구두눔", "disabled": False, "note": "미배정"}
                        ],
                        "end_of_list": True,
                        "scroll_anchor": None,
                        "reasoning": "왼쪽 도시 목록에서 미배정 도시를 확인함",
                    }
                )
            ]
        )

        observation = process.observe(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
        )

        assert observation is not None
        assert "왼쪽 팝업창" in provider.last_text
        assert "도시 이름 왼쪽 동그라미" in provider.last_text
        assert "총독 얼굴" in provider.last_text
        assert "비어 있으면" in provider.last_text

    def test_governor_appoint_branch_observes_city_blocks_before_clicking_city(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_click")

        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))

        assert memory.current_stage == "governor_appoint_city_observe"

        result = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "paris", "label": "파리", "disabled": True, "note": "총독 얼굴 보임 / 이미 배정됨"},
                    {"id": "lugdunum", "label": "루구두눔", "disabled": False, "note": "빈 동그라미 / 미배정"},
                ],
                end_of_list=True,
            ),
        )

        assert result is None
        assert memory.current_stage == "governor_appoint_city_decide"

    def test_governor_appoint_city_observation_scrolls_when_popup_has_more_cities(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_observe")

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "paris", "label": "파리", "disabled": False, "note": "빈 동그라미 / 미배정"},
                ],
                end_of_list=False,
                scroll_anchor={"x": 220, "y": 520, "left": 80, "top": 160, "right": 360, "bottom": 900},
            ),
        )

        assert action is not None
        assert action.action == "move"
        assert memory.current_stage == "governor_appoint_city_hover_scroll_anchor"

    def test_governor_appoint_city_decision_prompt_only_lists_unassigned_cities(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_observe")
        process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "paris", "label": "파리", "disabled": True, "note": "총독 얼굴 보임 / 이미 배정됨"},
                    {"id": "lugdunum", "label": "루구두눔", "disabled": False, "note": "빈 동그라미 / 미배정"},
                ],
                end_of_list=True,
            ),
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "lugdunum",
                        "reason": "미배정 도시 중 과학 전략과 가장 잘 맞음",
                    }
                )
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 산출이 높은 도시에 총독 배정",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, StageTransition)
        assert action.stage == "governor_appoint_city"
        assert memory.current_stage == "governor_appoint_city"
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().id == "lugdunum"
        assert "루구두눔" in provider.last_text
        assert "파리" not in provider.last_text

    def test_governor_appoint_city_scroll_semantic_verify_detects_city_list_progress(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_scroll_down")
        memory.remember_choices(
            [{"id": "paris", "label": "파리", "note": "미배정"}],
            end_of_list=False,
            scroll_anchor={"x": 220, "y": 520, "left": 80, "top": 160, "right": 360, "bottom": 900},
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"id": "lugdunum", "label": "루구두눔", "note": "미배정"}],
                        "end_of_list": False,
                        "scroll_anchor": {
                            "x": 220,
                            "y": 520,
                            "left": 80,
                            "top": 160,
                            "right": 360,
                            "bottom": 900,
                        },
                        "reasoning": "왼쪽 도시 목록에서 새 도시가 보임",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="scroll", x=220, y=520, scroll_amount=-120),
        )

        assert verify.handled is True
        assert verify.passed is True

    def test_governor_appoint_city_decision_restores_hidden_city_after_scroll_scan(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_observe")

        first = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "paris", "label": "파리", "disabled": False, "note": "빈 동그라미 / 미배정"},
                ],
                end_of_list=False,
                scroll_anchor={"x": 220, "y": 520, "left": 80, "top": 160, "right": 360, "bottom": 900},
            ),
        )
        assert first is not None
        process.on_action_success(memory, first)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert scroll_action is not None
        assert scroll_action.action == "scroll"
        process.on_action_success(memory, scroll_action)

        second = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "lugdunum", "label": "루구두눔", "disabled": False, "note": "빈 동그라미 / 미배정"},
                ],
                end_of_list=True,
                scroll_anchor={"x": 220, "y": 520, "left": 80, "top": 160, "right": 360, "bottom": 900},
            ),
        )
        assert second is None
        assert memory.current_stage == "governor_appoint_city_decide"

        provider = FakeProvider([json.dumps({"best_option_id": "paris", "reason": "과학 도시"})])
        transition = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(transition, StageTransition)
        assert transition.stage == "governor_appoint_city_restore_hover_scroll_anchor"
        assert memory.current_stage == "governor_appoint_city_restore_hover_scroll_anchor"

    def test_governor_appoint_branch_defers_memory_decision_to_branch_specific_stage(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city_decide")

        assert process.should_auto_decide_from_memory(memory) is False

    def test_governor_appoint_city_no_progress_reobserves_without_generic_fallback(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_city")

        first = process.handle_no_progress(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            last_action=AgentAction(action="click", x=420, y=360, task_status="in_progress"),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert first.handled is True
        assert first.reroute is False
        assert memory.current_stage == "governor_appoint_city_observe"
        assert memory.fallback_return_stage == ""

    def test_governor_entry_press_success_transitions_directly_to_observe_choices(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "governor_mode": "overview",
                        "governor_screen_ready": True,
                        "notification_visible": False,
                        "reasoning": "총독 카드와 진급 버튼이 보임",
                    }
                )
            ]
        )
        action = AgentAction(action="press", key="enter")

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            action,
        )
        assert verify.passed is True

        process.on_action_success(memory, action)

        assert memory.current_stage == "observe_choices"

    def test_governor_promote_branch_transitions_via_on_action_success(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_promote")
        memory.begin_stage("governor_promote_click")

        # promote_click -> promote_select
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_promote_select"

        # promote_select -> promote_confirm
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_promote_confirm"

        # promote_confirm -> promote_popup
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_promote_popup"

        # promote_popup -> exit_esc1
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_exit_esc1"

        # exit_esc1 -> exit_esc2
        process.on_action_success(memory, AgentAction(action="press", key="escape"))
        assert memory.current_stage == "governor_exit_esc2"

    def test_governor_appoint_branch_transitions_via_on_action_success(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_appoint_click")

        # appoint_click -> appoint_city_observe
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_appoint_city_observe"

        # appoint_city -> appoint_confirm
        memory.begin_stage("governor_appoint_city")
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_appoint_confirm"

        # appoint_confirm -> exit_esc1
        memory.begin_stage("governor_appoint_confirm")
        process.on_action_success(memory, AgentAction(action="click", x=500, y=500))
        assert memory.current_stage == "governor_exit_esc1"

    def test_governor_promote_esc_stages_are_deterministic(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_promote")
        memory.begin_stage("governor_exit_esc1")
        provider = FakeProvider([])

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
        assert action.action == "press"
        assert action.key == "escape"
        assert action.task_status == "in_progress"

    def test_governor_appoint_exit_esc_stages_are_deterministic(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_exit_esc1")

        action1 = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action1 is not None
        assert action1.action == "press"
        assert action1.key == "escape"
        assert action1.task_status == "in_progress"

        memory.begin_stage("governor_exit_esc2")
        action2 = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action2 is not None
        assert action2.action == "press"
        assert action2.key == "escape"
        assert action2.task_status == "complete"

    def test_governor_appoint_exit_terminal_state_completes_without_vlm_verification(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.set_branch("governor_appoint")
        memory.begin_stage("governor_exit_esc2")

        verification = process.verify_completion(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
        )

        assert verification.complete is True

    def test_governor_observation_requires_one_downward_scroll_before_choose_from_memory(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.begin_stage("observe_choices")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"id": "pingala", "label": "핑갈라", "note": "진급_가능"}],
                end_of_list=True,
                scroll_anchor={"x": 500, "y": 520, "left": 250, "top": 120, "right": 760, "bottom": 920},
            ),
        )

        assert first_action is not None
        assert first_action.action == "move"
        assert memory.current_stage == "hover_scroll_anchor"
        assert memory.choice_catalog.end_reached is False

    def test_governor_observation_chooses_from_memory_after_one_downward_scroll_and_no_new_candidates(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.begin_stage("observe_choices")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"id": "pingala", "label": "핑갈라", "note": "진급_가능"}],
                end_of_list=True,
                scroll_anchor={"x": 500, "y": 520, "left": 250, "top": 120, "right": 760, "bottom": 920},
            ),
        )
        assert first_action is not None
        process.on_action_success(memory, first_action)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert scroll_action is not None
        assert scroll_action.action == "scroll"
        process.on_action_success(memory, scroll_action)

        second_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"id": "pingala", "label": "핑갈라", "note": "진급_가능"}],
                end_of_list=False,
                scroll_anchor={"x": 500, "y": 520, "left": 250, "top": 120, "right": 760, "bottom": 920},
            ),
        )

        assert second_action is None
        assert memory.current_stage == "choose_from_memory"
        assert "full_scan_complete" in memory.completed_substeps

    def test_governor_secret_society_best_choice_with_appoint_uses_secret_branch(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "hermes_secret", "label": "허미즈", "note": "비밀결사 임명_가능"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "hermes_secret",
                        "action_type": "appoint",
                        "reason": "비밀결사 총독 임명",
                    }
                )
            ]
        )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="과학 승리",
        )

        assert decided is True
        assert memory.branch == "governor_secret_society"
        assert memory.current_stage == "governor_secret_society_appoint_click"

    def test_governor_secret_society_best_choice_with_promote_stays_on_promote_branch(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "hermes_secret", "label": "허미즈", "note": "비밀결사 진급_가능"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "best_option_id": "hermes_secret",
                        "action_type": "promote",
                        "reason": "비밀결사 총독 진급",
                    }
                )
            ]
        )

        decided = process.decide_from_memory(
            provider,
            memory,
            high_level_strategy="과학 승리",
        )

        assert decided is True
        assert memory.branch == "governor_promote"
        assert memory.current_stage == "governor_promote_click"

    def test_governor_secret_society_branch_merges_into_promote_when_green_promote_is_visible(self):
        process = get_multi_step_process("governor_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("governor_primitive", enable_choice_catalog=True)
        memory.mark_substep("governor_entry_done")
        memory.choice_catalog.candidates["hermes_secret"] = ChoiceCandidate(
            id="hermes_secret",
            label="허미즈",
            metadata={"note": "비밀결사 임명_가능"},
        )
        memory.set_best_choice(option_id="hermes_secret", reason="비밀결사 선택")
        memory.set_branch("governor_secret_society")
        memory.begin_stage("governor_secret_society_post_appoint_check")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "promote_visible": True,
                        "reasoning": "허미즈 카드에 초록색 진급 버튼이 활성화됨",
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

        assert isinstance(action, StageTransition)
        assert action.stage == "governor_promote_click"
        assert memory.branch == "governor_promote"
        assert memory.current_stage == "governor_promote_click"

    def test_governor_registry_uses_observation_assisted(self):
        assert PRIMITIVE_REGISTRY["governor_primitive"]["process_kind"] == "observation_assisted"
        assert PRIMITIVE_REGISTRY["governor_primitive"]["max_steps"] >= 20

    def test_voting_process_factory_returns_voting_process(self):
        process = get_multi_step_process("voting_primitive", "")

        assert type(process).__name__ == "VotingProcess"

    def test_voting_process_uses_vote_start_button_from_welcome_popup_entry(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "voting_mode": "welcome_popup",
                        "voting_screen_ready": False,
                        "welcome_popup_visible": True,
                        "globe_button_visible": False,
                        "reasoning": "세계 의회에 오신 것을 환영합니다 팝업과 투표시작 버튼이 보임",
                    }
                ),
                json.dumps(
                    {
                        "action": "click",
                        "x": 620,
                        "y": 710,
                        "reasoning": "투표시작 버튼 클릭",
                        "task_status": "in_progress",
                    }
                ),
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

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert memory.current_stage == "vote_entry"

    def test_voting_process_uses_lower_right_globe_button_for_entry(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "voting_mode": "globe_button",
                        "voting_screen_ready": False,
                        "welcome_popup_visible": False,
                        "globe_button_visible": True,
                        "reasoning": "오른쪽 아래 세계의회 지구본 버튼만 보임",
                    }
                ),
                json.dumps(
                    {
                        "action": "click",
                        "x": 930,
                        "y": 885,
                        "reasoning": "오른쪽 아래 지구본 버튼 클릭",
                        "task_status": "in_progress",
                    }
                ),
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

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert memory.current_stage == "vote_entry"

    def test_voting_entry_transitions_to_scan_when_world_congress_screen_is_ready(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "voting_mode": "agenda_screen",
                        "voting_screen_ready": True,
                        "welcome_popup_visible": False,
                        "globe_button_visible": False,
                        "reasoning": "합의안 block이 보여 실제 투표 화면이 열림",
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

        assert isinstance(action, StageTransition)
        assert action.stage == "vote_scan_agendas"
        assert memory.current_stage == "vote_scan_agendas"

    def test_voting_scan_end_transitions_to_select_agenda(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        process.initialize(memory)
        memory.mark_substep("voting_entry_done")
        memory.begin_stage("vote_scan_agendas")

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[
                    {"id": "luxury_ban", "label": "사치 자원 금지"},
                    {"id": "trade_bonus", "label": "교역 보너스"},
                ],
                end_of_list=True,
                scroll_anchor={"x": 520, "y": 510, "left": 280, "top": 120, "right": 760, "bottom": 920},
            ),
        )

        assert action is None
        assert memory.current_stage == "vote_select_agenda"
        assert "full_scan_complete" in memory.completed_substeps

    def test_voting_selects_visible_pending_agenda_before_resolution_stage(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        process.initialize(memory)
        memory.remember_choices(
            [
                {"id": "luxury_ban", "label": "사치 자원 금지"},
                {"id": "trade_bonus", "label": "교역 보너스", "selected": True},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.begin_stage("vote_select_agenda")

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, StageTransition)
        assert action.stage == "vote_choose_resolution"
        assert memory.current_stage == "vote_choose_resolution"
        assert memory.voting_state.current_agenda_id == "luxury_ban"
        assert memory.voting_state.current_agenda_label == "사치 자원 금지"

    def test_voting_direction_stage_emits_repeated_click_bundle_once(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.choice_catalog.candidates["luxury_ban"] = ChoiceCandidate(
            id="luxury_ban",
            label="사치 자원 금지",
        )
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.set_current_voting_agenda(option_id="luxury_ban")
        memory.begin_stage("vote_choose_direction")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "x": 540,
                        "y": 430,
                        "repeat_count": 3,
                        "selection": "upvote",
                        "reason": "필요한 표만큼 같은 찬성 버튼을 여러 번 눌러야 함",
                    }
                )
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
        assert len(actions) == 3
        assert all(action.action == "click" for action in actions)
        assert all(action.task_status == "in_progress" for action in actions)
        process.on_actions_success(memory, actions)
        assert memory.current_stage == "vote_hover_left_for_target"
        assert memory.voting_state.selected_vote_direction == "upvote"

    def test_voting_left_hover_runs_before_target_selection(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.begin_stage("vote_hover_left_for_target")

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, AgentAction)
        assert action.action == "move"
        assert action.x < 250
        assert action.task_status == "in_progress"
        process.on_action_success(memory, action)
        assert memory.current_stage == "vote_choose_target"

    def test_voting_target_stage_emits_repeated_click_bundle_before_resolve(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.choice_catalog.candidates["luxury_ban"] = ChoiceCandidate(
            id="luxury_ban",
            label="사치 자원 금지",
        )
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.set_current_voting_agenda(option_id="luxury_ban")
        memory.begin_stage("vote_choose_target")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "x": 660,
                        "y": 515,
                        "repeat_count": 2,
                        "selection": "말",
                        "reason": "같은 대상에 표를 몰아주기 위해 반복 선택",
                    }
                )
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
        assert len(actions) == 2
        assert all(action.action == "click" for action in actions)
        process.on_actions_success(memory, actions)
        assert memory.current_stage == "vote_resolve_agenda"
        assert memory.voting_state.selected_target_label == "말"

    def test_voting_resolve_stage_marks_current_agenda_complete_and_returns_to_remaining_agenda(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.remember_choices(
            [
                {"id": "luxury_ban", "label": "사치 자원 금지"},
                {"id": "trade_bonus", "label": "교역 보너스"},
            ],
            end_of_list=True,
            scroll_direction="down",
        )
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.set_current_voting_agenda(option_id="luxury_ban")
        memory.begin_stage("vote_resolve_agenda")
        provider = FakeProvider([json.dumps({"agenda_state": "complete", "reason": "현재 합의안 투표 완료"})])

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, StageTransition)
        assert action.stage == "vote_select_agenda"
        assert memory.current_stage == "vote_select_agenda"
        assert memory.voting_state.completed_agenda_ids == ["luxury_ban"]
        assert memory.choice_catalog.candidates["luxury_ban"].metadata["selected"] is True

    def test_voting_exit_stage_finishes_in_terminal_state(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.begin_stage("vote_exit")

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, AgentAction)
        assert action.action == "press"
        assert action.key == "escape"
        assert action.task_status == "in_progress"
        process.on_action_success(memory, action)
        assert memory.current_stage == "vote_complete"

        verification = process.verify_completion(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
        )

        assert verification.complete is True

    def test_voting_exit_semantic_verify_rejects_when_agenda_screen_still_visible(self):
        process = get_multi_step_process("voting_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("voting_primitive", enable_choice_catalog=True, enable_voting_state=True)
        memory.init_voting_state()
        memory.mark_substep("voting_entry_done")
        memory.begin_stage("vote_exit")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "voting_mode": "agenda_screen",
                        "voting_screen_ready": True,
                        "welcome_popup_visible": False,
                        "globe_button_visible": False,
                        "reasoning": "ESC 이후에도 합의안 block이 그대로 보여 아직 세계의회 안에 있음",
                    }
                )
            ]
        )

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            AgentAction(action="press", key="escape"),
        )

        assert verify.handled is True
        assert verify.passed is False

    def test_research_process_uses_entry_action_before_tree_is_ready(self):
        process = get_multi_step_process("research_select_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        process.initialize(memory)
        provider = FakeProvider([json.dumps({"research_screen_ready": False, "notification_visible": True})])

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
        assert action.action == "press"
        assert action.key == "enter"
        assert memory.current_stage == "research_entry"

    def test_culture_process_uses_entry_action_before_tree_is_ready(self):
        process = get_multi_step_process("culture_decision_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("culture_decision_primitive")
        process.initialize(memory)
        provider = FakeProvider([json.dumps({"culture_screen_ready": False, "notification_visible": True})])

        action = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="문화 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.action == "press"
        assert action.key == "enter"
        assert memory.current_stage == "culture_entry"

    def test_culture_entry_press_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("culture_decision_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("culture_decision_primitive")
        process.initialize(memory)

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="press", key="enter"),
        )

        assert should_verify is True

    def test_culture_entry_semantic_verify_promotes_to_direct_select(self):
        process = get_multi_step_process("culture_decision_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("culture_decision_primitive")
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "culture_screen_ready": True,
                        "notification_visible": False,
                        "reasoning": "좌측 상단 선택 창이 열림",
                    }
                )
            ]
        )
        action = AgentAction(action="press", key="enter")

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            action,
        )

        assert verify.handled is True
        assert verify.passed is True
        process.on_action_success(memory, action)
        assert "culture_entry_done" in memory.completed_substeps
        assert memory.current_stage == "direct_culture_select"

    def test_research_entry_press_runs_semantic_verify_without_ui_change(self):
        process = get_multi_step_process("research_select_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        process.initialize(memory)

        should_verify = process.should_verify_action_without_ui_change(
            memory,
            AgentAction(action="press", key="enter"),
        )

        assert should_verify is True

    def test_research_entry_semantic_verify_promotes_to_direct_select(self):
        process = get_multi_step_process("research_select_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("research_select_primitive")
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "research_screen_ready": True,
                        "notification_visible": False,
                        "reasoning": "좌측 상단 선택 창이 열림",
                    }
                )
            ]
        )
        action = AgentAction(action="press", key="enter")

        verify = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            action,
        )

        assert verify.handled is True
        assert verify.passed is True
        process.on_action_success(memory, action)
        assert "research_entry_done" in memory.completed_substeps
        assert memory.current_stage == "direct_research_select"

    def test_city_production_process_requires_entry_before_observation(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)

        assert process.should_observe(memory) is False

    def test_city_production_process_enters_list_branch_before_observation(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "production_mode": "list",
                        "production_screen_ready": True,
                        "notification_visible": False,
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

        assert isinstance(action, StageTransition)
        assert action.stage == "observe_choices"
        assert memory.current_stage == "observe_choices"
        assert memory.branch == "choice_list"
        assert process.should_observe(memory) is True

    def test_city_production_process_enters_placement_branch_without_observation(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "production_mode": "placement",
                        "production_screen_ready": True,
                        "notification_visible": False,
                    }
                ),
                json.dumps(
                    {
                        "placement_action": "click_tile",
                        "x": 640,
                        "y": 730,
                        "button": "right",
                        "tile_x": 640,
                        "tile_y": 730,
                        "tile_button": "right",
                        "tile_color": "green",
                        "reason": "초록색 타일에 특수지구 배치",
                    }
                ),
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

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert memory.current_stage == "production_place"
        assert memory.branch == "placement_map"
        assert process.should_observe(memory) is False

    def test_city_production_observe_repairs_missing_anchor_with_locator(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"label": "기념비"}, {"label": "개척자"}],
                        "end_of_list": False,
                        "scroll_anchor": None,
                        "reasoning": "생산 목록 상단이 보인다",
                    }
                ),
                json.dumps(
                    {
                        "scroll_anchor": {
                            "x": 760,
                            "y": 510,
                            "left": 620,
                            "top": 100,
                            "right": 900,
                            "bottom": 920,
                        },
                        "reasoning": "세로 생산 목록 중앙",
                    }
                ),
            ]
        )

        observation = process.observe(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
        )
        action = process.consume_observation(memory, observation)

        assert observation is not None
        assert observation.scroll_anchor is not None
        assert observation.scroll_anchor["x"] == 760
        assert action is not None
        assert action.action == "move"
        assert action.x > observation.scroll_anchor["x"]
        assert action.x >= 840
        assert action.y == 510
        assert memory.current_stage == "hover_scroll_anchor"
        assert "obs_summary=" in memory.to_prompt_string()
        assert "scroll_anchor=(760,510)" in memory.to_prompt_string()

    def test_city_production_rejects_left_side_locator_anchor_and_falls_back_to_default(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"label": "기념비"}, {"label": "개척자"}],
                        "end_of_list": False,
                        "scroll_anchor": None,
                        "reasoning": "생산 목록 상단이 보인다",
                    }
                ),
                json.dumps(
                    {
                        "scroll_anchor": {
                            "x": 280,
                            "y": 510,
                            "left": 120,
                            "top": 100,
                            "right": 440,
                            "bottom": 920,
                        },
                        "reasoning": "왼쪽 위 탭 쪽으로 잘못 잡힌 anchor",
                    }
                ),
            ]
        )

        observation = process.observe(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
        )
        action = process.consume_observation(memory, observation)
        expected_anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert observation is not None
        assert observation.scroll_anchor == expected_anchor
        assert action is not None
        assert action.action == "move"
        assert (action.x, action.y) == (expected_anchor["x"], expected_anchor["y"])
        assert action.x > 500

    def test_city_production_uses_safe_default_anchor_instead_of_screen_center(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [{"label": "기념비"}],
                        "end_of_list": False,
                        "scroll_anchor": None,
                        "reasoning": "생산 목록이 보이지만 anchor 없음",
                    }
                ),
                json.dumps({"scroll_anchor": None, "reasoning": "anchor를 확정하지 못함"}),
            ]
        )

        observation = process.observe(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
        )
        action = process.consume_observation(memory, observation)
        expected_anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert observation is not None
        assert observation.scroll_anchor == expected_anchor
        assert action is not None
        assert action.action == "move"
        assert (action.x, action.y) == (expected_anchor["x"], expected_anchor["y"])
        assert (action.x, action.y) != (500, 500)

    def test_city_production_default_anchor_targets_right_side_list_panel(self):
        process = get_multi_step_process("city_production_primitive", "")

        anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert anchor["x"] > 500
        assert anchor["left"] > 500
        assert anchor["right"] > anchor["left"]

    def test_city_production_hover_success_transitions_to_scroll_stage(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("hover_scroll_anchor")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_anchor={"x": 720, "y": 520, "left": 600, "top": 100, "right": 900, "bottom": 920},
        )

        process.on_action_success(
            memory,
            process.resolve_action(AgentAction(action="move", x=720, y=520), memory),
        )
        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert memory.current_stage == "scroll_down_for_hidden_choices"
        assert action is not None
        assert action.action == "scroll"
        assert action.scroll_amount == -120
        assert action.x >= 820
        assert action.y == 520

    def test_city_production_scan_does_not_complete_after_first_scrolled_observation_finds_no_new_choices(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )
        assert first_action is not None
        process.on_action_success(memory, first_action)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert scroll_action is not None
        process.on_action_success(memory, scroll_action)

        second_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert second_action is not None
        assert second_action.action == "move"
        assert memory.choice_catalog.end_reached is False
        assert memory.current_stage == "hover_scroll_anchor"

    def test_city_production_does_not_trust_first_scrolled_end_of_list_signal(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )
        assert first_action is not None
        process.on_action_success(memory, first_action)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert scroll_action is not None
        process.on_action_success(memory, scroll_action)

        second_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "개척자"}, {"label": "기념비"}],
                end_of_list=True,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert second_action is not None
        assert second_action.action == "move"
        assert memory.choice_catalog.end_reached is False
        assert memory.current_stage == "hover_scroll_anchor"

    def test_city_production_trusts_initial_end_of_list_when_no_scroll_has_happened(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=True,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert action is None
        assert memory.choice_catalog.end_reached is True
        assert memory.current_stage == "choose_from_memory"

    def test_city_production_scan_completes_after_third_scrolled_observation_finds_no_new_choices(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("observe_choices")

        first_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )
        assert first_action is not None
        process.on_action_success(memory, first_action)

        first_scroll = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert first_scroll is not None
        process.on_action_success(memory, first_scroll)

        second_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )
        assert second_action is not None
        process.on_action_success(memory, second_action)

        second_scroll = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert second_scroll is not None
        process.on_action_success(memory, second_scroll)

        third_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert third_action is not None
        assert third_action.action == "move"
        assert memory.choice_catalog.end_reached is False
        process.on_action_success(memory, third_action)

        third_scroll = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert third_scroll is not None
        process.on_action_success(memory, third_scroll)

        fourth_action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"label": "기념비"}, {"label": "건설자"}],
                end_of_list=False,
                scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert fourth_action is None
        assert memory.choice_catalog.end_reached is True
        assert memory.current_stage == "choose_from_memory"

    def test_city_production_repairs_stale_invalid_memory_anchor_before_hover_scroll(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("hover_scroll_anchor")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_anchor={"x": 280, "y": 510, "left": 120, "top": 100, "right": 440, "bottom": 920},
        )

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        expected_anchor = process._default_list_scroll_anchor(1000)  # noqa: SLF001

        assert action is not None
        assert action.action == "move"
        assert (action.x, action.y) == (expected_anchor["x"], expected_anchor["y"])
        assert memory.get_scroll_anchor() is not None
        assert memory.get_scroll_anchor().x == expected_anchor["x"]

    def test_city_production_restore_flow_moves_then_scrolls_up_for_hidden_best_choice(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("choose_from_memory")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=True,
            scroll_anchor={"x": 700, "y": 500, "left": 580, "top": 100, "right": 900, "bottom": 920},
        )
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=False,
            position_hint="above",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")

        move_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert move_action is not None
        assert move_action.action == "move"
        assert memory.current_stage == "restore_hover_scroll_anchor"

        process.on_action_success(memory, move_action)

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert memory.current_stage == "restore_best_choice_visibility"
        assert scroll_action is not None
        assert scroll_action.action == "scroll"
        assert scroll_action.scroll_amount > 0
        assert scroll_action.x >= 880
        assert scroll_action.y == 500

    def test_city_production_initial_placement_screen_returns_tile_action_immediately(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        process.initialize(memory)
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "production_mode": "placement",
                        "production_screen_ready": True,
                        "notification_visible": False,
                    }
                ),
                json.dumps(
                    {
                        "placement_action": "click_tile",
                        "x": 640,
                        "y": 730,
                        "button": "right",
                        "tile_x": 640,
                        "tile_y": 730,
                        "tile_button": "right",
                        "tile_color": "green",
                        "reason": "초록색 타일에 특수지구 배치",
                    }
                ),
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

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert action.button == "right"
        assert memory.current_stage == "production_place"
        assert memory.branch == "placement_map"

    def test_city_production_placement_upscales_legacy_1000_click_tile_coords_for_10000_range(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task(
            "city_production_primitive",
            normalizing_range=10000,
            enable_choice_catalog=True,
        )
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "placement_action": "click_tile",
                        "x": 438,
                        "y": 645,
                        "button": "right",
                        "tile_x": 438,
                        "tile_y": 645,
                        "tile_button": "right",
                        "tile_color": "green",
                        "reason": "산과 인접한 초록 타일",
                    }
                )
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert action.button == "right"
        assert (action.x, action.y) == (4380, 6450)

    def test_city_production_purchase_button_upscales_legacy_1000_coords_for_10000_range(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task(
            "city_production_primitive",
            normalizing_range=10000,
            enable_choice_catalog=True,
        )
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "placement_action": "click_purchase_button",
                        "x": 438,
                        "y": 645,
                        "button": "right",
                        "tile_x": 510,
                        "tile_y": 690,
                        "tile_button": "right",
                        "tile_color": "purple",
                        "reason": "구매형 타일 골드 배지 클릭",
                    }
                )
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=10000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert action.button == "right"
        assert (action.x, action.y) == (4380, 6450)
        assert memory.get_city_placement_target() == (5100, 6900, "right")

    def test_city_production_placement_tile_click_transitions_to_confirmation_stage(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")

        process.on_action_success(
            memory,
            AgentAction(
                action="click",
                x=640,
                y=730,
                button="right",
                reasoning="건설할 타일 선택",
                task_status="in_progress",
            ),
        )

        assert memory.current_stage == "resolve_placement_followup"

    def test_city_production_purchasable_tile_clicks_gold_badge_and_transitions_to_reclick_stage(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "placement_action": "click_purchase_button",
                        "x": 612,
                        "y": 706,
                        "button": "right",
                        "tile_x": 640,
                        "tile_y": 730,
                        "tile_button": "right",
                        "tile_color": "purple",
                        "reason": "보라 타일은 골드 배지 먼저 눌러야 함",
                    }
                ),
                json.dumps({"placement_followup_state": "placement", "reason": "보라 타일 구매 후 아직 배치 화면"}),
            ]
        )

        action = process.plan_action(
            provider,
            Image.new("RGB", (2000, 1000)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, AgentAction)
        assert action.action == "click"
        assert action.button == "right"
        assert (action.x, action.y) == (612, 706)
        assert "골드" in (action.reasoning or "")

        process.on_action_success(memory, action)

        transition = process.plan_action(
            provider,
            Image.new("RGB", (2000, 1000)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(transition, StageTransition)
        assert transition.stage == "production_place_reclick"
        assert memory.current_stage == "production_place_reclick"
        assert provider.last_use_thinking is False
        assert provider.last_max_tokens is not None
        assert provider.last_max_tokens <= 128
        assert provider.last_pil_size is not None
        assert provider.last_pil_size[0] <= 640
        assert "placement_followup_state" in provider.last_text

    def test_city_production_purchasable_tile_reclick_uses_saved_tile_coordinates_before_confirm(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "placement_action": "click_purchase_button",
                        "x": 612,
                        "y": 706,
                        "button": "right",
                        "tile_x": 640,
                        "tile_y": 730,
                        "tile_button": "right",
                        "tile_color": "blue",
                        "reason": "파란 타일 구매 버튼 클릭",
                    }
                ),
                json.dumps({"placement_followup_state": "placement", "reason": "타일 구매만 완료됨"}),
                json.dumps({"placement_followup_state": "confirm", "reason": "건설 확인 팝업 표시"}),
            ]
        )

        purchase_action = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert isinstance(purchase_action, AgentAction)
        assert (purchase_action.x, purchase_action.y) == (612, 706)

        process.on_action_success(memory, purchase_action)

        first_transition = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert isinstance(first_transition, StageTransition)
        assert memory.current_stage == "production_place_reclick"

        reclick = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert reclick is not None
        assert reclick.action == "click"
        assert reclick.button == "right"
        assert (reclick.x, reclick.y) == (640, 730)
        assert (reclick.x, reclick.y) != (purchase_action.x, purchase_action.y)
        assert "같은 타일" in (reclick.reasoning or "")

        process.on_action_success(memory, reclick)
        assert memory.current_stage == "resolve_placement_followup"

        second_transition = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(second_transition, StageTransition)
        assert second_transition.stage == "production_place_confirm"
        assert memory.current_stage == "production_place_confirm"

    def test_city_production_placement_followup_retries_planning_instead_of_same_tile_reclick(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "placement_action": "click_tile",
                        "x": 640,
                        "y": 730,
                        "button": "right",
                        "tile_color": "blue",
                        "reason": "파란 타일이라도 건설할 땅을 직접 클릭",
                    }
                ),
                json.dumps({"placement_followup_state": "placement", "reason": "여전히 배치 지도라 재평가 필요"}),
            ]
        )

        placement_action = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert isinstance(placement_action, AgentAction)
        assert (placement_action.x, placement_action.y) == (640, 730)

        process.on_action_success(memory, placement_action)

        first_transition = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )
        assert isinstance(first_transition, StageTransition)
        assert first_transition.stage == "production_place"
        assert memory.current_stage == "production_place"

    def test_city_production_placement_stage_prompt_mentions_gold_badge_blue_purple_and_adjacency(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")

        instruction = process.build_instruction(
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert "현재 보유 골드" in instruction
        assert "인접 보너스" in instruction
        assert "골드" in instruction
        assert "보라" in instruction
        assert "골드와 숫자가 있는 구매 버튼/배지를 먼저 클릭" in instruction
        assert "같은 타일 본체를 다시 클릭" in instruction
        assert "캠퍼스를 기본값처럼 고르지 마" in instruction

    def test_city_production_placement_no_progress_retries_without_generic_fallback(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
        )

        first = process.handle_no_progress(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            last_action=AgentAction(action="click", x=640, y=730, button="right", task_status="in_progress"),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert first.handled is True
        assert first.reroute is False
        assert memory.current_stage == "production_place"
        assert memory.fallback_return_stage == ""

        second = process.handle_no_progress(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            last_action=AgentAction(action="click", x=640, y=730, button="right", task_status="in_progress"),
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert second.handled is False
        assert second.reroute is True
        assert memory.current_stage == "production_place"
        assert memory.fallback_return_stage == ""

    def test_city_production_does_not_reanchor_scroll_during_placement_branch(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_anchor={"x": 900, "y": 520, "left": 620, "top": 100, "right": 940, "bottom": 920},
        )

        action = process.resolve_action(
            AgentAction(action="scroll", x=0, y=0, scroll_amount=-650, task_status="in_progress"),
            memory,
        )

        assert action.x == 0
        assert action.y == 0

    def test_choice_catalog_debug_includes_scan_end_reason(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_direction="down",
        )
        memory.register_choice_scroll(direction="down")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_direction="down",
        )
        memory.register_choice_scroll(direction="down")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_direction="down",
        )
        memory.register_choice_scroll(direction="down")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=False,
            scroll_direction="down",
        )

        prompt_memory = memory.to_prompt_string()

        assert "scan_end_reason=down_scroll_no_new_candidates" in prompt_memory

    def test_city_production_restore_observation_returns_to_selection_when_best_choice_visible(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("observe_choices")
        memory.choice_catalog.end_reached = True
        memory.choice_catalog.last_scroll_direction = "up"
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=False,
            position_hint="above",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")

        action = process.consume_observation(
            memory,
            ObservationBundle(
                visible_options=[{"id": "개척자", "label": "개척자"}],
                end_of_list=False,
                scroll_anchor={"x": 700, "y": 500, "left": 580, "top": 100, "right": 900, "bottom": 920},
            ),
        )

        assert action is None
        assert memory.current_stage == "select_from_memory"
        assert memory.get_best_choice() is not None
        assert memory.get_best_choice().visible_now is True

    def test_city_production_select_click_success_routes_to_post_select_resolve(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("select_from_memory")
        memory.choice_catalog.candidates["캠퍼스"] = ChoiceCandidate(
            id="캠퍼스",
            label="캠퍼스",
            visible_now=True,
            position_hint="visible",
        )
        memory.set_best_choice(option_id="캠퍼스", reason="과학 우선")
        process.on_action_success(
            memory,
            AgentAction(
                action="click",
                x=640,
                y=420,
                reasoning="캠퍼스 선택",
                task_status="in_progress",
            ),
        )

        assert memory.current_stage == "resolve_post_select_followup"

    def test_city_production_select_click_can_finish_immediately_when_initial_list_has_no_hidden_choices(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("select_from_memory")
        memory.choice_catalog.end_reached = True
        memory.choice_catalog.scan_end_reason = "observer_end_of_list"
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=True,
            position_hint="visible",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")
        provider = FakeProvider([json.dumps({"post_select_state": "done", "reason": "추가 단계 없음"})])

        result = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(
                action="click",
                x=640,
                y=420,
                reasoning="개척자 선택",
                task_status="in_progress",
            ),
        )

        assert result.handled is True
        assert result.passed is True
        assert memory.current_stage == "production_complete"
        assert process.is_terminal_state(memory) is True

    def test_city_production_select_click_can_route_to_placement_immediately_when_initial_list_has_no_hidden_choices(
        self,
    ):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("select_from_memory")
        memory.choice_catalog.end_reached = True
        memory.choice_catalog.scan_end_reason = "observer_end_of_list"
        memory.choice_catalog.candidates["캠퍼스"] = ChoiceCandidate(
            id="캠퍼스",
            label="캠퍼스",
            visible_now=True,
            position_hint="visible",
        )
        memory.set_best_choice(option_id="캠퍼스", reason="과학 우선")
        provider = FakeProvider([json.dumps({"post_select_state": "placement", "reason": "지구 배치 필요"})])

        result = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(
                action="click",
                x=640,
                y=420,
                reasoning="캠퍼스 선택",
                task_status="in_progress",
            ),
        )

        assert result.handled is True
        assert result.passed is True
        assert memory.branch == "placement_map"
        assert memory.current_stage == "production_place"

    def test_city_production_select_click_can_finish_immediately_even_after_hidden_choice_scan(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("select_from_memory")
        memory.choice_catalog.end_reached = True
        memory.choice_catalog.scan_end_reason = "observer_end_of_list"
        memory.choice_catalog.downward_scan_scrolls = 1
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=True,
            position_hint="visible",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")
        provider = FakeProvider([json.dumps({"post_select_state": "done", "reason": "추가 단계 없음"})])

        result = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(
                action="click",
                x=640,
                y=420,
                reasoning="개척자 선택",
                task_status="in_progress",
            ),
        )

        assert result.handled is True
        assert result.passed is True
        assert memory.current_stage == "production_complete"

    def test_city_production_select_click_keeps_deferred_followup_only_when_post_select_state_is_unknown(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("select_from_memory")
        memory.choice_catalog.end_reached = True
        memory.choice_catalog.scan_end_reason = "observer_end_of_list"
        memory.choice_catalog.downward_scan_scrolls = 1
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=True,
            position_hint="visible",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")
        provider = FakeProvider([json.dumps({"post_select_state": "unknown", "reason": "판별 불가"})])

        result = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(
                action="click",
                x=640,
                y=420,
                reasoning="개척자 선택",
                task_status="in_progress",
            ),
        )

        assert result.handled is False
        assert memory.current_stage == "select_from_memory"

    def test_city_production_post_select_resolve_routes_to_placement_followup(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("resolve_post_select_followup")
        provider = FakeProvider([json.dumps({"post_select_state": "placement", "reason": "지구 배치 필요"})])

        transition = process.plan_action(
            provider,
            Image.new("RGB", (2000, 1000)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(transition, StageTransition)
        assert transition.stage == "production_place"
        assert memory.branch == "placement_map"
        assert memory.current_stage == "production_place"
        assert provider.last_use_thinking is False
        assert provider.last_max_tokens is not None
        assert provider.last_max_tokens <= 128
        assert provider.last_pil_size is not None
        assert provider.last_pil_size[0] <= 640
        assert "post_select_state" in provider.last_text

    def test_city_production_post_select_resolve_routes_to_confirm_followup(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("resolve_post_select_followup")
        provider = FakeProvider([json.dumps({"post_select_state": "confirm", "reason": "확인 팝업 표시"})])

        transition = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="성장 우선",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(transition, StageTransition)
        assert transition.stage == "production_place_confirm"
        assert memory.current_stage == "production_place_confirm"
        assert provider.last_use_thinking is False

    def test_city_production_confirm_popup_click_is_terminal_complete(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place_confirm")
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
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert action is not None
        assert action.task_status == "in_progress"
        assert "'예' 또는 확인 버튼" in provider.last_text

        process.on_action_success(
            memory,
            AgentAction(
                action="click",
                x=610,
                y=560,
                reasoning="확인 팝업의 예 버튼 클릭",
                task_status="in_progress",
            ),
        )

        assert process.is_terminal_state(memory) is True
        assert memory.current_stage == "production_complete"

    def test_city_production_scroll_verification_rejects_unchanged_visible_options(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("scroll_down_for_hidden_choices")
        memory.remember_choices(
            [
                {"id": "monument", "label": "기념비"},
                {"id": "builder", "label": "건설자"},
                {"id": "campus", "label": "캠퍼스", "disabled": True},
            ],
            end_of_list=False,
            scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [
                            {"id": "monument", "label": "기념비"},
                            {"id": "builder", "label": "건설자"},
                            {"id": "campus", "label": "캠퍼스", "disabled": True},
                        ],
                        "end_of_list": False,
                        "scroll_anchor": {
                            "x": 760,
                            "y": 520,
                            "left": 620,
                            "top": 100,
                            "right": 900,
                            "bottom": 920,
                        },
                        "reasoning": "같은 목록이 그대로 보임",
                    }
                )
            ]
        )

        verification = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(action="scroll", x=880, y=520, scroll_amount=-650, task_status="in_progress"),
        )

        assert verification.handled is True
        assert verification.passed is False
        assert "같은 선택지" in verification.reason

    def test_city_production_observation_summary_hides_disabled_and_checked_entries(self):
        process = get_multi_step_process("city_production_primitive", "")
        summary = process._summarize_visible_options(  # noqa: SLF001
            ObservationBundle(
                visible_options=[
                    {"id": "campus", "label": "캠퍼스", "disabled": True},
                    {"id": "granary", "label": "곡창", "selected": True},
                    {"id": "monument", "label": "기념비"},
                ],
                end_of_list=False,
            )
        )

        assert "기념비" in summary
        assert "캠퍼스" not in summary
        assert "곡창" not in summary

    def test_city_production_scroll_verification_ignores_disabled_only_viewport_changes(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("scroll_down_for_hidden_choices")
        memory.remember_choices(
            [
                {"id": "monument", "label": "기념비"},
                {"id": "builder", "label": "건설자"},
            ],
            end_of_list=False,
            scroll_anchor={"x": 760, "y": 520, "left": 620, "top": 100, "right": 900, "bottom": 920},
        )
        provider = FakeProvider(
            [
                json.dumps(
                    {
                        "visible_options": [
                            {"id": "monument", "label": "기념비"},
                            {"id": "builder", "label": "건설자"},
                            {"id": "campus", "label": "캠퍼스", "disabled": True},
                        ],
                        "end_of_list": False,
                        "scroll_anchor": {
                            "x": 760,
                            "y": 520,
                            "left": 620,
                            "top": 100,
                            "right": 900,
                            "bottom": 920,
                        },
                        "reasoning": "새로 보인 것은 어두운 캠퍼스뿐",
                    }
                )
            ]
        )

        verification = process.verify_action_success(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            AgentAction(action="scroll", x=880, y=520, scroll_amount=-650, task_status="in_progress"),
        )

        assert verification.handled is True
        assert verification.passed is False
        assert "같은 선택지" in verification.reason

    def test_choice_catalog_treats_new_disabled_rows_as_non_selectable_scan_progress(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        memory.remember_choices(
            [{"id": "monument", "label": "기념비"}],
            end_of_list=False,
            scroll_direction="down",
        )
        memory.register_choice_scroll(direction="down")
        memory.remember_choices(
            [
                {"id": "monument", "label": "기념비"},
                {"id": "campus", "label": "캠퍼스", "disabled": True},
            ],
            end_of_list=False,
            scroll_direction="down",
        )

        assert memory.choice_catalog.last_new_candidate_count == 0
        assert memory.choice_catalog.last_visible_option_ids == ("monument",)
        assert memory.choice_catalog.end_reached is False

    def test_choice_catalog_removes_candidate_from_visible_prompt_when_it_turns_disabled(self):
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.remember_choices(
            [{"id": "campus", "label": "캠퍼스"}],
            end_of_list=False,
            scroll_direction="down",
        )
        memory.set_best_choice(option_id="campus", reason="과학 우선")

        memory.remember_choices(
            [{"id": "campus", "label": "캠퍼스", "disabled": True}],
            end_of_list=False,
            scroll_direction="down",
        )

        prompt_memory = memory.to_prompt_string()

        assert memory.get_best_choice() is None
        assert "- 캠퍼스" not in prompt_memory
        assert "캠퍼스 (visible)" not in prompt_memory

    def test_city_production_visible_progress_uses_branch_stage_not_action_count(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)

        entry_progress = process.get_visible_progress(memory, executed_steps=0, hard_max_steps=18)

        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("hover_scroll_anchor")
        scan_progress = process.get_visible_progress(memory, executed_steps=5, hard_max_steps=18)

        memory.begin_stage("choose_from_memory")
        choose_progress = process.get_visible_progress(memory, executed_steps=6, hard_max_steps=18)

        memory.begin_stage("select_from_memory")
        select_progress = process.get_visible_progress(memory, executed_steps=7, hard_max_steps=18)

        assert entry_progress == (1, 2)
        assert scan_progress == (2, 4)
        assert choose_progress == (3, 4)
        assert select_progress == (4, 4)

    def test_city_production_visible_progress_keeps_fallback_in_same_stage_bucket(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("generic_fallback")
        memory.set_fallback_return_stage("production_place", "city_production_placement:production_place")

        progress = process.get_visible_progress(memory, executed_steps=9, hard_max_steps=18)

        assert progress == (2, 3)

    def test_city_production_visible_progress_uses_three_step_placement_flow(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.begin_stage("production_place")

        placement_progress = process.get_visible_progress(memory, executed_steps=2, hard_max_steps=18)

        memory.begin_stage("production_place_confirm")
        confirm_progress = process.get_visible_progress(memory, executed_steps=3, hard_max_steps=18)

        assert placement_progress == (2, 3)
        assert confirm_progress == (3, 3)

    def test_city_production_visible_progress_expands_for_purchase_reclick_branch(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("placement_map")
        memory.remember_city_placement_target(
            x=620,
            y=700,
            button="right",
            origin="purchase_button",
            reason="구매형 타일",
            tile_color="purple",
        )
        memory.begin_stage("production_place_reclick")

        reclick_progress = process.get_visible_progress(memory, executed_steps=3, hard_max_steps=18)

        memory.begin_stage("production_place_confirm")
        confirm_progress = process.get_visible_progress(memory, executed_steps=4, hard_max_steps=18)

        assert reclick_progress == (3, 4)
        assert confirm_progress == (4, 4)

    def test_city_production_restore_scroll_uses_same_reduced_magnitude_upward(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("restore_best_choice_visibility")
        memory.remember_choices(
            [{"label": "기념비"}],
            end_of_list=True,
            scroll_anchor={"x": 700, "y": 500, "left": 580, "top": 100, "right": 900, "bottom": 920},
        )
        memory.choice_catalog.candidates["개척자"] = ChoiceCandidate(
            id="개척자",
            label="개척자",
            visible_now=False,
            position_hint="above",
        )
        memory.set_best_choice(option_id="개척자", reason="확장 우선")

        scroll_action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert scroll_action is not None
        assert scroll_action.action == "scroll"
        assert scroll_action.scroll_amount == 120

    def test_city_production_restore_scroll_success_records_upward_direction(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("restore_best_choice_visibility")
        memory.choice_catalog.last_scroll_direction = "down"

        process.on_action_success(
            memory,
            AgentAction(action="scroll", x=880, y=520, scroll_amount=120, task_status="in_progress"),
        )

        assert memory.current_stage == "observe_choices"
        assert memory.choice_catalog.last_scroll_direction == "up"

    def test_city_production_post_select_resolve_can_finish_without_followup(self):
        process = get_multi_step_process("city_production_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("city_production_primitive", enable_choice_catalog=True)
        memory.mark_substep("production_entry_done")
        memory.set_branch("choice_list")
        memory.begin_stage("resolve_post_select_followup")
        provider = FakeProvider([json.dumps({"post_select_state": "done", "reason": "추가 단계 없음"})])

        transition = process.plan_action(
            provider,
            Image.new("RGB", (1600, 900)),
            memory,
            normalizing_range=1000,
            high_level_strategy="초반 문화",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(transition, StageTransition)
        assert transition.stage == "production_complete"
        assert memory.branch == "choice_list"
        assert memory.current_stage == "production_complete"


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
                        "active_tab": "전체",
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

    def test_bootstrap_filters_queue_to_slot_tabs_without_wild_slot(self):
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
                        "active_tab": "전체",
                        "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],
                        "wild_slot_active": False,
                        "slot_inventory": [
                            {"slot_id": "military_1", "slot_type": "군사", "is_empty": True},
                            {"slot_id": "economic_1", "slot_type": "경제", "is_empty": False},
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
        assert memory.policy_state.eligible_tabs_queue == ["군사", "경제", "외교", "와일드카드", "암흑"]
        assert memory.get_policy_current_tab_name() == "군사"

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

    def test_click_cached_tab_verification_uses_card_list_as_primary_signal(self):
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
        provider = FakeProvider(
            [
                json.dumps({"match": True, "observed_tab": "경제", "reason": "노란 카드"}),
            ]
        )

        verified = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is True
        assert "기대 탭: 경제" in provider.last_text
        assert verified.details["card_list_observed"] == "경제"
        assert verified.details["tab_bar_observed"] == "skipped"
        assert provider.last_pil_size == _policy_crop_size(
            process, Image.new("RGB", (100, 100)), _POLICY_RIGHT_CARD_LIST_RATIOS
        )

    def test_click_cached_tab_verification_accepts_empty_diplomatic_panel(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "외교", "x": 820, "y": 100}],
            eligible_tabs_queue=["외교"],
            slot_inventory=[{"slot_id": "diplomatic_1", "slot_type": "외교", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["외교"],
        )
        memory.begin_stage("click_cached_tab")
        provider = FakeProvider(
            [
                json.dumps({"match": True, "observed_tab": "empty", "reason": "오른쪽 카드 없음"}),
            ]
        )

        verified = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is True
        assert verified.details["card_list_observed"] == "empty"
        assert provider.last_pil_size == _policy_crop_size(
            process, Image.new("RGB", (100, 100)), _POLICY_RIGHT_CARD_LIST_RATIOS
        )

    def test_click_cached_tab_verification_fails_for_wrong_color_panel(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            provisional_tabs=["군사"],
        )
        memory.begin_stage("click_cached_tab")
        provider = FakeProvider([json.dumps({"match": False, "observed_tab": "경제", "reason": "노란 카드"})])

        verified = process.verify_action_success(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            type("A", (), {"action": "click"})(),
        )

        assert verified.handled is True
        assert verified.passed is False
        assert verified.details["card_list_observed"] == "경제"

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

    def test_plan_current_tab_failure_resumes_from_plan_current_tab_after_generic_fallback(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "경제", "x": 760, "y": 100}],
            eligible_tabs_queue=["경제"],
            slot_inventory=[{"slot_id": "economic_1", "slot_type": "경제", "is_empty": True}],
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
                json.dumps({"action": "move", "x": 400, "y": 400, "reasoning": "정책 화면 복구"}),
            ]
        )

        planned = process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert planned is not None
        assert planned.action == "move"
        assert memory.current_stage == "generic_fallback"
        assert memory.fallback_return_stage == "plan_current_tab"
        assert memory.fallback_return_key == "plan_current_tab:경제"

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

    def test_click_cached_tab_stage_skips_reclick_when_selected_tab_already_matches_current_tab(self):
        process = get_multi_step_process("policy_primitive", "")
        memory = ShortTermMemory()
        memory.start_task("policy_primitive", enable_policy_state=True)
        memory.mark_policy_entry_done()
        memory.init_policy_state(
            tab_positions=[{"tab_name": "군사", "x": 700, "y": 100, "confirmed": True}],
            eligible_tabs_queue=["군사"],
            slot_inventory=[{"slot_id": "military_1", "slot_type": "군사", "is_empty": True}],
            wild_slot_active=False,
            overview_mode=False,
            selected_tab_name="군사",
        )
        memory.begin_stage("click_cached_tab")

        action = process.plan_action(
            FakeProvider([]),
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert isinstance(action, StageTransition)
        assert action.stage == "plan_current_tab"
        assert memory.current_stage == "plan_current_tab"

    def test_finalize_policy_without_changes_returns_escape_complete_when_assign_disabled(self):
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
                            "assign_enabled": False,
                            "assign_x": 0,
                            "assign_y": 0,
                            "reason": "변경이 없어 버튼이 비활성",
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
        assert action.action == "press"
        assert action.key == "escape"
        assert action.task_status == "complete"

    def test_finalize_policy_with_changes_clicks_assign_when_enabled(self):
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
        memory.policy_state.changes_made_this_run = True
        memory.mark_policy_tab_completed("군사")
        memory.advance_policy_tab()
        memory.begin_stage("finalize_policy")

        action = process.plan_action(
            FakeProvider(
                [
                    json.dumps(
                        {
                            "assign_enabled": True,
                            "assign_x": 860,
                            "assign_y": 930,
                            "reason": "모든 정책 배정 버튼 활성",
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
        assert action.action == "click"
        assert action.x == 860
        assert action.y == 930
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

    def test_confirm_policy_popup_prompt_includes_stage_specific_terminal_guidance(self):
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

        process.plan_action(
            provider,
            Image.new("RGB", (100, 100)),
            memory,
            normalizing_range=1000,
            high_level_strategy="과학 승리",
            recent_actions="없음",
            hitl_directive=None,
        )

        assert "=== 현재 프로세스 상태 ===" in provider.last_text
        assert "현재 stage: confirm_policy_popup" in provider.last_text
        assert "이 단계에서만 task_status='complete'로 마무리한다." in provider.last_text

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
