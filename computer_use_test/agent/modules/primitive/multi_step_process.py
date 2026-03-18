"""
Class-based multi-step primitive processes.

This layer separates:
- scripted multi-step processes
- observation-assisted multi-step processes that must scan hidden choices

Observation is a separate VLM-only pass that collects visible choices without
making the final decision. The planner then acts using short-term memory.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass

from computer_use_test.agent.modules.memory.short_term_memory import ScrollAnchor, ShortTermMemory
from computer_use_test.agent.modules.primitive.runtime_hooks import (
    NoopSemanticVerifyHook,
    NoProgressResolution,
    RetryFallbackHook,
    SemanticVerifyResult,
)
from computer_use_test.agent.modules.router.primitive_registry import get_primitive_prompt
from computer_use_test.utils.image_pipeline import PRESETS
from computer_use_test.utils.llm_provider.base import BaseVLMProvider
from computer_use_test.utils.llm_provider.parser import AgentAction, strip_markdown
from computer_use_test.utils.prompts.primitive_prompt import MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION
from computer_use_test.utils.screen import norm_to_real

logger = logging.getLogger(__name__)

_POLICY_TAB_NAMES = ["군사", "경제", "외교", "와일드카드", "암흑"]
_POLICY_TAB_BAR_ORDER = ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"]
_POLICY_RIGHT_TAB_BAR_RATIOS = (0.57, 0.24, 0.97, 0.31)
_POLICY_RIGHT_CARD_LIST_RATIOS = (0.57, 0.29, 0.97, 0.84)
_PRODUCTION_LIST_DEFAULT_RATIOS = (0.68, 0.10, 0.94, 0.92)
_PRODUCTION_LIST_HOVER_X_RATIO = 0.88
_PRODUCTION_LIST_HOVER_RIGHT_INSET_RATIO = 0.02
_PRODUCTION_LIST_HOVER_WIDTH_BIAS = 0.72


def _normalized_coord_note(normalizing_range: int, *, fields: str) -> str:
    """Return a shared normalization contract for structured JSON fields."""
    return (
        f"- {fields}는 현재 VLM이 보는 스크린샷 기준 0-{normalizing_range} normalized coordinates 여야 한다.\n"
        "- 픽셀 좌표나 실제 모니터 좌표를 반환하지 마."
    )


def _build_observation_json_instruction(normalizing_range: int) -> str:
    """Structured JSON contract for observation-only passes."""
    return f"""응답은 아래 JSON 하나만 출력해.
{{
  "visible_options": [
    {{
      "id": "stable_id",
      "label": "실제 보이는 항목 이름",
      "disabled": false,
      "selected": false,
      "note": "효과/턴수/추가정보"
    }}
  ],
  "end_of_list": false,
  "scroll_anchor": {{
    "x": 0, "y": 0,
    "left": 0, "top": 0, "right": {normalizing_range}, "bottom": {normalizing_range}
  }},
  "reasoning": "관찰 요약"
}}
- 지금 화면에 실제로 보이는 항목만 적어.
- 절대 최종 선택을 하지 마.
- scroll_anchor는 스크롤해야 하는 팝업/리스트의 중앙 hover 지점이다.
- 스크롤할 팝업/리스트가 명확하지 않으면 scroll_anchor는 null 로 반환해도 된다.
- 목록 아래에 아직 새 항목이 남아 있으면 end_of_list=false.
- 더 아래에 새 항목이 없으면 end_of_list=true.
{_normalized_coord_note(normalizing_range, fields="scroll_anchor.x/y 와 scroll_anchor.left/top/right/bottom")}
"""


@dataclass
class ObservationBundle:
    """Structured result from the observer."""

    visible_options: list[dict]
    end_of_list: bool
    scroll_anchor: dict | None = None
    reasoning: str = ""


@dataclass
class VerificationResult:
    """Result of completion verification."""

    complete: bool
    reason: str = ""


@dataclass
class StageTransition:
    """Internal no-action stage transition between two planning phases."""

    stage: str
    reason: str = ""


class BaseObserver:
    """Base class for independent VLM observation passes."""

    def observe(
        self,
        provider: BaseVLMProvider,
        pil_image,
        prompt: str,
        *,
        img_config=None,
    ) -> ObservationBundle | None:
        prepared = provider._prepare_pil_image(pil_image, img_config=img_config)
        jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
        build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
        content_parts = [
            provider._build_pil_image_content(prepared, **build_kwargs),
            provider._build_text_content(prompt),
        ]

        try:
            response = provider._send_to_api(content_parts, temperature=0.1, max_tokens=2048, use_thinking=False)
            content = strip_markdown(response.content)
            data = json.loads(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Observation parse failed: %s", exc)
            return None

        visible_options = data.get("visible_options", [])
        if not isinstance(visible_options, list):
            visible_options = []

        end_of_list = bool(data.get("end_of_list", False))
        scroll_anchor = data.get("scroll_anchor")
        if not isinstance(scroll_anchor, dict):
            scroll_anchor = None

        return ObservationBundle(
            visible_options=[item for item in visible_options if isinstance(item, dict)],
            end_of_list=end_of_list,
            scroll_anchor=scroll_anchor,
            reasoning=str(data.get("reasoning", "")).strip(),
        )


def _analyze_structured_json(
    provider: BaseVLMProvider,
    pil_image,
    prompt: str,
    *,
    img_config=None,
    max_tokens: int = 2048,
):
    """Call the VLM with a PIL image and parse a JSON object response."""
    prepared = provider._prepare_pil_image(pil_image, img_config=img_config)
    jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
    build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
    content_parts = [
        provider._build_pil_image_content(prepared, **build_kwargs),
        provider._build_text_content(prompt),
    ]

    response = provider._send_to_api(content_parts, temperature=0.1, max_tokens=max_tokens, use_thinking=False)
    content = strip_markdown(response.content)
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError(f"expected dict JSON, got {type(data).__name__}")
    return data


class ScrollableChoiceObserver(BaseObserver):
    """Observer used by primitives that must scan a hidden scrollable list."""

    def __init__(self, target_description: str):
        self.target_description = target_description

    def build_prompt(self, primitive_name: str, memory: ShortTermMemory, *, normalizing_range: int) -> str:
        memory_summary = memory.to_prompt_string()
        return (
            "너는 문명6 에이전트의 관찰 전용 서브에이전트야. 선택/판단/클릭을 하지 말고 보이는 선택지만 수집해.\n\n"
            f"{_build_observation_json_instruction(normalizing_range)}\n\n"
            f"대상 UI: {self.target_description}\n"
            f"현재 primitive: {primitive_name}\n"
            f"이미 기억된 항목:\n{memory_summary}\n\n"
            "규칙:\n"
            "- 현재 화면에 보이는 항목만 visible_options에 넣어.\n"
            "- 비활성/어두운 선택지는 disabled=true로 표시.\n"
            "- 이미 선택되었거나 체크된 항목은 selected=true로 표시.\n"
            "- 스크롤해야 하는 실제 패널 중앙을 scroll_anchor로 반환.\n"
            "- 숨겨진 항목을 아직 못 본 상태면 end_of_list=false.\n"
        )


class CityProductionObserver(ScrollableChoiceObserver):
    """Observer specialized for the tall city-production list popup."""

    def build_prompt(self, primitive_name: str, memory: ShortTermMemory, *, normalizing_range: int) -> str:
        base_prompt = super().build_prompt(primitive_name, memory, normalizing_range=normalizing_range)
        return (
            f"{base_prompt}\n"
            "- 생산 목록은 화면 오른쪽에 세로로 길게 뜨는 생산 품목 패널이다. "
            "건물/유닛/지구 이름과 턴 수가 보이는 실제 목록 내부만 기준으로 봐.\n"
            "- scroll_anchor는 반드시 그 생산 목록 내부 중앙이어야 한다.\n"
            "- 지도 육각형, 좌측 빈 영역, 우측 HUD 바깥, 우하단 '생산 품목' 알림 버튼을 scroll_anchor로 주지 마.\n"
        )


class GovernorObserver(ScrollableChoiceObserver):
    """Observer specialized for the governor card list panel."""

    def build_prompt(self, primitive_name: str, memory: ShortTermMemory, *, normalizing_range: int) -> str:
        base_prompt = super().build_prompt(primitive_name, memory, normalizing_range=normalizing_range)
        return (
            f"{base_prompt}\n"
            "- 총독 카드 목록 패널에서 각 총독 카드를 관찰해.\n"
            "- 각 카드의 id는 총독 이름 slug (예: 핑갈라, 리앙, 허미즈_비밀결사).\n"
            "- note에 상태 정보를 기록해: '임명_가능' / '재배정' / '진급_가능' / '비밀결사' 등.\n"
            "- 임명 버튼이 있으면 '임명_가능', 진급 버튼이 있으면 '진급_가능'.\n"
            "- 이미 도시에 배정된 총독은 '재배정'.\n"
            "- 활성 버튼이 없으면 disabled=true.\n"
            "- scroll_anchor는 총독 카드 리스트 중앙이어야 한다.\n"
            "- 비밀결사 총독은 스크롤 아래에 숨겨져 있을 수 있음.\n"
        )


class BaseMultiStepProcess:
    """Base class for a class-owned multi-step primitive process."""

    supports_observation: bool = False

    def __init__(self, primitive_name: str, completion_condition: str = ""):
        self.primitive_name = primitive_name
        self.completion_condition = completion_condition
        self.retry_fallback_hook = RetryFallbackHook()
        self.semantic_verify_hook = NoopSemanticVerifyHook()

    def initialize(self, memory: ShortTermMemory) -> None:
        """Set the initial stage when the process starts."""
        if not memory.current_stage:
            memory.begin_stage("step")

    def should_observe(self, memory: ShortTermMemory) -> bool:
        """Whether the next step should be an independent observation pass."""
        return False

    def observe(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        img_config=None,
    ) -> ObservationBundle | None:
        return None

    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        """Update memory from observation and optionally emit a deterministic action."""
        return None

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        """Extra per-stage guidance appended to the base primitive prompt."""
        stage = memory.current_stage or "step"
        note = [f"현재 멀티스텝 stage: {stage}"]
        if memory.branch:
            note.append(f"현재 branch: {memory.branch}")
        if memory.completed_substeps:
            note.append(f"완료된 하위 단계: {', '.join(memory.completed_substeps[-5:])}")
        return "\n".join(note)

    def build_instruction(
        self,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
    ) -> str:
        prompt = get_primitive_prompt(
            self.primitive_name,
            normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            short_term_memory=memory.to_prompt_string(),
        )
        return f"{prompt}\n\n=== 현재 프로세스 상태 ===\n{self.build_stage_note(memory)}"

    def build_generic_fallback_note(self, memory: ShortTermMemory) -> str:
        stage = memory.fallback_return_stage or memory.current_stage or "step"
        return (
            f"현재 멀티스텝 stage '{stage}' 에서 실패했다. "
            "현재 화면을 복구하거나 다음 정상 단계로 돌아가기 위한 가장 안전한 단일 action 1개만 수행해."
        )

    def _plan_generic_fallback_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
        extra_note: str = "",
    ) -> AgentAction | None:
        prompt = get_primitive_prompt(
            self.primitive_name,
            normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            short_term_memory=memory.to_prompt_string(),
        )
        combined_note = extra_note or self.build_generic_fallback_note(memory)
        if combined_note:
            prompt = f"{prompt}\n\n[현재 추가 지시]\n{combined_note}"
        return provider.analyze(
            pil_image=pil_image,
            instruction=prompt,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )

    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        if memory.current_stage == "generic_fallback":
            return self._plan_generic_fallback_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
            )

        instruction = self.build_instruction(
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
        )
        return provider.analyze(
            pil_image=pil_image,
            instruction=instruction,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )

    def decide_from_memory(
        self,
        provider: BaseVLMProvider,
        memory: ShortTermMemory,
        *,
        high_level_strategy: str,
    ) -> bool:
        """Optional text-only decision stage before action planning."""
        return True

    def resolve_action(self, action: AgentAction, memory: ShortTermMemory) -> AgentAction:
        """Apply runtime constraints before execution."""
        if action.action == "scroll":
            anchor = memory.get_scroll_anchor()
            if anchor is not None and not anchor.contains(action.x, action.y):
                action.x = anchor.x
                action.y = anchor.y
            elif anchor is not None and action.x == 0 and action.y == 0:
                action.x = anchor.x
                action.y = anchor.y
        return action

    def resolve_actions(self, actions: list[AgentAction], memory: ShortTermMemory) -> list[AgentAction]:
        """Apply runtime constraints for one or more actions."""
        return [self.resolve_action(action, memory) for action in actions]

    def get_visible_progress(
        self,
        memory: ShortTermMemory,
        *,
        executed_steps: int,
        hard_max_steps: int,
    ) -> tuple[int, int]:
        """Return the user-facing step/max pair for Rich and HITL."""
        step = max(0, min(executed_steps, hard_max_steps))
        return step, hard_max_steps

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        """Hook called when the action produced a meaningful UI change."""

    def on_actions_success(self, memory: ShortTermMemory, actions: list[AgentAction]) -> None:
        """Hook called when a multi-action bundle produced a meaningful UI change."""
        for action in actions:
            self.on_action_success(memory, action)

    def get_recovery_key(self, memory: ShortTermMemory, *, stage_name: str | None = None) -> str:
        """Return the shared retry/fallback scope key for the current stage."""
        return stage_name or memory.current_stage or "step"

    def on_stage_success(self, memory: ShortTermMemory, action: AgentAction, *, stage_name: str) -> None:
        """Shared success handling after process-specific state updates."""
        recovery_key = self.get_recovery_key(memory, stage_name=stage_name)
        self.retry_fallback_hook.reset(memory, recovery_key)
        if stage_name == "generic_fallback":
            self.retry_fallback_hook.on_fallback_success(memory)

    @staticmethod
    def _force_task_status(action: AgentAction | list[AgentAction] | None, task_status: str):
        """Overwrite task_status for planned action(s) when stage semantics require it."""
        if action is None:
            return None
        if isinstance(action, list):
            for item in action:
                item.task_status = task_status
            return action
        action.task_status = task_status
        return action

    def verify_action_success(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        action: AgentAction,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        """Run an optional semantic verification hook for the action."""
        return self.semantic_verify_hook.verify(
            provider,
            pil_image,
            memory,
            action,
            img_config=img_config,
        )

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        """Whether to run semantic verification even when raw UI diff is tiny."""
        return False

    def should_verify_action_after_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        """Whether to run semantic verification after raw UI change was already detected."""
        return True

    def is_terminal_state(self, memory: ShortTermMemory) -> bool:
        """Whether the current process state is an explicit terminal state."""
        return False

    def terminal_state_reason(self, memory: ShortTermMemory) -> str:
        """Human-readable reason for terminal-state completion."""
        return f"terminal state reached: {memory.current_stage or 'unknown'}"

    def handle_no_progress(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        last_action: AgentAction,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> NoProgressResolution:
        """Hook called when the UI did not change after an action."""
        if memory.current_stage == "generic_fallback":
            return NoProgressResolution(
                handled=False,
                reroute=True,
                error_message="Generic fallback produced no UI change",
            )
        return self.retry_fallback_hook.handle_failure(
            memory,
            stage_name=memory.current_stage or "step",
            stage_key=self.get_recovery_key(memory),
        )

    def verify_completion(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        img_config=None,
    ) -> VerificationResult:
        """Ask the VLM to verify the terminal UI state when a task claims completion."""
        if not self.completion_condition:
            return VerificationResult(True, "completion_condition 없음")

        prompt = (
            "너는 문명6 멀티스텝 작업 종료 검증기야. 스크린샷을 보고 종료 조건 충족 여부만 판단해.\n"
            'JSON만 출력: {"complete": true/false, "reason": "간단한 이유"}\n\n'
            f"Primitive: {self.primitive_name}\n"
            f"종료 조건: {self.completion_condition}\n"
            f"현재 memory 요약:\n{memory.to_prompt_string()}\n"
        )

        prepared = provider._prepare_pil_image(pil_image, img_config=img_config)
        jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
        build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
        content_parts = [
            provider._build_pil_image_content(prepared, **build_kwargs),
            provider._build_text_content(prompt),
        ]

        try:
            response = provider._send_to_api(content_parts, temperature=0.1, max_tokens=512, use_thinking=False)
            content = strip_markdown(response.content)
            data = json.loads(content)
            return VerificationResult(complete=bool(data.get("complete", False)), reason=str(data.get("reason", "")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Completion verification failed for %s: %s", self.primitive_name, exc)
            return VerificationResult(False, "verification parse failed")


class ObservationAssistedProcess(BaseMultiStepProcess):
    """Process for primitives that must scan scrollable hidden choices first."""

    supports_observation = True

    def __init__(
        self,
        primitive_name: str,
        completion_condition: str = "",
        *,
        target_description: str,
    ):
        super().__init__(primitive_name, completion_condition)
        self.observer = ScrollableChoiceObserver(target_description)

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("observe_choices")

    def should_observe(self, memory: ShortTermMemory) -> bool:
        return not memory.choice_catalog.end_reached

    def observe(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        img_config=None,
    ) -> ObservationBundle | None:
        prompt = self.observer.build_prompt(
            self.primitive_name,
            memory,
            normalizing_range=normalizing_range,
        )
        effective_img_config = (
            PRESETS.get("observation_fast")
            if img_config is None
            else PRESETS.get(
                "observation_fast",
                img_config,
            )
        )
        return self.observer.observe(provider, pil_image, prompt, img_config=effective_img_config)

    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        memory.begin_stage("observe_choices")
        memory.remember_choices(
            observation.visible_options,
            end_of_list=observation.end_of_list,
            scroll_anchor=observation.scroll_anchor,
            scroll_direction="down",
        )

        if observation.end_of_list:
            memory.mark_substep("full_scan_complete")
            memory.begin_stage("choose_from_memory")
            return None

        anchor = memory.get_scroll_anchor()
        x = anchor.x if anchor else 500
        y = anchor.y if anchor else 500
        memory.begin_stage("scroll_down_for_hidden_choices")
        return AgentAction(
            action="scroll",
            x=x,
            y=y,
            scroll_amount=-650,
            reasoning="아직 목록 끝이 아니므로 같은 팝업 중앙 hover 상태에서 아래로 스크롤",
            task_status="in_progress",
        )

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        if not memory.choice_catalog.end_reached:
            return (
                "현재 멀티스텝 stage: observation_scan\n"
                "- 아직 최종 선택을 하지 말고 hidden choice가 있는지 먼저 확인해.\n"
                "- scroll action은 저장된 popup/list 중앙 hover 지점에서만 수행한다."
            )

        best_choice = memory.get_best_choice()
        best_line = f"- 현재 best choice: {best_choice.label} ({best_choice.position_hint})" if best_choice else ""
        return (
            "현재 멀티스텝 stage: select_from_memory\n"
            "- short term memory의 choice catalog 전체를 기준으로 최적 선택을 결정해.\n"
            "- 선택지가 현재 안 보이면 scroll로 다시 찾아가고, 보이면 클릭/확정한다.\n"
            f"{best_line}"
        ).strip()

    def decide_from_memory(
        self,
        provider: BaseVLMProvider,
        memory: ShortTermMemory,
        *,
        high_level_strategy: str,
    ) -> bool:
        if memory.get_best_choice() is not None:
            return True

        memory.begin_stage("decide_best_choice")
        max_tokens = memory.choice_catalog_decision_max_tokens()
        prompt = (
            "너는 문명6 선택 결정 서브에이전트야. 아래 short-term memory에 누적된 전체 후보 중 "
            "상위 전략에 가장 적합한 하나를 고르고 JSON만 출력해.\n"
            'JSON: {"best_option_id":"stable_id","reason":"짧은 이유"}\n'
            "- best_option_id는 후보 catalog에 적힌 id를 그대로 복사해.\n"
            "- 체크됨, 이미 지음, 비활성 후보는 고르지 마.\n\n"
            f"Primitive: {self.primitive_name}\n"
            f"상위 전략:\n{high_level_strategy}\n\n"
            f"후보 catalog:\n{memory.choice_catalog_decision_prompt()}\n"
        )
        try:
            response = provider.call_vlm(
                prompt=prompt,
                image_path=None,
                temperature=0.2,
                max_tokens=max_tokens,
                use_thinking=False,
            )
            content = strip_markdown(response.content)
            data = json.loads(content)
            option_id = str(data.get("best_option_id", "")).strip()
            reason = str(data.get("reason", "")).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Best-choice decision failed for %s: %s", self.primitive_name, exc)
            return False

        if not option_id:
            return False

        memory.set_best_choice(option_id=option_id, reason=reason)
        return memory.get_best_choice() is not None


class ScriptedMultiStepProcess(BaseMultiStepProcess):
    """Process with a fixed sequence but no mandatory observer scan."""

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("scripted_flow")


class PolicyProcess(ScriptedMultiStepProcess):
    """Policy primitive with two entry branches that merge into card management."""

    def __init__(self, primitive_name: str, completion_condition: str = ""):
        super().__init__(primitive_name, completion_condition)

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("policy_entry")

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        stage = memory.current_stage
        if stage == "policy_entry":
            return (
                "현재 stage: policy_entry\n"
                "- '정책변경' 팝업 또는 '새 정부 선택' 분기만 처리해 정책 카드 화면으로 진입해."
            )
        if stage == "bootstrap_tabs":
            return (
                "현재 stage: bootstrap_tabs\n"
                "- policy_entry가 끝난 뒤 overview 정책 화면에서 "
                "좌측 슬롯 의미 정보와 오른쪽 탭바의 정책 탭 위치를 읽어 cache를 만든다."
            )
        if stage == "calibrate_tabs":
            current_tab = memory.get_policy_calibration_target_name() or "-"
            return (
                "현재 stage: calibrate_tabs\n"
                f"- 강제 재보정 단계다. 정책 탭 '{current_tab}'의 cached position을 클릭해 "
                "실제 탭 전환이 맞는지 확인한다.\n"
                "- 정상 bootstrap 경로에서는 이 stage를 쓰지 않고, 실패 복구가 필요할 때만 진입한다."
            )
        if stage == "click_cached_tab":
            current_tab = memory.get_policy_current_tab_name() or "-"
            return (
                "현재 stage: click_cached_tab\n"
                f"- cached position을 사용해 현재 탭 '{current_tab}'을 클릭해.\n"
                "- semantic verifier가 실제 탭 전환 성공 여부를 판정하고, 성공한 탭만 confirmed cache로 승격한다."
            )
        if stage == "plan_current_tab":
            current_tab = memory.get_policy_current_tab_name() or "-"
            return (
                "현재 stage: plan_current_tab\n"
                f"- 현재 탭 '{current_tab}'의 보이는 카드만 읽고 유지/교체를 한 번에 판단해.\n"
                "- 교체가 필요 없으면 빈 배열 []을 반환하고, 필요하면 drag action 배열을 즉시 만든다."
            )
        if stage == "click_next_tab":
            current_tab = memory.get_policy_current_tab_name() or "-"
            return (
                "현재 stage: click_next_tab\n"
                f"- 방금 완료한 이전 탭의 다음 순서인 현재 queued tab '{current_tab}'을 cached position으로 클릭해.\n"
                "- 클릭 전후 스크린샷 변화가 없으면 실패한 그 탭 하나만 다시 찾아 cached 좌표를 수정한다."
            )
        if stage == "generic_fallback":
            current_tab = memory.get_policy_current_tab_name() or "-"
            return (
                "현재 stage: generic_fallback\n"
                f"- 현재 탭 '{current_tab}'에서 structured flow가 막혔다. "
                "같은 policy primitive 안에서 화면을 복구하는 단일 action을 수행해."
            )
        if stage == "finalize_policy":
            return (
                "현재 stage: finalize_policy\n"
                "- 마지막으로 '모든 정책 배정' 버튼만 누른다.\n"
                "- 이 단계에서는 task_status를 complete로 끝내지 말고, 다음 확인 팝업 단계로 넘겨라."
            )
        if stage == "confirm_policy_popup":
            return (
                "현재 stage: confirm_policy_popup\n"
                "- '정책 안건이 확정되었습니까?' 확인 팝업의 '예' 또는 확인 버튼을 눌러 종료해.\n"
                "- 이 단계에서만 task_status='complete'로 마무리한다."
            )
        return (
            "현재 stage: policy_manage\n"
            "- 탭 queue를 따라 탭 클릭 -> 현재 탭 카드 판단 -> 즉시 drag-and-drop -> 다음 탭을 반복해."
        )

    def build_generic_fallback_note(self, memory: ShortTermMemory) -> str:
        current_tab = memory.get_policy_current_tab_name() or "-"
        queue = (
            ", ".join(memory.policy_state.eligible_tabs_queue) if memory.policy_state.eligible_tabs_queue else "없음"
        )
        stage = memory.fallback_return_stage or memory.current_stage or "step"
        return (
            f"현재 멀티스텝 stage '{stage}' 에서 실패했다. "
            f"같은 policy primitive 안에서 복구를 계속해야 한다. "
            f"현재 탭은 '{current_tab}', cached queue는 [{queue}] 이다. "
            "정책 화면을 안전하게 복구하거나 현재 단계가 다시 진행될 수 있도록 가장 안전한 단일 action 1개만 수행해."
        )

    def _build_policy_queue(self) -> list[str]:
        return list(_POLICY_TAB_NAMES)

    @staticmethod
    def _policy_crop_box(
        pil_image,
        ratios: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        """Return one clamped crop box from normalized image ratios."""
        width, height = pil_image.size
        left = max(0, min(width - 1, round(width * ratios[0])))
        top = max(0, min(height - 1, round(height * ratios[1])))
        right = max(left + 1, min(width, round(width * ratios[2])))
        bottom = max(top + 1, min(height, round(height * ratios[3])))
        return left, top, right, bottom

    @classmethod
    def _crop_policy_region(
        cls,
        pil_image,
        ratios: tuple[float, float, float, float],
    ) -> tuple[object, tuple[int, int, int, int]]:
        """Crop one fixed-ratio policy UI region from the current screenshot."""
        crop_box = cls._policy_crop_box(pil_image, ratios)
        return pil_image.crop(crop_box), crop_box

    @staticmethod
    def _crop_local_norm_to_global_norm(
        local_x: int,
        local_y: int,
        crop_box: tuple[int, int, int, int],
        pil_image,
        *,
        normalizing_range: int,
    ) -> tuple[int, int]:
        """Convert crop-local normalized coordinates back into full-image normalized coordinates."""
        width, height = pil_image.size
        left, top, right, bottom = crop_box
        crop_width = max(1, right - left)
        crop_height = max(1, bottom - top)

        local_px_x = norm_to_real(local_x, crop_width, normalizing_range)
        local_px_y = norm_to_real(local_y, crop_height, normalizing_range)
        global_px_x = min(width, left + local_px_x)
        global_px_y = min(height, top + local_px_y)
        return (
            round((global_px_x / max(1, width)) * normalizing_range),
            round((global_px_y / max(1, height)) * normalizing_range),
        )

    @classmethod
    def _reconcile_relocalized_policy_crop_position(
        cls,
        raw_x: int,
        raw_y: int,
        *,
        existing_x: int,
        existing_y: int,
        crop_box: tuple[int, int, int, int],
        pil_image,
        normalizing_range: int,
    ) -> tuple[int | None, int | None, str]:
        """Choose the best crop-local relocalize result after projecting it into full-image coordinates."""
        factor = cls._legacy_policy_scale_factor(normalizing_range)
        axis_budget = cls._policy_relocalize_axis_budget(normalizing_range)
        candidates: list[tuple[str, int, int]] = [("raw", raw_x, raw_y)]
        if factor is not None and raw_x <= 1000 and raw_y <= 1000:
            candidates.append((f"scaled(x{factor})", raw_x * factor, raw_y * factor))

        ranked: list[tuple[int, int, str, int, int, int, int]] = []
        for label, candidate_local_x, candidate_local_y in candidates:
            if not (0 <= candidate_local_x <= normalizing_range and 0 <= candidate_local_y <= normalizing_range):
                continue
            candidate_global = cls._crop_local_norm_to_global_norm(
                candidate_local_x,
                candidate_local_y,
                crop_box,
                pil_image,
                normalizing_range=normalizing_range,
            )
            dx = abs(candidate_global[0] - existing_x)
            dy = abs(candidate_global[1] - existing_y)
            ranked.append(
                (
                    max(dx, dy),
                    dx + dy,
                    label,
                    candidate_global[0],
                    candidate_global[1],
                    candidate_local_x,
                    candidate_local_y,
                )
            )

        if not ranked:
            return None, None, f"reject raw=({raw_x},{raw_y}) out-of-range"

        ranked.sort(key=lambda item: (item[0], item[1], 0 if item[2] == "raw" else 1))
        max_axis_delta, _, label, chosen_x, chosen_y, chosen_local_x, chosen_local_y = ranked[0]
        if max_axis_delta > axis_budget:
            return (
                None,
                None,
                f"reject raw=({raw_x},{raw_y}) existing=({existing_x},{existing_y}) budget={axis_budget}",
            )

        if label == "raw":
            return chosen_x, chosen_y, f"raw=({raw_x},{raw_y})"
        return (
            chosen_x,
            chosen_y,
            f"raw=({raw_x},{raw_y}) -> ({chosen_local_x},{chosen_local_y}) {label}",
        )

    def get_recovery_key(self, memory: ShortTermMemory, *, stage_name: str | None = None) -> str:
        stage = stage_name or memory.current_stage or "step"
        current_tab = memory.get_policy_current_tab_name()
        if stage == "click_next_tab":
            if current_tab:
                return f"{stage}:{current_tab}"
        if current_tab and stage in {"click_cached_tab", "plan_current_tab", "finalize_policy"}:
            return f"{stage}:{current_tab}"
        return stage

    @staticmethod
    def _legacy_policy_scale_factor(normalizing_range: int) -> int | None:
        """Return the legacy 0-1000 scale factor when the runtime range is a clean multiple."""
        if normalizing_range <= 1000 or normalizing_range % 1000 != 0:
            return None
        factor = normalizing_range // 1000
        return factor if factor > 1 else None

    @classmethod
    def _maybe_upscale_bootstrap_positions(
        cls,
        positions: list[dict[str, int | str | bool]],
        *,
        normalizing_range: int,
    ) -> tuple[list[dict[str, int | str | bool]], str]:
        """Upscale bootstrap tab coordinates when the VLM leaked legacy 0-1000 values."""
        factor = cls._legacy_policy_scale_factor(normalizing_range)
        if factor is None or not positions:
            return positions, ""
        if not all(
            isinstance(item.get("x"), int)
            and isinstance(item.get("y"), int)
            and 0 <= int(item["x"]) <= 1000
            and 0 <= int(item["y"]) <= 1000
            for item in positions
        ):
            return positions, ""

        scaled_positions: list[dict[str, int | str | bool]] = []
        for item in positions:
            scaled_item = dict(item)
            scaled_item["x"] = int(item["x"]) * factor
            scaled_item["y"] = int(item["y"]) * factor
            scaled_positions.append(scaled_item)
        return scaled_positions, f"legacy1000x{factor}"

    @staticmethod
    def _policy_relocalize_axis_budget(normalizing_range: int) -> int:
        """Maximum tolerated drift from the cached tab position for one relocalize result."""
        return max(40, normalizing_range // 25)

    @classmethod
    def _reconcile_relocalized_policy_position(
        cls,
        raw_x: int,
        raw_y: int,
        *,
        existing_x: int,
        existing_y: int,
        normalizing_range: int,
    ) -> tuple[int | None, int | None, str]:
        """Choose between raw/scaled relocalize coordinates using the existing cache as anchor."""
        factor = cls._legacy_policy_scale_factor(normalizing_range)
        axis_budget = cls._policy_relocalize_axis_budget(normalizing_range)
        candidates: list[tuple[str, int, int]] = [("raw", raw_x, raw_y)]
        if factor is not None and raw_x <= 1000 and raw_y <= 1000:
            candidates.append((f"scaled(x{factor})", raw_x * factor, raw_y * factor))

        ranked: list[tuple[int, int, str, int, int]] = []
        for label, candidate_x, candidate_y in candidates:
            if not (0 <= candidate_x <= normalizing_range and 0 <= candidate_y <= normalizing_range):
                continue
            dx = abs(candidate_x - existing_x)
            dy = abs(candidate_y - existing_y)
            ranked.append((max(dx, dy), dx + dy, label, candidate_x, candidate_y))

        if not ranked:
            return None, None, f"reject raw=({raw_x},{raw_y}) out-of-range"

        ranked.sort(key=lambda item: (item[0], item[1], 0 if item[2] == "raw" else 1))
        max_axis_delta, _, label, chosen_x, chosen_y = ranked[0]
        if max_axis_delta > axis_budget:
            return (
                None,
                None,
                f"reject raw=({raw_x},{raw_y}) existing=({existing_x},{existing_y}) budget={axis_budget}",
            )

        if label == "raw":
            return chosen_x, chosen_y, f"raw=({raw_x},{raw_y})"
        return chosen_x, chosen_y, f"raw=({raw_x},{raw_y}) -> ({chosen_x},{chosen_y}) {label}"

    @staticmethod
    def _normalized_policy_to_absolute(
        memory: ShortTermMemory,
        x: int,
        y: int,
        *,
        normalizing_range: int,
    ) -> tuple[int, int] | None:
        """Project normalized policy coordinates into logical absolute screen coordinates."""
        geometry = memory.policy_state.capture_geometry
        if geometry is None:
            return None
        return (
            norm_to_real(x, geometry.region_w, normalizing_range) + geometry.x_offset,
            norm_to_real(y, geometry.region_h, normalizing_range) + geometry.y_offset,
        )

    @staticmethod
    def _absolute_policy_to_normalized(
        memory: ShortTermMemory,
        screen_x: int,
        screen_y: int,
        *,
        normalizing_range: int,
    ) -> tuple[int, int] | None:
        """Project absolute policy coordinates back into the current normalized capture space."""
        geometry = memory.policy_state.capture_geometry
        if geometry is None or geometry.region_w <= 0 or geometry.region_h <= 0:
            return None
        local_x = screen_x - geometry.x_offset
        local_y = screen_y - geometry.y_offset
        if local_x < 0 or local_y < 0 or local_x > geometry.region_w or local_y > geometry.region_h:
            return None
        return (
            round((local_x / geometry.region_w) * normalizing_range),
            round((local_y / geometry.region_h) * normalizing_range),
        )

    def _build_policy_tab_click(
        self,
        memory: ShortTermMemory,
        *,
        stage_name: str,
        target_tab: str,
        reasoning: str,
    ) -> AgentAction | None:
        """Build one absolute policy-tab click from the cached screen coordinates."""
        tab_position = memory.policy_state.tab_positions.get(target_tab)
        if not target_tab or tab_position is None:
            return None
        memory.begin_stage(stage_name)
        return AgentAction(
            action="click",
            coord_space="absolute",
            x=tab_position.screen_x,
            y=tab_position.screen_y,
            reasoning=reasoning,
            task_status="in_progress",
        )

    def _mark_policy_tab_click_success(
        self,
        memory: ShortTermMemory,
        *,
        tab_name: str,
        next_stage: str,
        event_prefix: str,
        empty_queue_event: str = "",
    ) -> None:
        """Finalize one successful policy-tab click and promote the cache entry to confirmed."""
        if not tab_name:
            if empty_queue_event:
                memory.set_policy_event(empty_queue_event)
            memory.begin_stage(next_stage)
            memory.capture_checkpoint()
            return

        tab_position = memory.policy_state.tab_positions.get(tab_name)
        memory.set_policy_selected_tab(tab_name)
        if memory.policy_state.capture_geometry is not None:
            memory.policy_state.cache_geometry = copy.deepcopy(memory.policy_state.capture_geometry)
        memory.mark_policy_tab_confirmed(tab_name)
        memory.policy_state.overview_mode = False
        memory.reset_policy_tab_failure(tab_name)
        memory.clear_policy_failed_tabs()
        coord_note = (
            f"{tab_name}=({tab_position.screen_x},{tab_position.screen_y})" if tab_position is not None else tab_name
        )
        memory.set_policy_event(f"{event_prefix}={coord_note}")
        logger.info("Policy tab cache fixed -> %s", coord_note)
        memory.begin_stage(next_stage)
        memory.capture_checkpoint()

    def _plan_generic_policy_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
        extra_note: str = "",
    ) -> AgentAction | None:
        prompt = get_primitive_prompt(
            self.primitive_name,
            normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            short_term_memory=memory.to_prompt_string(),
        )
        if extra_note:
            prompt = f"{prompt}\n\n[현재 추가 지시]\n{extra_note}"
        return provider.analyze(
            pil_image=pil_image,
            instruction=prompt,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )

    def _plan_generic_policy_multi_actions(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
        extra_note: str = "",
    ) -> list[AgentAction] | None:
        prompt = get_primitive_prompt(
            self.primitive_name,
            normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            short_term_memory=memory.to_prompt_string(),
            json_instruction_override=MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION,
        )
        if extra_note:
            prompt = f"{prompt}\n\n[현재 추가 지시]\n{extra_note}"
        return provider.analyze_multi(
            pil_image=pil_image,
            instruction=prompt,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )

    def _build_next_policy_tab_click(self, memory: ShortTermMemory) -> AgentAction | None:
        current_tab = memory.get_policy_current_tab_name()
        return self._build_policy_tab_click(
            memory,
            stage_name="click_next_tab",
            target_tab=current_tab,
            reasoning=f"다음 정책 카테고리 탭 '{current_tab}'을 cached absolute position으로 클릭",
        )

    def _build_calibration_tab_click(self, memory: ShortTermMemory) -> AgentAction | None:
        target_tab = memory.get_policy_calibration_target_name()
        return self._build_policy_tab_click(
            memory,
            stage_name="calibrate_tabs",
            target_tab=target_tab,
            reasoning=f"정책 탭 보정: '{target_tab}' 탭을 cached absolute position으로 클릭",
        )

    def _policy_tab_check_img_config(self, img_config=None):
        return PRESETS.get("planner_high_quality", img_config)

    def _scan_policy_tab_bar_positions(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        img_config=None,
    ) -> tuple[list[dict[str, int | str | bool]] | None, str]:
        """Read the 5 queued policy-tab positions from the cropped right-side tab bar."""
        tab_bar_image, crop_box = self._crop_policy_region(pil_image, _POLICY_RIGHT_TAB_BAR_RATIOS)
        prompt = (
            "너는 문명6 정책 탭바 분석기야. 이 이미지는 정책 화면의 오른쪽 상단 탭바만 crop한 이미지다.\n"
            "JSON만 출력해.\n"
            "{\n"
            '  "tab_positions": [\n'
            '    {"tab_name":"군사","x":0,"y":0}, {"tab_name":"경제","x":0,"y":0}, '
            '{"tab_name":"외교","x":0,"y":0}, {"tab_name":"와일드카드","x":0,"y":0}, '
            '{"tab_name":"암흑","x":0,"y":0}\n'
            "  ]\n"
            "}\n"
            "규칙:\n"
            f"{_normalized_coord_note(normalizing_range, fields='tab_positions.x/y')}\n"
            "- 이 crop에는 정책 탭바만 보인다고 가정해. 좌측 슬롯이나 우측 카드 목록은 없다.\n"
            f"- 탭 순서는 {' -> '.join(_POLICY_TAB_BAR_ORDER)} 이다.\n"
            "- '전체'는 overview 상태 표시일 뿐 queue 대상이 아니다. 반환하지 마.\n"
            "- '황금기'가 보여도 이번 primitive 대상이 아니므로 반환하지 마.\n"
            "- 반드시 군사, 경제, 외교, 와일드카드, 암흑 5개 탭의 중심 좌표만 반환해.\n"
            "- 외교는 경제와 와일드카드 사이, 와일드카드는 외교와 암흑 사이에 있다.\n"
            f"- 상위 전략 참고:\n{high_level_strategy}\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                tab_bar_image,
                prompt,
                img_config=self._policy_tab_check_img_config(img_config),
                max_tokens=768,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Policy tab-bar scan failed: %s", exc)
            return None, ""

        tab_positions = data.get("tab_positions", [])
        if not isinstance(tab_positions, list):
            return None, ""

        local_positions: list[dict[str, int | str | bool]] = []
        seen_tabs: set[str] = set()
        for item in tab_positions:
            if not isinstance(item, dict):
                continue
            tab_name = str(item.get("tab_name", item.get("name", ""))).strip()
            if tab_name not in _POLICY_TAB_NAMES or tab_name in seen_tabs:
                continue
            try:
                x = int(item.get("x", 0))
                y = int(item.get("y", 0))
            except (TypeError, ValueError):
                continue
            if not (0 <= x <= normalizing_range and 0 <= y <= normalizing_range):
                continue
            local_positions.append({"tab_name": tab_name, "x": x, "y": y, "confirmed": False})
            seen_tabs.add(tab_name)

        if set(seen_tabs) != set(_POLICY_TAB_NAMES):
            return None, ""

        local_positions, scale_note = self._maybe_upscale_bootstrap_positions(
            local_positions,
            normalizing_range=normalizing_range,
        )
        absolute_positions: list[dict[str, int | str | bool]] = []
        for item in local_positions:
            global_norm = self._crop_local_norm_to_global_norm(
                int(item["x"]),
                int(item["y"]),
                crop_box,
                pil_image,
                normalizing_range=normalizing_range,
            )
            absolute = self._normalized_policy_to_absolute(
                memory,
                global_norm[0],
                global_norm[1],
                normalizing_range=normalizing_range,
            )
            if absolute is None:
                return None, scale_note
            absolute_positions.append(
                {
                    "tab_name": str(item["tab_name"]),
                    "screen_x": absolute[0],
                    "screen_y": absolute[1],
                    "confirmed": False,
                }
            )
        return absolute_positions, scale_note

    def _verify_policy_tab_switch(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        expected_tab: str,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        card_list_image, _ = self._crop_policy_region(pil_image, _POLICY_RIGHT_CARD_LIST_RATIOS)
        prompt = (
            "문명6 정책 탭 확인기. JSON만 출력.\n"
            '{"match": true, "observed_tab": "경제", "reason": "노란 카드"}\n'
            f"기대 탭: {expected_tab}\n"
            "이 이미지는 정책 화면 오른쪽 카드 목록만 crop한 이미지다.\n"
            "판정 기준:\n"
            "- 이 crop에는 좌측 파란 슬롯 영역이 포함되지 않는다.\n"
            "- 오른쪽 카드 목록의 분류와 필터 상태만으로 현재 탭을 판단해.\n"
            "- 군사: 오른쪽 카드 목록이 주로 군사 카드(빨강 계열)다.\n"
            "- 경제: 오른쪽 카드 목록이 주로 경제 카드(노랑 계열)다.\n"
            "- 외교: 오른쪽 카드 목록이 주로 외교 카드(초록/청록/파랑 계열)다.\n"
            "- 와일드카드: 오른쪽 카드 목록이 주로 와일드카드 카드이며 "
            "보라색, 검은색, 황금색 카드가 섞여 보일 수 있다.\n"
            "- 암흑: 오른쪽 카드 목록이 주로 암흑 카드(검정 계열)다.\n"
            "- '전체'는 여러 색이 섞인 혼합 overview 목록이다. 와일드카드와 혼동하지 마.\n"
            "- 방금 클릭한 탭의 오른쪽 카드 목록이 보이면 match=true다.\n"
            "- 혼합 overview 목록이면 observed_tab='전체' 로 반환해.\n"
            "- 애매하면 match=false, observed_tab='unknown'.\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                card_list_image,
                prompt,
                img_config=self._policy_tab_check_img_config(img_config),
                max_tokens=96,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy tab check failed to parse for %s: %s", expected_tab, exc)
            memory.set_policy_last_tab_check_result(f"{expected_tab}->unknown:parse-fail")
            return SemanticVerifyResult(handled=True, passed=False, reason="policy tab-check parse failed")

        observed_tab = str(data.get("observed_tab", "unknown")).strip() or "unknown"
        is_overview_observation = observed_tab == "전체"
        matched = bool(data.get("match", False)) and observed_tab == expected_tab and not is_overview_observation
        reason = str(data.get("reason", "")).strip()
        result_note = f"{expected_tab}->{observed_tab}:{'ok' if matched else 'fail'}"
        if reason:
            result_note += f" ({reason})"
        if observed_tab == "전체" and not memory.policy_state.overview_mode:
            result_note += " [unexpected-overview]"
        memory.set_policy_last_tab_check_result(result_note)
        return SemanticVerifyResult(handled=True, passed=matched, reason=reason or result_note)

    def _policy_screen_ready(
        self,
        provider: BaseVLMProvider,
        pil_image,
        *,
        img_config=None,
    ) -> bool:
        prompt = (
            "너는 문명6 정책 진입 상태 판별기야. 현재 화면이 '정책 카드 관리 화면'인지 여부만 JSON으로 판단해.\n"
            'JSON만 출력: {"policy_screen_ready": true/false}\n'
            "정책 카드 관리 화면의 기준:\n"
            "- 좌측에 정책 슬롯 영역이 보인다.\n"
            "- 우측에 정책 카드 목록이 보인다.\n"
            "- 상단 또는 중상단에 군사/경제/외교/와일드카드/암흑 탭이 보인다.\n"
            "- '사회 제도 완성' / '정책 변경' 팝업만 보이는 상태는 false다.\n"
            "- '새 정부 선택' 화면은 false다.\n"
        )
        try:
            data = _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Policy screen-ready parse failed: %s", exc)
            return False
        return bool(data.get("policy_screen_ready", False))

    def _bootstrap_policy_screen(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        high_level_strategy: str,
        normalizing_range: int,
        img_config=None,
    ) -> bool:
        memory.set_policy_mode("structured")
        using_session_cache = memory.has_full_policy_tab_cache() and memory.policy_cache_matches_current_geometry()
        if using_session_cache:
            existing_positions = memory.export_policy_tab_cache()["positions"]
            prompt = (
                "너는 문명6 정책 화면 bootstrap 분석기야. 현재 화면이 정책 카드 관리 화면이면 아래 JSON만 출력해.\n"
                "{\n"
                '  "policy_screen_ready": true,\n'
                '  "overview_mode": true,\n'
                '  "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],\n'
                '  "wild_slot_active": true,\n'
                '  "slot_inventory": [\n'
                '    {"slot_id":"military_1","slot_type":"군사",'
                '"current_card_name":"","is_empty":true,"active":true,"is_wild":false}\n'
                "  ]\n"
                "}\n"
                '정책 카드 화면이 아니면 {"policy_screen_ready": false} 만 출력해.\n'
                "규칙:\n"
                "- 이미 검증된 정책 탭 좌표 cache는 코드가 별도로 갖고 있다. tab_positions는 반환하지 마.\n"
                "- slot_inventory에는 슬롯 의미 정보만 넣고 좌표는 넣지 마.\n"
                "- visible_tabs에는 실제로 보이는 탭 이름을 적어도 된다. "
                "전체/황금기가 보여도 괜찮다.\n"
                f"- 상위 전략 참고:\n{high_level_strategy}\n"
            )
        else:
            existing_positions = {}
            prompt = (
                "너는 문명6 정책 화면 bootstrap 분석기야. 현재 화면이 정책 카드 관리 화면이면 아래 JSON만 출력해.\n"
                "{\n"
                '  "policy_screen_ready": true,\n'
                '  "overview_mode": true,\n'
                '  "visible_tabs": ["전체", "군사", "경제", "외교", "와일드카드", "암흑", "황금기"],\n'
                '  "wild_slot_active": true,\n'
                '  "slot_inventory": [\n'
                '    {"slot_id":"military_1","slot_type":"군사",'
                '"current_card_name":"","is_empty":true,"active":true,"is_wild":false}\n'
                "  ]\n"
                "}\n"
                '정책 카드 화면이 아니면 {"policy_screen_ready": false} 만 출력해.\n'
                "규칙:\n"
                "- policy entry 직후의 첫 정책 화면은 기본적으로 overview_mode=true 로 본다.\n"
                "- visible_tabs에는 실제로 보이는 탭 이름을 적어도 된다. "
                "전체/황금기가 보여도 괜찮다.\n"
                "- slot_inventory에는 슬롯 의미 정보만 넣고 좌표는 넣지 마.\n"
                f"- 상위 전략 참고:\n{high_level_strategy}\n"
            )
        try:
            data = _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=2048)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Policy bootstrap parse failed: %s", exc)
            return False

        if not bool(data.get("policy_screen_ready", False)):
            return False

        slot_inventory = data.get("slot_inventory", [])
        if not isinstance(slot_inventory, list):
            return False

        visible_tabs = data.get("visible_tabs", [])
        if not isinstance(visible_tabs, list):
            visible_tabs = []
        visible_tabs = [str(tab) for tab in visible_tabs if str(tab) in _POLICY_TAB_NAMES]

        if set(visible_tabs) != set(_POLICY_TAB_NAMES):
            logger.info("Policy bootstrap rejected: visible tabs incomplete (%s)", visible_tabs)
            memory.set_policy_event(f"bootstrap rejected: visible_tabs={visible_tabs}")
            return False

        cached_positions: list[dict[str, int | str | bool]] = []
        bootstrap_scale_note = ""
        if using_session_cache:
            for tab_name in _POLICY_TAB_NAMES:
                payload = existing_positions.get(tab_name, {})
                if not isinstance(payload, dict):
                    continue
                cached_positions.append(
                    {
                        "tab_name": tab_name,
                        "screen_x": int(payload.get("screen_x", 0)),
                        "screen_y": int(payload.get("screen_y", 0)),
                        "confirmed": bool(payload.get("confirmed", False)),
                    }
                )
            if len(cached_positions) != len(_POLICY_TAB_NAMES):
                memory.set_policy_event("session cache incomplete -> recalibration required")
                return False
        else:
            cached_positions, bootstrap_scale_note = self._scan_policy_tab_bar_positions(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                img_config=img_config,
            )
            if cached_positions is None:
                return False
            if set(item["tab_name"] for item in cached_positions) != set(_POLICY_TAB_NAMES):
                seen_tabs = sorted(str(item["tab_name"]) for item in cached_positions)
                logger.info("Policy bootstrap rejected: tab positions incomplete (%s)", seen_tabs)
                memory.set_policy_event(f"bootstrap rejected: tab_positions={seen_tabs}")
                return False

        normalized_queue = self._build_policy_queue()

        memory.init_policy_state(
            tab_positions=cached_positions,
            eligible_tabs_queue=normalized_queue,
            slot_inventory=slot_inventory,
            wild_slot_active=bool(data.get("wild_slot_active", False)),
            overview_mode=bool(data.get("overview_mode", True)),
            visible_tabs=visible_tabs,
            provisional_tabs=[] if using_session_cache else _POLICY_TAB_NAMES,
            calibration_pending_tabs=[],
            cache_source="session_cache" if using_session_cache else "bootstrap_scan",
            capture_geometry=memory.policy_state.capture_geometry,
        )
        if (
            len(memory.policy_state.tab_positions) != len(_POLICY_TAB_NAMES)
            or not memory.policy_state.slot_inventory
            or not memory.policy_state.eligible_tabs_queue
        ):
            return False
        cache_summary = ", ".join(
            (
                f"{tab}=("
                f"{memory.policy_state.tab_positions[tab].screen_x},"
                f"{memory.policy_state.tab_positions[tab].screen_y})"
            )
            for tab in _POLICY_TAB_NAMES
        )
        memory.set_policy_bootstrap_summary(
            "tabs=5 "
            f"queue={','.join(memory.policy_state.eligible_tabs_queue)} "
            f"wild={memory.policy_state.wild_slot_active} "
            f"source={'session_cache' if using_session_cache else 'bootstrap_scan'}"
            f"{f' coord_scale={bootstrap_scale_note}' if bootstrap_scale_note else ''}"
        )
        scale_suffix = f" | coord_scale={bootstrap_scale_note}" if bootstrap_scale_note else ""
        memory.set_policy_event(f"bootstrap ok | queue={','.join(normalized_queue)} | {cache_summary}{scale_suffix}")
        logger.info(
            "Policy bootstrap success | source=%s | queue=%s | visible_tabs=%s | cache=%s | wild=%s",
            "session_cache" if using_session_cache else "bootstrap_scan",
            ",".join(memory.policy_state.eligible_tabs_queue),
            ",".join(visible_tabs),
            cache_summary,
            memory.policy_state.wild_slot_active,
        )
        memory.set_policy_bundle_action_count(0)
        memory.reset_policy_bootstrap_failure()
        memory.reset_stage_failure("bootstrap_tabs")
        memory.clear_stage_fallback_used("bootstrap_tabs")
        memory.mark_substep("policy_bootstrap_complete")
        if memory.is_policy_complete():
            memory.begin_stage("finalize_policy")
        else:
            memory.begin_stage("click_cached_tab")
        return True

    def _build_current_tab_click(self, memory: ShortTermMemory) -> AgentAction | None:
        current_tab = memory.get_policy_current_tab_name()
        return self._build_policy_tab_click(
            memory,
            stage_name="click_cached_tab",
            target_tab=current_tab,
            reasoning=f"정책 카테고리 탭 '{current_tab}'을 cached absolute position으로 클릭",
        )

    def _complete_current_policy_tab(self, memory: ShortTermMemory) -> None:
        current_tab = memory.get_policy_current_tab_name()
        if not current_tab:
            return
        memory.mark_policy_tab_completed(current_tab)
        memory.advance_policy_tab()

    def _plan_next_tab_click_or_finalize(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | None:
        current_tab = memory.get_policy_current_tab_name()
        if not current_tab:
            memory.begin_stage("finalize_policy")
            memory.set_policy_event("no next tab -> finalize")
            return self._force_task_status(
                self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note="'모든 정책 배정' 버튼만 클릭하고 아직 종료하지 마. 다음 확인 팝업 단계로 넘겨라.",
                ),
                "in_progress",
            )

        if memory.get_policy_current_tab_position() is None:
            memory.set_fallback_return_stage(
                "click_next_tab",
                self.get_recovery_key(memory, stage_name="click_next_tab"),
            )
            memory.begin_stage("generic_fallback")
            memory.set_policy_mode("generic_recovery")
            return self._plan_generic_policy_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=(
                    f"현재 queued tab '{current_tab}'의 cached position이 없다. "
                    "같은 policy primitive 안에서 그 탭을 다시 찾을 수 있게 "
                    "정책 화면을 복구하는 가장 안전한 단일 action을 수행해."
                ),
            )

        memory.set_policy_event(f"click next tab={current_tab}")
        return self._build_next_policy_tab_click(memory)

    def _relocalize_failed_tab(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        tab_name: str,
        *,
        normalizing_range: int,
        img_config=None,
    ) -> bool:
        tab_bar_image, crop_box = self._crop_policy_region(pil_image, _POLICY_RIGHT_TAB_BAR_RATIOS)
        prompt = (
            "너는 문명6 정책 탭 재탐색기야. 이 이미지는 정책 화면의 오른쪽 상단 탭바만 crop한 이미지다.\n"
            "지정된 탭 하나의 현재 위치만 다시 찾아 JSON만 출력해.\n"
            '{"found": true, "tab_name": "군사", "x": 0, "y": 0}\n'
            '못 찾으면 {"found": false}만 출력해.\n'
            f"찾을 탭: {tab_name}\n"
            f"정책 탭 후보: {', '.join(_POLICY_TAB_NAMES)}\n"
            f"{_normalized_coord_note(normalizing_range, fields='x/y')}\n"
            f"- 탭 순서는 {' -> '.join(_POLICY_TAB_BAR_ORDER)} 이다.\n"
            "- '전체'는 overview 상태 표시일 뿐 찾을 대상이 아니다.\n"
            "- '황금기'는 보여도 무시해.\n"
            "- 외교는 경제와 와일드카드 사이, 와일드카드는 외교와 암흑 사이에 있다.\n"
            "- 다른 탭은 무시하고 요청된 탭 하나의 중심 좌표만 반환해.\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                tab_bar_image,
                prompt,
                img_config=self._policy_tab_check_img_config(img_config),
                max_tokens=512,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy tab relocalize failed for %s: %s", tab_name, exc)
            return False

        if not bool(data.get("found", False)):
            memory.set_policy_last_relocalize_result(f"{tab_name}:not-found")
            return False

        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        if not (0 <= x <= normalizing_range and 0 <= y <= normalizing_range):
            memory.set_policy_last_relocalize_result(f"{tab_name}:invalid")
            return False

        existing_position = memory.policy_state.tab_positions.get(tab_name)
        if existing_position is None:
            global_norm = self._crop_local_norm_to_global_norm(
                x,
                y,
                crop_box,
                pil_image,
                normalizing_range=normalizing_range,
            )
            reconciled_x, reconciled_y, reconcile_note = global_norm[0], global_norm[1], f"raw=({x},{y})"
        else:
            existing_normalized = self._absolute_policy_to_normalized(
                memory,
                existing_position.screen_x,
                existing_position.screen_y,
                normalizing_range=normalizing_range,
            )
            if existing_normalized is None:
                memory.set_policy_last_relocalize_result(f"{tab_name}:geometry-mismatch")
                logger.warning("Rejected relocalized policy tab %s due to geometry mismatch", tab_name)
                return False
            reconciled_x, reconciled_y, reconcile_note = self._reconcile_relocalized_policy_crop_position(
                x,
                y,
                existing_x=existing_normalized[0],
                existing_y=existing_normalized[1],
                crop_box=crop_box,
                pil_image=pil_image,
                normalizing_range=normalizing_range,
            )

        if reconciled_x is None or reconciled_y is None:
            memory.set_policy_last_relocalize_result(f"{tab_name}:{reconcile_note}")
            logger.warning("Rejected relocalized policy tab %s | %s", tab_name, reconcile_note)
            return False

        absolute_position = self._normalized_policy_to_absolute(
            memory,
            reconciled_x,
            reconciled_y,
            normalizing_range=normalizing_range,
        )
        if absolute_position is None:
            memory.set_policy_last_relocalize_result(f"{tab_name}:geometry-missing")
            logger.warning("Rejected relocalized policy tab %s due to missing capture geometry", tab_name)
            return False

        memory.update_policy_tab_position(tab_name, absolute_position[0], absolute_position[1], confirmed=False)
        memory.policy_state.cache_geometry = copy.deepcopy(memory.policy_state.capture_geometry)
        memory.mark_policy_tab_provisional(tab_name)
        memory.set_policy_last_relocalize_result(
            f"{tab_name}:{reconcile_note} -> abs=({absolute_position[0]},{absolute_position[1]})"
        )
        logger.info(
            "Updated cached policy tab %s -> abs(%s, %s) | %s",
            tab_name,
            absolute_position[0],
            absolute_position[1],
            reconcile_note,
        )
        memory.set_policy_event(
            f"relocalized {tab_name}=abs({absolute_position[0]},{absolute_position[1]}) | {reconcile_note}"
        )
        return True

    def _restart_policy_calibration(
        self,
        memory: ShortTermMemory,
        *,
        preserve_progress: bool,
        reason: str,
    ) -> None:
        preserve_entry_done = memory.is_policy_entry_done()
        memory.clear_fallback_return_stage()
        memory.clear_policy_bootstrap(
            preserve_entry_done=preserve_entry_done,
            preserve_progress=preserve_progress,
            preserve_tab_positions=False,
        )
        memory.reset_policy_bootstrap_failure()
        memory.reset_stage_failure("bootstrap_tabs")
        memory.clear_stage_fallback_used("bootstrap_tabs")
        memory.clear_policy_failed_tabs()
        memory.set_policy_mode("structured")
        memory.set_policy_event(reason)
        memory.begin_stage("bootstrap_tabs" if preserve_entry_done else "policy_entry")
        logger.info(
            "Policy tab cache invalidated -> restarting bootstrap from %s (preserve_progress=%s)",
            memory.current_stage,
            preserve_progress,
        )

    def _advance_after_current_policy_tab(self, memory: ShortTermMemory, current_tab: str) -> StageTransition:
        """Move to the next queued tab or finalize after the current tab is done."""
        next_tab = memory.get_policy_next_tab_name()
        self._complete_current_policy_tab(memory)
        if next_tab:
            memory.begin_stage("click_next_tab")
            memory.set_policy_event(f"plan complete={current_tab} -> advance to {next_tab} | popped={current_tab}")
            return StageTransition(stage="click_next_tab", reason=f"현재 탭 '{current_tab}' 완료 -> {next_tab}")
        memory.begin_stage("finalize_policy")
        memory.set_policy_event(f"plan complete={current_tab} -> finalize")
        return StageTransition(stage="finalize_policy", reason=f"현재 탭 '{current_tab}' 완료 -> finalize")

    def _normalize_policy_drag_bundle(
        self,
        memory: ShortTermMemory,
        current_tab: str,
        actions: list[AgentAction],
    ) -> list[AgentAction] | None:
        """Validate one direct-drag bundle and inject canonical policy metadata."""
        normalized_actions: list[AgentAction] = []
        for action in actions:
            if action.action != "drag":
                logger.warning("Policy direct drag plan rejected non-drag action for %s", current_tab)
                return None

            action.policy_card_name = action.policy_card_name.strip()
            action.policy_target_slot_id = action.policy_target_slot_id.strip()
            action.policy_source_tab = current_tab
            action.policy_reasoning = action.policy_reasoning.strip() or action.reasoning.strip()
            if not action.task_status:
                action.task_status = "in_progress"

            if not action.policy_card_name or not action.policy_target_slot_id:
                logger.warning(
                    "Policy direct drag metadata missing | tab=%s card=%s slot=%s",
                    current_tab,
                    action.policy_card_name or "-",
                    action.policy_target_slot_id or "-",
                )
                return None

            target_slot = memory.policy_state.slot_inventory.get(action.policy_target_slot_id)
            if target_slot is None:
                logger.warning(
                    "Policy direct drag target slot missing | tab=%s slot=%s",
                    current_tab,
                    action.policy_target_slot_id,
                )
                return None

            if not memory.policy_slot_accepts_source_tab(current_tab, target_slot):
                logger.warning(
                    "Policy direct drag rejected illegal slot | tab=%s card=%s target_slot=%s target_type=%s",
                    current_tab,
                    action.policy_card_name,
                    action.policy_target_slot_id,
                    target_slot.slot_type,
                )
                return None

            normalized_actions.append(action)

        return normalized_actions

    def _plan_current_tab_actions(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> list[AgentAction] | StageTransition | None:
        current_tab = memory.get_policy_current_tab_name()
        if not current_tab:
            return None

        slot_lines = []
        for slot in memory.policy_state.slot_inventory.values():
            status = "빈칸" if slot.is_empty else slot.current_card_name or "카드 있음"
            suffix = " / 와일드" if slot.is_wild else ""
            source_note = f" / 현재 출처:{slot.selected_from_tab}" if slot.selected_from_tab else ""
            reason_note = f" / 현재 선택 이유:{slot.selection_reason}" if slot.selection_reason else ""
            slot_lines.append(f"- {slot.slot_id}: {slot.slot_type} / {status}{suffix}{source_note}{reason_note}")

        multi_actions = self._plan_generic_policy_multi_actions(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
            extra_note=(
                f"현재 탭 '{current_tab}'의 보이는 카드만 기준으로 유지 또는 교체를 한 번에 판단해.\n"
                "스크롤하지 말고 현재 화면에 보이는 카드만 사용해.\n"
                "드래그 소스 카드는 오른쪽 카드 목록에서만 찾아. 왼쪽 슬롯에 꽂힌 카드는 소스 후보가 아니다.\n"
                "교체가 필요 없으면 빈 배열 []을 반환해.\n"
                "교체가 필요하면 필요한 drag action들만 JSON 배열로 반환해.\n"
                "- click, press, scroll, type은 반환하지 마.\n"
                "- 반환 순서는 실제 실행 순서다.\n"
                "- drag 좌표는 현재 화면 기준으로 잡아.\n"
                "- 각 drag action에는 policy_card_name, policy_target_slot_id, "
                "policy_source_tab, policy_reasoning 필드를 포함해.\n"
                f"- policy_source_tab은 반드시 '{current_tab}'이어야 한다.\n"
                "- policy_target_slot_id는 왼쪽 슬롯 ID를 정확히 써.\n"
                "- policy_reasoning은 그 카드 선택 이유를 짧게 써.\n"
                "- 현재 탭 카드로는 현재 탭과 같은 카테고리 슬롯에만 배치할 수 있다.\n"
                "- 예: 경제 탭 카드 -> 경제 슬롯만 가능. 군사 슬롯에는 넣지 마.\n"
                "- 와일드 슬롯은 short-term memory에 기록된 현재 카드/이전 선택 이유와 "
                "비교했을 때 현재 탭 카드가 전략적으로 명확히 우세할 때만 "
                "교체 후보로 포함한다.\n"
                "- 단순히 현재 탭에도 좋은 카드가 보인다는 이유만으로 와일드 슬롯을 매 탭 바꾸지 마.\n"
                "- 우세 근거가 약하거나 근소하면 현재 와일드 카드를 유지한다.\n"
                "- 남은 queue에 아직 다른 탭이 남아 있으면 와일드 슬롯 교체는 더 보수적으로 판단한다.\n\n"
                f"상위 전략:\n{high_level_strategy}\n\n"
                f"왼쪽 슬롯 상태:\n{chr(10).join(slot_lines)}\n"
            ),
        )
        if multi_actions is None:
            logger.warning("Policy direct drag plan failed for %s", current_tab)
            return None

        if not multi_actions:
            memory.set_policy_bundle_action_count(0)
            return self._advance_after_current_policy_tab(memory, current_tab)

        normalized_actions = self._normalize_policy_drag_bundle(memory, current_tab, multi_actions)
        if normalized_actions is None:
            memory.set_policy_event(f"invalid drag bundle rejected={current_tab}")
            return None

        memory.set_policy_bundle_action_count(len(normalized_actions))
        memory.set_policy_event(f"plan_current_tab={current_tab} drags={len(normalized_actions)}")
        logger.info(
            "Policy direct drag plan built | tab=%s | actions=%s",
            current_tab,
            " | ".join(
                (
                    f"{action.policy_card_name}->{action.policy_target_slot_id} "
                    f"drag({action.x},{action.y}->{action.end_x},{action.end_y})"
                )
                for action in normalized_actions
            )
            or "-",
        )
        return normalized_actions

    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        if memory.current_stage == "generic_fallback":
            memory.set_policy_mode("generic_recovery")
            logger.info(
                "Policy generic fallback planning | return_stage=%s | queue_idx=%s | current_tab=%s",
                memory.fallback_return_stage or "-",
                memory.policy_state.current_tab_index,
                memory.get_policy_current_tab_name() or "-",
            )
            return self._plan_generic_policy_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=self.build_generic_fallback_note(memory),
            )

        if not memory.is_policy_entry_done():
            if self._policy_screen_ready(provider, pil_image, img_config=img_config):
                memory.mark_policy_entry_done()
                memory.mark_substep("policy_entry_done")
                memory.set_policy_event("policy entry complete")
                logger.info("Policy entry complete -> bootstrap_tabs")
                memory.begin_stage("bootstrap_tabs")
            else:
                memory.begin_stage("policy_entry")
                memory.set_policy_mode("structured")
                return self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note=(
                        "정책 카드 화면으로 진입하기 위한 entry branch만 처리해. "
                        "아직 policy bootstrap이나 탭 클릭을 하지 마."
                    ),
                )

        if not memory.has_policy_bootstrap():
            memory.begin_stage("bootstrap_tabs")
            memory.set_policy_mode("structured")
            bootstrapped = self._bootstrap_policy_screen(
                provider,
                pil_image,
                memory,
                high_level_strategy=high_level_strategy,
                normalizing_range=normalizing_range,
                img_config=img_config,
            )
            if not bootstrapped:
                failures = memory.increment_policy_bootstrap_failure()
                memory.set_policy_event(f"bootstrap failed count={failures}")
                logger.info("Policy bootstrap failed | count=%s", failures)
                if failures <= 2:
                    return self._plan_generic_policy_action(
                        provider,
                        pil_image,
                        memory,
                        normalizing_range=normalizing_range,
                        high_level_strategy=high_level_strategy,
                        recent_actions=recent_actions,
                        hitl_directive=hitl_directive,
                        img_config=img_config,
                        extra_note=(
                            "정책 카드 화면 bootstrap에 실패했다. "
                            "정책 화면을 안정적으로 유지하고 5개 정책 탭이 모두 읽히도록 "
                            "가장 안전한 action 1개만 수행해."
                        ),
                    )
                memory.set_fallback_return_stage("bootstrap_tabs", "bootstrap_tabs")
                memory.begin_stage("generic_fallback")
                memory.set_policy_mode("generic_recovery")
                return self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note=(
                        "정책 카드 화면 bootstrap이 두 번 실패했다. "
                        "같은 policy primitive 안에서 화면을 복구하는 가장 안전한 단일 action을 수행해."
                    ),
                )

        if memory.current_stage == "confirm_policy_popup":
            return self._force_task_status(
                self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note=(
                        "방금 '모든 정책 배정' 클릭은 이미 끝났다. "
                        "지금 보이는 확인 팝업의 '예' 또는 확인 버튼만 클릭하고 task_status를 complete로 설정해."
                    ),
                ),
                "complete",
            )

        if memory.is_policy_complete():
            memory.begin_stage("finalize_policy")
            memory.set_policy_event("queue complete -> finalize")
            logger.info("Policy queue complete -> finalize")
            return self._force_task_status(
                self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note="'모든 정책 배정' 버튼만 클릭하고 아직 종료하지 마. 다음 확인 팝업 단계로 넘겨라.",
                ),
                "in_progress",
            )

        if memory.current_stage == "calibrate_tabs" and memory.has_policy_calibration_pending():
            calibration_action = self._build_calibration_tab_click(memory)
            if calibration_action is not None:
                memory.set_policy_event(f"calibrate click={memory.get_policy_calibration_target_name()}")
                return calibration_action

        current_tab = memory.get_policy_current_tab_name()
        if not current_tab:
            memory.begin_stage("finalize_policy")
            return self._force_task_status(
                self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note="'모든 정책 배정' 버튼만 클릭하고 아직 종료하지 마. 다음 확인 팝업 단계로 넘겨라.",
                ),
                "in_progress",
            )

        if memory.get_policy_current_tab_position() is None:
            memory.set_fallback_return_stage("bootstrap_tabs", "bootstrap_tabs")
            memory.begin_stage("generic_fallback")
            memory.set_policy_mode("generic_recovery")
            return self._plan_generic_policy_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=(
                    f"현재 queued tab '{current_tab}'의 cached position이 없다. "
                    "같은 policy primitive 안에서 정책 화면을 복구해 bootstrap을 다시 진행할 수 있게 해."
                ),
            )

        if memory.current_stage == "click_next_tab":
            return self._plan_next_tab_click_or_finalize(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
            )

        if memory.current_stage == "click_cached_tab":
            memory.set_policy_event(f"click tab={current_tab}")
            return self._build_current_tab_click(memory)

        if memory.current_stage == "plan_current_tab":
            planned = self._plan_current_tab_actions(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
            )
            if not planned:
                memory.set_fallback_return_stage(
                    "click_cached_tab",
                    self.get_recovery_key(memory, stage_name="click_cached_tab"),
                )
                memory.begin_stage("generic_fallback")
                memory.set_policy_mode("generic_recovery")
                return self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note=(
                        "현재 탭 계획에 실패했다. policy 화면을 복구한 뒤 반드시 현재 queued tab을 다시 클릭해 "
                        "semantic verification부터 재개할 수 있도록 가장 안전한 단일 action을 수행해."
                    ),
                )
            if isinstance(planned, StageTransition):
                reason = (
                    f"현재 탭 '{current_tab}' 계획이 끝나 next-tab stage로 전환"
                    if planned.stage == "click_next_tab"
                    else f"현재 탭 '{current_tab}'이 마지막 탭이라 finalize stage로 전환"
                )
                return StageTransition(stage=planned.stage, reason=reason)
            return planned

        if memory.current_stage == "finalize_policy" or memory.is_policy_complete():
            memory.begin_stage("finalize_policy")
            memory.set_policy_event("queue complete -> finalize")
            return self._force_task_status(
                self._plan_generic_policy_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note="'모든 정책 배정' 버튼만 클릭하고 아직 종료하지 마. 다음 확인 팝업 단계로 넘겨라.",
                ),
                "in_progress",
            )

        return None

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        if action.action not in {"click", "double_click"}:
            return False
        return memory.should_verify_policy_tab_click()

    def verify_action_success(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        action: AgentAction,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        if action.action not in {"click", "double_click"} or not memory.should_verify_policy_tab_click():
            return super().verify_action_success(
                provider,
                pil_image,
                memory,
                action,
                img_config=img_config,
            )

        expected_tab = memory.get_policy_click_target_name()
        if not expected_tab:
            return SemanticVerifyResult(handled=True, passed=False, reason="policy tab target missing")

        return self._verify_policy_tab_switch(
            provider,
            pil_image,
            memory,
            expected_tab,
            img_config=img_config,
        )

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        if memory.current_stage == "calibrate_tabs" and action.action in {"click", "double_click"}:
            target_tab = memory.get_policy_calibration_target_name()
            if target_tab:
                memory.complete_policy_calibration_target(target_tab)
                memory.set_policy_selected_tab(target_tab)
                memory.reset_policy_tab_failure(target_tab)
                memory.clear_policy_failed_tabs()
                memory.set_policy_event(f"calibration success={target_tab}")
                logger.info("Policy tab calibration success -> %s", target_tab)
                if memory.has_policy_calibration_pending():
                    memory.begin_stage("calibrate_tabs")
                    memory.set_policy_mode("calibrating")
                else:
                    memory.begin_stage("click_cached_tab")
                    memory.set_policy_mode("structured")
                memory.capture_checkpoint()
            return

        if memory.current_stage == "click_cached_tab" and action.action in {"click", "double_click"}:
            current_tab = memory.get_policy_current_tab_name()
            if current_tab:
                self._mark_policy_tab_click_success(
                    memory,
                    tab_name=current_tab,
                    next_stage="plan_current_tab",
                    event_prefix="tab click success",
                )
            return

        if memory.current_stage == "click_next_tab" and action.action in {"click", "double_click"}:
            current_tab = memory.get_policy_current_tab_name()
            if current_tab:
                self._mark_policy_tab_click_success(
                    memory,
                    tab_name=current_tab,
                    next_stage="plan_current_tab",
                    event_prefix="next tab click success",
                )
            else:
                self._mark_policy_tab_click_success(
                    memory,
                    tab_name="",
                    next_stage="finalize_policy",
                    event_prefix="next tab click success",
                    empty_queue_event="next tab click success with empty current -> finalize",
                )
            return

        if memory.current_stage == "generic_fallback":
            memory.set_policy_mode("structured")
            memory.set_policy_event("generic fallback success -> resume structured flow")
            logger.info("Policy generic fallback success -> resume structured flow")
            return

        if memory.current_stage == "finalize_policy" and action.action in {"click", "double_click"}:
            memory.set_policy_event("finalize click success -> confirm popup")
            logger.info("Policy finalize click success -> confirm popup")
            memory.begin_stage("confirm_policy_popup")
            memory.capture_checkpoint()
            return

    def on_actions_success(self, memory: ShortTermMemory, actions: list[AgentAction]) -> None:
        if memory.current_stage != "plan_current_tab" or any(action.action != "drag" for action in actions):
            return super().on_actions_success(memory, actions)

        current_tab = memory.get_policy_current_tab_name()
        if not current_tab:
            return

        memory.set_policy_bundle_action_count(len(actions))
        for action in actions:
            memory.mark_policy_slot_selected(
                card_name=action.policy_card_name,
                source_tab=action.policy_source_tab or current_tab,
                target_slot_id=action.policy_target_slot_id,
                reasoning=action.policy_reasoning,
            )
        next_tab = memory.get_policy_next_tab_name()
        if next_tab:
            self._complete_current_policy_tab(memory)
            memory.begin_stage("click_next_tab")
            memory.set_policy_event(f"drag complete={current_tab} -> advance to {next_tab} | popped={current_tab}")
            logger.info("Policy drag complete | tab=%s -> advance to %s", current_tab, next_tab)
        else:
            self._complete_current_policy_tab(memory)
            memory.begin_stage("finalize_policy")
            memory.set_policy_event(f"drag complete={current_tab} -> finalize | popped={current_tab}")
            logger.info("Policy drag complete -> finalize | tab=%s", current_tab)
        memory.capture_checkpoint()

    def handle_no_progress(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        last_action: AgentAction,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> NoProgressResolution:
        if memory.current_stage in {"calibrate_tabs", "click_cached_tab", "click_next_tab"}:
            if memory.current_stage == "calibrate_tabs":
                target_tab = memory.get_policy_calibration_target_name()
                retry_stage = "calibrate_tabs"
                retry_key = f"calibrate_tabs:{target_tab}" if target_tab else "calibrate_tabs"
                retry_prefix = "calibration"
                preserve_progress = False
                next_mode = "calibrating"
            elif memory.current_stage == "click_cached_tab":
                target_tab = memory.get_policy_current_tab_name()
                retry_stage = "click_cached_tab"
                retry_key = self.get_recovery_key(memory)
                retry_prefix = "tab click"
                preserve_progress = True
                next_mode = "structured"
            else:
                target_tab = memory.get_policy_current_tab_name()
                retry_stage = "click_cached_tab"
                retry_key = self.get_recovery_key(memory, stage_name="click_cached_tab")
                retry_prefix = "tab click"
                preserve_progress = True
                next_mode = "structured"

            if not target_tab:
                return NoProgressResolution()

            memory.record_policy_failed_tab(target_tab)
            if len(memory.policy_state.distinct_failed_tabs) >= 2:
                self._restart_policy_calibration(
                    memory,
                    preserve_progress=preserve_progress,
                    reason="distinct tab failures -> full recalibration",
                )
                return NoProgressResolution(handled=True)

            def _relocalize_and_retry_once() -> None:
                relocalized = self._relocalize_failed_tab(
                    provider,
                    pil_image,
                    memory,
                    target_tab,
                    normalizing_range=normalizing_range,
                    img_config=img_config,
                )
                memory.set_policy_mode(next_mode)
                memory.set_policy_event(
                    f"{retry_prefix} retry={target_tab} relocalized={'yes' if relocalized else 'no'}"
                )
                memory.begin_stage(retry_stage)
                logger.info(
                    "Policy stage %s failed for tab '%s'; relocalize attempted=%s and will retry",
                    retry_stage,
                    target_tab,
                    relocalized,
                )

            return self.retry_fallback_hook.handle_failure(
                memory,
                stage_name=retry_stage,
                stage_key=retry_key,
                on_first_retry=_relocalize_and_retry_once,
                reroute_message=f"Policy tab '{target_tab}' failed at stage '{retry_stage}' after retry+fallback",
            )

        if memory.current_stage == "generic_fallback":
            restart_key = (
                f"policy_restart:{memory.fallback_return_stage or memory.get_policy_current_tab_name() or 'policy'}"
            )
            if not memory.has_policy_generic_fallback_used(restart_key):
                memory.mark_policy_generic_fallback_used(restart_key)
                preserve_entry_done = memory.is_policy_entry_done()
                memory.clear_fallback_return_stage()
                memory.clear_policy_bootstrap(
                    preserve_entry_done=preserve_entry_done,
                    preserve_progress=True,
                )
                memory.reset_stage_failure("bootstrap_tabs")
                memory.clear_stage_fallback_used("bootstrap_tabs")
                memory.set_policy_mode("structured")
                memory.set_policy_event("generic fallback no-progress -> restart same primitive")
                memory.begin_stage("bootstrap_tabs" if preserve_entry_done else "policy_entry")
                logger.info(
                    "Policy generic fallback made no progress -> restarting same primitive from %s",
                    memory.current_stage,
                )
                return NoProgressResolution(handled=True)
            memory.set_policy_event("generic fallback exhausted -> reroute")
            return NoProgressResolution(
                handled=False,
                reroute=True,
                error_message="Policy generic fallback produced no progress after same-primitive restart",
            )

        return super().handle_no_progress(
            provider,
            pil_image,
            memory,
            last_action=last_action,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )


class CityProductionProcess(ObservationAssistedProcess):
    """Observation-assisted production flow with an explicit entry gate."""

    # macOS PyAutoGUI emits wheel scrolls as many 10-line events; keep each
    # deterministic burst small so the Civ list does not overshoot or queue.
    _ANCHOR_SCROLL_DELTA = 120
    _ENTRY_SUBSTEP = "production_entry_done"
    _LIST_BRANCH = "choice_list"
    _PLACEMENT_BRANCH = "placement_map"
    _PLACEMENT_STAGE = "production_place"
    _PLACEMENT_RESOLVE_STAGE = "resolve_placement_followup"
    _PLACEMENT_RECLICK_STAGE = "production_place_reclick"
    _PLACEMENT_CONFIRM_STAGE = "production_place_confirm"
    _POST_SELECT_RESOLVE_STAGE = "resolve_post_select_followup"
    _COMPLETE_STAGE = "production_complete"
    _HOVER_SCROLL_STAGE = "hover_scroll_anchor"
    _SCROLL_DOWN_STAGE = "scroll_down_for_hidden_choices"
    _RESTORE_HOVER_STAGE = "restore_hover_scroll_anchor"
    _RESTORE_SCROLL_STAGE = "restore_best_choice_visibility"
    _FOLLOWUP_STATES = {"done", "placement", "confirm", "unknown"}
    _PLACEMENT_FOLLOWUP_STATES = {"placement", "confirm", "unknown"}
    _PLACEMENT_PLAN_ACTIONS = {"click_tile", "click_purchase_button", "unknown"}
    _PLACEMENT_TILE_COLORS = {"green", "blue", "purple", "unknown"}
    _PLACEMENT_RECOVERY_STAGES = {
        _PLACEMENT_STAGE,
        _PLACEMENT_RECLICK_STAGE,
        _PLACEMENT_CONFIRM_STAGE,
        _PLACEMENT_RESOLVE_STAGE,
    }

    def __init__(self, primitive_name: str, completion_condition: str = ""):
        super().__init__(
            primitive_name,
            completion_condition,
            target_description="화면 오른쪽의 세로로 긴 생산 품목 선택 패널",
        )
        self.observer = CityProductionObserver("화면 오른쪽의 세로로 긴 생산 품목 선택 패널")

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("production_entry")

    def get_recovery_key(self, memory: ShortTermMemory, *, stage_name: str | None = None) -> str:
        current_stage = stage_name or memory.current_stage or "step"
        if memory.branch == self._PLACEMENT_BRANCH or current_stage in self._PLACEMENT_RECOVERY_STAGES:
            return f"city_production_placement:{current_stage}"
        return super().get_recovery_key(memory, stage_name=stage_name)

    def should_observe(self, memory: ShortTermMemory) -> bool:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return False
        if memory.branch and memory.branch != self._LIST_BRANCH:
            return False
        if memory.current_stage in {
            "generic_fallback",
            self._HOVER_SCROLL_STAGE,
            self._SCROLL_DOWN_STAGE,
            self._RESTORE_HOVER_STAGE,
            self._RESTORE_SCROLL_STAGE,
            self._POST_SELECT_RESOLVE_STAGE,
            self._COMPLETE_STAGE,
        }:
            return False
        if memory.current_stage == "observe_choices":
            return True
        return super().should_observe(memory)

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return (
                "현재 stage: production_entry\n"
                "- 생산 선택 팝업 또는 배치 화면이 실제로 열렸는지 먼저 확인해.\n"
                "- 우하단 '생산 품목' 알림만 보이면 press enter로 진입해.\n"
                "- 아직 목록 관찰/스크롤/품목 선택을 하지 마."
            )
        if memory.current_stage == self._PLACEMENT_CONFIRM_STAGE:
            return (
                "현재 stage: production_place_confirm\n"
                "- 직전에 고른 배치 타일에 대한 건설/구매 확인 팝업만 처리한다.\n"
                "- '이곳에 ... 을 건설하겠습니까?' 또는 구매 후 건설 확인이면 '예'/확인 버튼만 클릭해.\n"
                "- 클릭 후 primitive는 explicit terminal state로 종료된다. task_status는 in_progress로 유지해도 된다."
            )
        if memory.current_stage == "select_from_memory":
            best_choice = memory.get_best_choice()
            best_label = best_choice.label if best_choice is not None else "-"
            return (
                "현재 stage: select_from_memory\n"
                f"- memory에서 고른 생산 품목 '{best_label}' 을 클릭한다.\n"
                "- 클릭 직후 화면을 한 번 더 판별해 배치/확인/완료 중 어디로 갈지 state machine이 결정한다.\n"
                "- 여기서는 task_status를 complete로 끝내지 마."
            )
        if memory.current_stage == self._POST_SELECT_RESOLVE_STAGE:
            return (
                "현재 stage: resolve_post_select_followup\n"
                "- 방금 생산 품목 클릭 이후 화면이 배치 화면인지, 확인 팝업인지, 바로 완료인지 판별만 한다.\n"
                "- 추가 클릭/스크롤은 하지 마."
            )
        if memory.current_stage == self._COMPLETE_STAGE:
            return "현재 stage: production_complete\n- city production primitive의 explicit terminal state다."
        if memory.current_stage == self._PLACEMENT_RECLICK_STAGE:
            return (
                "현재 stage: production_place_reclick\n"
                "- 방금 골드 배지로 구매한 같은 타일 본체를 다시 클릭해 실제 건설 배치를 이어간다.\n"
                "- 다른 타일을 새로 고르지 말고, 저장된 같은 타일을 다시 클릭한다."
            )
        if memory.current_stage == self._PLACEMENT_RESOLVE_STAGE:
            return (
                "현재 stage: resolve_placement_followup\n"
                "- 직전 배치 타일 클릭 이후 화면이 아직 배치 화면인지, 확인 팝업인지 짧게 판별한다.\n"
                "- 파란색/보라색 구매형 타일 구매 후 배치 화면이 남아 있으면 같은 타일 재클릭 단계로 간다."
            )
        if memory.branch == self._PLACEMENT_BRANCH:
            return (
                "현재 stage: production_place\n"
                "- 지금은 스크롤 목록이 아니라 특수지구/불가사의 배치 화면이다.\n"
                "- 특수지구를 배치하더라도 캠퍼스를 기본값처럼 고르지 마.\n"
                "- 반드시 현재 high_level_strategy와 도시의 실제 상황을 함께 보고 어떤 지구/건물이 맞는지 결정해.\n"
                "- 현재 보유 골드와 파란색/보라색 타일에 표시된 구매 골드를 비교해, "
                "실제로 지불 가능한 경우에만 구매형 타일을 고른다.\n"
                "- 초록색 즉시 배치 가능 타일과 파란색/보라색 구매 후 배치 가능 타일을 비교할 때 인접 보너스, "
                "지형 시너지, 상위 전략 목표를 함께 고려한다.\n"
                "- 구매형 타일은 타일 위에 있는 골드와 숫자가 있는 구매 버튼/배지를 먼저 클릭한다.\n"
                "- 파란색/보라색 구매형 타일을 구매한 뒤에도 배치 화면이 유지되면 "
                "같은 타일 본체를 다시 클릭해 실제 배치를 이어간다.\n"
                "- 타일 선택 action을 바로 1회 수행하고, 아직 task_status를 complete로 끝내지 마.\n"
                "- 생산 목록 관찰/스크롤은 하지 마."
            )
        if memory.current_stage == self._HOVER_SCROLL_STAGE:
            return (
                "현재 stage: hover_scroll_anchor\n"
                "- 방금 찾은 오른쪽 생산 목록 패널 중앙으로 커서만 이동해 hover를 고정한다.\n"
                "- 클릭하지 말고, 다음 단계에서만 스크롤한다."
            )
        if memory.current_stage == self._SCROLL_DOWN_STAGE:
            return (
                "현재 stage: scroll_down_for_hidden_choices\n"
                "- 이미 hover된 생산 목록 패널 중앙에서 아래로 스크롤해 숨은 선택지를 더 본다."
            )
        if memory.current_stage == self._RESTORE_HOVER_STAGE:
            return (
                "현재 stage: restore_hover_scroll_anchor\n"
                "- 선택한 생산 품목이 현재 안 보인다. 생산 목록 패널 중앙에 다시 hover를 고정한다."
            )
        if memory.current_stage == self._RESTORE_SCROLL_STAGE:
            best_choice = memory.get_best_choice()
            best_label = best_choice.label if best_choice is not None else "-"
            return (
                "현재 stage: restore_best_choice_visibility\n"
                f"- 선택한 생산 품목 '{best_label}' 이 다시 보이도록 패널을 재복원 스크롤한다.\n"
                "- 스크롤 후에는 다시 observation으로 돌아가 실제로 보이는지 확인한다."
            )
        return super().build_stage_note(memory)

    @staticmethod
    def _is_selectable_visible_option(raw: dict) -> bool:
        """Whether one observed production row is currently actionable."""
        selected = bool(raw.get("selected", False))
        return not bool(raw.get("disabled", False)) and not selected and not bool(raw.get("built", selected))

    def _effective_progress_stage(self, memory: ShortTermMemory) -> str:
        """Resolve fallback/retry stages back to the user-facing main stage bucket."""
        if memory.current_stage == "generic_fallback" and memory.fallback_return_stage:
            return memory.fallback_return_stage
        return memory.current_stage or "production_entry"

    def _entered_from_choice_list(self, memory: ShortTermMemory) -> bool:
        """Whether the current placement/confirm flow originated from the list branch."""
        return bool(
            memory.choice_catalog.candidates
            or memory.choice_catalog.best_option_id
            or "full_scan_complete" in memory.completed_substeps
        )

    def _is_purchase_reclick_flow(self, memory: ShortTermMemory) -> bool:
        """Whether the placement branch currently includes a purchase-button reclick step."""
        stage = self._effective_progress_stage(memory)
        return (
            memory.city_placement_state.target_origin == "purchase_button"
            or memory.city_placement_state.reclick_attempts > 0
            or stage == self._PLACEMENT_RECLICK_STAGE
        )

    def get_visible_progress(
        self,
        memory: ShortTermMemory,
        *,
        executed_steps: int,
        hard_max_steps: int,
    ) -> tuple[int, int]:
        """Return branch-aware stage progress instead of raw action counts."""
        del executed_steps, hard_max_steps

        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return 1, 2

        stage = self._effective_progress_stage(memory)
        purchase_reclick_flow = self._is_purchase_reclick_flow(memory)
        from_choice_list = self._entered_from_choice_list(memory)

        if stage == self._PLACEMENT_CONFIRM_STAGE:
            if from_choice_list and memory.branch == self._LIST_BRANCH:
                return 5, 5
            if from_choice_list:
                return (7, 7) if purchase_reclick_flow else (6, 6)
            return (4, 4) if purchase_reclick_flow else (3, 3)

        if memory.branch == self._PLACEMENT_BRANCH:
            if stage in {self._PLACEMENT_STAGE, self._PLACEMENT_RESOLVE_STAGE}:
                if from_choice_list:
                    return 5, 7 if purchase_reclick_flow else 6
                return 2, 4 if purchase_reclick_flow else 3
            if stage == self._PLACEMENT_RECLICK_STAGE:
                if from_choice_list:
                    return 6, 7
                return 3, 4

        if memory.branch == self._LIST_BRANCH:
            if stage in {self._HOVER_SCROLL_STAGE, self._SCROLL_DOWN_STAGE}:
                return 2, 4
            if stage == "observe_choices":
                if memory.get_best_choice() is not None:
                    return 4, 4
                if memory.choice_catalog.end_reached:
                    return 3, 4
                return 2, 4
            if stage in {"choose_from_memory", "decide_best_choice"}:
                return 3, 4
            if stage in {
                self._RESTORE_HOVER_STAGE,
                self._RESTORE_SCROLL_STAGE,
                "select_from_memory",
                self._POST_SELECT_RESOLVE_STAGE,
                self._COMPLETE_STAGE,
            }:
                return 4, 4

        return super().get_visible_progress(memory, executed_steps=0, hard_max_steps=1)

    def _production_screen_state(self, provider: BaseVLMProvider, pil_image, *, img_config=None) -> dict | None:
        prompt = (
            "너는 문명6 도시 생산 진입 상태 판별기야. 현재 화면이 실제 생산 선택 화면인지 여부만 판단해.\n"
            'JSON만 출력: {"production_mode":"list|placement|notification|other",'
            ' "production_screen_ready": true/false,'
            ' "notification_visible": true/false, "reasoning": "짧은 이유"}\n'
            "- 생산 품목 목록(건물/유닛/지구 리스트)이 실제로 보이면 "
            "production_mode='list', production_screen_ready=true.\n"
            "- 특수지구/불가사의 배치 타일 화면이면 "
            "production_mode='placement', production_screen_ready=true.\n"
            "- 우하단 '생산 품목' 알림만 보이고 목록/배치 화면이 안 열렸으면 "
            "production_mode='notification', production_screen_ready=false.\n"
            "- 어떤 생산 UI도 확실하지 않으면 production_mode='other', production_screen_ready=false.\n"
            "- 우하단 '생산 품목' 알림이 분명히 보이면 notification_visible=true.\n"
        )
        try:
            return _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Production entry check failed: %s", exc)
            return None

    @staticmethod
    def _followup_img_config(img_config=None):
        return PRESETS.get(
            "city_production_followup_fast",
            img_config,
        )

    @staticmethod
    def _placement_img_config(img_config=None):
        return PRESETS.get(
            "city_production_placement_fast",
            img_config,
        )

    @staticmethod
    def _legacy_coord_scale_factor(normalizing_range: int) -> int | None:
        """Return legacy 0-1000 scale factor when runtime range is a clean multiple."""
        if normalizing_range <= 1000 or normalizing_range % 1000 != 0:
            return None
        factor = normalizing_range // 1000
        return factor if factor > 1 else None

    def _detect_post_select_followup(
        self,
        provider: BaseVLMProvider,
        pil_image,
        *,
        img_config=None,
    ) -> tuple[str, str]:
        """Classify whether a clicked production choice needs another city-production step."""
        prompt = (
            "문명6 도시 생산 후속상태 분류기다. JSON만 출력해.\n"
            '{"post_select_state":"done|placement|confirm|unknown","reason":"짧게"}\n'
            "- 방금 생산품목 선택 직후 화면이다.\n"
            "- 지도 위 초록/파랑 타일 배치 화면이면 placement.\n"
            "- '이곳에 ... 을 건설하겠습니까?' 같은 확인 팝업이면 confirm.\n"
            "- 추가 단계 없이 생산 선택이 끝났으면 done.\n"
            "- 확실하지 않으면 unknown.\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                pil_image,
                prompt,
                img_config=self._followup_img_config(img_config),
                max_tokens=128,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Production post-select follow-up detection failed: %s", exc)
            return "unknown", "post-select detection failed"
        state = str(data.get("post_select_state", "")).strip().lower()
        if state not in self._FOLLOWUP_STATES:
            state = "unknown"
        return state, str(data.get("reason", "")).strip()

    def _detect_placement_followup(
        self,
        provider: BaseVLMProvider,
        pil_image,
        *,
        img_config=None,
    ) -> tuple[str, str]:
        """Classify whether a placement click needs a same-tile re-click or confirmation."""
        prompt = (
            "문명6 도시 생산 배치 후속상태 분류기다. JSON만 출력해.\n"
            '{"placement_followup_state":"placement|confirm|unknown","reason":"짧게"}\n'
            "- 방금 배치 타일을 클릭한 직후 화면이다.\n"
            "- 특수지구/불가사의 배치 지도 화면이 그대로 남아 있으면 placement.\n"
            "- '이곳에 ... 을 건설하겠습니까?' 같은 건설/구매 확인 팝업이면 confirm.\n"
            "- 확실하지 않으면 unknown.\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                pil_image,
                prompt,
                img_config=self._followup_img_config(img_config),
                max_tokens=128,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Production placement follow-up detection failed: %s", exc)
            return "unknown", "placement follow-up detection failed"
        state = str(data.get("placement_followup_state", "")).strip().lower()
        if state not in self._PLACEMENT_FOLLOWUP_STATES:
            state = "unknown"
        return state, str(data.get("reason", "")).strip()

    def _plan_placement_click(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        img_config=None,
    ) -> AgentAction | None:
        """Plan one city-production placement click with explicit purchase-button support."""
        prompt = (
            "너는 문명6 도시 생산 배치 클릭 계획기다. JSON만 출력해.\n"
            "{"
            '"placement_action":"click_tile|click_purchase_button|unknown",'
            '"x":0,"y":0,"button":"right",'
            '"tile_x":0,"tile_y":0,"tile_button":"right",'
            '"tile_color":"green|blue|purple|unknown",'
            '"reason":"짧게"}\n'
            "- 지금 화면은 특수지구/불가사의/건물 배치 지도다.\n"
            "- 초록색 타일은 즉시 배치 가능 타일이다.\n"
            "- 파란색/보라색 타일은 타일 위에 있는 골드와 숫자가 있는 "
            "구매 버튼/배지를 먼저 눌러야 하는 구매형 타일이다.\n"
            "- 구매형 타일을 고르면 placement_action='click_purchase_button' 으로 하고, "
            "x/y 는 골드+숫자 구매 버튼 중심을 반환해.\n"
            "- 구매형 타일을 고를 때 tile_x/tile_y 는 같은 타일 본체 중심을 반환해. "
            "구매 후 배치 화면이 남으면 그 좌표를 다시 클릭한다.\n"
            "- 초록 타일을 고를 때는 placement_action='click_tile' 로 하고, x/y 는 타일 본체 중심을 반환해.\n"
            "- 현재 보유 골드, 타일 구매 비용, 인접 보너스, 지형 시너지, 상위 전략을 함께 고려해.\n"
            "- 구매형 타일은 현재 골드로 실제 구매 가능하고, 초록 타일보다 확실히 유리할 때만 선택해.\n"
            "- 확실하지 않으면 placement_action='unknown'.\n"
            f"{_normalized_coord_note(normalizing_range, fields='x/y 와 tile_x/tile_y')}\n"
            f"- 상위 전략 참고:\n{high_level_strategy}\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                pil_image,
                prompt,
                img_config=self._placement_img_config(img_config),
                max_tokens=256,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Production placement planning failed: %s", exc)
            return None

        action_kind = str(data.get("placement_action", "")).strip().lower()
        if action_kind not in self._PLACEMENT_PLAN_ACTIONS or action_kind == "unknown":
            return None

        try:
            x = int(data.get("x", 0))
            y = int(data.get("y", 0))
        except (TypeError, ValueError):
            return None
        if not (0 <= x <= normalizing_range and 0 <= y <= normalizing_range):
            return None

        button = str(data.get("button", "right")).strip().lower() or "right"
        if button not in {"left", "right"}:
            button = "right"
        tile_color = str(data.get("tile_color", "")).strip().lower()
        if tile_color not in self._PLACEMENT_TILE_COLORS:
            tile_color = "unknown"
        reason = str(data.get("reason", "")).strip()
        scale_factor = self._legacy_coord_scale_factor(normalizing_range)
        scale_note = ""

        if action_kind == "click_purchase_button":
            try:
                tile_x = int(data.get("tile_x", 0))
                tile_y = int(data.get("tile_y", 0))
            except (TypeError, ValueError):
                return None
            if not (0 <= tile_x <= normalizing_range and 0 <= tile_y <= normalizing_range):
                return None
            tile_button = str(data.get("tile_button", "right")).strip().lower() or "right"
            if tile_button not in {"left", "right"}:
                tile_button = "right"
            if (
                scale_factor is not None
                and 0 <= x <= 1000
                and 0 <= y <= 1000
                and 0 <= tile_x <= 1000
                and 0 <= tile_y <= 1000
            ):
                x *= scale_factor
                y *= scale_factor
                tile_x *= scale_factor
                tile_y *= scale_factor
                scale_note = f" | coord_scale=legacy1000x{scale_factor}"
            memory.remember_city_placement_target(
                x=tile_x,
                y=tile_y,
                button=tile_button,
                reason=reason or "구매형 타일 후속 배치",
                origin="purchase_button",
                tile_color=tile_color,
            )
            action_reason = reason or "구매형 타일의 골드 버튼을 먼저 눌러 배치를 준비"
        else:
            if scale_factor is not None and 0 <= x <= 1000 and 0 <= y <= 1000:
                x *= scale_factor
                y *= scale_factor
                scale_note = f" | coord_scale=legacy1000x{scale_factor}"
            memory.clear_city_placement_target()
            action_reason = reason or "즉시 배치 가능한 타일 본체를 클릭"

        action = AgentAction(
            action="click",
            x=x,
            y=y,
            button=button,
            reasoning=action_reason,
            task_status="in_progress",
        )
        memory.set_last_planned_action_debug(f"{self._format_action_debug(action)}{scale_note}")
        return action

    def _build_saved_placement_reclick_action(self, memory: ShortTermMemory) -> AgentAction | None:
        """Re-click the saved tile body after a purchasable-tile purchase."""
        target = memory.get_city_placement_target()
        if target is None:
            return None
        x, y, button = target
        action = AgentAction(
            action="click",
            x=x,
            y=y,
            button=button,
            reasoning="방금 골드 버튼으로 구매한 같은 타일 본체를 다시 클릭해 실제 건설 배치를 이어감",
            task_status="in_progress",
        )
        memory.set_last_planned_action_debug(self._format_action_debug(action))
        return action

    @staticmethod
    def _anchor_components(
        scroll_anchor: dict | ScrollAnchor | None,
        *,
        normalizing_range: int,
    ) -> tuple[int, int, int, int, int, int] | None:
        """Return validated normalized anchor coordinates for dict or memory-backed anchors."""
        if isinstance(scroll_anchor, dict):
            getter = scroll_anchor.get
        elif isinstance(scroll_anchor, ScrollAnchor):

            def getter(key: str, default: int = 0) -> int:
                return getattr(scroll_anchor, key, default)
        else:
            return None
        try:
            x = int(getter("x", 0))
            y = int(getter("y", 0))
            left = int(getter("left", 0))
            top = int(getter("top", 0))
            right = int(getter("right", normalizing_range))
            bottom = int(getter("bottom", normalizing_range))
        except (TypeError, ValueError):
            return None
        if 0 <= left <= x <= right <= normalizing_range and 0 <= top <= y <= bottom <= normalizing_range:
            return x, y, left, top, right, bottom
        return None

    @classmethod
    def _is_valid_anchor_dict(cls, scroll_anchor: dict | None, *, normalizing_range: int) -> bool:
        """Return whether a raw anchor dict fits the normalized coordinate contract."""
        return cls._anchor_components(scroll_anchor, normalizing_range=normalizing_range) is not None

    @classmethod
    def _is_plausible_list_anchor(
        cls,
        scroll_anchor: dict | ScrollAnchor | None,
        *,
        normalizing_range: int,
    ) -> bool:
        """Reject anchors that do not look like the right-side tall production list panel."""
        components = cls._anchor_components(scroll_anchor, normalizing_range=normalizing_range)
        if components is None:
            return False
        x, _, left, top, right, bottom = components
        width = right - left
        height = bottom - top
        return (
            x >= round(normalizing_range * 0.55)
            and left >= round(normalizing_range * 0.40)
            and width >= round(normalizing_range * 0.15)
            and height >= round(normalizing_range * 0.45)
            and height >= width
        )

    @staticmethod
    def _ratio_to_norm(value: float, normalizing_range: int) -> int:
        """Convert a UI ratio into a normalized coordinate."""
        return round(value * normalizing_range)

    def _default_list_scroll_anchor(self, normalizing_range: int) -> dict[str, int]:
        """Return a conservative fallback anchor inside the typical production popup."""
        left = self._ratio_to_norm(_PRODUCTION_LIST_DEFAULT_RATIOS[0], normalizing_range)
        top = self._ratio_to_norm(_PRODUCTION_LIST_DEFAULT_RATIOS[1], normalizing_range)
        right = self._ratio_to_norm(_PRODUCTION_LIST_DEFAULT_RATIOS[2], normalizing_range)
        bottom = self._ratio_to_norm(_PRODUCTION_LIST_DEFAULT_RATIOS[3], normalizing_range)
        hover_anchor = self._project_anchor_to_right_hover_lane(
            {
                "x": (left + right) // 2,
                "y": (top + bottom) // 2,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            normalizing_range=normalizing_range,
        )
        return {
            "x": hover_anchor.x,
            "y": hover_anchor.y,
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        }

    @classmethod
    def _project_anchor_to_right_hover_lane(
        cls,
        scroll_anchor: dict | ScrollAnchor,
        *,
        normalizing_range: int,
    ) -> ScrollAnchor:
        """Project a validated production anchor onto a reliable right-edge hover lane."""
        components = cls._anchor_components(scroll_anchor, normalizing_range=normalizing_range)
        if components is None:
            raise ValueError("scroll_anchor must be validated before projection")
        _, y, left, top, right, bottom = components
        width = right - left
        inset = max(round(normalizing_range * _PRODUCTION_LIST_HOVER_RIGHT_INSET_RATIO), 12)
        preferred_x = max(
            round(normalizing_range * _PRODUCTION_LIST_HOVER_X_RATIO),
            left + round(width * _PRODUCTION_LIST_HOVER_WIDTH_BIAS),
        )
        x = min(right - inset, preferred_x)
        x = max(left + inset, x)
        y = max(top + inset, min(bottom - inset, y))
        return ScrollAnchor(x=x, y=y, left=left, top=top, right=right, bottom=bottom)

    def _get_runtime_scroll_anchor(self, memory: ShortTermMemory) -> ScrollAnchor:
        """Return a production-list anchor, repairing invalid memory with the safe right-side fallback."""
        anchor = memory.get_scroll_anchor()
        if self._is_plausible_list_anchor(anchor, normalizing_range=memory.normalizing_range):
            projected_anchor = self._project_anchor_to_right_hover_lane(
                anchor, normalizing_range=memory.normalizing_range
            )
            memory.choice_catalog.scroll_anchor = projected_anchor
            return projected_anchor
        default_anchor = ScrollAnchor(**self._default_list_scroll_anchor(memory.normalizing_range))
        memory.choice_catalog.scroll_anchor = default_anchor
        return default_anchor

    def _locate_list_scroll_anchor(
        self,
        provider: BaseVLMProvider,
        pil_image,
        *,
        normalizing_range: int,
        img_config=None,
    ) -> dict | None:
        """Locate the right-side production-list hover anchor with a dedicated high-precision pass."""
        prompt = f"""너는 문명6 도시 생산 목록의 scroll hover anchor 위치만 찾는 서브에이전트야.
응답은 JSON 하나만 출력해.
{{
  "scroll_anchor": {{
    "x": 0, "y": 0,
    "left": 0, "top": 0, "right": {normalizing_range}, "bottom": {normalizing_range}
  }},
  "reasoning": "짧은 이유"
}}
- 생산 목록이 실제로 보이면 화면 오른쪽에 세로로 길게 있는 패널 내부 중앙을 scroll_anchor로 반환해.
- 지도 육각형, 좌측 빈 영역, 우측 HUD, 우하단 버튼/알림은 절대 scroll_anchor로 반환하지 마.
- 생산 목록이 확실하지 않으면 scroll_anchor는 null 로 반환해.
{_normalized_coord_note(normalizing_range, fields="scroll_anchor.x/y 와 scroll_anchor.left/top/right/bottom")}
"""
        try:
            data = _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Production scroll-anchor locator failed: %s", exc)
            return None
        scroll_anchor = data.get("scroll_anchor")
        if self._is_plausible_list_anchor(scroll_anchor, normalizing_range=normalizing_range):
            return scroll_anchor
        return None

    @staticmethod
    def _summarize_visible_options(observation: ObservationBundle) -> str:
        """Return a compact debug summary for the current visible production options."""
        labels = [
            str(item.get("label", "")).strip()
            for item in observation.visible_options
            if (str(item.get("label", "")).strip() and CityProductionProcess._is_selectable_visible_option(item))
        ]
        total_selectable = len(labels)
        ignored_count = max(0, len(observation.visible_options) - len(labels))
        labels = labels[:5]
        prefix = ", ".join(labels) if labels else "보이는 활성 품목 없음"
        if total_selectable > len(labels):
            prefix = f"{prefix} 외 {total_selectable - len(labels)}개"
        if ignored_count:
            prefix = f"{prefix} / 제외 {ignored_count}개"
        return f"{total_selectable} selectable visible / end={observation.end_of_list} / {prefix}"

    @staticmethod
    def _format_action_debug(action: AgentAction) -> str:
        """Return a compact one-line action summary for Rich/status debug."""
        parts = [action.action]
        if action.action in {"click", "double_click", "scroll"}:
            parts.append(f"@ ({action.x}, {action.y})")
        if action.action == "drag":
            parts.append(f"@ ({action.x}, {action.y}) -> ({action.end_x}, {action.end_y})")
        if action.action == "scroll":
            parts.append(f"amount={action.scroll_amount}")
        if action.action == "press" and action.key:
            parts.append(f"key={action.key}")
        if action.reasoning:
            parts.append(action.reasoning[:120])
        return " | ".join(parts)

    def _queue_generic_fallback(self, memory: ShortTermMemory, *, reason: str) -> None:
        """Queue a safe generic fallback when deterministic anchor-based flow cannot proceed."""
        memory.begin_stage("generic_fallback")
        memory.set_last_planned_action_debug(reason)

    def _build_anchor_move_action(self, memory: ShortTermMemory, *, stage_name: str, reason: str) -> AgentAction | None:
        """Return a hover-only cursor move to the saved list anchor."""
        anchor = self._get_runtime_scroll_anchor(memory)
        memory.begin_stage(stage_name)
        action = AgentAction(
            action="move",
            x=anchor.x,
            y=anchor.y,
            reasoning=reason,
            task_status="in_progress",
        )
        memory.set_last_planned_action_debug(self._format_action_debug(action))
        return action

    def _build_anchor_scroll_action(
        self,
        memory: ShortTermMemory,
        *,
        direction: str,
        reason: str,
    ) -> AgentAction | None:
        """Return a deterministic scroll action from the saved list anchor."""
        anchor = self._get_runtime_scroll_anchor(memory)
        memory.choice_catalog.last_scroll_direction = direction
        action = AgentAction(
            action="scroll",
            x=anchor.x,
            y=anchor.y,
            scroll_amount=-self._ANCHOR_SCROLL_DELTA if direction == "down" else self._ANCHOR_SCROLL_DELTA,
            reasoning=reason,
            task_status="in_progress",
        )
        memory.set_last_planned_action_debug(self._format_action_debug(action))
        return action

    def _build_restore_scroll_action(self, memory: ShortTermMemory) -> AgentAction | None:
        """Return a deterministic scroll that restores the chosen option into view."""
        best_choice = memory.get_best_choice()
        if best_choice is None:
            self._queue_generic_fallback(memory, reason="best choice missing during restore -> generic_fallback")
            return None
        if best_choice.position_hint == "above":
            memory.begin_stage(self._RESTORE_SCROLL_STAGE)
            return self._build_anchor_scroll_action(
                memory,
                direction="up",
                reason=f"선택한 생산 품목 '{best_choice.label}' 이 다시 보이도록 위로 재복원 스크롤",
            )
        if best_choice.position_hint == "below":
            memory.begin_stage(self._RESTORE_SCROLL_STAGE)
            return self._build_anchor_scroll_action(
                memory,
                direction="down",
                reason=f"선택한 생산 품목 '{best_choice.label}' 이 다시 보이도록 아래로 재복원 스크롤",
            )
        self._queue_generic_fallback(memory, reason="best choice visibility unknown -> generic_fallback")
        return None

    @staticmethod
    def _observation_option_signature(visible_options: list[dict]) -> tuple[str, ...]:
        """Build a stable signature for the currently visible production options."""
        signature: list[str] = []
        for raw in visible_options:
            if not CityProductionProcess._is_selectable_visible_option(raw):
                continue
            label = str(raw.get("label", "")).strip()
            if not label:
                continue
            signature.append(str(raw.get("id", "")).strip() or label)
        return tuple(signature)

    def _verify_scroll_progress(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        """Verify that a city-production scroll changed the visible list viewport."""
        prompt = self.observer.build_prompt(
            self.primitive_name,
            memory,
            normalizing_range=memory.normalizing_range,
        )
        effective_img_config = (
            PRESETS.get("observation_fast") if img_config is None else PRESETS.get("observation_fast", img_config)
        )
        observation = self.observer.observe(provider, pil_image, prompt, img_config=effective_img_config)
        if observation is None:
            return SemanticVerifyResult(handled=False)

        previous_signature = tuple(memory.choice_catalog.last_visible_option_ids)
        current_signature = self._observation_option_signature(observation.visible_options)
        if observation.end_of_list or current_signature != previous_signature:
            return SemanticVerifyResult(handled=True, passed=True, reason="production scroll changed visible options")

        return SemanticVerifyResult(
            handled=True,
            passed=False,
            reason="스크롤 후에도 같은 선택지가 보여 실제 목록 이동을 확인하지 못함",
        )

    def observe(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        img_config=None,
    ) -> ObservationBundle | None:
        observation = super().observe(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )
        if observation is None or memory.branch != self._LIST_BRANCH:
            return observation

        if not self._is_plausible_list_anchor(observation.scroll_anchor, normalizing_range=normalizing_range):
            observation.scroll_anchor = None

        saved_anchor = memory.get_scroll_anchor()
        if not self._is_plausible_list_anchor(saved_anchor, normalizing_range=normalizing_range):
            saved_anchor = None

        located_anchor = self._locate_list_scroll_anchor(
            provider,
            pil_image,
            normalizing_range=normalizing_range,
            img_config=img_config,
        )
        if located_anchor is not None:
            observation.scroll_anchor = located_anchor
        elif observation.scroll_anchor is None and saved_anchor is None:
            observation.scroll_anchor = self._default_list_scroll_anchor(normalizing_range)
        return observation

    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        effective_observation = observation
        if observation.end_of_list and memory.choice_catalog.downward_scan_scrolls < 2:
            logger.info(
                "Ignoring early city-production end_of_list after %s downward scroll(s)",
                memory.choice_catalog.downward_scan_scrolls,
            )
            effective_observation = copy.copy(observation)
            effective_observation.end_of_list = False

        summary = self._summarize_visible_options(effective_observation)
        debug_anchor = observation.scroll_anchor or memory.get_scroll_anchor()
        if debug_anchor is None:
            debug_anchor = self._default_list_scroll_anchor(memory.normalizing_range)
        memory.set_last_observation_debug(summary, scroll_anchor=debug_anchor)

        memory.begin_stage("observe_choices")
        scroll_direction = memory.choice_catalog.last_scroll_direction or "down"
        memory.remember_choices(
            observation.visible_options,
            end_of_list=effective_observation.end_of_list,
            scroll_anchor=observation.scroll_anchor or debug_anchor,
            scroll_direction=scroll_direction,
        )

        best_choice = memory.get_best_choice()
        if best_choice is not None:
            if best_choice.visible_now:
                memory.begin_stage("select_from_memory")
                memory.set_last_planned_action_debug(f"best choice '{best_choice.label}' visible -> select_from_memory")
                return None
            return self._build_anchor_move_action(
                memory,
                stage_name=self._RESTORE_HOVER_STAGE,
                reason=f"선택한 생산 품목 '{best_choice.label}' 을 다시 찾기 전에 생산 목록 패널 중앙 hover를 고정",
            )

        if effective_observation.end_of_list or memory.choice_catalog.end_reached:
            memory.mark_substep("full_scan_complete")
            memory.begin_stage("choose_from_memory")
            scan_reason = memory.choice_catalog.scan_end_reason or "unknown"
            memory.set_last_planned_action_debug(f"scan complete ({scan_reason}) -> choose_from_memory")
            return None

        return self._build_anchor_move_action(
            memory,
            stage_name=self._HOVER_SCROLL_STAGE,
            reason="생산 목록 패널 중앙으로 커서를 먼저 이동해 hover를 고정",
        )

    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            state = self._production_screen_state(provider, pil_image, img_config=img_config)
            production_mode = str(state.get("production_mode", "")).strip() if state else ""
            if state and bool(state.get("production_screen_ready", False)):
                memory.mark_substep(self._ENTRY_SUBSTEP)
                if production_mode == "placement":
                    memory.set_branch(self._PLACEMENT_BRANCH)
                    memory.begin_stage(self._PLACEMENT_STAGE)
                    planned_placement = self._plan_placement_click(
                        provider,
                        pil_image,
                        memory,
                        normalizing_range=normalizing_range,
                        high_level_strategy=high_level_strategy,
                        img_config=img_config,
                    )
                    if planned_placement is not None:
                        return planned_placement
                    memory.set_last_planned_action_debug(
                        "placement planner returned no safe click -> retry production_place"
                    )
                    return StageTransition(
                        stage=self._PLACEMENT_STAGE,
                        reason="placement planner returned no safe click",
                    )
                memory.set_branch(self._LIST_BRANCH)
                memory.begin_stage("observe_choices")
                return StageTransition(stage="observe_choices", reason="production list screen ready")
            if state and bool(state.get("notification_visible", False)):
                memory.begin_stage("production_entry")
                memory.set_last_planned_action_debug("press enter | lower-right production notification")
                return AgentAction(
                    action="press",
                    key="enter",
                    reasoning="우하단 '생산 품목' 알림을 열어 생산 선택 화면으로 진입",
                    task_status="in_progress",
                )
            memory.begin_stage("production_entry")
            return self._plan_generic_fallback_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=(
                    "지금은 production entry 단계다. 생산 선택 팝업 또는 배치 화면으로 진입하기 위한 "
                    "가장 안전한 단일 action만 수행해. 아직 목록 스캔/스크롤/품목 선택은 하지 마."
                ),
            )

        if memory.branch == self._PLACEMENT_BRANCH:
            if memory.current_stage == self._PLACEMENT_STAGE:
                planned_placement = self._plan_placement_click(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    img_config=img_config,
                )
                if planned_placement is not None:
                    return planned_placement
                memory.set_last_planned_action_debug(
                    "placement planner returned no safe click -> retry production_place"
                )
                return StageTransition(
                    stage=self._PLACEMENT_STAGE,
                    reason="placement planner returned no safe click",
                )
            if memory.current_stage == self._PLACEMENT_RESOLVE_STAGE:
                followup_state, reason = self._detect_placement_followup(
                    provider,
                    pil_image,
                    img_config=img_config,
                )
                if followup_state == "confirm":
                    memory.begin_stage(self._PLACEMENT_CONFIRM_STAGE)
                    memory.set_last_planned_action_debug("placement follow-up -> production_place_confirm")
                    return StageTransition(
                        stage=self._PLACEMENT_CONFIRM_STAGE,
                        reason=reason or "placement follow-up: confirm",
                    )
                if followup_state == "placement":
                    if (
                        memory.get_city_placement_target() is not None
                        and memory.city_placement_state.target_origin == "purchase_button"
                        and memory.city_placement_state.reclick_attempts < 1
                    ):
                        memory.bump_city_placement_reclick_attempt()
                        memory.begin_stage(self._PLACEMENT_RECLICK_STAGE)
                        memory.set_last_planned_action_debug("placement follow-up -> production_place_reclick")
                        return StageTransition(
                            stage=self._PLACEMENT_RECLICK_STAGE,
                            reason=reason or "placement follow-up: same tile re-click",
                        )
                    memory.begin_stage(self._PLACEMENT_STAGE)
                    memory.set_last_planned_action_debug("placement follow-up -> production_place")
                    return StageTransition(
                        stage=self._PLACEMENT_STAGE,
                        reason=reason or "placement follow-up: still placement map",
                    )
                memory.begin_stage(self._PLACEMENT_STAGE)
                memory.set_last_planned_action_debug("placement follow-up unknown -> production_place")
                return StageTransition(
                    stage=self._PLACEMENT_STAGE,
                    reason=reason or "placement follow-up: unknown -> re-evaluate placement",
                )

            if memory.current_stage == self._PLACEMENT_RECLICK_STAGE:
                reclick_action = self._build_saved_placement_reclick_action(memory)
                if reclick_action is not None:
                    return reclick_action
                memory.begin_stage(self._PLACEMENT_STAGE)
                return StageTransition(
                    stage=self._PLACEMENT_STAGE,
                    reason="missing saved placement target -> return to placement planning",
                )

        if memory.current_stage == self._PLACEMENT_CONFIRM_STAGE:
            return self._force_task_status(
                self._plan_generic_fallback_action(
                    provider,
                    pil_image,
                    memory,
                    normalizing_range=normalizing_range,
                    high_level_strategy=high_level_strategy,
                    recent_actions=recent_actions,
                    hitl_directive=hitl_directive,
                    img_config=img_config,
                    extra_note=(
                        "방금 도시 생산의 마지막 확인 팝업 단계다. "
                        "지금 보이는 '예' 또는 확인 버튼만 클릭해. "
                        "클릭 성공 후 state machine이 terminal state로 종료를 처리한다. "
                        "다른 목록 스크롤, 품목 선택, 타일 클릭은 하지 마."
                    ),
                ),
                "in_progress",
            )

        if memory.current_stage == self._POST_SELECT_RESOLVE_STAGE:
            followup_state, reason = self._detect_post_select_followup(
                provider,
                pil_image,
                img_config=img_config,
            )
            if followup_state == "placement":
                memory.set_branch(self._PLACEMENT_BRANCH)
                memory.begin_stage(self._PLACEMENT_STAGE)
                memory.set_last_planned_action_debug("post-select follow-up -> production_place")
                return StageTransition(
                    stage=self._PLACEMENT_STAGE,
                    reason=reason or "post-select follow-up: placement",
                )
            if followup_state == "confirm":
                memory.begin_stage(self._PLACEMENT_CONFIRM_STAGE)
                memory.set_last_planned_action_debug("post-select follow-up -> production_place_confirm")
                return StageTransition(
                    stage=self._PLACEMENT_CONFIRM_STAGE,
                    reason=reason or "post-select follow-up: confirm",
                )
            if followup_state == "done":
                memory.begin_stage(self._COMPLETE_STAGE)
                memory.set_last_planned_action_debug("post-select follow-up -> production_complete")
                return StageTransition(
                    stage=self._COMPLETE_STAGE,
                    reason=reason or "post-select follow-up: done",
                )
            self._queue_generic_fallback(memory, reason="post-select follow-up unknown -> generic_fallback")
            return StageTransition(stage="generic_fallback", reason=reason or "post-select follow-up: unknown")

        if memory.branch == self._LIST_BRANCH:
            if memory.current_stage == self._HOVER_SCROLL_STAGE:
                return self._build_anchor_move_action(
                    memory,
                    stage_name=self._HOVER_SCROLL_STAGE,
                    reason="생산 목록 패널 중앙으로 커서를 먼저 이동해 hover를 고정",
                )

            if memory.current_stage == self._SCROLL_DOWN_STAGE:
                memory.begin_stage(self._SCROLL_DOWN_STAGE)
                return self._build_anchor_scroll_action(
                    memory,
                    direction="down",
                    reason="hover된 생산 목록 패널 중앙에서 아래로 스크롤해 숨은 선택지를 확인",
                )

            best_choice = memory.get_best_choice()
            if best_choice is not None and not best_choice.visible_now:
                if memory.current_stage == self._RESTORE_HOVER_STAGE:
                    return self._build_anchor_move_action(
                        memory,
                        stage_name=self._RESTORE_HOVER_STAGE,
                        reason=(
                            f"선택한 생산 품목 '{best_choice.label}' 을 다시 찾기 전에 생산 목록 패널 중앙 hover를 고정"
                        ),
                    )
                if memory.current_stage == self._RESTORE_SCROLL_STAGE:
                    return self._build_restore_scroll_action(memory)
                return self._build_anchor_move_action(
                    memory,
                    stage_name=self._RESTORE_HOVER_STAGE,
                    reason=f"선택한 생산 품목 '{best_choice.label}' 을 다시 찾기 전에 생산 목록 패널 중앙 hover를 고정",
                )

        current_stage = memory.current_stage
        action = super().plan_action(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )
        if current_stage == "select_from_memory":
            return self._force_task_status(action, "in_progress")
        return action

    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        """Treat hover-only cursor moves as successful even when screenshots do not change."""
        if action.action == "move":
            return True
        if action.action == "scroll" and memory.branch == self._LIST_BRANCH:
            return True
        return super().should_verify_action_without_ui_change(memory, action)

    def should_verify_action_after_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        """Skip redundant semantic verification for list scrolls that already produced raw UI change."""
        if action.action == "scroll" and memory.branch == self._LIST_BRANCH:
            return False
        return super().should_verify_action_after_ui_change(memory, action)

    def verify_action_success(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        action: AgentAction,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        """Cursor-only hover moves are intentional non-visual actions."""
        if action.action == "move":
            return SemanticVerifyResult(handled=True, passed=True, reason="hover move is non-visual by design")
        if action.action == "scroll" and memory.branch == self._LIST_BRANCH:
            return self._verify_scroll_progress(
                provider,
                pil_image,
                memory,
                img_config=img_config,
            )
        return super().verify_action_success(provider, pil_image, memory, action, img_config=img_config)

    def verify_completion(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        img_config=None,
    ) -> VerificationResult:
        """City production completes only from explicit terminal states."""
        if self.is_terminal_state(memory):
            return VerificationResult(True, self.terminal_state_reason(memory))
        return VerificationResult(False, f"city production not terminal: {memory.current_stage or '-'}")

    def is_terminal_state(self, memory: ShortTermMemory) -> bool:
        return memory.current_stage == self._COMPLETE_STAGE

    def terminal_state_reason(self, memory: ShortTermMemory) -> str:
        return "city production reached explicit terminal state"

    def handle_no_progress(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        last_action: AgentAction,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> NoProgressResolution:
        if memory.branch == self._PLACEMENT_BRANCH:
            retry_stage = memory.current_stage or self._PLACEMENT_STAGE
            if retry_stage not in self._PLACEMENT_RECOVERY_STAGES:
                retry_stage = self._PLACEMENT_STAGE
            stage_key = self.get_recovery_key(memory, stage_name=retry_stage)
            failures = memory.increment_stage_failure(stage_key)
            if failures <= 1:
                memory.begin_stage(retry_stage)
                memory.set_last_planned_action_debug(f"placement no-progress -> retry {retry_stage}")
                logger.info("City production placement no-progress -> retry stage %s", retry_stage)
                return NoProgressResolution(handled=True)
            return NoProgressResolution(
                handled=False,
                reroute=True,
                error_message=f"City production placement stalled at stage '{retry_stage}'",
            )
        return super().handle_no_progress(
            provider,
            pil_image,
            memory,
            last_action=last_action,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )

    def resolve_action(self, action: AgentAction, memory: ShortTermMemory) -> AgentAction:
        """Keep city-production scrolls bound to a plausible right-side list anchor."""
        if action.action == "scroll":
            if memory.branch != self._LIST_BRANCH:
                return action
            anchor = self._get_runtime_scroll_anchor(memory)
            if not anchor.contains(action.x, action.y) or (action.x == 0 and action.y == 0):
                action.x = anchor.x
                action.y = anchor.y
            return action
        return super().resolve_action(action, memory)

    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        """Advance deterministic hover/scroll stages for city production."""
        if action.action == "move":
            if memory.current_stage == self._HOVER_SCROLL_STAGE:
                memory.begin_stage(self._SCROLL_DOWN_STAGE)
                return
            if memory.current_stage == self._RESTORE_HOVER_STAGE:
                memory.begin_stage(self._RESTORE_SCROLL_STAGE)
                return

        if (
            memory.branch == self._PLACEMENT_BRANCH
            and memory.current_stage == self._PLACEMENT_STAGE
            and action.action in {"click", "double_click"}
        ):
            if not (
                memory.city_placement_state.has_target
                and memory.city_placement_state.target_origin == "purchase_button"
            ):
                memory.remember_city_placement_target(
                    x=action.x,
                    y=action.y,
                    button=action.button or "right",
                    reason=action.reasoning or "",
                    origin="direct_tile",
                )
            memory.begin_stage(self._PLACEMENT_RESOLVE_STAGE)
            return

        if (
            memory.branch == self._PLACEMENT_BRANCH
            and memory.current_stage == self._PLACEMENT_RECLICK_STAGE
            and action.action in {"click", "double_click"}
        ):
            memory.begin_stage(self._PLACEMENT_RESOLVE_STAGE)
            return

        if memory.current_stage == "select_from_memory" and action.action in {"click", "double_click"}:
            memory.begin_stage(self._POST_SELECT_RESOLVE_STAGE)
            return

        if memory.current_stage == self._PLACEMENT_CONFIRM_STAGE and action.action in {"click", "double_click"}:
            memory.begin_stage(self._COMPLETE_STAGE)
            return

        if action.action == "scroll":
            if memory.current_stage == self._SCROLL_DOWN_STAGE:
                memory.register_choice_scroll(direction="down")
                memory.begin_stage("observe_choices")
                return
            if memory.current_stage == self._RESTORE_SCROLL_STAGE:
                memory.register_choice_scroll(direction="up")
                memory.begin_stage("observe_choices")
                return


class GovernorProcess(ObservationAssistedProcess):
    """Observation-assisted governor flow: scan → decide → fixed branch."""

    _ANCHOR_SCROLL_DELTA = 120

    # Entry
    _ENTRY_SUBSTEP = "governor_entry_done"
    _ENTRY_STAGE = "governor_entry"

    # Branches
    _PROMOTE_BRANCH = "governor_promote"
    _APPOINT_BRANCH = "governor_appoint"

    # Observation (reuse ObservationAssistedProcess base stages)
    _HOVER_SCROLL_STAGE = "hover_scroll_anchor"
    _SCROLL_DOWN_STAGE = "scroll_down_for_hidden_choices"
    _RESTORE_HOVER_STAGE = "restore_hover_scroll_anchor"
    _RESTORE_SCROLL_STAGE = "restore_best_choice_visibility"

    # Promote branch (fixed steps)
    _PROMOTE_CLICK = "governor_promote_click"
    _PROMOTE_SELECT = "governor_promote_select"
    _PROMOTE_CONFIRM = "governor_promote_confirm"
    _PROMOTE_POPUP = "governor_promote_popup"
    _EXIT_ESC1 = "governor_exit_esc1"
    _EXIT_ESC2 = "governor_exit_esc2"

    # Appoint branch (fixed steps)
    _APPOINT_CLICK = "governor_appoint_click"
    _APPOINT_CITY = "governor_appoint_city"
    _APPOINT_CONFIRM = "governor_appoint_confirm"

    _PROMOTE_STAGES = {
        "governor_promote_click",
        "governor_promote_select",
        "governor_promote_confirm",
        "governor_promote_popup",
        "governor_exit_esc1",
        "governor_exit_esc2",
    }
    _APPOINT_STAGES = {
        "governor_appoint_click",
        "governor_appoint_city",
        "governor_appoint_confirm",
    }

    def __init__(self, primitive_name: str, completion_condition: str = ""):
        super().__init__(
            primitive_name,
            completion_condition,
            target_description="총독 카드 목록이 표시되는 스크롤 가능 패널",
        )
        self.observer = GovernorObserver("총독 카드 목록이 표시되는 스크롤 가능 패널")

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage(self._ENTRY_STAGE)

    # ------------------------------------------------------------------
    # Observation gating
    # ------------------------------------------------------------------
    def should_observe(self, memory: ShortTermMemory) -> bool:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return False
        if memory.branch:
            return False
        if memory.current_stage in {
            "generic_fallback",
            self._HOVER_SCROLL_STAGE,
            self._SCROLL_DOWN_STAGE,
            self._RESTORE_HOVER_STAGE,
            self._RESTORE_SCROLL_STAGE,
        }:
            return False
        if memory.current_stage == "observe_choices":
            return True
        return super().should_observe(memory)

    # ------------------------------------------------------------------
    # Decision override — also captures action_type (promote/appoint)
    # ------------------------------------------------------------------
    def decide_from_memory(
        self,
        provider: BaseVLMProvider,
        memory: ShortTermMemory,
        *,
        high_level_strategy: str,
    ) -> bool:
        if memory.get_best_choice() is not None:
            return True

        memory.begin_stage("decide_best_choice")
        max_tokens = memory.choice_catalog_decision_max_tokens()
        prompt = (
            "너는 문명6 총독 선택 결정 서브에이전트야. 아래 short-term memory에 누적된 전체 총독 후보 중 "
            "상위 전략에 가장 적합한 하나를 고르고, 진급(promote)인지 임명(appoint)인지도 결정해.\n"
            'JSON: {"best_option_id":"stable_id","action_type":"promote|appoint","reason":"짧은 이유"}\n'
            "- best_option_id는 후보 catalog에 적힌 id를 그대로 복사해.\n"
            "- note에 '진급_가능'이면 action_type='promote', '임명_가능'이면 action_type='appoint'.\n"
            "- 비활성(disabled) 후보는 고르지 마.\n\n"
            f"Primitive: {self.primitive_name}\n"
            f"상위 전략:\n{high_level_strategy}\n\n"
            f"후보 catalog:\n{memory.choice_catalog_decision_prompt()}\n"
        )
        try:
            response = provider.call_vlm(
                prompt=prompt,
                image_path=None,
                temperature=0.2,
                max_tokens=max_tokens,
                use_thinking=False,
            )
            content = strip_markdown(response.content)
            data = json.loads(content)
            option_id = str(data.get("best_option_id", "")).strip()
            reason = str(data.get("reason", "")).strip()
            action_type = str(data.get("action_type", "")).strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Governor best-choice decision failed: %s", exc)
            return False

        if not option_id:
            logger.warning("Governor decide_from_memory: VLM returned empty best_option_id")
            return False

        catalog_ids = list(memory.choice_catalog.candidates.keys())
        logger.info(
            "Governor decide_from_memory: VLM chose id='%s' action_type='%s' | catalog_ids=%s",
            option_id,
            action_type,
            catalog_ids[:10],
        )

        memory.set_best_choice(option_id=option_id, reason=reason)
        if memory.get_best_choice() is None:
            logger.warning(
                "Governor decide_from_memory: set_best_choice failed — id='%s' not in catalog or not selectable",
                option_id,
            )
            # Fallback: try matching by label
            memory.set_best_choice(label=option_id, reason=reason)
            if memory.get_best_choice() is None:
                logger.warning("Governor decide_from_memory: label fallback also failed for '%s'", option_id)
                return False

        # Set branch based on action_type
        if action_type == "appoint":
            memory.set_branch(self._APPOINT_BRANCH)
            memory.begin_stage(self._APPOINT_CLICK)
        else:
            memory.set_branch(self._PROMOTE_BRANCH)
            memory.begin_stage(self._PROMOTE_CLICK)
        return True

    # ------------------------------------------------------------------
    # Entry check (one-time VLM state classification)
    # ------------------------------------------------------------------
    def _governor_entry_check(self, provider: BaseVLMProvider, pil_image, *, img_config=None) -> dict | None:
        prompt = (
            "너는 문명6 총독 화면 진입 상태 판별기야. JSON만 출력해.\n"
            '{"governor_mode":"overview|notification|other",'
            '"governor_screen_ready":true/false,'
            '"notification_visible":true/false,'
            '"reasoning":"짧은 이유"}\n'
            "- 총독 카드 목록과 [임명]/[진급] 버튼이 보이면 "
            "governor_mode='overview', governor_screen_ready=true.\n"
            "- 우하단 '총독 타이틀' 버튼 또는 펜 아이콘만 보이고 실제 총독 화면이 안 열렸으면 "
            "governor_mode='notification', notification_visible=true, governor_screen_ready=false.\n"
            "- 확실하지 않으면 governor_mode='other', governor_screen_ready=false.\n"
        )
        try:
            return _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Governor entry check failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Stage notes
    # ------------------------------------------------------------------
    def build_stage_note(self, memory: ShortTermMemory) -> str:
        stage = memory.current_stage or self._ENTRY_STAGE

        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return (
                "현재 stage: governor_entry\n"
                "- 실제 총독 목록 화면이 열렸는지 먼저 확인해.\n"
                "- 우하단 '총독 타이틀' 버튼 또는 펜 아이콘만 보이면 press enter로 진입해.\n"
                "- 아직 총독 카드, 스킬, 도시 선택을 하지 마."
            )

        # Promote branch stages
        if stage == self._PROMOTE_CLICK:
            best = memory.get_best_choice()
            label = best.label if best else "선택된 총독"
            return f"현재 stage: governor_promote_click\n- memory의 best choice 총독 '{label}'의 [진급] 버튼을 클릭해."
        if stage == self._PROMOTE_SELECT:
            return "현재 stage: governor_promote_select\n- 활성화된(밝은) 진급 스킬 박스 1개만 클릭. [확정] 누르지 마."
        if stage == self._PROMOTE_CONFIRM:
            return "현재 stage: governor_promote_confirm\n- 초록색 [확정] 버튼만 클릭."
        if stage == self._PROMOTE_POPUP:
            return "현재 stage: governor_promote_popup\n- '정말입니까?' 팝업에서 '예' 버튼 클릭."
        if stage in {self._EXIT_ESC1, self._EXIT_ESC2}:
            return f"현재 stage: {stage}\n- ESC 키를 1회 눌러 총독 화면을 닫는다."

        # Appoint branch stages
        if stage == self._APPOINT_CLICK:
            best = memory.get_best_choice()
            label = best.label if best else "선택된 총독"
            return f"현재 stage: governor_appoint_click\n- memory의 best choice 총독 '{label}'의 [임명] 버튼 클릭."
        if stage == self._APPOINT_CITY:
            return "현재 stage: governor_appoint_city\n- 왼쪽 도시 목록에서 사람얼굴 아이콘 없는 도시 클릭."
        if stage == self._APPOINT_CONFIRM:
            return "현재 stage: governor_appoint_confirm\n- 초록색 [배정] 버튼 클릭."

        # Observation/scroll stages
        if stage == self._HOVER_SCROLL_STAGE:
            return "현재 stage: hover_scroll_anchor\n- 총독 카드 리스트 패널 중앙으로 커서만 이동해 hover를 고정한다."
        if stage == self._SCROLL_DOWN_STAGE:
            return (
                "현재 stage: scroll_down_for_hidden_choices\n"
                "- 이미 hover된 총독 카드 리스트 중앙에서 아래로 스크롤해 숨은 총독을 더 본다."
            )
        if stage == self._RESTORE_HOVER_STAGE:
            return (
                "현재 stage: restore_hover_scroll_anchor\n"
                "- 선택한 총독이 현재 안 보인다. 카드 리스트 패널 중앙에 다시 hover를 고정한다."
            )
        if stage == self._RESTORE_SCROLL_STAGE:
            best = memory.get_best_choice()
            label = best.label if best else "-"
            return (
                "현재 stage: restore_best_choice_visibility\n"
                f"- 선택한 총독 '{label}' 이 다시 보이도록 패널을 재복원 스크롤한다."
            )

        return super().build_stage_note(memory)

    def build_generic_fallback_note(self, memory: ShortTermMemory) -> str:
        stage = memory.fallback_return_stage or memory.current_stage or self._ENTRY_STAGE
        return (
            f"현재 멀티스텝 stage '{stage}' 에서 총독 화면 진행이 막혔다. "
            "같은 governor primitive 안에서 화면을 복구하거나 다음 정상 단계로 돌아가기 위한 "
            "가장 안전한 단일 action 1개만 수행해. "
            "검은색/비활성 [확정], [배정] 버튼은 누르지 마."
        )

    # ------------------------------------------------------------------
    # Scroll anchor helpers (simplified vs CityProduction — no right-side constraint)
    # ------------------------------------------------------------------
    def _get_runtime_scroll_anchor(self, memory: ShortTermMemory) -> ScrollAnchor:
        anchor = memory.get_scroll_anchor()
        if anchor is not None:
            return anchor
        nr = memory.normalizing_range
        return ScrollAnchor(x=nr // 2, y=nr // 2, left=nr // 4, top=nr // 4, right=3 * nr // 4, bottom=3 * nr // 4)

    def _build_anchor_move_action(self, memory: ShortTermMemory, *, stage_name: str, reason: str) -> AgentAction:
        anchor = self._get_runtime_scroll_anchor(memory)
        memory.begin_stage(stage_name)
        return AgentAction(
            action="move",
            x=anchor.x,
            y=anchor.y,
            reasoning=reason,
            task_status="in_progress",
        )

    def _build_anchor_scroll_action(
        self,
        memory: ShortTermMemory,
        *,
        direction: str,
        reason: str,
    ) -> AgentAction:
        anchor = self._get_runtime_scroll_anchor(memory)
        memory.choice_catalog.last_scroll_direction = direction
        return AgentAction(
            action="scroll",
            x=anchor.x,
            y=anchor.y,
            scroll_amount=-self._ANCHOR_SCROLL_DELTA if direction == "down" else self._ANCHOR_SCROLL_DELTA,
            reasoning=reason,
            task_status="in_progress",
        )

    def _build_restore_scroll_action(self, memory: ShortTermMemory) -> AgentAction | None:
        best_choice = memory.get_best_choice()
        if best_choice is None:
            return None
        if best_choice.position_hint == "above":
            memory.begin_stage(self._RESTORE_SCROLL_STAGE)
            return self._build_anchor_scroll_action(
                memory,
                direction="up",
                reason=f"선택한 총독 '{best_choice.label}' 이 다시 보이도록 위로 재복원 스크롤",
            )
        if best_choice.position_hint == "below":
            memory.begin_stage(self._RESTORE_SCROLL_STAGE)
            return self._build_anchor_scroll_action(
                memory,
                direction="down",
                reason=f"선택한 총독 '{best_choice.label}' 이 다시 보이도록 아래로 재복원 스크롤",
            )
        return None

    # ------------------------------------------------------------------
    # consume_observation — update memory from observer scan
    # ------------------------------------------------------------------
    def consume_observation(self, memory: ShortTermMemory, observation: ObservationBundle) -> AgentAction | None:
        memory.begin_stage("observe_choices")
        scroll_direction = memory.choice_catalog.last_scroll_direction or "down"
        memory.remember_choices(
            observation.visible_options,
            end_of_list=observation.end_of_list,
            scroll_anchor=observation.scroll_anchor,
            scroll_direction=scroll_direction,
        )

        best_choice = memory.get_best_choice()
        if best_choice is not None:
            if best_choice.visible_now:
                memory.begin_stage("select_from_memory")
                return None
            return self._build_anchor_move_action(
                memory,
                stage_name=self._RESTORE_HOVER_STAGE,
                reason=f"선택한 총독 '{best_choice.label}' 을 다시 찾기 전에 카드 리스트 패널 중앙 hover를 고정",
            )

        if observation.end_of_list or memory.choice_catalog.end_reached:
            memory.mark_substep("full_scan_complete")
            memory.begin_stage("choose_from_memory")
            return None

        return self._build_anchor_move_action(
            memory,
            stage_name=self._HOVER_SCROLL_STAGE,
            reason="총독 카드 리스트 패널 중앙으로 커서를 먼저 이동해 hover를 고정",
        )

    # ------------------------------------------------------------------
    # Plan action — main dispatch
    # ------------------------------------------------------------------
    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        # 1. Entry gate (ONE-TIME VLM state check)
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            state = self._governor_entry_check(provider, pil_image, img_config=img_config)
            if state and bool(state.get("governor_screen_ready", False)):
                memory.mark_substep(self._ENTRY_SUBSTEP)
                memory.begin_stage("observe_choices")
                return StageTransition(stage="observe_choices", reason="governor overview screen ready")
            if state and bool(state.get("notification_visible", False)):
                memory.begin_stage(self._ENTRY_STAGE)
                return AgentAction(
                    action="press",
                    key="enter",
                    reasoning="우하단 '총독 타이틀' 알림을 열어 총독 화면으로 진입",
                    task_status="in_progress",
                )
            memory.begin_stage(self._ENTRY_STAGE)
            return self._plan_generic_fallback_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=(
                    "지금은 governor entry 단계다. 실제 총독 화면으로 진입하기 위한 가장 안전한 단일 action만 수행해. "
                    "아직 총독 카드/진급 스킬/도시 선택/배정 확정은 하지 마."
                ),
            )

        # 2. Promote branch (fixed)
        if memory.branch == self._PROMOTE_BRANCH:
            return self._plan_promote_branch(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
            )

        # 3. Appoint branch (fixed)
        if memory.branch == self._APPOINT_BRANCH:
            return self._plan_appoint_branch(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
            )

        # 4. Observation + scroll-back flow (handled by base + local overrides)
        if memory.current_stage == self._HOVER_SCROLL_STAGE:
            return self._build_anchor_move_action(
                memory,
                stage_name=self._HOVER_SCROLL_STAGE,
                reason="총독 카드 리스트 패널 중앙으로 커서를 먼저 이동해 hover를 고정",
            )
        if memory.current_stage == self._SCROLL_DOWN_STAGE:
            memory.begin_stage(self._SCROLL_DOWN_STAGE)
            return self._build_anchor_scroll_action(
                memory,
                direction="down",
                reason="hover된 총독 카드 리스트 중앙에서 아래로 스크롤해 숨은 총독을 확인",
            )

        best_choice = memory.get_best_choice()
        if best_choice is not None and not best_choice.visible_now:
            if memory.current_stage == self._RESTORE_HOVER_STAGE:
                return self._build_anchor_move_action(
                    memory,
                    stage_name=self._RESTORE_HOVER_STAGE,
                    reason=f"선택한 총독 '{best_choice.label}' 을 다시 찾기 전에 카드 리스트 패널 중앙 hover를 고정",
                )
            if memory.current_stage == self._RESTORE_SCROLL_STAGE:
                restore = self._build_restore_scroll_action(memory)
                if restore is not None:
                    return restore
            return self._build_anchor_move_action(
                memory,
                stage_name=self._RESTORE_HOVER_STAGE,
                reason=f"선택한 총독 '{best_choice.label}' 을 다시 찾기 전에 카드 리스트 패널 중앙 hover를 고정",
            )

        return super().plan_action(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )

    # ------------------------------------------------------------------
    # Promote branch — fixed step sequence
    # ------------------------------------------------------------------
    def _plan_promote_branch(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        stage = memory.current_stage

        # Deterministic ESC stages (no VLM needed)
        if stage == self._EXIT_ESC1:
            return AgentAction(
                action="press",
                key="escape",
                reasoning="진급 완료 후 ESC 1회 — 총독 화면 닫기",
                task_status="in_progress",
            )
        if stage == self._EXIT_ESC2:
            return AgentAction(
                action="press",
                key="escape",
                reasoning="ESC 2회 — 총독 화면 완전 종료",
                task_status="complete",
            )

        # VLM-guided stages
        return self._force_task_status(
            self._plan_generic_fallback_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=self.build_stage_note(memory),
            ),
            "in_progress",
        )

    # ------------------------------------------------------------------
    # Appoint branch — fixed step sequence
    # ------------------------------------------------------------------
    def _plan_appoint_branch(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        stage = memory.current_stage

        task_status = "complete" if stage == self._APPOINT_CONFIRM else "in_progress"
        return self._force_task_status(
            self._plan_generic_fallback_action(
                provider,
                pil_image,
                memory,
                normalizing_range=normalizing_range,
                high_level_strategy=high_level_strategy,
                recent_actions=recent_actions,
                hitl_directive=hitl_directive,
                img_config=img_config,
                extra_note=self.build_stage_note(memory),
            ),
            task_status,
        )

    # ------------------------------------------------------------------
    # on_action_success — deterministic stage transitions
    # ------------------------------------------------------------------
    def on_action_success(self, memory: ShortTermMemory, action: AgentAction) -> None:
        stage = memory.current_stage

        # Observation scroll stages
        if action.action == "move":
            if stage == self._HOVER_SCROLL_STAGE:
                memory.begin_stage(self._SCROLL_DOWN_STAGE)
                return
            if stage == self._RESTORE_HOVER_STAGE:
                memory.begin_stage(self._RESTORE_SCROLL_STAGE)
                return

        if action.action == "scroll":
            if stage == self._SCROLL_DOWN_STAGE:
                memory.register_choice_scroll(direction="down")
                memory.begin_stage("observe_choices")
                return
            if stage == self._RESTORE_SCROLL_STAGE:
                memory.register_choice_scroll(direction="up")
                memory.begin_stage("observe_choices")
                return

        # Promote branch transitions
        if action.action == "click":
            if stage == self._PROMOTE_CLICK:
                memory.begin_stage(self._PROMOTE_SELECT)
                return
            if stage == self._PROMOTE_SELECT:
                memory.begin_stage(self._PROMOTE_CONFIRM)
                return
            if stage == self._PROMOTE_CONFIRM:
                memory.begin_stage(self._PROMOTE_POPUP)
                return
            if stage == self._PROMOTE_POPUP:
                memory.begin_stage(self._EXIT_ESC1)
                return

        if action.action == "press":
            if stage == self._EXIT_ESC1:
                memory.begin_stage(self._EXIT_ESC2)
                return

        # Appoint branch transitions
        if action.action == "click":
            if stage == self._APPOINT_CLICK:
                memory.begin_stage(self._APPOINT_CITY)
                return
            if stage == self._APPOINT_CITY:
                memory.begin_stage(self._APPOINT_CONFIRM)
                return

    # ------------------------------------------------------------------
    # should_verify_action_without_ui_change
    # ------------------------------------------------------------------
    def should_verify_action_without_ui_change(self, memory: ShortTermMemory, action: AgentAction) -> bool:
        if action.action == "move":
            return True
        if action.action == "scroll" and memory.current_stage in {
            self._SCROLL_DOWN_STAGE,
            self._RESTORE_SCROLL_STAGE,
        }:
            return True
        if action.action == "press" and action.key == "escape":
            return True
        return super().should_verify_action_without_ui_change(memory, action)

    # ------------------------------------------------------------------
    # get_visible_progress — branch-aware progress display
    # ------------------------------------------------------------------
    def get_visible_progress(
        self,
        memory: ShortTermMemory,
        *,
        executed_steps: int,
        hard_max_steps: int,
    ) -> tuple[int, int]:
        del executed_steps, hard_max_steps

        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return 1, 2

        stage = memory.current_stage or "observe_choices"

        if memory.branch == self._PROMOTE_BRANCH:
            promote_order = [
                self._PROMOTE_CLICK,
                self._PROMOTE_SELECT,
                self._PROMOTE_CONFIRM,
                self._PROMOTE_POPUP,
                self._EXIT_ESC1,
                self._EXIT_ESC2,
            ]
            if stage in promote_order:
                return promote_order.index(stage) + 3, 8
            return 3, 8

        if memory.branch == self._APPOINT_BRANCH:
            appoint_order = [self._APPOINT_CLICK, self._APPOINT_CITY, self._APPOINT_CONFIRM]
            if stage in appoint_order:
                return appoint_order.index(stage) + 3, 5
            return 3, 5

        # Observation phase
        if stage in {self._HOVER_SCROLL_STAGE, self._SCROLL_DOWN_STAGE, "observe_choices"}:
            return 2, 4
        if stage in {"choose_from_memory", "decide_best_choice"}:
            return 3, 4

        return super().get_visible_progress(memory, executed_steps=0, hard_max_steps=1)


class CultureDecisionProcess(ScriptedMultiStepProcess):
    """Culture flow with an explicit lower-right notification entry stage."""

    _ENTRY_SUBSTEP = "culture_entry_done"

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("culture_entry")

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return (
                "현재 stage: culture_entry\n"
                "- 사회 제도 트리/선택 화면이 실제로 열렸는지 먼저 확인해.\n"
                "- 우하단 '사회 제도 선택' 알림만 보이면 press enter로 진입해.\n"
                "- 아직 제도 선택을 하지 마."
            )
        return "현재 멀티스텝 stage: direct_culture_select\n- 사회 제도 화면이 열린 상태다. 전략에 맞는 제도를 선택해."

    def _culture_screen_state(self, provider: BaseVLMProvider, pil_image, *, img_config=None) -> dict | None:
        prompt = (
            "너는 문명6 사회 제도 진입 상태 판별기야. 현재 화면이 실제 제도 선택 화면인지 여부만 판단해.\n"
            'JSON만 출력: {"culture_screen_ready": true/false,'
            ' "notification_visible": true/false, "reasoning": "짧은 이유"}\n'
            "- 사회 제도 트리 또는 제도 선택 팝업이 실제로 보이면 culture_screen_ready=true.\n"
            "- 우하단 '사회 제도 선택' 알림만 보이고 제도 트리가 안 열렸으면 culture_screen_ready=false.\n"
            "- 우하단 '사회 제도 선택' 알림이 분명히 보이면 notification_visible=true.\n"
        )
        try:
            return _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Culture entry check failed: %s", exc)
            return None

    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            state = self._culture_screen_state(provider, pil_image, img_config=img_config)
            if state and bool(state.get("culture_screen_ready", False)):
                memory.mark_substep(self._ENTRY_SUBSTEP)
                memory.begin_stage("direct_culture_select")
                return StageTransition(stage="direct_culture_select", reason="culture screen ready")
            if state and bool(state.get("notification_visible", False)):
                memory.begin_stage("culture_entry")
                return AgentAction(
                    action="press",
                    key="enter",
                    reasoning="우하단 '사회 제도 선택' 알림을 열어 제도 선택 화면으로 진입",
                    task_status="in_progress",
                )
            memory.begin_stage("culture_entry")
        else:
            memory.begin_stage("direct_culture_select")
        return super().plan_action(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )


class ResearchSelectProcess(ScriptedMultiStepProcess):
    """Research selection with an explicit lower-right notification entry stage."""

    _ENTRY_SUBSTEP = "research_entry_done"

    def initialize(self, memory: ShortTermMemory) -> None:
        if not memory.current_stage:
            memory.begin_stage("research_entry")

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            return (
                "현재 stage: research_entry\n"
                "- 기술 트리/연구 선택 팝업이 실제로 열렸는지 먼저 확인해.\n"
                "- 우하단 '연구 선택' 알림만 보이면 press enter로 진입해.\n"
                "- 아직 기술 선택을 하지 마."
            )
        return (
            "현재 멀티스텝 stage: direct_research_select\n"
            "- research selection은 별도 observation 없이 바로 적절한 연구를 선택한다."
        )

    def _research_screen_state(self, provider: BaseVLMProvider, pil_image, *, img_config=None) -> dict | None:
        prompt = (
            "너는 문명6 연구 진입 상태 판별기야. 현재 화면이 실제 연구 선택 화면인지 여부만 판단해.\n"
            'JSON만 출력: {"research_screen_ready": true/false,'
            ' "notification_visible": true/false, "reasoning": "짧은 이유"}\n'
            "- 기술 트리 또는 연구 선택 팝업이 실제로 보이면 research_screen_ready=true.\n"
            "- 우하단 '연구 선택' 알림만 보이고 기술 트리가 안 열렸으면 research_screen_ready=false.\n"
            "- 우하단 '연구 선택' 알림이 분명히 보이면 notification_visible=true.\n"
        )
        try:
            return _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=256)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Research entry check failed: %s", exc)
            return None

    def plan_action(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        *,
        normalizing_range: int,
        high_level_strategy: str,
        recent_actions: str,
        hitl_directive: str | None,
        img_config=None,
    ) -> AgentAction | list[AgentAction] | StageTransition | None:
        if self._ENTRY_SUBSTEP not in memory.completed_substeps:
            state = self._research_screen_state(provider, pil_image, img_config=img_config)
            if state and bool(state.get("research_screen_ready", False)):
                memory.mark_substep(self._ENTRY_SUBSTEP)
                memory.begin_stage("direct_research_select")
                return StageTransition(stage="direct_research_select", reason="research screen ready")
            if state and bool(state.get("notification_visible", False)):
                memory.begin_stage("research_entry")
                return AgentAction(
                    action="press",
                    key="enter",
                    reasoning="우하단 '연구 선택' 알림을 열어 기술 선택 화면으로 진입",
                    task_status="in_progress",
                )
            memory.begin_stage("research_entry")
        else:
            memory.begin_stage("direct_research_select")
        return super().plan_action(
            provider,
            pil_image,
            memory,
            normalizing_range=normalizing_range,
            high_level_strategy=high_level_strategy,
            recent_actions=recent_actions,
            hitl_directive=hitl_directive,
            img_config=img_config,
        )


def get_multi_step_process(primitive_name: str, completion_condition: str = "") -> BaseMultiStepProcess:
    """Factory for class-based multi-step primitive processes."""
    if primitive_name == "religion_primitive":
        return ObservationAssistedProcess(
            primitive_name,
            completion_condition,
            target_description="왼쪽 종교관 팝업의 종교관 박스와 '종교관 세우기' 직전 리스트",
        )
    if primitive_name == "city_production_primitive":
        return CityProductionProcess(primitive_name, completion_condition)
    if primitive_name == "voting_primitive":
        return ObservationAssistedProcess(
            primitive_name,
            completion_condition,
            target_description="세계의회 팝업의 합의안/agenda 블록 목록",
        )
    if primitive_name == "governor_primitive":
        return GovernorProcess(primitive_name, completion_condition)
    if primitive_name == "policy_primitive":
        return PolicyProcess(primitive_name, completion_condition)
    if primitive_name == "research_select_primitive":
        return ResearchSelectProcess(primitive_name, completion_condition)
    if primitive_name == "culture_decision_primitive":
        return CultureDecisionProcess(primitive_name, completion_condition)
    return ScriptedMultiStepProcess(primitive_name, completion_condition)
