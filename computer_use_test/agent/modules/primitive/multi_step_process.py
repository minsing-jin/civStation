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

from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
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
        prompt = (
            "너는 문명6 선택 결정 서브에이전트야. 아래 short-term memory에 누적된 전체 후보 중 "
            "상위 전략에 가장 적합한 하나를 고르고 JSON만 출력해.\n"
            'JSON: {"best_option_label":"후보 이름","reason":"짧은 이유"}\n\n'
            f"Primitive: {self.primitive_name}\n"
            f"상위 전략:\n{high_level_strategy}\n\n"
            f"후보 memory:\n{memory.to_prompt_string()}\n"
        )
        try:
            response = provider.call_vlm(prompt=prompt, image_path=None, temperature=0.2, max_tokens=512)
            content = strip_markdown(response.content)
            data = json.loads(content)
            label = str(data.get("best_option_label", "")).strip()
            reason = str(data.get("reason", "")).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Best-choice decision failed for %s: %s", self.primitive_name, exc)
            return False

        if not label:
            return False

        memory.set_best_choice(label=label, reason=reason)
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
                "좌측 슬롯 의미 정보와 정책 탭 위치를 읽어 cache를 만든다."
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
        return PRESETS.get("policy_tab_check_fast", img_config)

    def _verify_policy_tab_switch(
        self,
        provider: BaseVLMProvider,
        pil_image,
        memory: ShortTermMemory,
        expected_tab: str,
        *,
        img_config=None,
    ) -> SemanticVerifyResult:
        prompt = (
            "문명6 정책 탭 확인기. JSON만 출력.\n"
            '{"match": true, "observed_tab": "경제", "reason": "노란 카드"}\n'
            f"기대 탭: {expected_tab}\n"
            "판정 기준:\n"
            "- 오른쪽 흰 종이 배경 질감 위에 카드가 여러 장 모여 있는 정책 카드 목록만 본다.\n"
            "- 좌측 파란 슬롯 영역과 거기에 이미 장착된 카드들은 현재 탭 판정 근거에서 완전히 무시한다.\n"
            "- 좌측 슬롯에 다른 탭 카드가 끼워져 있어도 탭 전환 실패로 판단하지 마.\n"
            "- 상단 탭 강조 상태는 참고하지 말고, 오른쪽 카드 목록의 분류와 필터 상태만으로 판단해.\n"
            "- 군사: 오른쪽 카드 목록이 주로 군사 카드(빨강 계열)다.\n"
            "- 경제: 오른쪽 카드 목록이 주로 경제 카드(노랑 계열)다.\n"
            "- 외교: 오른쪽 카드 목록이 주로 외교 카드(초록/청록/파랑 계열)다.\n"
            "- 와일드카드: 오른쪽 카드 목록이 주로 와일드카드 카드이며 "
            "보라색, 검은색, 황금색 카드가 섞여 보일 수 있다.\n"
            "- 암흑: 오른쪽 카드 목록이 주로 암흑 카드(검정 계열)다.\n"
            "- '전체' 또는 혼합 목록은 여러 색이 섞여 보이므로 "
            "와일드카드와 혼동하지 마. 오른쪽 카드 목록이 실제로 "
            "해당 탭 필터 상태인지 본다.\n"
            "- 방금 클릭한 탭의 오른쪽 카드 목록이 보이면 match=true다.\n"
            "- 애매하면 match=false, observed_tab='unknown'.\n"
        )
        try:
            data = _analyze_structured_json(
                provider,
                pil_image,
                prompt,
                img_config=self._policy_tab_check_img_config(img_config),
                max_tokens=96,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy tab check failed to parse for %s: %s", expected_tab, exc)
            memory.set_policy_last_tab_check_result(f"{expected_tab}->unknown:parse-fail")
            return SemanticVerifyResult(handled=True, passed=False, reason="policy tab-check parse failed")

        observed_tab = str(data.get("observed_tab", "unknown")).strip() or "unknown"
        matched = bool(data.get("match", False)) and observed_tab == expected_tab
        reason = str(data.get("reason", "")).strip()
        result_note = f"{expected_tab}->{observed_tab}:{'ok' if matched else 'fail'}"
        if reason:
            result_note += f" ({reason})"
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
                '  "visible_tabs": ["군사", "경제", "외교", "와일드카드", "암흑"],\n'
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
                f"- 상위 전략 참고:\n{high_level_strategy}\n"
            )
        else:
            existing_positions = {}
            prompt = (
                "너는 문명6 정책 화면 bootstrap 분석기야. 현재 화면이 정책 카드 관리 화면이면 아래 JSON만 출력해.\n"
                "{\n"
                '  "policy_screen_ready": true,\n'
                '  "overview_mode": true,\n'
                '  "visible_tabs": ["군사", "경제", "외교", "와일드카드", "암흑"],\n'
                '  "wild_slot_active": true,\n'
                '  "slot_inventory": [\n'
                '    {"slot_id":"military_1","slot_type":"군사",'
                '"current_card_name":"","is_empty":true,"active":true,"is_wild":false}\n'
                "  ],\n"
                '  "tab_positions": [\n'
                '    {"tab_name":"군사","x":0,"y":0}, {"tab_name":"경제","x":0,"y":0}, '
                '{"tab_name":"외교","x":0,"y":0}, {"tab_name":"와일드카드","x":0,"y":0}, '
                '{"tab_name":"암흑","x":0,"y":0}\n'
                "  ]\n"
                "}\n"
                '정책 카드 화면이 아니면 {"policy_screen_ready": false} 만 출력해.\n'
                "규칙:\n"
                f"{_normalized_coord_note(normalizing_range, fields='tab_positions.x/y')}\n"
                "- policy entry 직후의 첫 정책 화면은 기본적으로 overview_mode=true 로 본다.\n"
                "- 탭 5종류(군사, 경제, 외교, 와일드카드, 암흑)는 모두 visible 이라고 보고, "
                "5개 전부의 중심 좌표를 반환해.\n"
                "- 탭 queue는 코드가 군사→경제→외교→와일드카드→암흑 순서로 고정 생성하므로, "
                "queue를 추론하거나 반환하지 마.\n"
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
            tab_positions = data.get("tab_positions", [])
            if not isinstance(tab_positions, list):
                return False
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
                cached_positions.append({"tab_name": tab_name, "x": x, "y": y, "confirmed": False})
                seen_tabs.add(tab_name)

            if set(seen_tabs) != set(_POLICY_TAB_NAMES):
                logger.info("Policy bootstrap rejected: tab positions incomplete (%s)", sorted(seen_tabs))
                memory.set_policy_event(f"bootstrap rejected: tab_positions={sorted(seen_tabs)}")
                return False
            cached_positions, bootstrap_scale_note = self._maybe_upscale_bootstrap_positions(
                cached_positions,
                normalizing_range=normalizing_range,
            )
            absolute_positions: list[dict[str, int | str | bool]] = []
            for item in cached_positions:
                absolute = self._normalized_policy_to_absolute(
                    memory,
                    int(item["x"]),
                    int(item["y"]),
                    normalizing_range=normalizing_range,
                )
                if absolute is None:
                    memory.set_policy_event("bootstrap rejected: capture geometry missing")
                    return False
                absolute_positions.append(
                    {
                        "tab_name": str(item["tab_name"]),
                        "screen_x": absolute[0],
                        "screen_y": absolute[1],
                        "confirmed": False,
                    }
                )
            cached_positions = absolute_positions

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
            f"{tab}=({memory.policy_state.tab_positions[tab].screen_x},{memory.policy_state.tab_positions[tab].screen_y})"
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
        prompt = (
            "너는 문명6 정책 탭 재탐색기야. 지정된 탭 하나의 현재 위치만 다시 찾아 JSON만 출력해.\n"
            '{"found": true, "tab_name": "군사", "x": 0, "y": 0}\n'
            '못 찾으면 {"found": false}만 출력해.\n'
            f"찾을 탭: {tab_name}\n"
            f"정책 탭 후보: {', '.join(_POLICY_TAB_NAMES)}\n"
            f"{_normalized_coord_note(normalizing_range, fields='x/y')}\n"
            "- 다른 탭은 무시하고 요청된 탭 하나의 중심 좌표만 반환해.\n"
        )
        try:
            data = _analyze_structured_json(provider, pil_image, prompt, img_config=img_config, max_tokens=512)
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
            reconciled_x, reconciled_y, reconcile_note = x, y, f"raw=({x},{y})"
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
            reconciled_x, reconciled_y, reconcile_note = self._reconcile_relocalized_policy_position(
                x,
                y,
                existing_x=existing_normalized[0],
                existing_y=existing_normalized[1],
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


class ResearchSelectProcess(ScriptedMultiStepProcess):
    """Research selection intentionally skips the observer path."""

    def build_stage_note(self, memory: ShortTermMemory) -> str:
        return (
            "현재 멀티스텝 stage: direct_research_select\n"
            "- research selection은 별도 observation 없이 바로 적절한 연구를 선택한다."
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
        return ObservationAssistedProcess(
            primitive_name,
            completion_condition,
            target_description="생산 품목 선택 팝업의 스크롤 가능한 생산 목록",
        )
    if primitive_name == "voting_primitive":
        return ObservationAssistedProcess(
            primitive_name,
            completion_condition,
            target_description="세계의회 팝업의 합의안/agenda 블록 목록",
        )
    if primitive_name == "policy_primitive":
        return PolicyProcess(primitive_name, completion_condition)
    if primitive_name == "research_select_primitive":
        return ResearchSelectProcess(primitive_name, completion_condition)
    return ScriptedMultiStepProcess(primitive_name, completion_condition)
