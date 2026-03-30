"""
Primitive Registry — Single source of truth for all primitives.

Each entry defines:
- criteria: Condition for the router to select this primitive (Korean, matching game UI)
  (Router가 이 primitive를 선택하는 조건)
- prompt: Action prompt template used by the primitive
  (Primitive가 사용하는 action 프롬프트 템플릿)
- priority: Evaluation order in the router prompt — lower = checked first
  (Router 프롬프트에서의 판단 우선순위, 낮을수록 먼저 판단)

To add a new primitive, add an entry to PRIMITIVE_REGISTRY below.
ROUTER_PROMPT, PRIMITIVE_NAMES, and get_primitive_prompt() auto-update.
"""

import logging
import warnings
from dataclasses import dataclass

from civStation.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DEAL_PROMPT,
    DIPLOMATIC_PROMPT,
    ERA_PROMPT,
    GOVERNOR_PROMPT,
    POLICY_PROMPT,
    POPUP_PROMPT,
    RELIGION_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
    VOTING_PROMPT,
    WAR_PROMPT,
    get_json_instruction_template,
    get_primitive_prompt_template,
    normalize_prompt_language,
)

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result from the Router's screenshot classification.

    Turn detection is handled by TurnDetector (background thread).
    """

    primitive: str
    reasoning: str = ""


# ==============================================================================
# Primitive Registry (TODO: Criteria will be replaced by small vlm model or image similarity metric)
# ==============================================================================
PRIMITIVE_REGISTRY: dict[str, dict] = {
    # --- Router-included primitives (sorted by priority) ---
    "religion_primitive": {
        "criteria": (
            "종교관 선택 화면. 왼쪽 종교관 목록 또는 초록색 '종교관 세우기' 버튼 표시. "
            "또는 우하단 천사 문양 원형 종교관 버튼 보임."
        ),
        "prompt": RELIGION_PROMPT,
        "priority": 1,
        "multi_step": True,
        "process_kind": "observation_assisted",
        "max_steps": 20,
        "completion_condition": (
            "종교관 준비 팝업을 Esc로 닫은 뒤 prep_popup_visible=false 이고 "
            "우하단 원형 버튼이 더 이상 천사 문양이 아니면 task_status='complete'."
        ),
        "completion_condition_en": (
            "After closing the pantheon-ready popup with Esc, task_status='complete' when "
            "prep_popup_visible=false and the lower-right circular button is no longer the angel icon."
        ),
    },
    "governor_primitive": {
        "criteria": (
            "총독 카드 나열 또는 '총독 배정' 텍스트 표시. 총독 임명 팝업 포함. "
            "또는 우하단 '총독 타이틀' 버튼/펜 아이콘 보임."
        ),
        "prompt": GOVERNOR_PROMPT,
        "priority": 2,
        "multi_step": True,
        "process_kind": "observation_assisted",
        "max_steps": 20,
        "completion_condition": (
            "총독 진급 [확정] 후 ESC 2회 완료, 또는 총독 임명 후 [배정] 버튼 클릭 완료 시 task_status='complete'."
        ),
        "completion_condition_en": (
            "task_status='complete' after governor promotion is confirmed with [Confirm] and ESC is pressed twice, "
            "or after governor appointment and the [Assign] button click are completed."
        ),
    },
    "voting_primitive": {
        "criteria": "세계의회 투표 화면. 정책 A/B 선택, 찬성/반대 기호, 합의안 투표.",
        "prompt": VOTING_PROMPT,
        "priority": 3,
        "multi_step": True,
        "process_kind": "observation_assisted",
        "max_steps": 14,
        "completion_condition": "'게임으로 돌아가기' 클릭 또는 esc 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' after clicking 'Return to Game' or pressing esc.",
    },
    "era_primitive": {
        "criteria": "시대 전략 선택 화면. 시대 헌신 4개 박스 표시, 확정 버튼.",
        "prompt": ERA_PROMPT,
        "priority": 4,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 6,
        "completion_condition": "'확정' 버튼 클릭 완료 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' after clicking the 'Confirm' button.",
    },
    "unit_ops_primitive": {
        "criteria": "유닛 선택됨 (우하단 유닛정보). 이동/공격/건설 필요. 하늘색 타일 또는 적 인접.",
        "prompt": UNIT_OPS_PROMPT,
        "priority": 20,
        "multi_step": False,
        "max_steps": 1,
        "completion_condition": "",
    },
    "research_select_primitive": {
        "criteria": (
            "연구 선택 팝업 표시 또는 기술 트리 화면 열림. 기술 목록/노드 보임. 또는 우하단 '연구 선택' 버튼 보임."
        ),
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 5,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 6,
        "completion_condition": "기술 클릭 완료 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' after the technology click is completed.",
    },
    "city_production_primitive": {
        "criteria": (
            "생산 품목 선택 팝업 표시 또는 배치 화면 열림. 건물/유닛 목록/배치 타일 보임. "
            "또는 우하단 '생산 품목' 버튼 보임."
        ),
        "prompt": CITY_PRODUCTION_PROMPT,
        "priority": 6,
        "multi_step": True,
        "process_kind": "observation_assisted",
        "max_steps": 18,
        "completion_condition": "생산 품목 클릭 완료 또는 배치 확인 시 task_status='complete'.",
        "completion_condition_en": (
            "task_status='complete' after clicking the production item or confirming placement."
        ),
        "img_config_preset": "planner_high_quality",
    },
    "culture_decision_primitive": {
        "criteria": "사회 제도 트리 화면 열림. 사회 제도 노드 트리 형태. 또는 우하단 '사회 제도 선택' 버튼 보임.",
        "prompt": CULTURE_MANAGER_PROMPT,
        "priority": 8,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 6,
        "completion_condition": "사회 제도 클릭 완료 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' after the civic click is completed.",
    },
    "diplomatic_primitive": {
        "criteria": "외교 화면. 도시국가 사절파견 또는 외교 상호작용.",
        "prompt": DIPLOMATIC_PROMPT,
        "priority": 9,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 10,
        "completion_condition": "모든 화살표가 어두워짐 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' once every arrow is dark.",
    },
    "combat_primitive": {
        "criteria": "전투 유닛이 적 인접. 공격/방어 결정 필요.",
        "prompt": COMBAT_PROMPT,
        "priority": 21,
        "multi_step": False,
        "max_steps": 1,
        "completion_condition": "",
    },
    "policy_primitive": {
        "criteria": "'사회제도 완성'/'정책변경' 팝업, 새 정부 선택 화면, 또는 정부/정책 카드 배정 화면.",
        "prompt": POLICY_PROMPT,
        "priority": 7,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 24,
        "completion_condition": (
            "'모든 정책 배정' 후 확인 팝업의 '예' 또는 확인 버튼 클릭 완료, "
            "또는 이번 정책 run에서 변경 없음이라 정책 화면에서 Esc 종료 완료 시 task_status='complete'."
        ),
        "completion_condition_en": (
            "task_status='complete' after clicking 'Yes' or the confirm button in the popup that follows "
            "'Confirm Policies', or after exiting the policy screen with Esc when no policy changes were made "
            "during this policy run."
        ),
        "img_config_preset": "planner_high_quality",
    },
    "popup_primitive": {
        "criteria": "기타 일반 팝업 표시됨. 또는 우하단에 '다음 턴' 버튼 보임.",
        "prompt": POPUP_PROMPT,
        "priority": 99,
        "multi_step": False,
        "max_steps": 1,
        "completion_condition": "",
    },
    # --- HITL-only primitives (no router criteria) ---
    "war_primitive": {
        "criteria": "",  # Not included in router — HITL forced only
        "prompt": WAR_PROMPT,
        "priority": -1,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 8,
        "completion_condition": "전쟁선포 완료 후 esc 시 task_status='complete'.",
        "completion_condition_en": "task_status='complete' after declaring war and then pressing esc.",
    },
    "deal_primitive": {
        "criteria": "",  # Not included in router — HITL forced only
        "prompt": DEAL_PROMPT,
        "priority": -1,
        "multi_step": True,
        "process_kind": "scripted",
        "max_steps": 10,
        "completion_condition": "거래수락 + esc 또는 esc x2 취소 시 task_status='complete'.",
        "completion_condition_en": (
            "task_status='complete' after accepting the deal and pressing esc, or after cancelling with esc twice."
        ),
    },
}

# ==============================================================================
# Auto-generated from registry
# ==============================================================================
PRIMITIVE_NAMES: list[str] = list(PRIMITIVE_REGISTRY.keys())


def _build_router_prompt() -> str:
    """Build router prompt automatically from PRIMITIVE_REGISTRY.

    Excludes HITL-only primitives (those with empty criteria).
    """
    routable = {k: v for k, v in PRIMITIVE_REGISTRY.items() if v.get("criteria")}
    sorted_entries = sorted(routable.items(), key=lambda x: x[1]["priority"])

    criteria_lines = []
    for i, (name, entry) in enumerate(sorted_entries, 1):
        criteria_lines.append(f'{i}. "{name}": {entry["criteria"]}')

    criteria_block = "\n".join(criteria_lines)

    return f"""문명6 스크린샷을 분류해.
우선순위 순서대로 판단:

{criteria_block}

해당 없으면 "popup_primitive" (다음 턴).
JSON만 응답:
{{"primitive":"카테고리","reasoning":"이유"}}
"""


ROUTER_PROMPT = _build_router_prompt()


# ==============================================================================
# Primitive ↔ Korean label mapping (matches primitive_directives keys)
# ==============================================================================
PRIMITIVE_TO_KOREAN: dict[str, str] = {
    "unit_ops_primitive": "유닛 조작",
    "popup_primitive": "팝업",
    "governor_primitive": "총독",
    "research_select_primitive": "기술 연구",
    "city_production_primitive": "도시 생산",
    "culture_decision_primitive": "사회 제도",
    "diplomatic_primitive": "외교",
    "combat_primitive": "전투",
    "policy_primitive": "정책",
    "religion_primitive": "종교",
    "war_primitive": "전쟁 선포",
    "deal_primitive": "거래",
    "voting_primitive": "세계의회",
    "era_primitive": "시대 전략",
}


def get_directive_for_primitive(
    primitive_name: str,
    primitive_directives: dict[str, str],
) -> str | None:
    """Look up the matching directive from strategy.primitive_directives.

    Args:
        primitive_name: Internal primitive name (e.g. "unit_ops_primitive")
        primitive_directives: Korean-keyed dict from StructuredStrategy

    Returns:
        The directive string, or None if no match found.
    """
    korean_key = PRIMITIVE_TO_KOREAN.get(primitive_name)
    if korean_key is None:
        logger.warning(f"No Korean mapping for primitive '{primitive_name}'")
        return None
    return primitive_directives.get(korean_key)


def get_primitive_prompt(
    primitive_name: str,
    normalizing_range: int = 1000,
    high_level_strategy: str | None = None,
    recent_actions: str | None = None,
    hitl_directive: str | None = None,
    short_term_memory: str | None = None,
    json_instruction_override: str | None = None,
    language: str = "eng",
    # Deprecated — kept for backward compat
    context: str | None = None,
    **kwargs,
) -> str:
    """
    Get the appropriate prompt for a primitive by name.

    Args:
        primitive_name: Name of the primitive (e.g., "unit_ops_primitive")
        normalizing_range: Coordinate normalization range (default: 1000)
        high_level_strategy: Optional high-level strategy context to guide decisions.
                         If None, uses default placeholder strategy.
        recent_actions: Compressed string of recent actions (for repetition avoidance).
        hitl_directive: Optional micro-level HITL directive (e.g., "병영을 최우선 선택").
                       Injected into the prompt with highest priority.
        short_term_memory: Optional short-term memory string from previous steps.
                          Used by multi-step primitives for step-to-step context.
        language: Prompt language. Defaults to `eng`. Legacy Korean prompts remain
                  available via `kor`.
        context: **Deprecated** — ignored. Use recent_actions instead.

    Returns:
        Prompt string for the primitive with formatted JSON instructions

    Raises:
        ValueError: If primitive name is not recognized
    """
    if context is not None:
        warnings.warn(
            "get_primitive_prompt(context=...) is deprecated. Use recent_actions= instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if primitive_name not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive: {primitive_name}. Available: {PRIMITIVE_NAMES}")

    prompt_language = normalize_prompt_language(language)

    if high_level_strategy is None:
        high_level_strategy = "Aim for a science victory." if prompt_language == "eng" else "과학 승리를 목표로 함"

    registry_entry = PRIMITIVE_REGISTRY[primitive_name]
    is_multi_step = registry_entry.get("multi_step", False)

    hitl_directive_section = hitl_directive or ""
    recent_actions_section = recent_actions or ("None" if prompt_language == "eng" else "없음")
    short_term_memory_section = short_term_memory or ("None" if prompt_language == "eng" else "없음")
    completion_condition_key = "completion_condition_en" if prompt_language == "eng" else "completion_condition"
    completion_condition_section = registry_entry.get(
        completion_condition_key, registry_entry.get("completion_condition", "")
    )

    if json_instruction_override is not None:
        json_instruction = json_instruction_override.format(normalizing_range=normalizing_range)
    elif is_multi_step:
        json_instruction = get_json_instruction_template(prompt_language, format_kind="multi_step").format(
            normalizing_range=normalizing_range
        )
    else:
        json_instruction = get_json_instruction_template(prompt_language, format_kind="single").format(
            normalizing_range=normalizing_range
        )

    prompt_template = get_primitive_prompt_template(primitive_name, language=prompt_language)
    return prompt_template.format(
        json_instruction=json_instruction,
        high_level_strategy=high_level_strategy,
        recent_actions=recent_actions_section,
        hitl_directive=hitl_directive_section,
        short_term_memory=short_term_memory_section,
        completion_condition=completion_condition_section,
        **kwargs,
    )
