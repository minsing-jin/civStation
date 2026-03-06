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

from computer_use_test.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DIPLOMATIC_PROMPT,
    GOVERNOR_PROMPT,
    JSON_FORMAT_INSTRUCTION,
    MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION,
    POLICY_PROMPT,
    POPUP_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
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
    "popup_primitive": {
        "criteria": "팝업 표시됨. 또는 우하단에 '다음 턴'/'연구 선택'/'생산 품목' 버튼 보임.",
        "prompt": POPUP_PROMPT,
        "priority": 1,
    },
    "governor_primitive": {
        "criteria": "총독 카드 나열 또는 '총독 배정' 텍스트 표시. 총독 임명 팝업 포함.",
        "prompt": GOVERNOR_PROMPT,
        "priority": 2,
    },
    "unit_ops_primitive": {
        "criteria": "유닛 선택됨 (우하단 유닛정보). 이동/공격/건설 필요. 하늘색 타일 또는 적 인접.",
        "prompt": UNIT_OPS_PROMPT,
        "priority": 3,
    },
    "research_select_primitive": {
        "criteria": "연구 선택 팝업 표시. 기술 목록 보임.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 4,
    },
    "city_production_primitive": {
        "criteria": "생산 품목 선택 팝업 표시. 건물/유닛 목록 보임.",
        "prompt": CITY_PRODUCTION_PROMPT,
        "priority": 5,
    },
    "science_decision_primitive": {
        "criteria": "기술 트리 화면 열림. 기술 노드 트리 형태.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 6,
    },
    "culture_decision_primitive": {
        "criteria": "사회 제도 트리 화면 열림. 사회 제도 노드 트리 형태.",
        "prompt": CULTURE_MANAGER_PROMPT,
        "priority": 7,
    },
    "diplomatic_primitive": {
        "criteria": "외교 화면. 대화/거래/전쟁 선포 등 외교 상호작용.",
        "prompt": DIPLOMATIC_PROMPT,
        "priority": 8,
    },
    "combat_primitive": {
        "criteria": "전투 유닛이 적 인접. 공격/방어 결정 필요.",
        "prompt": COMBAT_PROMPT,
        "priority": 9,
    },
    "policy_primitive": {
        "criteria": "정부/정책 변경 화면. 정부 선택 또는 '정책변경 미확정' 팝업.",
        "prompt": POLICY_PROMPT,
        "priority": 10,
    },
}

# ==============================================================================
# Auto-generated from registry
# ==============================================================================
PRIMITIVE_NAMES: list[str] = list(PRIMITIVE_REGISTRY.keys())


def _build_router_prompt() -> str:
    """Build router prompt automatically from PRIMITIVE_REGISTRY."""
    sorted_entries = sorted(PRIMITIVE_REGISTRY.items(), key=lambda x: x[1]["priority"])

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
    "science_decision_primitive": "기술 연구",  # shares key with research_select
    "city_production_primitive": "도시 생산",
    "culture_decision_primitive": "사회 제도",
    "diplomatic_primitive": "외교",
    "combat_primitive": "전투",
    "policy_primitive": "정책",
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

    if high_level_strategy is None:
        high_level_strategy = "과학 승리를 목표로 함"

    hitl_directive_section = hitl_directive or ""
    recent_actions_section = recent_actions or "없음"

    if primitive_name == "policy_primitive":
        json_instruction = MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)
    else:
        json_instruction = JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)
    prompt_template = PRIMITIVE_REGISTRY[primitive_name]["prompt"]
    return prompt_template.format(
        json_instruction=json_instruction,
        high_level_strategy=high_level_strategy,
        recent_actions=recent_actions_section,
        hitl_directive=hitl_directive_section,
        **kwargs,
    )
