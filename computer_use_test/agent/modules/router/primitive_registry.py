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

from dataclasses import dataclass

from computer_use_test.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DIPLOMATIC_PROMPT,
    GOVERNOR_PROMPT,
    JSON_FORMAT_INSTRUCTION,
    POLICY_PROMPT,
    POPUP_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
)


@dataclass
class RouterResult:
    """Result from the Router's screenshot classification.

    Contains the selected primitive and turn-recognition metadata.
    """

    primitive: str
    reasoning: str = ""
    observed_turn: int | None = None  # Turn number read from the screen (top-right)
    is_new_turn: bool = False  # True when the in-game turn number has incremented


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

    return f"""문명6 스크린샷을 분류하고 우상단 턴 숫자를 읽어라.
우선순위 순서대로 판단:

{criteria_block}

해당 없으면 "popup_primitive" (다음 턴).
JSON만 응답:
{{"primitive":"카테고리","reasoning":"이유","turn_number":숫자_또는_null}}
"""


ROUTER_PROMPT = _build_router_prompt()


def get_primitive_prompt(
    primitive_name: str,
    normalizing_range: int = 1000,
    high_level_strategy: str | None = None,
    context: str | None = None,
    hitl_directive: str | None = None,
    **kwargs,
) -> str:
    """
    Get the appropriate prompt for a primitive by name.

    Args:
        primitive_name: Name of the primitive (e.g., "unit_ops_primitive")
        normalizing_range: Coordinate normalization range (default: 1000)
        high_level_strategy: Optional high-level strategy context to guide decisions.
                         If None, uses default placeholder strategy.
        context: Primitive-specific information like statics, current state.
               ex) "city population: 5, production: 10, food: 8, science: 12.
                    Current turn: 150. Current Production Items: ..."
        hitl_directive: Optional micro-level HITL directive (e.g., "병영을 최우선 선택").
                       Injected into the prompt with highest priority.

    Returns:
        Prompt string for the primitive with formatted JSON instructions

    Raises:
        ValueError: If primitive name is not recognized
    """
    if primitive_name not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive: {primitive_name}. Available: {PRIMITIVE_NAMES}")

    if high_level_strategy is None:
        high_level_strategy = "과학 승리를 목표로 함"

    # Build hitl_directive section (empty string if no directive)
    hitl_directive_section = ""
    if hitl_directive:
        hitl_directive_section = hitl_directive

    json_instruction = JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)
    prompt_template = PRIMITIVE_REGISTRY[primitive_name]["prompt"]
    return prompt_template.format(
        json_instruction=json_instruction,
        high_level_strategy=high_level_strategy,
        context=context,
        hitl_directive=hitl_directive_section,
        **kwargs,
    )
