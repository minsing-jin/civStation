"""
Primitive Registry — Single source of truth for all primitives.

- criteria: Router가 이 primitive를 선택하는 조건 (한국어)
- prompt: Primitive가 사용하는 action 프롬프트 템플릿
- priority: Router 프롬프트에서의 판단 우선순위 (낮을수록 먼저 판단)

새로운 primitive 추가 시 여기에만 추가하면
ROUTER_PROMPT, PRIMITIVE_NAMES, get_primitive_prompt() 모두 자동 반영됨.
"""

from computer_use_test.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DIPLOMATIC_PROMPT,
    JSON_FORMAT_INSTRUCTION,
    POPUP_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
)

# ==============================================================================
# Primitive Registry
# ==============================================================================
PRIMITIVE_REGISTRY: dict[str, dict] = {
    "popup_primitive": {
        "criteria": (
            "화면에 팝업상자가 나타나 있다. "
            "또는 화면 오른쪽 아래에 '다음 턴', '연구 선택', '생산 품목', "
            "'사회 제도 선택' 같은 버튼이 보인다."
        ),
        "prompt": POPUP_PROMPT,
        "priority": 1,
    },
    "research_select_primitive": {
        "criteria": "화면 오른쪽 아래에 연구 선택 팝업이 나타나 있다. 연구할 기술 목록이 보인다.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 2,
    },
    "city_production_primitive": {
        "criteria": "화면 오른쪽에 생산 품목 선택 팝업이 나타나 있다. 생산할 수 있는 건물/유닛 목록이 보인다.",
        "prompt": CITY_PRODUCTION_PROMPT,
        "priority": 3,
    },
    "science_decision_primitive": {
        "criteria": "기술 트리 화면이 열려 있다. 기술 노드들이 연결된 트리 형태로 보인다.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 4,
    },
    "culture_decision_primitive": {
        "criteria": "사회 제도 트리 화면이 열려 있다. 사회 제도 노드들이 연결된 트리 형태로 보인다.",
        "prompt": CULTURE_MANAGER_PROMPT,
        "priority": 5,
    },
    "unit_ops_primitive": {
        "criteria": "유닛이 선택되어 있거나 맵에서 유닛 조작이 필요한 상황이다. 하늘색 이동 가능 타일이 보인다.",
        "prompt": UNIT_OPS_PROMPT,
        "priority": 6,
    },
    "diplomatic_primitive": {
        "criteria": "외교 화면이 열려 있다. 상대 문명과의 대화, 거래, 전쟁 선포 등 외교 상호작용이 필요하다.",
        "prompt": DIPLOMATIC_PROMPT,
        "priority": 8,
    },
    "combat_primitive": {
        "criteria": "전투 유닛이 적과 인접해 있거나 공격/방어 결정이 필요한 전투 상황이다.",
        "prompt": COMBAT_PROMPT,
        "priority": 9,
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

    criteria_block = "\n\n".join(criteria_lines)

    return f"""너는 문명6 게임 상태를 분석하는 라우터야.
스크린샷을 보고 현재 상황을 정확히 하나의 카테고리로 분류해.

분류 기준 (우선순위 순서대로 판단):

{criteria_block}

위 기준 중 해당하는 것이 없다면 "popup_primitive"로 분류해 (다음 턴 버튼을 눌러야 할 가능성이 높음).

반드시 아래 JSON 형식으로만 응답해:
{{
    "primitive": "위_카테고리_중_하나",
    "reasoning": "이유를 간단히 설명"
}}
"""


ROUTER_PROMPT = _build_router_prompt()


def get_primitive_prompt(primitive_name: str, normalizing_range: int = 1000) -> str:
    """
    Get the appropriate prompt for a primitive by name.

    Args:
        primitive_name: Name of the primitive (e.g., "unit_ops_primitive")
        normalizing_range: Coordinate normalization range (default: 1000)

    Returns:
        Prompt string for the primitive with formatted JSON instructions

    Raises:
        ValueError: If primitive name is not recognized
    """
    if primitive_name not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive: {primitive_name}. Available: {PRIMITIVE_NAMES}")

    json_instruction = JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range)
    prompt_template = PRIMITIVE_REGISTRY[primitive_name]["prompt"]
    return prompt_template.format(json_instruction=json_instruction)
