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
        "criteria": (
            "화면에 팝업상자가 나타나 있다. "
            "또는 좌측 화면에 스킬트리 선택화면이 없는상태에서 화면 오른쪽 아래에 '다음 턴', '연구 선택', '생산 품목', "
            "'사회 제도 선택' 같은 버튼이 보인다."
        ),
        "prompt": POPUP_PROMPT,
        "priority": 1,
    },
    "governor_primitive": {
        "criteria": (
            "화면에 '총독' 관련 텍스트가 보이거나 총독 카드들이 나열된 화면이다. "
            "또는 화면 좌측 상단에 '다음 위치에 총독 배정'이라는 텍스트가 보이는 지도 화면이다. "
            "우측 아래에 총독 임명 선택 창 팝업창이 보인다."
        ),
        "prompt": GOVERNOR_PROMPT,
        "priority": 2,
    },
    "unit_ops_primitive": {
        "criteria": (
            "유닛이 선택되어 있어서 오른쪽 아래에 유닛 이름과 이동력이 있는 정보가 보이고, "
            "맵에서 유닛 조작(이동, 공격, 건설 등)이 필요한 상황이다. "
            "하늘색 이동 가능 타일이 보이거나, 적 유닛/도시가 인접해 공격이 가능한 상황을 포함한다."
        ),
        "prompt": UNIT_OPS_PROMPT,
        "priority": 3,
    },
    "research_select_primitive": {
        "criteria": "화면 오른쪽 아래에 연구 선택 팝업이 나타나 있다. 연구할 기술 목록이 보인다.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 4,
    },
    "city_production_primitive": {
        "criteria": "화면 오른쪽에 생산 품목 선택 팝업이 나타나 있다. 생산할 수 있는 건물/유닛 목록이 보인다.",
        "prompt": CITY_PRODUCTION_PROMPT,
        "priority": 5,
    },
    "science_decision_primitive": {
        "criteria": "기술 트리 화면이 열려 있다. 기술 노드들이 연결된 트리 형태로 보인다.",
        "prompt": RESEARCH_MANAGER_PROMPT,
        "priority": 6,
    },
    "culture_decision_primitive": {
        "criteria": "사회 제도 트리 화면이 열려 있다. 사회 제도 노드들이 연결된 트리 형태로 보인다.",
        "prompt": CULTURE_MANAGER_PROMPT,
        "priority": 7,
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
    "policy_primitive": {
        "criteria": "1. '정부(Government)' 화면이 열려 있음. "
        "2. '정부 화면'에서 '정책 변경' 탭이 활성화됨. "
        "3. 새로운 정부(과두제 등)를 선택하는 화면임. "
        "4. 정책 화면을 닫을 때 발생하는 '정책변경이 확정되지 않았습니다' 팝업이 떠 있음.",
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

    criteria_block = "\n\n".join(criteria_lines)

    return f"""너는 문명6 게임 상태를 분석하는 라우터야.
스크린샷을 보고 현재 상황을 정확히 하나의 카테고리로 분류해.

추가로, 화면 오른쪽 위(Top-right)에 표시된 현재 턴 숫자를 읽어서 "turn_number" 필드에 정수로 기록해.
턴 숫자를 읽을 수 없는 경우 null로 표기해.

분류 기준 (우선순위 순서대로 판단):

{criteria_block}

위 기준 중 해당하는 것이 없다면 "popup_primitive"로 분류해 (다음 턴 버튼을 눌러야 할 가능성이 높음).

반드시 아래 JSON 형식으로만 응답해:
{{
    "primitive": "위_카테고리_중_하나",
    "reasoning": "이유를 간단히 설명",
    "turn_number": 현재_턴_숫자_또는_null
}}
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
