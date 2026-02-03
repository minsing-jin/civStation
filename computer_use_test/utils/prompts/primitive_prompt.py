"""
Civilization VI Game Agent Prompt Templates.

This module contains only prompt template strings for each primitive.
Registry, routing, and lookup logic lives in
computer_use_test.agent.modules.router.primitive_registry.
"""

# TODO: primitive action만 이 File에서는 구현하고,
# 행동 전략은 High-level order를 반영할수 있도록 파라미터로 받고 고도화 시키기
# TODO: policy selection prompt
# (drag and drop 필요, 설명 읽는것 필요 - visual grounding 필요)

# ==============================================================================
# Base JSON Format Instruction Template
# ==============================================================================
JSON_FORMAT_INSTRUCTION = """
CRITICAL: Generate a JSON response with ONE action in the following format:
{{
  "action": "REQUIRED - must be one of: click, press, drag, type",
  "x": 0,
  "y": 0,
  "end_x": 0,
  "end_y": 0,
  "button": "left",
  "key": "",
  "text": "",
  "reasoning": "Brief explanation of what you see and why this action is needed"
}}

COORDINATE SYSTEM:
- All coordinates must be NORMALIZED (0-{normalizing_range})
- (0, 0) = Top-Left corner
- ({normalizing_range}, {normalizing_range}) = Bottom-Right corner

Action types:
- "click": Click at position (x, y) with button (left/right)
- "press": Press a keyboard key (provide key name in "key" field) - Examples: "enter", "esc", "b", "space"
- "drag": Drag from (x, y) to (end_x, end_y)
- "type": Type text (provide text in "text" field)

IMPORTANT:
- The "action" field is MANDATORY and must be exactly one of the types above
- For "press" action, you MUST provide the "key" field (e.g., "key": "enter")
- Use 0 or empty string for unused fields
"""

# ==============================================================================
# Primitive Prompt Templates
# ==============================================================================
# Unit Operations Prompt
# ==============================================================================
UNIT_OPS_PROMPT = """너는 문명6 에이전트야. 스크린샷에 선택된 유닛의 정보를 보고 판단해.

{json_instruction}

판단 기준:
1. 만약 선택된 유닛이 '개척자'(Settler)라면:
   - 'b' 키를 눌러 도시를 건설해.

2. 만약 선택된 유닛이 '건설자'(Builder)라면:
   - 유닛이 자원 타일 위에 있으면 해당 개선 시설을 건설해 (클릭).
   - 아니면 근처 자원 타일로 우클릭해서 이동해.

3. 만약 선택된 유닛이 전투 유닛(전사, 궁수, 기병 등)이라면:
   - 유닛의 위치를 제외한 화면의 하늘색 타일(이동 가능 영역) 중
     전략적으로 좋은 곳을 골라 우클릭(button: "right")으로 이동해.
   - 적 유닛이 인접해 있으면 적 유닛을 우클릭해서 공격해.

4. 만약 선택된 유닛이 '정찰병'(Scout)이라면:
   - 아직 탐색하지 않은 가려진 영역 방향의 하늘색 타일을 우클릭해서 이동해.

5. 만약 유닛의 행동력이 없다면 (이동 불가 상태):
   - 아무 행동도 하지 않고 대기. press "enter" 키를 눌러 다음으로 넘겨.

유닛 이름, 현재 위치, 행동력 상태를 확인한 뒤 최적의 행동을 결정해."""

# ==============================================================================
# Popup Handling Prompt
# ==============================================================================
POPUP_PROMPT = """너는 문명6 에이전트야. 화면에 나타난 팝업이나 알림을 처리해야 해.

{json_instruction}

판단 기준:
1. 화면에 팝업이 나타났고, '예/아니오' 또는 '확인/취소' 버튼이 있다면:
   - 'enter' 키를 눌러 확인해.

2. 화면에 팝업이 나타났는데 선택 버튼이 없다면 (정보성 팝업):
   - 'esc' 키를 눌러 닫아.

3. 화면 오른쪽 맨 아래에 '다음 턴'이라는 글자와 화살표 아이콘이 보이면:
   - 'enter' 키를 눌러 다음 턴으로 넘겨.

4. 화면 오른쪽 맨 아래에 '연구 선택'이라는 글자와 파란색 플라스크 아이콘이 보이면:
   - 해당 '연구 선택' 버튼을 클릭해.

5. 화면 오른쪽 맨 아래에 '생산 품목'이라는 글자와 주황색 톱니바퀴 아이콘이 보이면:
   - 해당 '생산 품목' 버튼을 클릭해.

6. 화면 오른쪽 맨 아래에 '사회 제도 선택'이라는 글자가 보이면:
   - 해당 버튼을 클릭해.

7. 기타 닫을 수 있는 알림/팝업이 보이면:
   - 'esc' 키를 눌러 닫아.

화면의 팝업 내용을 정확히 읽고 적절한 행동을 결정해."""

# ==============================================================================
# Unified Research Manager Prompt
# ==============================================================================
RESEARCH_MANAGER_PROMPT = """너는 문명6 에이전트야. 현재 게임 상황에 맞춰 최적의 기술(Science)을 연구해야 해.

{json_instruction}

현재 화면 상태를 인식하고 아래 기준에 따라 행동해:

상태 1: '기술 발전표(Tech Tree)' 또는 '연구 선택 팝업'이 열려 있는 경우
   1. 화면에 보이는 기술들 중 '연구 가능' 상태인 항목들을 식별해.
   2. 우선순위 판단:
      - 1순위: '유레카(부스트)' 조건이 이미 달성되어 연구 시간이 단축된 기술.
      - 2순위: 현재 전략에 필수적이거나, 남은 턴 수가 가장 적은 기술.
   3. 행동: 해당 기술 아이콘을 클릭해. (선택 후 자동으로 닫히지 않으면 ESC로 닫기)

상태 2: 연구 선택 화면이 열려 있지 않은데, 연구가 필요한 경우 (우측 하단 '연구 선택' 알림, 플라스크 모양이 있을때)
   1. 상단 UI의 '플라즈마' 아이콘을 클릭하거나, 우측 하단 행동 알림을 클릭해 기술 트리를 열어.

판단 시 주의사항:
- 단순히 턴 수가 적은 것만 고르지 말고, 유레카가 발동된 효율 좋은 기술을 놓치지 마.
- 이미 완료된 기술이나, 선행 기술이 부족해 잠겨있는 기술은 선택하지 마.

현재 화면에 보이는 기술 이름, 턴 수, 유레카 여부를 분석해서 JSON으로 응답해."""

# ==============================================================================
# City Production Prompt
# TODO: 건물 지을때 위치 확인후 짓는건 따로 case 처리 필요
# ==============================================================================
CITY_PRODUCTION_PROMPT = """너는 문명6 에이전트야. 도시에서 생산할 품목을 선택해야 해.

{json_instruction}

판단 기준:
1. 화면 오른쪽에 '생산 품목 선택' 팝업이 나타났다면:
   - 생산 가능한 품목 리스트에서 각 항목의 남은 턴 수를 확인해.
   - 남은 턴 수가 가장 적은(가장 빨리 완료되는) 생산 품목을 찾아서 클릭해.

2. 도시 화면이 열려 있고 생산 큐가 비어있다면:
   - 생산 가능한 품목 중 턴이 가장 적은 것을 클릭해.

3. 생산 품목 선택 후에는 자동으로 닫히므로 추가 행동 불필요.

생산 품목의 이름과 남은 턴 수를 정확히 읽고 최적의 품목을 선택해."""

# ==============================================================================
# Unified Culture Manager Prompt
# ==============================================================================
CULTURE_MANAGER_PROMPT = """너는 문명6 에이전트야. 현재 게임 상황에 맞춰 최적의 사회 제도(Civics)를 연구해야 해.

{json_instruction}

현재 화면 상태를 인식하고 아래 기준에 따라 행동해:

상태 1: '사회 제도표(Civics Tree)' 또는 좌측에 '제도 선택 팝업'이 열려 있는 경우
   1. 화면에 보이는 사회 제도들 중 '연구 가능' 상태인 항목들을 식별해.
   2. 우선순위 판단:
      - 1순위: '영감(Inspiration)' 조건이 이미 달성되어 연구 시간이 단축된 제도.
      - 2순위: 현재 전략에 필수적이거나, 남은 턴 수가 가장 적은 제도.
   3. 행동: 해당 사회 제도 아이콘을 클릭해. (선택 후 자동으로 닫히지 않으면 ESC로 닫기)

상태 2: 제도 선택 화면이 열려 있지 않은데, 연구가 필요한 경우
   (우측 하단 '사회제도 선택' 알림, 보라색 음표/책 모양이 있을 때)
   1. 상단 UI의 '문화(책/음표)' 아이콘을 클릭하거나,
      우측 하단 행동 알림을 클릭해 사회 제도 트리를 열어.

판단 시 주의사항:
- 단순히 턴 수가 적은 것만 고르지 말고, 영감(부스트)이 발동된 효율 좋은 제도를 놓치지 마.
- 이미 완료된 제도나, 선행 제도가 부족해 잠겨있는 제도는 선택하지 마.

현재 화면에 보이는 사회 제도 이름, 턴 수, 영감 달성 여부를 분석해서 JSON으로 응답해."""

# ==============================================================================
# Diplomatic Prompt (for future use)
# ==============================================================================
DIPLOMATIC_PROMPT = """너는 문명6 에이전트야. 외교 상호작용을 처리해야 해.

{json_instruction}

판단 기준:
1. 외교 화면이 열려 있다면:
   - 상대 문명의 제안을 읽고 유리하면 수락, 불리하면 거절해.
   - 수락: 확인/수락 버튼 클릭.
   - 거절: 취소/거절 버튼 클릭 또는 'esc' 키 입력.

2. 전쟁 선포 알림이 나타났다면:
   - esc 버튼을 눌러

3. 거래 제안이 있다면:
   - 유리한 거래면 수락 클릭.
   - 불리하면 esc 버튼 눌러.

외교 상황을 정확히 파악하고 적절한 행동을 결정해."""

# ==============================================================================
# Combat Prompt (TODO: for future use)
# ==============================================================================
COMBAT_PROMPT = """너는 문명6 에이전트야. 전투 상황을 처리해야 해.

{json_instruction}

판단 기준:
1. 전투 유닛이 선택되어 있고 적이 인접해 있다면:
   - 적 유닛을 우클릭(button: "right")해서 공격해.

2. 공성 유닛이라면:
   - 사거리 내 적 유닛이나 도시를 우클릭해서 원거리 공격해.

3. 유닛 체력이 낮다면 (빨간색 HP 바):
   - 안전한 후방 타일로 우클릭해서 후퇴해.
   - 또는 'f' 키를 눌러 방어 태세.

4. 도시 공격이 가능하다면:
   - 적 도시를 우클릭해서 공격해.

적 위치, 아군 유닛 체력, 지형을 파악하고 최적의 전투 행동을 결정해."""

# ==============================================================================
# TODO: Policy selection Prompt
# (for future use - drag and drop, 설명 읽는것 필요 - visual grounding 필요)
# ==============================================================================


# ==============================================================================
# Custom prompt builder (TODO: Implement later)
# ==============================================================================
def build_custom_prompt(
    scenario: str,
    focus_areas: list[str],
    include_json_format: bool = True,
    normalizing_range: int = 1000,
) -> str:
    """
    Build a custom prompt for specific scenarios.

    Args:
        scenario: Description of the game scenario
        focus_areas: List of specific areas to focus on
        include_json_format: Whether to include JSON format instructions
        normalizing_range: Coordinate normalization range

    Returns:
        Custom prompt string
    """
    prompt = f"Analyze this Civilization VI screenshot showing {scenario}.\n\n"

    if include_json_format:
        prompt += JSON_FORMAT_INSTRUCTION.format(normalizing_range=normalizing_range) + "\n"

    prompt += "Focus on:\n"
    for area in focus_areas:
        prompt += f"- {area}\n"

    return prompt
