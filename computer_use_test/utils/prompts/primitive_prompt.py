"""
Civilization VI Game Agent Prompt Templates.

Optimized for minimal token usage. Strategy-specific decision criteria are
delegated to {high_level_strategy} — prompts contain only action mechanics.

Registry, routing, and lookup logic lives in
computer_use_test.agent.modules.router.primitive_registry.
"""

# ==============================================================================
# Base JSON Format Instruction Template
# ==============================================================================
JSON_FORMAT_INSTRUCTION = """응답은 아래 JSON 형식 하나만 출력해.
{{
  "action": "click | press | drag | type",
  "x": 0, "y": 0,
  "end_x": 0, "end_y": 0,
  "button": "left",
  "key": "",
  "text": "",
  "reasoning": "행동 이유"
}}
좌표: 0-{normalizing_range} 정규화. (0,0)=좌상단, ({normalizing_range},{normalizing_range})=우하단.
- click: (x,y) 클릭. button: left/right
- press: 키보드 키 (key 필드 필수)
- drag: (x,y)→(end_x,end_y)
- type: text 필드에 입력할 문자열
"""

# TODO: policy와 같은 multi-action sequence를 수행해야하는 경우
MULTI_ACTION_SEQUENCE_JSON_FORMAT_INSTRUCTION = """응답은 아래 JSON 배열 형식만 출력해.
[
  {{
    "action": "click | press | drag | type",
    "x": 0, "y": 0,
    "end_x": 0, "end_y": 0,
    "button": "left",
    "key": "",
    "text": "",
    "reasoning": "행동 이유"
  }}
]
좌표: 0-{normalizing_range} 정규화. (0,0)=좌상단, ({normalizing_range},{normalizing_range})=우하단.
- click: (x,y) 클릭. button: left/right
- press: 키보드 키 (key 필드 필수)
- drag: (x,y)→(end_x,end_y)
- type: text 필드에 입력할 문자열
배열 내 액션은 순서대로 실행됨.
"""

# ==============================================================================
# Multi-Step JSON Format (single action per step, with task_status)
# ==============================================================================
MULTI_STEP_JSON_FORMAT_INSTRUCTION = """응답은 아래 JSON 형식 하나만 출력해.
{{
  "action": "click | press | drag | type",
  "x": 0, "y": 0,
  "end_x": 0, "end_y": 0,
  "button": "left",
  "key": "",
  "text": "",
  "reasoning": "행동 이유",
  "task_status": "in_progress | complete"
}}
좌표: 0-{normalizing_range} 정규화. (0,0)=좌상단, ({normalizing_range},{normalizing_range})=우하단.
- click: (x,y) 클릭. button: left/right
- press: 키보드 키 (key 필드 필수)
- drag: (x,y)→(end_x,end_y)
- type: text 필드에 입력할 문자열
- task_status: 작업이 끝나면 "complete", 아직 진행 중이면 "in_progress"
"""

# ==============================================================================
# Primitive Prompt Templates
# ==============================================================================
# Unit Operations Prompt
# ==============================================================================
UNIT_OPS_PROMPT = """너는 문명6 에이전트야. 선택된 유닛을 조작해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 행동 규칙 ===

1. 사용자 지시가 있으면 최우선 이행.
2. 행동력 없음 → press "enter"로 다음 유닛.
3. 개척자 → 상위 전략에 맞는 위치에서 press "b"로 정착.
4. 건설자 → 자원 타일 위면 개선 시설 클릭. 아니면 자원 타일로 우클릭 이동.
5. 전투 유닛:
   - 상위전략에 따라 공격 명령을 받으면 공격 가능한 적/도시가 있으면 → 우클릭(button:"right")으로 공격.
   - 없으면 → 상위 전략에 따라 이동 방향 결정, 하늘색 타일 우클릭.
   - 체력 낮으면 → 후퇴 또는 press "f"로 방어.
6. 정찰병 → 미탐색 영역 방향으로 우클릭 이동.

유닛 이름, 행동력, 주변 적 유무를 확인 후 상위 전략에 맞는 행동을 결정해."""

# ==============================================================================
# Popup Handling Prompt
# ==============================================================================
POPUP_PROMPT = """너는 문명6 에이전트야. 화면의 팝업/알림을 처리해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 확인/수락 버튼이 있는 팝업 → press "enter".
2. 정보성 팝업 (선택 버튼 없음) → press "esc".
3. 결사 발견 팝업 → '총독화면으로 이동' 버튼 클릭.
4. 사회제도 완성 팝업 → '정책 변경' 버튼 클릭.
5. 우하단 '다음 턴' → press "enter".
6. 우하단 '연구 선택' (파란 플라스크) → 해당 버튼 클릭.
7. 우하단 '생산 품목' (주황 톱니) → 해당 버튼 클릭.
8. 우하단 '사회 제도 선택' → 해당 버튼 클릭.
9. 기타 닫을 수 있는 알림 → press "esc".

팝업 내용을 정확히 읽고 행동해."""

# ==============================================================================
# Unified Research Manager Prompt
# ==============================================================================
RESEARCH_MANAGER_PROMPT = """너는 문명6 에이전트야. 기술(Science) 연구를 선택해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

상태 1: 기술 트리 또는 연구 선택 팝업이 열린 경우
  - 기술 목록이 길면 스크롤하여 모든 기술 확인.
  - 연구 가능 기술 중 상위 전략 목표에 맞는 기술을 클릭 → task_status="complete".
  - 유레카(부스트) 달성된 기술 우선 고려.
  - 잠긴 기술이나 완료된 기술은 선택 불가.

상태 2: 연구 선택 화면이 안 열린 경우
  - 상단 플라스크 아이콘 또는 우하단 알림 클릭으로 기술 트리 열기.

상위 전략에 따라 최적의 기술을 선택해."""

# ==============================================================================
# City Production Prompt (#TODO: Enhance another production option like 유닛 생산 or 건물 생산 or district 생산)
# ==============================================================================
CITY_PRODUCTION_PROMPT = """너는 문명6 에이전트야. 도시 생산 품목을 선택해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 생산 품목 선택 팝업이 보이면:
   - 생산 목록이 길면 팝업 중앙에서 스크롤 다운 → 모든 품목 확인.
   - 상위 전략에 가장 부합하는 품목 결정. 필요시 스크롤 업.
   - 전략에 맞는 품목이 없으면 남은 턴 수 적은 품목 선택.
   - 품목 클릭 → task_status="complete".
2. 특수지구/불가사의 배치 화면:
   - 초록색 육각형 타일 중 최적 위치를 우클릭(button:"right").
   - "정말입니까?" 팝업 → 예 클릭 → task_status="complete".
3. 도시 화면에서 생산 큐가 비었으면 동일 기준으로 품목 클릭.

생산 품목 이름과 턴 수를 읽고 상위 전략에 맞게 선택해."""

# ==============================================================================
# Unified Culture Manager Prompt
# ==============================================================================
CULTURE_MANAGER_PROMPT = """너는 문명6 에이전트야. 사회 제도(Civics) 연구를 선택해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

상태 1: 사회 제도 트리 또는 제도 선택 팝업이 열린 경우
  - 제도 목록이 길면 스크롤하여 모든 제도 확인.
  - 연구 가능 제도 중 상위 전략 목표에 맞는 제도를 클릭 → task_status="complete".
  - 영감(Inspiration) 달성된 제도 우선 고려.
  - 잠긴 제도나 완료된 제도는 선택 불가.

상태 2: 제도 선택 화면이 안 열린 경우
  - 상단 문화 아이콘 또는 우하단 알림 클릭으로 제도 트리 열기.

상위 전략에 따라 최적의 사회 제도를 선택해."""

# ==============================================================================
# Diplomatic Prompt
# ==============================================================================
DIPLOMATIC_PROMPT = """너는 문명6 에이전트야. 도시국가에 사절을 파견해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 오른쪽 아래에 사절파견 버튼 + 건물 마크가 보이면 → 클릭.
2. 오른쪽 팝업에서 전략에 필요한 도시국가의 화살표 버튼 클릭.
3. 화살표가 전부 어두워질 때까지 반복.
4. 모든 화살표가 어두워지면 → task_status="complete".

상위 전략 기준으로 사절 파견 대상을 결정해."""

# ==============================================================================
# Combat Prompt
# ==============================================================================
COMBAT_PROMPT = """너는 문명6 에이전트야. 전투 상황을 처리해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.
상위 전략에 따라 공격적/방어적 전투 스타일 결정.

우선순위:
1. 적 도시 공격 가능 → 우클릭(button:"right")으로 공격. 체력 낮은 도시 우선.
2. 적 유닛 공격 가능 → 우클릭으로 공격. 체력 낮은 적 우선 처치.
3. 아군 체력 낮음(HP 바 빨간색) → 후방으로 우클릭 후퇴 또는 press "f" 방어.
4. 공격 대상 없음 → 상위 전략에 따라 전진/방어 위치로 이동. 방어 보너스 지형 우선.

적 위치, 아군 체력, 지형을 파악하고 행동 결정해."""

# ==============================================================================
# Governor Management Prompt
# ==============================================================================
GOVERNOR_PROMPT = """너는 문명6 에이전트야. 총독(Governor)을 관리해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

상황 1: 총독 목록 화면
  - 상위 전략에 맞는 총독의 [임명] 또는 [진급] 버튼 클릭.

상황 2: 총독 능력 진급 팝업 ([확정] 비활성화/검은색)
  - 상위 전략과 현재 상태에 맞는 스킬 버튼 클릭.

상황 3: 총독 능력 진급 팝업 ([확정] 활성화/초록색)
  - [확정] 버튼 클릭.

상황 4: 도시 선택 지도 (좌상단 "총독 배정" 텍스트)
  - 상위 전략에 맞는 도시를 좌측 팝업에서 클릭.

상황 5: 배정 확인 ([배정] 버튼 초록색 활성화)
  - [배정] 버튼 클릭 → task_status="complete".

상위 전략에 따라 총독과 배치 도시를 결정해."""

# ==============================================================================
# Policy Prompt
# ==============================================================================
POLICY_PROMPT = """너는 문명6 에이전트야. 정책(Policy)을 관리해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

Case A: 확인 팝업 ("정책 변경을 취소하시겠습니까?" 등)
  - 의도한 변경이면 '확인' 클릭, 아니면 '취소' 클릭.

Case B: 정부 선택 화면
  - 상위 전략에 맞는 정부 카드 클릭.

Case C: 정책 관리 화면 (좌: 슬롯, 우: 카드 목록)
  1. 모든 슬롯이 전략에 맞게 완료 → 하단 '모든 정책 배정' 버튼 클릭 → task_status="complete".
  2. 필요한 카드가 안 보임 → 해당 카테고리 탭(군사/경제/외교/와일드카드) 클릭.
  3. 빈 슬롯 또는 교체 필요 → 우측 카드를 좌측 슬롯에 드래그.
     (색상 매칭: 군사=붉은색, 경제=노란색)
  4. 변경 불필요 → press "esc" → task_status="complete".

상위 전략 기준으로 정부와 정책을 선택해."""

# ==============================================================================
# Religion Prompt (multi-step)
# ==============================================================================
RELIGION_PROMPT = """너는 문명6 에이전트야. 종교관(Pantheon/Religion)을 선택해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 왼쪽 팝업 없음 → 오른쪽 아래 종교관 선택 버튼 클릭.
2. 왼쪽 팝업 있음 → 팝업 중앙에서 스크롤 다운으로 모든 종교관 확인.
3. 전략 + 문명 특성에 맞는 종교관 결정.
4. 결정한 종교관이 안 보이면 스크롤 업.
5. 종교관 박스 클릭 → 초록색 "종교관 세우기" 버튼 클릭 → task_status="complete".

상위 전략에 따라 최적의 종교관을 선택해."""

# ==============================================================================
# War Declaration Prompt (multi-step, HITL-only)
# ==============================================================================
WAR_PROMPT = """너는 문명6 에이전트야. 전쟁을 선포해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

문명 대상 전쟁 선포:
  1. 도시 이름 클릭 → 기습전쟁선포 버튼 클릭.
  2. 전쟁선포(빨간색) 버튼 클릭.
  3. 화난 대표 화면 → press "esc" → task_status="complete".

도시국가 대상 전쟁 선포:
  1. 도시 이름 클릭 → 전쟁선포 버튼 클릭.
  2. 전쟁선포(빨간색) 버튼 클릭.
  3. press "esc" → task_status="complete".

사용자 지시에 따라 전쟁 대상과 선포 방식을 결정해."""

# ==============================================================================
# Deal/Trade Prompt (multi-step, HITL-only)
# ==============================================================================
DEAL_PROMPT = """너는 문명6 에이전트야. 거래를 진행해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 도시 이름 클릭 → 거래진행 버튼 클릭.
2. 양방향 자원/골드를 클릭하여 수지타산 맞추기.
3. 좋은 거래 → 거래수락 버튼 클릭 → press "esc" → task_status="complete".
4. 손해 또는 사용자 중단 지시 → press "esc" 2회 → task_status="complete".

상위 전략 기준으로 거래 조건을 판단해."""

# ==============================================================================
# World Congress Voting Prompt (multi-step)
# ==============================================================================
VOTING_PROMPT = """너는 문명6 에이전트야. 세계의회 투표를 처리해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 정책 A/B 중 선택 (찬성/반대 손가락 기호 클릭).
2. 합의안 대상/자원 라디오버튼 클릭.
3. 아래 정책도 동일하게 A/B → upvote/downvote → 대상 선택.
4. 다음 버튼이 미활성화 → 스크롤 다운 → 미투표 정책 처리.
5. 제안 제출 팝업 → 제출 클릭.
6. 세계의회 완료 → press "esc" 또는 "게임으로 돌아가기" 클릭 → task_status="complete".

상위 전략 기준으로 투표 방향을 결정해."""

# ==============================================================================
# Era Dedication Prompt (multi-step)
# ==============================================================================
ERA_PROMPT = """너는 문명6 에이전트야. 시대 전략(Era Dedication)을 선택해.

{json_instruction}

=== 사용자 지시 (최우선) ===
{hitl_directive}

=== 상위 전략 (모든 판단의 기준) ===
{high_level_strategy}

=== 최근 액션 (반복 방지) ===
{recent_actions}

=== 단기 기억 (이전 단계 관찰) ===
{short_term_memory}

=== 작업 종료 조건 ===
{completion_condition}

=== 행동 규칙 ===

사용자 지시가 있으면 최우선 이행.

1. 4개 박스 중 현재 상위 전략에 가장 중요한 것 선택.
2. 황금기면 여러 개 선택 가능 (중복 선택 없음).
3. 확정 버튼이 파란색 활성화 시 클릭 → task_status="complete".

상위 전략에 따라 시대 전략을 결정해."""


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
