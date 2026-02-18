"""
Strategy Prompts - LLM prompts for strategy generation and refinement.

Contains prompts used by the StrategyPlanner to:
1. Refine human input into structured strategy (HITL mode)
2. Generate strategy autonomously from context
3. Update existing strategies based on game changes
"""

STRATEGY_REFINEMENT_PROMPT = """너는 문명6 게임의 전략 분석가야.
플레이어의 자유형식 전략 입력을 분석하고, 구체적인 전략 본문(text)을 작성해야 해.

=== 현재 게임 상태 ===
{context_string}

=== 플레이어 입력 ===
{raw_input}

=== 작업 ===
플레이어의 의도를 파악하고, 아래 JSON 형식으로 전략을 생성해.

text 작성 가이드:
1. 핵심 전략 방향과 그 이유를 먼저 서술
2. 우선순위를 ">" 구분으로 명시 (예: "캠퍼스 건설 > 기술 연구 > 도시 성장")
3. 위협이나 제약사항을 구체적으로 언급
4. 즉각적 목표 (10-20턴 내)를 구체적 건물/기술/유닛 이름으로 제시
5. 장기 경로 (50-100턴)를 간략히 서술
6. 200-400자 분량의 한국어로 작성

예시:
플레이어 입력: "과학으로 가고 싶어, 근데 옆에 적이 있어"
좋은 text:
"[과학 승리 전략]
캠퍼스 건설을 최우선으로 하되, 인접 적 문명에 대한 방어를 병행한다.
초반에는 궁수 2-3기로 방어선을 구축하고, 안전이 확보되면 캠퍼스→도서관→대학 순으로 과학 인프라를 확보한다.
우선순위: 방어력 확보 > 캠퍼스 건설 > 핵심 기술 연구 > 도시 성장
제약사항: 공격적 전쟁 지양, 방어 유닛 최소 3기 유지
즉각적 목표: 궁수 2기 생산, 성벽 건설, 필기 연구
장기 목표: 캠퍼스 3개 건설, 적 문명과 평화 유지, 현대 시대 도달"

응답 형식 (JSON만 출력):
{{
    "victory_goal": "science" | "culture" | "domination" | "religious" | "diplomatic" | "score",
    "current_phase": "early_expansion" | "mid_development" | "late_consolidation" | "victory_push",
    "text": "전략 본문 (200-400자, 한국어)"
}}

중요:
- 플레이어 입력에서 명시되지 않은 부분은 게임 상태를 바탕으로 구체적으로 추론해
- 우선순위는 가장 중요한 것부터 순서대로 나열해
- 문명6 도메인 지식을 활용해 구체적인 건물, 기술, 유닛 이름을 사용해
- text에는 전략의 핵심 방향, 우선순위, 제약사항, 즉각적 목표, 장기 목표를 모두 포함해
"""

AUTONOMOUS_STRATEGY_PROMPT = """너는 문명6 게임의 전략 AI야.
현재 게임 상태를 분석하고 최적의 전략을 자율적으로 생성해야 해.

=== 현재 게임 상태 ===
{context_string}

=== 분석 지침 ===
1. 현재 턴과 시대를 고려해 적절한 단계(phase)를 결정해
2. 자원 상황과 문명 특성을 고려해 승리 유형을 선택해
3. 주변 상황(전쟁, 외교 관계)을 고려해 제약사항을 설정해
4. 현재 생산/연구 상황을 고려해 즉각적 목표를 설정해

text 작성 가이드:
1. 핵심 전략 방향과 그 이유를 먼저 서술
2. 우선순위를 ">" 구분으로 명시
3. 위협이나 제약사항을 구체적으로 언급
4. 즉각적 목표 (10-20턴 내)를 구체적 건물/기술/유닛 이름으로 제시
5. 장기 경로 (50-100턴)를 간략히 서술
6. 200-400자 분량의 한국어로 작성

응답 형식 (JSON만 출력):
{{
    "victory_goal": "science" | "culture" | "domination" | "religious" | "diplomatic" | "score",
    "current_phase": "early_expansion" | "mid_development" | "late_consolidation" | "victory_push",
    "text": "전략 본문 (200-400자, 한국어)"
}}

단계 판단 기준:
- early_expansion (턴 1-50): 정착, 탐험, 초기 인프라
- mid_development (턴 50-150): 특구 건설, 기술 발전, 군비 확충
- late_consolidation (턴 150-250): 승리 조건 집중, 외교 관리
- victory_push (턴 250+): 최종 승리 추진

중요:
- 현재 전쟁 중이라면 군사 우선순위를 높여
- 자원이 부족하면 경제 회복을 우선시해
- 위협 요소가 있다면 제약사항에 반영해
- 문명6 도메인 지식을 활용해 구체적인 건물, 기술, 유닛 이름을 사용해
"""

STRATEGY_UPDATE_PROMPT = """너는 문명6 게임의 전략 분석가야.
게임 상황 변화에 따라 기존 전략을 업데이트해야 해.

=== 현재 게임 상태 ===
{context_string}

=== 기존 전략 ===
{current_strategy}

=== 업데이트 이유 ===
{update_reason}

=== 작업 ===
게임 상황 변화를 반영해 전략을 업데이트해.
기존 전략의 핵심 방향은 유지하되, 필요한 조정을 해.

text 작성 가이드:
1. 기존 전략의 핵심 방향을 유지하면서 변경 사항을 반영
2. 우선순위를 ">" 구분으로 명시
3. 새로운 위협이나 제약사항을 구체적으로 언급
4. 즉각적 목표 (10-20턴 내)를 구체적 건물/기술/유닛 이름으로 제시
5. 장기 경로를 간략히 서술
6. 200-400자 분량의 한국어로 작성

응답 형식 (JSON만 출력):
{{
    "victory_goal": "기존과 동일하거나 변경된 승리 유형",
    "current_phase": "현재 단계",
    "text": "업데이트된 전략 본문 (200-400자, 한국어)"
}}

업데이트 원칙:
- 승리 유형은 특별한 이유 없이 변경하지 않아
- 새로운 위협이 발생하면 우선순위와 제약사항을 조정해
- 목표 달성 시 새로운 목표로 교체해
- 상황 악화 시 방어적 조정을 해
"""
