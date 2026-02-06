"""
Strategy Prompts - LLM prompts for strategy generation and refinement.

Contains prompts used by the StrategyPlanner to:
1. Refine human input into structured strategy (HITL mode)
2. Generate strategy autonomously from context
3. Update existing strategies based on game changes
"""

STRATEGY_REFINEMENT_PROMPT = """너는 문명6 게임의 전략 분석가야.
플레이어의 자유형식 전략 입력을 분석하고 구조화된 전략으로 변환해야 해.

=== 현재 게임 상태 ===
{context_string}

=== 플레이어 입력 ===
{raw_input}

=== 작업 ===
플레이어의 의도를 파악하고, 아래 JSON 형식으로 구조화된 전략을 생성해.
각 필드를 한국어로 채워야 해.

응답 형식 (JSON만 출력):
{{
    "victory_goal": "science" | "culture" | "domination" | "religious" | "diplomatic" | "score",
    "current_phase": "early_expansion" | "mid_development" | "late_consolidation" | "victory_push",
    "priorities": ["우선순위1", "우선순위2", "우선순위3", "우선순위4"],
    "focus_areas": ["집중분야1", "집중분야2", "집중분야3"],
    "constraints": ["제약사항1", "제약사항2"],
    "immediate_objectives": ["즉각목표1", "즉각목표2", "즉각목표3"],
    "long_term_objectives": ["장기목표1", "장기목표2", "장기목표3"],
    "notes": "추가 전략 노트"
}}

중요:
- 플레이어 입력에서 명시되지 않은 필드는 게임 상태를 바탕으로 합리적으로 추론해
- 우선순위는 가장 중요한 것부터 순서대로 나열해
- 즉각적 목표는 10-20턴 내 달성 가능한 것으로
- 장기 목표는 50-100턴 내 달성을 목표로
- 제약사항에는 피해야 할 행동이나 상황을 포함해
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

응답 형식 (JSON만 출력):
{{
    "victory_goal": "science" | "culture" | "domination" | "religious" | "diplomatic" | "score",
    "current_phase": "early_expansion" | "mid_development" | "late_consolidation" | "victory_push",
    "priorities": ["우선순위1", "우선순위2", "우선순위3", "우선순위4"],
    "focus_areas": ["집중분야1", "집중분야2", "집중분야3"],
    "constraints": ["제약사항1", "제약사항2"],
    "immediate_objectives": ["즉각목표1", "즉각목표2", "즉각목표3"],
    "long_term_objectives": ["장기목표1", "장기목표2", "장기목표3"],
    "notes": "전략 분석 요약"
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

응답 형식 (JSON만 출력):
{{
    "victory_goal": "기존과 동일하거나 변경된 승리 유형",
    "current_phase": "현재 단계",
    "priorities": ["업데이트된 우선순위 목록"],
    "focus_areas": ["업데이트된 집중 분야"],
    "constraints": ["업데이트된 제약사항"],
    "immediate_objectives": ["업데이트된 즉각적 목표"],
    "long_term_objectives": ["업데이트된 장기 목표"],
    "notes": "변경 사항 요약"
}}

업데이트 원칙:
- 승리 유형은 특별한 이유 없이 변경하지 않아
- 새로운 위협이 발생하면 우선순위와 제약사항을 조정해
- 목표 달성 시 새로운 목표로 교체해
- 상황 악화 시 방어적 조정을 해
"""
