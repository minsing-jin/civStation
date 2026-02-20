"""
Strategy Schemas - Data structures for strategic planning.

Defines the StructuredStrategy dataclass and related enums
for representing game strategies in a structured format.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VictoryType(str, Enum):
    """Types of victory conditions in Civilization VI."""

    SCIENCE = "science"
    CULTURE = "culture"
    DOMINATION = "domination"
    RELIGIOUS = "religious"
    DIPLOMATIC = "diplomatic"
    SCORE = "score"


class GamePhase(str, Enum):
    """General phases of a Civilization game."""

    EARLY_EXPANSION = "early_expansion"  # Turns 1-50: Settling, exploring
    MID_DEVELOPMENT = "mid_development"  # Turns 50-150: Building infrastructure
    LATE_CONSOLIDATION = "late_consolidation"  # Turns 150-250: Pushing toward victory
    VICTORY_PUSH = "victory_push"  # Turns 250+: Final push for victory


@dataclass
class StructuredStrategy:
    """
    Structured representation of a game strategy.

    This is the output of the Strategy Planner and is used to guide
    primitive-level decision making throughout the game.

    - text: LLM이 작성한 전략 본문 (한국어, free-form). 프롬프트에 직접 주입됨.
    - victory_goal / current_phase: 프로그래밍 접근용 메타데이터.
    """

    text: str = ""
    victory_goal: VictoryType = VictoryType.SCIENCE
    current_phase: str = "early_expansion"
    primitive_hint: str = ""  # micro-level behavioral directive from HITL

    def to_prompt_string(self) -> str:
        """
        Convert strategy to a string suitable for LLM prompts.

        Returns the text body directly — no lossy conversion needed.
        """
        return self.text

    @classmethod
    def default_science_strategy(cls) -> "StructuredStrategy":
        """Create a default science victory strategy."""
        return cls(
            text=(
                "[과학 승리 전략]\n"
                "캠퍼스 건설을 최우선으로 하고, 도서관→대학→연구소 순으로 과학 인프라를 확보한다. "
                "초반에는 도시 2-3개를 빠르게 정착하고 인구 성장에 집중한다.\n"
                "우선순위: 캠퍼스 건설 > 핵심 기술 연구 > 도시 성장 > 방어력 유지\n"
                "제약사항: 불필요한 전쟁 회피, 군사비 과다 지출 지양\n"
                "즉각적 목표: 도서관 건설, 필기 연구, 제2도시 정착\n"
                "장기 목표: 캠퍼스 3개 이상, 현대 시대 도달, 우주선 발사"
            ),
            victory_goal=VictoryType.SCIENCE,
            current_phase="early_expansion",
        )

    @classmethod
    def default_culture_strategy(cls) -> "StructuredStrategy":
        """Create a default culture victory strategy."""
        return cls(
            text=(
                "[문화 승리 전략]\n"
                "극장가 건설을 최우선으로 하고, 위인 확보와 유산 건설을 통해 관광을 극대화한다. "
                "초반에는 문화 출력을 높여 정책 카드를 빠르게 확보하고, 걸작을 수집한다.\n"
                "우선순위: 극장가 건설 > 위인 확보 > 유산 건설 > 관광 강화\n"
                "제약사항: 유산 경쟁 패배 시 대안 마련, 외교적 고립 방지\n"
                "즉각적 목표: 원형극장 건설, 걸작 확보, 축제 개최\n"
                "장기 목표: 5개 극장가 건설, 록밴드 활용, 해외 관광객 유치"
            ),
            victory_goal=VictoryType.CULTURE,
            current_phase="early_expansion",
        )

    @classmethod
    def default_domination_strategy(cls) -> "StructuredStrategy":
        """Create a default domination victory strategy."""
        return cls(
            text=(
                "[지배 승리 전략]\n"
                "군사 유닛 생산과 병영 건설을 최우선으로 하고, 전략 자원을 조기에 확보한다. "
                "초반에는 정찰로 적 위치를 파악하고, 궁수 업그레이드 후 공세를 시작한다.\n"
                "우선순위: 군사 유닛 생산 > 병영 건설 > 전략 자원 확보 > 적 수도 점령\n"
                "제약사항: 다수 문명과 동시 전쟁 회피, 보급선 유지\n"
                "즉각적 목표: 전사 2기 생산, 궁수 업그레이드, 정찰 수행\n"
                "장기 목표: 모든 수도 점령, 군사 기술 우위 유지, 동맹 활용"
            ),
            victory_goal=VictoryType.DOMINATION,
            current_phase="early_expansion",
        )


class HITLInputRequiredError(Exception):
    """Raised when HITL mode is enabled but no human input was provided."""

    pass


def parse_strategy_json(raw_json: str) -> StructuredStrategy:
    """
    Parse a JSON string into a StructuredStrategy.

    Expected format:
        {"victory_goal": "science", "current_phase": "early_expansion", "text": "..."}

    Raises:
        json.JSONDecodeError: If raw_json is not valid JSON
        ValueError: If the parsed text is too short (< 20 chars)
    """
    data = json.loads(raw_json)

    # Parse victory_goal with fallback
    victory_goal_str = data.get("victory_goal", "science").lower()
    try:
        victory_goal = VictoryType(victory_goal_str)
    except ValueError:
        logger.warning(f"Unknown victory type: {victory_goal_str}, defaulting to science")
        victory_goal = VictoryType.SCIENCE

    text = data.get("text", "")
    if len(text) < 20:
        raise ValueError(f"Strategy text too short ({len(text)} chars, minimum 20)")

    return StructuredStrategy(
        text=text,
        victory_goal=victory_goal,
        current_phase=data.get("current_phase", "early_expansion"),
        primitive_hint=data.get("primitive_hint", ""),
    )
