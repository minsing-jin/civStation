"""
Strategy Schemas - Data structures for strategic planning.

Defines the StructuredStrategy dataclass and related enums
for representing game strategies in a structured format.
"""

from dataclasses import dataclass, field
from enum import Enum


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
class StrategyConstraint:
    """A constraint or limitation on strategy execution."""

    description: str
    severity: str = "medium"  # "low", "medium", "high"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.description}"


@dataclass
class StructuredStrategy:
    """
    Structured representation of a game strategy.

    This is the output of the Strategy Planner and is used to guide
    primitive-level decision making throughout the game.
    """

    # Primary goal
    victory_goal: VictoryType = VictoryType.SCIENCE

    # Current game phase
    current_phase: str = "early_expansion"
    # Can be GamePhase value or custom string like "defending", "recovering"

    # Ordered list of priorities (most important first)
    priorities: list[str] = field(default_factory=list)
    # e.g., ["Build campus districts", "Research key techs", "Maintain defense"]

    # Areas to focus resources on
    focus_areas: list[str] = field(default_factory=list)
    # e.g., ["Science output", "City growth", "Military units"]

    # Constraints or things to avoid
    constraints: list[str] = field(default_factory=list)
    # e.g., ["Avoid war with Rome", "Don't expand too far from capital"]

    # Short-term objectives (next 10-20 turns)
    immediate_objectives: list[str] = field(default_factory=list)
    # e.g., ["Build library in capital", "Research Writing", "Settle 2nd city"]

    # Long-term objectives (next 50-100 turns)
    long_term_objectives: list[str] = field(default_factory=list)
    # e.g., ["Build 3 campuses", "Reach Modern Era", "Launch satellite"]

    # Additional notes or context
    notes: str = ""

    def to_prompt_string(self) -> str:
        """
        Convert strategy to a string suitable for LLM prompts.

        Returns a concise Korean summary of the strategy.
        """
        lines = []

        # Victory goal and phase
        victory_korean = {
            VictoryType.SCIENCE: "과학",
            VictoryType.CULTURE: "문화",
            VictoryType.DOMINATION: "지배",
            VictoryType.RELIGIOUS: "종교",
            VictoryType.DIPLOMATIC: "외교",
            VictoryType.SCORE: "점수",
        }
        lines.append(f"[{victory_korean.get(self.victory_goal, '과학')} 승리 전략]")
        lines.append(f"현재 단계: {self.current_phase}")

        # Priorities
        if self.priorities:
            lines.append(f"우선순위: {' > '.join(self.priorities[:4])}")

        # Focus areas
        if self.focus_areas:
            lines.append(f"집중 분야: {', '.join(self.focus_areas[:3])}")

        # Constraints
        if self.constraints:
            lines.append(f"제약사항: {', '.join(self.constraints[:3])}")

        # Immediate objectives
        if self.immediate_objectives:
            lines.append("즉각적 목표:")
            for obj in self.immediate_objectives[:4]:
                lines.append(f"  - {obj}")

        return "\n".join(lines)

    def to_short_string(self) -> str:
        """Get a single-line summary of the strategy."""
        victory_korean = {
            VictoryType.SCIENCE: "과학",
            VictoryType.CULTURE: "문화",
            VictoryType.DOMINATION: "지배",
            VictoryType.RELIGIOUS: "종교",
            VictoryType.DIPLOMATIC: "외교",
            VictoryType.SCORE: "점수",
        }
        goal = victory_korean.get(self.victory_goal, "과학")
        priority = self.priorities[0] if self.priorities else "일반"
        return f"{goal} 승리, {self.current_phase}, 우선: {priority}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "victory_goal": self.victory_goal.value,
            "current_phase": self.current_phase,
            "priorities": self.priorities,
            "focus_areas": self.focus_areas,
            "constraints": self.constraints,
            "immediate_objectives": self.immediate_objectives,
            "long_term_objectives": self.long_term_objectives,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredStrategy":
        """Create StructuredStrategy from dictionary."""
        victory_goal = VictoryType(data.get("victory_goal", "science"))
        return cls(
            victory_goal=victory_goal,
            current_phase=data.get("current_phase", "early_expansion"),
            priorities=data.get("priorities", []),
            focus_areas=data.get("focus_areas", []),
            constraints=data.get("constraints", []),
            immediate_objectives=data.get("immediate_objectives", []),
            long_term_objectives=data.get("long_term_objectives", []),
            notes=data.get("notes", ""),
        )

    @classmethod
    def default_science_strategy(cls) -> "StructuredStrategy":
        """Create a default science victory strategy."""
        return cls(
            victory_goal=VictoryType.SCIENCE,
            current_phase="early_expansion",
            priorities=["캠퍼스 건설", "핵심 기술 연구", "도시 성장", "방어력 유지"],
            focus_areas=["과학 출력", "인구 성장", "위인 포인트"],
            constraints=["불필요한 전쟁 회피", "군사비 과다 지출 지양"],
            immediate_objectives=["도서관 건설", "필기 연구", "제2도시 정착"],
            long_term_objectives=["캠퍼스 3개 이상 건설", "현대 시대 도달", "우주선 발사"],
        )

    @classmethod
    def default_culture_strategy(cls) -> "StructuredStrategy":
        """Create a default culture victory strategy."""
        return cls(
            victory_goal=VictoryType.CULTURE,
            current_phase="early_expansion",
            priorities=["극장가 건설", "위인 확보", "유산 건설", "관광 강화"],
            focus_areas=["문화 출력", "관광 산출", "위인 포인트"],
            constraints=["유산 경쟁 패배 시 대안 마련", "외교적 고립 방지"],
            immediate_objectives=["원형극장 건설", "걸작 확보", "축제 개최"],
            long_term_objectives=["5개 극장가 건설", "록밴드 활용", "해외 관광객 유치"],
        )

    @classmethod
    def default_domination_strategy(cls) -> "StructuredStrategy":
        """Create a default domination victory strategy."""
        return cls(
            victory_goal=VictoryType.DOMINATION,
            current_phase="early_expansion",
            priorities=["군사 유닛 생산", "병영 건설", "전략 자원 확보", "적 수도 점령"],
            focus_areas=["군사력", "생산력", "전략 자원"],
            constraints=["다수 문명과 동시 전쟁 회피", "보급선 유지"],
            immediate_objectives=["전사 2기 생산", "궁수 업그레이드", "정찰 수행"],
            long_term_objectives=["모든 수도 점령", "군사 기술 우위 유지", "동맹 활용"],
        )


class HITLInputRequiredError(Exception):
    """Raised when HITL mode is enabled but no human input was provided."""

    pass
