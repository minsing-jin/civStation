"""
Global Context - Stores overall game state information.

This context maintains high-level game statistics that persist across turns,
including map status, resources, diplomatic relations, and war status.
"""

# TODO: MCP 서버 연동 시, GlobalContext 필드를 게임 네이티브 데이터로 직접 채울 수 있음.
#       현재는 ContextUpdater(VLM 비전 분석)로 current_turn, game_era만 업데이트하고
#       나머지 필드(cities, units, resources 등)는 MCP로 정확한 값을 가져올 것.
# TODO: Context Length 관리 — cities, known_civilizations, units_by_type 등
#       리스트/딕셔너리가 게임 후반에 커질 수 있으므로
#       to_prompt_string() 출력 시 최대 길이 제한 또는 요약 로직 추가할 것.

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DiplomaticStatus(str, Enum):
    """Diplomatic relationship status with other civilizations."""

    UNKNOWN = "unknown"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    UNFRIENDLY = "unfriendly"
    DENOUNCED = "denounced"
    WAR = "war"
    ALLIED = "allied"


@dataclass
class CivilizationInfo:
    """Information about a known civilization."""

    name: str
    leader: str = ""
    diplomatic_status: DiplomaticStatus = DiplomaticStatus.UNKNOWN
    known_cities: list[str] = field(default_factory=list)
    military_strength: str = "unknown"  # "weak", "equal", "strong", "unknown"
    last_interaction_turn: int = 0
    notes: str = ""


@dataclass
class CityInfo:
    """Information about one of our cities."""

    name: str
    population: int = 1
    production_focus: str = ""
    current_production: str = ""
    production_turns_left: int = 0
    food_per_turn: float = 0.0
    production_per_turn: float = 0.0
    science_per_turn: float = 0.0
    culture_per_turn: float = 0.0
    gold_per_turn: float = 0.0
    faith_per_turn: float = 0.0
    has_governor: bool = False
    governor_name: str = ""
    amenities: int = 0
    housing: int = 0


@dataclass
class ResourceInfo:
    """Information about resources."""

    strategic: dict[str, int] = field(default_factory=dict)  # e.g., {"iron": 2, "horses": 4}
    luxury: dict[str, int] = field(default_factory=dict)  # e.g., {"silk": 1, "wine": 2}
    bonus: dict[str, int] = field(default_factory=dict)  # e.g., {"wheat": 3}


@dataclass
class GlobalContext:
    """
    Global game state context.

    Stores information that persists across turns and is relevant
    to high-level strategic decisions.
    """

    # Game progress
    current_turn: int = 1
    game_era: str = "Ancient"  # Ancient, Classical, Medieval, Renaissance, Industrial, Modern, Atomic, Information
    game_speed: str = "Standard"

    # Our civilization status
    civilization_name: str = ""
    leader_name: str = ""
    gold: int = 0
    gold_per_turn: float = 0.0
    science_per_turn: float = 0.0
    culture_per_turn: float = 0.0
    faith: int = 0
    faith_per_turn: float = 0.0

    # Cities
    cities: list[CityInfo] = field(default_factory=list)
    total_population: int = 0

    # Military
    military_strength: int = 0
    unit_count: int = 0
    units_by_type: dict[str, int] = field(default_factory=dict)  # e.g., {"warrior": 2, "settler": 1}

    # Diplomacy
    known_civilizations: list[CivilizationInfo] = field(default_factory=list)
    active_wars: list[str] = field(default_factory=list)  # List of civ names we're at war with
    active_alliances: list[str] = field(default_factory=list)

    # Resources
    resources: ResourceInfo = field(default_factory=ResourceInfo)

    # Research & Culture
    current_research: str = ""
    research_turns_left: int = 0
    current_civic: str = ""
    civic_turns_left: int = 0
    available_techs: list[str] = field(default_factory=list)
    available_civics: list[str] = field(default_factory=list)

    # Victory progress
    victory_type_focus: str = ""  # science, culture, domination, religious, diplomatic
    victory_progress: dict[str, Any] = field(default_factory=dict)

    # Map exploration
    explored_percentage: float = 0.0
    known_natural_wonders: list[str] = field(default_factory=list)
    known_city_states: list[str] = field(default_factory=list)

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)

    def update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()

    def get_city_by_name(self, name: str) -> CityInfo | None:
        """Get city info by name."""
        for city in self.cities:
            if city.name.lower() == name.lower():
                return city
        return None

    def get_civ_by_name(self, name: str) -> CivilizationInfo | None:
        """Get civilization info by name."""
        for civ in self.known_civilizations:
            if civ.name.lower() == name.lower():
                return civ
        return None

    def is_at_war(self) -> bool:
        """Check if we are currently at war."""
        return len(self.active_wars) > 0

    def to_prompt_string(self) -> str:
        """
        Convert global context to a string suitable for LLM prompts.

        Returns a concise summary of the most important game state information.
        """
        lines = [
            f"=== 게임 상태 (턴 {self.current_turn}, {self.game_era} 시대) ===",
            f"문명: {self.civilization_name} ({self.leader_name})",
            "",
            "📊 자원:",
            f"  - 골드: {self.gold} (+{self.gold_per_turn:.1f}/턴)",
            f"  - 과학: +{self.science_per_turn:.1f}/턴",
            f"  - 문화: +{self.culture_per_turn:.1f}/턴",
            f"  - 신앙: {self.faith} (+{self.faith_per_turn:.1f}/턴)",
        ]

        if self.cities:
            lines.append("")
            lines.append(f"🏛️ 도시 ({len(self.cities)}개, 총 인구 {self.total_population}):")
            for city in self.cities[:5]:  # Limit to 5 cities for brevity
                production_info = f", 생산: {city.current_production} ({city.production_turns_left}턴)" if city.current_production else ""
                lines.append(f"  - {city.name} (인구 {city.population}{production_info})")

        if self.current_research:
            lines.append("")
            lines.append(f"🔬 연구 중: {self.current_research} ({self.research_turns_left}턴 남음)")

        if self.current_civic:
            lines.append(f"📜 사회제도: {self.current_civic} ({self.civic_turns_left}턴 남음)")

        if self.active_wars:
            lines.append("")
            lines.append(f"⚔️ 전쟁 중: {', '.join(self.active_wars)}")

        if self.known_civilizations:
            lines.append("")
            lines.append("🌍 알려진 문명:")
            for civ in self.known_civilizations[:5]:
                lines.append(f"  - {civ.name}: {civ.diplomatic_status.value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_turn": self.current_turn,
            "game_era": self.game_era,
            "civilization_name": self.civilization_name,
            "gold": self.gold,
            "gold_per_turn": self.gold_per_turn,
            "science_per_turn": self.science_per_turn,
            "culture_per_turn": self.culture_per_turn,
            "city_count": len(self.cities),
            "total_population": self.total_population,
            "military_strength": self.military_strength,
            "is_at_war": self.is_at_war(),
            "active_wars": self.active_wars,
            "current_research": self.current_research,
            "victory_focus": self.victory_type_focus,
        }
