"""
Primitive Context - Stores short-term execution state.

This context maintains information about the current primitive execution,
recent actions, and local state needed for decision making.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ActionRecord:
    """Record of a single action execution."""

    action_type: str  # "click", "press", "drag", etc.
    primitive: str  # Primitive that generated the action
    x: int = 0
    y: int = 0
    end_x: int = 0
    end_y: int = 0
    key: str = ""
    text: str = ""
    result: str = "success"  # "success", "failed", "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "action_type": self.action_type,
            "primitive": self.primitive,
            "x": self.x,
            "y": self.y,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PrimitiveContext:
    """
    Primitive-level execution context.

    Stores short-term state relevant to the current execution,
    including recent actions and selected unit information.
    """

    # Current execution state
    current_primitive: str = ""
    execution_count: int = 0  # How many times this primitive has run this turn

    # Recent actions (limited history)
    recent_actions: list[ActionRecord] = field(default_factory=list)
    max_action_history: int = 10

    # Selected unit/entity information
    selected_unit_info: dict[str, Any] = field(default_factory=dict)
    # e.g., {"name": "전사", "moves": 2, "health": 100, "position": (10, 15)}

    # Error tracking
    consecutive_failures: int = 0
    last_error: str = ""

    # Primitive-specific state
    primitive_state: dict[str, Any] = field(default_factory=dict)
    # Flexible storage for primitive-specific data

    def set_current_primitive(self, primitive_name: str) -> None:
        """Set the current primitive and reset execution count."""
        if self.current_primitive != primitive_name:
            self.current_primitive = primitive_name
            self.execution_count = 0
            self.primitive_state = {}
        self.execution_count += 1

    def add_action(
        self,
        action_type: str,
        primitive: str,
        x: int = 0,
        y: int = 0,
        end_x: int = 0,
        end_y: int = 0,
        key: str = "",
        text: str = "",
        result: str = "success",
        error_message: str = "",
    ) -> None:
        """Add an action to the history."""
        record = ActionRecord(
            action_type=action_type,
            primitive=primitive,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            key=key,
            text=text,
            result=result,
            error_message=error_message,
        )
        self.recent_actions.append(record)

        # Trim history if needed
        if len(self.recent_actions) > self.max_action_history:
            self.recent_actions = self.recent_actions[-self.max_action_history :]

        # Track failures
        if result == "failed":
            self.consecutive_failures += 1
            self.last_error = error_message
        else:
            self.consecutive_failures = 0
            self.last_error = ""

    def update_selected_unit(self, unit_info: dict[str, Any]) -> None:
        """Update the selected unit information."""
        self.selected_unit_info = unit_info

    def clear_selected_unit(self) -> None:
        """Clear the selected unit information."""
        self.selected_unit_info = {}

    def get_last_actions(self, count: int = 5) -> list[ActionRecord]:
        """Get the last N actions."""
        return self.recent_actions[-count:]

    def get_actions_for_primitive(self, primitive_name: str) -> list[ActionRecord]:
        """Get all recent actions for a specific primitive."""
        return [a for a in self.recent_actions if a.primitive == primitive_name]

    def is_repeating_action(self, action_type: str, x: int, y: int, threshold: int = 3) -> bool:
        """Check if we're repeating the same action too many times."""
        recent = self.get_last_actions(threshold)
        same_action_count = sum(1 for a in recent if a.action_type == action_type and a.x == x and a.y == y)
        return same_action_count >= threshold

    def to_prompt_string(self) -> str:
        """
        Convert primitive context to a string suitable for LLM prompts.

        Returns a concise summary of execution state.
        """
        lines = ["=== 실행 컨텍스트 ==="]

        # Current primitive
        if self.current_primitive:
            lines.append(f"현재 프리미티브: {self.current_primitive} (실행 #{self.execution_count})")

        # Selected unit
        if self.selected_unit_info:
            unit_name = self.selected_unit_info.get("name", "알 수 없음")
            moves = self.selected_unit_info.get("moves", "?")
            health = self.selected_unit_info.get("health", "?")
            lines.append(f"선택된 유닛: {unit_name} (이동력: {moves}, 체력: {health})")

        # Recent actions
        if self.recent_actions:
            lines.append("")
            lines.append("최근 액션:")
            for action in self.get_last_actions(5):
                if action.action_type in ("click", "double_click"):
                    lines.append(f"  - {action.action_type} ({action.x}, {action.y}) [{action.result}]")
                elif action.action_type == "press":
                    lines.append(f"  - press '{action.key}' [{action.result}]")
                elif action.action_type == "drag":
                    lines.append(
                        f"  - drag ({action.x}, {action.y}) -> ({action.end_x}, {action.end_y}) [{action.result}]"
                    )
                else:
                    lines.append(f"  - {action.action_type} [{action.result}]")

        # Error tracking
        if self.consecutive_failures > 0:
            lines.append("")
            lines.append(f"⚠️ 연속 실패: {self.consecutive_failures}회")
            if self.last_error:
                lines.append(f"   마지막 오류: {self.last_error}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "current_primitive": self.current_primitive,
            "execution_count": self.execution_count,
            "recent_actions": [a.to_dict() for a in self.recent_actions],
            "selected_unit_info": self.selected_unit_info,
            "consecutive_failures": self.consecutive_failures,
        }

    def reset(self) -> None:
        """Reset the primitive context for a new turn."""
        self.current_primitive = ""
        self.execution_count = 0
        self.selected_unit_info = {}
        self.consecutive_failures = 0
        self.last_error = ""
        self.primitive_state = {}
        # Keep recent_actions for context across turns
