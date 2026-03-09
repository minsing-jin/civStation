"""
Short-Term Memory — Tracks observations across multi-step primitive execution.

Attached to multi-step primitives to:
- Remember previous step observations (reasoning + action summary)
- Prevent repeating the same action
- Provide context window of recent steps to the VLM prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Number of recent observations to include in prompt (token budget)
_PROMPT_WINDOW = 5


@dataclass
class Observation:
    """Single step observation."""

    step: int
    reasoning: str
    action_summary: str
    result: str = ""


@dataclass
class ShortTermMemory:
    """Per-task short-term memory for multi-step primitives."""

    primitive_name: str = ""
    observations: list[Observation] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 10

    def start_task(self, primitive_name: str, max_steps: int = 10) -> None:
        """Initialize memory for a new multi-step task."""
        self.primitive_name = primitive_name
        self.max_steps = max_steps
        self.observations = []
        self.step_count = 0
        logger.debug(f"ShortTermMemory started: {primitive_name} (max_steps={max_steps})")

    def add_observation(self, reasoning: str, action_summary: str, result: str = "") -> None:
        """Record an observation from a completed step."""
        self.step_count += 1
        self.observations.append(
            Observation(
                step=self.step_count,
                reasoning=reasoning,
                action_summary=action_summary,
                result=result,
            )
        )
        logger.debug(f"STM step {self.step_count}: {action_summary}")

    def to_prompt_string(self) -> str:
        """Format recent observations for VLM prompt injection.

        Returns the last ``_PROMPT_WINDOW`` observations to keep token usage bounded.
        """
        if not self.observations:
            return "없음"

        recent = self.observations[-_PROMPT_WINDOW:]
        lines = []
        for obs in recent:
            line = f"[Step {obs.step}] {obs.action_summary} — {obs.reasoning}"
            if obs.result:
                line += f" → 결과: {obs.result}"
            lines.append(line)
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all observations and reset state."""
        self.primitive_name = ""
        self.observations = []
        self.step_count = 0
        self.max_steps = 10

    def is_max_steps_reached(self) -> bool:
        """Check if the maximum number of steps has been reached."""
        return self.step_count >= self.max_steps
