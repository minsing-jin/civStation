"""
Macro-Turn Manager — Tracks game-level turns vs. primitive-level micro-turns.

A "micro-turn" is one primitive action (screenshot → route → plan → execute).
A "macro-turn" is one full in-game turn (ends when the agent clicks "Next Turn").

This module detects macro-turn boundaries and generates LLM summaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context.context_manager import ContextManager
    from computer_use_test.utils.llm_provider.base import BaseVLMProvider
    from computer_use_test.utils.llm_provider.parser import AgentAction

logger = logging.getLogger(__name__)

# Keywords in reasoning that indicate a "next turn" action
_NEXT_TURN_KEYWORDS = {"다음 턴", "next turn", "턴 종료", "end turn", "턴 넘기기"}


@dataclass
class MacroTurnSummary:
    """Summary of a completed macro-turn (one full game turn)."""

    macro_turn_number: int
    micro_turn_count: int
    primitives_used: list[str]
    llm_summary: str
    key_decisions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MacroTurnManager:
    """
    Manages the distinction between micro-turns and macro-turns.

    Records each micro-turn, detects when the agent ends a game turn
    (clicks "Next Turn"), and produces LLM summaries of completed macro-turns.
    """

    def __init__(self, context_manager: ContextManager, vlm_provider: BaseVLMProvider | None = None) -> None:
        self._ctx = context_manager
        self._vlm = vlm_provider

        self._macro_turn_number: int = 1
        self._micro_turn_count: int = 0
        self._primitives_used: list[str] = []
        self._micro_summaries: list[str] = []

    @property
    def macro_turn_number(self) -> int:
        return self._macro_turn_number

    @property
    def micro_turn_count(self) -> int:
        return self._micro_turn_count

    def record_micro_turn(self, primitive_name: str, summary: str) -> None:
        """Record one primitive-level micro-turn execution."""
        self._micro_turn_count += 1
        self._primitives_used.append(primitive_name)
        if summary:
            self._micro_summaries.append(f"[{primitive_name}] {summary}")

    def is_next_turn_action(self, primitive_name: str, action: AgentAction) -> bool:
        """
        Detect if an action represents clicking "Next Turn" in the game.

        Heuristic: popup_primitive + click + reasoning contains "next turn" keywords.
        """
        if action.action != "click":
            return False

        if primitive_name != "popup_primitive":
            return False

        reasoning = (action.reasoning or "").lower()
        return any(kw in reasoning for kw in _NEXT_TURN_KEYWORDS)

    def handle_macro_turn_end(self) -> MacroTurnSummary:
        """
        Process the end of a macro-turn.

        1. Generate LLM summary of this macro-turn
        2. Store summary in context manager
        3. Advance game turn (flush primitive context)
        4. Reset micro-turn tracking
        """
        # Generate summary
        llm_summary = self._generate_summary()

        summary = MacroTurnSummary(
            macro_turn_number=self._macro_turn_number,
            micro_turn_count=self._micro_turn_count,
            primitives_used=list(set(self._primitives_used)),
            llm_summary=llm_summary,
        )

        # Store in context manager
        self._ctx.add_macro_turn_summary(llm_summary)

        # Advance game turn with action flush
        self._ctx.advance_turn(
            primitive_used=", ".join(set(self._primitives_used)),
            success=True,
            notes=llm_summary[:200],
            flush_actions=True,
        )

        # Reset for next macro-turn
        self._macro_turn_number += 1
        self._micro_turn_count = 0
        self._primitives_used = []
        self._micro_summaries = []

        logger.info(f"Macro-turn {summary.macro_turn_number} completed ({summary.micro_turn_count} micro-turns)")
        return summary

    def _generate_summary(self) -> str:
        """Generate an LLM summary of the current macro-turn."""
        if not self._vlm or not self._micro_summaries:
            return self._fallback_summary()

        prompt = (
            "다음은 문명6 게임에서 한 턴 동안 수행한 행동들입니다.\n"
            "이 턴의 핵심 결정과 결과를 2-3문장으로 요약하세요.\n\n"
            f"턴 번호: {self._macro_turn_number}\n"
            f"수행한 행동 ({self._micro_turn_count}개):\n"
        )
        for s in self._micro_summaries[-10:]:  # Limit to last 10
            prompt += f"  - {s}\n"

        try:
            content_parts = [self._vlm._build_text_content(prompt)]
            response = self._vlm._send_to_api(content_parts, temperature=0.3, max_tokens=256)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return self._fallback_summary()

    def _fallback_summary(self) -> str:
        """Generate a simple text summary without LLM."""
        primitives = ", ".join(set(self._primitives_used)) if self._primitives_used else "없음"
        return f"턴 {self._macro_turn_number}: {self._micro_turn_count}개 행동 수행 (프리미티브: {primitives})"
