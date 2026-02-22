"""
Strategy Planner - Main strategy generation implementation.

Provides HITL (Human-in-the-Loop) and autonomous strategy generation
using VLM to process game context and human input.
"""

import json
import logging
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.hitl import HITLInputManager
from computer_use_test.agent.modules.strategy.base_strategy import BaseStrategyPlanner
from computer_use_test.agent.modules.strategy.prompts.strategy_prompts import (
    AUTONOMOUS_STRATEGY_PROMPT,
    STRATEGY_REFINEMENT_PROMPT,
    STRATEGY_UPDATE_PROMPT,
)
from computer_use_test.agent.modules.strategy.strategy_schemas import (
    HITLInputRequiredError,
    StructuredStrategy,
    VictoryType,
    parse_strategy_json,
)
from computer_use_test.utils.llm_provider.parser import strip_markdown

if TYPE_CHECKING:
    from computer_use_test.agent.modules.context import ContextManager
    from computer_use_test.utils.llm_provider.base import BaseVLMProvider

logger = logging.getLogger(__name__)


class StrategyPlanner(BaseStrategyPlanner):
    """
    Strategy planner with HITL and autonomous modes.

    In HITL mode, takes human input and uses VLM to refine it into
    a structured strategy. In autonomous mode, generates strategy
    purely from game context.

    Input is received exclusively via the configured chat-app provider
    (Discord, WhatsApp, etc.).  Local voice/text/auto modes have been removed.
    """

    def __init__(
        self,
        vlm_provider: "BaseVLMProvider",
        hitl_mode: bool = True,
        chatapp_provider=None,
        discussion_engine=None,
    ):
        """
        Initialize the strategy planner.

        Args:
            vlm_provider: VLM provider instance for LLM calls
            hitl_mode: If True, requires human input via chatapp; if False, autonomous
            chatapp_provider: Optional ChatAppInputProvider for chat app input
            discussion_engine: Optional StrategyDiscussion engine for multi-turn discussions
        """
        self.provider = vlm_provider
        self.hitl_mode = hitl_mode
        self.logger = logger
        self.discussion_engine = discussion_engine

        # Initialize HITL input manager if in HITL mode
        self._input_manager: HITLInputManager | None = None
        if hitl_mode:
            self._input_manager = HITLInputManager(chatapp_provider=chatapp_provider)

    def get_human_input(self, prompt: str = "전략을 입력하세요: ") -> str:
        """
        Get human input via configured method (voice/text).

        Args:
            prompt: Prompt to display to the user

        Returns:
            Human input as string

        Raises:
            HITLInputRequiredError: If HITL mode is disabled
        """
        if not self._input_manager:
            raise HITLInputRequiredError("HITL mode is disabled, cannot get human input")
        return self._input_manager.get_input(prompt)

    def generate_strategy(
        self,
        context: "ContextManager",
        human_input: str | None = None,
    ) -> StructuredStrategy:
        """
        Generate a strategy based on context and optional human input.

        In HITL mode, if human_input is not provided, it will be requested
        via the configured input method (voice/text/auto).

        Args:
            context: The ContextManager instance with current game state
            human_input: Optional human-provided strategy guidance

        Returns:
            A StructuredStrategy object

        Raises:
            HITLInputRequiredError: If HITL mode is enabled but input cannot be obtained
        """
        if self.hitl_mode:
            if human_input is None:
                # Get input via configured method (voice/text/auto)
                human_input = self.get_human_input()
            return self.refine_strategy(human_input, context)
        else:
            return self._autonomous_generation(context)

    def refine_strategy(
        self,
        raw_input: str,
        context: "ContextManager",
    ) -> StructuredStrategy:
        """
        Refine raw human input into a structured strategy.

        Args:
            raw_input: Free-form human strategy input
            context: The ContextManager instance for context

        Returns:
            A StructuredStrategy object
        """
        context_string = context.get_combined_context()

        prompt = STRATEGY_REFINEMENT_PROMPT.format(
            context_string=context_string,
            raw_input=raw_input,
        )

        self.logger.info(f"Refining strategy from human input: {raw_input[:100]}...")

        last_error = None
        for attempt in range(2):
            try:
                response = self._call_vlm(prompt)
                strategy = parse_strategy_json(strip_markdown(response))
                self.logger.info(f"Strategy refined: {strategy.victory_goal.value} victory")
                return strategy
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == 0:
                    self.logger.warning(f"Strategy parse failed (attempt 1): {e}, retrying...")
                    prompt += "\n\n이전 응답이 올바른 JSON이 아니었습니다. 반드시 유효한 JSON만 출력하세요."

        self.logger.error(f"Strategy refinement failed after retries: {last_error}")
        return self._fallback_strategy_from_input(raw_input)

    def update_strategy(
        self,
        current_strategy: StructuredStrategy,
        context: "ContextManager",
        reason: str = "",
    ) -> StructuredStrategy:
        """
        Update an existing strategy based on new context.

        Args:
            current_strategy: The current strategy to update
            context: The ContextManager instance with new game state
            reason: Optional reason for the update

        Returns:
            Updated StructuredStrategy object
        """
        context_string = context.get_combined_context()
        current_strategy_str = current_strategy.to_prompt_string()

        prompt = STRATEGY_UPDATE_PROMPT.format(
            context_string=context_string,
            current_strategy=current_strategy_str,
            update_reason=reason or "정기 업데이트",
        )

        self.logger.info(f"Updating strategy, reason: {reason or 'periodic update'}")

        last_error = None
        for attempt in range(2):
            try:
                response = self._call_vlm(prompt)
                strategy = parse_strategy_json(strip_markdown(response))
                self.logger.info(f"Strategy updated: {strategy.victory_goal.value} victory")
                return strategy
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == 0:
                    self.logger.warning(f"Strategy update parse failed (attempt 1): {e}, retrying...")
                    prompt += "\n\n이전 응답이 올바른 JSON이 아니었습니다. 반드시 유효한 JSON만 출력하세요."

        self.logger.error(f"Strategy update failed after retries: {last_error}")
        return current_strategy

    def _autonomous_generation(self, context: "ContextManager") -> StructuredStrategy:
        """Generate strategy autonomously from context."""
        context_string = context.get_combined_context()

        prompt = AUTONOMOUS_STRATEGY_PROMPT.format(
            context_string=context_string,
        )

        self.logger.info("Generating autonomous strategy from context")

        last_error = None
        for attempt in range(2):
            try:
                response = self._call_vlm(prompt)
                strategy = parse_strategy_json(strip_markdown(response))
                self.logger.info(f"Autonomous strategy generated: {strategy.victory_goal.value} victory")
                return strategy
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt == 0:
                    self.logger.warning(f"Autonomous strategy parse failed (attempt 1): {e}, retrying...")
                    prompt += "\n\n이전 응답이 올바른 JSON이 아니었습니다. 반드시 유효한 JSON만 출력하세요."

        self.logger.error(f"Autonomous strategy generation failed after retries: {last_error}")
        return StructuredStrategy.default_science_strategy()

    def _call_vlm(self, prompt: str) -> str:
        """Call the VLM with a text-only prompt."""
        content_parts = [self.provider._build_text_content(prompt)]

        response = self.provider._send_to_api(content_parts, temperature=0.3)

        return response.content

    def _fallback_strategy_from_input(self, raw_input: str) -> StructuredStrategy:
        """Create a fallback strategy based on keywords in input."""
        input_lower = raw_input.lower()

        # Detect victory type from keywords
        if any(word in input_lower for word in ["과학", "science", "기술", "연구", "캠퍼스"]):
            return StructuredStrategy.default_science_strategy()
        elif any(word in input_lower for word in ["문화", "culture", "관광", "유산", "극장"]):
            return StructuredStrategy.default_culture_strategy()
        elif any(word in input_lower for word in ["지배", "domination", "군사", "전쟁", "정복"]):
            return StructuredStrategy.default_domination_strategy()
        else:
            return StructuredStrategy(
                text=f"과학 승리를 목표로 한다. 플레이어 지시: {raw_input[:200]}",
                victory_goal=VictoryType.SCIENCE,
                current_phase="early_expansion",
            )
