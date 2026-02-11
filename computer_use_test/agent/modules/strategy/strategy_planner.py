"""
Strategy Planner - Main strategy generation implementation.

Provides HITL (Human-in-the-Loop) and autonomous strategy generation
using VLM to process game context and human input.
"""

import json
import logging
from typing import TYPE_CHECKING

from computer_use_test.agent.modules.hitl import HITLInputManager, InputMode
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

    Supports multiple input modes for HITL:
    - text: Terminal-based text input
    - voice: Microphone + STT (Speech-to-Text)
    - auto: Try voice first, fallback to text
    """

    def __init__(
        self,
        vlm_provider: "BaseVLMProvider",
        hitl_mode: bool = True,
        input_mode: InputMode | str = InputMode.TEXT,
        stt_provider: str = "whisper",
        voice_timeout: float = 10.0,
        chatapp_provider=None,
        discussion_engine=None,
    ):
        """
        Initialize the strategy planner.

        Args:
            vlm_provider: VLM provider instance for LLM calls
            hitl_mode: If True, requires human input; if False, autonomous
            input_mode: HITL input mode ("voice", "text", "auto", or "chatapp")
            stt_provider: STT provider for voice input ("whisper", "google", "openai")
            voice_timeout: Maximum seconds to wait for voice input
            chatapp_provider: Optional ChatAppInputProvider for chat app input
            discussion_engine: Optional StrategyDiscussion engine for multi-turn discussions
        """
        self.provider = vlm_provider
        self.hitl_mode = hitl_mode
        self.input_mode = InputMode(input_mode) if isinstance(input_mode, str) else input_mode
        self.logger = logger
        self.discussion_engine = discussion_engine

        # Initialize HITL input manager if in HITL mode
        self._input_manager: HITLInputManager | None = None
        if hitl_mode:
            self._input_manager = HITLInputManager(
                input_mode=self.input_mode,
                stt_provider=stt_provider,
                voice_timeout=voice_timeout,
                chatapp_provider=chatapp_provider,
            )

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

        try:
            response = self._call_vlm(prompt)
            strategy = self._parse_strategy_response(response)
            self.logger.info(f"Strategy refined: {strategy.victory_goal.value} victory")
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to refine strategy: {e}")
            # Return a default strategy based on input hints
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

        try:
            response = self._call_vlm(prompt)
            strategy = self._parse_strategy_response(response)
            self.logger.info(f"Strategy updated: {strategy.victory_goal.value} victory")
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to update strategy: {e}")
            # Return original strategy if update fails
            return current_strategy

    def _autonomous_generation(self, context: "ContextManager") -> StructuredStrategy:
        """Generate strategy autonomously from context."""
        context_string = context.get_combined_context()

        prompt = AUTONOMOUS_STRATEGY_PROMPT.format(
            context_string=context_string,
        )

        self.logger.info("Generating autonomous strategy from context")

        try:
            response = self._call_vlm(prompt)
            strategy = self._parse_strategy_response(response)
            self.logger.info(f"Autonomous strategy generated: {strategy.victory_goal.value} victory")
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to generate autonomous strategy: {e}")
            # Return default science strategy
            return StructuredStrategy.default_science_strategy()

    def _call_vlm(self, prompt: str) -> str:
        """Call the VLM with a text-only prompt."""
        content_parts = [self.provider._build_text_content(prompt)]

        response = self.provider._send_to_api(
            content_parts,
            temperature=0.3,
            max_tokens=2048,
        )

        return response.content

    def _parse_strategy_response(self, response: str) -> StructuredStrategy:
        """Parse VLM response into StructuredStrategy."""
        # Strip markdown if present
        content = strip_markdown(response)

        try:
            data = json.loads(content)

            # Parse victory goal
            victory_goal_str = data.get("victory_goal", "science").lower()
            try:
                victory_goal = VictoryType(victory_goal_str)
            except ValueError:
                self.logger.warning(f"Unknown victory type: {victory_goal_str}, defaulting to science")
                victory_goal = VictoryType.SCIENCE

            return StructuredStrategy(
                victory_goal=victory_goal,
                current_phase=data.get("current_phase", "early_expansion"),
                priorities=data.get("priorities", []),
                focus_areas=data.get("focus_areas", []),
                constraints=data.get("constraints", []),
                immediate_objectives=data.get("immediate_objectives", []),
                long_term_objectives=data.get("long_term_objectives", []),
                notes=data.get("notes", ""),
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse strategy JSON: {e}")
            self.logger.error(f"Raw response: {response}")
            raise ValueError(f"Failed to parse strategy response: {e}") from e

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
            # Default to science
            strategy = StructuredStrategy.default_science_strategy()
            strategy.notes = f"플레이어 입력 기반: {raw_input[:100]}"
            return strategy
