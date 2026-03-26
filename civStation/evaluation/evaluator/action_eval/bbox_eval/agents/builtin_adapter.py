"""
Built-in agent adapter wrapping existing VLM providers for bbox evaluation.

Uses BaseVLMProvider.call_vlm() + parse_to_agent_plan() to generate actions,
then converts AgentPlan -> AgentResponse.

Example:
    >>> from civStation.utils.llm_provider import create_provider
    >>> provider = create_provider("claude")
    >>> runner = BuiltinAgentRunner(provider=provider)
    >>> response = runner.run_case(case)
"""

from __future__ import annotations

import logging

from civStation.utils.llm_provider.base import BaseVLMProvider
from civStation.utils.llm_provider.parser import parse_to_agent_plan

from ..schema import AgentResponse, DatasetCase
from .base import AgentRunnerError, BaseAgentRunner

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """You are evaluating a game screenshot. Analyze the image and follow this instruction:

{instruction}

The image size is {width}x{height}. Use normalized coordinates in the range 0-{max_coord}.

Respond with JSON only (no markdown fences):
{{
  "reasoning": "your reasoning here",
  "actions": [
    {{"type": "click", "x": <int>, "y": <int>, "button": "left"}},
    {{"type": "press", "keys": ["enter"]}},
    {{"type": "drag", "start_x": <int>, "start_y": <int>, "end_x": <int>, "end_y": <int>}}
  ]
}}
"""


class BuiltinAgentRunner(BaseAgentRunner):
    """
    Wraps an existing BaseVLMProvider to run as an agent for evaluation.

    Builds a prompt from the case instruction, calls the provider,
    and parses the response into an AgentResponse.
    """

    def __init__(self, provider: BaseVLMProvider, primitive_name: str = "eval"):
        self.provider = provider
        self.primitive_name = primitive_name

    def run_case(self, case: DatasetCase) -> AgentResponse:
        max_coord = max(case.image_size.width, case.image_size.height)
        prompt = _PROMPT_TEMPLATE.format(
            instruction=case.instruction,
            width=case.image_size.width,
            height=case.image_size.height,
            max_coord=max_coord,
        )

        try:
            response = self.provider.call_vlm(
                prompt=prompt,
                image_path=case.screenshot_path,
                temperature=0.3,
            )
        except Exception as e:
            raise AgentRunnerError(f"VLM call failed for case {case.case_id}: {e}") from e

        try:
            plan = parse_to_agent_plan(response.content, self.primitive_name)
        except ValueError as e:
            raise AgentRunnerError(f"Failed to parse VLM response for case {case.case_id}: {e}") from e

        return AgentResponse(
            actions=list(plan.actions),
            meta={
                "provider": self.provider.get_provider_name(),
                "primitive_name": plan.primitive_name,
                "reasoning": plan.reasoning,
            },
        )
