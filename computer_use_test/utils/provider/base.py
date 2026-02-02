"""
Base interface for Vision-Language Model (VLM) providers.

All providers implement ONE core method `_send_to_api()` for API calls.
Everything else (call_vlm, analyze, parse) builds on top to avoid duplication.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from computer_use_test.agent.models.schema import AgentPlan
from computer_use_test.utils.prompts.civ6_action_prompt import get_system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """Response from a VLM API call."""

    content: str
    raw_response: Optional[object] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    finish_reason: Optional[str] = None  # "stop", "max_tokens", "length" etc.


@dataclass
class AgentAction:
    """
    Single action from VLM using normalized coordinates (0-1000).

    Supports: click, double_click, drag, press, type
    """

    action: str = ""          # "click", "double_click", "drag", "press", "type"
    x: int = 0                # Normalized x (0-1000)
    y: int = 0                # Normalized y (0-1000)
    end_x: int = 0            # Drag end x (0-1000)
    end_y: int = 0            # Drag end y (0-1000)
    button: str = "left"      # "left" or "right"
    key: str = ""             # Key name for "press"
    text: str = ""            # Text for "type"
    reasoning: str = ""


class BaseVLMProvider(ABC):
    """
    Abstract base class for VLM providers.

    Each provider only needs to implement:
    - _send_to_api(): Core API call with content parts
    - _encode_image_file(): Encode file path to provider-specific format
    - _encode_pil_image(): Encode PIL image to provider-specific format
    - get_provider_name(): Provider identifier
    """

    DEFAULT_MODEL: str = ""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.logger = logger

    # ==================== Core (implement in subclass) ====================

    @abstractmethod
    def _send_to_api(
        self,
        content_parts: list,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> VLMResponse:
        """
        Send content parts to the VLM API and return raw response.

        This is the ONLY method that touches the API. All other methods
        build content_parts and delegate to this.

        Args:
            content_parts: Provider-specific list of content parts (text, images)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            VLMResponse with the raw text response
        """
        pass

    @abstractmethod
    def _build_image_content(self, image_path: Union[str, Path]) -> object:
        """Build provider-specific image content from file path."""
        pass

    @abstractmethod
    def _build_pil_image_content(self, pil_image) -> object:
        """Build provider-specific image content from PIL image."""
        pass

    @abstractmethod
    def _build_text_content(self, text: str) -> object:
        """Build provider-specific text content."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    # ==================== Static Evaluation (file-based) ====================

    def call_vlm(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> VLMResponse:
        """Call VLM with text prompt + optional image file."""
        content_parts = []
        if image_path:
            content_parts.append(self._build_image_content(image_path))
        content_parts.append(self._build_text_content(prompt))
        return self._send_to_api(content_parts, temperature, max_tokens)

    def parse_to_agent_plan(self, response: VLMResponse, primitive_name: str) -> AgentPlan:
        """Parse VLM response into AgentPlan (for static evaluation)."""
        from computer_use_test.agent.models.schema import ClickAction, DragAction, KeyPressAction

        content = self._strip_markdown(response.content)

        try:
            data = json.loads(content)
            actions = []

            for ad in data.get("actions", []):
                action_type = ad.get("type")
                if action_type == "click":
                    actions.append(ClickAction(
                        type="click", x=ad["x"], y=ad["y"],
                        button=ad.get("button", "left"), description=ad.get("description"),
                    ))
                elif action_type == "press":
                    actions.append(KeyPressAction(
                        type="press", keys=ad["keys"],
                        interval=ad.get("interval", 0.1), description=ad.get("description"),
                    ))
                elif action_type == "drag":
                    actions.append(DragAction(
                        type="drag",
                        start_x=ad["start_x"], start_y=ad["start_y"],
                        end_x=ad["end_x"], end_y=ad["end_y"],
                        duration=ad.get("duration", 0.5), button=ad.get("button", "left"),
                        description=ad.get("description"),
                    ))

            return AgentPlan(
                primitive_name=primitive_name,
                reasoning=data.get("reasoning", ""),
                actions=actions,
            )
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to parse VLM response: {e}")
            self.logger.error(f"Raw response content:\n{response.content}")
            self.logger.error(f"After markdown stripping:\n{content}")
            raise ValueError(f"Failed to parse VLM response: {e}") from e

    def call_and_parse(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]],
        primitive_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AgentPlan:
        """Convenience: call VLM and parse response in one step."""
        response = self.call_vlm(prompt, image_path, temperature, max_tokens)
        return self.parse_to_agent_plan(response, primitive_name)

    # ==================== Live Agent (PIL image, normalized coords) ====================

    def analyze(
        self,
        pil_image,
        instruction: str,
        normalizing_range: int = 1000,
    ) -> Optional[AgentAction]:
        """
        Analyze PIL image and return next action with normalized coordinates.

        Uses the same _send_to_api() core as call_vlm() to avoid duplication.

        Args:
            pil_image: PIL Image (screenshot)
            instruction: User goal
            normalizing_range: Coordinate range (default: 1000)

        Returns:
            AgentAction with normalized coordinates, or None
        """
        prompt = get_system_prompt(instruction, normalizing_range)

        content_parts = [
            self._build_pil_image_content(pil_image),
            self._build_text_content(prompt),
        ]

        try:
            # TODO: For long-horizon tasks, reduce max_tokens and remove "reasoning"
            #       field from action JSON format to save tokens.
            response = self._send_to_api(content_parts, temperature=0.3, max_tokens=8192)

            if response.finish_reason in ("max_tokens", "length", "MAX_TOKENS"):
                self.logger.warning(
                    f"Action response TRUNCATED (finish_reason={response.finish_reason}). "
                    f"JSON will likely be incomplete."
                )

            return self._parse_action_json(response.content)
        except Exception as e:
            self.logger.error(f"analyze() failed: {e}")
            return None

    # ==================== Shared Utilities ====================

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Strip markdown code block wrappers from response text.

        Handles various markdown formats:
        - ```json...```
        - ```...```
        - Multiple closing ```
        - Extra whitespace
        """
        import re

        content = text.strip()

        # Remove opening fence
        if content.startswith("```json"):
            content = content[7:].lstrip()
        elif content.startswith("```"):
            content = content[3:].lstrip()

        # Remove all closing fences (handle multiple ``` with possible newlines)
        # Use regex to remove trailing ``` blocks
        content = re.sub(r'(\n```)+\s*$', '', content)
        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    def _parse_action_json(self, response_text: str) -> Optional[AgentAction]:
        """Parse VLM response into AgentAction (for live agent)."""
        try:
            content = self._strip_markdown(response_text)
            data = json.loads(content)

            # Handle list response
            if isinstance(data, list):
                if not data:
                    return None
                self.logger.info(f"List response: using first of {len(data)} items")
                data = data[0]

            return AgentAction(
                action=data.get("action", ""),
                x=int(data.get("x", 0)),
                y=int(data.get("y", 0)),
                end_x=int(data.get("end_x", 0)),
                end_y=int(data.get("end_y", 0)),
                button=data.get("button", "left"),
                key=data.get("key", ""),
                text=data.get("text", ""),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Failed to parse action JSON: {e}")
            self.logger.error(f"Raw response text:\n{response_text}")
            return None


class MockVLMProvider(BaseVLMProvider):
    """Mock VLM provider for testing without API calls."""

    DEFAULT_MODEL = "mock-vlm"

    def __init__(self, api_key=None, model=None):
        super().__init__(api_key="mock", model=model or self.DEFAULT_MODEL)

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=4096):
        return VLMResponse(
            content=json.dumps({
                "reasoning": "Mock analysis",
                "actions": [{"type": "click", "x": 500, "y": 300}, {"type": "press", "keys": ["esc"]}],
            }),
            tokens_used=100, cost=0.0,
            finish_reason="stop",
        )

    def _build_image_content(self, image_path):
        return {"type": "image", "path": str(image_path)}

    def _build_pil_image_content(self, pil_image):
        return {"type": "image", "data": "mock"}

    def _build_text_content(self, text):
        return {"type": "text", "text": text}

    def get_provider_name(self):
        return "mock"
