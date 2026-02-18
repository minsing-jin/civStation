"""
Base interface for Vision-Language Model (VLM) providers.

All providers implement ONE core method `_send_to_api()` for API calls.
Everything else (call_vlm, analyze, parse) builds on top to avoid duplication.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from computer_use_test.agent.models.schema import AgentPlan
from computer_use_test.utils.llm_provider.parser import (
    AgentAction,
    parse_action_json,
    parse_to_agent_plan,
    validate_action,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """Response from a VLM API call."""

    content: str
    raw_response: object | None = None
    tokens_used: int | None = None
    cost: float | None = None
    finish_reason: str | None = None  # "stop", "max_tokens", "length" etc.


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

    def __init__(self, api_key: str | None = None, model: str | None = None):
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
    def _build_image_content(self, image_path: str | Path) -> object:
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
        image_path: str | Path | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> VLMResponse:
        """Call VLM with text prompt + optional image file."""
        content_parts = []
        if image_path:
            content_parts.append(self._build_image_content(image_path))
        content_parts.append(self._build_text_content(prompt))
        return self._send_to_api(content_parts, temperature, max_tokens)

    def call_and_parse(
        self,
        prompt: str,
        image_path: str | Path | None,
        primitive_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AgentPlan:
        """Convenience: call VLM and parse response in one step."""
        response = self.call_vlm(prompt, image_path, temperature, max_tokens)
        return parse_to_agent_plan(response.content, primitive_name)

    # ==================== Live Agent (PIL image, normalized coords) ====================

    MAX_RETRIES: int = 3

    def analyze(
        self,
        pil_image,
        instruction: str,
        normalizing_range: int = 1000,
    ) -> AgentAction | None:
        """
        Analyze PIL image and return next action with normalized coordinates.

        Uses the same _send_to_api() core as call_vlm() to avoid duplication.
        The instruction should already contain JSON format instructions
        (via JSON_FORMAT_INSTRUCTION from primitive_prompt.py).

        Retries up to MAX_RETRIES times on parse/validation failure.

        Args:
            pil_image: PIL Image (screenshot)
            instruction: Complete prompt with JSON format instructions included
            normalizing_range: Coordinate normalization range (default: 1000)

        Returns:
            AgentAction with normalized coordinates, or None after all retries exhausted
        """
        content_parts = [
            self._build_pil_image_content(pil_image),
            self._build_text_content(instruction),
        ]

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # TODO: For long-horizon tasks, reduce max_tokens and remove "reasoning"
                #       field from action JSON format to save tokens.
                response = self._send_to_api(content_parts, temperature=0.3, max_tokens=8192)

                if response.finish_reason in ("max_tokens", "length", "MAX_TOKENS"):
                    self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Response TRUNCATED (finish_reason={response.finish_reason})")

                action = parse_action_json(response.content)
                if action is None:
                    self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Parse failed, retrying...")
                    continue

                errors = validate_action(action, normalizing_range)
                if errors:
                    for err in errors:
                        self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Validation: {err}")
                    self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Validation failed, retrying...")
                    continue

                if attempt > 1:
                    self.logger.info(f"Action succeeded on attempt {attempt}/{self.MAX_RETRIES}")
                return action

            except Exception as e:
                self.logger.error(f"[Attempt {attempt}/{self.MAX_RETRIES}] API error: {e}")

        self.logger.error(f"analyze() failed after {self.MAX_RETRIES} attempts")
        return None


class MockVLMProvider(BaseVLMProvider):
    """Mock VLM provider for testing without API calls."""

    DEFAULT_MODEL = "mock-vlm"

    def __init__(self, api_key=None, model=None):
        super().__init__(api_key="mock", model=model or self.DEFAULT_MODEL)

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=4096):
        return VLMResponse(
            content=json.dumps(
                {
                    "reasoning": "Mock analysis",
                    "actions": [{"type": "click", "x": 500, "y": 300}, {"type": "press", "keys": ["esc"]}],
                }
            ),
            tokens_used=100,
            cost=0.0,
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
