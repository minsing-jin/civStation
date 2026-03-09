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
    parse_action_json_list,
    parse_to_agent_plan,
    validate_action,
)

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

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        resize_for_vlm: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.resize_for_vlm = resize_for_vlm
        self.logger = logger

    # ==================== Core (implement in subclass) ====================

    @abstractmethod
    def _send_to_api(
        self,
        content_parts: list,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        use_thinking: bool = True,
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
    def _build_pil_image_content(self, pil_image, jpeg_quality: int | None = None) -> object:
        """Build provider-specific image content from PIL image."""
        pass

    @abstractmethod
    def _build_text_content(self, text: str) -> object:
        """Build provider-specific text content."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    # ==================== Image pre-processing ====================

    def _prepare_pil_image(self, pil_image, img_config=None):
        """Preprocess a PIL image before sending to the VLM.

        When *img_config* is provided, applies the parameterized pipeline.
        Otherwise falls back to PLANNER_DEFAULT when ``self.resize_for_vlm``
        is True.
        """
        if img_config is not None:
            from computer_use_test.utils.image_pipeline import process_image

            return process_image(pil_image, img_config).image

        if self.resize_for_vlm:
            from computer_use_test.utils.image_pipeline import PLANNER_DEFAULT, process_image

            return process_image(pil_image, PLANNER_DEFAULT).image
        return pil_image

    # ==================== Static Evaluation (file-based) ====================

    def call_vlm(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
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
        max_tokens: int = 8192,
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
        img_config=None,
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
            img_config: Optional ImagePipelineConfig for preprocessing

        Returns:
            AgentAction with normalized coordinates, or None after all retries exhausted
        """
        prepared = self._prepare_pil_image(pil_image, img_config=img_config)
        jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
        build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
        content_parts = [
            self._build_pil_image_content(prepared, **build_kwargs),
            self._build_text_content(instruction),
        ]

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # TODO: For long-horizon tasks, reduce max_tokens and remove "reasoning"
                #       field from action JSON format to save tokens.
                # max_tokens must be large enough to cover thinking tokens (Gemini)
                # plus the actual JSON response (~200 tokens).
                response = self._send_to_api(content_parts, temperature=0.3, max_tokens=8192)

                if response.finish_reason in ("max_tokens", "length", "MAX_TOKENS"):
                    self.logger.warning(
                        f"[Attempt {attempt}/{self.MAX_RETRIES}] Response TRUNCATED"
                        f" (finish_reason={response.finish_reason})"
                    )

                action = parse_action_json(response.content)
                if action is None:
                    self.logger.warning(
                        f"[Attempt {attempt}/{self.MAX_RETRIES}] Parse failed, retrying...\n"
                        f"  Raw response (first 500 chars): {response.content[:500]}"
                    )
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

    def analyze_multi(
        self,
        pil_image,
        instruction: str,
        normalizing_range: int = 1000,
        img_config=None,
    ) -> list[AgentAction] | None:
        """
        Analyze PIL image and return a list of actions for multi-step primitives.

        Like analyze() but parses a JSON array response into multiple AgentActions.
        Used for primitives that need sequential multi-action execution (e.g., policy).

        Args:
            pil_image: PIL Image (screenshot)
            instruction: Complete prompt with JSON format instructions included
            normalizing_range: Coordinate normalization range (default: 1000)
            img_config: Optional ImagePipelineConfig for preprocessing

        Returns:
            List of AgentActions, or None after all retries exhausted
        """
        prepared = self._prepare_pil_image(pil_image, img_config=img_config)
        jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
        build_kwargs = {"jpeg_quality": jpeg_quality} if jpeg_quality > 0 else {}
        content_parts = [
            self._build_pil_image_content(prepared, **build_kwargs),
            self._build_text_content(instruction),
        ]

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self._send_to_api(content_parts, temperature=0.3, max_tokens=8192)

                if response.finish_reason in ("max_tokens", "length", "MAX_TOKENS"):
                    self.logger.warning(
                        f"[Attempt {attempt}/{self.MAX_RETRIES}] Response TRUNCATED"
                        f" (finish_reason={response.finish_reason})"
                    )

                actions = parse_action_json_list(response.content)
                if not actions:
                    self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Multi-parse failed, retrying...")
                    continue

                # Validate each action
                all_valid = True
                for i, action in enumerate(actions):
                    errors = validate_action(action, normalizing_range)
                    if errors:
                        for err in errors:
                            self.logger.warning(f"[Attempt {attempt}] Action {i} validation: {err}")
                        all_valid = False
                        break

                if not all_valid:
                    self.logger.warning(f"[Attempt {attempt}/{self.MAX_RETRIES}] Validation failed, retrying...")
                    continue

                if attempt > 1:
                    self.logger.info(f"Multi-action succeeded on attempt {attempt}/{self.MAX_RETRIES}")
                return actions

            except Exception as e:
                self.logger.error(f"[Attempt {attempt}/{self.MAX_RETRIES}] API error: {e}")

        self.logger.error(f"analyze_multi() failed after {self.MAX_RETRIES} attempts")
        return None


class MockVLMProvider(BaseVLMProvider):
    """Mock VLM provider for testing without API calls."""

    DEFAULT_MODEL = "mock-vlm"

    def __init__(self, api_key=None, model=None):
        super().__init__(api_key="mock", model=model or self.DEFAULT_MODEL)

    def _send_to_api(self, content_parts, temperature=0.7, max_tokens=4096, use_thinking=True):
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

    def _build_pil_image_content(self, pil_image, jpeg_quality: int | None = None):
        return {"type": "image", "data": "mock"}

    def _build_text_content(self, text):
        return {"type": "text", "text": text}

    def get_provider_name(self):
        return "mock"
