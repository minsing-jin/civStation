"""
Gemini (Google) VLM Provider.

Implements only the provider-specific methods:
- _send_to_api(): Send content to Google Generative AI API
- _build_image_content(): Load file as PIL image (Gemini's native format)
- _build_pil_image_content(): Pass through PIL image
- _build_text_content(): Pass through text string

All shared logic (call_vlm, parse_to_agent_plan, analyze, etc.)
lives in BaseVLMProvider.
"""

import os
from pathlib import Path
from typing import Any

from google.genai import types

from computer_use_test.utils.llm_provider.base import BaseVLMProvider, VLMResponse


class GeminiVLMProvider(BaseVLMProvider):
    """
    Gemini VLM provider using Google Generative AI API.

    Supports Gemini Pro Vision, Gemini 1.5/2.0/3.0 models.
    """

    DEFAULT_MODEL = "gemini-3-flash-preview"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key, model)

        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")

        if self.model is None:
            self.model = self.DEFAULT_MODEL

        try:
            from google import genai

            self.client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
        except ImportError as e:
            raise ImportError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            ) from e

    # ==================== Abstract method implementations ====================

    def _send_to_api(
        self,
        content_parts: list,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        response_json_schema: dict | None = None,
        use_thinking: bool = True,
    ) -> VLMResponse:
        """Send content parts to Google Generative AI API."""
        try:
            # Build config kwargs dynamically
            config_kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Add JSON schema if provided (enforced)
            if response_json_schema is not None:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_json_schema"] = response_json_schema

            # Add thinking config
            if use_thinking:
                config_kwargs["thinking_config"] = types.ThinkingConfig()
            else:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            response = self.client.models.generate_content(
                contents=content_parts,
                model=self.model,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            response_text = response.text

            # Rough token estimation (Gemini doesn't always expose detailed usage)
            tokens_used = int(len(response_text.split()) * 1.3)

            if "flash" in self.model.lower():
                cost = tokens_used * 0.000075
            elif "pro" in self.model.lower():
                cost = tokens_used * 0.000125
            else:
                cost = tokens_used * 0.0001

            # Extract finish_reason from Gemini response (enum → string)
            finish_reason = None
            if response.candidates:
                raw_reason = response.candidates[0].finish_reason
                finish_reason = raw_reason.name if hasattr(raw_reason, "name") else str(raw_reason)

            if finish_reason == "MAX_TOKENS":
                self.logger.warning(
                    f"Gemini response TRUNCATED (finish_reason={finish_reason})."
                    " Output likely incomplete. Consider increasing max_tokens."
                )

            return VLMResponse(
                content=response_text,
                raw_response=response,
                tokens_used=tokens_used,
                cost=cost,
                finish_reason=finish_reason,
            )

        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}") from e

    def _build_image_content(self, image_path: str | Path) -> object:
        """Load image file as PIL Image (Gemini's native format)."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            from PIL import Image

            return Image.open(image_path)
        except ImportError as e:
            raise ImportError("PIL package not installed. Install with: pip install Pillow") from e

    def _build_pil_image_content(self, pil_image, jpeg_quality: int | None = None) -> object:
        """Pass through PIL image (Gemini uses PIL natively, JPEG quality N/A)."""
        return pil_image

    def _build_text_content(self, text: str) -> object:
        """Pass through text string (Gemini accepts plain strings)."""
        return text

    def get_provider_name(self) -> str:
        return "gemini"
