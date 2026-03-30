"""
Claude (Anthropic) VLM Provider.

Implements only the provider-specific methods:
- _send_to_api(): Send content to Anthropic Messages API
- _build_image_content(): Encode file path to Anthropic image format
- _build_pil_image_content(): Encode PIL image to Anthropic image format
- _build_text_content(): Build Anthropic text block

All shared logic (call_vlm, parse_to_agent_plan, analyze, etc.)
lives in BaseVLMProvider.
"""

import base64
import os
from pathlib import Path

from civStation.utils.llm_provider.base import BaseVLMProvider, VLMResponse


class ClaudeVLMProvider(BaseVLMProvider):
    """
    Claude VLM provider using Anthropic API.

    Supports Claude 3/4 models with vision capabilities.
    """

    DEFAULT_MODEL = "claude-4-5-sonnet-20241022"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key, model)

        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")

        if self.model is None:
            self.model = self.DEFAULT_MODEL

        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic") from e

    # ==================== Provider-specific helpers ====================

    def _encode_image(self, image_path: str | Path) -> tuple[str, str]:
        """Encode image file to (media_type, base64_data)."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = image_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        return media_type, image_data

    # ==================== Abstract method implementations ====================

    def _send_to_api(
        self,
        content_parts: list,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        use_thinking: bool = True,
    ) -> VLMResponse:
        """Send content parts to Anthropic Messages API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content_parts}],
            )

            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Cost estimation
            if "opus" in self.model.lower():
                cost = (response.usage.input_tokens * 15 + response.usage.output_tokens * 75) / 1_000_000
            elif "sonnet" in self.model.lower():
                cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
            else:  # haiku
                cost = (response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000

            finish_reason = response.stop_reason  # "end_turn" or "max_tokens"

            if finish_reason == "max_tokens":
                self.logger.warning(
                    f"Claude response TRUNCATED (stop_reason={finish_reason})."
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
            raise RuntimeError(f"Claude API call failed: {e}") from e

    def _build_image_content(self, image_path: str | Path) -> object:
        """Build Anthropic image content from file path."""
        media_type, image_data = self._encode_image(image_path)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        }

    def _build_pil_image_content(self, pil_image, jpeg_quality: int | None = None) -> object:
        """Build Anthropic image content from PIL image (JPEG for speed)."""
        import io

        from civStation.utils.screen import VLM_JPEG_QUALITY

        quality = jpeg_quality or VLM_JPEG_QUALITY
        buffer = io.BytesIO()
        img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        img.save(buffer, format="JPEG", quality=quality)
        image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            },
        }

    def _build_text_content(self, text: str) -> object:
        """Build Anthropic text content block."""
        return {"type": "text", "text": text}

    def get_provider_name(self) -> str:
        return "claude"
