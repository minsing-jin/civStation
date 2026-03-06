"""
GPT (OpenAI) VLM Provider.

Implements only the provider-specific methods:
- _send_to_api(): Send content to OpenAI Chat Completions API
- _build_image_content(): Encode file path to OpenAI image_url format
- _build_pil_image_content(): Encode PIL image to OpenAI image_url format
- _build_text_content(): Build OpenAI text block

All shared logic (call_vlm, parse_to_agent_plan, analyze, etc.)
lives in BaseVLMProvider.
"""

import base64
import os
from pathlib import Path

from computer_use_test.utils.llm_provider.base import BaseVLMProvider, VLMResponse


class GPTVLMProvider(BaseVLMProvider):
    """
    GPT VLM provider using OpenAI API.

    Supports GPT-4 Vision, GPT-4o, and GPT-4o-mini with vision capabilities.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key, model)

        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

        if self.model is None:
            self.model = self.DEFAULT_MODEL

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("openai package not installed. Install with: pip install openai") from e

    # ==================== Provider-specific helpers ====================

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode image file to base64 data URL."""
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
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{media_type};base64,{image_data}"

    # ==================== Abstract method implementations ====================

    # Models that require max_completion_tokens instead of max_tokens
    _NEW_API_PREFIXES = ("gpt-5", "o1", "o3", "o4")

    def _uses_new_token_param(self) -> bool:
        """Check if the current model requires max_completion_tokens."""
        model_lower = self.model.lower()
        return any(model_lower.startswith(p) for p in self._NEW_API_PREFIXES)

    def _send_to_api(
        self,
        content_parts: list,
        temperature: float = 0.7,
        max_tokens: int = 8192,
        use_thinking: bool = True,
    ) -> VLMResponse:
        """Send content parts to OpenAI Chat Completions API."""
        try:
            is_new_model = self._uses_new_token_param()

            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": content_parts}],
            }

            # GPT-5/o1/o3/o4: temperature not supported, use max_completion_tokens
            if is_new_model:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**kwargs)

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            # Cost estimation
            if "gpt-4o" in self.model.lower():
                if "mini" in self.model.lower():
                    cost = (response.usage.prompt_tokens * 0.15 + response.usage.completion_tokens * 0.6) / 1_000_000
                else:
                    cost = (response.usage.prompt_tokens * 2.5 + response.usage.completion_tokens * 10) / 1_000_000
            elif "gpt-4-turbo" in self.model.lower() or "gpt-4-vision" in self.model.lower():
                cost = (response.usage.prompt_tokens * 10 + response.usage.completion_tokens * 30) / 1_000_000
            else:
                cost = (response.usage.prompt_tokens * 30 + response.usage.completion_tokens * 60) / 1_000_000

            finish_reason = response.choices[0].finish_reason  # "stop" or "length"

            if finish_reason == "length":
                self.logger.warning(
                    f"GPT response TRUNCATED (finish_reason={finish_reason})."
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
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def _build_image_content(self, image_path: str | Path) -> object:
        """Build OpenAI image_url content from file path."""
        image_url = self._encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": image_url},
        }

    def _build_pil_image_content(self, pil_image) -> object:
        """Build OpenAI image_url content from PIL image (JPEG for speed)."""
        import io

        from computer_use_test.utils.screen import VLM_JPEG_QUALITY

        buffer = io.BytesIO()
        img = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        img.save(buffer, format="JPEG", quality=VLM_JPEG_QUALITY)
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{image_data}"

        return {
            "type": "image_url",
            "image_url": {"url": data_url},
        }

    def _build_text_content(self, text: str) -> object:
        """Build OpenAI text content block."""
        return {"type": "text", "text": text}

    def get_provider_name(self) -> str:
        return "gpt"
