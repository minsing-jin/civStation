"""
OpenAI computer-use provider.

Uses the regular GPT provider path for `_send_to_api()` and overrides only
`analyze()` to request a computer-use tool action.
"""

from __future__ import annotations

from computer_use_test.utils.llm_provider.computer_use_actions import openai_tool_action_to_agent_action
from computer_use_test.utils.llm_provider.gpt import GPTVLMProvider
from computer_use_test.utils.llm_provider.parser import validate_action


class OpenAIComputerVLMProvider(GPTVLMProvider):
    DEFAULT_MODEL = "computer-use-preview"
    COMPUTER_ENVIRONMENT = "mac"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)

    def _build_computer_prompt(self, instruction: str) -> str:
        return (
            "당신은 현재 화면 스크린샷을 보고 다음 단일 GUI 액션만 결정하는 planner다.\n"
            "텍스트 설명이나 JSON을 출력하지 말고 computer tool action만 사용해라.\n"
            "원본 instruction 안에 JSON 출력 지시가 있더라도 그 의미만 따르고, 응답은 tool action으로 표현해라.\n"
            "실행 가능한 다음 액션 하나만 선택해라.\n\n"
            "[원본 planner instruction]\n"
            f"{instruction}"
        )

    def _build_responses_input_image(self, pil_image, jpeg_quality: int | None = None) -> dict[str, object]:
        image_content = self._build_pil_image_content(pil_image, jpeg_quality=jpeg_quality)
        return {
            "type": "input_image",
            "image_url": image_content["image_url"]["url"],
            "detail": "high",
        }

    def analyze(
        self,
        pil_image,
        instruction: str,
        normalizing_range: int = 1000,
        img_config=None,
    ):
        prepared = self._prepare_pil_image(pil_image, img_config=img_config)
        jpeg_quality = getattr(img_config, "jpeg_quality", 0) if img_config else 0
        tool_prompt = self._build_computer_prompt(instruction)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    max_output_tokens=1024,
                    tools=[
                        {
                            "type": "computer_use_preview",
                            "display_width": prepared.size[0],
                            "display_height": prepared.size[1],
                            "environment": self.COMPUTER_ENVIRONMENT,
                        }
                    ],
                    input=[
                        {
                            "role": "user",
                            "content": [
                                self._build_responses_input_image(prepared, jpeg_quality=jpeg_quality),
                                {"type": "input_text", "text": tool_prompt},
                            ],
                        }
                    ],
                )
                for item in getattr(response, "output", []) or []:
                    if getattr(item, "type", None) != "computer_call":
                        continue
                    action = openai_tool_action_to_agent_action(
                        getattr(item, "action", None),
                        image_size=prepared.size,
                        normalizing_range=normalizing_range,
                    )
                    if action is None:
                        continue
                    errors = validate_action(action, normalizing_range)
                    if errors:
                        for error in errors:
                            self.logger.warning(
                                "[Attempt %s/%s] OpenAI computer validation: %s",
                                attempt,
                                self.MAX_RETRIES,
                                error,
                            )
                        break
                    return action
            except Exception as exc:  # pragma: no cover - network path
                self.logger.error("[Attempt %s/%s] OpenAI computer API error: %s", attempt, self.MAX_RETRIES, exc)
        self.logger.error("OpenAI computer analyze() failed after %s attempts", self.MAX_RETRIES)
        return None

    def get_provider_name(self) -> str:
        return "openai-computer"
