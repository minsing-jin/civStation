"""
Anthropic computer-use provider.

Uses the regular Claude provider path for `_send_to_api()` and overrides only
`analyze()` to request a computer-use tool action.
"""

from __future__ import annotations

from civStation.utils.llm_provider.claude import ClaudeVLMProvider
from civStation.utils.llm_provider.computer_use_actions import anthropic_tool_input_to_agent_action
from civStation.utils.llm_provider.parser import validate_action


class AnthropicComputerVLMProvider(ClaudeVLMProvider):
    COMPUTER_TOOL_BETA = "computer-use-2025-01-24"
    COMPUTER_TOOL_TYPE = "computer_20250124"
    LATEST_COMPUTER_TOOL_BETA = "computer-use-2025-11-24"
    LATEST_COMPUTER_TOOL_TYPE = "computer_20251124"

    def _uses_latest_computer_tool(self) -> bool:
        model = self.model.lower().replace(".", "-")
        return any(token in model for token in ("sonnet-4-6", "opus-4-6", "opus-4-5"))

    def _computer_tool_beta(self) -> str:
        if self._uses_latest_computer_tool():
            return self.LATEST_COMPUTER_TOOL_BETA
        return self.COMPUTER_TOOL_BETA

    def _computer_tool_type(self) -> str:
        if self._uses_latest_computer_tool():
            return self.LATEST_COMPUTER_TOOL_TYPE
        return self.COMPUTER_TOOL_TYPE

    def _build_computer_prompt(self, instruction: str) -> str:
        return (
            "당신은 현재 화면 스크린샷을 보고 다음 단일 GUI 액션만 결정하는 planner다.\n"
            "이 환경은 zoom action을 지원하지 않으니 zoom은 사용하지 마라.\n"
            "텍스트 설명이나 JSON을 출력하지 말고 computer tool action만 사용해라.\n"
            "원본 instruction 안에 JSON 출력 지시가 있더라도 그 의미만 따르고, 응답은 tool action으로 표현해라.\n"
            "실행 가능한 다음 액션 하나만 선택해라.\n\n"
            "[원본 planner instruction]\n"
            f"{instruction}"
        )

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
                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                self._build_pil_image_content(prepared, jpeg_quality=jpeg_quality),
                                {"type": "text", "text": tool_prompt},
                            ],
                        }
                    ],
                    tools=[
                        {
                            "type": self._computer_tool_type(),
                            "name": "computer",
                            "display_width_px": prepared.size[0],
                            "display_height_px": prepared.size[1],
                            "display_number": 1,
                        }
                    ],
                    betas=[self._computer_tool_beta()],
                )
                for block in getattr(response, "content", []) or []:
                    if getattr(block, "type", None) != "tool_use" or getattr(block, "name", None) != "computer":
                        continue
                    try:
                        action = anthropic_tool_input_to_agent_action(
                            getattr(block, "input", None),
                            image_size=prepared.size,
                            normalizing_range=normalizing_range,
                        )
                    except ValueError as exc:
                        self.logger.warning(
                            "[Attempt %s/%s] Anthropic computer unsupported action: %s",
                            attempt,
                            self.MAX_RETRIES,
                            exc,
                        )
                        continue
                    if action is None:
                        continue
                    errors = validate_action(action, normalizing_range)
                    if errors:
                        for error in errors:
                            self.logger.warning(
                                "[Attempt %s/%s] Anthropic computer validation: %s",
                                attempt,
                                self.MAX_RETRIES,
                                error,
                            )
                        break
                    return action
            except Exception as exc:  # pragma: no cover - network path
                self.logger.error(
                    "[Attempt %s/%s] Anthropic computer API error: %s",
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                )
        self.logger.error("Anthropic computer analyze() failed after %s attempts", self.MAX_RETRIES)
        return None

    def get_provider_name(self) -> str:
        return "anthropic-computer"
