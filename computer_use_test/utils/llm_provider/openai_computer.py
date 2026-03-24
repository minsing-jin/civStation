"""
OpenAI computer-use provider.

Uses the regular GPT provider path for `_send_to_api()` and overrides only
`analyze()` to request a computer-use tool action.
"""

from __future__ import annotations

import base64
import io

from computer_use_test.utils.llm_provider.computer_use_actions import openai_tool_action_to_agent_action
from computer_use_test.utils.llm_provider.gpt import GPTVLMProvider
from computer_use_test.utils.llm_provider.parser import validate_action


class OpenAIComputerVLMProvider(GPTVLMProvider):
    DEFAULT_MODEL = "gpt-5.4"
    COMPUTER_ENVIRONMENT = "mac"
    MAX_COMPUTER_STEPS = 3

    def __init__(self, api_key: str | None = None, model: str | None = None):
        super().__init__(api_key=api_key, model=model or self.DEFAULT_MODEL)

    def _build_computer_prompt(self, instruction: str) -> str:
        return (
            "당신은 현재 화면에서 다음 단일 GUI 액션만 결정하는 planner다.\n"
            "필요하면 먼저 screenshot action으로 화면을 확인해도 된다.\n"
            "텍스트 설명이나 JSON을 출력하지 말고 computer tool action만 사용해라.\n"
            "원본 instruction 안에 JSON 출력 지시가 있더라도 그 의미만 따르고, 응답은 tool action으로 표현해라.\n"
            "실행 가능한 다음 액션 하나만 선택해라.\n\n"
            "[원본 planner instruction]\n"
            f"{instruction}"
        )

    def _uses_builtin_computer_tool(self) -> bool:
        return self.model.lower().startswith("gpt-5")

    def _build_tool_spec(self, prepared) -> list[dict[str, object]]:
        if self._uses_builtin_computer_tool():
            return [{"type": "computer"}]
        return [
            {
                "type": "computer_use_preview",
                "display_width": prepared.size[0],
                "display_height": prepared.size[1],
                "environment": self.COMPUTER_ENVIRONMENT,
            }
        ]

    def _build_responses_input_image(self, pil_image, jpeg_quality: int | None = None) -> dict[str, object]:
        image_content = self._build_pil_image_content(pil_image, jpeg_quality=jpeg_quality)
        return {
            "type": "input_image",
            "image_url": image_content["image_url"]["url"],
            "detail": "high",
        }

    def _build_computer_screenshot_output(self, pil_image) -> dict[str, object]:
        image = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "type": "computer_screenshot",
            "image_url": f"data:image/png;base64,{image_data}",
            "detail": "original",
        }

    def _iter_tool_actions(self, item) -> list[object]:
        actions = getattr(item, "actions", None)
        if actions:
            return list(actions)
        legacy_action = getattr(item, "action", None)
        return [legacy_action] if legacy_action is not None else []

    def _extract_action_from_response(self, response, *, image_size: tuple[int, int], normalizing_range: int):
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "computer_call":
                continue
            for tool_action in self._iter_tool_actions(item):
                action = openai_tool_action_to_agent_action(
                    tool_action,
                    image_size=image_size,
                    normalizing_range=normalizing_range,
                )
                if action is None:
                    continue
                errors = validate_action(action, normalizing_range)
                if errors:
                    for error in errors:
                        self.logger.warning("OpenAI computer validation: %s", error)
                    return None
                return action
        return None

    def _first_computer_call(self, response):
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "computer_call":
                return item
        return None

    def _analyze_builtin_computer(
        self,
        prepared,
        tool_prompt: str,
        *,
        normalizing_range: int,
    ):
        response = self.client.responses.create(
            model=self.model,
            max_output_tokens=1024,
            tools=self._build_tool_spec(prepared),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": tool_prompt},
                    ],
                }
            ],
        )

        for _ in range(self.MAX_COMPUTER_STEPS):
            action = self._extract_action_from_response(
                response,
                image_size=prepared.size,
                normalizing_range=normalizing_range,
            )
            if action is not None:
                return action

            computer_call = self._first_computer_call(response)
            if computer_call is None:
                return None

            response_id = getattr(response, "id", None)
            call_id = getattr(computer_call, "call_id", None)
            if response_id is None or call_id is None:
                return None

            response = self.client.responses.create(
                model=self.model,
                max_output_tokens=1024,
                tools=self._build_tool_spec(prepared),
                previous_response_id=response_id,
                input=[
                    {
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": self._build_computer_screenshot_output(prepared),
                    }
                ],
            )
        return None

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
                if self._uses_builtin_computer_tool():
                    action = self._analyze_builtin_computer(
                        prepared,
                        tool_prompt,
                        normalizing_range=normalizing_range,
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        max_output_tokens=1024,
                        tools=self._build_tool_spec(prepared),
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
                    action = self._extract_action_from_response(
                        response,
                        image_size=prepared.size,
                        normalizing_range=normalizing_range,
                    )
                if action is not None:
                    return action
            except Exception as exc:  # pragma: no cover - network path
                self.logger.error("[Attempt %s/%s] OpenAI computer API error: %s", attempt, self.MAX_RETRIES, exc)
        self.logger.error("OpenAI computer analyze() failed after %s attempts", self.MAX_RETRIES)
        return None

    def get_provider_name(self) -> str:
        return "openai-computer"
