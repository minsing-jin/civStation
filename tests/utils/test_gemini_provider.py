from types import SimpleNamespace

from google.genai import types

from computer_use_test.utils.llm_provider.gemini import GeminiVLMProvider


class _DummyModels:
    def __init__(self):
        self.last_config = None

    def generate_content(self, *, contents, model, config):
        self.last_config = config
        return SimpleNamespace(text='{"primitive":"policy_primitive"}', candidates=[])


class _DummyClient:
    def __init__(self):
        self.models = _DummyModels()


def test_gemini_31_pro_preview_uses_medium_thinking_even_when_disabled():
    provider = GeminiVLMProvider(api_key="test-key", model="gemini-3.1-pro-preview")
    provider.client = _DummyClient()

    provider._send_to_api(["prompt"], use_thinking=False)

    thinking = provider.client.models.last_config.thinking_config
    assert thinking.thinking_level == types.ThinkingLevel.MEDIUM
    assert thinking.thinking_budget is None


def test_gemini_flash_still_disables_thinking_with_zero_budget():
    provider = GeminiVLMProvider(api_key="test-key", model="gemini-3-flash-preview")
    provider.client = _DummyClient()

    provider._send_to_api(["prompt"], use_thinking=False)

    thinking = provider.client.models.last_config.thinking_config
    assert thinking.thinking_level is None
    assert thinking.thinking_budget == 0
