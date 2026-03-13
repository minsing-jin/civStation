from types import SimpleNamespace

import pytest
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


def test_gemini_provider_uses_explicit_api_key_for_client(monkeypatch):
    captured = {}

    def fake_client(*, api_key):
        captured["api_key"] = api_key
        return _DummyClient()

    monkeypatch.delenv("GENAI_API_KEY", raising=False)
    monkeypatch.setattr("google.genai.Client", fake_client)

    GeminiVLMProvider(api_key="test-key", model="gemini-3.1-pro-preview")

    assert captured["api_key"] == "test-key"


def test_gemini_provider_uses_genai_env_key_when_api_key_missing(monkeypatch):
    captured = {}

    def fake_client(*, api_key):
        captured["api_key"] = api_key
        return _DummyClient()

    monkeypatch.setenv("GENAI_API_KEY", "env-key")
    monkeypatch.setattr("google.genai.Client", fake_client)

    GeminiVLMProvider(api_key=None, model="gemini-3.1-pro-preview")

    assert captured["api_key"] == "env-key"


def test_gemini_provider_requires_genai_env_key_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("GENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GENAI_API_KEY"):
        GeminiVLMProvider(api_key=None, model="gemini-3.1-pro-preview")


def test_gemini_31_pro_preview_uses_medium_thinking_even_when_disabled(monkeypatch):
    monkeypatch.setattr("google.genai.Client", lambda *, api_key: _DummyClient())
    provider = GeminiVLMProvider(api_key="test-key", model="gemini-3.1-pro-preview")

    provider._send_to_api(["prompt"], use_thinking=False)

    thinking = provider.client.models.last_config.thinking_config
    assert thinking.thinking_level == types.ThinkingLevel.MEDIUM
    assert thinking.thinking_budget is None


def test_gemini_flash_still_disables_thinking_with_zero_budget(monkeypatch):
    monkeypatch.setattr("google.genai.Client", lambda *, api_key: _DummyClient())
    provider = GeminiVLMProvider(api_key="test-key", model="gemini-3-flash-preview")

    provider._send_to_api(["prompt"], use_thinking=False)

    thinking = provider.client.models.last_config.thinking_config
    assert thinking.thinking_level is None
    assert thinking.thinking_budget == 0
