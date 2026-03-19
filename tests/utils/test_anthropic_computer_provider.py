from types import SimpleNamespace

from PIL import Image

from computer_use_test.utils.llm_provider import create_provider, get_available_providers
from computer_use_test.utils.llm_provider.anthropic_computer import AnthropicComputerVLMProvider


def test_anthropic_computer_provider_registered():
    providers = get_available_providers()

    assert "anthropic-computer" in providers

    provider = create_provider("anthropic-computer", api_key="test-key", model="claude-sonnet-4-5")
    assert isinstance(provider, AnthropicComputerVLMProvider)


def test_anthropic_computer_provider_translates_left_click(monkeypatch):
    monkeypatch.setattr("anthropic.Anthropic", lambda *, api_key: SimpleNamespace())
    provider = AnthropicComputerVLMProvider(api_key="test-key", model="claude-sonnet-4-5")

    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="computer",
                input={"action": "left_click", "coordinate": [320, 160]},
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=14),
        stop_reason="tool_use",
    )
    provider.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)))

    action = provider.analyze(
        pil_image=Image.new("RGB", (640, 320), "white"),
        instruction="클릭해",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "click"
    assert action.button == "left"
    assert action.x == 500
    assert action.y == 500


def test_anthropic_computer_provider_translates_key_action(monkeypatch):
    monkeypatch.setattr("anthropic.Anthropic", lambda *, api_key: SimpleNamespace())
    provider = AnthropicComputerVLMProvider(api_key="test-key", model="claude-sonnet-4-5")

    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                name="computer",
                input={"action": "key", "text": "Cmd+Shift+P"},
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=14),
        stop_reason="tool_use",
    )
    provider.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kwargs: response)))

    action = provider.analyze(
        pil_image=Image.new("RGB", (640, 320), "white"),
        instruction="명령 팔레트를 열어",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "press"
    assert action.key == "cmd+shift+p"
