from types import SimpleNamespace

from PIL import Image

from computer_use_test.utils.llm_provider import create_provider, get_available_providers
from computer_use_test.utils.llm_provider.openai_computer import OpenAIComputerVLMProvider


def test_openai_computer_provider_registered():
    providers = get_available_providers()

    assert "openai-computer" in providers

    provider = create_provider("openai-computer", api_key="test-key", model="gpt-5.4")
    assert isinstance(provider, OpenAIComputerVLMProvider)


def test_openai_computer_provider_translates_click_to_normalized_action(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="gpt-5.4")

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="computer_call",
                action=SimpleNamespace(type="click", x=256, y=384, button="left"),
                call_id="call_1",
                pending_safety_checks=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    action = provider.analyze(
        pil_image=Image.new("RGB", (512, 768), "white"),
        instruction="다음 액션을 정해라",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "click"
    assert action.coord_space == "normalized"
    assert action.x == 500
    assert action.y == 500
    assert action.button == "left"


def test_openai_computer_provider_translates_keypress_to_press(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="gpt-5.4")

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="computer_call",
                action=SimpleNamespace(type="keypress", keys=["CMD", "S"]),
                call_id="call_1",
                pending_safety_checks=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    action = provider.analyze(
        pil_image=Image.new("RGB", (800, 600), "white"),
        instruction="저장해",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "press"
    assert action.key == "cmd+s"


def test_openai_computer_provider_ignores_screenshot_action_and_uses_next_action(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="gpt-5.4")

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="computer_call",
                action=SimpleNamespace(type="screenshot"),
                call_id="call_1",
                pending_safety_checks=[],
                status="completed",
            ),
            SimpleNamespace(
                type="computer_call",
                action=SimpleNamespace(type="move", x=100, y=50),
                call_id="call_2",
                pending_safety_checks=[],
                status="completed",
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))

    action = provider.analyze(
        pil_image=Image.new("RGB", (200, 100), "white"),
        instruction="커서를 이동해",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "move"
    assert action.x == 500
    assert action.y == 500
