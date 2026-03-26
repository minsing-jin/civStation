from types import SimpleNamespace

from PIL import Image

from civStation.utils.llm_provider import create_provider, get_available_providers
from civStation.utils.llm_provider.openai_computer import OpenAIComputerVLMProvider


def test_openai_computer_provider_registered():
    providers = get_available_providers()

    assert "openai" in providers
    assert "openai-computer" in providers

    provider = create_provider("openai-computer", api_key="test-key", model="gpt-5.4")
    assert isinstance(provider, OpenAIComputerVLMProvider)


def test_openai_computer_provider_uses_computer_tool_for_gpt54(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="gpt-5.4")

    calls = []
    response = SimpleNamespace(
        id="resp_1",
        output=[
            SimpleNamespace(
                type="computer_call",
                actions=[SimpleNamespace(type="click", x=256, y=384, button="left")],
                call_id="call_1",
                pending_safety_checks=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

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
    assert calls[0]["tools"] == [{"type": "computer"}]


def test_openai_computer_provider_handles_screenshot_first_turn_for_gpt54(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="gpt-5.4")

    first_response = SimpleNamespace(
        id="resp_1",
        output=[
            SimpleNamespace(
                type="computer_call",
                actions=[SimpleNamespace(type="screenshot")],
                call_id="call_1",
                pending_safety_checks=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )
    second_response = SimpleNamespace(
        id="resp_2",
        output=[
            SimpleNamespace(
                type="computer_call",
                actions=[SimpleNamespace(type="keypress", keys=["CMD", "S"])],
                call_id="call_2",
                pending_safety_checks=[],
                status="completed",
            )
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return first_response
        return second_response

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

    action = provider.analyze(
        pil_image=Image.new("RGB", (800, 600), "white"),
        instruction="저장해",
        normalizing_range=1000,
        img_config=None,
    )

    assert action is not None
    assert action.action == "press"
    assert action.key == "cmd+s"
    assert calls[0]["tools"] == [{"type": "computer"}]
    assert calls[1]["previous_response_id"] == "resp_1"
    assert calls[1]["input"][0]["type"] == "computer_call_output"
    assert calls[1]["input"][0]["call_id"] == "call_1"
    assert calls[1]["input"][0]["output"]["type"] == "computer_screenshot"


def test_openai_computer_provider_supports_preview_contract(monkeypatch):
    monkeypatch.setattr("openai.OpenAI", lambda *, api_key: SimpleNamespace())
    provider = OpenAIComputerVLMProvider(api_key="test-key", model="computer-use-preview")

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
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return response

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))

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
    assert calls[0]["tools"][0]["type"] == "computer_use_preview"
