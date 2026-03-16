from computer_use_test.utils.llm_provider.parser import AgentAction
from computer_use_test.utils.screen import execute_action, norm_to_real


def test_norm_to_real_preserves_exact_scaled_coordinates():
    assert norm_to_real(6910, 10000, 10000) == 6910
    assert norm_to_real(6940, 10000, 10000) == 6940


def test_execute_action_absolute_click_bypasses_normalized_conversion(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        "computer_use_test.utils.screen.pyautogui.moveTo",
        lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
    )
    monkeypatch.setattr(
        "computer_use_test.utils.screen.pyautogui.click",
        lambda *args, **kwargs: calls.append(("click", args, kwargs)),
    )

    execute_action(
        AgentAction(
            action="click",
            coord_space="absolute",
            x=1234,
            y=567,
            button="left",
        ),
        screen_w=1600,
        screen_h=900,
        normalizing_range=10000,
        x_offset=111,
        y_offset=222,
    )

    assert calls == [
        ("moveTo", (1234, 567), {"duration": 0.5}),
        ("click", (), {"button": "left"}),
    ]


def test_execute_action_move_only_moves_cursor(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(
        "computer_use_test.utils.screen.pyautogui.moveTo",
        lambda *args, **kwargs: calls.append(("moveTo", args, kwargs)),
    )

    execute_action(
        AgentAction(
            action="move",
            x=600,
            y=300,
        ),
        screen_w=1600,
        screen_h=900,
        normalizing_range=1000,
        x_offset=0,
        y_offset=0,
    )

    assert calls == [
        ("moveTo", (960, 270), {"duration": 0.2}),
    ]
