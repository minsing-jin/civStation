from pathlib import Path

from PIL import Image

from civStation.utils.screen import capture_screen_pil
from civStation.utils.screenshot_trajectory import (
    get_screenshot_trajectory_root,
    start_screenshot_trajectory_session,
)


def test_get_screenshot_trajectory_root_defaults_to_project_tmp_root():
    expected = Path(__file__).resolve().parents[2] / ".tmp" / "civStation" / "screenshot_trajectories"
    assert get_screenshot_trajectory_root() == expected


def test_screenshot_trajectory_session_keeps_latest_20_images(tmp_path):
    session = start_screenshot_trajectory_session(base_dir=tmp_path, max_images=20)

    try:
        for index in range(22):
            session.record(Image.new("RGB", (8, 8), (index, index, index)))
    finally:
        session.close()

    images = sorted(session.path.glob("*.png"))
    assert len(images) == 20
    assert images[0].name.startswith("0003_")
    assert images[-1].name.startswith("0022_")


def test_capture_screen_pil_records_trajectory_when_session_is_active(tmp_path, monkeypatch):
    monkeypatch.setattr("civStation.utils.screen.pyautogui.size", lambda: (100, 50))
    monkeypatch.setattr(
        "civStation.utils.screen.pyautogui.screenshot",
        lambda: Image.new("RGB", (100, 50), "white"),
    )
    monkeypatch.setattr("civStation.utils.screen._detect_game_window", lambda: None)

    session = start_screenshot_trajectory_session(base_dir=tmp_path, max_images=20)

    try:
        image, width, height, x_offset, y_offset = capture_screen_pil()
    finally:
        session.close()

    assert image.size == (100, 50)
    assert (width, height, x_offset, y_offset) == (100, 50, 0, 0)
    assert len(list(session.path.glob("*.png"))) == 1
