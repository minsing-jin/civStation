"""Tests for action-aware UI change detection."""

from PIL import Image, ImageDraw

from computer_use_test.utils.llm_provider.parser import AgentAction
from computer_use_test.utils.ui_change_detector import screenshots_similar, ui_changed


def _blank() -> Image.Image:
    return Image.new("RGB", (400, 300), color="black")


def test_ui_changed_detects_local_click_region_change():
    before = _blank()
    after = _blank()
    draw = ImageDraw.Draw(after)
    draw.rectangle((180, 120, 230, 170), fill="white")

    action = AgentAction(action="click", x=500, y=500)
    assert ui_changed(before, after, action=action) is True
    assert screenshots_similar(before, after, action=action) is False


def test_screenshots_similar_when_images_match():
    before = _blank()
    after = _blank()
    assert screenshots_similar(before, after) is True
