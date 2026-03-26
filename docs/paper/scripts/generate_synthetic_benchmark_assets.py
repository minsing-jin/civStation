from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

PAPER_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PAPER_ROOT / "arxiv/benchmarks/synthetic_ui"
SHOT_DIR = OUT_DIR / "screenshots"
DATASET_PATH = OUT_DIR / "synthetic_bbox_dataset.jsonl"
HERO_IMAGE = OUT_DIR / "hero_screenshot.png"


def _draw_centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(255, 255, 255)) -> None:
    draw.text(xy, text, fill=fill)


def make_confirm_image(path: Path) -> dict:
    img = Image.new("RGB", (1000, 1000), (29, 36, 48))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((180, 180, 820, 760), radius=28, fill=(48, 58, 74), outline=(160, 170, 190), width=3)
    _draw_centered_text(draw, (300, 240), "Research Complete", fill=(240, 240, 240))
    _draw_centered_text(draw, (270, 320), "Choose your next technology.", fill=(210, 215, 225))
    draw.rounded_rectangle((350, 640, 650, 730), radius=18, fill=(38, 151, 86), outline=(220, 255, 220), width=3)
    _draw_centered_text(draw, (455, 675), "Confirm", fill=(255, 255, 255))
    img.save(path)
    return {
        "case_id": "synthetic_confirm_001",
        "instruction": "Click the green Confirm button",
        "screenshot_path": str(Path("screenshots") / path.name),
        "image_size": {"width": 1000, "height": 1000},
        "action_sets": [
            {
                "actions": [
                    {
                        "type": "click",
                        "target_bbox": {"x_min": 350, "y_min": 640, "x_max": 650, "y_max": 730},
                        "button": "left",
                    }
                ]
            }
        ],
        "metadata": {"primitive": "popup_like"},
    }


def make_escape_image(path: Path) -> dict:
    img = Image.new("RGB", (1000, 1000), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((210, 230, 790, 710), radius=24, fill=(232, 233, 235), outline=(30, 30, 30), width=3)
    _draw_centered_text(draw, (370, 305), "Alert", fill=(30, 30, 30))
    _draw_centered_text(draw, (300, 390), "Press ESC to close this dialog.", fill=(60, 60, 60))
    draw.rectangle((720, 250, 760, 290), fill=(220, 75, 75))
    img.save(path)
    return {
        "case_id": "synthetic_escape_002",
        "instruction": "Dismiss the dialog by pressing escape",
        "screenshot_path": str(Path("screenshots") / path.name),
        "image_size": {"width": 1000, "height": 1000},
        "action_sets": [{"actions": [{"type": "press", "keys": ["esc"]}]}],
        "metadata": {"primitive": "popup_like"},
    }


def make_drag_image(path: Path) -> dict:
    img = Image.new("RGB", (1000, 1000), (34, 48, 34))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1000, 1000), fill=(41, 61, 41))
    draw.rectangle((470, 470, 530, 530), fill=(85, 170, 255), outline=(255, 255, 255), width=3)
    draw.rectangle((80, 80, 140, 140), fill=(255, 200, 60), outline=(255, 255, 255), width=3)
    draw.line((500, 500, 110, 110), fill=(255, 255, 255), width=5)
    _draw_centered_text(draw, (320, 860), "Drag from the blue square to the yellow square", fill=(255, 255, 255))
    img.save(path)
    return {
        "case_id": "synthetic_drag_003",
        "instruction": "Drag from the blue center square to the yellow upper-left square",
        "screenshot_path": str(Path("screenshots") / path.name),
        "image_size": {"width": 1000, "height": 1000},
        "action_sets": [
            {
                "actions": [
                    {
                        "type": "drag",
                        "start_bbox": {"x_min": 470, "y_min": 470, "x_max": 530, "y_max": 530},
                        "end_bbox": {"x_min": 80, "y_min": 80, "x_max": 140, "y_max": 140},
                        "button": "left",
                    }
                ]
            }
        ],
        "metadata": {"primitive": "drag_like"},
    }


def make_hero_image(path: Path) -> None:
    img = Image.new("RGB", (1440, 900), (22, 28, 38))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((90, 90, 1350, 810), radius=36, fill=(44, 55, 72), outline=(120, 145, 180), width=4)
    draw.rounded_rectangle((180, 180, 620, 720), radius=24, fill=(56, 74, 95), outline=(180, 210, 255), width=3)
    draw.rounded_rectangle((810, 180, 1260, 720), radius=24, fill=(68, 90, 68), outline=(220, 245, 220), width=3)
    _draw_centered_text(draw, (260, 230), "Policy Card List", fill=(255, 255, 255))
    _draw_centered_text(draw, (910, 230), "Government Slots", fill=(255, 255, 255))
    draw.rounded_rectangle((260, 320, 540, 390), radius=14, fill=(198, 148, 68))
    draw.rounded_rectangle((260, 420, 540, 490), radius=14, fill=(80, 133, 196))
    draw.rounded_rectangle((920, 350, 1160, 440), radius=18, fill=(40, 150, 88))
    _draw_centered_text(draw, (330, 345), "Urban Planning", fill=(20, 20, 20))
    _draw_centered_text(draw, (330, 445), "God King", fill=(255, 255, 255))
    _draw_centered_text(draw, (990, 385), "Confirm Policies", fill=(255, 255, 255))
    img.save(path)


def main() -> int:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        make_confirm_image(SHOT_DIR / "confirm.png"),
        make_escape_image(SHOT_DIR / "escape.png"),
        make_drag_image(SHOT_DIR / "drag.png"),
    ]
    make_hero_image(HERO_IMAGE)

    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {DATASET_PATH}")
    print(f"Wrote {HERO_IMAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
