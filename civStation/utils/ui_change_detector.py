"""
UI Change Detector for multi-step primitive execution.

Uses:
- a cheap global grayscale diff
- an action-local diff around the last interaction point

This helps distinguish real progress from click misses on dense UI panels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from civStation.utils.llm_provider.parser import AgentAction

logger = logging.getLogger(__name__)

_GLOBAL_COMPARE_SIZE = (96, 96)
_LOCAL_COMPARE_SIZE = (64, 64)
_GLOBAL_DIFF_THRESHOLD = 0.02
_LOCAL_DIFF_THRESHOLD = 0.05
_LEGACY_SIMILARITY_THRESHOLD = 0.95


def _normalized_mean_abs_diff(img1: Image.Image, img2: Image.Image, size: tuple[int, int]) -> float:
    gray1 = img1.convert("L").resize(size)
    gray2 = img2.convert("L").resize(size)
    pixels1 = gray1.tobytes()
    pixels2 = gray2.tobytes()
    if not pixels1:
        return 0.0
    total = sum(abs(a - b) for a, b in zip(pixels1, pixels2, strict=True))
    return total / (len(pixels1) * 255.0)


def _crop_local_region(img: Image.Image, x: int, y: int, normalizing_range: int = 1000) -> Image.Image:
    width, height = img.size
    px = int((max(0, min(normalizing_range, x)) / normalizing_range) * width)
    py = int((max(0, min(normalizing_range, y)) / normalizing_range) * height)

    half_w = max(32, int(width * 0.12))
    half_h = max(32, int(height * 0.12))
    left = max(0, px - half_w)
    top = max(0, py - half_h)
    right = min(width, px + half_w)
    bottom = min(height, py + half_h)
    return img.crop((left, top, right, bottom))


def ui_changed(
    img1: Image.Image,
    img2: Image.Image,
    *,
    action: AgentAction | None = None,
    normalizing_range: int = 1000,
    global_threshold: float = _GLOBAL_DIFF_THRESHOLD,
    local_threshold: float = _LOCAL_DIFF_THRESHOLD,
) -> bool:
    """Return True when there is a meaningful UI change between two screenshots."""
    try:
        global_diff = _normalized_mean_abs_diff(img1, img2, _GLOBAL_COMPARE_SIZE)
        local_diff = 0.0

        if action is not None and action.action in {"click", "double_click", "drag", "scroll"}:
            crop1 = _crop_local_region(img1, action.x, action.y, normalizing_range=normalizing_range)
            crop2 = _crop_local_region(img2, action.x, action.y, normalizing_range=normalizing_range)
            local_diff = _normalized_mean_abs_diff(crop1, crop2, _LOCAL_COMPARE_SIZE)

        changed = global_diff >= global_threshold or local_diff >= local_threshold
        logger.debug(
            "UI change diff: global=%.4f local=%.4f -> changed=%s",
            global_diff,
            local_diff,
            changed,
        )
        return changed
    except Exception as exc:  # noqa: BLE001
        logger.warning("UI change detection failed: %s", exc)
        return True


def screenshots_similar(
    img1: Image.Image,
    img2: Image.Image,
    threshold: float = _LEGACY_SIMILARITY_THRESHOLD,
    *,
    action: AgentAction | None = None,
    normalizing_range: int = 1000,
) -> bool:
    """Backward-compatible wrapper returning True when no meaningful change occurred."""
    if threshold != _LEGACY_SIMILARITY_THRESHOLD and action is None:
        diff = _normalized_mean_abs_diff(img1, img2, _GLOBAL_COMPARE_SIZE)
        similarity = 1.0 - diff
        logger.debug("Legacy screenshot similarity: %.4f (threshold=%.4f)", similarity, threshold)
        return similarity >= threshold
    return not ui_changed(img1, img2, action=action, normalizing_range=normalizing_range)
