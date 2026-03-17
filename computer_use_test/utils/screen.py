"""
Screen capture and action execution utilities.

Handles the Mac Retina display coordinate mismatch by using
normalized coordinates (0-1000) between the VLM and PyAutoGUI.

Flow:
1. capture_screen_pil() → PIL image + logical resolution
2. VLM analyzes image → returns normalized coords (0-1000)
3. execute_action() → converts normalized → logical → PyAutoGUI

When the game runs in windowed mode, capture_screen_pil() automatically
detects the game window via macOS Quartz and crops the screenshot to
just the game area, eliminating desktop noise (Dock, menu bar, etc.).
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from PIL import Image

from computer_use_test.utils.llm_provider.base import AgentAction

logger = logging.getLogger(__name__)

# Default VLM image settings — trade-off between quality and token cost.
# 1280px long edge keeps game-UI text readable while cutting image tokens ~75%.
VLM_MAX_LONG_EDGE: int = 1280
VLM_JPEG_QUALITY: int = 80

# Keywords to match the Civilization VI window (case-insensitive)
_GAME_WINDOW_KEYWORDS = ("civilization", "civ6", "civ vi")
_SCROLL_HOVER_SETTLE_SECONDS = 0.18


def _unavailable_pyautogui_method(*args, **kwargs):
    raise RuntimeError("pyautogui is unavailable in this environment")


try:
    import pyautogui as _pyautogui
except Exception:  # pragma: no cover - depends on runtime display availability
    pyautogui = SimpleNamespace(
        size=_unavailable_pyautogui_method,
        screenshot=_unavailable_pyautogui_method,
        moveTo=_unavailable_pyautogui_method,
        click=_unavailable_pyautogui_method,
        doubleClick=_unavailable_pyautogui_method,
        mouseDown=_unavailable_pyautogui_method,
        mouseUp=_unavailable_pyautogui_method,
        scroll=_unavailable_pyautogui_method,
        press=_unavailable_pyautogui_method,
        write=_unavailable_pyautogui_method,
    )
else:
    pyautogui = _pyautogui


def _detect_game_window() -> tuple[int, int, int, int] | None:
    """Detect the Civilization VI game window bounds (logical coords).

    Uses macOS Quartz CGWindowList API. Returns None on non-macOS
    platforms or when the game window is not found.

    Returns:
        (x, y, width, height) in logical screen coordinates, or None.
    """
    try:
        import Quartz
    except ImportError:
        return None

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    if not windows:
        return None

    for w in windows:
        name = (w.get("kCGWindowName") or "").lower()
        owner = (w.get("kCGWindowOwnerName") or "").lower()
        combined = f"{name} {owner}"

        if any(kw in combined for kw in _GAME_WINDOW_KEYWORDS):
            bounds = w.get("kCGWindowBounds")
            if bounds:
                gw, gh = int(bounds["Width"]), int(bounds["Height"])
                # Skip tiny windows (e.g. helper processes)
                if gw < 400 or gh < 300:
                    continue
                return (int(bounds["X"]), int(bounds["Y"]), gw, gh)

    return None


def capture_screen_pil(crop_to_game: bool = True):
    """
    Capture current screen as PIL image, optionally cropped to the game window.

    When crop_to_game is True and the game window is detected, the screenshot
    is cropped to just the game area. Normalized coordinates from the VLM
    will then map to the game window, and execute_action() uses the returned
    offsets to convert back to absolute screen coordinates.

    Returns:
        Tuple of (pil_image, region_w, region_h, x_offset, y_offset)
        - pil_image: PIL.Image (cropped to game window if detected)
        - region_w: Logical width of the captured region
        - region_h: Logical height of the captured region
        - x_offset: Logical X offset of the region on screen
        - y_offset: Logical Y offset of the region on screen
    """
    # PyAutoGUI size returns logical resolution (not physical/retina)
    screen_w, screen_h = pyautogui.size()

    # Capture screenshot (returns PIL Image)
    screenshot = pyautogui.screenshot()

    x_offset, y_offset = 0, 0
    region_w, region_h = screen_w, screen_h

    if crop_to_game:
        bounds = _detect_game_window()
        if bounds:
            gx, gy, gw, gh = bounds
            # Retina scale factor: screenshot pixels vs logical pixels
            scale = screenshot.size[0] / screen_w
            crop_box = (
                int(gx * scale),
                int(gy * scale),
                int((gx + gw) * scale),
                int((gy + gh) * scale),
            )
            screenshot = screenshot.crop(crop_box)
            x_offset, y_offset = gx, gy
            region_w, region_h = gw, gh
            logger.debug(f"Cropped to game window: {gw}x{gh} at ({gx},{gy})")

    return screenshot, region_w, region_h, x_offset, y_offset


def resize_for_vlm(
    pil_image: Image.Image,
    max_long_edge: int = VLM_MAX_LONG_EDGE,
) -> Image.Image:
    """Downscale a screenshot so the VLM processes fewer image tokens.

    The aspect ratio is preserved. If the image is already small enough
    it is returned unchanged.

    Args:
        pil_image: Original PIL image (e.g. from ``capture_screen_pil``).
        max_long_edge: Maximum pixels on the longer edge (default 1280).

    Returns:
        A (possibly resized) PIL image in RGB mode.
    """
    w, h = pil_image.size
    long_edge = max(w, h)

    if long_edge <= max_long_edge:
        return pil_image.convert("RGB")

    scale = max_long_edge / long_edge
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"Resized screenshot for VLM: {w}x{h} → {new_w}x{new_h}")
    return resized.convert("RGB")


def norm_to_real(norm_val: int, screen_size: int, normalizing_range: int = 1000) -> int:
    """Convert a normalized coordinate to real screen coordinate."""
    clamped = max(0, min(normalizing_range, norm_val))
    return round((clamped / normalizing_range) * screen_size)


def execute_action(
    action: AgentAction | None,
    screen_w: int,
    screen_h: int,
    normalizing_range: int = 1000,
    x_offset: int = 0,
    y_offset: int = 0,
) -> None:
    """
    Execute an AgentAction by converting normalized coordinates to screen coordinates.

    Normalized coords (0-1000) → region coords (0-screen_w) → absolute screen coords (+offset)
    Absolute coords bypass the normalization step and are treated as logical
    monitor coordinates directly.

    When the screenshot was cropped to a game window, x_offset/y_offset shift
    the coordinates back to absolute screen positions for PyAutoGUI.

    Args:
        action: AgentAction with normalized or absolute coordinates
        screen_w: Logical width of the captured region (game window or full screen)
        screen_h: Logical height of the captured region
        normalizing_range: Coordinate range used in the prompt (default: 1000)
        x_offset: Logical X offset of the captured region on screen (default: 0)
        y_offset: Logical Y offset of the captured region on screen (default: 0)
    """
    if action is None:
        logger.warning("No action to execute (None)")
        return

    # Handle list input (legacy compatibility)
    if isinstance(action, list):
        if len(action) == 0:
            logger.warning("Empty action list")
            return
        logger.debug(f"List input detected, using first of {len(action)} items")
        action = action[0]

    # Handle dict input (legacy compatibility)
    if isinstance(action, dict):
        action = AgentAction(**action)

    action_type = action.action
    reasoning = action.reasoning
    logger.debug(f"Action: {action_type} | Reasoning: {reasoning}")

    # Validate action type
    if not action_type or action_type not in ["click", "double_click", "drag", "scroll", "move", "press", "type"]:
        logger.error(f"Invalid or empty action type: '{action_type}'")
        logger.error(f"Full action object: {action}")
        logger.error("Valid action types are: click, double_click, drag, scroll, move, press, type")
        return

    coord_space = getattr(action, "coord_space", "normalized") or "normalized"

    if action_type == "click":
        if coord_space == "absolute":
            real_x = action.x
            real_y = action.y
            logger.debug(
                "Click: absolute(%s, %s) -> real(%s, %s) - %s",
                action.x,
                action.y,
                real_x,
                real_y,
                action.button,
            )
        else:
            real_x = norm_to_real(action.x, screen_w, normalizing_range) + x_offset
            real_y = norm_to_real(action.y, screen_h, normalizing_range) + y_offset
            logger.debug(
                "Click: normalized(%s, %s) -> real(%s, %s) - %s",
                action.x,
                action.y,
                real_x,
                real_y,
                action.button,
            )
        pyautogui.moveTo(real_x, real_y, duration=0.5)
        pyautogui.click(button=action.button)

    elif action_type == "double_click":
        if coord_space == "absolute":
            real_x = action.x
            real_y = action.y
            logger.debug(
                "Double-click: absolute(%s, %s) -> real(%s, %s) - %s",
                action.x,
                action.y,
                real_x,
                real_y,
                action.button,
            )
        else:
            real_x = norm_to_real(action.x, screen_w, normalizing_range) + x_offset
            real_y = norm_to_real(action.y, screen_h, normalizing_range) + y_offset
            logger.debug(
                "Double-click: normalized(%s, %s) -> real(%s, %s) - %s",
                action.x,
                action.y,
                real_x,
                real_y,
                action.button,
            )
        pyautogui.moveTo(real_x, real_y, duration=0.5)
        pyautogui.doubleClick(button=action.button)

    elif action_type == "drag":
        if coord_space == "absolute":
            start_x = action.x
            start_y = action.y
            end_x = action.end_x
            end_y = action.end_y
        else:
            start_x = norm_to_real(action.x, screen_w, normalizing_range) + x_offset
            start_y = norm_to_real(action.y, screen_h, normalizing_range) + y_offset
            end_x = norm_to_real(action.end_x, screen_w, normalizing_range) + x_offset
            end_y = norm_to_real(action.end_y, screen_h, normalizing_range) + y_offset

        logger.debug(
            "Drag: %s(%s,%s)->(%s,%s) real (%s,%s)->(%s,%s)",
            coord_space,
            action.x,
            action.y,
            action.end_x,
            action.end_y,
            start_x,
            start_y,
            end_x,
            end_y,
        )

        # Move to start position
        pyautogui.moveTo(start_x, start_y, duration=0.3)
        # Hold mouse button down
        pyautogui.mouseDown(button=action.button)
        # Move to end position while holding
        pyautogui.moveTo(end_x, end_y, duration=0.5)
        # Release mouse button
        pyautogui.mouseUp(button=action.button)

    elif action_type == "scroll":
        if coord_space == "absolute":
            real_x = action.x
            real_y = action.y
        else:
            real_x = norm_to_real(action.x, screen_w, normalizing_range) + x_offset
            real_y = norm_to_real(action.y, screen_h, normalizing_range) + y_offset
        logger.debug(
            "Scroll: %s(%s, %s) -> real(%s, %s) amount=%s",
            coord_space,
            action.x,
            action.y,
            real_x,
            real_y,
            action.scroll_amount,
        )
        pyautogui.moveTo(real_x, real_y, duration=0.2)
        time.sleep(_SCROLL_HOVER_SETTLE_SECONDS)
        pyautogui.scroll(action.scroll_amount)

    elif action_type == "move":
        if coord_space == "absolute":
            real_x = action.x
            real_y = action.y
        else:
            real_x = norm_to_real(action.x, screen_w, normalizing_range) + x_offset
            real_y = norm_to_real(action.y, screen_h, normalizing_range) + y_offset
        logger.debug(
            "Move: %s(%s, %s) -> real(%s, %s)",
            coord_space,
            action.x,
            action.y,
            real_x,
            real_y,
        )
        pyautogui.moveTo(real_x, real_y, duration=0.2)

    elif action_type == "press":
        key = action.key
        if key:
            logger.debug(f"Press key: {key}")
            pyautogui.press(key)
        else:
            logger.error(f"Press action missing 'key' field. Full action: {action}")

    elif action_type == "type":
        text = action.text
        if text:
            logger.debug(f"Type text: {text}")
            pyautogui.write(text, interval=0.1)
        else:
            logger.error(f"Type action missing 'text' field. Full action: {action}")


def move_cursor_to_center(screen_w: int, screen_h: int, x_offset: int = 0, y_offset: int = 0) -> None:
    """Move cursor to screen center after click to prevent hover tooltips."""
    center_x = screen_w // 2 + x_offset
    center_y = screen_h // 2 + y_offset
    pyautogui.moveTo(center_x, center_y, duration=0.1)


def agent_loop(
    provider,
    instruction: str,
    max_steps: int = 10,
    normalizing_range: int = 1000,
) -> None:
    """
    Run agent loop: capture → analyze → execute → repeat.

    Args:
        provider: BaseVLMProvider instance (claude, gemini, gpt)
        instruction: User goal / task description
        max_steps: Maximum number of steps to execute
        normalizing_range: Coordinate normalization range
    """
    logger.info(f"Agent started ({provider.get_provider_name()})")
    logger.info(f"Instruction: {instruction}")

    for step in range(1, max_steps + 1):
        logger.info(f"\n--- Step {step}/{max_steps} ---")

        # 1. Capture screen
        pil_image, screen_w, screen_h, x_off, y_off = capture_screen_pil()
        logger.info(f"Screen captured: {screen_w}x{screen_h} (logical, offset=({x_off},{y_off}))")

        # 2. Analyze with VLM
        logger.info("Analyzing...")
        action = provider.analyze(
            pil_image=pil_image,
            instruction=instruction,
            normalizing_range=normalizing_range,
        )

        if action is None:
            logger.warning("VLM returned no action, stopping")
            break

        # 3. Execute action
        execute_action(action, screen_w, screen_h, normalizing_range, x_off, y_off)

    logger.info("Agent loop completed")
