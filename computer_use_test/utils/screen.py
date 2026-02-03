"""
Screen capture and action execution utilities.

Handles the Mac Retina display coordinate mismatch by using
normalized coordinates (0-1000) between the VLM and PyAutoGUI.

Flow:
1. capture_screen_pil() → PIL image + logical resolution
2. VLM analyzes image → returns normalized coords (0-1000)
3. execute_action() → converts normalized → logical → PyAutoGUI
"""

import logging

import pyautogui

from computer_use_test.utils.provider.base import AgentAction

logger = logging.getLogger(__name__)


def capture_screen_pil():
    """
    Capture current screen as PIL image.

    Returns the logical resolution (what PyAutoGUI uses) so that
    normalized coordinates can be correctly mapped.

    Returns:
        Tuple of (pil_image, screen_width, screen_height)
        - pil_image: PIL.Image of the screenshot
        - screen_width: Logical width (PyAutoGUI coordinate space)
        - screen_height: Logical height (PyAutoGUI coordinate space)
    """
    # PyAutoGUI size returns logical resolution (not physical/retina)
    screen_w, screen_h = pyautogui.size()

    # Capture screenshot (returns PIL Image)
    screenshot = pyautogui.screenshot()

    return screenshot, screen_w, screen_h


def norm_to_real(norm_val: int, screen_size: int, normalizing_range: int) -> int:
    """Convert a normalized coordinate to real screen coordinate."""
    clamped = max(0, min(normalizing_range, norm_val))
    return int((clamped / normalizing_range) * screen_size)


def execute_action(
    action: AgentAction | None,
    screen_w: int,
    screen_h: int,
    normalizing_range: int = 1000,
) -> None:
    """
    Execute an AgentAction by converting normalized coordinates to screen coordinates.

    Normalized coords (0-1000) → Real screen coords (0-screen_w, 0-screen_h)

    This solves the Mac Retina display mismatch where:
    - Screenshot pixel coords ≠ PyAutoGUI coords
    - Screenshot is 2x the logical resolution on Retina displays

    Args:
        action: AgentAction with normalized coordinates (0-{normalizing_range})
        screen_w: Logical screen width (from pyautogui.size())
        screen_h: Logical screen height (from pyautogui.size())
        normalizing_range: Coordinate range used in the prompt (default: 1000)
    """
    if action is None:
        logger.warning("No action to execute (None)")
        return

    # Handle list input (legacy compatibility)
    if isinstance(action, list):
        if len(action) == 0:
            logger.warning("Empty action list")
            return
        logger.info(f"List input detected, using first of {len(action)} items")
        action = action[0]

    # Handle dict input (legacy compatibility)
    if isinstance(action, dict):
        action = AgentAction(**action)

    action_type = action.action
    reasoning = action.reasoning
    logger.info(f"Action: {action_type} | Reasoning: {reasoning}")

    if action_type == "click":
        real_x = norm_to_real(action.x, screen_w)
        real_y = norm_to_real(action.y, screen_h)

        logger.info(f"Click: normalized({action.x}, {action.y}) → real({real_x}, {real_y}) - {action.button}")
        pyautogui.moveTo(real_x, real_y, duration=0.5)
        pyautogui.click(button=action.button)

    elif action_type == "double_click":
        real_x = norm_to_real(action.x, screen_w)
        real_y = norm_to_real(action.y, screen_h)

        logger.info(f"Double-click: normalized({action.x}, {action.y}) → real({real_x}, {real_y}) - {action.button}")
        pyautogui.moveTo(real_x, real_y, duration=0.5)
        pyautogui.doubleClick(button=action.button)

    elif action_type == "drag":
        start_x = norm_to_real(action.x, screen_w)
        start_y = norm_to_real(action.y, screen_h)
        end_x = norm_to_real(action.end_x, screen_w)
        end_y = norm_to_real(action.end_y, screen_h)

        logger.info(
            f"Drag: ({action.x},{action.y})→({action.end_x},{action.end_y}) "
            f"real ({start_x},{start_y})→({end_x},{end_y})"
        )

        # Move to start position
        pyautogui.moveTo(start_x, start_y, duration=0.3)
        # Hold mouse button down
        pyautogui.mouseDown(button=action.button)
        # Move to end position while holding
        pyautogui.moveTo(end_x, end_y, duration=0.5)
        # Release mouse button
        pyautogui.mouseUp(button=action.button)

    elif action_type == "press":
        key = action.key
        if key:
            logger.info(f"Press key: {key}")
            pyautogui.press(key)

    elif action_type == "type":
        text = action.text
        if text:
            logger.info(f"Type text: {text}")
            pyautogui.write(text, interval=0.1)

    else:
        logger.warning(f"Unknown action type: {action_type}")


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
        pil_image, screen_w, screen_h = capture_screen_pil()
        logger.info(f"Screen captured: {screen_w}x{screen_h} (logical)")

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
        execute_action(action, screen_w, screen_h, normalizing_range)

    logger.info("Agent loop completed")
