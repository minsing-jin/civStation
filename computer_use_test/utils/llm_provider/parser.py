"""
Response parsing and validation for VLM outputs.

Pure functions that transform VLM response text into structured objects.
No provider state required — only text in, structured data out.
"""

import json
import logging
import re
from dataclasses import dataclass

from computer_use_test.agent.models.schema import AgentPlan, ClickAction, DragAction, KeyPressAction

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """
    Single action from VLM using normalized coordinates (0-1000).

    Supports: click, double_click, drag, press, type
    """

    action: str = ""  # "click", "double_click", "drag", "press", "type"
    x: int = 0  # Normalized x (0-1000)
    y: int = 0  # Normalized y (0-1000)
    end_x: int = 0  # Drag end x (0-1000)
    end_y: int = 0  # Drag end y (0-1000)
    button: str = "left"  # "left" or "right"
    key: str = ""  # Key name for "press"
    text: str = ""  # Text for "type"
    reasoning: str = ""


def strip_markdown(text: str) -> str:
    """
    Strip markdown code block wrappers from response text.

    Handles various markdown formats:
    - ```json...```
    - ```...```
    - Multiple closing ```
    - Extra whitespace
    """
    content = text.strip()

    # Remove opening fence
    if content.startswith("```json"):
        content = content[7:].lstrip()
    elif content.startswith("```"):
        content = content[3:].lstrip()

    # Remove all closing fences (handle multiple ``` with possible newlines)
    # Use regex to remove trailing ``` blocks
    content = re.sub(r"(\n```)+\s*$", "", content)
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def validate_action(action: AgentAction, normalizing_range: int = 1000) -> list[str]:
    """
    Validate an AgentAction for correctness.

    Checks:
    - Required fields per action type (click->x,y / press->key / type->text / drag->endpoints)
    - Coordinate range (0 <= coord <= normalizing_range)
    - Button value validity
    - Data type sanity

    Args:
        action: The parsed AgentAction to validate
        normalizing_range: Max coordinate value (default 1000)

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []

    # --- Required fields per action type ---
    if action.action in ("click", "double_click"):
        if not (isinstance(action.x, int) and isinstance(action.y, int)):
            errors.append(f"click requires int x,y, got x={type(action.x).__name__}, y={type(action.y).__name__}")

    elif action.action == "drag":
        if not all(isinstance(v, int) for v in (action.x, action.y, action.end_x, action.end_y)):
            errors.append("drag requires int x, y, end_x, end_y")
        if action.x == action.end_x and action.y == action.end_y:
            errors.append(f"drag start==end ({action.x},{action.y}), no movement")

    elif action.action == "press":
        if not action.key:
            errors.append("press action requires non-empty 'key' field")

    elif action.action == "type":
        if not action.text:
            errors.append("type action requires non-empty 'text' field")

    # --- Coordinate range (only for actions that use coordinates) ---
    if action.action in ("click", "double_click", "drag"):
        for name, val in [("x", action.x), ("y", action.y)]:
            if not (0 <= val <= normalizing_range):
                errors.append(f"{name}={val} out of range [0, {normalizing_range}]")

    if action.action == "drag":
        for name, val in [("end_x", action.end_x), ("end_y", action.end_y)]:
            if not (0 <= val <= normalizing_range):
                errors.append(f"{name}={val} out of range [0, {normalizing_range}]")

    # --- Button validation ---
    if action.action in ("click", "double_click", "drag"):
        if action.button not in ("left", "right"):
            errors.append(f"invalid button='{action.button}', must be 'left' or 'right'")

    return errors


def parse_action_json(response_text: str) -> AgentAction | None:
    """
    Parse VLM response into AgentAction (for live agent).

    Strictly validates field types instead of silently defaulting.
    Returns None on any type mismatch so the retry logic can re-attempt.
    """
    content = ""
    try:
        content = strip_markdown(response_text)
        logger.debug(f"Stripped content for parsing:\n{content}")

        data = json.loads(content)

        # Handle list response
        if isinstance(data, list):
            if not data:
                logger.warning("Parse: VLM returned empty list")
                return None
            logger.info(f"List response: using first of {len(data)} items")
            data = data[0]

        if not isinstance(data, dict):
            logger.warning(f"Parse: expected dict, got {type(data).__name__}")
            return None

        # --- action field (required for all) ---
        action_value = data.get("action", "")
        if not isinstance(action_value, str) or not action_value:
            logger.warning(f"Parse: missing or invalid 'action' field: {data.get('action')!r}")
            return None

        valid_actions = ["click", "double_click", "drag", "press", "type"]
        if action_value not in valid_actions:
            logger.warning(f"Parse: unknown action '{action_value}', expected one of {valid_actions}")
            return None

        # --- Type-strict field extraction ---
        parse_errors: list[str] = []

        def _parse_int(field: str) -> int:
            """Extract int field; log and record error on bad type."""
            raw = data.get(field)
            if raw is None:
                return 0
            if isinstance(raw, int):
                return raw
            if isinstance(raw, float) and raw == int(raw):
                return int(raw)
            if isinstance(raw, str):
                try:
                    return int(raw)
                except ValueError:
                    pass
            parse_errors.append(f"'{field}' expected int, got {type(raw).__name__}={raw!r}")
            return 0

        def _parse_str(field: str, default: str = "") -> str:
            """Extract string field; log error on bad type."""
            raw = data.get(field)
            if raw is None:
                return default
            if isinstance(raw, str):
                return raw
            parse_errors.append(f"'{field}' expected str, got {type(raw).__name__}={raw!r}")
            return default

        x = _parse_int("x")
        y = _parse_int("y")
        end_x = _parse_int("end_x")
        end_y = _parse_int("end_y")
        button = _parse_str("button", "left")
        key = _parse_str("key")
        text = _parse_str("text")
        reasoning = _parse_str("reasoning")

        if parse_errors:
            for err in parse_errors:
                logger.warning(f"Parse type error: {err}")
            logger.warning(f"Parse: returning None due to type errors in: {data}")
            return None

        agent_action = AgentAction(
            action=action_value,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            button=button,
            key=key,
            text=text,
            reasoning=reasoning,
        )

        logger.debug(f"Successfully parsed action: {agent_action}")
        return agent_action

    except json.JSONDecodeError as e:
        logger.warning(f"Parse: invalid JSON: {e}")
        logger.debug(f"Raw: {response_text}\nStripped: {content}")
        return None


def parse_to_agent_plan(response_content: str, primitive_name: str) -> AgentPlan:
    """
    Parse VLM response text into AgentPlan (for static evaluation).

    Args:
        response_content: Raw VLM response text (may include markdown fences)
        primitive_name: Name of the primitive that generated this response

    Returns:
        AgentPlan with parsed actions

    Raises:
        ValueError: If JSON parsing fails
    """
    content = strip_markdown(response_content)

    try:
        data = json.loads(content)
        actions = []

        for ad in data.get("actions", []):
            action_type = ad.get("type")
            if action_type == "click":
                actions.append(
                    ClickAction(
                        type="click",
                        x=ad["x"],
                        y=ad["y"],
                        button=ad.get("button", "left"),
                        description=ad.get("description"),
                    )
                )
            elif action_type == "press":
                actions.append(
                    KeyPressAction(
                        type="press",
                        keys=ad["keys"],
                        interval=ad.get("interval", 0.1),
                        description=ad.get("description"),
                    )
                )
            elif action_type == "drag":
                actions.append(
                    DragAction(
                        type="drag",
                        start_x=ad["start_x"],
                        start_y=ad["start_y"],
                        end_x=ad["end_x"],
                        end_y=ad["end_y"],
                        duration=ad.get("duration", 0.5),
                        button=ad.get("button", "left"),
                        description=ad.get("description"),
                    )
                )

        return AgentPlan(
            primitive_name=primitive_name,
            reasoning=data.get("reasoning", ""),
            actions=actions,
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse VLM response: {e}")
        logger.error(f"Raw response content:\n{response_content}")
        logger.error(f"After markdown stripping:\n{content}")
        raise ValueError(f"Failed to parse VLM response: {e}") from e
