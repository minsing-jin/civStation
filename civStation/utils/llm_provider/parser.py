"""
Response parsing and validation for VLM outputs.

Pure functions that transform VLM response text into structured objects.
No provider state required — only text in, structured data out.
"""

import json
import logging
import re
from dataclasses import dataclass

from civStation.agent.models.schema import AgentPlan, ClickAction, DragAction, KeyPressAction

logger = logging.getLogger(__name__)


@dataclass
class AgentAction:
    """
    Single action from VLM using normalized or absolute coordinates.

    Supports: click, double_click, drag, scroll, move, press, type
    """

    action: str = ""  # "click", "double_click", "drag", "scroll", "move", "press", "type"
    coord_space: str = "normalized"  # "normalized" or "absolute"
    x: int = 0  # X coordinate in coord_space
    y: int = 0  # Y coordinate in coord_space
    end_x: int = 0  # Drag end x in coord_space
    end_y: int = 0  # Drag end y in coord_space
    scroll_amount: int = 0  # Mouse wheel delta. Positive=up, negative=down.
    button: str = "left"  # "left" or "right"
    key: str = ""  # Key name for "press"
    text: str = ""  # Text for "type"
    reasoning: str = ""
    task_status: str = ""  # "in_progress", "complete", or "" (single-step)
    policy_card_name: str = ""
    policy_source_tab: str = ""
    policy_target_slot_id: str = ""
    policy_reasoning: str = ""


def strip_markdown(text: str) -> str:
    """
    Strip markdown code block wrappers from response text.

    Handles various markdown formats:
    - ```json...``` (whole response is a code block)
    - Preamble text + ```json...``` (VLM adds reasoning before JSON)
    - ```...```
    - Multiple closing ```
    - Extra whitespace
    """
    content = text.strip()

    # Remove opening fence (response starts with code block)
    if content.startswith("```json"):
        content = content[7:].lstrip()
    elif content.startswith("```"):
        content = content[3:].lstrip()
    else:
        # Preamble text before code block — extract the code block content
        fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", content)
        if fence_match:
            content = fence_match.group(1).strip()
            return content

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
    if action.coord_space not in ("normalized", "absolute"):
        errors.append(f"coord_space='{action.coord_space}' is invalid, must be 'normalized' or 'absolute'")

    # --- Required fields per action type ---
    if action.action in ("click", "double_click", "move"):
        if not (isinstance(action.x, int) and isinstance(action.y, int)):
            errors.append(
                f"{action.action} requires int x,y, got x={type(action.x).__name__}, y={type(action.y).__name__}"
            )

    elif action.action == "drag":
        if not all(isinstance(v, int) for v in (action.x, action.y, action.end_x, action.end_y)):
            errors.append("drag requires int x, y, end_x, end_y")
        if action.x == action.end_x and action.y == action.end_y:
            errors.append(f"drag start==end ({action.x},{action.y}), no movement")

    elif action.action == "scroll":
        if not isinstance(action.scroll_amount, int):
            errors.append("scroll requires int scroll_amount")
        elif action.scroll_amount == 0:
            errors.append("scroll_amount must be non-zero")

    elif action.action == "press":
        if not action.key:
            errors.append("press action requires non-empty 'key' field")

    elif action.action == "type":
        if not action.text:
            errors.append("type action requires non-empty 'text' field")

    # --- Coordinate validation (only for actions that use coordinates) ---
    if action.action in ("click", "double_click", "drag", "scroll", "move"):
        for name, val in [("x", action.x), ("y", action.y)]:
            if action.coord_space == "normalized":
                if not (0 <= val <= normalizing_range):
                    errors.append(f"{name}={val} out of range [0, {normalizing_range}]")
            elif val < 0:
                errors.append(f"{name}={val} must be >= 0 for absolute coordinates")

    if action.action == "drag":
        for name, val in [("end_x", action.end_x), ("end_y", action.end_y)]:
            if action.coord_space == "normalized":
                if not (0 <= val <= normalizing_range):
                    errors.append(f"{name}={val} out of range [0, {normalizing_range}]")
            elif val < 0:
                errors.append(f"{name}={val} must be >= 0 for absolute coordinates")

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

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: find first JSON object by matching outermost { }
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                data = json.loads(content[first_brace : last_brace + 1])
            else:
                raise

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

        valid_actions = ["click", "double_click", "drag", "scroll", "move", "press", "type"]
        if action_value not in valid_actions:
            logger.warning(f"Parse: unknown action '{action_value}', expected one of {valid_actions}")
            return None

        # --- Required field presence before type-strict extraction ---
        required_coord_fields = {
            "click": ("x", "y"),
            "double_click": ("x", "y"),
            "scroll": ("x", "y", "scroll_amount"),
            "move": ("x", "y"),
            "drag": ("x", "y", "end_x", "end_y"),
        }
        missing_fields = [field for field in required_coord_fields.get(action_value, ()) if field not in data]
        if missing_fields:
            logger.warning(
                "Parse: missing required field(s) for %s: %s | payload=%s",
                action_value,
                ", ".join(missing_fields),
                data,
            )
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
        scroll_amount = _parse_int("scroll_amount")
        coord_space = _parse_str("coord_space", "normalized")
        button = _parse_str("button", "left")
        key = _parse_str("key")
        text = _parse_str("text")
        reasoning = _parse_str("reasoning")
        task_status = _parse_str("task_status")
        policy_card_name = _parse_str("policy_card_name")
        policy_source_tab = _parse_str("policy_source_tab")
        policy_target_slot_id = _parse_str("policy_target_slot_id")
        policy_reasoning = _parse_str("policy_reasoning")

        if parse_errors:
            for err in parse_errors:
                logger.warning(f"Parse type error: {err}")
            logger.warning(f"Parse: returning None due to type errors in: {data}")
            return None

        agent_action = AgentAction(
            action=action_value,
            coord_space=coord_space,
            x=x,
            y=y,
            end_x=end_x,
            end_y=end_y,
            scroll_amount=scroll_amount,
            button=button,
            key=key,
            text=text,
            reasoning=reasoning,
            task_status=task_status,
            policy_card_name=policy_card_name,
            policy_source_tab=policy_source_tab,
            policy_target_slot_id=policy_target_slot_id,
            policy_reasoning=policy_reasoning,
        )

        logger.debug(f"Successfully parsed action: {agent_action}")
        return agent_action

    except json.JSONDecodeError as e:
        logger.warning(f"Parse: invalid JSON: {e}")
        logger.debug(f"Raw: {response_text}\nStripped: {content}")
        return None


def parse_action_json_list(response_text: str) -> list[AgentAction] | None:
    """
    Parse VLM response into a list of AgentActions (for multi-action primitives).

    Accepts either a JSON array of action objects or a single action object
    (which is wrapped into a one-element list).

    Returns:
        List of AgentAction objects. Returns an empty list for a valid no-op `[]`
        response, or None on parse failure.
    """
    content = ""
    try:
        content = strip_markdown(response_text)
        logger.debug(f"Stripped content for multi-action parsing:\n{content}")

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: find JSON array [...] or object {...}
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                first = content.find(start_char)
                last = content.rfind(end_char)
                if first != -1 and last > first:
                    try:
                        data = json.loads(content[first : last + 1])
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise

        # Single object → wrap in list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            logger.warning(f"Parse multi: expected list or dict, got {type(data).__name__}")
            return None

        if not data:
            return []

        actions: list[AgentAction] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                logger.warning(f"Parse multi: item {i} is not a dict, skipping")
                continue

            action = parse_action_json(json.dumps(item))
            if action is not None:
                actions.append(action)
            else:
                logger.warning(f"Parse multi: item {i} failed to parse, skipping")

        if not actions:
            logger.warning("Parse multi: no valid actions parsed from list")
            return None

        return actions

    except json.JSONDecodeError as e:
        logger.warning(f"Parse multi: invalid JSON: {e}")
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
