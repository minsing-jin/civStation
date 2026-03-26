"""
Helpers for translating computer-use tool payloads into AgentAction objects.
"""

from __future__ import annotations

from collections.abc import Sequence

from civStation.utils.llm_provider.parser import AgentAction


def _field(obj, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalize_coordinate(value: int, extent: int, normalizing_range: int) -> int:
    if extent <= 0:
        raise ValueError("extent must be > 0")
    clamped = max(0, min(int(value), extent))
    return round((clamped / extent) * normalizing_range)


def _normalize_point(
    x: int,
    y: int,
    *,
    image_size: tuple[int, int],
    normalizing_range: int,
) -> tuple[int, int]:
    width, height = image_size
    return (
        _normalize_coordinate(x, width, normalizing_range),
        _normalize_coordinate(y, height, normalizing_range),
    )


def _normalize_keys(keys: Sequence[str]) -> str:
    cleaned = [str(key).strip().lower() for key in keys if str(key).strip()]
    return "+".join(cleaned)


def _coordinate_pair(raw) -> tuple[int, int]:
    if isinstance(raw, Sequence) and not isinstance(raw, str) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    raise ValueError(f"Invalid coordinate payload: {raw!r}")


def openai_tool_action_to_agent_action(
    action,
    *,
    image_size: tuple[int, int],
    normalizing_range: int,
) -> AgentAction | None:
    action_type = _field(action, "type", "")

    if action_type in {"screenshot", "wait", ""}:
        return None

    if action_type in {"click", "double_click", "move"}:
        x, y = _normalize_point(
            int(_field(action, "x", 0)),
            int(_field(action, "y", 0)),
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        button = str(_field(action, "button", "left") or "left").lower()
        if button not in {"left", "right"}:
            button = "left"
        return AgentAction(
            action=action_type,
            coord_space="normalized",
            x=x,
            y=y,
            button=button,
            reasoning="OpenAI computer-use action",
        )

    if action_type == "drag":
        path = list(_field(action, "path", []) or [])
        if len(path) < 2:
            raise ValueError("OpenAI drag action must include at least two path points")
        start_x, start_y = _normalize_point(
            int(_field(path[0], "x", 0)),
            int(_field(path[0], "y", 0)),
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        end_x, end_y = _normalize_point(
            int(_field(path[-1], "x", 0)),
            int(_field(path[-1], "y", 0)),
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        return AgentAction(
            action="drag",
            coord_space="normalized",
            x=start_x,
            y=start_y,
            end_x=end_x,
            end_y=end_y,
            button="left",
            reasoning="OpenAI computer-use drag action",
        )

    if action_type == "scroll":
        x, y = _normalize_point(
            int(_field(action, "x", 0)),
            int(_field(action, "y", 0)),
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        scroll_amount = int(_field(action, "scroll_y", 0) or _field(action, "scroll_x", 0))
        if scroll_amount == 0:
            return None
        return AgentAction(
            action="scroll",
            coord_space="normalized",
            x=x,
            y=y,
            scroll_amount=scroll_amount,
            reasoning="OpenAI computer-use scroll action",
        )

    if action_type == "keypress":
        keys = _field(action, "keys", []) or []
        key = _normalize_keys(keys)
        if not key:
            return None
        return AgentAction(
            action="press",
            key=key,
            reasoning="OpenAI computer-use keypress action",
        )

    if action_type == "type":
        text = str(_field(action, "text", "") or "")
        if not text:
            return None
        return AgentAction(
            action="type",
            text=text,
            reasoning="OpenAI computer-use type action",
        )

    raise ValueError(f"Unsupported OpenAI computer-use action: {action_type}")


def anthropic_tool_input_to_agent_action(
    payload,
    *,
    image_size: tuple[int, int],
    normalizing_range: int,
) -> AgentAction | None:
    action_type = str(_field(payload, "action", "") or "")

    if action_type in {"screenshot", "wait", ""}:
        return None

    click_map = {
        "left_click": ("click", "left"),
        "right_click": ("click", "right"),
        "middle_click": ("click", "left"),
        "double_click": ("double_click", "left"),
        "triple_click": ("double_click", "left"),
        "mouse_move": ("move", "left"),
    }
    if action_type in click_map:
        raw_x, raw_y = _coordinate_pair(_field(payload, "coordinate"))
        x, y = _normalize_point(
            raw_x,
            raw_y,
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        mapped_action, button = click_map[action_type]
        return AgentAction(
            action=mapped_action,
            coord_space="normalized",
            x=x,
            y=y,
            button=button,
            reasoning="Anthropic computer-use action",
        )

    if action_type == "left_click_drag":
        start_raw = _field(payload, "start_coordinate") or _field(payload, "from_coordinate")
        end_raw = _field(payload, "coordinate") or _field(payload, "to_coordinate")
        if start_raw is None or end_raw is None:
            raise ValueError("Anthropic left_click_drag requires start and end coordinates")
        start_x, start_y = _coordinate_pair(start_raw)
        end_x, end_y = _coordinate_pair(end_raw)
        norm_start_x, norm_start_y = _normalize_point(
            start_x,
            start_y,
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        norm_end_x, norm_end_y = _normalize_point(
            end_x,
            end_y,
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        return AgentAction(
            action="drag",
            coord_space="normalized",
            x=norm_start_x,
            y=norm_start_y,
            end_x=norm_end_x,
            end_y=norm_end_y,
            button="left",
            reasoning="Anthropic computer-use drag action",
        )

    if action_type == "scroll":
        raw_x, raw_y = _coordinate_pair(_field(payload, "coordinate", [0, 0]))
        x, y = _normalize_point(
            raw_x,
            raw_y,
            image_size=image_size,
            normalizing_range=normalizing_range,
        )
        amount = int(_field(payload, "scroll_amount", 0) or _field(payload, "amount", 0))
        direction = str(_field(payload, "scroll_direction", "") or _field(payload, "direction", "")).lower()
        if direction == "down":
            amount = -abs(amount)
        elif direction == "up":
            amount = abs(amount)
        if amount == 0:
            return None
        return AgentAction(
            action="scroll",
            coord_space="normalized",
            x=x,
            y=y,
            scroll_amount=amount,
            reasoning="Anthropic computer-use scroll action",
        )

    if action_type == "key":
        text = str(_field(payload, "text", "") or _field(payload, "key", "") or "").strip().lower()
        if not text:
            return None
        return AgentAction(
            action="press",
            key=text,
            reasoning="Anthropic computer-use key action",
        )

    if action_type == "type":
        text = str(_field(payload, "text", "") or "")
        if not text:
            return None
        return AgentAction(
            action="type",
            text=text,
            reasoning="Anthropic computer-use type action",
        )

    raise ValueError(f"Unsupported Anthropic computer-use action: {action_type}")
