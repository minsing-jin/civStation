from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints


def serialize_value(value: Any) -> Any:
    """Recursively serialize dataclasses, enums, and datetimes into JSON-safe values."""
    if is_dataclass(value):
        return {field.name: serialize_value(getattr(value, field.name)) for field in fields(type(value))}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [serialize_value(item) for item in value]
    if isinstance(value, set):
        return [serialize_value(item) for item in sorted(value, key=str)]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    return value


def deserialize_value(expected_type: Any, value: Any) -> Any:
    """Recursively hydrate JSON-safe values back into Python dataclasses."""
    if expected_type in (Any, object, None):
        return value
    if value is None:
        return None

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin in (list, list[Any]):
        item_type = args[0] if args else Any
        return [deserialize_value(item_type, item) for item in value]
    if origin in (set, set[Any]):
        item_type = args[0] if args else Any
        return {deserialize_value(item_type, item) for item in value}
    if origin in (tuple, tuple[Any, ...]):
        item_type = args[0] if args else Any
        return tuple(deserialize_value(item_type, item) for item in value)
    if origin in (dict, dict[Any, Any]):
        key_type = args[0] if len(args) > 0 else Any
        item_type = args[1] if len(args) > 1 else Any
        return {deserialize_value(key_type, key): deserialize_value(item_type, item) for key, item in value.items()}
    if origin in (UnionType, getattr(__import__("typing"), "Union", object)):
        non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none_args) == 1:
            return deserialize_value(non_none_args[0], value)
        for candidate in non_none_args:
            try:
                return deserialize_value(candidate, value)
            except Exception:  # noqa: BLE001
                continue
        return value

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return expected_type(value)
    if expected_type is datetime:
        return datetime.fromisoformat(value)
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        type_hints = get_type_hints(expected_type)
        kwargs = {}
        for field in fields(expected_type):
            if field.name not in value:
                continue
            field_type = type_hints.get(field.name, field.type)
            kwargs[field.name] = deserialize_value(field_type, value[field.name])
        return expected_type(**kwargs)
    return value


def patch_dataclass(target: Any, patch: dict[str, Any]) -> Any:
    """Apply a recursive patch to a dataclass instance."""
    if not is_dataclass(target):
        raise TypeError(f"patch_dataclass requires a dataclass instance, got {type(target)!r}")

    type_hints = get_type_hints(type(target))
    for field_name, patch_value in patch.items():
        if not hasattr(target, field_name):
            continue
        current_value = getattr(target, field_name)
        field_type = type_hints.get(field_name, type(current_value))

        if is_dataclass(current_value) and isinstance(patch_value, dict):
            patch_dataclass(current_value, patch_value)
            continue
        if isinstance(current_value, dict) and isinstance(patch_value, dict):
            merged = dict(current_value)
            merged.update(patch_value)
            setattr(target, field_name, merged)
            continue
        setattr(target, field_name, deserialize_value(field_type, patch_value))
    return target
