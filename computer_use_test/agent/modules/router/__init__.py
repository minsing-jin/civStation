"""Primitive router — classifies screenshots to select the right primitive."""

from computer_use_test.agent.modules.router.base_router import PrimitiveRouter
from computer_use_test.agent.modules.router.primitive_registry import (
    PRIMITIVE_NAMES,
    PRIMITIVE_REGISTRY,
    ROUTER_PROMPT,
    RouterResult,
    get_primitive_prompt,
)
from computer_use_test.agent.modules.router.router import Civ6MockRouter, Civ6Router

__all__ = [
    "PrimitiveRouter",
    "Civ6Router",
    "Civ6MockRouter",
    "RouterResult",
    "PRIMITIVE_NAMES",
    "PRIMITIVE_REGISTRY",
    "ROUTER_PROMPT",
    "get_primitive_prompt",
]
