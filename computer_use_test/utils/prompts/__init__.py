"""
Prompts package for VLM interactions.

This package contains reusable prompts for different game primitives
and scenarios in Civilization VI.
"""

from computer_use_test.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DIPLOMATIC_PROMPT,
    POPUP_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
    build_custom_prompt,
    get_primitive_prompt,
)
from computer_use_test.utils.prompts.router_prompt import ROUTER_PROMPT

__all__ = [
    "UNIT_OPS_PROMPT",
    "POPUP_PROMPT",
    "RESEARCH_MANAGER_PROMPT",
    "CITY_PRODUCTION_PROMPT",
    "CULTURE_MANAGER_PROMPT",
    "DIPLOMATIC_PROMPT",
    "COMBAT_PROMPT",
    "ROUTER_PROMPT",
    "get_primitive_prompt",
    "build_custom_prompt",
]
