"""
Prompts package for VLM interactions.

This package contains reusable prompt templates for different game primitives
and scenarios in Civilization VI.

Prompt templates live here in primitive_prompt.py.
Registry, routing, and lookup logic lives in
computer_use_test.agent.modules.router.primitive_registry.
"""

from computer_use_test.utils.prompts.primitive_prompt import (
    CITY_PRODUCTION_PROMPT,
    COMBAT_PROMPT,
    CULTURE_MANAGER_PROMPT,
    DEAL_PROMPT,
    DIPLOMATIC_PROMPT,
    ERA_PROMPT,
    GOVERNOR_PROMPT,
    POLICY_PROMPT,
    POPUP_PROMPT,
    RELIGION_PROMPT,
    RESEARCH_MANAGER_PROMPT,
    UNIT_OPS_PROMPT,
    VOTING_PROMPT,
    WAR_PROMPT,
    build_custom_prompt,
)

__all__ = [
    "UNIT_OPS_PROMPT",
    "POPUP_PROMPT",
    "RESEARCH_MANAGER_PROMPT",
    "CITY_PRODUCTION_PROMPT",
    "CULTURE_MANAGER_PROMPT",
    "DIPLOMATIC_PROMPT",
    "GOVERNOR_PROMPT",
    "COMBAT_PROMPT",
    "POLICY_PROMPT",
    "RELIGION_PROMPT",
    "WAR_PROMPT",
    "DEAL_PROMPT",
    "VOTING_PROMPT",
    "ERA_PROMPT",
    "build_custom_prompt",
]
