"""civ6-mcp tool-call planner.

Drives an LLM (any text-capable provider that civStation already supports)
to emit a JSON tool-call sequence given:
- the player's strategy
- the current game state (rendered from civ6-mcp `get_*` output)
- a tool catalog
- recent tool-call history
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from civStation.agent.modules.backend.civ6_mcp import planner_types
from civStation.agent.modules.backend.civ6_mcp._payload import _load_json_payload, planner_tool_call_items
from civStation.agent.modules.backend.civ6_mcp.executor import coerce_tool_calls
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    normalize_raw_mcp_game_state,
)
from civStation.agent.modules.backend.civ6_mcp.operations import (
    END_TURN_REFLECTION_FIELDS,
    END_TURN_TOOL,
)
from civStation.agent.modules.backend.civ6_mcp.planner_types import (
    Civ6McpPlannerProvider,
    PlannerResult,
)
from civStation.agent.modules.backend.civ6_mcp.prompts import (
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from civStation.agent.modules.backend.civ6_mcp.results import ToolCall
from civStation.agent.modules.backend.civ6_mcp.turn_planning import build_prioritized_turn_plan

logger = logging.getLogger(__name__)

_DEFAULT_PLANNER_TOOL_ALLOWLIST = planner_types.DEFAULT_PLANNER_TOOL_ALLOWLIST
"""Private canonical planner tool allow-list for this module."""

DEFAULT_PLANNER_TOOL_ALLOWLIST = _DEFAULT_PLANNER_TOOL_ALLOWLIST
"""Public compatibility alias for the canonical planner tool allow-list."""


class MissingEndTurnPlannerOutputError(RuntimeError):
    """Error raised when civ6-mcp planner output omits the required final ``end_turn``."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str = "",
        parsed_payload: dict[str, Any] | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.parsed_payload = parsed_payload
        self.tool_calls = list(tool_calls or [])


class _MissingEndTurnPlanError(ValueError):
    """Internal retryable validation error that preserves parsed planner output."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: str,
        parsed_payload: dict[str, Any],
        tool_calls: list[ToolCall],
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.parsed_payload = parsed_payload
        self.tool_calls = list(tool_calls)


class Civ6McpToolPlanner:
    """Coordinate LLM-backed planning for allowlisted civ6-mcp tool calls."""

    def __init__(
        self,
        provider: Civ6McpPlannerProvider,
        *,
        tool_catalog: dict[str, dict],
        allowed_tools: tuple[str, ...] | None = None,
        max_retries: int = 2,
    ) -> None:
        self._provider = provider
        self._tool_catalog = tool_catalog
        self._allowed_tools = tuple(allowed_tools or _DEFAULT_PLANNER_TOOL_ALLOWLIST)
        self._max_retries = max_retries

    @property
    def allowed_tools(self) -> tuple[str, ...]:
        """Return the tool names this planner may emit."""
        return self._allowed_tools

    def render_tool_catalog(self) -> str:
        """Render allowlisted upstream tool metadata for the planner prompt."""
        lines: list[str] = []
        for name in self._allowed_tools:
            entry = self._tool_catalog.get(name)
            if not entry:
                continue
            description = (entry.get("description") or "").strip().splitlines()
            short_desc = description[0] if description else ""
            schema = entry.get("input_schema") or {}
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            param_parts: list[str] = []
            for param_name, param_schema in properties.items():
                if not isinstance(param_schema, dict):
                    continue
                param_type = param_schema.get("type") or "any"
                marker = "" if param_name in required else "?"
                default = param_schema.get("default")
                default_str = ""
                if default is not None and param_name not in required:
                    default_str = f"={default!r}"
                param_parts.append(f"{param_name}{marker}: {param_type}{default_str}")
            params_text = ", ".join(param_parts) if param_parts else ""
            lines.append(f"- {name}({params_text}) — {short_desc}")
        return "\n".join(lines) if lines else "(no tools loaded)"

    def plan(
        self,
        *,
        strategy: str,
        state_context: str,
        recent_calls: str,
        hitl_directive: str = "",
    ) -> PlannerResult:
        """Return a validated planner result, retrying invalid provider responses."""
        system_prompt = build_planner_system_prompt(
            tool_catalog=self.render_tool_catalog(),
            allowed_tools=self._allowed_tools,
        )
        user_prompt = build_planner_user_prompt(
            strategy=strategy,
            state_context=state_context,
            recent_calls=recent_calls,
            hitl_directive=hitl_directive,
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            if attempt > 0:
                full_prompt += (
                    "\n\nThe previous response was invalid JSON, missing the "
                    "expected schema, or violated backend turn-planning rules."
                )
                if last_error is not None:
                    full_prompt += f" Planner validation error: {last_error}."
                full_prompt += " Re-emit ONLY a valid JSON object as specified."
            try:
                content_parts = [self._provider._build_text_content(full_prompt)]
                response = self._provider._send_to_api(
                    content_parts,
                    temperature=0.2,
                    max_tokens=4096,
                )
                raw = response.content
                payload = _load_strict_planner_payload(raw)
                tool_calls = coerce_tool_calls(payload)
                if not tool_calls:
                    raise ValueError("planner emitted zero tool calls")
                payload, tool_calls = _repair_or_reject_final_end_turn(
                    payload,
                    tool_calls,
                    raw_response=raw,
                )
                _validate_successful_plan_reflections(tool_calls)
                return PlannerResult(
                    tool_calls=tool_calls,
                    raw_response=raw,
                    parsed_payload=payload,
                )
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "civ6-mcp planner attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        if isinstance(last_error, _MissingEndTurnPlanError):
            raise MissingEndTurnPlannerOutputError(
                f"civ6-mcp planner exhausted retries: {last_error}",
                raw_response=last_error.raw_response,
                parsed_payload=last_error.parsed_payload,
                tool_calls=last_error.tool_calls,
            ) from last_error

        raise RuntimeError(f"civ6-mcp planner exhausted retries: {last_error}")

    def plan_from_observation(
        self,
        *,
        observation: Civ6McpNormalizedObservation | object,
        strategy: str,
        recent_calls: str,
        hitl_directive: str = "",
    ) -> PlannerResult:
        """Return a prioritized and validated tool-call plan from an observation payload."""
        normalized_observation = _normalize_planner_observation(observation)

        turn_plan = build_prioritized_turn_plan(normalized_observation, strategy=strategy)
        state_context = (
            f"{normalized_observation.planner_context}\n\n## PRIORITIZED_MCP_INTENTS\n{turn_plan.render_for_prompt()}"
        )
        observation_hints = _render_observation_payload_hints(normalized_observation)
        if observation_hints:
            state_context = f"{state_context}\n\n## OBSERVATION_PAYLOAD_HINTS\n{observation_hints}"
        return self.plan(
            strategy=strategy,
            state_context=state_context,
            recent_calls=recent_calls,
            hitl_directive=hitl_directive,
        )


def _load_strict_planner_payload(raw: str) -> dict[str, Any]:
    """Load the planner response as strict JSON with the required top-level shape."""
    payload = _load_json_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("Planner output must be a top-level JSON object with a 'tool_calls' array.")
    try:
        planner_tool_call_items(payload, reject_vlm_actions=True)
    except ValueError as exc:
        message = str(exc)
        if "VLM/computer-use action-plan payload cannot run" in message:
            raise ValueError(message) from exc
        if "Planner payload 'tool_calls' must be a list." in message:
            raise ValueError("Planner output 'tool_calls' must be a list.") from exc
        if "Unexpected planner payload shape" in message:
            raise ValueError("Planner output must include a 'tool_calls' array.") from exc
        raise
    return payload


def _repair_or_reject_final_end_turn(
    payload: dict[str, Any],
    tool_calls: list[ToolCall],
    *,
    raw_response: str,
) -> tuple[dict[str, Any], list[ToolCall]]:
    """Ensure the executable planner sequence ends with ``end_turn``.

    A call after ``end_turn`` cannot safely run in the same turn. When the
    model emitted a valid ``end_turn`` and then drifted into extra calls, keep
    the closed-turn prefix. If there is no ``end_turn`` at all, reject the plan
    so retry prompting can request a proper turn close.
    """
    if tool_calls[-1].tool == END_TURN_TOOL:
        return payload, tool_calls

    end_turn_index = next((index for index, call in enumerate(tool_calls) if call.tool == END_TURN_TOOL), None)
    if end_turn_index is None:
        raise _MissingEndTurnPlanError(
            "final tool call must be end_turn",
            raw_response=raw_response,
            parsed_payload=payload,
            tool_calls=tool_calls,
        )

    repaired_payload = dict(payload)
    repaired_payload["tool_calls"] = list(payload["tool_calls"][: end_turn_index + 1])
    logger.warning(
        "civ6-mcp planner emitted %d trailing call(s) after end_turn; dropping them.",
        len(tool_calls) - end_turn_index - 1,
    )
    return repaired_payload, tool_calls[: end_turn_index + 1]


def _validate_successful_plan_reflections(tool_calls: list[ToolCall]) -> None:
    """Validate the successful planner sequence has complete end-turn reflections."""
    if not tool_calls or tool_calls[-1].tool != END_TURN_TOOL:
        raise ValueError("final tool call must be end_turn")

    arguments = tool_calls[-1].arguments
    non_string = [
        field for field in END_TURN_REFLECTION_FIELDS if field in arguments and not isinstance(arguments[field], str)
    ]
    if non_string:
        raise ValueError(f"end_turn reflection fields must be strings: {', '.join(non_string)}")

    missing = [field for field in END_TURN_REFLECTION_FIELDS if not str(arguments.get(field) or "").strip()]
    if missing:
        raise ValueError(f"end_turn requires non-empty reflection fields: {', '.join(missing)}")


def _normalize_planner_observation(observation: object) -> Civ6McpNormalizedObservation:
    """Normalize planner input while rejecting explicit non-civ6 backend payloads."""
    if isinstance(observation, Civ6McpNormalizedObservation):
        normalized = observation
    else:
        if isinstance(observation, Mapping) and observation.get("backend") not in (None, "civ6-mcp"):
            raise ValueError(f"civ6-mcp planner cannot consume backend {observation.get('backend')!r} observations.")
        normalized = normalize_raw_mcp_game_state(observation)

    if normalized.backend != "civ6-mcp":
        raise ValueError(f"civ6-mcp planner cannot consume backend {normalized.backend!r} observations.")
    return normalized


def _render_observation_payload_hints(observation: Civ6McpNormalizedObservation) -> str:
    """Render non-executable state-gap hints preserved from the observation payload."""
    overview = observation.raw_sections.get("OVERVIEW", "").lower()
    if "research:" not in overview or "civic:" not in overview:
        return ""
    if not any(pattern in overview for pattern in ("research:\ncivic:", "research: \ncivic:", "research:\r\ncivic:")):
        return ""
    return (
        "P070 get_tech_civics - Overview indicates blank research and civic choices. "
        "| source=OVERVIEW | trigger=blank research/civic labels"
    )


__all__ = [
    "Civ6McpToolPlanner",
    "DEFAULT_PLANNER_TOOL_ALLOWLIST",
    "MissingEndTurnPlannerOutputError",
    "PlannerResult",
]
