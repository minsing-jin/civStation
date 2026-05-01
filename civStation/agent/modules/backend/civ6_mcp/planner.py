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
from dataclasses import dataclass

from civStation.agent.modules.backend.civ6_mcp.executor import (
    ToolCall,
    coerce_tool_calls,
)
from civStation.agent.modules.backend.civ6_mcp.prompts import (
    build_planner_system_prompt,
    build_planner_user_prompt,
)
from civStation.utils.llm_provider.parser import strip_markdown

logger = logging.getLogger(__name__)


_DEFAULT_PLANNER_TOOL_ALLOWLIST: tuple[str, ...] = (
    # observation
    "get_game_overview",
    "get_units",
    "get_cities",
    "get_city_production",
    "get_empire_resources",
    "get_diplomacy",
    "get_tech_civics",
    "get_policies",
    "get_governors",
    "get_pantheon_beliefs",
    "get_religion_beliefs",
    "get_world_congress",
    "get_great_people",
    "get_notifications",
    "get_pending_diplomacy",
    "get_pending_trades",
    "get_victory_progress",
    "get_strategic_map",
    "get_dedications",
    "get_unit_promotions",
    "get_purchasable_tiles",
    "get_district_advisor",
    "get_wonder_advisor",
    "get_settle_advisor",
    "get_global_settle_advisor",
    "get_pathing_estimate",
    "get_trade_routes",
    "get_trade_destinations",
    "get_trade_options",
    "get_builder_tasks",
    "get_religion_spread",
    # action
    "unit_action",
    "skip_remaining_units",
    "city_action",
    "set_city_production",
    "set_city_focus",
    "purchase_item",
    "purchase_tile",
    "set_research",
    "set_policies",
    "change_government",
    "appoint_governor",
    "assign_governor",
    "promote_governor",
    "promote_unit",
    "upgrade_unit",
    "send_envoy",
    "send_diplomatic_action",
    "form_alliance",
    "propose_trade",
    "respond_to_trade",
    "propose_peace",
    "respond_to_diplomacy",
    "choose_pantheon",
    "found_religion",
    "choose_dedication",
    "queue_wc_votes",
    "patronize_great_person",
    "recruit_great_person",
    "reject_great_person",
    "spy_action",
    "dismiss_popup",
    "end_turn",
)


@dataclass
class PlannerResult:
    tool_calls: list[ToolCall]
    raw_response: str = ""
    parsed_payload: dict | None = None


class Civ6McpToolPlanner:
    """Asks an LLM to produce a tool-call plan for one turn.

    The provider must implement civStation's `BaseVLMProvider` interface;
    we call `_send_to_api()` with text-only content (no image), the same
    way `StrategyPlanner._call_vlm` does today.
    """

    def __init__(
        self,
        provider,
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
        return self._allowed_tools

    def render_tool_catalog(self) -> str:
        """Render the tool catalog text shown to the planner LLM."""
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
        """Produce a PlannerResult or raise if every retry yields invalid JSON."""
        system_prompt = build_planner_system_prompt(tool_catalog=self.render_tool_catalog())
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
                    "\n\nThe previous response was invalid JSON or missing the "
                    "expected schema. Re-emit ONLY a valid JSON object as specified."
                )
            try:
                content_parts = [self._provider._build_text_content(full_prompt)]
                response = self._provider._send_to_api(
                    content_parts,
                    temperature=0.2,
                    max_tokens=4096,
                )
                raw = response.content
                cleaned = strip_markdown(raw)
                payload = json.loads(cleaned)
                tool_calls = coerce_tool_calls(payload)
                if not tool_calls:
                    raise ValueError("planner emitted zero tool calls")
                return PlannerResult(
                    tool_calls=tool_calls,
                    raw_response=raw,
                    parsed_payload=payload if isinstance(payload, dict) else {"tool_calls": payload},
                )
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "civ6-mcp planner attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )

        raise RuntimeError(f"civ6-mcp planner exhausted retries: {last_error}")
