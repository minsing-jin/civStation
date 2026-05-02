"""Prompts for the civ6-mcp tool-call planner.

The planner LLM does NOT see screenshots. It sees:
- the player's high-level strategy (StructuredStrategy text)
- a state bundle rendered from civ6-mcp `get_*` tools
- a tool catalog of available action tools (subset of upstream civ6-mcp)
- a list of recent tool calls (to avoid loops)

It outputs a JSON list of tool calls to execute this turn, ending with
`end_turn` (which in civ6-mcp requires five non-empty reflection strings).
"""

from __future__ import annotations

from collections.abc import Iterable

from civStation.agent.modules.backend.civ6_mcp.operations import (
    END_TURN_REFLECTION_FIELDS,
    END_TURN_TOOL,
)

PLANNER_SYSTEM_PROMPT_EN = """\
You are the Civilization VI tool-call planner for the civStation agent.

Your job: pick a sequence of MCP tool calls to advance the current turn,
ending with `end_turn`. You drive the game directly through the upstream
civ6-mcp server — there are NO screenshots, NO clicks, no pixels.

Rules:
1. Output STRICT JSON. No prose, no markdown fences. The top-level value
   MUST be an object with key `"tool_calls"` whose value is an ordered list.
2. Each item in the list is an object with `"tool"` (string, exact tool
   name) and `"arguments"` (object). `"reasoning"` is optional but useful.
3. Read tools (names starting with `get_*`/`list_*`) are FREE — call them
   liberally to inspect state. Action tools mutate state; be deliberate.
4. The final tool call MUST be `end_turn` with all five reflections
   populated:  tactical, strategic, tooling, planning, hypothesis
   (each a non-empty string). The server rejects empty reflections.
5. Honor the player's strategy. Do not pursue victories the strategy
   does not endorse.
6. If `pending_trades_text` or `pending_diplomacy_text` is non-empty,
   handle them before calling `end_turn`.
7. Never call `run_lua` unless the strategy explicitly authorizes raw
   gameplay scripts.
8. Cap your tool-call list at 20 items. If you need more, skip non-essential
   actions and end the turn.

Available tools (subset; full schema is provided separately):
{tool_catalog}

Output schema:
{{
  "tool_calls": [
{tool_call_examples}
  ]
}}
"""


PLANNER_USER_PROMPT_EN = """\
[STRATEGY]
{strategy}

[GAME STATE]
{state_context}

[RECENT TOOL CALLS]
{recent_calls}

[HITL DIRECTIVE]
{hitl_directive}

Plan the next tool-call sequence for this turn. End with `end_turn`.
Output JSON only.
"""


def build_planner_user_prompt(
    *,
    strategy: str,
    state_context: str,
    recent_calls: str,
    hitl_directive: str,
) -> str:
    """Build the per-turn planner prompt from strategy, state, history, and HITL input."""
    return PLANNER_USER_PROMPT_EN.format(
        strategy=strategy or "(no strategy provided)",
        state_context=state_context or "(no state available)",
        recent_calls=recent_calls or "(none)",
        hitl_directive=hitl_directive or "(none)",
    )


def build_planner_system_prompt(
    *,
    tool_catalog: str,
    allowed_tools: Iterable[str] | None = None,
) -> str:
    """Build the planner system prompt with catalog text and tool examples."""
    planner_tools = tuple(_canonical_planner_tool_allowlist() if allowed_tools is None else allowed_tools)
    return PLANNER_SYSTEM_PROMPT_EN.format(
        tool_catalog=tool_catalog or "(no tools loaded)",
        tool_call_examples=_render_tool_call_examples(planner_tools),
    )


def _canonical_planner_tool_allowlist() -> tuple[str, ...]:
    """Return the planner module's canonical allow-list without importing it eagerly."""
    from civStation.agent.modules.backend.civ6_mcp.planner import (
        DEFAULT_PLANNER_TOOL_ALLOWLIST as planner_tool_allowlist,
    )

    return planner_tool_allowlist


def __getattr__(name: str) -> object:
    """Preserve legacy prompt-module access to the planner allow-list."""
    if name == "DEFAULT_PLANNER_TOOL_ALLOWLIST":
        return _canonical_planner_tool_allowlist()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _render_tool_call_examples(allowed_tools: Iterable[str] | None) -> str:
    """Render schema examples from the planner allow-list instead of fixed tool names."""
    ordered_tools = tuple(dict.fromkeys(tool for tool in allowed_tools or () if isinstance(tool, str) and tool))
    observation_tool = next((tool for tool in ordered_tools if tool.startswith(("get_", "list_"))), None)
    action_tool = next(
        (tool for tool in ordered_tools if tool != END_TURN_TOOL and not tool.startswith(("get_", "list_"))),
        None,
    )

    examples: list[str] = []
    if observation_tool:
        examples.append(f'    {{"tool": "{observation_tool}", "arguments": {{}}, "reasoning": "..."}},')
    if action_tool:
        examples.append(f'    {{"tool": "{action_tool}", "arguments": {{"...": "..."}}}},')
    if observation_tool or action_tool:
        examples.append("    ...")

    reflections = "\n".join(f'        "{field}": "...",' for field in END_TURN_REFLECTION_FIELDS)
    reflections = reflections.rstrip(",")
    examples.append(
        f'''    {{
      "tool": "{END_TURN_TOOL}",
      "arguments": {{
{reflections}
      }}
    }}'''
    )
    return "\n".join(examples)
