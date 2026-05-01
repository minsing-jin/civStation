"""civ6-mcp variants of the LayerAdapterRegistry slots.

These adapters keep the same Python callable signatures as the existing
``builtin_*`` adapters (so the outward MCP server `civStation/mcp/server.py`
can dispatch to them transparently), but route the call to the upstream
civ6-mcp server instead of the VLM/computer-use stack.

They are off by default. Register them through
``LayerAdapterRegistry(... action_routers={"civ6_mcp": ..., ...})`` or call
``register_civ6_mcp_adapters(registry, client)`` explicitly. The session's
``adapter_overrides`` dict then names them per-slot, e.g.::

    {"action_router": "civ6_mcp", "context_observer": "civ6_mcp",
     "action_planner": "civ6_mcp", "action_executor": "civ6_mcp"}
"""

from __future__ import annotations

import logging
from typing import Any

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClient
from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    ToolCall,
    coerce_tool_calls,
)
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner
from civStation.utils.llm_provider.parser import AgentAction

logger = logging.getLogger(__name__)


CIV6_MCP_ADAPTER_NAME = "civ6_mcp"


def make_civ6_mcp_router_adapter(_client: Civ6McpClient):
    """Routing is a no-op for civ6-mcp — there is no screen to classify."""

    def adapter(session, pil_image):  # noqa: ARG001 - signature parity with builtin_router
        return {
            "primitive": "civ6_mcp",
            "reasoning": "civ6-mcp backend does not classify screenshots; planner picks tools directly.",
        }

    return adapter


def make_civ6_mcp_planner_adapter(client: Civ6McpClient, *, provider_factory=None):
    """Returns the same shape as builtin_planner but ignores pil_image.

    The output is encoded as a single AgentAction whose `text` field is the
    JSON payload of the planner result (so the outward MCP server can ferry
    it through `action_plan`/`workflow_decide` without schema changes).
    """

    def adapter(
        session,
        pil_image,  # noqa: ARG001
        primitive_name,
        *,
        strategy_override=None,
        recent_actions_override=None,
    ):
        if provider_factory is None:
            provider = session  # the runtime registry passes a factory; if none, fail soft
            raise RuntimeError("civ6-mcp planner adapter requires a provider_factory")
        provider = provider_factory(session.runtime.planner.provider, session.runtime.planner.model)
        observer = Civ6McpObserver(client, _CtxView(session))
        bundle = observer.observe()
        planner = Civ6McpToolPlanner(provider=provider, tool_catalog=client.tool_schemas())
        plan = planner.plan(
            strategy=strategy_override or _safe_strategy(session),
            state_context=bundle.to_planner_context(),
            recent_calls="(none)",
            hitl_directive="",
        )
        # Pack the tool list into AgentAction.text for transport over the
        # outward MCP server. The civ6_mcp executor adapter unpacks it.
        import json as _json

        payload = {
            "tool_calls": [
                {"tool": call.tool, "arguments": call.arguments, "reasoning": call.reasoning}
                for call in plan.tool_calls
            ]
        }
        return AgentAction(
            action="civ6_mcp_tool_plan",
            text=_json.dumps(payload),
            reasoning=f"civ6-mcp planner emitted {len(plan.tool_calls)} tool calls",
        )

    return adapter


def make_civ6_mcp_observer_adapter(client: Civ6McpClient):
    def adapter(session, pil_image):  # noqa: ARG001
        observer = Civ6McpObserver(client, _CtxView(session))
        bundle = observer.observe()
        return {
            "situation_summary": " | ".join(
                bit
                for bit in (
                    f"Turn {bundle.overview.current_turn}" if bundle.overview.current_turn else "",
                    f"Era {bundle.overview.game_era}" if bundle.overview.game_era else "",
                    bundle.overview.current_research and f"Research {bundle.overview.current_research}",
                    bundle.overview.current_civic and f"Civic {bundle.overview.current_civic}",
                )
                if bit
            ),
            "threats": [],
            "opportunities": [],
            "raw_state": bundle.to_planner_context(),
        }

    return adapter


def make_civ6_mcp_executor_adapter(client: Civ6McpClient):
    """Executor that consumes the AgentAction emitted by the civ6-mcp planner.

    AgentAction.action == "civ6_mcp_tool_plan" with AgentAction.text holding
    a JSON tool-call payload. Falls back to a single ToolCall payload if
    AgentAction.action == "civ6_mcp_tool_call".
    """
    executor = Civ6McpExecutor(client)

    def adapter(session, action: AgentAction, capture):  # noqa: ARG001
        import json as _json

        payload: Any
        try:
            payload = _json.loads(action.text or "")
        except Exception as exc:  # noqa: BLE001
            return {
                "executed": False,
                "blocked": True,
                "reason": f"civ6-mcp executor could not parse AgentAction.text JSON: {exc}",
            }

        if action.action == "civ6_mcp_tool_call":
            if not isinstance(payload, dict) or "tool" not in payload:
                return {
                    "executed": False,
                    "blocked": True,
                    "reason": "civ6_mcp_tool_call payload missing 'tool'",
                }
            calls = [
                ToolCall(
                    tool=str(payload["tool"]),
                    arguments=dict(payload.get("arguments") or {}),
                    reasoning=str(payload.get("reasoning") or ""),
                )
            ]
        else:
            try:
                calls = coerce_tool_calls(payload)
            except ValueError as exc:
                return {
                    "executed": False,
                    "blocked": True,
                    "reason": f"civ6-mcp executor invalid plan: {exc}",
                }

        outcomes = executor.execute_many(calls)
        return {
            "executed": True,
            "tool_call_count": len(outcomes),
            "results": [
                {
                    "tool": outcome.call.tool,
                    "success": outcome.success,
                    "classification": outcome.classification,
                    "text": outcome.text[:480],
                    "error": outcome.error[:480],
                }
                for outcome in outcomes
            ],
        }

    return adapter


def register_civ6_mcp_adapters(registry, client: Civ6McpClient, *, provider_factory=None) -> None:
    """Mutate ``registry`` to add civ6-mcp variants under the name ``"civ6_mcp"``."""
    factory = provider_factory or registry.provider_factory
    registry.action_routers[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_router_adapter(client)
    registry.action_planners[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_planner_adapter(client, provider_factory=factory)
    registry.context_observers[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_observer_adapter(client)
    registry.action_executors[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_executor_adapter(client)


# ----- helpers ----------------------------------------------------------


def _safe_strategy(session) -> str:
    try:
        return session.high_level_context.get_strategy_string() or ""
    except Exception:  # noqa: BLE001
        return ""


class _CtxView:
    """Minimal ContextManager-shaped facade over a LayeredSession.

    Civ6McpObserver writes to a ContextManager via ``update_global_context``
    and ``update_game_observation``. The outward MCP server uses a session
    object with similar fields but slightly different methods, so we adapt.
    """

    def __init__(self, session) -> None:
        self._session = session

    def update_global_context(self, **kwargs) -> None:
        gc = getattr(self._session, "global_context", None)
        if gc is None:
            return
        for key, value in kwargs.items():
            if hasattr(gc, key):
                try:
                    setattr(gc, key, value)
                except Exception:  # noqa: BLE001
                    continue

    def update_game_observation(self, situation_summary: str, **_kwargs) -> None:
        hl = getattr(self._session, "high_level_context", None)
        if hl is None:
            return
        if not hasattr(hl, "notes") or hl.notes is None:
            return
        try:
            hl.notes.append(situation_summary)
            hl.notes = list(hl.notes)[-3:]
        except Exception:  # noqa: BLE001
            return
