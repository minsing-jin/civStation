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

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypedDict

from civStation.agent.modules.backend.civ6_mcp.client import Civ6McpClientProtocol
from civStation.agent.modules.backend.civ6_mcp.executor import (
    Civ6McpExecutor,
    coerce_tool_call,
    coerce_tool_calls,
)
from civStation.agent.modules.backend.civ6_mcp.observation_schema import (
    Civ6McpNormalizedObservation,
    normalize_observation_bundle,
)
from civStation.agent.modules.backend.civ6_mcp.observer import Civ6McpObserver
from civStation.agent.modules.backend.civ6_mcp.planner import Civ6McpToolPlanner
from civStation.agent.modules.backend.civ6_mcp.planner_types import (
    Civ6McpPlannerProvider,
    PlannerResult,
)
from civStation.utils.llm_provider.parser import AgentAction

CIV6_MCP_ADAPTER_NAME = "civ6_mcp"
CIV6_MCP_TOOL_PLAN_ACTION = "civ6_mcp_tool_plan"
CIV6_MCP_TOOL_CALL_ACTION = "civ6_mcp_tool_call"


class Civ6McpRouteResult(TypedDict):
    """Outward MCP action_route payload produced by the civ6-mcp router."""

    primitive: str
    reasoning: str


class Civ6McpPlannerTransportPayload(TypedDict):
    """Internal payload ferried through the existing outward AgentAction schema."""

    tool_calls: list[dict[str, Any]]


class Civ6McpObservationResult(TypedDict):
    """Outward MCP context_observer payload produced by the civ6-mcp observer."""

    situation_summary: str
    threats: list[str]
    opportunities: list[str]


class Civ6McpPlannerLike(Protocol):
    """Planner surface used by the MCP adapter."""

    def plan(
        self,
        *,
        strategy: str,
        state_context: str,
        recent_calls: str,
        hitl_directive: str = "",
    ) -> PlannerResult:
        """Produce a parsed civ6-mcp tool-call plan."""
        ...


ProviderFactory = Callable[[str, str | None], Civ6McpPlannerProvider]
PlannerFactory = Callable[[Civ6McpPlannerProvider, Mapping[str, Mapping[str, Any]]], Civ6McpPlannerLike]


def make_civ6_mcp_router_adapter(
    _client: Civ6McpClientProtocol,
) -> Callable[[Any, Any], Civ6McpRouteResult]:
    """Routing is a no-op for civ6-mcp — there is no screen to classify."""

    def adapter(session: Any, pil_image: Any) -> Civ6McpRouteResult:  # noqa: ARG001 - builtin_router parity
        """Return the fixed civ6-mcp primitive route for outward MCP callers."""
        return {
            "primitive": "civ6_mcp",
            "reasoning": "civ6-mcp backend does not classify screenshots; planner picks tools directly.",
        }

    return adapter


def make_civ6_mcp_planner_adapter(
    client: Civ6McpClientProtocol,
    *,
    provider_factory: ProviderFactory | None = None,
    planner_factory: PlannerFactory | None = None,
) -> Callable[..., AgentAction]:
    """Returns the same shape as builtin_planner but ignores pil_image.

    The output is encoded as a single AgentAction whose `text` field is the
    JSON payload of the planner result (so the outward MCP server can ferry
    it through `action_plan`/`workflow_decide` without schema changes).
    """
    effective_planner_factory = planner_factory or _default_planner_factory

    def adapter(
        session: Any,
        pil_image: Any,  # noqa: ARG001
        primitive_name: str,
        *,
        strategy_override: str | None = None,
        recent_actions_override: str | None = None,
    ) -> AgentAction:
        """Observe state, plan civ6-mcp tool calls, and encode them as AgentAction."""
        _raise_if_non_civ6_mcp_primitive(primitive_name)
        if provider_factory is None:
            raise RuntimeError("civ6-mcp planner adapter requires a provider_factory")
        provider = provider_factory(session.runtime.planner.provider, session.runtime.planner.model)
        observer = Civ6McpObserver(client, _CtxView(session))
        bundle = observer.observe()
        planner = effective_planner_factory(provider, client.tool_schemas())
        plan = planner.plan(
            strategy=strategy_override or _safe_strategy(session),
            state_context=bundle.to_planner_context(),
            recent_calls=recent_actions_override or "(none)",
            hitl_directive=_safe_hitl_directive(session),
        )
        payload = encode_civ6_mcp_planner_result(plan)
        return AgentAction(
            action=CIV6_MCP_TOOL_PLAN_ACTION,
            text=json.dumps(payload),
            reasoning=f"civ6-mcp planner emitted {len(plan.tool_calls)} tool calls",
        )

    return adapter


def encode_civ6_mcp_planner_result(plan: PlannerResult) -> Civ6McpPlannerTransportPayload:
    """Translate a backend PlannerResult into the unchanged outward plan envelope.

    The MCP server still exposes an ``AgentAction`` under the ``action`` key.
    Only the backend-specific tool sequence is serialized into ``AgentAction.text``.
    """
    return {
        "tool_calls": [
            {"tool": call.tool, "arguments": dict(call.arguments), "reasoning": call.reasoning}
            for call in plan.tool_calls
        ]
    }


def make_civ6_mcp_observer_adapter(client: Civ6McpClientProtocol) -> Callable[[Any, Any], Civ6McpObservationResult]:
    """Translate upstream civ6-mcp observations into the existing outward schema."""

    def adapter(session: Any, pil_image: Any) -> Civ6McpObservationResult:  # noqa: ARG001
        """Observe civ6-mcp state and encode it for the outward MCP observer slot."""
        observer = Civ6McpObserver(client, _CtxView(session))
        bundle = observer.observe()
        observation = observer.last_observation or normalize_observation_bundle(bundle)
        return encode_civ6_mcp_observation_result(observation)

    return adapter


def encode_civ6_mcp_observation_result(observation: Civ6McpNormalizedObservation) -> Civ6McpObservationResult:
    """Encode normalized civ6-mcp state using the unchanged MCP observer result shape."""
    game_updates = observation.game_observation_updates
    return {
        "situation_summary": str(game_updates.get("situation_summary") or ""),
        "threats": [],
        "opportunities": [],
    }


def make_civ6_mcp_executor_adapter(client: Civ6McpClientProtocol) -> Callable[[Any, AgentAction, Any], dict[str, Any]]:
    """Executor that consumes the AgentAction emitted by the civ6-mcp planner.

    ``CIV6_MCP_TOOL_PLAN_ACTION`` expects ``AgentAction.text`` to hold a JSON
    tool-call payload. ``CIV6_MCP_TOOL_CALL_ACTION`` accepts a single tool-call
    object for direct MCP clients.
    """
    executor = Civ6McpExecutor(client)

    def adapter(session: Any, action: AgentAction, capture: Any) -> dict[str, Any]:  # noqa: ARG001
        """Decode a civ6-mcp AgentAction envelope and execute its tool calls."""
        if action.action not in (CIV6_MCP_TOOL_PLAN_ACTION, CIV6_MCP_TOOL_CALL_ACTION):
            return {
                "executed": False,
                "blocked": True,
                "reason": (
                    f"civ6-mcp executor only accepts civ6-mcp AgentAction envelopes; received action={action.action!r}"
                ),
            }

        payload: Any
        try:
            payload = json.loads(action.text or "")
        except Exception as exc:  # noqa: BLE001
            return {
                "executed": False,
                "blocked": True,
                "reason": f"civ6-mcp executor could not parse AgentAction.text JSON: {exc}",
            }

        if action.action == CIV6_MCP_TOOL_CALL_ACTION:
            if not isinstance(payload, dict):
                return {
                    "executed": False,
                    "blocked": True,
                    "reason": "civ6_mcp_tool_call payload must be an object",
                }
            try:
                calls = [coerce_tool_call(payload)]
            except ValueError as exc:
                return {
                    "executed": False,
                    "blocked": True,
                    "reason": f"civ6-mcp executor invalid tool call: {exc}",
                }
        else:
            try:
                calls = coerce_tool_calls(payload)
            except ValueError as exc:
                return {
                    "executed": False,
                    "blocked": True,
                    "reason": f"civ6-mcp executor invalid plan: {exc}",
                }

        outcomes = executor.execute_many(calls, stop_requested=lambda: _session_stop_requested(session))
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


def register_civ6_mcp_adapters(
    registry: Any,
    client: Civ6McpClientProtocol,
    *,
    provider_factory: ProviderFactory | None = None,
    planner_factory: PlannerFactory | None = None,
) -> None:
    """Mutate ``registry`` to add civ6-mcp variants under the name ``"civ6_mcp"``."""
    factory = provider_factory or registry.provider_factory
    registry.action_routers[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_router_adapter(client)
    registry.action_planners[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_planner_adapter(
        client,
        provider_factory=factory,
        planner_factory=planner_factory,
    )
    registry.context_observers[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_observer_adapter(client)
    registry.action_executors[CIV6_MCP_ADAPTER_NAME] = make_civ6_mcp_executor_adapter(client)


# ----- helpers ----------------------------------------------------------


def _default_planner_factory(
    provider: Civ6McpPlannerProvider,
    tool_catalog: Mapping[str, Mapping[str, Any]],
) -> Civ6McpToolPlanner:
    return Civ6McpToolPlanner(provider=provider, tool_catalog=dict(tool_catalog))


def _raise_if_non_civ6_mcp_primitive(primitive_name: str) -> None:
    if primitive_name not in (CIV6_MCP_ADAPTER_NAME, "civ6-mcp"):
        raise ValueError(
            "civ6-mcp planner adapter cannot plan VLM/computer-use primitives; "
            f"received primitive_name={primitive_name!r}"
        )


def _safe_strategy(session: Any) -> str:
    try:
        return session.high_level_context.get_strategy_string() or ""
    except Exception:  # noqa: BLE001
        return ""


def _safe_hitl_directive(session: Any) -> str:
    queue = getattr(session, "command_queue", None)
    peek = getattr(queue, "peek", None)
    if not callable(peek):
        return ""
    try:
        directives = peek()
    except Exception:  # noqa: BLE001
        return ""
    return _render_hitl_directives(directives)


def _session_stop_requested(session: Any) -> bool:
    gate = getattr(session, "agent_gate", None)
    is_stopped = getattr(gate, "is_stopped", False)
    try:
        if bool(is_stopped() if callable(is_stopped) else is_stopped):
            return True
    except Exception:  # noqa: BLE001
        pass

    queue = getattr(session, "command_queue", None)
    peek = getattr(queue, "peek", None)
    if not callable(peek):
        return False
    try:
        directives = peek()
    except Exception:  # noqa: BLE001
        return False
    for directive in directives:
        directive_type = getattr(directive, "directive_type", None)
        value = getattr(directive_type, "value", directive_type)
        if str(value).lower() == "stop":
            return True
    return False


def _render_hitl_directives(directives: Sequence[Any]) -> str:
    rendered: list[str] = []
    for directive in directives:
        directive_type = getattr(directive, "directive_type", None)
        if directive_type is not None:
            directive_type = getattr(directive_type, "value", directive_type)
        payload = getattr(directive, "payload", "")
        if directive_type or payload:
            rendered.append(f"{directive_type or 'directive'}: {payload}".strip())
    return "\n".join(rendered)


class _CtxView:
    """Minimal ContextManager-shaped facade over a LayeredSession.

    Civ6McpObserver writes to a ContextManager via ``update_global_context``
    and ``update_game_observation``. The outward MCP server uses a session
    object with similar fields but slightly different methods, so we adapt.
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    def update_global_context(self, **kwargs: Any) -> None:
        gc = getattr(self._session, "global_context", None)
        if gc is None:
            return
        for key, value in kwargs.items():
            if hasattr(gc, key):
                try:
                    setattr(gc, key, value)
                except Exception:  # noqa: BLE001
                    continue

    def update_game_observation(
        self,
        situation_summary: str,
        threats: list[str] | None = None,
        opportunities: list[str] | None = None,
        observation_fields: Mapping[str, object] | None = None,
    ) -> None:
        # The outward MCP server appends ``situation_summary`` exactly once
        # from the returned adapter payload. This facade only syncs structured
        # fields that the server cannot infer from that public schema.
        hl = getattr(self._session, "high_level_context", None)
        if hl is None:
            return
        if threats is not None:
            hl.active_threats = [str(item) for item in threats]
        if opportunities is not None:
            hl.opportunities = [str(item) for item in opportunities]
        if observation_fields is not None:
            hl.latest_game_observation = dict(observation_fields)
        return


__all__ = [
    "CIV6_MCP_ADAPTER_NAME",
    "CIV6_MCP_TOOL_CALL_ACTION",
    "CIV6_MCP_TOOL_PLAN_ACTION",
    "Civ6McpObservationResult",
    "Civ6McpPlannerTransportPayload",
    "Civ6McpRouteResult",
    "encode_civ6_mcp_observation_result",
    "encode_civ6_mcp_planner_result",
    "make_civ6_mcp_executor_adapter",
    "make_civ6_mcp_observer_adapter",
    "make_civ6_mcp_planner_adapter",
    "make_civ6_mcp_router_adapter",
    "register_civ6_mcp_adapters",
]
