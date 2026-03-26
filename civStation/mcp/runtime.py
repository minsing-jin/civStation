from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from civStation.agent.modules.context.context_updater import ContextUpdater
from civStation.agent.modules.strategy import StrategyPlanner
from civStation.agent.modules.strategy.strategy_schemas import StructuredStrategy
from civStation.agent.turn_executor import plan_action, route_primitive
from civStation.utils.llm_provider import create_provider
from civStation.utils.llm_provider.parser import AgentAction
from civStation.utils.screen import capture_screen_pil, execute_action


@dataclass
class ProviderConfig:
    provider: str = "gemini"
    model: str | None = None


@dataclass
class SessionRuntimeConfig:
    router: ProviderConfig = field(default_factory=ProviderConfig)
    planner: ProviderConfig = field(default_factory=ProviderConfig)
    strategy: ProviderConfig = field(default_factory=ProviderConfig)
    normalizing_range: int = 1000
    delay_before_action: float = 0.5
    prompt_language: str = "eng"
    execution_mode: str = "dry_run"
    require_execute_confirmation: bool = True

    @classmethod
    def from_project_defaults(cls, config_path: Path | None = None) -> SessionRuntimeConfig:
        path = config_path or Path("config.yaml")
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        provider = data.get("provider", "gemini")
        model = data.get("model")
        return cls(
            router=ProviderConfig(
                provider=data.get("router-provider", provider),
                model=data.get("router-model", model),
            ),
            planner=ProviderConfig(
                provider=data.get("planner-provider", provider),
                model=data.get("planner-model", model),
            ),
            strategy=ProviderConfig(
                provider=data.get("planner-provider", provider),
                model=data.get("planner-model", model),
            ),
            normalizing_range=int(data.get("range", 1000)),
            delay_before_action=float(data.get("delay-action", 0.5)),
            prompt_language=str(data.get("prompt-language", "eng")),
            execution_mode=str(data.get("mcp-execution-mode", "dry_run")),
            require_execute_confirmation=bool(data.get("mcp-require-execute-confirmation", True)),
        )


@dataclass
class CaptureArtifact:
    image_path: str = ""
    screen_w: int = 0
    screen_h: int = 0
    x_offset: int = 0
    y_offset: int = 0


ActionRouterAdapter = Callable[[Any, Any], dict[str, Any]]
ActionPlannerAdapter = Callable[[Any, Any, str], AgentAction | list[AgentAction] | None]
ContextObserverAdapter = Callable[[Any, Any], dict[str, Any]]
StrategyRefinerAdapter = Callable[[Any, str], StructuredStrategy]
ActionExecutorAdapter = Callable[[Any, AgentAction, CaptureArtifact], dict[str, Any]]


class LayerAdapterRegistry:
    """Named adapter registry for MCP-layer customization."""

    def __init__(
        self,
        *,
        action_routers: dict[str, ActionRouterAdapter] | None = None,
        action_planners: dict[str, ActionPlannerAdapter] | None = None,
        context_observers: dict[str, ContextObserverAdapter] | None = None,
        strategy_refiners: dict[str, StrategyRefinerAdapter] | None = None,
        action_executors: dict[str, ActionExecutorAdapter] | None = None,
        provider_factory: Callable[[str, str | None], Any] = create_provider,
        screen_capture: Callable[[], tuple[Any, int, int, int, int]] = capture_screen_pil,
        low_level_executor: Callable[[AgentAction, int, int, int, int, int], None] = execute_action,
        include_builtins: bool = True,
    ) -> None:
        self.provider_factory = provider_factory
        self.screen_capture = screen_capture
        self.low_level_executor = low_level_executor

        self.action_routers: dict[str, ActionRouterAdapter] = {}
        self.action_planners: dict[str, ActionPlannerAdapter] = {}
        self.context_observers: dict[str, ContextObserverAdapter] = {}
        self.strategy_refiners: dict[str, StrategyRefinerAdapter] = {}
        self.action_executors: dict[str, ActionExecutorAdapter] = {}

        if include_builtins:
            self._register_builtins()

        self.action_routers.update(action_routers or {})
        self.action_planners.update(action_planners or {})
        self.context_observers.update(context_observers or {})
        self.strategy_refiners.update(strategy_refiners or {})
        self.action_executors.update(action_executors or {})

    def _register_builtins(self) -> None:
        def builtin_router(session, pil_image):
            provider = self.provider_factory(session.runtime.router.provider, session.runtime.router.model)
            result = route_primitive(provider, pil_image)
            return {"primitive": result.primitive, "reasoning": result.reasoning}

        def builtin_planner(
            session,
            pil_image,
            primitive_name,
            *,
            strategy_override=None,
            recent_actions_override=None,
        ):
            provider = self.provider_factory(session.runtime.planner.provider, session.runtime.planner.model)
            with session.bound_context_manager() as ctx:
                from civStation.agent.turn_executor import (
                    _build_recent_actions_string,
                    _build_strategy_with_directive,
                )

                ctx.set_current_primitive(primitive_name)
                strategy_text = strategy_override or _build_strategy_with_directive(ctx, primitive_name)
                recent_actions = recent_actions_override or _build_recent_actions_string(ctx)
                return plan_action(
                    provider,
                    pil_image,
                    primitive_name,
                    normalizing_range=session.runtime.normalizing_range,
                    high_level_strategy=strategy_text,
                    recent_actions_string=recent_actions,
                    prompt_language=session.runtime.prompt_language,
                )

        def builtin_context_observer(session, pil_image):
            provider = self.provider_factory(session.runtime.strategy.provider, session.runtime.strategy.model)
            with session.bound_context_manager() as ctx:
                updater = ContextUpdater(ctx, provider)
                updater._analyze_and_update(pil_image)
                session.sync_from_context_manager(ctx)
                return {
                    "situation_summary": ctx.high_level_context.notes[-1] if ctx.high_level_context.notes else "",
                    "threats": list(ctx.high_level_context.active_threats),
                    "opportunities": list(ctx.high_level_context.opportunities),
                }

        def builtin_strategy_refiner(session, raw_input):
            provider = self.provider_factory(session.runtime.strategy.provider, session.runtime.strategy.model)
            planner = StrategyPlanner(vlm_provider=provider, hitl_mode=False)
            with session.bound_context_manager() as ctx:
                strategy = planner.refine_strategy(raw_input, ctx)
                ctx.set_strategy(strategy)
                session.sync_from_context_manager(ctx)
                return strategy

        def builtin_executor(session, action, capture):
            self.low_level_executor(
                action,
                capture.screen_w,
                capture.screen_h,
                session.runtime.normalizing_range,
                capture.x_offset,
                capture.y_offset,
            )
            with session.bound_context_manager() as ctx:
                primitive_name = session.primitive_context.current_primitive or session.last_route.get("primitive", "")
                ctx.record_action(
                    action_type=action.action,
                    primitive=primitive_name or "mcp_action",
                    x=action.x,
                    y=action.y,
                    end_x=action.end_x,
                    end_y=action.end_y,
                    key=action.key,
                    text=action.text,
                    result="success",
                )
                session.sync_from_context_manager(ctx)
            return {
                "executed": True,
                "action": action.action,
                "x": action.x,
                "y": action.y,
            }

        self.action_routers["builtin"] = builtin_router
        self.action_planners["builtin"] = builtin_planner
        self.context_observers["builtin"] = builtin_context_observer
        self.strategy_refiners["builtin"] = builtin_strategy_refiner
        self.action_executors["builtin"] = builtin_executor

    def list_available(self) -> dict[str, list[str]]:
        return {
            "action_router": sorted(self.action_routers),
            "action_planner": sorted(self.action_planners),
            "context_observer": sorted(self.context_observers),
            "strategy_refiner": sorted(self.strategy_refiners),
            "action_executor": sorted(self.action_executors),
        }
