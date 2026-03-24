from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from PIL import Image

from computer_use_test.agent.modules.hitl.command_queue import Directive, DirectiveType
from computer_use_test.agent.modules.memory.short_term_memory import ShortTermMemory
from computer_use_test.agent.modules.strategy.strategy_schemas import StructuredStrategy
from computer_use_test.mcp.codec import deserialize_value, serialize_value
from computer_use_test.mcp.runtime import CaptureArtifact, LayerAdapterRegistry, SessionRuntimeConfig
from computer_use_test.mcp.session import LayeredSession, SessionRegistry
from computer_use_test.utils.llm_provider.parser import AgentAction


class LayeredComputerUseMCP:
    """Layered MCP facade around the existing computer-use architecture."""

    def __init__(
        self,
        *,
        adapter_registry: LayerAdapterRegistry | None = None,
        sessions: SessionRegistry | None = None,
        server_name: str = "computer-use-layered-mcp",
    ) -> None:
        self.adapters = adapter_registry or LayerAdapterRegistry()
        self.sessions = sessions or SessionRegistry()
        self.server = FastMCP(server_name, json_response=True)
        self._register_tools()
        self._register_resources()
        self._register_prompts()

    def _get_session(self, session_id: str) -> LayeredSession:
        return self.sessions.get(session_id)

    def _resolve_adapter_name(self, session: LayeredSession, key: str) -> str:
        return session.adapter_overrides.get(key, "builtin")

    def _artifact_dir(self) -> Path:
        path = Path(".tmp/computer_use_test/mcp_artifacts")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_capture(self, session: LayeredSession, pil_image, *, suffix: str = "capture") -> CaptureArtifact:
        filename = f"{session.session_id}_{suffix}.png"
        image_path = self._artifact_dir() / filename
        pil_image.save(image_path)
        artifact = CaptureArtifact(image_path=str(image_path))
        session.last_capture = artifact
        session.touch()
        return artifact

    def _load_image(self, session: LayeredSession, *, image_path: str | None, use_last_capture: bool = True):
        if image_path:
            pil_image = Image.open(image_path)
            artifact = CaptureArtifact(image_path=image_path)
            session.last_capture = artifact
            session.touch()
            return pil_image, artifact
        if use_last_capture and session.last_capture and session.last_capture.image_path:
            pil_image = Image.open(session.last_capture.image_path)
            return pil_image, session.last_capture
        pil_image, screen_w, screen_h, x_offset, y_offset = self.adapters.screen_capture()
        artifact = self._save_capture(session, pil_image)
        artifact.screen_w = screen_w
        artifact.screen_h = screen_h
        artifact.x_offset = x_offset
        artifact.y_offset = y_offset
        session.last_capture = artifact
        session.touch()
        return pil_image, artifact

    def _serialize_action(self, action: AgentAction | list[AgentAction] | None) -> dict[str, Any]:
        if action is None:
            return {"action": None}
        if isinstance(action, list):
            return {"actions": [serialize_value(item) for item in action]}
        return {"action": serialize_value(action)}

    def _run_router(self, session: LayeredSession, pil_image) -> dict[str, Any]:
        adapter_name = self._resolve_adapter_name(session, "action_router")
        adapter = self.adapters.action_routers[adapter_name]
        result = adapter(session, pil_image)
        session.last_route = serialize_value(result)
        session.touch()
        return session.last_route

    def _run_planner(
        self,
        session: LayeredSession,
        pil_image,
        primitive_name: str,
        *,
        strategy_override: str | None = None,
        recent_actions_override: str | None = None,
    ) -> dict[str, Any]:
        adapter_name = self._resolve_adapter_name(session, "action_planner")
        adapter = self.adapters.action_planners[adapter_name]
        action = adapter(
            session,
            pil_image,
            primitive_name,
            strategy_override=strategy_override,
            recent_actions_override=recent_actions_override,
        )
        payload = {"primitive": primitive_name, **self._serialize_action(action)}
        session.last_plan = payload
        session.touch()
        return payload

    def _run_context_observer(self, session: LayeredSession, pil_image) -> dict[str, Any]:
        adapter_name = self._resolve_adapter_name(session, "context_observer")
        adapter = self.adapters.context_observers[adapter_name]
        result = adapter(session, pil_image)
        if isinstance(result, dict):
            summary = str(result.get("situation_summary", "")).strip()
            threats = result.get("threats")
            opportunities = result.get("opportunities")
            if summary:
                session.high_level_context.notes.append(summary)
                session.high_level_context.notes = session.high_level_context.notes[-3:]
            if isinstance(threats, list):
                session.high_level_context.active_threats = [str(item) for item in threats]
            if isinstance(opportunities, list):
                session.high_level_context.opportunities = [str(item) for item in opportunities]
        session.touch()
        return session.context_payload()

    def _run_strategy_refiner(self, session: LayeredSession, raw_input: str) -> StructuredStrategy:
        adapter_name = self._resolve_adapter_name(session, "strategy_refiner")
        adapter = self.adapters.strategy_refiners[adapter_name]
        strategy = adapter(session, raw_input)
        session.high_level_context.set_strategy(strategy)
        session.touch()
        return strategy

    def _run_executor(
        self,
        session: LayeredSession,
        action: AgentAction,
        capture: CaptureArtifact | None = None,
    ) -> dict[str, Any]:
        adapter_name = self._resolve_adapter_name(session, "action_executor")
        adapter = self.adapters.action_executors[adapter_name]
        capture_artifact = capture or session.last_capture
        if capture_artifact is None:
            _, capture_artifact = self._load_image(session, image_path=None, use_last_capture=False)
        result = adapter(session, action, capture_artifact)
        session.touch()
        return result

    def _register_tools(self) -> None:
        mcp = self.server

        @mcp.tool(name="adapter_list", structured_output=True)
        def adapter_list() -> dict[str, Any]:
            return self.adapters.list_available()

        @mcp.tool(name="session_create", structured_output=True)
        def session_create(
            name: str | None = None,
            runtime: dict[str, Any] | None = None,
            adapter_overrides: dict[str, str] | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            runtime_config = (
                deserialize_value(SessionRuntimeConfig, runtime)
                if runtime
                else SessionRuntimeConfig.from_project_defaults()
            )
            session = self.sessions.create(
                name=name,
                runtime=runtime_config,
                adapter_overrides=adapter_overrides,
                metadata=metadata,
            )
            return {
                "session_id": session.session_id,
                "name": session.name,
                "runtime": serialize_value(session.runtime),
                "adapter_overrides": dict(session.adapter_overrides),
            }

        @mcp.tool(name="session_list", structured_output=True)
        def session_list() -> dict[str, Any]:
            return {
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "name": session.name,
                        "updated_at": session.updated_at.isoformat(),
                        "agent_state": session.agent_gate.state.value,
                    }
                    for session in self.sessions.list()
                ]
            }

        @mcp.tool(name="session_get", structured_output=True)
        def session_get(session_id: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            return {
                "session_id": session.session_id,
                "name": session.name,
                "runtime": serialize_value(session.runtime),
                "adapter_overrides": dict(session.adapter_overrides),
                "metadata": serialize_value(session.metadata),
                "agent_state": session.agent_gate.state.value,
                "last_capture": serialize_value(session.last_capture),
                "last_route": serialize_value(session.last_route),
                "last_plan": serialize_value(session.last_plan),
            }

        @mcp.tool(name="session_export", structured_output=True)
        def session_export(session_id: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            return {"session_id": session_id, "state": session.export_state()}

        @mcp.tool(name="session_import", structured_output=True)
        def session_import(state: dict[str, Any], name: str | None = None) -> dict[str, Any]:
            session = self.sessions.import_state(state, name=name)
            return {
                "session_id": session.session_id,
                "name": session.name,
                "runtime": serialize_value(session.runtime),
            }

        @mcp.tool(name="session_delete", structured_output=True)
        def session_delete(session_id: str) -> dict[str, Any]:
            return {"deleted": self.sessions.delete(session_id)}

        @mcp.tool(name="session_config_get", structured_output=True)
        def session_config_get(session_id: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            return {
                "session_id": session_id,
                "runtime": serialize_value(session.runtime),
                "adapter_overrides": dict(session.adapter_overrides),
            }

        @mcp.tool(name="session_config_update", structured_output=True)
        def session_config_update(
            session_id: str,
            runtime_patch: dict[str, Any] | None = None,
            adapter_overrides: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            if runtime_patch:
                merged = serialize_value(session.runtime)
                merged.update(runtime_patch)
                session.runtime = deserialize_value(SessionRuntimeConfig, merged)
            if adapter_overrides:
                session.adapter_overrides.update(adapter_overrides)
            session.touch()
            return {
                "session_id": session_id,
                "runtime": serialize_value(session.runtime),
                "adapter_overrides": dict(session.adapter_overrides),
            }

        @mcp.tool(name="context_get", structured_output=True)
        def context_get(session_id: str) -> dict[str, Any]:
            return self._get_session(session_id).context_payload()

        @mcp.tool(name="context_update", structured_output=True)
        def context_update(session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
            session = self._get_session(session_id)
            session.patch_context(patch)
            return session.context_payload()

        @mcp.tool(name="context_observe", structured_output=True)
        def context_observe(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, _ = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            return {"session_id": session_id, "context": self._run_context_observer(session, pil_image)}

        @mcp.tool(name="strategy_get", structured_output=True)
        def strategy_get(session_id: str) -> dict[str, Any]:
            return self._get_session(session_id).strategy_payload()

        @mcp.tool(name="strategy_set", structured_output=True)
        def strategy_set(session_id: str, strategy: dict[str, Any]) -> dict[str, Any]:
            session = self._get_session(session_id)
            structured = deserialize_value(StructuredStrategy, strategy)
            session.high_level_context.set_strategy(structured)
            session.touch()
            return session.strategy_payload()

        @mcp.tool(name="strategy_refine", structured_output=True)
        def strategy_refine(session_id: str, raw_input: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            strategy = self._run_strategy_refiner(session, raw_input)
            return {"session_id": session_id, "strategy": serialize_value(strategy)}

        @mcp.tool(name="memory_get", structured_output=True)
        def memory_get(session_id: str) -> dict[str, Any]:
            return self._get_session(session_id).memory_payload()

        @mcp.tool(name="memory_start_task", structured_output=True)
        def memory_start_task(
            session_id: str,
            primitive_name: str,
            max_steps: int = 10,
            enable_choice_catalog: bool = False,
            enable_policy_state: bool = False,
            enable_voting_state: bool = False,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            memory = ShortTermMemory()
            memory.start_task(
                primitive_name=primitive_name,
                max_steps=max_steps,
                normalizing_range=session.runtime.normalizing_range,
                enable_choice_catalog=enable_choice_catalog,
                enable_policy_state=enable_policy_state,
                enable_voting_state=enable_voting_state,
            )
            session.memory = memory
            session.touch()
            return session.memory_payload()

        @mcp.tool(name="memory_update", structured_output=True)
        def memory_update(session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
            session = self._get_session(session_id)
            session.patch_memory(patch)
            return session.memory_payload()

        @mcp.tool(name="memory_reset", structured_output=True)
        def memory_reset(session_id: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            session.memory = None
            session.touch()
            return session.memory_payload()

        @mcp.tool(name="action_route", structured_output=True)
        def action_route(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            route = self._run_router(session, pil_image)
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                **route,
            }

        @mcp.tool(name="action_plan", structured_output=True)
        def action_plan(
            session_id: str,
            primitive_name: str | None = None,
            image_path: str | None = None,
            use_last_capture: bool = True,
            strategy_override: str | None = None,
            recent_actions_override: str | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            effective_primitive = primitive_name or session.last_route.get("primitive")
            if not effective_primitive:
                raise ValueError("primitive_name is required when no prior route exists")
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            plan = self._run_planner(
                session,
                pil_image,
                effective_primitive,
                strategy_override=strategy_override,
                recent_actions_override=recent_actions_override,
            )
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                **plan,
            }

        @mcp.tool(name="action_execute", structured_output=True)
        def action_execute(
            session_id: str,
            action: dict[str, Any],
            capture: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            agent_action = deserialize_value(AgentAction, action)
            capture_artifact = deserialize_value(CaptureArtifact | None, capture) if capture else session.last_capture
            result = self._run_executor(session, agent_action, capture_artifact)
            return {"session_id": session_id, **result}

        @mcp.tool(name="action_route_and_plan", structured_output=True)
        def action_route_and_plan(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
            strategy_override: str | None = None,
            recent_actions_override: str | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            route = self._run_router(session, pil_image)
            plan = self._run_planner(
                session,
                pil_image,
                route["primitive"],
                strategy_override=strategy_override,
                recent_actions_override=recent_actions_override,
            )
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                "route": route,
                "plan": plan,
            }

        @mcp.tool(name="workflow_observe", structured_output=True)
        def workflow_observe(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            context = self._run_context_observer(session, pil_image)
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                "context": context,
            }

        @mcp.tool(name="workflow_decide", structured_output=True)
        def workflow_decide(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
            strategy_override: str | None = None,
            recent_actions_override: str | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            route = self._run_router(session, pil_image)
            plan = self._run_planner(
                session,
                pil_image,
                route["primitive"],
                strategy_override=strategy_override,
                recent_actions_override=recent_actions_override,
            )
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                "route": route,
                "plan": plan,
            }

        @mcp.tool(name="workflow_act", structured_output=True)
        def workflow_act(
            session_id: str,
            action: dict[str, Any] | None = None,
            capture: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            action_payload = action or session.last_plan.get("action")
            if not action_payload:
                raise ValueError("action is required when no previous plan is available")
            agent_action = deserialize_value(AgentAction, action_payload)
            capture_artifact = deserialize_value(CaptureArtifact | None, capture) if capture else session.last_capture
            result = self._run_executor(session, agent_action, capture_artifact)
            return {"session_id": session_id, **result}

        @mcp.tool(name="workflow_step", structured_output=True)
        def workflow_step(
            session_id: str,
            image_path: str | None = None,
            use_last_capture: bool = True,
            execute: bool = False,
            strategy_override: str | None = None,
            recent_actions_override: str | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            pil_image, capture = self._load_image(session, image_path=image_path, use_last_capture=use_last_capture)
            route = self._run_router(session, pil_image)
            plan = self._run_planner(
                session,
                pil_image,
                route["primitive"],
                strategy_override=strategy_override,
                recent_actions_override=recent_actions_override,
            )
            execution = None
            if execute and plan.get("action"):
                execution = self._run_executor(session, deserialize_value(AgentAction, plan["action"]), capture)
            return {
                "session_id": session_id,
                "capture": serialize_value(capture),
                "route": route,
                "plan": plan,
                "execution": execution,
            }

        @mcp.tool(name="hitl_send", structured_output=True)
        def hitl_send(
            session_id: str,
            directive_type: str,
            payload: str = "",
            source: str = "mcp",
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            session = self._get_session(session_id)
            normalized = directive_type.lower()
            if normalized == "start":
                ok = session.agent_gate.start()
            elif normalized == "pause":
                ok = session.agent_gate.pause()
            elif normalized == "resume":
                ok = session.agent_gate.resume()
            elif normalized == "stop":
                ok = session.agent_gate.stop()
            else:
                directive_map = {
                    "change_strategy": DirectiveType.CHANGE_STRATEGY,
                    "primitive_override": DirectiveType.PRIMITIVE_OVERRIDE,
                    "custom": DirectiveType.CUSTOM,
                }
                dtype = directive_map.get(normalized)
                if dtype is None:
                    raise ValueError(f"Unsupported directive_type: {directive_type}")
                session.command_queue.push(
                    Directive(
                        directive_type=dtype,
                        payload=payload,
                        source=source,
                        metadata=metadata or {},
                    )
                )
                ok = True
            session.touch()
            return self._hitl_status_payload(session) | {"ok": ok}

        @mcp.tool(name="hitl_status", structured_output=True)
        def hitl_status(session_id: str) -> dict[str, Any]:
            session = self._get_session(session_id)
            return self._hitl_status_payload(session)

    def _hitl_status_payload(self, session: LayeredSession) -> dict[str, Any]:
        return {
            "session_id": session.session_id,
            "agent_state": session.agent_gate.state.value,
            "queued_directives": [serialize_value(item) for item in session.command_queue.peek()],
            "last_route": serialize_value(session.last_route),
            "last_plan": serialize_value(session.last_plan),
            "current_strategy": serialize_value(session.high_level_context.current_strategy),
            "relay_connected": session.relay_client is not None,
        }

    def _register_resources(self) -> None:
        mcp = self.server

        @mcp.resource("civ6://sessions", mime_type="application/json")
        def sessions_resource() -> str:
            payload = {
                "sessions": [
                    {
                        "session_id": session.session_id,
                        "name": session.name,
                        "updated_at": session.updated_at.isoformat(),
                        "agent_state": session.agent_gate.state.value,
                    }
                    for session in self.sessions.list()
                ]
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        @mcp.resource("civ6://sessions/{session_id}/state", mime_type="application/json")
        def session_state_resource(session_id: str) -> str:
            session = self._get_session(session_id)
            return json.dumps(session.export_state(), ensure_ascii=False, indent=2)

        @mcp.resource("civ6://sessions/{session_id}/context", mime_type="application/json")
        def session_context_resource(session_id: str) -> str:
            session = self._get_session(session_id)
            return json.dumps(
                {
                    "session_id": session_id,
                    "context": session.context_payload(),
                },
                ensure_ascii=False,
                indent=2,
            )

        @mcp.resource("civ6://sessions/{session_id}/memory", mime_type="application/json")
        def session_memory_resource(session_id: str) -> str:
            session = self._get_session(session_id)
            return json.dumps(session.memory_payload(), ensure_ascii=False, indent=2)

    def _register_prompts(self) -> None:
        mcp = self.server

        @mcp.prompt(name="strategy_only_workflow")
        def strategy_only_workflow(session_id: str) -> str:
            return (
                f"Use session `{session_id}` in strategy-only mode.\n"
                "Recommended sequence:\n"
                "1. context_get\n"
                "2. memory_get\n"
                "3. strategy_refine or strategy_set\n"
                "4. strategy_get\n"
                "Do not call action_execute unless the user explicitly wants execution."
            )

        @mcp.prompt(name="plan_only_workflow")
        def plan_only_workflow(session_id: str) -> str:
            return (
                f"Use session `{session_id}` in plan-only mode.\n"
                "Recommended sequence:\n"
                "1. workflow_observe\n"
                "2. action_route\n"
                "3. action_plan\n"
                "4. Return the proposed action without execution."
            )

        @mcp.prompt(name="full_orchestration_workflow")
        def full_orchestration_workflow(session_id: str) -> str:
            return (
                f"Use session `{session_id}` for full orchestration.\n"
                "Recommended sequence:\n"
                "1. workflow_observe\n"
                "2. workflow_decide\n"
                "3. workflow_act or workflow_step(execute=true)\n"
                "4. hitl_status to surface pending directives."
            )

        @mcp.prompt(name="relay_controlled_workflow")
        def relay_controlled_workflow(session_id: str) -> str:
            return (
                f"Use session `{session_id}` for relay-driven HITL.\n"
                "Use hitl_send for start/pause/resume/stop and change_strategy directives.\n"
                "Use hitl_status to inspect queued directives, current strategy, and last planned action."
            )


def main() -> None:
    LayeredComputerUseMCP().server.run(transport="stdio")


if __name__ == "__main__":
    main()
