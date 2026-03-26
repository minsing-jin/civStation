from __future__ import annotations

import copy
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any

from civStation.agent.modules.context.context_manager import (
    ContextManager,
    PolicyTabCalibrationCache,
    TurnRecord,
)
from civStation.agent.modules.context.global_context import GlobalContext
from civStation.agent.modules.context.high_level_context import HighLevelContext
from civStation.agent.modules.context.primitive_context import PrimitiveContext
from civStation.agent.modules.hitl.agent_gate import AgentGate
from civStation.agent.modules.hitl.command_queue import CommandQueue, Directive
from civStation.agent.modules.memory.short_term_memory import ShortTermMemory
from civStation.mcp.codec import deserialize_value, patch_dataclass, serialize_value
from civStation.mcp.runtime import CaptureArtifact, SessionRuntimeConfig


@dataclass
class LayeredSession:
    session_id: str
    name: str
    runtime: SessionRuntimeConfig = field(default_factory=SessionRuntimeConfig.from_project_defaults)
    adapter_overrides: dict[str, str] = field(default_factory=dict)
    global_context: GlobalContext = field(default_factory=GlobalContext)
    high_level_context: HighLevelContext = field(default_factory=HighLevelContext)
    primitive_context: PrimitiveContext = field(default_factory=PrimitiveContext)
    policy_tab_cache: PolicyTabCalibrationCache = field(default_factory=PolicyTabCalibrationCache)
    turn_history: list[TurnRecord] = field(default_factory=list)
    macro_turn_summaries: list[str] = field(default_factory=list)
    memory: ShortTermMemory | None = None
    last_capture: CaptureArtifact | None = None
    last_route: dict[str, Any] = field(default_factory=dict)
    last_plan: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    command_queue: CommandQueue = field(default_factory=CommandQueue)
    agent_gate: AgentGate = field(init=False)
    relay_client: Any | None = None
    _context_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.agent_gate = AgentGate(self.command_queue)

    def touch(self) -> None:
        self.updated_at = datetime.now()

    def sync_from_context_manager(self, ctx: ContextManager) -> None:
        self.global_context = copy.deepcopy(ctx.global_context)
        self.high_level_context = copy.deepcopy(ctx.high_level_context)
        self.primitive_context = copy.deepcopy(ctx.primitive_context)
        self.policy_tab_cache = copy.deepcopy(ctx.get_policy_tab_cache())
        self.turn_history = list(ctx.get_turn_history())
        self.macro_turn_summaries = list(ctx.get_macro_turn_summaries(last_n=len(ctx.get_macro_turn_summaries())))
        self.touch()

    def apply_to_context_manager(self, ctx: ContextManager) -> None:
        ctx.global_context = copy.deepcopy(self.global_context)
        ctx.high_level_context = copy.deepcopy(self.high_level_context)
        ctx.primitive_context = copy.deepcopy(self.primitive_context)
        ctx._policy_tab_cache = copy.deepcopy(self.policy_tab_cache)  # noqa: SLF001
        ctx._turn_history = list(self.turn_history)  # noqa: SLF001
        ctx._macro_turn_summaries = list(self.macro_turn_summaries)  # noqa: SLF001

    @contextmanager
    def bound_context_manager(self):
        with self._context_lock:
            ContextManager.reset_instance()
            ctx = ContextManager.get_instance()
            self.apply_to_context_manager(ctx)
            try:
                yield ctx
            finally:
                self.sync_from_context_manager(ctx)
                ContextManager.reset_instance()

    def ensure_memory(self) -> ShortTermMemory:
        if self.memory is None:
            self.memory = ShortTermMemory()
        return self.memory

    def export_state(self) -> dict[str, Any]:
        pending_directives = [serialize_value(item) for item in self.command_queue.peek()]
        return {
            "session_id": self.session_id,
            "name": self.name,
            "runtime": serialize_value(self.runtime),
            "adapter_overrides": dict(self.adapter_overrides),
            "global_context": serialize_value(self.global_context),
            "high_level_context": serialize_value(self.high_level_context),
            "primitive_context": serialize_value(self.primitive_context),
            "policy_tab_cache": serialize_value(self.policy_tab_cache),
            "turn_history": serialize_value(self.turn_history),
            "macro_turn_summaries": list(self.macro_turn_summaries),
            "memory": serialize_value(self.memory) if self.memory is not None else None,
            "last_capture": serialize_value(self.last_capture) if self.last_capture is not None else None,
            "last_route": serialize_value(self.last_route),
            "last_plan": serialize_value(self.last_plan),
            "metadata": serialize_value(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_state": self.agent_gate.state.value,
            "queued_directives": pending_directives,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        name: str | None = None,
        session_id: str | None = None,
    ) -> LayeredSession:
        session = cls(
            session_id=session_id or state.get("session_id", f"session_{uuid.uuid4().hex[:12]}"),
            name=name or state.get("name", "imported-session"),
            runtime=deserialize_value(SessionRuntimeConfig, state.get("runtime", {})),
            adapter_overrides=dict(state.get("adapter_overrides", {})),
            global_context=deserialize_value(GlobalContext, state.get("global_context", {})),
            high_level_context=deserialize_value(HighLevelContext, state.get("high_level_context", {})),
            primitive_context=deserialize_value(PrimitiveContext, state.get("primitive_context", {})),
            policy_tab_cache=deserialize_value(PolicyTabCalibrationCache, state.get("policy_tab_cache", {})),
            turn_history=deserialize_value(list[TurnRecord], state.get("turn_history", [])),
            macro_turn_summaries=list(state.get("macro_turn_summaries", [])),
            memory=deserialize_value(ShortTermMemory | None, state.get("memory")),
            last_capture=deserialize_value(CaptureArtifact | None, state.get("last_capture")),
            metadata=deserialize_value(dict[str, Any], state.get("metadata", {})),
            created_at=datetime.fromisoformat(state.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(state.get("updated_at", datetime.now().isoformat())),
        )
        session.last_route = deserialize_value(dict[str, Any], state.get("last_route", {}))
        session.last_plan = deserialize_value(dict[str, Any], state.get("last_plan", {}))
        for item in state.get("queued_directives", []):
            session.command_queue.push(deserialize_value(Directive, item))
        return session

    def patch_context(self, patch: dict[str, Any]) -> None:
        if "global_context" in patch:
            patch_dataclass(self.global_context, patch["global_context"])
        if "high_level_context" in patch:
            patch_dataclass(self.high_level_context, patch["high_level_context"])
        if "primitive_context" in patch:
            patch_dataclass(self.primitive_context, patch["primitive_context"])
        if "policy_tab_cache" in patch:
            patch_dataclass(self.policy_tab_cache, patch["policy_tab_cache"])
        self.touch()

    def patch_memory(self, patch: dict[str, Any]) -> ShortTermMemory:
        memory = self.ensure_memory()
        patch_dataclass(memory, patch)
        self.touch()
        return memory

    def context_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "global_context": serialize_value(self.global_context),
            "high_level_context": serialize_value(self.high_level_context),
            "primitive_context": serialize_value(self.primitive_context),
            "policy_tab_cache": serialize_value(self.policy_tab_cache),
            "turn_history": serialize_value(self.turn_history),
            "macro_turn_summaries": list(self.macro_turn_summaries),
        }

    def strategy_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "strategy": serialize_value(self.high_level_context.current_strategy),
        }

    def memory_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "memory": serialize_value(self.memory) if self.memory is not None else None,
        }


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, LayeredSession] = {}
        self._lock = Lock()

    def create(
        self,
        *,
        name: str | None = None,
        runtime: SessionRuntimeConfig | None = None,
        adapter_overrides: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LayeredSession:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        session = LayeredSession(
            session_id=session_id,
            name=name or session_id,
            runtime=runtime or SessionRuntimeConfig.from_project_defaults(),
            adapter_overrides=adapter_overrides or {},
            metadata=metadata or {},
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def import_state(self, state: dict[str, Any], *, name: str | None = None) -> LayeredSession:
        session = LayeredSession.from_state(state, name=name, session_id=f"session_{uuid.uuid4().hex[:12]}")
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> LayeredSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Unknown session_id: {session_id}")
            return self._sessions[session_id]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if session.relay_client is not None:
            try:
                session.relay_client.stop()
            except Exception:  # noqa: BLE001
                pass
        return True

    def list(self) -> list[LayeredSession]:
        with self._lock:
            return list(self._sessions.values())
