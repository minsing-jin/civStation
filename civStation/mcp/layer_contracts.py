from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ThreadAffinity(str, Enum):
    BACKGROUND = "background"
    MAIN_THREAD = "main_thread"
    EXTERNAL = "external"


@dataclass(frozen=True)
class LayerContract:
    name: str
    thread_affinity: ThreadAffinity
    ownership: str
    responsibilities: list[str] = field(default_factory=list)
    customization_points: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["thread_affinity"] = self.thread_affinity.value
        return payload


STRATEGY_CONTEXT_CONTRACT = LayerContract(
    name="strategy_context",
    thread_affinity=ThreadAffinity.BACKGROUND,
    ownership="Background observers/updaters own context observation and strategy refresh cadence.",
    responsibilities=[
        "observe and summarize state in the background",
        "update long-lived context and strategy state",
        "expose read/update hooks without blocking action execution",
    ],
    customization_points=[
        "context_observer",
        "strategy_refiner",
        "session runtime config",
    ],
    notes=[
        "This layer should not perform live primitive action execution.",
        "External callers can refine/update, but the layer remains background-oriented.",
    ],
)

PRIMITIVE_ACTION_CONTRACT = LayerContract(
    name="primitive_action",
    thread_affinity=ThreadAffinity.MAIN_THREAD,
    ownership="The main turn loop owns routing, planning, and action execution sequencing.",
    responsibilities=[
        "route the current screen to a primitive",
        "plan primitive actions",
        "execute primitive actions in order",
    ],
    customization_points=[
        "action_router",
        "action_planner",
        "action_executor",
    ],
    notes=[
        "Live execution should remain explicitly gated.",
        "This layer must preserve single-threaded action ordering.",
    ],
)

HITL_CONTRACT = LayerContract(
    name="hitl",
    thread_affinity=ThreadAffinity.EXTERNAL,
    ownership="External clients, relays, and queues own operator interaction.",
    responsibilities=[
        "receive lifecycle and directive signals",
        "surface queue and relay status",
        "bridge human overrides without collapsing layer boundaries",
    ],
    customization_points=[
        "relay transport/client adapter",
        "directive source integration",
    ],
    notes=[
        "HITL should stay outside the core action loop and communicate through queue/gate surfaces.",
    ],
)


def describe_layer_contracts() -> dict[str, dict[str, Any]]:
    return {
        "strategy_context": STRATEGY_CONTEXT_CONTRACT.to_dict(),
        "primitive_action": PRIMITIVE_ACTION_CONTRACT.to_dict(),
        "hitl": HITL_CONTRACT.to_dict(),
    }
