from abc import ABC, abstractmethod
from dataclasses import dataclass

from civStation.agent.models.schema import Action, AgentPlan
from civStation.agent.modules.primitive.base_primitive import BasePrimitive
from civStation.agent.modules.router.base_router import PrimitiveRouter


# Ground truth data structure (Ground Truth 데이터 구조)
@dataclass
class GroundTruth:
    screenshot_path: str
    expected_primitive: str
    expected_actions: list[Action]


# Evaluation result data structure (평가 결과 데이터 구조)
@dataclass
class EvalResult:
    screenshot_path: str
    selected_primitive: str
    primitive_match: bool
    action_sequence_match: bool
    levenshtein_distance: int  # Action sequence similarity (Action 순서 유사도)


# Evaluator interface / pipeline engine (Evaluator 인터페이스 / 파이프라인 엔진)
class BaseEvaluator(ABC):
    def __init__(self, router: PrimitiveRouter, primitives: dict[str, BasePrimitive]):
        self.router = router
        self.primitives = primitives

    def evaluate_single(self, gt: GroundTruth) -> EvalResult:
        # Step 1: Load Screenshot (Implicit in path)

        # Step 2: VLM Primitive Selection Eval
        selected_prim_name = self.router.route(gt.screenshot_path)

        # Step 3: Planning & Action Set Generation
        if selected_prim_name in self.primitives:
            primitive_instance = self.primitives[selected_prim_name]
            agent_plan = primitive_instance.generate_plan_and_action(gt.screenshot_path)
        else:
            agent_plan = AgentPlan(primitive_name="None", reasoning="Fail", actions=[])

        # Step 4: Compare & Metric (Logic implementation required)
        return self._compare(gt, selected_prim_name, agent_plan)

    @abstractmethod
    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        pass
