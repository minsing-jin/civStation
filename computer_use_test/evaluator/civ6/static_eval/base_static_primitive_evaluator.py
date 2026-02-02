from computer_use_test.agent.models.civ6_models import AgentPlan, Action
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


# Ground Truth 데이터 구조
@dataclass
class GroundTruth:
    screenshot_path: str
    expected_primitive: str
    expected_actions: List[Action]


# 평가 결과 데이터 구조
@dataclass
class EvalResult:
    screenshot_path: str
    selected_primitive: str
    primitive_match: bool
    action_sequence_match: bool
    levenshtein_distance: int  # Action 순서 유사도 등


# 1. Primitive 추상 클래스 (Strategy Pattern)
class BasePrimitive(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_plan(self, screenshot_path: str) -> AgentPlan:
        """스크린샷을 받아 구체적인 행동(Action Set)을 반환"""
        pass


# 2. Router 인터페이스 (VLM Primitive Selector)
class PrimitiveRouter(ABC):
    @abstractmethod
    def route(self, screenshot_path: str) -> str:
        """스크린샷을 보고 적절한 Primitive 이름을 반환"""
        pass


# 3. Evaluator 인터페이스 (파이프라인 엔진)
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
            agent_plan = primitive_instance.generate_plan(gt.screenshot_path)
        else:
            agent_plan = AgentPlan(primitive_name="None", reasoning="Fail", actions=[])

        # Step 4: Compare & Metric (Logic implementation required)
        return self._compare(gt, selected_prim_name, agent_plan)

    @abstractmethod
    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        pass
