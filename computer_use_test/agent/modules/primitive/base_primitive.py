from abc import ABC, abstractmethod

from computer_use_test.agent.models.schema import AgentPlan


class BasePrimitive(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        """Generate a concrete action set from a screenshot.

        (스크린샷을 받아 구체적인 행동(Action Set)을 반환)
        """
        pass
