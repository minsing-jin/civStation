from abc import ABC, abstractmethod


class PrimitiveRouter(ABC):
    @abstractmethod
    def route(self, screenshot_path: str) -> str:
        """Select the appropriate primitive name from a screenshot.

        (스크린샷을 보고 적절한 Primitive 이름을 반환)
        """
        pass
