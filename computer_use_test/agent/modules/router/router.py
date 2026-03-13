import os

from computer_use_test.agent.modules.router.base_router import PrimitiveRouter


class Civ6Router(PrimitiveRouter):
    def route(self, screenshot_path: str) -> str:
        # [TODO] VLM에게 스크린샷을 주고 Primitive 분류 요청
        # 여기서는 파일명에 힌트가 있다고 가정하고 Mocking
        if "unit" in screenshot_path:
            return "unit_ops_primitive"
        if "mayor" in screenshot_path:
            return "country_mayer_primitive"
        if "science" in screenshot_path:
            return "research_select_primitive"
        return "culture_decision_primitive"


class Civ6MockRouter(PrimitiveRouter):
    """
    Mock router that selects primitives based on screenshot filename keywords.

    This is for testing purposes - in production, this would use a VLM
    to analyze the screenshot and determine the appropriate primitive.
    """

    def route(self, screenshot_path: str) -> str:
        """
        Route to appropriate primitive based on filename keywords.

        Args:
            screenshot_path: Path to the screenshot file

        Returns:
            Name of the selected primitive
        """
        filename = os.path.basename(screenshot_path).lower()

        # Keyword-based routing (check more specific patterns first)
        if "research_select" in filename or "tech_select" in filename:
            return "research_select_primitive"
        if "production" in filename or "city_production" in filename:
            return "city_production_primitive"
        if "popup" in filename or "next_turn" in filename or "dialog" in filename:
            return "popup_primitive"
        if "science" in filename or "tech" in filename:
            return "research_select_primitive"
        if "culture" in filename or "civic" in filename:
            return "culture_decision_primitive"
        if "unit" in filename:
            return "unit_ops_primitive"

        # Default fallback
        return "popup_primitive"
