import hashlib
import random

from computer_use_test.agent.models.schema import Action, AgentPlan, ClickAction, KeyPressAction
from computer_use_test.agent.modules.primitive.base_primitive import BasePrimitive
from computer_use_test.agent.modules.router.primitive_registry import get_primitive_prompt
from computer_use_test.utils.llm_provider.base import BaseVLMProvider


# TODO: This file is not used at test runner time, but is kept for reference
class UnitOpsPrimitive(BasePrimitive):
    """
    Primitive for unit operations in Civilization VI.

    Handles unit movement, combat, fortification, and other unit-related actions.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        """
        Initialize primitive.

        Args:
            vlm_provider: Optional VLM provider for real model calls.
                         If None, uses deterministic mocking.
            custom_prompt: Optional custom prompt to override default.
                          If None, uses default prompt from prompts module.
        """
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "unit_ops_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        """
        Generate unit operations plan.

        If VLM provider is available, uses real model.
        Otherwise, uses deterministic mock.

        Args:
            screenshot_path: Path to screenshot

        Returns:
            AgentPlan with actions
        """
        # If VLM provider available, use it
        if self.vlm_provider:
            # Use custom prompt if provided, otherwise use default from prompts module
            prompt = self.custom_prompt or get_primitive_prompt(self.name)

            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Fallback to mock
        seed = int(hashlib.md5(screenshot_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        base_x, base_y = 960, 540
        offset_x = rng.randint(-200, 200)
        offset_y = rng.randint(-200, 200)

        actions: list[Action] = [
            ClickAction(x=base_x, y=base_y, description="Select unit"),
            KeyPressAction(keys=["m"], description="Initiate move command"),
            ClickAction(x=base_x + offset_x, y=base_y + offset_y, description="Click destination tile"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected unit and issued movement command to nearby tile",
            actions=actions,
        )


class ScienceDecisionPrimitive(BasePrimitive):
    """
    Primitive for science/technology tree decisions.

    Handles tech research selection and related scientific choices.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        """Initialize primitive with optional VLM provider and custom prompt."""
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "research_select_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        """Generate science decision plan (with VLM or mock)."""
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)

            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock fallback
        seed = int(hashlib.md5(screenshot_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        tech_tree_x = 100 + rng.randint(0, 100)
        tech_tree_y = 200 + rng.randint(0, 100)

        actions: list[Action] = [
            ClickAction(x=tech_tree_x, y=tech_tree_y, description="Select technology"),
            KeyPressAction(keys=["esc"], description="Close tech tree"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected next technology from science tree",
            actions=actions,
        )


class CultureDecisionPrimitive(BasePrimitive):
    """
    Primitive for culture/civics tree decisions.

    Handles civic research selection and cultural policy choices.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        """Initialize primitive with optional VLM provider and custom prompt."""
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "culture_decision_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        """Generate culture decision plan (with VLM or mock)."""
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)

            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock fallback
        seed = int(hashlib.md5(screenshot_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        civic_tree_x = 640 + rng.randint(-50, 50)
        civic_tree_y = 360 + rng.randint(-50, 50)

        actions: list[Action] = [
            ClickAction(x=civic_tree_x, y=civic_tree_y, description="Select civic"),
            KeyPressAction(keys=["enter"], description="Confirm civic choice"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected next civic from culture tree",
            actions=actions,
        )


class PopupPrimitive(BasePrimitive):
    """
    Primitive for handling popups, notifications, and turn-end prompts.

    Handles: yes/no dialogs, info popups, next-turn button,
    research-select button, production-select button.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "popup_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)
            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock: press enter to dismiss popup / advance turn
        actions: list[Action] = [
            KeyPressAction(keys=["enter"], description="Confirm popup or next turn"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Dismissed popup by pressing enter",
            actions=actions,
        )


class ResearchSelectPrimitive(BasePrimitive):
    """
    Primitive for research/technology selection popup.

    When the research selection popup appears, picks the tech
    with the fewest remaining turns.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "research_select_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)
            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock: click on a research item
        seed = int(hashlib.md5(screenshot_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        research_x = 300 + rng.randint(-50, 50)
        research_y = 400 + rng.randint(-100, 100)

        actions: list[Action] = [
            ClickAction(x=research_x, y=research_y, description="Select research with fewest turns"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected research with fewest remaining turns",
            actions=actions,
        )


class CityProductionPrimitive(BasePrimitive):
    """
    Primitive for city production selection popup.

    When the production selection popup appears, picks the item
    with the fewest remaining turns.
    """

    def __init__(
        self,
        vlm_provider: BaseVLMProvider | None = None,
        custom_prompt: str | None = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "city_production_primitive"

    def generate_plan_and_action(self, screenshot_path: str) -> AgentPlan:
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)
            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock: click on a production item
        seed = int(hashlib.md5(screenshot_path.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        production_x = 700 + rng.randint(-50, 50)
        production_y = 400 + rng.randint(-100, 100)

        actions: list[Action] = [
            ClickAction(x=production_x, y=production_y, description="Select production with fewest turns"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected production item with fewest remaining turns",
            actions=actions,
        )
