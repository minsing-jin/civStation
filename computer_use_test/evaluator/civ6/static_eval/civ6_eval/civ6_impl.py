"""
Civilization VI specific implementation of the Static Primitive Evaluator.

This module contains:
- Seven distinct primitive implementations for different Civ6 game scenarios
- Mock router for primitive selection based on screenshot filename
- Enhanced evaluator with 5-pixel coordinate tolerance
"""

import hashlib
import os
import random
from typing import List, Optional

from computer_use_test.agent.models.schema import (
    Action,
    AgentPlan,
    ClickAction,
    DragAction,
    KeyPressAction,
)
from computer_use_test.evaluator.civ6.static_eval.interfaces import (
    BaseEvaluator,
    BasePrimitive,
    EvalResult,
    GroundTruth,
    PrimitiveRouter,
    within_tolerance,
)
from computer_use_test.utils.provider.base import BaseVLMProvider
from computer_use_test.utils.prompts.civ6_prompts import get_primitive_prompt


# ==================== Primitive Implementations ====================


class UnitOpsPrimitive(BasePrimitive):
    """
    Primitive for unit operations in Civilization VI.

    Handles unit movement, combat, fortification, and other unit-related actions.
    """

    def __init__(
        self,
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
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

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
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

        actions: List[Action] = [
            ClickAction(x=base_x, y=base_y, description="Select unit"),
            KeyPressAction(keys=["m"], description="Initiate move command"),
            ClickAction(x=base_x + offset_x, y=base_y + offset_y, description="Click destination tile"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected unit and issued movement command to nearby tile",
            actions=actions,
        )


class CountryMayerPrimitive(BasePrimitive):
    """
    Primitive for city (mayor) management operations.

    Handles production queue, citizen assignment, building purchases, etc.
    """

    def __init__(
        self,
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        """Initialize primitive with optional VLM provider and custom prompt."""
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "country_mayer_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
        """Generate city management plan (with VLM or mock)."""
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

        city_panel_x = 800 + rng.randint(-50, 50)
        city_panel_y = 600 + rng.randint(-50, 50)

        actions: List[Action] = [
            ClickAction(x=city_panel_x, y=city_panel_y, description="Open city panel"),
            ClickAction(x=900, y=700, description="Select production item"),
            KeyPressAction(keys=["enter"], description="Confirm production choice"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Opened city management and adjusted production queue",
            actions=actions,
        )


class ScienceDecisionPrimitive(BasePrimitive):
    """
    Primitive for science/technology tree decisions.

    Handles tech research selection and related scientific choices.
    """

    def __init__(
        self,
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        """Initialize primitive with optional VLM provider and custom prompt."""
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "science_decision_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
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

        actions: List[Action] = [
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
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        """Initialize primitive with optional VLM provider and custom prompt."""
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "culture_decision_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
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

        actions: List[Action] = [
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
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "popup_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
        if self.vlm_provider:
            prompt = self.custom_prompt or get_primitive_prompt(self.name)
            return self.vlm_provider.call_and_parse(
                prompt=prompt,
                image_path=screenshot_path,
                primitive_name=self.name,
            )

        # Mock: press enter to dismiss popup / advance turn
        actions: List[Action] = [
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
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "research_select_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
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

        actions: List[Action] = [
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
        vlm_provider: Optional[BaseVLMProvider] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.vlm_provider = vlm_provider
        self.custom_prompt = custom_prompt

    @property
    def name(self) -> str:
        return "city_production_primitive"

    def generate_plan(self, screenshot_path: str) -> AgentPlan:
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

        actions: List[Action] = [
            ClickAction(x=production_x, y=production_y, description="Select production with fewest turns"),
        ]

        return AgentPlan(
            primitive_name=self.name,
            reasoning="Mock: Selected production item with fewest remaining turns",
            actions=actions,
        )


# ==================== Router Implementation ====================


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
            return "science_decision_primitive"
        if "culture" in filename or "civic" in filename:
            return "culture_decision_primitive"
        if "unit" in filename:
            return "unit_ops_primitive"

        # Default fallback
        return "popup_primitive"


# ==================== Evaluator Implementation ====================


class Civ6StaticEvaluator(BaseEvaluator):
    """
    Enhanced evaluator with 5-pixel coordinate tolerance.

    Compares predicted actions against ground truth with appropriate
    tolerances for different action types.
    """

    COORD_TOLERANCE = 5

    def _compare_actions(self, gt_action: Action, pred_action: Action) -> bool:
        """
        Compare two actions with appropriate tolerance.

        Args:
            gt_action: Ground truth action
            pred_action: Predicted action

        Returns:
            True if actions match within tolerance, False otherwise
        """
        # Type must match exactly
        if type(gt_action) != type(pred_action):
            return False

        # ClickAction: allow coordinate tolerance
        if isinstance(gt_action, ClickAction):
            assert isinstance(pred_action, ClickAction)  # Type narrowing
            return (
                within_tolerance(gt_action.x, pred_action.x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.y, pred_action.y, self.COORD_TOLERANCE)
                and gt_action.button == pred_action.button
            )

        # DragAction: allow coordinate tolerance for all coordinates
        elif isinstance(gt_action, DragAction):
            assert isinstance(pred_action, DragAction)
            return (
                within_tolerance(gt_action.start_x, pred_action.start_x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.start_y, pred_action.start_y, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.end_x, pred_action.end_x, self.COORD_TOLERANCE)
                and within_tolerance(gt_action.end_y, pred_action.end_y, self.COORD_TOLERANCE)
            )

        # KeyPressAction: exact match required for keys
        elif isinstance(gt_action, KeyPressAction):
            assert isinstance(pred_action, KeyPressAction)
            return gt_action.keys == pred_action.keys

        # Unknown action type
        return False

    def _compare(self, gt: GroundTruth, selected_prim: str, plan: AgentPlan) -> EvalResult:
        """
        Compare predicted plan against ground truth.

        Args:
            gt: Ground truth data
            selected_prim: Name of selected primitive
            plan: Generated action plan

        Returns:
            Evaluation result with match status
        """
        # 1. Check primitive selection
        prim_match = gt.expected_primitive == selected_prim

        # 2. Check action sequence
        actions_match = True
        if len(gt.expected_actions) != len(plan.actions):
            actions_match = False
        else:
            for gt_act, pred_act in zip(gt.expected_actions, plan.actions):
                if not self._compare_actions(gt_act, pred_act):
                    actions_match = False
                    break

        return EvalResult(
            screenshot_path=gt.screenshot_path,
            selected_primitive=selected_prim,
            primitive_match=prim_match,
            action_sequence_match=actions_match,
            levenshtein_distance=0,  # TODO: Implement edit distance for partial credit
        )
