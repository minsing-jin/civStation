"""
Pydantic models for bounding-box-based static screenshot evaluation.

Defines GT action types with bbox targets, dataset cases, and result/report models.
Agent predictions reuse the existing Action union from schema.py (point coordinates).

Example:
    >>> from computer_use_test.evaluation.evaluator.action_eval.bbox_eval import BBox, GTClickAction
    >>> bbox = BBox(x_min=80, y_min=180, x_max=120, y_max=220)
    >>> bbox.contains_point(100, 200)
    True
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

from computer_use_test.agent.models.schema import Action

# ==================== Bounding Box ====================


class BBox(BaseModel):
    """
    Axis-aligned bounding box in normalized coordinates (0-1000).

    Attributes:
        x_min: Left edge (inclusive).
        y_min: Top edge (inclusive).
        x_max: Right edge (inclusive).
        y_max: Bottom edge (inclusive).
    """

    x_min: int = Field(..., ge=0)
    y_min: int = Field(..., ge=0)
    x_max: int = Field(..., ge=0)
    y_max: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> BBox:
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be <= x_max ({self.x_max})")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be <= y_max ({self.y_max})")
        return self

    def contains_point(self, x: int, y: int) -> bool:
        """Check whether (x, y) lies inside the bbox (inclusive)."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def center(self) -> tuple[float, float]:
        """Return (cx, cy) center of the bbox."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    def distance_to_center(self, x: int, y: int) -> float:
        """Euclidean distance from (x, y) to the bbox center, normalized by bbox diagonal."""
        cx, cy = self.center()
        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        diagonal = math.sqrt(dx * dx + dy * dy) if (dx > 0 or dy > 0) else 1.0
        return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / diagonal


# ==================== GT Action Types ====================


class GTClickAction(BaseModel):
    """Ground-truth click with a target bounding box."""

    type: Literal["click"] = "click"
    target_bbox: BBox
    button: str = "left"


class GTDoubleClickAction(BaseModel):
    """Ground-truth double-click with a target bounding box."""

    type: Literal["double_click"] = "double_click"
    target_bbox: BBox
    button: str = "left"


class GTDragAction(BaseModel):
    """Ground-truth drag with start and end bounding boxes."""

    type: Literal["drag"] = "drag"
    start_bbox: BBox
    end_bbox: BBox
    button: str = "left"


class GTKeyPressAction(BaseModel):
    """Ground-truth key press."""

    type: Literal["press"] = "press"
    keys: list[str]


class GTWaitAction(BaseModel):
    """Ground-truth wait action."""

    type: Literal["wait"] = "wait"
    duration: float = 1.0


GTAction = Annotated[
    GTClickAction | GTDoubleClickAction | GTDragAction | GTKeyPressAction | GTWaitAction,
    Field(discriminator="type"),
]


# ==================== Dataset ====================


class ImageSize(BaseModel):
    """Image dimensions."""

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class GTActionSet(BaseModel):
    """One acceptable ground-truth action sequence."""

    actions: list[GTAction]


class DatasetCase(BaseModel):
    """A single evaluation case loaded from JSONL."""

    case_id: str
    instruction: str
    screenshot_path: str
    image_size: ImageSize
    action_sets: list[GTActionSet] = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==================== Agent Response ====================


class AgentResponse(BaseModel):
    """Response returned by an agent for one case."""

    actions: list[Action] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


# ==================== Result Models ====================


class StepResult(BaseModel):
    """Per-step comparison result."""

    step_index: int
    correct: bool
    gt_type: str
    pred_type: str | None = None
    distance_to_center: float | None = None
    levenshtein_similarity: float | None = None


class SequenceResult(BaseModel):
    """Result of comparing a predicted sequence against one GT action set."""

    strict_success: bool
    prefix_len: int
    step_accuracy: float
    step_results: list[StepResult]


class CaseResult(BaseModel):
    """Evaluation result for a single dataset case."""

    case_id: str
    best_sequence: SequenceResult | None = None
    agent_actions_count: int = 0
    gt_set_index: int | None = None
    error: str | None = None
    timed_out: bool = False


class PerActionTypeMetric(BaseModel):
    """Metrics broken down by action type."""

    action_type: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0


class AggregateMetrics(BaseModel):
    """Aggregated metrics across all cases."""

    total_cases: int = 0
    strict_success_rate: float = 0.0
    avg_step_accuracy: float = 0.0
    avg_prefix_len: float = 0.0
    error_count: int = 0
    timeout_count: int = 0
    per_action_type: list[PerActionTypeMetric] = Field(default_factory=list)


class EvalConfig(BaseModel):
    """Configuration snapshot for reproducibility."""

    dataset_path: str = ""
    agent_cmd: str | None = None
    provider: str | None = None
    model: str | None = None
    timeout: float = 10.0
    ignore_wait: bool = False


class EvalReport(BaseModel):
    """Complete evaluation report."""

    aggregate: AggregateMetrics
    cases: list[CaseResult]
    config: EvalConfig
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
