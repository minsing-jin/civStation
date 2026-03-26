"""
Bounding-box-based static screenshot evaluation framework.

This package evaluates game agents by comparing their predicted point-coordinate
actions against ground-truth bounding-box targets. It supports multiple acceptable
GT action sets per case, external agent integration via stdin/stdout, and built-in
VLM provider wrapping.

Quick Start:
    >>> from civStation.evaluation.evaluator.action_eval.bbox_eval import (
    ...     run_evaluation,
    ...     MockAgentRunner,
    ... )
    >>> report = run_evaluation("dataset.jsonl", MockAgentRunner())
    >>> print(f"Success: {report.aggregate.strict_success_rate:.1%}")

CLI:
    python -m civStation.evaluation.evaluator.action_eval.bbox_eval \\
        --dataset dataset.jsonl --provider mock --verbose
"""

# --- Agents ---
from .agents import (
    AgentRunnerError,
    BaseAgentRunner,
    BuiltinAgentRunner,
    MockAgentRunner,
    SubprocessAgentRunner,
)

# --- Dataset ---
from .dataset_loader import DatasetLoadError, load_dataset, validate_dataset

# --- Core evaluation ---
from .runner import evaluate_case, run_evaluation

# --- Schema ---
from .schema import (
    AgentResponse,
    AggregateMetrics,
    BBox,
    CaseResult,
    DatasetCase,
    EvalConfig,
    EvalReport,
    GTAction,
    GTActionSet,
    GTClickAction,
    GTDoubleClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    ImageSize,
    PerActionTypeMetric,
    SequenceResult,
    StepResult,
)

# --- Scoring ---
from .scorer import (
    aggregate_results,
    compare_sequence,
    compare_step,
    levenshtein_distance,
    levenshtein_similarity,
    select_best_gt_set,
)

__all__ = [
    # Core
    "evaluate_case",
    "run_evaluation",
    # Dataset
    "DatasetLoadError",
    "load_dataset",
    "validate_dataset",
    # Schema: GT
    "BBox",
    "GTAction",
    "GTActionSet",
    "GTClickAction",
    "GTDoubleClickAction",
    "GTDragAction",
    "GTKeyPressAction",
    "GTWaitAction",
    "ImageSize",
    # Schema: Dataset & Response
    "AgentResponse",
    "DatasetCase",
    # Schema: Results
    "AggregateMetrics",
    "CaseResult",
    "EvalConfig",
    "EvalReport",
    "PerActionTypeMetric",
    "SequenceResult",
    "StepResult",
    # Scoring
    "aggregate_results",
    "compare_sequence",
    "compare_step",
    "levenshtein_distance",
    "levenshtein_similarity",
    "select_best_gt_set",
    # Agents
    "AgentRunnerError",
    "BaseAgentRunner",
    "BuiltinAgentRunner",
    "MockAgentRunner",
    "SubprocessAgentRunner",
]
