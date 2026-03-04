"""
Scoring algorithms for bounding-box-based evaluation.

Compares agent-predicted point-coordinate actions against GT bounding-box actions.

Key functions:
    - compare_step():        Single GT action vs predicted action.
    - compare_sequence():    Full action sequence vs one GT action_set.
    - select_best_gt_set():  Pick the best-matching GT set from multiple.
    - aggregate_results():   Compute aggregate metrics across cases.
"""

from __future__ import annotations

from collections import defaultdict

from computer_use_test.agent.models.schema import (
    Action,
    ClickAction,
    DoubleClickAction,
    DragAction,
    KeyPressAction,
    WaitAction,
)

from .schema import (
    AggregateMetrics,
    CaseResult,
    GTAction,
    GTClickAction,
    GTDoubleClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    PerActionTypeMetric,
    SequenceResult,
    StepResult,
)

# ==================== Levenshtein ====================


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein edit distance between two lists of strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def levenshtein_similarity(a: list[str], b: list[str]) -> float:
    """Return similarity in [0, 1] based on Levenshtein distance."""
    if not a and not b:
        return 1.0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len


# ==================== Step Comparison ====================


def compare_step(gt: GTAction, pred: Action, ignore_wait: bool = False) -> StepResult:
    """Compare a single GT action against a predicted action."""

    # --- Click ---
    if isinstance(gt, GTClickAction):
        if not isinstance(pred, ClickAction):
            return StepResult(step_index=0, correct=False, gt_type="click", pred_type=_pred_type(pred))
        inside = gt.target_bbox.contains_point(pred.x, pred.y)
        button_ok = gt.button == pred.button
        dist = gt.target_bbox.distance_to_center(pred.x, pred.y)
        return StepResult(
            step_index=0,
            correct=inside and button_ok,
            gt_type="click",
            pred_type="click",
            distance_to_center=dist,
        )

    # --- DoubleClick ---
    if isinstance(gt, GTDoubleClickAction):
        if not isinstance(pred, DoubleClickAction):
            return StepResult(step_index=0, correct=False, gt_type="double_click", pred_type=_pred_type(pred))
        inside = gt.target_bbox.contains_point(pred.x, pred.y)
        button_ok = gt.button == pred.button
        dist = gt.target_bbox.distance_to_center(pred.x, pred.y)
        return StepResult(
            step_index=0,
            correct=inside and button_ok,
            gt_type="double_click",
            pred_type="double_click",
            distance_to_center=dist,
        )

    # --- Drag ---
    if isinstance(gt, GTDragAction):
        if not isinstance(pred, DragAction):
            return StepResult(step_index=0, correct=False, gt_type="drag", pred_type=_pred_type(pred))
        start_ok = gt.start_bbox.contains_point(pred.start_x, pred.start_y)
        end_ok = gt.end_bbox.contains_point(pred.end_x, pred.end_y)
        return StepResult(
            step_index=0,
            correct=start_ok and end_ok,
            gt_type="drag",
            pred_type="drag",
        )

    # --- KeyPress ---
    if isinstance(gt, GTKeyPressAction):
        if not isinstance(pred, KeyPressAction):
            return StepResult(step_index=0, correct=False, gt_type="press", pred_type=_pred_type(pred))
        keys_match = gt.keys == pred.keys
        sim = levenshtein_similarity(gt.keys, pred.keys)
        return StepResult(
            step_index=0,
            correct=keys_match,
            gt_type="press",
            pred_type="press",
            levenshtein_similarity=sim,
        )

    # --- Wait ---
    if isinstance(gt, GTWaitAction):
        if ignore_wait:
            return StepResult(step_index=0, correct=True, gt_type="wait", pred_type=_pred_type(pred))
        if not isinstance(pred, WaitAction):
            return StepResult(step_index=0, correct=False, gt_type="wait", pred_type=_pred_type(pred))
        return StepResult(step_index=0, correct=True, gt_type="wait", pred_type="wait")

    return StepResult(step_index=0, correct=False, gt_type="unknown", pred_type=_pred_type(pred))


def _pred_type(pred: Action) -> str:
    """Extract type string from a predicted action."""
    if isinstance(pred, ClickAction):
        return "click"
    if isinstance(pred, DoubleClickAction):
        return "double_click"
    if isinstance(pred, DragAction):
        return "drag"
    if isinstance(pred, KeyPressAction):
        return "press"
    if isinstance(pred, WaitAction):
        return "wait"
    return "unknown"


# ==================== Sequence Comparison ====================


def _filter_wait_actions_gt(gt_actions: list[GTAction], ignore_wait: bool) -> list[GTAction]:
    """Filter out GTWaitActions if ignore_wait is True."""
    if not ignore_wait:
        return gt_actions
    return [a for a in gt_actions if not isinstance(a, GTWaitAction)]


def _filter_wait_actions_pred(pred_actions: list[Action], ignore_wait: bool) -> list[Action]:
    """Filter out WaitActions if ignore_wait is True."""
    if not ignore_wait:
        return pred_actions
    return [a for a in pred_actions if not isinstance(a, WaitAction)]


def compare_sequence(
    gt_actions: list[GTAction],
    pred_actions: list[Action],
    ignore_wait: bool = False,
) -> SequenceResult:
    """Compare a predicted action sequence against a single GT action set."""
    gt_filtered = _filter_wait_actions_gt(gt_actions, ignore_wait)
    pred_filtered = _filter_wait_actions_pred(pred_actions, ignore_wait)

    if not gt_filtered:
        return SequenceResult(
            strict_success=len(pred_filtered) == 0,
            prefix_len=0,
            step_accuracy=1.0 if len(pred_filtered) == 0 else 0.0,
            step_results=[],
        )

    step_results: list[StepResult] = []
    paired_len = min(len(gt_filtered), len(pred_filtered))

    for i in range(paired_len):
        result = compare_step(gt_filtered[i], pred_filtered[i], ignore_wait=False)
        step_results.append(StepResult(**{**result.model_dump(), "step_index": i}))

    # Compute prefix_len (consecutive correct from index 0)
    prefix_len = 0
    for sr in step_results:
        if sr.correct:
            prefix_len += 1
        else:
            break

    matched = sum(1 for sr in step_results if sr.correct)
    step_accuracy = matched / len(gt_filtered)
    strict_success = len(gt_filtered) == len(pred_filtered) and all(sr.correct for sr in step_results)

    return SequenceResult(
        strict_success=strict_success,
        prefix_len=prefix_len,
        step_accuracy=step_accuracy,
        step_results=step_results,
    )


# ==================== Best GT Set Selection ====================


def select_best_gt_set(
    gt_action_sets: list[list[GTAction]],
    pred_actions: list[Action],
    ignore_wait: bool = False,
) -> tuple[SequenceResult, int]:
    """
    Compare pred against each GT action set and return the best match.

    Selection order: strict_success (desc) > prefix_len (desc) > step_accuracy (desc).

    Returns:
        (best_sequence_result, index_of_best_gt_set)
    """
    best_result: SequenceResult | None = None
    best_index = 0

    for i, gt_actions in enumerate(gt_action_sets):
        result = compare_sequence(gt_actions, pred_actions, ignore_wait)
        if best_result is None or _is_better(result, best_result):
            best_result = result
            best_index = i

    assert best_result is not None
    return best_result, best_index


def _is_better(a: SequenceResult, b: SequenceResult) -> bool:
    """Return True if result `a` is strictly better than `b`."""
    if a.strict_success != b.strict_success:
        return a.strict_success
    if a.prefix_len != b.prefix_len:
        return a.prefix_len > b.prefix_len
    return a.step_accuracy > b.step_accuracy


# ==================== Aggregation ====================


def aggregate_results(cases: list[CaseResult]) -> AggregateMetrics:
    """Compute aggregate metrics from a list of case results."""
    total = len(cases)
    if total == 0:
        return AggregateMetrics()

    success_count = 0
    accuracy_sum = 0.0
    prefix_sum = 0.0
    error_count = 0
    timeout_count = 0
    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    scored_count = 0
    for case in cases:
        if case.error:
            error_count += 1
            continue
        if case.timed_out:
            timeout_count += 1
            continue
        if case.best_sequence is None:
            continue

        scored_count += 1
        seq = case.best_sequence
        if seq.strict_success:
            success_count += 1
        accuracy_sum += seq.step_accuracy
        prefix_sum += seq.prefix_len

        for sr in seq.step_results:
            type_stats[sr.gt_type]["total"] += 1
            if sr.correct:
                type_stats[sr.gt_type]["correct"] += 1

    per_action = []
    for atype, stats in sorted(type_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        per_action.append(
            PerActionTypeMetric(action_type=atype, total=stats["total"], correct=stats["correct"], accuracy=acc)
        )

    return AggregateMetrics(
        total_cases=total,
        strict_success_rate=success_count / scored_count if scored_count > 0 else 0.0,
        avg_step_accuracy=accuracy_sum / scored_count if scored_count > 0 else 0.0,
        avg_prefix_len=prefix_sum / scored_count if scored_count > 0 else 0.0,
        error_count=error_count,
        timeout_count=timeout_count,
        per_action_type=per_action,
    )
