"""
Main orchestrator for bbox-based evaluation.

Loads dataset, runs agent on each case, scores results, and produces an EvalReport.

Example:
    >>> from civStation.evaluation.evaluator.action_eval.bbox_eval import (
    ...     run_evaluation, MockAgentRunner,
    ... )
    >>> report = run_evaluation("dataset.jsonl", MockAgentRunner())
    >>> print(f"Success rate: {report.aggregate.strict_success_rate:.1%}")
"""

from __future__ import annotations

import logging

from .agents.base import AgentRunnerError, BaseAgentRunner
from .dataset_loader import load_dataset
from .schema import CaseResult, DatasetCase, EvalConfig, EvalReport
from .scorer import aggregate_results, select_best_gt_set

logger = logging.getLogger(__name__)


def evaluate_case(
    case: DatasetCase,
    runner: BaseAgentRunner,
    ignore_wait: bool = False,
) -> CaseResult:
    """Run agent on a single case and score the result."""
    try:
        response = runner.run_case(case)
    except AgentRunnerError as e:
        timed_out = "timed out" in str(e).lower()
        return CaseResult(
            case_id=case.case_id,
            error=str(e),
            timed_out=timed_out,
        )

    gt_action_lists = [action_set.actions for action_set in case.action_sets]

    best_seq, gt_idx = select_best_gt_set(
        gt_action_lists,
        list(response.actions),
        ignore_wait=ignore_wait,
    )

    return CaseResult(
        case_id=case.case_id,
        best_sequence=best_seq,
        agent_actions_count=len(response.actions),
        gt_set_index=gt_idx,
    )


def run_evaluation(
    dataset_path: str,
    runner: BaseAgentRunner,
    ignore_wait: bool = False,
    config: EvalConfig | None = None,
    verbose: bool = False,
) -> EvalReport:
    """
    Run full evaluation pipeline.

    Args:
        dataset_path: Path to JSONL dataset file.
        runner: Agent runner to use.
        ignore_wait: Whether to ignore wait actions in scoring.
        config: Optional config snapshot for the report.
        verbose: Whether to log per-case results.

    Returns:
        Complete EvalReport with aggregate metrics and per-case results.
    """
    cases = load_dataset(dataset_path)
    logger.info(f"Loaded {len(cases)} cases from {dataset_path}")

    results: list[CaseResult] = []
    for i, case in enumerate(cases):
        if verbose:
            logger.info(f"[{i + 1}/{len(cases)}] Evaluating case {case.case_id}...")

        result = evaluate_case(case, runner, ignore_wait=ignore_wait)
        results.append(result)

        if verbose:
            if result.error:
                logger.info(f"  -> ERROR: {result.error}")
            elif result.best_sequence:
                seq = result.best_sequence
                logger.info(
                    f"  -> success={seq.strict_success}, prefix={seq.prefix_len}, accuracy={seq.step_accuracy:.2f}"
                )

    aggregate = aggregate_results(results)

    if config is None:
        config = EvalConfig(dataset_path=dataset_path, ignore_wait=ignore_wait)

    return EvalReport(
        aggregate=aggregate,
        cases=results,
        config=config,
    )
