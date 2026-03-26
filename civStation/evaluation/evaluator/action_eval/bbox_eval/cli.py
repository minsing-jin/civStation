"""
CLI entry point for bbox-based static screenshot evaluation.

Usage:
    # External agent:
    python -m civStation.evaluation.evaluator.action_eval.bbox_eval \
        --dataset path/to/dataset.jsonl \
        --agent-cmd "python my_agent.py" \
        --timeout 10 \
        --output results.json

    # Built-in VLM provider:
    python -m civStation.evaluation.evaluator.action_eval.bbox_eval \
        --dataset path/to/dataset.jsonl \
        --provider claude \
        --model claude-4-5-sonnet-20241022

    # Mock provider for testing:
    python -m civStation.evaluation.evaluator.action_eval.bbox_eval \
        --dataset path/to/dataset.jsonl \
        --provider mock --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bbox-based static screenshot evaluation for game agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL dataset file.",
    )

    # Agent source (mutually exclusive)
    agent_group = parser.add_mutually_exclusive_group(required=True)
    agent_group.add_argument(
        "--agent-cmd",
        help="Shell command to run the external agent (stdin/stdout JSON protocol).",
    )
    agent_group.add_argument(
        "--provider",
        help="Built-in VLM provider name (claude, gemini, gpt, mock).",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier for built-in provider.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-case timeout in seconds for subprocess agent (default: 10).",
    )
    parser.add_argument(
        "--ignore-wait",
        action="store_true",
        help="Ignore wait actions when scoring.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON evaluation report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose per-case logging.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from .schema import EvalConfig

    # Build runner
    if args.agent_cmd:
        from .agents import SubprocessAgentRunner

        runner = SubprocessAgentRunner(cmd=args.agent_cmd, timeout=args.timeout)
        config = EvalConfig(
            dataset_path=args.dataset,
            agent_cmd=args.agent_cmd,
            timeout=args.timeout,
            ignore_wait=args.ignore_wait,
        )
    else:
        from civStation.utils.llm_provider import create_provider

        from .agents import BuiltinAgentRunner

        provider = create_provider(args.provider, model=args.model)
        runner = BuiltinAgentRunner(provider=provider)
        config = EvalConfig(
            dataset_path=args.dataset,
            provider=args.provider,
            model=args.model,
            ignore_wait=args.ignore_wait,
        )

    # Run evaluation
    from .runner import run_evaluation

    report = run_evaluation(
        dataset_path=args.dataset,
        runner=runner,
        ignore_wait=args.ignore_wait,
        config=config,
        verbose=args.verbose,
    )

    # Output
    report_json = report.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)
        print(f"Report written to {args.output}")
    else:
        print(report_json)

    # Summary to stderr
    agg = report.aggregate
    print(
        f"\n--- Summary ---\n"
        f"Total cases: {agg.total_cases}\n"
        f"Strict success rate: {agg.strict_success_rate:.1%}\n"
        f"Avg step accuracy: {agg.avg_step_accuracy:.1%}\n"
        f"Avg prefix length: {agg.avg_prefix_len:.1f}\n"
        f"Errors: {agg.error_count}, Timeouts: {agg.timeout_count}",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
