from __future__ import annotations

import json
from pathlib import Path

from civStation.evaluation.evaluator.action_eval.bbox_eval import EvalConfig, run_evaluation
from civStation.evaluation.evaluator.action_eval.bbox_eval.agents import BuiltinAgentRunner
from civStation.utils.llm_provider import create_provider

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
DATASET = PAPER_ROOT / "arxiv/benchmarks/synthetic_ui/synthetic_bbox_dataset.jsonl"
RESULTS_DIR = PAPER_ROOT / "arxiv/results"


class ResolvedBuiltinAgentRunner(BuiltinAgentRunner):
    def __init__(self, provider, dataset_path: Path, primitive_name: str = "eval"):
        super().__init__(provider=provider, primitive_name=primitive_name)
        self.dataset_path = dataset_path

    def run_case(self, case):
        original = case.screenshot_path
        screenshot = Path(original)
        if not screenshot.is_absolute():
            case = case.model_copy(update={"screenshot_path": str((self.dataset_path.parent / screenshot).resolve())})
        return super().run_case(case)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# Cross-model bbox benchmark",
        "",
        f"- Dataset: `{DATASET.relative_to(REPO_ROOT)}`",
        "",
        "| provider | model | strict_success_rate | avg_step_accuracy | avg_prefix_len | errors | timeouts |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['provider']} | {row['model']} | error | error | error | 1 | 0 |")
            continue
        agg = row["aggregate"]
        lines.append(
            f"| {row['provider']} | {row['model']} | {agg['strict_success_rate']:.3f} | "
            f"{agg['avg_step_accuracy']:.3f} | {agg['avg_prefix_len']:.3f} | "
            f"{agg['error_count']} | {agg['timeout_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        {"provider": "gpt", "model": "gpt-4o-mini"},
        {"provider": "gpt", "model": "gpt-4o"},
        {"provider": "claude", "model": "claude-sonnet-4-20250514"},
    ]

    rows: list[dict] = []
    for cfg in configs:
        provider_name = cfg["provider"]
        model_name = cfg["model"]
        print(f"Running provider={provider_name} model={model_name}")
        try:
            provider = create_provider(provider_name, model=model_name)
            runner = ResolvedBuiltinAgentRunner(provider=provider, dataset_path=DATASET)
            report = run_evaluation(
                dataset_path=str(DATASET),
                runner=runner,
                verbose=True,
                config=EvalConfig(dataset_path=str(DATASET), provider=provider_name, model=model_name),
            )
            rows.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "aggregate": json.loads(report.aggregate.model_dump_json()),
                    "cases": json.loads(report.model_dump_json()).get("cases", []),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "error": str(exc),
                }
            )

    json_path = RESULTS_DIR / "cross_model_bbox_benchmark.json"
    md_path = RESULTS_DIR / "cross_model_bbox_benchmark.md"
    write_json(json_path, {"dataset": str(DATASET.relative_to(REPO_ROOT)), "results": rows})
    md_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
