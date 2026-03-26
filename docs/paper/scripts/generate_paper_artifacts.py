from __future__ import annotations

import json
from pathlib import Path

from civStation.agent.models.schema import ClickAction, DoubleClickAction, DragAction, KeyPressAction, WaitAction
from civStation.evaluation.evaluator.action_eval.bbox_eval import (
    AgentResponse,
    BaseAgentRunner,
    GTClickAction,
    GTDoubleClickAction,
    GTDragAction,
    GTKeyPressAction,
    GTWaitAction,
    MockAgentRunner,
    load_dataset,
    run_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "tests/evaluator/civ6_eval/fixtures/sample_bbox_dataset.jsonl"
RESULTS_DIR = PAPER_ROOT / "arxiv/results"
LATENCY_SUMMARY = RESULTS_DIR / "latency_tradeoff_summary.md"
LATENCY_SOURCE = REPO_ROOT / "tests/rough_test/reports/vlm_policy_speed_report_allmodes_20260305_231703.md"


class PerfectAgentRunner(BaseAgentRunner):
    """Reference runner that exactly hits the first GT action set."""

    def __init__(self, dataset_cases):
        self._cases = {case.case_id: case for case in dataset_cases}

    def run_case(self, case):
        gt_actions = case.action_sets[0].actions
        pred_actions = []
        for gt in gt_actions:
            if isinstance(gt, GTClickAction):
                cx, cy = gt.target_bbox.center()
                pred_actions.append(ClickAction(type="click", x=int(cx), y=int(cy), button=gt.button))
            elif isinstance(gt, GTDoubleClickAction):
                cx, cy = gt.target_bbox.center()
                pred_actions.append(DoubleClickAction(type="double_click", x=int(cx), y=int(cy), button=gt.button))
            elif isinstance(gt, GTDragAction):
                sx, sy = gt.start_bbox.center()
                ex, ey = gt.end_bbox.center()
                pred_actions.append(
                    DragAction(type="drag", start_x=int(sx), start_y=int(sy), end_x=int(ex), end_y=int(ey))
                )
            elif isinstance(gt, GTKeyPressAction):
                pred_actions.append(KeyPressAction(type="press", keys=gt.keys))
            elif isinstance(gt, GTWaitAction):
                pred_actions.append(WaitAction(type="wait", duration=gt.duration))
        return AgentResponse(actions=pred_actions, meta={"agent": "perfect_reference"})


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_dataset(DATASET)

    mock_report = run_evaluation(dataset_path=str(DATASET), runner=MockAgentRunner(), verbose=False)
    perfect_report = run_evaluation(dataset_path=str(DATASET), runner=PerfectAgentRunner(cases), verbose=False)

    mock_path = RESULTS_DIR / "mock_bbox_eval.json"
    perfect_path = RESULTS_DIR / "perfect_bbox_eval.json"
    manifest_path = RESULTS_DIR / "validation_manifest.json"

    write_json(mock_path, json.loads(mock_report.model_dump_json(indent=2)))
    write_json(perfect_path, json.loads(perfect_report.model_dump_json(indent=2)))

    manifest = {
        "dataset": str(DATASET.relative_to(REPO_ROOT)),
        "outputs": {
            "mock_bbox_eval": str(mock_path.relative_to(REPO_ROOT)),
            "perfect_bbox_eval": str(perfect_path.relative_to(REPO_ROOT)),
            "latency_tradeoff_summary": str(LATENCY_SUMMARY.relative_to(REPO_ROOT)),
        },
        "latency_source_report": str(LATENCY_SOURCE.relative_to(REPO_ROOT)),
        "mock_aggregate": json.loads(mock_report.aggregate.model_dump_json(indent=2)),
        "perfect_aggregate": json.loads(perfect_report.aggregate.model_dump_json(indent=2)),
    }
    write_json(manifest_path, manifest)

    print(f"Wrote {mock_path}")
    print(f"Wrote {perfect_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
