# 🧪 Validation Artifacts

## 📚 Index

- [📁 Files](#-files)
- [🎯 Purpose](#-purpose)

This folder stores small validation artifacts referenced by the paper draft.

## 📁 Files

- `mock_bbox_eval.json`
  - Produced by running `bbox_eval` with `MockAgentRunner`
  - Expected to perform poorly on the fixture dataset
- `perfect_bbox_eval.json`
  - Produced by a reference runner that clicks bbox centers and reproduces the first ground-truth action set
  - Expected to score perfectly on the fixture dataset
- `latency_tradeoff_summary.md`
  - Summarizes the saved rough latency benchmark report under `tests/rough_test/reports/`
  - Focuses on preprocessing, downscaling, background padding, and prompt/range effects
- `validation_manifest.json`
  - Produced by `scripts/generate_paper_artifacts.py`
  - Records the fixture dataset path and the aggregate metrics for the mock and perfect-reference evaluator runs
- `cross_model_bbox_benchmark.json` / `cross_model_bbox_benchmark.md`
  - Produced by `scripts/run_paper_cross_model_benchmark.py`
  - Real provider benchmark on the synthetic UI bbox dataset
- `gpt_tradeoff_benchmark.json` / `gpt_tradeoff_benchmark.md`
  - Produced by `scripts/run_paper_gemini_tradeoff.py`
  - Real provider trade-off benchmark for preprocessing, prompt/range choices, and quality gating

## 🎯 Purpose

These are not competitive benchmark results. They are supporting artifacts for the paper draft:

- the bbox JSON files show that the evaluation pipeline distinguishes deliberately mismatched actions from geometrically correct reference actions on the sample fixture dataset
- the latency summary captures exploratory speed trade-offs for image preprocessing and prompt/range choices
- the validation manifest ties the generated evaluator outputs back to the script and dataset used
- the cross-model benchmark captures model-dependent behavior on a small synthetic UI benchmark
- the GPT trade-off benchmark provides a reproducible latency/quality artifact on synthetic assets
