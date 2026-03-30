# Paper Index

This directory contains all paper-related materials for the CivStation project.

## Recommended public artifact

If publishing a single document now, use:

- `technical_report/main.pdf`

This is the most honest representation of the current project state and the best candidate for a public archival upload.

## Documents

- `technical_report/`
  - Main experimental systems write-up
  - Best candidate for arXiv or public archival upload
- `research_memo/`
  - Failure-oriented memo explaining why the project is not yet a validated benchmark
- `arxiv/`
  - Heavier paper-style draft and supporting results
- `neurips/`
  - NeurIPS-formatted build of the shared manuscript
- `iclr/`
  - ICLR-formatted build of the shared manuscript

## Shared sources

- `shared/civstation_body.tex`
  - Shared manuscript body used by the paper-style venue builds

## Scripts

- `scripts/generate_paper_artifacts.py`
- `scripts/generate_synthetic_benchmark_assets.py`
- `scripts/run_paper_cross_model_benchmark.py`
- `scripts/run_paper_gemini_tradeoff.py`

## Recommendation

At the current maturity level:

1. keep `technical_report` as the primary public document
2. keep `research_memo` as the companion honesty document
3. treat `arxiv/`, `neurips/`, and `iclr/` as exploratory paper-style packaging rather than definitive claims of benchmark maturity
4. prefer the upload bundle under `arxiv_upload_technical_report/` when preparing an archival submission

## Ready-to-upload bundle

- directory: `arxiv_upload_technical_report/`
- source archive: `arxiv_upload_technical_report.tar.gz`
