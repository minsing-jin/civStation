# Checklist Draft

This file is a working draft for later transfer into the official NeurIPS or ICLR checklist form.

## Claims

- The paper claims a layered system architecture, not state-of-the-art game performance.
- All numeric claims in the current draft are limited to:
  - repository structure counts
  - MCP tool count
  - primitive count
  - targeted test counts
  - sample bbox evaluator sanity-check results

## Limitations

- No live-agent benchmark results are reported yet.
- No ablation study is reported yet.
- No human-subject or operator-efficiency study is reported yet.
- The current quantitative evidence is software-validation oriented.

## Reproducibility

- Public repository available: `https://github.com/minsing-jin/civStation`
- Core architecture documented in README and architecture docs
- Evaluation framework exposed in public code
- Sample datasets/fixtures available in `tests/evaluator/civ6_eval/fixtures/`
- Current local blocker: no LaTeX toolchain installed on this machine

## Ethics / Broader Impact

- The project is a computer-use agent that can act on a host machine.
- Main risk: misuse of live desktop control.
- Main mitigation in the current design:
  - explicit pause/resume/stop controls
  - strategy overrides
  - primitive overrides
  - multiple operator-facing control surfaces

## Compute / Resources

- No expensive large-scale training compute is claimed.
- No paid API evaluation results are included in the current draft.
- Current validation used local tests and local fixture-based evaluation only.

## Missing Before Submission

1. Add benchmark results with real agent policies.
2. Add at least one ablation on layered decomposition.
3. Add failure-case analysis with screenshots.
4. Convert this note into the official conference checklist format.
