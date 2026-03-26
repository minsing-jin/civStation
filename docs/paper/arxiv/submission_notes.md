# Submission Notes

## Current status

- The manuscript in `main.tex` is now shaped more like a top-conference system paper than a software-note or review article.
- The draft emphasizes:
  - problem formulation
  - explicit contributions
  - layered method/system design
  - evaluation protocol and current validation
  - limitations, reproducibility, and broader impact
- The paper now also includes:
  - exploratory latency trade-off discussion for image preprocessing and prompt/range choices
  - a stronger benchmark-substrate argument for long-horizon agent and human comparisons

## Official venue constraints checked

- `NeurIPS 2025` main text limit: 9 content pages, plus references and checklist
- `ICLR 2025` main text limit: 6-10 pages, plus references and optional appendices
- Both venues allow arXiv/preprint posting
- Both venues require careful handling of LLM use, ethics, and submission policy

## What is still missing for a competitive NeurIPS/ICLR oral submission

- Real benchmark results on live or static tasks, not only architecture description
- Strong quantitative comparisons against baselines
- Ablations showing why the layered decomposition matters
- Human-in-the-loop intervention studies or operator-efficiency results
- Failure analysis with representative qualitative examples
- A proper appendix and venue-specific checklist
- Final anonymized venue-specific style conversion

## Immediate next paper upgrades

1. Add a real experiment section with:
   - static evaluator results
   - baseline prompts or monolithic-agent comparison
   - MCP/no-MCP or primitive/no-primitive ablations
2. Add qualitative case studies with screenshots and intervention traces
3. Convert from current IEEE-style preprint to:
   - NeurIPS style with checklist
   - ICLR style with appendix and anonymization
4. Compile PDF after LaTeX toolchain installation

## Codex / skill note

- No restart is needed to continue work in the current session.
- Restart helps only if you want future sessions to auto-discover newly installed skills without manually opening them.
