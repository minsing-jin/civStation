# Exploratory Latency Trade-off Summary

Source artifact:

- `tests/rough_test/reports/vlm_policy_speed_report_allmodes_20260305_231703.md`

Setup from the source report:

- Model: `gemini-3-flash-preview`
- Screenshot: `tests/rough_test/screenshot.png`
- Prompt variants: `long_prompt`, `concise_prompt`
- Size modes: `raw`, `compressed`, `downscale_restore`, `compressed_downscale_restore`
- Background modes: `none`, `color`
- Normalizing ranges: `250`, `500`, `1000`
- Runs per variant: `2`

Key observations from the saved report:

- Mean latency by normalizing range:
  - `250`: `4.167s`
  - `500`: `4.392s`
  - `1000`: `4.579s`
- Mean latency by prompt:
  - `concise_prompt`: `4.249s`
  - `long_prompt`: `4.510s`
- Mean latency by image size mode:
  - `compressed`: `3.609s`
  - `compressed_downscale_restore`: `3.667s`
  - `downscale_restore`: `4.063s`
  - `raw`: `6.179s`
- Mean latency by background mode:
  - `none`: `4.367s`
  - `color`: `4.391s`

Fastest variants in the report:

1. `range=500`, `long_prompt`, `compressed_downscale_restore+bg_none`: `2.739s`
2. `range=500`, `long_prompt`, `compressed+bg_none`: `2.816s`
3. `range=250`, `long_prompt`, `compressed_downscale_restore+bg_none`: `2.857s`

Interpretation for the paper draft:

- Raw screenshots were substantially slower than compressed or compressed-downscaled variants.
- Lower normalization ranges were modestly faster on average than `1000`.
- Background color padding had much smaller effect than image size mode.
- Prompt length mattered, but less than image transport/size decisions in the saved report.

Caveat:

- This is an exploratory rough-test artifact, not a full benchmark section.
- The saved report is latency-oriented. The repository also contains quality-benchmark code with a quality gate, but no full saved quality report is currently checked in for this run.
