# Trade-off benchmark

- Provider: `gpt`
- Model: `gpt-4o-mini`
- Dataset: `docs/paper/arxiv/benchmarks/synthetic_ui/synthetic_bbox_dataset.jsonl`

## Latency Leaders
| range | prompt | image_variant | avg_latency_sec | payload_bytes |
| --- | --- | --- | --- | --- |
| 500 | concise_prompt | raw+bg_none+baseline | 1.952 | 11798 |
| 500 | concise_prompt | compressed+bg_none+contrast_jpeg | 2.004 | 37200 |
| 250 | concise_prompt | downscale_restore+bg_none+baseline | 2.006 | 18428 |
| 250 | long_prompt | raw+bg_none+contrast_jpeg | 2.137 | 41563 |
| 250 | concise_prompt | compressed+bg_none+baseline | 2.142 | 48931 |
| 250 | long_prompt | raw+bg_none+baseline | 2.165 | 11798 |
| 500 | long_prompt | downscale_restore+bg_none+contrast_jpeg | 2.190 | 24412 |
| 250 | long_prompt | compressed+bg_none+contrast_jpeg | 2.233 | 37200 |

## Quality Leaders
| range | prompt | image_variant | avg_latency_sec | strict_success_rate | avg_step_accuracy | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 250 | concise_prompt | compressed+bg_none+contrast_jpeg | 3.439 | 0.500 | 0.500 | pass |
| 250 | concise_prompt | downscale_restore+bg_none+baseline | 3.426 | 0.333 | 0.333 | pass |
| 500 | concise_prompt | compressed+bg_none+baseline | 2.958 | 0.000 | 0.000 | pass |
| 250 | concise_prompt | compressed+bg_none+baseline | 3.556 | 0.000 | 0.000 | pass |
| 500 | concise_prompt | compressed+bg_none+contrast_jpeg | 3.933 | 0.000 | 0.000 | pass |
| 500 | concise_prompt | downscale_restore+bg_none+baseline | 4.188 | 0.000 | 0.000 | pass |
| 500 | concise_prompt | downscale_restore+bg_none+contrast_jpeg | 4.543 | 0.000 | 0.000 | pass |
| 250 | concise_prompt | downscale_restore+bg_none+contrast_jpeg | 3.941 | 0.000 | 0.000 | fail |
