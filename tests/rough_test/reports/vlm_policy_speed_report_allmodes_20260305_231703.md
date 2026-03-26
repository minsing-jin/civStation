# VLM Policy Inference Speed Report (All Modes)

## Setup
- Model: `gemini-3-flash-preview`
- Image: `tests/rough_test/screenshot.png`
- Prompts: `long_prompt`, `concise_prompt`
- Size modes: `raw, compressed, downscale_restore, compressed_downscale_restore`
- Background modes: `none, color`
- Normalizing ranges: `250, 500, 1000`
- Runs per variant: `2`
- Total variants: `48` (= 3 ranges x 2 prompts x 4 size modes x 2 background modes)

## 1) Normalizing Range Comparison (overall)
| normalizing_range | mean_avg_latency_sec | delta_vs_1000 |
| --- | --- | --- |
| 250 | 4.167 | -9.00% |
| 500 | 4.392 | -4.08% |
| 1000 | 4.579 | +0.00% |

## 2) Prompt Effect (overall)
| prompt_variant | mean_avg_latency_sec |
| --- | --- |
| concise_prompt | 4.249 |
| long_prompt | 4.510 |

## 3) Image Size Mode Effect (overall)
| size_variant | mean_avg_latency_sec |
| --- | --- |
| compressed | 3.609 |
| compressed_downscale_restore | 3.667 |
| downscale_restore | 4.063 |
| raw | 6.179 |

## 4) Background Effect (overall)
| background_variant | mean_avg_latency_sec |
| --- | --- |
| color | 4.391 |
| none | 4.367 |

## 5) Range x Prompt
| normalizing_range | prompt_variant | mean_avg_latency_sec |
| --- | --- | --- |
| 250 | concise_prompt | 4.002 |
| 250 | long_prompt | 4.332 |
| 500 | concise_prompt | 4.303 |
| 500 | long_prompt | 4.481 |
| 1000 | concise_prompt | 4.442 |
| 1000 | long_prompt | 4.716 |

## 6) Fastest 10 Variants
| range | prompt | image_variant | avg_latency_sec |
| --- | --- | --- | --- |
| 500 | long_prompt | compressed_downscale_restore+bg_none | 2.739 |
| 500 | long_prompt | compressed+bg_none | 2.816 |
| 250 | long_prompt | compressed_downscale_restore+bg_none | 2.857 |
| 500 | concise_prompt | downscale_restore+bg_none | 2.894 |
| 500 | long_prompt | compressed+bg_color | 2.918 |
| 250 | concise_prompt | downscale_restore+bg_none | 2.946 |
| 250 | long_prompt | compressed_downscale_restore+bg_color | 2.954 |
| 500 | concise_prompt | compressed_downscale_restore+bg_color | 3.035 |
| 250 | long_prompt | downscale_restore+bg_color | 3.115 |
| 250 | concise_prompt | downscale_restore+bg_color | 3.163 |

## 7) Slowest 10 Variants
| range | prompt | image_variant | avg_latency_sec |
| --- | --- | --- | --- |
| 500 | long_prompt | raw+bg_color | 7.249 |
| 1000 | long_prompt | compressed_downscale_restore+bg_none | 6.804 |
| 1000 | long_prompt | raw+bg_none | 6.686 |
| 500 | concise_prompt | raw+bg_none | 6.659 |
| 1000 | concise_prompt | raw+bg_color | 6.529 |
| 250 | long_prompt | raw+bg_color | 6.300 |
| 500 | long_prompt | downscale_restore+bg_color | 6.108 |
| 500 | concise_prompt | raw+bg_color | 5.991 |
| 1000 | long_prompt | raw+bg_color | 5.964 |
| 250 | long_prompt | raw+bg_none | 5.896 |

## 8) Full Matrix
| range | prompt | image_variant | size | bg | img_size | avg | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.500 | 3.022 | 3.977 |
| 250 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.532 | 3.005 | 4.059 |
| 250 | concise_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 3.381 | 2.871 | 3.891 |
| 250 | concise_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 4.167 | 3.332 | 5.002 |
| 250 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 3.163 | 2.844 | 3.482 |
| 250 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 2.946 | 2.742 | 3.149 |
| 250 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 5.884 | 5.741 | 6.028 |
| 250 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 5.441 | 5.301 | 5.582 |
| 250 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.736 | 3.608 | 3.863 |
| 250 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 4.520 | 4.001 | 5.038 |
| 250 | long_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 2.954 | 2.834 | 3.075 |
| 250 | long_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 2.857 | 2.733 | 2.982 |
| 250 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 3.115 | 3.047 | 3.182 |
| 250 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 5.279 | 4.211 | 6.346 |
| 250 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 6.300 | 5.883 | 6.717 |
| 250 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 5.896 | 5.317 | 6.475 |
| 500 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.544 | 3.424 | 3.664 |
| 500 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 4.006 | 2.895 | 5.117 |
| 500 | concise_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 3.035 | 2.798 | 3.273 |
| 500 | concise_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 3.319 | 2.848 | 3.789 |
| 500 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 4.975 | 4.319 | 5.630 |
| 500 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 2.894 | 2.848 | 2.940 |
| 500 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 5.991 | 5.912 | 6.071 |
| 500 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 6.659 | 5.945 | 7.372 |
| 500 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 2.918 | 2.770 | 3.066 |
| 500 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 2.816 | 2.422 | 3.211 |
| 500 | long_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 3.479 | 2.764 | 4.193 |
| 500 | long_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 2.739 | 2.354 | 3.124 |
| 500 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 6.108 | 6.021 | 6.196 |
| 500 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 4.719 | 3.696 | 5.743 |
| 500 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 7.249 | 6.915 | 7.583 |
| 500 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 5.824 | 5.695 | 5.952 |
| 1000 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 4.070 | 3.731 | 4.409 |
| 1000 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.675 | 2.987 | 4.363 |
| 1000 | concise_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 3.753 | 3.073 | 4.432 |
| 1000 | concise_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 3.691 | 3.449 | 3.933 |
| 1000 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 4.290 | 3.071 | 5.509 |
| 1000 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 3.796 | 3.531 | 4.062 |
| 1000 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 6.529 | 6.425 | 6.633 |
| 1000 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 5.730 | 5.598 | 5.861 |
| 1000 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.675 | 3.574 | 3.776 |
| 1000 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.314 | 3.259 | 3.369 |
| 1000 | long_prompt | compressed_downscale_restore+bg_color | compressed_downscale_restore | color | 1190x794 | 3.819 | 3.621 | 4.017 |
| 1000 | long_prompt | compressed_downscale_restore+bg_none | compressed_downscale_restore | none | 960x640 | 6.804 | 5.054 | 8.554 |
| 1000 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 3.958 | 3.391 | 4.525 |
| 1000 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 3.509 | 3.062 | 3.956 |
| 1000 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 5.964 | 5.920 | 6.007 |
| 1000 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 6.686 | 6.556 | 6.816 |
