# VLM Policy Inference Speed Report

## Setup
- Model: `gemini-3-flash-preview`
- Image: `tests/rough_test/screenshot.png`
- Prompts: `long_prompt`, `concise_prompt`
- Size modes: `raw, compressed, downscale_restore`
- Background modes: `none, color`
- Normalizing ranges: `250, 500, 1000`
- Runs per variant: `2`
- Total variants: `36` (= 3 ranges x 2 prompts x 3 size modes x 2 background modes)

## 1) Normalizing Range Comparison (overall)
| normalizing_range | mean_avg_latency_sec | delta_vs_1000 |
| --- | --- | --- |
| 250 | 4.232 | +7.19% |
| 500 | 4.095 | +3.72% |
| 1000 | 3.948 | +0.00% |

## 2) Prompt Effect (overall)
| prompt_variant | mean_avg_latency_sec |
| --- | --- |
| concise_prompt | 4.129 |
| long_prompt | 4.054 |

## 3) Image Size Mode Effect (overall)
| size_variant | mean_avg_latency_sec |
| --- | --- |
| compressed | 3.177 |
| downscale_restore | 3.171 |
| raw | 5.926 |

## 4) Background Effect (overall)
| background_variant | mean_avg_latency_sec |
| --- | --- |
| color | 3.990 |
| none | 4.193 |

## 5) Range x Prompt
| normalizing_range | prompt_variant | mean_avg_latency_sec |
| --- | --- | --- |
| 250 | concise_prompt | 4.260 |
| 250 | long_prompt | 4.203 |
| 500 | concise_prompt | 4.091 |
| 500 | long_prompt | 4.098 |
| 1000 | concise_prompt | 4.036 |
| 1000 | long_prompt | 3.860 |

## 6) Fastest 8 Variants
| range | prompt | image_variant | avg_latency_sec |
| --- | --- | --- | --- |
| 500 | long_prompt | compressed+bg_none | 2.398 |
| 1000 | concise_prompt | downscale_restore+bg_color | 2.659 |
| 1000 | long_prompt | downscale_restore+bg_color | 2.662 |
| 500 | long_prompt | compressed+bg_color | 2.683 |
| 250 | long_prompt | downscale_restore+bg_color | 2.684 |
| 500 | long_prompt | downscale_restore+bg_color | 2.684 |
| 250 | concise_prompt | compressed+bg_color | 2.754 |
| 1000 | concise_prompt | compressed+bg_none | 2.769 |

## 7) Slowest 8 Variants
| range | prompt | image_variant | avg_latency_sec |
| --- | --- | --- | --- |
| 500 | long_prompt | raw+bg_color | 6.520 |
| 250 | long_prompt | raw+bg_none | 6.458 |
| 250 | concise_prompt | raw+bg_color | 6.347 |
| 500 | long_prompt | raw+bg_none | 6.254 |
| 1000 | concise_prompt | raw+bg_color | 5.947 |
| 500 | concise_prompt | raw+bg_none | 5.902 |
| 1000 | concise_prompt | raw+bg_none | 5.889 |
| 1000 | long_prompt | raw+bg_none | 5.813 |

## 8) Full Matrix
| range | prompt | image_variant | size | bg | img_size | avg | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 250 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 2.754 | 2.363 | 3.144 |
| 250 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.951 | 2.976 | 4.926 |
| 250 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 3.660 | 3.056 | 4.263 |
| 250 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 3.641 | 3.321 | 3.962 |
| 250 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 6.347 | 5.952 | 6.742 |
| 250 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 5.208 | 5.155 | 5.261 |
| 250 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 4.116 | 3.670 | 4.562 |
| 250 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.182 | 2.741 | 3.623 |
| 250 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 2.684 | 2.534 | 2.834 |
| 250 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 3.115 | 2.592 | 3.637 |
| 250 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 5.664 | 4.569 | 6.758 |
| 250 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 6.458 | 5.967 | 6.948 |
| 500 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.391 | 3.378 | 3.404 |
| 500 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.776 | 3.490 | 4.063 |
| 500 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 3.095 | 3.014 | 3.176 |
| 500 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 2.935 | 2.688 | 3.182 |
| 500 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 5.449 | 5.074 | 5.824 |
| 500 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 5.902 | 5.864 | 5.940 |
| 500 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 2.683 | 2.585 | 2.781 |
| 500 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 2.398 | 2.235 | 2.561 |
| 500 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 2.684 | 2.653 | 2.715 |
| 500 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 4.048 | 3.579 | 4.517 |
| 500 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 6.520 | 6.371 | 6.668 |
| 500 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 6.254 | 6.190 | 6.318 |
| 1000 | concise_prompt | compressed+bg_color | compressed | color | 1588x1057 | 3.043 | 2.818 | 3.267 |
| 1000 | concise_prompt | compressed+bg_none | compressed | none | 1280x853 | 2.769 | 2.670 | 2.868 |
| 1000 | concise_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 2.659 | 2.615 | 2.703 |
| 1000 | concise_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 3.910 | 3.102 | 4.717 |
| 1000 | concise_prompt | raw+bg_color | raw | color | 3500x2332 | 5.947 | 5.640 | 6.255 |
| 1000 | concise_prompt | raw+bg_none | raw | none | 2822x1880 | 5.889 | 5.592 | 6.186 |
| 1000 | long_prompt | compressed+bg_color | compressed | color | 1588x1057 | 2.794 | 2.483 | 3.106 |
| 1000 | long_prompt | compressed+bg_none | compressed | none | 1280x853 | 3.268 | 2.554 | 3.982 |
| 1000 | long_prompt | downscale_restore+bg_color | downscale_restore | color | 1190x794 | 2.662 | 2.511 | 2.813 |
| 1000 | long_prompt | downscale_restore+bg_none | downscale_restore | none | 960x640 | 2.957 | 2.868 | 3.046 |
| 1000 | long_prompt | raw+bg_color | raw | color | 3500x2332 | 5.663 | 5.462 | 5.865 |
| 1000 | long_prompt | raw+bg_none | raw | none | 2822x1880 | 5.813 | 5.754 | 5.872 |
