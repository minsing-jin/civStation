# 提供商与图像管线

provider 的选择与图像预处理，是影响质量和成本的两个最大杠杆。

## 支持的 provider 名称

以 `civStation.utils.llm_provider.create_provider()` 为准：

| provider flag | 含义 | 默认模型 |
| --- | --- | --- |
| `claude` | Anthropic VLM provider | `claude-4-5-sonnet-20241022` |
| `gemini` | Google GenAI VLM provider | `gemini-3-flash-preview` |
| `gpt` | OpenAI VLM provider | `gpt-4o` |
| `openai` | `gpt` 的别名 | `gpt-4o` |
| `openai-computer` | OpenAI computer-use 风格 provider | `gpt-5.4` |
| `anthropic-computer` | Claude computer-use 风格 provider | 继承 Claude 默认模型 |
| `mock` | 用于测试的 deterministic fake provider | `mock-vlm` |

## 实用选择指南

| 需求 | 推荐设置 |
| --- | --- |
| 最便宜的实验 | `gemini` 或 `mock` |
| 更强的 planner 质量 | planning 用 `claude` |
| OpenAI 视觉规划 | `gpt` |
| tool-native computer-use 实验 | `openai-computer` 或 `anthropic-computer` |
| 零 API 调用测试 | `mock` |

## 按角色拆分 provider

CLI 为以下位置提供独立 provider slots：

- router
- planner
- turn detector

因为它们承担不同工作，所以这种拆分很有价值。

示例：

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turn-detector-provider gemini
```

## 图像管线的调用位置

每个调用位置都可以有不同的图像预处理：

- `router`
- `planner`
- `context`
- `turn-detector`

## 内置 preset

来自 `civStation/utils/image_pipeline.py`：

- `router_default`
- `planner_default`
- `context_default`
- `turn_detector_default`
- `planner_high_quality`
- `observation_fast`
- `policy_tab_check_fast`
- `city_production_followup_fast`
- `city_production_placement_fast`

## 主要图像控制项

| 参数后缀 | 含义 |
| --- | --- |
| `img-preset` | preset 名称 |
| `img-max-long-edge` | resize 上限 |
| `img-ui-filter` | UI filtering mode |
| `img-color` | color policy |
| `img-encode` | transport encoding simulation |
| `img-jpeg-quality` | JPEG quality override |

示例：

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-img-preset router_default \
  --planner-img-preset planner_high_quality \
  --context-img-max-long-edge 1280
```

## 为什么重要

- routing 往往适合更激进的简化
- planning 往往需要更多细节
- turn detection 可能与 planner 有不同权衡
- resize 与 encoding 会显著改变延迟和成本

把图像预处理当作一等调优表面，而不是隐藏的 plumbing。
