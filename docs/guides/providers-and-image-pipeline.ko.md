# 프로바이더와 이미지 파이프라인

provider 선택과 이미지 전처리는 품질과 비용을 좌우하는 가장 큰 레버 중 두 가지입니다.

## 지원 provider 이름

`civStation.utils.llm_provider.create_provider()` 기준:

| provider flag | 의미 | 기본 모델 |
| --- | --- | --- |
| `claude` | Anthropic VLM provider | `claude-4-5-sonnet-20241022` |
| `gemini` | Google GenAI VLM provider | `gemini-3-flash-preview` |
| `gpt` | OpenAI VLM provider | `gpt-4o` |
| `openai` | `gpt`의 alias | `gpt-4o` |
| `openai-computer` | OpenAI computer-use 스타일 provider | `gpt-5.4` |
| `anthropic-computer` | Claude computer-use 스타일 provider | Claude 기본 모델 상속 |
| `mock` | 테스트용 deterministic fake provider | `mock-vlm` |

## 실전 선택 가이드

| 필요 | 추천 설정 |
| --- | --- |
| 가장 저렴한 실험 | `gemini` 또는 `mock` |
| planner 품질 우선 | planning에 `claude` |
| OpenAI 비전 플래닝 | `gpt` |
| tool-native computer-use 실험 | `openai-computer` 또는 `anthropic-computer` |
| API 호출 없는 테스트 | `mock` |

## 역할별 provider 분리

CLI는 다음에 대해 독립 provider slots를 제공합니다.

- router
- planner
- turn detector

이들은 서로 다른 일을 하므로 분리 가치가 큽니다.

예시:

```bash
python -m civStation.agent.turn_runner \
  --router-provider gemini \
  --planner-provider claude \
  --turn-detector-provider gemini
```

## 이미지 파이프라인 호출 지점

각 호출 지점마다 다른 이미지 전처리를 설정할 수 있습니다.

- `router`
- `planner`
- `context`
- `turn-detector`

## 내장 preset

`civStation/utils/image_pipeline.py` 기준:

- `router_default`
- `planner_default`
- `context_default`
- `turn_detector_default`
- `planner_high_quality`
- `observation_fast`
- `policy_tab_check_fast`
- `city_production_followup_fast`
- `city_production_placement_fast`

## 주요 이미지 제어 항목

| 플래그 suffix | 의미 |
| --- | --- |
| `img-preset` | preset 이름 |
| `img-max-long-edge` | resize limit |
| `img-ui-filter` | UI filtering mode |
| `img-color` | color policy |
| `img-encode` | transport encoding simulation |
| `img-jpeg-quality` | JPEG quality override |

예시:

```bash
python -m civStation.agent.turn_runner \
  --provider gemini \
  --router-img-preset router_default \
  --planner-img-preset planner_high_quality \
  --context-img-max-long-edge 1280
```

## 왜 중요한가

- routing은 공격적인 단순화가 잘 맞는 경우가 많고
- planning은 더 많은 디테일을 필요로 하며
- turn detection은 planner와 다른 tradeoff가 필요할 수 있고
- resize와 encoding은 latency와 cost를 크게 바꿉니다

이미지 전처리를 보이지 않는 plumbing이 아니라 1급 튜닝 표면으로 다루는 것이 좋습니다.
