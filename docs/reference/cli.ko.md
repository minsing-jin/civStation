# CLI 레퍼런스

권장 CLI 진입점:

```bash
uv run civstation
uv run civstation run --help
```

설치 후 실행 명령:

```bash
civstation run --help
```

fallback 모듈 실행:

```bash
python -m civStation
```

루트 CLI는 온보딩, 모바일/운영자 체크리스트, GitHub star 빠른 액션을 먼저 보여줍니다.
`civstation run ...`은 나머지 플래그를 기존 `turn_runner` parser로 그대로 넘기므로, 여전히 ConfigArgParse가 CLI 인자와 `config.yaml`을 함께 읽습니다.

## Provider 설정

| 플래그 그룹 | 목적 |
| --- | --- |
| `--provider`, `--model` | 기본 provider/model |
| `--router-provider`, `--router-model` | router만 override |
| `--planner-provider`, `--planner-model` | planner만 override |
| `--turn-detector-provider`, `--turn-detector-model` | turn detection만 override |

현재 help 출력 기준 지원 provider:

```text
claude, gemini, gpt, openai, openai-computer, anthropic-computer, mock
```

## 실행 파라미터

| 플래그 | 의미 |
| --- | --- |
| `--turns` | 턴 수 |
| `--range` | normalized coordinate range |
| `--delay-action` | 행동 사이 딜레이 |
| `--delay-turn` | 턴 사이 딜레이 |
| `--prompt-language` | primitive prompt 언어 |
| `--debug` | `context`, `turns`, `all` 같은 debug 기능 목록 |

## Strategy와 HITL

| 플래그 | 의미 |
| --- | --- |
| `--strategy` | 고수준 strategy 텍스트 |
| `--hitl` | human-in-the-loop 모드 활성화 |
| `--autonomous` | autonomous 모드 활성화 |
| `--hitl-mode` | interrupt mode, 현재는 `async` |

## Chat App Integration

| 플래그 계열 | 의미 |
| --- | --- |
| `--chatapp` | `original`, `discord`, `whatsapp` |
| `--discord-*` | Discord token, channel, user 설정 |
| `--whatsapp-*` | WhatsApp token, user 설정 |
| `--enable-discussion` | strategy discussion engine 활성화 |

## Knowledge Retrieval

| 플래그 | 의미 |
| --- | --- |
| `--knowledge-index` | 로컬 Civopedia index 경로 |
| `--enable-web-search` | 가능하면 Tavily web search 활성화 |

## Control API와 Status UI

| 플래그 | 의미 |
| --- | --- |
| `--status-ui` | dashboard와 control API 활성화 |
| `--control-api` | 전체 dashboard 없이 lifecycle API 활성화 |
| `--status-port` | UI/API bind 포트 |
| `--wait-for-start` | 외부 start 신호를 기다림 |

## Image Pipeline 플래그

다음 각 site마다 같은 형태의 이미지 플래그가 있습니다.

- `router`
- `planner`
- `context`
- `turn-detector`

site별:

```text
--{site}-img-preset
--{site}-img-max-long-edge
--{site}-img-ui-filter
--{site}-img-color
--{site}-img-encode
--{site}-img-jpeg-quality
```

## Relay 플래그

| 플래그 | 의미 |
| --- | --- |
| `--relay-url` | relay server의 WebSocket URL |
| `--relay-token` | relay auth token |

## 대표 명령

빠른 로컬 실행:

```bash
uv run civstation run \
  --provider gemini \
  --turns 50 \
  --status-ui
```

제어 중심 실행:

```bash
uv run civstation run \
  --router-provider gemini \
  --planner-provider claude \
  --status-ui \
  --wait-for-start
```
