# 아키텍처 가이드

이 페이지는 루트 레벨 아키텍처 노트를 docs 포털 안으로 가져온 버전입니다.

## 상위 데이터 흐름

```text
turn_runner.py
  -> provider setup
  -> HITL setup
  -> logging and run sessions
  -> run_multi_turn()

run_multi_turn()
  -> run_one_turn()
      -> capture screenshot
      -> route primitive
      -> plan action
      -> execute action
      -> update context and checkpoints
```

## 핵심 런타임 파일

| 파일 | 역할 |
| --- | --- |
| `civStation/agent/turn_runner.py` | CLI와 runtime wiring |
| `civStation/agent/turn_executor.py` | observe, route, plan, execute loop |
| `civStation/mcp/server.py` | layered MCP facade |
| `civStation/utils/image_pipeline.py` | per-site image preprocessing |
| `civStation/utils/llm_provider/` | provider implementations |

## 왜 이렇게 나뉘어 있나

- routing과 planning은 다른 문제
- context는 단일 클릭보다 오래 살아야 함
- strategy는 전체 prompt를 다시 쓰지 않고도 수정 가능해야 함
- human control은 안전하게 루프를 중단할 수 있어야 함
- MCP는 내부 import 없이도 아키텍처를 노출해야 함

## 디렉터리 뷰

```text
civStation/
  agent/
    modules/
      context/
      strategy/
      router/
      primitive/
      hitl/
      knowledge/
      memory/
  evaluation/
  mcp/
  utils/
```

## 함께 읽으면 좋은 페이지

- [멘탈 모델](../concepts/mental-model.md)
- [레이어](../concepts/layers.md)
- [실행 루프](../concepts/execution-loop.md)
