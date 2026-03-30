# Computer-use 프로바이더 설계

이 문서는 OpenAI와 Anthropic computer-use provider를 기존 `BaseVLMProvider` 계약 안에 맞춰 넣는 설계를 다룹니다.

## 목표

- 현재 router/executor architecture를 교체하지 않고
- `openai-computer`, `anthropic-computer` provider를 추가
- planner-side `analyze()`에서 provider-specific computer-use API/tool 사용

## 핵심 결정

- 별도의 autonomous agent loop로 가지 않고
- planner adapter 형태로 computer-use provider를 붙입니다

## 이유

코드베이스는 이미 다음을 갖고 있기 때문입니다.

1. screenshot capture
2. primitive routing
3. action planning
4. local execution

따라서 가장 좋은 통합 지점은 별도 루프가 아니라 planner-side `analyze()`입니다.

이 페이지는 설계 요약이며, 원문은 option 비교와 accepted shape를 더 자세히 설명합니다.
