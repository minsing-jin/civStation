# Computer-use 프로바이더 구현 계획

이 구현 계획은 기존 provider factory와 planner `analyze()` 경로 안에 OpenAI / Anthropic computer-use provider를 추가하는 작업을 다룹니다.

## 목표

- `openai-computer`, `anthropic-computer`를 provider factory에서 선택 가능하게 만들기
- 현재 router와 multi-action flow를 깨지 않기
- `analyze()`만 computer-use API를 쓰고 나머지는 기존 경로를 유지

## 아키텍처

- shared action-mapping helper 추가
- `OpenAIComputerVLMProvider`, `AnthropicComputerVLMProvider` 구현
- 기존 GPT / Claude provider를 상속
- `_send_to_api()`와 `analyze_multi()`는 기존 JSON path 유지

## 구현 포인트

- action translation tests 추가
- coordinate normalization tests 추가
- `create_provider()` 등록 검증

이 페이지는 구현 요약이며, 원문은 단계별 테스트/구현 계획을 포함합니다.
