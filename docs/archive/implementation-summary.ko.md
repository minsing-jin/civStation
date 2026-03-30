# 구현 이력 요약

이 페이지는 오래된 implementation summary를 docs 포털 안에 보존하기 위한 문서입니다.

## 원래 요약이 다루던 것

당시 구현 요약은 프로젝트의 이전 단계에 집중했습니다.

- static evaluator tests의 pytest migration
- VLM provider integration
- primitive 수준 provider support
- prompt module 정리
- evaluator용 문서

## 그 시점의 주요 마일스톤

### 테스트 마이그레이션

tolerance, parsing, integration coverage를 갖춘 evaluator tests의 pytest 전환이 기록되어 있었습니다.

### provider integration

다음 지원 추가가 문서화되어 있었습니다.

- Claude
- Gemini
- GPT
- provider factory helpers

### primitive integration

여러 primitive에 optional VLM provider와 custom prompt가 연결되기 시작한 시기이기도 했습니다.

## 왜 남겨 두는가

이 문서는 프로젝트 고고학용입니다. 현재 layered runtime과 MCP surface가 자리잡기 전 evaluator와 provider stack이 어떻게 발전했는지 보여 줍니다.

## 현재 기준 문서

최신 사용법은 다음을 우선 보세요.

- [프로바이더와 이미지 파이프라인](../guides/providers-and-image-pipeline.md)
- [평가](../guides/evaluation.md)
- [프로젝트 구조](../reference/project-layout.md)
