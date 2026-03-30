# 스택 확장

이 프로젝트는 레이어 단위로 확장되도록 설계되어 있습니다.

## primitive 추가

여기서 시작합니다.

- `civStation/agent/modules/router/primitive_registry.py`
- `civStation/agent/modules/primitive/`

일반적인 흐름:

1. registry에 primitive 정의 추가
2. prompt logic 추가 또는 수정
3. planning 또는 multi-step behavior 처리
4. tests 추가
5. 문서 갱신

## MCP adapter 추가 또는 교체

여기서 시작합니다.

- `civStation/mcp/runtime.py`
- `civStation/mcp/server.py`

public MCP surface를 바꾸지 않고 다른 router, planner, context observer, strategy refiner, executor를 붙이고 싶다면 adapter override를 사용하세요.

## HitL 확장

여기서 시작합니다.

- `civStation/agent/modules/hitl/command_queue.py`
- `civStation/agent/modules/hitl/status_ui/server.py`
- `civStation/agent/modules/hitl/relay/relay_client.py`

directive priority와 lifecycle semantics는 편의 기능이 아니라 운영 안전 문제입니다. 특히 조심해야 합니다.

## provider 또는 image handling 확장

여기서 시작합니다.

- `civStation/utils/llm_provider/`
- `civStation/utils/image_pipeline.py`

모델별 동작, preprocessing preset, transport-specific image tuning을 넣기 좋은 자리입니다.

## skill 확장

프로젝트용 skills는 다음 아래에 있습니다.

- `.codex/skills/`
- `.agents/skills/`

권장 패턴:

1. skill은 얇게 유지
2. MCP를 안정적인 control plane으로 사용
3. 재사용 workflow는 `SKILL.md`에 둠
4. 필요하면 helper script를 skill 옆에 둠

## evaluation 확장

여기서 시작합니다.

- `civStation/evaluation/evaluator/action_eval/bbox_eval/`
- `civStation/evaluation/evaluator/action_eval/civ6_eval/`

더 재사용 가능한 action evaluator가 필요하면 `bbox_eval`을 우선 고려하세요.
