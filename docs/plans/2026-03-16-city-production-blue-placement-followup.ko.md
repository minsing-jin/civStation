# 도시 생산 파란 타일 배치 후속 구현 계획

이 설계 노트는 `city production placement` 이후 파란색 구매 타일이 나타나는 경우를 처리하기 위한 후속 플로우를 설명합니다.

## 핵심 목표

- 구매 가능한 파란 타일이 개입하는 배치 화면 처리
- 마지막 build confirmation 전에 같은 타일을 다시 클릭하는 deterministic re-click
- current gold, adjacency, strategy reasoning을 placement prompt에 명시

## 설계 요약

- `CityProductionProcess`에 lightweight post-placement resolver stage를 추가
- short-term memory에 마지막 placement click 저장
- 화면이 여전히 placement인지, confirm 단계인지 분기

## 구현 포인트

- `tests/agent/modules/primitive/test_multi_step_process.py`로 동작 고정
- `civStation/agent/modules/memory/short_term_memory.py`에 follow-up state 추가
- `civStation/agent/modules/primitive/multi_step_process.py`에 후속 stage 추가

원문은 상세 task-by-task plan을 포함하고, 이 페이지는 한국어 요약입니다.
