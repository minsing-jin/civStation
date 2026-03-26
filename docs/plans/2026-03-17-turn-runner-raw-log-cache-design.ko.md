# Turn Runner Raw Log Cache 설계

이 문서는 `turn_runner`가 Rich live 출력은 유지하면서도 plain-text 최신 실행 로그를 따로 남기도록 만드는 설계를 다룹니다.

## 문제

- 현재는 Rich 출력이 운영자 가시성에는 좋지만
- 실패 후 외부 coding agent가 읽을 수 있는 raw text log가 부족합니다
- 유용한 문맥이 터미널 상태와 traceback에 흩어집니다

## 목표

- 매 실행마다 최신 로그 파일 하나만 유지
- Python logging records와 uncaught traceback을 함께 보존
- 기존 Rich UX는 그대로 유지

## 추천안

가장 적절한 방식은:

- root logger file handler
- `sys.excepthook`

를 함께 쓰는 방식입니다.

## 설계 요약

- `civStation/utils/run_log_cache.py` 추가
- latest-run temp cache path를 deterministic하게 계산
- 실행 시작 시 파일을 덮어쓰고 종료 시 정리

이 페이지는 설계 요약이며, 원문은 더 자세한 옵션 비교를 포함합니다.
