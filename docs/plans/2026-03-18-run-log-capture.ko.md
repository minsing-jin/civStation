# 실행 로그 캡처 구현 계획

이 구현 계획은 `turn_runner` 실행 중 보이는 주요 출력을 최신 로그 파일 하나에 모두 기록하는 작업을 다룹니다.

## 목표

- `logging`, stdout, stderr, traceback을 하나의 latest-run 파일에 기록
- 새 실행 시 동일 파일을 덮어쓰기

## 아키텍처

- `RunLogSession`이 root logging handler와 stdout/stderr tee를 함께 소유
- close 시 모든 monkeypatch와 핸들러를 원복

## 구현 포인트

- `tests/utils/test_run_log_cache.py`에 print/stderr capture regression 추가
- `civStation/utils/run_log_cache.py`에 tee stream wrapper 구현
- `session.close()`가 stdout/stderr를 복원하는지 검증

이 페이지는 구현 요약이며, 원문은 실패 테스트부터 시작하는 상세 실행 계획을 포함합니다.
