# Turn Runner Raw Log Cache 구현 계획

이 구현 계획은 `turn_runner`에 opt-in raw run log cache를 추가하는 작업을 다룹니다.

## 목표

- 실행마다 최신 로그 파일 하나를 덮어쓰기
- plain logging과 uncaught tracebacks를 함께 저장
- 기존 Rich output은 유지

## 아키텍처

- 새로운 utility module이 temp-file path와 logger file handler, temporary `sys.excepthook` wrapper를 소유
- `turn_runner.main()`이 startup에서 opt-in
- cleanup에서 항상 teardown

## 구현 포인트

- `tests/utils/test_run_log_cache.py`에 regression tests 추가
- `RunLogSession` 같은 session object 도입
- root logger + traceback capture를 묶어서 관리

원문은 task-by-task implementation steps를 포함하고, 이 페이지는 한국어 요약입니다.
