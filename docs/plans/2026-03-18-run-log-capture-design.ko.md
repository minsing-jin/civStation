# 통합 실행 로그 캡처 설계

이 문서는 `turn_runner`의 `logging`, Rich, `print`, uncaught traceback을 하나의 최신 실행 로그 파일에 함께 남기는 설계를 설명합니다.

## 목표

- 터미널에 보이는 주요 출력을 모두 같은 latest-run 파일에 기록
- 새 실행 시 같은 파일을 덮어쓰기
- 터미널에서의 기존 출력 경험은 유지

## 핵심 설계

- `RunLogSession`이 latest-run 파일 하나를 소유
- root `FileHandler`는 유지
- `sys.stdout`과 `sys.stderr`를 tee stream wrapper로 감싸서 원래 스트림과 파일에 동시에 기록
- 종료 시 stdout, stderr, excepthook, logger handler를 모두 원복

## 왜 필요한가

기존 raw-log cache는 logging과 uncaught exception에 강하지만, Rich 출력과 `print()`를 일관되게 보존하지 못했습니다.

이 페이지는 설계 요약이며, 원문은 더 세밀한 current state와 restore semantics를 포함합니다.
