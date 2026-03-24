# Unified Run Log Capture Design

**Date:** 2026-03-18

**Goal:** `turn_runner` 실행 중 터미널에 보이는 주요 출력(`logging`, Rich, `print`, uncaught traceback)을 최신 실행 로그 파일 하나에 함께 남기고, 새 실행 시에는 같은 파일을 덮어쓴다.

## Current State

- 루트 `logging` 레코드는 `RunLogSession`이 붙이는 `FileHandler`를 통해 최신 로그 파일에 남는다.
- Rich 대시보드와 일부 디버그 출력은 `Console.print()` 또는 `print()`를 사용하므로 최신 로그 파일에 일관되게 남지 않는다.
- uncaught exception traceback은 `sys.excepthook` 교체로 파일에 남긴다.
- 최신 로그 파일은 실행마다 동일 경로를 재사용하고 `mode="w"`로 덮어쓴다.

## Design

### 1. One session owns one latest-run file

`RunLogSession`은 기존처럼 deterministic path 하나를 소유하고, 새 실행 시 `mode="w"`로 내용을 초기화한다. 이 동작은 유지한다.

### 2. Tee stdout/stderr into the same file

`RunLogSession`이 시작될 때:

- 기존 root `FileHandler`를 유지한다.
- `sys.stdout` / `sys.stderr`를 감싸는 tee stream wrapper를 설치한다.
- wrapper는 원래 스트림으로도 쓰고, 같은 로그 파일 스트림에도 즉시 flush 한다.

이렇게 하면 `Console.print()`와 `print()`가 같은 최신 로그 파일에 기록된다.

### 3. Preserve terminal behavior

tee wrapper는 원래 stdout/stderr에 그대로 전달해야 하므로, 사용자는 지금처럼 터미널에서 로그를 계속 본다. 파일 기록만 추가된다.

### 4. Restore everything on close

세션 종료 시:

- `sys.stdout` / `sys.stderr`
- `sys.excepthook`
- root logger handler

를 원래 상태로 되돌린다.

## Non-Goals

- 과거 실행 로그를 누적 저장하지 않는다.
- 모든 내부 상태를 별도 구조화 JSON으로 저장하지 않는다.
- `DEBUG` 레벨까지 강제로 켜지는 동작은 이번 범위에 포함하지 않는다.

## Testing

- root logger 레코드는 계속 파일에 남는지
- `print()` / `stderr.write()`가 파일에 남는지
- 세션 종료 후 stdout/stderr가 원복되는지
- 새 실행이 이전 내용을 덮어쓰는지
- uncaught traceback 기록이 계속 유지되는지
