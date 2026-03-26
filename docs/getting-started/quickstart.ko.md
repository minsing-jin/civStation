# 빠른 시작

clone 이후 실제로 제어 가능한 실행까지 가는 가장 짧은 경로입니다.

## 1. 설치

```bash
make install
```

## 2. API 키 설정

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
```

## 3. 상태 UI와 함께 에이전트 실행

```bash
python -m civStation.agent.turn_runner \
  --provider claude \
  --turns 100 \
  --strategy "Focus on science victory" \
  --status-ui \
  --wait-for-start \
  --status-port 8765
```

이 명령은 다음을 수행합니다.

- 라이브 turn loop 시작
- 내장 dashboard와 control API 활성화
- 바로 행동하지 않고 명시적인 start 신호 대기
- 실행 중에도 strategy를 보이고 수정 가능하게 유지

## 4. 대시보드 열기

```text
http://127.0.0.1:8765
```

여기서 start, pause, resume, stop, directive 전송이 가능합니다.

## 5. 선택 사항: layered MCP 서버 실행

다른 터미널에서:

```bash
python -m civStation.mcp.server
```

외부 도구나 skill이 내부 Python 모듈 직접 import 대신 MCP를 통해 이 아키텍처를 제어해야 할 때 사용합니다.

## 6. 선택 사항: 기본 `config.yaml` 사용

저장소에는 이미 프로젝트 레벨 기본 설정 파일이 있습니다. 따라서 필요한 것만 override해 실행할 수 있습니다.

```bash
python -m civStation.agent.turn_runner \
  --config config.yaml \
  --provider gemini \
  --status-ui
```

CLI 플래그가 `config.yaml`보다 우선합니다.

## 다음 단계

- 운영자 흐름을 보려면 [첫 실전 실행](first-live-run.md)
- 더 다양한 실행 패턴은 [에이전트 실행](../guides/running-the-agent.md)
- 다른 시스템과 통합하려면 [레이어드 MCP](../concepts/layered-mcp.md)
