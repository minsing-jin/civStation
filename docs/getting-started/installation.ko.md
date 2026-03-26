# 설치

로컬에서 실제로 실행 가능한 환경을 만드는 가장 직접적인 경로부터 시작합니다.

## 요구 사항

- Python `3.10+`
- 에이전트가 관찰하고 제어할 수 있는 로컬 Civilization VI 환경
- 사용할 모델 provider의 API 키
- 운영체제의 화면 캡처 및 입력 제어 권한

Linux에서 voice 기능이나 전체 dashboard 스택을 쓸 경우 PortAudio 같은 시스템 패키지가 필요할 수 있습니다. CI도 이 때문에 Ubuntu에서 `portaudio19-dev`와 `gcc`를 설치합니다.

## 기본 설치

일반적인 프로젝트 작업용:

```bash
make install
```

이 명령은 프로젝트를 editable 모드로 설치하고 test dependencies와 `pre-commit`까지 설정합니다.

## docs 지원까지 포함한 설치

문서 사이트를 로컬에서 띄우거나 빌드하려면:

```bash
uv pip install -e ".[docs,test]"
```

이후에는 다음 단축 명령을 쓸 수 있습니다.

```bash
make docs-serve
make docs-build
```

## 환경 변수

실제로 사용할 provider만 설정하면 됩니다.

```env
ANTHROPIC_API_KEY=...
GENAI_API_KEY=...
OPENAI_API_KEY=...
RELAY_TOKEN=...
```

provider 별칭은 [프로바이더와 이미지 파이프라인](../guides/providers-and-image-pipeline.md)에 정리되어 있습니다.

## 설치 확인

CLI 확인:

```bash
python -m civStation.agent.turn_runner --help
```

MCP 서버 엔트리 확인:

```bash
python -m civStation.mcp.server
```

MCP 서버는 기본적으로 stdio transport를 사용하므로, 브라우저를 열지 않고 클라이언트를 기다리는 상태가 정상입니다.

## 다음 단계

[빠른 시작](quickstart.md)으로 최소 실행 루프를 돌려보세요.
