# 테스트와 품질

프로젝트는 이미 레이어별 테스트 표면을 갖고 있습니다. 바뀐 것을 증명하는 가장 작은 테스트부터 돌리는 것이 좋습니다.

## 주요 명령

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

`just` 사용 시:

```bash
just qa
just test
just coverage
```

## 테스트 영역

| 폴더 | 검증 대상 |
| --- | --- |
| `tests/agent/` | turn loop, runtime modules, checkpoints |
| `tests/evaluator/` | bbox와 Civ6 evaluation logic |
| `tests/utils/` | providers, parsers, screen helpers, run-log cache |
| `tests/mcp/` | layered MCP server behavior |
| `tests/rough_test/` | exploratory/heavier tests와 reports |

## integration marker

pytest config에는 `integration` marker가 정의되어 있습니다.

```bash
pytest -m "not integration"
```

빠른 로컬 패스를 원할 때 유용합니다.

## CI

메인 테스트 workflow는 현재:

- Ruff checks
- Python `3.12`, `3.13` 테스트
- 오디오 관련 dependency를 위한 Linux system packages 설치

docs workflow는 pull request마다 docs 사이트를 strict mode로 build하고, `main` 또는 `master` push 시 GitHub Pages에 배포합니다.

## 머지 전 확인

- 변경된 모듈에 대한 가장 작은 targeted test
- 중요한 runtime 변경이면 full `pytest`
- docs나 nav를 바꿨다면 `make docs-build`
- HitL나 status UI 코드를 건드렸다면 실제 dashboard 흐름 점검
