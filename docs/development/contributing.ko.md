# 기여하기

이 페이지는 `CONTRIBUTING.md`의 실전 버전입니다.

## setup

```bash
make install
```

docs까지 함께 편집한다면:

```bash
uv pip install -e ".[docs,test]"
```

## 일상 명령

```bash
make lint
make format
make check
make test
make coverage
make docs-build
```

또는 `just` 사용:

```bash
just qa
just docs-build
```

## 기여 원칙

- 변경은 작고 되돌리기 쉽게 유지
- 동작이 바뀌면 docs도 같이 갱신
- 로직이 바뀌면 tests도 같이 추가/수정
- 아키텍처를 우회하기보다 기존 레이어를 자연스럽게 확장

## 좋은 PR 범위

- primitive 개선 + 대응 tests
- 새 MCP adapter 또는 tool 개선
- HitL surface 수정 + dashboard/API 검증
- 현재 코드와 일치하는 문서 보완

## 이슈와 제안

- bugs/regressions: GitHub issues
- feature proposals: GitHub issues 또는 명확한 범위의 PR
- docs gap: 보통은 PR이 가장 빠릅니다

## 로컬 docs 워크플로

```bash
make docs-serve
```

브라우저에서 문서 변경을 바로 검토할 수 있습니다.
