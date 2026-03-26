# 설정

런타임은 `config.yaml`에서 기본값을 읽고, CLI 플래그가 이를 override합니다.

## 예시 프로젝트 설정

현재 저장소의 config는 대략 다음과 같은 형태입니다.

```yaml
provider: gemini
model: gemini-3-flash-preview
router-provider: gemini
router-model: gemini-3-flash-preview
planner-provider: gemini
planner-model: gemini-3-flash-preview
turns: 100
range: 10000
prompt-language: eng
debug: "all"
router-img-max-long-edge: 1024
strategy: "과학 승리에 집중하고 정찰을 강화해."
hitl: true
autonomous: true
hitl-mode: async
chatapp: original
control-api: true
status-ui: true
```

정확한 현재 기본값은 저장소 루트의 실제 파일을 확인하세요.

## 우선순위

```text
CLI flags > config.yaml > parser defaults
```

즉 `config.yaml`은 안정적인 baseline이고, 실험별 차이는 CLI에서 올리는 구조입니다.

## 주요 섹션

### provider 설정

- `provider`
- `model`
- `router-provider`
- `router-model`
- `planner-provider`
- `planner-model`
- `turn-detector-provider`
- `turn-detector-model`

### 실행 설정

- `turns`
- `range`
- `delay-action`
- `delay-turn`
- `prompt-language`
- `debug`

### strategy와 HITL

- `strategy`
- `hitl`
- `autonomous`
- `hitl-mode`

### chat app integration

- `chatapp`
- `discord-token`
- `discord-channel`
- `discord-user`
- `whatsapp-token`
- `whatsapp-phone-number-id`
- `whatsapp-user`
- `enable-discussion`

### knowledge retrieval

- `knowledge-index`
- `enable-web-search`

### status UI와 control

- `control-api`
- `status-ui`
- `status-port`
- `wait-for-start`

### image pipeline overrides

site별:

- `router-img-*`
- `planner-img-*`
- `context-img-*`
- `turn-detector-img-*`

### relay

- `relay-url`
- `relay-token`

## 권장 방식

- 저장소 레벨 `config.yaml`은 보수적으로 유지
- 실험 차이는 CLI override로 분리
- image pipeline tweak는 한 번에 하나씩 비교
- run setup을 공유할 때는 non-default provider split을 명시
